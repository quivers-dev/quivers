"""Scalar two-boundary Wiener laws and certified first-passage rejection.

The variance of the driving Brownian motion is one. Signed reaction times
encode the upper boundary positively and the lower boundary negatively.
The image and spectral representations follow Navarro and Fuss (2009),
DOI 10.1016/j.jmp.2009.02.003. The sampler uses one-sided first-passage laws
as a joint envelope: its acceptance probability is between one half and one.
No time-discretized diffusion or approximate inverse CDF is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import (
    Context,
    Decimal,
    DecimalException,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
)
from functools import lru_cache
import math
from typing import Protocol

from quivers.qiec.reference_support import DistributionError, real_parameter


_SMALL_TIME = 0.125
_IMAGE_TERMS = 4
_SPECTRAL_TERMS = 12


class _RandomSource(Protocol):
    """Random primitives shared by the scalar and native host samplers."""

    def random(self) -> float:
        """Return a uniform variate in the half-open unit interval."""
        ...

    def gauss(self, mu: float, sigma: float) -> float:
        """Return a Gaussian variate with the supplied location and scale.

        Parameters
        ----------
        mu, sigma : float
            Location and scale.

        Returns
        -------
        float
            Gaussian variate.
        """
        ...


def _parameters(a: Mapping[str, object]) -> tuple[float, float, float, float]:
    """Validate the four diffusion parameters.

    Parameters
    ----------
    a : Mapping[str, object]
        Boundary separation, nondecision delay, relative bias, and drift.

    Returns
    -------
    tuple of float
        Finite boundary, ndt, bias, and drift.

    Raises
    ------
    DistributionError
        If a parameter is absent, boolean, nonfinite, or outside its domain.
    """
    boundary, ndt, bias, drift = (
        real_parameter(a, name, "Wiener")
        for name in ("boundary", "ndt", "bias", "drift")
    )
    if not all(map(math.isfinite, (boundary, ndt, bias, drift))):
        raise DistributionError("Wiener parameters must be finite")
    if boundary <= 0 or ndt < 0 or not 0 < bias < 1:
        raise DistributionError("Wiener requires boundary>0, ndt>=0, and 0<bias<1")
    return boundary, ndt, bias, drift


def _exprel(x: float) -> float:
    """Evaluate expm1(x)/x continuously at zero.

    Parameters
    ----------
    x : float
        Nonpositive argument.

    Returns
    -------
    float
        Stable divided exponential.
    """
    if abs(x) > 0.01:
        return math.expm1(x) / x
    result = 1 / math.factorial(12)
    for power in range(10, -1, -1):
        result = 1 / math.factorial(power + 1) + x * result
    return result


def _log1mexp(x: float) -> float:
    """Log a positive complement without cancellation.

    Parameters
    ----------
    x : float
        Strictly negative logarithm.

    Returns
    -------
    float
        Logarithm of one minus its exponential.
    """
    return math.log(-math.expm1(x)) if x > -math.log(2) else math.log1p(-math.exp(x))


def _unit_log_ratio(t: float, w: float, opposite: float) -> float:
    """Evaluate the unit-boundary two-boundary/one-boundary density ratio.

    Parameters
    ----------
    t : float
        Strictly positive decision time divided by boundary squared.
    w, opposite : float
        Distances to the selected and other boundaries as fractions of the
        separation. They are retained separately near the parameter endpoints.

    Returns
    -------
    float
        Log density ratio, without choice normalization.

    Raises
    ------
    DistributionError
        If the normalized time exceeds representable arithmetic capacity.
    """
    if t <= 0 or not math.isfinite(t):
        raise DistributionError(
            "Wiener normalized time exceeds floating-point capacity"
        )
    if t <= _SMALL_TIME:
        if w <= 0.5:
            corrections = []
            for n in range(1, _IMAGE_TERMS + 1):
                exponent = -2 * n * (n - w) / t
                if exponent < -745:
                    corrections.append(0.0)
                    continue
                x = -4 * n * w / t
                correction = (8 * n * n / t) * _exprel(x) - 1 - math.exp(x)
                corrections.append(math.exp(exponent) * correction)
            return math.log1p(-math.fsum(corrections))
        terms = []
        for n in range(_IMAGE_TERMS):
            distance = 2 * n + w
            gap = math.log1p(2 * opposite / distance) - 2 * opposite * (2 * n + 1) / t
            terms.append(math.log(distance / w) - 2 * n * (n + w) / t + _log1mexp(gap))
        largest = max(terms)
        return largest + math.log(math.fsum(math.exp(term - largest) for term in terms))
    near = min(w, opposite)
    z = math.pi * near
    sinc = (
        1 - z * z / 6 + z**4 / 120 - z**6 / 5040 + z**8 / 362880
        if near < 0.01
        else math.sin(z) / z
    )
    log_sine_over_w = math.log(math.pi) + math.log(sinc)
    if w > 0.5:
        log_sine_over_w += math.log(near) - math.log(w)
    cosine = math.cos(math.pi * near) * (1 if w <= 0.5 else -1)
    coefficient, previous = 1.0, 0.0
    total = 1.0
    for k in range(2, _SPECTRAL_TERMS + 1):
        previous, coefficient = coefficient, 2 * cosine * coefficient - previous
        total += k * coefficient * math.exp(-(k * k - 1) * math.pi * math.pi * t / 2)
    return (
        math.log(math.pi)
        + log_sine_over_w
        - math.pi * math.pi * t / 2
        + math.log(total)
        + 0.5 * math.log(2 * math.pi)
        + 1.5 * math.log(t)
        + w * w / (2 * t)
    )


def density(a: Mapping[str, object], value: object) -> float:
    """Score a signed reaction time under the joint decision/time law.

    Parameters
    ----------
    a : Mapping[str, object]
        Boundary, nondecision time, relative bias, and drift.
    value : object
        Positive upper-boundary RT or negative lower-boundary RT.

    Returns
    -------
    float
        Joint log density; negative infinity outside signed reaction-time support.

    Raises
    ------
    DistributionError
        If parameters or normalized-time arithmetic are invalid.
    """
    boundary, ndt, bias, drift = _parameters(a)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return -math.inf
    point = float(value)
    if not math.isfinite(point) or abs(point) <= ndt:
        return -math.inf
    decision = abs(point) - ndt
    upper = point > 0
    w, opposite = (1 - bias, bias) if upper else (bias, 1 - bias)
    toward = drift if upper else -drift
    root_time = math.sqrt(decision)
    standardized = (boundary / root_time) * w - toward * root_time
    return (
        math.log(boundary)
        + math.log(w)
        - 0.5 * math.log(2 * math.pi)
        - 1.5 * math.log(decision)
        - standardized * standardized / 2
        + _unit_log_ratio(decision / boundary / boundary, w, opposite)
    )


# Every interval endpoint below is rounded outwards. Decimal exp and sqrt are
# correctly rounded; widening their nearest-rounded result by one representable
# neighbour encloses the exact transcendental value. Pi and sin/cos use bounded
# alternating series, so no external special-function dependency is needed.
type _Interval = tuple[Decimal, Decimal]


class _IntervalArithmetic:
    """Directed decimal operations used only by rejection decisions.

    Parameters
    ----------
    precision : int
        Decimal significant digits in the endpoint computations.
    """

    def __init__(self, precision: int) -> None:
        self.down = Context(prec=precision, rounding=ROUND_FLOOR)
        self.up = Context(prec=precision, rounding=ROUND_CEILING)
        self.near = Context(prec=precision, rounding=ROUND_HALF_EVEN)
        self.precision = precision

    def constant(self, value: int | float | Decimal) -> _Interval:
        """Enclose an exact integer, binary float, or decimal.

        Parameters
        ----------
        value : int or float or Decimal
            Known exact host value.

        Returns
        -------
        tuple of Decimal
            Rounded lower and upper bounds.
        """
        exact = (
            Decimal.from_float(value) if isinstance(value, float) else Decimal(value)
        )
        return self.down.plus(exact), self.up.plus(exact)

    def add(self, a: _Interval, b: _Interval) -> _Interval:
        """Add two intervals with directed rounding.

        Parameters
        ----------
        a, b : tuple of Decimal
            Operand intervals.

        Returns
        -------
        tuple of Decimal
            Sum enclosure.
        """
        return self.down.add(a[0], b[0]), self.up.add(a[1], b[1])

    def sub(self, a: _Interval, b: _Interval) -> _Interval:
        """Subtract intervals.

        Parameters
        ----------
        a, b : tuple of Decimal
            Operand intervals.

        Returns
        -------
        tuple of Decimal
            Difference enclosure.
        """
        return self.down.subtract(a[0], b[1]), self.up.subtract(a[1], b[0])

    def mul(self, a: _Interval, b: _Interval) -> _Interval:
        """Multiply intervals of arbitrary sign.

        Parameters
        ----------
        a, b : tuple of Decimal
            Operand intervals.

        Returns
        -------
        tuple of Decimal
            Product enclosure.
        """
        return min(self.down.multiply(x, y) for x in a for y in b), max(
            self.up.multiply(x, y) for x in a for y in b
        )

    def div(self, a: _Interval, b: _Interval) -> _Interval:
        """Divide by an interval bounded away from zero.

        Parameters
        ----------
        a, b : tuple of Decimal
            Numerator and denominator intervals.

        Returns
        -------
        tuple of Decimal
            Quotient enclosure.

        Raises
        ------
        DistributionError
            If the denominator enclosure includes zero.
        """
        if b[0] <= 0 <= b[1]:
            raise DistributionError(
                "Wiener interval division cannot certify a nonzero denominator"
            )
        return min(self.down.divide(x, y) for x in a for y in b), max(
            self.up.divide(x, y) for x in a for y in b
        )

    def exp(self, a: _Interval) -> _Interval:
        """Enclose the exponential using its correctly rounded decimal value.

        Parameters
        ----------
        a : tuple of Decimal
            Argument enclosure.

        Returns
        -------
        tuple of Decimal
            Exponential enclosure.
        """
        low, high = self.near.exp(a[0]), self.near.exp(a[1])
        return max(Decimal(0), self.near.next_minus(low)), self.near.next_plus(high)

    def sqrt(self, a: _Interval) -> _Interval:
        """Enclose a nonnegative square root.

        Parameters
        ----------
        a : tuple of Decimal
            Nonnegative argument enclosure.

        Returns
        -------
        tuple of Decimal
            Square-root enclosure.
        """
        low, high = self.near.sqrt(a[0]), self.near.sqrt(a[1])
        return max(Decimal(0), self.near.next_minus(low)), self.near.next_plus(high)

    def scale(self, a: _Interval, value: int) -> _Interval:
        """Multiply by an exact integer.

        Parameters
        ----------
        a : tuple of Decimal
            Operand enclosure.
        value : int
            Integer multiplier.

        Returns
        -------
        tuple of Decimal
            Product enclosure.
        """
        return self.mul(a, self.constant(value))

    def exprel(self, a: _Interval) -> _Interval:
        """Enclose expm1(x)/x, including arbitrarily small arguments.

        Parameters
        ----------
        a : tuple of Decimal
            Nonpositive argument enclosure.

        Returns
        -------
        tuple of Decimal
            Divided-exponential enclosure with a rigorous Taylor remainder.
        """
        one = self.constant(1)
        if a[1] < Decimal("-0.5"):
            return self.div(self.sub(self.exp(a), one), a)
        term = total = one
        for n in range(1, self.precision + 8):
            term = self.div(self.mul(term, a), self.constant(n + 1))
            total = self.add(total, term)
        following = self.div(self.mul(term, a), self.constant(self.precision + 9))
        error = max(endpoint.copy_abs() for endpoint in following)
        return self.down.subtract(total[0], error), self.up.add(total[1], error)

    def trigonometric(self, x: _Interval, *, cosine: bool) -> _Interval:
        """Bound sine or cosine on [0, pi/2] by an alternating Taylor series.

        Parameters
        ----------
        x : tuple of Decimal
            Argument contained in the first quadrant.
        cosine : bool
            Select cosine rather than sine.

        Returns
        -------
        tuple of Decimal
            Function enclosure, including the first omitted term.
        """
        square = self.mul(x, x)
        term = self.constant(1) if cosine else x
        total = term
        for n in range(1, self.precision + 8):
            denominator = (2 * n - int(cosine)) * (2 * n + 1 - int(cosine))
            term = self.scale(
                self.div(self.mul(term, square), self.constant(denominator)), -1
            )
            total = self.add(total, term)
        n = self.precision + 8
        denominator = (2 * n - int(cosine)) * (2 * n + 1 - int(cosine))
        following = self.div(self.mul(term, square), self.constant(denominator))
        error = max(endpoint.copy_abs() for endpoint in following)
        return self.down.subtract(total[0], error), self.up.add(total[1], error)


@lru_cache(maxsize=8)
def _pi_interval(precision: int) -> _Interval:
    """Bound pi using Machin's arctangent identity.

    Parameters
    ----------
    precision : int
        Decimal significant digits.

    Returns
    -------
    tuple of Decimal
        Rigorous lower and upper bounds for pi.
    """
    arithmetic = _IntervalArithmetic(precision)

    def arctangent(denominator: int) -> _Interval:
        """Enclose atan(1/denominator) by an alternating series.

        Parameters
        ----------
        denominator : int
            Positive reciprocal argument denominator.

        Returns
        -------
        tuple of Decimal
            Bounded alternating sum.
        """
        term = arithmetic.div(arithmetic.constant(1), arithmetic.constant(denominator))
        total = term
        square = arithmetic.constant(denominator * denominator)
        for n in range(1, precision + 8):
            term = arithmetic.div(term, square)
            addition = arithmetic.div(term, arithmetic.constant(2 * n + 1))
            total = arithmetic.add(
                total, arithmetic.scale(addition, -1 if n % 2 else 1)
            )
        following = arithmetic.div(
            arithmetic.div(term, square), arithmetic.constant(2 * (precision + 8) + 1)
        )
        error = following[1]
        return arithmetic.down.subtract(total[0], error), arithmetic.up.add(
            total[1], error
        )

    return arithmetic.sub(
        arithmetic.scale(arctangent(5), 16), arithmetic.scale(arctangent(239), 4)
    )


@lru_cache(maxsize=256)
def _spectral_geometry(
    bias: float, upper: bool, precision: int
) -> tuple[_Interval, _Interval, _Interval]:
    """Reuse immutable certified boundary geometry across sampler proposals.

    Parameters
    ----------
    bias : float
        Starting fraction.
    upper : bool
        Selected boundary.
    precision : int
        Decimal significant digits.

    Returns
    -------
    tuple of intervals
        Pi, sine, and cosine enclosures for the selected boundary distance.
    """
    a = _IntervalArithmetic(precision)
    start, one = a.constant(bias), a.constant(1)
    w = a.sub(one, start) if upper else start
    opposite = start if upper else a.sub(one, start)
    pi = _pi_interval(precision)
    near = w if w[1] <= Decimal("0.5") else opposite
    angle = a.mul(pi, near)
    sine = a.trigonometric(angle, cosine=False)
    cosine = a.trigonometric(angle, cosine=True)
    if w[0] > Decimal("0.5"):
        cosine = a.scale(cosine, -1)
    return pi, sine, cosine


def _ratio_interval(
    t: float, bias: float, upper: bool, terms: int, precision: int
) -> _Interval:
    """Certify the drift-free two-boundary/one-boundary density ratio.

    Parameters
    ----------
    t : float
        Unit-boundary decision time.
    bias : float
        Original starting fraction.
    upper : bool
        Whether the proposed first hit is at the upper boundary.
    terms : int
        Number of image pairs or spectral terms.
    precision : int
        Decimal significant digits.

    Returns
    -------
    tuple of Decimal
        Outward-rounded probability bounds including analytical series tails.

    Raises
    ------
    DistributionError
        If the supplied precision cannot certify valid interval operations.
    """
    a = _IntervalArithmetic(precision)
    zero, one, two = a.constant(0), a.constant(1), a.constant(2)
    time, start = a.constant(t), a.constant(bias)
    w = a.sub(one, start) if upper else start
    opposite = start if upper else a.sub(one, start)
    if t <= _SMALL_TIME:
        total = zero
        if w[1] <= Decimal("0.5"):
            for n in range(1, terms + 1):
                exponent = a.div(a.scale(a.sub(a.constant(n), w), -2 * n), time)
                x = a.div(a.scale(w, -4 * n), time)
                bracket = a.sub(
                    a.sub(a.mul(a.div(a.constant(8 * n * n), time), a.exprel(x)), one),
                    a.exp(x),
                )
                total = a.add(total, a.mul(a.exp(exponent), bracket))
            n = terms + 1
            leading = a.mul(
                a.add(a.div(a.constant(8 * n * n), time), two),
                a.exp(a.div(a.scale(a.sub(a.constant(n), w), -2 * n), time)),
            )
            ratio = a.mul(
                a.div(a.constant((n + 1) ** 2), a.constant(n * n)),
                a.exp(a.div(a.scale(a.sub(a.constant(2 * n + 1), w), -2), time)),
            )
            tail = a.div(leading, a.sub(one, ratio))[1]
            result = a.sub(one, total)
            low, high = a.down.subtract(result[0], tail), result[1]
        else:
            for n in range(terms):
                distance = a.add(a.constant(2 * n), w)
                exponent = a.div(a.scale(a.add(a.constant(n), w), -2 * n), time)
                r = a.div(a.scale(opposite, 2 * (2 * n + 1)), time)
                q = a.div(a.scale(opposite, 2), distance)
                negative_r = a.scale(r, -1)
                complement = a.sub(
                    a.mul(r, a.exprel(negative_r)), a.mul(q, a.exp(negative_r))
                )
                total = a.add(
                    total, a.mul(a.mul(a.div(distance, w), a.exp(exponent)), complement)
                )
            n = terms
            distance = a.add(a.constant(2 * n), w)
            leading = a.mul(
                a.div(distance, w),
                a.exp(a.div(a.scale(a.add(a.constant(n), w), -2 * n), time)),
            )
            ratio = a.mul(
                a.div(a.add(distance, two), distance),
                a.exp(a.div(a.scale(a.add(a.constant(2 * n + 1), w), -2), time)),
            )
            tail = a.div(leading, a.sub(one, ratio))[1]
            low, high = total[0], a.up.add(total[1], tail)
    else:
        pi, sine, cosine = _spectral_geometry(bias, upper, precision)
        c = a.div(a.mul(pi, pi), two)
        previous, coefficient, total = zero, one, one
        for k in range(2, terms + 1):
            previous, coefficient = (
                coefficient,
                a.sub(a.scale(a.mul(cosine, coefficient), 2), previous),
            )
            total = a.add(
                total,
                a.scale(
                    a.mul(coefficient, a.exp(a.scale(a.mul(c, time), -(k * k - 1)))), k
                ),
            )
        n = terms + 1
        leading = a.scale(a.exp(a.scale(a.mul(c, time), -(n * n - 1))), n * n)
        ratio = a.mul(
            a.div(a.constant((n + 1) ** 2), a.constant(n * n)),
            a.exp(a.scale(a.mul(c, time), -(2 * n + 1))),
        )
        tail = a.div(leading, a.sub(one, ratio))[1]
        total = (a.down.subtract(total[0], tail), a.up.add(total[1], tail))
        exponent = a.sub(a.div(a.mul(w, w), a.scale(time, 2)), a.mul(c, time))
        prefactor = a.mul(
            a.div(a.mul(pi, sine), w),
            a.mul(
                a.sqrt(a.mul(a.scale(pi, 2), a.mul(a.mul(time, time), time))),
                a.exp(exponent),
            ),
        )
        low, high = a.mul(prefactor, total)
    low, high = max(Decimal(0), low), min(Decimal(1), high)
    if low > high:
        raise DistributionError(
            "Wiener interval precision cannot certify a probability enclosure"
        )
    return low, high


def _accept(
    t: float,
    bias: float,
    upper: bool,
    uniform: float,
    *,
    max_terms: int,
    max_precision: int,
) -> bool:
    """Resolve a rejection decision using certified interval refinement.

    Parameters
    ----------
    t, bias : float
        Unit-boundary proposal time and original starting fraction.
    upper : bool
        Proposed boundary.
    uniform : float
        Independent acceptance uniform.
    max_terms, max_precision : int
        Maximum series terms and decimal significant digits.

    Returns
    -------
    bool
        True exactly when the uniform is certified below the density ratio.

    Raises
    ------
    DistributionError
        If finite refinement budgets cannot resolve the comparison.
    """
    if uniform == 0:
        return True  # The interior first-passage density is strictly positive.
    threshold = Decimal.from_float(uniform)
    terms, precision = min(4, max_terms), min(48, max_precision)
    while True:
        try:
            low, high = _ratio_interval(t, bias, upper, terms, precision)
        except DistributionError, DecimalException:
            # Coarse interval arithmetic may enclose zero in a denominator;
            # this is an unresolved certificate, never a rejection decision.
            low, high = Decimal(0), Decimal(1)
        if threshold < low:
            return True
        if threshold > high:
            return False
        if terms == max_terms and precision == max_precision:
            raise DistributionError(
                "Wiener rejection decision exhausted certified series/precision budgets"
            )
        terms, precision = min(max_terms, 2 * terms), min(max_precision, 2 * precision)


def _draw(
    boundary: float,
    ndt: float,
    bias: float,
    drift: float,
    rng: _RandomSource,
    *,
    max_proposals: int,
    max_terms: int,
    max_precision: int,
) -> float:
    """Sample a validated signed reaction time by first-passage domination.

    Parameters
    ----------
    boundary, ndt, bias, drift : float
        Validated diffusion parameters.
    rng : _RandomSource
        Uniform and Gaussian random source.
    max_proposals, max_terms, max_precision : int
        Explicit proposal, series, and interval-precision budgets.

    Returns
    -------
    float
        Signed reaction time from the joint law.

    Raises
    ------
    DistributionError
        If sampling or floating-point capacity is exhausted.
    """
    scaled_drift = drift * boundary
    if not math.isfinite(scaled_drift):
        raise DistributionError("Wiener scaled drift exceeds floating-point capacity")
    magnitude = abs(scaled_drift)
    upper_mass = math.exp(min(0.0, 2 * scaled_drift * (1 - bias)))
    lower_mass = math.exp(min(0.0, -2 * scaled_drift * bias))
    upper_probability = upper_mass / (upper_mass + lower_mass)
    for _ in range(max_proposals):
        upper = rng.random() < upper_probability
        distance = 1 - bias if upper else bias
        normal = rng.gauss(0.0, 1.0)
        square = normal * normal
        b = magnitude * distance
        denominator = (
            b + square / 2 + math.hypot(square / 2, abs(normal) * math.sqrt(b))
        )
        if denominator <= 0 or not math.isfinite(denominator):
            raise DistributionError("Wiener proposal exceeds floating-point capacity")
        proposal = distance * distance / denominator
        if magnitude and rng.random() > 1 / (1 + magnitude * proposal / distance):
            mean = distance / magnitude
            proposal = mean * (mean / proposal)
        if not math.isfinite(proposal) or proposal <= 0:
            raise DistributionError("Wiener proposal exceeds floating-point capacity")
        if _accept(
            proposal,
            bias,
            upper,
            rng.random(),
            max_terms=max_terms,
            max_precision=max_precision,
        ):
            result = ndt + (proposal * boundary) * boundary
            if not math.isfinite(result) or result <= ndt:
                raise DistributionError(
                    "Wiener sampled time exceeds representable reaction-time capacity"
                )
            return result if upper else -result
    raise DistributionError(
        "Wiener sampler exhausted its exact rejection proposal budget"
    )


def sample(
    a: Mapping[str, object],
    rng: _RandomSource,
    *,
    max_proposals: int = 10000,
    max_terms: int = 64,
    max_precision: int = 192,
) -> float:
    """Draw a joint choice and reaction time without time discretization.

    Parameters
    ----------
    a : Mapping[str, object]
        Boundary, nondecision time, bias, and drift parameters.
    rng : _RandomSource
        Uniform/Gaussian random generator, such as random.Random.
    max_proposals : int
        Rejection attempts before explicit failure.
    max_terms : int
        Image pairs or spectral terms in an acceptance certificate.
    max_precision : int
        Decimal significant digits in an acceptance certificate.

    Returns
    -------
    float
        Signed joint reaction time.

    Raises
    ------
    DistributionError
        If parameters or budgets are invalid, or certification fails.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (max_proposals, max_terms, max_precision)
    ):
        raise DistributionError("Wiener numerical budgets must be positive integers")
    return _draw(
        *_parameters(a),
        rng,
        max_proposals=max_proposals,
        max_terms=max_terms,
        max_precision=max_precision,
    )
