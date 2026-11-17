"""Joint first-passage time and decision law for a unit-variance Wiener process.

Positive signed reaction times denote upper-boundary decisions; negative times
lower-boundary decisions. Scores retain the decision probability. The image
and spectral expansions follow Navarro and Fuss (2009),
https://doi.org/10.1016/j.jmp.2009.02.003. Sampling uses certified rejection
from one-boundary hitting times, with no time discretization.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from quivers.continuous._ordinal import _log1mexp

from quivers.continuous._zip_hurdle import _OpenUnitInterval
from quivers.qiec._wiener import _draw


_SMALL_TIME = 0.125
_IMAGE_TERMS = 4
_SPECTRAL_TERMS = 12


def _exprel(x: Tensor) -> Tensor:
    """Evaluate the smooth divided exponential for nonpositive arguments.

    Parameters
    ----------
    x : Tensor
        Nonpositive values.

    Returns
    -------
    Tensor
        expm1(x)/x, with its analytic limit and derivatives at zero.
    """
    small = x.abs() < 0.01
    z = torch.where(small, x, 0.0)
    polynomial = torch.full_like(x, 1 / math.factorial(12))
    for power in range(10, -1, -1):
        polynomial = 1 / math.factorial(power + 1) + z * polynomial
    regular = torch.where(small, -1.0, x)
    return torch.where(small, polynomial, regular.expm1() / regular)


def _log_ratio(time: Tensor, distance: Tensor, opposite: Tensor) -> Tensor:
    """Log the drift-independent two-boundary/one-boundary density ratio.

    Parameters
    ----------
    time : Tensor
        Positive unit-boundary decision times.
    distance, opposite : Tensor
        Selected and opposing distances as fractions of boundary separation.

    Returns
    -------
    Tensor
        Log ratios from fixed expansions with bounded omitted tails. The
        four image pairs and twelve spectral terms give absolute relative
        kernel errors below 1e-34 at the regime boundary; arithmetic rounding
        may dominate this truncation error.
    """
    small = time <= _SMALL_TIME
    close = distance <= 0.5
    # Mask before division/exponentiation so inactive regimes cannot inject
    # NaNs into reverse- or forward-mode derivatives.
    t = torch.where(small & close, time, 0.1)
    w = torch.where(small & close, distance, 0.25)
    corrections = torch.zeros_like(t)
    for n in range(1, _IMAGE_TERMS + 1):
        significant = t > 2 * n * (n - w) / 700
        ti = torch.where(significant, t, 0.1)
        wi = torch.where(significant, w, 0.25)
        x = -4 * n * wi / ti
        correction = (-2 * n * (n - wi) / ti).exp() * (
            (8 * n * n / ti) * _exprel(x) - 1 - x.exp()
        )
        corrections = corrections + torch.where(significant, correction, 0.0)
    close_result = torch.log1p(-corrections)

    t = torch.where(small & ~close, time, 0.1)
    w = torch.where(small & ~close, distance, 0.75)
    c = torch.where(small & ~close, opposite, 0.25)
    pairs = []
    for n in range(_IMAGE_TERMS):
        p = 2 * n + w
        log_difference = torch.log1p(2 * c / p) - 2 * c * (2 * n + 1) / t
        pairs.append((p / w).log() - 2 * n * (n + w) / t + _log1mexp(log_difference))
    far_result = torch.logsumexp(torch.stack(pairs), dim=0)

    t = torch.where(small, 0.25, time)
    w = torch.where(small, 0.5, distance)
    c = torch.where(small, 0.5, opposite)
    near = torch.where(w <= 0.5, w, c)
    z = math.pi * near
    tiny = near < 0.01
    safe_z = torch.where(tiny, 1.0, z)
    z2 = torch.where(tiny, z, 0.0).square()
    sinc_series = 1 + z2 * (-1 / 6 + z2 * (1 / 120 + z2 * (-1 / 5040 + z2 / 362880)))
    log_sinc = torch.where(tiny, sinc_series, safe_z.sin() / safe_z).log()
    log_sine_over_w = (
        math.log(math.pi) + log_sinc + torch.where(w <= 0.5, 0.0, c.log() - w.log())
    )
    cosine = (math.pi * near).cos() * torch.where(w <= 0.5, 1.0, -1.0)
    previous = torch.zeros_like(t)
    coefficient = total = torch.ones_like(t)
    for k in range(2, _SPECTRAL_TERMS + 1):
        previous, coefficient = coefficient, 2 * cosine * coefficient - previous
        total = total + k * coefficient * (-(k * k - 1) * math.pi**2 * t / 2).exp()
    spectral = (
        math.log(math.pi)
        + log_sine_over_w
        + 0.5 * math.log(2 * math.pi)
        + 1.5 * t.log()
        + w.square() / (2 * t)
        - math.pi**2 * t / 2
        + total.log()
    )
    return torch.where(small, torch.where(close, close_result, far_result), spectral)


# Coefficients of z*coth(z), from Bernoulli numbers B_2 through B_24.
# The nearest singularities are at +/-i*pi; the small-drift branch is |z|<=.5.
_COTH_COEFFICIENTS = tuple(
    2 ** (2 * n) * numerator / denominator / math.factorial(2 * n)
    for n, (numerator, denominator) in enumerate(
        [
            (1, 6),
            (-1, 30),
            (1, 42),
            (-1, 30),
            (5, 66),
            (-691, 2730),
            (7, 6),
            (-3617, 510),
            (43867, 798),
            (-174611, 330),
            (854513, 138),
            (-236364091, 2730),
        ],
        start=1,
    )
)


def _xcoth(x: Tensor) -> Tensor:
    """Evaluate x*coth(x) through its analytic zero limit.

    Parameters
    ----------
    x : Tensor
        Nonnegative arguments.

    Returns
    -------
    Tensor
        Stable hyperbolic product and its derivatives.
    """
    small = x <= 0.5
    z = torch.where(small, x, 0.0).square()
    polynomial = torch.full_like(x, _COTH_COEFFICIENTS[-1])
    for coefficient in reversed(_COTH_COEFFICIENTS[:-1]):
        polynomial = coefficient + z * polynomial
    near = 1 + z * polynomial
    regular = torch.where(small, 1.0, x)
    e = (-2 * regular).exp()
    far = regular * (1 + e) / (-torch.expm1(-2 * regular))
    return torch.where(small, near, far)


def _conditional_moments(
    boundary: Tensor, signed_drift: Tensor, r: Tensor, opposite: Tensor
) -> tuple[Tensor, Tensor]:
    """Differentiate the joint first-passage Laplace transform analytically.

    Parameters
    ----------
    boundary : Tensor
        Positive boundary separation.
    signed_drift : Tensor
        Drift times boundary separation.
    r, opposite : Tensor
        Distance from the opposing endpoint and its complement, retained
        separately to preserve fractions near zero and one.

    Returns
    -------
    tuple of Tensor
        Conditional mean and variance of decision time given this endpoint.
        The underlying transform is sinh(r*q)/sinh(q), q=sqrt(v**2+2*s),
        times a drift tilt independent of the Laplace argument s.
    """
    small = signed_drift.abs() <= 0.5
    # Using signed_drift**2 directly preserves second derivatives at zero;
    # composing abs with square would give the wrong AD Hessian there.
    z = torch.where(small, signed_drift, 0.0).square()
    geometric = 1 + r
    power = r.square()
    mean_coefficients = []
    variance_coefficients = []
    for n, coefficient in enumerate(_COTH_COEFFICIENTS, start=1):
        if n > 1:
            geometric = geometric + power * (1 + r)
            power = power * r.square()
        mean_coefficients.append(coefficient * geometric)
        if n > 1:
            variance_coefficients.append(-(2 * n - 2) * coefficient * geometric)
    mean_series = mean_coefficients[-1]
    for coefficient in reversed(mean_coefficients[:-1]):
        mean_series = coefficient + z * mean_series
    variance_series = variance_coefficients[-1]
    for coefficient in reversed(variance_coefficients[:-1]):
        variance_series = coefficient + z * variance_series
    mean_series = ((opposite * mean_series) * boundary) * boundary
    variance_series = (
        (((opposite * variance_series) * boundary) * boundary) * boundary
    ) * boundary

    u = torch.where(small, 1.0, signed_drift.abs())
    e = (-2 * u).exp()
    denominator = -torch.expm1(-2 * u)
    coth = (1 + e) / denominator
    csch2 = 4 * e / denominator.square()
    active = r * u < 350
    ub = torch.where(active, u, 1.0)
    rb = torch.where(active, r, 0.5)
    cb = torch.where(active, opposite, 0.5)
    b = (
        (-2 * rb * ub).exp()
        * (-torch.expm1(-2 * cb * ub))
        / (ub * (-torch.expm1(-2 * ub)) * _exprel(-2 * rb * ub))
    )
    b = torch.where(active, b, 0.0)
    h = opposite * coth - b
    scaled_derivative = _xcoth(cb * ub) - _xcoth(ub) - _xcoth(rb * ub)
    variance_numerator = opposite * (coth + u * csch2) + b * (scaled_derivative - 1)
    inverse_drift = boundary / u
    mean_regular = (h * inverse_drift) * boundary
    variance_regular = (
        ((variance_numerator * inverse_drift) * inverse_drift) * inverse_drift
    ) * boundary
    return torch.where(small, mean_series, mean_regular), torch.where(
        small, variance_series, variance_regular
    )


def _choice_probabilities(signed_drift: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Compute lower and upper hit probabilities without rare-choice subtraction.

    Parameters
    ----------
    signed_drift : Tensor
        Drift times boundary separation.
    bias : Tensor
        Starting fraction strictly inside the unit interval.

    Returns
    -------
    tuple of Tensor
        Lower- and upper-boundary probabilities, including analytic drift-zero
        values and derivatives.
    """
    small = signed_drift.abs() <= 0.5
    v = torch.where(small, signed_drift, 0.0)
    upper_small = bias * _exprel(-2 * v * bias) / _exprel(-2 * v)
    lower_small = (1 - bias) * _exprel(2 * v * (1 - bias)) / _exprel(2 * v)
    u = torch.where(small, 1.0, signed_drift.abs())
    denominator = -torch.expm1(-2 * u)
    upper = (-torch.expm1(-2 * u * bias)) / denominator
    lower = (-torch.expm1(-2 * u * (1 - bias))) / denominator
    upper = upper * torch.where(signed_drift >= 0, 1.0, (-2 * u * (1 - bias)).exp())
    lower = lower * torch.where(signed_drift <= 0, 1.0, (-2 * u * bias).exp())
    return torch.where(small, lower_small, lower), torch.where(
        small, upper_small, upper
    )


class _SignedSupport(constraints.Constraint):
    """Reaction times whose absolute magnitude exceeds their nondecision delay.

    Parameters
    ----------
    ndt : Tensor
        Nonnegative nondecision times.
    """

    def __init__(self, ndt: Tensor) -> None:
        self.ndt = ndt

    def check(self, value: Tensor) -> Tensor:
        """Check signed reaction-time support.

        Parameters
        ----------
        value : Tensor
            Candidate signed reaction times.

        Returns
        -------
        Tensor
            Boolean validity mask.
        """
        return torch.isfinite(value) & (value.abs() > self.ndt)


class _TorchRandom:
    """Buffered CPU random primitives owned by Torch's seeded random stream."""

    def __init__(self) -> None:
        self.uniforms: list[float] = []
        self.normals: list[float] = []

    def random(self) -> float:
        """Return a Torch uniform variate.

        Returns
        -------
        float
            Uniform value in [0, 1).
        """
        if not self.uniforms:
            self.uniforms = torch.rand(256, dtype=torch.float64, device="cpu").tolist()
        return self.uniforms.pop()

    def gauss(self, mu: float, sigma: float) -> float:
        """Return a Torch Gaussian variate.

        Parameters
        ----------
        mu, sigma : float
            Location and scale.

        Returns
        -------
        float
            Gaussian value.
        """
        if not self.normals:
            self.normals = torch.randn(256, dtype=torch.float64, device="cpu").tolist()
        return mu + sigma * self.normals.pop()


class Wiener(Distribution):
    """Joint signed response-time law for two absorbing diffusion boundaries.

    Parameters
    ----------
    boundary : Tensor or float
        Positive boundary separation, with Brownian variance fixed to one.
    ndt : Tensor or float
        Nonnegative delay added to decision time.
    bias : Tensor or float
        Starting position as a fraction of separation, strictly between zero
        and one.
    drift : Tensor or float
        Finite drift towards the upper boundary.
    validate_args : bool or None
        Whether off-support observations raise rather than score as -infinity.
        Parameter validation is always performed.
    max_proposals, max_terms, max_precision : int
        Finite rejection, series, and decimal precision budgets. Exhaustion
        raises; the sampler never substitutes an approximate draw.

    Notes
    -----
    Positive observations encode upper hits and negative observations lower
    hits. The likelihood is joint in time and decision, without conditioning
    on the chosen boundary. Scores and samples use Float64; input gradients
    propagate to the original parameter dtype. Sampling is not reparameterized.
    Its exact-arithmetic construction has acceptance probability at least 1/2;
    numerical acceptance decisions use outward-rounded series enclosures.
    """

    has_rsample = False

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Return the domains of diffusion parameters.

        Returns
        -------
        dict of str to Constraint
            Positive separation, nonnegative delay, open-unit bias, real drift.
        """
        return {
            "boundary": constraints.positive,
            "ndt": constraints.nonnegative,
            "bias": _OpenUnitInterval(),
            "drift": constraints.real,
        }

    def __init__(
        self,
        boundary: Tensor | float,
        ndt: Tensor | float,
        bias: Tensor | float,
        drift: Tensor | float,
        validate_args: bool | None = False,
        *,
        max_proposals: int = 10000,
        max_terms: int = 64,
        max_precision: int = 192,
    ) -> None:
        values = (boundary, ndt, bias, drift)
        if any(
            isinstance(v, bool)
            or isinstance(v, Tensor)
            and (v.dtype == torch.bool or v.is_complex())
            for v in values
        ):
            raise ValueError("Wiener parameters must be real and nonboolean")
        device = next(
            (v.device for v in values if isinstance(v, Tensor)), torch.device("cpu")
        )
        tensors = broadcast_all(
            *(torch.as_tensor(v, dtype=torch.float64, device=device) for v in values)
        )
        self.boundary, self.ndt, self.bias, self.drift = tensors
        if not all(bool(torch.isfinite(v).all()) for v in tensors):
            raise ValueError("Wiener parameters must be finite")
        if not bool(
            (
                (self.boundary > 0)
                & (self.ndt >= 0)
                & (self.bias > 0)
                & (self.bias < 1)
            ).all()
        ):
            raise ValueError("Wiener requires boundary>0, ndt>=0, and 0<bias<1")
        budgets = (max_proposals, max_terms, max_precision)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in budgets):
            raise ValueError("Wiener numerical budgets must be positive integers")
        self.max_proposals, self.max_terms, self.max_precision = budgets
        super().__init__(self.boundary.shape, torch.Size(), validate_args=validate_args)

    @property
    def support(self) -> _SignedSupport:
        """Return the signed reaction-time constraint.

        Returns
        -------
        _SignedSupport
            Absolute reaction time strictly above nondecision delay.
        """
        return _SignedSupport(self.ndt)

    def log_prob(self, value: Tensor | float) -> Tensor:
        """Score a signed time without losing the probability of its decision.

        Parameters
        ----------
        value : Tensor or float
            Signed reaction time, broadcastable with parameter batch axes.

        Returns
        -------
        Tensor
            Float64 joint log density, negative infinity outside support when
            observation validation is disabled.

        Raises
        ------
        ValueError
            If observations are boolean or complex, or normalized times exceed
            floating-point arithmetic capacity.
        """
        if (
            isinstance(value, bool)
            or isinstance(value, Tensor)
            and (value.dtype == torch.bool or value.is_complex())
        ):
            raise ValueError("Wiener observations must be real and nonboolean")
        point = torch.as_tensor(value, dtype=torch.float64, device=self.boundary.device)
        if self._validate_args:
            self._validate_sample(point)
        valid = self.support.check(point)
        decision = torch.where(valid, point.abs() - self.ndt, self.boundary.square())
        time = decision / self.boundary / self.boundary
        if not bool((torch.isfinite(time) & (time > 0)).all()):
            raise ValueError("Wiener normalized time exceeds floating-point capacity")
        upper = point > 0
        w = torch.where(upper, 1 - self.bias, self.bias)
        c = torch.where(upper, self.bias, 1 - self.bias)
        toward = torch.where(upper, self.drift, -self.drift)
        log_distance = self.boundary.log() + w.log()
        # Completing the square avoids cancellation between the drift tilt
        # and the very small-time zero-drift kernel for large drifts.
        standardized = (self.boundary / decision.sqrt()) * w - toward * decision.sqrt()
        result = (
            log_distance
            - 0.5 * math.log(2 * math.pi)
            - 1.5 * decision.log()
            - standardized.square() / 2
            + _log_ratio(time, w, c)
        )
        return torch.where(valid, result, -torch.inf)

    def _choice_moments(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return probabilities and conditional decision-time moments.

        Returns
        -------
        tuple of Tensor
            Lower/upper probabilities, means, and variances respectively.

        Raises
        ------
        ValueError
            If drift times boundary exceeds floating-point capacity.
        """
        scaled = self.boundary * self.drift
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError("Wiener scaled drift exceeds floating-point capacity")
        lower, upper = _choice_probabilities(scaled, self.bias)
        lower_mean, lower_var = _conditional_moments(
            self.boundary, scaled, 1 - self.bias, self.bias
        )
        upper_mean, upper_var = _conditional_moments(
            self.boundary, scaled, self.bias, 1 - self.bias
        )
        return lower, upper, lower_mean, upper_mean, lower_var, upper_var

    @property
    def choice_probs(self) -> Tensor:
        """Return lower/upper boundary probabilities along a final length-two axis.

        Returns
        -------
        Tensor
            Choice probabilities with axes batch x (lower, upper).
        """
        lower, upper = _choice_probabilities(self.boundary * self.drift, self.bias)
        return torch.stack((lower, upper), dim=-1)

    @property
    def mean(self) -> Tensor:
        """Return the mean signed reaction time.

        Returns
        -------
        Tensor
            Positive upper-choice and negative lower-choice joint mean.
        """
        lower, upper, ml, mu, _, _ = self._choice_moments()
        return upper * (self.ndt + mu) - lower * (self.ndt + ml)

    @property
    def variance(self) -> Tensor:
        """Return signed reaction-time variance using a stable mixture formula.

        Returns
        -------
        Tensor
            Variance of signed joint reaction times.
        """
        lower, upper, ml, mu, vl, vu = self._choice_moments()
        return (
            lower * vl + upper * vu + lower * upper * (2 * self.ndt + ml + mu).square()
        )

    @property
    def mean_response_time(self) -> Tensor:
        """Return mean absolute reaction time, including nondecision delay.

        Returns
        -------
        Tensor
            Expected unsigned response time.
        """
        lower, upper, ml, mu, _, _ = self._choice_moments()
        return self.ndt + lower * ml + upper * mu

    @property
    def variance_response_time(self) -> Tensor:
        """Return unsigned reaction-time variance.

        Returns
        -------
        Tensor
            Variance of the magnitude of signed joint samples.
        """
        lower, upper, ml, mu, vl, vu = self._choice_moments()
        return lower * vl + upper * vu + lower * upper * (mu - ml).square()

    @torch.no_grad()
    def sample(self, sample_shape: Sequence[int] = torch.Size()) -> Tensor:
        """Draw signed reaction times using certified first-passage rejection.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sample dimensions.

        Returns
        -------
        Tensor
            Float64 signed times, with sample and parameter-batch dimensions.
        """
        shape = self._extended_shape(torch.Size(sample_shape))
        output = torch.empty(shape, dtype=torch.float64, device="cpu")
        if output.numel() == 0:
            return output.to(self.boundary.device)
        parameters = [
            v.detach().cpu().reshape(-1).tolist()
            for v in (self.boundary, self.ndt, self.bias, self.drift)
        ]
        rows = [
            (parameters[0][i], parameters[1][i], parameters[2][i], parameters[3][i])
            for i in range(self.boundary.numel())
        ]
        rng = _TorchRandom()
        flat = output.reshape(-1)
        for i in range(flat.numel()):
            flat[i] = _draw(
                *rows[i % len(rows)],
                rng,
                max_proposals=self.max_proposals,
                max_terms=self.max_terms,
                max_precision=self.max_precision,
            )
        return output.to(self.boundary.device)

    def expand(
        self, batch_shape: torch.Size | Sequence[int], _instance: Wiener | None = None
    ) -> Wiener:
        """Expand diffusion parameter views without copying data.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            New batch dimensions.
        _instance : Wiener or None
            Optional preallocated distribution instance.

        Returns
        -------
        Wiener
            Distribution with broadcast parameter views and unchanged budgets.
        """
        new = self._get_checked_instance(Wiener, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in ("boundary", "ndt", "bias", "drift"):
            setattr(new, name, getattr(self, name).expand(batch_shape))
        new.max_proposals, new.max_terms, new.max_precision = (
            self.max_proposals,
            self.max_terms,
            self.max_precision,
        )
        super(Wiener, new).__init__(batch_shape, torch.Size(), validate_args=False)
        new._validate_args = self._validate_args
        return new


__all__ = ["Wiener"]
