"""Independent scalar reference laws for zero-inflated and hurdle counts."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import math
import random

from quivers.qiec.reference_support import (
    DistributionError,
    real_parameter,
    _integer_observation,
)
from quivers.qiec._sampling import _binomial_sample, _dirichlet_weights

NAMES = (
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedBinomial",
    "ZeroInflatedBetaBinomial",
    "HurdleNegativeBinomial",
)


def _require_sample_capacity(value: float | int) -> float | int:
    """Refuse host sampling outside the native exact-integer capacity.

    Parameters
    ----------
    value : float or int
        Generated intensity or count.

    Returns
    -------
    float or int
        The unchanged admissible value.

    Raises
    ------
    DistributionError
        If the sampler would need a negative, nonfinite, or greater-than-2**53 value.
    """
    if not math.isfinite(value) or not 0 <= value <= 2**53:
        raise DistributionError(
            "count sampling exceeds the exact float64 integer capacity 2**53"
        )
    return value


def _stirling_error(x: float) -> float:
    """Evaluate the scalar log-factorial Stirling correction.

    Parameters
    ----------
    x : float
        Strictly positive real count.

    Returns
    -------
    float
        Log-gamma correction, with an asymptotic expansion above fifteen.
    """
    if x <= 15:
        return math.lgamma(x + 1) - (x + 0.5) * math.log(x) + x - 0.91893853320467274178
    inverse = 1 / x
    square = inverse * inverse
    return inverse * (
        1 / 12
        - square * (1 / 360 - square * (1 / 1260 - square * (1 / 1680 - square / 1188)))
    )


def _deviance(x: float, log_expected: float) -> float:
    """Evaluate scalar Poisson deviance stably near its minimum.

    Parameters
    ----------
    x : float
        Positive observed count.
    log_expected : float
        Logarithm of the positive expected count.

    Returns
    -------
    float
        The nonnegative Poisson deviance term.
    """
    expected = math.exp(log_expected)
    difference = x - expected
    if abs(difference) >= 0.1 * (x + expected):
        return x * (math.log(x) - log_expected) + expected - x
    v = difference / (x + expected)
    square = v * v
    polynomial = 1 / 21
    for denominator in range(19, 2, -2):
        polynomial = 1 / denominator + square * polynomial
    return difference * v + 2 * x * v * square * polynomial


def _log_gamma_increment(x: float, shift: float) -> float:
    """Evaluate a gamma increment without extensive cancellation.

    Parameters
    ----------
    x : float
        Positive gamma argument.
    shift : float
        Increment with positive shifted argument.

    Returns
    -------
    float
        Log-gamma difference.
    """
    if x > 16 and x + shift > 16:
        return (
            shift * math.log(x)
            + (x + shift - 0.5) * math.log1p(shift / x)
            - shift
            + _stirling_error(x + shift)
            - _stirling_error(x)
        )
    return math.lgamma(x + shift) - math.lgamma(x)


def _poisson_log_mass(count: int, rate: float) -> float:
    """Score a Poisson count without cancellation in large factorials.

    Parameters
    ----------
    count : int
        Integer count candidate.
    rate : float
        Nonnegative finite Poisson intensity.

    Returns
    -------
    float
        Stable log mass, including zero-rate and outside-support cases.

    Raises
    ------
    DistributionError
        If the rate is negative or nonfinite.
    """
    if not math.isfinite(rate) or rate < 0:
        raise DistributionError("Poisson rate must be finite and nonnegative")
    if count < 0:
        return -math.inf
    if rate == 0:
        return 0.0 if count == 0 else -math.inf
    if count == 0:
        return -rate
    return (
        -_stirling_error(float(count))
        - 0.5 * (1.83787706640934548356 + math.log(count))
        - _deviance(float(count), math.log(rate))
    )


def _binomial_log_mass(n: float, k: int, log_p: float, log_q: float) -> float:
    """Evaluate binomial masses from log probabilities.

    Parameters
    ----------
    n : float
        Nonnegative integer trial count.
    k : int
        Observation count.
    log_p : float
        Log success probability, possibly negative infinity.
    log_q : float
        Log failure probability, possibly negative infinity.

    Returns
    -------
    float
        Stable log mass, including boundary atoms.
    """
    if not 0 <= k <= n:
        return -math.inf
    if n == 0:
        return 0.0
    if k == 0:
        return n * log_q
    if k == n:
        return n * log_p
    if not math.isfinite(log_p) or not math.isfinite(log_q):
        return -math.inf
    a, b = float(k), n - k
    return (
        _stirling_error(n)
        - _stirling_error(a)
        - _stirling_error(b)
        - _deviance(a, math.log(n) + log_p)
        - _deviance(b, math.log(n) + log_q)
        - 0.5 * (1.83787706640934548356 + math.log(a) + math.log(b) - math.log(n))
    )


def _negative_binomial_log_mass(r: float, k: int, log_p: float, log_q: float) -> float:
    """Evaluate negative-binomial masses without large gamma subtraction.

    Parameters
    ----------
    r : float
        Nonnegative real shape.
    k : int
        Observation count.
    log_p : float
        Log success probability.
    log_q : float
        Log failure probability.

    Returns
    -------
    float
        Stable log mass, including degenerate zero counts.
    """
    if r == 0:
        return 0.0 if k == 0 else -math.inf
    if k == 0:
        return r * log_q
    if k < 0 or not math.isfinite(log_p):
        return -math.inf
    log_k, log_r = math.log(k), math.log(r)
    larger, smaller = max(log_k, log_r), min(log_k, log_r)
    log_total = larger + math.log1p(math.exp(smaller - larger))
    return (
        _stirling_error(k + r)
        - _stirling_error(float(k))
        - _stirling_error(r)
        - _deviance(float(k), log_total + log_p)
        - _deviance(r, log_total + log_q)
        - 0.5 * (1.83787706640934548356 + log_k + log_total - log_r)
    )


def _validate_beta_concentrations(a: float, b: float) -> None:
    """Validate scalar Beta count parameters before any degenerate shortcut.

    Parameters
    ----------
    a, b : float
        Success and failure concentrations.

    Raises
    ------
    DistributionError
        If concentrations or their sum are not positive and finite.
    """
    if not (
        math.isfinite(a)
        and math.isfinite(b)
        and a > 0
        and b > 0
        and math.isfinite(a + b)
    ):
        raise DistributionError(
            "beta-binomial concentrations and their sum must be finite and positive"
        )


def _beta_binomial_log_mass(n: float, k: int, a: float, b: float) -> float:
    """Evaluate beta-binomial factors without extensive gamma subtraction.

    Parameters
    ----------
    k, n : int
        Counts and trials within support.
    a, b : float
        Positive beta concentrations.

    Returns
    -------
    float
        Stable normalized log mass.

    Raises
    ------
    DistributionError
        If concentrations or their sum exceed positive finite capacity.
    """
    _validate_beta_concentrations(a, b)
    if not 0 <= k <= n:
        return -math.inf
    if n == 0:
        return 0.0
    if k == 0:
        return (
            _log_gamma_increment(b, a) - _log_gamma_increment(b + n, a)
            if a < n
            else _log_gamma_increment(b, n) - _log_gamma_increment(b + a, n)
        )
    if k == n:
        return (
            _log_gamma_increment(a, b) - _log_gamma_increment(a + n, b)
            if b < n
            else _log_gamma_increment(a, n) - _log_gamma_increment(a + b, n)
        )
    x, y = float(k), float(n - k)
    count = x + y
    prior = a + b
    posterior = count + prior
    xp, yp = x + a, y + b
    lp, lq = math.log(xp) - math.log(posterior), math.log(yp) - math.log(posterior)
    return (
        _stirling_error(count)
        - _stirling_error(x)
        - _stirling_error(y)
        + _stirling_error(xp)
        + _stirling_error(yp)
        - _stirling_error(posterior)
        - _stirling_error(a)
        - _stirling_error(b)
        + _stirling_error(prior)
        - _deviance(x, math.log(count) + lp)
        - _deviance(y, math.log(count) + lq)
        - _deviance(a, math.log(prior) + lp)
        - _deviance(b, math.log(prior) + lq)
        + 0.5
        * (
            math.log(count)
            - math.log(x)
            - math.log(y)
            - math.log(xp)
            - math.log(yp)
            + math.log(posterior)
            + math.log(a)
            + math.log(b)
            - math.log(prior)
            - 1.83787706640934548356
        )
    )


def _parameters(arguments: Mapping[str, object], family: str) -> dict[str, float]:
    """Validate scalar count-mixture parameters.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named host parameters.
    family : str
        Count family name.

    Returns
    -------
    dict[str, float]
        Finite scalar parameters satisfying their domains.

    Raises
    ------
    DistributionError
        If any parameter violates its family constraint.
    """
    names = (
        ("zero_prob", "total_count", "concentration1", "concentration0")
        if family == "ZeroInflatedBetaBinomial"
        else ("zero_prob", "total_count", "probs")
    )
    p = {name: real_parameter(arguments, name, family) for name in names}
    valid = (
        all(math.isfinite(value) for value in p.values())
        and 0 <= p["zero_prob"] <= 1
        and p["total_count"] >= 0
    )
    if family in ("ZeroInflatedBinomial", "ZeroInflatedBetaBinomial"):
        valid = valid and p["total_count"].is_integer()
    if family == "HurdleNegativeBinomial":
        valid = valid and p["total_count"] > 0
    if "probs" in p:
        valid = (
            valid
            and 0 <= p["probs"] <= 1
            and (family == "ZeroInflatedBinomial" or p["probs"] < 1)
        )
    else:
        valid = valid and p["concentration1"] > 0 and p["concentration0"] > 0
    if not valid:
        raise DistributionError(f"{family} parameters violate the count-family domains")
    return p


def _log_count_mass(p: Mapping[str, float], count: int, family: str) -> float:
    """Score the uninflated count law in log space.

    Parameters
    ----------
    p : Mapping[str, float]
        Validated scalar parameters.
    count : int
        Nonnegative response count.
    family : str
        Mixture family selecting the count component.

    Returns
    -------
    float
        The component log mass, including exact endpoint atoms.
    """
    n = p["total_count"]
    if family in ("ZeroInflatedNegativeBinomial", "HurdleNegativeBinomial"):
        probability = p["probs"]
        return _negative_binomial_log_mass(
            n,
            count,
            math.log(probability) if probability else -math.inf,
            math.log1p(-probability),
        )
    if family == "ZeroInflatedBinomial":
        probability = p["probs"]
        return _binomial_log_mass(
            n,
            count,
            math.log(probability) if probability else -math.inf,
            math.log1p(-probability) if probability < 1 else -math.inf,
        )
    return _beta_binomial_log_mass(n, count, p["concentration1"], p["concentration0"])


def count_mixture_log_prob(
    arguments: Mapping[str, object], value: object, *, family: str
) -> float:
    """Evaluate a normalized count-mixture mass independently of PyTorch.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named scalar parameters.
    value : object
        Count observation.
    family : str
        Registered count-mixture name.

    Returns
    -------
    float
        Exact log probability, negative infinity outside the support.
    """
    p = _parameters(arguments, family)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return -math.inf
    count = _integer_observation(value, family)
    if count is None or count < 0:
        return -math.inf
    zero = p["zero_prob"]
    if family == "HurdleNegativeBinomial":
        if count == 0:
            return math.log(zero) if zero else -math.inf
        if zero == 1:
            return -math.inf
        if p["probs"] == 0:
            return math.log1p(-zero) if count == 1 else -math.inf
        t = -math.log1p(-p["probs"])
        rate = p["total_count"] * t
        if rate > 1e-4:
            return (
                math.log1p(-zero)
                + _log_count_mass(p, count, family)
                - math.log(-math.expm1(-rate))
            )
        numerator = (
            1 + rate / 2 + rate * rate / 12
            if rate < 1e-5
            else rate / -math.expm1(-rate)
        )
        denominator = 1 + t / 2 + t * t / 12 if t < 1e-5 else t / -math.expm1(-t)
        return (
            math.log1p(-zero)
            + _log_gamma_increment(p["total_count"] + 1, count - 1)
            - math.lgamma(count + 1)
            + (count - 1) * math.log(p["probs"])
            + p["total_count"] * math.log1p(-p["probs"])
            + math.log(numerator / denominator)
        )
    base = _log_count_mass(p, count, family)
    if count == 0:
        if zero == 0:
            return base
        if zero == 1:
            return 0.0
        a, b = math.log(zero), math.log1p(-zero) + base
        maximum = max(a, b)
        return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))
    return math.log1p(-zero) + base if zero < 1 else -math.inf


def _poisson(rate: float, rng: random.Random) -> int:
    """Draw a Poisson count without work proportional to a large rate.

    Parameters
    ----------
    rate : float
        Nonnegative finite intensity.
    rng : random.Random
        Host random generator.

    Returns
    -------
    int
        Poisson count using inversion at small rates and PTRS at large rates.

    References
    ----------
    Hoermann (1993), transformed rejection method PTRS.
    """
    _require_sample_capacity(rate)
    if rate < 10:
        limit, product, count = math.exp(-rate), rng.random(), 0
        while product > limit:
            product *= rng.random()
            count += 1
        return count
    root = math.sqrt(rate)
    b = 0.931 + 2.53 * root
    a = -0.059 + 0.02483 * b
    inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
    squeeze = 0.9277 - 3.6224 / (b - 2)
    while True:
        u, v = rng.random() - 0.5, 1 - rng.random()
        us = 0.5 - abs(u)
        if us == 0:
            continue
        count = math.floor((2 * a / us + b) * u + rate + 0.43)
        if us >= 0.07 and v <= squeeze:
            _require_sample_capacity(count)
            return count
        if count < 0 or (us < 0.013 and v > us):
            continue
        log_mass = _poisson_log_mass(count, rate)
        if math.log(v * inverse_alpha / (a / (us * us) + b)) <= log_mass:
            _require_sample_capacity(count)
            return count


def _logseries(probability: float, rng: random.Random) -> int:
    """Draw a logarithmic-series jump with its p=0 limit.

    Parameters
    ----------
    probability : float
        Success probability in [0, 1).
    rng : random.Random
        Host random generator.

    Returns
    -------
    int
        A positive logarithmic-series count.
    """
    v = 1 - rng.random()
    if v >= probability:
        return 1
    q = -math.expm1(math.log1p(-probability) * rng.random())
    if v <= q * q:
        count = math.floor(1 + math.log(v) / math.log(q))
        _require_sample_capacity(count)
        return count
    return 1 if v >= q else 2


def count_mixture_sample(
    arguments: Mapping[str, object], rng: random.Random, *, family: str
) -> int:
    """Sample a count mixture through scalar generative components.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named scalar parameters.
    rng : random.Random
        Host random generator.
    family : str
        Registered count-mixture name.

    Returns
    -------
    int
        An integer response. Positive NB sampling conditions a compound
        Poisson count directly, so tiny positive mass does not cause rejection.
    """
    p = _parameters(arguments, family)
    if rng.random() < p["zero_prob"]:
        return 0
    n = p["total_count"]
    if family in ("ZeroInflatedNegativeBinomial", "HurdleNegativeBinomial"):
        probability = p["probs"]
        rate = -n * math.log1p(-probability)
        if rate >= math.log(2):
            # Gamma-Poisson needs constant expected work even when the compound
            # Poisson representation would contain millions of jumps.
            while True:
                intensity = rng.gammavariate(n, probability / (1 - probability))
                draw = _poisson(intensity, rng)
                if family != "HurdleNegativeBinomial" or draw > 0:
                    return draw
        if family == "HurdleNegativeBinomial":
            u = rng.random()
            remaining = (
                math.log1p((1 - u) * math.expm1(rate))
                if rate <= 8
                else rate + math.log1p(u * math.expm1(-rate))
            )
            components = 1 + _poisson(remaining, rng)
        else:
            components = _poisson(rate, rng)
        count = sum(_logseries(probability, rng) for _ in range(components))
        _require_sample_capacity(count)
        return count
    probability = (
        p["probs"]
        if family == "ZeroInflatedBinomial"
        else _dirichlet_weights((p["concentration1"], p["concentration0"]), rng)[0]
    )
    return _binomial_sample(int(n), probability, rng)


SAMPLERS = {name: partial(count_mixture_sample, family=name) for name in NAMES}
DENSITIES = {name: partial(count_mixture_log_prob, family=name) for name in NAMES}


__all__: list[str] = []
