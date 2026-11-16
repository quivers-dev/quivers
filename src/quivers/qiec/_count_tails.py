"""Reference scalar count tails using shared Beta/Gamma and count kernels."""

from __future__ import annotations

import math

from quivers.qiec._beta_tails import beta_fraction, log_beta_front
from quivers.qiec._gamma_tails import _regularized_gamma_logs, _log_complement
from quivers.qiec._count_mixtures import _parameters, _beta_binomial_log_mass
from quivers.qiec._sampling import _trial_count
from quivers.qiec.reference_support import (
    DistributionError,
    log_probabilities,
    real_parameter,
    simplex,
    vector_parameter,
)

COUNT_TAIL_FAMILIES = frozenset(
    {
        "Bernoulli",
        "Categorical",
        "Binomial",
        "NegativeBinomial",
        "Poisson",
        "Geometric",
        "BetaBinomial",
        "ZeroInflatedPoisson",
        "ZeroInflatedNegativeBinomial",
        "ZeroInflatedBinomial",
        "ZeroInflatedBetaBinomial",
        "HurdlePoisson",
        "HurdleNegativeBinomial",
    }
)


def _logadd(first, second):
    """Add two nonnegative masses in logarithmic coordinates.

    Parameters
    ----------
    first, second : float
        Logarithmic nonnegative masses.

    Returns
    -------
    float
        Logarithm of their sum.
    """
    peak = max(first, second)
    return (
        peak + math.log1p(math.exp(min(first, second) - peak))
        if peak > -math.inf
        else peak
    )


def _pair(a, b, lx, lq):
    """Evaluate incomplete-Beta tails while retaining finite logit coordinates.

    Parameters
    ----------
    a, b, lx, lq : float
        Positive shapes and complementary log coordinates.

    Returns
    -------
    tuple[float, float]
        Paired inclusive cumulative and strict survival logarithms.
    """
    if lx == -math.inf:
        return -math.inf, 0.0
    if lq == -math.inf:
        return 0.0, -math.inf
    swapped = lx > math.log(a + 1) - math.log(a + b + 2)
    if swapped:
        a, b, lx, lq = b, a, lq, lx
    x = math.exp(lx)
    front = (
        log_beta_front(a, b, x)
        if 0 < x < 1
        else log_beta_front(a, b, 0.5) + a * (lx + math.log(2)) + b * (lq + math.log(2))
    )

    def nonzero(value):
        """Preserve a continued-fraction denominator sign near zero.

        Parameters
        ----------
        value : float
            Recurrence denominator.

        Returns
        -------
        float
            Nonzero denominator.
        """
        return math.copysign(max(abs(value), 1e-300), value)

    fraction = beta_fraction(
        a,
        b,
        x,
        nonzero=nonzero,
        converged=lambda change: abs(change - 1) <= 1.7763568394002505e-15,
    )
    lower = min(0.0, front + math.log(fraction) - math.log(a))
    upper = _log_complement(lower)
    return (upper, lower) if swapped else (lower, upper)


def _coordinates(arguments, name):
    """Validate proper original probability or logit coordinates.

    Parameters
    ----------
    arguments : Mapping
        Named scalar distribution parameters.
    name : str
        Family used for parameter diagnostics.

    Returns
    -------
    tuple[float, float]
        Log success and log failure probabilities.

    Raises
    ------
    DistributionError
        If probabilities fall outside the proper family domain.
    """
    if "probs" in arguments:
        probability = real_parameter(arguments, "probs", name)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise DistributionError(f"{name} requires probabilities in [0,1]")
    lp, lq = log_probabilities(arguments, name)
    if math.isnan(lp) or math.isnan(lq) or lp > 0 or lq > 0:
        raise DistributionError(f"{name} requires proper probability coordinates")
    if name == "Geometric" and lp == -math.inf:
        raise DistributionError(
            "Geometric requires proper success probabilities in (0,1]"
        )
    if name == "NegativeBinomial" and lq == -math.inf:
        raise DistributionError(
            "NegativeBinomial requires success probability below one"
        )
    return lp, lq


def geometric_coordinates(arguments):
    """Read a proper geometric law, retaining finite extreme log odds.

    Parameters
    ----------
    arguments : Mapping
        Success probability or logit parameters.

    Returns
    -------
    tuple[float, float]
        Log success and log failure probabilities.
    """
    return _coordinates(arguments, "Geometric")


def _bb_sum(n, k, a, b, upper, max_terms):
    """Sum one complete finite tail using the shared stable mass and recurrence.

    Parameters
    ----------
    n, k : int
        Trials and query threshold.
    a, b : float
        Positive beta concentrations.
    upper : bool
        Whether to sum downwards from the upper support endpoint.
    max_terms : int
        Maximum number of complete finite-tail cells.

    Returns
    -------
    float
        Log complete tail mass.

    Raises
    ------
    DistributionError
        If the requested tail exceeds its resource budget.
    """
    length = n - k if upper else k + 1
    if length > max_terms:
        raise DistributionError("BetaBinomial CDF finite-tail capacity exceeded")
    j = n if upper else 0
    mass = _beta_binomial_log_mass(float(n), j, a, b)
    total = mass
    for _ in range(1, length):
        if upper:
            mass += (
                math.log(j)
                - math.log(n - j + 1)
                + math.log(b + n - j)
                - math.log(a + j - 1)
            )
            j -= 1
        else:
            mass += (
                math.log(n - j)
                - math.log(j + 1)
                + math.log(a + j)
                - math.log(b + n - j - 1)
            )
            j += 1
        total = _logadd(total, mass)
    return min(0.0, total)


def log_count_tails(name, arguments, value, *, max_terms=262144):
    """Evaluate scalar counting-law tails without a truncated infinite sum.

    Parameters
    ----------
    name : str
        Registered scalar counting family.
    arguments : Mapping
        Native-scale family parameters.
    value : float
        Real threshold; noninteger thresholds use floor.
    max_terms : int
        Complete finite-tail budget for beta-binomial calculations.

    Returns
    -------
    tuple[float, float]
        Inclusive log CDF and strict log survival.

    Raises
    ------
    DistributionError
        If parameters or thresholds are invalid, or a finite-tail budget is exceeded.
    """
    point = float(value)
    if math.isnan(point):
        raise DistributionError("count CDF query must not be NaN")
    if name == "Categorical":
        probabilities = simplex(arguments, name)
        if point < 0:
            return -math.inf, 0.0
        if point >= len(probabilities) - 1:
            return 0.0, -math.inf
        k = math.floor(point)
        if "logits" in arguments and "probs" not in arguments:
            from quivers.qiec.reference_families import _logsumexp

            weights = vector_parameter(arguments, "logits", name)
            normalizer = _logsumexp(weights)
            return (
                _logsumexp(weights[: k + 1]) - normalizer,
                _logsumexp(weights[k + 1 :]) - normalizer,
            )
        lower, upper = (
            math.fsum(probabilities[: k + 1]),
            math.fsum(probabilities[k + 1 :]),
        )
        return (
            math.log(lower) if lower else -math.inf,
            math.log(upper) if upper else -math.inf,
        )
    if name in {"Bernoulli", "Binomial", "NegativeBinomial", "Geometric"}:
        lp, lq = _coordinates(arguments, name)
        if name == "Binomial":
            n = _trial_count(arguments.get("total_count"))
        elif name == "NegativeBinomial":
            n = real_parameter(arguments, "total_count", name)
            if not math.isfinite(n) or n < 0:
                raise DistributionError(
                    "NegativeBinomial shape must be finite and nonnegative"
                )
        else:
            n = 1
        if point < 0:
            return -math.inf, 0.0
        if point == math.inf:
            return 0.0, -math.inf
        k = math.floor(point)
        if name == "Bernoulli":
            return (lq, lp) if k == 0 else (0.0, -math.inf)
        if name == "Geometric":
            sf = (k + 1) * lq
            if k == 0:
                return lp, sf
            if sf < 0:
                return _log_complement(sf), sf
            # A finite negative logit can underflow its ordinary hazard.
            logit = real_parameter(arguments, "logits", name)
            return math.log(k + 1) + logit, sf
        if name == "Binomial":
            if k >= n:
                return 0.0, -math.inf
            return _pair(n - k, k + 1, lq, lp)
        if n == 0:
            return 0.0, -math.inf
        return _pair(n, k + 1, lq, lp)
    if name == "Poisson":
        rate = real_parameter(arguments, "rate", name)
        if not math.isfinite(rate) or rate < 0:
            raise DistributionError("Poisson rate must be finite and nonnegative")
        if point < 0:
            return -math.inf, 0.0
        if point == math.inf:
            return 0.0, -math.inf
        cdf, sf = _regularized_gamma_logs(math.floor(point) + 1, rate)
        return sf, cdf
    if name == "BetaBinomial":
        n = _trial_count(arguments.get("total_count"))
        a, b = (
            real_parameter(arguments, v, name)
            for v in ("concentration1", "concentration0")
        )
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            raise DistributionError(
                "BetaBinomial concentrations must be finite and positive"
            )
        if point < 0:
            return -math.inf, 0.0
        if point >= n:
            return 0.0, -math.inf
        k = math.floor(point)
        upper = n - k < k + 1
        tail = _bb_sum(n, k, a, b, upper, max_terms)
        if tail > -math.log(2):
            upper = not upper
            tail = _bb_sum(n, k, a, b, upper, max_terms)
        other = _log_complement(tail)
        return (other, tail) if upper else (tail, other)
    if name in {"ZeroInflatedPoisson", "HurdlePoisson"}:
        z = real_parameter(arguments, "zero_prob", name)
        base = "Poisson"
        params = {"rate": arguments["rate"]}
        if not math.isfinite(z) or not 0 <= z <= 1:
            raise DistributionError("zero_prob must lie in [0,1]")
    else:
        p = _parameters(arguments, name)
        z = p.pop("zero_prob")
        base = {
            "ZeroInflatedBinomial": "Binomial",
            "ZeroInflatedBetaBinomial": "BetaBinomial",
            "ZeroInflatedNegativeBinomial": "NegativeBinomial",
            "HurdleNegativeBinomial": "NegativeBinomial",
        }.get(name)
        if base is None:
            raise DistributionError(f"{name} has no scalar count tail kernel")
        params = p
    lc, ls = log_count_tails(base, params, point, max_terms=max_terms)
    if point < 0:
        return -math.inf, 0.0
    if name.startswith("Hurdle"):
        lzero = log_count_tails(base, params, 0.0, max_terms=max_terms)[0]
        if point < 1:
            lc, ls = -math.inf, 0.0
        elif lzero == 0:
            lc, ls = 0.0, -math.inf
        else:
            normalizer = _log_complement(lzero)
            lc = (
                lc + _log_complement(lzero - lc) - normalizer
                if lc > lzero
                else -math.inf
            )
            ls -= normalizer
            if ls < -math.log(2):
                lc = _log_complement(ls)
    lz = math.log(z) if z else -math.inf
    ln = math.log1p(-z) if z < 1 else -math.inf
    return _logadd(lz, ln + lc), ln + ls
