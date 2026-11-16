"""Plain-Python heterogeneous-trial count vectors and stable beta factors."""

import math
import random
from collections.abc import Mapping, Sequence

from quivers.qiec._sampling import _binomial_sample, _dirichlet_weights, _trial_count
from quivers.qiec._count_mixtures import (
    _binomial_log_mass,
    _beta_binomial_log_mass,
)
from quivers.qiec.reference_support import (
    DistributionError,
    vector_parameter,
    simplex,
)


def _parameters(arguments: Mapping[str, object], family: str):
    """Validate exact trials and count-vector parameters.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named scalar trials and vector probabilities or concentrations.
    family : str
        Count-vector family name.

    Returns
    -------
    tuple[int, tuple[float, ...]]
        Exact trials and category parameters or normalized log weights.

    Raises
    ------
    DistributionError
        If parameters violate the count or vector contract.
    """
    n = _trial_count(arguments.get("total_count"))
    if family == "DirichletMultinomial":
        values = vector_parameter(arguments, "concentration", family)
        if (
            not values
            or any(not math.isfinite(v) or v <= 0 for v in values)
            or not math.isfinite(sum(values))
        ):
            raise DistributionError(
                "DirichletMultinomial concentrations and their total must be finite and positive"
            )
        return int(n), values
    if ("probs" in arguments) == ("logits" in arguments):
        raise DistributionError("Multinomial requires exactly one of probs and logits")
    if "probs" in arguments:
        probabilities = simplex(arguments, family)
        return int(n), tuple(math.log(v) if v > 0 else -math.inf for v in probabilities)
    logits = vector_parameter(arguments, "logits", family)
    if (
        not logits
        or any(math.isnan(v) or v == math.inf for v in logits)
        or max(logits) == -math.inf
    ):
        raise DistributionError(
            "Multinomial logits require a finite category and no NaN or positive infinity"
        )
    peak = max(logits)
    log_total = peak + math.log(sum(math.exp(v - peak) for v in logits))
    return int(n), tuple(v - log_total for v in logits)


def _event(value: object, n: int, categories: int):
    """Validate event width and exact integer counts.

    Parameters
    ----------
    value : object
        Candidate count vector.
    n : int
        Exact trial total.
    categories : int
        Declared event width.

    Returns
    -------
    tuple[int, ...] or None
        Counts, or None for a point outside support.

    Raises
    ------
    DistributionError
        If event width or integer representation is invalid.
    """
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != categories
    ):
        raise DistributionError(f"count event requires {categories} categories")
    counts = []
    for v in value:
        if isinstance(v, int) and v > 2**53:
            raise DistributionError("count event exceeds exact Float64 capacity 2**53")
        if (
            isinstance(v, bool)
            or not isinstance(v, int | float)
            or not math.isfinite(v)
            or v < 0
            or v != int(v)
        ):
            return None
        counts.append(int(v))
    return tuple(counts) if sum(counts) == n else None


def _logadd(a: float, b: float) -> float:
    """Add two log weights with an exact all-zero limit.

    Parameters
    ----------
    a, b : float
        Log weights, possibly negative infinity.

    Returns
    -------
    float
        Logarithm of their summed weights.
    """
    peak = max(a, b)
    return (
        peak + math.log1p(math.exp(min(a, b) - peak))
        if peak != -math.inf
        else -math.inf
    )


def multinomial_density(arguments: Mapping[str, object], value: object) -> float:
    """Score one multinomial vector using conditional binomial factors.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Trials and probability or logit vectors.
    value : object
        Candidate integer event.

    Returns
    -------
    float
        Normalized log mass, negative infinity outside support.
    """
    n, logs = _parameters(arguments, "Multinomial")
    counts = _event(value, n, len(logs))
    if counts is None:
        return -math.inf
    tail = [logs[-1]]
    for logp in logs[-2::-1]:
        tail.append(_logadd(logp, tail[-1]))
    tail.reverse()
    result = 0.0
    for i, count in enumerate(counts[:-1]):
        if n == 0:
            break
        if tail[i] == -math.inf:
            return -math.inf
        result += _binomial_log_mass(
            float(n), count, logs[i] - tail[i], tail[i + 1] - tail[i]
        )
        n -= count
    return result


def dirichlet_multinomial_density(
    arguments: Mapping[str, object], value: object
) -> float:
    """Score Dirichlet-multinomial events using stable beta-binomial factors.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Trials and concentration vectors.
    value : object
        Candidate integer event.

    Returns
    -------
    float
        Normalized log mass, negative infinity outside support.
    """
    n, a = _parameters(arguments, "DirichletMultinomial")
    counts = _event(value, n, len(a))
    if counts is None:
        return -math.inf
    tail = [a[-1]]
    for value in a[-2::-1]:
        tail.append(value + tail[-1])
    tail.reverse()
    result = 0.0
    for i, count in enumerate(counts[:-1]):
        result += _beta_binomial_log_mass(n, count, a[i], tail[i + 1])
        n -= count
    return result


def _sample_counts(n: int, p: Sequence[float], rng: random.Random) -> tuple[int, ...]:
    """Draw K-1 efficient conditional binomials without iterating over trials.

    Parameters
    ----------
    n : int
        Nonnegative exact trial count.
    p : Sequence[float]
        Nonnegative probabilities with a positive total.
    rng : random.Random
        Random generator.

    Returns
    -------
    tuple[int, ...]
        Counts summing exactly to the input trial total.

    Raises
    ------
    DistributionError
        If a binomial draw violates the remaining trial count.
    """
    tail = [p[-1]]
    for value in p[-2::-1]:
        tail.append(value + tail[-1])
    tail.reverse()
    counts = []
    for i, prob in enumerate(p[:-1]):
        draw = _binomial_sample(n, prob / tail[i], rng) if n and tail[i] > 0 else 0
        counts.append(draw)
        n -= draw
    return (*counts, n)


def multinomial_sample(
    arguments: Mapping[str, object], rng: random.Random
) -> tuple[int, ...]:
    """Draw an exact-total multinomial count vector.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Trials and probability or logit vectors.
    rng : random.Random
        Random generator.

    Returns
    -------
    tuple[int, ...]
        Sampled category counts.
    """
    n, logs = _parameters(arguments, "Multinomial")
    return _sample_counts(n, tuple(math.exp(v) for v in logs), rng)


def dirichlet_multinomial_sample(
    arguments: Mapping[str, object], rng: random.Random
) -> tuple[int, ...]:
    """Draw log-space gamma weights before efficient conditional count draws.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Trials and concentration vectors.
    rng : random.Random
        Random generator.

    Returns
    -------
    tuple[int, ...]
        Sampled category counts.

    Raises
    ------
    DistributionError
        If gamma sampling exceeds finite floating-point capacity.
    """
    n, a = _parameters(arguments, "DirichletMultinomial")
    if n == 0:
        return tuple(0 for _ in a)
    return _sample_counts(n, _dirichlet_weights(a, rng), rng)


SAMPLERS = {
    "Multinomial": multinomial_sample,
    "DirichletMultinomial": dirichlet_multinomial_sample,
}
DENSITIES = {
    "Multinomial": multinomial_density,
    "DirichletMultinomial": dirichlet_multinomial_density,
}


__all__: list[str] = []
