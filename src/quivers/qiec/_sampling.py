"""Shared host samplers for exact finite counts and log-space Gamma weights."""

from collections.abc import Sequence
import math
import random

from quivers.qiec.reference_support import DistributionError


def _log_gamma_parts(shape: float, rng: random.Random) -> tuple[float, float]:
    """Draw the two components of the Gamma shape-augmentation identity.

    Parameters
    ----------
    shape : float
        Finite strictly positive Gamma shape.
    rng : random.Random
        Generator supplying the augmented Gamma and uniform draws.

    Returns
    -------
    tuple[float, float]
        Log of Gamma(shape+1,1) and an independent unit exponential.

    Raises
    ------
    DistributionError
        If a shape or the underlying Gamma draw exceeds numerical capacity.
    """
    if not math.isfinite(shape) or shape <= 0:
        raise DistributionError("Gamma sampling requires finite positive shapes")
    boosted = rng.gammavariate(shape + 1.0, 1.0)
    if not math.isfinite(boosted) or boosted <= 0:
        raise DistributionError("Gamma sampling exceeded finite positive capacity")
    return math.log(boosted), -math.log1p(-rng.random())


def _log_standard_gamma(shape: float, rng: random.Random) -> float:
    """Sample a log Gamma value before any small-shape exponentiation.

    Parameters
    ----------
    shape : float
        Strictly positive Gamma shape.
    rng : random.Random
        Generator, with the same draw order as the LogGamma reference law.

    Returns
    -------
    float
        Log of a Gamma(shape,1) draw.
    """
    log_gamma, exponential = _log_gamma_parts(shape, rng)
    return log_gamma - exponential / shape


def _dirichlet_weights(a: Sequence[float], rng: random.Random) -> tuple[float, ...]:
    """Normalize log-Gamma draws, including shapes below floating underflow.

    Parameters
    ----------
    a : Sequence[float]
        Nonempty positive finite concentrations.
    rng : random.Random
        Host random generator.

    Returns
    -------
    tuple[float, ...]
        Normalized category probabilities.

    Raises
    ------
    DistributionError
        If no category is supplied or Gamma sampling exceeds capacity.
    """
    if not a:
        raise DistributionError("Dirichlet sampling requires a nonempty vector")
    parts = [_log_gamma_parts(v, rng) for v in a]
    logs = [g - e / v for (g, e), v in zip(parts, a, strict=True)]
    peak = max(logs)
    if peak == -math.inf:
        scale = min(a)
        scaled = [
            scale * g - e * (scale / v) for (g, e), v in zip(parts, a, strict=True)
        ]
        peak = max(scaled)
        logs = [(v - peak) / scale for v in scaled]
    else:
        logs = [v - peak for v in logs]
    weights = [math.exp(v) for v in logs]
    total = sum(weights)
    return tuple(v / total for v in weights)


def _trial_count(value: object) -> int:
    """Read an exact finite-trial count before converting it to Float64.

    Parameters
    ----------
    value : object
        Scalar integer or real trial count.

    Returns
    -------
    int
        Nonnegative trials through 2**53.

    Raises
    ------
    DistributionError
        If the count is not exactly representable within the supported range.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DistributionError("binomial trials must be numeric integers")
    if not 0 <= value <= 2**53 or value != int(value):
        raise DistributionError(
            "count trials exceed exact Float64 capacity or are not nonnegative integers through 2**53"
        )
    return int(value)


def _binomial_sample(n: int, p: float, rng: random.Random) -> int:
    """Use the standard library's efficient exact-binomial sampling kernel.

    Parameters
    ----------
    n : int
        Integer trials within exact Float64 capacity.
    p : float
        Probability in the closed unit interval.
    rng : random.Random
        Generator whose state is advanced only for nondegenerate draws.

    Returns
    -------
    int
        Sampled count in the trial support.

    Raises
    ------
    DistributionError
        If trials, probability, or the returned count violates its contract.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 2**53:
        raise DistributionError("binomial trials must be exact integers through 2**53")
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise DistributionError("binomial probability must lie in [0,1]")
    if n == 0 or p == 0:
        return 0
    if p == 1:
        return n
    value = rng.binomialvariate(n, p)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= n:
        raise DistributionError(
            "binomial sample exceeded remaining trials or was not an integer"
        )
    return value


__all__: list[str] = []
