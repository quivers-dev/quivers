"""Paired reference tails for Normal-exponential and inverse Gaussian laws."""

from __future__ import annotations

from collections.abc import Mapping
import math
from quivers.qiec.reference_support import DistributionError, real_parameter as _real
from quivers.qiec.primitives import _log1mexp, _log_ndtr
from quivers.qiec._gaussian_tails import (
    log_erfcx_positive,
    log_scaled_erfcx_difference,
    exgaussian_log_component,
)


def _exgaussian_parameters(a: Mapping[str, object]) -> tuple[float, float, float]:
    """Read a Normal location/scale and a positive Exponential rate.

    Parameters
    ----------
    a : Mapping[str, object]
        Named location, scale, and Exponential rate.

    Returns
    -------
    tuple[float, float, float]
        Validated parameters in constructor order.

    Raises
    ------
    DistributionError
        If scale or rate is nonpositive, or any parameter is malformed.
    """
    loc = _real(a, "loc", "ExGaussian")
    scale = _real(a, "scale", "ExGaussian")
    rate = _real(a, "rate", "ExGaussian")
    if scale <= 0 or rate <= 0:
        raise DistributionError("ExGaussian needs scale > 0 and rate > 0")
    return loc, scale, rate


def _inverse_gaussian_parameters(a: Mapping[str, object]) -> tuple[float, float]:
    """Read the positive mean and shape of an inverse Gaussian.

    Parameters
    ----------
    a : Mapping[str, object]
        Named response mean and positive shape.

    Returns
    -------
    tuple[float, float]
        Validated parameters in constructor order.

    Raises
    ------
    DistributionError
        If mean or shape is nonpositive, or any parameter is malformed.
    """
    mu = _real(a, "mu", "InverseGaussian")
    shape = _real(a, "shape", "InverseGaussian")
    if mu <= 0 or shape <= 0:
        raise DistributionError("InverseGaussian needs mu > 0 and shape > 0")
    return mu, shape


def log_response_time_tails(family, arguments, value):
    """Evaluate complementary log tails through one shared Gaussian factor.

    Parameters
    ----------
    family : str
        ExGaussian or InverseGaussian.
    arguments : Mapping[str, object]
        Scalar constructor parameters.
    value : float
        Extended-real evaluation point.

    Returns
    -------
    tuple[float, float]
        Log cumulative and strict upper-tail probabilities.

    Raises
    ------
    DistributionError
        If the point is NaN, parameters are invalid or numerical capacity fails.
    """
    if math.isnan(value):
        raise DistributionError("response-time tails require non-NaN evaluation points")
    if family == "ExGaussian":
        loc, scale, rate = _exgaussian_parameters(arguments)
        if not math.isfinite(value):
            return (0.0, -math.inf) if value > 0 else (-math.inf, 0.0)
        z, r = (value - loc) / scale, rate * scale
        lower = log_scaled_erfcx_difference(
            -z / math.sqrt(2), r / math.sqrt(2)
        ) - math.log(2)
        terms = (_log_ndtr(-z), exgaussian_log_component(z, r))
        peak = max(terms)
        upper = peak + math.log(math.fsum(math.exp(t - peak) for t in terms))
    elif family == "InverseGaussian":
        mu, shape = _inverse_gaussian_parameters(arguments)
        if value <= 0:
            return -math.inf, 0.0
        if value == math.inf:
            return 0.0, -math.inf
        root = math.sqrt(shape) / math.sqrt(value)
        a = (value - mu) / math.sqrt(value) * (math.sqrt(shape) / mu)
        b = (math.sqrt(value) * (math.sqrt(shape) / mu) + root) / math.sqrt(2)
        upper = log_scaled_erfcx_difference(
            a / math.sqrt(2), math.sqrt(2) * root
        ) - math.log(2)
        terms = (_log_ndtr(a), -0.5 * a * a - math.log(2) + log_erfcx_positive(b))
        peak = max(terms)
        lower = peak + math.log(math.fsum(math.exp(t - peak) for t in terms))
    else:
        raise DistributionError(f"no response-time tail rule for {family}")
    return (lower, _log1mexp(lower)) if lower < upper else (_log1mexp(upper), upper)


def log_density_variation(family, arguments, lower, upper):
    """Bound absolute log-density change without integrating probabilities.

    Parameters
    ----------
    family : str
        ExGaussian or InverseGaussian.
    arguments : Mapping[str, object]
        Retained scalar constructor parameters.
    lower, upper : float
        Extended-real interval endpoints.

    Returns
    -------
    float
        Nonnegative variation bound, or infinity outside the smooth support.

    Raises
    ------
    DistributionError
        If the family or its parameters are invalid.
    """
    valid = math.isfinite(lower) and math.isfinite(upper) and upper > lower
    if family == "ExGaussian":
        loc, scale, rate = _exgaussian_parameters(arguments)
        if not valid:
            return math.inf
        t = (lower - loc) / scale - rate * scale
        return (upper - lower) * (rate + (max(-t, 0) + 1) / scale)
    if family == "InverseGaussian":
        mu, shape = _inverse_gaussian_parameters(arguments)
        if not valid or lower <= 0:
            return math.inf
        return (upper - lower) * (
            1.5 / lower + 0.5 * (shape / lower) / lower + 0.5 * (shape / mu) / mu
        )
    raise DistributionError(f"no response-time variation bound for {family}")


__all__ = []
