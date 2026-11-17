"""Standard-library Gaussian divided differences for response-time tails."""

import math
from quivers._gaussian_series import erfcx_derivative_polynomial, erfcx_denominator
from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8, _NODES_32, _WEIGHTS_32
from quivers.qiec.primitives import _log1mexp, _log_ndtr
from quivers.qiec.reference_support import DistributionError


def log_erfcx_positive(value: float) -> float:
    """Log scaled complementary error function for a nonnegative argument.

    Parameters
    ----------
    value : float
        A nonnegative argument.

    Returns
    -------
    float
        Log scaled complementary error function.
    """
    if value < 5:
        return value * value + math.log(math.erfc(value))
    return -0.5 * math.log(math.pi) - math.log(erfcx_denominator(value))


def _scaled_derivative(x: float, delta: float) -> float:
    """Log a positive erfcx derivative retaining the common Gaussian scale.

    Parameters
    ----------
    x, delta : float
        Base argument and nonnegative displacement.

    Returns
    -------
    float
        Scaled log derivative magnitude.
    """
    value = x + delta
    if value < -1:
        rest = 2 / math.sqrt(math.pi) * math.exp(
            -value * value
        ) - 2 * value * math.erfc(value)
        return delta * (2 * x + delta) + math.log(rest)
    if value >= 8:
        polynomial = erfcx_derivative_polynomial(0.5 / (value * value))
        return (
            math.log(polynomial) - 0.5 * math.log(math.pi) - 2 * math.log(value) - x * x
        )
    erfcx = math.exp(value * value) * math.erfc(value)
    return math.log(2 / math.sqrt(math.pi) - 2 * value * erfcx) - x * x


def _local(x, delta, nodes, weights):
    """Integrate a short positive derivative interval in log space.

    Parameters
    ----------
    x, delta : float
        Base argument and interval width.
    nodes, weights : tuple[float, ...]
        Gaussian quadrature rule.

    Returns
    -------
    float
        Scaled log difference.
    """
    values = [
        math.log(w / 2) + _scaled_derivative(x, delta * (n + 1) / 2)
        for n, w in zip(nodes, weights, strict=True)
    ]
    peak = max(values)
    return (
        math.log(delta) + peak + math.log(math.fsum(math.exp(v - peak) for v in values))
    )


def log_scaled_erfcx_difference(x: float, delta: float) -> float:
    """Log a positive scaled erfcx difference, checking short-interval accuracy.

    Parameters
    ----------
    x, delta : float
        Finite base argument and strictly positive separation.

    Returns
    -------
    float
        Log(exp(-x²)(erfcx(x)-erfcx(x+delta))).

    Raises
    ------
    DistributionError
        If arguments or bounded quadrature exceed numerical capacity.
    """
    if not math.isfinite(x) or not math.isfinite(delta) or delta <= 0:
        raise DistributionError(
            "Gaussian divided differences require finite arguments and positive separation"
        )
    if abs(x) > 1e150 or delta > 1e150:
        raise DistributionError(
            "Gaussian divided difference exceeds Float64 log-tail capacity"
        )
    variation = delta / (1 + abs(x)) if x >= 0 else delta * (1 + abs(x))
    if variation < 1e-3:
        coarse = _local(x, delta, _NODES_8, _WEIGHTS_8)
        fine = _local(x, delta, _NODES_32, _WEIGHTS_32)
        if not math.isfinite(fine) or abs(coarse - fine) > 5e-13 * (1 + abs(fine)):
            raise DistributionError(
                "Gaussian divided difference quadrature failed to converge"
            )
        return fine
    if x >= 0:
        first, second = log_erfcx_positive(x), log_erfcx_positive(x + delta)
        return -x * x + first + _log1mexp(second - first)
    left = math.log(math.erfc(x))
    y = x + delta
    right = (
        (delta * (2 * x + delta) + math.log(math.erfc(y)))
        if y < 0
        else (-x * x + log_erfcx_positive(y))
    )
    return left + _log1mexp(min(0.0, right - left))


def exgaussian_log_component(z: float, rate_scale: float) -> float:
    """Log the exponentially tilted Gaussian shared by density and survival.

    Parameters
    ----------
    z : float
        Standardized response.
    rate_scale : float
        Positive exponential rate times Gaussian scale.

    Returns
    -------
    float
        Log(exp(rate_scale**2/2-rate_scale*z)*Phi(z-rate_scale)).
    """
    difference = rate_scale - z
    if difference > 0:
        return (
            -math.log(2) - 0.5 * z * z + log_erfcx_positive(difference / math.sqrt(2))
        )
    return 0.5 * rate_scale**2 - rate_scale * z + _log_ndtr(-difference)


__all__ = []
