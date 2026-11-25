"""Positive Gaussian divided differences shared by response-time tails."""

import math
import torch
from torch import Tensor

from quivers._positions import demand, elementwise
from quivers._gaussian_series import erfcx_derivative_polynomial, erfcx_denominator
from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8, _NODES_32, _WEIGHTS_32
from quivers.continuous._ordered import _log1mexp


def log_erfcx_positive(value: Tensor) -> Tensor:
    """Evaluate log erfcx on nonnegative arguments with stable derivatives.

    Parameters
    ----------
    value : Tensor
        Nonnegative finite arguments.

    Returns
    -------
    Tensor
        Log scaled complementary error function.
    """
    large = value >= 5
    small = torch.where(large, 0.0, value)
    tail = torch.where(large, value, 5.0)
    result = -0.5 * math.log(math.pi) - erfcx_denominator(tail).log()
    return torch.where(large, result, torch.special.erfcx(small).log())


def exgaussian_log_component(z: Tensor, rate_scale: Tensor) -> Tensor:
    """Log the exponentially tilted Gaussian term shared by density and SF.

    Parameters
    ----------
    z : Tensor
        Standardized response.
    rate_scale : Tensor
        Positive product of exponential rate and Gaussian scale.

    Returns
    -------
    Tensor
        Log(exp(rate_scale**2/2-rate_scale*z)*Phi(z-rate_scale)).
    """
    difference = rate_scale - z
    left = (
        -math.log(2.0)
        - 0.5 * z.square()
        + log_erfcx_positive(difference.clamp_min(0) / math.sqrt(2.0))
    )
    r = torch.where(difference > 0, 0.0, rate_scale)
    zz = torch.where(difference > 0, 0.0, z)
    right = 0.5 * r.square() - r * zz + torch.special.log_ndtr(-difference.clamp_max(0))
    return torch.where(difference > 0, left, right)


def paired_log_tails(lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
    """Retain the smaller log tail and complement it without cancellation.

    Parameters
    ----------
    lower, upper : Tensor
        Independently evaluated log CDF and log survival.

    Returns
    -------
    tuple[Tensor, Tensor]
        Complementary log tail probabilities.
    """
    lower_small = lower < upper
    safe_lower = torch.where(lower_small, lower, -1.0)
    safe_upper = torch.where(lower_small, -1.0, upper)
    return (
        torch.where(lower_small, lower, _log1mexp(safe_upper)),
        torch.where(lower_small, _log1mexp(safe_lower), upper),
    )


def _log_scaled_derivative(x: Tensor, delta: Tensor) -> Tensor:
    """Evaluate log(exp(-x**2)*(-erfcx'(x+delta))) without cancellation.

    Parameters
    ----------
    x, delta : Tensor
        Base argument and a nonnegative short displacement.

    Returns
    -------
    Tensor
        Log positive derivative magnitude on the common Gaussian scale.
    """
    value = x + delta
    negative, large = value < -1, value >= 8
    central = torch.where(negative | large, 0.0, value)
    derivative = 2 / math.sqrt(math.pi) - 2 * central * torch.special.erfcx(central)
    result = derivative.log() - x.square()
    positive = torch.where(large, value, 8.0)
    polynomial = erfcx_derivative_polynomial(0.5 / positive.square())
    asymptotic = (
        polynomial.log() - 0.5 * math.log(math.pi) - 2 * positive.log() - x.square()
    )
    neg = torch.where(negative, value, -1.0)
    rest = 2 / math.sqrt(math.pi) * torch.exp(-neg.square()) - 2 * neg * torch.erfc(neg)
    origin = torch.where(negative, x, 0.0)
    step = torch.where(negative, delta, 0.0)
    left = step * (2 * origin + step) + rest.log()
    return torch.where(negative, left, torch.where(large, asymptotic, result))


def _local_difference(x: Tensor, delta: Tensor, nodes, weights) -> Tensor:
    """Integrate the positive derivative over a short finite interval.

    Parameters
    ----------
    x, delta : Tensor
        Base argument and positive interval width.
    nodes, weights : tuple[float, ...]
        Gauss-Legendre rule on [-1,1].

    Returns
    -------
    Tensor
        Scaled log erfcx difference.
    """
    node = x.new_tensor(nodes)
    weight = x.new_tensor(weights)
    displacement = delta.unsqueeze(-1) * ((node + 1) / 2)
    values = _log_scaled_derivative(x.unsqueeze(-1), displacement)
    return delta.log() + torch.logsumexp(values + (weight / 2).log(), -1)


@elementwise
def log_scaled_erfcx_difference(x: Tensor, delta: Tensor) -> Tensor:
    """Evaluate exp(-x²)(erfcx(x)-erfcx(x+delta)) in log space.

    Direct branches retain their common Gaussian factor. Short intervals
    instead integrate the strictly positive derivative with agreeing 8- and
    32-point rules; disagreement raises instead of returning a zero tail.

    Parameters
    ----------
    x : Tensor
        Finite base arguments.
    delta : Tensor
        Finite strictly positive separations.

    Returns
    -------
    Tensor
        Float64 logarithm of the positive scaled difference.

    Raises
    ------
    ValueError
        If arguments are nonfinite or separations are nonpositive.
    RuntimeError
        If quadrature fails to agree or log-tail capacity is exceeded.
    """
    x, delta = torch.broadcast_tensors(x.double(), delta.double())
    demand(
        torch.isfinite(x) & torch.isfinite(delta) & (delta > 0),
        lambda: ValueError(
            "Gaussian divided differences require finite arguments and positive separation"
        ),
    )
    demand(
        ~((x.abs() > 1e150) | (delta > 1e150)),
        lambda: RuntimeError(
            "Gaussian divided difference exceeds Float64 log-tail capacity"
        ),
    )
    variation = torch.where(x >= 0, delta / (1 + x.abs()), delta * (1 + x.abs()))
    local = variation < 1e-3
    xx, dd = torch.where(local, 0.0, x), torch.where(local, 1.0, delta)
    positive = xx >= 0
    xp = torch.where(positive, xx, 0.0)
    yp = xp + dd
    first = log_erfcx_positive(xp)
    second = log_erfcx_positive(yp)
    positive_result = -xp.square() + first + _log1mexp(second - first)
    xn = torch.where(positive, -1.0, xx)
    yn = xn + dd
    left = torch.erfc(xn).log()
    below = yn < 0
    yn_safe = torch.where(below, yn, 0.0)
    right_negative = dd * (2 * xn + dd) + torch.erfc(yn_safe).log()
    right_positive = -xn.square() + log_erfcx_positive(torch.where(below, 0.0, yn))
    right = torch.where(below, right_negative, right_positive)
    negative_result = left + _log1mexp((right - left).clamp_max(0))
    result = torch.where(positive, positive_result, negative_result)
    if bool(local.any()):
        a, d = x[local], delta[local]
        coarse = _local_difference(a, d, _NODES_8, _WEIGHTS_8)
        fine = _local_difference(a, d, _NODES_32, _WEIGHTS_32)
        agreed = (coarse - fine).abs() <= 5e-13 * (1 + fine.abs())
        if not bool((torch.isfinite(fine) & agreed).all()):
            raise RuntimeError(
                "Gaussian divided difference quadrature failed to converge"
            )
        positions = local.reshape(-1).nonzero().squeeze(-1)
        result = result.reshape(-1).index_copy(0, positions, fine).reshape(result.shape)
    return result
