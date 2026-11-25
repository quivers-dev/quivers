"""Log regularized incomplete gamma functions with bounded convergence."""

from __future__ import annotations

import torch
from torch import Tensor

from quivers._positions import anywhere, demand, everywhere
from quivers.continuous._ordered import _log1mexp


def _complement(value: Tensor) -> Tensor:
    """Log complement without inactive singular derivatives at mass one."""
    negative = value < 0
    safe = torch.where(negative, value, -1.0)
    return torch.where(negative, _log1mexp(safe), -torch.inf)


def _log_gamma1p(shape: Tensor) -> Tensor:
    """Retain log Gamma increments smaller than the spacing near one."""
    small = shape < 0.01
    value = torch.where(small, shape, 0.0)
    coefficients = (
        -0.5772156649015328606,
        0.8224670334241132182,
        -0.4006856343865314285,
        0.2705808084277845479,
        -0.2073855510286739853,
        0.1695571769974081899,
        -0.1440498967688461181,
        0.1255096695247430424,
        -0.1113342658695646905,
        0.1000994575127818085,
    )
    polynomial = torch.full_like(value, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        polynomial = coefficient + value * polynomial
    regular = torch.where(small, 0.1, shape)
    return torch.where(small, value * polynomial, torch.lgamma(1 + regular))


def log_gamma_tails(shape: Tensor, point: Tensor) -> tuple[Tensor, Tensor]:
    """Log lower and upper regularized gamma functions.

    Computation uses double precision, a positive lower-tail series, and
    a modified Lentz upper-tail continued fraction. Both iteration counts
    are bounded; failure to reach double-precision convergence raises
    rather than returning a silently truncated probability. Differentiation
    follows the converged expressions, including the shape parameter.
    Constant exponential/Erlang-two shapes have direct survival formulas.
    """
    a, x = torch.broadcast_tensors(shape.double(), point.double().clamp_min(0))
    demand(
        torch.isfinite(a) & (a > 0) & ~torch.isnan(x),
        lambda: ValueError(
            "gamma log tails require finite positive shape and a nonnegative point"
        ),
    )
    if not a.requires_grad and everywhere(a == 1):
        log_q = -x
        return _complement(log_q), log_q
    if (
        not a.requires_grad
        and everywhere(a == 2)
        and everywhere((x >= 1e-4) | (x == 0))
    ):
        safe_x = torch.where(torch.isfinite(x), x, 1.0)
        log_q = torch.where(
            torch.isfinite(x), -safe_x + torch.log1p(safe_x), -torch.inf
        )
        return _complement(log_q), log_q

    positive = (x > 0) & torch.isfinite(x)
    near_one = positive & (a < 1) & (x <= 1)
    lower = positive & ~near_one & (x < a + 1) & ~((a < 0.01) & (x > 1))
    tolerance = 8 * torch.finfo(torch.float64).eps
    small = 1e-300
    iterations = 512

    near_log_p = torch.full_like(x, -1.0)
    if anywhere(near_one):
        near_a = torch.where(near_one, a, 0.5)
        near_x = torch.where(near_one, x, 0.5)
        factor = torch.ones_like(x)
        total = torch.zeros_like(x)
        compensation = torch.zeros_like(x)
        for n in range(1, iterations + 1):
            factor = factor * (-near_x / n)
            term = factor * near_a / (near_a + n)
            corrected = term - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
            if everywhere(term.abs() <= tolerance * total.abs()):
                break
        else:
            raise NotImplementedError(
                "gamma small-shape series did not converge within 512 iterations"
            )
        near_log_p = (
            near_a * near_x.log() - _log_gamma1p(near_a) + torch.log1p(total)
        ).clamp_max(0)

    # Mask inactive rows before the recurrences, so one tail cannot
    # overflow while its result is discarded by the final selection.
    series_a = torch.where(lower, a, 1.0)
    series_x = torch.where(lower, x, 0.0)
    term = 1 / series_a
    total = term
    for n in range(1, iterations + 1):
        term = term * series_x / (series_a + n)
        total = total + term
        if everywhere(term.abs() <= tolerance * total.abs()):
            break
    else:
        raise NotImplementedError(
            "gamma lower-tail series did not converge within 512 iterations"
        )
    safe_series_x = torch.where(lower, x, 1.0)
    log_p_series = (
        series_a * safe_series_x.log() - series_x - torch.lgamma(series_a) + total.log()
    )

    upper = positive & ~lower & ~near_one
    fraction_a = torch.where(upper, a, 1.0)
    fraction_x = torch.where(upper, x, 3.0)
    b = fraction_x + 1 - fraction_a
    c = torch.full_like(b, 1 / small)
    d = 1 / b
    fraction = d

    def nonzero(value: Tensor) -> Tensor:
        return torch.where(
            value.abs() < small, torch.where(value < 0, -small, small), value
        )

    for n in range(1, iterations + 1):
        coefficient = -n * (n - fraction_a)
        b = b + 2
        d = 1 / nonzero(coefficient * d + b)
        c = nonzero(b + coefficient / c)
        update = d * c
        fraction = fraction * update
        # At an integer shape the fraction can terminate exactly while
        # its shape derivatives still depend on subsequent coefficients.
        # Retain those terms when differentiating a learned shape.
        if (not a.requires_grad or n >= 64) and everywhere(
            (update - 1).abs() <= tolerance
        ):
            break
    else:
        raise NotImplementedError(
            "gamma upper-tail fraction did not converge within 512 iterations"
        )
    log_q_fraction = (
        fraction_a * fraction_x.log()
        - fraction_x
        - torch.lgamma(fraction_a)
        + fraction.log()
    )
    log_p = torch.where(
        lower, log_p_series, _complement(torch.where(upper, log_q_fraction, -1.0))
    )
    log_q = torch.where(
        upper, log_q_fraction, _complement(torch.where(lower, log_p_series, -1.0))
    )
    log_p = torch.where(near_one, near_log_p, log_p)
    log_q = torch.where(near_one, _complement(near_log_p), log_q)
    return (
        torch.where(positive, log_p, torch.where(x == 0, -torch.inf, 0.0)),
        torch.where(positive, log_q, torch.where(x == 0, 0.0, -torch.inf)),
    )
