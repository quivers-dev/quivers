"""Differentiable Beta log tails for truncation and censored responses."""

from __future__ import annotations

import torch
from torch import Tensor

from quivers._positions import demand, everywhere, reject_unless
from quivers.continuous._closed_probability import _log_gamma_increment, _stirling_error
from quivers.continuous._ordered import _log1mexp
from quivers.qiec._beta_tails import beta_fraction


def _log_front(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Retain the prefactor and its derivatives at concentrated Beta means."""
    large = (a >= 8) & (b >= 8)
    # One coherent orientation retains the full Hessian at a==b. The
    # averaged subgradients of minimum/maximum would project both shape
    # derivatives onto the diagonal and discard antisymmetric curvature.
    first = a <= b
    small = torch.where(first, a, b)
    other = torch.where(first, b, a)
    sa, sb = torch.where(large, 1.0, small), torch.where(large, 1.0, other)
    direct = (
        a * x.log()
        + b * torch.log1p(-x)
        - (torch.lgamma(sa) - _log_gamma_increment(sb, sa))
    )
    aa, bb = torch.where(large, a, 8.0), torch.where(large, b, 8.0)
    total = aa + bb
    p, q = aa / total, bb / total
    delta = x - p
    near = delta.abs() < 0.01 * torch.minimum(p, q)
    u = torch.where(near, delta / p, 0.0)
    v = torch.where(near, -delta / q, 0.0)

    def polynomial(value: Tensor) -> Tensor:
        result = torch.full_like(value, -1.0 / 18)
        for degree in range(17, 1, -1):
            result = (-1.0 if degree % 2 == 0 else 1.0) / degree + value * result
        return result

    centered = torch.where(
        near,
        aa * u.square() * polynomial(u) + bb * v.square() * polynomial(v),
        aa * (x.log() - p.log()) + bb * (torch.log1p(-x) - q.log()),
    )
    stable = (
        centered
        + 0.5 * (aa.log() + bb.log() - total.log())
        - 0.91893853320467274178
        - _stirling_error(aa)
        - _stirling_error(bb)
        + _stirling_error(total)
    )
    return torch.where(large, stable, direct)


def tensor_beta_fraction(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Evaluate the incomplete-Beta continued fraction over tensors.

    Parameters
    ----------
    a, b : Tensor
        Concentrations; ``b`` may be zero for a negative-binomial limit.
    x : Tensor
        Points below ``(a + 1) / (a + b + 2)``, where the fraction converges.

    Returns
    -------
    Tensor
        The continued fraction.

    Raises
    ------
    RuntimeError
        If an entry exhausts the recurrence's convergence budget.
    """

    def settled(change: Tensor) -> Tensor:
        """Whether the last update changed the fraction by rounding alone.

        Parameters
        ----------
        change : Tensor
            The multiplicative update.

        Returns
        -------
        Tensor
            Whether each entry has converged.
        """
        return (change - 1).abs() <= 8 * torch.finfo(torch.float64).eps

    return beta_fraction(
        a,
        b,
        x,
        nonzero=lambda value: torch.where(
            value.abs() < 1e-300, torch.where(value < 0, -1e-300, 1e-300), value
        ),
        converged=lambda change: everywhere(settled(change)),
        unconverged=lambda change, error: reject_unless(settled(change), lambda: error),
    )


def log_beta_tails(a: Tensor, b: Tensor, point: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate Beta log CDF and survival with gradients in all arguments.

    Parameters
    ----------
    a, b : Tensor
        Finite positive concentrations with a finite sum.
    point : Tensor
        Evaluation point, broadcast with the concentrations.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 log CDF and log survival, preserving exact support tails.

    Raises
    ------
    ValueError
        If concentrations are invalid or a point is NaN.
    RuntimeError
        If the continued fraction exceeds its bounded convergence budget.
    """
    a, b, point = torch.broadcast_tensors(a.double(), b.double(), point.double())
    demand(
        (a > 0) & (b > 0) & torch.isfinite(a + b) & ~point.isnan(),
        lambda: ValueError(
            "Beta tails require finite positive concentrations and a non-NaN point"
        ),
    )
    interior = (point > 0) & (point < 1)
    a, b = torch.where(interior, a, 1.0), torch.where(interior, b, 1.0)
    x = torch.where(interior, point, 0.5)
    swapped = x > (a + 1) / (a + b + 2)
    aa, bb = torch.where(swapped, b, a), torch.where(swapped, a, b)
    xx = torch.where(swapped, 1 - x, x)

    fraction = tensor_beta_fraction(aa, bb, xx)
    lower = (_log_front(aa, bb, xx) + fraction.log() - aa.log()).clamp_max(0)
    negative = lower < 0
    upper = torch.where(
        negative, _log1mexp(torch.where(negative, lower, -1.0)), -torch.inf
    )
    cdf, sf = torch.where(swapped, upper, lower), torch.where(swapped, lower, upper)
    return (
        torch.where(interior, cdf, torch.where(point <= 0, -torch.inf, 0.0)),
        torch.where(interior, sf, torch.where(point <= 0, 0.0, -torch.inf)),
    )
