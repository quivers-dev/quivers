"""Positive quadrature for scalar laws with two endpoint atoms."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8


def _positive_unit_moments(
    log_density: Callable[[Tensor, Tensor], Tensor],
    log_error: Callable[[Tensor, Tensor, Tensor], Tensor],
    lower_mass: Tensor,
    upper_mass: Tensor,
    *,
    atol: float,
    rtol: float,
    max_panels: int,
) -> tuple[Tensor, Tensor]:
    """Integrate bounded moments with explicit per-panel quadrature errors.

    Parameters
    ----------
    log_density : Callable
        Evaluates log densities at points on the final Gauss-node axis and
        selected flattened batch rows on the leading axis.
    log_error : Callable
        Bounds absolute Gauss8 errors for density times one, y, y-squared,
        or a centered square with center in the unit interval.
    lower_mass, upper_mass : Tensor
        Flattened probability masses at zero and one.
    atol, rtol : float
        Absolute and relative error budgets for the quadrature integrals.
    max_panels : int
        Maximum accepted and pending panels per flattened batch row.

    Returns
    -------
    tuple[Tensor, Tensor]
        Mean and variance; positive central-moment contributions avoid
        subtracting nearly equal raw moments.

    Raises
    ------
    RuntimeError
        If certification exceeds the panel budget or interval resolution.
    """
    rows = len(lower_mass)
    indices = torch.arange(rows, device=lower_mass.device)
    left, right = torch.zeros_like(lower_mass), torch.ones_like(lower_mass)
    nodes, weights = lower_mass.new_tensor(_NODES_8), lower_mass.new_tensor(_WEIGHTS_8)
    accepted_indices, masses, means, centered = [], [], [], []
    panels = torch.ones(rows, device=lower_mass.device, dtype=torch.long)
    while indices.numel():
        midpoint, half = (left + right) / 2, (right - left) / 2
        point = midpoint[:, None] + half[:, None] * nodes
        log_weights = log_density(point, indices) + weights.log() + half.log()[:, None]
        log_mass = torch.logsumexp(log_weights, -1)
        finite = torch.isfinite(log_mass)
        relative = torch.softmax(torch.where(finite[:, None], log_weights, 0.0), -1)
        # No finite density is representable when every log weight is -inf.
        relative = torch.where(finite[:, None], relative, 0.0)
        mean = (relative * point).sum(-1)
        variance = (relative * (point - mean[:, None]).square()).sum(-1)
        with torch.no_grad():
            error = log_error(left, right, indices)
            budget = torch.logaddexp(
                (right - left).log() + math.log(atol),
                log_mass.detach() + math.log(rtol),
            )
            good = error <= budget
            if bool(torch.isnan(error).any()):
                raise RuntimeError(
                    "bounded moment derivative error could not be represented"
                )
        if bool(good.any()):
            accepted_indices.append(indices[good])
            mass = log_mass[good].exp()
            masses.append(mass)
            means.append(mean[good])
            centered.append(mass * variance[good])
        pending = ~good
        if not bool(pending.any()):
            break
        indices, lo, hi, mid = (
            indices[pending],
            left[pending],
            right[pending],
            midpoint[pending],
        )
        with torch.no_grad():
            panels.scatter_add_(0, indices, torch.ones_like(indices))
            if bool((panels > max_panels).any()):
                raise RuntimeError(
                    "bounded moment quadrature could not certify its panel budget"
                )
            if bool(((mid == lo) | (mid == hi)).any()):
                raise RuntimeError(
                    "bounded moment quadrature exhausted Float64 interval resolution"
                )
        indices = torch.cat((indices, indices))
        left, right = torch.cat((lo, mid)), torch.cat((mid, hi))
    at = torch.cat(accepted_indices)
    mass, mean, central = torch.cat(masses), torch.cat(means), torch.cat(centered)
    zero = torch.zeros_like(lower_mass)
    interior_mass = zero.scatter_add(0, at, mass)
    total = lower_mass + upper_mass + interior_mass
    first = zero.scatter_add(0, at, mass * mean)
    complement = zero.scatter_add(0, at, mass * (1 - mean))
    location = torch.where(
        lower_mass >= upper_mass,
        (upper_mass + first) / total,
        1 - (lower_mass + complement) / total,
    )
    between = central + mass * (mean - location[at]).square()
    variance = (
        lower_mass * location.square()
        + upper_mass * (1 - location).square()
        + zero.scatter_add(0, at, between)
    ) / total
    return location, variance


__all__: list[str] = []
