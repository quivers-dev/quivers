"""Real retained-sample conversion shared by empirical posterior diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from quivers._count_capacity import _MAX_EXACT_COUNT


def real_samples(
    values: torch.Tensor, *, label: str, finite: bool = True
) -> torch.Tensor:
    """Detach real samples with shared integer and optional finiteness checks.

    Parameters
    ----------
    values : torch.Tensor
        Real samples of arbitrary shape, including Boolean or integer draws.
    label : str
        Diagnostic name used in validation errors.
    finite : bool
        Require finite values. Summary diagnostics may instead retain invalid
        coordinates and report NaN without deleting any draws.

    Returns
    -------
    torch.Tensor
        Float64 values on the original device, without an autograd graph.

    Raises
    ------
    TypeError
        If values are not a real tensor.
    ValueError
        If finiteness is required and values are nonfinite, or an original
        integer exceeds the exact Float64 range. Missing values are not
        deleted or imputed.
    """
    if not isinstance(values, torch.Tensor) or values.is_complex():
        raise TypeError(f"{label} requires a real sample tensor")
    values = values.detach()
    if not values.is_floating_point():
        integers = values.to(torch.int64)
        unsigned_overflow = values.dtype == torch.uint64 and bool((integers < 0).any())
        if unsigned_overflow or bool(
            ((integers < -_MAX_EXACT_COUNT) | (integers > _MAX_EXACT_COUNT)).any()
        ):
            raise ValueError(f"{label} integer inputs must lie within +/-2**53")
    result = values.to(dtype=torch.float64)
    if finite and not bool(torch.isfinite(result).all()):
        raise ValueError(
            f"{label} requires finite samples; missing values are not deleted"
        )
    return result


def probabilities(probs: Sequence[float]) -> tuple[float, ...]:
    """Validate interior empirical quantile probabilities.

    Parameters
    ----------
    probs : Sequence[float]
        Requested probabilities, retaining their order and repetitions.

    Returns
    -------
    tuple[float, ...]
        Validated probabilities; an empty sequence is permitted.

    Raises
    ------
    ValueError
        If a probability is Boolean, nonfinite, or outside the open unit interval.
    """
    levels = tuple(probs)
    if any(
        isinstance(p, bool) or not math.isfinite(p) or not 0 < p < 1 for p in levels
    ):
        raise ValueError("summary probabilities must lie strictly between zero and one")
    return levels


def linear_quantile(sorted_values: torch.Tensor, prob: float) -> torch.Tensor:
    """Interpolate a type-7 quantile without overflowing opposite-sign endpoints.

    Parameters
    ----------
    sorted_values : torch.Tensor
        Values sorted ascending along a nonempty first axis.
    prob : float
        Probability in the closed unit interval.

    Returns
    -------
    torch.Tensor
        The interpolated quantile with the first axis reduced.
    """
    position = (sorted_values.shape[0] - 1) * prob
    lower = min(math.floor(position), sorted_values.shape[0] - 1)
    fraction = position - lower
    below = sorted_values[lower]
    if fraction == 0:
        return below
    above = sorted_values[min(lower + 1, sorted_values.shape[0] - 1)]
    opposite = torch.signbit(below) != torch.signbit(above)
    # same-sign subtraction preserves small spreads near large offsets.
    difference = torch.where(opposite, torch.zeros_like(above), above - below)
    interpolated = (
        above - difference * (1 - fraction)
        if fraction >= 0.5
        else below + difference * fraction
    )
    return torch.where(
        opposite, (1 - fraction) * below + fraction * above, interpolated
    )


def scaled_samples(
    pooled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center and scale retained draws without overflowing an opposite-sign span.

    Parameters
    ----------
    pooled : torch.Tensor
        Real values with a nonempty leading sample axis. Finiteness and
        original integer capacity remain the caller's validation policy.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Normalized values, retained offset, and nonnegative scale. When all
        draws agree, the scale is zero and normalized values are zero.
        An overflowing subtraction uses zero offset and scales original values.
    """
    anchor = pooled[0]
    deltas = pooled - anchor
    shifted = torch.isfinite(deltas).all(0)
    working = torch.where(shifted, deltas, pooled)
    scale = working.abs().amax(0)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    return working / divisor, torch.where(shifted, anchor, 0), scale


def empirical_summary(
    pooled: torch.Tensor, levels: Sequence[float], *, robust: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize retained draws using scaled arithmetic and shared type-7 quantiles.

    Parameters
    ----------
    pooled : torch.Tensor
        Float64 values with a nonempty leading sample axis.
    levels : Sequence[float]
        Validated quantile probabilities.
    robust : bool
        Use median and normal-consistent median absolute deviation instead of
        mean and sample standard deviation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Location, dispersion, and quantiles on the remaining event axes.
        One draw has undefined sample standard deviation, represented by NaN.
        A coordinate with any nonfinite draw returns NaN for every summary,
        without deleting or imputing draws.
    """
    normalized, offset, scale = scaled_samples(pooled)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    ordered = pooled.sort(dim=0).values if levels or robust else None
    if robust:
        assert ordered is not None
        estimate = linear_quantile(ordered, 0.5)
        ordered_normalized = (ordered - offset) / divisor
        normalized_median = linear_quantile(ordered_normalized, 0.5)
        deviations = (normalized - normalized_median).abs().sort(dim=0).values
        dispersion = linear_quantile(deviations, 0.5) * 1.482602218505602 * scale
    else:
        center = normalized.mean(0) * scale
        estimate = offset + center
        dispersion = (
            normalized.std(dim=0, correction=1) * scale
            if pooled.shape[0] > 1
            else torch.full_like(estimate, torch.nan)
        )
    quantiles = (
        torch.stack([linear_quantile(ordered, p) for p in levels])
        if ordered is not None and levels
        else pooled.new_empty((0, *pooled.shape[1:]))
    )
    finite = torch.isfinite(pooled).all(0)
    return (
        torch.where(finite, estimate, torch.nan),
        torch.where(finite, dispersion, torch.nan),
        torch.where(finite, quantiles, torch.nan),
    )


__all__: list[str] = []
