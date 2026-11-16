"""Recorded outcome events scored through the existing measure-mass algebra."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution

from quivers.continuous._censored import _interior_point
from quivers.continuous._interval import _reference, log_interval_mass


def event_log_prob(
    base: Distribution,
    value: Tensor,
    *,
    status: Tensor | int | None = None,
    interval_upper: Tensor | float | None = None,
    trunc_lower: Tensor | float | None = None,
    trunc_upper: Tensor | float | None = None,
) -> Tensor:
    """Score exact outcomes or recorded censoring events under truncation.

    The event numerator and truncation denominator are masses of the same
    underlying measure. Right censoring means Y>value, left means Y<=value,
    and interval censoring means value<Y<=interval_upper. Truncation includes
    both endpoints. This function applies no likelihood weights or reductions.

    Parameters
    ----------
    base : torch.distributions.Distribution
        Scalar-event law with the existing interval-mass capability.
    value : Tensor
        Exact outcome or first censoring endpoint, at its checked batch layout.
    status : Tensor or int or None
        Integral codes -1 left, 0 exact, 1 right, 2 interval; None means exact.
    interval_upper : Tensor or float or None
        Upper endpoints, required only for rows with interval censoring.
    trunc_lower, trunc_upper : Tensor or float or None
        Inclusive sampling bounds at the same checked batch layout.

    Returns
    -------
    Tensor
        Unweighted log likelihood per independent batch entry.

    Raises
    ------
    ValueError
        If codes, bounds, or layouts are invalid, or truncation has zero mass.
    NotImplementedError
        If the response has a coupled event or lacks interval-mass support.
    """
    if base.event_shape:
        raise NotImplementedError(
            "response censoring/truncation requires scalar events"
        )
    reference = _reference(base)
    value = torch.as_tensor(value, device=reference.device)
    support = base.support
    if support is None:
        raise NotImplementedError("response law does not expose its support")
    if value.dtype == torch.bool or (
        not support.is_discrete and not value.is_floating_point()
    ):
        value = value.to(reference.dtype)
    code = (
        torch.zeros_like(value, dtype=torch.long)
        if status is None
        else torch.as_tensor(status, device=value.device)
    )
    if (
        code.dtype == torch.bool
        or code.is_floating_point()
        or code.is_complex()
        or not bool(((code >= -1) & (code <= 2)).all())
    ):
        raise ValueError("censoring status requires integer codes -1, 0, 1, or 2")
    code = code.long()
    shape = torch.broadcast_shapes(base.batch_shape, value.shape, code.shape)
    value, code = value.expand(shape), code.expand(shape)
    left = torch.as_tensor(
        -torch.inf if trunc_lower is None else trunc_lower,
        device=reference.device,
        dtype=None if isinstance(trunc_lower, Tensor) else torch.float64,
    )
    right = torch.as_tensor(
        torch.inf if trunc_upper is None else trunc_upper,
        device=reference.device,
        dtype=None if isinstance(trunc_upper, Tensor) else torch.float64,
    )
    if bool(left.isnan().any() | right.isnan().any()) or bool((left > right).any()):
        raise ValueError("truncation bounds must be ordered and not NaN")
    if torch.broadcast_shapes(shape, left.shape, right.shape) != shape:
        raise ValueError("truncation bounds introduce undeclared response batch axes")
    left, right = left.expand(shape), right.expand(shape)
    truncated = trunc_lower is not None or trunc_upper is not None
    denominator = (
        log_interval_mass(base, left, right)
        if truncated
        else reference.new_zeros(shape)
    )
    if truncated and not bool(torch.isfinite(denominator).all()):
        raise ValueError(
            "truncation interval must have positive finite mass in every row"
        )
    exact = code == 0
    result = reference.new_full(shape, -torch.inf)
    if bool(exact.any()):
        supported = support.check(value)
        accepted = exact & supported & (value >= left) & (value <= right)
        point = value
        if not bool(accepted.all()):
            interior = _interior_point(base, reference)
            point = torch.where(accepted, value, interior.to(value.dtype))
        density = base.log_prob(point)
        result = torch.where(accepted, density - denominator, result)
    censored = ~exact
    if bool(censored.any()):
        intervals = code == 2
        if bool(intervals.any()) and interval_upper is None:
            raise ValueError("interval censoring requires an upper endpoint")
        upper = (
            value
            if interval_upper is None
            else torch.as_tensor(
                interval_upper,
                device=value.device,
                dtype=None if isinstance(interval_upper, Tensor) else torch.float64,
            )
        )
        if torch.broadcast_shapes(shape, upper.shape) != shape:
            raise ValueError(
                "interval endpoints introduce undeclared response batch axes"
            )
        upper = upper.expand(shape)
        if bool(
            (intervals & (~upper.isfinite() | value.isnan() | (upper <= value))).any()
        ):
            raise ValueError(
                "interval censoring requires an upper endpoint greater than its lower endpoint"
            )
        lower_event = torch.where((code == 1) | intervals, value, -torch.inf)
        upper_event = torch.where(
            code == -1, value, torch.where(intervals, upper, torch.inf)
        )
        # Intersect the recorded event with the sampling support. An active
        # censoring lower endpoint is open even when it equals a closed
        # truncation endpoint. Inactive rows query the whole base law.
        low = torch.where(censored, torch.maximum(lower_event, left), -torch.inf)
        high = torch.where(censored, torch.minimum(upper_event, right), torch.inf)
        lower_closed = ~(((code == 1) | intervals) & (value >= left))
        mass = log_interval_mass(base, low, high, lower_closed=lower_closed)
        result = torch.where(censored, mass - denominator, result)
    return result
