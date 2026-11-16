"""Deterministic interval probabilities for scalar distributions."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import cast
from torch import Tensor
from torch.distributions import (
    AffineTransform,
    Beta,
    Distribution,
    Exponential,
    Gamma,
    Normal,
    TransformedDistribution,
    Uniform,
)

from quivers.continuous._ordered import _log1mexp
from quivers.continuous._truncated_normal import TruncatedNormal
from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8


def _log_probability(probability: Tensor) -> Tensor:
    """Take log without leaking inactive zero-probability derivatives.

    Parameters
    ----------
    probability : Tensor
        Probability values, including exact support endpoints.

    Returns
    -------
    Tensor
        Log probabilities with exact negative infinity at zero."""
    positive = probability > 0
    safe = torch.where(positive, probability, 1.0)
    return torch.where(positive, safe.log(), -torch.inf)


def _reference(base: Distribution) -> Tensor:
    """Find a parameter carrying the base's floating dtype and device.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.

    Returns
    -------
    Tensor
        Floating parameter carrying the distribution dtype and device.

    Raises
    ------
    NotImplementedError
        If no floating or integer parameter supplies dtype and device."""
    pending = [base]
    visited: set[int] = set()
    integer_reference: Tensor | None = None
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        values = tuple(vars(current).values())
        for value in values:
            if isinstance(value, Tensor) and value.is_floating_point():
                return value
            if isinstance(value, Tensor) and integer_reference is None:
                integer_reference = value
        pending.extend(
            value
            for value in (
                *values,
                getattr(current, "base", None),
                getattr(current, "base_dist", None),
            )
            if isinstance(value, Distribution)
        )
    if integer_reference is not None:
        return integer_reference.to(dtype=torch.get_default_dtype())
    raise NotImplementedError(
        f"{type(base).__name__} does not expose a floating distribution parameter"
    )


def _log_tails(base: Distribution, value: Tensor) -> tuple[Tensor, Tensor]:
    """Return log CDF and log survival, without Monte Carlo estimation.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    value : Tensor
        Evaluation points, broadcast against the distribution batch.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log lower and strict upper tail masses."""
    log_cdf = getattr(base, "log_cdf", None)
    log_sf = getattr(base, "log_survival_function", None)
    if callable(log_cdf) and callable(log_sf):
        return cast(Callable[[Tensor], Tensor], log_cdf)(value), cast(
            Callable[[Tensor], Tensor], log_sf
        )(value)
    if isinstance(base, Normal):
        standardized = (value.double() - base.loc.double()) / base.scale.double()
        return torch.special.log_ndtr(standardized), torch.special.log_ndtr(
            -standardized
        )
    if isinstance(base, Beta):
        from quivers.continuous._beta_tails import log_beta_tails

        return log_beta_tails(base.concentration1, base.concentration0, value)
    if isinstance(base, TransformedDistribution) and all(
        isinstance(transform, AffineTransform)
        and bool((torch.as_tensor(transform.scale) > 0).all())
        for transform in base.transforms
    ):
        inverse = value
        for transform in reversed(base.transforms):
            inverse = cast(Tensor, transform.inv(inverse))
        return _log_tails(base.base_dist, inverse)
    if isinstance(base, Uniform):
        lower, upper = base.low.double(), base.high.double()
        point = value.double()
        inside = (point > lower) & (point < upper)
        safe = torch.where(inside, point, lower + (upper - lower) / 2)
        width = (upper - lower).log()
        cdf = (safe - lower).log() - width
        sf = (upper - safe).log() - width
        return (
            torch.where(inside, cdf, torch.where(point <= lower, -torch.inf, 0.0)),
            torch.where(inside, sf, torch.where(point <= lower, 0.0, -torch.inf)),
        )
    if isinstance(base, Exponential):
        exponent = -base.rate * value.clamp_min(0)
        return _log1mexp(exponent), exponent
    if isinstance(base, Gamma):
        from quivers.continuous._gamma_tails import log_gamma_tails

        return log_gamma_tails(
            base.concentration, base.rate.double() * value.double().clamp_min(0)
        )
    from quivers.continuous._count_tails import distribution_count_log_tails

    count_tails = distribution_count_log_tails(base, value)
    if count_tails is not None:
        return count_tails
    probability = base.cdf(value)
    return _log_probability(probability), _log_probability(1 - probability)


def _log_difference(larger: Tensor, smaller: Tensor) -> Tensor:
    """Subtract probabilities in log space, preserving zero mass.

    Parameters
    ----------
    larger : Tensor
        Log mass of the containing event.
    smaller : Tensor
        Log mass of the nested event to subtract.

    Returns
    -------
    Tensor
        Log of the positive mass difference, or negative infinity."""
    positive = torch.isfinite(larger) & (smaller < larger)
    safe_larger = torch.where(positive, larger, 0.0)
    safe_smaller = torch.where(positive, smaller, -1.0)
    result = safe_larger + _log1mexp(safe_smaller - safe_larger)
    return torch.where(positive, result, -torch.inf)


def _log_tails_left(base: Distribution, value: Tensor) -> tuple[Tensor, Tensor]:
    """Return log probabilities of ``Y < value`` and ``Y >= value``.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    value : Tensor
        Evaluation points, broadcast against the distribution batch.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log masses strictly below and at or above the cutoff."""
    log_cdf = getattr(base, "log_cdf_left", None)
    log_sf = getattr(base, "log_survival_function_left", None)
    if callable(log_cdf) and callable(log_sf):
        return cast(Callable[[Tensor], Tensor], log_cdf)(value), cast(
            Callable[[Tensor], Tensor], log_sf
        )(value)
    if getattr(base.support, "is_discrete", False):
        return _log_tails(base, value.ceil() - 1)
    return _log_tails(base, value)


def _closed_flag(value: bool | Tensor, name: str) -> bool | Tensor:
    """Validate scalar or batched Boolean interval endpoint metadata.

    Parameters
    ----------
    value : bool | Tensor
        Evaluation points, broadcast against the distribution batch.
    name : str
        Parameter name used in a diagnostic.

    Returns
    -------
    bool | Tensor
        Validated Boolean scalar or tensor endpoint policy.

    Raises
    ------
    TypeError
        If the policy is neither Boolean nor a Boolean tensor."""
    if type(value) is bool or isinstance(value, Tensor) and value.dtype == torch.bool:
        return value
    raise TypeError(f"{name} must be a bool or a Boolean tensor")


def _endpoint_tails(
    base: Distribution, value: Tensor, left: bool | Tensor
) -> tuple[Tensor, Tensor]:
    """Select one-sided limits while preserving explicit row endpoint flags.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    value : Tensor
        Evaluation points, broadcast against the distribution batch.
    left : bool | Tensor
        Whether to select the CDF left limit.

    Returns
    -------
    tuple[Tensor, Tensor]
        Selected log CDF and survival limits."""
    if isinstance(left, bool):
        return _log_tails_left(base, value) if left else _log_tails(base, value)
    left_cdf, left_sf = _log_tails_left(base, value)
    right_cdf, right_sf = _log_tails(base, value)
    flag = left.to(value.device)
    return torch.where(flag, left_cdf, right_cdf), torch.where(flag, left_sf, right_sf)


def _narrow_interval(
    base: Distribution, lower: Tensor, upper: Tensor
) -> tuple[Tensor, Tensor] | None:
    """Integrate a locally flat density without subtracting rounded tails.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    lower : Tensor
        Whether the intersection concerns lower bounds.
    upper : Tensor
        Input to the retained distribution calculation.

    Returns
    -------
    tuple[Tensor, Tensor] | None
        Local quadrature mask and log masses, or None when no local rule applies."""
    width = upper - lower
    if isinstance(base, Gamma):
        shape, rate = base.concentration.double(), base.rate.double()
        safe_lower = lower.clamp_min(torch.finfo(torch.float64).tiny)
        variation = width * ((shape - 1).abs() / safe_lower + rate)
        narrow = (lower > 0) & (width > 0) & (variation < 1e-4)
    elif isinstance(base, Normal):
        location, scale = base.loc.double(), base.scale.double()
        variation = (
            width
            * torch.maximum((lower - location).abs(), (upper - location).abs())
            / scale.square()
        )
        narrow = (width > 0) & (variation < 1e-4)
    elif isinstance(base, Beta):
        safe_lower = lower.clamp_min(torch.finfo(torch.float64).tiny)
        safe_remaining = (1 - upper).clamp_min(torch.finfo(torch.float64).tiny)
        variation = width * (
            (base.concentration1.double() - 1).abs() / safe_lower
            + (base.concentration0.double() - 1).abs() / safe_remaining
        )
        narrow = (lower > 0) & (upper < 1) & (width > 0) & (variation < 1e-4)
    else:
        return None
    if not bool(narrow.any()):
        return None
    midpoint = torch.where(narrow, lower + width / 2, 0.5)
    half_width = torch.where(narrow, width / 2, 0.1)
    nodes = midpoint.new_tensor(_NODES_8)
    weights = midpoint.new_tensor(_WEIGHTS_8)
    node_shape = (8,) + (1,) * midpoint.ndim
    points = midpoint.unsqueeze(0) + half_width.unsqueeze(0) * nodes.reshape(node_shape)
    if isinstance(base, Beta):
        from quivers.continuous._beta_tails import _log_front

        density = (
            _log_front(
                base.concentration1.double(), base.concentration0.double(), points
            )
            - points.log()
            - torch.log1p(-points)
        )
    else:
        density = base.log_prob(points)
    result = half_width.log() + torch.logsumexp(
        density + weights.log().reshape(node_shape), dim=0
    )
    return narrow, result


def log_interval_mass(
    base: Distribution,
    low: Tensor | float | None,
    high: Tensor | float | None,
    *,
    lower_closed: bool | Tensor = True,
    upper_closed: bool | Tensor = True,
) -> Tensor:
    """Log mass of a scalar interval with explicit endpoint membership.

    Independent event coordinates can sum these log masses. A coupled
    event requires a rectangle probability and is deliberately refused.
    Specialized log-tail methods avoid cancellation; the generic CDF
    path retains exact zero probabilities rather than flooring them.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    low : Tensor | float | None
        Lower event bound; None leaves that side unbounded.
    high : Tensor | float | None
        Upper event bound; None leaves that side unbounded.
    lower_closed : bool | Tensor
        Whether the lower endpoint belongs to the event.
    upper_closed : bool | Tensor
        Whether the upper endpoint belongs to the event.

    Returns
    -------
    Tensor
        Raw log mass on the requested interval, retaining independent batch axes.

    Raises
    ------
    NotImplementedError, ValueError
        If an endpoint is NaN or the base lacks the required interval capability."""
    lower_closed = _closed_flag(lower_closed, "lower_closed")
    upper_closed = _closed_flag(upper_closed, "upper_closed")
    interval = getattr(base, "log_interval_mass", None)
    if callable(interval):
        return interval(low, high, lower_closed=lower_closed, upper_closed=upper_closed)
    if base.event_shape:
        raise NotImplementedError(
            "interval normalization requires scalar events; a coupled event "
            "requires a joint rectangle probability"
        )
    reference = _reference(base)
    lower = (
        None
        if low is None
        else torch.as_tensor(low, device=reference.device, dtype=torch.float64)
    )
    upper = (
        None
        if high is None
        else torch.as_tensor(high, device=reference.device, dtype=torch.float64)
    )
    if (
        lower is not None
        and bool(lower.isnan().any())
        or upper is not None
        and bool(upper.isnan().any())
    ):
        raise ValueError("interval endpoints must not be NaN")
    if isinstance(base, Normal):
        lower = reference.new_tensor(-torch.inf) if lower is None else lower
        upper = reference.new_tensor(torch.inf) if upper is None else upper
        ordered = lower < upper
        ordinary = TruncatedNormal(
            base.loc,
            base.scale,
            torch.where(ordered, lower, 0.0),
            torch.where(ordered, upper, 1.0),
        ).log_normalizer()
        ordinary = torch.where(ordered, ordinary, -torch.inf)
        narrow = _narrow_interval(base, lower, upper)
        return (
            ordinary if narrow is None else torch.where(narrow[0], narrow[1], ordinary)
        )
    if lower is None and upper is None:
        return torch.zeros(
            base.batch_shape, device=reference.device, dtype=reference.dtype
        )
    if lower is None:
        assert upper is not None
        left = not upper_closed if isinstance(upper_closed, bool) else ~upper_closed
        return _endpoint_tails(base, upper, left)[0]
    if upper is None:
        return _endpoint_tails(base, lower, lower_closed)[1]
    low_cdf, low_sf = _endpoint_tails(base, lower, lower_closed)
    left = not upper_closed if isinstance(upper_closed, bool) else ~upper_closed
    high_cdf, high_sf = _endpoint_tails(base, upper, left)
    # Use the side with the smaller cumulative probability: a right-tail
    # interval subtracts survival probabilities, never rounded CDFs near 1.
    use_cdf = high_cdf < low_sf
    ordinary = torch.where(
        use_cdf,
        _log_difference(
            torch.where(use_cdf, high_cdf, 0.0), torch.where(use_cdf, low_cdf, -1.0)
        ),
        _log_difference(
            torch.where(use_cdf, 0.0, low_sf), torch.where(use_cdf, -1.0, high_sf)
        ),
    )
    narrow = _narrow_interval(base, lower, upper)
    return ordinary if narrow is None else torch.where(narrow[0], narrow[1], ordinary)
