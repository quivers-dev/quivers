"""Deterministic interval probabilities for scalar distributions."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import cast
from torch import Tensor
from torch.distributions import Distribution, Exponential, Gamma, Normal, Poisson

from quivers.continuous._ordered import _log1mexp
from quivers.continuous._truncated_normal import TruncatedNormal


class _PoissonTails(torch.autograd.Function):
    """Regularized-gamma tails with the finite Poisson boundary derivative."""

    @staticmethod
    def forward(ctx, rate: Tensor, count: Tensor) -> tuple[Tensor, Tensor]:
        ctx.save_for_backward(rate, count)
        return (
            torch.special.gammaincc(count + 1, rate),
            torch.special.gammainc(count + 1, rate),
        )

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        cdf_gradient, sf_gradient = grad_outputs
        rate, count = ctx.saved_tensors
        positive = rate > 0
        safe_rate = torch.where(positive, rate, 1.0)
        ordinary = torch.exp(
            count * safe_rate.log() - safe_rate - torch.lgamma(count + 1)
        )
        zero_rate = torch.where(positive, 0.0, rate)
        boundary = zero_rate.pow(count) * torch.exp(
            -zero_rate - torch.lgamma(count + 1)
        )
        mass = torch.where(positive, ordinary, boundary)
        return ((sf_gradient - cdf_gradient) * mass).sum_to_size(rate.shape), None


def _log_probability(probability: Tensor) -> Tensor:
    """Take log without leaking inactive zero-probability derivatives."""
    positive = probability > 0
    safe = torch.where(positive, probability, 1.0)
    return torch.where(positive, safe.log(), -torch.inf)


def _reference(base: Distribution) -> Tensor:
    """Find a parameter carrying the base's floating dtype and device."""
    for value in vars(base).values():
        if isinstance(value, Tensor) and value.is_floating_point():
            return value
    for attribute in ("base", "base_dist"):
        inner = getattr(base, attribute, None)
        if isinstance(inner, Distribution):
            return _reference(inner)
    raise NotImplementedError(
        f"{type(base).__name__} does not expose a floating distribution parameter"
    )


def _log_tails(base: Distribution, value: Tensor) -> tuple[Tensor, Tensor]:
    """Return log CDF and log survival, without Monte Carlo estimation."""
    log_cdf = getattr(base, "log_cdf", None)
    log_sf = getattr(base, "log_survival_function", None)
    if callable(log_cdf) and callable(log_sf):
        return cast(Callable[[Tensor], Tensor], log_cdf)(value), cast(
            Callable[[Tensor], Tensor], log_sf
        )(value)
    if isinstance(base, Exponential):
        exponent = -base.rate * value.clamp_min(0)
        return _log1mexp(exponent), exponent
    if isinstance(base, Gamma):
        from quivers.continuous._gamma_tails import log_gamma_tails

        return log_gamma_tails(
            base.concentration, base.rate.double() * value.double().clamp_min(0)
        )
    if isinstance(base, Poisson):
        nonnegative = value >= 0
        count = torch.where(nonnegative, value.floor(), 0.0)
        cdf, sf = _PoissonTails.apply(base.rate, count)
        # Out-of-support arguments must not introduce log(0) derivatives
        # in inactive branches of a parameter-dependent interval mass.
        cdf = torch.where(nonnegative, cdf, 1.0)
        sf = torch.where(nonnegative, sf, 1.0)
        return (
            torch.where(nonnegative, _log_probability(cdf), -torch.inf),
            torch.where(nonnegative, _log_probability(sf), 0.0),
        )
    probability = base.cdf(value)
    return _log_probability(probability), _log_probability(1 - probability)


def _log_difference(larger: Tensor, smaller: Tensor) -> Tensor:
    """Subtract probabilities in log space, preserving zero mass."""
    positive = torch.isfinite(larger) & (smaller < larger)
    safe_larger = torch.where(positive, larger, 0.0)
    safe_smaller = torch.where(positive, smaller, -1.0)
    result = safe_larger + _log1mexp(safe_smaller - safe_larger)
    return torch.where(positive, result, -torch.inf)


def _narrow_interval(
    base: Distribution, lower: Tensor, upper: Tensor
) -> tuple[Tensor, Tensor] | None:
    """Integrate a locally flat density without subtracting rounded tails."""
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
    else:
        return None
    if not bool(narrow.any()):
        return None
    midpoint = torch.where(narrow, lower + width / 2, 1.0)
    half_width = torch.where(narrow, width / 2, 0.1)
    nodes = midpoint.new_tensor(
        [
            -0.9602898564975363,
            -0.7966664774136267,
            -0.5255324099163290,
            -0.1834346424956498,
            0.1834346424956498,
            0.5255324099163290,
            0.7966664774136267,
            0.9602898564975363,
        ]
    )
    weights = midpoint.new_tensor(
        [
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.3626837833783620,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ]
    )
    node_shape = (8,) + (1,) * midpoint.ndim
    points = midpoint.unsqueeze(0) + half_width.unsqueeze(0) * nodes.reshape(node_shape)
    density = base.log_prob(points)
    result = half_width.log() + torch.logsumexp(
        density + weights.log().reshape(node_shape), dim=0
    )
    return narrow, result


def log_interval_mass(
    base: Distribution, low: Tensor | float | None, high: Tensor | float | None
) -> Tensor:
    """Log mass of an inclusive interval for a scalar-event distribution.

    Independent event coordinates can sum these log masses. A coupled
    event requires a rectangle probability and is deliberately refused.
    Specialized log-tail methods avoid cancellation; the generic CDF
    path retains exact zero probabilities rather than flooring them.
    """
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
    if isinstance(base, Normal):
        lower = reference.new_tensor(-torch.inf) if lower is None else lower
        upper = reference.new_tensor(torch.inf) if upper is None else upper
        ordinary = TruncatedNormal(base.loc, base.scale, lower, upper).log_normalizer()
        narrow = _narrow_interval(base, lower, upper)
        return (
            ordinary if narrow is None else torch.where(narrow[0], narrow[1], ordinary)
        )
    if getattr(base.support, "is_discrete", False) and lower is not None:
        lower = lower.ceil() - 1
    if lower is None and upper is None:
        return torch.zeros(
            base.batch_shape, device=reference.device, dtype=reference.dtype
        )
    if lower is None:
        assert upper is not None
        return _log_tails(base, upper)[0]
    if upper is None:
        return _log_tails(base, lower)[1]
    low_cdf, low_sf = _log_tails(base, lower)
    high_cdf, high_sf = _log_tails(base, upper)
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
