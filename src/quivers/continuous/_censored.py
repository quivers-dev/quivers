"""Scalar censoring as a pushforward with exact boundary probability atoms."""

from __future__ import annotations

import torch
from typing import cast
from torch import Tensor
from torch.distributions import (
    Distribution,
    TransformedDistribution,
    biject_to,
    constraints,
)

from quivers.continuous._interval import _log_tails, _reference


def _interior_point(base: Distribution, reference: Tensor) -> Tensor:
    """Choose a deterministic scoring point inside the base support."""
    if isinstance(base, TransformedDistribution):
        value = _interior_point(base.base_dist, reference)
        for transform in base.transforms:
            value = cast(Tensor, transform(value))
        return value
    support = base.support
    if support is None:
        raise NotImplementedError("censoring requires a declared scalar support")
    if support.is_discrete:
        return torch.zeros(
            base.batch_shape, dtype=reference.dtype, device=reference.device
        ) + getattr(support, "lower_bound", 0)
    return biject_to(support)(
        torch.zeros(base.batch_shape, dtype=reference.dtype, device=reference.device)
    )


class Censored(Distribution):
    """Clamp a scalar law to fixed bounds, retaining exact tail atoms.

    Parameters
    ----------
    base : Distribution
        Scalar-event probability law with a deterministic CDF capability.
    low, high : Tensor or float or None
        Ordered censoring bounds. None leaves the corresponding side open.

    Raises
    ------
    ValueError
        If the base has a coupled event, a bound is NaN, or bounds are unordered.
    """

    @property
    def arg_constraints(self):
        """Delegate parameter validation to the underlying distribution."""
        return {}

    def __init__(self, base: Distribution, low=None, high=None):
        if base.event_shape:
            raise ValueError("scalar censoring requires scalar events")
        reference = _reference(base)
        self.low = torch.as_tensor(
            -torch.inf if low is None else low,
            dtype=reference.dtype,
            device=reference.device,
        )
        self.high = torch.as_tensor(
            torch.inf if high is None else high,
            dtype=reference.dtype,
            device=reference.device,
        )
        shape = torch.broadcast_shapes(
            base.batch_shape, self.low.shape, self.high.shape
        )
        self.base = base.expand(shape)
        self.low, self.high = self.low.expand(shape), self.high.expand(shape)
        if not bool((self.low < self.high).all()):
            raise ValueError("censoring bounds must be ordered and non-NaN")
        self.has_rsample = base.has_rsample
        super().__init__(
            batch_shape=shape, event_shape=torch.Size(), validate_args=False
        )

    @property
    def support(self):
        """Closed response interval, including the censoring atoms."""
        return constraints.interval(self.low, self.high)

    def expand(self, batch_shape, _instance=None):
        """Broadcast the same base law and censoring bounds."""
        return Censored(
            self.base.expand(batch_shape),
            self.low.expand(batch_shape),
            self.high.expand(batch_shape),
        )

    def log_prob(self, value):
        """Score boundary probability masses and interior base densities."""
        value = torch.as_tensor(value, dtype=self.low.dtype, device=self.low.device)
        if bool(value.isnan().any()):
            raise ValueError("censored response values must not be NaN")
        safe = _interior_point(self.base, self.low)
        interior = (value > self.low) & (value < self.high)
        density = self.base.log_prob(torch.where(interior, value, safe))
        lower = _log_tails(
            self.base, torch.where(torch.isfinite(self.low), self.low, safe)
        )[0]
        upper_point = (
            self.high.ceil() - 1 if self.base.support.is_discrete else self.high
        )
        upper = _log_tails(
            self.base, torch.where(torch.isfinite(upper_point), upper_point, safe)
        )[1]
        lower = torch.where(torch.isfinite(self.low), lower, -torch.inf)
        upper = torch.where(torch.isfinite(self.high), upper, -torch.inf)
        result = torch.where(
            value == self.low, lower, torch.where(value == self.high, upper, density)
        )
        return torch.where(
            (value >= self.low) & (value <= self.high) & torch.isfinite(value),
            result,
            -torch.inf,
        )

    def _tails(self, value):
        """Keep exact CDF support boundaries without invalid inactive scores."""
        value = torch.as_tensor(value, dtype=self.low.dtype, device=self.low.device)
        if bool(value.isnan().any()):
            raise ValueError("censored response values must not be NaN")
        interior = (value >= self.low) & (value < self.high) & torch.isfinite(value)
        below = (value < self.low) | torch.isneginf(value)
        safe = _interior_point(self.base, self.low)
        cdf, sf = _log_tails(self.base, torch.where(interior, value, safe))
        return (
            torch.where(interior, cdf, torch.where(below, -torch.inf, 0.0)),
            torch.where(interior, sf, torch.where(below, 0.0, -torch.inf)),
        )

    def log_cdf(self, value):
        """Evaluate the CDF including the lower boundary atom."""
        return self._tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate strict survival, excluding the upper boundary atom."""
        return self._tails(value)[1]

    def cdf(self, value):
        """Evaluate the CDF of the censored response."""
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Map base quantiles to the censoring interval."""
        return self.base.icdf(value).clamp(self.low, self.high)

    def sample(self, sample_shape=torch.Size()):
        """Draw from the censoring pushforward, including its boundary atoms."""
        with torch.no_grad():
            return self.base.sample(sample_shape).clamp(self.low, self.high)

    def rsample(self, sample_shape=torch.Size()):
        """Apply censoring to a pathwise base draw."""
        return self.base.rsample(sample_shape).clamp(self.low, self.high)
