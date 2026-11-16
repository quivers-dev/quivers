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

from quivers.continuous._interval import _log_tails, _log_tails_left, _reference


def _interior_point(base: Distribution, reference: Tensor) -> Tensor:
    """Choose a deterministic scoring point inside the base support.

    Parameters
    ----------
    base : Distribution
        Underlying distribution or raw measure.
    reference : Tensor
        Tensor carrying the required dtype and device.

    Returns
    -------
    Tensor
        A deterministic base-support point for inactive score branches.

    Raises
    ------
    NotImplementedError
        If the base has no declared scalar support."""
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

    _atomicity = "mixed"

    @property
    def arg_constraints(self):
        """Delegate parameter validation to the underlying distribution.

        Returns
        -------
        object
            Empty parameter metadata; the base validates its own parameters."""
        return {}

    def __init__(self, base: Distribution, low=None, high=None):
        if base.event_shape:
            raise ValueError("scalar censoring requires scalar events")
        from quivers.continuous.measure import normalize_at_boundary

        base = normalize_at_boundary(base)
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
        """Closed response interval, including the censoring atoms.

        Returns
        -------
        object
            Constraint describing valid response values."""
        return constraints.interval(self.low, self.high)

    def expand(self, batch_shape, _instance=None):
        """Broadcast the same base law and censoring bounds.

        Parameters
        ----------
        batch_shape : object
            Requested broadcast batch shape.
        _instance : object
            Unused Torch expansion instance.

        Returns
        -------
        object
            Distribution with the requested batch shape and retained construction."""
        return Censored(
            self.base.expand(batch_shape),
            self.low.expand(batch_shape),
            self.high.expand(batch_shape),
        )

    def select_batch(self, active, *, output_shape=None):
        """Select only active base rows and censoring bounds before sampling.

        Parameters
        ----------
        active : Tensor
            Boolean mask over the native batch axes.
        output_shape : torch.Size or None
            Optional selected batch layout, including nested event batch axes.

        Returns
        -------
        Distribution
            Selected law, preserving registered subclasses and their policies.
        """
        from quivers.continuous._atoms import _family_metadata
        from quivers.continuous._batch_selection import (
            _reconstruct,
            select_batch,
            select_parameter,
        )

        if _family_metadata(self) is not None:

            def select(value, rank):
                """Select one subclass constructor parameter under its rank.

                Parameters
                ----------
                value : Tensor
                    Parameter tensor.
                rank : int
                    Intrinsic parameter rank.

                Returns
                -------
                Tensor
                    Selected tensor parameter.
                """
                return select_parameter(
                    value, self.batch_shape, rank, active, output_shape=output_shape
                )

            return _reconstruct(self, select)
        base = select_batch(self.base, active, output_shape=output_shape)
        low = select_parameter(
            self.low, self.batch_shape, 0, active, output_shape=output_shape
        )
        high = select_parameter(
            self.high, self.batch_shape, 0, active, output_shape=output_shape
        )
        return Censored(base, low, high)

    def atom_reference(self, value):
        """Retain finite censoring boundaries and the base's interior atoms.

        Parameters
        ----------
        value : Tensor or float
            Scalar query points.

        Returns
        -------
        Tensor
            Potential atomic carrier, independent of current atom masses.
        """
        from quivers.continuous._atoms import atom_reference, _point

        value = _point(self, value)
        interior = (value > self.low) & (value < self.high)
        boundary = ((value == self.low) | (value == self.high)) & torch.isfinite(value)
        safe = _interior_point(self.base, self.low)
        return boundary | (
            interior & atom_reference(self.base, torch.where(interior, value, safe))
        )

    def log_atom_mass(self, value):
        """Score censoring endpoint masses and retained interior base atoms.

        Parameters
        ----------
        value : Tensor or float
            Scalar query points.

        Returns
        -------
        Tensor
            Log singleton probability mass, with no continuous contribution.
        """
        from quivers.continuous._atoms import log_atom_mass, _point

        value = _point(self, value)
        interior = (value > self.low) & (value < self.high)
        safe = _interior_point(self.base, self.low)
        lower = _log_tails(
            self.base, torch.where(torch.isfinite(self.low), self.low, safe)
        )[0]
        upper = _log_tails_left(
            self.base, torch.where(torch.isfinite(self.high), self.high, safe)
        )[1]
        middle = log_atom_mass(self.base, torch.where(interior, value, safe))
        result = torch.where(
            value == self.low, lower, torch.where(value == self.high, upper, middle)
        )
        return torch.where(self.atom_reference(value), result, -torch.inf)

    def log_prob(self, value):
        """Score boundary probability masses and interior base densities.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Raw log density or mass at the evaluation points.

        Raises
        ------
        ValueError
            If a response is NaN."""
        value = torch.as_tensor(value, dtype=self.low.dtype, device=self.low.device)
        if bool(value.isnan().any()):
            raise ValueError("censored response values must not be NaN")
        safe = _interior_point(self.base, self.low)
        interior = (value > self.low) & (value < self.high)
        density = self.base.log_prob(torch.where(interior, value, safe))
        lower = _log_tails(
            self.base, torch.where(torch.isfinite(self.low), self.low, safe)
        )[0]
        upper = _log_tails_left(
            self.base, torch.where(torch.isfinite(self.high), self.high, safe)
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

    def _tails(self, value, *, left=False):
        """Keep exact CDF support boundaries without invalid inactive scores.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.
        left : object
            Whether to select the CDF left limit.

        Returns
        -------
        object
            Paired log CDF and survival values.

        Raises
        ------
        ValueError
            If an evaluation point is NaN."""
        value = torch.as_tensor(value, dtype=self.low.dtype, device=self.low.device)
        if bool(value.isnan().any()):
            raise ValueError("censored response values must not be NaN")
        interior = (
            ((value > self.low) & (value <= self.high))
            if left
            else ((value >= self.low) & (value < self.high))
        ) & torch.isfinite(value)
        below = (value <= self.low if left else value < self.low) | torch.isneginf(
            value
        )
        safe = _interior_point(self.base, self.low)
        tails = _log_tails_left if left else _log_tails
        cdf, sf = tails(self.base, torch.where(interior, value, safe))
        return (
            torch.where(interior, cdf, torch.where(below, -torch.inf, 0.0)),
            torch.where(interior, sf, torch.where(below, 0.0, -torch.inf)),
        )

    def log_cdf(self, value):
        """Evaluate the CDF including the lower boundary atom.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Result in the distribution dtype and device."""
        return self._tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate strict survival, excluding the upper boundary atom.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Result in the distribution dtype and device."""
        return self._tails(value)[1]

    def log_cdf_left(self, value):
        """Evaluate ``log P(Y < value)``, excluding either boundary atom.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Result in the distribution dtype and device."""
        return self._tails(value, left=True)[0]

    def log_survival_function_left(self, value):
        """Evaluate ``log P(Y >= value)``, including the upper boundary atom.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Result in the distribution dtype and device."""
        return self._tails(value, left=True)[1]

    def cdf(self, value):
        """Evaluate the CDF of the censored response.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Lower-half-line probability, including boundary atoms."""
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Map base quantiles to the censoring interval.

        Parameters
        ----------
        value : object
            Evaluation points, broadcast against the distribution batch.

        Returns
        -------
        object
            Response quantiles at the requested probabilities."""
        return self.base.icdf(value).clamp(self.low, self.high)

    def sample(self, sample_shape=torch.Size()):
        """Draw from the censoring pushforward, including its boundary atoms.

        Parameters
        ----------
        sample_shape : object
            Leading independent sampling dimensions.

        Returns
        -------
        object
            Draws from the normalized law with leading sample dimensions."""
        with torch.no_grad():
            return self.base.sample(sample_shape).clamp(self.low, self.high)

    def rsample(self, sample_shape=torch.Size()):
        """Apply censoring to a pathwise base draw.

        Parameters
        ----------
        sample_shape : object
            Leading independent sampling dimensions.

        Returns
        -------
        object
            Pathwise draws with leading sample dimensions."""
        return self.base.rsample(sample_shape).clamp(self.low, self.high)
