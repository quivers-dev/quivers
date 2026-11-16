"""Correlated Gaussian additive log ratios on a simplex."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, constraints


class AdditiveLogRatioNormal(Distribution):
    """A Gaussian law for log ratios against the final simplex component.

    Parameters
    ----------
    loc : Tensor or Sequence[float]
        Gaussian location with trailing width ``D``.
    scale_tril : Tensor or Sequence[Sequence[float]]
        Lower-triangular Gaussian scale with trailing shape ``(D, D)``.
    validate_args : bool or None
        Torch distribution validation policy.

    Notes
    -----
    A draw has ``D + 1`` components. Its first ``D`` components satisfy
    ``log(y[..., :-1] / y[..., -1:]) ~ Normal(loc, scale_tril)`` jointly.
    Analytic response means and variances are unavailable.
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale_tril, validate_args=None):
        device = (
            loc.device
            if isinstance(loc, Tensor)
            else scale_tril.device
            if isinstance(scale_tril, Tensor)
            else None
        )
        loc = torch.as_tensor(loc, device=device)
        scale_tril = torch.as_tensor(scale_tril, device=device)
        dtype = torch.promote_types(loc.dtype, scale_tril.dtype)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        self._normal = MultivariateNormal(
            loc.to(dtype), scale_tril=scale_tril.to(dtype), validate_args=validate_args
        )
        self.loc = self._normal.loc
        self.scale_tril = self._normal.scale_tril
        if self.loc.shape[-1] < 1:
            raise ValueError(
                "AdditiveLogRatioNormal requires at least two simplex components"
            )
        super().__init__(
            self._normal.batch_shape,
            torch.Size((self.loc.shape[-1] + 1,)),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        """Broadcast the batch dimensions while retaining the coordinate event.

        Parameters
        ----------
        batch_shape : torch.Size
            New broadcast-compatible batch shape.
        _instance : AdditiveLogRatioNormal or None
            Optional instance populated by Torch's expansion protocol.

        Returns
        -------
        AdditiveLogRatioNormal
            Distribution with the requested batch shape.
        """
        new = self._get_checked_instance(AdditiveLogRatioNormal, _instance)
        new._normal = self._normal.expand(batch_shape)
        new.loc = new._normal.loc
        new.scale_tril = new._normal.scale_tril
        super(AdditiveLogRatioNormal, new).__init__(
            torch.Size(batch_shape), self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        """Draw simplex values by reparameterizing the Gaussian log ratios.

        Parameters
        ----------
        sample_shape : torch.Size
            Independent sample dimensions preceding the batch dimensions.

        Returns
        -------
        Tensor
            Strictly positive simplex draws.

        Raises
        ------
        FloatingPointError
            If floating point underflow cannot represent an interior draw.
        """
        ratios = self._normal.rsample(sample_shape)
        values = torch.cat((ratios, torch.zeros_like(ratios[..., :1])), -1).softmax(-1)
        if not bool((values > 0).all()):
            raise FloatingPointError(
                "AdditiveLogRatioNormal draw underflowed the simplex interior"
            )
        return values

    def log_prob(self, value):
        """Score an interior simplex event, including the ALR Jacobian.

        Parameters
        ----------
        value : Tensor
            Candidate simplex events with the trailing component axis.

        Returns
        -------
        Tensor
            Gaussian log density minus the sum of log components; negative
            infinity outside the simplex interior.

        Raises
        ------
        ValueError
            If the event has the wrong number of components.
        """
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        if value.ndim < 1 or value.shape[-1] != self.event_shape[0]:
            raise ValueError(
                "AdditiveLogRatioNormal event has the wrong component count"
            )
        valid = (
            torch.isfinite(value).all(-1)
            & (value > 0).all(-1)
            & constraints.simplex.check(value)
        )
        safe = torch.where(
            valid[..., None], value, torch.full_like(value, 1 / value.shape[-1])
        )
        log_y = safe.log()
        score = self._normal.log_prob(log_y[..., :-1] - log_y[..., -1:]) - log_y.sum(-1)
        return torch.where(valid, score, torch.full_like(score, -torch.inf))


__all__ = ["AdditiveLogRatioNormal"]
