"""Asymmetric Laplace distributions in the quantile regression parameterization."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from quivers.continuous._zip_hurdle import _OpenUnitInterval


class AsymmetricLaplace(Distribution):
    r"""Asymmetric Laplace distribution parameterized by a response quantile.

    The density is ``q * (1-q) / scale * exp(-rho_q((y-loc)/scale))``,
    where ``rho_q(u) = u * (q - I[u < 0])``. Thus ``P(Y < loc) = q``;
    ``loc`` is the modeled quantile, and generally differs from the mean.
    At ``q=0.5``, this is ``Laplace(loc, 2 * scale)``.

    Parameters
    ----------
    loc : Tensor or float
        Location of the specified quantile.
    scale : Tensor or float
        Strictly positive scale in the check-loss parameterization.
    quantile : Tensor or float
        Probability strictly between zero and one.
    validate_args : bool or None
        Whether to validate distribution parameters and evaluation points.

    Notes
    -----
    Sampling uses the difference of independent exponential variables.
    It is reparameterized in all three parameters and requires no
    rejection loop, inverse-CDF endpoint clamp, or discrete branch draw.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "quantile": _OpenUnitInterval(),
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        quantile: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale, self.quantile = broadcast_all(loc, scale, quantile)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(shape)
        new.scale = self.scale.expand(shape)
        new.quantile = self.quantile.expand(shape)
        super(AsymmetricLaplace, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        q = self.quantile
        return self.loc + self.scale * (1 - 2 * q) / (q * (1 - q))

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        q = self.quantile
        return self.scale.square() * (
            q.reciprocal().square() + (1 - q).reciprocal().square()
        )

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        q = self.quantile
        residual = (value - self.loc) / self.scale
        loss = torch.where(residual < 0, (q - 1) * residual, q * residual)
        return q.log() + torch.log1p(-q) - self.scale.log() - loss

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        q = self.quantile
        residual = (value - self.loc) / self.scale
        # Restrict the inactive exponent to avoid overflow in either tail.
        left = q * torch.exp((1 - q) * residual.clamp_max(0))
        right = -torch.expm1(torch.log1p(-q) - q * residual.clamp_min(0))
        return torch.where(residual < 0, left, right)

    def icdf(self, value: Tensor) -> Tensor:
        if self._validate_args and not bool(
            constraints.unit_interval.check(value).all()
        ):
            raise ValueError("AsymmetricLaplace.icdf requires probabilities in [0, 1]")
        q = self.quantile
        left = (value.log() - q.log()) / (1 - q)
        right = -(torch.log1p(-value) - torch.log1p(-q)) / q
        return self.loc + self.scale * torch.where(value < q, left, right)

    def entropy(self) -> Tensor:
        q = self.quantile
        return 1 + self.scale.log() - q.log() - torch.log1p(-q)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        noise = torch.empty(
            (2, *shape), dtype=self.loc.dtype, device=self.loc.device
        ).exponential_()
        return self.loc + self.scale * (
            noise[0] / self.quantile - noise[1] / (1 - self.quantile)
        )


__all__ = ["AsymmetricLaplace"]
