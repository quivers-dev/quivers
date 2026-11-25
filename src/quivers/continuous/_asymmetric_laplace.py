"""Asymmetric Laplace distributions in the quantile regression parameterization."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from quivers._positions import demand
from quivers.continuous._zip_hurdle import _OpenUnitInterval
from quivers.continuous._shape_families import _log1mexp


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
        """Expand parameter views over a new batch.

        Parameters
        ----------
        batch_shape : torch.Size
            Target batch dimensions.
        _instance : Distribution or None
            Optional reusable distribution instance.

        Returns
        -------
        AsymmetricLaplace
            Expanded parameter views.
        """
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
        """Expected response, which generally differs from the quantile location.

        Returns
        -------
        Tensor
            Response mean.
        """
        q = self.quantile
        return self.loc + self.scale * (1 - 2 * q) / (q * (1 - q))

    @property
    def mode(self) -> Tensor:
        """Most probable continuous response.

        Returns
        -------
        Tensor
            Component quantile location.
        """
        return self.loc

    @property
    def variance(self) -> Tensor:
        """Response variance from the two exponential components.

        Returns
        -------
        Tensor
            Response variance.
        """
        q = self.quantile
        return self.scale.square() * (
            q.reciprocal().square() + (1 - q).reciprocal().square()
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluate the check-loss density.

        Parameters
        ----------
        value : Tensor
            Real response values.

        Returns
        -------
        Tensor
            Log Lebesgue density.
        """
        if self._validate_args:
            self._validate_sample(value)
        q = self.quantile
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        residual = (point - self.loc) / self.scale
        loss = torch.where(residual < 0, (q - 1) * residual, q * residual)
        result = q.log() + torch.log1p(-q) - self.scale.log() - loss
        return torch.where(finite, result, -torch.inf)

    def _log_tails(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate complementary tails without unsafe infinite arithmetic.

        Parameters
        ----------
        value : Tensor
            Real evaluation points, including infinite interval bounds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Log cumulative and strict survival probabilities.
        """
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        residual = (point - self.loc) / self.scale
        q = self.quantile
        left = q.log() + (1 - q) * residual.clamp_max(0)
        right = torch.log1p(-q) - q * residual.clamp_min(0)
        log_cdf = torch.where(residual < 0, left, _log1mexp(right))
        log_sf = torch.where(residual < 0, _log1mexp(left), right)
        log_cdf = torch.where(value == -torch.inf, -torch.inf, log_cdf)
        log_cdf = torch.where(value == torch.inf, 0.0, log_cdf)
        log_sf = torch.where(value == -torch.inf, 0.0, log_sf)
        log_sf = torch.where(value == torch.inf, -torch.inf, log_sf)
        return log_cdf, log_sf

    def log_cdf(self, value: Tensor) -> Tensor:
        """Evaluate cumulative probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Real evaluation points.

        Returns
        -------
        Tensor
            Log probability of a response at or below each point.
        """
        return self._log_tails(value)[0]

    def log_survival_function(self, value: Tensor) -> Tensor:
        """Evaluate strict survival probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Real evaluation points.

        Returns
        -------
        Tensor
            Log probability of a response strictly above each point.
        """
        return self._log_tails(value)[1]

    def cdf(self, value: Tensor) -> Tensor:
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Real evaluation points.

        Returns
        -------
        Tensor
            Probability of a response at or below each point.
        """
        return self.log_cdf(value).exp()

    def icdf(self, value: Tensor) -> Tensor:
        """Invert cumulative probabilities with safe endpoint derivatives.

        Parameters
        ----------
        value : Tensor
            Probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Continuous response quantiles.

        Raises
        ------
        ValueError
            If probabilities are outside the closed unit interval.
        """
        if self._validate_args:
            demand(
                constraints.unit_interval.check(value),
                lambda: ValueError(
                    "AsymmetricLaplace.icdf requires probabilities in [0, 1]"
                ),
            )
        q = self.quantile
        interior = (value > 0) & (value < 1)
        probability = torch.where(interior, value, 0.5)
        is_left = probability < q
        left_point = torch.where(is_left, probability, q)
        right_point = torch.where(is_left, q, probability)
        left = (left_point.log() - q.log()) / (1 - q)
        right = -(torch.log1p(-right_point) - torch.log1p(-q)) / q
        result = self.loc + self.scale * torch.where(is_left, left, right)
        result = torch.where(value == 0, -torch.inf, result)
        return torch.where(value == 1, torch.inf, result)

    def entropy(self) -> Tensor:
        """Differential entropy of the component law.

        Returns
        -------
        Tensor
            Entropy in natural logarithmic units.
        """
        q = self.quantile
        return 1 + self.scale.log() - q.log() - torch.log1p(-q)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw reparameterized differences of independent exponentials.

        Parameters
        ----------
        sample_shape : torch.Size
            Dimensions prepended to the parameter batch.

        Returns
        -------
        Tensor
            Reparameterized response samples.
        """
        shape = self._extended_shape(sample_shape)
        noise = torch.empty(
            (2, *shape), dtype=self.loc.dtype, device=self.loc.device
        ).exponential_()
        return self.loc + self.scale * (
            noise[0] / self.quantile - noise[1] / (1 - self.quantile)
        )


__all__ = ["AsymmetricLaplace"]
