"""Zero-inflated, hurdle, and Normal mixture families.

The inflation and Normal-mixture families compose probability measures.
The hurdle family scores a normalized positive Poisson component directly,
including its point-mass limit at rate zero.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution

from quivers.continuous._closed_probability import ClosedPoisson, _count_log
from quivers.continuous._zero_truncated_poisson import (
    positive_poisson_mean,
    positive_poisson_sample,
)
from quivers.continuous.measure import Mixture, PointMass


def _two_component_inflation(
    zero_prob: Tensor,
    base: Distribution,
) -> Mixture:
    """Build the two-component `Mixture([pi, 1-pi], [PointMass(0), base])`
    used by inflation / hurdle constructions. Stacks the weights
    along the last axis so per-row `pi` broadcasts against the base's
    batch shape.
    """
    weights = torch.stack([zero_prob, 1.0 - zero_prob], dim=-1)
    return Mixture(weights, [PointMass(0.0), base])


class ZeroInflatedPoisson(Distribution):
    """`Mixture([pi, 1-pi], [PointMass(0), Poisson(rate)])`.

    Total zero mass `pi + (1-pi) * exp(-rate)` per Lambert 1992;
    the count component is the unrestricted Poisson, so excess zeros
    come from both the structural-zero spike and the natural Poisson
    zeros. Contrast with
    [`HurdlePoisson`][quivers.continuous._zip_hurdle.HurdlePoisson]
    in which the count component is zero-truncated.

    Reference: [Lambert 1992](https://doi.org/10.2307/1269547).
    """

    arg_constraints = {
        "zero_prob": _constraints.unit_interval,
        "rate": _constraints.nonnegative,
    }
    support = _constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        zero_prob: Tensor,
        rate: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.zero_prob = zero_prob
        self.rate = rate
        base = ClosedPoisson(rate)
        self._mixture = _two_component_inflation(zero_prob, base)
        super().__init__(
            batch_shape=self._mixture.batch_shape,
            event_shape=self._mixture.event_shape,
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return (1.0 - self.zero_prob) * self.rate

    def log_prob(self, value: Tensor) -> Tensor:
        return self._mixture.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self._mixture.sample(sample_shape).long()


class HurdlePoisson(Distribution):
    """A point mass at zero mixed with a normalized positive Poisson.

    A Bernoulli with probability ``1 - zero_prob`` selects the positive
    component. At rate zero that component is the point mass at one,
    giving the continuous extension in the rate. Negative rates are
    refused. The disjoint supports permit direct scoring of each branch.

    Reference: [Mullahy 1986](https://doi.org/10.1016/0304-4076(86)90002-3).
    """

    arg_constraints = {
        "zero_prob": _constraints.unit_interval,
        "rate": _constraints.nonnegative,
    }
    support = _constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        zero_prob: Tensor,
        rate: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.zero_prob, self.rate = torch.broadcast_tensors(zero_prob, rate)
        super().__init__(
            batch_shape=self.rate.shape,
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        """The hurdle mean, with the positive component's limit at zero."""
        return (1.0 - self.zero_prob) * positive_poisson_mean(self.rate)

    def log_prob(self, value: Tensor) -> Tensor:
        """Score counts against the normalized hurdle distribution.

        Parameters
        ----------
        value : Tensor
            Nonnegative integer observations.

        Returns
        -------
        Tensor
            Log probabilities broadcast over observations and parameters.

        Raises
        ------
        ValueError
            If validation is enabled and an observation is outside the support.
        """
        if self._validate_args:
            self._validate_sample(value)
        safe_value = torch.where(value > 0, value, torch.ones_like(value))
        one = torch.ones_like(self.rate)
        positive = (
            _count_log(one, 1 - self.zero_prob)
            + _count_log(safe_value - 1, self.rate)
            - self.rate
            + positive_poisson_mean(self.rate).log()
            - torch.lgamma(safe_value + 1)
        )
        result = torch.where(value == 0, _count_log(one, self.zero_prob), positive)
        return torch.where(
            _constraints.nonnegative_integer.check(value), result, -torch.inf
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw counts using a conditioned first arrival for the positive branch.

        Parameters
        ----------
        sample_shape : torch.Size
            Dimensions prepended to the parameter batch shape.

        Returns
        -------
        Tensor
            Nonnegative integer draws, including the rate-zero limit.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            positive = positive_poisson_sample(self.rate.expand(shape))
            zero = torch.bernoulli(self.zero_prob.expand(shape)).bool()
            return torch.where(zero, torch.zeros_like(positive), positive).long()


class MixtureNormal(Distribution):
    """`Mixture(weights, [Normal(loc_k, scale_k) for k])`.

    Same-family Gaussian mixture with K = `weights.shape[-1]`
    components. Mixing weights live in the K-simplex along the last
    axis; component means and scales also vary along the last axis
    and broadcast over the leading batch.

    For mixing arbitrary families together, construct the
    [`Mixture`][quivers.continuous.measure.Mixture] directly with a
    list of heterogeneous components.

    Reference: [McLachlan & Peel 2000](https://doi.org/10.1002/0471721182).
    """

    arg_constraints = {
        "weights": _constraints.simplex,
        "loc": _constraints.real,
        "scale": _constraints.positive,
    }
    support = _constraints.real
    has_rsample = False

    def __init__(
        self,
        weights: Tensor,
        loc: Tensor,
        scale: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        if weights.shape[-1] != loc.shape[-1]:
            raise ValueError(
                "MixtureNormal: weights and loc must agree on the "
                f"component axis (last dim); got weights={tuple(weights.shape)}, "
                f"loc={tuple(loc.shape)}"
            )
        if weights.shape[-1] != scale.shape[-1]:
            raise ValueError(
                "MixtureNormal: weights and scale must agree on the "
                f"component axis (last dim); got weights={tuple(weights.shape)}, "
                f"scale={tuple(scale.shape)}"
            )
        self.weights = weights
        self.loc = loc
        self.scale = scale
        self._num_components = int(weights.shape[-1])
        components = [
            torch.distributions.Normal(loc[..., k], scale[..., k])
            for k in range(self._num_components)
        ]
        self._mixture = Mixture(weights, components)
        super().__init__(
            batch_shape=self._mixture.batch_shape,
            event_shape=self._mixture.event_shape,
            validate_args=validate_args,
        )

    @property
    def num_components(self) -> int:
        return self._num_components

    @property
    def mean(self) -> Tensor:
        return (self.weights * self.loc).sum(dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        return self._mixture.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self._mixture.sample(sample_shape)


class _OpenUnitInterval(_constraints.Constraint):
    """Strictly interior probabilities."""

    def check(self, value: Tensor) -> Tensor:
        return (value > 0) & (value < 1)


class ZeroOneInflatedBeta(Distribution):
    """Zero-one inflated beta distribution.

    Mixture of point masses at 0 and 1 with a continuous Beta on (0, 1):

    * ``P(y = 0) = zoi * (1 - coi)``
    * ``P(y = 1) = zoi * coi``
    * ``P(0 < y < 1) = 1 - zoi``
    * On the open interval, the density is ``(1 - zoi)`` times
      the density of ``Beta(mu * phi, (1 - mu) * phi)``.

    Parameters follow the mean-precision Beta reparameterization and the
    brms ``zoi`` / ``coi`` inflation probabilities. The mean is strictly
    between zero and one, precision is positive, and inflation
    probabilities include both endpoints. Formula frontend emits
    ``ZeroOneInflatedBeta(mu, phi, zoi, coi)``.

    For the distributional background, see [Ospina and Ferrari
    (2010)](https://doi.org/10.1007/s00362-008-0125-4). The parameter
    convention follows the [brms family
    documentation](https://paulbuerkner.com/brms/articles/brms_families.html).
    """

    arg_constraints = {
        "mu": _OpenUnitInterval(),
        "phi": _constraints.positive,
        "zoi": _constraints.unit_interval,
        "coi": _constraints.unit_interval,
    }
    support = _constraints.unit_interval
    has_rsample = False

    def __init__(
        self,
        mu: Tensor,
        phi: Tensor,
        zoi: Tensor,
        coi: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.phi, self.zoi, self.coi = torch.broadcast_tensors(
            mu, phi, zoi, coi
        )
        batch_shape = self.mu.shape
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size(),
            validate_args=validate_args,
        )

    def _beta(self) -> torch.distributions.Beta:
        return torch.distributions.Beta(
            self.mu * self.phi,
            (1.0 - self.mu) * self.phi,
            validate_args=self._validate_args,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        is_zero = value == 0.0
        is_one = value == 1.0
        is_interior = (value > 0.0) & (value < 1.0)
        # Inactive components use interior arguments so their derivatives
        # cannot introduce NaNs into the selected component.
        zero_zoi = torch.where(is_zero, self.zoi, 1.0)
        zero_coi = torch.where(is_zero, self.coi, 0.0)
        one_zoi = torch.where(is_one, self.zoi, 1.0)
        one_coi = torch.where(is_one, self.coi, 1.0)
        mid_zoi = torch.where(is_interior, self.zoi, 0.0)
        mid_value = torch.where(is_interior, value, 0.5)
        lp_zero = zero_zoi.log() + torch.log1p(-zero_coi)
        lp_one = one_zoi.log() + one_coi.log()
        lp_mid = torch.log1p(-mid_zoi) + self._beta().log_prob(mid_value)
        return torch.where(
            is_zero,
            lp_zero,
            torch.where(is_one, lp_one, torch.where(is_interior, lp_mid, -torch.inf)),
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = sample_shape + self.batch_shape
        u = torch.rand(shape, device=self.mu.device, dtype=self.mu.dtype)
        zoi = self.zoi.expand(shape)
        coi = self.coi.expand(shape)
        at_boundary = u < zoi
        u2 = torch.rand(shape, device=self.mu.device, dtype=self.mu.dtype)
        boundary = torch.where(u2 < coi, torch.ones_like(u2), torch.zeros_like(u2))
        mid = self._beta().sample(sample_shape)
        return torch.where(at_boundary, boundary, mid)


__all__ = [
    "HurdlePoisson",
    "MixtureNormal",
    "ZeroInflatedPoisson",
    "ZeroOneInflatedBeta",
]
