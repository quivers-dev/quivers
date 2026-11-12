"""Sugar wrappers for zero-inflated, hurdle, and mixture families.

Each class here desugars at construction time to a composition in the
[measure algebra][quivers.continuous.measure]:

* `ZeroInflatedPoisson(pi, rate) := Mixture([pi, 1-pi], [PointMass(0), Poisson(rate)])`
* `HurdlePoisson(pi, rate) := Mixture([pi, 1-pi], [PointMass(0), Restrict(Poisson(rate), low=1)])`
* `MixtureNormal(w, loc, scale) := Mixture(w, [Normal(loc_k, scale_k) for k])`

Internally each is a [`Mixture`][quivers.continuous.measure.Mixture]
of [`PointMass`][quivers.continuous.measure.PointMass] and
[`Restrict`][quivers.continuous.measure.Restrict] / base components,
which the rewrite pass recognises and lifts to the canonical
Bernoulli-branch + conditional-density form Stan / PyMC / NumPyro
expect. A single Mixture-with-PointMass emitter covers every family
in place of per-family per-base Cartesian-product renderer logic.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution

from quivers.continuous._closed_probability import ClosedPoisson
from quivers.continuous.measure import Mixture, Normalize, PointMass, Restrict


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
    """`Mixture([pi, 1-pi], [PointMass(0), Restrict(Poisson(rate), low=1)])`.

    Two-stage hurdle model: a Bernoulli`(1 - pi)` decides whether the
    response is zero or positive; conditional on positivity, the
    response is a zero-truncated Poisson. Because the two components
    have measure-disjoint supports, the joint log-density factors
    into a Bernoulli plus a zero-truncated Poisson term, and the
    rewrite pass emits that factored form to every backend.

    Reference: [Mullahy 1986](https://doi.org/10.1016/0304-4076(86)90002-3).
    """

    arg_constraints = {
        "zero_prob": _constraints.unit_interval,
        "rate": _constraints.positive,
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
        base = torch.distributions.Poisson(rate)
        # Wrap the truncated component in `Normalize` so the hurdle's
        # second component is a probability measure on `{1, 2, ...}`
        # before mixing; the resulting Mixture has total mass one and
        # the user-facing `log_prob` returns the proper hurdle
        # density rather than the unnormalised sub-measure form.
        truncated_base = Normalize(Restrict(base, low=1.0))
        self._mixture = _two_component_inflation(zero_prob, truncated_base)
        super().__init__(
            batch_shape=self._mixture.batch_shape,
            event_shape=self._mixture.event_shape,
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        survival = -torch.expm1(-self.rate)
        return (1.0 - self.zero_prob) * self.rate / survival.clamp_min(1e-12)

    def log_prob(self, value: Tensor) -> Tensor:
        return self._mixture.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self._mixture.sample(sample_shape).long()


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


class ZeroOneInflatedBeta(Distribution):
    """Zero-one inflated beta distribution.

    Mixture of point masses at 0 and 1 with a continuous Beta on (0, 1):

    * ``P(y = 0) = zoi * (1 - coi)``
    * ``P(y = 1) = zoi * coi``
    * ``P(0 < y < 1) = 1 - zoi``
    * On the open interval, the density is ``(1 - zoi)`` times
      the density of ``Beta(mu * phi, (1 - mu) * phi)``.

    Parameters follow the mean-precision Beta reparameterization and the
    brms ``zoi`` / ``coi`` inflation probabilities (all on the unit
    interval). Formula frontend emits
    ``ZeroOneInflatedBeta(mu, phi, zoi, coi)``.

    For the distributional background, see [Ospina and Ferrari
    (2010)](https://doi.org/10.1007/s00362-008-0125-4). The parameter
    convention follows the [brms family
    documentation](https://paulbuerkner.com/brms/articles/brms_families.html).
    """

    arg_constraints = {
        "mu": _constraints.unit_interval,
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
        self.mu = mu
        self.phi = phi
        self.zoi = zoi
        self.coi = coi
        batch_shape = torch.broadcast_shapes(mu.shape, phi.shape, zoi.shape, coi.shape)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size(),
            validate_args=validate_args,
        )

    def _beta(self) -> torch.distributions.Beta:
        mu = self.mu.clamp(1e-6, 1.0 - 1e-6)
        phi = self.phi.clamp_min(1e-6)
        return torch.distributions.Beta(mu * phi, (1.0 - mu) * phi)

    def log_prob(self, value: Tensor) -> Tensor:
        zoi = self.zoi.clamp(1e-6, 1.0 - 1e-6)
        coi = self.coi.clamp(1e-6, 1.0 - 1e-6)
        # Exact boundary atoms (censoring / rounding produce exact 0/1).
        is_zero = value <= 0.0
        is_one = value >= 1.0
        lp_zero = torch.log(zoi * (1.0 - coi))
        lp_one = torch.log(zoi * coi)
        y_mid = value.clamp(1e-6, 1.0 - 1e-6)
        lp_mid = torch.log1p(-zoi) + self._beta().log_prob(y_mid)
        return torch.where(is_zero, lp_zero, torch.where(is_one, lp_one, lp_mid))

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
