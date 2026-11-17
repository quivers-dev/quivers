"""Zero-inflated, hurdle, and Normal mixture families.

The inflation and Normal-mixture families compose probability measures.
The hurdle family scores a normalized positive Poisson component directly,
including its point-mass limit at rate zero.
"""

from __future__ import annotations

import torch
from torch import Tensor
from quivers.continuous._count_cdf import _CountCDF
from quivers.continuous._closed_probability import _check_count_observation_source
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution

from quivers.continuous._closed_probability import ClosedPoisson, _count_log
from quivers.continuous._recorded_event_tails import (
    log_normal_mixture_tails,
    log_zero_one_inflated_beta_tails,
)
from quivers.continuous._zero_inflated_count import _InflatedCount
from quivers.continuous._zero_truncated_poisson import (
    positive_poisson_mean,
    positive_poisson_log_mean,
    positive_poisson_sample,
)
from quivers.continuous._observation_events import event_log_prob
from quivers.continuous.measure import Mixture, Normalize, Restrict


class ZeroInflatedPoisson(_InflatedCount):
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
    _count_class = ClosedPoisson
    _zero_count_parameters = ("rate",)

    def __init__(
        self,
        zero_prob: Tensor,
        rate: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.zero_prob, self.rate = torch.broadcast_tensors(zero_prob, rate)
        super().__init__(
            batch_shape=self.zero_prob.shape,
            validate_args=validate_args,
        )


class HurdlePoisson(_CountCDF, Distribution):
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

    def _all_count_tails(self, value):
        """Condition shared Poisson tails before structural-zero mixing.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Paired log and ordinary tail probabilities.
        """
        from quivers.continuous._count_tails import (
            positive_count_log_tails,
            inflated_count_tails,
        )

        rate = self.rate.double()
        base = ClosedPoisson(rate, validate_args=False)
        first = -rate + positive_poisson_log_mean(rate)
        second = first + torch.log1p(rate / 2)
        lc, ls = positive_count_log_tails(
            value,
            base.log_cdf(value),
            base.log_survival_function(value),
            -rate,
            first,
            second,
        )
        return inflated_count_tails(self.zero_prob.double(), value, lc, ls)

    def _count_log_tails(self, value):
        """Read the paired log tails from the shared hurdle event kernel.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Inclusive log CDF and strict log survival.
        """
        result = self._all_count_tails(value)
        return result[0], result[1]

    def _count_probability_tails(self, value):
        """Retain finite probability derivatives at the hurdle boundary.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Inclusive cumulative and strict survival probabilities.
        """
        result = self._all_count_tails(value)
        return result[2], result[3]

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
        _check_count_observation_source(value)
        if self._validate_args:
            self._validate_sample(value)
        valid = _constraints.nonnegative_integer.check(value)
        value = value.to(torch.float64)
        rate, zero = self.rate.to(torch.float64), self.zero_prob.to(torch.float64)
        safe_value = torch.where(valid & (value > 0), value, 1.0)
        # Reuse the stable unrestricted count kernel. The one-count identity
        # retains the finite first/second rate derivatives at the zero limit.
        safe_rate = torch.where(rate > 0, rate, 1.0)
        conditioned = (
            ClosedPoisson(safe_rate, validate_args=False).log_prob(safe_value)
            - (-torch.expm1(-safe_rate)).log()
        )
        component = torch.where(
            safe_value == 1,
            -rate + positive_poisson_log_mean(rate),
            torch.where(rate > 0, conditioned, -torch.inf),
        )
        one = torch.ones_like(rate)
        positive = _count_log(one, 1 - zero) + component
        result = torch.where(value == 0, _count_log(one, zero), positive)
        return torch.where(valid, result, -torch.inf)

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
            probability = self.zero_prob.expand(shape)
            active = (
                torch.rand(shape, dtype=probability.dtype, device=probability.device)
                >= probability
            )
            result = torch.zeros(shape, dtype=torch.long, device=probability.device)
            if active.any():
                result[active] = positive_poisson_sample(
                    self.rate.expand(shape)[active]
                ).long()
            return result


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

    def _log_tails(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the log CDF and log survival as weighted component sums.

        Parameters
        ----------
        value : Tensor
            Real query points without the component axis.

        Returns
        -------
        tuple[Tensor, Tensor]
            Float64 ``log P(Y <= value)`` and ``log P(Y > value)``.
        """
        return log_normal_mixture_tails(self.weights, self.loc, self.scale, value)

    def _components(self) -> list[torch.distributions.Normal]:
        """Build the Normal components in component-axis order.

        Returns
        -------
        list[torch.distributions.Normal]
            One Normal law per mixture component.
        """
        return [
            torch.distributions.Normal(self.loc[..., k], self.scale[..., k])
            for k in range(self._num_components)
        ]

    def recorded_event_log_prob(
        self,
        value: Tensor,
        *,
        status: Tensor | int | None = None,
        interval_upper: Tensor | float | None = None,
        trunc_lower: Tensor | float | None = None,
        trunc_upper: Tensor | float | None = None,
    ) -> Tensor:
        """Score a recorded event with each component truncated separately.

        brms normalizes every mixture component on the truncation interval and
        then mixes the normalized components with the mixing weights. The
        score is ``log sum_k w_k P_k(event) / P_k(lower <= Y <= upper)``, and
        censoring alone reduces to the mass of the mixture. Cost: O(n K).

        Parameters
        ----------
        value : Tensor
            Exact outcomes or first censoring endpoints.
        status : Tensor or int or None
            Censoring codes -1 left, 0 exact, 1 right, 2 interval.
        interval_upper : Tensor or float or None
            Upper endpoints of interval-censored rows.
        trunc_lower, trunc_upper : Tensor or float or None
            Inclusive truncation bounds.

        Returns
        -------
        Tensor
            Unweighted log likelihoods per row.
        """
        scores = torch.stack(
            [
                event_log_prob(
                    component,
                    value,
                    status=status,
                    interval_upper=interval_upper,
                    trunc_lower=trunc_lower,
                    trunc_upper=trunc_upper,
                )
                for component in self._components()
            ],
            dim=-1,
        )
        log_weights = torch.where(
            self.weights > 0,
            torch.where(self.weights > 0, self.weights, 1.0).log(),
            -torch.inf,
        )
        return torch.logsumexp(log_weights + scores, dim=-1)

    def componentwise_truncation(
        self, lower: Tensor | float | None, upper: Tensor | float | None
    ) -> Mixture:
        """Mix the components after normalizing each on the truncation interval.

        Parameters
        ----------
        lower, upper : Tensor or float or None
            Inclusive truncation bounds; None leaves a side open.

        Returns
        -------
        Mixture
            The brms truncated mixture, sampled by component then by the
            component's truncated Normal law.
        """
        return Mixture(
            self.weights,
            [
                Normalize(Restrict(component, lower, upper))
                for component in self._components()
            ],
        )

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

    _semantic_family = "ZeroOneInflatedBeta"

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

    @property
    def mean(self) -> Tensor:
        """Expected response, including the atom at one and the interior mean.

        Returns
        -------
        Tensor
            The atom-at-one contribution plus the weighted interior mean.
        """
        return self.zoi * self.coi + (1 - self.zoi) * self.mu

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

    def _log_tails(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the log CDF and log survival, including both atoms.

        Parameters
        ----------
        value : Tensor
            Real query points.

        Returns
        -------
        tuple[Tensor, Tensor]
            Float64 ``log P(Y <= value)`` and ``log P(Y > value)``.
        """
        return log_zero_one_inflated_beta_tails(
            self.mu, self.phi, self.zoi, self.coi, value, left=False
        )

    def log_cdf_left(self, value: Tensor) -> Tensor:
        """Evaluate ``log P(Y < value)``, excluding an atom at the query.

        Parameters
        ----------
        value : Tensor
            Real query points.

        Returns
        -------
        Tensor
            Float64 log left limits of the CDF.
        """
        return log_zero_one_inflated_beta_tails(
            self.mu, self.phi, self.zoi, self.coi, value, left=True
        )[0]

    def log_survival_function_left(self, value: Tensor) -> Tensor:
        """Evaluate ``log P(Y >= value)``, including an atom at the query.

        Parameters
        ----------
        value : Tensor
            Real query points.

        Returns
        -------
        Tensor
            Float64 log survival including the query point.
        """
        return log_zero_one_inflated_beta_tails(
            self.mu, self.phi, self.zoi, self.coi, value, left=True
        )[1]

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
