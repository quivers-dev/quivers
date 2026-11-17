"""Zero atoms combined with normalized binomial and negative-binomial counts."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from quivers.continuous._count_cdf import _CountCDF
from quivers.continuous._closed_probability import _check_count_observation_source
from torch.distributions import Distribution, constraints

from quivers.continuous._closed_probability import (
    ClosedBinomial,
    ClosedNegativeBinomial,
    _count_log,
    _require_sample_capacity,
    _beta_binomial_log_mass,
    _log_gamma_increment,
    _negative_binomial_log_mass,
    _check_trial_count_source,
)
from quivers.continuous._simplex_sampling import _dirichlet_weights
from quivers.continuous._zero_truncated_poisson import (
    positive_poisson_log_mean,
    positive_poisson_sample,
)


def _broadcast_count_parameters(*values) -> tuple[Tensor, ...]:
    """Broadcast count parameters with float64 arithmetic before any count casts.

    Parameters
    ----------
    *values : Tensor or float or int
        Count and probability parameters, on compatible devices.

    Returns
    -------
    tuple[Tensor, ...]
        Broadcast parameter views, preserving integer counts through 2**53.
    """
    device = next((value.device for value in values if isinstance(value, Tensor)), None)
    return tuple(
        torch.broadcast_tensors(
            *(
                value.to(torch.float64)
                if isinstance(value, Tensor)
                else torch.as_tensor(value, dtype=torch.float64, device=device)
                for value in values
            )
        )
    )


class _ZeroAtomLogMass(torch.autograd.Function):
    """Log-add a structural zero with a count zero without endpoint singularities."""

    generate_vmap_rule = True

    @staticmethod
    def forward(zero_prob: Tensor, base_log_mass: Tensor) -> Tensor:
        """Evaluate the atom in log space.

        Parameters
        ----------
        zero_prob : Tensor
            Structural zero probabilities, including zero and one.
        base_log_mass : Tensor
            Log probabilities of zero under the count component.

        Returns
        -------
        Tensor
            Log total mass at zero, preserving underflowing base probabilities.
        """
        mass = torch.logaddexp(zero_prob.log(), torch.log1p(-zero_prob) + base_log_mass)
        return torch.where(base_log_mass == 0, torch.zeros_like(mass), mass)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        """Retain inputs and output for differentiable analytic derivatives.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        inputs : tuple
            Structural probability and base log mass.
        output : Tensor
            Their log mixture mass.
        """
        ctx.save_for_backward(*inputs, output)
        ctx.save_for_forward(*inputs, output)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, Tensor]:
        """Differentiate finite atom scores at closed mixture endpoints.

        Parameters
        ----------
        ctx : object
            Saved inputs and log mixture mass.
        gradient : Tensor
            Upstream derivatives, possibly zero for inactive observation branches.

        Returns
        -------
        tuple[Tensor, Tensor]
            Derivatives for the structural probability and base log mass.
        """
        zero_prob, base_log_mass, atom = ctx.saved_tensors
        active = gradient != 0
        zero_prob = torch.where(active, zero_prob, 0.0)
        base_log_mass = torch.where(active, base_log_mass, 0.0)
        atom = torch.where(active, atom, 0.0)
        return (
            gradient * -torch.expm1(base_log_mass) * torch.exp(-atom),
            gradient * (1 - zero_prob) * torch.exp(base_log_mass - atom),
        )

    @staticmethod
    def jvp(ctx, zero_tangent: Tensor, base_tangent: Tensor) -> Tensor:
        """Propagate forward sensitivities through closed mixture endpoints.

        Parameters
        ----------
        ctx : object
            Saved inputs and output from the primal evaluation.
        zero_tangent : Tensor
            Structural-probability tangent.
        base_tangent : Tensor
            Base-log-mass tangent.

        Returns
        -------
        Tensor
            Directional derivative of the log mixture atom.
        """
        zero_prob, base_log_mass, atom = ctx.saved_tensors
        zero_derivative = -torch.expm1(base_log_mass) * torch.exp(-atom)
        base_derivative = (1 - zero_prob) * torch.exp(base_log_mass - atom)
        return torch.where(
            zero_tangent == 0, 0.0, zero_tangent * zero_derivative
        ) + torch.where(base_tangent == 0, 0.0, base_tangent * base_derivative)


def _zero_atom_log_mass(zero_prob: Tensor, base_log_mass: Tensor) -> Tensor:
    """Combine the two zero masses with exact endpoint derivatives.

    Parameters
    ----------
    zero_prob : Tensor
        Structural zero probability.
    base_log_mass : Tensor
        Count-component log mass at zero.

    Returns
    -------
    Tensor
        Broadcast log probability of zero under the full mixture.
    """
    zero_prob, base_log_mass = torch.broadcast_tensors(zero_prob, base_log_mass)
    return _ZeroAtomLogMass.apply(zero_prob, base_log_mass)


class BetaBinomial(_CountCDF, Distribution):
    """A broadcast beta-binomial component, including zero trials.

    Parameters
    ----------
    total_count : Tensor or int
        Nonnegative integer trial count.
    concentration1 : Tensor or float
        Positive success concentration.
    concentration0 : Tensor or float
        Positive failure concentration.
    validate_args : bool or None
        Whether to validate arguments.
    """

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def _count_log_tails(self, value):
        """Evaluate exact finite beta-binomial tails within the shared budget.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Inclusive log CDF and strict log survival.
        """
        from quivers.continuous._count_tails import beta_binomial_log_tails

        return beta_binomial_log_tails(
            self.total_count, value, self.concentration1, self.concentration0
        )

    def __init__(self, total_count, concentration1, concentration0, validate_args=None):
        _check_trial_count_source(total_count)
        self.total_count, self.concentration1, self.concentration0 = (
            _broadcast_count_parameters(total_count, concentration1, concentration0)
        )
        if not bool(
            (
                torch.isfinite(self.concentration1)
                & (self.concentration1 > 0)
                & torch.isfinite(self.concentration0)
                & (self.concentration0 > 0)
                & torch.isfinite(self.concentration1 + self.concentration0)
            ).all()
        ):
            raise ValueError(
                "beta-binomial concentrations and their sum must be finite"
            )
        super().__init__(self.total_count.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Expected successes.

        Returns
        -------
        Tensor
            Trial count times the beta mean.
        """
        active = self.total_count > 0
        a = torch.where(active, self.concentration1, 1.0)
        b = torch.where(active, self.concentration0, 1.0)
        return self.total_count * (a / (a + b))

    @property
    def variance(self) -> Tensor:
        """Variance after integrating over the beta probability.

        Returns
        -------
        Tensor
            Beta-binomial response variance.
        """
        n = self.total_count
        a = torch.where(n > 0, self.concentration1, 1.0)
        b = torch.where(n > 0, self.concentration0, 1.0)
        phi = a + b
        p = a / phi
        return n * p * (b / phi) * ((phi + n) / (phi + 1))

    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluate the finite count mass in the parameter dtype.

        Parameters
        ----------
        value : Tensor
            Counts broadcastable with the parameter batch.

        Returns
        -------
        Tensor
            Log mass, negative infinity outside the trial support.
        """
        _check_count_observation_source(value)
        n, a, b, value = torch.broadcast_tensors(
            self.total_count,
            self.concentration1,
            self.concentration0,
            value.to(self.concentration1.dtype),
        )
        valid = constraints.nonnegative_integer.check(value) & (value <= n)
        k = torch.where(valid, value, 0.0)
        result = _beta_binomial_log_mass(k, torch.where(valid, n, 0.0), a, b)
        return torch.where(valid, result, -torch.inf)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw a beta probability followed by a binomial count.

        Parameters
        ----------
        sample_shape : torch.Size
            Sample dimensions prepended to the batch.

        Returns
        -------
        Tensor
            Nonnegative integer counts.
        """
        shape = sample_shape + self.batch_shape
        n = self.total_count.expand(shape)
        result = torch.zeros(shape, dtype=torch.int64, device=n.device)
        active = n > 0
        if bool(active.any()):
            concentrations = torch.stack(
                (self.concentration1, self.concentration0), -1
            ).expand(shape + (2,))[active]
            probability = _dirichlet_weights(concentrations)[..., 0]
            result[active] = (
                ClosedBinomial(n[active], probs=probability).sample().long()
            )
        return result


class _InflatedCount(_CountCDF, Distribution):
    """Shared zero-inflation scoring, moments, expansion, and discrete sampling."""

    support = constraints.nonnegative_integer
    has_rsample = False
    zero_prob: Tensor
    _count_class: type[Distribution]
    _zero_count_parameters: tuple[str, ...] = ("total_count", "probs")

    def _all_count_tails(self, value):
        """Reuse the shared structural-zero atom algebra for count events.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Paired logarithmic and ordinary tail probabilities.
        """
        from quivers.continuous._count_tails import inflated_count_tails

        base = self._component()
        return inflated_count_tails(
            self.zero_prob,
            value,
            *base._count_log_tails(value),
        )

    def _count_log_tails(self, value):
        """Select log tails from the common count-mixture event kernel.

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
        """Retain finite atom-probability derivatives at mixture endpoints.

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

    def _component(
        self, parameters: Mapping[str, Tensor] | None = None
    ) -> Distribution:
        """Construct the count component from current or selected parameter views.

        Parameters
        ----------
        parameters : Mapping[str, Tensor] or None
            Selected parameters, or all current batch parameters.

        Returns
        -------
        Distribution
            The count component without its structural zero atom.
        """
        return self._count_class(
            **{
                name: getattr(self, name) if parameters is None else parameters[name]
                for name in self.arg_constraints
                if name != "zero_prob"
            },
            validate_args=False,
        )

    @property
    def mean(self) -> Tensor:
        """Expected count including structural zeros.

        Returns
        -------
        Tensor
            Count-component mean multiplied by its mixture probability.
        """
        return (1 - self.zero_prob) * self._component().mean

    @property
    def variance(self) -> Tensor:
        """Variance including component and structural-zero variation.

        Returns
        -------
        Tensor
            Full response variance.
        """
        base = self._component()
        return (1 - self.zero_prob) * (
            base.variance + self.zero_prob * base.mean.square()
        )

    def expand(self, batch_shape: torch.Size, _instance=None) -> _InflatedCount:
        """Expand parameter views without copying their storage.

        Parameters
        ----------
        batch_shape : torch.Size
            Requested batch shape.
        _instance : _InflatedCount or None
            Optional PyTorch distribution instance.

        Returns
        -------
        _InflatedCount
            Expanded distribution retaining validation and endpoint behavior.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(shape))
        super(_InflatedCount, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value: Tensor) -> Tensor:
        """Score a zero atom or an unrestricted count-component observation.

        Parameters
        ----------
        value : Tensor
            Nonnegative integer observations.

        Returns
        -------
        Tensor
            Exact log probability, including closed mixture endpoints.
        """
        _check_count_observation_source(value)
        if self._validate_args:
            self._validate_sample(value)
        valid = self.support.check(value)
        point = torch.where(valid, value, 0)
        base = self._component()
        at_zero = _zero_atom_log_mass(
            self.zero_prob, base.log_prob(torch.zeros_like(point))
        )
        ordinary_probability = torch.where(valid & (value != 0), self.zero_prob, 0.0)
        ordinary = torch.log1p(-ordinary_probability) + base.log_prob(point)
        return torch.where(
            valid, torch.where(value == 0, at_zero, ordinary), -torch.inf
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample the count component only where its mixture branch is selected.

        Parameters
        ----------
        sample_shape : torch.Size
            Dimensions prepended to the parameter batch.

        Returns
        -------
        Tensor
            Integer counts on the parameters' device.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            active = torch.rand(
                shape, dtype=self.zero_prob.dtype, device=self.zero_prob.device
            ) >= self.zero_prob.expand(shape)
            certain_zero = torch.zeros_like(self.zero_prob, dtype=torch.bool)
            for name in self._zero_count_parameters:
                if name in self.arg_constraints:
                    certain_zero = certain_zero | (getattr(self, name) == 0)
            active = active & ~certain_zero.expand(shape)
            result = torch.zeros(shape, dtype=torch.long, device=self.zero_prob.device)
            if active.any():
                parameters = {
                    name: getattr(self, name).expand(shape)[active]
                    for name in self.arg_constraints
                }
                draws = _require_sample_capacity(self._component(parameters).sample())
                result[active] = draws.long()
            return result


class ZeroInflatedNegativeBinomial(_InflatedCount):
    """Structural zeros plus NB successes before total_count failures.

    Parameters
    ----------
    zero_prob : Tensor or float
        Structural zero probability in [0, 1].
    total_count : Tensor or float
        Nonnegative negative-binomial shape.
    probs : Tensor or float
        Success probability in [0, 1), with count mean r*p/(1-p).
    validate_args : bool or None
        Whether to validate parameters and observations.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "total_count": constraints.nonnegative,
        "probs": constraints.half_open_interval(0.0, 1.0),
    }
    _count_class = ClosedNegativeBinomial

    def __init__(self, zero_prob, total_count, probs, validate_args=None):
        self.zero_prob, self.total_count, self.probs = _broadcast_count_parameters(
            zero_prob, total_count, probs
        )
        super().__init__(self.zero_prob.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Expected count with an exact zero-probability endpoint.

        Returns
        -------
        Tensor
            Mixture probability times r*p/(1-p).
        """
        return (1 - self.zero_prob) * self.total_count * self.probs / (1 - self.probs)

    @property
    def variance(self) -> Tensor:
        """Variance in the original probability parameterization.

        Returns
        -------
        Tensor
            Within-count variance plus structural-zero variation.
        """
        base_mean = self.total_count * self.probs / (1 - self.probs)
        return (1 - self.zero_prob) * (
            base_mean / (1 - self.probs) + self.zero_prob * base_mean.square()
        )


class ZeroInflatedBinomial(_InflatedCount):
    """Structural zeros plus a binomial, including zero trials.

    Parameters
    ----------
    zero_prob : Tensor or float
        Structural zero probability in [0, 1].
    total_count : Tensor or int
        Nonnegative integer trial counts.
    probs : Tensor or float
        Binomial success probability in [0, 1].
    validate_args : bool or None
        Whether to validate parameters and observations.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
    }
    _count_class = ClosedBinomial

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        """Counts between zero and the trial count.

        Returns
        -------
        constraints.Constraint
            The parameter-dependent integer support.
        """
        return constraints.integer_interval(0, self.total_count)

    def __init__(self, zero_prob, total_count, probs, validate_args=None):
        _check_trial_count_source(total_count)
        self.zero_prob, self.total_count, self.probs = _broadcast_count_parameters(
            zero_prob, total_count, probs
        )
        super().__init__(self.zero_prob.shape, validate_args=validate_args)


class ZeroInflatedBetaBinomial(ZeroInflatedBinomial):
    """Structural zeros plus a beta-binomial with integrated success probability.

    Parameters
    ----------
    zero_prob : Tensor or float
        Structural zero probability in [0, 1].
    total_count : Tensor or int
        Nonnegative integer trial counts.
    concentration1 : Tensor or float
        Positive beta success concentration.
    concentration0 : Tensor or float
        Positive beta failure concentration.
    validate_args : bool or None
        Whether to validate parameters and observations.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "total_count": constraints.nonnegative_integer,
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    _count_class = BetaBinomial

    def __init__(
        self, zero_prob, total_count, concentration1, concentration0, validate_args=None
    ):
        _check_trial_count_source(total_count)
        self.zero_prob, self.total_count, self.concentration1, self.concentration0 = (
            _broadcast_count_parameters(
                zero_prob, total_count, concentration1, concentration0
            )
        )
        Distribution.__init__(self, self.zero_prob.shape, validate_args=validate_args)


def _logseries_sample(probs: Tensor) -> Tensor:
    """Draw logarithmic-series jumps using the constant-work Kemp algorithm.

    Parameters
    ----------
    probs : Tensor
        Probabilities in [0, 1), with the point-mass-at-one limit at zero.

    Returns
    -------
    Tensor
        Positive integer jumps, one per input probability.

    References
    ----------
    Kemp (1981), algorithm LS; also used by NumPy random_logseries.
    https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c
    """
    v = 1 - torch.rand_like(probs)
    q = -torch.expm1(torch.log1p(-probs) * torch.rand_like(probs))
    relevant = v < probs
    large = relevant & (v <= q.square())
    result = torch.ones_like(probs, dtype=torch.long)
    jumps = _require_sample_capacity(torch.floor(1 + v[large].log() / q[large].log()))
    result[large] = jumps.long()
    result[relevant & ~large & (v < q)] = 2
    return result


def _positive_negative_binomial_sample(total_count: Tensor, probs: Tensor) -> Tensor:
    """Draw a positive NB without rare-event rejection.

    Parameters
    ----------
    total_count : Tensor
        Positive shapes, already expanded to the desired sample layout.
    probs : Tensor
        Corresponding success probabilities in [0, 1).

    Returns
    -------
    Tensor
        Positive counts. Low positive-mass cells use a conditioned compound
        Poisson sum; other cells use rejection with acceptance at least one half.
    """
    rate = -total_count * torch.log1p(-probs)
    small = rate < math.log(2)
    result = torch.empty_like(probs, dtype=torch.long)
    if small.any():
        rates, probabilities = rate[small], probs[small]
        multiplicities = positive_poisson_sample(rates).long()
        positions = torch.arange(rates.numel(), device=probs.device).repeat_interleave(
            multiplicities
        )
        jumps = _logseries_sample(probabilities.repeat_interleave(multiplicities))
        totals = torch.zeros_like(rates, dtype=torch.float64).index_add(
            0, positions, jumps.double()
        )
        result[small] = _require_sample_capacity(totals).long()
    remaining = torch.nonzero(~small.reshape(-1), as_tuple=False).flatten()
    flat_result, flat_r, flat_p = (
        result.reshape(-1),
        total_count.reshape(-1),
        probs.reshape(-1),
    )
    while remaining.numel():
        draws = (
            ClosedNegativeBinomial(
                flat_r[remaining], probs=flat_p[remaining], validate_args=False
            )
            .sample()
            .long()
        )
        flat_result[remaining] = draws
        remaining = remaining[draws == 0]
    return result


class HurdleNegativeBinomial(ZeroInflatedNegativeBinomial):
    """Zero atom plus a normalized strictly positive negative binomial.

    Parameters
    ----------
    zero_prob : Tensor or float
        Hurdle probability in [0, 1].
    total_count : Tensor or float
        Strictly positive negative-binomial shape.
    probs : Tensor or float
        Success probability in [0, 1). At zero the positive component is
        its point-mass-at-one limit.
    validate_args : bool or None
        Whether to validate parameters and observations.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "total_count": constraints.positive,
        "probs": constraints.half_open_interval(0.0, 1.0),
    }

    def __init__(self, zero_prob, total_count, probs, validate_args=None):
        super().__init__(zero_prob, total_count, probs, validate_args=validate_args)

    def _all_count_tails(self, value):
        """Reuse positive conditioning before structural-zero event mixing.

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

        base = self._component()
        log_zero = self.total_count * torch.log1p(-self.probs)
        first = log_zero + self._positive_log_ratio()
        second = first + torch.log1p((self.total_count + 1) * self.probs / 2)
        lc, ls = positive_count_log_tails(
            value,
            *base._count_log_tails(value),
            log_zero,
            first,
            second,
        )
        return inflated_count_tails(self.zero_prob, value, lc, ls)

    def _positive_log_ratio(self) -> Tensor:
        """Retain tiny perturbations in the positive-NB normalization ratio.

        Returns
        -------
        Tensor
            Logarithm of r*p/P(NB>0), including its zero-probability limit.
        """
        t = -torch.log1p(-self.probs)
        return positive_poisson_log_mean(
            self.total_count * t
        ) - positive_poisson_log_mean(t)

    def _positive_ratio(self) -> Tensor:
        """Evaluate r*p/P(NB>0) including its smooth p=0 limit.

        Returns
        -------
        Tensor
            A stable ratio equal to one at p=0.
        """
        return self._positive_log_ratio().exp()

    @property
    def mean(self) -> Tensor:
        """Expected hurdle count.

        Returns
        -------
        Tensor
            Positive conditional mean times the nonzero probability.
        """
        return (1 - self.zero_prob) * self._positive_ratio() / (1 - self.probs)

    @property
    def variance(self) -> Tensor:
        """Variance of the full hurdle response.

        Returns
        -------
        Tensor
            Second moment minus the squared full mean.
        """
        mean = self.mean
        return mean * ((1 + self.total_count * self.probs) / (1 - self.probs) - mean)

    def log_prob(self, value: Tensor) -> Tensor:
        """Score the atom or normalized positive component without p=0 divisions.

        Parameters
        ----------
        value : Tensor
            Nonnegative integer counts.

        Returns
        -------
        Tensor
            Exact log probabilities including the p=0 limit.
        """
        _check_count_observation_source(value)
        if self._validate_args:
            self._validate_sample(value)
        valid = self.support.check(value)
        point = torch.where(valid & (value > 0), value, 1).to(self.probs.dtype)
        positive = (
            _log_gamma_increment(self.total_count + 1, point - 1)
            - torch.lgamma(point + 1)
            + _count_log(point - 1, self.probs)
            + self.total_count * torch.log1p(-self.probs)
            + self._positive_log_ratio()
        )
        rate = -self.total_count * torch.log1p(-self.probs)
        ordinary = rate > 1e-4
        safe_p = torch.where(ordinary, self.probs, torch.full_like(self.probs, 0.5))
        safe_r = torch.where(
            ordinary, self.total_count, torch.ones_like(self.total_count)
        )
        stable = _negative_binomial_log_mass(
            point, safe_r, safe_p.log(), torch.log1p(-safe_p)
        )
        stable = stable - torch.log(-torch.expm1(safe_r * torch.log1p(-safe_p)))
        positive = torch.where(ordinary, stable, positive)
        atom_probability = torch.where(valid & (value == 0), self.zero_prob, 1.0)
        positive_probability = torch.where(valid & (value > 0), self.zero_prob, 0.0)
        result = torch.where(
            value == 0,
            atom_probability.log(),
            torch.log1p(-positive_probability) + positive,
        )
        return torch.where(valid, result, -torch.inf)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw structural zeros and efficiently conditioned positive NB counts.

        Parameters
        ----------
        sample_shape : torch.Size
            Dimensions prepended to the batch shape.

        Returns
        -------
        Tensor
            Nonnegative integer draws without reparameterized gradients.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            active = torch.rand(
                shape, dtype=self.zero_prob.dtype, device=self.zero_prob.device
            ) >= self.zero_prob.expand(shape)
            result = torch.zeros(shape, dtype=torch.long, device=self.zero_prob.device)
            if active.any():
                result[active] = _positive_negative_binomial_sample(
                    self.total_count.expand(shape)[active],
                    self.probs.expand(shape)[active],
                )
            return result


__all__ = [
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedBinomial",
    "ZeroInflatedBetaBinomial",
    "HurdleNegativeBinomial",
]
