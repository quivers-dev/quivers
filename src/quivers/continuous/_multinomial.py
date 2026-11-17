"""Vector counts with heterogeneous trials and stable conditional scores."""

from __future__ import annotations

from typing import cast
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import lazy_property

from quivers.continuous._simplex_sampling import _dirichlet_weights
from quivers.continuous._closed_probability import (
    ClosedBinomial,
    _ValidatedCategorical,
    _binomial_log_mass,
    _beta_binomial_log_mass,
    _check_trial_count_source,
    _check_count_observation_source,
)


def _log_tails(logits: Tensor) -> Tensor:
    """Sum suffix log weights without undefined all-zero-tail derivatives.

    Parameters
    ----------
    logits : Tensor
        Normalized log weights, allowing negative-infinite zero categories.

    Returns
    -------
    Tensor
        Stable suffix log sums on the original category axis.
    """
    running = logits[..., -1]
    values = [running]
    for index in range(logits.shape[-1] - 2, -1, -1):
        value = logits[..., index]
        absent = (value == -torch.inf) & (running == -torch.inf)
        combined = torch.logaddexp(
            torch.where(absent, 0.0, value), torch.where(absent, 0.0, running)
        )
        running = torch.where(absent, -torch.inf, combined)
        values.append(running)
    return torch.stack(values[::-1], -1)


def _count_parameters(total_count, vector):
    """Broadcast a scalar trial parameter against a trailing event vector.

    Parameters
    ----------
    total_count : Tensor or int
        Nonnegative, exactly representable trial counts.
    vector : Tensor or Sequence[float]
        Parameter vectors with one trailing category axis.

    Returns
    -------
    tuple[Tensor, Tensor]
        Broadcast Float64 trials and vector parameters.

    Raises
    ------
    ValueError
        If trials or the category axis violate their contracts.
    """
    device = (
        vector.device
        if isinstance(vector, Tensor)
        else total_count.device
        if isinstance(total_count, Tensor)
        else None
    )
    _check_trial_count_source(total_count)
    source_n = torch.as_tensor(total_count, device=device)
    _check_trial_count_source(source_n)
    if source_n.dtype == torch.bool:
        raise ValueError("vector count trials must be integers, not booleans")
    # Only vector.shape[:-1] participates in scalar trial broadcasting.
    raw_vector = torch.as_tensor(vector, dtype=torch.float64, device=device)
    if raw_vector.ndim < 1 or raw_vector.shape[-1] < 1:
        raise ValueError("vector counts require a nonempty trailing category axis")
    raw_n = torch.as_tensor(total_count, dtype=torch.float64, device=device)
    shape = torch.broadcast_shapes(raw_n.shape, raw_vector.shape[:-1])
    n = raw_n.expand(shape)
    values = raw_vector.expand(*shape, raw_vector.shape[-1])
    if not bool((torch.isfinite(n) & (n >= 0) & (n == n.floor()) & (n <= 2**53)).all()):
        raise ValueError(
            "vector count trials must be finite nonnegative integers <= 2**53"
        )
    return n, values


def _count_event(value, total_count: Tensor, categories: int):
    """Check an integer event and mask invalid events before count arithmetic.

    Parameters
    ----------
    value : Tensor or Sequence[int]
        Candidate events with a trailing category axis.
    total_count : Tensor
        Trial counts on the distribution batch.
    categories : int
        Declared event width.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Safe count events, broadcast trial counts, and event validity.

    Raises
    ------
    ValueError
        If event width or exact count representation is invalid.
    """
    _check_count_observation_source(value)
    point = torch.as_tensor(value, dtype=torch.float64, device=total_count.device)
    if point.ndim < 1 or point.shape[-1] != categories:
        raise ValueError(
            f"count event requires a trailing category axis of size {categories}"
        )
    shape = torch.broadcast_shapes(point.shape[:-1], total_count.shape)
    point = point.expand(*shape, categories)
    n = total_count.expand(shape)
    valid = (
        torch.isfinite(point)
        & (point >= 0)
        & (point == point.floor())
        & (point <= n.unsqueeze(-1))
    ).all(-1)
    # Accumulate integers so a rounded Float64 sum cannot admit an invalid
    # event at the exact-integer boundary. Clamping only guards conversion
    # of invalid values; those events are scored as impossible below.
    safe = torch.where(valid.unsqueeze(-1), point, 0.0)
    remaining = n
    for index in range(categories):
        valid = valid & (safe[..., index] <= remaining)
        remaining = (remaining - safe[..., index]).clamp_min(0)
    valid = valid & (remaining == 0)
    return torch.where(valid.unsqueeze(-1), safe, 0.0), n, valid


def _other_categories(value: Tensor) -> Tensor:
    """Sum other category weights without subtracting a dominant category.

    Parameters
    ----------
    value : Tensor
        Nonnegative category weights on the trailing axis.

    Returns
    -------
    Tensor
        The sum of every category except the corresponding component.
    """
    zero = torch.zeros_like(value[..., :1])
    before = torch.cat((zero, value[..., :-1].cumsum(-1)), -1)
    after = torch.cat((value[..., 1:].flip(-1).cumsum(-1).flip(-1), zero), -1)
    return before + after


def _remaining_counts(point: Tensor, n: Tensor) -> Tensor:
    """Resolve the trial count before each conditional binomial event.

    Parameters
    ----------
    point : Tensor
        Valid count events.
    n : Tensor
        Broadcast trial counts.

    Returns
    -------
    Tensor
        Remaining counts before each of the first K-1 categories.
    """
    prefix = point[..., :-1].cumsum(-1)
    previous = torch.cat((torch.zeros_like(n).unsqueeze(-1), prefix[..., :-1]), -1)
    return n.unsqueeze(-1) - previous


def _multinomial_sample(
    n: Tensor, probs: Tensor, sample_shape: torch.Size | Sequence[int] = torch.Size()
) -> Tensor:
    """Draw heterogeneous multinomials using K-1 vectorized binomial draws.

    Parameters
    ----------
    n : Tensor
        Exact Float64 trial counts on the batch axes.
    probs : Tensor
        Normalized category probabilities with a trailing event axis.
    sample_shape : torch.Size or Sequence[int]
        Leading sample axes.

    Returns
    -------
    Tensor
        Int64 events, each summing exactly to its trial count.

    Raises
    ------
    RuntimeError
        If an underlying Binomial draw exceeds the remaining trial support.
    """
    shape = torch.Size(sample_shape) + n.shape
    remaining = n.expand(shape)
    tail = probs.flip(-1).cumsum(-1).flip(-1)
    draws = []
    for category in range(probs.shape[-1] - 1):
        denominator = tail[..., category]
        conditional = torch.where(
            denominator > 0,
            probs[..., category] / torch.where(denominator > 0, denominator, 1.0),
            0.0,
        )
        value = ClosedBinomial(
            remaining, probs=conditional.expand(shape), validate_args=False
        ).sample()
        draws.append(value)
        remaining = remaining - value
    return torch.stack((*draws, remaining), -1).to(torch.int64)


class Multinomial(Distribution):
    """Multinomial events with a separate trial count on each batch row.

    Parameters
    ----------
    total_count : Tensor or int
        Nonnegative integer trials, exactly representable through 2**53.
    probs : Tensor or Sequence[float] or None
        Finite nonnegative category weights with a positive row total.
    logits : Tensor or Sequence[float] or None
        Alternative category log weights; provide exactly one representation.
    validate_args : bool or None
        Whether to validate observation support through torch.
    """

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.simplex,
        "logits": constraints.independent(constraints.real, 1),
    }
    support = constraints.independent(constraints.nonnegative_integer, 1)
    has_rsample = False

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Multinomial requires exactly one of probs and logits")
        vector = probs if probs is not None else logits
        self.total_count, parameter = _count_parameters(total_count, vector)
        self._from_probs = probs is not None
        self._parameter = parameter
        # Validate the original representation before zero-trial masking.
        with torch.no_grad():
            if probs is not None:
                _ValidatedCategorical(probs=parameter)
            else:
                valid = torch.isfinite(parameter) | (parameter == -torch.inf)
                if not bool(valid.all() and torch.isfinite(parameter).any(-1).all()):
                    raise ValueError(
                        "Multinomial logits require a finite category and no NaN or positive infinity"
                    )
                _ValidatedCategorical(logits=parameter)
        super().__init__(
            self.total_count.shape,
            torch.Size((parameter.shape[-1],)),
            validate_args=False,
        )
        self._validate_args = bool(validate_args)

    def _categorical(self, active=None):
        """Normalize only active parameters without inactive singular derivatives.

        Parameters
        ----------
        active : Tensor or None
            Mask of rows entering a response operation, or None for all rows.

        Returns
        -------
        _ValidatedCategorical
            Normalized probabilities and log weights.
        """
        parameter = self._parameter
        if active is not None:
            parameter = torch.where(
                active.unsqueeze(-1), parameter, 1.0 if self._from_probs else 0.0
            )
        return (
            _ValidatedCategorical(probs=parameter)
            if self._from_probs
            else _ValidatedCategorical(logits=parameter)
        )

    @lazy_property
    def probs(self):
        """Normalized category probabilities, retaining the original zero-trial input.

        Returns
        -------
        Tensor
            Probabilities on batch and category axes.
        """
        return self._categorical().probs

    @lazy_property
    def logits(self):
        """Normalized category log weights with exact structural zeros.

        Returns
        -------
        Tensor
            Log probabilities on batch and category axes.
        """
        return self._categorical().logits

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views while preserving category event axes.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested batch shape.
        _instance : Multinomial or None
            Optional torch expansion instance.

        Returns
        -------
        Multinomial
            Expanded distribution with unchanged category count.
        """
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new._parameter = self._parameter.expand(*batch_shape, *self.event_shape)
        new._from_probs = self._from_probs
        Distribution.__init__(new, batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        """Score whole count vectors by stable conditional binomial masses.

        Parameters
        ----------
        value : Tensor or Sequence[int]
            Count events with the declared trailing category axis.

        Returns
        -------
        Tensor
            One Float64 log mass per sample and batch event.
        """
        point, n, valid = _count_event(value, self.total_count, self.event_shape[0])
        active = valid & (n > 0)
        categorical = self._categorical(active)
        n = torch.where(valid, n, 0.0)
        if self.event_shape[0] == 1:
            return torch.where(valid, n * 0 + categorical.probs.sum(-1) * 0, -torch.inf)
        remaining = _remaining_counts(point, n)
        k = point[..., :-1]
        if self._from_probs:
            probabilities = categorical.probs
            tail = probabilities.flip(-1).cumsum(-1).flip(-1)
            denom = tail[..., :-1]
            p = torch.where(
                denom > 0,
                probabilities[..., :-1] / torch.where(denom > 0, denom, 1.0),
                0.0,
            )
            scores = ClosedBinomial(remaining, probs=p, validate_args=False).log_prob(k)
        else:
            logits = cast(Tensor, categorical.logits)
            tail = _log_tails(logits)
            active = torch.isfinite(tail[..., :-1])
            base = torch.where(active, tail[..., :-1], 0.0)
            log_p = logits[..., :-1] - base
            log_q = tail[..., 1:] - base
            ordinary = active & torch.isfinite(log_p) & torch.isfinite(log_q)
            scores = _binomial_log_mass(
                k,
                remaining,
                torch.where(ordinary, log_p, -0.6931471805599453),
                torch.where(ordinary, log_q, -0.6931471805599453),
            )
            left = torch.where(
                k == 0, 0.0, k * torch.where(torch.isfinite(log_p), log_p, 0.0)
            )
            right = torch.where(
                remaining == k,
                0.0,
                (remaining - k) * torch.where(torch.isfinite(log_q), log_q, 0.0),
            )
            endpoint = left + right
            impossible = ((k > 0) & ~torch.isfinite(log_p)) | (
                (remaining > k) & ~torch.isfinite(log_q)
            )
            endpoint = torch.where(impossible, -torch.inf, endpoint)
            scores = torch.where(ordinary, scores, endpoint)
        return torch.where(valid, scores.sum(-1), -torch.inf)

    def sample(self, sample_shape: torch.Size | Sequence[int] = torch.Size()):
        """Draw Int64 events in O(categories times rows) work.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Int64 events with exactly the requested trial totals.
        """
        with torch.no_grad():
            return _multinomial_sample(
                self.total_count,
                self._categorical(self.total_count > 0).probs,
                sample_shape,
            )

    @property
    def mean(self):
        """Expected category counts.

        Returns
        -------
        Tensor
            Float64 counts on the batch and category axes.
        """
        return (
            self.total_count.unsqueeze(-1)
            * self._categorical(self.total_count > 0).probs
        )

    @property
    def variance(self):
        """Marginal category variances.

        Returns
        -------
        Tensor
            Float64 variances on batch and category axes.
        """
        p = self._categorical(self.total_count > 0).probs
        return self.total_count.unsqueeze(-1) * p * _other_categories(p)


class DirichletMultinomial(Distribution):
    """Multinomial events with Dirichlet-distributed category probabilities.

    Parameters
    ----------
    total_count : Tensor or int
        Nonnegative integer trials through 2**53 on the batch axes.
    concentration : Tensor or Sequence[float]
        Finite strictly positive category concentrations.
    validate_args : bool or None
        Whether torch validates normalized parameters and observations.
    """

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "concentration": constraints.independent(constraints.positive, 1),
    }
    support = constraints.independent(constraints.nonnegative_integer, 1)
    has_rsample = False

    def __init__(self, total_count, concentration, validate_args=None):
        self.total_count, self.concentration = _count_parameters(
            total_count, concentration
        )
        if not bool(
            (torch.isfinite(self.concentration) & (self.concentration > 0)).all()
        ):
            raise ValueError(
                "DirichletMultinomial concentrations must be finite and positive"
            )
        if not bool(torch.isfinite(self.concentration.sum(-1)).all()):
            raise ValueError(
                "DirichletMultinomial concentration totals exceed Float64 capacity"
            )
        super().__init__(
            self.total_count.shape,
            torch.Size((self.concentration.shape[-1],)),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        """Expand trials and concentration views without merging event axes.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested batch shape.
        _instance : DirichletMultinomial or None
            Optional torch expansion instance.

        Returns
        -------
        DirichletMultinomial
            Expanded count-vector distribution.
        """
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.concentration = self.concentration.expand(*batch_shape, *self.event_shape)
        Distribution.__init__(new, batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        """Score the event using its exact conditional beta-binomial factorization.

        Parameters
        ----------
        value : Tensor or Sequence[int]
            Count vectors on the trailing category axis.

        Returns
        -------
        Tensor
            Float64 log masses on the sample and batch axes.
        """
        point, n, valid = _count_event(value, self.total_count, self.event_shape[0])
        n = torch.where(valid, n, 0.0)
        concentration = torch.where((n > 0).unsqueeze(-1), self.concentration, 1.0)
        if self.event_shape[0] == 1:
            return torch.where(valid, n * 0 + concentration.sum(-1) * 0, -torch.inf)
        tail = concentration.flip(-1).cumsum(-1).flip(-1)
        scores = _beta_binomial_log_mass(
            point[..., :-1],
            _remaining_counts(point, n),
            concentration[..., :-1],
            tail[..., 1:],
        )
        return torch.where(valid, scores.sum(-1), -torch.inf)

    def sample(self, sample_shape: torch.Size | Sequence[int] = torch.Size()):
        """Draw Dirichlet weights and use the shared heterogeneous-trial sampler.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Int64 count events with exact trial totals.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            totals = self.total_count.expand(shape[:-1])
            concentration = self.concentration.expand(shape)
            active = totals > 0
            result = torch.zeros(shape, dtype=torch.int64, device=totals.device)
            if bool(active.any()):
                probabilities = _dirichlet_weights(concentration[active])
                result[active] = _multinomial_sample(totals[active], probabilities)
            return result

    @property
    def mean(self):
        """Expected category counts.

        Returns
        -------
        Tensor
            Float64 counts including exact zero-trial rows.
        """
        concentration = torch.where(
            (self.total_count > 0).unsqueeze(-1), self.concentration, 1.0
        )
        return self.total_count.unsqueeze(-1) * (
            concentration / concentration.sum(-1, keepdim=True)
        )

    @property
    def variance(self):
        """Marginal category variances, including Dirichlet overdispersion.

        Returns
        -------
        Tensor
            Float64 marginal variances.
        """
        concentration = torch.where(
            (self.total_count > 0).unsqueeze(-1), self.concentration, 1.0
        )
        total = concentration.sum(-1, keepdim=True)
        p = concentration / total
        return (
            self.total_count.unsqueeze(-1)
            * p
            * (_other_categories(concentration) / total)
            * ((self.total_count.unsqueeze(-1) + total) / (1 + total))
        )


__all__ = ["Multinomial", "DirichletMultinomial"]
