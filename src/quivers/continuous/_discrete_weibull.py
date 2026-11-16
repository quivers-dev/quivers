"""Discrete Weibull counts with stable tails and bounded moment calculations."""

import math

import torch
from quivers.continuous._closed_probability import _check_count_observation_source
from torch import Tensor
from torch.distributions import Distribution, constraints

from quivers.continuous._closed_probability import _require_sample_capacity
from quivers.continuous._gamma_tails import log_gamma_tails
from quivers.continuous._shape_families import (
    _check_probability,
    _log1mexp_neg_exp,
)
from quivers.continuous._zero_inflated_count import _broadcast_count_parameters

_LOG_MAX = math.log(torch.finfo(torch.float64).max)


def _negative_exp(log_value: Tensor) -> Tensor:
    """Negate an exponential while masking overflowing inactive derivatives.

    Parameters
    ----------
    log_value : Tensor
        Logarithms of nonnegative magnitudes.

    Returns
    -------
    Tensor
        Negative exponentials, including negative infinity.
    """
    finite = log_value <= _LOG_MAX
    safe = torch.where(finite, log_value, 0.0)
    return torch.where(finite, -safe.exp(), -torch.inf)


def _log_expm1_from_log(log_value: Tensor) -> Tensor:
    """Evaluate log(expm1(exp(x))) across small and large positive increments.

    Parameters
    ----------
    log_value : Tensor
        Logarithms of strictly positive increments.

    Returns
    -------
    Tensor
        Log exponentials minus one without intermediate overflow.
    """
    small = log_value < -20
    large = log_value > math.log(40)
    safe = torch.where(small | large, 0.0, log_value).exp()
    regular = torch.log(torch.expm1(safe))
    very_small = torch.exp(torch.where(small, log_value, 0.0))
    small_result = log_value + torch.log1p(very_small / 2 + very_small.square() / 6)
    large_result = torch.exp(torch.where(large, log_value.clamp_max(_LOG_MAX), 0.0))
    return torch.where(small, small_result, torch.where(large, large_result, regular))


def _integer_interval_log_mass(
    probs: Tensor, concentration: Tensor, first: Tensor, last: Tensor
) -> Tensor:
    """Factor the probability between nonnegative integer endpoints.

    Parameters
    ----------
    probs, concentration : Tensor
        Survival parameter and positive Weibull power.
    first, last : Tensor
        Inclusive integer endpoints, with last possibly positive infinity.

    Returns
    -------
    Tensor
        Log interval masses without subtracting rounded CDFs or powers.
    """
    positive = first > 0
    m = torch.where(positive, first, 1.0)
    q = torch.where(probs > 0, probs, 0.5)
    log_rate = (-q.log()).log()
    log_magnitude = log_rate + concentration * m.log()
    survival = _negative_exp(log_magnitude)
    finite = torch.isfinite(last)
    n = torch.where(finite, last, first)
    # Subtract the integer endpoints before adding one; first+1 could round
    # back to first at the exact Float64 count boundary.
    log_ratio = torch.log1p((n - first + 1) / m)
    log_increment = concentration.log() + log_ratio.log()
    log_difference = log_magnitude + _log_expm1_from_log(log_increment)
    ordinary = survival + _log1mexp_neg_exp(log_difference)
    ordinary = torch.where(finite, ordinary, survival)
    cdf = _log1mexp_neg_exp(log_rate + concentration * torch.log1p(n))
    cdf = torch.where(n == 0, torch.log1p(-probs), cdf)
    cdf = torch.where(finite, cdf, 0.0)
    result = torch.where(positive, ordinary, cdf)
    result = torch.where(probs > 0, result, torch.where(positive, -torch.inf, 0.0))
    # The zero cell has finite one-sided probability derivatives at q=0.
    return torch.where((first == 0) & (last == 0), torch.log1p(-probs), result)


def _log_mass(probs: Tensor, concentration: Tensor, value: Tensor) -> Tensor:
    """Score one integer through the common factored interval kernel.

    Parameters
    ----------
    probs, concentration : Tensor
        Survival probability and Weibull power.
    value : Tensor
        Candidate count observations.

    Returns
    -------
    Tensor
        Log mass, negative infinity outside integer support.
    """
    valid = torch.isfinite(value) & (value >= 0) & (value == value.floor())
    point = torch.where(valid, value, 0.0)
    result = _integer_interval_log_mass(probs, concentration, point, point)
    return torch.where(valid, result, -torch.inf)


class DiscreteWeibull(Distribution):
    """Counts with survival ``P(Y >= k) = probs**(k**concentration)``.

    Parameters
    ----------
    probs : Tensor or float
        Survival parameter in [0, 1), with zero giving the point mass at zero.
    concentration : Tensor or float
        Strictly positive power parameter.
    validate_args : bool or None
        Whether to validate distribution parameters.
    moment_rtol : float
        Relative tolerance certified by omitted-moment integral bounds.
    moment_max_terms : int
        Maximum number of count cells used for a moment calculation.

    Notes
    -----
    Density and inverse-transform sampling require constant work per row.
    Moments use adaptive blocks and analytic upper bounds on the omitted
    tails. Failure to certify the requested tolerance raises explicitly.
    Native count arithmetic uses Float64 and integer samples use Int64.
    """

    arg_constraints = {
        "probs": constraints.half_open_interval(0.0, 1.0),
        "concentration": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        probs,
        concentration,
        validate_args=None,
        *,
        moment_rtol=1e-10,
        moment_max_terms=262144,
    ):
        if (
            not 0 < moment_rtol < 1
            or not isinstance(moment_max_terms, int)
            or moment_max_terms < 1
        ):
            raise ValueError(
                "DiscreteWeibull moment tolerance must lie in (0,1) and term budget must be positive"
            )
        self.probs, self.concentration = _broadcast_count_parameters(
            probs, concentration
        )
        self.moment_rtol = moment_rtol
        self.moment_max_terms = moment_max_terms
        super().__init__(self.probs.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """Expand the parameter views and retain the moment resource policy.

        Parameters
        ----------
        batch_shape : torch.Size
            Target batch dimensions.
        _instance : Distribution or None
            Optional reusable instance.

        Returns
        -------
        DiscreteWeibull
            Expanded distribution.
        """
        new = self._get_checked_instance(DiscreteWeibull, _instance)
        shape = torch.Size(batch_shape)
        new.probs, new.concentration = (
            self.probs.expand(shape),
            self.concentration.expand(shape),
        )
        new.moment_rtol, new.moment_max_terms = self.moment_rtol, self.moment_max_terms
        super(DiscreteWeibull, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        """Evaluate a count mass in Float64.

        Parameters
        ----------
        value : Tensor
            Candidate integer counts.

        Returns
        -------
        Tensor
            Log mass, negative infinity outside the support.
        """
        _check_count_observation_source(value)
        return _log_mass(self.probs, self.concentration, value.double())

    def log_interval_mass(self, low, high, *, lower_closed=True, upper_closed=True):
        """Evaluate an integer interval without subtracting rounded tail values.

        Parameters
        ----------
        low, high : Tensor or float or None
            Real endpoints; None leaves a side unbounded.
        lower_closed, upper_closed : bool or Tensor
            Endpoint membership, optionally varying across batch rows.

        Returns
        -------
        Tensor
            Exact factored log probabilities on the distribution batch.

        Raises
        ------
        ValueError
            If an endpoint is NaN.
        TypeError
            If endpoint membership is not Boolean.
        """
        from quivers.continuous._interval import _closed_flag

        lower_closed = _closed_flag(lower_closed, "lower_closed")
        upper_closed = _closed_flag(upper_closed, "upper_closed")
        lower = torch.as_tensor(
            -torch.inf if low is None else low,
            dtype=torch.float64,
            device=self.probs.device,
        )
        upper = torch.as_tensor(
            torch.inf if high is None else high,
            dtype=torch.float64,
            device=self.probs.device,
        )
        if bool(torch.isnan(lower).any()) or bool(torch.isnan(upper).any()):
            raise ValueError("interval endpoints must not be NaN")
        first = torch.where(
            torch.as_tensor(lower_closed, device=lower.device),
            lower.ceil(),
            lower.floor() + 1,
        ).clamp_min(0)
        last = torch.where(
            torch.as_tensor(upper_closed, device=upper.device),
            upper.floor(),
            upper.ceil() - 1,
        )
        valid = torch.isfinite(first) & (last >= first)
        first, last = torch.where(valid, first, 0.0), torch.where(valid, last, 0.0)
        mass = _integer_interval_log_mass(self.probs, self.concentration, first, last)
        return torch.where(valid, mass, -torch.inf)

    def _log_tails(self, value):
        """Evaluate log count tails with finite interval-bound derivatives.

        Parameters
        ----------
        value : Tensor
            Real count bounds, including infinite endpoints.

        Returns
        -------
        tuple[Tensor, Tensor]
            Log cumulative and strict survival probabilities.
        """
        ordinary = torch.isfinite(value) & (value >= 0)
        count = torch.where(ordinary, value.floor() + 1, 1.0)
        q = torch.where(self.probs > 0, self.probs, 0.5)
        log_magnitude = (-q.log()).log() + self.concentration * count.log()
        cdf = _log1mexp_neg_exp(log_magnitude)
        survival = _negative_exp(log_magnitude)
        cdf = torch.where(self.probs == 0, 0.0, cdf)
        survival = torch.where(self.probs == 0, -torch.inf, survival)
        # The first count cell retains its closed-endpoint derivative at q=0.
        cdf = torch.where(count == 1, torch.log1p(-self.probs), cdf)
        cdf = torch.where(value < 0, -torch.inf, cdf)
        survival = torch.where(value < 0, 0.0, survival)
        cdf = torch.where(value == torch.inf, 0.0, cdf)
        survival = torch.where(value == torch.inf, -torch.inf, survival)
        return cdf, survival

    def log_cdf(self, value):
        """Evaluate probability at or below a real count bound in log space.

        Parameters
        ----------
        value : Tensor
            Real count bounds.

        Returns
        -------
        Tensor
            Log cumulative probabilities.
        """
        return self._log_tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate probability strictly above a real count bound in log space.

        Parameters
        ----------
        value : Tensor
            Real count bounds.

        Returns
        -------
        Tensor
            Log strict survival probabilities.
        """
        return self._log_tails(value)[1]

    def cdf(self, value):
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Real count bounds.

        Returns
        -------
        Tensor
            Cumulative probabilities.
        """
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Invert the staircase cumulative distribution.

        Parameters
        ----------
        value : Tensor
            Probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Float64 count quantiles, positive infinity at probability one
            except for the degenerate-zero law.

        Raises
        ------
        RuntimeError
            If a finite requested quantile exceeds exact Float64 count capacity.
        """
        value = value.double()
        if self._validate_args:
            _check_probability(value)
        interior = (value > 0) & (value < 1) & (self.probs > 0)
        probability = torch.where(interior, value, 0.5)
        q = torch.where(self.probs > 0, self.probs, 0.5)
        log_root = (
            (-torch.log1p(-probability)).log() - (-q.log()).log()
        ) / self.concentration
        root = log_root.exp()
        candidate = (root.ceil() - 1).clamp_min(0)
        _require_sample_capacity(torch.where(interior, candidate, 0.0))
        candidate = torch.where(
            self.cdf(candidate) < probability, candidate + 1, candidate
        )
        previous = (candidate - 1).clamp_min(0)
        candidate = torch.where(
            (candidate > 0) & (self.cdf(previous) >= probability), previous, candidate
        )
        result = torch.where(interior, candidate, 0.0)
        return torch.where((value == 1) & (self.probs > 0), torch.inf, result)

    def sample(self, sample_shape=torch.Size()):
        """Draw integer counts by a vectorized exponential transform.

        Parameters
        ----------
        sample_shape : torch.Size
            Dimensions prepended to the parameter batch.

        Returns
        -------
        Tensor
            Int64 counts on the parameters' device.

        Raises
        ------
        RuntimeError
            If a generated count exceeds exact Float64 integer capacity.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            exponential = torch.empty(
                shape, dtype=self.probs.dtype, device=self.probs.device
            ).exponential_()
            q = torch.where(self.probs > 0, self.probs, 0.5)
            log_count = (exponential.log() - (-q.log()).log()) / self.concentration
            count = torch.where(self.probs > 0, log_count.exp().floor(), 0.0)
            return _require_sample_capacity(count).long()

    @staticmethod
    def _tail_bound_logs(probs, concentration, last):
        """Bound omitted count moments using continuous survival integrals.

        Parameters
        ----------
        probs : Tensor
            Active positive survival parameters.
        concentration : Tensor
            Active positive powers.
        last : int
            Largest count included in the partial moments.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Log bounds for omitted mass, first moment, and second moment.
        """
        with torch.no_grad():
            q, power = probs.detach(), concentration.detach()
            log_rate = (-q.log()).log()
            log_last = -math.inf if last == 0 else math.log(last)
            point = torch.exp(log_rate + power * log_last)
            orders = torch.stack((power.reciprocal(), 2 / power))
            _, log_tail = log_gamma_tails(orders, point.unsqueeze(0))
            integrals = (
                torch.lgamma(orders)
                - power.log().unsqueeze(0)
                - orders * log_rate.unsqueeze(0)
                + log_tail
            )
            mass = _negative_exp(log_rate + power * math.log(last + 1))
            first = torch.logaddexp(log_last + mass, integrals[0])
            second = torch.logaddexp(
                2 * log_last + mass,
                torch.logaddexp(math.log(2) + integrals[1], integrals[0]),
            )
            # Round the analytic upper bound outward relative to the numerical
            # incomplete-gamma evaluation tolerance.
            return mass, first + 1e-8, second + 1e-8

    def _moments(self, variance):
        """Accumulate adaptive count blocks with explicit omitted-moment bounds.

        Parameters
        ----------
        variance : bool
            Whether to certify the central second moment as well as the mean.

        Returns
        -------
        tuple[Tensor, Tensor]
            Mean and central variance lower approximations within tolerance.

        Raises
        ------
        NotImplementedError
            If the finite term budget or Gamma-tail evaluator cannot certify
            the requested relative moment tolerance.
        """
        flat_q, flat_power = self.probs.reshape(-1), self.concentration.reshape(-1)
        shape_tangent = torch.autograd.forward_ad.unpack_dual(flat_power).tangent
        geometric = (
            (flat_power == 1)
            if not flat_power.requires_grad and shape_tangent is None
            else torch.zeros_like(flat_q, dtype=torch.bool)
        )
        direct = (flat_q == 0) | geometric
        direct_mean = flat_q / (1 - flat_q)
        direct_variance = flat_q / (1 - flat_q).square()
        indices = torch.arange(flat_q.numel(), device=flat_q.device)
        output_indices, means, variances = (
            [indices[direct]],
            [direct_mean[direct]],
            [direct_variance[direct]],
        )
        active = ~direct
        indices, q, power = indices[active], flat_q[active], flat_power[active]
        mass, center, m2 = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
        log_tolerance = math.log(self.moment_rtol)
        # Bound temporary row-by-count tables as well as the count-term budget.
        chunk_size = max(1, min(256, 262144 // max(1, indices.numel())))
        for start in range(0, self.moment_max_terms, chunk_size):
            if indices.numel() == 0:
                break
            end = min(start + chunk_size, self.moment_max_terms)
            counts = torch.arange(start, end, dtype=q.dtype, device=q.device)
            weights = _log_mass(q[:, None], power[:, None], counts[None, :]).exp()
            block_mass = weights.sum(-1)
            safe_block = torch.where(block_mass > 0, block_mass, 1.0)
            block_center = (weights * counts).sum(-1) / safe_block
            block_m2 = (weights * (counts - block_center[:, None]).square()).sum(-1)
            total = mass + block_mass
            safe_total = torch.where(total > 0, total, 1.0)
            difference = block_center - center
            m2 = m2 + block_m2 + difference.square() * mass * block_mass / safe_total
            center = center + difference * block_mass / safe_total
            mass = total
            mean = mass * center
            try:
                tail_mass, tail_mean, tail_second = self._tail_bound_logs(
                    q, power, end - 1
                )
            except (ValueError, NotImplementedError) as error:
                raise NotImplementedError(
                    "DiscreteWeibull moments could not certify the Gamma integral remainder"
                ) from error
            certified = tail_mean <= log_tolerance + mean.detach().log()
            if variance:
                log_center = center.detach().abs().log()
                bound = torch.logaddexp(
                    tail_second,
                    torch.logaddexp(
                        math.log(2) + log_center + tail_mean, 2 * log_center + tail_mass
                    ),
                )
                certified = certified & (bound <= log_tolerance + m2.detach().log())
            output_indices.append(indices[certified])
            means.append(mean[certified])
            variances.append(m2[certified])
            remaining = ~certified
            indices, q, power = indices[remaining], q[remaining], power[remaining]
            mass, center, m2 = mass[remaining], center[remaining], m2[remaining]
        if indices.numel():
            raise NotImplementedError(
                f"DiscreteWeibull moments did not certify relative tolerance {self.moment_rtol:g} within {self.moment_max_terms} count terms"
            )
        order = torch.cat(output_indices)
        mean_result = torch.empty_like(flat_q).index_copy(0, order, torch.cat(means))
        variance_result = torch.empty_like(flat_q).index_copy(
            0, order, torch.cat(variances)
        )
        return mean_result.reshape(self.batch_shape), variance_result.reshape(
            self.batch_shape
        )

    @property
    def mean(self):
        """Compute the mean with an analytic omitted-tail error bound.

        Returns
        -------
        Tensor
            Float64 response means within the configured relative tolerance.
        """
        return self._moments(False)[0]

    @property
    def variance(self):
        """Compute variance without subtracting two large raw moments.

        Returns
        -------
        Tensor
            Float64 variances within the configured relative tolerance.
        """
        return self._moments(True)[1]


__all__ = ["DiscreteWeibull"]
