"""Bernoulli-trial families scored on the closed unit interval.

`torch.distributions.Bernoulli`, `torch.distributions.Binomial` and
`torch.distributions.NegativeBinomial` given ``probs`` score through
``probs_to_logits``, which clamps the probability into
``[eps, 1 - eps]`` before taking its logit. A success probability of
exactly zero or one (zero only, for the negative binomial, whose
``probs`` lies in ``[0, 1)``) is then scored as if it were ``eps`` away: a success under ``probs = 0`` gets a log mass of
about ``-15.9`` in single precision rather than ``-inf``. The classes
here keep torch's construction, sampling and moments, and score a
distribution built from ``probs`` from the probability itself:

* Bernoulli: ``x log p + (1 - x) log(1 - p)``;
* Binomial: ``log C(n, k) + k log p + (n - k) log(1 - p)``;
* NegativeBinomial (torch's convention, ``k`` successes before the
  ``r``-th failure): ``log Gamma(r + k) - log Gamma(k + 1)
  - log Gamma(r) + r log(1 - p) + k log p``.

Each product of a count and a log probability is taken as zero where
the count is zero, with a zero gradient there, so ``0 log 0 = 0``, the
convention that makes the masses at the boundary probabilities the
point masses the families tend to there. Finite logits are scored directly through their log probabilities. A ``total_count`` of zero is admitted, the point mass at zero.
Binomial, negative-binomial, and Poisson parameters and count arithmetic
use float64 and stable deviance/Stirling corrections: subtracting large
log-gamma terms can otherwise give a positive log probability. Finite
binomial trial counts are restricted to exactly representable integers
through 2**53.

References
----------
Johnson, N. L., Kemp, A. W., and Kotz, S. (2005). *Univariate Discrete
Distributions*, 3rd ed., chapters 3 and 5. Wiley.
https://doi.org/10.1002/0471715816
"""

from __future__ import annotations

from typing import cast
from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor
import torch.distributions as D
from torch.distributions import constraints as _constraints
from torch.distributions.utils import lazy_property


class _ValidatedCategorical(D.Categorical):
    """Validate raw category weights before torch normalizes their signs away.

    Parameters
    ----------
    probs : torch.Tensor or None
        Finite nonnegative category weights, with a positive total per row.
    logits : torch.Tensor or None
        Alternative unnormalized log weights. Supply only one representation.
    validate_args : bool or None
        Torch sample and normalized-parameter validation policy. Raw probability
        validation is mandatory under either policy.
    """

    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            probs = torch.as_tensor(probs)
            if probs.ndim < 1 or probs.shape[-1] == 0:
                raise ValueError("Categorical probs require a nonempty category axis")
            if not bool((torch.isfinite(probs) & (probs >= 0)).all()):
                raise ValueError("Categorical probs must be finite and nonnegative")
            largest = probs.amax(-1, keepdim=True)
            if not bool((largest > 0).all()):
                raise ValueError(
                    "Categorical probs must have a positive total in every row"
                )
            # Rescaling also keeps sums of valid finite weights from overflowing.
            probs = probs / largest
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    @lazy_property
    def logits(self):
        """Preserve exact zero category probabilities in log-space scoring.

        Returns
        -------
        torch.Tensor
            Normalized log weights, negative infinity for zero categories.
        """
        positive = self.probs > 0
        return torch.where(
            positive, torch.where(positive, self.probs, 1.0).log(), -torch.inf
        )

    def expand(self, batch_shape, _instance=None):
        """Expand category weights using torch's view-preserving implementation.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Desired batch shape.
        _instance : _ValidatedCategorical or None
            Optional torch expansion instance.

        Returns
        -------
        _ValidatedCategorical
            Expanded categorical distribution.
        """
        new = self._get_checked_instance(_ValidatedCategorical, _instance)
        return super().expand(batch_shape, new)


_MAX_EXACT_TRIALS = 2**53


def _check_trial_count_source(total_count) -> None:
    """Reject integer trial inputs that a float64 count kernel cannot preserve.

    Parameters
    ----------
    total_count : torch.Tensor or int or float
        Source trial count, sequence, or array before a floating-point conversion.

    Raises
    ------
    ValueError
        If any real trial count exceeds 2**53.
    TypeError
        If trials use unsigned64 instead of the canonical signed integer dtype.
    """
    message = (
        "native count trials must be <= 2**53 for exact float64 integer representation"
    )
    if isinstance(total_count, Real):
        if total_count > _MAX_EXACT_TRIALS:
            raise ValueError(message)
        return
    if isinstance(total_count, torch.Tensor):
        if total_count.dtype == getattr(torch, "uint64", None):
            raise TypeError("native trial tensors use signed64 integer counts")
        if (
            total_count.is_floating_point() or total_count.dtype == torch.int64
        ) and bool((total_count > _MAX_EXACT_TRIALS).any()):
            raise ValueError(message)
        return
    if isinstance(total_count, Sequence) and not isinstance(total_count, str | bytes):
        # Inspect every scalar before mixed Python sequences can infer a
        # floating dtype and round an otherwise exact large integer.
        for item in total_count:
            _check_trial_count_source(item)
        return
    if hasattr(total_count, "dtype") and hasattr(total_count, "shape"):
        # Array providers retain their existing storage dtype. No NumPy
        # dependency or intermediate default-floating conversion is needed.
        _check_trial_count_source(torch.as_tensor(total_count))


def _require_sample_capacity(value: torch.Tensor) -> torch.Tensor:
    """Refuse sampler values beyond exact float64 integer capacity.

    Parameters
    ----------
    value : torch.Tensor
        Generated intensities or count samples.

    Returns
    -------
    torch.Tensor
        The unchanged representable values.

    Raises
    ------
    RuntimeError
        If a value is negative, nonfinite, or exceeds 2**53. This is an
        operational failure, not truncation or conditioning of the law.
    """
    if not bool(
        (torch.isfinite(value) & (value >= 0) & (value <= _MAX_EXACT_TRIALS)).all()
    ):
        raise RuntimeError(
            "count sampling exceeds the exact float64 integer capacity 2**53"
        )
    return value


def _sample_poisson_exact(rate: torch.Tensor) -> torch.Tensor:
    """Draw Poisson counts after checking intensity and output capacity.

    Parameters
    ----------
    rate : torch.Tensor
        Nonnegative Poisson intensities.

    Returns
    -------
    torch.Tensor
        Float64 integer-valued samples on the input device.

    Raises
    ------
    RuntimeError
        If an intensity or draw exceeds the sampler's exact integer capacity.
    """
    rate = _require_sample_capacity(rate.to(torch.float64))
    return _require_sample_capacity(torch.poisson(rate))


def _stirling_error(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the log-factorial Stirling correction without large subtraction.

    Parameters
    ----------
    x : torch.Tensor
        Strictly positive real counts.

    Returns
    -------
    torch.Tensor
        ``lgamma(x+1) - (x+.5)*log(x) + x - log(2*pi)/2``.
    """
    large = x > 15
    small_x = torch.where(large, torch.ones_like(x), x)
    exact = (
        torch.lgamma(small_x + 1)
        - (small_x + 0.5) * small_x.log()
        + small_x
        - 0.91893853320467274178
    )
    inverse = 1 / torch.where(large, x, torch.full_like(x, 16))
    square = inverse.square()
    series = inverse * (
        1 / 12
        - square * (1 / 360 - square * (1 / 1260 - square * (1 / 1680 - square / 1188)))
    )
    return torch.where(large, series, exact)


def _deviance(x: torch.Tensor, log_expected: torch.Tensor) -> torch.Tensor:
    """Evaluate Poisson deviance with a convergent series near its minimum.

    Parameters
    ----------
    x : torch.Tensor
        Strictly positive observed counts.
    log_expected : torch.Tensor
        Logarithm of the positive expected counts.

    Returns
    -------
    torch.Tensor
        ``x*log(x/expected) + expected - x`` with stable central values.
    """
    expected = log_expected.exp()
    difference = x - expected
    near = difference.abs() < 0.1 * (x + expected)
    direct_x = torch.where(near, torch.ones_like(x), x)
    direct_log = torch.where(near, torch.zeros_like(log_expected), log_expected)
    direct = direct_x * (direct_x.log() - direct_log) + direct_log.exp() - direct_x
    v = torch.where(near, difference / (x + expected), torch.zeros_like(x))
    square = v.square()
    polynomial = 1 / 21
    for denominator in range(19, 2, -2):
        polynomial = 1 / denominator + square * polynomial
    series = difference * v + 2 * x * v * square * polynomial
    return torch.where(near, series, direct)


def _binomial_log_mass(
    k: torch.Tensor, n: torch.Tensor, log_p: torch.Tensor, log_q: torch.Tensor
) -> torch.Tensor:
    """Score a binomial without subtracting extensive log-gamma terms.

    Parameters
    ----------
    k : torch.Tensor
        Counts in the trial support.
    n : torch.Tensor
        Nonnegative trial counts.
    log_p : torch.Tensor
        Finite log success probabilities.
    log_q : torch.Tensor
        Finite log failure probabilities.

    Returns
    -------
    torch.Tensor
        Stable double-precision log masses.
    """
    interior = (k > 0) & (k < n)
    a = torch.where(interior, k, torch.ones_like(k))
    b = torch.where(interior, n - k, torch.ones_like(k))
    total = a + b
    result = (
        _stirling_error(total)
        - _stirling_error(a)
        - _stirling_error(b)
        - _deviance(a, total.log() + log_p)
        - _deviance(b, total.log() + log_q)
        - 0.5 * (1.83787706640934548356 + a.log() + b.log() - total.log())
    )
    boundary = torch.where(k == 0, n * log_q, n * log_p)
    return torch.where(interior, result, boundary)


def _negative_binomial_log_mass(
    k: torch.Tensor, r: torch.Tensor, log_p: torch.Tensor, log_q: torch.Tensor
) -> torch.Tensor:
    """Score positive negative-binomial counts through stable deviance terms.

    Parameters
    ----------
    k : torch.Tensor
        Strictly positive observed counts.
    r : torch.Tensor
        Strictly positive shape parameters.
    log_p : torch.Tensor
        Finite log success probabilities.
    log_q : torch.Tensor
        Finite log failure probabilities.

    Returns
    -------
    torch.Tensor
        Negative-binomial log masses without extensive gamma subtraction.
    """
    log_k, log_r = k.log(), r.log()
    log_total = torch.logaddexp(log_k, log_r)
    log_ratio = torch.logaddexp(torch.zeros_like(log_k), log_k - log_r)
    return (
        _stirling_error(k + r)
        - _stirling_error(k)
        - _stirling_error(r)
        - _deviance(k, log_total + log_p)
        - _deviance(r, log_total + log_q)
        - 0.5 * (1.83787706640934548356 + log_k + log_ratio)
    )


def _log_gamma_increment(x: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
    """Evaluate ``lgamma(x+increment)-lgamma(x)`` without large cancellation.

    Parameters
    ----------
    x : torch.Tensor
        Positive gamma argument.
    increment : torch.Tensor
        Shift with ``x+increment > 0``.

    Returns
    -------
    torch.Tensor
        Stable difference, including gradients of the shift.
    """
    large = (x > 16) & (x + increment > 16)
    direct_x = torch.where(large, torch.ones_like(x), x)
    direct_shift = torch.where(large, torch.zeros_like(increment), increment)
    direct = torch.lgamma(direct_x + direct_shift) - torch.lgamma(direct_x)
    large_x = torch.where(large, x, torch.full_like(x, 17))
    shift = torch.where(large, increment, torch.zeros_like(increment))
    asymptotic = (
        shift * large_x.log()
        + (large_x + shift - 0.5) * torch.log1p(shift / large_x)
        - shift
        + _stirling_error(large_x + shift)
        - _stirling_error(large_x)
    )
    return torch.where(large, asymptotic, direct)


def _beta_binomial_log_mass(k: Tensor, n: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Score beta-binomial factors without subtracting extensive gamma values.

    Parameters
    ----------
    k, n : Tensor
        Valid observed counts and trial totals.
    a, b : Tensor
        Positive beta concentrations.

    Returns
    -------
    Tensor
        Float64 log masses with stable count and concentration limits.
    """
    active = n > 0
    a = torch.where(active, a, 1.0)
    b = torch.where(active, b, 1.0)
    interior = (k > 0) & (k < n)
    x = torch.where(interior, k, 1.0)
    y = torch.where(interior, n - k, 1.0)
    count = x + y
    prior = a + b
    posterior = count + prior
    xp, yp = x + a, y + b
    log_p = torch.logaddexp(x.log(), a.log()) - posterior.log()
    log_q = torch.logaddexp(y.log(), b.log()) - posterior.log()
    result = (
        _stirling_error(count)
        - _stirling_error(x)
        - _stirling_error(y)
        + _stirling_error(xp)
        + _stirling_error(yp)
        - _stirling_error(posterior)
        - _stirling_error(a)
        - _stirling_error(b)
        + _stirling_error(prior)
        - _deviance(x, count.log() + log_p)
        - _deviance(y, count.log() + log_q)
        - _deviance(a, prior.log() + log_p)
        - _deviance(b, prior.log() + log_q)
        + 0.5
        * (
            count.log()
            - x.log()
            - y.log()
            - xp.log()
            - yp.log()
            + posterior.log()
            + a.log()
            + b.log()
            - prior.log()
            - 1.83787706640934548356
        )
    )
    zero = torch.where(
        a < n,
        _log_gamma_increment(b, a) - _log_gamma_increment(b + n, a),
        _log_gamma_increment(b, n) - _log_gamma_increment(b + a, n),
    )
    full = torch.where(
        b < n,
        _log_gamma_increment(a, b) - _log_gamma_increment(a + n, b),
        _log_gamma_increment(a, n) - _log_gamma_increment(a + b, n),
    )
    return torch.where(
        n == 0, 0.0, torch.where(interior, result, torch.where(k == 0, zero, full))
    )


def _out_of_support(value: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    """Where a count lies outside ``0 .. total``.

    Parameters
    ----------
    value : torch.Tensor
        The counts.
    total : torch.Tensor
        The trial counts.

    Returns
    -------
    torch.Tensor
        A Boolean mask.
    """
    return (value < 0) | (value > total)


def _count_log(count: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """``count * log(probs)``: zero where the count is, ``-inf`` where a
    positive count meets a zero probability.

    Both boundary cases are constants, so the logarithm is taken of a
    probability replaced by one there, and the gradient through them is
    zero rather than the ``0 * inf`` of differentiating the logarithm
    at zero (which a reduction that weighs an impossible branch by zero
    would otherwise turn into ``nan``).

    Parameters
    ----------
    count : torch.Tensor
        Nonnegative counts.
    probs : torch.Tensor
        Nonnegative values, broadcast against ``count``.

    Returns
    -------
    torch.Tensor
        The products.
    """
    count, probs = torch.broadcast_tensors(count, probs)
    absent = count == 0
    impossible = ~absent & (probs == 0)
    safe = torch.where(absent | impossible, torch.ones_like(probs), probs)
    product = count * torch.log(safe)
    product = torch.where(impossible, torch.full_like(product, -torch.inf), product)
    return torch.where(absent, torch.zeros_like(product), product)


def _count_log_complement(count: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """``count * log(1 - probs)``, with the boundary cases of `_count_log`.

    Parameters
    ----------
    count : torch.Tensor
        Nonnegative counts.
    probs : torch.Tensor
        Probabilities, broadcast against ``count``.

    Returns
    -------
    torch.Tensor
        The products.
    """
    count, probs = torch.broadcast_tensors(count, probs)
    return _count_log(count, 1.0 - probs)


def _require_counts(value: torch.Tensor) -> None:
    """Check that every value is a nonnegative integer.

    A count above the number of trials is a point of the sample space
    the binomial puts no mass on, scored ``-inf``; only a value that is
    no count at all is an error.

    Parameters
    ----------
    value : torch.Tensor
        The values.

    Raises
    ------
    ValueError
        If a value is negative or not an integer.
    """
    if not bool(_constraints.nonnegative_integer.check(value).all()):
        raise ValueError(
            f"a binomial count must be a nonnegative integer, got {value.tolist()}"
        )


class ClosedBernoulli(D.Bernoulli):
    """A Bernoulli that scores ``probs`` of zero and one where they stand.

    Parameters
    ----------
    probs : torch.Tensor, optional
        The success probability, in ``[0, 1]``.
    logits : torch.Tensor, optional
        The log-odds of a success.
    validate_args : bool, optional
        Whether to validate arguments and samples.
    """

    def __init__(
        self,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        self._closed_probs = probs

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """The log mass at an outcome.

        Parameters
        ----------
        value : torch.Tensor
            The outcomes, zero or one.

        Returns
        -------
        torch.Tensor
            The log masses: ``-inf`` for a success under ``probs = 0``
            or a failure under ``probs = 1``.
        """
        probs = self._closed_probs
        if probs is None:
            return super().log_prob(value)
        if self._validate_args:
            self._validate_sample(value)
        value, probs = torch.broadcast_tensors(value.to(probs.dtype), probs)
        return _count_log(value, probs) + _count_log_complement(1.0 - value, probs)


class ClosedBinomial(D.Binomial):
    """A binomial that scores ``probs`` of zero and one, and zero trials.

    Parameters
    ----------
    total_count : torch.Tensor or int
        The number of trials, a nonnegative integer.
    probs : torch.Tensor, optional
        The success probability, in ``[0, 1]``.
    logits : torch.Tensor, optional
        The log-odds of a success.
    validate_args : bool, optional
        Whether to validate arguments and samples.
    """

    arg_constraints = dict(D.Binomial.arg_constraints) | {
        "total_count": _constraints.nonnegative_integer
    }

    def __init__(
        self,
        total_count: torch.Tensor | int = 1,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        _check_trial_count_source(total_count)
        probs = None if probs is None else probs.to(torch.float64)
        logits = None if logits is None else logits.to(torch.float64)
        parameter = probs if probs is not None else logits
        total_count = torch.as_tensor(
            total_count,
            dtype=torch.float64,
            device=None if parameter is None else parameter.device,
        )
        super().__init__(
            total_count=cast(int, total_count),
            probs=probs,
            logits=logits,
            validate_args=validate_args,
        )
        self._closed_probs = probs

    def sample(
        self, sample_shape: torch.Size | Sequence[int] = torch.Size()
    ) -> torch.Tensor:
        """Draw finite-trial counts without rounding an out-of-range result.

        Parameters
        ----------
        sample_shape : torch.Size
            Sample dimensions prepended to the distribution batch.

        Returns
        -------
        torch.Tensor
            Exactly representable Float64 counts in the trial support.

        Raises
        ------
        RuntimeError
            If the underlying sampler returns an unrepresentable count or
            a value above the supplied trials.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            n = self.total_count.expand(shape)
            value = _require_sample_capacity(
                torch.binomial(n, self.probs.expand(shape))
            )
            if not bool(((value <= n) & (value == value.floor())).all()):
                raise RuntimeError(
                    "binomial sampler returned a noninteger count or a count above remaining trials"
                )
            return value

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """The log mass at a count of successes.

        Parameters
        ----------
        value : torch.Tensor
            The counts.

        Returns
        -------
        torch.Tensor
            The log masses, ``-inf`` above ``total_count``.

        Raises
        ------
        ValueError
            If argument validation is on and a value is not a
            nonnegative integer.
        """
        probs = self._closed_probs
        if self._validate_args:
            _require_counts(value)
        value, total = torch.broadcast_tensors(
            value.to(torch.float64), self.total_count
        )
        outside = _out_of_support(value, total)
        k = torch.where(outside, torch.zeros_like(value), value)
        if probs is None:
            log_p = torch.nn.functional.logsigmoid(self.logits)
            log_q = torch.nn.functional.logsigmoid(-self.logits)
            result = _binomial_log_mass(k, total, log_p, log_q)
        else:
            interior_p = (probs > 0) & (probs < 1)
            safe = torch.where(interior_p, probs, torch.full_like(probs, 0.5))
            stable = _binomial_log_mass(k, total, safe.log(), torch.log1p(-safe))
            endpoint = _count_log(k, probs) + _count_log_complement(total - k, probs)
            result = torch.where(interior_p, stable, endpoint)
        return torch.where(outside, torch.full_like(result, -torch.inf), result)

    def expand(self, batch_shape, _instance=None):
        """Retain probability-endpoint state through batch expansion.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested broadcast batch shape.
        _instance : ClosedBinomial or None
            Optional PyTorch instance.

        Returns
        -------
        ClosedBinomial
            Expanded distribution with float64 count arithmetic.
        """
        new = self._get_checked_instance(type(self), _instance)
        super().expand(batch_shape, new)
        new._closed_probs = (
            None
            if self._closed_probs is None
            else self._closed_probs.expand(batch_shape)
        )
        return new


class ClosedNegativeBinomial(D.NegativeBinomial):
    """A negative binomial that scores ``probs = 0`` and ``r = 0``.

    Follows torch's convention: the count is of successes, each with
    probability ``probs`` in ``[0, 1)``, before the ``total_count``-th
    failure. A ``total_count`` of zero and a ``probs`` of zero are each
    the point mass at zero. A ``probs`` of one is no distribution (the
    count is almost surely infinite) and stays outside the family.

    Parameters
    ----------
    total_count : torch.Tensor or float
        The number of failures waited for, nonnegative.
    probs : torch.Tensor, optional
        The success probability, in ``[0, 1)``.
    logits : torch.Tensor, optional
        The log-odds of a success.
    validate_args : bool, optional
        Whether to validate arguments and samples.
    """

    def __init__(
        self,
        total_count: torch.Tensor | float,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        probs = None if probs is None else probs.to(torch.float64)
        logits = None if logits is None else logits.to(torch.float64)
        parameter = probs if probs is not None else logits
        total_count = torch.as_tensor(
            total_count,
            dtype=torch.float64,
            device=None if parameter is None else parameter.device,
        )
        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=validate_args,
        )
        self._closed_probs = probs

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """The log mass at a count of successes.

        Parameters
        ----------
        value : torch.Tensor
            The counts.

        Returns
        -------
        torch.Tensor
            The log masses, ``-inf`` below zero.
        """
        if self._validate_args:
            self._validate_sample(value)
        value, total = torch.broadcast_tensors(
            value.to(torch.float64), self.total_count
        )
        positive = (value > 0) & (total > 0)
        k = torch.where(positive, value, torch.ones_like(value))
        r = torch.where(positive, total, torch.ones_like(total))
        probs = self._closed_probs
        if probs is None:
            log_p = torch.nn.functional.logsigmoid(self.logits)
            log_q = torch.nn.functional.logsigmoid(-self.logits)
        else:
            interior = probs > 0
            safe_p = torch.where(interior, probs, torch.full_like(probs, 0.5))
            log_p, log_q = safe_p.log(), torch.log1p(-safe_p)
        stable = _negative_binomial_log_mass(k, r, log_p, log_q)
        if probs is not None:
            stable = torch.where(probs > 0, stable, -torch.inf)
            zero = _count_log_complement(total, probs)
        else:
            zero = total * log_q
        result = torch.where(
            value == 0, zero, torch.where(positive, stable, -torch.inf)
        )
        return result

    @property
    def mean(self) -> torch.Tensor:
        """Expected successes with the exact p=0 endpoint.

        Returns
        -------
        torch.Tensor
            Float64 negative-binomial mean.
        """
        if self._closed_probs is None:
            return super().mean
        return self.total_count * self._closed_probs / (1 - self._closed_probs)

    @property
    def variance(self) -> torch.Tensor:
        """Response variance with the exact p=0 endpoint.

        Returns
        -------
        torch.Tensor
            Float64 negative-binomial variance.
        """
        if self._closed_probs is None:
            return super().variance
        return self.mean / (1 - self._closed_probs)

    def expand(self, batch_shape, _instance=None):
        """Retain exact probability endpoints through expansion.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested broadcast batch shape.
        _instance : ClosedNegativeBinomial or None
            Optional PyTorch instance.

        Returns
        -------
        ClosedNegativeBinomial
            Expanded distribution retaining its closed probability parameter.
        """
        new = self._get_checked_instance(type(self), _instance)
        super().expand(batch_shape, new)
        new._closed_probs = (
            None
            if self._closed_probs is None
            else self._closed_probs.expand(batch_shape)
        )
        return new

    def sample(self, sample_shape=torch.Size()):
        """Draw gamma-Poisson counts using true closed probabilities.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Sample dimensions prepended to the batch.

        Returns
        -------
        torch.Tensor
            Float64 counts, exactly zero at zero shape or probability.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            total = self.total_count.expand(shape)
            if self._closed_probs is None:
                active = total > 0
                gamma_rate = torch.ones_like(total)
                gamma_scale = self.logits.exp().expand(shape)
            else:
                probability = self._closed_probs.expand(shape)
                active = (total > 0) & (probability > 0)
                safe_probability = torch.where(active, probability, 0.5)
                gamma_rate = (1 - safe_probability) / safe_probability
                gamma_scale = torch.ones_like(total)
            safe_total = torch.where(active, total, 1.0)
            safe_rate = torch.where(active, gamma_rate, 1.0)
            intensity = D.Gamma(safe_total, safe_rate).sample() * gamma_scale
            intensity = torch.where(active, intensity, 0.0)
            return _sample_poisson_exact(intensity)


class ClosedPoisson(D.Poisson):
    """A Poisson that scores a rate of zero as the point mass at zero.

    ``log p(k) = k log(rate) - rate - log k!``, with the count's
    logarithm term taken by `_count_log`, so a positive count under a
    zero rate is ``-inf`` with a zero gradient, and a zero count under
    it is ``0``.

    Parameters
    ----------
    rate : torch.Tensor
        The rate, nonnegative.
    validate_args : bool, optional
        Whether to validate arguments and samples.
    """

    def __init__(
        self, rate: torch.Tensor | float, validate_args: bool | None = None
    ) -> None:
        super().__init__(
            torch.as_tensor(rate, dtype=torch.float64), validate_args=validate_args
        )

    def sample(self, sample_shape=torch.Size()):
        """Draw counts within the sampler's exact integer representation.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Float64 integer-valued counts.

        Raises
        ------
        RuntimeError
            If an intensity or sample exceeds 2**53.
        """
        with torch.no_grad():
            return _sample_poisson_exact(
                self.rate.expand(self._extended_shape(sample_shape))
            )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """The log mass at a count.

        Parameters
        ----------
        value : torch.Tensor
            The counts.

        Returns
        -------
        torch.Tensor
            The log masses.
        """
        if self._validate_args:
            self._validate_sample(value)
        value, rate = torch.broadcast_tensors(value.to(self.rate.dtype), self.rate)
        positive = value > 0
        k = torch.where(positive, value, torch.ones_like(value))
        safe_rate = torch.where(rate > 0, rate, torch.ones_like(rate))
        stable = (
            -_stirling_error(k)
            - 0.5 * (1.83787706640934548356 + k.log())
            - _deviance(k, safe_rate.log())
        )
        stable = torch.where(rate > 0, stable, -torch.inf)
        return torch.where(positive, stable, -rate)


__all__ = [
    "ClosedBernoulli",
    "ClosedBinomial",
    "ClosedNegativeBinomial",
    "ClosedPoisson",
]
