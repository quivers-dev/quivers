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

Each product of a count and a log probability is computed with
`torch.xlogy` / `torch.special.xlog1py`, so ``0 log 0 = 0``, which is the
convention that makes the masses at the boundary probabilities the
point masses the families tend to there. A distribution built from ``logits``
is scored by torch's own logit path, which is exact for every finite
logit. A ``total_count`` of zero is admitted, the point mass at zero.

References
----------
Johnson, N. L., Kemp, A. W., and Kotz, S. (2005). *Univariate Discrete
Distributions*, 3rd ed., chapters 3 and 5. Wiley.
https://doi.org/10.1002/0471715816
"""

from __future__ import annotations

from typing import cast

import torch
import torch.distributions as D
from torch.distributions import constraints as _constraints


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
        return torch.xlogy(value, probs) + torch.special.xlog1py(1.0 - value, -probs)


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

    def __init__(
        self,
        total_count: torch.Tensor | int = 1,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(
            # The type checker reads torch's `total_count` as `int`; the
            # constructor broadcasts a tensor of trial counts as well.
            total_count=cast(int, total_count),
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
            The log masses, ``-inf`` above ``total_count``.

        Raises
        ------
        ValueError
            If argument validation is on and a value is not a
            nonnegative integer.
        """
        probs = self._closed_probs
        if probs is None:
            return super().log_prob(value)
        if self._validate_args:
            _require_counts(value)
        value, total, probs = torch.broadcast_tensors(
            value.to(probs.dtype), torch.as_tensor(self.total_count), probs
        )
        outside = _out_of_support(value, total)
        inside_value = torch.where(outside, torch.zeros_like(value), value)
        inside_total = torch.where(outside, torch.zeros_like(total), total)
        log_choose = (
            torch.lgamma(inside_total + 1.0)
            - torch.lgamma(inside_value + 1.0)
            - torch.lgamma(inside_total - inside_value + 1.0)
        )
        log_mass = (
            log_choose
            + torch.xlogy(inside_value, probs)
            + torch.special.xlog1py(inside_total - inside_value, -probs)
        )
        return torch.where(outside, torch.full_like(log_mass, -torch.inf), log_mass)


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
        probs = self._closed_probs
        if probs is None:
            return super().log_prob(value)
        if self._validate_args:
            self._validate_sample(value)
        value, total, probs = torch.broadcast_tensors(
            value.to(probs.dtype), torch.as_tensor(self.total_count), probs
        )
        negative = value < 0
        count = torch.where(negative, torch.zeros_like(value), value)
        # The normalizer Gamma(r + k) / (Gamma(k + 1) Gamma(r)) is one at
        # r = k = 0, where its log is the indeterminate inf - inf.
        degenerate = (total + count) == 0
        safe_total = torch.where(degenerate, torch.ones_like(total), total)
        log_normalizer = (
            torch.lgamma(safe_total + count)
            - torch.lgamma(count + 1.0)
            - torch.lgamma(safe_total)
        )
        log_normalizer = torch.where(
            degenerate, torch.zeros_like(log_normalizer), log_normalizer
        )
        log_mass = (
            log_normalizer
            + torch.special.xlog1py(total, -probs)
            + torch.xlogy(count, probs)
        )
        return torch.where(negative, torch.full_like(log_mass, -torch.inf), log_mass)


__all__ = ["ClosedBernoulli", "ClosedBinomial", "ClosedNegativeBinomial"]
