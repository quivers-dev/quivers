"""Stable moments and draws for a Poisson conditioned to be positive."""

from __future__ import annotations

import torch
from torch import Tensor

from quivers.continuous._closed_probability import (
    _sample_poisson_exact,
    _require_sample_capacity,
)


def positive_poisson_mean(rate: Tensor) -> Tensor:
    """Evaluate rate / (1 - exp(-rate)), including its limit at zero.

    Parameters
    ----------
    rate : Tensor
        Nonnegative Poisson rates.

    Returns
    -------
    Tensor
        The conditional mean. Its derivative at zero is one half.
    """
    small = rate < 0.01
    x = torch.where(small, rate, torch.zeros_like(rate))
    square = x.square()
    # The next Bernoulli-series term is at most 1e-16 / 1209600.
    series = 1 + x / 2 + square * (1 / 12 + square * (-1 / 720 + square / 30240))
    regular = torch.where(small, torch.ones_like(rate), rate)
    ratio = regular / -torch.expm1(-regular)
    return torch.where(small, series, ratio)


def positive_poisson_sample(rate: Tensor) -> Tensor:
    """Draw a positive Poisson count, with the point-mass limit at zero.

    Condition the first arrival in a unit-rate Poisson process to occur
    before ``rate``. Its remaining interval has length
    ``log(1 + (1-u) * expm1(rate))``. One plus a Poisson draw on that
    interval has the required conditional distribution.

    Parameters
    ----------
    rate : Tensor
        Nonnegative rates, already expanded to the requested sample shape.

    Returns
    -------
    Tensor
        Positive integer-valued samples in the input floating-point dtype.

    References
    ----------
    Harding, T. (2005). Simulate zero-truncated Poisson distribution.
    https://stat.ethz.ch/pipermail/r-help/2005-May/070678.html
    """
    rate = _require_sample_capacity(rate.to(torch.float64))
    u = torch.rand_like(rate)
    small = rate <= 8
    small_rate = torch.where(small, rate, torch.zeros_like(rate))
    small_remaining = torch.log1p((1 - u) * torch.expm1(small_rate))
    large_rate = torch.where(small, torch.full_like(rate, 8), rate)
    large_remaining = large_rate + torch.log1p(u * torch.expm1(-large_rate))
    remaining = torch.where(small, small_remaining, large_remaining)
    return _require_sample_capacity(1 + _sample_poisson_exact(remaining))


__all__: list[str] = []


def positive_poisson_log_mean(rate: Tensor) -> Tensor:
    """Retain log(rate/P(Poisson(rate)>0)) before a near-one ratio rounds.

    Parameters
    ----------
    rate : Tensor
        Nonnegative Poisson rates, including zero.

    Returns
    -------
    Tensor
        Smooth logarithmic conditional mean. The small-rate Bernoulli series
        preserves perturbations smaller than the spacing around one.
    """
    small = rate < 0.01
    x = torch.where(small, rate, 0.0)
    square = x.square()
    series = x / 2 + square * (-1 / 24 + square * (1 / 2880 - square / 181440))
    regular = torch.where(small, 1.0, rate)
    value = regular.log() - (-torch.expm1(-regular)).log()
    return torch.where(small, series, value)
