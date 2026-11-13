"""Stable moments and draws for a Poisson conditioned to be positive."""

from __future__ import annotations

import torch
from torch import Tensor


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
    u = torch.rand_like(rate)
    small = rate <= 8
    small_rate = torch.where(small, rate, torch.zeros_like(rate))
    small_remaining = torch.log1p((1 - u) * torch.expm1(small_rate))
    large_rate = torch.where(small, torch.full_like(rate, 8), rate)
    large_remaining = large_rate + torch.log1p(u * torch.expm1(-large_rate))
    remaining = torch.where(small, small_remaining, large_remaining)
    return 1 + torch.poisson(remaining)


__all__: list[str] = []
