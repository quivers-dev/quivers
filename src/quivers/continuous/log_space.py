"""Distributions of the logit of a Beta variable and the log of a Gamma variable.

A Beta or Gamma site with a small shape parameter puts most of its mass
within a few multiples of the smallest floating-point number of zero (or
of one). Scored on the constrained scale, through a sigmoid or an
exponential and the Jacobian of that map, such a site loses its tails:
the sigmoid saturates long before the density of the logit does, and a
Gamma sampler that returns the smallest normal number in place of a
smaller draw no longer samples the declared prior.

The two families here live on the real line and carry the density of
the transformed variable directly.

`LogitBeta`
    The distribution of ``x = logit(p)`` for ``p ~ Beta(a, b)``, with
    density ``a * logsigmoid(x) + b * logsigmoid(-x) - log B(a, b)``.
`LogGamma`
    The distribution of ``y = log(g)`` for ``g ~ Gamma(a, r)``, with
    density ``a * (log r + y) - r * exp(y) - log Gamma(a)``.

Both densities are exact for every finite argument: no intermediate
quantity is formed on the constrained scale. Sampling stays in log space
through the shape-augmentation identity: if ``G' ~ Gamma(a + 1)`` and
``U ~ Uniform(0, 1)`` are independent, then ``G' * U^(1/a) ~ Gamma(a)``
([Marsaglia and Tsang 2000](https://doi.org/10.1145/358407.358414)), so ``log G' + log(U) / a`` is a log-Gamma draw
that never forms the Gamma variable itself. A logit-Beta draw is the
difference of two such draws, since ``G_a / (G_a + G_b) ~ Beta(a, b)``
for independent unit-rate Gamma variables. ``G'`` has shape at least
one, so it never approaches the underflow threshold, and its implicit
reparameterization gradient carries through, which makes both samplers
reparameterized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

type _Parameter = Tensor | float
"""A distribution parameter: a tensor, or a Python number broadcast as one."""


def _log_uniform(shape: torch.Size, like: Tensor) -> Tensor:
    """Draw ``log(U)`` for ``U ~ Uniform(0, 1]`` without forming zero.

    Parameters
    ----------
    shape : torch.Size
        The shape of the draw.
    like : Tensor
        A tensor whose dtype and device the draw takes.

    Returns
    -------
    Tensor
        Finite, nonpositive draws of ``log(U)``.
    """
    uniform = torch.rand(shape, dtype=like.dtype, device=like.device)
    return torch.log1p(-uniform)


def log_standard_gamma(concentration: Tensor) -> Tensor:
    """Draw ``log(G)`` for ``G ~ Gamma(concentration, 1)`` in log space.

    The draw is ``log G' + log(U) / concentration`` with ``G' ~
    Gamma(concentration + 1, 1)`` and ``U ~ Uniform(0, 1]``, which has
    the law of the log of a Gamma variable for every positive
    concentration and never forms that variable. Gradients reach the
    concentration through the implicit reparameterization of ``G'`` and
    through the division.

    Parameters
    ----------
    concentration : Tensor
        Positive shape parameters, one per draw.

    Returns
    -------
    Tensor
        One log-Gamma draw per entry of ``concentration``, finite for
        every concentration at or above 64 times the smallest normal
        number of its dtype.
    """
    boosted = torch._standard_gamma(concentration + 1.0)
    return torch.log(boosted) + _log_uniform(concentration.shape, concentration) / (
        concentration
    )


def _log_beta(first: Tensor, second: Tensor) -> Tensor:
    """The log of the Beta function, ``log B(first, second)``.

    Parameters
    ----------
    first : Tensor
        The first positive argument.
    second : Tensor
        The second positive argument.

    Returns
    -------
    Tensor
        ``lgamma(first) + lgamma(second) - lgamma(first + second)``.
    """
    return torch.lgamma(first) + torch.lgamma(second) - torch.lgamma(first + second)


class LogGamma(Distribution):
    """The log of a Gamma variable: ``y = log(g)`` for ``g ~ Gamma(a, r)``.

    The density on the real line is

    ``log p(y) = a * (log r + y) - r * exp(y) - lgamma(a)``,

    exact for every finite ``y`` and every positive shape. Draws are
    made in log space by shape augmentation (see
    [`log_standard_gamma`][quivers.continuous.log_space.log_standard_gamma]),
    so a shape of ``1e-8`` yields draws near ``-1e8`` rather than the
    clamped smallest normal number a constrained-scale sampler returns.
    TensorFlow Probability ships the same family as ``ExpGamma``.

    Parameters
    ----------
    concentration : Tensor or float
        The Gamma shape ``a``, positive.
    rate : Tensor or float
        The Gamma rate ``r``, positive.
    validate_args : bool, optional
        Whether to check parameter constraints and the support of scored
        values.
    """

    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "concentration": _constraints.positive,
        "rate": _constraints.positive,
    }
    support = _constraints.real  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    has_rsample = True

    def __init__(
        self,
        concentration: _Parameter,
        rate: _Parameter,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration, self.rate = broadcast_all(concentration, rate)
        super().__init__(self.concentration.shape, validate_args=validate_args)

    def expand(
        self, batch_shape: torch.Size, _instance: LogGamma | None = None
    ) -> LogGamma:
        """Broadcast the parameters to a larger batch shape.

        Parameters
        ----------
        batch_shape : torch.Size
            The batch shape to expand to.
        _instance : LogGamma, optional
            The instance torch's ``expand`` protocol fills.

        Returns
        -------
        LogGamma
            The expanded distribution.
        """
        new = self._get_checked_instance(LogGamma, _instance)
        shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(shape)
        new.rate = self.rate.expand(shape)
        Distribution.__init__(new, shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        """``digamma(a) - log(r)``."""
        return torch.digamma(self.concentration) - torch.log(self.rate)

    @property
    def mode(self) -> Tensor:
        """``log(a / r)``, where ``a * y - r * exp(y)`` peaks."""
        return torch.log(self.concentration) - torch.log(self.rate)

    @property
    def variance(self) -> Tensor:
        """``trigamma(a)``, independent of the rate."""
        return torch.polygamma(1, self.concentration)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw reparameterized samples in log space.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Draws of shape ``sample_shape + batch_shape``.
        """
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        return log_standard_gamma(concentration) - torch.log(self.rate.expand(shape))

    def log_prob(self, value: Tensor) -> Tensor:
        """The log density at ``value``.

        Parameters
        ----------
        value : Tensor
            Points on the real line.

        Returns
        -------
        Tensor
            ``a * (log r + y) - r * exp(y) - lgamma(a)``.
        """
        if self._validate_args:
            self._validate_sample(value)
        log_rate = torch.log(self.rate)
        return (
            self.concentration * (log_rate + value)
            - torch.exp(value + log_rate)
            - torch.lgamma(self.concentration)
        )

    def entropy(self) -> Tensor:
        """``lgamma(a) - a * digamma(a) + a``, independent of the rate.

        Returns
        -------
        Tensor
            The differential entropy, one per batch element.
        """
        a = self.concentration
        return torch.lgamma(a) - a * torch.digamma(a) + a + torch.zeros_like(self.rate)


class LogitBeta(Distribution):
    """The logit of a Beta variable: ``x = logit(p)`` for ``p ~ Beta(a, b)``.

    The density on the real line is

    ``log p(x) = a * logsigmoid(x) + b * logsigmoid(-x) - log B(a, b)``,

    exact for every finite ``x``: at ``a = 1e-3``, ``b = 1`` and ``x =
    -800`` it is about ``-7.71``, where scoring ``sigmoid(x)`` under a
    Beta density with the sigmoid's Jacobian saturates and returns about
    ``-99``. A draw is the difference of two log-Gamma draws made in log
    space (see
    [`log_standard_gamma`][quivers.continuous.log_space.log_standard_gamma]),
    so it is finite and reparameterized for shapes far below where a
    Beta sampler returns exactly zero or one.

    Parameters
    ----------
    concentration1 : Tensor or float
        The Beta shape ``a`` on ``p``, positive.
    concentration0 : Tensor or float
        The Beta shape ``b`` on ``1 - p``, positive.
    validate_args : bool, optional
        Whether to check parameter constraints and the support of scored
        values.
    """

    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "concentration1": _constraints.positive,
        "concentration0": _constraints.positive,
    }
    support = _constraints.real  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    has_rsample = True

    def __init__(
        self,
        concentration1: _Parameter,
        concentration0: _Parameter,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration1, self.concentration0 = broadcast_all(
            concentration1, concentration0
        )
        super().__init__(self.concentration1.shape, validate_args=validate_args)

    def expand(
        self, batch_shape: torch.Size, _instance: LogitBeta | None = None
    ) -> LogitBeta:
        """Broadcast the parameters to a larger batch shape.

        Parameters
        ----------
        batch_shape : torch.Size
            The batch shape to expand to.
        _instance : LogitBeta, optional
            The instance torch's ``expand`` protocol fills.

        Returns
        -------
        LogitBeta
            The expanded distribution.
        """
        new = self._get_checked_instance(LogitBeta, _instance)
        shape = torch.Size(batch_shape)
        new.concentration1 = self.concentration1.expand(shape)
        new.concentration0 = self.concentration0.expand(shape)
        Distribution.__init__(new, shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        """``digamma(a) - digamma(b)``."""
        return torch.digamma(self.concentration1) - torch.digamma(self.concentration0)

    @property
    def mode(self) -> Tensor:
        """``log(a / b)``, where the logit density peaks."""
        return torch.log(self.concentration1) - torch.log(self.concentration0)

    @property
    def variance(self) -> Tensor:
        """``trigamma(a) + trigamma(b)``."""
        return torch.polygamma(1, self.concentration1) + torch.polygamma(
            1, self.concentration0
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw reparameterized samples as a difference of log-Gamma draws.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Draws of shape ``sample_shape + batch_shape``.
        """
        shape = self._extended_shape(sample_shape)
        return log_standard_gamma(self.concentration1.expand(shape)) - (
            log_standard_gamma(self.concentration0.expand(shape))
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """The log density at ``value``.

        Parameters
        ----------
        value : Tensor
            Points on the real line.

        Returns
        -------
        Tensor
            ``a * logsigmoid(x) + b * logsigmoid(-x) - log B(a, b)``.
        """
        if self._validate_args:
            self._validate_sample(value)
        return (
            self.concentration1 * F.logsigmoid(value)
            + self.concentration0 * F.logsigmoid(-value)
            - _log_beta(self.concentration1, self.concentration0)
        )

    def entropy(self) -> Tensor:
        """``log B(a, b) - a (digamma(a) - digamma(a + b)) - b (digamma(b) - digamma(a + b))``.

        Returns
        -------
        Tensor
            The differential entropy, one per batch element.
        """
        a, b = self.concentration1, self.concentration0
        total = torch.digamma(a + b)
        return (
            _log_beta(a, b)
            - a * (torch.digamma(a) - total)
            - b * (torch.digamma(b) - total)
        )


__all__ = ["LogGamma", "LogitBeta", "log_standard_gamma"]
