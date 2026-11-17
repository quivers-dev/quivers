"""Exponentially modified Gaussian and inverse Gaussian response distributions."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from quivers.continuous._gaussian_tails import (
    exgaussian_log_component,
    log_erfcx_positive,
    log_scaled_erfcx_difference,
    paired_log_tails,
)


class ExGaussian(Distribution):
    """Sum of independent Normal and Exponential random variables.

    Parameters
    ----------
    loc : Tensor or float
        Mean of the Normal component, distinct from the response mean.
    scale : Tensor or float
        Strictly positive standard deviation of the Normal component.
    rate : Tensor or float
        Strictly positive rate of the Exponential component.
    validate_args : bool or None
        Whether to validate parameters and evaluation points.

    Notes
    -----
    The response mean is ``loc + 1/rate``. The brms formula interface
    predicts that mean and calls the exponential mean ``beta``.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale, self.rate = broadcast_all(loc, scale, rate)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExGaussian, _instance)
        shape = torch.Size(batch_shape)
        new.loc, new.scale, new.rate = (
            parameter.expand(shape) for parameter in (self.loc, self.scale, self.rate)
        )
        super(ExGaussian, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.loc + self.rate.reciprocal()

    @property
    def variance(self) -> Tensor:
        return self.scale.square() + self.rate.reciprocal().square()

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        standardized = (value - self.loc) / self.scale
        return self.rate.log() + exgaussian_log_component(
            standardized, self.rate * self.scale
        )

    def _log_tails(self, value: Tensor) -> tuple[Tensor, Tensor]:
        if bool(torch.isnan(value).any()):
            raise ValueError("ExGaussian tails require non-NaN evaluation points")
        finite = torch.isfinite(value)
        loc, scale, rate = self.loc.double(), self.scale.double(), self.rate.double()
        point = torch.where(finite, value.double(), loc)
        z = (point - loc) / scale
        rate_scale = rate * scale
        lower = log_scaled_erfcx_difference(
            -z / math.sqrt(2), rate_scale / math.sqrt(2)
        ) - math.log(2)
        upper = torch.logsumexp(
            torch.stack(
                (torch.special.log_ndtr(-z), exgaussian_log_component(z, rate_scale))
            ),
            dim=0,
        )
        lower, upper = paired_log_tails(lower, upper)
        return (
            torch.where(finite, lower, torch.where(value > 0, 0.0, -torch.inf)),
            torch.where(finite, upper, torch.where(value > 0, -torch.inf, 0.0)),
        )

    def log_cdf(self, value: Tensor) -> Tensor:
        """Return log P(Y <= value), retaining small Gaussian-exponential tails.

        Parameters
        ----------
        value : Tensor
            Extended-real evaluation points, broadcast with the batch.

        Returns
        -------
        Tensor
            Log cumulative probabilities. Computed in Float64 to retain
            probabilities that ordinary CDF evaluation rounds away.
        """
        return self._log_tails(value)[0]

    def log_survival_function(self, value: Tensor) -> Tensor:
        """Return log P(Y > value), without subtracting a rounded CDF.

        Parameters
        ----------
        value : Tensor
            Extended-real evaluation points, broadcast with the batch.

        Returns
        -------
        Tensor
            Log strict upper-tail probabilities. Computed in Float64 to retain
            probabilities that ordinary CDF evaluation rounds away.
        """
        return self._log_tails(value)[1]

    def log_density_variation(self, lower: Tensor, upper: Tensor) -> Tensor:
        """Bound the absolute log-density change across a finite interval.

        Parameters
        ----------
        lower, upper : Tensor
            Finite interval endpoints, broadcast against the parameters.

        Returns
        -------
        Tensor
            A nonnegative Float64 variation bound, or infinity outside a
            finite positive-width interval. The inverse Mills ratio obeys
            m(t) <= max(-t, 0) + 1, yielding a conservative score bound.
        """
        lower, upper = lower.double(), upper.double()
        valid = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
        start = torch.where(valid, lower, self.loc.double())
        width = torch.where(valid, upper - lower, 1.0)
        rate, scale = self.rate.double(), self.scale.double()
        t = (start - self.loc.double()) / scale - rate * scale
        bound = width * (rate + ((-t).clamp_min(0) + 1) / scale)
        return torch.where(valid, bound, torch.inf)

    def cdf(self, value: Tensor) -> Tensor:
        return self.log_cdf(value).exp().to(dtype=torch.result_type(value, self.loc))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        normal = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        exponential = torch.empty_like(normal).exponential_()
        return self.loc + self.scale * normal + exponential / self.rate


class _InverseGaussianSample(torch.autograd.Function):
    """Exact vectorized sampling with implicit CDF derivatives."""

    @staticmethod
    def forward(ctx, mu: Tensor, shape: Tensor, sample_shape: torch.Size) -> Tensor:
        size = sample_shape + mu.shape
        noise = torch.randn(size, dtype=mu.dtype, device=mu.device).square()
        t = mu * noise / (2 * shape)
        root = t.sqrt() * (t + 2).sqrt()
        smaller = mu / (1 + t + root)
        uniform = torch.rand(size, dtype=mu.dtype, device=mu.device)
        result = torch.where(
            uniform <= mu / (mu + smaller), smaller, mu * (mu / smaller)
        )
        ctx.save_for_backward(mu, shape, result)
        return result

    @staticmethod
    def backward(ctx, gradient: Tensor):
        mu, shape, value = ctx.saved_tensors
        # dF/dtheta divided by f, simplified before evaluation. The
        # exp(-a**2/2) factors cancel, so tail densities never underflow.
        argument = (shape / (2 * value)).sqrt() * (value / mu + 1)
        ratio = (
            math.sqrt(2 * math.pi)
            * (shape * value).sqrt()
            * torch.special.erfcx(argument)
        )
        d_mu = value * ratio / mu.square()
        d_shape = value / shape * (1 - ratio / mu)
        return (
            (gradient * d_mu).sum_to_size(mu.shape),
            (gradient * d_shape).sum_to_size(shape.shape),
            None,
        )


class InverseGaussian(Distribution):
    """Inverse Gaussian distribution in mean/shape parameterization.

    Parameters
    ----------
    mu : Tensor or float
        Strictly positive response mean.
    shape : Tensor or float
        Strictly positive shape, conventionally lambda.
    validate_args : bool or None
        Whether to validate parameters and evaluation points.

    Notes
    -----
    Sampling uses the exact Michael–Schucany–Haas transformation without
    rejection loops. Its gradients use implicit differentiation of the
    CDF, rather than differentiating the sampler's discrete branch choice.
    The gradient is ``-partial_theta CDF(y) / density(y)``; common
    exponential factors cancel analytically before numerical evaluation.
    """

    arg_constraints = {"mu": constraints.positive, "shape": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        mu: Tensor | float,
        shape: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.shape = broadcast_all(mu, shape)
        super().__init__(self.mu.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGaussian, _instance)
        shape = torch.Size(batch_shape)
        new.mu, new.shape = self.mu.expand(shape), self.shape.expand(shape)
        super(InverseGaussian, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def variance(self) -> Tensor:
        return self.mu.square() * (self.mu / self.shape)

    @property
    def mode(self) -> Tensor:
        ratio = 1.5 * self.mu / self.shape
        return self.mu / ((1 + ratio.square()).sqrt() + ratio)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        inside = (value > 0) & torch.isfinite(value)
        point = torch.where(inside, value, self.mu)
        residual = (point - self.mu) / point.sqrt() * (self.shape.sqrt() / self.mu)
        result = (
            0.5 * (self.shape.log() - math.log(2 * math.pi) - 3 * point.log())
            - 0.5 * residual.square()
        )
        return torch.where(inside, result, torch.full_like(result, -math.inf))

    def _log_tails(self, value: Tensor) -> tuple[Tensor, Tensor]:
        if bool(torch.isnan(value).any()):
            raise ValueError("InverseGaussian tails require non-NaN evaluation points")
        inside = (value > 0) & torch.isfinite(value)
        mu, shape = self.mu.double(), self.shape.double()
        point = torch.where(inside, value.double(), mu)
        root = shape.sqrt() / point.sqrt()
        a = (point - mu) / point.sqrt() * (shape.sqrt() / mu)
        separation = math.sqrt(2) * root
        upper = log_scaled_erfcx_difference(a / math.sqrt(2), separation) - math.log(2)
        b = (point.sqrt() * (shape.sqrt() / mu) + root) / math.sqrt(2)
        lower = torch.logsumexp(
            torch.stack(
                (
                    torch.special.log_ndtr(a),
                    -0.5 * a.square() - math.log(2) + log_erfcx_positive(b),
                )
            ),
            dim=0,
        )
        lower, upper = paired_log_tails(lower, upper)
        return (
            torch.where(inside, lower, torch.where(value > 0, 0.0, -torch.inf)),
            torch.where(inside, upper, torch.where(value > 0, -torch.inf, 0.0)),
        )

    def log_cdf(self, value: Tensor) -> Tensor:
        """Return log P(Y <= value), retaining finite extreme-tail values.

        Parameters
        ----------
        value : Tensor
            Extended-real evaluation points, broadcast with the batch.

        Returns
        -------
        Tensor
            Log cumulative probabilities. Computed in Float64 to retain
            probabilities that ordinary CDF evaluation rounds away.
        """
        return self._log_tails(value)[0]

    def log_survival_function(self, value: Tensor) -> Tensor:
        """Return log P(Y > value), retaining the Gaussian divided difference.

        Parameters
        ----------
        value : Tensor
            Extended-real evaluation points, broadcast with the batch.

        Returns
        -------
        Tensor
            Log strict upper-tail probabilities. Computed in Float64 to retain
            probabilities that ordinary CDF evaluation rounds away.
        """
        return self._log_tails(value)[1]

    def log_density_variation(self, lower: Tensor, upper: Tensor) -> Tensor:
        """Bound the absolute log-density change over a positive interval.

        Parameters
        ----------
        lower, upper : Tensor
            Interval endpoints, broadcast against the parameters.

        Returns
        -------
        Tensor
            A nonnegative Float64 variation bound, or infinity outside
            finite strictly positive endpoints.
        """
        lower, upper = lower.double(), upper.double()
        valid = (
            torch.isfinite(lower)
            & torch.isfinite(upper)
            & (lower > 0)
            & (upper > lower)
        )
        start = torch.where(valid, lower, self.mu.double())
        width = torch.where(valid, upper - lower, 1.0)
        mu, shape = self.mu.double(), self.shape.double()
        bound = width * (
            1.5 / start + 0.5 * (shape / start) / start + 0.5 * (shape / mu) / mu
        )
        return torch.where(valid, bound, torch.inf)

    def cdf(self, value: Tensor) -> Tensor:
        return self.log_cdf(value).exp().to(dtype=torch.result_type(value, self.mu))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return _InverseGaussianSample.apply(
            self.mu, self.shape, torch.Size(sample_shape)
        )


__all__ = ["ExGaussian", "InverseGaussian"]
