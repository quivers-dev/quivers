"""Exponentially modified Gaussian and inverse Gaussian response distributions."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


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
        rate_scale = self.rate * self.scale
        difference = rate_scale - standardized
        # erfcx removes cancellation between the positive quadratic and
        # the Gaussian left tail. Mask its inactive negative argument.
        left = (
            self.rate.log()
            - math.log(2.0)
            - 0.5 * standardized.square()
            + torch.special.erfcx(difference.clamp_min(0) / math.sqrt(2.0)).log()
        )
        right_rate_scale = torch.where(
            difference > 0, torch.zeros_like(rate_scale), rate_scale
        )
        right_standardized = torch.where(
            difference > 0, torch.zeros_like(standardized), standardized
        )
        right = (
            self.rate.log()
            + 0.5 * right_rate_scale.square()
            - right_rate_scale * right_standardized
            + torch.special.log_ndtr(-difference.clamp_max(0))
        )
        return torch.where(difference > 0, left, right)

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        standardized = (point - self.loc) / self.scale
        first = torch.special.log_ndtr(standardized)
        second = self.log_prob(point) - self.rate.log()
        # The second term is bounded by the first; preserve that bound
        # against rounding before taking the stable exponential difference.
        result = first.exp() * -torch.expm1((second - first).clamp_max(0))
        return torch.where(finite, result, (value > 0).to(result))

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

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args and not bool(constraints.real.check(value).all()):
            raise ValueError("InverseGaussian.cdf requires real evaluation points")
        inside = (value > 0) & torch.isfinite(value)
        point = torch.where(inside, value, self.mu)
        root = (self.shape / point).sqrt()
        a = root * (point / self.mu - 1)
        b = root * (point / self.mu + 1)
        # exp(2*lambda/mu)*Phi(-b) equals
        # 0.5*erfcx(b/sqrt(2))*exp(-a*a/2), without an overflowing factor.
        result = 0.5 * torch.erfc(-a / math.sqrt(2.0)) + 0.5 * torch.special.erfcx(
            b / math.sqrt(2.0)
        ) * torch.exp(-0.5 * a.square())
        return torch.where(inside, result, (value > 0).to(result))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return _InverseGaussianSample.apply(
            self.mu, self.shape, torch.Size(sample_shape)
        )


__all__ = ["ExGaussian", "InverseGaussian"]
