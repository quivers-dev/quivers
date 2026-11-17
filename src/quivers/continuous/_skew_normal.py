"""Skew-normal responses with stable positive-integral tail probabilities."""

from __future__ import annotations

from functools import lru_cache
import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from quivers._quadrature_constants import (
    _NODES_32,
    _WEIGHTS_32,
    _NODES_128,
    _WEIGHTS_128,
)
from quivers.continuous._shape_families import _check_probability, _log1mexp


@lru_cache(maxsize=None)
def _quadrature(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cache Gaussian quadrature constants on the evaluation device.

    Parameters
    ----------
    device : torch.device
        Torch device receiving the constants.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Unit-interval nodes/weights and finite-tail nodes/log weights.
    """
    nodes = torch.tensor(_NODES_32, dtype=torch.float64, device=device)
    weights = torch.tensor(_WEIGHTS_32, dtype=torch.float64, device=device)
    tail_nodes = torch.tensor(_NODES_128, dtype=torch.float64, device=device)
    tail_weights = torch.tensor(_WEIGHTS_128, dtype=torch.float64, device=device)
    return (
        torch.as_tensor((nodes + 1) / 2, dtype=torch.float64, device=device),
        torch.as_tensor(weights / 2, dtype=torch.float64, device=device),
        torch.as_tensor(20 * (tail_nodes + 1), dtype=torch.float64, device=device),
        (20 * tail_weights).log(),
    )


def _positive_left_logcdf(h: Tensor, alpha: Tensor) -> Tensor:
    """Score negative standardized values with nonnegative skewness.

    Parameters
    ----------
    h : Tensor
        Nonnegative absolute standardized responses.
    alpha : Tensor
        Nonnegative skewness.

    Returns
    -------
    Tensor
        Stable log probabilities, without subtracting nearly equal Owen terms.
    """
    nodes, weights, v, log_weights = _quadrature(h.device)
    a = h * alpha
    central = (alpha <= 1) & (a <= 0.5)
    near_zero = (alpha > 1) & (a <= 0.5)
    tail = a > 0.5

    h0 = torch.where(central, h, torch.zeros_like(h))
    alpha0 = torch.where(central, alpha, torch.zeros_like(alpha))
    angle0 = alpha0.atan()
    integrand = torch.exp(
        -0.5 * (h0[..., None] * torch.tan(angle0[..., None] * nodes)).square()
    )
    ratio = (
        (2 / math.pi)
        * angle0
        / torch.special.erfcx(h0 / math.sqrt(2))
        * (integrand * weights).sum(-1)
    )
    first = torch.special.log_ndtr(-h0) + torch.log1p(-ratio)

    h1 = torch.where(near_zero, h, torch.zeros_like(h))
    alpha1 = torch.where(near_zero, alpha, torch.ones_like(alpha))
    angle1 = alpha1.reciprocal().atan()
    r = h1 / angle1
    leading = angle1 * (
        torch.exp(-0.5 * r.square())
        - r * math.sqrt(math.pi / 2) * torch.erfc(r / math.sqrt(2))
    )
    phi = angle1[..., None] * nodes
    small_phi = phi < 0.05
    safe_phi = torch.where(small_phi, torch.full_like(phi, 0.1), phi)
    offset = torch.where(
        small_phi,
        1 / 3 + phi.square() * (1 / 15 + phi.square() * (2 / 189 + phi.square() / 675)),
        safe_phi.sin().reciprocal().square() - safe_phi.reciprocal().square(),
    )
    correction = angle1 * (
        weights
        * torch.exp(-0.5 * (h1[..., None] / phi).square())
        * torch.expm1(-0.5 * h1[..., None].square() * offset)
    ).sum(-1)
    second = (leading + correction).log() - math.log(math.pi)

    h2 = torch.where(tail, h, torch.ones_like(h))
    a2 = torch.where(tail, a, torch.ones_like(a))
    x = v / a2[..., None]
    terms = (
        log_weights
        - v
        - 0.5 * x.square()
        - torch.log(h2[..., None].square() + (a2[..., None] + x).square())
    )
    third = (
        -0.5 * (h2.square() + a2.square())
        + h2.log()
        - math.log(math.pi)
        - a2.log()
        + torch.logsumexp(terms, -1)
    )
    return torch.where(central, first, torch.where(near_zero, second, third))


def _positive_log_tails(z: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Score both tails for nonnegative skewness, each from its small side.

    Parameters
    ----------
    z : Tensor
        Standardized real values.
    alpha : Tensor
        Nonnegative skewness.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log cumulative and log survival probabilities. Below zero the
        cumulative probability is computed directly and the survival is its
        complement; above zero the survival is computed directly, so it
        stays finite where the cumulative probability rounds to one.
    """
    h = z.abs()
    left = _positive_left_logcdf(h, alpha)
    normal = torch.special.log_ndtr(-h)
    log_survival = normal + torch.log(2 - torch.exp((left - normal).clamp_max(0)))
    lower = z <= 0
    return (
        torch.where(lower, left, _log1mexp(log_survival)),
        torch.where(lower, _log1mexp(left), log_survival),
    )


def _log_density(z: Tensor, alpha: Tensor) -> Tensor:
    """Evaluate the standardized skew-normal log density.

    Parameters
    ----------
    z : Tensor
        Standardized values.
    alpha : Tensor
        Real skewness.

    Returns
    -------
    Tensor
        Log densities on the real line.
    """
    return (
        math.log(2)
        - 0.91893853320467274178
        - 0.5 * z.square()
        + torch.special.log_ndtr(alpha * z)
    )


class _SkewLogCDF(torch.autograd.Function):
    """Numerically evaluated CDF with analytic reverse and forward derivatives."""

    generate_vmap_rule = True

    @staticmethod
    def forward(z, alpha):
        """Evaluate the numerical distribution transform without recording its integration graph.

        Parameters
        ----------
        z : Tensor
            Standardized real response values.
        alpha : Tensor
            Real standardized skewness parameters.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        reflected = alpha < 0
        point = torch.where(reflected, -z, z)
        log_cdf, log_survival = _positive_log_tails(point, alpha.abs())
        return torch.where(reflected, log_survival, log_cdf)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors for analytic reverse and forward differentiation.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        inputs : tuple[Tensor, ...]
            Inputs of the forward transformation.
        output : Tensor
            Computed forward output.

        Returns
        -------
        None
            Context updated in place.
        """
        ctx.save_for_backward(*inputs, output)
        ctx.save_for_forward(*inputs, output)

    @staticmethod
    def backward(ctx, gradient):
        """Propagate derivatives using analytic distribution identities.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        gradient : Tensor
            Upstream reverse-mode derivatives.

        Returns
        -------
        tuple[Tensor, Tensor]
            Broadcast result with the parameter device and dtype.
        """
        z, alpha, log_cdf = ctx.saved_tensors
        active = gradient != 0
        z = torch.where(active, z, 0.0)
        alpha = torch.where(active, alpha, 0.0)
        log_cdf = torch.where(active, log_cdf, 0.0)
        dz = torch.exp(_log_density(z, alpha) - log_cdf)
        da = -torch.exp(
            -0.5 * z.square() * (1 + alpha.square())
            - math.log(math.pi)
            - torch.log1p(alpha.square())
            - log_cdf
        )
        return gradient * dz, gradient * da

    @staticmethod
    def jvp(ctx, z_tangent, alpha_tangent):
        """Propagate forward sensitivities using analytic distribution identities.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        z_tangent : Tensor
            Standardized-response directional derivatives.
        alpha_tangent : Tensor
            Skewness directional derivatives.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        z, alpha, log_cdf = ctx.saved_tensors
        dz = torch.exp(_log_density(z, alpha) - log_cdf)
        da = -torch.exp(
            -0.5 * z.square() * (1 + alpha.square())
            - math.log(math.pi)
            - torch.log1p(alpha.square())
            - log_cdf
        )
        return torch.where(z_tangent == 0, 0.0, z_tangent * dz) + torch.where(
            alpha_tangent == 0, 0.0, alpha_tangent * da
        )


def _normal_quantile_from_log(log_probability: Tensor) -> Tensor:
    """Invert log-normal probabilities without underflowing subnormal tails.

    Parameters
    ----------
    log_probability : Tensor
        Log probabilities below minus log two.

    Returns
    -------
    Tensor
        Standard normal quantiles.
    """
    tail = log_probability < -700
    safe = torch.where(tail, torch.full_like(log_probability, -1), log_probability)
    ordinary = torch.special.ndtri(safe.exp())
    target = torch.where(tail, log_probability, torch.full_like(log_probability, -1))
    z = -torch.sqrt(-2 * target)
    for _ in range(6):
        current = torch.special.log_ndtr(z)
        inverse_slope = torch.exp(current + 0.5 * z.square() + 0.91893853320467274178)
        z = z - (current - target) * inverse_slope
    return torch.where(tail, z, ordinary)


class _SkewQuantile(torch.autograd.Function):
    """Safeguarded Newton quantiles with implicit distribution derivatives."""

    generate_vmap_rule = True

    @staticmethod
    def forward(p, alpha):
        """Evaluate the numerical distribution transform without recording its integration graph.

        Parameters
        ----------
        p : Tensor
            Probabilities strictly between zero and one.
        alpha : Tensor
            Real standardized skewness parameters.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        reflected = p > 0.5
        p = torch.where(reflected, 1 - p, p)
        alpha = torch.where(reflected, -alpha, alpha)
        log_p = p.log()
        lower = _normal_quantile_from_log(log_p - math.log(2))
        upper = -torch.special.ndtri((1 - p) / 2)
        z = (lower + upper) / 2
        for _ in range(48):
            log_f = _log_density(z, alpha)
            log_cdf = _SkewLogCDF.apply(z, alpha)
            above = log_cdf >= log_p
            upper = torch.where(above, z, upper)
            lower = torch.where(above, lower, z)
            step = (log_cdf - log_p) * torch.exp(log_cdf - log_f)
            proposal = z - step
            accepted = (
                torch.isfinite(proposal) & (proposal >= lower) & (proposal <= upper)
            )
            update = torch.where(accepted, proposal, (lower + upper) / 2)
            z = torch.where((log_cdf - log_p).abs() < 2e-14, z, update)
        return torch.where(reflected, -z, z)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors for analytic reverse and forward differentiation.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        inputs : tuple[Tensor, ...]
            Inputs of the forward transformation.
        output : Tensor
            Computed forward output.

        Returns
        -------
        None
            Context updated in place.
        """
        ctx.save_for_backward(inputs[1], output)
        ctx.save_for_forward(inputs[1], output)

    @staticmethod
    def backward(ctx, gradient):
        """Propagate derivatives using analytic distribution identities.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        gradient : Tensor
            Upstream reverse-mode derivatives.

        Returns
        -------
        tuple[Tensor, Tensor]
            Broadcast result with the parameter device and dtype.
        """
        alpha, z = ctx.saved_tensors
        log_f = _log_density(z, alpha)
        dp = torch.exp(-log_f)
        da = torch.exp(
            -0.5 * z.square() * (1 + alpha.square())
            - math.log(math.pi)
            - torch.log1p(alpha.square())
            - log_f
        )
        return gradient * dp, gradient * da

    @staticmethod
    def jvp(ctx, p_tangent, alpha_tangent):
        """Propagate forward sensitivities using analytic distribution identities.

        Parameters
        ----------
        ctx : object
            PyTorch differentiation context.
        p_tangent : Tensor
            Probability directional derivatives.
        alpha_tangent : Tensor
            Skewness directional derivatives.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        alpha, z = ctx.saved_tensors
        log_f = _log_density(z, alpha)
        dp = torch.exp(-log_f)
        da = torch.exp(
            -0.5 * z.square() * (1 + alpha.square())
            - math.log(math.pi)
            - torch.log1p(alpha.square())
            - log_f
        )
        return torch.where(p_tangent == 0, 0.0, p_tangent * dp) + torch.where(
            alpha_tangent == 0, 0.0, alpha_tangent * da
        )


class SkewNormal(Distribution):
    """Skew-normal law in Azzalini location, scale, and skewness parameters.

    Parameters
    ----------
    loc : Tensor or float
        Location of the underlying skewed normal, distinct from its mean.
    scale : Tensor or float
        Positive scale, distinct from its standard deviation.
    skewness : Tensor or float
        Real Azzalini shape parameter, conventionally alpha.
    validate_args : bool or None
        Whether to validate parameters.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "skewness": constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, skewness, validate_args=None):
        self.loc, self.scale, self.skewness = broadcast_all(loc, scale, skewness)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views without copying their underlying storage.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested broadcast batch dimensions.
        _instance : Distribution or None
            Optional instance supplied by PyTorch.

        Returns
        -------
        Distribution
            Broadcast result with the parameter device and dtype.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(shape))
        super(SkewNormal, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        """Return the response expectation, or infinity when its first moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        delta = self.skewness / torch.hypot(
            self.skewness, torch.ones_like(self.skewness)
        )
        return self.loc + self.scale * delta * math.sqrt(2 / math.pi)

    @property
    def variance(self):
        """Return the response variance, or infinity when its second moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        delta = self.skewness / torch.hypot(
            self.skewness, torch.ones_like(self.skewness)
        )
        return self.scale.square() * (1 - (2 / math.pi) * delta.square())

    def log_prob(self, value):
        """Evaluate log densities, returning negative infinity outside the support.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        z = (point - self.loc) / self.scale
        return torch.where(
            finite, _log_density(z, self.skewness) - self.scale.log(), -torch.inf
        )

    def log_cdf(self, value):
        """Evaluate cumulative probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        z, alpha = torch.broadcast_tensors(
            (point - self.loc) / self.scale, self.skewness
        )
        output = _SkewLogCDF.apply(z.double(), alpha.double()).to(z.dtype)
        return torch.where(finite, output, torch.where(value > 0, 0.0, -torch.inf))

    def log_survival_function(self, value):
        """Evaluate complementary probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        z, alpha = torch.broadcast_tensors(
            (self.loc - point) / self.scale, -self.skewness
        )
        output = _SkewLogCDF.apply(z.double(), alpha.double()).to(z.dtype)
        return torch.where(finite, output, torch.where(value > 0, -torch.inf, 0.0))

    def cdf(self, value):
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Transform probabilities to response quantiles.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        if self._validate_args:
            _check_probability(value)
        inside = (value > 0) & (value < 1)
        p, alpha = torch.broadcast_tensors(
            torch.where(inside, value, torch.full_like(value, 0.5)), self.skewness
        )
        z = _SkewQuantile.apply(p.double(), alpha.double()).to(p.dtype)
        result = self.loc + self.scale * z
        return torch.where(
            inside, result, torch.where(value == 0, -torch.inf, torch.inf)
        )

    def rsample(self, sample_shape=torch.Size()):
        """Draw responses with pathwise gradients through the distribution parameters.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        shape = self._extended_shape(sample_shape)
        first = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        second = torch.randn_like(first)
        denominator = torch.hypot(self.skewness, torch.ones_like(self.skewness))
        return self.loc + self.scale * (
            (self.skewness / denominator) * first.abs() + second / denominator
        )


__all__ = ["SkewNormal"]
