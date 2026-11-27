"""Discretized parametric distribution families.

Provides learnable morphisms backed by discretized versions of
standard probability distributions (normal, logit-normal, beta,
truncated normal).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.core.objects import SetObject, FinSet
from quivers.core.morphisms import FiniteMorphism
from quivers.core._util import EPS
from quivers.core.algebras import MARKOV


class _DiscretizedModule(nn.Module):
    """Base module for discretized parametric distributions."""

    pass


class DiscretizedNormal(FiniteMorphism):
    """A morphism whose codomain fibers are discretized normal densities.

    For each domain element, produces a probability distribution over
    the codomain by evaluating a normal density at evenly spaced bin
    centers and normalizing.

    The location μ and log-scale log(σ) are learnable parameters.
    When domain has size > 1, each domain element gets its own (μ, σ).

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : FinSet
        Target object (bins of the discretized distribution).
    low : float
        Lower bound of the discretization interval.
    high : float
        Upper bound of the discretization interval.

    Examples
    --------
    >>> A = FinSet(name="A", cardinality=2)
    >>> B = FinSet(name="response", cardinality=7)
    >>> f = DiscretizedNormal(A, B, low=0.0, high=1.0)
    >>> t = f.tensor  # shape (2, 7), rows sum to ~1
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: FinSet,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        super().__init__(domain, codomain, algebra=MARKOV)
        self._low = low
        self._high = high
        n_bins = codomain.cardinality

        # bin centers
        centers = torch.linspace(low, high, n_bins)

        self._module = _DiscretizedModule()
        self._module.register_buffer("centers", centers)

        # learnable parameters: one (μ, log_σ) per domain element
        dom_size = domain.size
        self._module.register_parameter("mu", nn.Parameter(torch.zeros(dom_size)))
        self._module.register_parameter(
            "log_sigma", nn.Parameter(torch.zeros(dom_size))
        )

    @property
    def tensor(self) -> torch.Tensor:
        """Discretized normal probabilities.

        Returns
        -------
        torch.Tensor
            Shape (*domain.shape, codomain.cardinality), rows sum to ~1.
        """
        mu = cast(torch.Tensor, self._module.mu)  # (dom_size,)
        sigma = (
            cast(torch.Tensor, self._module.log_sigma).exp().clamp(min=EPS)
        )  # (dom_size,)
        centers = cast(torch.Tensor, self._module.centers)  # (n_bins,)

        # compute log-density at each bin center for each domain element
        # mu: (dom, 1), centers: (1, bins)
        mu_expanded = mu.unsqueeze(-1)
        sigma_expanded = sigma.unsqueeze(-1)
        centers_expanded = centers.unsqueeze(0)

        log_density = -0.5 * ((centers_expanded - mu_expanded) / sigma_expanded) ** 2
        probs = F.softmax(log_density, dim=-1)

        return probs.reshape(self.tensor_shape)

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return self._module


class DiscretizedLogitNormal(FiniteMorphism):
    """A morphism with discretized logit-normal codomain fibers.

    The logit-normal distribution is a normal distribution on the logit
    scale: if X ~ Normal(μ, σ), then logistic(X) ~ LogitNormal(μ, σ).
    Useful for probabilities and bounded quantities.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : FinSet
        Target object (bins on (0, 1)).
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: FinSet,
    ) -> None:
        super().__init__(domain, codomain, algebra=MARKOV)
        n_bins = codomain.cardinality

        # bin centers in (0, 1), excluding endpoints
        centers = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1]

        self._module = _DiscretizedModule()
        self._module.register_buffer("centers", centers)

        dom_size = domain.size
        self._module.register_parameter("mu", nn.Parameter(torch.zeros(dom_size)))
        self._module.register_parameter(
            "log_sigma", nn.Parameter(torch.zeros(dom_size))
        )

    @property
    def tensor(self) -> torch.Tensor:
        """Discretized logit-normal probabilities.

        Returns
        -------
        torch.Tensor
            Shape (*domain.shape, codomain.cardinality), rows sum to ~1.
        """
        mu = cast(torch.Tensor, self._module.mu)
        sigma = cast(torch.Tensor, self._module.log_sigma).exp().clamp(min=EPS)
        centers = cast(torch.Tensor, self._module.centers)

        # logit transform of bin centers
        logit_centers = torch.log(centers / (1.0 - centers))

        mu_expanded = mu.unsqueeze(-1)
        sigma_expanded = sigma.unsqueeze(-1)
        logit_expanded = logit_centers.unsqueeze(0)

        # log-density of normal at logit(center), plus jacobian
        z = (logit_expanded - mu_expanded) / sigma_expanded
        log_density = -0.5 * z**2 - torch.log(centers * (1.0 - centers)).unsqueeze(0)

        probs = F.softmax(log_density, dim=-1)
        return probs.reshape(self.tensor_shape)

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return self._module


class DiscretizedBeta(FiniteMorphism):
    """A morphism with discretized beta-distribution codomain fibers.

    The beta distribution Beta(α, β) is parameterized via learnable
    log-concentration parameters. Bin centers are placed in (0, 1).

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : FinSet
        Target object (bins on (0, 1)).
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: FinSet,
    ) -> None:
        super().__init__(domain, codomain, algebra=MARKOV)
        n_bins = codomain.cardinality

        # bin centers in (0, 1)
        centers = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1]

        self._module = _DiscretizedModule()
        self._module.register_buffer("centers", centers)

        dom_size = domain.size
        # parameterize via log(α), log(β) for positivity
        self._module.register_parameter(
            "log_alpha", nn.Parameter(torch.zeros(dom_size))
        )
        self._module.register_parameter("log_beta", nn.Parameter(torch.zeros(dom_size)))

    @property
    def tensor(self) -> torch.Tensor:
        """Discretized beta probabilities.

        Returns
        -------
        torch.Tensor
            Shape (*domain.shape, codomain.cardinality), rows sum to ~1.
        """
        alpha = cast(torch.Tensor, self._module.log_alpha).exp().clamp(min=EPS)
        beta = cast(torch.Tensor, self._module.log_beta).exp().clamp(min=EPS)
        centers = cast(torch.Tensor, self._module.centers)

        # log Beta(x; α, β) ∝ (α-1)log(x) + (β-1)log(1-x)
        alpha_expanded = alpha.unsqueeze(-1)
        beta_expanded = beta.unsqueeze(-1)
        centers_expanded = centers.unsqueeze(0)

        log_density = (alpha_expanded - 1.0) * torch.log(
            centers_expanded.clamp(min=EPS)
        ) + (beta_expanded - 1.0) * torch.log((1.0 - centers_expanded).clamp(min=EPS))

        probs = F.softmax(log_density, dim=-1)
        return probs.reshape(self.tensor_shape)

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return self._module


class DiscretizedTruncatedNormal(FiniteMorphism):
    """A morphism with discretized truncated-normal codomain fibers.

    Normal distribution truncated to [low, high]. Bin centers are
    placed within the truncation interval.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : FinSet
        Target object (bins within [low, high]).
    low : float
        Lower truncation bound.
    high : float
        Upper truncation bound.
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: FinSet,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        super().__init__(domain, codomain, algebra=MARKOV)
        self._low = low
        self._high = high
        n_bins = codomain.cardinality

        # bin centers strictly within [low, high]
        centers = torch.linspace(low, high, n_bins)

        self._module = _DiscretizedModule()
        self._module.register_buffer("centers", centers)

        dom_size = domain.size
        self._module.register_parameter(
            "mu", nn.Parameter(torch.full((dom_size,), (low + high) / 2.0))
        )
        self._module.register_parameter(
            "log_sigma", nn.Parameter(torch.zeros(dom_size))
        )

    @property
    def tensor(self) -> torch.Tensor:
        """Discretized truncated-normal probabilities.

        Returns
        -------
        torch.Tensor
            Shape (*domain.shape, codomain.cardinality), rows sum to ~1.
        """
        mu = cast(torch.Tensor, self._module.mu)
        sigma = cast(torch.Tensor, self._module.log_sigma).exp().clamp(min=EPS)
        centers = cast(torch.Tensor, self._module.centers)

        mu_expanded = mu.unsqueeze(-1)
        sigma_expanded = sigma.unsqueeze(-1)
        centers_expanded = centers.unsqueeze(0)

        # normal log-density at bin centers (truncation handled by
        # softmax renormalization since bins are within [low, high])
        log_density = -0.5 * ((centers_expanded - mu_expanded) / sigma_expanded) ** 2
        probs = F.softmax(log_density, dim=-1)

        return probs.reshape(self.tensor_shape)

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return self._module


__all__ = [
    "DiscretizedNormal",
    "DiscretizedLogitNormal",
    "DiscretizedBeta",
    "DiscretizedTruncatedNormal",
]
