"""Boundary morphisms between discrete and continuous spaces.

These morphisms bridge the gap between the finite tensor world
(FinSet, discrete Morphism) and the continuous sampling world
(ContinuousSpace, ContinuousMorphism).

Morphisms provided
------------------
Discretize — continuous space -> finite set (binning)
Embed      — finite set -> continuous space (kernel density)
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.core.objects import FinSet
from quivers.continuous.spaces import Euclidean
from quivers.continuous.morphisms import (
    ContinuousMorphism,
    _event_size,
    dimension_probe,
)
from quivers.core._util import EPS


class Discretize(ContinuousMorphism):
    """Map a continuous space to a finite set by binning.

    Divides a bounded continuous space into n_bins equal-width bins
    and assigns each continuous input to the bin containing it. The
    resulting distribution is deterministic (one-hot on the correct bin).

    This is useful for converting continuous outputs into discrete
    categories for downstream processing in the finite tensor world.

    For log_prob: returns log(1) = 0 if y equals the correct bin,
    log(0) otherwise. In practice uses a soft assignment based on
    distance to bin centers for gradient flow.

    Parameters
    ----------
    domain : Euclidean
        Source continuous space (must be bounded, 1-dimensional).
    n_bins : int
        Number of discrete bins.
    soft : bool
        If True (default), use soft binning via softmax over negative
        squared distances. If False, use hard (argmax) assignment.
    temperature : float
        Temperature for soft binning. Lower = sharper.
    """

    mappable: bool = True

    def __init__(
        self,
        domain: Euclidean,
        n_bins: int,
        soft: bool = True,
        temperature: float = 0.1,
    ) -> None:
        if not isinstance(domain, Euclidean) or not domain.is_bounded:
            raise ValueError("Discretize requires a bounded Euclidean domain")

        if domain.dim != 1:
            raise ValueError(
                f"Discretize currently supports 1-d spaces only, got dim={domain.dim}"
            )

        codomain = FinSet(name="bins", cardinality=n_bins)
        super().__init__(domain, codomain)

        self._n_bins = n_bins
        self._soft = soft
        self._temperature = temperature

        # compute bin centers and register as buffer
        assert domain.low is not None and domain.high is not None
        centers = torch.linspace(domain.low, domain.high, n_bins)
        self.register_buffer("centers", centers)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of bin assignment y given input x.

        Parameters
        ----------
        x : torch.Tensor
            Continuous inputs. Shape (batch, 1).
        y : torch.Tensor
            Bin indices. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        # soft assignment probabilities
        probs = self._soft_assign(x)  # (batch, n_bins)
        y_long = y.long()
        selected = probs[torch.arange(len(y_long)), y_long]
        return torch.log(selected.clamp(min=EPS))

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Assign continuous inputs to discrete bins.

        Parameters
        ----------
        x : torch.Tensor
            Continuous inputs. Shape (batch, 1).
        sample_shape : torch.Size
            Ignored (assignment is deterministic).

        Returns
        -------
        torch.Tensor
            Bin indices. Shape (batch,) or (*sample_shape, batch).
        """
        if self._soft:
            probs = self._soft_assign(x)
            bins = torch.multinomial(probs, 1).squeeze(-1)

        else:
            # hard assignment: closest bin center
            dists = (x - cast(torch.Tensor, self.centers).unsqueeze(0)).abs()
            bins = dists.argmin(dim=-1)

        if len(sample_shape) > 0:
            return bins.unsqueeze(0).expand(*sample_shape, -1)

        return bins

    def _soft_assign(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft bin assignment probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Continuous inputs. Shape (batch, 1).

        Returns
        -------
        torch.Tensor
            Assignment probabilities. Shape (batch, n_bins).
        """
        # x: (batch, 1), centers: (n_bins,)
        if x.dim() == 2:
            x_flat = x.squeeze(-1)

        else:
            x_flat = x

        # squared distance to each bin center
        dists_sq = (
            x_flat.unsqueeze(-1) - cast(torch.Tensor, self.centers).unsqueeze(0)
        ) ** 2

        # softmax over negative distances
        return F.softmax(-dists_sq / self._temperature, dim=-1)


class Embed(ContinuousMorphism):
    """Map a finite set to a continuous space via kernel density placement.

    Each element i of the domain FinSet is associated with a point
    in the continuous codomain. Sampling from Embed(x=i) produces
    a value near that point, with learnable spread.

    Concretely, Embed places a Gaussian kernel at each bin center:

        p(y | i) = Normal(center_i, sigma_i)

    The centers and log-sigmas are learnable parameters.

    Parameters
    ----------
    domain : FinSet
        Source discrete set.
    codomain : Euclidean
        Target continuous space (should be bounded for initialization).
    """

    mappable: bool = True

    def __init__(
        self,
        domain: FinSet,
        codomain: Euclidean,
    ) -> None:
        super().__init__(domain, codomain)
        n = domain.size
        d = codomain.dim

        # initialize centers evenly spaced across codomain
        if codomain.is_bounded:
            assert codomain.low is not None and codomain.high is not None
            if d == 1:
                init_centers = torch.linspace(
                    codomain.low,
                    codomain.high,
                    n,
                ).unsqueeze(-1)

            else:
                init_centers = (
                    torch.rand(n, d) * (codomain.high - codomain.low) + codomain.low
                )

        else:
            init_centers = torch.randn(n, d) * 0.5

        self.centers = nn.Parameter(init_centers)
        self.log_sigma = nn.Parameter(torch.zeros(n, d))

    def _kernel_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the Gaussian kernel's location and scale at ``x``.

        Both gathers index the learnable tables by the domain index,
        so the returned shapes follow ``x``: a ``(batch,)`` index
        vector gives ``(batch, d)``, and a ``(batch, seq)`` index
        matrix (a token sequence) gives ``(batch, seq, d)``.
        """
        index = x.long()
        mu = self.centers[index]
        sigma = self.log_sigma[index].exp().clamp(min=EPS)
        return mu, sigma

    def base_dimension(self, x: torch.Tensor) -> int:
        """One standard-normal coordinate per embedded output axis.

        A ``(batch, seq)`` index matrix places one kernel per
        position, so the budget grows with the sequence length as well
        as the codomain dimension.
        """
        mu, _ = self._kernel_params(dimension_probe(x))
        return _event_size(mu.shape)

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """The location-scale map around the embedding of ``x``."""
        mu, sigma = self._kernel_params(x)
        return mu + sigma * base.reshape(mu.shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of y under the kernel centered at x's embedding.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).
        y : torch.Tensor
            Continuous outputs. Shape (batch, d).

        Returns
        -------
        torch.Tensor
            Log-densities. Shape (batch,).
        """
        mu, sigma = self._kernel_params(x)

        log_p = (
            -0.5 * ((y - mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
        )

        return log_p.sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from the Gaussian kernel at x's embedding point.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).
        sample_shape : torch.Size
            Additional sample dimensions.

        Returns
        -------
        torch.Tensor
            Continuous samples. Shape (*sample_shape, batch, d).
        """
        mu, sigma = self._kernel_params(x)

        eps = torch.randn(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )

        return mu + sigma * eps


__all__ = [
    "Discretize",
    "Embed",
]
