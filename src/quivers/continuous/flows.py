"""Normalizing flows as continuous morphisms.

A normalizing flow defines a bijective map from a simple base
distribution (standard normal) to a complex target distribution.
The map is composed of invertible layers with tractable Jacobians,
enabling exact log-density computation via the change-of-variables
formula.

For conditional flows, each layer's parameters depend on the
conditioning input x, making the flow a ContinuousMorphism:

    p(y | x) = p_base(f^{-1}(y; x)) * |det df^{-1}/dy|

This module provides:

    AffineCouplingLayer — single invertible affine coupling layer
    ConditionalFlow     — stack of coupling layers as a ContinuousMorphism
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from quivers.core.objects import SetObject
from quivers.continuous.spaces import ContinuousSpace, Euclidean
from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
)


class _ConditionedNet(nn.Module):
    """Small MLP that takes (x, z_fixed) and outputs (shift, log_scale).

    For discrete domains, x is first embedded via a learnable table.
    For continuous domains, x is used directly.

    Parameters
    ----------
    domain : AnySpace
        The conditioning space.
    input_dim : int
        Dimensionality of the fixed portion of z.
    output_dim : int
        Dimensionality of the transformed portion (outputs 2 * output_dim).
    hidden_dim : int
        Hidden layer width.
    """

    mappable: bool = True

    def __init__(
        self,
        domain: AnySpace,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if isinstance(domain, SetObject):
            self.embed = nn.Embedding(domain.size, hidden_dim)
            cond_dim = hidden_dim

        else:
            self.embed = None
            cond_dim = cast(ContinuousSpace, domain).dim

        self.net = nn.Sequential(
            nn.Linear(cond_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * output_dim),
        )
        self._output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        z_fixed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shift and log-scale.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning input.
        z_fixed : torch.Tensor
            Fixed portion of the latent vector. Shape (batch, input_dim).

        Returns
        -------
        shift : torch.Tensor
            Shift. Shape (batch, output_dim).
        log_scale : torch.Tensor
            Log-scale (clamped for stability). Shape (batch, output_dim).
        """
        if self.embed is not None:
            x_repr = self.embed(x.long())

        else:
            x_repr = x

        combined = torch.cat([x_repr, z_fixed], dim=-1)
        out = self.net(combined)
        shift = out[..., : self._output_dim]
        log_scale = out[..., self._output_dim :].clamp(-5.0, 5.0)
        return shift, log_scale


class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer (RealNVP-style).

    Splits the input z into two halves (z_a, z_b). One half passes
    through unchanged while the other is affinely transformed based
    on the first half (and the conditioning variable x):

        z_a' = z_a                               (unchanged)
        z_b' = z_b * exp(s(x, z_a)) + t(x, z_a) (transformed)

    The Jacobian is triangular, so its log-determinant is
    sum(s(x, z_a)).

    Parameters
    ----------
    domain : AnySpace
        Conditioning space.
    dim : int
        Total dimensionality of z.
    mask_even : bool
        If True, z_a = even indices, z_b = odd indices.
        If False, reversed.
    hidden_dim : int
        Hidden layer width for the scale/shift network.
    """

    mappable: bool = True

    def __init__(
        self,
        domain: AnySpace,
        dim: int,
        mask_even: bool = True,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self._dim = dim
        self._mask_even = mask_even

        # determine split sizes
        if mask_even:
            self._fixed_idx = torch.arange(0, dim, 2)
            self._transform_idx = torch.arange(1, dim, 2)

        else:
            self._fixed_idx = torch.arange(1, dim, 2)
            self._transform_idx = torch.arange(0, dim, 2)

        n_fixed = len(self._fixed_idx)
        n_transform = len(self._transform_idx)

        self.net = _ConditionedNet(
            domain,
            n_fixed,
            n_transform,
            hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: base -> target.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning input.
        z : torch.Tensor
            Input vector. Shape (batch, dim).

        Returns
        -------
        z_out : torch.Tensor
            Transformed vector. Shape (batch, dim).
        log_det : torch.Tensor
            Log-determinant of the Jacobian. Shape (batch,).
        """
        z_fixed = z[..., self._fixed_idx]
        z_transform = z[..., self._transform_idx]

        shift, log_scale = self.net(x, z_fixed)
        z_transformed = z_transform * log_scale.exp() + shift

        z_out = torch.empty_like(z)
        z_out[..., self._fixed_idx] = z_fixed
        z_out[..., self._transform_idx] = z_transformed

        log_det = log_scale.sum(dim=-1)
        return z_out, log_det

    def inverse(
        self,
        x: torch.Tensor,
        z_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass: target -> base.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning input.
        z_out : torch.Tensor
            Transformed vector. Shape (batch, dim).

        Returns
        -------
        z : torch.Tensor
            Original vector. Shape (batch, dim).
        log_det : torch.Tensor
            Log-determinant (negative of forward). Shape (batch,).
        """
        z_fixed = z_out[..., self._fixed_idx]
        z_transformed = z_out[..., self._transform_idx]

        shift, log_scale = self.net(x, z_fixed)
        z_original = (z_transformed - shift) * (-log_scale).exp()

        z = torch.empty_like(z_out)
        z[..., self._fixed_idx] = z_fixed
        z[..., self._transform_idx] = z_original

        log_det = -log_scale.sum(dim=-1)
        return z, log_det


class ConditionalFlow(ContinuousMorphism):
    """Conditional normalizing flow as a continuous morphism.

    Stacks affine coupling layers to form an invertible transformation
    from a standard normal base to the target distribution, conditioned
    on input x.

    The flow supports exact log-density computation:

        log p(y | x) = log N(f^{-1}(y; x); 0, I)
                       + sum_k log |det df_k^{-1}/dz_k|

    Sampling uses:

        z ~ N(0, I)
        y = f_K(... f_2(f_1(z; x); x) ...; x)

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Conditioning space.
    codomain : Euclidean
        Target continuous space.
    n_layers : int
        Number of coupling layers.
    hidden_dim : int
        Hidden layer width for scale/shift networks.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.continuous.spaces import Euclidean
    >>> A = FinSet(name="context", cardinality=10)
    >>> Y = Euclidean(name="output", dim=4)
    >>> flow = ConditionalFlow(A, Y, n_layers=6)
    >>> x = torch.tensor([0, 1, 2])
    >>> samples = flow.rsample(x)  # shape (3, 4)
    """

    mappable: bool = True

    def __init__(
        self,
        domain: AnySpace,
        codomain: Euclidean,
        n_layers: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim

        if d < 2:
            raise ValueError(
                f"ConditionalFlow requires codomain dim >= 2, got {d}. "
                "Use ConditionalNormal for 1-d targets."
            )

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(
                AffineCouplingLayer(
                    domain,
                    d,
                    mask_even=(i % 2 == 0),
                    hidden_dim=hidden_dim,
                )
            )

        self._d = d

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact log-density via change of variables.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning inputs.
        y : torch.Tensor
            Target values. Shape (batch, d).

        Returns
        -------
        torch.Tensor
            Log-densities. Shape (batch,).
        """
        z = y
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        # inverse pass through layers (reverse order)
        for layer in reversed(self.layers):
            z, log_det = cast(AffineCouplingLayer, layer).inverse(x, z)
            total_log_det = total_log_det + log_det

        # base distribution log-density (standard normal)
        log_base = -0.5 * z.pow(2).sum(dim=-1) - 0.5 * self._d * math.log(2 * math.pi)

        return log_base + total_log_det

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample via forward pass through the flow.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning inputs.
        sample_shape : torch.Size
            Additional sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples. Shape (*sample_shape, batch, d).
        """
        batch = x.shape[0]

        if len(sample_shape) > 0:
            n_extra = int(torch.Size(sample_shape).numel())
            total = n_extra * batch

            # replicate x for all samples
            x_rep = (
                x.unsqueeze(0)
                .expand(
                    n_extra,
                    *x.shape,
                )
                .reshape(total, *x.shape[1:])
                if x.dim() > 1
                else (x.unsqueeze(0).expand(n_extra, batch).reshape(total))
            )

            z = torch.randn(total, self._d, device=x.device)

            for layer in self.layers:
                z, _ = layer.forward(x_rep, z)

            if z.dim() > 1:
                return z.reshape(*sample_shape, batch, self._d)

            return z.reshape(*sample_shape, batch)

        else:
            z = torch.randn(batch, self._d, device=x.device)

            for layer in self.layers:
                z, _ = layer.forward(x, z)

            return z


__all__ = [
    "AffineCouplingLayer",
    "ConditionalFlow",
]
