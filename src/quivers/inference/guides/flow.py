"""Normalising-flow variational guides.

A normalizing-flow guide parameterizes the variational posterior
as the pushforward of a fixed base distribution
:math:`p_0 = \\mathcal{N}(0, I)` through a learnable stack of
invertible transforms :math:`T_1, \\dots, T_K`:

.. math::

    z_0 \\sim p_0, \\quad
    z_K = T_K \\circ \\dots \\circ T_1(z_0).

The density at :math:`z_K` follows from the change-of-variables
formula:

.. math::

    \\log q(z_K) = \\log p_0(z_0)
        - \\sum_{k=1}^{K} \\log |\\det J_{T_k}(z_{k-1})|.

After the flow runs in flat unconstrained space, the registry's
per-site bijectors map :math:`z_K` to constrained per-site
tensors, adding their own change-of-variables term.

The flow stack runs on the flat ``(D,)`` vector produced by
`LatentRegistry.unflatten_unconstrained`'s inverse; per-site
shape contortions live in the registry, not in the flow. This is
the standard Pyro / NumPyro convention.

This module ships three concrete guides:

* `AutoNormalizingFlow`: user-supplied list of
  [`quivers.inference.transforms.TransformModule`][quivers.inference.transforms.TransformModule]
  instances. Use this when you want a custom architecture.
* `AutoIAFGuide`: preconfigured stack of
  [`quivers.inference.transforms.InverseAutoregressiveTransform`][quivers.inference.transforms.InverseAutoregressiveTransform]
  layers separated by reverse permutations. The default flow
  guide for variational inference (Pyro's flagship NF guide).
* `AutoNeuralSplineGuide`: preconfigured stack of
  [`quivers.inference.transforms.NeuralSplineCouplingTransform`][quivers.inference.transforms.NeuralSplineCouplingTransform]
  layers with alternating coupling masks. Sharper than IAF for
  posteriors with sharp modes or near-bounded support.
"""

from __future__ import annotations

import torch
import torch.distributions as D
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.transforms import (
    MADE,
    InverseAutoregressiveTransform,
    NeuralSplineCouplingTransform,
    TransformModule,
    alternating_mask,
    make_coupling_mlp,
)


class AutoNormalizingFlow(Guide):
    """Normalising-flow variational guide over the flat latent vector.

    Parameters
    ----------
    model : MonadicProgram
        Generative model to build a guide for.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    transforms : list[TransformModule]
        Flow stack applied to the standard-Normal base. Each
        `TransformModule` must accept a
        ``(..., D)``-shaped tensor where ``D`` is the registry's
        total unconstrained dimension.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        transforms: list[TransformModule],
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._registry = self.build_registry(
            model, observed_names, x=x, observations=observations
        )
        D_total = self._registry.total_unconstrained_dim
        if D_total == 0:
            raise ValueError(
                f"{type(self).__name__}: registry has zero total "
                f"unconstrained dimension; model has no continuous "
                f"latents to guide"
            )
        if not transforms:
            raise ValueError(
                f"{type(self).__name__}: transforms list must be non-empty"
            )
        self._D = D_total
        self.flow = nn.ModuleList(transforms)
        self.register_buffer("base_loc", torch.zeros(D_total))
        self.register_buffer("base_scale", torch.ones(D_total))

    # ------------------------------------------------------------------
    # Flow forward / inverse
    # ------------------------------------------------------------------

    def _forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Push ``z0`` through the flow; return ``(z_K, sum log|det|)``."""
        log_det_total = torch.zeros(z0.shape[:-1], device=z0.device)
        z = z0
        for layer in self.flow:
            z_next = layer(z)
            log_det_total = log_det_total + layer.log_abs_det_jacobian(z, z_next)
            z = z_next
        return z, log_det_total

    def _inverse(self, z_K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse the flow; return ``(z_0, sum log|det|)`` where the
        log-det is accumulated in the *forward* direction so adding
        it to ``log p_0(z_0)`` produces the correct ``log q(z_K)``."""
        log_det_total = torch.zeros(z_K.shape[:-1], device=z_K.device)
        z = z_K
        for layer in reversed(self.flow):
            z_prev = layer.inv(z)
            log_det_total = log_det_total + layer.log_abs_det_jacobian(z_prev, z)
            z = z_prev
        return z, log_det_total

    def _base_dist(self) -> D.Distribution:
        return D.Independent(D.Normal(self.base_loc, self.base_scale), 1)

    # ------------------------------------------------------------------
    # Sample / log-prob
    # ------------------------------------------------------------------

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One flow draw, unflattened and bijected to constrained space."""
        batch = x.shape[0]
        z0 = self._base_dist().rsample()
        z_K, _ = self._forward(z0)
        per_site = self._registry.unflatten_unconstrained(z_K)
        result: dict[str, torch.Tensor] = {}
        for site in self._registry.sites.values():
            z_site = per_site[site.name]
            if not site.is_plate:
                z_site = z_site.unsqueeze(0).expand(batch, *site.unconstrained_shape)
            v = site.bijector(z_site)
            if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                v = v.squeeze(-1)
            result[site.name] = v
        return result

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-density at the supplied constrained sites."""
        batch = x.shape[0]
        unconstrained_per_site: dict[str, torch.Tensor] = {}
        bijector_log_det = torch.zeros((), device=x.device)
        for site in self._registry.sites.values():
            if site.name not in sites:
                raise KeyError(
                    f"{type(self).__name__}.log_prob: missing site {site.name!r}"
                )
            v = sites[site.name]
            if site.constrained_dim == 1 and v.dim() == (1 if site.is_plate else 1):
                v_e = v.unsqueeze(-1)
            else:
                v_e = v
            if not site.is_plate and v_e.dim() == len(site.unconstrained_shape) + 1:
                v_e = v_e[0]
            z_site = site.bijector.inv(v_e)
            unconstrained_per_site[site.name] = z_site
            bijector_log_det = bijector_log_det + (
                site.bijector.inv.log_abs_det_jacobian(v_e, z_site).sum()
            )

        z_K = self._registry.flatten_unconstrained(unconstrained_per_site)
        z_0, flow_log_det = self._inverse(z_K)
        log_p_base = self._base_dist().log_prob(z_0)
        # log q(z_K) = log p_0(z_0) - log|det dT/dz_0|
        #            = log p_0(z_0) - flow_log_det
        log_q_z = log_p_base - flow_log_det
        return (log_q_z + bijector_log_det).expand(batch)

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)


# ---------------------------------------------------------------------------
# Preconfigured architectures
# ---------------------------------------------------------------------------


class _ReversePermutation(TransformModule):
    """Deterministic reverse-order permutation. Used between IAF
    blocks to ensure every coordinate sees every other in some
    layer of the stack."""

    bijective = True

    def __init__(self, dim: int) -> None:
        super().__init__(cache_size=0)
        from torch.distributions import constraints as _constraints

        self.domain = _constraints.real_vector
        self.codomain = _constraints.real_vector
        self.register_buffer("perm", torch.arange(dim - 1, -1, -1, dtype=torch.long))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.perm)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y.index_select(-1, self.inv_perm)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        return torch.zeros(x.shape[:-1], device=x.device)


class AutoIAFGuide(AutoNormalizingFlow):
    """Inverse-autoregressive-flow guide.

    Default normalizing-flow guide for variational inference
    (Kingma-Salimans-Jozefowicz et al. 2016). Stack of
    `InverseAutoregressiveTransform` layers, each separated
    by a reverse permutation so successive layers have different
    autoregressive orderings.

    Sampling is parallel (one MLP forward per layer); density
    evaluation is sequential (one coordinate at a time per layer),
    so this guide should be used with objectives that sample more
    than they score the same flow (ELBO, IWAE).

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    num_flows : int
        Number of IAF blocks in the stack. Default ``4``.
    hidden_dim : int
        Hidden width of every MADE inside the stack. Default
        ``2 * D`` where ``D`` is the latent dimension.
    num_hidden_layers : int
        Number of hidden layers in each MADE. Default ``2``.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        num_flows: int = 4,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 2,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if num_flows < 1:
            raise ValueError(f"AutoIAFGuide: num_flows must be >= 1, got {num_flows}")
        registry = Guide.build_registry(
            model, observed_names, x=x, observations=observations
        )
        D_total = registry.total_unconstrained_dim
        if hidden_dim is None:
            hidden_dim = max(8, 2 * D_total)
        if D_total < 2:
            raise ValueError(
                f"AutoIAFGuide: model must have >= 2 unconstrained "
                f"latent dimensions for an IAF (got {D_total}); use "
                f"AutoNormalGuide for 1-D models"
            )
        layers: list[TransformModule] = []
        for i in range(num_flows):
            made = MADE(
                dim=D_total,
                n_per_dim=2,
                hidden=hidden_dim,
                n_hidden_layers=num_hidden_layers,
            )
            layers.append(InverseAutoregressiveTransform(made))
            if i < num_flows - 1:
                layers.append(_ReversePermutation(D_total))
        super().__init__(model, observed_names, layers, x=x, observations=observations)


class AutoNeuralSplineGuide(AutoNormalizingFlow):
    """Neural-spline-flow guide (Durkan-Bekasov-Murray-Papamakarios 2019).

    Stack of monotone rational-quadratic spline coupling layers
    (`NeuralSplineCouplingTransform`) with alternating
    half-masks. Sharper than IAF for posteriors with bounded
    support or sharp modes; comparable runtime.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    num_flows : int
        Number of coupling layers. Default ``4``.
    num_bins : int
        Number of spline bins per coordinate. Default ``8``.
    tail_bound : float
        Inputs outside ``[-tail_bound, tail_bound]`` pass through
        as identity. Default ``3.0``.
    hidden_dim : int
        Hidden width of the coupling MLPs. Default ``max(64, 2*D)``.
    num_hidden_layers : int
        Hidden layers in each coupling MLP. Default ``2``.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        num_flows: int = 4,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 2,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if num_flows < 1:
            raise ValueError(
                f"AutoNeuralSplineGuide: num_flows must be >= 1, got {num_flows}"
            )
        registry = Guide.build_registry(
            model, observed_names, x=x, observations=observations
        )
        D_total = registry.total_unconstrained_dim
        if D_total < 2:
            raise ValueError(
                f"AutoNeuralSplineGuide: model must have >= 2 "
                f"unconstrained latent dimensions for a spline "
                f"coupling flow (got {D_total})"
            )
        if hidden_dim is None:
            hidden_dim = max(64, 2 * D_total)
        layers: list[TransformModule] = []
        for i in range(num_flows):
            mask = alternating_mask(D_total, even=(i % 2 == 0))
            num_unmasked = int(mask.sum().item())
            num_masked = D_total - num_unmasked
            # Net produces (3 * num_bins - 1) parameters per masked coord.
            out_dim = num_masked * (3 * num_bins - 1)
            net = make_coupling_mlp(
                n_in=num_unmasked,
                n_out=out_dim,
                hidden=hidden_dim,
                n_hidden_layers=num_hidden_layers,
            )
            layers.append(
                NeuralSplineCouplingTransform(
                    dim=D_total,
                    net=net,
                    mask=mask,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                )
            )
        super().__init__(model, observed_names, layers, x=x, observations=observations)


__all__ = [
    "AutoNormalizingFlow",
    "AutoIAFGuide",
    "AutoNeuralSplineGuide",
]
