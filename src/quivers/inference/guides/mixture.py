"""Mixture variational guide.

A finite mixture of component guides:

.. math::

    q(z) = \\sum_{k=1}^{K} \\pi_k\\, q_k(z),

with :math:`\\pi \\in \\Delta^{K-1}` parameterized by an
unconstrained logit vector. Log-density uses ``logsumexp``:

.. math::

    \\log q(z) = \\mathrm{logsumexp}_k\\bigl(
        \\log \\pi_k + \\log q_k(z)\\bigr).

Sampling (`AutoMixtureGuide.rsample`) draws a component index
:math:`k \\sim \\mathrm{Categorical}(\\pi)` and then
:math:`z \\sim q_k`, so its draws follow exactly the density that
`AutoMixtureGuide.log_prob` scores. The component choice is
discrete, so these draws carry no gradient to the mixture logits.

Training instead goes through `AutoMixtureGuide.rsample_strata`,
which enumerates the components: it returns one reparameterized
draw per component together with the log mixture weights. Every
variational objective in [`quivers.inference.objectives`][quivers.inference.objectives]
consumes these strata, so the ELBO becomes the stratified ELBO

.. math::

    \\mathcal{L}(\\phi) = \\sum_{k=1}^{K} \\pi_k\\,
        \\mathbb{E}_{z \\sim q_k}\\bigl[\\log p(z, y) - \\log q(z)\\bigr],

and the importance-weighted bounds become their stratified
counterparts (SIWAE), following Morningstar, Vikram, Ham, Gallagher,
and Dillon (2021),
[Automatic Differentiation Variational Inference with Mixtures](https://proceedings.mlr.press/v130/morningstar21b.html)
([doi:10.48550/arXiv.2003.01687](https://doi.org/10.48550/arXiv.2003.01687)).
The estimate is unbiased for the objective of the mixture density
that `log_prob` scores; gradients reach the component parameters
through each component's reparameterized draw and reach the mixture
logits through the finite sum over components.

The component guides may be of any `Guide` subclass, including
another `AutoMixtureGuide`, whose strata are then enumerated
recursively. The canonical use case is to wrap several
`AutoNormalGuide` instances to recover a multimodal posterior that
no single unimodal guide can capture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.inference.guides.base import Guide, GuideDraw


class AutoMixtureGuide(Guide):
    """Finite mixture variational guide.

    Parameters
    ----------
    components : list[Guide]
        Component guides. All components must share the same
        `LatentRegistry` (i.e. be built against the same
        model + observed-name set).

    Raises
    ------
    ValueError
        If fewer than two components are given, or if the
        components cover different latent names.
    """

    def __init__(self, components: list[Guide]) -> None:
        super().__init__()
        if len(components) < 2:
            raise ValueError(
                f"AutoMixtureGuide: need at least 2 components, got {len(components)}"
            )
        reference = components[0]
        ref_names = tuple(reference.registry.names)
        for i, comp in enumerate(components[1:], 1):
            comp_names = tuple(comp.registry.names)
            if comp_names != ref_names:
                raise ValueError(
                    f"AutoMixtureGuide: component {i} has different "
                    f"latent names {comp_names!r} than component 0 "
                    f"{ref_names!r}"
                )
        self._registry = reference.registry
        self._component_guides: list[Guide] = list(components)
        self.components = nn.ModuleList(components)
        self.mixture_logits = nn.Parameter(torch.zeros(len(components)))

    @property
    def num_components(self) -> int:
        """Number of mixture components.

        Returns
        -------
        int
            The component count :math:`K`.
        """
        return len(self.components)

    def mixture_weights(self) -> torch.Tensor:
        """Mixture weights :math:`\\pi = \\mathrm{softmax}(\\text{logits})`.

        Returns
        -------
        torch.Tensor
            Shape ``(K,)``, non-negative and summing to one.
        """
        return F.softmax(self.mixture_logits, dim=-1)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Draw from the mixture density that `log_prob` scores.

        Samples a component index :math:`k \\sim
        \\mathrm{Categorical}(\\pi)` and returns component ``k``'s
        reparameterized draw. One index is drawn per call and shared
        across the batch rows, so a plate latent and the per-row
        latents of every row come from the same component, matching
        the per-row joint density `log_prob` evaluates.

        Gradients reach the chosen component's parameters but not
        the mixture logits, since the component choice is discrete.
        The variational objectives train the logits through
        `rsample_strata` instead.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Per-site constrained draws from the mixture.
        """
        with torch.no_grad():
            index = int(
                torch.distributions.Categorical(logits=self.mixture_logits)
                .sample()
                .item()
            )
        return self._component_guides[index].rsample(x)

    def rsample_strata(self, x: torch.Tensor) -> tuple[GuideDraw, ...]:
        """Enumerate the components as strata.

        Each component contributes its own strata (one for an
        ordinary component, several for a nested mixture), with the
        component's log mixture weight added to their log-weights and
        each draw scored by the mixture density. The resulting weights
        sum to one and the strata partition the mixture density, so
        :math:`\\sum_s \\omega_s f(z_s)` estimates
        :math:`\\mathbb{E}_q[f]` without bias, with gradients to the
        mixture logits through :math:`\\omega_s` and the density.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.

        Returns
        -------
        tuple[GuideDraw, ...]
            One weighted draw per stratum, ``S`` in total over the
            components.
        """
        log_pi = F.log_softmax(self.mixture_logits, dim=-1)
        draws: list[GuideDraw] = []
        for k, component in enumerate(self._component_guides):
            for draw in component.rsample_strata(x):
                draws.append(
                    GuideDraw(
                        sites=draw.sites,
                        log_density=self.log_prob(x, draw.sites),
                        log_weight=log_pi[k] + draw.log_weight.to(dtype=log_pi.dtype),
                    )
                )
        return tuple(draws)

    # ------------------------------------------------------------------
    # Log-density
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Mixture log-density via logsumexp over components.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        sites : dict[str, torch.Tensor]
            Per-site constrained values to score.

        Returns
        -------
        torch.Tensor
            :math:`\\log \\sum_k \\pi_k q_k(z)`, shape ``(batch,)``.
        """
        log_pi = F.log_softmax(self.mixture_logits, dim=-1)
        component_log_probs = torch.stack(
            [comp.log_prob(x, sites) for comp in self._component_guides], dim=0
        )
        return torch.logsumexp(log_pi.unsqueeze(-1) + component_log_probs, dim=0)

    @property
    def latent_names(self) -> list[str]:
        """Names of the latent sites the mixture covers.

        Returns
        -------
        list[str]
            The registry's site names, in order.
        """
        return list(self._registry.names)


__all__ = ["AutoMixtureGuide"]
