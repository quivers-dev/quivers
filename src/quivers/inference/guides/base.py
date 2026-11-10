"""Variational guide abstract base.

A `Guide` is a parameterized distribution
:math:`q_\\phi(z \\mid x)` over a model's latent variables. The
ABC fixes a uniform contract:

* `Guide.rsample` returns a dict ``{site_name: tensor}``
  whose per-site shapes match the model's trace-side convention
  (plate latents at ``(|A|, *B.shape)``, scalar latents at
  ``(batch, *B.shape)`` or ``(batch,)`` for scalar event shape).
* `Guide.log_prob` evaluates the variational density of a
  per-site dict and returns a ``(batch,)``-shaped tensor.
* `Guide.rsample_strata` returns a stratified set of draws whose
  weighted average estimates expectations under the same density
  that `Guide.log_prob` scores. The variational objectives consume
  this method, so a guide whose draws are not reparameterizable as a
  whole (a finite mixture, whose component choice is discrete) can
  still be trained with pathwise gradients by enumerating its
  discrete part exactly.

Every concrete subclass under [`quivers.inference.guides`][quivers.inference.guides] is
constructed against a `quivers.inference.registry.LatentRegistry`,
with no per-guide model walk. New guides supply the *variational
family's structure* (Normal vs MVN vs flow), not its
introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.registry import LatentRegistry


class Guide(nn.Module, ABC):
    """Abstract variational guide.

    Subclasses MUST implement `rsample` and `log_prob`
    and expose `latent_names`. They MAY override
    `registry` if they construct their registry lazily, but
    the default implementation expects ``self._registry`` to be
    set in ``__init__``.
    """

    _registry: LatentRegistry

    @classmethod
    def build_registry(
        cls,
        model: MonadicProgram,
        observed_names: set[str] | frozenset[str],
    ) -> LatentRegistry:
        """Convenience wrapper around
        `LatentRegistry.from_model` so guide constructors
        can do ``self._registry = self.build_registry(model, obs)``
        without an extra import."""
        return LatentRegistry.from_model(model, observed_names)

    @property
    def registry(self) -> LatentRegistry:
        """The `LatentRegistry` this guide was built
        against."""
        return self._registry

    @abstractmethod
    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reparameterized sample from :math:`q_\\phi(z \\mid x)`.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``. Used only for
            its batch dim and device; the variational parameters
            are stored on the guide itself.

        Returns
        -------
        dict[str, torch.Tensor]
            Per-site constrained samples shaped to match the
            model's trace-side convention.
        """
        ...

    def rsample_strata(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Stratified reparameterized draws from :math:`q_\\phi(z \\mid x)`.

        Returns log-weights :math:`\\log \\omega_s` and one draw
        :math:`z_s \\sim q_s` per stratum :math:`s` such that the
        strata partition the guide's density,
        :math:`q_\\phi = \\sum_s \\omega_s\\, q_s` with
        :math:`\\sum_s \\omega_s = 1`. Thus
        :math:`\\sum_s \\omega_s f(z_s)` is an unbiased estimate of
        :math:`\\mathbb{E}_{q_\\phi}[f(z)]` for every ``f``, and
        :math:`\\sum_s \\omega_s\\, p(z_s) / q_\\phi(z_s)` is an
        unbiased estimate of :math:`\\int p`. Gradients reach the
        variational parameters through the reparameterized draws and
        through the log-weights.

        A guide whose `rsample` is reparameterized returns a single
        stratum with log-weight zero, which is this default. A guide
        with a discrete choice inside its sampler (such as
        `AutoMixtureGuide`) overrides it to enumerate that choice.

        Parameters
        ----------
        x : torch.Tensor
            Program input, as for `rsample`.

        Returns
        -------
        tuple[torch.Tensor, list[dict[str, torch.Tensor]]]
            The stratum log-weights, shape ``(S,)`` with
            ``logsumexp == 0``, and the ``S`` per-site draws, each
            shaped as `rsample`'s output.
        """
        log_weights = torch.zeros(1, device=x.device, dtype=torch.get_default_dtype())
        return log_weights, [self.rsample(x)]

    @abstractmethod
    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log density of ``sites`` under :math:`q_\\phi(z \\mid x)`,
        with the change-of-variables Jacobian correction baked in.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.
        """
        ...

    @property
    @abstractmethod
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        ...


__all__ = ["Guide"]
