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
* `Guide.rsample_strata` returns weighted draws, each a `GuideDraw`
  carrying its own guide log-density, whose weighted average
  estimates expectations under the guide. The variational objectives
  consume this method. A guide whose draws are not reparameterizable
  as a whole (a finite mixture, whose component choice is discrete)
  enumerates its discrete part exactly, and a guide whose density is
  defined only along the path that produced a draw (`AutoDAIS`)
  returns that path's density with the draw.

Every concrete subclass under [`quivers.inference.guides`][quivers.inference.guides] is
constructed against a `quivers.inference.registry.LatentRegistry`,
with no per-guide model walk. New guides supply the *variational
family's structure* (Normal vs MVN vs flow), not its
introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import didactic.api as dx
import torch
import torch.nn as nn

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.effects.subsample import active_subsets
from quivers.inference.registry import LatentRegistry

# A module reference, resolved at call time: the effects package imports
# this module while `quivers.inference.trace` is still initializing.
import quivers.inference.trace as _trace


class GuideDraw(ValueModel):
    """One weighted draw from a guide, scored by the guide.

    The site tensors, density, and weight compare by identity: two
    draws are equal exactly when they hold the same tensor objects.

    Parameters
    ----------
    sites
        Per-site constrained values, shaped as `Guide.rsample`'s output.
    log_density
        The guide log-density of this draw, shape ``(batch,)``, with
        gradients to the variational parameters through the draw and
        directly. For most guides it is `Guide.log_prob` at ``sites``;
        for `AutoDAIS` it is the extended-space density of the
        trajectory that produced the draw.
    log_weight
        Scalar log-weight :math:`\\log \\omega` of the stratum the draw
        represents; zero for a guide that is not stratified.
    """

    sites: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_density: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_weight: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


class Guide(nn.Module, ABC):
    """Abstract variational guide.

    Subclasses MUST implement `rsample` and `log_prob`
    and expose `latent_names`. They MAY override
    `registry` if they construct their registry lazily, but
    the default implementation expects ``self._registry`` to be
    set in ``__init__``.
    """

    _registry: LatentRegistry

    scores_subsampled_rows: ClassVar[bool] = False
    """Whether `log_prob` scores a latent in a subsampled plate at the
    subset's rows, scaled as the model's sites are. A guide whose density
    does not split over a plate's rows leaves it ``False`` and refuses
    such a latent through `check_subsampling`."""

    def check_subsampling(self) -> None:
        """Refuse a latent in a subsampled plate this guide cannot score by rows.

        Under a [`subsample`][quivers.effects.subsample] block the model
        scores a plate's latents at the subset's rows, scaled; the
        guide's density must match, or the objective is not an unbiased
        estimate of the full one. A guide that scores rows declares
        `scores_subsampled_rows`.

        Raises
        ------
        ValueError
            If a subsampled plate holds one of the guide's latents and
            the guide does not score a plate's rows separately.
        """
        subsets = active_subsets()
        if not subsets or self.scores_subsampled_rows:
            return
        plates = self.registry.model.site_plates
        for name in self.registry.names:
            axes = plates.get(name.split("@")[0], ())
            held = sorted(plate for plate, _ in axes if plate in subsets)
            if held:
                raise ValueError(
                    f"{type(self).__name__} cannot score latent {name!r} in the "
                    f"subsampled plate {held[0]!r} at the subset's rows, since "
                    "its density does not split over the plate; use "
                    "AutoNormalGuide, AmortizedGuide or AutoDeltaGuide for the "
                    "plate's latents, or leave the plate unsubsampled"
                )

    @classmethod
    def build_registry(
        cls,
        model: MonadicProgram,
        observed_names: set[str] | frozenset[str],
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> LatentRegistry:
        """Build the latent registry a guide is constructed against.

        Parameters
        ----------
        model : MonadicProgram
            The model.
        observed_names : set[str] | frozenset[str]
            The names observed, for a registry read off the model's
            steps.
        x : torch.Tensor | None
            A program input. When given, the registry is built from a
            run of the model at ``x`` and ``observations`` with
            `LatentRegistry.from_trace`, so every site the run reaches
            is a latent, every position of a scan included; the run
            draws from a forked stream, leaving Torch's global RNG as it
            was. Required for a program that scans or calls a program
            that draws.
        observations : dict[str, torch.Tensor] | None
            The observed values and host data of that run.

        Returns
        -------
        LatentRegistry
            The registry.
        """
        if x is None:
            return LatentRegistry.from_model(model, observed_names)
        with torch.random.fork_rng(devices=[]):
            recorded = _trace.trace(model, x, observations=dict(observations or {}))
        return LatentRegistry.from_trace(model, recorded)

    @property
    def registry(self) -> LatentRegistry:
        """The `LatentRegistry` this guide was built
        against."""
        return self._registry

    def observe(self, observations: Mapping[str, torch.Tensor]) -> None:
        """Receive the data of the step about to sample the guide.

        Every objective calls this with its step's observations and host
        data before it draws from the guide, so a guide that reads the
        data (an `AmortizedGuide`) samples the posterior of the data the
        step scores. A guide whose parameters do not depend on the data
        ignores them, as this base implementation does.

        Parameters
        ----------
        observations : Mapping[str, torch.Tensor]
            Observed values and host data.
        """
        del observations

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

    def rsample_strata(self, x: torch.Tensor) -> tuple[GuideDraw, ...]:
        """Weighted reparameterized draws from :math:`q_\\phi(z \\mid x)`.

        Returns one `GuideDraw` per stratum :math:`s`, with log-weight
        :math:`\\log \\omega_s` and a draw :math:`z_s \\sim q_s`, such
        that the strata partition the guide's density,
        :math:`q_\\phi = \\sum_s \\omega_s\\, q_s` with
        :math:`\\sum_s \\omega_s = 1`. Thus
        :math:`\\sum_s \\omega_s f(z_s)` is an unbiased estimate of
        :math:`\\mathbb{E}_{q_\\phi}[f(z)]` for every ``f``, and
        :math:`\\sum_s \\omega_s\\, p(z_s) / q_\\phi(z_s)` is an
        unbiased estimate of :math:`\\int p`. Each draw carries the guide
        log-density the objectives pair it with. Gradients reach the
        variational parameters through the reparameterized draws, the
        densities, and the log-weights.

        The default returns a single draw from `rsample` with log-weight
        zero, scored by `log_prob`. A guide with a discrete choice inside
        its sampler (such as `AutoMixtureGuide`) overrides it to
        enumerate that choice; a guide without a density at arbitrary
        points (such as `AutoDAIS`) overrides it to return the density
        of the path that produced the draw.

        Parameters
        ----------
        x : torch.Tensor
            Program input, as for `rsample`.

        Returns
        -------
        tuple[GuideDraw, ...]
            The ``S`` weighted draws; their log-weights have
            ``logsumexp == 0``.
        """
        sites = self.rsample(x)
        log_density = self.log_prob(x, sites)
        return (
            GuideDraw(
                sites=sites,
                log_density=log_density,
                log_weight=torch.zeros((), device=x.device, dtype=log_density.dtype),
            ),
        )

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


__all__ = ["Guide", "GuideDraw"]
