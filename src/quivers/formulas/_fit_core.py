"""Fitted formula state and compilation shared by all response models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.elpd import NO_CRITERIA, ELPDEstimate
from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import Module
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._brms_priors import derived_site_draws, with_flat_catalog
from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._priors import PriorSet
from quivers.inference._posterior import constrained_draws
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.prior import PriorDraws, PriorMode
from quivers.inference.registry import LatentRegistry
from quivers.program import Program


class FormulaFit(ValueModel):
    """State shared by fitted single-response and joint formula models."""

    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | PriorDraws | Guide = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observations: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sample_prior: PriorMode = "no"
    prior: PriorDraws | MCMCResult | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    losses: tuple[float, ...] = ()
    reparameterize: Literal["centered", "noncentered"] = "noncentered"
    fixed_prior: str = "Normal(0.0, 5.0)"
    random_scale_prior: str = "HalfNormal(1.0)"
    correlation_prior: str = "LKJCholesky(1.0)"
    group_covariances: dict[str, GroupCovariance] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    priors: PriorSet = dx.field(
        default_factory=PriorSet, opaque=True, extras=OPAQUE_BY_VALUE
    )
    criteria: Mapping[str, ELPDEstimate | torch.Tensor] = dx.field(
        default=NO_CRITERIA, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def draws(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Return constrained posterior draws with separate chain and draw axes.

        Parameters
        ----------
        num_samples : int or None
            Maximum draws per MCMC chain, or fresh guide draws. Omission
            keeps all MCMC draws and draws 100 variational samples.
        seed : int
            Seed for variational draws. The caller's CPU RNG is preserved.

        Returns
        -------
        dict[str, torch.Tensor]
            Native site values with shape ``(chain, draw, *site_shape)``.
            Guides produce one pseudo-chain, which supplies no MCMC diagnostics.

        Raises
        ------
        ValueError
            If the count is invalid or posterior sites mismatch the model.
        """
        registry = LatentRegistry.from_model(
            self.program, observed_names=set(self.observations)
        )
        draws = constrained_draws(
            self.posterior,
            torch.zeros(1, 1, dtype=torch.long),
            num_samples=num_samples,
            seed=seed,
            registry=registry,
        )
        bindings = self._derived_draw_bindings(draws)
        if bindings is None:
            return draws
        module, names = bindings
        return derived_site_draws(module, draws, names)

    def _derived_draw_bindings(
        self, draws: Mapping[str, torch.Tensor]
    ) -> tuple[Module, list[str]] | None:
        """Identify reported deterministic bindings of the sampled coordinates."""
        raise NotImplementedError


def compile_model(module: Module, families: FamilyCatalog | None) -> Program:
    """Compile a formula model with the laws used by its emitted sites.

    Parameters
    ----------
    module : Module
        The emitted model, including any missing-value and prior sites.
    families : FamilyCatalog or None
        The custom laws of the model's response blocks and terms.

    Returns
    -------
    Program
        The compiled program with its local family catalogue.
    """
    return Compiler(module, families=with_flat_catalog(families, module)).compile()


__all__: list[str] = []
