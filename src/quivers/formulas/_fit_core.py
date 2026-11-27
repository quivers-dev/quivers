"""Compilation of formula models for fitting, replay, and prediction."""

from __future__ import annotations

from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import Module
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._brms_priors import with_flat_catalog
from quivers.program import Program


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
