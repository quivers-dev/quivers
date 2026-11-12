"""The transpilers' view of step resolution.

Resolution itself lives with the source language in
:mod:`quivers.dsl.step_resolution`; this module re-exports it and turns
its failures into the transpile API's
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct], so a
renderer reports an unresolvable step the way it reports every other
construct it cannot emit.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.dsl.ast_nodes import DrawArg, Expr, Module, MorphismDecl
from quivers.dsl.step_resolution import (
    ResolvedDist,
    StepResolutionError,
    build_let_table,
)
from quivers.dsl.step_resolution import (
    build_morphism_table as _build_morphism_table,
)
from quivers.dsl.step_resolution import (
    param_source_kind as _param_source_kind,
)
from quivers.dsl.step_resolution import (
    resolve_step_dist as _resolve_step_dist,
)
from quivers.transpile._api import UnsupportedConstruct


def resolve_step_dist(
    morphism_name: str,
    raw_args: tuple[DrawArg, ...] | None,
    *,
    morphisms: Mapping[str, MorphismDecl],
    lets: Mapping[str, Expr],
    family_registry: frozenset[str],
    target: str,
) -> ResolvedDist:
    """Resolve a step's morphism slot, reporting failure as unsupported.

    Parameters
    ----------
    morphism_name : str
        The step's morphism slot.
    raw_args : tuple[DrawArg, ...] | None
        The step's own arguments.
    morphisms : Mapping[str, MorphismDecl]
        The module's morphism declarations by name.
    lets : Mapping[str, Expr]
        The module's let aliases by name.
    family_registry : frozenset[str]
        The canonical family names.
    target : str
        The transpile target, for the diagnostic.

    Returns
    -------
    ResolvedDist
        The family and its wire arguments.

    Raises
    ------
    UnsupportedConstruct
        If the slot cannot be resolved.
    """
    try:
        return _resolve_step_dist(
            morphism_name,
            raw_args,
            morphisms=dict(morphisms),
            lets=dict(lets),
            family_registry=family_registry,
            target=target,
        )
    except StepResolutionError as error:
        raise UnsupportedConstruct(error.target, error.explained_kinds) from error


def build_morphism_table(module: Module) -> dict[str, MorphismDecl]:
    """Index a module's morphisms, reporting a consumed network as unsupported.

    Parameters
    ----------
    module : Module
        The parsed module.

    Returns
    -------
    dict[str, MorphismDecl]
        Morphism name to declaration.

    Raises
    ------
    UnsupportedConstruct
        If a step consumes a morphism parameterized by a network.
    """
    try:
        return _build_morphism_table(module)
    except StepResolutionError as error:
        raise UnsupportedConstruct(error.target, error.explained_kinds) from error


def param_source_kind(decl: MorphismDecl, *, target: str) -> str | None:
    """Name a morphism's parameter source, reporting failure as unsupported.

    Parameters
    ----------
    decl : MorphismDecl
        The morphism.
    target : str
        The transpile target, for the diagnostic.

    Returns
    -------
    str | None
        The source kind, or ``None`` for a morphism without one.

    Raises
    ------
    UnsupportedConstruct
        If the declaration's parameter source cannot be read.
    """
    try:
        return _param_source_kind(decl, target=target)
    except StepResolutionError as error:
        raise UnsupportedConstruct(error.target, error.explained_kinds) from error


__all__ = [
    "ResolvedDist",
    "build_let_table",
    "build_morphism_table",
    "param_source_kind",
    "resolve_step_dist",
]
