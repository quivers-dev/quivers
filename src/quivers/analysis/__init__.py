"""Algebra-guided training tooling.

This package inspects a compiled QVR program and returns chain metadata,
initialization recipes, and source-keyed diagnostics without rewriting the
program. It accepts either a
[`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] AST or its
runtime [`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram].

The pieces:

* `ChainShape` ([`quivers.analysis.chain_shape`][quivers.analysis.chain_shape]): walks
  a program's let / latent steps and tags each one with its
  source location, governing algebra, and intermediate
  dimensionality. This metadata supplies the initialization and saturation
  analyses.
* `recommend_init` ([`quivers.analysis.init_spec`][quivers.analysis.init_spec]): given
  a program, produces a per-latent `InitSpec` from each
  algebra's initialization recipe. Pair with
  `apply_init_spec` to materialize the initial values onto
  the program's learnable parameters.
* `saturation_warnings`
  ([`quivers.analysis.saturation`][quivers.analysis.saturation]): given a program, returns
  source-keyed warnings about latents that, under the
  recommended init, would saturate the surrounding algebra's
  value range.

See [*Analysis Pipelines: Fitting and Diagnostics*](https://quivers.dev/guides/analysis-fitting-and-diagnostics/)
for the user-facing workflow.
"""

from __future__ import annotations

from quivers.analysis.chain_shape import ChainShape, StepKind, StepShape
from quivers.analysis.init_spec import (
    InitSpec,
    apply_init_spec,
    recommend_init,
)
from quivers.analysis.plate_graph import (
    Edge,
    NodeKind,
    Plate,
    PlateGraph,
    PlateNode,
    build_plate_graph,
)
from quivers.analysis.plate_render import (
    render_daft,
    render_dot,
    render_mermaid,
    render_table,
    render_table_plain,
    render_tikz,
)
from quivers.analysis.saturation import SaturationWarning, saturation_warnings
from quivers.analysis.scope import (
    SCOPE_SEPARATOR,
    TOP_LEVEL_ACCESSORS,
    TOP_LEVEL_KINDS,
    ScopedRef,
    ScopeKind,
    find_all_references,
    resolve_scoped_path,
    scope_children,
    split_path,
    top_level_bindings,
)

__all__ = [
    # chain_shape
    "ChainShape",
    "StepKind",
    "StepShape",
    # init_spec
    "InitSpec",
    "recommend_init",
    "apply_init_spec",
    # plate_graph
    "Edge",
    "NodeKind",
    "Plate",
    "PlateGraph",
    "PlateNode",
    "build_plate_graph",
    # plate_render
    "render_daft",
    "render_dot",
    "render_mermaid",
    "render_table",
    "render_table_plain",
    "render_tikz",
    # saturation
    "SaturationWarning",
    "saturation_warnings",
    # scope
    "SCOPE_SEPARATOR",
    "TOP_LEVEL_ACCESSORS",
    "TOP_LEVEL_KINDS",
    "ScopedRef",
    "ScopeKind",
    "find_all_references",
    "resolve_scoped_path",
    "scope_children",
    "split_path",
    "top_level_bindings",
]
