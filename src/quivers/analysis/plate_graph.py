"""Plate-graph extraction for QVR programs.

A plate diagram in the Bayesian-statistics tradition has three
ingredients:

* **Nodes**: random variables (sample sites, observe sites) and
  deterministic intermediate values (let-bound terms). Observed
  nodes are conventionally shaded; latent nodes are unshaded;
  deterministic nodes are conventionally drawn as smaller dots.
* **Edges**: directed dependencies between nodes, computed from
  the family-argument list of each draw / observe step. ``z ~
  Categorical(probs=theta)`` produces an edge from ``theta`` to ``z``.
* **Plates**: rectangular regions enclosing groups of nodes that
  share a common indexing axis. A plate is induced by an ``index``
  annotation on a draw (``sample theta : Doc <- ...`` puts ``theta``
  on the ``Doc`` plate), by ``iid_over`` on the same step, by the
  ``over`` axis on a marginalize block (which scopes every inner
  observation), or by the grouping index map ``[via=...]`` on an
  observe step.

The extractor in this module walks a compiled
:class:`quivers.dsl.compiler.Compiler`'s program declarations and
builds an immutable :class:`PlateGraph` describing exactly this
structure. Renderers (in-TUI table view, Mermaid output, Daft /
TikZ / DOT code generation) consume the graph; the graph itself is
the single source of truth and never embeds layout decisions.

The walker handles arbitrarily nested ``marginalize`` blocks by
treating each block as a new plate scope that lives *inside* the
enclosing plates. ``[via=idx]`` on an observe inside a
marginalize-over-G block puts the observe on G's plate in addition
to its own ``index`` plate, matching the standard convention for
grouped observations in pgm notation.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    LetStep,
    MarginalizeStep,
    ObserveStep,
    ProgramDecl,
    ReturnStep,
    SampleStep,
)

NodeKind = Literal["latent", "observed", "marginalized", "deterministic"]
"""The kind tag of a `PlateNode`, which selects how a renderer draws it."""


class Edge(dx.Model):
    """One directed dependency edge in the plate diagram."""

    src: str
    dst: str


class Plate(dx.Model):
    """One plate (indexing axis) of the diagram.

    Attributes
    ----------
    name : str
        The plate object's name (``Doc``, ``Topic``, ``Word``).
        Multiple plates can share a name when the same axis is
        indexed at multiple levels in the program; the extractor
        de-duplicates them.
    cardinality : int | None
        The plate's size if statically inferable from the
        module's ``object NAME : FinSet N`` declarations; ``None``
        otherwise.
    parent : str | None
        Name of an enclosing plate, or ``None`` when this plate
        is at the program's outermost level. Set when a
        marginalize block nests inside another marginalize.
    """

    name: str
    cardinality: int | None = None
    parent: str | None = None


class PlateNode(dx.Model):
    """One random variable / deterministic intermediate.

    Attributes
    ----------
    name : str
        The variable's bound name in the program (``theta``, ``z``,
        ``w``, ``mu``).
    kind : NodeKind
        ``"latent"`` for unobserved samples, ``"observed"`` for
        observe sites (clamped at runtime), ``"marginalized"`` for
        sample sites that are integrated out by an enclosing
        ``marginalize`` block, ``"deterministic"`` for ``let``
        bindings whose value is a closed-form function of upstream
        nodes (no stochastic draw).
    family : str | None
        The distribution family name (``"Normal"``, ``"Dirichlet"``,
        ``"Categorical"``) for stochastic nodes. ``None`` for
        ``let`` nodes.
    plates : tuple[str, ...]
        The plates this node sits inside, ordered from outermost
        to innermost. Empty tuple for scalar nodes.
    args : tuple[str, ...]
        The variable names this node's family / let-expression
        references. The graph's edges are derived from these.
    scope_path : str
        The ``::``-separated scope path to this binding (e.g.
        ``lda::theta``, ``lda::z::w``). Used by renderers that
        link plate-diagram clicks back to ``:info`` lookups.
    """

    name: str
    kind: NodeKind
    family: str | None = None
    plates: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    scope_path: str = ""


class PlateGraph(dx.Model):
    """Structural plate diagram of a single QVR program."""

    program_name: str
    domain: str = ""
    codomain: str = ""
    nodes: tuple[PlateNode, ...] = ()
    plates: tuple[Plate, ...] = ()
    edges: tuple[Edge, ...] = ()

    @property
    def latents(self) -> tuple[PlateNode, ...]:
        return tuple(n for n in self.nodes if n.kind == "latent")

    @property
    def observed(self) -> tuple[PlateNode, ...]:
        return tuple(n for n in self.nodes if n.kind == "observed")

    @property
    def marginalized(self) -> tuple[PlateNode, ...]:
        return tuple(n for n in self.nodes if n.kind == "marginalized")

    @property
    def deterministic(self) -> tuple[PlateNode, ...]:
        return tuple(n for n in self.nodes if n.kind == "deterministic")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def _index_name(index_expr: object) -> str | None:
    """Return the bare name of an index expression (``Doc``) or
    ``None`` for unindexed / non-name shapes."""
    if index_expr is None:
        return None
    return getattr(index_expr, "name", None)


def _option_name_value(opt: object) -> str | None:
    """Read a ``[k = NAME]`` option entry's value as a bare name."""
    value = getattr(opt, "value", None)
    if value is None:
        return None
    inner = getattr(value, "value", None)
    if isinstance(inner, str):
        return inner
    if isinstance(value, str):
        return value
    return None


def _options_dict(step: object) -> dict[str, str | tuple[str, ...]]:
    """Return ``{key: value}`` for a step's option block.

    Values that resolve to a single bare name come back as a
    string; ``[over=[A, B]]``-style list options come back as a
    tuple of names.
    """
    out: dict[str, str | tuple[str, ...]] = {}
    for opt in getattr(step, "options", None) or ():
        key = getattr(opt, "key", None)
        if not isinstance(key, str):
            continue
        value = getattr(opt, "value", None)
        # Bare name
        bare = _option_name_value(opt)
        if bare is not None:
            out[key] = bare
            continue
        # Option-list shape: ``OptionList`` of OptionName entries
        items = getattr(value, "items", None)
        if items is not None:
            names: list[str] = []
            for item in items:
                inner = getattr(item, "value", None)
                if isinstance(inner, str):
                    names.append(inner)
            if names:
                out[key] = tuple(names)
                continue
    return out


def _plates_for_step(
    step: object,
    parent_plates: tuple[str, ...],
) -> tuple[str, ...]:
    """Compute the plate stack a step's binding lives on.

    Combines, in order:

    1. The step's enclosing plate stack (from the surrounding
       marginalize / program scope).
    2. The step's own ``index`` annotation (``sample v : T <-
       ...`` puts ``v`` on the ``T`` plate).
    3. The step's ``iid_over`` axes (each adds a plate).

    The ``via`` option on an observe is intentionally *not* a
    plate: it names an index map variable that routes
    observations into the enclosing marginalize's grouping plate,
    which the walker has already added to ``parent_plates``.

    De-duplicates while preserving order so a nested plate that
    inherits from its parent doesn't double-list it.
    """
    plates: list[str] = list(parent_plates)
    own_index = _index_name(getattr(step, "index", None))
    if own_index is not None and own_index not in plates:
        plates.append(own_index)
    axes = getattr(step, "axes", None)
    if axes is not None:
        for ax in getattr(axes, "iid_over", ()) or ():
            if ax not in plates:
                plates.append(ax)
    opts = _options_dict(step)
    iid = opts.get("iid_over")
    if isinstance(iid, str) and iid not in plates:
        plates.append(iid)
    elif isinstance(iid, tuple):
        for ax in iid:
            if ax not in plates:
                plates.append(ax)
    return tuple(plates)


def _child_plates_for_marginalize(
    step: object,
    parent_plates: tuple[str, ...],
) -> tuple[str, ...]:
    """Plate stack inherited by a marginalize block's body.

    A ``marginalize z : T <- F [over=G]`` block's children inherit
    their enclosing plates *plus* the ``over`` axis (the grouping
    plate the marginalized observations live on). The marginalize
    variable's own index ``T`` is *not* inherited because the
    variable is integrated out: inner observations don't lie on
    the marginalized axis. Without ``[over=...]``, no new plate is
    added and children inherit only ``parent_plates``.
    """
    plates: list[str] = list(parent_plates)
    opts = _options_dict(step)
    over = opts.get("over") or getattr(step, "over", None)
    if isinstance(over, str) and over not in plates:
        plates.append(over)
    elif isinstance(over, tuple):
        for ax in over:
            if ax not in plates:
                plates.append(ax)
    return tuple(plates)


def _step_args(step: object) -> tuple[str, ...]:
    """Variable references a step depends on. Pattern-matches each
    `DrawArg` tagged variant to its structured content: a
    `DrawArgName` contributes one identifier; a `DrawArgIndex`
    contributes its base name plus every index identifier. Numeric
    literals (`DrawArgScalar`), nested distribution calls
    (`DrawArgDist`), and list literals (`DrawArgList`) contribute
    no plate-graph edges. Legacy bare-string args from
    compiler-synthesized steps pass through unchanged.
    """
    args = getattr(step, "args", None)
    if not args:
        return ()
    out: list[str] = []

    def collect(a: DrawArg | str) -> None:
        if isinstance(a, DrawArgNamed):
            collect(a.value)
        elif isinstance(a, DrawArgDist):
            for nested in a.args:
                collect(nested)
        elif isinstance(a, DrawArgList):
            for nested in a.items:
                collect(nested)
        if isinstance(a, DrawArgIndex):
            if a.name:
                out.append(a.name)
            out.extend(a.indices)
        elif isinstance(a, DrawArgName):
            out.append(a.text)
        elif isinstance(a, str):
            out.append(a)

    for a in args:
        collect(a)
    return tuple(out)


def _let_value_args(value: object) -> tuple[str, ...]:
    """Walk a let-expression's value AST and collect every
    referenced bare identifier. Best-effort; handles the common
    shapes (Ident / BinOp / Call) without depending on the full
    let-expr AST hierarchy."""
    found: list[str] = []

    def _walk(node: object) -> None:
        if node is None:
            return
        # Common ident-bearing fields
        for attr in ("name", "ident", "var"):
            v = getattr(node, attr, None)
            if isinstance(v, str) and v.isidentifier():
                found.append(v)
        for attr in ("left", "right", "lhs", "rhs", "operand", "expr", "body"):
            sub = getattr(node, attr, None)
            if sub is not None:
                _walk(sub)
        for attr in ("args", "items", "operands", "children"):
            sub = getattr(node, attr, None)
            if isinstance(sub, (tuple, list)):
                for x in sub:
                    _walk(x)

    _walk(value)
    return tuple(dict.fromkeys(found))


def _cardinality_lookup(
    compiler,
    plate_name: str,
) -> int | None:  # type: ignore[no-untyped-def]
    obj = (getattr(compiler, "objects", None) or {}).get(plate_name)
    if obj is None:
        return None
    return getattr(obj, "cardinality", None)


def build_plate_graph(  # type: ignore[no-untyped-def]
    compiler,
    program_name: str,
) -> PlateGraph | None:
    """Build the :class:`PlateGraph` for ``program_name``.

    Returns ``None`` if ``program_name`` isn't a registered program
    on ``compiler``. Walks the program's ``draws`` and any nested
    ``marginalize`` body recursively. Marginalized sample sites
    (the variable bound by the marginalize header) get
    ``kind="marginalized"`` so renderers can shade them
    differently from a plain ``"latent"``.

    Parameters
    ----------
    compiler : Compiler
        A `quivers.dsl.Compiler` that has compiled the module holding
        the program.
    program_name : str
        The name of the program to diagram.

    Returns
    -------
    PlateGraph or None
        The program's plate graph, or ``None`` when no program of that
        name is registered.
    """
    decl = compiler.programs.get(program_name)
    if not isinstance(decl, ProgramDecl):
        return None

    nodes: list[PlateNode] = []
    plate_names: dict[str, str | None] = {}  # plate -> parent plate
    edges: list[Edge] = []

    def _walk(  # type: ignore[no-untyped-def]
        steps: tuple[object, ...],
        parent_plates: tuple[str, ...],
        path_prefix: str,
    ) -> None:
        for step in steps:
            if isinstance(step, SampleStep):
                name = step.vars[0] if step.vars else "?"
                plates = _plates_for_step(step, parent_plates)
                for p in plates:
                    if p not in plate_names:
                        plate_names[p] = parent_plates[-1] if parent_plates else None
                args = _step_args(step)
                nodes.append(
                    PlateNode(
                        name=name,
                        kind="latent",
                        family=step.morphism,
                        plates=plates,
                        args=args,
                        scope_path=f"{path_prefix}::{name}",
                    )
                )
                for a in args:
                    edges.append(Edge(src=a, dst=name))
            elif isinstance(step, ObserveStep):
                name = step.vars[0]
                plates = _plates_for_step(step, parent_plates)
                for p in plates:
                    if p not in plate_names:
                        plate_names[p] = parent_plates[-1] if parent_plates else None
                args = _step_args(step)
                nodes.append(
                    PlateNode(
                        name=name,
                        kind="observed",
                        family=step.morphism,
                        plates=plates,
                        args=args,
                        scope_path=f"{path_prefix}::{name}",
                    )
                )
                for a in args:
                    edges.append(Edge(src=a, dst=name))
            elif isinstance(step, MarginalizeStep):
                name = step.var
                plates = _plates_for_step(step, parent_plates)
                for p in plates:
                    if p not in plate_names:
                        plate_names[p] = parent_plates[-1] if parent_plates else None
                args = _step_args(step)
                nodes.append(
                    PlateNode(
                        name=name,
                        kind="marginalized",
                        family=step.morphism,
                        plates=plates,
                        args=args,
                        scope_path=f"{path_prefix}::{name}",
                    )
                )
                for a in args:
                    edges.append(Edge(src=a, dst=name))
                # Recurse into the marginalize body. Inner plates
                # inherit ``parent_plates + [over]`` rather than
                # the marginalize variable's own index plate,
                # because the marginalized variable is integrated
                # out and inner observations don't lie on its axis.
                inner_prefix = f"{path_prefix}::{name}"
                inner_plates = _child_plates_for_marginalize(step, parent_plates)
                _walk(step.scope, inner_plates, inner_prefix)
            elif isinstance(step, LetStep):
                name = step.name
                args = _let_value_args(step.value)
                nodes.append(
                    PlateNode(
                        name=name,
                        kind="deterministic",
                        family=None,
                        plates=parent_plates,
                        args=args,
                        scope_path=f"{path_prefix}::{name}",
                    )
                )
                for a in args:
                    edges.append(Edge(src=a, dst=name))
            elif isinstance(step, ReturnStep):
                # Return doesn't introduce a node; downstream
                # consumers see the return vars via the program
                # decl.
                continue

    _walk(decl.draws, (), program_name)

    plates_out: list[Plate] = []
    for p, parent in plate_names.items():
        plates_out.append(
            Plate(
                name=p,
                cardinality=_cardinality_lookup(compiler, p),
                parent=parent,
            )
        )

    domain = getattr(decl.domain, "name", "") if decl.domain is not None else ""
    codomain = getattr(decl.codomain, "name", "") if decl.codomain is not None else ""
    return PlateGraph(
        program_name=program_name,
        domain=domain,
        codomain=codomain,
        nodes=tuple(nodes),
        plates=tuple(plates_out),
        edges=tuple(edges),
    )


__all__ = [
    "Edge",
    "NodeKind",
    "Plate",
    "PlateGraph",
    "PlateNode",
    "build_plate_graph",
]
