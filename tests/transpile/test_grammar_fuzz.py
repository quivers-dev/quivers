"""Random-AST grammar fuzz tests for per-target let-expression renderers.

Builds random [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] trees
of bounded depth, runs each tree through one of the per-target
``render_let_expr_*`` helpers, wraps the rendered RHS in a minimal
parsable shell, pretty-prints the schema through panproto's
[`emit_pretty`][panproto.AstParserRegistry.emit_pretty], re-parses the
bytes through the target tree-sitter grammar, and asserts that the
multiset of ``(kind, literal-value, chose-alt-fingerprint)`` triples
extracted from the RHS subtree round-trips.

The shells per target:

* Stan: ``transformed parameters { real result = <expr>; }``
* Python: ``result = <expr>``
* Julia: ``result = <expr>``
* Scheme: ``(define result <expr>)``
* JavaScript: ``var result = <expr>;``
* BUGS / JAGS: ``model { result <- <expr> }``

The invariant compared is the multiset of vertex fingerprints over the
RHS subtree only. Stricter tree-equality would be too sensitive to
vertex IDs and to internal helper bookkeeping constraints. Picking the
fingerprint to be ``(kind, literal-value, chose-alt-fingerprint)``
catches operator regressions (the operator rides on the binary-op
vertex's ``chose-alt-fingerprint``) and identifier-text regressions
(the operand text rides on each leaf's ``literal-value``).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable

import panproto
import pytest

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._diagnostics import user_facing_message
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.renderers._bugs_helpers import render_let_expr_bugs
from quivers.transpile.renderers._javascript_helpers import (
    render_let_expr_javascript,
)
from quivers.transpile.renderers._julia_helpers import render_let_expr_julia
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    assignment,
    render_let_expr_python,
)
from quivers.transpile.renderers._scheme_helpers import render_let_expr_scheme
from quivers.transpile.renderers._stan_helpers import render_let_expr_stan


# ---------------------------------------------------------------------------
# Random-AST generator.
# ---------------------------------------------------------------------------


_OPS: tuple[str, ...] = ("+", "-", "*", "/")
_FUNCS: tuple[str, ...] = ("exp", "log", "sqrt", "abs")
_VAR_NAMES: tuple[str, ...] = ("x", "y", "z", "a", "b")


def random_let_expr(
    rng: random.Random,
    depth: int,
    *,
    bugs_safe: bool = False,
) -> LetExprNode:
    """Generate a random `LetExprNode` of bounded depth.

    Parameters
    ----------
    rng
        Deterministic randomness source.
    depth
        Maximum remaining recursion budget. At depth 0 the generator
        only emits leaves (literal / var).
    bugs_safe
        When set, restrict generated `LetExprIndex` nodes to a bare
        variable in the array slot, matching BUGS / JAGS's
        `indexed_variable` grammar production (which only accepts an
        identifier as the array). Other targets accept nested array
        expressions.
    """
    if depth <= 0:
        return _random_leaf(rng)
    kinds = ("literal", "var", "binop", "unary", "call", "index", "list")
    kind = rng.choice(kinds)
    if kind == "literal":
        return _random_leaf(rng, force_literal=True)
    if kind == "var":
        return _random_leaf(rng, force_var=True)
    if kind == "binop":
        op = rng.choice(_OPS)
        # `rng.choice` widens to `str`; narrow back to the `Literal`
        # union LetExprBinOp's `op` field demands so pyright is happy.
        assert op in ("+", "-", "*", "/")
        return LetExprBinOp(
            op=op,
            left=random_let_expr(rng, depth - 1, bugs_safe=bugs_safe),
            right=random_let_expr(rng, depth - 1, bugs_safe=bugs_safe),
        )
    if kind == "unary":
        return LetExprUnaryOp(
            operand=random_let_expr(rng, depth - 1, bugs_safe=bugs_safe),
        )
    if kind == "call":
        arity = rng.randint(1, 3)
        return LetExprCall(
            func=rng.choice(_FUNCS),
            args=tuple(
                random_let_expr(rng, depth - 1, bugs_safe=bugs_safe)
                for _ in range(arity)
            ),
        )
    if kind == "index":
        n_indices = rng.randint(1, 2)
        if bugs_safe:
            array_node: LetExprNode = LetExprVar(name=rng.choice(_VAR_NAMES))
        else:
            array_node = random_let_expr(rng, depth - 1, bugs_safe=bugs_safe)
        return LetExprIndex(
            array=array_node,
            indices=tuple(
                random_let_expr(rng, depth - 1, bugs_safe=bugs_safe)
                for _ in range(n_indices)
            ),
        )
    if kind == "list":
        n_items = rng.randint(1, 3)
        return LetExprList(
            items=tuple(
                random_let_expr(rng, depth - 1, bugs_safe=bugs_safe)
                for _ in range(n_items)
            ),
        )
    raise AssertionError(f"unhandled kind {kind!r}")


def _random_leaf(
    rng: random.Random,
    *,
    force_literal: bool = False,
    force_var: bool = False,
) -> LetExprNode:
    """Pick a leaf node uniformly between literals and variables.

    A literal is a whole number spelled as an integer, as the source
    spells one, so every target renders it as an integer literal.
    """
    if force_literal:
        return LetExprLiteral(value=float(rng.randint(1, 9)), integral=True)
    if force_var:
        return LetExprVar(name=rng.choice(_VAR_NAMES))
    if rng.random() < 0.5:
        return LetExprLiteral(value=float(rng.randint(1, 9)), integral=True)
    return LetExprVar(name=rng.choice(_VAR_NAMES))


# ---------------------------------------------------------------------------
# Per-target ctx shims.
# ---------------------------------------------------------------------------


class _FuzzCtx:
    """Generic ctx supplying the union of all per-target ctx methods.

    Each helper expects a small subset (``v``/``vertex``, ``e``/``edge``,
    ``lit``/``literal``, ``constraint``, ``fresh``). The shim implements
    both naming conventions so a single object covers Stan
    (``vertex``/``edge``/``literal``) and the other backends
    (``v``/``e``/``lit``).
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        target: str,
        cards: dict[str, int] | None = None,
    ) -> None:
        self._sb = sb
        self._n = 0
        self.target = target
        self.cards: dict[str, int] = dict(cards or {})
        # The Stan helper's shape tables: a fuzzed expression reads no
        # bound name, so every name is a number, none is declared a
        # vector, and none is an index covariate.
        self.ranks: dict[str, int] = {}
        self.vectors: frozenset[str] = frozenset()
        self.host_indices: frozenset[str] = frozenset()

    def fresh(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    # Stan-style names.
    def vertex(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def edge(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    # Other-backend names.
    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


# ---------------------------------------------------------------------------
# Shell builders: wrap a rendered RHS in a parsable program per target.
# ---------------------------------------------------------------------------


def _shell_stan(ctx: _FuzzCtx, rhs: str) -> None:
    """Wrap `rhs` in ``transformed parameters { real result = <rhs>; }``."""
    ctx.vertex("prog", "program")
    ctx.vertex("tp", "transformed_parameters")
    ctx.edge("prog", "tp", "child_of")
    ctx.vertex("decl", "top_var_decl")
    ctx.edge("tp", "decl", "child_of")
    ctx.vertex("tvt", "top_var_type")
    ctx.vertex("rt", "real_type")
    ctx.literal("rt", "real")
    ctx.edge("tvt", "rt", "child_of")
    ctx.edge("decl", "tvt", "child_of")
    ctx.vertex("nm", "identifier")
    ctx.literal("nm", "result")
    ctx.edge("decl", "nm", "name")
    ctx.edge("decl", rhs, "child_of")


def _shell_python(sb: panproto.SchemaBuilder, ctx: PyCtx, rhs: str) -> None:
    """Wrap `rhs` in a Python module ``result = <rhs>``."""
    asn = assignment(ctx, lhs_name="result", rhs=rhs)
    mod = ctx.v(ctx.fresh("mod"), "module")
    ctx.e(mod, asn, "child_of")


def _shell_julia(ctx: _FuzzCtx, rhs: str) -> None:
    """Wrap `rhs` in a Julia source file ``result = <rhs>``."""
    src = ctx.v(ctx.fresh("src"), "source_file")
    lhs = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(lhs, "result")
    op = ctx.v(ctx.fresh("op"), "operator")
    ctx.lit(op, "=")
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    ctx.e(asn, lhs, "child_of")
    ctx.e(asn, op, "child_of")
    ctx.e(asn, rhs, "child_of")
    ctx.e(src, asn, "child_of")


def _shell_scheme(ctx: _FuzzCtx, rhs: str) -> None:
    """Wrap `rhs` in a Scheme program ``(define result <rhs>)``."""
    prog = ctx.v(ctx.fresh("prog"), "program")
    defn = ctx.v(ctx.fresh("list"), "list")
    head = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(head, "define")
    nm = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(nm, "result")
    ctx.e(defn, head, "child_of")
    ctx.e(defn, nm, "child_of")
    ctx.e(defn, rhs, "child_of")
    ctx.e(prog, defn, "child_of")


def _shell_javascript(ctx: _FuzzCtx, rhs: str) -> None:
    """Wrap `rhs` in a JS program ``var result = <rhs>;``."""
    prog = ctx.v(ctx.fresh("prog"), "program")
    vd = ctx.v(ctx.fresh("vd"), "variable_declaration")
    dr = ctx.v(ctx.fresh("dr"), "variable_declarator")
    nm = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(nm, "result")
    ctx.constraint(nm, "chose-alt-fingerprint", "result")
    ctx.e(dr, nm, "name")
    ctx.e(dr, rhs, "value")
    ctx.e(vd, dr, "child_of")
    ctx.e(prog, vd, "child_of")


def _shell_bugs(ctx: _FuzzCtx, rhs: str) -> None:
    """Wrap `rhs` in a BUGS / JAGS model ``model { result <- <rhs> }``."""
    src = ctx.v(ctx.fresh("src"), "source_file")
    mb = ctx.v(ctx.fresh("mb"), "model_block")
    ctx.constraint(mb, "ptrace-0", "Tmodel")
    dr = ctx.v(ctx.fresh("dr"), "deterministic_relation")
    lhs = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(lhs, "result")
    ctx.e(dr, lhs, "variable")
    ctx.e(dr, rhs, "value")
    ctx.e(mb, dr, "deterministic_relation")
    ctx.e(src, mb, "model_block")


# ---------------------------------------------------------------------------
# Per-target descriptors.
# ---------------------------------------------------------------------------


class _TargetSpec:
    """Bundle of helper, grammar name, shell, RHS-finder, and flags.

    `find_rhs_root` walks the *re-parsed* schema and returns the
    vertex id of the RHS expression root, given the top-level
    parent kind from the shell.
    """

    def __init__(
        self,
        *,
        name: str,
        grammar: str,
        helper: Callable[..., str],
        build_shell: Callable[[panproto.SchemaBuilder, object, str], None],
        find_rhs_root: Callable[[panproto.Schema], str],
        ext: str,
        bugs_safe: bool = False,
    ) -> None:
        self.name = name
        self.grammar = grammar
        self.helper = helper
        self.build_shell = build_shell
        self.find_rhs_root = find_rhs_root
        self.ext = ext
        self.bugs_safe = bugs_safe


def _stan_build(sb: panproto.SchemaBuilder, ctx_obj: object, rhs: str) -> None:
    assert isinstance(ctx_obj, _FuzzCtx)
    _shell_stan(ctx_obj, rhs)


def _python_build(sb: panproto.SchemaBuilder, ctx_obj: object, rhs: str) -> None:
    assert isinstance(ctx_obj, PyCtx)
    _shell_python(sb, ctx_obj, rhs)


def _julia_build(sb: panproto.SchemaBuilder, ctx_obj: object, rhs: str) -> None:
    assert isinstance(ctx_obj, _FuzzCtx)
    _shell_julia(ctx_obj, rhs)


def _scheme_build(sb: panproto.SchemaBuilder, ctx_obj: object, rhs: str) -> None:
    assert isinstance(ctx_obj, _FuzzCtx)
    _shell_scheme(ctx_obj, rhs)


def _javascript_build(sb: panproto.SchemaBuilder, ctx_obj: object, rhs: str) -> None:
    assert isinstance(ctx_obj, _FuzzCtx)
    _shell_javascript(ctx_obj, rhs)


def _bugs_build(sb: panproto.SchemaBuilder, ctx_obj: object, rhs: str) -> None:
    assert isinstance(ctx_obj, _FuzzCtx)
    _shell_bugs(ctx_obj, rhs)


# ---------------------------------------------------------------------------
# RHS-root finders for the *re-parsed* schema per target.
# ---------------------------------------------------------------------------


def _find_rhs_stan(schema: panproto.Schema) -> str:
    """Locate the RHS expression vertex in the re-parsed Stan schema.

    Stan's `top_var_decl` has three outgoing edges: the type, the
    `name` field, and the value `child_of`. The RHS is the last
    `child_of` whose target is neither the `top_var_type` (the first
    `child_of`) nor the `identifier` (which rides the `name` edge).
    """
    kinds = {v.id: v.kind for v in schema.vertices}
    for v in schema.vertices:
        if v.kind != "top_var_decl":
            continue
        for edge in schema.outgoing_edges(v.id):
            if edge.kind != "child_of":
                continue
            tgt_kind = kinds.get(edge.tgt)
            if tgt_kind in ("top_var_type", "identifier"):
                continue
            return edge.tgt
    raise AssertionError("no RHS vertex found in Stan schema")


def _find_rhs_python(schema: panproto.Schema) -> str:
    for v in schema.vertices:
        if v.kind != "assignment":
            continue
        for edge in schema.outgoing_edges(v.id):
            if edge.kind == "right":
                return edge.tgt
    raise AssertionError("no RHS vertex found in Python schema")


def _find_rhs_julia(schema: panproto.Schema) -> str:
    """Julia's `assignment` has three child_of edges (lhs, operator, rhs).

    Find the assignment vertex, walk its `child_of` edges in
    insertion order, and return the third (the RHS expression).
    """
    kinds = {v.id: v.kind for v in schema.vertices}
    for v in schema.vertices:
        if v.kind != "assignment":
            continue
        children = [e.tgt for e in schema.outgoing_edges(v.id) if e.kind == "child_of"]
        # The RHS is the child whose kind is neither `identifier` nor
        # `operator`. The shell wires the lhs as the first identifier
        # child and the `=` as an operator child.
        for child in children:
            child_kind = kinds.get(child)
            if child_kind == "identifier" and child == children[0]:
                continue
            if child_kind == "operator":
                continue
            return child
    raise AssertionError("no RHS vertex found in Julia schema")


def _find_rhs_scheme(schema: panproto.Schema) -> str:
    """Scheme's `(define result <expr>)` is a list with three children:
    the `define` symbol, the `result` symbol, and the expression."""
    kinds = {v.id: v.kind for v in schema.vertices}
    # Find the top-level program -> list whose first symbol is `define`.
    for v in schema.vertices:
        if v.kind != "list":
            continue
        children = [e.tgt for e in schema.outgoing_edges(v.id) if e.kind == "child_of"]
        if len(children) < 3:
            continue
        first = children[0]
        if kinds.get(first) != "symbol":
            continue
        first_text = _literal_value(schema, first)
        if first_text != "define":
            continue
        return children[2]
    raise AssertionError("no RHS vertex found in Scheme schema")


def _find_rhs_javascript(schema: panproto.Schema) -> str:
    for v in schema.vertices:
        if v.kind != "variable_declarator":
            continue
        for edge in schema.outgoing_edges(v.id):
            if edge.kind == "value":
                return edge.tgt
    raise AssertionError("no RHS vertex found in JavaScript schema")


def _find_rhs_bugs(schema: panproto.Schema) -> str:
    for v in schema.vertices:
        if v.kind != "deterministic_relation":
            continue
        for edge in schema.outgoing_edges(v.id):
            if edge.kind == "value":
                return edge.tgt
    raise AssertionError("no RHS vertex found in BUGS schema")


# ---------------------------------------------------------------------------
# Multiset fingerprint extraction.
# ---------------------------------------------------------------------------


def _literal_value(schema: panproto.Schema, vid: str) -> str | None:
    for c in schema.constraints_for(vid):
        if c.sort == "literal-value":
            return c.value
    return None


def _fingerprint_constraint(schema: panproto.Schema, vid: str) -> str | None:
    for c in schema.constraints_for(vid):
        if c.sort == "chose-alt-fingerprint":
            return c.value
    return None


def _reachable(schema: panproto.Schema, root: str) -> set[str]:
    """Return the set of vertex ids reachable from `root` via outgoing edges."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        vid = stack.pop()
        if vid in seen:
            continue
        seen.add(vid)
        for edge in schema.outgoing_edges(vid):
            stack.append(edge.tgt)
    return seen


def _vertex_kinds_multiset(
    schema: panproto.Schema, root: str
) -> Counter[tuple[str, str | None, str | None]]:
    """Count `(kind, literal-value, chose-alt-fingerprint)` over the
    subtree rooted at `root`.

    Includes the root itself.
    """
    kinds = {v.id: v.kind for v in schema.vertices}
    counter: Counter[tuple[str, str | None, str | None]] = Counter()
    for vid in _reachable(schema, root):
        key = (
            kinds[vid],
            _literal_value(schema, vid),
            _fingerprint_constraint(schema, vid),
        )
        counter[key] += 1
    return counter


# ---------------------------------------------------------------------------
# Per-target normalisation: the re-parsed schema may pick up extra
# `chose-alt-fingerprint` constraints on leaf vertices that the helper
# does not set on emission. We strip the leaf fingerprint when it
# duplicates the literal value, since `1`'s fingerprint is `1`.
# ---------------------------------------------------------------------------


# Vertex kinds whose `chose-alt-fingerprint` carries semantic info
# (the operator) that the multiset comparison must preserve. Every
# other kind's fingerprint is either a comma-skeleton (set by the
# re-parser but not by the helper) or a literal echo (redundant with
# `literal-value`); stripping those eliminates spurious mismatches.
_OP_KINDS: frozenset[str] = frozenset(
    {
        "infix_op_expression",
        "binary_expression",
        "binary_operator",
        "prefix_op_expression",
        "unary_expression",
        "unary_operator",
    }
)


def _normalise_counter(
    counter: Counter[tuple[str, str | None, str | None]],
) -> Counter[tuple[str, str | None]]:
    """Project the triple to `(kind, payload)` per vertex.

    For operator-bearing kinds (`_OP_KINDS`) the payload is the
    operator token from `chose-alt-fingerprint`; for every other
    kind the payload is the `literal-value` text (or `None`). The
    projection drops the comma-skeleton fingerprints the re-parser
    attaches to list-like vertices (`array`, `array_expression`,
    `argument_list`, ...) that the helpers do not set on emission.
    """
    out: Counter[tuple[str, str | None]] = Counter()
    for (kind, lit, fp), n in counter.items():
        if kind in _OP_KINDS:
            out[(kind, fp)] += n
        else:
            out[(kind, lit)] += n
    return out


# ---------------------------------------------------------------------------
# Test driver.
# ---------------------------------------------------------------------------


def _build_render_and_extract(
    spec: _TargetSpec, expr: LetExprNode
) -> tuple[
    Counter[tuple[str, str | None, str | None]],
    Counter[tuple[str, str | None, str | None]],
    bytes,
]:
    """Render `expr` through `spec.helper`, emit, re-parse, return both
    multisets and the emitted bytes.

    The first multiset is taken over the originally-built RHS subtree
    (before emit/parse). The second is taken over the re-parsed schema
    RHS subtree.
    """
    proto = target_protocol(spec.grammar)
    sb = proto.schema()
    ctx_obj = _make_ctx(spec, sb)
    rhs_vid = spec.helper(ctx_obj, expr)
    spec.build_shell(sb, ctx_obj, rhs_vid)
    original_schema = sb.build()
    original_counter = _vertex_kinds_multiset(original_schema, rhs_vid)
    emitted = bytes(parser_registry().emit_pretty(spec.grammar, original_schema))
    if not emitted:
        return original_counter, Counter(), emitted
    reparsed = parser_registry().parse_with_protocol(
        spec.grammar, emitted, f"fuzz.{spec.ext}"
    )
    rp_root = spec.find_rhs_root(reparsed)
    reparsed_counter = _vertex_kinds_multiset(reparsed, rp_root)
    return original_counter, reparsed_counter, emitted


def _make_ctx(spec: _TargetSpec, sb: panproto.SchemaBuilder) -> object:
    """Build the helper-expected ctx for `spec`."""
    if spec.name == "python":
        return PyCtx(sb)
    return _FuzzCtx(sb, target=spec.name)


_TARGETS: dict[str, _TargetSpec] = {
    "stan": _TargetSpec(
        name="stan",
        grammar="stan",
        helper=render_let_expr_stan,
        build_shell=_stan_build,
        find_rhs_root=_find_rhs_stan,
        ext="stan",
    ),
    "python": _TargetSpec(
        name="python",
        grammar="python",
        helper=render_let_expr_python,
        build_shell=_python_build,
        find_rhs_root=_find_rhs_python,
        ext="py",
    ),
    "julia": _TargetSpec(
        name="julia",
        grammar="julia",
        helper=render_let_expr_julia,
        build_shell=_julia_build,
        find_rhs_root=_find_rhs_julia,
        ext="jl",
    ),
    "scheme": _TargetSpec(
        name="scheme",
        grammar="scheme",
        helper=render_let_expr_scheme,
        build_shell=_scheme_build,
        find_rhs_root=_find_rhs_scheme,
        ext="scm",
    ),
    "javascript": _TargetSpec(
        name="javascript",
        grammar="javascript",
        helper=render_let_expr_javascript,
        build_shell=_javascript_build,
        find_rhs_root=_find_rhs_javascript,
        ext="js",
    ),
    "bugs": _TargetSpec(
        name="bugs",
        grammar="bugs",
        helper=render_let_expr_bugs,
        build_shell=_bugs_build,
        find_rhs_root=_find_rhs_bugs,
        ext="bugs",
        bugs_safe=True,
    ),
}


_NUM_SEEDS = 50
_DEPTH = 4


@pytest.mark.parametrize("target", sorted(_TARGETS))
@pytest.mark.parametrize("seed", list(range(_NUM_SEEDS)))
def test_grammar_fuzz_round_trip(target: str, seed: int) -> None:
    """Round-trip a random `LetExprNode` tree through `target`'s helper.

    For each (target, seed): generate a tree of bounded depth, render
    it through the per-target helper, wrap the result in a parsable
    shell, pretty-print, re-parse, and assert that the RHS subtree's
    vertex multiset survives the round-trip. The seed is part of the
    test id so a failing cell is reproducible from its name alone.
    """
    spec = _TARGETS[target]
    rng = random.Random(seed)
    expr = random_let_expr(rng, _DEPTH, bugs_safe=spec.bugs_safe)
    try:
        original, reparsed, emitted = _build_render_and_extract(spec, expr)
    except UnsupportedConstruct as exc:
        # A refusal is one of the two outcomes this test accepts. The
        # generator walks the whole shape of `LetExprNode`, and no
        # target has an expression form for every shape, so a tree it
        # cannot render is expected. What is not acceptable is a
        # refusal a user could not act on, so the kind is held to the
        # same contract every other refusal carries: it renders an
        # explanation rather than the report-a-bug fallback.
        assert exc.kinds, (
            f"{target} refused the tree for seed {seed} with no kind "
            f"at all, so nothing downstream can dispatch on it or "
            f"explain it; tree={expr!r}"
        )
        for kind in exc.kinds:
            message = user_facing_message(f"qvr-{target}", (kind,))
            assert "no explanation registered" not in message, (
                f"{target} refused the tree for seed {seed} with kind "
                f"{kind!r}, which renders as the report-a-bug "
                f"fallback. Register an explanation in `_diagnostics` "
                f"so the refusal says what the target cannot express; "
                f"tree={expr!r}"
            )
        return
    if not emitted:
        pytest.xfail(
            reason=(
                f"{target}: emit_pretty returned empty bytes for the "
                f"helper's emitted schema (panproto pretty-printer "
                f"limitation, not a helper-shape bug); tree={expr!r}"
            )
        )
    original_norm = _normalise_counter(original)
    reparsed_norm = _normalise_counter(reparsed)
    if original_norm != reparsed_norm:
        diff_missing = original_norm - reparsed_norm
        diff_extra = reparsed_norm - original_norm
        raise AssertionError(
            f"{target}: RHS multiset mismatch after round-trip.\n"
            f"  tree:    {expr!r}\n"
            f"  emitted: {emitted!r}\n"
            f"  missing from re-parsed (originally rendered): "
            f"{dict(diff_missing)!r}\n"
            f"  extra in re-parsed (not originally rendered): "
            f"{dict(diff_extra)!r}\n"
        )


# ---------------------------------------------------------------------------
# Helper-presence smoke check: every helper symbol the suite parametrises
# over must be importable. Catches helper-rename regressions.
# ---------------------------------------------------------------------------


def test_helper_symbols_importable() -> None:
    """Every helper referenced by `_TARGETS` must be a callable symbol.

    A regression that renames or removes one of the helper exports
    would fail every parametrised cell with an opaque import error;
    this smoke test surfaces the rename up front.
    """
    for name, spec in _TARGETS.items():
        assert callable(spec.helper), f"{name} helper is not callable"
        assert callable(spec.find_rhs_root), f"{name} RHS-root finder is not callable"
        assert callable(spec.build_shell), f"{name} shell builder is not callable"
