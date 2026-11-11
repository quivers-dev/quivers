"""Render [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] subtrees
to JavaScript / WebPPL tree-sitter schema vertices.

The JavaScript grammar exposes the following expression-level vertex
kinds the helper builds:

* `number` for numeric leaves
* `identifier` for variable references
* `string` wrapping a `string_fragment` for string literals
* `binary_expression` with per-operator alt discriminated by
  ``field:operator`` plus a matching ``chose-alt-fingerprint``
* `unary_expression` for the unary minus
* `call_expression` with `function` and `arguments` field edges
* `subscript_expression` with `object` and `index` field edges
* `array` for list literals (alt fingerprint encodes ``[ , , ]``)
* `function_expression` wrapping `formal_parameters` and
  `statement_block` for lambdas
* `member_expression` wrapping the receiver and a
  `property_identifier` for method calls

Every vertex sets ``chose-alt-child-kinds`` to the space-separated
sequence of its children's grammar kinds; the pretty-printer uses this
to disambiguate grammar productions and silently drops vertices whose
constraint is missing or stale. The recursive worker returns
``(vertex_id, vertex_kind)`` so parents can populate
``chose-alt-child-kinds`` from real child kinds.

[`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] is unrolled at
render time: the cases form becomes an `array` whose children are the
case bodies in label order; the uniform-body multi-binder form becomes
nested `array` vertices populated by substituting each binder through
its axis's static cardinality (looked up via ``ctx.cards``, populated
from [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]). The
shared `_substitute_let_expr`
walk from `_stan_helpers` performs the capture-avoiding substitution.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.ast_nodes.let_expressions import LetFactorBinder
from quivers.dsl.ast_nodes.objects import TypeName
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._builtin_spellings import native_path
from quivers.transpile.ir import LetExprAffineMap
from quivers.transpile.renderers._stan_helpers import _substitute_let_expr


@runtime_checkable
class _JsLetCtx(Protocol):
    """Structural protocol for the helper's ctx parameter.

    Mirrors the carrier shape used by
    `_bugs_helpers` and
    `_stan_helpers`:
    bound methods for vertex / edge / literal / constraint emission,
    a fresh-id generator, and the static-cardinality table used when
    unrolling [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor].
    """

    target: str
    cards: dict[str, int]

    def fresh(self, prefix: str) -> str: ...
    def v(self, vid: str, kind: str) -> str: ...
    def e(self, src: str, tgt: str, kind: str) -> None: ...
    def lit(self, vid: str, text: str) -> None: ...
    def constraint(self, vid: str, sort: str, value: str) -> None: ...


def render_let_expr_javascript(ctx: _JsLetCtx, expr: LetExprNode) -> str:
    """Build a JavaScript / WebPPL expression schema for ``expr`` in
    ``ctx``.

    Returns the root vertex id. Wraps the recursive worker
    `_render`
    so external callers see the same signature as the other
    per-target helpers.
    """
    vid, _kind = _render(ctx, expr)
    return vid


def _render(ctx: _JsLetCtx, expr: LetExprNode) -> tuple[str, str]:
    """Recursive renderer returning ``(vertex_id, vertex_kind)``.

    Parents read the kind to populate their own
    ``chose-alt-child-kinds`` constraint with the real child kinds.
    """
    if isinstance(expr, LetExprLiteral):
        return _emit_number(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_identifier(ctx, expr.name)
    if isinstance(expr, LetExprString):
        return _emit_string(ctx, expr.value)
    if isinstance(expr, LetExprBinOp):
        return _emit_binop(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_unary(ctx, expr)
    if isinstance(expr, LetExprCall):
        # A pointwise math builtin is called by the runtime global
        # `native_path` names.
        path = native_path(ctx.target, expr.func, len(expr.args))
        return _emit_call(
            ctx,
            expr.func if path is None else ".".join(path),
            tuple(_render(ctx, a) for a in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return _emit_index(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_array(ctx, tuple(_render(ctx, item) for item in expr.items))
    if isinstance(expr, LetExprLambda):
        return _emit_lambda(ctx, expr)
    if isinstance(expr, LetExprMethodCall):
        return _emit_method_call(ctx, expr)
    if isinstance(expr, LetExprAffineMap):
        return _emit_affine_map(ctx, expr)
    if isinstance(expr, LetExprFactor):
        return _emit_factor(ctx, expr)
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


# ---------------------------------------------------------------------------
# Per-kind emitters.
# ---------------------------------------------------------------------------


def _emit_number(ctx: _JsLetCtx, value: float) -> tuple[str, str]:
    """Emit a `number` vertex with the textual rendering of ``value``."""
    vid = ctx.v(ctx.fresh("num"), "number")
    text = str(int(value)) if float(value).is_integer() else repr(float(value))
    ctx.lit(vid, text)
    ctx.constraint(vid, "chose-alt-fingerprint", text)
    return vid, "number"


def _emit_identifier(ctx: _JsLetCtx, name: str) -> tuple[str, str]:
    """Emit a bare `identifier` vertex carrying ``name``."""
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(vid, name)
    ctx.constraint(vid, "chose-alt-fingerprint", name)
    return vid, "identifier"


def _emit_string(ctx: _JsLetCtx, value: str) -> tuple[str, str]:
    """Emit a `string` vertex wrapping a `string_fragment` child.

    The JavaScript grammar's `string` production has a CHOICE alt
    between single-quoted and double-quoted forms; the helper picks
    the double-quoted alt via the ``" "`` fingerprint, matching the
    shape parse trees produce for `"text"` literals.
    """
    vid = ctx.v(ctx.fresh("str"), "string")
    ctx.constraint(vid, "chose-alt-fingerprint", '" "')
    ctx.constraint(vid, "chose-alt-child-kinds", "string_fragment")
    frag = ctx.v(ctx.fresh("sfrag"), "string_fragment")
    ctx.lit(frag, value)
    ctx.constraint(frag, "chose-alt-fingerprint", value)
    ctx.e(vid, frag, "child_of")
    return vid, "string"


_JS_PAREN_REQUIRED_OPERAND_KINDS: frozenset[str] = frozenset(
    {
        "binary_expression",
        "unary_expression",
    }
)
"""Operand kinds wrapped in `parenthesized_expression` when they appear
under a binary or unary operator. WebPPL's pretty printer emits children
in source order without re-grouping, so an unparenthesised
`binary_expression` operand reassociates: ``(a + b) * c`` would print as
``a + b * c``."""


def _maybe_paren(ctx: _JsLetCtx, rendered: tuple[str, str]) -> tuple[str, str]:
    """Wrap `rendered` in a `parenthesized_expression` when its vertex
    kind is in
    `_JS_PAREN_REQUIRED_OPERAND_KINDS`."""
    _vid, kind = rendered
    if kind not in _JS_PAREN_REQUIRED_OPERAND_KINDS:
        return rendered
    return _emit_paren(ctx, rendered)


def _emit_paren(ctx: _JsLetCtx, rendered: tuple[str, str]) -> tuple[str, str]:
    """Wrap `rendered` in a `parenthesized_expression` vertex."""
    vid, kind = rendered
    paren = ctx.v(ctx.fresh("paren"), "parenthesized_expression")
    ctx.constraint(paren, "chose-alt-fingerprint", "( )")
    ctx.constraint(paren, "chose-alt-child-kinds", kind)
    ctx.e(paren, vid, "child_of")
    return paren, "parenthesized_expression"


def _emit_binop(ctx: _JsLetCtx, expr: LetExprBinOp) -> tuple[str, str]:
    """Emit a `binary_expression` with `left` / `right` field edges.

    JavaScript's `binary_expression` discriminates the operator via
    the grammar's CHOICE alternatives; the panproto walker picks the
    alt from the ``field:operator`` + ``chose-alt-fingerprint`` pair.
    """
    left_vid, left_kind = _maybe_paren(ctx, _render(ctx, expr.left))
    right_vid, right_kind = _maybe_paren(ctx, _render(ctx, expr.right))
    vid = ctx.v(ctx.fresh("bin"), "binary_expression")
    ctx.constraint(vid, "field:operator", expr.op)
    ctx.constraint(vid, "chose-alt-fingerprint", expr.op)
    ctx.constraint(vid, "chose-alt-child-kinds", f"{left_kind} {right_kind}")
    ctx.e(vid, left_vid, "left")
    ctx.e(vid, right_vid, "right")
    return vid, "binary_expression"


def _emit_unary(ctx: _JsLetCtx, expr: LetExprUnaryOp) -> tuple[str, str]:
    """Emit a unary-minus `unary_expression` carrying its operand on
    the `argument` field edge."""
    operand_vid, operand_kind = _maybe_paren(ctx, _render(ctx, expr.operand))
    vid = ctx.v(ctx.fresh("uop"), "unary_expression")
    ctx.constraint(vid, "field:operator", "-")
    ctx.constraint(vid, "chose-alt-fingerprint", "-")
    ctx.constraint(vid, "chose-alt-child-kinds", operand_kind)
    ctx.e(vid, operand_vid, "argument")
    return vid, "unary_expression"


def _emit_call(
    ctx: _JsLetCtx,
    func: str,
    rendered: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    """Emit ``<func>(<arg_0>, <arg_1>, ...)`` as a `call_expression`.

    The `function` edge points at an `identifier` callee; the
    `arguments` edge points at an `arguments` vertex whose
    fingerprint encodes the comma count (``( )`` for one arg,
    ``( , )`` for two, ``( , , )`` for three, ...).
    """
    fn_vid, fn_kind = _emit_identifier(ctx, func)
    args_vid, args_kind = _emit_arguments(ctx, rendered)
    vid = ctx.v(ctx.fresh("call"), "call_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", f"{fn_kind} {args_kind}")
    ctx.e(vid, fn_vid, "function")
    ctx.e(vid, args_vid, "arguments")
    return vid, "call_expression"


def _emit_arguments(
    ctx: _JsLetCtx, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit an `arguments` vertex with the right comma fingerprint."""
    vid = ctx.v(ctx.fresh("args"), "arguments")
    if not rendered:
        ctx.lit(vid, "()")
        ctx.constraint(vid, "chose-alt-fingerprint", "()")
        return vid, "arguments"
    fingerprint = _paren_fingerprint(len(rendered))
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.e(vid, child_vid, "child_of")
    return vid, "arguments"


def _emit_affine_map(ctx: _JsLetCtx, expr: LetExprAffineMap) -> tuple[str, str]:
    """Emit one head's row block of ``W x + b`` as a call to the
    runtime's `_qvr_affine`.

    WebPPL has neither a matrix product nor a numeric array type, and
    its CPS transform rejects the loops a contraction would otherwise
    need at the call site, so the contraction lives in
    [`runtime_webppl.js`][quivers.transpile.runtime_webppl] and the
    emit passes it the weight, the bias, the ordered factors of the
    conditioning row, the head's row block, and its elementwise link.
    Every index is zero-based, which is JavaScript's origin as well as
    QVR's, so nothing is rebased.
    """
    return _emit_call(
        ctx,
        "_qvr_affine",
        (
            _render(ctx, expr.weight),
            _render(ctx, expr.bias),
            _emit_array(
                ctx,
                tuple(_render(ctx, s.value) for s in expr.sources),
            ),
            _emit_number(ctx, float(expr.row_offset)),
            _emit_number(ctx, float(expr.rows)),
            _emit_string(ctx, expr.transform),
        ),
    )


def _emit_index(ctx: _JsLetCtx, expr: LetExprIndex) -> tuple[str, str]:
    """Emit a `subscript_expression` (``arr[i][j]...`` form).

    Each level of subscripting is its own `subscript_expression`
    vertex; multi-index access nests them right-to-left over the
    index list so the printed form matches the surface syntax.
    """
    current_vid, current_kind = _render(ctx, expr.array)
    for idx in expr.indices:
        idx_vid, idx_kind = _render(ctx, idx)
        vid = ctx.v(ctx.fresh("sub"), "subscript_expression")
        ctx.constraint(
            vid,
            "chose-alt-child-kinds",
            f"{current_kind} {idx_kind}",
        )
        ctx.e(vid, current_vid, "object")
        ctx.e(vid, idx_vid, "index")
        current_vid = vid
        current_kind = "subscript_expression"
    return current_vid, current_kind


def _emit_array(
    ctx: _JsLetCtx, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit an `array` literal ``[e0, e1, ...]``.

    The grammar's CHOICE production picks the alt from the bracket
    fingerprint: ``[]`` for empty, ``[ ]`` for one element, then
    ``[ , ]`` / ``[ , , ]`` / ... for two-or-more (N-1 commas for N
    elements).
    """
    vid = ctx.v(ctx.fresh("arr"), "array")
    if not rendered:
        ctx.lit(vid, "[]")
        ctx.constraint(vid, "chose-alt-fingerprint", "[]")
        return vid, "array"
    fingerprint = _bracket_fingerprint(len(rendered))
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.e(vid, child_vid, "child_of")
    return vid, "array"


def _emit_lambda(ctx: _JsLetCtx, expr: LetExprLambda) -> tuple[str, str]:
    """Emit a `function_expression` ``function(<param>){return <body>;}``.

    WebPPL has no arrow-function or unnamed-function shorthand in
    its parsed subset; the helper renders every
    [`LetExprLambda`][quivers.dsl.ast_nodes.LetExprLambda] as the
    canonical ``function(...){return ...;}`` form so the result is
    valid input to both the WebPPL interpreter and the tree-sitter
    JavaScript grammar.
    """
    # Formal parameters: `( <param> )`.
    params_vid = ctx.v(ctx.fresh("ps"), "formal_parameters")
    ctx.constraint(params_vid, "chose-alt-fingerprint", "( )")
    ctx.constraint(params_vid, "chose-alt-child-kinds", "identifier")
    pid_vid, _pid_kind = _emit_identifier(ctx, expr.param)
    ctx.e(params_vid, pid_vid, "child_of")
    # Body: `{ return <expr>; }`.
    body_vid, body_kind = _render(ctx, expr.body)
    ret_vid = ctx.v(ctx.fresh("ret"), "return_statement")
    ctx.constraint(ret_vid, "chose-alt-fingerprint", "return ;")
    ctx.constraint(ret_vid, "chose-alt-child-kinds", body_kind)
    ctx.e(ret_vid, body_vid, "child_of")
    block_vid = ctx.v(ctx.fresh("blk"), "statement_block")
    ctx.constraint(block_vid, "chose-alt-fingerprint", "{ }")
    ctx.constraint(block_vid, "chose-alt-child-kinds", "return_statement")
    ctx.e(block_vid, ret_vid, "child_of")
    # Function expression: `function( <params> ) <block>`.
    fn_vid = ctx.v(ctx.fresh("fn"), "function_expression")
    ctx.constraint(fn_vid, "chose-alt-fingerprint", "function")
    ctx.constraint(
        fn_vid,
        "chose-alt-child-kinds",
        "formal_parameters statement_block",
    )
    ctx.e(fn_vid, params_vid, "parameters")
    ctx.e(fn_vid, block_vid, "body")
    return fn_vid, "function_expression"


def _emit_method_call(ctx: _JsLetCtx, expr: LetExprMethodCall) -> tuple[str, str]:
    """Emit a `call_expression` whose callee is a `member_expression`.

    Shape: ``<receiver>.<method>(<args>)`` -> `call_expression` with
    a `member_expression` callee carrying the receiver and a
    `property_identifier` for the method name.
    """
    receiver_vid, receiver_kind = _render(ctx, expr.receiver)
    prop_vid = ctx.v(ctx.fresh("pid"), "property_identifier")
    ctx.lit(prop_vid, expr.method)
    ctx.constraint(prop_vid, "chose-alt-fingerprint", expr.method)
    member_vid = ctx.v(ctx.fresh("memb"), "member_expression")
    ctx.constraint(member_vid, "chose-alt-fingerprint", ".")
    ctx.constraint(
        member_vid,
        "chose-alt-child-kinds",
        f"{receiver_kind} property_identifier",
    )
    ctx.e(member_vid, receiver_vid, "object")
    ctx.e(member_vid, prop_vid, "property")
    rendered = tuple(_render(ctx, a) for a in expr.args)
    args_vid, args_kind = _emit_arguments(ctx, rendered)
    call_vid = ctx.v(ctx.fresh("mcall"), "call_expression")
    ctx.constraint(
        call_vid,
        "chose-alt-child-kinds",
        f"member_expression {args_kind}",
    )
    ctx.e(call_vid, member_vid, "function")
    ctx.e(call_vid, args_vid, "arguments")
    return call_vid, "call_expression"


# ---------------------------------------------------------------------------
# Factor unrolling.
# ---------------------------------------------------------------------------


def _emit_factor(ctx: _JsLetCtx, expr: LetExprFactor) -> tuple[str, str]:
    """Unroll a [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    into nested `array` literals.

    The cases form (binders contain a single axis, body is None,
    cases enumerate labels in ``[0, |axis|)``) emits an `array`
    whose children are each case's body in label order. The
    uniform-body form (one or more binders, body is the repeated
    expression, cases is empty) emits a tower of `array` vertices of
    shape ``(|b0|, |b1|, ..., |bn-1|)``, with each binder
    substituted for its 0-indexed integer value (JavaScript arrays
    are 0-based, matching QVR's surface convention). The shared
    `_substitute_let_expr`
    walk takes the same value for both `index_value` and
    `scalar_value` because no index-base shift is needed here.
    """
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprFactor: cases form requires "
                    f"exactly one binder; got {len(expr.binders)}"
                ],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        rendered = tuple(_render(ctx, c.value) for c in ordered)
        return _emit_array(ctx, rendered)
    if expr.body is not None and not expr.cases:
        sizes = tuple(_card_for(ctx, b) for b in expr.binders)
        return _build_nested_array(ctx, expr.binders, sizes, expr.body, ())
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            "let-expr:LetExprFactor: mixed cases-plus-body form is "
            "not a valid surface construct"
        ],
    )


def _card_for(ctx: _JsLetCtx, binder: LetFactorBinder) -> int:
    """Resolve the static cardinality of ``binder.index``.

    `LetExprFactor` only unrolls when every binder's axis has a
    statically-known size; the helper consults ``ctx.cards``
    (populated from
    [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
    """
    idx = binder.index
    if isinstance(idx, TypeName):
        cards = ctx.cards
        if idx.name not in cards:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprFactor: binder {binder.var!r} "
                    f"references object {idx.name!r} whose cardinality "
                    "is unknown at render time"
                ],
            )
        return cards[idx.name]
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprFactor: binder {binder.var!r} index is "
            f"{type(idx).__name__}; only TypeName binders unroll"
        ],
    )


def _build_nested_array(
    ctx: _JsLetCtx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> tuple[str, str]:
    """Materialise the nested `array` tower for the uniform-body
    factor form.

    Substitutes each binder for its 1-indexed integer literal once
    the index tuple is fully fixed, then dispatches the substituted
    body through the recursive
    `_render`
    worker.
    """
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            literal = LetExprLiteral(value=value, integral=True)
            subst = _substitute_let_expr(
                subst,
                binder.var,
                index_value=literal,
                scalar_value=literal,
            )
        return _render(ctx, subst)
    level = len(fixed)
    rendered: list[tuple[str, str]] = []
    for i in range(sizes[level]):
        rendered.append(_build_nested_array(ctx, binders, sizes, body, fixed + (i,)))
    return _emit_array(ctx, tuple(rendered))


# ---------------------------------------------------------------------------
# Small fingerprint helpers.
# ---------------------------------------------------------------------------


def _paren_fingerprint(n: int) -> str:
    """Build the parenthesised comma-fingerprint for `n` children.

    ``( )`` for one child, ``( , )`` for two, ``( , , )`` for three.
    """
    if n == 1:
        return "( )"
    return "( " + (", " * (n - 1)).rstrip() + " )"


def _bracket_fingerprint(n: int) -> str:
    """Build the bracket comma-fingerprint for an `n`-element array.

    ``[ ]`` for one element, ``[ , ]`` for two, ``[ , , ]`` for three.
    """
    if n == 1:
        return "[ ]"
    return "[ " + (", " * (n - 1)).rstrip() + " ]"


def _target(ctx: _JsLetCtx) -> str:
    """Read the ctx's `target` tag for error messages."""
    return ctx.target


__all__ = ["render_let_expr_javascript"]
