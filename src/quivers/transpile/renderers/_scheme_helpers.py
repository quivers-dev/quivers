"""Render `LetExprNode` to Scheme/Church tree-sitter schema vertices.

Scheme's tree-sitter grammar exposes a small uniform vertex
vocabulary (`list`, `symbol`, `number`, `string`, `program`); every
compound form (`(define ...)`, `(lambda ...)`, `(list ...)`,
`(map ...)`) is a `list` whose first child is the head symbol. The
helpers in this module map each
[`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] variant onto that
vocabulary directly.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    DiscreteConstructor,
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
    LetFactorBinder,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.ir import LetAffineSource, LetExprAffineMap

_TARGET = "qvr-scheme-helper"

#: QVR builtins Scheme spells under another name: the power function is
#: `expt`, and Scheme has no `pow`.
_SCHEME_CALL_NAMES: dict[str, str] = {"pow": "expt"}


def render_let_expr_scheme(ctx, expr: LetExprNode) -> str:
    """Build a Scheme expression schema for `expr`.

    Scheme is prefix: `a + b` becomes `(+ a b)`. Most expression forms
    map to a `list` vertex with the operator/function name as the
    first child and the arguments as subsequent children.
    """
    if isinstance(expr, LetExprLiteral):
        # An integral literal is an exact Scheme integer, which a
        # subscript (`list-ref`) requires; any other literal keeps its
        # decimal point and is inexact.
        v = ctx.v(ctx.fresh("num"), "number")
        ctx.lit(v, str(int(expr.value)) if expr.integral else repr(float(expr.value)))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(v, expr.name)
        return v
    if isinstance(expr, LetExprString):
        v = ctx.v(ctx.fresh("str"), "string")
        ctx.lit(v, f'"{expr.value}"')
        return v
    if isinstance(expr, LetExprBinOp):
        lst = ctx.v(ctx.fresh("list"), "list")
        op = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(op, expr.op)
        ctx.e(lst, op)
        ctx.e(lst, render_let_expr_scheme(ctx, expr.left))
        ctx.e(lst, render_let_expr_scheme(ctx, expr.right))
        return lst
    if isinstance(expr, LetExprUnaryOp):
        lst = ctx.v(ctx.fresh("list"), "list")
        op = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(op, "-")
        ctx.e(lst, op)
        ctx.e(lst, render_let_expr_scheme(ctx, expr.operand))
        return lst
    if isinstance(expr, LetExprCall):
        lst = ctx.v(ctx.fresh("list"), "list")
        fn = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(fn, _SCHEME_CALL_NAMES.get(expr.func, expr.func))
        ctx.e(lst, fn)
        for a in expr.args:
            ctx.e(lst, render_let_expr_scheme(ctx, a))
        return lst
    if isinstance(expr, LetExprIndex):
        # A table is a list of rows, and `list-ref` takes one index, so
        # `a[i, j]` reads as `(list-ref (list-ref a i) j)`.
        current = render_let_expr_scheme(ctx, expr.array)
        for idx in expr.indices:
            lst = ctx.v(ctx.fresh("list"), "list")
            fn = ctx.v(ctx.fresh("sym"), "symbol")
            ctx.lit(fn, "list-ref")
            ctx.e(lst, fn)
            ctx.e(lst, current)
            ctx.e(lst, _render_index_scheme(ctx, idx))
            current = lst
        return current
    if isinstance(expr, LetExprList):
        lst = ctx.v(ctx.fresh("list"), "list")
        fn = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(fn, "list")
        ctx.e(lst, fn)
        for item in expr.items:
            ctx.e(lst, render_let_expr_scheme(ctx, item))
        return lst
    if isinstance(expr, LetExprLambda):
        return _render_lambda(ctx, expr)
    if isinstance(expr, LetExprAffineMap):
        return _render_affine_map(ctx, expr)
    if isinstance(expr, LetExprFactor):
        return _render_factor(ctx, expr)
    if isinstance(expr, LetExprMethodCall):
        return _render_method_call(ctx, expr)
    raise UnsupportedConstruct(
        _TARGET,
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


def _render_index_scheme(ctx, expr: LetExprNode) -> str:
    """Render one subscript of a `list-ref`.

    Scheme's `list-ref` takes an exact integer, and
    [`LetExprLiteral`][quivers.dsl.ast_nodes.LetExprLiteral] carries
    its value as a float, so a literal subscript is emitted as its
    integer spelling rather than through the general numeric path.
    Scheme's index origin is QVR's own, so the value passes through
    unshifted. Any other index expression renders normally.
    """
    if isinstance(expr, LetExprLiteral) and float(expr.value).is_integer():
        return _scheme_number(ctx, int(expr.value))
    return render_let_expr_scheme(ctx, expr)


def _scheme_form(ctx, head: str, children: tuple[str, ...]) -> str:
    """Emit ``(<head> <c0> <c1> ...)`` as a `list` vertex."""
    lst = ctx.v(ctx.fresh("list"), "list")
    sym = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(sym, head)
    ctx.e(lst, sym)
    for child in children:
        ctx.e(lst, child)
    return lst


def _scheme_number(ctx, value: int) -> str:
    """Emit an integer `number` vertex."""
    vid = ctx.v(ctx.fresh("num"), "number")
    ctx.lit(vid, str(value))
    return vid


def _affine_conditioning_row(ctx, sources: tuple[LetAffineSource, ...]) -> str:
    """Emit the map's conditioning row: the factors joined by
    `append` in declaration order.

    A one-factor row is the factor itself, so the common case emits
    no call at all.
    """
    if not sources:
        raise UnsupportedConstruct(
            _TARGET,
            [
                "let-expr:LetExprAffineMap: the map's conditioning "
                "row carries no factors"
            ],
        )
    rendered = tuple(render_let_expr_scheme(ctx, source.value) for source in sources)
    if len(rendered) == 1:
        return rendered[0]
    return _scheme_form(ctx, "append", rendered)


def _render_affine_map(ctx, expr: LetExprAffineMap) -> str:
    """Emit one head's row block of ``W x + b`` as a list of
    contracted rows.

    Scheme has no matrix product, but `map` over two equal-length
    lists paired with `apply +` is exactly the row contraction, so
    each codomain coordinate costs one form rather than one per
    domain coordinate. The row block is materialised as a `list` of
    `rows` such contractions, which is `rows` forms whatever the
    domain width.
    """
    row_expr = _affine_conditioning_row(ctx, expr.sources)
    weight = render_let_expr_scheme(ctx, expr.weight)
    bias = render_let_expr_scheme(ctx, expr.bias)
    rows: list[str] = []
    for i in range(expr.rows):
        row = expr.row_offset + i
        contracted = _scheme_form(
            ctx,
            "+",
            (
                _scheme_form(
                    ctx,
                    "apply",
                    (
                        _scheme_symbol(ctx, "+"),
                        _scheme_form(
                            ctx,
                            "map",
                            (
                                _scheme_symbol(ctx, "*"),
                                _scheme_form(
                                    ctx,
                                    "list-ref",
                                    (weight, _scheme_number(ctx, row)),
                                ),
                                row_expr,
                            ),
                        ),
                    ),
                ),
                _scheme_form(
                    ctx,
                    "list-ref",
                    (bias, _scheme_number(ctx, row)),
                ),
            ),
        )
        rows.append(
            _scheme_form(ctx, "exp", (contracted,))
            if expr.transform == "exp"
            else contracted
        )
    return _scheme_form(ctx, "list", tuple(rows))


def _scheme_symbol(ctx, name: str) -> str:
    """Emit a bare `symbol` vertex."""
    vid = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(vid, name)
    return vid


def _render_lambda(ctx, expr: LetExprLambda) -> str:
    """Emit ``(lambda (<param>) <body>)``.

    Scheme's lambda form takes a parameter list; the single-parameter
    QVR surface form maps to a one-element list literal in the
    parameter position.
    """
    lam = ctx.v(ctx.fresh("list"), "list")
    head = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(head, "lambda")
    ctx.e(lam, head)
    params = ctx.v(ctx.fresh("list"), "list")
    pid = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(pid, expr.param)
    ctx.e(params, pid)
    ctx.e(lam, params)
    ctx.e(lam, render_let_expr_scheme(ctx, expr.body))
    return lam


def _render_method_call(ctx, expr: LetExprMethodCall) -> str:
    """Emit ``(<method> <receiver> <args...>)``.

    Scheme has no dot-method dispatch; the closest faithful rendering
    is a positional call with the method name as the head and the
    receiver threaded as the first argument. This matches how Scheme
    libraries (e.g. SRFI-1) expose collection operations: `(length
    xs)`, `(car xs)`, `(weight chart item)`. The receiver and
    argument schemas are emitted recursively without rewriting.
    """
    lst = ctx.v(ctx.fresh("list"), "list")
    head = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(head, expr.method)
    ctx.e(lst, head)
    ctx.e(lst, render_let_expr_scheme(ctx, expr.receiver))
    for a in expr.args:
        ctx.e(lst, render_let_expr_scheme(ctx, a))
    return lst


def _render_factor(ctx, expr: LetExprFactor) -> str:
    """Emit a Scheme rendering for ``factor v : I in body`` / cases.

    Three branches:

    1. Single-axis ``cases`` form (``factor v : I in { 0 -> e0, ... }``).
       The case labels statically enumerate the index set, so the
       expression unrolls into ``(list e_0 e_1 ... e_{N-1})`` in label
       order, regardless of whether the index type is named or
       inline.
    2. Single-axis ``body`` form whose binder's index is an inline
       ``FinSet(N)`` with a literal integer ``N``. The body is
       quasi-quoted once per index via ``(map (lambda (<v>) <body>)
       (iota N))``; Scheme's substitution semantics handle the
       per-index binding at evaluation time, so no compile-time alpha
       substitution is needed.
    3. Anything else (multi-axis form, ``TypeName`` binder whose size
       the helper cannot resolve from the AST alone, mixed binders
       across axes) raises
       `UnsupportedConstruct`
       with a precise ``let-expr:LetExprFactor:<reason>`` kind so the
       caller can decide whether to surface the construct as a real
       failure or skip the cell in a backend-matrix test.
    """
    if expr.cases:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                _TARGET,
                ["let-expr:LetExprFactor:cases-with-multi-axis-binders"],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        expected = list(range(len(ordered)))
        if [c.label for c in ordered] != expected:
            raise UnsupportedConstruct(
                _TARGET,
                ["let-expr:LetExprFactor:cases-labels-not-dense"],
            )
        lst = ctx.v(ctx.fresh("list"), "list")
        head = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(head, "list")
        ctx.e(lst, head)
        for case in ordered:
            ctx.e(lst, render_let_expr_scheme(ctx, case.value))
        return lst
    if expr.body is None:
        raise UnsupportedConstruct(_TARGET, ["let-expr:LetExprFactor:no-body-no-cases"])
    if len(expr.binders) != 1:
        raise UnsupportedConstruct(_TARGET, ["let-expr:LetExprFactor:multi-axis-body"])
    binder = expr.binders[0]
    size_text = _binder_static_size(binder)
    if size_text is None:
        # Fall back to the renderer ctx's cards map for binders whose
        # index is a `TypeName` referencing a named object (e.g.
        # ``c : Comp`` with ``object Comp : FinSet 3``).
        index = binder.index
        cards = getattr(ctx, "cards", None) or {}
        if isinstance(index, TypeName):
            card = cards.get(index.name)
            if card is not None:
                size_text = str(card)
    if size_text is None:
        raise UnsupportedConstruct(
            _TARGET,
            [f"let-expr:LetExprFactor:unresolved-binder-size:{binder.var}"],
        )
    # `(map (lambda (<var>) <body>) (iota <N>))` -- one Scheme list
    # per fresh vertex; the body schema is emitted once and bound at
    # evaluation time.
    outer = ctx.v(ctx.fresh("list"), "list")
    map_head = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(map_head, "map")
    ctx.e(outer, map_head)
    # Inner lambda.
    lam = ctx.v(ctx.fresh("list"), "list")
    lam_head = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(lam_head, "lambda")
    ctx.e(lam, lam_head)
    params = ctx.v(ctx.fresh("list"), "list")
    pid = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(pid, binder.var)
    ctx.e(params, pid)
    ctx.e(lam, params)
    ctx.e(lam, render_let_expr_scheme(ctx, expr.body))
    ctx.e(outer, lam)
    # `(iota <N>)`.
    iota = ctx.v(ctx.fresh("list"), "list")
    iota_head = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(iota_head, "iota")
    ctx.e(iota, iota_head)
    size_vid = ctx.v(ctx.fresh("num"), "number")
    ctx.lit(size_vid, size_text)
    ctx.e(iota, size_vid)
    ctx.e(outer, iota)
    return outer


def _binder_static_size(binder: LetFactorBinder) -> str | None:
    """Return the binder's index cardinality as a literal integer
    string when statically known from the AST alone.

    Inline ``FinSet(N)`` with a literal integer argument resolves to
    ``str(N)``; every other shape (``TypeName`` referencing a named
    object, ``FinSet`` whose argument is not a literal integer, any
    non-discrete constructor) returns ``None`` so the caller raises
    `UnsupportedConstruct`
    with the appropriate reason. Resolving named-type sizes would
    require threading the module symbol table through every renderer's
    let-expression helper; the helper deliberately stays
    AST-local.
    """
    index = binder.index
    if not isinstance(index, DiscreteConstructor):
        return None
    if index.constructor != "FinSet":
        return None
    if len(index.args) != 1:
        return None
    (raw,) = index.args
    try:
        n = int(raw)
    except TypeError, ValueError:
        return None
    if n < 0:
        return None
    return str(n)


__all__ = ["render_let_expr_scheme"]
