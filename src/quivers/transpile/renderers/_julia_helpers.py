"""Render [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] subtrees
to Julia tree-sitter schema vertices.

The Julia grammar exposes the following expression-level vertex kinds
the helper builds:

* `integer_literal` / `float_literal` for numeric leaves
* `string_literal` wrapping a `content` child for double-quoted strings
* `identifier` for variable references
* `binary_expression` (per-operator alts via the `operator` child's
  `chose-alt-fingerprint`)
* `unary_expression` for the unary minus
* `call_expression` whose children are `identifier argument_list`
* `field_expression` for the `receiver.method` access path used to
  encode method-dispatch chains (Julia is multiple-dispatch; the
  conventional surface for `chart.goal_weight()` is
  `goal_weight(chart)`, but a parse-roundtrip helper must preserve the
  source syntax; the helper renders the dot form to keep the
  user-written shape)
* `index_expression` whose children are `identifier vector_expression`
  where the inner `vector_expression` carries the comma-separated
  index expressions
* `vector_expression` for `[a, b, c]` list literals
* `arrow_function_expression` for `param -> body` (single-param uses an
  `identifier` for the param; multi-param uses an `argument_list`)

Every vertex sets `chose-alt-child-kinds` to the space-separated
sequence of its children's surface kinds, and (where the grammar
distinguishes alternatives by punctuation) sets
`chose-alt-fingerprint` to the canonical punctuation skeleton. The
panproto pretty-printer consults both slots to pick the right grammar
production; missing constraints cause silent emission gaps. The helper
returns `(vertex_id, vertex_kind)` from each recursive call so parents
can build the `chose-alt-child-kinds` string from real child kinds.

`LetExprFactor` is unrolled at render time:

* the **cases form** (binders contain a single axis, body is `None`,
  cases enumerate labels in `[0, |axis|)`) becomes a
  `vector_expression` whose children are the case bodies in label
  order;
* the **uniform-body form** (one or more binders, body is the repeated
  expression, cases is empty) becomes a tower of `vector_expression`
  vertices of shape `(|b0|, |b1|, ..., |bn-1|)`, with each binder
  substituted for its 1-indexed integer value (Julia arrays are
  1-indexed, matching QVR's surface indexing).

Static cardinalities for binder axes come from the ctx's `cards` map
(populated from [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
When an axis size is missing, the helper raises
`UnsupportedConstruct`
tagged with the ctx's `target` ("turing" / "gen") rather than emitting
a placeholder.
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
from quivers.dsl.ast_nodes.let_expressions import LetFactorBinder, LetFactorCase
from quivers.dsl.ast_nodes.objects import (
    DiscreteConstructor,
    ObjectExpr,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.ir import (
    IRArg,
    LetAffineSource,
    LetExprAffineMap,
    IRArgList,
    IRArgNumber,
    IRArgRef,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRSample,
    IRScore,
    Plate,
)
from quivers.transpile.renderers._one_based import (
    one_based_subscript,
    reads_index_covariate,
)


@runtime_checkable
class _JlEmitCtx(Protocol):
    """Structural protocol for the helper's ctx parameter.

    The ctx is the caller's schema-building surface and nothing else:
    `target` tags the error messages with the backend name
    (``"turing"`` / ``"gen"``), `cards` is the static axis-size table
    sourced from [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]
    and consulted when unrolling a
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor], and the
    five methods write vertices, edges, literals and constraints into
    the caller's builder.

    Array-shape knowledge is *not* part of this protocol. It travels
    separately in a
    `JuliaShapes`,
    so a caller that only wants an expression rendered (with no IR
    program behind it) needs to supply nothing beyond the emission
    surface.
    """

    target: str
    cards: dict[str, int]

    def fresh(self, prefix: str) -> str: ...
    def v(self, vid: str, kind: str) -> str: ...
    def e(self, src: str, tgt: str, kind: str = ...) -> None: ...
    def lit(self, vid: str, text: str) -> None: ...
    def constraint(self, vid: str, sort: str, value: str) -> None: ...


class JuliaShapes:
    """Track array shapes used by Julia renderers.

    `name_event_rank` records event-axis counts, and `name_array_rank` records
    total array ranks. `batch_rank` locates event axes within the enclosing
    binding. These values determine whether reductions target event axes and
    whether row gathers need trailing ``:`` slices.

    `nested_names` identifies bindings emitted as nested vector literals, whose
    indices are chained as ``t[i][j]`` rather than emitted as ``t[i, j]``.
    `index_covariates` names the integer data inputs, whose entries the data
    contract delivers as one-based subscripts.
    `row_arrays` maps each binding stored as an array of event vectors to
    the number of its outer array axes: a subscript into it reads those axes
    first and then the event, as ``p[i, j][k]``.
    Defaults describe a standalone scalar expression with no declared plates.
    """

    name_event_rank: dict[str, int]
    name_array_rank: dict[str, int]
    nested_names: frozenset[str]
    batch_rank: int
    index_covariates: frozenset[str]
    row_arrays: dict[str, int]

    def __init__(
        self,
        *,
        name_event_rank: dict[str, int] | None = None,
        name_array_rank: dict[str, int] | None = None,
        nested_names: frozenset[str] = frozenset(),
        batch_rank: int = 0,
        index_covariates: frozenset[str] = frozenset(),
        row_arrays: dict[str, int] | None = None,
    ) -> None:
        self.name_event_rank = dict(name_event_rank or {})
        self.name_array_rank = dict(name_array_rank or {})
        self.nested_names = nested_names
        self.batch_rank = batch_rank
        self.index_covariates = index_covariates
        self.row_arrays = dict(row_arrays or {})

    def scoped_to(self, batch_rank: int) -> JuliaShapes:
        """The same rank tables, scoped to a binding whose plate has
        ``batch_rank`` leading batch axes.

        The tables are program-wide; `batch_rank` is per-binding, so a
        renderer builds the tables once and scopes them at each
        deterministic it emits.
        """
        return JuliaShapes(
            name_event_rank=self.name_event_rank,
            name_array_rank=self.name_array_rank,
            nested_names=self.nested_names,
            batch_rank=batch_rank,
            index_covariates=self.index_covariates,
            row_arrays=self.row_arrays,
        )


#: The shape environment of an expression rendered outside any IR
#: program: every name scalar-ranked, every subscript dense.
_STANDALONE_SHAPES = JuliaShapes()


@runtime_checkable
class _JlShapeView(Protocol):
    """The shape tables the shape-inference walks read.

    Satisfied by both
    `JuliaShapes`
    (which the renderers hand to the public inference entry points)
    and `_JlState` (which the emitters carry).
    """

    name_event_rank: dict[str, int]
    name_array_rank: dict[str, int]
    nested_names: frozenset[str]


class _JlState:
    """Render state: the caller's emission ctx, the shape environment,
    and the mutable elementwise flag.

    `dotted` selects elementwise emission: binary operators render as
    ``.+`` / ``.*`` and function calls as ``f.(x)``, which is what the
    caller needs once it stops wrapping the whole binding in Julia's
    ``@.`` macro. The reduction path flips it on while it renders the
    reduced argument and restores the caller's value afterwards. It
    lives here rather than on the caller's ctx so a render never
    mutates an object the caller owns.
    """

    def __init__(
        self,
        ctx: _JlEmitCtx,
        shapes: JuliaShapes,
        *,
        dotted: bool,
    ) -> None:
        self._ctx = ctx
        self.target = ctx.target
        self.cards = ctx.cards
        self.name_event_rank = shapes.name_event_rank
        self.name_array_rank = shapes.name_array_rank
        self.nested_names = shapes.nested_names
        self.batch_rank = shapes.batch_rank
        self.index_covariates = shapes.index_covariates
        self.row_arrays = shapes.row_arrays
        self.dotted = dotted

    def fresh(self, prefix: str) -> str:
        return self._ctx.fresh(prefix)

    def v(self, vid: str, kind: str) -> str:
        return self._ctx.v(vid, kind)

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._ctx.e(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._ctx.lit(vid, text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._ctx.constraint(vid, sort, value)


#: QVR reduction primitives paired with the Julia function that
#: reduces along a named axis. Julia spells the extremal reductions
#: `maximum` / `minimum`; `max` / `min` are the elementwise binary
#: forms and would silently return their argument unchanged.
_AXIS_REDUCING_CALLS: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "prod": "prod",
    "max": "maximum",
    "min": "minimum",
}


#: Let calls Julia spells under another name: integer division and
#: remainder truncate toward zero, as Julia's `div` and `rem` do.
_JULIA_CALL_NAMES: dict[str, str] = {"quot": "div", "rem": "rem"}

#: QVR builtins whose two-argument form is pointwise, which Julia spells
#: as a broadcast call.
_BINARY_POINTWISE_CALLS: frozenset[str] = frozenset({"max", "min"})


def render_let_expr_julia(
    ctx: _JlEmitCtx,
    expr: LetExprNode,
    *,
    shapes: JuliaShapes = _STANDALONE_SHAPES,
    dotted: bool = False,
) -> str:
    """Build a Julia expression schema for ``expr`` in ``ctx``.

    Returns the root vertex id. Wraps `_render` to discard the kind
    return value at the public boundary so callers see the same
    signature as the other per-target helpers.

    Pass ``shapes`` when the expression sits in an IR program whose
    names carry plates: the array-aware paths (row gathers, event-axis
    reductions, nested-tower subscripts) all read it. Omitting it
    renders the expression exactly as written, which is what a caller
    with no IR behind the expression wants.

    Pass ``dotted=True`` when the caller does not wrap the emitted
    binding in Julia's ``@.`` macro and still wants elementwise
    arithmetic; see
    `let_expr_has_axis_reduction`
    for the case that forces it.
    """
    vid, _kind = _render(_JlState(ctx, shapes, dotted=dotted), expr)
    return vid


def let_expr_has_axis_reduction(ctx: _JlShapeView, expr: LetExprNode) -> bool:
    """True iff ``expr`` contracts an axis anywhere in its tree:
    either a reduction along the last axis of an array with an event
    axis or with two or more axes, or the matrix-vector product an
    affine parameter map denotes.

    A binding whose body contracts an axis cannot be wrapped in
    Julia's ``@.`` macro: the macro broadcasts the contraction
    itself, applying `sum` to each scalar element (or turning a
    `matrix * vector` into an elementwise `.*`) and leaving the
    contraction undone. The renderers ask this question to decide
    between the ``@.`` form and the explicitly dotted form.
    """
    if isinstance(expr, LetExprCall):
        if _reduces_named_axis(ctx, expr.func, expr.args):
            return True
        return any(let_expr_has_axis_reduction(ctx, a) for a in expr.args)
    if isinstance(expr, LetExprBinOp):
        return let_expr_has_axis_reduction(
            ctx, expr.left
        ) or let_expr_has_axis_reduction(ctx, expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return let_expr_has_axis_reduction(ctx, expr.operand)
    if isinstance(expr, LetExprIndex):
        return let_expr_has_axis_reduction(ctx, expr.array) or any(
            let_expr_has_axis_reduction(ctx, i) for i in expr.indices
        )
    if isinstance(expr, LetExprList):
        return any(let_expr_has_axis_reduction(ctx, i) for i in expr.items)
    if isinstance(expr, LetExprLambda):
        return let_expr_has_axis_reduction(ctx, expr.body)
    if isinstance(expr, LetExprMethodCall):
        return let_expr_has_axis_reduction(ctx, expr.receiver) or any(
            let_expr_has_axis_reduction(ctx, a) for a in expr.args
        )
    if isinstance(expr, LetExprFactor):
        if expr.body is not None and let_expr_has_axis_reduction(ctx, expr.body):
            return True
        return any(let_expr_has_axis_reduction(ctx, c.value) for c in expr.cases)
    if isinstance(expr, LetExprAffineMap):
        # The matrix-vector product contracts the map's column axis.
        # `@.` would broadcast the `*` into a `.*` and leave the
        # contraction undone, so the binding takes the explicitly
        # dotted form and the emitter writes its own operators.
        return True
    return False


def infer_array_rank(ctx: _JlShapeView, expr: LetExprNode) -> int:
    """Infer the full Julia array rank a let-expression evaluates to.

    Leaf variables read `ctx.name_array_rank`; compound expressions
    propagate structurally. A subscript consumes one axis per index
    but adds back whatever rank the index itself carries, so a gather
    by a plate-shaped covariate (`Z_mat[item_idx]`) keeps the rank it
    started with while a literal subscript drops one.

    The renderers compare this against the binding's own plate rank to
    tell a value that already spans its plate from a scalar the plate
    replicates.
    """
    if isinstance(expr, (LetExprLiteral, LetExprString)):
        return 0
    if isinstance(expr, LetExprVar):
        return ctx.name_array_rank.get(expr.name, 0)
    if isinstance(expr, LetExprBinOp):
        return max(
            infer_array_rank(ctx, expr.left),
            infer_array_rank(ctx, expr.right),
        )
    if isinstance(expr, LetExprUnaryOp):
        return infer_array_rank(ctx, expr.operand)
    if isinstance(expr, LetExprCall):
        inner = max((infer_array_rank(ctx, a) for a in expr.args), default=0)
        if expr.func in _AXIS_REDUCING_CALLS and len(expr.args) == 1:
            return max(0, inner - 1)
        return inner
    if isinstance(expr, LetExprIndex):
        rank = infer_array_rank(ctx, expr.array)
        for index in expr.indices:
            rank = max(0, rank - 1) + infer_array_rank(ctx, index)
        return rank
    if isinstance(expr, LetExprList):
        return 1 + max(
            (infer_array_rank(ctx, item) for item in expr.items),
            default=0,
        )
    if isinstance(expr, LetExprFactor):
        return len(expr.binders)
    if isinstance(expr, LetExprAffineMap):
        # One head's row block is a length-`rows` vector.
        return 1
    return 0


def _infer_event_rank(ctx: _JlShapeView, expr: LetExprNode) -> int:
    """Infer the event rank of a let-expression at emit time.

    Leaf variables read their rank from `ctx.name_event_rank`;
    compound expressions propagate it structurally, mirroring the
    Python helper's walk: binary operators broadcast to the wider
    operand, unary minus and elementwise math preserve the rank, a
    reducing primitive collapses to 0, and each index a subscript
    supplies consumes one axis.
    """
    if isinstance(expr, (LetExprLiteral, LetExprString)):
        return 0
    if isinstance(expr, LetExprVar):
        return ctx.name_event_rank.get(expr.name, 0)
    if isinstance(expr, LetExprBinOp):
        return max(
            _infer_event_rank(ctx, expr.left),
            _infer_event_rank(ctx, expr.right),
        )
    if isinstance(expr, LetExprUnaryOp):
        return _infer_event_rank(ctx, expr.operand)
    if isinstance(expr, LetExprCall):
        if expr.func in _AXIS_REDUCING_CALLS and len(expr.args) == 1:
            return 0
        return max((_infer_event_rank(ctx, a) for a in expr.args), default=0)
    if isinstance(expr, LetExprIndex):
        arr_rank = _infer_event_rank(ctx, expr.array)
        return max(0, arr_rank - len(expr.indices))
    if isinstance(expr, LetExprList):
        return max(
            (_infer_event_rank(ctx, item) for item in expr.items),
            default=0,
        )
    return 0


def _render(ctx: _JlState, expr: LetExprNode) -> tuple[str, str]:
    """Recursive renderer returning ``(vertex_id, vertex_kind)`` so
    parents can populate ``chose-alt-child-kinds`` accurately."""
    if isinstance(expr, LetExprLiteral):
        return _emit_literal(ctx, expr.value, integral=expr.integral)
    if isinstance(expr, LetExprString):
        return _emit_string(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_identifier(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_binop(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_unary(ctx, expr)
    if isinstance(expr, LetExprCall):
        return _emit_call(ctx, expr.func, expr.args)
    if isinstance(expr, LetExprIndex):
        return _emit_index(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_vector(ctx, tuple(_render(ctx, item) for item in expr.items))
    if isinstance(expr, LetExprLambda):
        return _emit_lambda(ctx, expr)
    if isinstance(expr, LetExprMethodCall):
        return _emit_method_call(ctx, expr)
    if isinstance(expr, LetExprAffineMap):
        return _emit_affine_map(ctx, expr)
    if isinstance(expr, LetExprFactor):
        return _render_factor(ctx, expr)
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


# ---------------------------------------------------------------------------
# Per-kind emitters.
# ---------------------------------------------------------------------------


def _emit_literal(
    ctx: _JlState, value: float | int, *, integral: bool
) -> tuple[str, str]:
    """Emit an `integer_literal` or a `float_literal` of the literal's own type.

    An integral literal (a source integer, a factor binder's value, a
    subscript, an axis count) is a Julia `Int`; every other literal is a
    `Float64` spelled with its decimal point, so real arithmetic on
    whole-number reals stays in floating point instead of wrapping
    around past `typemax(Int)`.

    Parameters
    ----------
    ctx : _JlState
        The emit state.
    value : float | int
        The literal's value.
    integral : bool
        Whether the literal is an integer.

    Returns
    -------
    tuple[str, str]
        The vertex and its kind.

    Raises
    ------
    UnsupportedConstruct
        If an integral literal's value is not a whole number.
    """
    if integral:
        if not float(value).is_integer():
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprLiteral:{_target(ctx)}: integral literal "
                    f"{value!r} is not a whole number"
                ],
            )
        text = str(int(value))
        vid = ctx.v(ctx.fresh("il"), "integer_literal")
        ctx.lit(vid, text)
        ctx.constraint(vid, "chose-alt-fingerprint", text)
        return vid, "integer_literal"
    text = repr(float(value))
    vid = ctx.v(ctx.fresh("fl"), "float_literal")
    ctx.lit(vid, text)
    ctx.constraint(vid, "chose-alt-fingerprint", text)
    return vid, "float_literal"


def _emit_string(ctx: _JlState, text: str) -> tuple[str, str]:
    """Emit a `string_literal` whose single `content` child carries
    the unescaped text body.

    The Julia tree-sitter grammar models a double-quoted string as
    `string_literal -> '"' content '"'`; the fingerprint
    ``'" "'`` selects the double-quoted alternative and the
    `content` vertex's `literal-value` slot holds the text body.
    """
    vid = ctx.v(ctx.fresh("sl"), "string_literal")
    ctx.constraint(vid, "chose-alt-fingerprint", '" "')
    ctx.constraint(vid, "chose-alt-child-kinds", "content")
    content = ctx.v(ctx.fresh("sc"), "content")
    ctx.lit(content, text)
    ctx.constraint(content, "chose-alt-fingerprint", text)
    ctx.e(vid, content, "child_of")
    return vid, "string_literal"


def _emit_identifier(ctx: _JlState, name: str) -> tuple[str, str]:
    """Emit a bare `identifier` vertex carrying ``name``."""
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(vid, name)
    ctx.constraint(vid, "chose-alt-fingerprint", name)
    return vid, "identifier"


def _emit_operator(ctx: _JlState, op: str) -> tuple[str, str]:
    """Emit an `operator` vertex whose literal text is ``op``."""
    vid = ctx.v(ctx.fresh("op"), "operator")
    ctx.lit(vid, op)
    ctx.constraint(vid, "chose-alt-fingerprint", op)
    return vid, "operator"


_JL_PAREN_REQUIRED_OPERAND_KINDS: frozenset[str] = frozenset(
    {
        "binary_expression",
        "unary_expression",
        "arrow_function_expression",
    }
)

_JL_INDEX_CALLEE_KINDS: frozenset[str] = frozenset(
    {
        "identifier",
        "field_expression",
        "call_expression",
        "index_expression",
        "parenthesized_expression",
        "vector_expression",
    }
)
"""Vertex kinds Julia's `index_expression` accepts as the array
callee directly. Anything else (numeric literals, binary
expressions, unary expressions) must be wrapped in
`parenthesized_expression`; otherwise the pretty-printer drops
the offending subtree silently."""
"""Operand kinds that must be wrapped in `parenthesized_expression`
when they appear as a sub-expression of a binary or unary operator.

`binary_expression`: precedence preservation (the printer emits in
source order without re-grouping; without parens `(a + b) * c`
becomes `a + b * c`).

`unary_expression`: token-collision. `a - -b` tokenises as `a -- b`
which Julia parses as the post-decrement-like operator `--`, dropping
the right operand entirely.

`arrow_function_expression`: precedence (arrow binds looser than any
binary operator)."""


def _maybe_paren(
    ctx: _JlState,
    rendered: tuple[str, str],
) -> tuple[str, str]:
    """Wrap `rendered` in a `parenthesized_expression` if its vertex
    kind is in `_JL_PAREN_REQUIRED_OPERAND_KINDS`."""
    vid, kind = rendered
    if kind not in _JL_PAREN_REQUIRED_OPERAND_KINDS:
        return rendered
    return _force_paren(ctx, rendered)


def _force_paren(
    ctx: _JlState,
    rendered: tuple[str, str],
) -> tuple[str, str]:
    """Always wrap `rendered` in a `parenthesized_expression`. Used
    where the surrounding grammar production rejects the rendered
    kind directly (e.g. an `integer_literal` as an
    `index_expression` callee)."""
    vid, kind = rendered
    paren = ctx.v(ctx.fresh("pe"), "parenthesized_expression")
    ctx.constraint(paren, "chose-alt-fingerprint", "( )")
    ctx.constraint(paren, "chose-alt-child-kinds", kind)
    ctx.e(paren, vid, "child_of")
    return paren, "parenthesized_expression"


def _emit_binop(ctx: _JlState, expr: LetExprBinOp) -> tuple[str, str]:
    """Emit a `binary_expression` whose children are
    ``<left> operator <right>``.

    Under `ctx.dotted` the operator is prefixed with Julia's
    broadcasting dot (``.*``, ``.+``), which is what the caller needs
    when the binding is not wrapped in the ``@.`` macro.
    """
    left_vid, left_kind = _maybe_paren(ctx, _render(ctx, expr.left))
    op_text = f".{expr.op}" if ctx.dotted else expr.op
    op_vid, _op_kind = _emit_operator(ctx, op_text)
    right_vid, right_kind = _maybe_paren(ctx, _render(ctx, expr.right))
    vid = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        f"{left_kind} operator {right_kind}",
    )
    ctx.e(vid, left_vid, "child_of")
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, right_vid, "child_of")
    return vid, "binary_expression"


def _emit_power(
    ctx: _JlState, base: LetExprNode, exponent: LetExprNode
) -> tuple[str, str]:
    """Emit QVR's pointwise ``pow(base, exponent)`` as ``base .^ exponent``.

    Julia's `Base` has no `pow` function; the power is the `^` operator,
    and its broadcast form applies pointwise to arrays as to scalars.

    Parameters
    ----------
    ctx : _JlState
        The emit state.
    base : LetExprNode
        The base.
    exponent : LetExprNode
        The exponent.

    Returns
    -------
    tuple[str, str]
        The `binary_expression` vertex and its kind.
    """
    left_vid, left_kind = _maybe_paren(ctx, _render(ctx, base))
    op_vid, _op_kind = _emit_operator(ctx, ".^")
    right_vid, right_kind = _maybe_paren(ctx, _render(ctx, exponent))
    vid = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", f"{left_kind} operator {right_kind}")
    ctx.e(vid, left_vid, "child_of")
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, right_vid, "child_of")
    return vid, "binary_expression"


def _emit_unary(ctx: _JlState, expr: LetExprUnaryOp) -> tuple[str, str]:
    """Emit a `unary_expression` whose children are ``operator <operand>``.

    [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] only
    encodes unary minus (the parser does not produce other unary
    operators in let expressions), so the helper emits a literal
    ``-`` operator. Operand is wrapped in parens when it is itself
    a unary or binary expression to avoid Julia's `--` tokenisation
    that would otherwise eat the operand.
    """
    op_vid, _op_kind = _emit_operator(ctx, "-")
    operand_vid, operand_kind = _maybe_paren(ctx, _render(ctx, expr.operand))
    vid = ctx.v(ctx.fresh("ue"), "unary_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", f"operator {operand_kind}")
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, operand_vid, "child_of")
    return vid, "unary_expression"


def _sigmoid_expansion(arg: LetExprNode) -> LetExprNode:
    """Rewrite ``sigmoid(x)`` to the arithmetic body ``1 / (1 + exp(-x))``.

    Julia's `Base` carries no `sigmoid`; `StatsFuns.logistic` needs a
    package import the probe container does not bring into `Main`. The
    logit-link identity `1 / (1 + exp(-x))` is a closed-form Julia
    expression that broadcasts elementwise under `@.`, so the renderer
    expands the call into that body and lets the normal binop / unary
    path parenthesise it.
    """
    return LetExprBinOp(
        op="/",
        left=LetExprLiteral(value=1.0),
        right=LetExprBinOp(
            op="+",
            left=LetExprLiteral(value=1.0),
            right=LetExprCall(func="exp", args=(LetExprUnaryOp(operand=arg),)),
        ),
    )


def _emit_call(
    ctx: _JlState, func: str, args: tuple[LetExprNode, ...]
) -> tuple[str, str]:
    """Emit ``<func>(<arg_0>, <arg_1>, ...)`` as a `call_expression`
    whose children are ``identifier argument_list``.

    QVR math primitives without a `Base` Julia counterpart are rewritten
    to a closed-form body before emission: `sigmoid(x)` becomes
    `1 / (1 + exp(-x))`.

    A reduction primitive collapses the last axis of its argument. An
    argument with an event axis, or with two or more axes, routes to
    `_emit_axis_reduction`, which names that axis; a nested tower
    reduces each of its rows; a vector reduces whole, which is the
    same.
    A two-argument `max` or `min` is pointwise and emits `f.(a, b)`;
    `pow(a, b)` emits the broadcast power `a .^ b`.
    Every other call emits `f.(args)` under `ctx.dotted` and `f(args)`
    otherwise.
    """
    if func == "sigmoid" and len(args) == 1:
        return _render(ctx, _sigmoid_expansion(args[0]))
    if func == "pow" and len(args) == 2:
        return _emit_power(ctx, args[0], args[1])
    if _reduces_tower_rows(ctx, func, args):
        return _emit_tower_row_reduction(ctx, func, args[0])
    if _reduces_named_axis(ctx, func, args):
        return _emit_axis_reduction(ctx, func, args[0], infer_array_rank(ctx, args[0]))
    if func in _AXIS_REDUCING_CALLS and len(args) == 1 and ctx.dotted:
        # A reduction of a vector reduces it whole; broadcasting the
        # call would apply it to each entry and reduce nothing.
        rendered_argument = _render(ctx, args[0])
        callee_vid, _callee_kind = _emit_identifier(ctx, _AXIS_REDUCING_CALLS[func])
        al_vid = _emit_argument_list(ctx, (rendered_argument,))
        vid = ctx.v(ctx.fresh("ce"), "call_expression")
        ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
        ctx.e(vid, callee_vid, "child_of")
        ctx.e(vid, al_vid, "child_of")
        return vid, "call_expression"
    rendered = tuple(_render(ctx, a) for a in args)
    callee_vid, _callee_kind = _emit_identifier(ctx, _JULIA_CALL_NAMES.get(func, func))
    al_vid = _emit_argument_list(ctx, rendered)
    # QVR's two-argument `max` / `min` are pointwise. Julia's undotted
    # `max(a, b)` of two arrays compares them as wholes and returns one,
    # so the binary form always broadcasts.
    pointwise = func in _BINARY_POINTWISE_CALLS and len(args) == 2
    kind = "broadcast_call_expression" if ctx.dotted or pointwise else "call_expression"
    vid = ctx.v(ctx.fresh("ce"), kind)
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
    ctx.e(vid, callee_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, kind


def _emit_named_argument(
    ctx: _JlState, name: str, value: tuple[str, str]
) -> tuple[str, str]:
    """Emit the keyword argument ``<name> = <value>``.

    Julia's tree-sitter grammar models a call's keyword argument as an
    `assignment` vertex sitting directly in the `argument_list`, so
    the shape here is the same one `_assignment` would build at
    statement level.
    """
    value_vid, value_kind = value
    name_vid, name_kind = _emit_identifier(ctx, name)
    op_vid, _op_kind = _emit_operator(ctx, "=")
    vid = ctx.v(ctx.fresh("kw"), "assignment")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        f"{name_kind} operator {value_kind}",
    )
    ctx.e(vid, name_vid, "child_of")
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, value_vid, "child_of")
    return vid, "assignment"


def _emit_keyword_call(
    ctx: _JlState,
    func: str,
    positional: tuple[str, str],
    keywords: tuple[tuple[str, tuple[str, str]], ...],
) -> tuple[str, str]:
    """Emit ``<func>(<positional>, <k0> = <v0>, ...)``.

    Kept separate from `_emit_call` because the keyword arguments are
    renderer-synthesised axis positions rather than QVR let-expression
    nodes, and because the callee must never pick up the broadcasting
    dot: `sum.(x, dims = 2)` reduces nothing.
    """
    rendered = (
        positional,
        *(_emit_named_argument(ctx, k, v) for k, v in keywords),
    )
    callee_vid, _callee_kind = _emit_identifier(ctx, func)
    al_vid = _emit_argument_list(ctx, rendered)
    vid = ctx.v(ctx.fresh("ce"), "call_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
    ctx.e(vid, callee_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "call_expression"


def _is_tower(ctx: _JlShapeView, expr: LetExprNode) -> bool:
    """Whether an expression names a nested vector tower.

    Parameters
    ----------
    ctx : _JlShapeView
        The shape tables.
    expr : LetExprNode
        The expression.

    Returns
    -------
    bool
        True for a bare name the emit materialises as nested vectors.
    """
    return isinstance(expr, LetExprVar) and expr.name in ctx.nested_names


def _reduces_tower_rows(
    ctx: _JlShapeView, func: str, args: tuple[LetExprNode, ...]
) -> bool:
    """Whether a call reduces the rows of a nested vector tower.

    Parameters
    ----------
    ctx : _JlShapeView
        The shape tables.
    func : str
        The callee.
    args : tuple[LetExprNode, ...]
        The arguments.

    Returns
    -------
    bool
        True for a one-argument reduction of a tower name.
    """
    return func in _AXIS_REDUCING_CALLS and len(args) == 1 and _is_tower(ctx, args[0])


def _reduces_named_axis(
    ctx: _JlShapeView, func: str, args: tuple[LetExprNode, ...]
) -> bool:
    """Whether a reduction must name the last axis of a dense array.

    Parameters
    ----------
    ctx : _JlShapeView
        The shape tables.
    func : str
        The callee.
    args : tuple[LetExprNode, ...]
        The arguments.

    Returns
    -------
    bool
        True for a one-argument reduction of an array, not a tower, that
        has an event axis or two or more axes: Julia's `sum(x)` would
        reduce every axis of it, not only the last.
    """
    return (
        func in _AXIS_REDUCING_CALLS
        and len(args) == 1
        and not _is_tower(ctx, args[0])
        and (_infer_event_rank(ctx, args[0]) > 0 or infer_array_rank(ctx, args[0]) >= 2)
    )


def _emit_tower_row_reduction(
    ctx: _JlState, func: str, arg: LetExprNode
) -> tuple[str, str]:
    """Emit the reduction of each row of a two-level nested tower.

    A tower over two binders is a vector of row vectors, so its last
    axis is each row: the broadcast call `sum.(t)` reduces every row
    and keeps one number per row.

    Parameters
    ----------
    ctx : _JlState
        The emit state.
    func : str
        The QVR reduction.
    arg : LetExprNode
        The tower's name.

    Returns
    -------
    tuple[str, str]
        The broadcast call's vertex and kind.

    Raises
    ------
    UnsupportedConstruct
        If the tower nests more than two levels, whose last axis lies
        below a row.
    """
    if infer_array_rank(ctx, arg) != 2:
        raise UnsupportedConstruct(
            f"qvr-{ctx.target}-helper",
            [
                f"let-expr:axis-reduction:{ctx.target}: `{func}(...)` reduces the "
                "last axis of a factor over more than two binders, which the "
                "nested vector form has no single broadcast for"
            ],
        )
    rendered = _render(ctx, arg)
    callee_vid, _callee_kind = _emit_identifier(ctx, _AXIS_REDUCING_CALLS[func])
    al_vid = _emit_argument_list(ctx, (rendered,))
    vid = ctx.v(ctx.fresh("ce"), "broadcast_call_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
    ctx.e(vid, callee_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "broadcast_call_expression"


def _emit_axis_reduction(
    ctx: _JlState, func: str, arg: LetExprNode, rank: int
) -> tuple[str, str]:
    """Emit a reduction of the last axis of a ``rank``-axis ``arg``.

    Julia's `sum(x)` reduces every axis to a scalar and `@. sum(x)`
    reduces none of them, so a last-axis reduction has to name the
    axis: `sum(x, dims = rank)` keeps the array rank and `dropdims`
    removes the collapsed axis.

    The argument itself renders in dotted mode: it is the elementwise
    body the reduction consumes (`z_row .* w_row`), and the caller has
    already declined to wrap the binding in ``@.``.

    Parameters
    ----------
    ctx : _JlState
        The emit state.
    func : str
        The QVR reduction.
    arg : LetExprNode
        The argument.
    rank : int
        Its number of axes; the last is reduced.

    Returns
    -------
    tuple[str, str]
        The reduction's vertex and kind.
    """
    julia_func = _AXIS_REDUCING_CALLS[func]
    outer = ctx.dotted
    ctx.dotted = True
    rendered = _render(ctx, arg)
    ctx.dotted = outer
    reduced = _emit_keyword_call(
        ctx,
        julia_func,
        rendered,
        (("dims", _emit_literal(ctx, rank, integral=True)),),
    )
    return _emit_keyword_call(
        ctx,
        "dropdims",
        reduced,
        (("dims", _emit_literal(ctx, rank, integral=True)),),
    )


def _emit_argument_list(ctx: _JlState, rendered: tuple[tuple[str, str], ...]) -> str:
    """Emit an `argument_list` with the right comma fingerprint.

    Julia's `argument_list` fingerprint is ``( )`` for zero args,
    ``( , )`` for one arg, ``( , , )`` for two args, etc. (one comma
    per argument including the closing one); the panproto pretty
    printer reads the fingerprint to pick the right grammar
    alternative.
    """
    vid = ctx.v(ctx.fresh("al"), "argument_list")
    n = len(rendered)
    if n == 0:
        ctx.constraint(vid, "chose-alt-fingerprint", "()")
        ctx.lit(vid, "()")
        return vid
    fingerprint = "( " + " ".join("," for _ in range(n)) + " )"
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.e(vid, child_vid, "child_of")
    return vid


def _emit_index(ctx: _JlState, expr: LetExprIndex) -> tuple[str, str]:
    """Emit `arr[i0, i1, ...]` as an `index_expression` whose children
    are ``<arr> <vector_expression of indices>``.

    Julia's tree-sitter grammar represents subscripting as
    `arr[i,j]` -> `index_expression(arr, vector_expression(i, j))`
    where the inner `vector_expression` carries the comma-separated
    index list. Array callees whose vertex kind is outside
    `_JL_INDEX_CALLEE_KINDS`
    must be wrapped in `parenthesized_expression`; otherwise the
    pretty-printer drops them and the `index_expression` collapses.

    When the subscripted name's declared plate carries more axes than
    the subscript supplies indices, the remainder is a row gather and
    each residual axis is spelled with an explicit ``:``. Without it
    `Z_mat[item_idx]` against a 32-by-2 matrix is a *linear* index
    into the 64 entries rather than a gather of 2-vectors, which is a
    different (and silently finite) quantity.
    """
    arr_vid, arr_kind = _render(ctx, expr.array)
    if arr_kind not in _JL_INDEX_CALLEE_KINDS:
        arr_vid, arr_kind = _force_paren(ctx, (arr_vid, arr_kind))
    inner_rendered = tuple(
        _render(
            ctx,
            i
            if reads_index_covariate(i, ctx.index_covariates)
            else one_based_subscript(i),
        )
        for i in expr.indices
    )
    if isinstance(expr.array, LetExprVar) and expr.array.name in ctx.nested_names:
        current = (arr_vid, arr_kind)
        for one in inner_rendered:
            current = _emit_subscript(ctx, current, (one,))
        return current
    outer = (
        ctx.row_arrays.get(expr.array.name)
        if isinstance(expr.array, LetExprVar)
        else None
    )
    if outer is not None and len(inner_rendered) > outer:
        rows = _emit_subscript(ctx, (arr_vid, arr_kind), inner_rendered[:outer])
        return _emit_subscript(ctx, rows, inner_rendered[outer:])
    residual = _residual_index_axes(ctx, expr)
    inner_rendered += tuple(_emit_operator(ctx, ":") for _ in range(residual))
    return _emit_subscript(ctx, (arr_vid, arr_kind), inner_rendered)


def _emit_subscript(
    ctx: _JlState,
    array: tuple[str, str],
    indices: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    """Build one `index_expression` from a rendered array and index
    list."""
    arr_vid, arr_kind = array
    inner_vid, _inner_kind = _emit_vector(ctx, indices)
    vid = ctx.v(ctx.fresh("ix"), "index_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        f"{arr_kind} vector_expression",
    )
    ctx.e(vid, arr_vid, "child_of")
    ctx.e(vid, inner_vid, "child_of")
    return vid, "index_expression"


def _emit_range(ctx: _JlState, lower: int, upper: int) -> tuple[str, str]:
    """Emit the inclusive index range ``<lower>:<upper>``.

    Both bounds arrive in QVR's zero-based origin and are lifted to
    Julia's one-based one by the caller.
    """
    vid = ctx.v(ctx.fresh("rng"), "range_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", ":")
    ctx.constraint(vid, "chose-alt-child-kinds", "integer_literal integer_literal")
    lo_vid, _lo_kind = _emit_literal(ctx, lower, integral=True)
    hi_vid, _hi_kind = _emit_literal(ctx, upper, integral=True)
    ctx.e(vid, lo_vid, "child_of")
    ctx.e(vid, hi_vid, "child_of")
    return vid, "range_expression"


def _emit_row_block(
    ctx: _JlState,
    array: LetExprNode,
    offset: int,
    rows: int,
    *,
    trailing_colon: bool,
) -> tuple[str, str]:
    """Emit ``<array>[lo:hi]``, or ``<array>[lo:hi, :]`` for the
    rank-2 weight, in Julia's one-based inclusive origin.
    """
    arr = _render(ctx, array)
    if arr[1] not in _JL_INDEX_CALLEE_KINDS:
        arr = _force_paren(ctx, arr)
    block = _emit_range(ctx, offset + 1, offset + rows)
    indices = (block, _emit_operator(ctx, ":")) if trailing_colon else (block,)
    return _emit_subscript(ctx, arr, indices)


def _emit_conditioning_row(
    ctx: _JlState, sources: tuple[LetAffineSource, ...]
) -> tuple[str, str]:
    """Emit the map's conditioning row: the factors stacked in
    declaration order with Julia's `vcat`.

    A one-factor row is the factor itself, so the common case emits
    no call at all.
    """
    if not sources:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprAffineMap:{_target(ctx)}: the map's "
                f"conditioning row carries no factors"
            ],
        )
    rendered = tuple(_render(ctx, source.value) for source in sources)
    if len(rendered) == 1:
        return rendered[0]
    callee_vid, _callee_kind = _emit_identifier(ctx, "vcat")
    al_vid = _emit_argument_list(ctx, rendered)
    vid = ctx.v(ctx.fresh("ce"), "call_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
    ctx.e(vid, callee_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "call_expression"


def _emit_infix(
    ctx: _JlState,
    op: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> tuple[str, str]:
    """Emit a `binary_expression` over two already-rendered operands
    with a literal operator, bypassing `ctx.dotted`."""
    left_vid, left_kind = _maybe_paren(ctx, left)
    op_vid, _op_kind = _emit_operator(ctx, op)
    right_vid, right_kind = _maybe_paren(ctx, right)
    vid = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        f"{left_kind} operator {right_kind}",
    )
    ctx.e(vid, left_vid, "child_of")
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, right_vid, "child_of")
    return vid, "binary_expression"


def _emit_affine_map(ctx: _JlState, expr: LetExprAffineMap) -> tuple[str, str]:
    """Emit one head's row block of ``W x + b`` as a matrix-vector
    product.

    Julia's ``*`` on a matrix and a vector is exactly the contraction
    the map denotes, so the whole head is one product rather than a
    row per codomain coordinate. The operators are written literally
    rather than through `ctx.dotted`: the product must stay a
    contraction, the bias add is vector-plus-vector, and only the
    `exp` link broadcasts, which it does through its own
    `broadcast_call_expression`.
    """
    outer = ctx.dotted
    ctx.dotted = False
    try:
        total = _emit_infix(
            ctx,
            "+",
            _emit_infix(
                ctx,
                "*",
                _emit_row_block(
                    ctx,
                    expr.weight,
                    expr.row_offset,
                    expr.rows,
                    trailing_colon=True,
                ),
                _emit_conditioning_row(ctx, expr.sources),
            ),
            _emit_row_block(
                ctx,
                expr.bias,
                expr.row_offset,
                expr.rows,
                trailing_colon=False,
            ),
        )
    finally:
        ctx.dotted = outer
    if expr.transform != "exp":
        return total
    callee_vid, _callee_kind = _emit_identifier(ctx, "exp")
    al_vid = _emit_argument_list(ctx, (total,))
    vid = ctx.v(ctx.fresh("ce"), "broadcast_call_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
    ctx.e(vid, callee_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "broadcast_call_expression"


def _residual_index_axes(ctx: _JlState, expr: LetExprIndex) -> int:
    """Count the axes a subscript leaves unindexed.

    Answers only for a subscript whose array is a bare name with a
    known rank in `ctx.name_array_rank`; anything else returns 0, so
    the emitted subscript stays exactly as written. A name the table
    does not carry has no declared plate to compare against, and
    inventing residual axes for it would produce a subscript the
    source never asked for.
    """
    if not isinstance(expr.array, LetExprVar):
        return 0
    rank = ctx.name_array_rank.get(expr.array.name)
    if rank is None:
        return 0
    return max(0, rank - len(expr.indices))


def _emit_vector(
    ctx: _JlState, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit a `vector_expression` ``[e0, e1, ...]`` list literal.

    Fingerprint is ``[ ]`` for empty, ``[ , ]`` for one element,
    ``[ , , ]`` for two elements, etc. (one comma per element
    including the closing one).
    """
    vid = ctx.v(ctx.fresh("ve"), "vector_expression")
    n = len(rendered)
    if n == 0:
        ctx.constraint(vid, "chose-alt-fingerprint", "[ ]")
        return vid, "vector_expression"
    fingerprint = "[ " + " ".join("," for _ in range(n)) + " ]"
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.e(vid, child_vid, "child_of")
    return vid, "vector_expression"


def _emit_lambda(ctx: _JlState, expr: LetExprLambda) -> tuple[str, str]:
    """Emit ``param -> body`` as an `arrow_function_expression`.

    The Julia grammar's single-parameter form uses a bare `identifier`
    for the parameter; the multi-parameter form wraps the parameter
    list in an `argument_list`. [`LetExprLambda`][quivers.dsl.ast_nodes.LetExprLambda]
    only carries a single `param` field, so the single-parameter form
    is the only shape the helper emits.
    """
    param_vid, param_kind = _emit_identifier(ctx, expr.param)
    body_vid, body_kind = _render(ctx, expr.body)
    vid = ctx.v(ctx.fresh("af"), "arrow_function_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "->")
    ctx.constraint(vid, "chose-alt-child-kinds", f"{param_kind} {body_kind}")
    ctx.e(vid, param_vid, "child_of")
    ctx.e(vid, body_vid, "child_of")
    return vid, "arrow_function_expression"


def _emit_method_call(ctx: _JlState, expr: LetExprMethodCall) -> tuple[str, str]:
    """Emit ``receiver.method(args...)`` as a `call_expression` whose
    callee is a `field_expression`.

    Julia is multiple-dispatch; the dot-syntax form ``a.b(c, d)``
    parses as a `call_expression` with a `field_expression` callee.
    The helper preserves the source surface (rather than rewriting to
    the equivalent ``b(a, c, d)`` Julia function-call form) so that
    format-preserving round-trips of QVR programs reach the same
    Julia surface they were authored in.
    """
    receiver_vid, receiver_kind = _render(ctx, expr.receiver)
    method_vid, _method_kind = _emit_identifier(ctx, expr.method)
    field_vid = ctx.v(ctx.fresh("fe"), "field_expression")
    ctx.constraint(field_vid, "chose-alt-fingerprint", ".")
    ctx.constraint(
        field_vid,
        "chose-alt-child-kinds",
        f"{receiver_kind} identifier",
    )
    ctx.e(field_vid, receiver_vid, "child_of")
    ctx.e(field_vid, method_vid, "child_of")
    rendered = tuple(_render(ctx, a) for a in expr.args)
    al_vid = _emit_argument_list(ctx, rendered)
    vid = ctx.v(ctx.fresh("ce"), "call_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        "field_expression argument_list",
    )
    ctx.e(vid, field_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "call_expression"


def _render_factor(ctx: _JlState, expr: LetExprFactor) -> tuple[str, str]:
    """Unroll a `LetExprFactor` into nested `vector_expression` vertices.

    The cases form (binders contain a single axis, body is `None`,
    cases enumerate labels in `[0, |axis|)`) emits a
    `vector_expression` whose children are each case's body in label
    order.

    The uniform-body form (one or more binders, body is the repeated
    expression, cases is empty) emits a tower of `vector_expression`
    vertices of shape `(|b0|, |b1|, ..., |bn-1|)`, with each binder
    substituted for its 1-indexed integer value (Julia arrays are
    1-indexed).
    """
    if not expr.binders:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: empty binder "
                f"list is structurally ill-formed"
            ],
        )
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprFactor:{_target(ctx)}: cases form "
                    f"requires exactly one binder; got "
                    f"{len(expr.binders)}"
                ],
            )
        size = _factor_axis_size(ctx, expr.binders[0])
        return _emit_factor_cases(ctx, expr.binders[0], size, expr.cases)
    if expr.body is not None and not expr.cases:
        sizes = tuple(_factor_axis_size(ctx, b) for b in expr.binders)
        return _build_nested_vector(ctx, expr.binders, sizes, expr.body, ())
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprFactor:{_target(ctx)}: mixed "
            f"cases-plus-body form is not a valid surface construct"
        ],
    )


def _emit_factor_cases(
    ctx: _JlState,
    binder: LetFactorBinder,
    size: int,
    cases: tuple[LetFactorCase, ...],
) -> tuple[str, str]:
    """Build `[<case[0].value>, ..., <case[size-1].value>]` from the
    label-keyed case list."""
    by_label = {c.label: c.value for c in cases}
    missing = sorted(set(range(size)) - by_label.keys())
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: cases form "
                f"missing labels {missing} for binder "
                f"{binder.var!r} of size {size}"
            ],
        )
    rendered = tuple(_render(ctx, by_label[label]) for label in range(size))
    return _emit_vector(ctx, rendered)


def _build_nested_vector(
    ctx: _JlState,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> tuple[str, str]:
    """Recursive helper that materialises the nested `vector_expression`
    tower for the uniform-body factor form.

    Returns ``(vertex_id, vertex_kind)`` so the outer call site can
    populate ``chose-alt-child-kinds`` with the right child kinds.
    """
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            subst = _substitute(
                subst, binder.var, LetExprLiteral(value=value, integral=True)
            )
        return _render(ctx, subst)
    level = len(fixed)
    rendered: list[tuple[str, str]] = []
    for i in range(sizes[level]):
        rendered.append(_build_nested_vector(ctx, binders, sizes, body, fixed + (i,)))
    return _emit_vector(ctx, tuple(rendered))


# ---------------------------------------------------------------------------
# Factor-binder support: axis-size lookup and body substitution.
# ---------------------------------------------------------------------------


def _factor_axis_size(ctx: _JlState, binder: LetFactorBinder) -> int:
    """Resolve a factor binder's axis to a static integer size.

    Looks up the binder's index expression in ``ctx.cards``. Raises
    `UnsupportedConstruct`
    when the axis is a constructor with no static size or when the
    name is unknown.
    """
    name = _object_expr_axis_name(ctx, binder.index)
    size = ctx.cards.get(name)
    if size is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: axis "
                f"{name!r} (binder {binder.var!r}) has no static "
                f"cardinality in `ctx.cards`"
            ],
        )
    return int(size)


def _object_expr_axis_name(ctx: _JlState, obj: ObjectExpr) -> str:
    """Resolve an `ObjectExpr` to the axis name a `cards` lookup wants.

    Handles `TypeName` directly and
    `DiscreteConstructor("FinSet", N)` as a literal anonymous axis
    (the size is the integer arg).
    """
    if isinstance(obj, TypeName):
        return obj.name
    if isinstance(obj, DiscreteConstructor) and obj.constructor == "FinSet":
        if len(obj.args) == 1 and obj.args[0].isdigit():
            return obj.args[0]
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: "
                f"non-literal FinSet binder {obj.args!r}"
            ],
        )
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprFactor:{_target(ctx)}: binder "
            f"index {type(obj).__name__} not supported"
        ],
    )


def _substitute(expr: LetExprNode, name: str, value: LetExprNode) -> LetExprNode:
    """Capture-avoiding substitution of every free occurrence of
    `LetExprVar(name=name)` in `expr` with `value`.

    Mirrors the shared substitution walk in `_stan_helpers`; the two
    helpers carry independent copies so the Stan helper can stay tied
    to its Stan-specific imports while the Julia helper stays tied
    to its Julia-specific protocol ctx.
    """
    if isinstance(expr, LetExprVar):
        return value if expr.name == name else expr
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprString):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute(expr.left, name, value),
            right=_substitute(expr.right, name, value),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_substitute(expr.operand, name, value),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(_substitute(a, name, value) for a in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute(expr.array, name, value),
            indices=tuple(_substitute(i, name, value) for i in expr.indices),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(_substitute(i, name, value) for i in expr.items),
        )
    if isinstance(expr, LetExprLambda):
        if expr.param == name:
            return expr
        return LetExprLambda(
            param=expr.param,
            body=_substitute(expr.body, name, value),
        )
    if isinstance(expr, LetExprFactor):
        if any(b.var == name for b in expr.binders):
            return expr
        return LetExprFactor(
            binders=expr.binders,
            body=(
                _substitute(expr.body, name, value) if expr.body is not None else None
            ),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute(c.value, name, value),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute(expr.receiver, name, value),
            method=expr.method,
            args=tuple(_substitute(a, name, value) for a in expr.args),
        )
    raise UnsupportedConstruct(
        "qvr-let-substitution",
        [f"let-expr:{type(expr).__name__}: substitution unhandled"],
    )


def _target(ctx: _JlState) -> str:
    """Read the ctx's `target` tag for error messages."""
    return ctx.target


def rebase_index_literals(nodes: tuple[IRNode, ...]) -> tuple[IRNode, ...]:
    """Shift every literal subscript in an IR body from QVR's 0-based
    origin to Julia's 1-based one.

    Only the `indices` slots of an
    [`IRArgRef`][quivers.transpile.ir.IRArgRef] move: the same
    constant in a scalar slot (`gated_rate = z * rate`) is a value,
    not a position, and shifting it would change the density. Used on
    a marginalize atom's scope, where the latent has been pinned to
    the integer naming its support point and that integer reaches
    both kinds of slot.
    """
    return tuple(_rebase_node(node) for node in nodes)


def _rebase_node(node: IRNode) -> IRNode:
    """Rebase the literal subscripts of one IR node's arguments."""
    if isinstance(node, IRObserve):
        return IRObserve(
            name=node.name,
            family=node.family,
            args=tuple(_rebase_arg(a) for a in node.args),
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
            via=node.via,
        )
    if isinstance(node, IRSample):
        return IRSample(
            name=node.name,
            family=node.family,
            args=tuple(_rebase_arg(a) for a in node.args),
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
        )
    return node


def _rebase_arg(arg: IRArg) -> IRArg:
    """Rebase the literal subscripts inside one argument."""
    if isinstance(arg, IRArgRef):
        return IRArgRef(
            name=arg.name,
            indices=tuple(_rebase_index(i) for i in arg.indices),
        )
    if isinstance(arg, IRArgList):
        return IRArgList(elements=tuple(_rebase_arg(e) for e in arg.elements))
    return arg


def _rebase_index(index: IRArg) -> IRArg:
    """One subscript position, shifted by the Julia index origin."""
    if isinstance(index, IRArgNumber):
        return IRArgNumber(value=index.value + 1.0)
    return _rebase_arg(index)


def nested_tower_names(ir: IRProgram) -> frozenset[str]:
    """Names the Julia emit materialises as a nested vector tower.

    A [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] with more
    than one binder unrolls into `vector_expression` literals nested
    one level per binder, because the Julia grammar's dense
    `matrix_expression` separates its columns with whitespace alone
    and a synthesised schema carries no interstitial text to place it.
    A subscript into such a name thus reads one axis at a time.
    """
    out: set[str] = set()
    _walk_for_towers(ir.body, out)
    return frozenset(out)


def _walk_for_towers(body: tuple[IRNode, ...], out: set[str]) -> None:
    """Collect multi-binder factor bindings over an IR body."""
    for node in body:
        if isinstance(node, IRDeterministic) and isinstance(node.expr, LetExprFactor):
            if len(node.expr.binders) > 1 and node.expr.body is not None:
                out.add(node.name)
        elif isinstance(node, IRMarginalize):
            _walk_for_towers(node.scope, out)


def name_array_rank_map(ir: IRProgram) -> dict[str, int]:
    """Map every IR-bound name to its full Julia array rank.

    The rank is ``len(plate.batch_dims) + len(plate.event_dims)``: a
    Julia array materialises both plate halves as ordinary axes, batch
    outermost, so `Z_mat` with a 32-wide `Item` batch dim and a 2-wide
    `LatentDim` event dim is a 32-by-2 `Matrix`. The let-expression
    walk consults the table to tell a full subscript from a row
    gather, which decides whether the emitted index needs trailing
    ``:`` axes.
    """
    out: dict[str, int] = {}
    for inp in ir.inputs:
        out[inp.name] = _plate_rank(inp.plate)
    _walk_for_array_ranks(ir.body, out)
    return out


def _plate_rank(plate: Plate) -> int:
    """Full array rank of a plate: batch axes plus event axes."""
    return len(plate.batch_dims) + len(plate.event_dims)


def _walk_for_array_ranks(body: tuple[IRNode, ...], out: dict[str, int]) -> None:
    """Accumulate array ranks over an IR body, descending into every
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scope."""
    for node in body:
        if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
            out[node.name] = _plate_rank(node.plate)
        elif isinstance(node, IRMarginalize):
            out[node.latent] = _plate_rank(node.plate)
            _walk_for_array_ranks(node.scope, out)
        elif isinstance(node, IRScore):
            out[node.name] = 0


__all__ = [
    "JuliaShapes",
    "infer_array_rank",
    "let_expr_has_axis_reduction",
    "name_array_rank_map",
    "nested_tower_names",
    "rebase_index_literals",
    "render_let_expr_julia",
]
