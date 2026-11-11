"""Render `LetExprNode` to Stan tree-sitter schema vertices.

The Stan grammar exposes the following expression-level vertex
kinds the helper builds:

* `real_literal` / `integer_literal` for numeric leaves
* `variable_expression` wrapping an `identifier` for variable refs
* `infix_op_expression` (per-operator alts via `chose-alt-fingerprint`)
* `prefix_op_expression` for unary minus
* `function_expression` with `name`-edged `identifier` callee and a
  `argument_list` child whose fingerprint encodes the comma count
* `indexed_expression` with `[ ]` fingerprint, `variable_expression`
  callee, and `index`-wrapped index children
* `array_expression` for `{e0, e1, ...}` list literals

Every vertex sets `chose-alt-child-kinds` to the space-separated
sequence of its children's kinds; the pretty-printer uses this to
disambiguate grammar productions and silently drops vertices whose
constraint is missing or stale. The helper returns
`(vertex_id, kind)` from every recursive call so parents can build
the `chose-alt-child-kinds` string from real child kinds.

Some `LetExprNode` kinds do not map to Stan user-program syntax
(strings, lambdas, method calls). The helper raises
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
with a precise kind rather than emitting a fake placeholder token.
`LetExprFactor` is unrolled at render time: the cases form becomes
an `array_expression` whose children are the case bodies in label
order; the uniform-body multi-binder form becomes nested
`array_expression` vertices populated by substituting each binder
through its axis's static cardinality (looked up via the
`_StanLetCtx`'s `cards` map sourced from
[`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
"""

from __future__ import annotations

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
from quivers.dsl.ast_nodes.objects import TypeName
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._builtin_spellings import native_path
from quivers.transpile.renderers._one_based import one_based_subscript
from quivers.transpile.renderers._ranks import let_expr_rank
from quivers.transpile.ir import LetAffineSource, LetExprAffineMap


#: The pointwise two-argument builtins whose Stan spelling for real
#: operands differs from the integer one.
_REAL_POINTWISE: dict[str, str] = {"max": "fmax", "min": "fmin"}


def _integral(expr: LetExprNode) -> bool:
    """Whether an expression is integer-valued by its form.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    bool
        ``True`` for an integral literal, integer arithmetic on
        integral operands, and a ``max`` / ``min`` of integral operands;
        ``False`` for anything whose integrality the form does not
        show, which then takes the real spelling.
    """
    if isinstance(expr, LetExprLiteral):
        return expr.integral
    if isinstance(expr, LetExprBinOp):
        return expr.op in ("+", "-", "*", "%") and (
            _integral(expr.left) and _integral(expr.right)
        )
    if isinstance(expr, LetExprUnaryOp):
        return _integral(expr.operand)
    if isinstance(expr, LetExprCall):
        return (
            expr.func in _REAL_POINTWISE
            and len(expr.args) == 2
            and all(_integral(arg) for arg in expr.args)
        )
    return False


def render_let_expr_stan(ctx, expr: LetExprNode) -> str:
    """Build a Stan expression schema for `expr` in `ctx`. Returns
    the root vertex id.

    Wraps `_render` to discard the kind return value at the public
    boundary so callers see the same signature as the other
    per-target helpers.
    """
    vid, _kind = _render(ctx, expr)
    return vid


def _render(ctx, expr: LetExprNode) -> tuple[str, str]:
    """Recursive renderer returning ``(vertex_id, vertex_kind)`` so
    parents can populate ``chose-alt-child-kinds`` accurately."""
    if isinstance(expr, LetExprLiteral):
        return _emit_literal(ctx, expr.value, integral=expr.integral)
    if isinstance(expr, LetExprVar):
        return _emit_variable_expression(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_infix(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_prefix(ctx, expr)
    if isinstance(expr, LetExprCall):
        # `sum(<a> * <b>)` over two vector operands maps cleanly to
        # Stan's `dot_product(<a>, <b>)`; emitting it as
        # `sum(a * b)` makes stanc reject `vector * vector` (Stan
        # treats `*` as matrix-style multiply, which is ambiguous for
        # two same-length vectors).
        if (
            expr.func == "sum"
            and len(expr.args) == 1
            and isinstance(expr.args[0], LetExprBinOp)
            and expr.args[0].op == "*"
            and _rank(ctx, expr.args[0].left) == 1
            and _rank(ctx, expr.args[0].right) == 1
        ):
            inner = expr.args[0]
            return _emit_call_rendered(
                ctx,
                "dot_product",
                (_vector_operand(ctx, inner.left), _vector_operand(ctx, inner.right)),
            )
        if (
            expr.func in _REAL_POINTWISE
            and len(expr.args) == 2
            and not all(_integral(arg) for arg in expr.args)
        ):
            # Stan's `max` / `min` of two scalars take integers only;
            # the pointwise real forms are `fmax` / `fmin`.
            return _emit_function_expression(ctx, _REAL_POINTWISE[expr.func], expr.args)
        return _emit_function_expression(ctx, expr.func, expr.args)
    if isinstance(expr, LetExprIndex):
        return _emit_indexed(ctx, expr)
    if isinstance(expr, LetExprAffineMap):
        return _emit_affine_map(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_array_expression(
            ctx, tuple(_render(ctx, item) for item in expr.items)
        )
    if isinstance(expr, LetExprFactor):
        return _render_factor(ctx, expr)
    if isinstance(expr, LetExprString):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprString: Stan has no string literal "
                "in expression position"
            ],
        )
    if isinstance(expr, LetExprLambda):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprLambda: Stan has no anonymous "
                "function syntax in user-program expression position"
            ],
        )
    if isinstance(expr, LetExprMethodCall):
        # Stan has no `receiver.method(args)` dispatch syntax. The
        # principled emission for a deduction-receiver method call
        # such as `chart.goal_weight()` is a per-deduction
        # `functions { ... }` block defining `parse` and
        # `goal_weight` as user-defined Stan functions that compute
        # the inside-algorithm log-Z over the rule weight vector.
        # Two structural prerequisites block that emission:
        #
        # 1. `deduction_decl` belongs to
        #    `CATEGORICAL_METADATA_IGNORABLE`,
        #    so the IR pipeline elides deductions before the
        #    renderer runs. The atoms, rules, and lexicon needed to
        #    build the inside DP table are not present in the IR
        #    seen here, and there is no IR shape for the chart or
        #    its learnable rule-weight vector.
        # 2. PCFG inside requires a token-sequence input. The
        #    `parse(D, sentence)` callsite's `sentence` parameter
        #    types as `Real` (a scalar), so even given the rule
        #    weights the chart-parser graft has no input shape to
        #    dimension the DP table over spans.
        #
        # Rewriting `m.f(a)` as a static `f(m, a)` call without
        # supplying the `f` definition produces an undefined-
        # function reference that `stanc` rejects, so the helper
        # raises instead of emitting a placeholder.
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprMethodCall:stan: Stan has no method "
                "dispatch syntax; the chart-parser deduction graft "
                "that would supply the called function as a Stan "
                "`functions { ... }` block requires (a) plumbing "
                "`DeductionDecl` through the IR (currently dropped "
                "by `CATEGORICAL_METADATA_IGNORABLE`), and (b) a "
                "token-sequence input shape (the fixture's "
                "`sentence : Real` is a scalar)"
            ],
        )
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


def _emit_literal(ctx, value: float, *, integral: bool) -> tuple[str, str]:
    """Emit a `real_literal` or `integer_literal` vertex.

    The literal keeps the type the source gave it: an integral
    literal (a source integer, a factor binder's value, a subscript)
    is a Stan `int`, and every other literal is a Stan `real` spelled
    with its decimal point. Stan's `/` truncates when both operands
    are `int`, so `1.0 / 4.0` must reach it as two reals.

    Parameters
    ----------
    ctx : _StanLetCtx
        The render context.
    value : float
        The literal's value.
    integral : bool
        Whether the literal is an integer.

    Returns
    -------
    tuple[str, str]
        The vertex and its kind.
    """
    if integral:
        vid = ctx.vertex(ctx.fresh("il"), "integer_literal")
        ctx.literal(vid, str(int(value)))
        return vid, "integer_literal"
    vid = ctx.vertex(ctx.fresh("rl"), "real_literal")
    ctx.literal(vid, repr(float(value)))
    return vid, "real_literal"


def _emit_variable_expression(ctx, name: str) -> tuple[str, str]:
    """Emit a `variable_expression` wrapping an `identifier`."""
    vid = ctx.vertex(ctx.fresh("vex"), "variable_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier")
    ident = ctx.vertex(ctx.fresh("id"), "identifier")
    ctx.literal(ident, name)
    ctx.edge(vid, ident, "child_of")
    return vid, "variable_expression"


_STAN_PAREN_REQUIRED_OPERAND_KINDS: frozenset[str] = frozenset(
    {
        "infix_op_expression",
        "prefix_op_expression",
    }
)
"""Operand kinds that must be wrapped in `parenthized_expression`
when they appear as a sub-expression of a binary or unary operator
in Stan. Stan's printer emits operands left-to-right without
re-grouping, so `c * (theta_1 - theta_0)` would otherwise print as
`c * theta_1 - theta_0` and re-parse as `(c * theta_1) - theta_0`.

Wrapping `prefix_op_expression` operands keeps `-(-x)` from
collapsing to `--x` (which Stan rejects)."""


def _stan_paren(ctx, rendered: tuple[str, str]) -> tuple[str, str]:
    """Wrap `rendered` in a `parenthized_expression` vertex. Caller
    must check
    `_STAN_PAREN_REQUIRED_OPERAND_KINDS`
    before calling."""
    vid, kind = rendered
    paren = ctx.vertex(ctx.fresh("paren"), "parenthized_expression")
    ctx.constraint(paren, "chose-alt-fingerprint", "( )")
    ctx.constraint(paren, "chose-alt-child-kinds", kind)
    ctx.edge(paren, vid, "child_of")
    return paren, "parenthized_expression"


def _stan_maybe_paren(ctx, rendered: tuple[str, str]) -> tuple[str, str]:
    """Wrap `rendered` in a `parenthized_expression` if its kind is in
    `_STAN_PAREN_REQUIRED_OPERAND_KINDS`;
    otherwise return it unchanged."""
    _vid, kind = rendered
    if kind not in _STAN_PAREN_REQUIRED_OPERAND_KINDS:
        return rendered
    return _stan_paren(ctx, rendered)


def _rank(ctx, expr: LetExprNode) -> int:
    """The number of axes an expression's value carries.

    Parameters
    ----------
    ctx : _StanLetCtx
        The render context, whose `ranks` map gives each bound name's
        rank.
    expr : LetExprNode
        The expression.

    Returns
    -------
    int
        Its rank under QVR's shape rules.
    """
    return let_expr_rank(expr, ctx.ranks)


def _vector_operand(ctx, expr: LetExprNode) -> tuple[str, str]:
    """Render a one-axis operand as a Stan `vector`.

    QVR arithmetic on a one-axis value is elementwise. Stan defines
    elementwise arithmetic on `vector` but not on `array[] real`, so an
    operand that is not declared a `vector` is converted with
    `to_vector`, which accepts either.

    Parameters
    ----------
    ctx : _StanLetCtx
        The render context, whose `vectors` set names the bindings
        declared as Stan vectors.
    expr : LetExprNode
        A one-axis operand.

    Returns
    -------
    tuple[str, str]
        The rendered operand and its kind.
    """
    rendered = _render(ctx, expr)
    if isinstance(expr, LetExprVar) and expr.name in ctx.vectors:
        return rendered
    if rendered[1] == "infix_op_expression":
        # An elementwise operation over a one-axis operand is already
        # a vector, because its own operands were converted.
        return rendered
    return _emit_call_rendered(ctx, "to_vector", (rendered,))


#: The arithmetic operators QVR applies elementwise.
_ELEMENTWISE_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})


def _emit_infix(ctx, expr: LetExprBinOp) -> tuple[str, str]:
    """Emit an `infix_op_expression` for a binary operator.

    Both operands are routed through
    `_stan_maybe_paren`
    so a binary sub-expression keeps its grouping (Stan's printer
    emits operands left-to-right without re-grouping; without
    explicit parens `(theta_1 - theta_0)` as the right operand of
    `c * (...)` would re-parse as `(c * theta_1) - theta_0` and
    silently change the meaning).

    Arithmetic over a one-axis operand is elementwise in QVR. Stan
    spells it on `vector` operands, with `.*` for the product of two
    vectors and `./` for a quotient whose divisor is a vector, so each
    one-axis operand is rendered as a vector.

    Raises
    ------
    UnsupportedConstruct
        If an arithmetic operand carries two or more axes, which Stan
        has no elementwise arithmetic for.
    """
    if expr.op in _ELEMENTWISE_OPERATORS:
        left_rank = _rank(ctx, expr.left)
        right_rank = _rank(ctx, expr.right)
        if max(left_rank, right_rank) >= 2:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    f"let-expr:LetExprBinOp:stan: `{expr.op}` over an "
                    f"operand with {max(left_rank, right_rank)} axes has no "
                    f"elementwise Stan form outside a loop over its rows"
                ],
            )
        if max(left_rank, right_rank) == 1:
            op = expr.op
            if op == "*" and left_rank == 1 and right_rank == 1:
                op = ".*"
            if op == "/" and right_rank == 1:
                op = "./"
            left = (
                _vector_operand(ctx, expr.left)
                if left_rank == 1
                else _render(ctx, expr.left)
            )
            right = (
                _vector_operand(ctx, expr.right)
                if right_rank == 1
                else _render(ctx, expr.right)
            )
            return _emit_infix_rendered(ctx, op, left, right)
    return _emit_infix_rendered(
        ctx, expr.op, _render(ctx, expr.left), _render(ctx, expr.right)
    )


def _emit_infix_rendered(
    ctx,
    op: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> tuple[str, str]:
    """Emit an `infix_op_expression` over two already-rendered
    operands, parenthesising each where Stan's printer needs it."""
    left_vid, left_kind = _stan_maybe_paren(ctx, left)
    right_vid, right_kind = _stan_maybe_paren(ctx, right)
    vid = ctx.vertex(ctx.fresh("bin"), "infix_op_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", op)
    ctx.constraint(vid, "chose-alt-child-kinds", f"{left_kind} {right_kind}")
    ctx.edge(vid, left_vid, "child_of")
    ctx.edge(vid, right_vid, "child_of")
    return vid, "infix_op_expression"


def _emit_prefix(ctx, expr: LetExprUnaryOp) -> tuple[str, str]:
    """Emit a `prefix_op_expression` for the unary minus.

    The operand is routed through
    `_stan_maybe_paren`
    so a nested unary or binary operand keeps its grouping; without
    that wrap, `-(-x)` would print as `--x`, which Stan's lexer
    rejects, and `-(a + b)` would print as `-a + b` (i.e.
    `(-a) + b`).
    """
    operand_vid, operand_kind = _stan_maybe_paren(ctx, _render(ctx, expr.operand))
    vid = ctx.vertex(ctx.fresh("uop"), "prefix_op_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "-")
    ctx.constraint(vid, "chose-alt-child-kinds", operand_kind)
    ctx.edge(vid, operand_vid, "child_of")
    return vid, "prefix_op_expression"


def _emit_function_expression(
    ctx, func: str, args: tuple[LetExprNode, ...]
) -> tuple[str, str]:
    """Emit a `function_expression` with `name` edge to the callee
    identifier and `child_of` edge to the `argument_list`.

    A pointwise math builtin is called by the Stan function
    [`native_path`][quivers.transpile._builtin_spellings.native_path]
    names (`sigmoid` is `inv_logit`, ...); any other name is Stan's own.
    """
    path = native_path("stan", func, len(args))
    return _emit_call_rendered(
        ctx,
        func if path is None else ".".join(path),
        tuple(_render(ctx, a) for a in args),
    )


def _emit_call_rendered(
    ctx, func: str, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit a `function_expression` calling `func` on operands that
    are already rendered.

    `func` is the Stan stdlib identifier; a caller that starts from a
    QVR name spells it through `_emit_function_expression`.
    """
    vid = ctx.vertex(ctx.fresh("call"), "function_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier argument_list")
    fn = ctx.vertex(ctx.fresh("fn"), "identifier")
    ctx.literal(fn, func)
    ctx.edge(vid, fn, "name")
    al_vid = _emit_argument_list(ctx, rendered)
    ctx.edge(vid, al_vid, "child_of")
    return vid, "function_expression"


def _emit_argument_list(ctx, rendered: tuple[tuple[str, str], ...]) -> str:
    """Emit an `argument_list` with the right comma fingerprint and
    child-kinds string."""
    vid = ctx.vertex(ctx.fresh("args"), "argument_list")
    if rendered:
        fingerprint = "( " + ", ".join(["" for _ in rendered]) + " )"
        # Stan's grammar prints the fingerprint as `( , , )` with
        # N-1 commas for N args (one comma between each pair).
        fingerprint = "( " + ", ".join("" for _ in rendered).rstrip() + " )"
        # Build the canonical form: "( )" for one arg, "( , )" for
        # two args, "( , , )" for three args, etc.
        if len(rendered) == 1:
            fingerprint = "( )"
        else:
            commas = ", " * (len(rendered) - 1)
            fingerprint = f"( {commas.rstrip()} )"
    else:
        fingerprint = "( )"
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered) or "",
    )
    for child_vid, _kind in rendered:
        ctx.edge(vid, child_vid, "child_of")
    return vid


_STAN_INDEXED_CALLEE_KINDS: frozenset[str] = frozenset(
    {
        "variable_expression",
        "function_expression",
        "indexed_expression",
        "parenthized_expression",
        "array_expression",
    }
)
"""Stan grammar `indexed_expression` accepts a narrow set of array
callee kinds. Anything else (`infix_op_expression`, a literal, ...)
must be wrapped in `parenthized_expression` for the printer to
accept it; otherwise emit_pretty silently drops the entire
indexed expression and prints `[]`."""


def _emit_indexed(ctx, expr: LetExprIndex) -> tuple[str, str]:
    """Emit an `indexed_expression` (the `arr[i][j]...` form).

    When `expr.array` resolves to a kind Stan's `indexed_expression`
    production does not accept directly (every kind outside
    `_STAN_INDEXED_CALLEE_KINDS`),
    wrap it in `parenthized_expression` so the printer keeps the
    subtree intact instead of bailing to `[]`.

    Every subscript is valued in QVR's zero-based origin, and this is
    the one place it is translated to Stan's one-based one, through
    [`one_based_subscript`][quivers.transpile.renderers._one_based.one_based_subscript].
    A Stan loop counter reaches an expression as its QVR value
    `counter - 1` and so subscripts as the counter itself; an index
    covariate or a drawn class is a QVR value and is read one higher.
    """
    arr_vid, arr_kind = _render(ctx, expr.array)
    if arr_kind not in _STAN_INDEXED_CALLEE_KINDS:
        paren = ctx.vertex(ctx.fresh("paren"), "parenthized_expression")
        ctx.constraint(paren, "chose-alt-fingerprint", "( )")
        ctx.constraint(paren, "chose-alt-child-kinds", arr_kind)
        ctx.edge(paren, arr_vid, "child_of")
        arr_vid = paren
        arr_kind = "parenthized_expression"
    index_vids: list[str] = []
    child_kinds: list[str] = [arr_kind]
    for idx in expr.indices:
        inner_vid, inner_kind = _render(ctx, one_based_subscript(idx))
        wrap = ctx.vertex(ctx.fresh("idx"), "index")
        ctx.constraint(wrap, "chose-alt-child-kinds", inner_kind)
        ctx.edge(wrap, inner_vid, "child_of")
        index_vids.append(wrap)
        child_kinds.append("index")
    vid = ctx.vertex(ctx.fresh("ix"), "indexed_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "[ ]")
    ctx.constraint(vid, "chose-alt-child-kinds", " ".join(child_kinds))
    ctx.edge(vid, arr_vid, "child_of")
    for wrap in index_vids:
        ctx.edge(vid, wrap, "child_of")
    return vid, "indexed_expression"


def _emit_row_block(ctx, array: LetExprNode, offset: int, rows: int) -> tuple[str, str]:
    """Emit ``<array>[lo:hi]``, one head's contiguous row block.

    `offset` and `rows` arrive in QVR's zero-based origin; Stan
    indexes from one and its ``lo:hi`` slice is inclusive at both
    ends, so the block spans ``offset + 1`` to ``offset + rows``.
    """
    arr_vid, arr_kind = _render(ctx, array)
    if arr_kind not in _STAN_INDEXED_CALLEE_KINDS:
        arr_vid, arr_kind = _stan_paren(ctx, (arr_vid, arr_kind))
    colon = ctx.vertex(ctx.fresh("colon"), "colon_expression")
    ctx.constraint(colon, "chose-alt-fingerprint", ":")
    ctx.constraint(
        colon,
        "chose-alt-child-kinds",
        "integer_literal integer_literal",
    )
    lo_vid, _lo_kind = _emit_literal(ctx, float(offset + 1), integral=True)
    hi_vid, _hi_kind = _emit_literal(ctx, float(offset + rows), integral=True)
    ctx.edge(colon, lo_vid, "child_of")
    ctx.edge(colon, hi_vid, "child_of")
    wrap = ctx.vertex(ctx.fresh("idx"), "index")
    ctx.constraint(wrap, "chose-alt-child-kinds", "colon_expression")
    ctx.edge(wrap, colon, "child_of")
    vid = ctx.vertex(ctx.fresh("ix"), "indexed_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "[ ]")
    ctx.constraint(vid, "chose-alt-child-kinds", f"{arr_kind} index")
    ctx.edge(vid, arr_vid, "child_of")
    ctx.edge(vid, wrap, "child_of")
    return vid, "indexed_expression"


def _emit_conditioning_row(
    ctx, sources: tuple[LetAffineSource, ...]
) -> tuple[str, str]:
    """Emit the map's conditioning row: the factors concatenated in
    declaration order, as one Stan `vector`.

    Each factor is coerced with `to_vector` so a `vector`-typed
    object and an ``array[] real``-typed one stack the same way, and
    the factors are joined with `append_row`.
    """
    row: tuple[str, str] | None = None
    for source in sources:
        column = _emit_call_rendered(ctx, "to_vector", (_render(ctx, source.value),))
        row = (
            column
            if row is None
            else _emit_call_rendered(ctx, "append_row", (row, column))
        )
    if row is None:
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprAffineMap:stan: the map's "
                "conditioning row carries no factors"
            ],
        )
    return row


def _emit_affine_map(ctx, expr: LetExprAffineMap) -> tuple[str, str]:
    """Emit one head's row block of ``W x + b`` as a matrix-vector
    product.

    Stan's ``*`` on a `matrix` and a `vector` is exactly the
    contraction the map denotes, so the whole head is one product
    rather than a row per codomain coordinate. `to_matrix` and
    `to_vector` coerce the ``array[,] real`` / ``array[] real``
    shapes the map's data inputs arrive on, `exp` is Stan's
    vectorised exponential, and `to_array_1d` converts the resulting
    `vector` back to the ``array[N] real`` the binding is declared
    at.
    """
    total = _emit_infix_rendered(
        ctx,
        "+",
        _emit_infix_rendered(
            ctx,
            "*",
            _emit_call_rendered(
                ctx,
                "to_matrix",
                (_emit_row_block(ctx, expr.weight, expr.row_offset, expr.rows),),
            ),
            _emit_conditioning_row(ctx, expr.sources),
        ),
        _emit_call_rendered(
            ctx,
            "to_vector",
            (_emit_row_block(ctx, expr.bias, expr.row_offset, expr.rows),),
        ),
    )
    if expr.transform == "exp":
        total = _emit_call_rendered(ctx, "exp", (total,))
    return _emit_call_rendered(ctx, "to_array_1d", (total,))


def _emit_array_expression(
    ctx, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit an `array_expression` ``{e0, e1, ...}`` list literal."""
    vid = ctx.vertex(ctx.fresh("arr"), "array_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.edge(vid, child_vid, "child_of")
    return vid, "array_expression"


def _render_factor(ctx, expr: LetExprFactor) -> tuple[str, str]:
    """Unroll a `LetExprFactor` into nested `array_expression`
    vertices.

    The cases form (binders contain a single axis, body is None,
    cases enumerate labels in [0, |axis|)) emits an
    `array_expression` whose children are each case's body in
    label order.

    The uniform-body form (one or more binders, body is the
    repeated expression, cases is empty) emits a tower of
    `array_expression` vertices of shape
    `(|b0|, |b1|, ..., |bn-1|)`. The shared
    `_substitute_let_expr`
    walk receives the binder's zero-based value, an integer literal,
    in both slots, since
    `_emit_indexed`
    is the one place that lifts a subscript to Stan's one-based
    origin.
    """
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    "let-expr:LetExprFactor: cases form requires "
                    f"exactly one binder; got {len(expr.binders)}"
                ],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        rendered = tuple(_render(ctx, c.value) for c in ordered)
        return _emit_array_expression(ctx, rendered)
    if expr.body is not None and not expr.cases:
        sizes = tuple(_card_for(ctx, b) for b in expr.binders)
        return _build_nested_array(ctx, expr.binders, sizes, expr.body, ())
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [
            "let-expr:LetExprFactor: mixed cases-plus-body form "
            "is not a valid surface construct"
        ],
    )


def _card_for(ctx, binder: LetFactorBinder) -> int:
    """Resolve the static cardinality of `binder.index`.

    `LetExprFactor` only unrolls when every binder's axis has a
    statically-known size; the helper consults `ctx.cards`
    (populated from
    [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
    """
    idx = binder.index
    if isinstance(idx, TypeName):
        cards = getattr(ctx, "cards", None)
        if cards is None or idx.name not in cards:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    f"let-expr:LetExprFactor: binder {binder.var!r} "
                    f"references object {idx.name!r} whose cardinality "
                    "is unknown at render time"
                ],
            )
        return cards[idx.name]
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [
            f"let-expr:LetExprFactor: binder {binder.var!r} index is "
            f"{type(idx).__name__}; only TypeName binders unroll"
        ],
    )


def _build_nested_array(
    ctx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> tuple[str, str]:
    """Recursive helper that materialises the nested
    `array_expression` tower for the uniform-body factor form.

    Returns ``(vertex_id, vertex_kind)`` so the outer call site can
    populate ``chose-alt-child-kinds`` with the right child kinds.
    """
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            subst = _substitute_let_expr(
                subst,
                binder.var,
                index_value=LetExprLiteral(value=float(value), integral=True),
                scalar_value=LetExprLiteral(value=float(value), integral=True),
            )
        return _render(ctx, subst)
    level = len(fixed)
    rendered: list[tuple[str, str]] = []
    for i in range(sizes[level]):
        rendered.append(_build_nested_array(ctx, binders, sizes, body, fixed + (i,)))
    return _emit_array_expression(ctx, tuple(rendered))


def _substitute_let_expr(
    expr: LetExprNode,
    name: str,
    *,
    index_value: LetExprNode,
    scalar_value: LetExprNode,
) -> LetExprNode:
    """Capture-avoiding, context-aware substitution of every free
    occurrence of `LetExprVar(name=name)` in `expr`.

    Two replacement values are required because backends with
    1-based array indexing (Stan) want the binder substituted with
    its 1-indexed integer in *index slots* (`arr[v]` -> `arr[1]`
    for `v=0`) yet with its 0-indexed integer everywhere else
    (`2 * v` -> `2 * 0` for `v=0`). Backends with 0-based
    indexing (NumPyro, Pyro, PyMC, Edward2, JavaScript, WebPPL)
    pass the same value for both arguments, since QVR's surface
    semantics agrees with the host language in every slot.

    The walk distinguishes "index slot" exactly when it recurses
    into the `indices` tuple of a
    [`LetExprIndex`][quivers.dsl.ast_nodes.LetExprIndex] node; the
    `array` child of the same node and every other position is a
    scalar slot. Nested indexing (`arr[idx[v]]`) re-classifies the
    inner `v` as an index slot because it lives inside the inner
    `LetExprIndex.indices` tuple, which matches Stan's semantics
    where every integer fed to `[ ]` is 1-based.

    Shared substitution helper for every per-target renderer that
    needs to unroll [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    by binding indices to integer literals. Lives in `_stan_helpers`
    because Stan was the first target to need it; other helper
    modules import from here when they grow the same need (one
    source of truth for the walk).
    """
    return _substitute_let_expr_walk(
        expr,
        name,
        index_value=index_value,
        scalar_value=scalar_value,
        in_index_slot=False,
    )


def _substitute_let_expr_walk(
    expr: LetExprNode,
    name: str,
    *,
    index_value: LetExprNode,
    scalar_value: LetExprNode,
    in_index_slot: bool,
) -> LetExprNode:
    """Inner walk for
    `_substitute_let_expr`
    that carries the `in_index_slot` flag.

    The flag is set to `True` only when descending into the
    `indices` tuple of a `LetExprIndex`; every other recursive
    descent (including the `array` child of `LetExprIndex` itself)
    resets the flag to `False`. This matches Stan's grammar: the
    body of `arr[i]` is `arr` in scalar position and `i` in index
    position, and the helper substitutes accordingly.
    """
    if isinstance(expr, LetExprVar):
        if expr.name != name:
            return expr
        return index_value if in_index_slot else scalar_value
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprString):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute_let_expr_walk(
                expr.left,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
            right=_substitute_let_expr_walk(
                expr.right,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_substitute_let_expr_walk(
                expr.operand,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _substitute_let_expr_walk(
                    a,
                    name,
                    index_value=index_value,
                    scalar_value=scalar_value,
                    in_index_slot=False,
                )
                for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute_let_expr_walk(
                expr.array,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
            indices=tuple(
                _substitute_let_expr_walk(
                    i,
                    name,
                    index_value=index_value,
                    scalar_value=scalar_value,
                    in_index_slot=True,
                )
                for i in expr.indices
            ),
        )
    if isinstance(expr, LetExprAffineMap):
        return LetExprAffineMap(
            weight=_substitute_let_expr_walk(
                expr.weight,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
            bias=_substitute_let_expr_walk(
                expr.bias,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
            sources=tuple(
                LetAffineSource(
                    value=_substitute_let_expr_walk(
                        source.value,
                        name,
                        index_value=index_value,
                        scalar_value=scalar_value,
                        in_index_slot=False,
                    ),
                    width=source.width,
                )
                for source in expr.sources
            ),
            row_offset=expr.row_offset,
            rows=expr.rows,
            transform=expr.transform,
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(
                _substitute_let_expr_walk(
                    i,
                    name,
                    index_value=index_value,
                    scalar_value=scalar_value,
                    in_index_slot=False,
                )
                for i in expr.items
            ),
        )
    if isinstance(expr, LetExprLambda):
        if expr.param == name:
            return expr
        return LetExprLambda(
            param=expr.param,
            body=_substitute_let_expr_walk(
                expr.body,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
        )
    if isinstance(expr, LetExprFactor):
        if any(b.var == name for b in expr.binders):
            return expr
        return LetExprFactor(
            binders=expr.binders,
            body=(
                _substitute_let_expr_walk(
                    expr.body,
                    name,
                    index_value=index_value,
                    scalar_value=scalar_value,
                    in_index_slot=False,
                )
                if expr.body is not None
                else None
            ),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute_let_expr_walk(
                        c.value,
                        name,
                        index_value=index_value,
                        scalar_value=scalar_value,
                        in_index_slot=False,
                    ),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute_let_expr_walk(
                expr.receiver,
                name,
                index_value=index_value,
                scalar_value=scalar_value,
                in_index_slot=False,
            ),
            method=expr.method,
            args=tuple(
                _substitute_let_expr_walk(
                    a,
                    name,
                    index_value=index_value,
                    scalar_value=scalar_value,
                    in_index_slot=False,
                )
                for a in expr.args
            ),
        )
    raise UnsupportedConstruct(
        "qvr-let-substitution",
        [f"let-expr:{type(expr).__name__}: substitution unhandled"],
    )


__all__ = ["render_let_expr_stan"]
