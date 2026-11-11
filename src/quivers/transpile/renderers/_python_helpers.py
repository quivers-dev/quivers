"""Shared SchemaBuilder helpers for the Python tree-sitter grammar.

Used by every renderer whose target is Python source (NumPyro, Pyro,
PyMC, Edward2). Each helper takes a builder context and produces a
vertex id for the constructed sub-AST. Vertex kinds match Python's
tree-sitter `node-types.json` exactly.

This module also exposes
`render_let_expr_python`
which lowers a [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode]
sub-tree into the same Python schema, used by every Python-grammar
renderer to emit `let <name> = <expr>` as a deterministic
`assignment` inside the model body.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import didactic.api as dx

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
from quivers.dsl.pure_builtins import EAGER_BUILTIN_NAMES
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.family_meta import FAMILY_META, marginalize_support
from quivers.transpile.ir import (
    CSIntegerInterval,
    Dim,
    LetAffineSource,
    LetExprAffineMap,
    IRArg,
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
from quivers.transpile._builtin_spellings import native_import, native_path
from quivers.transpile.renderers._ranks import (
    name_plate_map,
    reduces_last_of_several_axes,
)
from quivers.transpile.renderers._stan_helpers import (
    _substitute_let_expr,
)

if TYPE_CHECKING:
    import panproto


class PyCtx:
    """Owns a `panproto.SchemaBuilder` plus
    a fresh-id counter and the per-render
    [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards] map.

    `cards` is consulted when unrolling
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] binders;
    every backend that wires the IR-walk into this ctx is expected
    to pass `cards` at construction.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        cards: dict[str, int] | None = None,
        target: str = "python",
        name_event_rank: dict[str, int] | None = None,
        gather_symbol: tuple[str, ...] | None = None,
        factor_towers: frozenset[str] | None = None,
        name_plates: dict[str, Plate] | None = None,
        name_rank: dict[str, int] | None = None,
    ) -> None:
        self._sb = sb
        self._n = 0
        self.cards: dict[str, int] = dict(cards or {})
        # The concrete Python backend ("pyro" / "numpyro" / "pymc" /
        # "edward2"); selects the per-target builtin symbol table when a
        # ``LetExprCall`` is lowered.
        self.target = target
        # Dotted callee for the target's array-gather primitive (e.g.
        # ``("tf", "gather")``). When set, a subscript whose single
        # index is a plate-vector name renders as
        # ``gather(array, index)`` instead of ``array[index]``:
        # TensorFlow's tensor / Edward2 RandomVariable ``__getitem__``
        # rejects a vector index, so advanced-index gathers must route
        # through ``tf.gather``. ``None`` keeps the native subscript
        # form (JAX / PyTorch / PyTensor all support vector subscripts).
        self.gather_symbol: tuple[str, ...] | None = gather_symbol
        # Per-name event rank read from the IR: ``len(plate.event_dims)``
        # for every IRDataInput / IRSample / IRObserve / IRDeterministic.
        # The let-expression walk infers the event rank of compound
        # subexpressions from this base; a reducing call (``sum``,
        # ``mean``, ...) whose argument has positive inferred rank emits
        # the backend's per-axis aggregator with an ``axis=-1`` (or
        # ``dim=-1``) keyword instead of the scalar Python builtin.
        self.name_event_rank: dict[str, int] = dict(name_event_rank or {})
        # Names bound to a `LetExprFactor` tower. The tower is
        # materialised as a nested Python list, one level per binder,
        # so a subscript into it chains one index per level
        # (``t[i][j]``) where a subscript into a numeric array would
        # spell the same access as a single multi-index (``t[i, j]``).
        self.factor_towers: frozenset[str] = factor_towers or frozenset()
        # Per-name `Plate` read from the IR. The marginalize
        # enumeration reads the batch dims to decide whether a weight
        # tensor needs the observe's `via` gather, and the trailing
        # event extent to size a `class_index` atom set.
        self.name_plates: dict[str, Plate] = dict(name_plates or {})
        # Per-name rank read from the IR: the batch and event axes of
        # every bound name together. A reducing call whose argument has
        # two or more axes reduces its last one, as the checked
        # elaboration and the compiled runtime do, so it takes the
        # target's axis keyword even when no axis is an event axis.
        self.name_rank: dict[str, int] = dict(name_rank or {})
        # Imports a lowered ``LetExprCall`` symbol requires, as
        # ``(dotted-chain, alias)`` pairs. Only NumPyro emits an import
        # block, so only it drains this; the other backends assume their
        # framework runtime is already in scope.
        self.required_imports: set[tuple[tuple[str, ...], str]] = set()

    @property
    def builder(self) -> panproto.SchemaBuilder:
        """The schema builder the context writes vertices into."""
        return self._sb

    def fresh(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str) -> None:
        self._sb.edge(src, tgt, kind)

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


def identifier(ctx: PyCtx, text: str) -> str:
    """Emit an ``identifier`` vertex carrying ``text``."""
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.literal(vid, text)
    return vid


def string_literal(ctx: PyCtx, text: str) -> str:
    """Emit a double-quoted ``string`` vertex with ``text`` as its content."""
    s = ctx.v(ctx.fresh("s"), "string")
    start = ctx.v(ctx.fresh("ss"), "string_start")
    ctx.literal(start, '"')
    content = ctx.v(ctx.fresh("sc"), "string_content")
    ctx.literal(content, text)
    end = ctx.v(ctx.fresh("se"), "string_end")
    ctx.literal(end, '"')
    ctx.e(s, start, "child_of")
    ctx.e(s, content, "child_of")
    ctx.e(s, end, "child_of")
    return s


def number_literal(ctx: PyCtx, value: float) -> str:
    """Emit ``integer`` if ``value`` is a whole number, else ``float``."""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        vid = ctx.v(ctx.fresh("int"), "integer")
        ctx.literal(vid, str(int(value)))
    else:
        vid = ctx.v(ctx.fresh("flt"), "float")
        ctx.literal(vid, repr(float(value)))
    return vid


def float_literal(ctx: PyCtx, value: float) -> str:
    """Emit a ``float`` vertex, keeping the decimal point on a whole
    number.

    `number_literal`
    prints a whole-valued float as an integer, which is the right
    spelling almost everywhere. It is the wrong spelling for the value
    that fills a real-valued tensor: `tf.fill` and `np.full` read the
    fill value's own type, so `Dirichlet(1.0)` broadcast to a
    concentration vector would build an integer tensor and TFP rejects
    an integer concentration outright.
    """
    vid = ctx.v(ctx.fresh("flt"), "float")
    ctx.literal(vid, repr(float(value)))
    return vid


def arg_expr(ctx: PyCtx, raw: str | float) -> str:
    """Build an expression vertex for a sample argument."""
    if isinstance(raw, str):
        return identifier(ctx, raw)
    return number_literal(ctx, raw)


def attribute(ctx: PyCtx, chain: tuple[str, ...]) -> str:
    """Build a left-recursive ``a.b.c.d`` attribute access.

    Tree-sitter Python represents `a.b.c` as ``attribute(object:
    attribute(object: id 'a', attribute: 'b'), attribute: 'c')``.
    """
    if len(chain) < 2:
        msg = f"attribute needs at least 2 names; got {chain!r}"
        raise ValueError(msg)
    current = identifier(ctx, chain[0])
    for attr_name in chain[1:]:
        attr = ctx.v(ctx.fresh("attr"), "attribute")
        attr_id = identifier(ctx, attr_name)
        ctx.e(attr, current, "object")
        ctx.e(attr, attr_id, "attribute")
        current = attr
    return current


def call(
    ctx: PyCtx,
    function: str,
    *,
    positional: tuple[str, ...] = (),
    keyword: tuple[tuple[str, str], ...] = (),
) -> str:
    """Build a ``call`` vertex with positional and keyword args."""
    c = ctx.v(ctx.fresh("call"), "call")
    args = ctx.v(ctx.fresh("args"), "argument_list")
    ctx.e(c, function, "function")
    ctx.e(c, args, "arguments")
    for pid in positional:
        ctx.e(args, pid, "child_of")
    for name, vid in keyword:
        kw = ctx.v(ctx.fresh("kw"), "keyword_argument")
        kw_name = identifier(ctx, name)
        ctx.e(kw, kw_name, "name")
        ctx.e(kw, vid, "value")
        ctx.e(args, kw, "child_of")
    return c


def python_binary_op(
    ctx: PyCtx,
    op: str,
    left: str,
    right: str,
) -> str:
    """Build a ``binary_operator`` vertex carrying ``op`` as the
    field:operator constraint.

    The Python tree-sitter `binary_operator` has CHOICE alternatives
    per operator; both the field constraint and the alt fingerprint
    must be set so the pretty printer picks the right surface form.
    """
    b = ctx.v(ctx.fresh("bop"), "binary_operator")
    ctx.constraint(b, "field:operator", op)
    ctx.constraint(b, "chose-alt-fingerprint", op)
    ctx.e(b, left, "left")
    ctx.e(b, right, "right")
    return b


def python_unary_minus(ctx: PyCtx, operand: str) -> str:
    """Build a ``unary_operator`` vertex for ``-<operand>``."""
    u = ctx.v(ctx.fresh("uop"), "unary_operator")
    ctx.constraint(u, "field:operator", "-")
    ctx.constraint(u, "chose-alt-fingerprint", "-")
    ctx.e(u, operand, "argument")
    return u


def python_paren(ctx: PyCtx, inner: str) -> str:
    """Build a ``parenthesized_expression`` vertex around ``inner``.

    The Python pretty printer drops parens around nested
    binary_operator children, which scrambles operator precedence;
    callers that depend on explicit grouping wrap subexpressions in
    parenthesized_expression to force the printer to emit ``( ... )``.
    """
    p = ctx.v(ctx.fresh("paren"), "parenthesized_expression")
    ctx.constraint(p, "chose-alt-fingerprint", "( )")
    ctx.e(p, inner, "child_of")
    return p


def python_method_call(
    ctx: PyCtx,
    receiver: str,
    method: str,
    args: tuple[str, ...],
) -> str:
    """Build ``<receiver>.<method>(<args>)``."""
    a = ctx.v(ctx.fresh("attr"), "attribute")
    ctx.e(a, receiver, "object")
    method_id = identifier(ctx, method)
    ctx.e(a, method_id, "attribute")
    c = ctx.v(ctx.fresh("call"), "call")
    arglist = ctx.v(ctx.fresh("args"), "argument_list")
    ctx.e(c, a, "function")
    ctx.e(c, arglist, "arguments")
    for arg in args:
        ctx.e(arglist, arg, "child_of")
    return c


def assignment(ctx: PyCtx, *, lhs_name: str, rhs: str) -> str:
    """Build ``<lhs_name> = <rhs>``."""
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    lhs = identifier(ctx, lhs_name)
    ctx.e(asn, lhs, "left")
    ctx.e(asn, rhs, "right")
    return asn


def function_def(
    ctx: PyCtx,
    *,
    name: str,
    default_params: tuple[str, ...],
    body_vid: str,
) -> str:
    """Build ``def <name>(<p1>=None, <p2>=None, ...): <body>``."""
    func = ctx.v(ctx.fresh("fn"), "function_definition")
    fname = identifier(ctx, name)
    params = ctx.v(ctx.fresh("ps"), "parameters")
    ctx.e(func, fname, "name")
    ctx.e(func, params, "parameters")
    ctx.e(func, body_vid, "body")
    for pname in default_params:
        dp = ctx.v(ctx.fresh("dp"), "default_parameter")
        dp_name = identifier(ctx, pname)
        dp_val = ctx.v(ctx.fresh("none"), "none")
        ctx.literal(dp_val, "None")
        ctx.e(dp, dp_name, "name")
        ctx.e(dp, dp_val, "value")
        ctx.e(params, dp, "child_of")
    return func


def with_statement(
    ctx: PyCtx,
    *,
    expression: str,
    alias: str | None,
    body_vid: str,
) -> str:
    """Build ``with <expression> [as <alias>]: <body>``.

    Tree-sitter Python's shape for ``with X as Y: body`` is:

    ```text
    with_statement
      with_clause
        with_item   (value field -> either expression, or as_pattern wrapping it)
      body field    -> block
    ```

    When ``alias`` is set, the ``with_item``'s value field points at
    the as_pattern (which itself owns the expression via its child_of
    edge and the target via its alias field). The
    expression is referenced exactly ONCE in the schema graph; routing
    it under both the with_item's value field and the as_pattern's
    child_of edge would cause `emit_pretty` to traverse it twice and
    emit the call twice (producing
    ``with pymc.Model() pymc.Model() as: ...``).
    """
    ws = ctx.v(ctx.fresh("with"), "with_statement")
    clause = ctx.v(ctx.fresh("wc"), "with_clause")
    item = ctx.v(ctx.fresh("wi"), "with_item")
    if alias is not None:
        as_pat = ctx.v(ctx.fresh("asp"), "as_pattern")
        target = ctx.v(ctx.fresh("astgt"), "as_pattern_target")
        ctx.e(target, identifier(ctx, alias), "child_of")
        ctx.e(as_pat, expression, "child_of")
        ctx.e(as_pat, target, "alias")
        ctx.e(item, as_pat, "value")
    else:
        ctx.e(item, expression, "value")
    ctx.e(clause, item, "child_of")
    ctx.e(ws, clause, "child_of")
    ctx.e(ws, body_vid, "body")
    return ws


#: A non-pointwise builtin's per-target lowering: the symbol path to
#: emit as a dotted attribute, plus the import it needs (``None`` when the
#: symbol lives on a namespace the target already has in scope). The
#: pointwise math builtins are spelled by
#: [`BUILTIN_SPELLINGS`][quivers.transpile._builtin_spellings.BUILTIN_SPELLINGS].
_CallEntry = tuple[tuple[str, ...], "tuple[tuple[str, ...], str] | None"]

#: QVR primitives whose semantics is per-axis reduction over the event
#: axis. When one of these calls sees a single argument of positive
#: inferred event rank, the emit routes to the target's array-aware
#: aggregator (mapped in ``_LET_CALL_SYMBOLS`` below) and appends the
#: reduction-axis keyword. A scalar argument keeps the Python builtin.
_AXIS_REDUCING_CALLS: frozenset[str] = frozenset({"sum", "mean", "prod", "max", "min"})

# Shape-preserving primitives whose semantics fixes the trailing event axis.
_TRAILING_AXIS_CALLS: frozenset[str] = frozenset({"softmax"})

#: Per-target keyword name for the reduction axis. NumPy / JAX / PyMC /
#: TensorFlow spell it ``axis``; PyTorch spells it ``dim``.
_AXIS_KEYWORD_BY_TARGET: dict[str, str] = {
    "pyro": "dim",
    "numpyro": "axis",
    "pymc": "axis",
    "edward2": "axis",
}

#: ``target -> {reduction name -> lowering}`` for the one-argument
#: reductions and ``softmax``. A one-argument ``max`` / ``min`` routes to
#: the reduction that returns a bare array (torch's ``amax`` / ``amin``
#: rather than the ``(values, indices)`` pair).
_LET_CALL_SYMBOLS: dict[str, dict[str, _CallEntry]] = {
    "pyro": {
        "sum": (("torch", "sum"), None),
        "mean": (("torch", "mean"), None),
        "prod": (("torch", "prod"), None),
        "max": (("torch", "amax"), None),
        "min": (("torch", "amin"), None),
        "softmax": (("torch", "softmax"), None),
    },
    "numpyro": {
        "sum": (("jnp", "sum"), None),
        "mean": (("jnp", "mean"), None),
        "prod": (("jnp", "prod"), None),
        "max": (("jnp", "max"), None),
        "min": (("jnp", "min"), None),
        "softmax": (("jnn", "softmax"), (("jax", "nn"), "jnn")),
    },
    "pymc": {
        "sum": (("pymc", "math", "sum"), None),
        "mean": (("pymc", "math", "mean"), None),
        "prod": (("pymc", "math", "prod"), None),
        "max": (("pymc", "math", "max"), None),
        "min": (("pymc", "math", "min"), None),
        "softmax": (("pymc", "math", "softmax"), None),
    },
    "edward2": {
        "sum": (("tf", "reduce_sum"), None),
        "mean": (("tf", "reduce_mean"), None),
        "prod": (("tf", "reduce_prod"), None),
        "max": (("tf", "reduce_max"), None),
        "min": (("tf", "reduce_min"), None),
        "softmax": (("tf", "nn", "softmax"), None),
    },
}

#: ``(stack, convert)``: how a list literal becomes an array. Python
#: arithmetic reads a list as a sequence rather than a vector (`[1.0,
#: 2.0] * mu` repeats or raises, and the product of two lists raises on
#: every target), so the entries are stacked into the target's array,
#: each first converted where the stack refuses a Python number
#: (`convert`); PyTensor's and TensorFlow's stacks promote a number
#: themselves.
_LIST_STACK_SYMBOLS: dict[str, tuple[tuple[str, ...], tuple[str, ...] | None]] = {
    "pyro": (("torch", "stack"), ("torch", "as_tensor")),
    "numpyro": (("jnp", "stack"), ("jnp", "asarray")),
    "pymc": (("pymc", "math", "stack"), None),
    "edward2": (("tf", "stack"), None),
}

#: The array constructor each target's pointwise binary call converts its
#: operands with, for the targets whose elementwise functions reject a
#: host operand: ``torch.maximum`` takes tensors only, and ``jnp.maximum``
#: rejects a Python list, such as a list-literal `let`. An operand may
#: also be a Python number, such as an unrolled factor binder.
_POINTWISE_OPERAND_SYMBOLS: dict[str, tuple[str, ...]] = {
    "pyro": ("torch", "as_tensor"),
    "numpyro": ("jnp", "asarray"),
}

#: The conversion PyMC subscripts a host array through when an index is a
#: symbolic expression, such as a pointwise ``max`` in index arithmetic:
#: a Python list or a NumPy array cannot be indexed by a tensor.
_PYMC_TENSOR: tuple[str, ...] = ("pymc", "pytensorf", "pt", "as_tensor_variable")


def _symbolic_index(expr: LetExprNode) -> bool:
    """Whether an index expression calls a builtin, so PyMC builds it as a tensor.

    Parameters
    ----------
    expr : LetExprNode
        The index expression.

    Returns
    -------
    bool
        ``True`` if a call occurs anywhere in it.
    """
    if isinstance(expr, LetExprCall):
        return True
    if isinstance(expr, LetExprBinOp):
        return _symbolic_index(expr.left) or _symbolic_index(expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return _symbolic_index(expr.operand)
    if isinstance(expr, LetExprIndex):
        return _symbolic_index(expr.array) or any(
            _symbolic_index(index) for index in expr.indices
        )
    return False


#: Eager builtin names come from the same capability registry as the compiler.
_MATH_BUILTIN_NAMES = EAGER_BUILTIN_NAMES


def _resolve_python_call(ctx: PyCtx, func: str, arity: int = 1) -> str:
    """Emit the callee vertex for a ``LetExprCall`` to ``func``.

    A pointwise math builtin emits the function
    [`native_path`][quivers.transpile._builtin_spellings.native_path]
    names as a dotted attribute (e.g. ``torch.erf``), recording any
    import it needs on ``ctx.required_imports``; a two-argument ``max``,
    ``min``, or ``pow`` is pointwise and never reaches a reduction. A
    reduction emits the target's array-aware aggregator. Any other
    builtin raises
    `UnsupportedConstruct`
    rather than emit an undefined name. Any other callee (a domain
    function such as a chart-parser ``parse``, or a user helper) is
    emitted verbatim.

    Parameters
    ----------
    ctx : PyCtx
        The emit context, whose target selects the symbol table and
        whose imports collect the symbol's import.
    func : str
        The called name.
    arity : int
        The number of arguments the call passes.

    Returns
    -------
    str
        The callee vertex.

    Raises
    ------
    UnsupportedConstruct
        If ``func`` is a math builtin with no symbol on the target.
    """
    path = native_path(ctx.target, func, arity)
    if path is not None:
        import_spec = native_import(ctx.target, func, arity)
        if import_spec is not None:
            ctx.required_imports.add(import_spec)
        if len(path) == 1:
            return identifier(ctx, path[0])
        return attribute(ctx, path)
    entry = _LET_CALL_SYMBOLS.get(ctx.target, {}).get(func)
    if entry is not None:
        segments, import_spec = entry
        if import_spec is not None:
            ctx.required_imports.add(import_spec)
        if len(segments) == 1:
            return identifier(ctx, segments[0])
        return attribute(ctx, segments)
    if func in _MATH_BUILTIN_NAMES:
        raise UnsupportedConstruct(
            "qvr-python-helper",
            [
                f"let-expr:LetExprCall:{ctx.target}: builtin {func!r} has "
                f"no {ctx.target} symbol mapping; it cannot be rendered as "
                f"a single call in this target"
            ],
        )
    return identifier(ctx, func)


def _render_python_operand(
    ctx: PyCtx, expr: LetExprNode, *, index_slot: bool = False
) -> str:
    """Render `expr` as an operand of a binary or unary operator.

    A nested [`LetExprBinOp`][quivers.dsl.ast_nodes.LetExprBinOp] or
    [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] operand is
    wrapped in a `parenthesized_expression` via
    `python_paren`.
    The Python pretty printer drops parens around nested
    `binary_operator` children, so ``(a + b) * c`` would otherwise print
    as ``a + b * c`` and reassociate under Python's precedence. Wrapping
    every nested operator preserves the source grouping, matching the
    Stan and Julia renderers.

    `index_slot` rides along so arithmetic inside a subscript
    (``t[i + 1]``) keeps its integer spelling.
    """
    vid = render_let_expr_python(ctx, expr, index_slot=index_slot)
    if isinstance(expr, (LetExprBinOp, LetExprUnaryOp)):
        return python_paren(ctx, vid)
    return vid


def _is_factor_tower(ctx: PyCtx, expr: LetExprNode) -> bool:
    """True iff `expr` denotes a `LetExprFactor` tower, so a subscript
    into it chains one index per nesting level.

    Either the expression is the factor itself, or it names a
    deterministic bound to one (`PyCtx.factor_towers`, populated from
    the IR by
    `factor_tower_names`).
    """
    if isinstance(expr, LetExprFactor):
        return True
    return isinstance(expr, LetExprVar) and expr.name in ctx.factor_towers


#: Each NumPy-like target's flooring division, product, and difference,
#: from which, with the sign and absolute value the shared builtin table
#: spells, its truncating integer division and remainder are built as
#: calls, which need no parentheses wherever they appear.
_TRUNCATION_SYMBOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "numpyro": {
        "floor_divide": ("jnp", "floor_divide"),
        "multiply": ("jnp", "multiply"),
        "subtract": ("jnp", "subtract"),
    },
    "pymc": {
        "floor_divide": ("pymc", "pytensorf", "pt", "int_div"),
        "multiply": ("pymc", "pytensorf", "pt", "mul"),
        "subtract": ("pymc", "pytensorf", "pt", "sub"),
    },
    "edward2": {
        "floor_divide": ("tf", "math", "floordiv"),
        "multiply": ("tf", "math", "multiply"),
        "subtract": ("tf", "math", "subtract"),
    },
}


def _render_truncating(
    ctx: PyCtx, func: str, dividend: LetExprNode, divisor: LetExprNode
) -> str:
    """Render integer division or remainder, which truncate toward zero.

    Python's `//` and `%` floor, and so differ from truncation when the
    operands' signs differ. Pyro uses `torch.div(..., rounding_mode="trunc")`
    and `torch.fmod` on tensors of its operands. The other targets build
    the quotient as ``sign(a) * sign(b) * (|a| // |b|)``, which keeps
    integer operands integral, and the remainder as ``a - b * quot``,
    each spelled as the library's calls.

    Parameters
    ----------
    ctx : PyCtx
        The emit context.
    func : str
        ``"quot"`` or ``"rem"``.
    dividend : LetExprNode
        The dividend.
    divisor : LetExprNode
        The divisor.

    Returns
    -------
    str
        The expression vertex.
    """
    if ctx.target == "pyro":
        operands = tuple(
            call(
                ctx,
                attribute(ctx, ("torch", "as_tensor")),
                positional=(render_let_expr_python(ctx, operand),),
            )
            for operand in (dividend, divisor)
        )
        if func == "rem":
            return call(ctx, attribute(ctx, ("torch", "fmod")), positional=operands)
        return call(
            ctx,
            attribute(ctx, ("torch", "div")),
            positional=operands,
            keyword=(("rounding_mode", string_literal(ctx, "trunc")),),
        )
    symbols = _TRUNCATION_SYMBOLS[ctx.target]

    def apply(name: str, *operands: str) -> str:
        """Call a math builtin by the target's spelling.

        Parameters
        ----------
        name : str
            The builtin.
        *operands : str
            The rendered operands.

        Returns
        -------
        str
            The call's vertex.
        """
        # `sign` and `abs` are math builtins, spelled by the shared table.
        path = native_path(ctx.target, name, len(operands))
        return call(
            ctx,
            attribute(ctx, symbols[name] if path is None else path),
            positional=operands,
        )

    def operand(expr: LetExprNode) -> str:
        """Render one operand.

        Parameters
        ----------
        expr : LetExprNode
            The operand.

        Returns
        -------
        str
            Its vertex.
        """
        return render_let_expr_python(ctx, expr)

    quotient = apply(
        "multiply",
        apply(
            "multiply",
            apply("sign", operand(dividend)),
            apply("sign", operand(divisor)),
        ),
        apply(
            "floor_divide",
            apply("abs", operand(dividend)),
            apply("abs", operand(divisor)),
        ),
    )
    if func == "quot":
        return quotient
    return apply(
        "subtract", operand(dividend), apply("multiply", operand(divisor), quotient)
    )


def render_let_expr_python(
    ctx: PyCtx, expr: LetExprNode, *, index_slot: bool = False
) -> str:
    """Recursively build a Python expression schema for `expr` in
    `ctx` (a `PyCtx`).
    Returns the root vertex id.

    `index_slot` marks the subscript position. A
    [`LetExprLiteral`][quivers.dsl.ast_nodes.LetExprLiteral] carries a
    `float` payload, so a source-level index (`cell_score[0, 0]`)
    arrives as ``0.0``; Python subscripts reject a float, so an
    index-slot literal emits as an integer. A genuinely fractional index
    has no integer spelling and raises instead. Outside a subscript, an
    `integral` literal, such as the value a factor unroll substitutes
    for its `Int` binder, emits as an integer, so index arithmetic built
    from it stays integer-valued wherever it is used.
    """
    if isinstance(expr, LetExprLiteral):
        if index_slot:
            if not float(expr.value).is_integer():
                raise UnsupportedConstruct(
                    "qvr-python-helper",
                    [
                        f"let-expr:LetExprIndex: non-integral literal "
                        f"index {expr.value!r}"
                    ],
                )
            v = ctx.v(ctx.fresh("lit"), "integer")
            ctx.literal(v, str(int(expr.value)))
            return v
        if expr.integral:
            v = ctx.v(ctx.fresh("lit"), "integer")
            ctx.literal(v, str(int(expr.value)))
            return v
        v = ctx.v(ctx.fresh("lit"), "float")
        ctx.literal(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.literal(v, expr.name)
        return v
    if isinstance(expr, LetExprString):
        # Use json.dumps for a fully escaped Python string literal
        # (handles embedded quotes, backslashes, and control chars
        # without letting a payload break out of the quoted region).
        v = ctx.v(ctx.fresh("str"), "string")
        ctx.literal(v, json.dumps(expr.value))
        return v
    if isinstance(expr, LetExprBinOp):
        # Python tree-sitter's `binary_operator` has CHOICE alts per
        # operator (`+`, `-`, `*`, `**`, etc.). The walker discriminates
        # via the `field:operator` constraint on the binary_operator
        # vertex itself; no separate operator vertex is needed.
        b = ctx.v(ctx.fresh("bop"), "binary_operator")
        ctx.constraint(b, "field:operator", expr.op)
        ctx.constraint(b, "chose-alt-fingerprint", expr.op)
        ctx.e(
            b,
            _render_python_operand(ctx, expr.left, index_slot=index_slot),
            "left",
        )
        ctx.e(
            b,
            _render_python_operand(ctx, expr.right, index_slot=index_slot),
            "right",
        )
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("uop"), "unary_operator")
        ctx.constraint(u, "field:operator", "-")
        ctx.constraint(u, "chose-alt-fingerprint", "-")
        ctx.e(
            u,
            _render_python_operand(ctx, expr.operand, index_slot=index_slot),
            "argument",
        )
        return u
    if isinstance(expr, LetExprCall):
        if expr.func == "real" and len(expr.args) == 1 and ctx.target == "edward2":
            return call(
                ctx,
                attribute(ctx, ("tf", "cast")),
                positional=(
                    render_let_expr_python(ctx, expr.args[0]),
                    attribute(ctx, ("tf", "float32")),
                ),
            )
        if expr.func in ("quot", "rem") and len(expr.args) == 2:
            return _render_truncating(ctx, expr.func, expr.args[0], expr.args[1])
        if expr.func == "int" and len(expr.args) == 1:
            argument = render_let_expr_python(ctx, expr.args[0])
            if ctx.target == "pyro":
                tensor = call(
                    ctx,
                    attribute(ctx, ("torch", "as_tensor")),
                    positional=(argument,),
                )
                return python_method_call(ctx, tensor, "long", ())
            if ctx.target == "numpyro":
                return call(
                    ctx,
                    attribute(ctx, ("jnp", "int32")),
                    positional=(argument,),
                )
            if ctx.target == "pymc":
                return call(
                    ctx,
                    attribute(ctx, ("pymc", "pytensorf", "pt", "cast")),
                    positional=(argument, string_literal(ctx, "int64")),
                )
            if ctx.target == "edward2":
                return call(
                    ctx,
                    attribute(ctx, ("tf", "cast")),
                    positional=(argument, attribute(ctx, ("tf", "int32"))),
                )
        fn = _resolve_python_call(ctx, expr.func, len(expr.args))
        c = ctx.v(ctx.fresh("call"), "call")
        ctx.e(c, fn, "function")
        args = ctx.v(ctx.fresh("args"), "argument_list")
        # torch's functions take tensors only, so every operand of a
        # builtin Pyro spells as one is converted: a constant, or the
        # Python number an unrolled factor binder becomes, is a tensor of
        # its own type (`float` to the default dtype, `int` to `int64`).
        # `jnp.maximum` and its kind reject a Python list operand.
        pointwise = native_path(ctx.target, expr.func, len(expr.args)) is not None
        operand = (
            _POINTWISE_OPERAND_SYMBOLS.get(ctx.target)
            if (
                ctx.target == "pyro"
                and (pointwise or expr.func in _LET_CALL_SYMBOLS["pyro"])
            )
            or (pointwise and len(expr.args) == 2)
            else None
        )
        for a in expr.args:
            rendered = render_let_expr_python(ctx, a)
            if operand is not None:
                rendered = call(ctx, attribute(ctx, operand), positional=(rendered,))
            ctx.e(args, rendered, "child_of")
        # A reducing primitive (``sum``, ``mean``, ...) collapses the
        # last axis of its argument. The renamed callee already targets
        # the target's array-aware aggregator, so an argument with an
        # event axis, or with two or more axes of any kind, takes the
        # reduction-axis keyword (``axis=-1`` / ``dim=-1``); a vector
        # with no event axis reduces to the same number either way and
        # keeps the call with no keyword.
        needs_trailing_axis = expr.func in _TRAILING_AXIS_CALLS and len(expr.args) == 1
        reduces_event_axis = (
            expr.func in _AXIS_REDUCING_CALLS
            and len(expr.args) == 1
            and (
                _infer_event_rank(ctx, expr.args[0]) > 0
                or reduces_last_of_several_axes(expr, ctx.name_rank)
            )
        )
        if needs_trailing_axis or reduces_event_axis:
            axis_kw = _AXIS_KEYWORD_BY_TARGET.get(ctx.target, "axis")
            kw = ctx.v(ctx.fresh("kw"), "keyword_argument")
            ctx.e(kw, identifier(ctx, axis_kw), "name")
            ctx.e(kw, python_unary_minus(ctx, number_literal(ctx, 1)), "value")
            ctx.e(args, kw, "child_of")
        ctx.e(c, args, "arguments")
        return c
    if isinstance(expr, LetExprIndex):
        # A single plate-vector index (a bare `LetExprVar`, always a
        # runtime gather covariate in QVR) routes through the target's
        # gather primitive when one is declared, since TensorFlow /
        # Edward2 reject a vector subscript. Every other case keeps the
        # native `array[index]` subscript.
        if (
            ctx.gather_symbol is not None
            and len(expr.indices) == 1
            and isinstance(expr.indices[0], LetExprVar)
        ):
            return call(
                ctx,
                attribute(ctx, ctx.gather_symbol),
                positional=(
                    render_let_expr_python(ctx, expr.array),
                    render_let_expr_python(ctx, expr.indices[0]),
                ),
            )
        array_vid = render_let_expr_python(ctx, expr.array)
        if _is_factor_tower(ctx, expr.array):
            # The tower is a nested Python list, one nesting level per
            # binder, so each index consumes one level: `t[i][j]`. A
            # flat multi-index would be a tuple subscript, which a list
            # rejects.
            current = array_vid
            for idx in expr.indices:
                s = ctx.v(ctx.fresh("subs"), "subscript")
                ctx.e(s, current, "value")
                ctx.e(
                    s,
                    render_let_expr_python(ctx, idx, index_slot=True),
                    "subscript",
                )
                current = s
            return current
        if ctx.target == "pymc" and any(_symbolic_index(i) for i in expr.indices):
            array_vid = call(ctx, attribute(ctx, _PYMC_TENSOR), positional=(array_vid,))
        s = ctx.v(ctx.fresh("subs"), "subscript")
        ctx.e(s, array_vid, "value")
        for idx in expr.indices:
            ctx.e(
                s,
                render_let_expr_python(ctx, idx, index_slot=True),
                "subscript",
            )
        return s
    if isinstance(expr, LetExprList):
        lst = ctx.v(ctx.fresh("list"), "list")
        stack = _LIST_STACK_SYMBOLS.get(ctx.target)
        for item in expr.items:
            rendered = render_let_expr_python(ctx, item)
            if stack is not None and stack[1] is not None:
                rendered = call(ctx, attribute(ctx, stack[1]), positional=(rendered,))
            ctx.e(lst, rendered, "child_of")
        if stack is None or not expr.items:
            return lst
        return call(ctx, attribute(ctx, stack[0]), positional=(lst,))
    if isinstance(expr, LetExprLambda):
        lam = ctx.v(ctx.fresh("lam"), "lambda")
        params = ctx.v(ctx.fresh("ps"), "lambda_parameters")
        pid = ctx.v(ctx.fresh("p"), "identifier")
        ctx.literal(pid, expr.param)
        ctx.e(params, pid, "child_of")
        ctx.e(lam, params, "parameters")
        ctx.e(lam, render_let_expr_python(ctx, expr.body), "body")
        return lam
    if isinstance(expr, LetExprMethodCall):
        a = ctx.v(ctx.fresh("attr"), "attribute")
        ctx.e(a, render_let_expr_python(ctx, expr.receiver), "object")
        m = ctx.v(ctx.fresh("m"), "identifier")
        ctx.literal(m, expr.method)
        ctx.e(a, m, "attribute")
        c = ctx.v(ctx.fresh("call"), "call")
        ctx.e(c, a, "function")
        args = ctx.v(ctx.fresh("args"), "argument_list")
        for a_node in expr.args:
            ctx.e(args, render_let_expr_python(ctx, a_node), "child_of")
        ctx.e(c, args, "arguments")
        return c
    if isinstance(expr, LetExprAffineMap):
        return _render_affine_map_python(ctx, expr)
    if isinstance(expr, LetExprFactor):
        return _render_factor_python(ctx, expr)
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


#: Per-target array-concatenation primitive, as
#: ``(dotted-chain, needs_axis_argument)``. The affine map's
#: conditioning row is the concatenation of its factors, and every
#: Python array library spells that join differently. TensorFlow's
#: `concat` takes the axis positionally; the others default to the
#: leading axis, which is the one a rank-1 row stacks along.
_PY_CONCAT_SYMBOLS: dict[str, tuple[tuple[str, ...], bool]] = {
    "pyro": (("torch", "cat"), False),
    "numpyro": (("jnp", "concatenate"), False),
    "pymc": (("pymc", "math", "concatenate"), False),
    "edward2": (("tf", "concat"), True),
}


#: Targets whose ``@`` operator refuses a rank-1 right operand and so
#: need a named matrix-vector primitive. TensorFlow's `matmul` demands
#: rank two on both sides; JAX, PyTorch and PyTensor all contract a
#: matrix against a vector under ``@`` directly.
_PY_MATVEC_SYMBOLS: dict[str, tuple[str, ...]] = {
    "edward2": ("tf", "linalg", "matvec"),
}


def python_slice(ctx: PyCtx, lower: int, upper: int) -> str:
    """Build a ``<lower>:<upper>`` slice node.

    Python's origin is QVR's own, so the bounds pass through
    unshifted; the upper bound is exclusive in both.
    """
    s = ctx.v(ctx.fresh("slice"), "slice")
    ctx.constraint(s, "chose-alt-fingerprint", ":")
    ctx.constraint(s, "chose-alt-child-kinds", "integer integer")
    ctx.e(s, number_literal(ctx, lower), "child_of")
    ctx.e(s, number_literal(ctx, upper), "child_of")
    return s


def _render_conditioning_row_python(
    ctx: PyCtx, sources: tuple[LetAffineSource, ...]
) -> str:
    """Render the map's conditioning row: the factors concatenated
    along their single axis, in declaration order.

    A one-factor row is the factor itself, so the common case emits
    no call at all.
    """
    if not sources:
        raise UnsupportedConstruct(
            "qvr-python-helper",
            [
                f"let-expr:LetExprAffineMap:{ctx.target}: the map's "
                f"conditioning row carries no factors"
            ],
        )
    rendered = tuple(render_let_expr_python(ctx, source.value) for source in sources)
    if len(rendered) == 1:
        return rendered[0]
    entry = _PY_CONCAT_SYMBOLS.get(ctx.target)
    if entry is None:
        raise UnsupportedConstruct(
            "qvr-python-helper",
            [
                f"let-expr:LetExprAffineMap:{ctx.target}: no array "
                f"concatenation primitive is mapped for this target, "
                f"so a product-domain conditioning row cannot be built"
            ],
        )
    segments, needs_axis = entry
    positional: tuple[str, ...] = (python_list(ctx, rendered),)
    if needs_axis:
        positional = (*positional, number_literal(ctx, 0))
    return call(ctx, attribute(ctx, segments), positional=positional)


def _render_affine_map_python(ctx: PyCtx, expr: LetExprAffineMap) -> str:
    """Render one head's row block of ``W x + b`` as a matmul.

    Most Python array backends spell the contraction ``@``; a target
    listed in
    `_PY_MATVEC_SYMBOLS`
    spells it as a named call instead. Either way the whole head is
    one product rather than a row per codomain coordinate, and the
    row block is a plain slice: Python's index origin is QVR's own,
    and its upper bound is exclusive in both.
    """
    weight = subscript(
        ctx,
        render_let_expr_python(ctx, expr.weight),
        (python_slice(ctx, expr.row_offset, expr.row_offset + expr.rows),),
    )
    bias = subscript(
        ctx,
        render_let_expr_python(ctx, expr.bias),
        (python_slice(ctx, expr.row_offset, expr.row_offset + expr.rows),),
    )
    row = _render_conditioning_row_python(ctx, expr.sources)
    matvec = _PY_MATVEC_SYMBOLS.get(ctx.target)
    product = (
        python_binary_op(ctx, "@", weight, row)
        if matvec is None
        else call(ctx, attribute(ctx, matvec), positional=(weight, row))
    )
    total = python_binary_op(ctx, "+", product, bias)
    if expr.transform == "exp":
        return call(ctx, _resolve_python_call(ctx, "exp"), positional=(total,))
    return total


def _infer_event_rank(ctx: PyCtx, expr: LetExprNode) -> int:
    """Infer the event rank of a let-expression at emit time.

    The rank of a leaf variable is ``len(plate.event_dims)`` for the
    IRDataInput / IRSample / IRObserve / IRDeterministic it references,
    read from `PyCtx.name_event_rank`; compound expressions propagate
    the rank structurally:

    - [`LetExprLiteral`][quivers.dsl.ast_nodes.LetExprLiteral] /
      [`LetExprString`][quivers.dsl.ast_nodes.LetExprString] -> 0.
    - [`LetExprVar`][quivers.dsl.ast_nodes.LetExprVar] ->
      ``ctx.name_event_rank.get(name, 0)``.
    - [`LetExprBinOp`][quivers.dsl.ast_nodes.LetExprBinOp] -> max of the
      operand ranks (broadcasting lifts the lower-rank operand).
    - [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] -> the
      operand rank (negation is shape-preserving).
    - [`LetExprCall`][quivers.dsl.ast_nodes.LetExprCall] to a reducing
      primitive -> 0 (the reduction collapses the event axis); any
      other callee -> max of its arg ranks (treating element-wise math
      like ``sigmoid`` / ``exp`` as shape-preserving).
    - [`LetExprIndex`][quivers.dsl.ast_nodes.LetExprIndex] ->
      ``max(0, arr.rank - len(indices))`` (each scalar index consumes
      one event axis).
    - [`LetExprList`][quivers.dsl.ast_nodes.LetExprList] -> max of the
      element ranks.

    Other node kinds are treated as scalar; the helper is conservative
    and never lifts the rank when it is not certain.
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
        return max(
            (_infer_event_rank(ctx, a) for a in expr.args),
            default=0,
        )
    if isinstance(expr, LetExprIndex):
        arr_rank = _infer_event_rank(ctx, expr.array)
        return max(0, arr_rank - len(expr.indices))
    if isinstance(expr, LetExprList):
        return max(
            (_infer_event_rank(ctx, item) for item in expr.items),
            default=0,
        )
    return 0


def name_event_rank_map(ir: IRProgram) -> dict[str, int]:
    """Map every IR-bound name to its event rank ``len(plate.event_dims)``.

    Covers every [`IRDataInput`][quivers.transpile.ir.IRDataInput],
    [`IRSample`][quivers.transpile.ir.IRSample],
    [`IRObserve`][quivers.transpile.ir.IRObserve], and
    [`IRDeterministic`][quivers.transpile.ir.IRDeterministic] in the
    program. Each Python renderer threads the result into its
    `PyCtx` so the
    let-expression walk can decide whether a reducing call
    (``sum(z_row * w_row)``) collapses a genuine event axis and thus
    needs the backend's per-axis aggregator.
    """
    out: dict[str, int] = {}
    for inp in ir.inputs:
        out[inp.name] = len(inp.plate.event_dims)
    _walk_for_name_ranks(ir.body, out)
    return out


def _walk_for_name_ranks(body: tuple[IRNode, ...], out: dict[str, int]) -> None:
    for node in body:
        if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
            out[node.name] = len(node.plate.event_dims)
        elif isinstance(node, IRMarginalize):
            out[node.latent] = 0
            _walk_for_name_ranks(node.scope, out)
        elif isinstance(node, IRScore):
            out[node.name] = 0


def factor_tower_names(ir: IRProgram) -> frozenset[str]:
    """Names bound to a [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    tower anywhere in `ir`.

    The Python renderers materialise a tower as a nested list literal,
    so a subscript into one of these names chains a single index per
    nesting level rather than spelling a flat multi-index.
    """
    out: set[str] = set()
    _walk_for_factor_towers(ir.body, out)
    return frozenset(out)


def _walk_for_factor_towers(body: tuple[IRNode, ...], out: set[str]) -> None:
    for node in body:
        if isinstance(node, IRDeterministic):
            if isinstance(node.expr, LetExprFactor):
                out.add(node.name)
        elif isinstance(node, IRMarginalize):
            _walk_for_factor_towers(node.scope, out)


#: Per-target array-stacking primitive. A binding whose value is
#: materialised coordinate by coordinate still denotes a vector, and a
#: family argument reads it as one: a Python list reaches a
#: distribution constructor unpromoted, and the library's own
#: arithmetic then refuses to broadcast it against the array-valued
#: arguments beside it. Stacking states the array-ness once, at the
#: binding, rather than at every use.
_PY_STACK_SYMBOLS: dict[str, tuple[str, ...]] = {
    "pyro": ("torch", "stack"),
    "numpyro": ("jnp", "stack"),
    "pymc": ("pymc", "math", "stack"),
    "edward2": ("tf", "stack"),
}


#: Targets whose stack primitive demands an already-array element.
#: `torch.stack` rejects a Python float, and a coordinate built from
#: literals alone renders as one, so PyTorch's elements each go
#: through `torch.as_tensor`: the identity on a tensor (the same
#: object, its gradient history intact) and a promotion on a float.
#: The other three targets promote inside `stack` itself.
_PY_STACK_ELEMENT_SYMBOLS: dict[str, tuple[str, ...]] = {
    "pyro": ("torch", "as_tensor"),
}


def _vector_binding_coordinates(
    ctx: PyCtx, node: IRDeterministic
) -> tuple[LetExprNode, ...] | None:
    """The coordinate expressions of a rank-1 array binding, or `None`
    when the binding is not one.

    A [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] over a
    single binder denotes a vector over that binder's axis, which is
    why its `IRDeterministic` carries exactly one batch dim. Both of
    its forms enumerate the coordinates: the cases form names them,
    and the uniform-body form substitutes the binder for each index
    of the axis. A tower over two or more binders is rank two or
    more, and stacking it would need a per-target nesting rule the
    coordinate list cannot supply, so it stays a nested list literal.
    """
    if len(node.plate.batch_dims) != 1 or node.plate.event_dims:
        return None
    expr = node.expr
    if not isinstance(expr, LetExprFactor) or len(expr.binders) != 1:
        return None
    if expr.cases and expr.body is None:
        ordered = sorted(expr.cases, key=lambda c: c.label)
        return tuple(case.value for case in ordered)
    if expr.body is not None and not expr.cases:
        (binder,) = expr.binders
        coordinates: list[LetExprNode] = []
        for index in range(_card_for(ctx, binder)):
            literal = LetExprLiteral(value=index, integral=True)
            coordinates.append(
                _substitute_let_expr(
                    expr.body,
                    binder.var,
                    index_value=literal,
                    scalar_value=literal,
                )
            )
        return tuple(coordinates)
    return None


def render_deterministic_python(ctx: PyCtx, node: IRDeterministic) -> str:
    """Render the right-hand side of a deterministic let-binding.

    A binding that denotes a vector but is built coordinate by
    coordinate is stacked into a target array, and so is a uniform
    factor over two or more binders, one stack per binder; every other
    binding is its expression rendered directly.

    Parameters
    ----------
    ctx : PyCtx
        The render state.
    node : IRDeterministic
        The binding.

    Returns
    -------
    str
        The vertex of the binding's right-hand side.
    """
    expr = node.expr
    if (
        isinstance(expr, LetExprFactor)
        and len(expr.binders) >= 2
        and expr.body is not None
        and not expr.cases
    ):
        sizes = tuple(_card_for(ctx, binder) for binder in expr.binders)
        return _build_stacked_python(ctx, expr.binders, sizes, expr.body, ())
    coordinates = _vector_binding_coordinates(ctx, node)
    if coordinates is None:
        return render_let_expr_python(ctx, node.expr)
    stack = _target_stack_symbol(ctx, node.name)
    element = _PY_STACK_ELEMENT_SYMBOLS.get(ctx.target)
    rendered: list[str] = []
    for coordinate in coordinates:
        vid = render_let_expr_python(ctx, coordinate)
        if element is not None:
            vid = call(ctx, attribute(ctx, element), positional=(vid,))
        rendered.append(vid)
    return call(
        ctx,
        attribute(ctx, stack),
        positional=(python_list(ctx, tuple(rendered)),),
    )


def _target_stack_symbol(ctx: PyCtx, name: str) -> tuple[str, ...]:
    """The target's array-stacking primitive.

    Parameters
    ----------
    ctx : PyCtx
        The render context.
    name : str
        The binding being stacked, for the diagnostic.

    Returns
    -------
    tuple[str, ...]
        The dotted stacking callee.

    Raises
    ------
    UnsupportedConstruct
        If the target maps no stacking primitive.
    """
    stack = _PY_STACK_SYMBOLS.get(ctx.target)
    if stack is None:
        raise UnsupportedConstruct(
            "qvr-python-helper",
            [
                f"let-expr:LetExprFactor:{ctx.target}: no array "
                f"stacking primitive is mapped for this target, so "
                f"the binding {name!r} cannot be materialised as an array"
            ],
        )
    return stack


def _build_stacked_python(
    ctx: PyCtx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> str:
    """Materialise a uniform factor as nested stacks of target arrays.

    Each binder contributes one stack, outermost first, so the binding
    is an array of shape ``(|b0|, ..., |bn-1|, *body_shape)`` that a
    reduction can collapse along its last axis and a subscript can read
    one level per index.

    Parameters
    ----------
    ctx : PyCtx
        The render context.
    binders : tuple[LetFactorBinder, ...]
        The factor's binders.
    sizes : tuple[int, ...]
        Each binder's extent.
    body : LetExprNode
        The factor's body.
    fixed : tuple[int, ...]
        The values of the binders already enumerated.

    Returns
    -------
    str
        The vertex of the stacked array, or of one cell's value when
        every binder is fixed.

    Raises
    ------
    UnsupportedConstruct
        If the target maps no stacking primitive.
    """
    if len(fixed) == len(binders):
        cell = body
        for binder, value in zip(binders, fixed, strict=True):
            literal = LetExprLiteral(value=value, integral=True)
            cell = _substitute_let_expr(
                cell, binder.var, index_value=literal, scalar_value=literal
            )
        vid = render_let_expr_python(ctx, cell)
        element = _PY_STACK_ELEMENT_SYMBOLS.get(ctx.target)
        if element is not None:
            vid = call(ctx, attribute(ctx, element), positional=(vid,))
        return vid
    level = len(fixed)
    items = tuple(
        _build_stacked_python(ctx, binders, sizes, body, (*fixed, index))
        for index in range(sizes[level])
    )
    stack = _target_stack_symbol(ctx, binders[0].var)
    return call(ctx, attribute(ctx, stack), positional=(python_list(ctx, items),))


def _render_factor_python(ctx: PyCtx, expr: LetExprFactor) -> str:
    """Unroll a `LetExprFactor` into the target's array.

    A factor denotes a tensor wherever it appears, an operand of
    arithmetic included, so it is stacked into the target's array rather
    than left a Python list, which ``+`` would concatenate.

    Cases form (single binder, body=None): stack `[case_0, case_1,
    ...]` in label order. Uniform-body form (one or more binders):
    substitute each binder for its 0-indexed integer value (Python
    arrays are 0-based, matching QVR's surface convention) and
    unroll into a nested list literal of shape
    (|b0|, |b1|, ..., |bn-1|). The shared
    `_substitute_let_expr`
    walk takes the same value for both `index_value` and
    `scalar_value` because no index-base shift is needed here.
    """
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                "qvr-python-helper",
                [
                    "let-expr:LetExprFactor: cases form requires "
                    f"exactly one binder; got {len(expr.binders)}"
                ],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        element = _PY_STACK_ELEMENT_SYMBOLS.get(ctx.target)
        items: list[str] = []
        for case in ordered:
            vid = render_let_expr_python(ctx, case.value)
            if element is not None:
                vid = call(ctx, attribute(ctx, element), positional=(vid,))
            items.append(vid)
        stack = _target_stack_symbol(ctx, expr.binders[0].var)
        return call(
            ctx, attribute(ctx, stack), positional=(python_list(ctx, tuple(items)),)
        )
    if expr.body is not None and not expr.cases:
        sizes = tuple(_card_for(ctx, b) for b in expr.binders)
        return _build_stacked_python(ctx, expr.binders, sizes, expr.body, ())
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [
            "let-expr:LetExprFactor: mixed cases-plus-body form is "
            "not a valid surface construct"
        ],
    )


def _emit_python_list(ctx: PyCtx, items: tuple[str, ...]) -> str:
    """Emit a Python list literal `[e0, e1, ...]`."""
    vid = ctx.v(ctx.fresh("list"), "list")
    for item in items:
        ctx.e(vid, item, "child_of")
    return vid


def _card_for(ctx: PyCtx, binder: LetFactorBinder) -> int:
    """Resolve the static cardinality of `binder.index` via the
    `PyCtx.cards` snapshot of `IRProgram.cards`."""
    idx = binder.index
    if isinstance(idx, TypeName):
        size = ctx.cards.get(idx.name)
        if size is None:
            raise UnsupportedConstruct(
                "qvr-python-helper",
                [
                    f"let-expr:LetExprFactor: binder {binder.var!r} "
                    f"references object {idx.name!r} whose cardinality "
                    "is unknown at render time"
                ],
            )
        return size
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [
            f"let-expr:LetExprFactor: binder {binder.var!r} index is "
            f"{type(idx).__name__}; only TypeName binders unroll"
        ],
    )


# ---------------------------------------------------------------------------
# Marginalize: the IR-level analysis every Python backend's
# enumeration reads before it emits.
# ---------------------------------------------------------------------------


class MarginalizeBody(dx.Model):
    """The scope shape a Python-family `marginalize` lowering accepts.

    Every Python backend integrates the latent by scoring the scope
    once per atom of its finite support and reducing the per-atom
    log-densities with `logsumexp`. That reduction has a closed form
    only when the scope carries exactly one density-bearing site, so
    the accepted shape is a run of deterministic bindings followed by
    a single [`IRObserve`][quivers.transpile.ir.IRObserve]. The
    deterministics are re-emitted once per atom, each time under the
    same names, so the observe that follows reads that atom's values.
    """

    deterministics: tuple[IRDeterministic, ...]
    observe: IRObserve


def marginalize_body(
    scope: tuple[IRNode, ...], *, latent: str, target: str
) -> MarginalizeBody:
    """Split a marginalize scope into its deterministic prefix and its
    single observe.

    Called both on the raw
    [`IRMarginalize.scope`][quivers.transpile.ir.IRMarginalize] and on
    each atom's substituted copy, which share their node order.

    Raises `UnsupportedConstruct` with a `marginalize:` kind when the
    scope carries a nested latent, more than one observe, or a
    deterministic after the observe. Emitting a live draw in place of
    the integral would denote a measure on a strictly larger space
    than the QVR reference integrates, so there is no correct code to
    emit for a scope this lowering cannot reduce.
    """
    deterministics: list[IRDeterministic] = []
    observe: IRObserve | None = None
    for child in scope:
        if isinstance(child, IRDeterministic):
            if observe is not None:
                raise UnsupportedConstruct(
                    f"qvr-{target}",
                    [
                        f"marginalize:scope-order:{latent}: a "
                        f"deterministic binding follows the observed "
                        f"site, so the atoms cannot be scored in one "
                        f"pass"
                    ],
                )
            deterministics.append(child)
            continue
        if isinstance(child, IRObserve):
            if observe is not None:
                raise UnsupportedConstruct(
                    f"qvr-{target}",
                    [
                        f"marginalize:multi-observe:{latent}: the "
                        f"scope carries more than one observed site"
                    ],
                )
            observe = child
            continue
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:scope:{type(child).__name__}: only "
                f"deterministic bindings and one observed site reduce "
                f"to a finite mixture"
            ],
        )
    if observe is None:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:no-observe:{latent}: the scope "
                f"carries no observed site to weight the atoms against"
            ],
        )
    return MarginalizeBody(deterministics=tuple(deterministics), observe=observe)


def marginal_support_size(node: IRMarginalize) -> int | None:
    """Class count of a `class_index` atom set, or `None` when the
    family's atoms carry their own count.

    The plan states the latent's support as an integer range sized by
    the latent's declared index object, or by the width of its
    probability argument when it declares none (see
    `_enumerated_support` in [`quivers.transpile.plan`][quivers.transpile.plan]),
    so the count is that range's length whatever the probability
    argument is: a sampled simplex, a literal, or a `let`. A `binary`
    atom set carries its own size and ignores this.

    Parameters
    ----------
    node : IRMarginalize
        The marginalize block.

    Returns
    -------
    int | None
        The number of atoms, or `None` for a family with no
        class-index atom set or one whose atoms carry their own count.

    Raises
    ------
    UnsupportedConstruct
        If a class-index latent's planned support is not an integer
        range.
    """
    meta = FAMILY_META.get(node.family)
    if meta is None:
        return None
    support = marginalize_support(meta)
    if support is None or support.size is not None:
        return None
    if not isinstance(node.constraint, CSIntegerInterval):
        raise UnsupportedConstruct(
            "qvr-transpile",
            [
                f"marginalize:unknown-cardinality:{node.family}: the support "
                f"of `{node.latent}` is not a finite integer range"
            ],
        )
    return node.constraint.upper - node.constraint.lower + 1


def marginal_weight_probs(
    node: IRMarginalize,
    observe: IRObserve,
    weight_args: tuple[IRArg, ...],
    weight_arg_names: tuple[str, ...],
    *,
    name_plates: dict[str, Plate],
    target: str,
) -> IRArg:
    """The probability tensor weighting the atoms, aligned to the axis
    the observed site is fibred over.

    A weight tensor allocated per grouping row (LDA's per-document
    `theta`) has to be gathered through the observe's `via` fibration
    before it can be added to a per-observation log-density; a weight
    tensor shared across the whole grouping plate (a single simplex
    over the latent's classes) broadcasts as it stands and must not be
    gathered, since its only axis is the class axis.
    """
    meta = FAMILY_META.get(node.family)
    if meta is None:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"family:unknown:{node.family}"],
        )
    support = marginalize_support(meta)
    if support is None:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"marginalize:non-finite-support:{node.family}"],
        )
    probs = _named_arg(weight_args, weight_arg_names, support.weight_arg)
    if observe.via is None:
        return probs
    if not node.plate.batch_dims:
        return probs
    if observe.plate.batch_dims == node.plate.batch_dims:
        return probs
    if not isinstance(probs, IRArgRef):
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:via-weight:{node.latent}: the atom "
                f"weights are a computed expression, so the "
                f"{observe.via!r} fibration has no index position to "
                f"gather them through"
            ],
        )
    weight_plate = name_plates.get(probs.name)
    if weight_plate is None or weight_plate.batch_dims != node.plate.batch_dims:
        return probs
    return IRArgRef(
        name=probs.name,
        indices=(IRArgRef(name=observe.via), *probs.indices),
    )


def marginalize_fibration(
    node: IRMarginalize,
    observe: IRObserve,
    weight_args: tuple[IRArg, ...],
    weight_arg_names: tuple[str, ...],
    *,
    name_plates: dict[str, Plate],
    target: str,
) -> tuple[str, Dim] | None:
    """The fibration a grouped block gathers its rows through, if any.

    `docs/semantics/programs.md` §2.7 keys a grouped block's accumulator
    by group: the rows an observe fibres into one group are summed
    before the reduction over the latent. A renderer that reduces each
    row on its own scores that measure only when the rows and the
    groups coincide, or when the prior is allocated per group, which
    denotes one latent per row. Otherwise the per-row per-class
    log-likelihoods have to be scattered into a ``(|G|, K)`` accumulator
    through the observe's ``via`` fibration first.

    Parameters
    ----------
    node : IRMarginalize
        The block.
    observe : IRObserve
        The scope's observe.
    weight_args : tuple[IRArg, ...]
        The prior's arguments.
    weight_arg_names : tuple[str, ...]
        The prior's argument names.
    name_plates : dict[str, Plate]
        The plate of every named tensor.
    target : str
        The transpile target, for diagnostics.

    Returns
    -------
    tuple[str, Dim] | None
        The fibration's name and the grouping plate's leading axis when
        the rows must be scattered; ``None`` when each row reduces on
        its own.
    """
    if observe.via is None or not node.plate.batch_dims:
        return None
    if observe.plate.batch_dims == node.plate.batch_dims:
        return None
    probs = marginal_weight_probs(
        node,
        observe,
        weight_args,
        weight_arg_names,
        name_plates=name_plates,
        target=target,
    )
    if isinstance(probs, IRArgRef) and probs.indices:
        # A per-group prior gathered through the fibration: one latent
        # per row, each reducing on its own.
        return None
    return observe.via, node.plate.batch_dims[0]


def marginal_atom_axis(family: str, arg_names: tuple[str, ...], *, target: str) -> int:
    """Negative axis along which one atom's distribution arguments
    stack into the batched component distribution.

    The atom axis is the innermost *batch* axis, preceding whatever
    trailing axes the family's arguments carry as
    their own event shape: a `Poisson` rate is scalar-per-row and
    stacks at ``-1``, a `Categorical` probability vector carries the
    class axis and stacks at ``-2``. The trailing rank comes from the
    torch `arg_constraints` entry of each argument the call site
    names.
    """
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"family:unknown:{family}"],
        )
    constraints = meta.distribution_class.arg_constraints
    if not isinstance(constraints, dict):
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:atom-axis:{family}: the family declares "
                f"`arg_constraints` as a property rather than a "
                f"class-level table, so an argument's trailing rank is "
                f"not known before the call is built"
            ],
        )
    ranks: set[int] = set()
    for name in arg_names:
        constraint = constraints.get(name)
        if constraint is None:
            raise UnsupportedConstruct(
                f"qvr-{target}",
                [
                    f"marginalize:atom-axis:{family}: argument "
                    f"{name!r} is not declared in the family's "
                    f"`arg_constraints`"
                ],
            )
        ranks.add(int(constraint.event_dim))
    if len(ranks) > 1:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:atom-axis:{family}: the call site's "
                f"arguments carry different trailing ranks "
                f"{sorted(ranks)!r}, so no single axis stacks them"
            ],
        )
    return -(1 + (ranks.pop() if ranks else 0))


def _named_arg(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...], wanted: str
) -> IRArg:
    """The argument bound to `wanted`, positionally paired with
    `arg_names`."""
    for name, arg in zip(arg_names, args, strict=True):
        if name == wanted:
            return arg
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [
            f"marginalize:missing-probability-argument:{wanted}: the "
            f"call site names no {wanted!r} argument"
        ],
    )


def python_list(ctx: PyCtx, items: tuple[str, ...]) -> str:
    """Emit a Python list literal ``[e0, e1, ...]``."""
    return _emit_python_list(ctx, items)


def subscript(ctx: PyCtx, value: str, indices: tuple[str, ...]) -> str:
    """Build ``<value>[<i0>, <i1>, ...]``."""
    s = ctx.v(ctx.fresh("subs"), "subscript")
    ctx.e(s, value, "value")
    for idx in indices:
        ctx.e(s, idx, "subscript")
    return s


def shape_tuple(ctx: PyCtx, shape: tuple[int, ...]) -> str:
    """Build a Python ``tuple`` node from an integer shape.

    Emits ``()`` for an empty shape, ``(<n>,)`` for a singleton (with
    the required trailing comma), and ``(<r>, <c>, ...)`` for higher
    arity. Tree-sitter Python's `tuple` production needs an explicit
    `ptrace-*` punctuation trace to render the comma; without it the
    emitter drops the comma and produces a `parenthesized_expression`.
    """
    tup = ctx.v(ctx.fresh("tup"), "tuple")
    n = len(shape)
    if n == 0:
        ctx.constraint(tup, "chose-alt-fingerprint", "()")
        ctx.constraint(tup, "ptrace-0", "T(")
        ctx.constraint(tup, "ptrace-1", "T)")
        return tup
    kind_list = " ".join("integer" for _ in range(n))
    if n == 1:
        ctx.constraint(tup, "chose-alt-fingerprint", "( ,)")
        ctx.constraint(tup, "ptrace-0", "T(")
        ctx.constraint(tup, "ptrace-1", "Cinteger")
        ctx.constraint(tup, "ptrace-2", "T,")
        ctx.constraint(tup, "ptrace-3", "T)")
    else:
        fingerprint = "( " + " ".join("," for _ in range(n - 1)) + " )"
        ctx.constraint(tup, "chose-alt-fingerprint", fingerprint)
        ctx.constraint(tup, "ptrace-0", "T(")
        slot = 1
        for i in range(n):
            ctx.constraint(tup, f"ptrace-{slot}", "Cinteger")
            slot += 1
            if i < n - 1:
                ctx.constraint(tup, f"ptrace-{slot}", "T,")
                slot += 1
        ctx.constraint(tup, f"ptrace-{slot}", "T)")
    ctx.constraint(tup, "chose-alt-child-kinds", kind_list)
    for size in shape:
        ctx.e(tup, number_literal(ctx, size), "child_of")
    return tup


__all__ = [
    "MarginalizeBody",
    "PyCtx",
    "arg_expr",
    "assignment",
    "attribute",
    "call",
    "factor_tower_names",
    "float_literal",
    "function_def",
    "identifier",
    "marginal_atom_axis",
    "marginal_support_size",
    "marginal_weight_probs",
    "marginalize_body",
    "name_event_rank_map",
    "name_plate_map",
    "number_literal",
    "python_binary_op",
    "python_list",
    "python_method_call",
    "python_paren",
    "python_unary_minus",
    "render_deterministic_python",
    "render_let_expr_python",
    "shape_tuple",
    "string_literal",
    "subscript",
    "with_statement",
]
