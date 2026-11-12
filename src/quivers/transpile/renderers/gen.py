"""`GenRenderer`: emit IR as a Gen.jl `@gen function` definition.

Gen.jl has no native plate construct. Each batch axis becomes an
explicit `for` loop in Julia that writes per-element draws into a
pre-allocated `Vector{T}(undef, B)` so the loop body can read prior
draws by index. The trace address attached to each per-element call
includes the loop index so Gen sees a distinct address per iteration:

```julia
@gen function model(alpha, beta, word_idx, w)
    theta = Vector{Vector{Float64}}(undef, 20)
    for m_Doc in 1:20
        theta[m_Doc] = @trace(dirichlet(fill(alpha, 3)), (:theta, m_Doc))
    end
    ...
end
```

The renderer inherits the target-independent IR walk from
[`RendererBase`][quivers.transpile.renderers.RendererBase] and
implements `declare`, `sample`, `marginalize`, and `broadcast` per the
Gen.jl idiom. `marginalize` lowers
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] to an explicit
[`IRSample`][quivers.transpile.ir.IRSample] followed by the scope
inline (Gen samples discrete latents natively, unlike Stan's
`log_sum_exp` enumeration).
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable

import panproto

from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
    LetFactorCase,
)
from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.renderers._rows import scalar_parameter, visited_position
from quivers.transpile.renderers._ranks import (
    carried_dims,
    name_plate_map,
    name_rank_map,
)
from quivers.transpile.renderers._julia_helpers import (
    JuliaShapes,
    infer_array_rank,
    let_expr_has_axis_reduction,
    name_array_rank_map,
    nested_tower_names,
    refuse_nested_slice_gather,
    render_let_expr_julia,
)
from quivers.transpile.renderers._python_helpers import (
    name_event_rank_map,
)
from quivers.transpile.ir import (
    ConstraintSpec,
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgFamilyRef,
    IRArgKernel,
    IRArgList,
    IRArgMatrix,
    IRArgNumber,
    IRArgRef,
    IRCall,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
    Plate,
    is_int_bit,
    is_int_category,
    is_int_count,
    is_real_corr_chol,
    is_real_cov_matrix,
    is_real_matrix,
    is_real_one_hot,
    is_real_positive,
    is_real_scalar,
    is_real_simplex,
    is_real_unit_interval,
    is_real_vector,
)
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    mixture_normal_components,
)
from quivers.transpile.renderers.turing import (
    TuringRenderer,
    _seed_batch_shaped,
    _seed_sample_plates,
    _TuringCtx,
)
from quivers.transpile.renderers._qiec import (
    emit_call_julia,
    render_computations_dynamic,
    qiec_helper_families_used,
)


# ---------------------------------------------------------------------------
# Julia tree-sitter SchemaBuilder helpers.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _GenCtx:
    """Per-render Julia schema-construction carrier.

    Wraps the panproto `SchemaBuilder` with a
    fresh-id counter, the accumulators the IR walk fills in for the
    function signature (parameters) and body (statements), and the
    cross-step tables the renderer consults when threading batch
    indices through references (which prior-step samples live on
    which axes, which inputs are scalar vs vector / index, which
    morphism decls back wrapper families).
    """

    sb: panproto.SchemaBuilder
    n: int = 0
    params: list[str] = dataclasses.field(default_factory=list)
    # The model function's body block; a statement is placed in it as
    # soon as it is built, so emitters that write into a block
    # directly (the shared marginalize emitter) interleave in order.
    body: str = ""
    return_names: tuple[str, ...] = ()
    # The bodies whose native operation table a call has bound.
    operations_bound: set[str] = dataclasses.field(default_factory=set)
    # Each previously-declared sample / observe / data-input mapped to
    # the tuple of batch Dims it lives on, in declaration order. Used
    # to thread loop indices through `IRArgRef` instances.
    decl_axes: dict[str, tuple[Dim, ...]] = dataclasses.field(default_factory=dict)
    # Each name the plan binds mapped to its whole plate, batch and
    # event: a gather `mu[z]` carries the axes of `mu` after the indexed
    # ones, which each take the loop over that axis or a whole `:`.
    name_plates: dict[str, Plate] = dataclasses.field(default_factory=dict)
    # Cards harvested from the IR for axis-name lookups.
    cards: dict[str, int] = dataclasses.field(default_factory=dict)
    # Morphism declarations needed by wrapper-family rendering.
    morphisms: dict[str, object] = dataclasses.field(default_factory=dict)
    # IRDataInput records by name, to recover constraint info on the
    # fly during arg rendering.
    inputs_by_name: dict[str, IRDataInput] = dataclasses.field(default_factory=dict)
    # Axis names a prior `for` loop has already iterated over in the
    # function body. The loop-variable namer consults this set to
    # decide whether to emit the bare `m_<Axis>` form or the
    # disambiguated `m_<Axis>_<step>` form.
    used_axes: set[str] = dataclasses.field(default_factory=set)
    # The array-shape environment the shared Julia let-expression
    # helper consults: per-name event rank and full array rank, plus
    # the bindings materialised as a nested tower of vector literals.
    shapes: JuliaShapes = dataclasses.field(default_factory=JuliaShapes)

    def fresh(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n}"

    def v(self, kind: str, prefix: str | None = None) -> str:
        vid = self.fresh(prefix or kind[:3])
        self.sb.vertex(vid, kind)
        return vid

    def vlit(self, kind: str, text: str, prefix: str | None = None) -> str:
        vid = self.v(kind, prefix)
        self.sb.constraint(vid, "literal-value", text)
        return vid

    def e(self, src: str, tgt: str) -> None:
        self.sb.edge(src, tgt, "child_of")

    def place(self, statement: str) -> None:
        """Append one statement to the model body.

        Parameters
        ----------
        statement : str
            The statement vertex.
        """
        self.e(self.body, statement)


class _JlCtxAdapter:
    """Adapt a `_GenCtx` to
    the protocol expected by
    `render_let_expr_julia`.

    The shared Julia let-expression helper expects a ctx with explicit
    `v(vid, kind)` / `e(src, tgt, kind=child_of)` / `lit(vid, text)` /
    `constraint(vid, sort, value)` / `fresh(prefix)` methods plus
    `cards` and `target` attributes. The Gen renderer's [`_GenCtx`]
    uses a different signature (`v(kind, prefix)`, `e(src, tgt)`); the
    adapter bridges the two without touching the underlying
    SchemaBuilder.
    """

    target: str
    cards: dict[str, int]

    def __init__(self, gx: _GenCtx, target: str) -> None:
        self._gx = gx
        self.cards = gx.cards
        self.target = target

    def fresh(self, prefix: str) -> str:
        return self._gx.fresh(prefix)

    def v(self, vid: str, kind: str) -> str:
        self._gx.sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._gx.sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._gx.sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._gx.sb.constraint(vid, sort, value)


def _ident(gx: _GenCtx, name: str) -> str:
    return gx.vlit("identifier", name, "id")


def _integer(gx: _GenCtx, value: int) -> str:
    return gx.vlit("integer_literal", str(value), "int")


def _float_lit(gx: _GenCtx, value: float) -> str:
    return gx.vlit("float_literal", repr(value), "flt")


def _number(gx: _GenCtx, value: float) -> str:
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return _integer(gx, int(value))
    return _float_lit(gx, float(value))


def _operator(gx: _GenCtx, text: str) -> str:
    return gx.vlit("operator", text, "op")


def _quote_sym(gx: _GenCtx, name: str) -> str:
    """`:name` as a Julia quote_expression containing an identifier."""
    q = gx.v("quote_expression", "qe")
    gx.e(q, _ident(gx, name))
    return q


def _call(gx: _GenCtx, callee_vid: str, args: tuple[str, ...]) -> str:
    """`<callee>(<a1>, <a2>, ...)`."""
    c = gx.v("call_expression", "call")
    al = gx.v("argument_list", "al")
    gx.e(c, callee_vid)
    gx.e(c, al)
    for a in args:
        gx.e(al, a)
    return c


def _index_into(gx: _GenCtx, base: str, indices: tuple[str, ...]) -> str:
    """`base[i1, i2, ...]` as `index_expression(base, vector_expression(...))`.

    Julia tree-sitter encodes the bracket pair as a `vector_expression`
    child of `index_expression`, not as parallel child edges of the
    `index_expression` itself.
    """
    ix = gx.v("index_expression", "ix")
    gx.e(ix, base)
    ve = gx.v("vector_expression", "ve")
    for i in indices:
        gx.e(ve, i)
    gx.e(ix, ve)
    return ix


def _range(gx: _GenCtx, lo: str, hi: str) -> str:
    """`<lo>:<hi>` as a range_expression."""
    r = gx.v("range_expression", "rg")
    gx.e(r, lo)
    gx.e(r, hi)
    return r


def _tuple(gx: _GenCtx, parts: tuple[str, ...]) -> str:
    t = gx.v("tuple_expression", "tup")
    for p in parts:
        gx.e(t, p)
    return t


def _assignment(gx: _GenCtx, lhs: str, rhs: str) -> str:
    """`<lhs> = <rhs>`."""
    asn = gx.v("assignment", "asn")
    gx.e(asn, lhs)
    gx.e(asn, _operator(gx, "="))
    gx.e(asn, rhs)
    return asn


def _macro_call_space(gx: _GenCtx, macro_name: str, args: tuple[str, ...]) -> str:
    """`@<macro_name> arg1 arg2 ...` (space-separated macro args).

    Julia accepts both `@trace(dist, addr)` and `@trace dist addr`. The
    panproto emitter does not consistently round-trip the parenthesised
    form for by-construction schemas (it requires byte-position
    metadata the walker cannot synthesise), so we emit the space-
    separated form. Gen.jl parses both identically.
    """
    mc = gx.v("macrocall_expression", "mc")
    mid = gx.v("macro_identifier", "mid")
    gx.e(mid, _ident(gx, macro_name))
    mal = gx.v("macro_argument_list", "mal")
    for a in args:
        gx.e(mal, a)
    gx.e(mc, mid)
    gx.e(mc, mal)
    return mc


def _macro_call_body(gx: _GenCtx, macro_name: str, body_vid: str) -> str:
    """`@<macro_name> <body>` for a long-form macro (function def, etc.)."""
    return _macro_call_space(gx, macro_name, (body_vid,))


def _function_def(
    gx: _GenCtx, *, name: str, params: tuple[str, ...], body_vid: str
) -> str:
    """`function <name>(<p1>, <p2>, ...) <body> end`."""
    fn = gx.v("function_definition", "fn")
    sig = gx.v("signature", "sig")
    sigcall = gx.v("call_expression", "scall")
    al = gx.v("argument_list", "sargs")
    for p in params:
        gx.e(al, _ident(gx, p))
    gx.e(sigcall, _ident(gx, name))
    gx.e(sigcall, al)
    gx.e(sig, sigcall)
    gx.e(fn, sig)
    gx.e(fn, body_vid)
    return fn


def _for_stmt(
    gx: _GenCtx, *, var: str, lo: str, hi: str, body_stmts: tuple[str, ...]
) -> str:
    """`for <var> in <lo>:<hi> <body> end`."""
    fs = gx.v("for_statement", "fs")
    fb = gx.v("for_binding", "fb")
    gx.e(fb, _ident(gx, var))
    gx.e(fb, _range(gx, lo, hi))
    blk = gx.v("block", "blk")
    for s in body_stmts:
        gx.e(blk, s)
    gx.e(fs, fb)
    gx.e(fs, blk)
    return fs


def _vector_alloc(gx: _GenCtx, *, elem_type: str, size_vid: str) -> str:
    """`Vector{<elem_type>}(undef, <size>)`.

    `elem_type` is a Julia type-name string (e.g. ``"Float64"``,
    ``"Vector{Float64}"``, ``"Matrix{Float64}"``, ``"Int"``).
    """
    callee = _parametrized_type(gx, "Vector", elem_type)
    al = gx.v("argument_list", "al")
    gx.e(al, _ident(gx, "undef"))
    gx.e(al, size_vid)
    call = gx.v("call_expression", "call")
    gx.e(call, callee)
    gx.e(call, al)
    return call


def _array_alloc(gx: _GenCtx, *, elem_type: str, size_vids: tuple[str, ...]) -> str:
    """`Array{<elem_type>, <N>}(undef, <s0>, ..., <sN-1>)`.

    The rank-1 case still spells `Vector{T}(undef, s)`, the idiom the
    single-axis plate loop has always emitted. Higher ranks allocate a
    genuine dense `Array` rather than a nested `Vector{Vector{T}}`,
    whose inner slots would be `#undef` when the loop nest first
    assigns into them.
    """
    if len(size_vids) == 1:
        return _vector_alloc(gx, elem_type=elem_type, size_vid=size_vids[0])
    pt = gx.v("parametrized_type_expression", "pt")
    gx.e(pt, _ident(gx, "Array"))
    curly = gx.v("curly_expression", "cu")
    gx.e(curly, _type_expr(gx, elem_type))
    gx.e(curly, _integer(gx, len(size_vids)))
    gx.e(pt, curly)
    al = gx.v("argument_list", "al")
    gx.e(al, _ident(gx, "undef"))
    for size_vid in size_vids:
        gx.e(al, size_vid)
    call = gx.v("call_expression", "call")
    gx.e(call, pt)
    gx.e(call, al)
    return call


def _parametrized_type(gx: _GenCtx, head: str, inner: str) -> str:
    """Build a `head{inner}` parametrised type expression.

    `inner` is a string spelling another Julia type expression
    (atomic identifier or another parametrised type). Recursive
    parametrisations like ``Vector{Vector{Float64}}`` parse cleanly.
    """
    pt = gx.v("parametrized_type_expression", "pt")
    gx.e(pt, _ident(gx, head))
    curly = gx.v("curly_expression", "cu")
    gx.e(curly, _type_expr(gx, inner))
    gx.e(pt, curly)
    return pt


def _type_expr(gx: _GenCtx, spec: str) -> str:
    """Convert a Julia type-name string into a vertex.

    Supports atomic identifiers and one or more nested parametrised
    type heads (e.g. ``"Vector{Vector{Float64}}"``,
    ``"Matrix{Float64}"``). The grammar is `Head{Inner}` where
    `Inner` is another type-name.
    """
    spec = spec.strip()
    if "{" not in spec:
        return _ident(gx, spec)
    head, rest = spec.split("{", 1)
    if not rest.endswith("}"):
        raise UnsupportedConstruct(
            "qvr-gen", [f"type-expr:{spec}: missing closing brace"]
        )
    inner = rest[:-1]
    return _parametrized_type(gx, head.strip(), inner)


# ---------------------------------------------------------------------------
# Dim-size resolution and element-type tables.
# ---------------------------------------------------------------------------


def _dim_size_vid(gx: _GenCtx, dim: Dim) -> str:
    """Return the vertex id for a dim's size.

    Static dims emit the literal cardinality (`20`). Dynamic dims
    emit `length(<size_name>)` so the generated function reads the
    runtime size from the argument the size was bound to.
    """
    if isinstance(dim, DimStatic):
        return _integer(gx, dim.size)
    if isinstance(dim, DimDynamic):
        # For dynamic dims with a `size_name`, prefer the
        # `length(<name>)` form unless the name is already an integer
        # input the program reads directly.
        return _call(gx, _ident(gx, "length"), (_ident(gx, dim.size_name),))
    raise UnsupportedConstruct("qvr-gen", [f"dim:{type(dim).__name__}"])


# ---------------------------------------------------------------------------
# Per-support storage-type tables.
# ---------------------------------------------------------------------------


def _element_type_for(spec: ConstraintSpec, plate: Plate) -> str:
    """Return the Julia element-type string for one draw of the family.

    Used as the `T` in `Vector{T}(undef, B)` when batch_dims is
    non-empty. Dispatches on the renderer's shared support predicates.
    """
    del plate
    c = spec.to_constraint()
    if is_real_scalar(c) or is_real_positive(c) or is_real_unit_interval(c):
        return "Float64"
    if is_real_vector(c) or is_real_simplex(c) or is_real_one_hot(c):
        return "Vector{Float64}"
    if is_real_matrix(c) or is_real_cov_matrix(c) or is_real_corr_chol(c):
        return "Matrix{Float64}"
    if is_int_bit(c) or is_int_category(c) or is_int_count(c):
        return "Int"
    raise UnsupportedConstruct(
        "qvr-gen",
        [f"support:{type(c).__name__}: no Gen.jl storage-type rule"],
    )


# ---------------------------------------------------------------------------
# Loop-variable naming.
#
# A step's loop variable per axis is `m_<AxisName>` by default; when
# the IR contains multiple steps batched on the same axis, the
# renderer disambiguates by appending the step name (`m_Doc_z`).
# ---------------------------------------------------------------------------


def _loop_var_for(gx: _GenCtx, axis_name: str, step_name: str) -> str:
    """Return the loop-variable identifier for `axis_name` in `step_name`.

    Sticks with the bare `m_<Axis>` form unless another previously
    emitted step already used a loop on that axis, in which case the
    step name is appended to disambiguate (`m_Doc_z`).
    """
    base = f"m_{axis_name}"
    if axis_name in gx.used_axes:
        return f"{base}_{step_name}"
    return base


# ---------------------------------------------------------------------------
# IR arg rendering.
# ---------------------------------------------------------------------------


def _expected_event_rank(family: str, arg_name: str) -> int:
    """Return the expected `event_dim` of the family's `arg_name`.

    Reads `FAMILY_META[family].distribution_class.arg_constraints`
    and walks it for an `IndependentConstraint(base, n)` constraint
    whose `event_dim` indicates how many vector / matrix axes the
    arg should carry. Returns 0 for scalar args.
    """
    meta = FAMILY_META.get(family)
    if meta is None:
        return 0
    cls_attr = getattr(meta.distribution_class, "arg_constraints", None)
    if not isinstance(cls_attr, dict):
        return 0
    constraint = cls_attr.get(arg_name)
    if constraint is None:
        return 0
    # `torch.distributions.constraints.independent(base, n)` returns
    # an `_IndependentConstraint` carrying `event_dim` and
    # `base_constraint`. Duck-typing on both avoids the private class
    # name while excluding other constraints (`simplex`,
    # `lower_triangular`, etc.) whose `event_dim > 0` reflects the
    # support's intrinsic shape, not a `Family.expand(...)` wrapper.
    if hasattr(constraint, "base_constraint") and hasattr(constraint, "event_dim"):
        return int(constraint.event_dim)
    return 0


def _broadcast_arg_to_event(
    gx: _GenCtx,
    arg_vid: str,
    *,
    arg: IRArg,
    family: str,
    arg_name: str,
    event_dims: tuple[Dim, ...],
    inputs_by_name: dict[str, IRDataInput],
) -> str:
    """Wrap a rendered scalar arg in `fill(...)` when the family arg
    expects a vector / matrix and the user supplied a scalar.

    Triggers when:

    * the IR arg is a literal number; or
    * the IR arg is a reference whose binding is a scalar input.

    The fill shape is the event_dims cardinality.
    """
    rank = _expected_event_rank(family, arg_name)
    if rank == 0:
        return arg_vid
    if isinstance(arg, IRArgBroadcast):
        # Lower already supplied a broadcast wrapper.
        return arg_vid
    needs_fill = False
    if isinstance(arg, IRArgNumber):
        needs_fill = True
    elif isinstance(arg, IRArgRef) and not arg.indices:
        binding = inputs_by_name.get(arg.name)
        if binding is not None and _is_scalar_input(binding):
            needs_fill = True
    if not needs_fill:
        return arg_vid
    sizes: list[str] = []
    for d in event_dims[:rank]:
        sizes.append(_dim_size_vid(gx, d))
    if not sizes:
        return arg_vid
    return _call(gx, _ident(gx, "fill"), (arg_vid, *sizes))


def _is_scalar_input(inp: IRDataInput) -> bool:
    """True iff the data input is a scalar (no batch / event dims)."""
    if inp.plate.batch_dims or inp.plate.event_dims:
        return False
    c = inp.constraint.to_constraint()
    if is_real_vector(c) or is_real_simplex(c) or is_real_one_hot(c):
        return False
    if is_real_matrix(c) or is_real_cov_matrix(c) or is_real_corr_chol(c):
        return False
    return True


@dataclasses.dataclass
class _ArgCtx:
    """Per-call rendering context for the args of one distribution call.

    Carries the per-step batch-loop bindings (axis name → loop
    variable identifier), the current step's `via` fibration (if
    any), and a flag indicating whether the inner observe sees the
    fibration semantics so reference threading routes through it.
    """

    batch_loops: dict[str, str] = dataclasses.field(default_factory=dict)
    via_indexer: str | None = None
    via_loop_var: str | None = None
    # Every loop of the step, its residual event axes included: the axis
    # a gathered element carries reads the loop over that axis.
    axis_loops: dict[str, str] = dataclasses.field(default_factory=dict)
    # Whether the argument being rendered fills a rank-0 family
    # parameter, which reads one entry per drawn element.
    scalar_slot: bool = False


def _render_arg(gx: _GenCtx, arg: IRArg, *, arg_ctx: _ArgCtx) -> str:
    """Render one IR arg into a Julia expression vertex.

    See `_render_arg_ref` for how `arg_ctx` participates in threading
    loop indices through references.
    """
    if isinstance(arg, IRArgNumber):
        return _number(gx, arg.value)
    if isinstance(arg, IRArgRef):
        return _render_arg_ref(gx, arg, arg_ctx=arg_ctx)
    if isinstance(arg, IRArgBroadcast):
        value_vid = _render_arg(gx, arg.value, arg_ctx=arg_ctx)
        return _broadcast_to_shape(gx, value_vid, arg.target_shape)
    if isinstance(arg, IRArgList):
        return _render_list(gx, arg, arg_ctx=arg_ctx)
    if isinstance(arg, IRArgMatrix):
        return _render_matrix(gx, arg, arg_ctx=arg_ctx)
    if isinstance(arg, IRArgFamilyRef):
        raise UnsupportedConstruct(
            "qvr-gen",
            [f"arg:family_ref:{arg.name}: no inline rendering"],
        )
    raise UnsupportedConstruct("qvr-gen", [f"arg:{type(arg).__name__}"])


def _render_arg_ref(gx: _GenCtx, ref: IRArgRef, *, arg_ctx: _ArgCtx) -> str:
    """Render an [`IRArgRef`][quivers.transpile.ir.IRArgRef], threading
    the surrounding loop's index into the reference.

    A reference to a batched declaration acquires one bracket per
    batch axis when the user did not supply explicit indices for
    that reference. The bracket is filled from `arg_ctx.batch_loops`
    when the declaration's axis has a live loop variable, or via the
    `via=` fibration when no direct loop variable matches.

    If the user supplied explicit indices, the renderer treats them
    as the full bracket expression and skips automatic axis
    threading; the inner index args still see the surrounding
    `arg_ctx` and route through it as appropriate.
    """
    base = _ident(gx, ref.name)
    decl_axes = gx.decl_axes.get(ref.name, ())

    if ref.indices:
        rendered = tuple(
            _one_higher(
                gx,
                _render_arg(
                    gx, idx, arg_ctx=dataclasses.replace(arg_ctx, scalar_slot=False)
                ),
                idx,
            )
            for idx in ref.indices
        )
        # Each axis the gathered element carries is read at the step's
        # loop over that axis, or whole when no loop runs over it, so a
        # row of a matrix is never read through one linear index.
        carried = tuple(
            _ident(gx, arg_ctx.axis_loops[str(dim.name)])
            if str(dim.name) in arg_ctx.axis_loops
            else _operator(gx, ":")
            for dim in carried_dims(ref, gx.name_plates)
        )
        return _index_into(gx, base, (*rendered, *carried))

    visited = visited_position(gx.name_plates.get(ref.name), arg_ctx.axis_loops)
    if arg_ctx.scalar_slot and visited:
        # Every axis of the value is one the step visits, a vector over
        # a residual event axis included, and the parameter reads one
        # entry per draw: the draw at a position reads the entry there.
        return _index_into(gx, base, tuple(_ident(gx, lv) for lv in visited))

    if not decl_axes:
        return base

    leading: list[str] = []
    for axis in decl_axes:
        loop_var = arg_ctx.batch_loops.get(axis.name)
        if loop_var is not None:
            leading.append(_ident(gx, loop_var))
            continue
        # No direct loop variable: route through the via fibration
        # if one is active; otherwise leave the axis unindexed.
        if arg_ctx.via_indexer is not None and arg_ctx.via_loop_var is not None:
            via_idx = _index_into(
                gx,
                _ident(gx, arg_ctx.via_indexer),
                (_ident(gx, arg_ctx.via_loop_var),),
            )
            # The fibration holds zero-based QVR group codes.
            leading.append(_gen_binary_expr(gx, via_idx, ".+", _integer(gx, 1)))
            continue

    if not leading:
        return base
    return _index_into(gx, base, tuple(leading))


def _one_higher(gx: _GenCtx, index: str, arg: IRArg) -> str:
    """Read a plan argument's subscript one higher when it holds a QVR value.

    A program's index covariate or drawn class holds a zero-based QVR
    value, which Julia reads one higher; a renderer's own loop variable
    already counts from one.

    Parameters
    ----------
    gx : _GenCtx
        The render context.
    index : str
        The rendered subscript.
    arg : IRArg
        The subscript's plan argument.

    Returns
    -------
    str
        The subscript, broadcast-shifted with `.+ 1` for a program value.
    """
    if not isinstance(arg, IRArgRef) or arg.name not in gx.shapes.program_values:
        return index
    return _gen_binary_expr(gx, index, ".+", _integer(gx, 1))


def _broadcast_to_shape(
    gx: _GenCtx, value_vid: str, target_shape: tuple[int, ...]
) -> str:
    """`fill(<value>, <s0>, <s1>, ...)` per the Gen broadcast idiom."""
    if not target_shape:
        return value_vid
    args: list[str] = [value_vid]
    for s in target_shape:
        args.append(_integer(gx, int(s)))
    return _call(gx, _ident(gx, "fill"), tuple(args))


def _render_list(gx: _GenCtx, arg: IRArgList, *, arg_ctx: _ArgCtx) -> str:
    """`[e0, e1, ...]` as a Julia vector_expression."""
    ve = gx.v("vector_expression", "ve")
    for e in arg.elements:
        gx.e(ve, _render_arg(gx, e, arg_ctx=arg_ctx))
    return ve


def _render_matrix(gx: _GenCtx, arg: IRArgMatrix, *, arg_ctx: _ArgCtx) -> str:
    """`[<r0>; <r1>; ...]` as a Julia matrix_expression of matrix_rows."""
    me = gx.v("matrix_expression", "me")
    for row in arg.rows:
        mr = gx.v("matrix_row", "mr")
        for e in row.elements:
            gx.e(mr, _render_arg(gx, e, arg_ctx=arg_ctx))
        gx.e(me, mr)
    return me


# ---------------------------------------------------------------------------
# Family-name resolution.
# ---------------------------------------------------------------------------


#: Wrapper-family fallback names. Each entry pairs the wrapper QVR
#: family with the Gen.jl call used to render its emitted distribution.
_WRAPPER_TARGET_NAMES: dict[str, str] = {
    "Truncated": "truncated",
}


#: QVR families with no native Gen.jl distribution whose canonical
#: Gen.jl encoding is the underlying base distribution centred at zero
#: (e.g. `HalfNormal(scale)` -> `normal(0, scale)`). Gen.jl's `assess`
#: enumerates the choicemap as written, so reflecting a half-support
#: family back into a centred two-tail draw differs from the QVR
#: HalfNormal density by a constant `log 2` per draw; the
#: log-density-equivalence harness absorbs that constant.
_HALF_BASE_TARGETS: dict[str, tuple[str, int]] = {
    "HalfNormal": ("normal", 0),
    "HalfCauchy": ("cauchy", 0),
}


#: QVR families whose `rate` arg maps to a `scale = 1/rate` slot in
#: Gen.jl's constructor. Gen.jl's `gamma(shape, scale)` is
#: scale-parameterized, so the QVR rate is reciprocated. The mapped
#: value is the 0-based positional index of the rate arg.
_GEN_RATE_TO_SCALE_INVERT_POSITIONS: dict[str, int] = {
    "Gamma": 1,
}


#: QVR families whose probability parameter is the complement of the
#: Gen.jl distribution's. torch `NegativeBinomial(r, p)` has pmf
#: proportional to `(1 - p)^r p^k`; Gen.jl's `neg_binom(r, p)` (built on
#: Distributions.jl) has pmf proportional to `p^r (1 - p)^k`, so the
#: probs slot carries `1 - p`. The mapped value is the 0-based
#: positional index of the probs arg.
_GEN_PROB_COMPLEMENT_POSITIONS: dict[str, int] = {
    "NegativeBinomial": 1,
}


def _gen_binary_expr(gx: _GenCtx, left: str, op: str, right: str) -> str:
    """`<left> <op> <right>` as a Julia `binary_expression`."""
    be = gx.v("binary_expression", "be")
    gx.e(be, left)
    gx.e(be, _operator(gx, op))
    gx.e(be, right)
    return be


def _gen_transform_args(gx: _GenCtx, arg_vids: list[str], family: str) -> list[str]:
    """Apply the family's value-level arg transforms to rendered args.

    Reciprocates the rate arg (Gamma) into Gen.jl's scale slot and
    complements the probs arg (NegativeBinomial). Both are
    position-stable rewrites on already-rendered vertices, mirroring
    the Turing renderer's `inv(rate)` and `1 - probs` substitutions.
    """
    pos = _GEN_RATE_TO_SCALE_INVERT_POSITIONS.get(family)
    if pos is not None and pos < len(arg_vids):
        inv = _call(gx, _ident(gx, "inv"), (arg_vids[pos],))
        arg_vids = arg_vids[:pos] + [inv] + arg_vids[pos + 1 :]
    cpos = _GEN_PROB_COMPLEMENT_POSITIONS.get(family)
    if cpos is not None and cpos < len(arg_vids):
        comp = _gen_binary_expr(gx, _integer(gx, 1), "-", arg_vids[cpos])
        arg_vids = arg_vids[:cpos] + [comp] + arg_vids[cpos + 1 :]
    return arg_vids


def _row_array_names(body: tuple[IRNode, ...]) -> dict[str, int]:
    """The sampled names stored as arrays of event vectors.

    A latent drawn under one or more loop axes from a family with its own
    event axes is allocated as an array, over those axes, whose elements
    are the family's event vectors (`_emit_storage_alloc`). An entrywise
    read must index the array and then the vector.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The IR body, searched through marginalization scopes.

    Returns
    -------
    dict[str, int]
        Each such name and the number of its outer array axes.
    """
    out: dict[str, int] = {}
    for node in body:
        if isinstance(node, IRSample):
            own = _family_event_rank(node.family)
            outer = len(node.plate.batch_dims) + len(node.plate.event_dims) - own
            if own > 0 and outer > 0:
                out[node.name] = outer
        elif isinstance(node, IRMarginalize):
            out.update(_row_array_names(node.scope))
    return out


def _family_event_rank(family: str) -> int:
    """The rank of the family's own event shape.

    Reads
    [`FamilyMeta.event_rank`][quivers.transpile.family_meta.FamilyMeta],
    which is 0 for a scalar family, 1 for a vector family and 2 for a
    matrix one. `Truncated` is the sole wrapper family without a
    `FAMILY_META` entry; it wraps a scalar base distribution and so
    produces a scalar draw.
    """
    meta = FAMILY_META.get(family)
    if meta is not None:
        return meta.event_rank
    if family in _WRAPPER_TARGET_NAMES:
        return 0
    raise UnsupportedConstruct("qvr-gen", [f"family:{family}: no Gen.jl event rank"])


#: The grafted Gen.jl distribution of each family call written through
#: an alternative parameterization: a Bernoulli given by its logits
#: scores each outcome by the log-sigmoid of the signed logits.
_LOGIT_CALLEES: dict[tuple[str, tuple[str, ...]], str] = {
    ("Bernoulli", ("logits",)): "qvr_bernoulli_logit",
}


def _gen_target_name(family: str) -> str:
    """Look up the Gen.jl distribution constructor for a family.

    Consults `FAMILY_META[family].target_names["gen"]` first; falls
    back to the wrapper-family table when the family is a
    composite-distribution wrapper that has no flat Gen.jl name.
    """
    if family == "Categorical":
        # QVR class codes count from zero, Gen's `categorical` from one;
        # the grafted zero-based form draws and scores QVR codes.
        return "qvr_categorical"
    meta = FAMILY_META.get(family)
    if meta is not None and "gen" in meta.target_names:
        return meta.target_names["gen"]
    fallback = _WRAPPER_TARGET_NAMES.get(family)
    if fallback is not None:
        return fallback
    raise UnsupportedConstruct("qvr-gen", [f"family:{family}:no-gen-target"])


# ---------------------------------------------------------------------------
# Wrapper-family rendering.
# ---------------------------------------------------------------------------


def _render_truncated_call(
    gx: _GenCtx,
    *,
    args: tuple[IRArg, ...],
    arg_ctx: _ArgCtx,
) -> str:
    """`truncated(<inner-dist>, <lo>, <hi>)` via Distributions.truncated.

    The wrapped distribution is encoded as an
    [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] referring
    to a morphism whose `init_family` is the base distribution.
    """
    if not args or not isinstance(args[0], IRArgFamilyRef):
        raise UnsupportedConstruct(
            "qvr-gen",
            ["family:Truncated: first arg must be IRArgFamilyRef"],
        )
    inner_vid = _render_inner_family(
        gx,
        args[0].name,
        arg_ctx=arg_ctx,
    )
    bounds = tuple(_render_arg(gx, a, arg_ctx=arg_ctx) for a in args[1:])
    return _call(gx, _ident(gx, "truncated"), (inner_vid, *bounds))


def _render_inner_family(gx: _GenCtx, morphism_name: str, *, arg_ctx: _ArgCtx) -> str:
    """Render the distribution call referenced by a morphism name.

    Reads the morphism's `init_family` clause from the carried
    morphism table to recover `(family_name, args)`, then dispatches
    through the family-name lookup as if it were an inline sample
    call.
    """
    decl = gx.morphisms.get(morphism_name)
    if decl is None:
        raise UnsupportedConstruct(
            "qvr-gen",
            [f"family_ref:{morphism_name}: morphism not declared in transpile context"],
        )
    init = getattr(decl, "init_family", None)
    if init is None or getattr(init, "family", None) is None:
        raise UnsupportedConstruct(
            "qvr-gen",
            [f"family_ref:{morphism_name}: morphism has no init_family declaration"],
        )
    family = init.family
    raw_args = tuple(
        value for _, value in bind_family_arguments(family, init.args or ())
    )
    callee_name = _gen_target_name(family)
    arg_vids = tuple(_lift_raw_arg(gx, a, arg_ctx=arg_ctx) for a in raw_args)
    return _call(gx, _ident(gx, callee_name), arg_vids)


def _lift_raw_arg(gx: _GenCtx, raw: object, *, arg_ctx: _ArgCtx) -> str:
    """Lift a morphism-table raw arg (str / number / IRArg) to a vertex."""
    if not isinstance(raw, (IRArg, int, float, str)):
        raw = atom_value(raw)
    if isinstance(raw, IRArg):
        return _render_arg(gx, raw, arg_ctx=arg_ctx)
    if isinstance(raw, (int, float)):
        return _number(gx, float(raw))
    if isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            return _ident(gx, raw)
        return _number(gx, value)
    raise UnsupportedConstruct("qvr-gen", [f"raw-arg:{type(raw).__name__}"])


_WrapperBuilder = Callable[[_GenCtx, tuple[IRArg, ...], _ArgCtx], str]


#: Per-wrapper-family builder dispatch.
_WRAPPER_BUILDERS: dict[str, _WrapperBuilder] = {
    "Truncated": (
        lambda gx, args, arg_ctx: _render_truncated_call(
            gx,
            args=args,
            arg_ctx=arg_ctx,
        )
    ),
}


def _build_wrapper_call(
    gx: _GenCtx,
    *,
    family: str,
    args: tuple[IRArg, ...],
    arg_ctx: _ArgCtx,
) -> str:
    builder = _WRAPPER_BUILDERS.get(family)
    if builder is None:
        raise UnsupportedConstruct(
            "qvr-gen", [f"wrapper-family:{family}: no Gen.jl builder"]
        )
    return builder(gx, args, arg_ctx)


# ---------------------------------------------------------------------------
# Trace-call assembly.
# ---------------------------------------------------------------------------


def _trace_address(gx: _GenCtx, name: str, loop_indices: tuple[str, ...]) -> str:
    """Build the trace address: ``:name`` or ``(:name, m_0, m_1, ...)``."""
    if not loop_indices:
        return _quote_sym(gx, name)
    parts = [_quote_sym(gx, name), *[_ident(gx, i) for i in loop_indices]]
    return _tuple(gx, tuple(parts))


def _trace_call(
    gx: _GenCtx,
    *,
    dist_vid: str,
    name: str,
    loop_indices: tuple[str, ...],
) -> str:
    """`@trace(<dist>, <address>)` in parenthesised macro form.

    The space-separated form ``@trace <dist> <addr>`` is ambiguous when
    ``<dist>`` is a function call and ``<addr>`` is a tuple literal:
    Julia's parser reads ``@trace truncated_normal(args) (:y, i)`` as
    ``@trace truncated_normal(args)(:y, i)`` (a chained call) and
    rejects the result with "extra token after end of expression". The
    parenthesised form binds unambiguously.
    """
    addr = _trace_address(gx, name, loop_indices)
    return _macro_call_parens(gx, "trace", (dist_vid, addr))


def _macro_call_parens(gx: _GenCtx, macro_name: str, args: tuple[str, ...]) -> str:
    """`@<macro_name>(arg1, arg2, ...)` (parenthesised macro args).

    Use this form when any of `args` is a function call whose trailing
    parenthesis would chain with the next argument under
    space-separated juxtaposition. Used by `_trace_call` for the
    `@trace(<dist>, <addr>)` shape.
    """
    mc = gx.v("macrocall_expression", "mc")
    mid = gx.v("macro_identifier", "mid")
    gx.e(mid, _ident(gx, macro_name))
    al = gx.v("argument_list", "mal")
    for a in args:
        gx.e(al, a)
    gx.e(mc, mid)
    gx.e(mc, al)
    return mc


# ---------------------------------------------------------------------------
# The renderer itself.
# ---------------------------------------------------------------------------


class GenRenderer(RendererBase):
    """Render an IR program as a Gen.jl `@gen function` definition.

    Inherits the IR-walk dispatch from
    [`RendererBase`][quivers.transpile.renderers.RendererBase].
    Overrides the per-node dispatch in `render` to collect declarations
    plus statements into the `@gen function` envelope; implements
    `declare`, `sample`, `marginalize`, `broadcast` per the Gen.jl
    idiom described in the module docstring.
    """

    target: str = "gen"

    # ------------------------------------------------------------------
    # protocol setup
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("julia")

    # ------------------------------------------------------------------
    # render: wrap the IR walk in the `@gen function` envelope
    # ------------------------------------------------------------------

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Render a program to the target's panproto schema.

        Parameters
        ----------
        ir
            The lowered program.

        Returns
        -------
        panproto.Schema
            The target program, in the target grammar's theory.
        """
        assert_no_dangling_refs(ir)
        assert_no_dropped_param_map(ir, self.target)
        proto = self.target_protocol()
        sb = proto.schema()
        gx = _GenCtx(sb=sb, cards={}, morphisms={}, name_plates=name_plate_map(ir))

        _harvest_cards(ir, gx)
        gx.shapes = JuliaShapes(
            name_event_rank=name_event_rank_map(ir),
            name_array_rank=name_array_rank_map(ir),
            nested_names=nested_tower_names(ir),
            program_values=frozenset(name_rank_map(ir)),
            row_arrays=_row_array_names(ir.body),
        )

        # Sort inputs alphabetically so the emitted `model(...)` signature
        # matches the probe driver's positional-arg convention (the probe
        # iterates the per-point `data` dict in sorted-key order).
        for inp in sorted(ir.inputs, key=lambda i: i.name):
            gx.params.append(inp.name)
            gx.inputs_by_name[inp.name] = inp
            gx.decl_axes[inp.name] = inp.plate.batch_dims

        # `RenderContext` is the inherited carrier; we re-use it for the
        # required dispatch signatures but the per-render scratch
        # lives on `_GenCtx`.
        ctx = RenderContext(sb=sb, morphisms={}, defines={})
        blk = gx.v("block", "body")
        gx.body = blk
        self._gx = gx
        self._module = ir.module
        self._ir = ir
        try:
            for node in ir.body:
                self._emit_node(ctx, node)
        finally:
            del self._gx
            del self._module
            del self._ir

        if gx.return_names:
            ret = gx.v("return_statement", "ret")
            if len(gx.return_names) == 1:
                gx.e(ret, _ident(gx, gx.return_names[0]))
            else:
                tup = gx.v("tuple_expression", "rtup")
                for n in gx.return_names:
                    gx.e(tup, _ident(gx, n))
                gx.e(ret, tup)
            gx.e(blk, ret)

        # Probe drivers eval the source and look up `Main.model`; the
        # function is always named `model` so the harness has a
        # canonical entry point regardless of the QVR module name.
        fn = _function_def(
            gx,
            name="model",
            params=tuple(gx.params),
            body_vid=blk,
        )
        mc = _macro_call_body(gx, "gen", fn)
        src = gx.v("source_file", "src")
        # Gen.jl has no built-in `truncated_normal`. When the IR samples
        # or observes from `TruncatedNormal`, graft the helper from
        # [`runtime_gen.jl`][quivers.transpile.runtime_gen] onto the
        # module above the `@gen function model` so the model body can
        # call `truncated_normal(loc, scale, low, high)`. The graft
        # carries its own `using Gen` / `using Distributions`
        # statements; subsequent `@gen` macrocalls see the imported
        # names through normal Julia name lookup.
        if (
            any(_ir_uses_family(ir.body, f) for f in _GEN_RUNTIME_HELPER_FAMILIES)
            or _ir_uses_logit_callee(ir.body)
            or qiec_helper_families_used(ir, self.target)
            or _ir_has_factor(ir.body)
        ):
            _graft_runtime_gen_helper(gx, src)
        gx.e(src, mc)
        render_computations_dynamic(sb, ir, target=self.target, root=src)
        return sb.build()

    # ------------------------------------------------------------------
    # Per-IRNode dispatch
    # ------------------------------------------------------------------

    def _emit_node(self, ctx: RenderContext, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(node)
                return
            self._emit_sample(node, observed=False, via=None)
            return
        if isinstance(node, IRObserve):
            self._emit_sample(
                IRSample(
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    constraint=node.constraint,
                    plate=node.plate,
                ),
                observed=True,
                via=node.via,
            )
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(node)
            return
        if isinstance(node, IRScore):
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRCall):
            self._emit_call(node)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._gx.return_names = tuple(node.names)
            return
        raise UnsupportedConstruct("qvr-gen", [f"node:{type(node).__name__}"])

    # ------------------------------------------------------------------
    # Sample / observe emission
    # ------------------------------------------------------------------

    def _emit_sample(
        self,
        node: IRSample,
        *,
        observed: bool,
        via: str | None,
    ) -> None:
        """Emit declaration + for-loop nest for a batched sample / observe.

        For the non-batched case (`plate.batch_dims` empty) emits a
        single scalar `name = @trace(<dist>, :name)`. For one-or-more
        batch dims, allocates `Vector{T}(undef, B)` then emits nested
        for-loops that fill the storage element-wise via `@trace`.

        `observed=True` skips the per-step storage allocation (the
        observed value is supplied externally via the function
        signature) and drops the `name[m] = ...` LHS so the trace
        macro emits as a bare statement.
        """
        gx = self._gx
        # Split the site's event dims into the trailing ones the
        # family produces natively and the leading residual the source
        # declared with `[over=<Axis>]`. The residual is iid
        # replication of a scalar draw, so it becomes extra innermost
        # loop axes rather than part of the distribution call: a
        # `Normal` site declared `over=LatentDim` fills a 32-by-2
        # `Array` at addresses `(:Z_mat, m_Item, m_LatentDim)`.
        own_start = len(node.plate.event_dims) - _family_event_rank(node.family)
        residual_event = node.plate.event_dims[:own_start]
        own_event = node.plate.event_dims[own_start:]
        loop_dims = (*node.plate.batch_dims, *residual_event)
        if not loop_dims:
            self._emit_scalar_sample(node, observed=observed)
            return
        if not observed:
            self._emit_storage_alloc(node, loop_dims)
        # The declaration is now visible to subsequent steps as a
        # batched draw on `node.plate.batch_dims`. The residual event
        # axes stay out of the table: index threading matches a
        # reference against the batch loops that surround it, and no
        # downstream step iterates a residual event axis.
        gx.decl_axes[node.name] = node.plate.batch_dims
        self._emit_loop_nest(
            node,
            observed=observed,
            via=via,
            loop_dims=loop_dims,
            own_event=own_event,
        )

    def _emit_gp_block(self, node: IRSample) -> None:
        """Emit a Gaussian-process sample as three Gen.jl statements:

            __gp_mean_<name> = zeros(N)
            __gp_cov_<name>  = _qvr_rbf_kernel(x, length_scale, jitter)
            <name> = @trace(mvnormal(__gp_mean_<name>, __gp_cov_<name>), :<name>)

        The ``_qvr_rbf_kernel`` helper lives in
        [`runtime_gen.jl`][quivers.transpile.runtime_gen] and is
        grafted into the emit via the existing runtime-helper graft
        when GP appears in the IR.
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-gen",
                ["family:GP: expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-gen",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        gx = self._gx
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = zeros(N)
        mean_rhs = _call(
            gx,
            _ident(gx, "zeros"),
            (_integer(gx, n),),
        )
        gx.place(_assignment(gx, _ident(gx, mean_name), mean_rhs))
        # __gp_cov_<name> = _qvr_rbf_kernel(x, ls, jitter)
        cov_rhs = _call(
            gx,
            _ident(gx, "_qvr_rbf_kernel"),
            (
                _ident(gx, x),
                _float_lit(gx, ls),
                _float_lit(gx, jitter),
            ),
        )
        gx.place(_assignment(gx, _ident(gx, cov_name), cov_rhs))
        # <name> = @trace(mvnormal(__gp_mean_<name>, __gp_cov_<name>), :<name>)
        mvn_call = _call(
            gx,
            _ident(gx, "mvnormal"),
            (_ident(gx, mean_name), _ident(gx, cov_name)),
        )
        trace = _trace_call(
            gx,
            dist_vid=mvn_call,
            name=node.name,
            loop_indices=(),
        )
        gx.place(_assignment(gx, _ident(gx, node.name), trace))

    def _emit_scalar_sample(self, node: IRSample, *, observed: bool) -> None:
        gx = self._gx
        dist_vid = self._build_dist_call(
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            event_dims=node.plate.event_dims,
            arg_ctx=_ArgCtx(),
        )
        trace = _trace_call(gx, dist_vid=dist_vid, name=node.name, loop_indices=())
        if observed:
            gx.place(trace)
        else:
            stmt = _assignment(gx, _ident(gx, node.name), trace)
            gx.place(stmt)
            gx.decl_axes[node.name] = ()

    def _emit_storage_alloc(self, node: IRSample, loop_dims: tuple[Dim, ...]) -> None:
        """Pre-allocate the dense array the plate loop fills.

        One axis gives `Vector{T}(undef, B)`; several give
        `Array{T, N}(undef, B0, ..., BN-1)`, matching the flat
        multi-index the loop nest assigns through.
        """
        gx = self._gx
        elem_type = _element_type_for(node.constraint, node.plate)
        size_vids = tuple(_dim_size_vid(gx, dim) for dim in loop_dims)
        alloc = _array_alloc(
            gx,
            elem_type=elem_type,
            size_vids=size_vids,
        )
        stmt = _assignment(gx, _ident(gx, node.name), alloc)
        gx.place(stmt)

    def _emit_loop_nest(
        self,
        node: IRSample,
        *,
        observed: bool,
        via: str | None,
        loop_dims: tuple[Dim, ...],
        own_event: tuple[Dim, ...],
    ) -> None:
        gx = self._gx
        loop_names = tuple(
            _loop_var_for(gx, str(dim.name), node.name) for dim in loop_dims
        )
        # The batch-loop binding for the current step: axis name →
        # loop variable identifier. Each declared ref whose
        # declaration sits on an axis present here picks up the
        # corresponding loop variable as an index. Only the batch
        # half participates: a residual event axis replicates the
        # draw and indexes nothing the arguments carry.
        batch_loops = {
            str(dim.name): lv
            for dim, lv in zip(
                node.plate.batch_dims,
                loop_names[: len(node.plate.batch_dims)],
                strict=True,
            )
        }
        arg_ctx = _ArgCtx(
            batch_loops=batch_loops,
            via_indexer=via,
            via_loop_var=loop_names[-1] if via else None,
            axis_loops={
                str(dim.name): lv for dim, lv in zip(loop_dims, loop_names, strict=True)
            },
        )
        dist_vid = self._build_dist_call(
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            event_dims=own_event,
            arg_ctx=arg_ctx,
        )
        trace = _trace_call(
            gx,
            dist_vid=dist_vid,
            name=node.name,
            loop_indices=loop_names,
        )
        if observed:
            inner_stmt = trace
        else:
            lhs = self._build_indexed_lhs(node.name, loop_names)
            inner_stmt = _assignment(gx, lhs, trace)

        current_stmts: tuple[str, ...] = (inner_stmt,)
        for dim, loop_var in zip(
            reversed(loop_dims),
            reversed(loop_names),
            strict=True,
        ):
            size_vid = _dim_size_vid(gx, dim)
            lo = _integer(gx, 1)
            fs = _for_stmt(
                gx,
                var=loop_var,
                lo=lo,
                hi=size_vid,
                body_stmts=current_stmts,
            )
            current_stmts = (fs,)
        gx.place(current_stmts[0])
        for dim in loop_dims:
            gx.used_axes.add(str(dim.name))

    def _build_indexed_lhs(self, name: str, loop_names: tuple[str, ...]) -> str:
        gx = self._gx
        if not loop_names:
            return _ident(gx, name)
        return _index_into(
            gx,
            _ident(gx, name),
            tuple(_ident(gx, lv) for lv in loop_names),
        )

    def _build_dist_call(
        self,
        *,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        event_dims: tuple[Dim, ...],
        arg_ctx: _ArgCtx,
    ) -> str:
        """Build the distribution call vertex for one element draw.

        Wrapper families route through their dedicated builder;
        otherwise the family looks up the Gen.jl callee in
        `FAMILY_META` and renders args inline, threading the
        scalar→vector broadcast for any arg whose constraint expects
        a vector / matrix.
        """
        if family in _WRAPPER_BUILDERS:
            return _build_wrapper_call(
                self._gx,
                family=family,
                args=args,
                arg_ctx=arg_ctx,
            )
        gx = self._gx
        if family == "MixtureNormal":
            return self._mixture_normal_call(args, arg_names, arg_ctx)
        callee_name = _LOGIT_CALLEES.get((family, tuple(arg_names)))
        if callee_name is None:
            callee_name = _gen_target_name(family)
        arg_vids: list[str] = []
        for arg, name in zip(args, arg_names, strict=False):
            rendered = _render_arg(
                gx,
                arg,
                arg_ctx=dataclasses.replace(
                    arg_ctx, scalar_slot=scalar_parameter(family, name)
                ),
            )
            rendered = _broadcast_arg_to_event(
                gx,
                rendered,
                arg=arg,
                family=family,
                arg_name=name,
                event_dims=event_dims,
                inputs_by_name=gx.inputs_by_name,
            )
            arg_vids.append(rendered)
        arg_vids = _gen_transform_args(gx, arg_vids, family)
        prefix = _HALF_BASE_TARGETS.get(family)
        if prefix is not None:
            _, location = prefix
            arg_vids.insert(0, _number(gx, float(location)))
        return _call(gx, _ident(gx, callee_name), tuple(arg_vids))

    def _mixture_normal_call(
        self,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        arg_ctx: _ArgCtx,
    ) -> str:
        """Build `HomogeneousMixture(normal, [0, 0])(w, mu, sigma)`.

        Gen.jl ships no `mixture_normal` distribution but it ships the
        mixture *combinator*: `HomogeneousMixture(base, dims)` lifts a
        base distribution to a finite mixture whose first argument is
        the weight vector and whose remaining arguments are the base's
        own parameters stacked along a trailing component axis. `dims`
        records the rank of each base parameter, both zero for
        `normal(mu, sigma)`. The resulting distribution's `logpdf` is
        the weighted log-sum-exp the QVR likelihood scores, so the emit
        is the same closed form rather than an approximation.

        The component axis lives inside the mixture, so the three
        arguments are the whole per-component vectors and pick up no
        loop index from the surrounding plate.
        """
        gx = self._gx
        weights, loc, scale = mixture_normal_components("gen", args, arg_names)
        dims = gx.v("vector_expression", "mxdims")
        gx.e(dims, _integer(gx, 0))
        gx.e(dims, _integer(gx, 0))
        mixture = _call(
            gx,
            _ident(gx, _gen_target_name("MixtureNormal")),
            (_ident(gx, "normal"), dims),
        )
        return _call(
            gx,
            mixture,
            tuple(
                _render_arg(gx, arg, arg_ctx=arg_ctx) for arg in (weights, loc, scale)
            ),
        )

    # ------------------------------------------------------------------
    # Deterministic let-bindings
    # ------------------------------------------------------------------

    def _emit_deterministic(self, node: IRDeterministic) -> None:
        gx = self._gx
        refuse_nested_slice_gather("gen", node, gx.shapes)
        adapter = _JlCtxAdapter(gx, "gen")
        shapes = gx.shapes.scoped_to(len(node.plate.batch_dims))
        # A body that reduces an event axis (`mu = sum(z_row * w_row)`)
        # cannot take the `@.` form below: the macro broadcasts the
        # reducing call itself, applying `sum` to each scalar element
        # and leaving the inner product uncomputed. Such a body emits
        # its own dotted operators instead.
        reduces_axis = let_expr_has_axis_reduction(shapes, node.expr)
        # A binding of one event, a vector `z = tanh(a)` drawn over an
        # `over` axis, is no batch the `@.` form below covers; its
        # elementwise calls are dotted, as Julia applies `tanh` to a
        # number and not to a vector.
        one_event = bool(node.plate.event_dims) and not node.plate.batch_dims
        rhs = render_let_expr_julia(
            adapter, node.expr, shapes=shapes, dotted=reduces_axis or one_event
        )
        # A binding whose expression produces fewer axes than its
        # plate is one value the plate replicates: `cell0 =
        # cell_score[0, 0]` under a `Resp` plate reaches the observe
        # loop as a per-response vector. Julia has no implicit
        # replication, so the missing axes are filled explicitly.
        missing = (
            len(node.plate.batch_dims)
            + len(node.plate.event_dims)
            - infer_array_rank(shapes, node.expr)
            if node.plate.batch_dims
            else 0
        )
        # The batch axes of a let-binding are its plate's, which the
        # plan reads off the binding's checked type. References to
        # `node.name` inside a loop pick up the loop index for every
        # such axis; a binding with none is one value every row reads.
        if missing > 0:
            rhs = _call(
                gx,
                _ident(gx, "fill"),
                (
                    rhs,
                    *(
                        _dim_size_vid(gx, dim)
                        for dim in node.plate.batch_dims[:missing]
                    ),
                ),
            )
        axes = node.plate.batch_dims
        gx.decl_axes[node.name] = axes
        if axes and not reduces_axis and missing <= 0:
            # The let body evaluates to a Vector / matrix shaped along
            # the batch axes. Julia's scalar `+ * - /` operators reject
            # a scalar mixed with a vector, so wrap the RHS in `@.`
            # (fused-broadcast macro) to promote every arithmetic
            # operator inside the body to its dotted form.
            rhs = _macro_call_space(gx, ".", (rhs,))
        stmt = _assignment(gx, _ident(gx, node.name), rhs)
        gx.place(stmt)

    # ------------------------------------------------------------------
    # Marginalize: lower to IRSample + scope inline
    # ------------------------------------------------------------------

    def _emit_marginalize(self, ctx: RenderContext, node: IRMarginalize) -> None:
        """Integrate a finite latent out and trace the reduced density.

        The atoms are scored through Distributions.jl, which the
        module's runtime graft brings into scope, in the form the
        Turing renderer emits for the same Julia; the reduced log
        weight is then traced as a factor choice under the
        `:qvr_factor` namespace, so Gen's trace scores it the way it
        scores a draw.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        node : IRMarginalize
            The block.
        """
        del ctx
        gx = self._gx
        emitter = _GenMarginalizer()
        context = emitter.context(gx, self._ir)
        emitter.marginalize(context, self._matrix_views(gx, context, node))

    def _matrix_views(
        self, gx: _GenCtx, context: _TuringCtx, node: IRMarginalize
    ) -> IRMarginalize:
        """Read the block's per-row draws as matrices.

        Gen draws an event-shaped site over a plate one row at a time,
        as a vector of vectors, while the marginalize emission reads
        such a site as a matrix with one row per draw. Each such site
        the block reads is stacked into a matrix bound under a fresh
        name, and the block is rewritten to read that name.

        Parameters
        ----------
        gx : _GenCtx
            The Gen render context.
        context : _TuringCtx
            The marginalize context, whose plate tables learn the
            stacked names.
        node : IRMarginalize
            The block.

        Returns
        -------
        IRMarginalize
            The block over the stacked names.
        """
        renames: dict[str, str] = {}
        for name in sorted(_marginal_ref_names(node)):
            plate = context.sample_plates.get(name)
            if plate is None or not (plate.batch_dims and plate.event_dims):
                continue
            if name not in gx.decl_axes:
                continue
            stacked = f"__marg_{name}_rows"
            renames[name] = stacked
            # `stack` lays the rows out as columns; the transpose reads
            # them back as rows.
            columns = _call(gx, _ident(gx, "stack"), (_ident(gx, name),))
            rhs = _call(gx, _ident(gx, "permutedims"), (columns,))
            gx.place(_assignment(gx, _ident(gx, stacked), rhs))
            context.sample_plates[stacked] = plate
            if name in context.batch_shaped_names:
                context.batch_shaped_names.add(stacked)
        if not renames:
            return node
        return _rename_marginal_refs(node, renames)

    # ------------------------------------------------------------------
    # Score: bind value, then `@addlogprob!`
    # ------------------------------------------------------------------

    def _emit_score(self, ctx: RenderContext, node: IRScore) -> None:
        """``<name> = <expr>; @trace(_qvr_qiec_factor(<name>), :<name>)``.

        Gen scores the choices a trace holds, so the weight is traced
        as a factor choice: a one-point distribution whose log density
        is the weight. The expression renders elementwise, so a weight
        it combines with a plated value scales each row before a
        reduction sums them.
        """
        gx = self._gx
        del ctx
        rhs = render_let_expr_julia(
            _JlCtxAdapter(gx, "gen"), node.expr, shapes=gx.shapes, dotted=True
        )
        bind = _assignment(gx, _ident(gx, node.name), rhs)
        gx.place(bind)
        factor = _call(gx, _ident(gx, "_qvr_qiec_factor"), (_ident(gx, node.name),))
        address = _gen_binary_expr(
            gx, _quote_sym(gx, _FACTOR_NAMESPACE), "=>", _quote_sym(gx, node.name)
        )
        trace = _macro_call_parens(gx, "trace", (factor, address))
        gx.place(trace)

    def _emit_call(self, node: IRCall) -> None:
        """Place a call of a module computation in the model body.

        Parameters
        ----------
        node : IRCall
            The call.
        """
        gx = self._gx

        def bind_argument(name: str, expression: LetExprNode) -> None:
            rhs = render_let_expr_julia(
                _JlCtxAdapter(gx, "gen"), expression, shapes=gx.shapes
            )
            gx.place(_assignment(gx, _ident(gx, name), rhs))

        emit_call_julia(
            gx.sb,
            node,
            self._module,
            gx.operations_bound,
            target="gen",
            body="model",
            bind_argument=bind_argument,
            place=gx.place,
        )

    # ------------------------------------------------------------------
    # Protocol-required dispatch points. The Gen renderer overrides
    # `_emit_node` at a higher level (it emits the whole Gen
    # `@gen function ... end` shell inline rather than block-by-block),
    # so the granular `declare` / `sample` / `marginalize` /
    # `broadcast` methods below are no-ops: the work happens in the
    # `_emit_*` methods further up.
    # ------------------------------------------------------------------

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """Emit the declaration of a named variable in ``block``.

        Parameters
        ----------
        ctx
            The render call's mutable state.
        name
            The variable's name.
        constraint
            The variable's support.
        plate
            The variable's event and batch dimensions.
        block
            Where the declaration lands in the target program.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        del ctx, name, constraint, plate, block
        return ""

    def sample(
        self,
        ctx: RenderContext,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        """Emit the ``~`` statement of a sample or observe step.

        Parameters
        ----------
        ctx
            The render call's mutable state.
        name
            The variable's name.
        family
            The distribution family's QVR name.
        args
            The family's arguments, in family order.
        arg_names
            The parameter name of each argument.
        constraint
            The variable's support.
        plate
            The variable's event and batch dimensions.
        observed
            Whether the site is conditioned on data.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        del (
            ctx,
            name,
            family,
            args,
            arg_names,
            constraint,
            plate,
            observed,
        )
        return ""

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Emit the scope integrating a discrete latent out.

        Parameters
        ----------
        ctx
            The render call's mutable state.
        node
            The marginalize block.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        self._emit_marginalize(ctx, node)
        return ""

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit the target's broadcast of ``value`` to ``target_shape``.

        Parameters
        ----------
        ctx
            The render call's mutable state.
        value
            The argument broadcast.
        target_shape
            The shape broadcast to.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        del ctx
        gx = self._gx
        value_vid = _render_arg(gx, value, arg_ctx=_ArgCtx())
        return _broadcast_to_shape(gx, value_vid, target_shape)


# ---------------------------------------------------------------------------
# Card harvesting from the IR.
# ---------------------------------------------------------------------------


def _harvest_cards(ir: IRProgram, gx: _GenCtx) -> None:
    """Walk the IR collecting `DimStatic` sizes by name into `gx.cards`."""
    for inp in ir.inputs:
        _harvest_plate(inp.plate, gx)
    for node in ir.body:
        _harvest_node(node, gx)


def _harvest_node(node: IRNode, gx: _GenCtx) -> None:
    if isinstance(node, (IRSample, IRObserve, IRDataInput, IRDeterministic)):
        _harvest_plate(node.plate, gx)
    if isinstance(node, IRMarginalize):
        _harvest_plate(node.plate, gx)
        for inner in node.scope:
            _harvest_node(inner, gx)


def _harvest_plate(plate: Plate, gx: _GenCtx) -> None:
    for dim in (*plate.event_dims, *plate.batch_dims):
        if isinstance(dim, DimStatic):
            gx.cards[dim.name] = dim.size


# ---------------------------------------------------------------------------
# Runtime-helper graft: `truncated_normal` as a Gen.Distribution subclass.
#
# Gen.jl ships `normal`, `uniform`, `beta`, ... as built-in distributions
# but no `truncated_normal`. The transpile-time graft parses the
# hand-written helper at [`runtime_gen.jl`][quivers.transpile.runtime_gen]
# once at module load through panproto's Julia tree-sitter grammar; per-
# render, it copies every grafted vertex / constraint / edge into the
# per-render schema (with fresh vertex ids) and attaches the runtime's
# top-level statements as `child_of` of the emitted `source_file` above
# the `@gen function model` macrocall.
#
# The emit is structurally a normal Julia source file: `using Gen`,
# `using Distributions`, the `TruncatedNormalDist` struct, the
# `Gen.random` / `Gen.logpdf` / `Gen.logpdf_grad` methods, the gradient
# predicates, and a callable instance bound to `truncated_normal`.
# `@trace truncated_normal(loc, scale, low, high) (:y, m)` call sites
# in the model body then resolve to the grafted callable via normal
# Julia name lookup.
# ---------------------------------------------------------------------------


_RUNTIME_GEN_PATH = pathlib.Path(__file__).resolve().parent.parent / "runtime_gen.jl"


#: Families whose Gen.jl emit relies on the
#: [`runtime_gen.jl`][quivers.transpile.runtime_gen] helper subtree.
#: Gen.jl ships `normal`, `uniform`, `beta`, etc. as built-in
#: distributions but lacks these; the renderer grafts the helper
#: when the IR samples or observes from any of them.
_GEN_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset(
    {
        "Categorical",
        "TruncatedNormal",
        "Logistic",
        "BetaBinomial",
        "HalfStudentT",
        "Kumaraswamy",
        "ContinuousBernoulli",
        "LKJCholesky",
        "MatrixNormal",
        "LogNormal",
        "Weibull",
        "GP",
    }
)


def _load_runtime_gen_schema() -> tuple[panproto.Schema, str, tuple[str, ...]]:
    """Parse [`runtime_gen.jl`][quivers.transpile.runtime_gen] through
    panproto's Julia tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `source_file` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `source_file` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "julia",
        _RUNTIME_GEN_PATH.read_bytes(),
        str(_RUNTIME_GEN_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "source_file"),
        None,
    )
    if src_id is None:
        raise RuntimeError(f"`source_file` not found in parse of {_RUNTIME_GEN_PATH}")
    children_with_sb: list[tuple[int, str]] = []
    for edge in schema.edges:
        if edge.src != src_id:
            continue
        sb_val = next(
            (
                int(c.value)
                for c in schema.constraints_for(edge.tgt)
                if c.sort == "start-byte"
            ),
            0,
        )
        children_with_sb.append((sb_val, edge.tgt))
    children_with_sb.sort()
    return schema, src_id, tuple(child for _, child in children_with_sb)


_RUNTIME_GEN_SCHEMA, _RUNTIME_GEN_SOURCE_ID, _RUNTIME_GEN_TOP_LEVEL = (
    _load_runtime_gen_schema()
)


def _subtree_vertex_ids(schema: panproto.Schema, roots: tuple[str, ...]) -> set[str]:
    """Return every vertex id reachable from `roots` via outgoing edges."""
    seen: set[str] = set(roots)
    frontier: list[str] = list(roots)
    while frontier:
        src = frontier.pop()
        for edge in schema.edges:
            if edge.src == src and edge.tgt not in seen:
                seen.add(edge.tgt)
                frontier.append(edge.tgt)
    return seen


_RUNTIME_GEN_SUBTREE = _subtree_vertex_ids(_RUNTIME_GEN_SCHEMA, _RUNTIME_GEN_TOP_LEVEL)


#: The address namespace factor choices are traced under, so a reader
#: of the trace tells a scored weight from a draw by its address.
_FACTOR_NAMESPACE = "qvr_factor"


def _ir_uses_logit_callee(body: tuple[IRNode, ...]) -> bool:
    """Whether a site draws or scores through a grafted logit distribution.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The IR body, searched through marginalization scopes.

    Returns
    -------
    bool
        True when a sample or observe calls a family under a
        parameterization `_LOGIT_CALLEES` spells.
    """
    for node in body:
        if isinstance(node, IRSample | IRObserve) and (
            (node.family, tuple(node.arg_names)) in _LOGIT_CALLEES
        ):
            return True
        if isinstance(node, IRMarginalize) and _ir_uses_logit_callee(node.scope):
            return True
    return False


def _ir_has_factor(body: tuple[IRNode, ...]) -> bool:
    """True iff `body` scores or calls, either of which traces a factor.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The plan body.

    Returns
    -------
    bool
        Whether the module needs the factor distribution.
    """
    return any(isinstance(node, (IRScore, IRCall, IRMarginalize)) for node in body)


def _ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any [`IRSample`][quivers.transpile.ir.IRSample] or
    [`IRObserve`][quivers.transpile.ir.IRObserve] in `body` (including
    nested [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes)
    samples from `family`."""
    for node in body:
        if isinstance(node, (IRSample, IRObserve)) and node.family == family:
            return True
        if isinstance(node, IRMarginalize) and (
            node.family == family or _ir_uses_family(node.scope, family)
        ):
            return True
    return False


def _graft_runtime_gen_helper(gx: _GenCtx, source_vid: str) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_gen.jl` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each
    top-level child as a `child_of` of `source_vid` in source order.
    The grafted top-level children appear above the `@gen function
    model` macrocall in the emit.
    """
    src_schema = _RUNTIME_GEN_SCHEMA
    subtree = _RUNTIME_GEN_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        new = gx.fresh("rg")
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        gx.sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            gx.sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            gx.sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_GEN_TOP_LEVEL:
        gx.sb.edge(source_vid, id_map[child_old], "child_of")


__all__ = ["GenRenderer"]


def _marginal_ref_names(node: IRMarginalize) -> set[str]:
    """Every name a marginalize block's arguments and scope read.

    Parameters
    ----------
    node : IRMarginalize
        The block.

    Returns
    -------
    set[str]
        The referenced names.
    """
    names: set[str] = set()
    for arg in node.args:
        _collect_arg_names(arg, names)
    for inner in node.scope:
        if isinstance(inner, (IRSample, IRObserve)):
            for arg in inner.args:
                _collect_arg_names(arg, names)
        elif isinstance(inner, (IRDeterministic, IRScore)):
            _collect_let_expr_names(inner.expr, names)
    return names


def _collect_arg_names(arg: IRArg, names: set[str]) -> None:
    """Add the names an argument tree references.

    Parameters
    ----------
    arg : IRArg
        The argument.
    names : set[str]
        The names found so far.
    """
    if isinstance(arg, IRArgRef):
        names.add(arg.name)
        for index in arg.indices:
            _collect_arg_names(index, names)
    elif isinstance(arg, IRArgBroadcast):
        _collect_arg_names(arg.value, names)
    elif isinstance(arg, IRArgList):
        for element in arg.elements:
            _collect_arg_names(element, names)
    elif isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            _collect_arg_names(row, names)


def _collect_let_expr_names(expr: LetExprNode, names: set[str]) -> None:
    """Add the names a let expression references.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    names : set[str]
        The names found so far.
    """
    if isinstance(expr, LetExprVar):
        names.add(expr.name)
    elif isinstance(expr, LetExprBinOp):
        _collect_let_expr_names(expr.left, names)
        _collect_let_expr_names(expr.right, names)
    elif isinstance(expr, LetExprUnaryOp):
        _collect_let_expr_names(expr.operand, names)
    elif isinstance(expr, LetExprCall):
        for argument in expr.args:
            _collect_let_expr_names(argument, names)
    elif isinstance(expr, LetExprIndex):
        _collect_let_expr_names(expr.array, names)
        for index in expr.indices:
            _collect_let_expr_names(index, names)
    elif isinstance(expr, LetExprList):
        for item in expr.items:
            _collect_let_expr_names(item, names)
    elif isinstance(expr, LetExprFactor):
        if expr.body is not None:
            _collect_let_expr_names(expr.body, names)
        for case in expr.cases:
            _collect_let_expr_names(case.value, names)


def _rename_marginal_refs(
    node: IRMarginalize, renames: dict[str, str]
) -> IRMarginalize:
    """Rewrite a marginalize block to read renamed bindings.

    Parameters
    ----------
    node : IRMarginalize
        The block.
    renames : dict[str, str]
        Each name's replacement.

    Returns
    -------
    IRMarginalize
        The block with every reference renamed.
    """
    scope: list[IRNode] = []
    for inner in node.scope:
        if isinstance(inner, (IRSample, IRObserve)):
            scope.append(
                inner.with_(args=tuple(_rename_arg(arg, renames) for arg in inner.args))
            )
        elif isinstance(inner, (IRDeterministic, IRScore)):
            scope.append(inner.with_(expr=_rename_let_expr(inner.expr, renames)))
        else:
            scope.append(inner)
    return node.with_(
        args=tuple(_rename_arg(arg, renames) for arg in node.args),
        scope=tuple(scope),
    )


def _rename_arg(arg: IRArg, renames: dict[str, str]) -> IRArg:
    """Rename the references in an argument tree.

    Parameters
    ----------
    arg : IRArg
        The argument.
    renames : dict[str, str]
        Each name's replacement.

    Returns
    -------
    IRArg
        The renamed argument.
    """
    if isinstance(arg, IRArgRef):
        return arg.with_(
            name=renames.get(arg.name, arg.name),
            indices=tuple(_rename_arg(index, renames) for index in arg.indices),
        )
    if isinstance(arg, IRArgBroadcast):
        return arg.with_(value=_rename_arg(arg.value, renames))
    if isinstance(arg, IRArgList):
        return arg.with_(
            elements=tuple(_rename_arg(element, renames) for element in arg.elements)
        )
    if isinstance(arg, IRArgMatrix):
        return arg.with_(
            rows=tuple(
                IRArgList(
                    elements=tuple(
                        _rename_arg(element, renames) for element in row.elements
                    )
                )
                for row in arg.rows
            )
        )
    return arg


def _rename_let_expr(expr: LetExprNode, renames: dict[str, str]) -> LetExprNode:
    """Rename the references in a let expression.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    renames : dict[str, str]
        Each name's replacement.

    Returns
    -------
    LetExprNode
        The renamed expression.
    """
    if isinstance(expr, LetExprVar):
        return LetExprVar(name=renames.get(expr.name, expr.name))
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_rename_let_expr(expr.left, renames),
            right=_rename_let_expr(expr.right, renames),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            op=expr.op, operand=_rename_let_expr(expr.operand, renames)
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(_rename_let_expr(argument, renames) for argument in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_rename_let_expr(expr.array, renames),
            indices=tuple(_rename_let_expr(index, renames) for index in expr.indices),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(_rename_let_expr(item, renames) for item in expr.items)
        )
    if isinstance(expr, LetExprFactor):
        shadowed = {
            name: value
            for name, value in renames.items()
            if name not in {binder.var for binder in expr.binders}
        }
        return LetExprFactor(
            binders=expr.binders,
            body=None if expr.body is None else _rename_let_expr(expr.body, shadowed),
            cases=tuple(
                LetFactorCase(
                    label=case.label, value=_rename_let_expr(case.value, shadowed)
                )
                for case in expr.cases
            ),
        )
    return expr


class _GenMarginalizer(TuringRenderer):
    """The Turing renderer's marginalize emission, writing into a Gen body.

    Both renderers emit Julia, and a Gen model has Distributions.jl in
    scope through its runtime graft, so the atoms' densities, weights,
    and reduction are the same statements; only the step that adds the
    reduced weight to the joint differs, and here it is a traced factor
    choice.
    """

    target: str = "gen"
    #: Gen exports a `logpdf` of its own, so the atoms are scored
    #: through Distributions.jl's by its qualified name.
    log_density_function: str = "Distributions.logpdf"

    def __init__(self) -> None:
        super().__init__()
        self._gx: _GenCtx | None = None

    def context(self, gx: _GenCtx, ir: IRProgram) -> _TuringCtx:
        """The Turing-style context over a Gen model body.

        Parameters
        ----------
        gx : _GenCtx
            The Gen render context, whose body the statements go into.
        ir : IRProgram
            The lowered root, read for its plates and shapes.

        Returns
        -------
        _TuringCtx
            A context sharing the Gen schema and body.
        """
        self._gx = gx
        context = _TuringCtx(
            sb=gx.sb,
            morphisms={},
            lets={},
            counter=[gx.n + _MARGINALIZE_ID_OFFSET],
            cards=dict(ir.cards),
            body=gx.body,
            input_plates={item.name: item.plate for item in ir.inputs},
            sample_plates={},
            batch_shaped_names=set(),
            name_plates=gx.name_plates,
            shapes=gx.shapes,
            module=ir.module,
        )
        _seed_sample_plates(context, ir.body)
        _seed_batch_shaped(context, ir.body)
        return context

    def _add_log_weight(self, ctx: _TuringCtx, weight: str, name: str) -> None:
        del ctx
        gx = self._gx
        if gx is None:
            raise RuntimeError("the Gen marginalizer has no body to write into")
        factor = _call(gx, _ident(gx, "_qvr_qiec_factor"), (weight,))
        address = _gen_binary_expr(
            gx, _quote_sym(gx, _FACTOR_NAMESPACE), "=>", _quote_sym(gx, name)
        )
        gx.place(_macro_call_parens(gx, "trace", (factor, address)))


#: Turing-style vertex ids are `v<n>`; the offset keeps the ids a
#: marginalize emission allocates apart from any a second emission in
#: the same model allocates, each starting past the Gen counter.
_MARGINALIZE_ID_OFFSET = 1_000_000
