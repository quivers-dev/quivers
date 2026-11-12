"""Turing.jl renderer: consume an [`IRProgram`][quivers.transpile.ir.IRProgram]
and emit a Julia panproto schema for the Turing.jl model idiom.

Output shape (LDA canonical):

```julia
@model function model(alpha, beta, word_idx, w)
    theta ~ filldist(Dirichlet(fill(alpha, 3)), 20)
    phi ~ filldist(Dirichlet(fill(beta, 200)), 3)
    z ~ filldist(Categorical(theta), 20)
    w .~ Categorical.(eachrow(phi)[z[word_idx]])
    return theta
end
```

Per the Turing.jl idiom the renderer:

* declares no separate variable bindings; declarations are the `~` /
  `.~` statements themselves;
* lifts each batch axis into a `filldist(<Family>(<args>), B)` call
  when none of the family's args depend on that batch's index; into
  an `arraydist([<Family>(<args[i]>) for i in 1:B])` call when at
  least one arg has an index expression rooted in that batch;
* integrates an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
  latent out: one scored copy of the scope per atom of its support,
  a max-shifted `logsumexp` across the atoms written in `Base`, and
  the reduced weight added through `@addlogprob!`;
* broadcasts scalars to vector / matrix shapes via Julia's `fill`;
* renders list literals as `[<e0>, <e1>, ...]` (Julia vector_expression)
  and matrix literals as `[<row0>; <row1>; ...]`
  (Julia matrix_expression);
* renders an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
  reference to a `Truncated` wrapper as `truncated(<base>, lower=L,
  upper=U)`.
"""

from __future__ import annotations

from collections.abc import Callable
import pathlib

import panproto
import torch.distributions.constraints as _torch_constraints

from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprIdent,
    Module,
    ProgramDecl,
)
from quivers.dsl.ast_nodes.let_expressions import LetExprNode
from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.qiec_ir import IRQiecModule
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._pipeline import (
    EmitPretty,
    parser_registry,
    target_protocol,
)
from quivers.transpile.renderers._ranks import (
    carried_dims,
    name_plate_map,
    name_rank_map,
)
from quivers.transpile.renderers._rows import scalar_parameter, visited_position
from quivers.transpile.renderers._julia_helpers import (
    JuliaShapes,
    infer_array_rank,
    let_expr_has_axis_reduction,
    name_array_rank_map,
    nested_tower_names,
    refuse_nested_slice_gather,
    rebase_index_literals,
    render_let_expr_julia,
)
from quivers.transpile._resolve import (
    build_let_table,
    build_morphism_table,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
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
)
from quivers.transpile.plan import Lower
from quivers.transpile.renderers._python_helpers import (
    MarginalizeBody,
    marginal_support_size,
    marginal_weight_probs,
    marginalize_body,
    marginalize_fibration,
    name_event_rank_map,
)
from quivers.transpile.renderers._base import (
    refuse_ungrouped_row_marginalize,
    BlockKind,
    IRMarginalAtom,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dropped_param_map,
    logit_weighted,
    mixture_normal_components,
)
from quivers.transpile.renderers._qiec import (
    emit_call_julia,
    render_computations_dynamic,
    qiec_helper_families_used,
)


# ---------------------------------------------------------------------------
# Julia tree-sitter helpers: scoped to the Turing renderer; not promoted
# to `_base.py` because no other backend builds Julia source.
# ---------------------------------------------------------------------------


def _vertex(sb: panproto.SchemaBuilder, counter: list[int], kind: str) -> str:
    """Allocate a fresh vertex id and register it under `kind`."""
    counter[0] += 1
    vid = f"v{counter[0]}"
    sb.vertex(vid, kind)
    return vid


def _literal(
    sb: panproto.SchemaBuilder, counter: list[int], kind: str, text: str
) -> str:
    """Register a vertex of `kind` whose `literal-value` constraint
    carries `text`."""
    vid = _vertex(sb, counter, kind)
    sb.constraint(vid, "literal-value", text)
    return vid


def _identifier(sb: panproto.SchemaBuilder, counter: list[int], text: str) -> str:
    return _literal(sb, counter, "identifier", text)


def _operator(sb: panproto.SchemaBuilder, counter: list[int], text: str) -> str:
    return _literal(sb, counter, "operator", text)


def _integer(sb: panproto.SchemaBuilder, counter: list[int], value: int) -> str:
    return _literal(sb, counter, "integer_literal", str(value))


def _float(sb: panproto.SchemaBuilder, counter: list[int], value: float) -> str:
    return _literal(sb, counter, "float_literal", repr(value))


def _number(sb: panproto.SchemaBuilder, counter: list[int], value: float) -> str:
    """Pick `integer_literal` for whole values; `float_literal` for the
    rest. Mirrors the Julia source idiom of writing `3` instead of
    `3.0` for plate sizes and integer arguments."""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return _integer(sb, counter, int(value))
    return _float(sb, counter, float(value))


def _argument_list(
    sb: panproto.SchemaBuilder, counter: list[int], children: tuple[str, ...]
) -> str:
    """Build an `argument_list` vertex aliased from `tuple_expression`.

    Julia tree-sitter aliases `argument_list` from `tuple_expression`;
    panproto's emitter uses the `pre-alias-symbol` constraint to route
    the production to the parenthesised form.
    """
    vid = _vertex(sb, counter, "argument_list")
    sb.constraint(vid, "pre-alias-symbol", "tuple_expression")
    for child in children:
        sb.edge(vid, child, "child_of")
    return vid


def _call(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    callee: str,
    children: tuple[str, ...],
) -> str:
    """`<callee>(<children[0]>, <children[1]>, ...)`."""
    vid = _vertex(sb, counter, "call_expression")
    sb.edge(vid, callee, "child_of")
    sb.edge(vid, _argument_list(sb, counter, children), "child_of")
    return vid


def _broadcast_call(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    callee: str,
    children: tuple[str, ...],
) -> str:
    """`<callee>.(<children[0]>, <children[1]>, ...)` (the dot-broadcast
    form of a function call)."""
    vid = _vertex(sb, counter, "broadcast_call_expression")
    sb.edge(vid, callee, "child_of")
    sb.edge(vid, _argument_list(sb, counter, children), "child_of")
    return vid


def _index_expr(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    array: str,
    indices: tuple[str, ...],
) -> str:
    """`<array>[<indices[0]>, <indices[1]>, ...]`.

    Julia's `index_expression` production is `primary [ _array ]` where
    `_array` is one of `vector_expression`, `matrix_expression`, or
    `comprehension_expression`. The index list is wrapped in a
    `vector_expression` so the pretty emitter selects the bracketed
    form.
    """
    vid = _vertex(sb, counter, "index_expression")
    sb.edge(vid, array, "child_of")
    inner = _vertex(sb, counter, "vector_expression")
    for idx in indices:
        sb.edge(inner, idx, "child_of")
    sb.edge(vid, inner, "child_of")
    return vid


def _assignment(
    sb: panproto.SchemaBuilder, counter: list[int], lhs: str, rhs: str
) -> str:
    """`<lhs> = <rhs>` (regular Julia assignment)."""
    vid = _vertex(sb, counter, "assignment")
    sb.edge(vid, lhs, "child_of")
    sb.edge(vid, _operator(sb, counter, "="), "child_of")
    sb.edge(vid, rhs, "child_of")
    return vid


def _tilde(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    lhs: str,
    rhs: str,
    *,
    broadcast: bool = False,
) -> str:
    """`<lhs> ~ <rhs>` or `<lhs> .~ <rhs>` (Turing.jl sampling).

    Encoded as a `compound_assignment_expression` whose operator is the
    literal `~` or `.~` token.
    """
    vid = _vertex(sb, counter, "compound_assignment_expression")
    sb.edge(vid, lhs, "child_of")
    sb.edge(vid, _operator(sb, counter, ".~" if broadcast else "~"), "child_of")
    sb.edge(vid, rhs, "child_of")
    return vid


def _vector_literal(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    elements: tuple[str, ...],
) -> str:
    """`[<e0>, <e1>, ...]` (Julia `vector_expression`)."""
    vid = _vertex(sb, counter, "vector_expression")
    for el in elements:
        sb.edge(vid, el, "child_of")
    return vid


def _matrix_literal(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    rows: tuple[tuple[str, ...], ...],
) -> str:
    """`[<r0[0]> <r0[1]> ...; <r1[0]> <r1[1]> ...; ...]` (Julia
    `matrix_expression`)."""
    vid = _vertex(sb, counter, "matrix_expression")
    for row in rows:
        row_vid = _vertex(sb, counter, "matrix_row")
        for el in row:
            sb.edge(row_vid, el, "child_of")
        sb.edge(vid, row_vid, "child_of")
    return vid


def _range(sb: panproto.SchemaBuilder, counter: list[int], lo: str, hi: str) -> str:
    """`<lo>:<hi>` (Julia `range_expression`, an infix `:`)."""
    vid = _vertex(sb, counter, "range_expression")
    sb.edge(vid, lo, "child_of")
    sb.edge(vid, hi, "child_of")
    return vid


def _comprehension(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    body: str,
    binder_var: str,
    binder_range: str,
) -> str:
    """`[<body> for <binder_var> in <binder_range>]` (Julia comprehension).

    Used by the `arraydist([... for i in 1:B])` index-dependent fallback.
    """
    comp = _vertex(sb, counter, "comprehension_expression")
    sb.edge(comp, body, "child_of")
    fc = _vertex(sb, counter, "for_clause")
    fb = _vertex(sb, counter, "for_binding")
    sb.edge(fb, _identifier(sb, counter, binder_var), "child_of")
    sb.edge(fb, binder_range, "child_of")
    sb.edge(fc, fb, "child_of")
    sb.edge(comp, fc, "child_of")
    return comp


def _for_stmt(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    *,
    var: str,
    lo: str,
    hi: str,
    body_stmts: tuple[str, ...],
) -> str:
    """`for <var> in <lo>:<hi> <body> end` (Julia `for_statement`)."""
    fs = _vertex(sb, counter, "for_statement")
    fb = _vertex(sb, counter, "for_binding")
    sb.edge(fb, _identifier(sb, counter, var), "child_of")
    sb.edge(fb, _range(sb, counter, lo, hi), "child_of")
    blk = _vertex(sb, counter, "block")
    for stmt in body_stmts:
        sb.edge(blk, stmt, "child_of")
    sb.edge(fs, fb, "child_of")
    sb.edge(fs, blk, "child_of")
    return fs


def _colon(sb: panproto.SchemaBuilder, counter: list[int]) -> str:
    """The `:` full-axis subscript, an `operator` vertex in the Julia
    grammar's `index_expression` bracket list."""
    return _operator(sb, counter, ":")


def _array_alloc(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    *,
    elem_type: str,
    sizes: tuple[str, ...],
) -> str:
    """`Array{<elem_type>, <N>}(undef, <s0>, ..., <sN-1>)`.

    Pre-allocates the dense container a plated `~` statement fills one
    slice at a time, in the (batch, event) axis order the QVR site
    declares.
    """
    pt = _vertex(sb, counter, "parametrized_type_expression")
    sb.edge(pt, _identifier(sb, counter, "Array"), "child_of")
    curly = _vertex(sb, counter, "curly_expression")
    sb.edge(curly, _identifier(sb, counter, elem_type), "child_of")
    sb.edge(curly, _integer(sb, counter, len(sizes)), "child_of")
    sb.edge(pt, curly, "child_of")
    call = _vertex(sb, counter, "call_expression")
    sb.edge(call, pt, "child_of")
    sb.edge(
        call,
        _argument_list(sb, counter, (_identifier(sb, counter, "undef"), *sizes)),
        "child_of",
    )
    return call


def _return(sb: panproto.SchemaBuilder, counter: list[int], value: str) -> str:
    """`return <value>`."""
    vid = _vertex(sb, counter, "return_statement")
    sb.edge(vid, value, "child_of")
    return vid


def _function_def(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    *,
    name: str,
    params: tuple[str, ...],
    body_vid: str,
) -> str:
    """`function <name>(<params...>) <body> end`."""
    fn = _vertex(sb, counter, "function_definition")
    sig = _vertex(sb, counter, "signature")
    call = _vertex(sb, counter, "call_expression")
    sb.edge(call, _identifier(sb, counter, name), "child_of")
    args = _argument_list(
        sb, counter, tuple(_identifier(sb, counter, p) for p in params)
    )
    sb.edge(call, args, "child_of")
    sb.edge(sig, call, "child_of")
    sb.edge(fn, sig, "child_of")
    sb.edge(fn, body_vid, "child_of")
    return fn


def _macro_call(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    macro_name: str,
    body_vid: str,
) -> str:
    """`@<macro_name> <body>` (no parens; long-form).

    Used for `@model function ... end`.
    """
    mc = _vertex(sb, counter, "macrocall_expression")
    mid = _vertex(sb, counter, "macro_identifier")
    sb.edge(mid, _identifier(sb, counter, macro_name), "child_of")
    margs = _vertex(sb, counter, "macro_argument_list")
    sb.edge(margs, body_vid, "child_of")
    sb.edge(mc, mid, "child_of")
    sb.edge(mc, margs, "child_of")
    return mc


# ---------------------------------------------------------------------------
# `TuringRenderer`
# ---------------------------------------------------------------------------


class TuringRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as a
    Turing.jl model.

    Overrides [`render`][quivers.transpile.renderers.RendererBase.render]
    to wrap the IR-walk with the `@model function model(...) ... end`
    prologue / epilogue Turing.jl requires. The per-node dispatches
    (`declare`, `sample`, `marginalize`, `broadcast`) accumulate Julia
    schema vertices into a single function body block.
    """

    target: str = "turing"
    #: The function scoring a marginalized block's atoms; Turing reads
    #: Distributions.jl's `logpdf` unqualified.
    log_density_function: str = "logpdf"

    def __init__(
        self,
        *,
        morphisms: dict | None = None,
        lets: dict | None = None,
    ) -> None:
        """Construct a renderer.

        `morphisms` and `lets` are the resolved tables from the surface
        module; the renderer consults `morphisms` when an
        [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] needs
        the referenced morphism's `~ Family(...)` init clause. Both
        default to empty so callers driving a synthetic
        [`IRProgram`][quivers.transpile.ir.IRProgram] (tests) can
        instantiate the renderer with no surface context.

        Parameters
        ----------
        morphisms
            The module's morphism declarations, by name.
        lets
            The module's define expressions, by name.
        """
        self._morphisms: dict = morphisms or {}
        self._lets: dict = lets or {}

    # ----- protocol / context plumbing -----

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("julia")

    # ----- top-level render -----

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
        assert_no_dropped_param_map(ir, self.target)
        proto = self.target_protocol()
        sb = proto.schema()
        counter: list[int] = [0]
        # Source file shell.
        source = _vertex(sb, counter, "source_file")
        body = _vertex(sb, counter, "block")

        # The Turing.jl function signature carries every IRDataInput
        # in alphabetical name order. The in-container probe (which
        # has no access to the IR's lowering order) passes observed
        # values as positional args in `sort(collect(keys(data)))`
        # order; sorting here keeps the two sides in lockstep so
        # callers never have to know the original IR order.
        params = tuple(sorted(inp.name for inp in ir.inputs))

        # Walk the body. Each node emits a `~` / `=` / `return`
        # statement into `body`.
        ctx = _TuringCtx(
            sb=sb,
            morphisms=self._morphisms,
            lets=self._lets,
            counter=counter,
            cards=dict(ir.cards),
            body=body,
            input_plates={inp.name: inp.plate for inp in ir.inputs},
            sample_plates={},
            batch_shaped_names=set(),
            name_plates=name_plate_map(ir),
            shapes=JuliaShapes(
                name_event_rank=name_event_rank_map(ir),
                name_array_rank=name_array_rank_map(ir),
                nested_names=nested_tower_names(ir),
                program_values=frozenset(name_rank_map(ir)),
            ),
            module=ir.module,
        )
        # Pre-populate the sample-plate table by walking the body so
        # observe / marginalize bodies can detect index-dependent args
        # against the originating sample's plate.
        _seed_sample_plates(ctx, ir.body)
        # Pre-populate the batch-shaped-names table: every
        # IRDataInput / IRSample with non-empty batch_dims, plus every
        # IRDeterministic whose RHS transitively references one such
        # name. Drives the broadcast-dot fallback for observes whose
        # `loc=` etc. is a let-bound vector (e.g. `mu = a + b *
        # x_design` in bayes_linear_regression).
        _seed_batch_shaped(ctx, ir.body)
        for node in ir.body:
            self._dispatch(ctx, node)

        fn = _function_def(sb, counter, name="model", params=params, body_vid=body)
        macro = _macro_call(sb, counter, "model", fn)
        # Turing.jl + Distributions.jl ship a large catalogue of
        # distributions but lack `ContinuousBernoulli` and an RBF
        # kernel builder. When the IR samples or observes through
        # either, graft the helper at
        # [`runtime_turing.jl`][quivers.transpile.runtime_turing] onto
        # the source above the `@model function model` macrocall so the
        # body's `~ ContinuousBernoulli(...)` / `_qvr_rbf_kernel(...)`
        # call sites resolve through normal Julia name lookup.
        if any(
            _ir_uses_family(ir.body, f) for f in _TURING_RUNTIME_HELPER_FAMILIES
        ) or qiec_helper_families_used(ir, self.target):
            _graft_runtime_turing_helper(sb, counter, source)
        sb.edge(source, macro, "child_of")
        render_computations_dynamic(sb, ir, target=self.target, root=source)
        return sb.build()

    # ----- IRNode dispatch (overrides RendererBase._dispatch_node) -----

    def _dispatch(self, ctx: _TuringCtx, node: IRNode) -> None:
        """Route one IR body node to the right Turing.jl emission."""
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=False,
            )
            return
        if isinstance(node, IRObserve):
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=True,
                via=node.via,
            )
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRCall):
            self._emit_call(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_return(ctx, node.names)
            return
        if isinstance(node, IRDataInput):
            return
        raise UnsupportedConstruct("qvr-turing", [f"node:{type(node).__name__}"])

    def _emit_call(self, ctx: _TuringCtx, node: IRCall) -> None:
        """Place a call of a module computation in the model body.

        Parameters
        ----------
        ctx : _TuringCtx
            The render context.
        node : IRCall
            The call.
        """
        sb, counter = ctx.sb, ctx.counter

        def bind_argument(name: str, expression: LetExprNode) -> None:
            rhs = render_let_expr_julia(
                _JlCtxShim(sb, counter, ctx.cards, "turing"),
                expression,
                shapes=ctx.shapes,
            )
            stmt = _assignment(sb, counter, _identifier(sb, counter, name), rhs)
            sb.edge(ctx.body, stmt, "child_of")

        emit_call_julia(
            sb,
            node,
            ctx.module,
            ctx.operations_bound,
            target="turing",
            body=ctx.body,
            bind_argument=bind_argument,
            place=lambda statement: sb.edge(ctx.body, statement, "child_of"),
        )

    def _emit_gp_block(
        self,
        ctx: _TuringCtx,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as a triple of Julia
        statements inside the ``@model`` body:

            __gp_mean_f = zeros(N)
            __gp_cov_f  = _qvr_rbf_kernel(x, length_scale, jitter)
            f ~ MvNormal(__gp_mean_f, __gp_cov_f)

        The ``_qvr_rbf_kernel`` helper is defined in
        [`runtime_turing.jl`][quivers.transpile.runtime_turing] and
        is grafted onto the emit when GP is in the IR (handled by
        the existing
        `_graft_runtime_turing_helper`
        path with GP added to the helper-family set).
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-turing",
                ["family:GP: expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-turing",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        sb, counter = ctx.sb, ctx.counter
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = zeros(N)
        mean_lhs = _identifier(sb, counter, mean_name)
        mean_rhs = _call(
            sb,
            counter,
            _identifier(sb, counter, "zeros"),
            (_integer(sb, counter, n),),
        )
        sb.edge(
            ctx.body,
            _assignment(sb, counter, mean_lhs, mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = _qvr_rbf_kernel(x, ls, jitter)
        cov_lhs = _identifier(sb, counter, cov_name)
        cov_rhs = _call(
            sb,
            counter,
            _identifier(sb, counter, "_qvr_rbf_kernel"),
            (
                _identifier(sb, counter, x),
                _float(sb, counter, ls),
                _float(sb, counter, jitter),
            ),
        )
        sb.edge(
            ctx.body,
            _assignment(sb, counter, cov_lhs, cov_rhs),
            "child_of",
        )
        # f ~ MvNormal(__gp_mean_f, __gp_cov_f)
        f_lhs = _identifier(sb, counter, node.name)
        mvn_rhs = _call(
            sb,
            counter,
            _identifier(sb, counter, "MvNormal"),
            (
                _identifier(sb, counter, mean_name),
                _identifier(sb, counter, cov_name),
            ),
        )
        sb.edge(
            ctx.body,
            _tilde(sb, counter, f_lhs, mvn_rhs),
            "child_of",
        )
        ctx.sample_plates[node.name] = node.plate

    # ----- declare: Turing.jl has no separate declaration block -----

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op: in Turing.jl, declarations are the `~` statements
        themselves. The function signature is populated from the
        IRDataInput list in [`render`][quivers.transpile.renderers.TuringRenderer.render].

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

    # ----- sample / observe emission -----

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
        *,
        via: str | None = None,
    ) -> SchemaFragment:
        """Emit one `~` / `.~` statement for a sample or observe step.

        Plate handling:
        * `batch_dims=()` and no `via`: plain `name ~ Family(args)`.
        * `batch_dims=(B,)` and no arg references a batch-plated name
          via index: `name ~ filldist(Family(args), B)`.
        * Otherwise (index-dependent args or a `via` fibration):
          broadcast-dot form `name .~ Family.(idx_args)`, or the
          `arraydist([Family(args[i]) for i in 1:B])` fallback when no
          via is present.

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
        via
            The fibration an observe gathers its rows through, or ``None``.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        del constraint  # output support already encoded in the family choice
        assert isinstance(ctx, _TuringCtx)
        ctx.sample_plates[name] = plate
        sb, counter = ctx.sb, ctx.counter

        meta = _meta_or_raise(family)
        target_dist = _ZERO_BASED_FAMILIES.get(
            family, _turing_parameterized_name(family, arg_names, meta)
        )
        if target_dist is None:
            raise UnsupportedConstruct(
                "qvr-turing", [f"family:{family}: no Turing.jl mapping"]
            )

        if family == "MixtureNormal":
            return self._emit_mixture_normal(
                ctx,
                name=name,
                target_dist=target_dist,
                args=args,
                arg_names=arg_names,
                plate=plate,
            )

        # Split the site's event dims into the trailing ones the
        # family produces natively (`Normal` none, `Dirichlet` one,
        # `LKJCholesky` two) and the leading residual the source
        # declared with `[over=<Axis>]`. The residual is iid
        # replication, so it joins the batch dims in the `filldist`
        # wrapper; without it a `Normal` site declared `over=LatentDim`
        # renders rank-1 where the reference measure is rank-2.
        own_start = len(plate.event_dims) - meta.event_rank
        residual_event = plate.event_dims[:own_start]
        own_event = plate.event_dims[own_start:]

        # Detect index-dependent arg shapes against the surrounding
        # observe/sample's batch axes. The presence of `via` is the
        # strongest signal: it always rewrites the indexing to thread
        # through the fibration variable.
        index_dep = via is not None or _args_have_batch_index(
            args,
            ctx.sample_plates,
            plate,
            ctx.batch_shaped_names,
        )

        if own_event and (plate.batch_dims or residual_event):
            # A batched draw from a family with its own event shape.
            # `filldist(Dirichlet(...), 8)` would produce a 16-by-8
            # matrix: every Turing product of a multivariate family
            # is event-major, where the QVR site is batch-major
            # (`array[8] simplex[16]` in Stan, `dims=("State","Obs")`
            # in PyMC). The pre-allocated container plus a per-row `~`
            # is the Turing idiom that lands the axes in the declared
            # order, and DynamicPPL resolves each `name[i, :]` varname
            # against the whole-array entry the caller conditions on.
            return self._emit_row_plate(
                ctx,
                name=name,
                target_dist=target_dist,
                args=args,
                arg_names=arg_names,
                family=family,
                meta=meta,
                plate=plate,
                own_event=own_event,
                residual_event=residual_event,
                index_dep=index_dep,
            )

        # A parameter whose every axis the site replicates over, such as
        # a vector `z` over `Feat` filling a `Normal` replicated over
        # `Feat`, holds one value per draw. Broadcasting the family over
        # it lays one distribution per position; a plain `Family(z)`
        # would hand the scalar constructor the whole vector. A
        # parameter carrying the site's batch axis is read per row by
        # the index-dependent forms below.
        loop_dims = (*plate.batch_dims, *residual_event)
        if (
            via is None
            and not own_event
            and (residual_event or not index_dep)
            and _visits_parameter(ctx, family, args, arg_names, loop_dims)
        ):
            return self._emit_elementwise(
                ctx,
                name=name,
                target_dist=target_dist,
                args=args,
                arg_names=arg_names,
                family=family,
                loop_dims=loop_dims,
                observed=observed,
            )

        if index_dep and residual_event:
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    f"sample:event-axis-lift:{name}: the site declares "
                    f"the residual event "
                    f"{'axis' if len(residual_event) == 1 else 'axes'} "
                    f"{[d.name for d in residual_event]!r} and its "
                    f"arguments index the surrounding batch axis. The "
                    f"broadcast-dot and `arraydist` forms both fix the "
                    f"drawn value at the family's own rank, so the "
                    f"residual axis has no Turing.jl spelling on "
                    f"either path"
                ],
            )

        lhs = _identifier(sb, counter, name)

        if index_dep and observed and via is not None:
            # Index-dependent observe with a `via` fibration: emit the
            # broadcast-dot form `name .~ Family.(rewritten_args)`.
            rhs_args = tuple(
                _arg_to_julia(ctx, a, via=via, family=family) for a in args
            )
            rhs = _dist_expr(
                sb,
                counter,
                target_dist,
                rhs_args,
                family,
                dotted=True,
                event_dims=own_event,
            )
            stmt = _tilde(sb, counter, lhs, rhs, broadcast=True)
            sb.edge(ctx.body, stmt, "child_of")
            return ""

        if index_dep and observed and via is None:
            # Index-dependent observe whose arg(s) reference a
            # batch-shaped name (e.g. a let-bound vector) directly,
            # without a `via` fibration. The plain
            # `filldist(Family(args), B)` form requires the args to
            # be scalar; wrap the per-element distribution array in
            # `product_distribution(Family.(args))` and emit the
            # scalar tilde. DynamicPPL no longer accepts `.~` over
            # an array of distributions; `product_distribution` is
            # the supported replacement. A table read as one event per
            # row (a gathered `probs` matrix) broadcasts over its rows.
            rhs_args = tuple(
                _rows_of_table(ctx, _arg_to_julia(ctx, a, family=family), a, meta, n)
                for a, n in zip(args, arg_names, strict=True)
            )
            elemwise = _dist_expr(
                sb,
                counter,
                target_dist,
                rhs_args,
                family,
                dotted=True,
                event_dims=own_event,
            )
            rhs = _call(
                sb,
                counter,
                _identifier(sb, counter, "product_distribution"),
                (elemwise,),
            )
            stmt = _tilde(sb, counter, lhs, rhs)
            sb.edge(ctx.body, stmt, "child_of")
            return ""

        if index_dep and not observed:
            # Index-dependent latent: fall back to `arraydist([Family(
            # args[i]) for i in 1:B])`. Pick the first batch dim as
            # the comprehension axis.
            if not plate.batch_dims:
                # No batch axis to iterate; treat as a scalar call.
                rhs = self._family_call(
                    ctx, target_dist, args, family, plate.event_dims
                )
            else:
                rhs = self._arraydist_call(
                    ctx, target_dist, args, plate.batch_dims[0], family
                )
            stmt = _tilde(sb, counter, lhs, rhs)
            sb.edge(ctx.body, stmt, "child_of")
            return ""

        # Plain / batch-wrapped form. Apply filldist over the batch
        # dims plus any residual event dims the family does not itself
        # produce. Pre-process args: a scalar IRArgRef against a
        # vector-shaped arg constraint (e.g. Dirichlet
        # `concentration`) is wrapped in `fill(<ref>, <event_size>)` so
        # the emitted call matches the Turing.jl idiom of a
        # fully-shaped vector for the event-dim arg.
        promoted = tuple(
            _promote_scalar_ref(a, name, plate, meta, ctx.shapes.name_array_rank)
            for name, a in zip(arg_names, args, strict=False)
        )
        dist = self._family_call(ctx, target_dist, promoted, family, own_event)
        fill_dims = (*plate.batch_dims, *residual_event)
        if fill_dims:
            size_vids = tuple(_dim_to_size(sb, counter, dim) for dim in fill_dims)
            dist = _call(
                sb,
                counter,
                _identifier(sb, counter, "filldist"),
                (dist, *size_vids),
            )
        stmt = _tilde(sb, counter, lhs, dist)
        sb.edge(ctx.body, stmt, "child_of")
        return ""

    def _emit_elementwise(
        self,
        ctx: _TuringCtx,
        *,
        name: str,
        target_dist: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        family: str,
        loop_dims: tuple[Dim, ...],
        observed: bool,
    ) -> SchemaFragment:
        """Emit a scalar family broadcast over the parameters it replicates.

        ``y ~ product_distribution(Normal.(z, 0.5))`` for an observation
        and ``b ~ arraydist(Normal.(z, 0.5))`` for a latent: each entry
        of the site is drawn at the entries of its parameters at the
        same position, scalars shared by every position.

        Parameters
        ----------
        ctx : _TuringCtx
            The render call's mutable state.
        name : str
            The site's name.
        target_dist : str
            The Turing.jl constructor.
        args : tuple[IRArg, ...]
            The family's arguments.
        arg_names : tuple[str, ...]
            The parameter each argument fills.
        family : str
            The family's QVR name.
        loop_dims : tuple[Dim, ...]
            The axes the site replicates the family over.
        observed : bool
            Whether the site is conditioned on data.

        Returns
        -------
        SchemaFragment
            ``""``; the statement is attached to the body.

        Raises
        ------
        UnsupportedConstruct
            If a parameter varies along some of the replicated axes but
            not all, which a broadcast lays out at the parameter's shape
            rather than the site's, or a parameter reading a whole event
            carries axes, which a broadcast would split.
        """
        sb, counter = ctx.sb, ctx.counter
        axes = tuple(str(dim.name) for dim in loop_dims)
        for arg, arg_name in zip(args, arg_names, strict=False):
            plate = _arg_plate(ctx, arg)
            if plate is None:
                continue
            carried = tuple(
                str(dim.name) for dim in (*plate.batch_dims, *plate.event_dims)
            )
            if carried and not scalar_parameter(family, arg_name):
                raise UnsupportedConstruct(
                    "qvr-turing",
                    [
                        f"sample:partial-axis-parameter:{name}: the {family} "
                        f"parameter {arg_name!r} reads a whole event, which a "
                        f"family call broadcast over the site's axes would split"
                    ],
                )
            if carried and carried != axes:
                raise UnsupportedConstruct(
                    "qvr-turing",
                    [
                        f"sample:partial-axis-parameter:{name}: a parameter "
                        f"varies along {list(carried)!r} of the axes "
                        f"{list(axes)!r} the site replicates over, which a "
                        f"broadcast family call lays out at the parameter's "
                        f"shape rather than the site's"
                    ],
                )
        elemwise = _dist_expr(
            sb,
            counter,
            target_dist,
            tuple(_arg_to_julia(ctx, a, family=family) for a in args),
            family,
            dotted=True,
        )
        wrapper = "product_distribution" if observed else "arraydist"
        rhs = _call(sb, counter, _identifier(sb, counter, wrapper), (elemwise,))
        stmt = _tilde(sb, counter, _identifier(sb, counter, name), rhs)
        sb.edge(ctx.body, stmt, "child_of")
        return ""

    def _emit_mixture_normal(
        self,
        ctx: _TuringCtx,
        *,
        name: str,
        target_dist: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> SchemaFragment:
        """Emit `name ~ filldist(MixtureModel(Normal.(mu, sigma), w), N)`.

        Distributions.jl ships no `MixtureNormal` constructor, but it
        ships `MixtureModel(components, prior)`, whose `logpdf` is the
        log-sum-exp of the prior-weighted component log-densities: the
        QVR likelihood's own closed form. `Normal.(mu, sigma)`
        broadcasts the per-component location and scale vectors into
        the component vector the constructor takes.

        The component axis lives inside the mixture, so the plate axes
        replicate the whole mixture rather than indexing its
        parameters, which is exactly `filldist`. A residual event axis
        would ask for a mixture-valued event shape, which
        `MixtureModel` over univariate components cannot carry, so it
        raises instead of emitting a differently-shaped draw.
        """
        sb, counter = ctx.sb, ctx.counter
        if plate.event_dims:
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    f"family:MixtureNormal:event-axis:{name}: a "
                    f"`MixtureModel` over univariate components carries "
                    f"no event shape, so the residual "
                    f"{[d.name for d in plate.event_dims]!r} axis has no "
                    f"Turing.jl spelling"
                ],
            )
        weights, loc, scale = mixture_normal_components("turing", args, arg_names)
        components = _broadcast_call(
            sb,
            counter,
            _identifier(sb, counter, "Normal"),
            (_arg_to_julia(ctx, loc), _arg_to_julia(ctx, scale)),
        )
        dist = _call(
            sb,
            counter,
            _identifier(sb, counter, target_dist),
            (components, _arg_to_julia(ctx, weights)),
        )
        if plate.batch_dims:
            size_vids = tuple(
                _dim_to_size(sb, counter, dim) for dim in plate.batch_dims
            )
            dist = _call(
                sb,
                counter,
                _identifier(sb, counter, "filldist"),
                (dist, *size_vids),
            )
        stmt = _tilde(sb, counter, _identifier(sb, counter, name), dist)
        sb.edge(ctx.body, stmt, "child_of")
        return ""

    def _emit_row_plate(
        self,
        ctx: _TuringCtx,
        *,
        name: str,
        target_dist: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        family: str,
        meta: FamilyMeta,
        plate: Plate,
        own_event: tuple[Dim, ...],
        residual_event: tuple[Dim, ...],
        index_dep: bool,
    ) -> SchemaFragment:
        """Emit a batch-major plated draw from a family with its own
        event shape:

        ```julia
        emission_rows = Array{Float64, 2}(undef, 8, 16)
        for m_State in 1 : 8
          emission_rows[m_State, :] ~ Dirichlet(fill(1, 16))
        end
        ```

        One loop per batch axis and per residual event axis, in the
        declared order, with one trailing `:` per axis the family
        produces itself. Arguments that index a batch-shaped name pick
        up the innermost loop variable, so a per-row concentration
        reads `alpha[m_State]`.
        """
        sb, counter = ctx.sb, ctx.counter
        loop_dims = (*plate.batch_dims, *residual_event)
        loop_vars = _distinct_loop_vars(loop_dims)
        rewritten = args
        if index_dep:
            rewritten = tuple(
                _replace_first_index(a, loop_vars[-1], ctx.batch_shaped_names)
                for a in args
            )
        promoted = tuple(
            _promote_scalar_ref(a, arg_name, plate, meta, ctx.shapes.name_array_rank)
            for arg_name, a in zip(arg_names, rewritten, strict=False)
        )
        dist = self._family_call(ctx, target_dist, promoted, family, own_event)
        lhs = _index_expr(
            sb,
            counter,
            _identifier(sb, counter, name),
            (
                *(_identifier(sb, counter, v) for v in loop_vars),
                *(_colon(sb, counter) for _ in own_event),
            ),
        )
        stmts: tuple[str, ...] = (_tilde(sb, counter, lhs, dist),)
        for dim, loop_var in zip(reversed(loop_dims), reversed(loop_vars), strict=True):
            stmts = (
                _for_stmt(
                    sb,
                    counter,
                    var=loop_var,
                    lo=_integer(sb, counter, 1),
                    hi=_dim_to_size(sb, counter, dim),
                    body_stmts=stmts,
                ),
            )
        alloc = _array_alloc(
            sb,
            counter,
            elem_type="Float64",
            sizes=tuple(
                _dim_to_size(sb, counter, dim) for dim in (*loop_dims, *own_event)
            ),
        )
        sb.edge(
            ctx.body,
            _assignment(sb, counter, _identifier(sb, counter, name), alloc),
            "child_of",
        )
        sb.edge(ctx.body, stmts[0], "child_of")
        return ""

    # ----- marginalize: the integrated-density lowering -----

    def marginalize(self, ctx: RenderContext, node: IRMarginalize) -> SchemaFragment:
        """Integrate an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        latent out, adding the reduced density to the model's
        log-joint with `Turing.@addlogprob!`.

        One scored copy of the scope per atom of the latent's finite
        support, then an elementwise reduction across atoms:

        ```julia
        __marg_state_0 = logpdf.(Categorical(eachrow(emission_rows)[1]), obs)
        ...
        __marg_state_w = log.(initial_row)
        __marg_state_t_0 = __marg_state_w[1] .+ __marg_state_0
        ...
        __marg_state_m = max.(__marg_state_t_0, ..., __marg_state_t_7)
        __marg_state = __marg_state_m .+ log.(
          exp.(__marg_state_t_0 .- __marg_state_m) + ...
        )
        @addlogprob! sum(__marg_state)
        ```

        The max-shifted form is the numerically stable `logsumexp`
        written in `Base`, so the emitted model needs no import beyond
        what `using Turing` already brings into scope. No site is
        declared for the latent: the atoms replace it, and the emitted
        program denotes the same measure the QVR reference integrates.

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
        refuse_ungrouped_row_marginalize(f"qvr-{self.target}", node)
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        plates = {**ctx.input_plates, **ctx.sample_plates}
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        prefix = f"__marg_{node.latent}"
        term_names: list[str] = []
        for position, atom in enumerate(atoms):
            scored = marginalize_body(
                rebase_index_literals(atom.scope),
                latent=node.latent,
                target=self.target,
            )
            for det in scored.deterministics:
                self._emit_deterministic(ctx, det)
            term = f"{prefix}_{position}"
            self._emit_assignment(
                ctx, term, self._atom_log_density(ctx, scored.observe)
            )
            term_names.append(term)
        # A grouped block keys its accumulator by group: the rows the
        # fibration sends to one group are summed before the reduction.
        fibration = marginalize_fibration(node, raw.observe)
        if fibration is not None:
            via, group = fibration
            for term in term_names:
                self._emit_assignment(
                    ctx, term, self._group_sums(ctx, term, via, group)
                )
        weight_names = self._emit_atom_weights(ctx, node, raw, atoms, prefix, plates)
        shifted: list[str] = []
        for position, (weight, term) in enumerate(
            zip(weight_names, term_names, strict=True)
        ):
            name = f"{prefix}_t_{position}"
            self._emit_assignment(
                ctx,
                name,
                _dotted_binary(
                    sb,
                    counter,
                    _identifier(sb, counter, weight),
                    "+",
                    _identifier(sb, counter, term),
                ),
            )
            shifted.append(name)
        max_name = f"{prefix}_m"
        self._emit_assignment(
            ctx,
            max_name,
            _broadcast_call(
                sb,
                counter,
                _identifier(sb, counter, "max"),
                tuple(_identifier(sb, counter, n) for n in shifted),
            ),
        )
        exp_terms = tuple(
            _broadcast_call(
                sb,
                counter,
                _identifier(sb, counter, "exp"),
                (
                    _dotted_binary(
                        sb,
                        counter,
                        _identifier(sb, counter, n),
                        "-",
                        _identifier(sb, counter, max_name),
                    ),
                ),
            )
            for n in shifted
        )
        total = exp_terms[0]
        for extra in exp_terms[1:]:
            total = _dotted_binary(sb, counter, total, "+", extra)
        reduced = _dotted_binary(
            sb,
            counter,
            _identifier(sb, counter, max_name),
            "+",
            _broadcast_call(sb, counter, _identifier(sb, counter, "log"), (total,)),
        )
        self._emit_assignment(ctx, prefix, reduced)
        summed = _call(
            sb,
            counter,
            _identifier(sb, counter, "sum"),
            (_identifier(sb, counter, prefix),),
        )
        self._add_log_weight(ctx, summed, prefix)
        return ""

    def _group_sums(self, ctx: _TuringCtx, term: str, via: str, group: Dim) -> str:
        """`[sum(<term>[<via> .+ 1 .== g]) for g in 1:<extent>]`.

        The per-row log-likelihoods of one atom summed within each
        group the fibration sends the rows to, one entry per group.
        The fibration holds QVR's zero-based group codes, read one
        higher to match the one-based group counter.

        Parameters
        ----------
        ctx : _TuringCtx
            The render context.
        term : str
            The name of the per-row vector.
        via : str
            The name of the fibration sending rows to groups.
        group : Dim
            The grouping plate's leading axis.

        Returns
        -------
        str
            The comprehension's vertex id.
        """
        sb, counter = ctx.sb, ctx.counter
        binder = "g"
        selected = _index_expr(
            sb,
            counter,
            _identifier(sb, counter, term),
            (
                _dotted_binary(
                    sb,
                    counter,
                    _one_higher(ctx, _identifier(sb, counter, via), via),
                    "==",
                    _identifier(sb, counter, binder),
                ),
            ),
        )
        body = _call(sb, counter, _identifier(sb, counter, "sum"), (selected,))
        return _comprehension(
            sb,
            counter,
            body,
            binder,
            _range(
                sb, counter, _integer(sb, counter, 1), _dim_to_size(sb, counter, group)
            ),
        )

    def _add_log_weight(self, ctx: _TuringCtx, weight: str, name: str) -> None:
        """Add a reduced log weight to the model's joint.

        Turing's own statement is `@addlogprob!`; a renderer that
        traces its weights as factor choices replaces this.

        Parameters
        ----------
        ctx : _TuringCtx
            The render context.
        weight : str
            The expression vertex of the weight.
        name : str
            The name the weight was bound under.
        """
        del name
        ctx.sb.edge(
            ctx.body,
            _macro_call(ctx.sb, ctx.counter, "addlogprob!", weight),
            "child_of",
        )

    def _emit_assignment(self, ctx: _TuringCtx, name: str, rhs: str) -> None:
        """Append `<name> = <rhs>` to the model body."""
        sb, counter = ctx.sb, ctx.counter
        sb.edge(
            ctx.body,
            _assignment(sb, counter, _identifier(sb, counter, name), rhs),
            "child_of",
        )

    def _atom_log_density(self, ctx: _TuringCtx, observe: IRObserve) -> str:
        """`logpdf.(<Dist>, <observed value>)` for one atom's scope.

        The distribution constructor broadcasts when its arguments are
        per-row vectors (`Poisson.(gated_rate)`) and stays scalar when
        a single event-shaped argument feeds it
        (`Categorical(eachrow(emission_rows)[1])`). Distributions.jl
        indexes a `Categorical` from 1, so an observed class index
        arrives from the point payload one short and is lifted here.
        """
        sb, counter = ctx.sb, ctx.counter
        meta = _meta_or_raise(observe.family)
        target_dist = _turing_parameterized_name(
            observe.family, observe.arg_names, meta
        )
        if target_dist is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:{observe.family}: no Turing.jl mapping"],
            )
        if len(observe.plate.event_dims) > meta.event_rank:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"marginalize:observe-event-axis:{observe.name}: the "
                    f"scored site declares a residual event axis the "
                    f"per-atom `logpdf.` broadcast cannot replicate"
                ],
            )
        elementwise = _args_have_batch_index(
            observe.args,
            ctx.sample_plates,
            observe.plate,
            ctx.batch_shaped_names,
        )
        rhs_args = tuple(
            _arg_to_julia(ctx, a, via=observe.via, family=observe.family)
            for a in observe.args
        )
        dist = _dist_expr(
            sb,
            counter,
            target_dist,
            rhs_args,
            observe.family,
            dotted=elementwise,
            event_dims=observe.plate.event_dims,
        )
        value = _identifier(sb, counter, observe.name)
        if observe.family in _ONE_BASED_SUPPORT_FAMILIES:
            value = _dotted_binary(sb, counter, value, "+", _integer(sb, counter, 1))
        return _broadcast_call(
            sb,
            counter,
            _identifier(sb, counter, self.log_density_function),
            (dist, value),
        )

    def _emit_atom_weights(
        self,
        ctx: _TuringCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
        prefix: str,
        plates: dict[str, Plate],
    ) -> tuple[str, ...]:
        """Bind one log-weight name per atom and return the names.

        A `Bernoulli` atom set weights the atoms 0 and 1 by
        `log1p(-p)` and `log(p)`, both shaped like the probability
        itself. A `Categorical` atom set reads the class axis of the
        probability tensor, so the weights are slices of a single
        `log.(probs)` binding. A `Bernoulli` atom set given by its
        logits ``l`` weights the atoms by ``log sigmoid(-l)`` and
        ``log sigmoid(l)``, written in log space as
        ``min.(x, 0) .- log1p.(exp.(-abs.(x)))``, which neither
        overflows nor rounds where the probability would.
        """
        sb, counter = ctx.sb, ctx.counter
        probs = marginal_weight_probs(
            node, atoms[0].weight_args, atoms[0].weight_arg_names, target=self.target
        )
        family = atoms[0].weight_family
        if logit_weighted(atoms[0]):
            zero = f"{prefix}_w_0"
            one = f"{prefix}_w_1"
            self._emit_assignment(
                ctx,
                zero,
                _log_sigmoid(
                    ctx,
                    lambda: _unary_minus(sb, counter, _arg_to_julia(ctx, probs)),
                ),
            )
            self._emit_assignment(
                ctx, one, _log_sigmoid(ctx, lambda: _arg_to_julia(ctx, probs))
            )
            return (zero, one)
        if family == "Bernoulli":
            zero = f"{prefix}_w_0"
            one = f"{prefix}_w_1"
            self._emit_assignment(
                ctx,
                zero,
                _broadcast_call(
                    sb,
                    counter,
                    _identifier(sb, counter, "log1p"),
                    (_unary_minus(sb, counter, _arg_to_julia(ctx, probs)),),
                ),
            )
            self._emit_assignment(
                ctx,
                one,
                _broadcast_call(
                    sb,
                    counter,
                    _identifier(sb, counter, "log"),
                    (_arg_to_julia(ctx, probs),),
                ),
            )
            return (zero, one)
        if family != "Categorical":
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"marginalize:weight-family:{family}: no Turing.jl "
                    f"log-weight form for this atom set"
                ],
            )
        leading = _class_axis_slice(probs, plates, self.target)
        log_probs = f"{prefix}_w"
        self._emit_assignment(
            ctx,
            log_probs,
            _broadcast_call(
                sb,
                counter,
                _identifier(sb, counter, "log"),
                (_arg_to_julia(ctx, probs),),
            ),
        )
        names: list[str] = []
        for position in range(len(atoms)):
            name = f"{prefix}_w_{position}"
            index = _integer(sb, counter, position + 1)
            if leading is None:
                slice_vid = _broadcast_call(
                    sb,
                    counter,
                    _identifier(sb, counter, "getindex"),
                    (_identifier(sb, counter, log_probs), index),
                )
            else:
                slice_vid = _index_expr(
                    sb,
                    counter,
                    _identifier(sb, counter, log_probs),
                    (
                        *(_colon(sb, counter) for _ in range(leading)),
                        index,
                    ),
                )
            self._emit_assignment(ctx, name, slice_vid)
            names.append(name)
        return tuple(names)

    # ----- broadcast: Julia's fill(<value>, K) / fill(<value>, R, C) -----

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
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        value_vid = _arg_to_julia(ctx, value)
        shape_args = tuple(_integer(sb, counter, s) for s in target_shape)
        return _call(
            sb,
            counter,
            _identifier(sb, counter, "fill"),
            (value_vid, *shape_args),
        )

    # ----- helpers: family call construction -----

    def _family_call(
        self,
        ctx: _TuringCtx,
        target_dist: str,
        args: tuple[IRArg, ...],
        family: str,
        event_dims: tuple[Dim, ...] = (),
    ) -> str:
        """Render `args` and build the family's scalar call expression.

        The composition itself lives in
        `_dist_expr`,
        which every emission path shares; this wrapper only supplies
        the rendered arguments and the site's own event axes (the
        `LKJCholesky` matrix dimension).
        """
        rhs_args = tuple(_arg_to_julia(ctx, a, family=family) for a in args)
        return _dist_expr(
            ctx.sb,
            ctx.counter,
            target_dist,
            rhs_args,
            family,
            event_dims=event_dims,
        )

    def _arraydist_call(
        self,
        ctx: _TuringCtx,
        target_dist: str,
        args: tuple[IRArg, ...],
        batch_dim: Dim,
        family: str,
    ) -> str:
        """Build `arraydist([<TargetDist>(<args[i]>) for i in 1:B])`."""
        sb, counter = ctx.sb, ctx.counter
        binder = _ARRAYDIST_BINDER
        # Rewrite arg references so a `name[k]` index becomes
        # `name[<binder>]` (when the existing index is itself a
        # reference into the batch axis).
        rewritten = tuple(
            _replace_first_index(a, binder, ctx.batch_shaped_names) for a in args
        )
        body_call = _dist_expr(
            sb,
            counter,
            target_dist,
            tuple(_arg_to_julia(ctx, a, family=family) for a in rewritten),
            family,
        )
        size_vid = _dim_to_size(sb, counter, batch_dim)
        rng = _range(sb, counter, _integer(sb, counter, 1), size_vid)
        comp = _comprehension(sb, counter, body_call, binder, rng)
        return _call(
            sb,
            counter,
            _identifier(sb, counter, "arraydist"),
            (comp,),
        )

    # ----- deterministic / score / return -----

    def _emit_deterministic(self, ctx: _TuringCtx, node: IRDeterministic) -> None:
        """Emit one `<name> = <rhs>` or `<name> = @. <rhs>` assignment.

        Wraps the RHS in Julia's `@.` macro when the deterministic is
        batch-shaped (its `name` was pre-seeded into
        [`ctx.batch_shaped_names`][_TuringCtx]); this turns every
        arithmetic operator and function call in the body into its
        broadcasting variant so a `mu = a + b * x_design` against a
        vector `x_design` produces a Vector{Float64} mu rather than
        raising "no method matching +(::Int64, ::Vector)".

        A body that reduces an event axis (`mu = sum(z_row * w_row)`)
        takes the explicitly dotted form instead. `@.` would broadcast
        the reducing call itself, applying `sum` to each scalar
        element and leaving the inner product uncomputed.
        """
        refuse_nested_slice_gather("turing", node, ctx.shapes)
        sb, counter = ctx.sb, ctx.counter
        lhs = _identifier(sb, counter, node.name)
        shim = _JlCtxShim(sb, counter, ctx.cards, "turing")
        shapes = ctx.shapes.scoped_to(len(node.plate.batch_dims))
        reduces_axis = let_expr_has_axis_reduction(shapes, node.expr)
        # A binding of one event, a vector `z = tanh(a)` drawn over an
        # `over` axis, is no batch the `@.` form below covers; its
        # elementwise calls are dotted, as Julia applies `tanh` to a
        # number and not to a vector.
        one_event = bool(node.plate.event_dims) and not node.plate.batch_dims
        rhs = render_let_expr_julia(
            shim, node.expr, shapes=shapes, dotted=reduces_axis or one_event
        )
        missing = _plate_axes_missing(shapes, node)
        if node.name in ctx.batch_shaped_names and not reduces_axis and not missing:
            rhs = _macro_call(sb, counter, ".", rhs)
        if missing:
            rhs = _fan_out_to_plate(ctx, node, rhs, missing)
        stmt = _assignment(sb, counter, lhs, rhs)
        sb.edge(ctx.body, stmt, "child_of")

    def _emit_score(self, ctx: RenderContext, node: IRScore) -> None:
        """Bind the score expression then add it to the log-joint via
        `Turing.@addlogprob!`.

        The expression renders elementwise, so a weight it combines with
        a plated value scales each row before a reduction sums them."""
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        lhs = _identifier(sb, counter, node.name)
        rhs = render_let_expr_julia(
            _JlCtxShim(sb, counter, ctx.cards, "turing"),
            node.expr,
            shapes=ctx.shapes,
            dotted=True,
        )
        stmt = _assignment(sb, counter, lhs, rhs)
        sb.edge(ctx.body, stmt, "child_of")
        mac = _macro_call(
            sb, counter, "addlogprob!", _identifier(sb, counter, node.name)
        )
        sb.edge(ctx.body, mac, "child_of")

    def _emit_return(self, ctx: RenderContext, names: tuple[str, ...]) -> None:
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        if not names:
            return
        if len(names) == 1:
            value = _identifier(sb, counter, names[0])
        else:
            value = _vertex(sb, counter, "tuple_expression")
            for n in names:
                sb.edge(value, _identifier(sb, counter, n), "child_of")
        ret = _return(sb, counter, value)
        sb.edge(ctx.body, ret, "child_of")


# Comprehension binder reserved for `arraydist([... for i in 1:B])`.
_ARRAYDIST_BINDER = "i"


# Recipe table: QVR families with no native Turing.jl distribution
# whose canonical Turing.jl encoding is `truncated(<base>(0, scale),
# 0, Inf)`. The mapped value is the inner base-distribution name.
# Keyed on the QVR family name; consulted from `_dist_expr`.
_HALF_TRUNCATED_BASES: dict[str, str] = {
    "HalfNormal": "Normal",
    "HalfCauchy": "Cauchy",
}


# Recipe table: QVR families whose `rate` arg corresponds to a
# `scale = 1/rate` argument slot in Distributions.jl. The mapped
# value is the 0-based positional index of the rate arg in the QVR
# call (per the family's arg-name tuple).
#
# * Exponential(rate) -> Exponential(1/rate), rate at position 0.
# * Gamma(concentration, rate) -> Gamma(concentration, 1/rate),
#   rate at position 1.
_RATE_TO_SCALE_INVERT_POSITIONS: dict[str, int] = {
    "Exponential": 0,
    "Gamma": 1,
}


# Recipe table: QVR families whose probability parameter is the
# complement of the target distribution's. torch `NegativeBinomial(r, p)`
# has pmf proportional to `(1 - p)^r p^k` (mean `r p / (1 - p)`);
# Distributions.jl `NegativeBinomial(r, p)` has pmf proportional to
# `p^r (1 - p)^k` (mean `r (1 - p) / p`), so the probs slot must carry
# `1 - p`. The mapped value is the 0-based positional index of the
# probs arg in the QVR call.
_PROB_COMPLEMENT_POSITIONS: dict[str, int] = {
    "NegativeBinomial": 1,
}


# Recipe table: QVR families whose Distributions.jl counterpart takes
# the same parameters in a different positional order. The mapped
# value is the permutation to apply to the rendered args, given as the
# 0-based QVR positions in target order.
#
# torch `Weibull(scale, concentration)` has density
# `(k/s) (x/s)^(k-1) exp(-(x/s)^k)` with `k = concentration`;
# Distributions.jl `Weibull(alpha, theta)` has the same density with
# `alpha` the shape and `theta` the scale, so the target call is
# shape-first: `Weibull(concentration, scale)`.
_ARG_ORDER_PERMUTATIONS: dict[str, tuple[int, ...]] = {
    "Weibull": (1, 0),
}


def _binary_expr(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    left: str,
    op: str,
    right: str,
) -> str:
    """`<left> <op> <right>` as a Julia `binary_expression`."""
    vid = _vertex(sb, counter, "binary_expression")
    sb.edge(vid, left, "child_of")
    sb.edge(vid, _operator(sb, counter, op), "child_of")
    sb.edge(vid, right, "child_of")
    return vid


def _complement_prob_arg(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    rhs_args: tuple[str, ...],
    position: int,
) -> tuple[str, ...]:
    """Replace `rhs_args[position]` with `1 .- <rhs_args[position]>`.

    The dotted subtraction broadcasts over both scalar and vector
    operands, so the same complement is correct whether the probs
    slot is a scalar (`NegativeBinomial(r, 1 .- 0.3)`) or a per-
    element vector inside a broadcast call
    (`NegativeBinomial.(r, 1 .- probs)`).
    """
    if position >= len(rhs_args):
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"prob-arg-complement: position {position} out of range "
                f"for {len(rhs_args)} args"
            ],
        )
    one = _integer(sb, counter, 1)
    comp = _binary_expr(sb, counter, one, ".-", rhs_args[position])
    return rhs_args[:position] + (comp,) + rhs_args[position + 1 :]


def _transform_rhs_args(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    rhs_args: tuple[str, ...],
    family: str,
    *,
    dotted: bool = False,
) -> tuple[str, ...]:
    """Apply the family's value-level arg transforms to rendered args.

    Covers the rate->scale reciprocation (Gamma / Exponential), the
    probs complement (NegativeBinomial) and the argument reordering
    (Weibull). Structural rewrites that change the call shape
    (StudentT affine, LKJCholesky dimension prepend, half-truncated
    wrapping) sit beside these in
    `_dist_expr`,
    which is the sole caller: these arg transforms preserve the
    rendered arg strings themselves and are safe to apply on every
    path that renders a family's args, whether the call is scalar,
    broadcast (`Family.(args)`) or inside an `arraydist`
    comprehension.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    counter : list[int]
        The fresh-identifier counter.
    rhs_args : tuple[str, ...]
        The rendered family arguments.
    family : str
        The QVR family name.
    dotted : bool
        Whether the family call is broadcast over rows, so every
        value-level transform must broadcast too.

    Returns
    -------
    tuple[str, ...]
        The transformed arguments.

    Raises
    ------
    UnsupportedConstruct
        If a transform's argument position is outside `rhs_args`.
    """
    if family in _RATE_TO_SCALE_INVERT_POSITIONS:
        rhs_args = _invert_rate_arg(
            sb,
            counter,
            rhs_args,
            _RATE_TO_SCALE_INVERT_POSITIONS[family],
            dotted=dotted,
        )
    if family in _PROB_COMPLEMENT_POSITIONS:
        rhs_args = _complement_prob_arg(
            sb, counter, rhs_args, _PROB_COMPLEMENT_POSITIONS[family]
        )
    if family in _ARG_ORDER_PERMUTATIONS:
        rhs_args = _permute_args(rhs_args, _ARG_ORDER_PERMUTATIONS[family], family)
    return rhs_args


def _dist_expr(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    target_dist: str,
    rhs_args: tuple[str, ...],
    family: str,
    *,
    dotted: bool = False,
    event_dims: tuple[Dim, ...] = (),
) -> str:
    """Build a Distributions.jl expression for one family call.

    Most families map directly to a constructor. Structural rewrites handle:

    - `HalfNormal(sigma)` as ``truncated(Normal(0, sigma), 0, Inf)``;
    - `HalfCauchy(gamma)` as ``truncated(Cauchy(0, gamma), 0, Inf)``;
    - `HalfStudentT(df, scale)` as
      ``truncated(scale * TDist(df), 0, Inf)``;
    - `TruncatedNormal` as a truncated `Normal`;
    - location-scale `StudentT` as ``loc + scale * TDist(df)``; and
    - `LKJCholesky` with its event dimension as a leading argument.

    `_transform_rhs_args` applies value-level argument rewrites. The `dotted`
    flag broadcasts every call and operator produced by these transformations.
    """
    call = _broadcast_call if dotted else _call
    binary = _dotted_binary if dotted else _binary_expr
    if family == "HalfStudentT":
        # `target_dist` is bypassed here: the family resolves through
        # `FAMILY_META` to a one-name spelling, and the fold needs the
        # two-name `truncated(<affine TDist>, 0, Inf)` composition
        # instead. `TDist(df)` is the unit-scale symmetric base, so
        # `scale * TDist(df)` is the symmetric `StudentT(df, 0, scale)`
        # and truncating it at zero folds it.
        if len(rhs_args) != 2:
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    f"family:HalfStudentT: expected 2 args "
                    f"(df, scale), got {len(rhs_args)}"
                ],
            )
        df, scale = rhs_args
        tdist = call(sb, counter, _identifier(sb, counter, "TDist"), (df,))
        base_call = binary(sb, counter, scale, "*", tdist)
        return call(
            sb,
            counter,
            _identifier(sb, counter, "truncated"),
            (
                base_call,
                _integer(sb, counter, 0),
                _identifier(sb, counter, "Inf"),
            ),
        )
    base_name = _HALF_TRUNCATED_BASES.get(family)
    if base_name is not None:
        base_call = call(
            sb,
            counter,
            _identifier(sb, counter, base_name),
            (_integer(sb, counter, 0), *rhs_args),
        )
        return call(
            sb,
            counter,
            _identifier(sb, counter, target_dist),
            (
                base_call,
                _integer(sb, counter, 0),
                _identifier(sb, counter, "Inf"),
            ),
        )
    if family == "TruncatedNormal":
        # args order: (loc, scale, low, high).
        if len(rhs_args) != 4:
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    f"family:TruncatedNormal: expected 4 args "
                    f"(loc, scale, low, high), got {len(rhs_args)}"
                ],
            )
        base_call = call(
            sb,
            counter,
            _identifier(sb, counter, "Normal"),
            (rhs_args[0], rhs_args[1]),
        )
        return call(
            sb,
            counter,
            _identifier(sb, counter, target_dist),
            (base_call, rhs_args[2], rhs_args[3]),
        )
    if family == "StudentT":
        if len(rhs_args) != 3:
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    f"family:StudentT: expected 3 args "
                    f"(df, loc, scale), got {len(rhs_args)}"
                ],
            )
        df, loc, scale = rhs_args
        tdist = call(sb, counter, _identifier(sb, counter, target_dist), (df,))
        scaled = binary(sb, counter, scale, "*", tdist)
        return binary(sb, counter, loc, "+", scaled)
    if family == "LKJCholesky":
        if not event_dims:
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    "family:LKJCholesky: missing matrix dimension "
                    "(no event axis on the sample)"
                ],
            )
        dim = _dim_to_size(sb, counter, event_dims[0])
        rhs_args = (dim, *rhs_args)
    rhs_args = _transform_rhs_args(sb, counter, rhs_args, family, dotted=dotted)
    return call(
        sb,
        counter,
        _identifier(sb, counter, target_dist),
        rhs_args,
    )


def _permute_args(
    rhs_args: tuple[str, ...],
    permutation: tuple[int, ...],
    family: str,
) -> tuple[str, ...]:
    """Reorder the leading `len(permutation)` args of `rhs_args`.

    `permutation[j]` is the 0-based QVR position whose rendered arg
    belongs in target position `j`. Any args beyond the permuted
    prefix keep their relative order.
    """
    if len(rhs_args) < len(permutation):
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"family:{family}: expected at least "
                f"{len(permutation)} args for the Distributions.jl "
                f"argument order, got {len(rhs_args)}"
            ],
        )
    return tuple(rhs_args[p] for p in permutation) + rhs_args[len(permutation) :]


def _invert_rate_arg(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    rhs_args: tuple[str, ...],
    position: int,
    *,
    dotted: bool = False,
) -> tuple[str, ...]:
    """Replace `rhs_args[position]` with its reciprocal.

    Julia's `Base.inv` returns the multiplicative inverse for any
    numeric type; emitting `inv(rate)` rather than the literal
    `1/rate` avoids encoding a `binary_expression` whose right
    child kind would need to be inferred from the arg's emitter
    output. The reciprocal is the canonical Distributions.jl
    `theta = 1/rate` substitution for the `Gamma` and `Exponential`
    families. Inside a broadcast family call (`Gamma.(a, inv.(rate))`)
    the rate may be one value per row, and `inv` of a vector is a
    `MethodError`, so the reciprocal is broadcast as well.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    counter : list[int]
        The fresh-identifier counter.
    rhs_args : tuple[str, ...]
        The rendered family arguments.
    position : int
        The 0-based position of the rate argument.
    dotted : bool
        Whether the family call is broadcast over rows, so the
        reciprocal is the dot-broadcast `inv.(rate)`.

    Returns
    -------
    tuple[str, ...]
        The arguments with the rate replaced by its reciprocal.

    Raises
    ------
    UnsupportedConstruct
        If `position` is outside `rhs_args`.
    """
    if position >= len(rhs_args):
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"rate-arg-invert: position {position} out of range for "
                f"{len(rhs_args)} args"
            ],
        )
    call = _broadcast_call if dotted else _call
    inv_call = call(
        sb,
        counter,
        _identifier(sb, counter, "inv"),
        (rhs_args[position],),
    )
    return rhs_args[:position] + (inv_call,) + rhs_args[position + 1 :]


# ---------------------------------------------------------------------------
# Renderer context: extends RenderContext with Turing-specific carriers.
# ---------------------------------------------------------------------------


class _TuringCtx(RenderContext):
    """Turing-renderer-internal context. Adds the function-body block
    vid and the plate tables a `~` emission needs to detect index
    dependence between args and surrounding batch axes.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    morphisms : dict
        The resolved morphism table.
    lets : dict
        The resolved define table.
    counter : list[int]
        The fresh-identifier counter.
    cards : dict[str, int]
        Object cardinalities.
    body : str
        The function-body block vertex.
    input_plates : dict[str, Plate]
        Each input's plate.
    sample_plates : dict[str, Plate]
        Each sample site's plate.
    batch_shaped_names : set[str]
        The names carrying a batch axis at runtime.
    name_plates : dict[str, Plate]
        Every bound name's plate, inputs, sites, and lets alike.
    shapes : JuliaShapes
        The array-shape environment let expressions render under.
    module : IRQiecModule
        The checked module the plan's calls read.
    """

    def __init__(
        self,
        *,
        sb: panproto.SchemaBuilder,
        morphisms: dict,
        lets: dict,
        counter: list[int],
        cards: dict[str, int],
        body: str,
        input_plates: dict[str, Plate],
        sample_plates: dict[str, Plate],
        batch_shaped_names: set[str],
        name_plates: dict[str, Plate],
        shapes: JuliaShapes,
        module: IRQiecModule,
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=lets)
        self.counter = counter
        self.cards = cards
        self.body = body
        self.input_plates = input_plates
        self.sample_plates = sample_plates
        self.batch_shaped_names = batch_shaped_names
        self.name_plates = name_plates
        self.shapes = shapes
        self.module = module
        self.operations_bound: set[str] = set()


# `_JlCtxShim` lets us reuse
# `render_let_expr_julia`
# (which expects a `JlCtx` exposing `v`, `e`, `lit`, `fresh`,
# `constraint`, `cards`, `target`) without pulling in the legacy
# backend's whole helper module.
class _JlCtxShim:
    """Minimal adapter exposing the methods
    `render_let_expr_julia`
    reads off its ctx parameter.

    Carries the static-axis-size table `cards` so
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] unrolling
    can resolve binder cardinalities, and the `target` tag so error
    messages identify the backend ("turing").
    """

    target: str
    cards: dict[str, int]

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        counter: list[int],
        cards: dict[str, int],
        target: str,
    ) -> None:
        self._sb = sb
        self._counter = counter
        self.cards = cards
        self.target = target

    def fresh(self, prefix: str) -> str:
        self._counter[0] += 1
        return f"{prefix}_{self._counter[0]}"

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
# Internal helpers: arg conversion, dim sizing, index-dependence detection.
# ---------------------------------------------------------------------------


#: Families whose Distributions.jl counterpart indexes its support
#: from 1 where the QVR reference counts from 0. `Categorical` is the
#: only such family the renderer maps; every other discrete family in
#: `FAMILY_META` shares torch's origin.
_ONE_BASED_SUPPORT_FAMILIES: frozenset[str] = frozenset({"Categorical"})


def _distinct_loop_vars(dims: tuple[Dim, ...]) -> tuple[str, ...]:
    """One `m_<Axis>` loop variable per dim, disambiguated on repeat.

    A site can carry the same axis twice (`Dirichlet ... [over=State,
    iid_over=State]`), and two nested loops sharing a variable name
    would let the inner one shadow the outer, collapsing the plate.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for dim in dims:
        base = f"m_{dim.name}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count}")
    return tuple(out)


def _plate_axes_missing(shapes: JuliaShapes, node: IRDeterministic) -> int:
    """How many of a binding's plate axes its expression does not
    itself produce.

    A `let cell0 = cell_score[0, 0]` bound under a `Resp` plate is one
    number the plate replicates 200 times, so one axis is missing; a
    `mu = a + b * x` already spans its plate and none are.
    """
    if not node.plate.batch_dims:
        return 0
    plate_rank = len(node.plate.batch_dims) + len(node.plate.event_dims)
    return max(0, plate_rank - infer_array_rank(shapes, node.expr))


def _fan_out_to_plate(
    ctx: _TuringCtx,
    node: IRDeterministic,
    rhs: str,
    missing: int,
) -> str:
    """Materialise a binding at its plate's shape with `fill`.

    Julia has no implicit replication, and the site that consumes the
    binding reads it at the plate's shape, so the leading `missing`
    batch axes are filled explicitly.
    """
    sb, counter = ctx.sb, ctx.counter
    sizes = tuple(
        _dim_to_size(sb, counter, dim) for dim in node.plate.batch_dims[:missing]
    )
    return _call(sb, counter, _identifier(sb, counter, "fill"), (rhs, *sizes))


def _dotted_binary(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    left: str,
    op: str,
    right: str,
) -> str:
    """`<left> .<op> <right>`: the broadcasting form of a binary
    operator, for expressions the renderer builds outside the `@.`
    macro."""
    return _binary_expr(sb, counter, left, f".{op}", right)


def _log_sigmoid(ctx: _TuringCtx, operand: Callable[[], str]) -> str:
    """``log sigmoid(x)`` elementwise, as ``min.(x, 0) .- log1p.(exp.(-abs.(x)))``.

    The form is exact in floating point wherever ``x`` is finite: the
    exponential's argument is never positive, so it neither overflows
    nor, through ``log1p``, loses the small tail a large ``|x|`` leaves.

    Parameters
    ----------
    ctx : _TuringCtx
        The emission context.
    operand : Callable[[], str]
        Builds a fresh vertex for ``x`` each time it is called; the
        form reads ``x`` twice.

    Returns
    -------
    str
        The expression's vertex.
    """
    sb, counter = ctx.sb, ctx.counter
    floor = _broadcast_call(
        sb,
        counter,
        _identifier(sb, counter, "min"),
        (operand(), _number(sb, counter, 0.0)),
    )
    magnitude = _broadcast_call(
        sb, counter, _identifier(sb, counter, "abs"), (operand(),)
    )
    tail = _broadcast_call(
        sb,
        counter,
        _identifier(sb, counter, "log1p"),
        (
            _broadcast_call(
                sb,
                counter,
                _identifier(sb, counter, "exp"),
                (_unary_minus(sb, counter, magnitude),),
            ),
        ),
    )
    return _dotted_binary(sb, counter, floor, "-", tail)


def _unary_minus(sb: panproto.SchemaBuilder, counter: list[int], operand: str) -> str:
    """`- <operand>` as a Julia `unary_expression`."""
    vid = _vertex(sb, counter, "unary_expression")
    sb.edge(vid, _operator(sb, counter, "-"), "child_of")
    sb.edge(vid, operand, "child_of")
    return vid


def _class_axis_slice(
    probs: IRArg, plates: dict[str, Plate], target: str
) -> int | None:
    """How to slice atom ``k`` off a probability tensor's class axis.

    Returns the number of leading `:` axes for a dense array, or
    A `Categorical` atom set reads its weights off the innermost axis
    of the probability argument, and where that axis sits follows from
    the referenced name's declared plate: `initial_row` is one simplex
    sliced as `w[k]`, and a `theta` declared over a `Doc` batch is a
    dense `Doc`-by-`Topic` matrix sliced as `w[:, k]`. A `theta`
    gathered through a fibration is the same dense shape, one row per
    gathered index, because the gather slices the rows out rather than
    viewing them.
    """
    if not isinstance(probs, IRArgRef):
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                "marginalize:weight-expression: the atom weights are a "
                "computed expression, so the class axis has no static "
                "position to slice"
            ],
        )
    plate = plates.get(probs.name)
    if plate is None:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:weight-plate:{probs.name}: the atom "
                f"weights reference a name with no declared plate"
            ],
        )
    if len(plate.event_dims) != 1:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"marginalize:weight-event-rank:{probs.name}: the class "
                f"axis is the probability tensor's only event axis, "
                f"but the name declares {len(plate.event_dims)}"
            ],
        )
    if probs.indices:
        # One row per gathered index, and the class axis behind it.
        return 1
    return len(plate.batch_dims)


def _meta_or_raise(family: str) -> FamilyMeta:
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(
            "qvr-turing", [f"family:{family}: not in FAMILY_META"]
        )
    return meta


def _dim_to_size(sb: panproto.SchemaBuilder, counter: list[int], dim: Dim) -> str:
    """Render a [`Dim`][quivers.transpile.ir.Dim] as a Julia size
    expression. Static dims become integer literals; dynamic dims
    become identifiers referencing the runtime-supplied size."""
    if isinstance(dim, DimStatic):
        return _integer(sb, counter, dim.size)
    if isinstance(dim, DimDynamic):
        return _identifier(sb, counter, dim.size_name)
    raise UnsupportedConstruct("qvr-turing", [f"dim:{type(dim).__name__}"])


def _arg_to_julia(
    ctx: _TuringCtx,
    arg: IRArg,
    *,
    via: str | None = None,
    family: str | None = None,
) -> str:
    """Convert one IR arg into a Julia schema vertex.

    `via` rewrites any one-level `IRArgRef(name, indices=(IRArgRef(z),))`
    into `eachrow(name)[z[via]]` form when the indexed parent has
    event_dim > 0 (a matrix-shaped distribution result); plain
    `name[z[via]]` when it is a vector. The detection consults
    `ctx.sample_plates` for the parent's event_dims.

    `family` reserved for future per-family arg rewriting (none today).
    """
    del family
    sb, counter = ctx.sb, ctx.counter
    if isinstance(arg, IRArgNumber):
        return _number(sb, counter, arg.value)
    if isinstance(arg, IRArgRef):
        return _ref_to_julia(ctx, arg, via=via)
    if isinstance(arg, IRArgBroadcast):
        return _broadcast_arg(ctx, arg)
    if isinstance(arg, IRArgList):
        return _vector_literal(
            sb,
            counter,
            tuple(_arg_to_julia(ctx, e) for e in arg.elements),
        )
    if isinstance(arg, IRArgMatrix):
        return _matrix_literal(
            sb,
            counter,
            tuple(
                tuple(_arg_to_julia(ctx, e) for e in row.elements) for row in arg.rows
            ),
        )
    if isinstance(arg, IRArgFamilyRef):
        return _family_ref_to_julia(ctx, arg)
    raise UnsupportedConstruct("qvr-turing", [f"arg:{type(arg).__name__}"])


def _ref_to_julia(ctx: _TuringCtx, ref: IRArgRef, *, via: str | None) -> str:
    """Render an [`IRArgRef`][quivers.transpile.ir.IRArgRef].

    Specialises on the LDA-style case: `phi[z]` with a surrounding
    observe whose `via` fibration is `word_idx` and whose indexed
    parent `phi` has event_dim > 0 becomes
    `eachrow(phi)[z[word_idx]]`. The vector-parent case becomes
    `phi[z[word_idx]]` (no `eachrow`).
    """
    sb, counter = ctx.sb, ctx.counter
    base = _identifier(sb, counter, ref.name)
    if not ref.indices:
        return base
    # Each gathered element carries the parent's axes after the indexed
    # ones: none for an entry of a vector, the columns for a row of a
    # matrix, each read whole with a trailing `:`.
    carried = carried_dims(ref, {**ctx.input_plates, **ctx.sample_plates})
    # Rewrite each index: a name-ref index may be threaded through the
    # via fibration.
    rendered_indices: list[str] = []
    for idx in ref.indices:
        if via is not None and isinstance(idx, IRArgRef) and not idx.indices:
            # `z` becomes `z[<via>]`.
            inner = _identifier(sb, counter, idx.name)
            via_id = _one_higher(ctx, _identifier(sb, counter, via), via)
            rendered = _index_expr(sb, counter, inner, (via_id,))
        else:
            rendered = _arg_to_julia(ctx, idx)
        if isinstance(idx, IRArgRef):
            rendered = _one_higher(ctx, rendered, idx.name)
        rendered_indices.append(rendered)
    if carried:
        # `phi` is a matrix-shaped distribution sample; take its rows.
        #
        # `phi[k, :]` rather than `eachrow(phi)[k]`, because the two
        # differ in more than spelling: the slice copies and the view
        # does not. A `Categorical` built from a row *view* is
        # parameterised by the view's type, and Distributions.jl then
        # cannot store the normalised `Vector` it computes back into
        # it. The copy costs one row and the constructor accepts it.
        return _index_expr(
            sb,
            counter,
            base,
            (*rendered_indices, *(_colon(sb, counter) for _ in carried)),
        )
    return _index_expr(sb, counter, base, tuple(rendered_indices))


def _rows_of_table(
    ctx: _TuringCtx, rendered: str, arg: IRArg, meta: FamilyMeta, arg_name: str
) -> str:
    """Broadcast a class-code family's table argument row by row.

    Categorical's `probs` reads one vector per observation. A binding
    with two axes, one row per observation, broadcasts through the
    grafted `qvr_rows(...)`, which takes a matrix's rows and passes a
    vector of vectors through.

    Parameters
    ----------
    ctx : _TuringCtx
        The render context.
    rendered : str
        The rendered argument.
    arg : IRArg
        The plan argument.
    meta : FamilyMeta
        The family.
    arg_name : str
        The argument's name.

    Returns
    -------
    str
        The argument, wrapped in `qvr_rows` when it is such a table.
    """
    if meta.qvr_name not in _ZERO_BASED_FAMILIES:
        return rendered
    constraints = meta.distribution_class.arg_constraints
    expected = constraints.get(arg_name) if isinstance(constraints, dict) else None
    if int(getattr(expected, "event_dim", 0)) != 1:
        return rendered
    if not isinstance(arg, IRArgRef) or arg.indices:
        return rendered
    if ctx.shapes.name_array_rank.get(arg.name, 0) != 2:
        return rendered
    return _call(
        ctx.sb, ctx.counter, _identifier(ctx.sb, ctx.counter, "qvr_rows"), (rendered,)
    )


def _one_higher(ctx: _TuringCtx, index: str, name: str) -> str:
    """Read a subscript one higher when it holds a QVR value.

    A program's index covariate or drawn class holds a zero-based QVR
    value, which Julia reads one higher; a renderer's own loop variable
    already counts from one.

    Parameters
    ----------
    ctx : _TuringCtx
        The render context.
    index : str
        The rendered subscript.
    name : str
        The name the subscript reads.

    Returns
    -------
    str
        The subscript, broadcast-shifted with `.+ 1` for a program value.
    """
    if name not in ctx.shapes.program_values:
        return index
    return _binary_expr(
        ctx.sb, ctx.counter, index, ".+", _integer(ctx.sb, ctx.counter, 1)
    )


def _broadcast_arg(ctx: _TuringCtx, arg: IRArgBroadcast) -> str:
    """Render an [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast]
    as `fill(<value>, <shape...>)`."""
    sb, counter = ctx.sb, ctx.counter
    value_vid = _arg_to_julia(ctx, arg.value)
    shape_args = tuple(_integer(sb, counter, s) for s in arg.target_shape)
    return _call(
        sb,
        counter,
        _identifier(sb, counter, "fill"),
        (value_vid, *shape_args),
    )


def _family_ref_to_julia(ctx: _TuringCtx, ref: IRArgFamilyRef) -> str:
    """Render an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef].

    Looks up the referenced morphism's `~ Family(...)` init clause and
    emits the inner distribution call directly. For wrapper families
    like Truncated the caller's `_render_truncated_wrapper` builds the
    enclosing `truncated(<base>, lower=L, upper=U)` form.
    """
    sb, counter = ctx.sb, ctx.counter
    decl = ctx.morphisms.get(ref.name)
    if decl is None or decl.init_family is None:
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"arg:family-ref:{ref.name}: morphism not declared with "
                f"`~ Family(...)` init"
            ],
        )
    inner_meta = _meta_or_raise(decl.init_family.family)
    inner_target = inner_meta.target_names.get("turing")
    if inner_target is None:
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"family:{decl.init_family.family}: no Turing.jl mapping "
                f"for wrapper-referenced inner family"
            ],
        )
    callee = _identifier(sb, counter, inner_target)
    inner_args = tuple(
        _raw_init_arg_to_julia(ctx, value)
        for _, value in bind_family_arguments(
            decl.init_family.family, decl.init_family.args
        )
    )
    return _call(sb, counter, callee, inner_args)


def _raw_init_arg_to_julia(ctx: _TuringCtx, raw: object) -> str:
    """Render a parser-form `init_family` arg. The parser uses bare
    `str` for identifiers and `float` for literals; the wrapper
    `IRArgFamilyRef` rendering pre-dates the structured DrawArg
    encoding for nested family calls."""
    sb, counter = ctx.sb, ctx.counter
    if not isinstance(raw, (int, float, str)):
        raw = atom_value(raw)
    if isinstance(raw, (int, float)):
        return _number(sb, counter, float(raw))
    if isinstance(raw, str):
        try:
            return _number(sb, counter, float(raw))
        except ValueError:
            return _identifier(sb, counter, raw)
    raise UnsupportedConstruct(
        "qvr-turing", [f"arg:family-ref-init:{type(raw).__name__}"]
    )


def _args_have_batch_index(
    args: tuple[IRArg, ...],
    sample_plates: dict[str, Plate],
    plate: Plate,
    batch_shaped_names: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Detect whether any arg references a name whose plate has a
    nonempty batch axis (i.e. an index-dependent call site for the
    surrounding step's plate).

    Used by [`sample`][quivers.transpile.renderers.TuringRenderer.sample] to choose between the
    `filldist(...)` form (no dependence) and the `arraydist(...)` /
    broadcast-dot form (dependence). The `batch_shaped_names` set
    additionally flags direct (un-indexed) references to a
    batch-shaped IRDataInput or to a let-bound deterministic whose
    RHS transitively references one, so a vector-valued `loc` like
    `mu = a + b * x_design` in bayes_linear_regression is recognised
    even though its IRDeterministic carries an empty plate.
    """
    del plate
    return any(_arg_indexes_plated(a, sample_plates, batch_shaped_names) for a in args)


def _arg_indexes_plated(
    arg: IRArg,
    sample_plates: dict[str, Plate],
    batch_shaped_names: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Recursive helper: True iff `arg` carries a bracket index whose
    inner reference is itself a previously-bound sample with a
    nonempty plate, OR `arg` (or a sub-arg) references a name in
    `batch_shaped_names` directly (un-indexed)."""
    if isinstance(arg, IRArgRef):
        if not arg.indices and arg.name in batch_shaped_names:
            return True
        if arg.indices:
            for idx in arg.indices:
                if isinstance(idx, IRArgRef) and idx.name in sample_plates:
                    return True
        for idx in arg.indices:
            if _arg_indexes_plated(idx, sample_plates, batch_shaped_names):
                return True
        return False
    if isinstance(arg, IRArgBroadcast):
        return _arg_indexes_plated(arg.value, sample_plates, batch_shaped_names)
    if isinstance(arg, IRArgList):
        return any(
            _arg_indexes_plated(e, sample_plates, batch_shaped_names)
            for e in arg.elements
        )
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for e in row.elements:
                if _arg_indexes_plated(e, sample_plates, batch_shaped_names):
                    return True
    return False


def _arg_plate(ctx: _TuringCtx, arg: IRArg) -> Plate | None:
    """The declared plate of an unindexed reference argument.

    Parameters
    ----------
    ctx : _TuringCtx
        The render context, carrying every bound name's plate.
    arg : IRArg
        The argument.

    Returns
    -------
    Plate | None
        The plate of the name ``arg`` reads whole, else ``None``.
    """
    if not isinstance(arg, IRArgRef) or arg.indices:
        return None
    return ctx.name_plates.get(arg.name)


def _visits_parameter(
    ctx: _TuringCtx,
    family: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
    loop_dims: tuple[Dim, ...],
) -> bool:
    """Whether a parameter holds one value per position the site replicates.

    Parameters
    ----------
    ctx : _TuringCtx
        The render context, carrying every bound name's plate.
    family : str
        The family's QVR name.
    args : tuple[IRArg, ...]
        The family's arguments.
    arg_names : tuple[str, ...]
        The parameter each argument fills.
    loop_dims : tuple[Dim, ...]
        The axes the site replicates the family over.

    Returns
    -------
    bool
        True when an argument filling a rank-0 parameter reads whole a
        value whose every axis is one of ``loop_dims``.
    """
    if not loop_dims:
        return False
    loops = {str(dim.name): str(dim.name) for dim in loop_dims}
    return any(
        scalar_parameter(family, arg_name)
        and bool(visited_position(_arg_plate(ctx, arg), loops))
        for arg, arg_name in zip(args, arg_names, strict=False)
    )


def _promote_scalar_ref(
    arg: IRArg,
    arg_name: str,
    plate: Plate,
    meta: FamilyMeta,
    name_array_rank: dict[str, int],
) -> IRArg:
    """Promote an [`IRArgRef`][quivers.transpile.ir.IRArgRef] that
    targets a vector-shaped arg constraint into an
    [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast].

    Lower leaves scalar references unwrapped (a tensor reference may
    already carry the right shape at runtime), but the Turing.jl
    idiom for a Dirichlet's `concentration` and similar vector
    parameters writes `fill(alpha, K)` for the scalar-broadcast case.
    The promotion rule: if the arg has no bracket indices and the
    family's `arg_constraints[arg_name]` is an
    `IndependentConstraint` of rank `n >= 1`, wrap in `IRArgBroadcast`
    whose `target_shape` is read from the surrounding step's
    `event_dims`. A reference is promoted only when the plan binds its
    name to a value of no axes: a reference to a vector (a sampled
    `loc` or `concentration`) already carries the slot's shape.

    Parameters
    ----------
    arg : IRArg
        The argument.
    arg_name : str
        The family parameter the argument binds.
    plate : Plate
        The site's plate, whose event axes size the fill.
    meta : FamilyMeta
        The family's metadata.
    name_array_rank : dict[str, int]
        The Julia array rank of each name the plan binds.

    Returns
    -------
    IRArg
        The argument filled to the slot's shape, or the argument as it
        stands.
    """
    if not isinstance(arg, IRArgRef) or arg.indices or name_array_rank.get(arg.name, 0):
        return arg
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return arg
    expected = cls_attr.get(arg_name)
    if expected is None or not isinstance(
        expected, _torch_constraints._IndependentConstraint
    ):
        return arg
    if len(plate.event_dims) < expected.event_dim:
        return arg
    sizes: tuple[int, ...] = ()
    for dim in plate.event_dims[: expected.event_dim]:
        if isinstance(dim, DimStatic):
            sizes = (*sizes, dim.size)
        else:
            return arg
    if not sizes:
        return arg
    return IRArgBroadcast(value=arg, target_shape=sizes)


def _replace_first_index(
    arg: IRArg,
    binder: str,
    batch_shaped_names: frozenset[str] | set[str] = frozenset(),
) -> IRArg:
    """For the `arraydist([... for i in 1:B])` fallback: thread the
    comprehension binder into each per-element arg.

    An indexed reference `arg[k]` becomes `arg[<binder>]` so the
    comprehension walks the batch axis. A bare reference to a
    batch-shaped name (a let-bound vector like `h_mean`, or any name
    in `batch_shaped_names`) gains a fresh `[<binder>]` index, because
    the per-element distribution consumes one scalar entry per step
    rather than the whole vector. Numeric / literal / scalar args pass
    through unchanged."""
    if isinstance(arg, IRArgRef):
        if arg.indices:
            new_indices = tuple(
                IRArgRef(name=binder) if isinstance(idx, IRArgRef) else idx
                for idx in arg.indices
            )
            return IRArgRef(name=arg.name, indices=new_indices)
        if arg.name in batch_shaped_names:
            return IRArgRef(name=arg.name, indices=(IRArgRef(name=binder),))
    if isinstance(arg, IRArgBroadcast):
        return IRArgBroadcast(
            value=_replace_first_index(arg.value, binder, batch_shaped_names),
            target_shape=arg.target_shape,
        )
    return arg


def _seed_sample_plates(ctx: _TuringCtx, body: tuple[IRNode, ...]) -> None:
    """Populate `ctx.sample_plates` from every IRSample / IRObserve /
    IRMarginalize in `body` (descending into marginalize scopes).

    Used by [`_args_have_batch_index`][_args_have_batch_index] and
    [`_ref_to_julia`][_ref_to_julia] to look up the plate of an
    indexed reference's parent so the renderer can dispatch on the
    parent's event_dim (eachrow vs. plain index).
    """
    for node in body:
        if isinstance(node, IRSample):
            ctx.sample_plates[node.name] = node.plate
        elif isinstance(node, IRObserve):
            ctx.sample_plates[node.name] = node.plate
        elif isinstance(node, IRMarginalize):
            ctx.sample_plates[node.latent] = node.plate
            _seed_sample_plates(ctx, node.scope)


def _seed_batch_shaped(ctx: _TuringCtx, body: tuple[IRNode, ...]) -> None:
    """Populate `ctx.batch_shaped_names` with every name that carries a
    batch dimension at runtime.

    A name is batch-shaped when its plate has a batch axis: the plan
    reads each binding's plate off its checked type, so an input, a
    sample, an observation, a marginalized latent, or a let binding
    is per-element exactly when its plate says so. A scalar binding a
    plated site reads (`cell0 = cell_score[0, 0]` under a `Resp`
    observe) stays one value the site's `filldist` shares across its
    rows.

    The Turing renderer uses this set to pick the broadcast-dot form
    `y .~ Family.(...)` for observes whose plain-call form would
    fail at the boundary between a scalar Julia distribution
    constructor and a vector argument, and to wrap each
    batch-shaped deterministic's RHS in `@.` so its arithmetic
    broadcasts elementwise.

    Parameters
    ----------
    ctx : _TuringCtx
        The render context whose name table is populated.
    body : tuple[IRNode, ...]
        The plan body.
    """
    for name, plate in ctx.input_plates.items():
        if plate.batch_dims:
            ctx.batch_shaped_names.add(name)
    # Marginalize scopes recurse: the atoms the block enumerates
    # re-emit their bindings into the enclosing body, so their shapes
    # are decided by the same rule.
    _collect_sample_batch_shaped(ctx, body)
    for node in _flatten_body(body):
        if isinstance(node, IRDeterministic) and node.plate.batch_dims:
            ctx.batch_shaped_names.add(node.name)


def _flatten_body(body: tuple[IRNode, ...]) -> tuple[IRNode, ...]:
    """Flatten an IR body, splicing every marginalize scope inline.

    The marginalize lowering re-emits each scope node into the
    enclosing body once per atom, so the shape analyses that run over
    the body must see those nodes too.
    """
    out: list[IRNode] = []
    for node in body:
        out.append(node)
        if isinstance(node, IRMarginalize):
            out.extend(_flatten_body(node.scope))
    return tuple(out)


def _collect_sample_batch_shaped(ctx: _TuringCtx, body: tuple[IRNode, ...]) -> None:
    """Add every IRSample / IRObserve / IRMarginalize name with
    non-empty batch_dims to `ctx.batch_shaped_names`."""
    for node in body:
        if isinstance(node, IRSample) and node.plate.batch_dims:
            ctx.batch_shaped_names.add(node.name)
        elif isinstance(node, IRObserve) and node.plate.batch_dims:
            ctx.batch_shaped_names.add(node.name)
        elif isinstance(node, IRMarginalize):
            if node.plate.batch_dims:
                ctx.batch_shaped_names.add(node.latent)
            _collect_sample_batch_shaped(ctx, node.scope)


# ---------------------------------------------------------------------------
# Public convenience entry point: parse a Module, lower, render, pretty.
# ---------------------------------------------------------------------------


def render_module(module: Module) -> bytes:
    """Lower a parsed [`Module`][quivers.dsl.ast_nodes.Module] and emit
    Turing.jl source bytes.

    Convenience wrapper around the `Module -> IRProgram -> Schema ->
    bytes` pipeline. Tests and the CLI driver call this; the renderer
    is composable directly via [`TuringRenderer.render`][quivers.transpile.renderers.TuringRenderer.render]
    for callers that already hold an IRProgram.
    """
    # `target="stan"` keeps `MarginalizeStep` intact through composite
    # expansion; the Turing renderer's `marginalize` handler lowers it
    # to an explicit `IRSample(latent)` plus the scoped body inline.
    # Driving the lowering through the renderer (not through
    # `expand_composite_lets`) preserves the marginalize step's
    # `over=` axis spec, which the in-place flatten path drops.
    expanded = expand_composite_lets(module, target="stan")
    morphisms = build_morphism_table(expanded)
    lets = build_let_table(expanded)
    # `_pick_program` raises early when the module has no
    # ProgramDecl, surfacing the failure here rather than letting
    # `Lower` raise an opaquer error downstream.
    _pick_program(expanded)
    ir = Lower()(expanded)
    schema = TuringRenderer(morphisms=morphisms, lets=lets).render(ir)
    return bytes(EmitPretty("julia")(schema))


def _pick_program(module: Module) -> ProgramDecl:
    """Pick the program decl to render, preferring the exported one."""
    programs: list[ProgramDecl] = []
    exported: set[str] = set()
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            programs.append(stmt)
        elif isinstance(stmt, ExportDecl) and isinstance(stmt.expr, ExprIdent):
            exported.add(stmt.expr.name)
    if not programs:
        raise UnsupportedConstruct("qvr-turing", ["program:absent"])
    return next((p for p in programs if p.name in exported), programs[-1])


# ---------------------------------------------------------------------------
# Runtime-helper graft: `ContinuousBernoulli` as a
# Distributions.ContinuousUnivariateDistribution subclass, plus the RBF
# kernel builder a `GP` site needs.
#
# Distributions.jl ships `Normal`, `Beta`, `TDist`, `Kumaraswamy`, ... as
# built-in distributions but lacks `ContinuousBernoulli`.
# The transpile-time graft parses the hand-written helper at
# [`runtime_turing.jl`][quivers.transpile.runtime_turing] once at module
# load through panproto's Julia tree-sitter grammar; per-render, it
# copies every grafted vertex / constraint / edge into the per-render
# schema (with fresh vertex ids) and attaches the runtime's top-level
# statements as `child_of` of the emitted `source_file` above the
# `@model function model` macrocall.
#
# The emit is structurally a normal Julia source file: the `using`
# preamble, the `ContinuousBernoulli` struct with its
# `Distributions.logpdf` / `Distributions.rand` / support methods, and
# the `_qvr_rbf_kernel` builder. Subsequent
# `~ ContinuousBernoulli(probs)` call sites and `_qvr_rbf_kernel(...)`
# covariance builds in the model body then resolve to the grafted
# definitions via normal Julia name lookup.
# ---------------------------------------------------------------------------


_RUNTIME_TURING_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_turing.jl"
)


#: Families whose Turing.jl emit relies on the
#: [`runtime_turing.jl`][quivers.transpile.runtime_turing] helper subtree.
#: Distributions.jl ships `Normal`, `Beta`, `TDist`, `Kumaraswamy`, etc.
#: as built-in distributions but lacks these; the renderer grafts the
#: helper when the IR samples or observes from any of them.
#:
#: `HalfStudentT` is deliberately absent. Distributions.jl has no such
#: type either, but it does have every piece the fold is built from, so
#: `_dist_expr`
#: composes `truncated(scale * TDist(df), 0, Inf)` out of the library's
#: own combinators rather than paying for a grafted scorer. Composing
#: keeps the emitted file a single top-level `@model` macrocall, which
#: a graft would turn into a sequence of top-level statements.
_TURING_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset(
    {
        "Categorical",
        "ContinuousBernoulli",
        "GP",
    }
)

#: Families whose QVR values are zero-based class codes, drawn and
#: scored through the grafted zero-based form in
#: [`runtime_turing.jl`][quivers.transpile.runtime_turing] rather than
#: Distributions.jl's one-based one.
_ZERO_BASED_FAMILIES: dict[str, str] = {"Categorical": "qvr_categorical"}

#: The Turing.jl distribution that takes each alternative parameterization
#: a family call can name, keyed by family and the parameter selecting it.
_TURING_PARAMETERIZED_NAMES: dict[tuple[str, str], str] = {
    ("Bernoulli", "logits"): "BernoulliLogit",
    ("Binomial", "logits"): "BinomialLogit",
}


def _turing_parameterized_name(
    family: str, arg_names: tuple[str, ...], meta: FamilyMeta
) -> str | None:
    """The Turing.jl distribution for the parameterization a call names.

    Parameters
    ----------
    family : str
        The family.
    arg_names : tuple[str, ...]
        The parameters the call supplies.
    meta : FamilyMeta
        The family's transpile metadata.

    Returns
    -------
    str | None
        The distribution taking the named parameterization, else the
        family's Turing.jl name, or ``None`` when it has none.
    """
    for name in arg_names:
        selected = _TURING_PARAMETERIZED_NAMES.get((family, name))
        if selected is not None:
            return selected
    return meta.target_names.get("turing")


def _load_runtime_turing_schema() -> tuple[panproto.Schema, str, tuple[str, ...]]:
    """Parse [`runtime_turing.jl`][quivers.transpile.runtime_turing] through
    panproto's Julia tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `source_file` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `source_file` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "julia",
        _RUNTIME_TURING_PATH.read_bytes(),
        str(_RUNTIME_TURING_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "source_file"),
        None,
    )
    if src_id is None:
        raise RuntimeError(
            f"`source_file` not found in parse of {_RUNTIME_TURING_PATH}"
        )
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


_RUNTIME_TURING_SCHEMA, _RUNTIME_TURING_SOURCE_ID, _RUNTIME_TURING_TOP_LEVEL = (
    _load_runtime_turing_schema()
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


_RUNTIME_TURING_SUBTREE = _subtree_vertex_ids(
    _RUNTIME_TURING_SCHEMA, _RUNTIME_TURING_TOP_LEVEL
)


def _ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any [`IRSample`][quivers.transpile.ir.IRSample] or
    [`IRObserve`][quivers.transpile.ir.IRObserve] in `body` (including
    nested [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes)
    samples from `family`."""
    for node in body:
        if isinstance(node, (IRSample, IRObserve)) and node.family == family:
            return True
        if isinstance(node, IRMarginalize) and _ir_uses_family(node.scope, family):
            return True
    return False


def _graft_runtime_turing_helper(
    sb: panproto.SchemaBuilder, counter: list[int], source_vid: str
) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_turing.jl` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each
    top-level child as a `child_of` of `source_vid` in source order.
    The grafted top-level children appear above the `@model function
    model` macrocall in the emit.
    """
    src_schema = _RUNTIME_TURING_SCHEMA
    subtree = _RUNTIME_TURING_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        counter[0] += 1
        new = f"rt{counter[0]}"
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_TURING_TOP_LEVEL:
        sb.edge(source_vid, id_map[child_old], "child_of")


__all__ = ["TuringRenderer"]
