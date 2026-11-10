"""Render transpilation IR as a WebPPL model function.

Sites use ``sample`` or ``observe`` inside ``repeat`` or ``mapIndexed``
plate forms. Finite marginalizations score each atom and add their
reduced density with ``factor``. Broadcasts use nested ``repeat``
calls, and wrapper families inline their referenced distribution.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Callable

import panproto

from quivers.dsl.ast_nodes import Expr, DefineDecl, Module, MorphismDecl
from quivers.dsl.ast_nodes.let_expressions import (
    LetFactorCase,
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
from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    IRQiecValueExpr,
    LetExprAffineMap,
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
from quivers.transpile.renderers._ranks import let_expr_rank, name_rank_map
from quivers.transpile.renderers._rows import take_row
from quivers.transpile.renderers._base import (
    marginalize_row_rank,
    BlockKind,
    IRMarginalAtom,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    mixture_component_count,
    mixture_normal_components,
)
from quivers.transpile.renderers._qiec import (
    graft_javascript_statements,
    javascript_call_source,
    javascript_operations_source,
    qiec_helper_families_used,
    value_argument_source,
    render_computations_dynamic,
)
from quivers.transpile.renderers._javascript_helpers import (
    render_let_expr_javascript,
)
from quivers.transpile.renderers._python_helpers import (
    MarginalizeBody,
    marginal_support_size,
    marginal_weight_probs,
    marginalize_body,
    marginalize_fibration,
    name_event_rank_map,
)


#: The model-body name of the native operation table.
_OPERATIONS = "_qvr_qiec_operations"


class _JsLetCtx:
    """Bridge `RenderContext.sb` to the
    `render_let_expr_javascript`
    helper's protocol (`v`, `e`, `lit`, `constraint`, `fresh`).

    Carries the object-name -> static-cardinality map consulted when
    unrolling [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    and the `target` tag used in error messages.

    The WebPPL IR-walk operates over
    [`RenderContext`][quivers.transpile.renderers.RenderContext]; the
    let-expression helper expects a small carrier exposing
    `panproto.SchemaBuilder` operations
    under terse method names. The shim keeps the helper independent
    of any specific renderer class.
    """

    target: str = "webppl"

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        fresh: Callable[[str], str],
        cards: dict[str, int],
    ) -> None:
        self._sb = sb
        self._fresh_fn = fresh
        self.cards = cards

    def fresh(self, prefix: str) -> str:
        return self._fresh_fn(prefix)

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str) -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


class WebPPLRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    WebPPL `panproto.Schema`.

    Subclasses
    [`RendererBase`][quivers.transpile.renderers.RendererBase]
    and overrides the four dispatch points
    (`declare`, `sample`, `marginalize`, `broadcast`) plus the two
    list / matrix arg helpers per the spec.
    """

    target: str = "webppl"

    def __init__(self, *, source_module: Module | None = None) -> None:
        """Initialise the renderer.

        Parameters
        ----------
        source_module
            Optional original [`Module`][quivers.dsl.ast_nodes.Module]
            the IR came from. Carries `MorphismDecl` / `DefineDecl`
            entries the renderer reads when resolving
            [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
            args. When omitted, the renderer raises on
            `IRArgFamilyRef` rather than guess.
        """
        self._source_module = source_module
        # Per-render scratch state. Reset at the top of every
        # `render` call so a single renderer instance can serve
        # multiple programs back-to-back.
        self._fresh_n = 0
        # The body block id under the top-level function expression.
        self._body_vid: str = ""
        # Per-render map from a sample / observe / input name to its
        # binding plate. Powers the index-dependent / broadcast
        # decisions on later sample steps.
        self._binding_plates: dict[str, Plate] = {}
        # Per-render map from a sample / observe / input name to its
        # support constraint. Powers broadcast detection (a scalar
        # ref into a vector-arg slot needs `repeat` wrapping).
        self._binding_supports: dict[str, ConstraintSpec] = {}
        # `via` fibrations seen on the active observe scope; used by
        # ref substitution to thread the per-observe loop var through
        # references whose binding plate matches the fibration's
        # group axis.
        self._observe_via: str | None = None
        # The per-observe loop variable name, when an observe's
        # mapIndexed body is being emitted; None outside that scope.
        self._observe_loop_var: str | None = None
        # The active marginalize / group plate batch_dim names. When
        # an observe inside a marginalize scope renders, refs to
        # names whose binding plate aligns with this tuple get
        # `via[loop_var]` indexed through the fibration; refs to
        # other plates pass through unchanged.
        self._group_plate_axes: tuple[str, ...] = ()
        # Static-cardinality table snapshotted from `IRProgram.cards`
        # at the top of every `render` call. The let-expression shim
        # reads this when unrolling `LetExprFactor` binders whose
        # axis size needs to be known at render time.
        self._cards: dict[str, int] = {}
        # Counter for fresh loop-variable names generated by the
        # deterministic-binding lift (`__i_<n>`); kept independent of
        # the schema-vertex counter so the names stay short and
        # legible in the rendered source.
        self._lift_n = 0
        # Names registered as IRDataInputs at render start; used by
        # ref-substitution to discriminate function parameters from
        # in-scope sample bindings.
        self._function_parameters_state: set[str] = set()
        # Per-name event rank (`len(plate.event_dims)`) read from the
        # IR. A reducing primitive applied to a positive-rank argument
        # collapses the innermost axis through a `_qvr_<f>_last`
        # runtime helper; WebPPL's own `sum` flattens every axis.
        self._name_event_rank: dict[str, int] = {}
        self._name_rank: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Abstract overrides: target_protocol + four dispatch points.
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("javascript")

    # ----- the full render override -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a WebPPL schema.

        Override of the base `render` so the program-level shape
        (a single `var model = function(<inputs>) { ... };`
        declaration) is built once per call.

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
        morphisms, lets = self._resolve_morphisms_and_lets()
        ctx = RenderContext(sb=sb, morphisms=morphisms, defines=lets)
        # Reset per-render state.
        self._fresh_n = 0
        self._binding_plates = {}
        self._binding_supports = {}
        self._observe_via = None
        self._observe_loop_var = None
        self._lift_n = 0
        # Snapshot the IR's static-cardinality table so the let-expr
        # shim can resolve factor binders.
        self._cards = dict(ir.cards)
        # Per-name event rank, so a reduction over an event axis
        # (`sum(z_row * w_row)`) collapses only the innermost axis
        # instead of flattening the whole nested array to one scalar.
        self._name_event_rank = name_event_rank_map(ir)
        self._name_rank = name_rank_map(ir)
        # Program root: a single var-decl wrapping a function expression.
        ctx.sb.vertex("prog", "program")
        # WebPPL's `dists` module ships `Gaussian`, `Beta`, `Categorical`,
        # etc. as built-in distributions but lacks `Logistic`,
        # `BetaBinomial`, `HalfStudentT`, `Kumaraswamy`, `LKJCholesky`,
        # and `ContinuousBernoulli`. When the IR samples or observes
        # from any of these, graft the helper at
        # [`runtime_webppl.js`][quivers.transpile.runtime_webppl] onto
        # the source above the `var model = function (...) {...};`
        # declaration so the body's `sample(<Family>({...}))` call
        # sites resolve through normal JS name lookup.
        if (
            any(_ir_uses_family(ir.body, f) for f in _WEBPPL_RUNTIME_HELPER_FAMILIES)
            or _ir_emits_qvr_bcast(ir, self._name_rank)
            or _ir_uses_webppl_math(ir.body)
            or _ir_reduces_event_axis(ir.body, self._name_event_rank, self._name_rank)
            or _ir_has_marginalize(ir.body)
            or _ir_has_affine_map(ir.body)
            # A `Categorical` call site reads its support from
            # `_qvr_support`, which lives in the same helper block
            # although the family itself is scored through `.score`
            # rather than through the helper path.
            or _ir_uses_family(ir.body, "Categorical")
            or qiec_helper_families_used(ir, self.target)
        ):
            _graft_runtime_webppl_helper(ctx.sb, self, "prog")
        var_decl = self._fresh(ctx, "vd")
        ctx.sb.vertex(var_decl, "variable_declaration")
        declarator = self._fresh(ctx, "dr")
        ctx.sb.vertex(declarator, "variable_declarator")
        model_name = self._ident(ctx, "model")
        ctx.sb.edge(declarator, model_name, "name")
        fn = self._fresh(ctx, "fn")
        ctx.sb.vertex(fn, "function_expression")
        params = self._fresh(ctx, "ps")
        ctx.sb.vertex(params, "formal_parameters")
        self._function_parameters.clear()
        for inp in ir.inputs:
            ctx.sb.edge(params, self._ident(ctx, inp.name), "child_of")
            self._binding_plates[inp.name] = inp.plate
            self._binding_supports[inp.name] = inp.constraint
            self._function_parameters.add(inp.name)
        body = self._fresh(ctx, "body")
        ctx.sb.vertex(body, "statement_block")
        ctx.sb.edge(fn, params, "parameters")
        ctx.sb.edge(fn, body, "body")
        ctx.sb.edge(declarator, fn, "value")
        ctx.sb.edge(var_decl, declarator, "child_of")
        ctx.sb.edge("prog", var_decl, "child_of")
        self._body_vid = body
        self._module = ir.module
        self._operations_bound: set[str] = set()
        # Walk the body. Inputs were registered above; declare is a
        # no-op for WebPPL, so we dispatch only the body nodes.
        for node in ir.body:
            self._dispatch_node(ctx, node)
        render_computations_dynamic(ctx.sb, ir, target=self.target, root="prog")
        return ctx.sb.build()

    # ----- declare dispatch -----

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op: WebPPL's `var <name> = sample(...);` binding is
        emitted by `sample` directly. The declaration table threads
        the name's binding plate / support into the renderer's
        bookkeeping so later sample steps can index into the name
        correctly.

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
        del ctx, block
        self._binding_plates[name] = plate
        self._binding_supports[name] = constraint
        return ""

    # ----- sample / observe dispatch -----

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
        """Emit a `sample` / `observe` statement for a sample / observe
        step.

        Wraps the call in `repeat(B, function() { ... })` per batch
        dim when no arg uses the per-element index; uses
        `mapIndexed(function(m, _) { ... }, repeat(B, function() {
        return 0; }))` when an arg refers to a name whose binding
        plate aligns with the surrounding plate's batch dims.

        For an observed step the emit shape is
        `mapIndexed(function(n, <obs>_n) { observe(<dist>, <obs>_n);
        }, <obs>);` when batched; the unbatched case emits a bare
        `observe(<dist>, <obs>);` statement.

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
        del constraint  # Constraint shaped the (no-op) declaration.
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:unknown:{family}"],
            )
        webppl_name = meta.target_names.get("webppl")
        if webppl_name is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:no-webppl-target:{family}"],
            )
        if family == "MixtureNormal":
            return self._emit_mixture_normal(
                ctx, name, args, arg_names, plate, observed=observed
            )
        injected_args, injected_arg_names = _inject_webppl_specific_args(
            family, args, arg_names
        )
        if observed:
            return self._emit_observe(
                ctx,
                name,
                meta,
                webppl_name,
                injected_args,
                injected_arg_names,
                plate,
            )
        return self._emit_sample(
            ctx,
            name,
            meta,
            webppl_name,
            injected_args,
            injected_arg_names,
            plate,
        )

    def _emit_mixture_normal(
        self,
        ctx: RenderContext,
        name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        *,
        observed: bool,
    ) -> SchemaFragment:
        """Emit a `MixtureNormal` site as WebPPL's own `Mixture`.

        WebPPL ships no Gaussian-mixture constructor but it ships
        `Mixture({dists, ps})`, a finite mixture over an explicit array
        of component distributions whose `score` is the weighted
        log-sum-exp of the components' scores: the QVR likelihood's own
        closed form. `MixtureNormal(w, mu, sigma)` unrolls to `K`
        `Gaussian` components read off the per-component vectors, with
        the weight vector passed through as `ps`.

        The component array is the same for every row of the plate, so
        the emitted site keeps the ordinary batched shape: the
        `mapIndexed` observe for an observed site, `repeat` for a
        latent one.
        """
        weights, loc, scale = mixture_normal_components("webppl", args, arg_names)
        components = mixture_component_count(
            "webppl",
            weights,
            self._binding_plates.get(
                weights.name if isinstance(weights, IRArgRef) else ""
            ),
        )
        meta = FAMILY_META["MixtureNormal"]
        webppl_name = meta.target_names["webppl"]
        return self._emit_prepared_site(
            ctx,
            name,
            plate,
            observed=observed,
            dist_call=lambda: self._call(
                ctx,
                self._ident(ctx, webppl_name),
                (
                    self._object_literal(
                        ctx,
                        (
                            (
                                "dists",
                                self._mixture_component_array(
                                    ctx, loc, scale, components
                                ),
                            ),
                            ("ps", self._mixture_vector_ref(ctx, weights)),
                        ),
                    ),
                ),
            ),
        )

    def _mixture_component_array(
        self,
        ctx: RenderContext,
        loc: IRArg,
        scale: IRArg,
        components: int,
    ) -> str:
        """Build `[Gaussian({mu: loc[0], sigma: scale[0]}), ...]`.

        One entry per mixture component, reading the component's
        location and scale out of the per-component vectors the QVR
        call supplies.
        """
        arr = self._fresh(ctx, "mxarr")
        ctx.sb.vertex(arr, "array")
        normal_name = FAMILY_META["Normal"].target_names["webppl"]
        aliases = FAMILY_META["Normal"].arg_aliases["webppl"]
        for index in range(components):
            entry = self._call(
                ctx,
                self._ident(ctx, normal_name),
                (
                    self._object_literal(
                        ctx,
                        (
                            (
                                aliases["loc"],
                                self._subscript(
                                    ctx,
                                    self._mixture_vector_ref(ctx, loc),
                                    self._number_literal(ctx, index),
                                ),
                            ),
                            (
                                aliases["scale"],
                                self._subscript(
                                    ctx,
                                    self._mixture_vector_ref(ctx, scale),
                                    self._number_literal(ctx, index),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            ctx.sb.edge(arr, entry, "child_of")
        return arr

    def _mixture_vector_ref(self, ctx: RenderContext, arg: IRArg) -> str:
        """Render one of a `MixtureNormal` call's per-component vectors.

        The component axis belongs to the mixture rather than to the
        surrounding plate, so the reference stays whole and picks up no
        row index; anything other than a bare reference would carry an
        index the unrolled components cannot honour.
        """
        if not isinstance(arg, IRArgRef) or arg.indices:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"family:MixtureNormal:component-arg:"
                    f"{type(arg).__name__}: each of the weight, "
                    f"location and scale arguments must be a bare "
                    f"reference to a per-component vector"
                ],
            )
        return self._ident(ctx, arg.name)

    def _emit_prepared_site(
        self,
        ctx: RenderContext,
        name: str,
        plate: Plate,
        *,
        observed: bool,
        dist_call: Callable[[], str],
    ) -> SchemaFragment:
        """Emit a sample / observe statement around a distribution
        expression the caller builds itself.

        `dist_call` is invoked once per emitted occurrence rather than
        once overall: a schema vertex belongs to exactly one parent, so
        a batched site whose body is re-entered needs its own copy of
        the expression.
        """
        if observed:
            if not plate.batch_dims:
                observe_call = self._call(
                    ctx,
                    self._ident(ctx, "observe"),
                    (dist_call(), self._ident(ctx, name)),
                )
                self._emit_expression_statement(ctx, self._body_vid, observe_call)
                return observe_call
            loop_var = "n"
            per_elem_var = f"{name}_n"
            body = self._fresh(ctx, "obody")
            ctx.sb.vertex(body, "statement_block")
            self._emit_expression_statement(
                ctx,
                body,
                self._call(
                    ctx,
                    self._ident(ctx, "observe"),
                    (dist_call(), self._ident(ctx, per_elem_var)),
                ),
            )
            mi_call = self._call(
                ctx,
                self._ident(ctx, "mapIndexed"),
                (
                    self._function_expression(ctx, (loop_var, per_elem_var), body),
                    self._ident(ctx, name),
                ),
            )
            self._emit_expression_statement(ctx, self._body_vid, mi_call)
            return mi_call
        rhs = self._call(ctx, self._ident(ctx, "sample"), (dist_call(),))
        for dim in reversed(plate.batch_dims):
            rhs = self._wrap_in_repeat(ctx, rhs, dim)
        self._emit_var_decl(ctx, self._body_vid, name, rhs)
        self._binding_plates[name] = plate
        return rhs

    def _emit_gp_block(
        self,
        ctx: RenderContext,
        node,
    ) -> None:
        """Emit a Gaussian-process sample as three WebPPL var-decls:

            var __gp_mean_<name> = _qvr_zeros(N);
            var __gp_cov_<name>  = _qvr_rbf_kernel(x, length_scale, jitter);
            var <name> = sample(MultivariateGaussian({mu: __gp_mean_<name>,
                                                       cov: __gp_cov_<name>}));

        The ``_qvr_rbf_kernel`` / ``_qvr_zeros`` helpers live in
        [`runtime_webppl.js`][quivers.transpile.runtime_webppl] and
        are grafted into the emit through the existing runtime-helper
        graft path; GP is added to the helper-family set so the
        helpers appear above the model body when the IR uses GP.
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-webppl",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # var __gp_mean_<name> = _qvr_zeros(N);
        mean_rhs = self._call(
            ctx,
            self._ident(ctx, "_qvr_zeros"),
            (self._number_literal(ctx, n),),
        )
        self._emit_var_decl(ctx, self._body_vid, mean_name, mean_rhs)
        # var __gp_cov_<name> = _qvr_rbf_kernel(x, ls, jitter);
        cov_rhs = self._call(
            ctx,
            self._ident(ctx, "_qvr_rbf_kernel"),
            (
                self._ident(ctx, x),
                self._number_literal(ctx, ls),
                self._number_literal(ctx, jitter),
            ),
        )
        self._emit_var_decl(ctx, self._body_vid, cov_name, cov_rhs)
        # var <name> = sample(MultivariateGaussian({mu, cov}));
        params = self._object_literal(
            ctx,
            (
                ("mu", self._ident(ctx, mean_name)),
                ("cov", self._ident(ctx, cov_name)),
            ),
        )
        dist_call = self._call(
            ctx,
            self._ident(ctx, "MultivariateGaussian"),
            (params,),
        )
        sample_call = self._call(
            ctx,
            self._ident(ctx, "sample"),
            (dist_call,),
        )
        self._emit_var_decl(
            ctx,
            self._body_vid,
            node.name,
            sample_call,
        )

    def _emit_sample(
        self,
        ctx: RenderContext,
        name: str,
        meta: FamilyMeta,
        webppl_name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> SchemaFragment:
        """Emit `var <name> = <maybe-batched-call>;` for a latent sample."""
        # Decide the batching strategy: `repeat` vs `mapIndexed`.
        loop_name = self._loop_name_for(name, plate)
        index_dependent = self._args_use_surrounding_index(args, plate)
        # Build the inner sample call: `sample(<Dist>({...}))`.
        # When index-dependent, arg refs whose binding plate aligns
        # with the surrounding plate get indexed by the loop var.
        rendered_args = self._render_arg_tuple(
            ctx, args, arg_names, meta, plate, loop_name, index_dependent
        )
        dist_obj = self._object_literal(ctx, rendered_args)
        dist_call = self._call(ctx, self._ident(ctx, webppl_name), (dist_obj,))
        sample_call = self._call(ctx, self._ident(ctx, "sample"), (dist_call,))
        # Wrap the residual event axes (the ones the source declared
        # with `[over=<Axis>]` beyond the family's own event rank)
        # innermost, then the batch axis outermost. A `Normal` site
        # declared `over=LatentDim` under `iid_over=Item` therefore
        # binds a 32-by-2 nested array rather than a flat 32-vector.
        own_start = len(plate.event_dims) - meta.event_rank
        for dim in reversed(plate.event_dims[:own_start]):
            sample_call = self._wrap_in_repeat(ctx, sample_call, dim)
        # Wrap the call per the batching strategy.
        value_vid = self._wrap_for_batch(
            ctx, sample_call, plate, loop_name, index_dependent
        )
        # Bind the result to `var <name> = ...;`.
        self._emit_var_decl(ctx, self._body_vid, name, value_vid)
        # Record the binding plate so downstream refs see the right shape.
        return value_vid

    def _emit_observe(
        self,
        ctx: RenderContext,
        name: str,
        meta: FamilyMeta,
        webppl_name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> SchemaFragment:
        """Emit `mapIndexed(function(n, <obs>_n) { observe(<dist>,
        <obs>_n); }, <obs>);` for a batched observe; emit a bare
        `observe(<dist>, <obs>);` statement for the unbatched form.

        The per-observe loop variable is bound to `n` (mapIndexed's
        first arg); the per-element observed value is bound to
        `<name>_n`. The body's `<dist>` rendering threads `n` through
        the `via` fibration when present.
        """
        # When unbatched, emit a top-level observe(...) statement.
        if not plate.batch_dims:
            rendered_args = self._render_arg_tuple(
                ctx,
                args,
                arg_names,
                meta,
                plate,
                loop_name=None,
                index_dependent=False,
            )
            dist_obj = self._object_literal(ctx, rendered_args)
            dist_call = self._call(ctx, self._ident(ctx, webppl_name), (dist_obj,))
            observe_call = self._call(
                ctx,
                self._ident(ctx, "observe"),
                (dist_call, self._ident(ctx, name)),
            )
            self._emit_expression_statement(ctx, self._body_vid, observe_call)
            return observe_call
        if len(plate.batch_dims) > 1:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"batch-rank:{len(plate.batch_dims)}: WebPPL emits one "
                    f"batch axis per observation; multi-axis plates not "
                    f"implemented"
                ],
            )
        # Batched observe: mapIndexed over the observed data.
        # The loop var is `n`; the per-element bound var is `<name>_n`.
        loop_var = "n"
        per_elem_var = f"{name}_n"
        # Push the per-observe loop state so ref rendering can use it.
        prev_loop = self._observe_loop_var
        prev_via = self._observe_via
        self._observe_loop_var = loop_var
        # The observe's `via` is not surfaced on IRObserve directly
        # here; the caller passes plate. The dispatch site populates
        # `_observe_via` before calling _emit_observe via the
        # IRObserve node.
        try:
            rendered_args = self._render_arg_tuple(
                ctx,
                args,
                arg_names,
                meta,
                plate,
                loop_name=loop_var,
                index_dependent=True,
            )
            dist_obj = self._object_literal(ctx, rendered_args)
            dist_call = self._call(ctx, self._ident(ctx, webppl_name), (dist_obj,))
            observe_call = self._call(
                ctx,
                self._ident(ctx, "observe"),
                (dist_call, self._ident(ctx, per_elem_var)),
            )
            lambda_body = self._fresh(ctx, "obody")
            ctx.sb.vertex(lambda_body, "statement_block")
            self._emit_expression_statement(ctx, lambda_body, observe_call)
            lambda_expr = self._function_expression(
                ctx, (loop_var, per_elem_var), lambda_body
            )
            mi_call = self._call(
                ctx,
                self._ident(ctx, "mapIndexed"),
                (lambda_expr, self._ident(ctx, name)),
            )
            self._emit_expression_statement(ctx, self._body_vid, mi_call)
            return mi_call
        finally:
            self._observe_loop_var = prev_loop
            self._observe_via = prev_via

    # ----- marginalize dispatch -----

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Integrate an
        [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] latent
        out, adding the reduced density to the program's log-density
        with WebPPL's `factor`.

        One scored copy of the scope per atom of the latent's finite
        support, then a reduction across atoms:

        ```javascript
        var __marg_z_atom_0 = function () {
          var gated_rate = _qvr_bcast("*", 0, rate);
          return mapIndexed(function (n, y_n) {
            return (Poisson({mu:gated_rate[n]})).score(y_n);
          }, y);
        };
        var __marg_z_0 = __marg_z_atom_0();
        ...
        var __marg_z_t_0 = _qvr_bcast("+", __marg_z_w_0, __marg_z_0);
        ...
        var __marg_z = _qvr_logsumexp([__marg_z_t_0, __marg_z_t_1]);
        factor(_qvr_total(__marg_z));
        ```

        Each atom's deterministic bindings live inside that atom's own
        function, which is what gives them a scope: WebPPL's
        single-assignment subset rejects a second `var gated_rate` in
        the same block. No site is declared for the latent: the atoms
        replace it, and the emitted program denotes the same measure
        the QVR reference integrates.

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
        plates = dict(self._binding_plates)
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        prefix = f"__marg_{node.latent}"
        prev_group = self._group_plate_axes
        self._group_plate_axes = tuple(str(d.name) for d in node.plate.batch_dims)
        try:
            term_names = tuple(
                self._emit_atom_scope(ctx, node, atom, prefix, position)
                for position, atom in enumerate(atoms)
            )
            weight_names = self._emit_atom_weights(
                ctx, node, raw, atoms, prefix, plates
            )
        finally:
            self._group_plate_axes = prev_group
        # A grouped block keys its accumulator by group: the rows the
        # fibration sends to one group are summed before the reduction.
        fibration = marginalize_fibration(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=plates,
            target=self.target,
        )
        # An ungrouped block shares one latent across the body's rows,
        # so each class's per-row log-likelihoods are accumulated to a
        # scalar before the weight is added and the reduction runs
        # over the classes. Reducing the rows elementwise instead
        # would give every row its own draw.
        accumulate_rows = marginalize_row_rank(node) > 0
        shifted: list[str] = []
        for position, (weight, term) in enumerate(
            zip(weight_names, term_names, strict=True)
        ):
            name = f"{prefix}_t_{position}"
            term_vid = self._ident(ctx, term)
            if fibration is not None:
                via, group = fibration
                term_vid = self._call(
                    ctx,
                    self._ident(ctx, "_qvr_group_sums"),
                    (
                        term_vid,
                        self._ident(ctx, via),
                        self._dim_size_value(ctx, group),
                    ),
                )
            if accumulate_rows:
                term_vid = self._call(
                    ctx,
                    self._ident(ctx, "_qvr_total"),
                    (term_vid,),
                )
            self._emit_var_decl(
                ctx,
                self._body_vid,
                name,
                self._call(
                    ctx,
                    self._ident(ctx, "_qvr_bcast"),
                    (
                        self._string_literal(ctx, "+"),
                        self._ident(ctx, weight),
                        term_vid,
                    ),
                ),
            )
            shifted.append(name)
        terms_array = self._fresh(ctx, "arr")
        ctx.sb.vertex(terms_array, "array")
        for name in shifted:
            ctx.sb.edge(terms_array, self._ident(ctx, name), "child_of")
        self._emit_var_decl(
            ctx,
            self._body_vid,
            prefix,
            self._call(ctx, self._ident(ctx, "_qvr_logsumexp"), (terms_array,)),
        )
        self._emit_expression_statement(
            ctx,
            self._body_vid,
            self._call(
                ctx,
                self._ident(ctx, "factor"),
                (
                    self._call(
                        ctx,
                        self._ident(ctx, "_qvr_total"),
                        (self._ident(ctx, prefix),),
                    ),
                ),
            ),
        )
        return ""

    def _emit_atom_scope(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
        atom: IRMarginalAtom,
        prefix: str,
        position: int,
    ) -> str:
        """Bind one atom's scope log-density and return its name.

        The atom's deterministic bindings and its scored site go into
        a nullary function so each atom's `var` bindings get their own
        scope; the call site binds the returned per-row log-density
        array.
        """
        scored = marginalize_body(atom.scope, latent=node.latent, target=self.target)
        block = self._fresh(ctx, "abody")
        ctx.sb.vertex(block, "statement_block")
        outer_body = self._body_vid
        self._body_vid = block
        try:
            for det in scored.deterministics:
                self._emit_deterministic(ctx, det)
            score = self._atom_score_expression(ctx, scored.observe)
        finally:
            self._body_vid = outer_body
        self._emit_return_statement(ctx, block, score)
        fn_name = f"{prefix}_atom_{position}"
        self._emit_var_decl(
            ctx,
            self._body_vid,
            fn_name,
            self._function_expression(ctx, (), block),
        )
        term = f"{prefix}_{position}"
        self._emit_var_decl(
            ctx,
            self._body_vid,
            term,
            self._call(ctx, self._ident(ctx, fn_name), ()),
        )
        return term

    def _atom_score_expression(self, ctx: RenderContext, observe: IRObserve) -> str:
        """The per-row log-density of one atom's scored site.

        Batched sites map over the observed array so the result keeps
        one entry per row, which is the shape the atom reduction adds
        the log-weights to; unbatched sites return the bare scalar
        score.
        """
        meta = FAMILY_META.get(observe.family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-webppl", [f"family:unknown:{observe.family}"]
            )
        webppl_name = meta.target_names.get("webppl")
        if webppl_name is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:no-webppl-target:{observe.family}"],
            )
        args, arg_names = _inject_webppl_specific_args(
            observe.family, observe.args, observe.arg_names
        )
        prev_via = self._observe_via
        self._observe_via = observe.via
        try:
            if not observe.plate.batch_dims:
                rendered = self._render_arg_tuple(
                    ctx,
                    args,
                    arg_names,
                    meta,
                    observe.plate,
                    loop_name=None,
                    index_dependent=False,
                )
                return self._atom_score_of(
                    ctx,
                    observe.family,
                    webppl_name,
                    rendered,
                    self._ident(ctx, observe.name),
                )
            loop_var = "n"
            per_elem_var = f"{observe.name}_n"
            prev_loop = self._observe_loop_var
            self._observe_loop_var = loop_var
            try:
                rendered = self._render_arg_tuple(
                    ctx,
                    args,
                    arg_names,
                    meta,
                    observe.plate,
                    loop_name=loop_var,
                    index_dependent=True,
                )
            finally:
                self._observe_loop_var = prev_loop
            body = self._fresh(ctx, "sbody")
            ctx.sb.vertex(body, "statement_block")
            self._emit_return_statement(
                ctx,
                body,
                self._atom_score_of(
                    ctx,
                    observe.family,
                    webppl_name,
                    rendered,
                    self._ident(ctx, per_elem_var),
                ),
            )
            return self._call(
                ctx,
                self._ident(ctx, "mapIndexed"),
                (
                    self._function_expression(ctx, (loop_var, per_elem_var), body),
                    self._ident(ctx, observe.name),
                ),
            )
        finally:
            self._observe_via = prev_via

    def _atom_score_of(
        self,
        ctx: RenderContext,
        family: str,
        webppl_name: str,
        params: tuple[tuple[str, str], ...],
        value_vid: str,
    ) -> str:
        """The log-density of one value under one family, picking the
        calling convention the family needs.

        Three conventions:

        * a family in
          `_WEBPPL_BOUNDARY_SAFE_SCORERS`
          scores through a runtime helper that takes the parameter
          object directly, because WebPPL's constructor rejects a
          parameter value the reference measure admits;
        * a family this renderer grafts is a plain object whose
          `score` body reaches for WebPPL combinators, and WebPPL
          compiles a member call outside its CPS transform, so it
          routes through the top-level `_qvr_score` helper;
        * a built-in distribution reads `this` in its own `score` and
          keeps the member form.
        """
        params_vid = self._object_literal(ctx, params)
        boundary_safe = _WEBPPL_BOUNDARY_SAFE_SCORERS.get(family)
        if boundary_safe is not None:
            return self._call(
                ctx,
                self._ident(ctx, boundary_safe),
                (params_vid, value_vid),
            )
        dist_vid = self._call(ctx, self._ident(ctx, webppl_name), (params_vid,))
        if family in _WEBPPL_RUNTIME_HELPER_FAMILIES:
            return self._call(
                ctx,
                self._ident(ctx, "_qvr_score"),
                (dist_vid, value_vid),
            )
        member = self._fresh(ctx, "mem")
        ctx.sb.vertex(member, "member_expression")
        ctx.sb.edge(member, self._paren(ctx, dist_vid, "call_expression"), "object")
        ctx.sb.edge(member, self._prop_ident(ctx, "score"), "property")
        return self._call(ctx, member, (value_vid,))

    def _emit_atom_weights(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
        prefix: str,
        plates: dict[str, Plate],
    ) -> tuple[str, ...]:
        """Bind one log-weight name per atom and return the names.

        A `Bernoulli` atom set weights the atoms 0 and 1 by
        `log(1 - p)` and `log(p)`, both shaped like the probability
        itself. A `Categorical` atom set reads the class axis of the
        probability tensor, which `_qvr_take_last` slices however many
        grouping axes sit above it.
        """
        probs = marginal_weight_probs(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=plates,
            target=self.target,
        )
        family = atoms[0].weight_family
        if family == "Bernoulli":
            zero = f"{prefix}_w_0"
            one = f"{prefix}_w_1"
            self._emit_var_decl(
                ctx,
                self._body_vid,
                zero,
                self._call(
                    ctx,
                    self._ident(ctx, "log"),
                    (
                        self._call(
                            ctx,
                            self._ident(ctx, "_qvr_bcast"),
                            (
                                self._string_literal(ctx, "-"),
                                self._number_literal(ctx, 1),
                                self._render_arg(ctx, probs),
                            ),
                        ),
                    ),
                ),
            )
            self._emit_var_decl(
                ctx,
                self._body_vid,
                one,
                self._call(
                    ctx,
                    self._ident(ctx, "log"),
                    (self._render_arg(ctx, probs),),
                ),
            )
            return (zero, one)
        if family != "Categorical":
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"marginalize:weight-family:{family}: no WebPPL "
                    f"log-weight form for this atom set"
                ],
            )
        log_probs = f"{prefix}_w"
        if isinstance(probs, IRArgRef) and probs.indices:
            # A grouped block reads its prior through a fibration, so
            # the weights are one row of the grouping plate per
            # observation. JavaScript subscripts an array with a
            # single integer and yields `undefined` for an array of
            # them, so the gather is a map and the logarithm is taken
            # inside it.
            weight_vid = self._call(
                ctx,
                self._ident(ctx, "_qvr_gather_log"),
                (
                    self._ident(ctx, probs.name),
                    self._render_arg(ctx, probs.indices[0]),
                ),
            )
        else:
            weight_vid = self._call(
                ctx,
                self._ident(ctx, "log"),
                (self._render_arg(ctx, probs),),
            )
        self._emit_var_decl(ctx, self._body_vid, log_probs, weight_vid)
        names: list[str] = []
        for position in range(len(atoms)):
            name = f"{prefix}_w_{position}"
            self._emit_var_decl(
                ctx,
                self._body_vid,
                name,
                self._call(
                    ctx,
                    self._ident(ctx, "_qvr_take_last"),
                    (
                        self._ident(ctx, log_probs),
                        self._number_literal(ctx, position),
                    ),
                ),
            )
            names.append(name)
        return tuple(names)

    # ----- broadcast dispatch -----

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit a WebPPL broadcast call:
        `repeat(K, function() { return <value>; })` for 1D, nested
        for higher rank.

        Rank-0 returns the value unchanged. The renderer raises with
        a precise kind tag for rank > 2 because the canonical
        WebPPL Dirichlet / MVN call shapes only exercise rank-1 and
        rank-2 broadcast positions.

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
        if len(target_shape) == 0:
            return self._render_arg(ctx, value)
        if len(target_shape) > 2:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"broadcast:rank:{len(target_shape)}: WebPPL "
                    f"broadcast supports up to rank 2"
                ],
            )
        # Innermost: a function() returning the value.
        inner_value = self._render_arg(ctx, value)
        current = inner_value
        for size in reversed(target_shape):
            body = self._fresh(ctx, "rbody")
            ctx.sb.vertex(body, "statement_block")
            self._emit_return_statement(ctx, body, current)
            lam = self._function_expression(ctx, (), body)
            current = self._call(
                ctx,
                self._ident(ctx, "repeat"),
                (self._number_literal(ctx, size), lam),
            )
        return current

    # ----- arg rendering helpers (_render_list, _render_matrix) -----

    def _render_list(
        self,
        ctx: RenderContext,
        arg: IRArgList,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, WebPPL list args render as a JS
        array literal `[<e0>, <e1>, ...]`.
        """
        arr = self._fresh(ctx, "arr")
        ctx.sb.vertex(arr, "array")
        for element in arg.elements:
            child_vid = self._render_arg(ctx, element)
            ctx.sb.edge(arr, child_vid, "child_of")
        return arr

    def _render_matrix(
        self,
        ctx: RenderContext,
        arg: IRArgMatrix,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, WebPPL matrix args render as nested
        JS array literals `[[<r0_e0>, ...], [<r1_e0>, ...], ...]`.
        """
        outer = self._fresh(ctx, "arr")
        ctx.sb.vertex(outer, "array")
        for row in arg.rows:
            row_vid = self._render_list(ctx, row)
            ctx.sb.edge(outer, row_vid, "child_of")
        return outer

    # ------------------------------------------------------------------
    # IR-walk override: WebPPL-specific dispatch needs to thread the
    # observe's `via` fibration through to ref-rendering.
    # ------------------------------------------------------------------

    def _dispatch_node(self, ctx: RenderContext, node: IRNode) -> None:
        """WebPPL-specific dispatch.

        Tracks the active observe's `via` fibration so ref rendering
        can thread the per-element loop variable through references
        whose binding plate matches the fibration's group axis.
        """
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="parameters",
            )
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
            self.declare(ctx, node.name, node.constraint, node.plate, block="data")
            prev_via = self._observe_via
            self._observe_via = node.via
            try:
                self.sample(
                    ctx,
                    node.name,
                    node.family,
                    node.args,
                    node.arg_names,
                    node.constraint,
                    node.plate,
                    observed=True,
                )
            finally:
                self._observe_via = prev_via
            return
        if isinstance(node, IRDataInput):
            self.declare(ctx, node.name, node.constraint, node.plate, block="data")
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
        raise UnsupportedConstruct(
            "qvr-webppl",
            [f"node:{type(node).__name__}"],
        )

    def _emit_call(self, ctx: RenderContext, node: IRCall) -> None:
        """Place a call of a module computation in the model body.

        The first call in a body binds the native operation table,
        under which the callee's draws are `sample` sites and its
        scores `factor` terms; an argument that is not a bare name is
        bound first, so the call reads names alone.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        node : IRCall
            The call.
        """
        names: list[str] = []
        source = ""
        for position, argument in enumerate(node.arguments):
            if isinstance(argument, LetExprVar):
                names.append(argument.name)
                continue
            bound = f"{node.name}_arg{position}"
            if isinstance(argument, IRQiecValueExpr):
                source += value_argument_source(
                    "javascript",
                    "webppl",
                    bound,
                    argument.value,
                    lambda local: local,
                    body=self._body_vid,
                    defined=self._operations_bound,
                )
                names.append(bound)
                continue
            rhs = render_let_expr_javascript(
                _JsLetCtx(ctx.sb, lambda p: self._fresh(ctx, p), self._cards),
                _reduce_last_axis(argument, self._name_event_rank, self._name_rank),
            )
            self._emit_var_decl(ctx, self._body_vid, bound, rhs)
            names.append(bound)
        if self._body_vid not in self._operations_bound:
            self._operations_bound.add(self._body_vid)
            source += javascript_operations_source(node, self._module, _OPERATIONS)
        source += javascript_call_source(node, self._module, tuple(names), _OPERATIONS)
        graft_javascript_statements(
            ctx.sb, source, self._body_vid, f"qiec_call_{node.name}"
        )

    def _emit_deterministic(
        self,
        ctx: RenderContext,
        node: IRDeterministic,
    ) -> None:
        """Emit `var <name> = <expr>;` for a deterministic let-binding.

        Lowers [`node.expr`][quivers.transpile.ir.IRDeterministic.expr]
        through
        `render_let_expr_javascript`
        so binary ops, calls, indices, lambdas, list literals, and
        method calls reach the emitter as real JavaScript expression
        vertices rather than a self-referential identifier.

        The binding's own rank decides its form. A scalar binding is
        one expression, whatever plated names it reads at an index. A
        binding with a leading axis is built one row at a time,
        ``mapIndexed(function(__i, _) { return <row>; }, <axis>)``,
        where the row is the expression read at ``__i`` through
        [`take_row`][quivers.transpile.renderers._rows.take_row]: each
        operand carrying the leading axis is read at the row, and an
        operand of lower rank passes through unchanged. Operators over
        the arrays a row may still hold go through the elementwise
        `_qvr_bcast` helper, since a JavaScript operator would coerce
        an array to `NaN`.
        """
        expr = _reduce_last_axis(node.expr, self._name_event_rank, self._name_rank)
        rank = let_expr_rank(expr, self._name_rank)
        dims = (*node.plate.batch_dims, *node.plate.event_dims)
        js = _JsLetCtx(ctx.sb, lambda p: self._fresh(ctx, p), self._cards)
        if rank == 0 or not dims:
            rhs = render_let_expr_javascript(
                js, _vectorize_let_expr(expr, self._name_rank)
            )
            self._emit_var_decl(ctx, self._body_vid, node.name, rhs)
            self._binding_plates[node.name] = node.plate
            self._binding_supports[node.name] = node.constraint
            return
        loop_var = self._fresh_loop_var()
        row = take_row(expr, LetExprVar(name=loop_var), self._name_rank, "webppl")
        body_block = self._fresh(ctx, "lbody")
        ctx.sb.vertex(body_block, "statement_block")
        body_vid = render_let_expr_javascript(
            js, _vectorize_let_expr(row, self._name_rank)
        )
        self._emit_return_statement(ctx, body_block, body_vid)
        lam = self._function_expression(ctx, (loop_var, "_"), body_block)
        rows = self._zero_array(ctx, self._dim_size_value(ctx, dims[0]))
        mi_call = self._call(ctx, self._ident(ctx, "mapIndexed"), (lam, rows))
        self._emit_var_decl(ctx, self._body_vid, node.name, mi_call)
        self._binding_plates[node.name] = node.plate
        self._binding_supports[node.name] = node.constraint

    def _fresh_loop_var(self) -> str:
        """Return a fresh `__i_<n>` loop-variable name for a lifted
        deterministic binding.

        The renderer counts these independently of the schema vertex
        counter so that two lifted bindings in the same program do
        not collide on the same loop name.
        """
        self._lift_n += 1
        return f"__i_{self._lift_n}"

    def _emit_score(self, ctx: RenderContext, node: IRScore) -> None:
        """Emit `var <name> = <expr>; factor(<name>);` for a score
        increment.

        WebPPL's `factor(value)` adds `value` to the log-density;
        the convention is to bind the expression to a local var
        first so the factor reads a name. The expression is rendered
        through
        `render_let_expr_javascript`
        so the bound value is a real JavaScript expression. Operators
        over array-valued operands, such as a per-row weight times a
        per-row residual, go through the elementwise `_qvr_bcast` helper,
        since a JavaScript operator would coerce an array to `NaN`.
        """
        expr = _reduce_last_axis(node.expr, self._name_event_rank, self._name_rank)
        rhs = render_let_expr_javascript(
            _JsLetCtx(ctx.sb, lambda p: self._fresh(ctx, p), self._cards),
            _vectorize_let_expr(expr, self._name_rank),
        )
        self._emit_var_decl(ctx, self._body_vid, node.name, rhs)
        factor_call = self._call(
            ctx,
            self._ident(ctx, "factor"),
            (self._ident(ctx, node.name),),
        )
        self._emit_expression_statement(ctx, self._body_vid, factor_call)

    def _emit_return(
        self,
        ctx: RenderContext,
        names: tuple[str, ...],
    ) -> None:
        """Emit `return <var>;` for a single return; `return [a, b,
        ...];` for multiple returns.
        """
        if not names:
            return
        rs = self._fresh(ctx, "rs")
        ctx.sb.vertex(rs, "return_statement")
        if len(names) == 1:
            ctx.sb.edge(rs, self._ident(ctx, names[0]), "child_of")
        else:
            arr = self._fresh(ctx, "rarr")
            ctx.sb.vertex(arr, "array")
            for var in names:
                ctx.sb.edge(arr, self._ident(ctx, var), "child_of")
            ctx.sb.edge(rs, arr, "child_of")
        ctx.sb.edge(self._body_vid, rs, "child_of")

    # ------------------------------------------------------------------
    # Argument tuple rendering: orchestrates broadcast / index
    # threading / arg-aliases per-call.
    # ------------------------------------------------------------------

    def _render_arg_tuple(
        self,
        ctx: RenderContext,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        meta: FamilyMeta,
        plate: Plate,
        loop_name: str | None,
        index_dependent: bool,
    ) -> tuple[tuple[str, str], ...]:
        """Render every arg in `args`, returning a tuple of
        `(keyword, vertex_id)` pairs suitable for the WebPPL
        distribution object literal.

        The keyword for each position comes from `arg_names`
        (carried on the IR per spec §2.4), filtered through the
        family's per-backend `arg_aliases["webppl"]` rename map.

        Vector / matrix-valued arg slots that receive a scalar ref
        get wrapped in a `repeat` broadcast call so the resulting
        WebPPL distribution sees a vector. Refs whose binding plate
        aligns with the surrounding sample plate get prepended
        loop-index access when the call site is index-dependent.
        """
        alias_map = meta.arg_aliases.get("webppl", {})
        # Pull the per-arg constraint table for broadcast detection.
        cls_attr = getattr(meta.distribution_class, "arg_constraints", None)
        per_arg_constraints: tuple[object, ...]
        if isinstance(cls_attr, dict):
            per_arg_constraints = tuple(cls_attr.values())
        else:
            per_arg_constraints = ()
        if len(args) != len(arg_names):
            # The IR contract is parallel arrays; a mismatch is a
            # genuine bug rather than a renderer concern.
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"arg-names-mismatch:{meta.qvr_name}: "
                    f"{len(args)} args vs {len(arg_names)} names"
                ],
            )
        out: list[tuple[str, str]] = []
        for idx, (arg, raw_name) in enumerate(zip(args, arg_names, strict=True)):
            keyword = alias_map.get(raw_name, raw_name)
            expected = (
                per_arg_constraints[idx] if idx < len(per_arg_constraints) else None
            )
            broadcasted = self._maybe_broadcast(arg, expected, plate)
            substituted = self._substitute_for_indexing(
                broadcasted, plate, loop_name, index_dependent
            )
            reciprocal = raw_name in _WEBPPL_ARG_RECIPROCAL.get(
                meta.qvr_name, frozenset()
            )
            if reciprocal:
                vid = self._render_reciprocal(ctx, substituted)
            else:
                vid = self._render_arg(ctx, substituted)
            if raw_name in _WEBPPL_TENSOR_ARGS.get(meta.qvr_name, frozenset()):
                vid = self._call(ctx, self._ident(ctx, "Vector"), (vid,))
            elif raw_name in _WEBPPL_MATRIX_ARGS.get(meta.qvr_name, frozenset()):
                vid = self._call(ctx, self._ident(ctx, "Matrix"), (vid,))
            out.append((keyword, vid))
        if meta.qvr_name == "Categorical":
            # WebPPL's `Categorical` ranges over the values it is
            # given and has no default for them, so a `{ps}` alone is
            # rejected outright. QVR's ranges over the positions of
            # its own probability vector, which is what the observed
            # index means, so the support is those positions.
            probabilities = next(
                (vid for key, vid in out if key == "ps"),
                None,
            )
            if probabilities is None:
                raise UnsupportedConstruct(
                    "qvr-webppl",
                    [
                        f"arg-names-mismatch:{meta.qvr_name}: no `ps` "
                        f"argument to take the support from"
                    ],
                )
            out.append(
                (
                    "vs",
                    self._call(
                        ctx,
                        self._ident(ctx, "_qvr_support"),
                        (probabilities,),
                    ),
                )
            )
        return tuple(out)

    def _maybe_broadcast(
        self,
        arg: IRArg,
        expected_constraint: object,
        plate: Plate,
    ) -> IRArg:
        """Wrap `arg` in
        [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] when
        the call-site demands a vector / matrix and the user passed
        a scalar reference / literal.

        The expected event rank is read from the family's
        `arg_constraints` entry's `event_dim`; the target shape is
        taken from the surrounding sample's `plate.event_dims`.
        Refs whose binding plate is already vector-shaped pass
        through unchanged.
        """
        if expected_constraint is None:
            return arg
        expected_rank = int(getattr(expected_constraint, "event_dim", 0))
        if expected_rank < 1:
            return arg
        if len(plate.event_dims) < expected_rank:
            return arg
        # Skip wrapping when the IR already encodes the broadcast.
        if isinstance(arg, IRArgBroadcast):
            return arg
        # Lists / matrices are already vector-shaped.
        if isinstance(arg, (IRArgList, IRArgMatrix)):
            return arg
        # Family-refs resolve inline; broadcast wrapping is wrong.
        if isinstance(arg, IRArgFamilyRef):
            return arg
        # Refs to scalar bindings need wrapping; refs to a name
        # whose binding plate is non-scalar already carry the right
        # shape.
        if isinstance(arg, IRArgRef):
            decl_plate = self._binding_plates.get(arg.name)
            if decl_plate is not None and decl_plate.event_dims:
                return arg
        # Compute the target shape from the surrounding plate's
        # event dims (first `expected_rank` entries).
        sizes: list[int] = []
        for dim in plate.event_dims[:expected_rank]:
            if isinstance(dim, DimStatic):
                sizes.append(dim.size)
            else:
                # Dynamic-sized event dims would need a different
                # call form (e.g. `repeat(K, ...)` where K is a
                # data input); raise to keep the surface explicit.
                raise UnsupportedConstruct(
                    "qvr-webppl",
                    [
                        f"broadcast:dynamic-event-dim:{dim.name}: "
                        f"dynamic event-dim broadcast not "
                        f"implemented for WebPPL"
                    ],
                )
        return IRArgBroadcast(value=arg, target_shape=tuple(sizes))

    def _substitute_for_indexing(
        self,
        arg: IRArg,
        plate: Plate,
        loop_name: str | None,
        index_dependent: bool,
    ) -> IRArg:
        """Thread the surrounding plate's loop variable through any
        [`IRArgRef`][quivers.transpile.ir.IRArgRef] whose binding
        plate's batch axes align with the surrounding plate's batch
        axes.

        For an unbatched call site or an index-independent call
        site, returns `arg` unchanged. For an observed step under a
        `via` fibration, the per-observe loop var is prepended to
        the ref's index list along the fibration's group axis.
        """
        if not index_dependent or loop_name is None:
            return arg
        return self._substitute_ref_indexing(arg, plate, loop_name)

    def _substitute_ref_indexing(
        self,
        arg: IRArg,
        plate: Plate,
        loop_name: str,
    ) -> IRArg:
        """Recursive worker for `_substitute_for_indexing`. Walks
        every IR-arg subtree threading the loop name into ref
        indices that align with the surrounding plate.
        """
        if isinstance(arg, IRArgRef):
            decl_plate = self._binding_plates.get(arg.name)
            new_indices = tuple(
                self._substitute_ref_indexing(idx, plate, loop_name)
                for idx in arg.indices
            )
            if decl_plate is not None and self._plates_align(decl_plate, plate):
                # Prepend the loop var to the existing indices.
                loop_ref = IRArgRef(name=loop_name)
                return IRArgRef(
                    name=arg.name,
                    indices=(loop_ref, *new_indices),
                )
            # Apply `via` fibration to refs whose binding plate
            # equals the active marginalize / group plate. The
            # observe loop var goes through `<via>[<loop>]` so the
            # per-row observation indexes into the per-group latent
            # / parameter.
            if (
                self._observe_via is not None
                and decl_plate is not None
                and self._matches_group_plate(decl_plate)
                and not self._is_data_input(arg.name)
            ):
                via_ref = IRArgRef(
                    name=self._observe_via,
                    indices=(IRArgRef(name=loop_name),),
                )
                return IRArgRef(
                    name=arg.name,
                    indices=(via_ref, *new_indices),
                )
            return IRArgRef(name=arg.name, indices=new_indices)
        if isinstance(arg, IRArgBroadcast):
            return IRArgBroadcast(
                value=self._substitute_ref_indexing(arg.value, plate, loop_name),
                target_shape=arg.target_shape,
            )
        if isinstance(arg, IRArgList):
            return IRArgList(
                elements=tuple(
                    self._substitute_ref_indexing(e, plate, loop_name)
                    for e in arg.elements
                )
            )
        if isinstance(arg, IRArgMatrix):
            return IRArgMatrix(
                rows=tuple(
                    IRArgList(
                        elements=tuple(
                            self._substitute_ref_indexing(e, plate, loop_name)
                            for e in row.elements
                        )
                    )
                    for row in arg.rows
                )
            )
        return arg

    def _is_data_input(self, name: str) -> bool:
        """True iff `name` was registered as an IRDataInput (versus
        a sample / observe / let binding). Data inputs are the
        program's function parameters; their refs do not need
        per-element threading through a `via` fibration.
        """
        plate = self._binding_plates.get(name)
        if plate is None:
            return False
        # Data inputs are registered with no event_dims; samples
        # whose batch_dims encode a plate also live in
        # _binding_plates. The discriminator is whether the name
        # corresponds to a function parameter; we use the
        # function-parameter list as the authoritative source.
        return name in self._function_parameters

    @property
    def _function_parameters(self) -> set[str]:
        """The names registered as IRDataInputs at render start."""
        return self._function_parameters_state

    def _matches_group_plate(self, decl: Plate) -> bool:
        """True iff `decl`'s batch axes equal the active marginalize
        group plate axes.

        Drives the `via` fibration threading: a ref to a name bound
        on the marginalize plate (Doc) referenced from an observe
        on a different plate (Word) needs to be indexed by
        `<via>[<loop>]` when `via` maps the observe's plate to the
        marginalize's plate.
        """
        if not self._group_plate_axes:
            return False
        decl_names = tuple(d.name for d in decl.batch_dims)
        return decl_names == self._group_plate_axes

    def _plates_align(self, decl: Plate, surrounding: Plate) -> bool:
        """True iff `decl` and `surrounding` share their leading batch axes.

        The alignment rule is name-based: a ref to a name bound on
        a plate with batch axes `(A,)` referenced from a sample on
        a plate with batch axes `(A, B)` aligns on `A`, and a ref to a
        name bound over `(A, B)` referenced from a site over `(A,)`
        aligns on `A` too, the site reading the name's row, whose
        remaining axes are the row's own. The check is purely
        name-based on `Dim.name`.
        """
        if not decl.batch_dims or not surrounding.batch_dims:
            return False
        decl_names = tuple(d.name for d in decl.batch_dims)
        surr_names = tuple(d.name for d in surrounding.batch_dims)
        shared = min(len(decl_names), len(surr_names))
        return decl_names[:shared] == surr_names[:shared]

    def _args_use_surrounding_index(
        self,
        args: tuple[IRArg, ...],
        plate: Plate,
    ) -> bool:
        """True iff at least one arg in `args` references a name
        whose binding plate aligns with the surrounding plate's
        batch axes.

        Drives the `repeat` vs `mapIndexed` decision in
        `_emit_sample`: index-dependent call sites need the loop
        variable in scope to index into the referenced names.
        """
        if not plate.batch_dims:
            return False
        return any(self._arg_uses_surrounding_index(a, plate) for a in args)

    def _arg_uses_surrounding_index(
        self,
        arg: IRArg,
        plate: Plate,
    ) -> bool:
        """Recursive worker for `_args_use_surrounding_index`."""
        if isinstance(arg, IRArgRef):
            decl_plate = self._binding_plates.get(arg.name)
            if decl_plate is not None and self._plates_align(decl_plate, plate):
                return True
            return any(
                self._arg_uses_surrounding_index(idx, plate) for idx in arg.indices
            )
        if isinstance(arg, IRArgBroadcast):
            return self._arg_uses_surrounding_index(arg.value, plate)
        if isinstance(arg, IRArgList):
            return any(self._arg_uses_surrounding_index(e, plate) for e in arg.elements)
        if isinstance(arg, IRArgMatrix):
            return any(
                self._arg_uses_surrounding_index(e, plate)
                for row in arg.rows
                for e in row.elements
            )
        return False

    def _loop_name_for(self, sample_name: str, plate: Plate) -> str | None:
        """Pick the loop-variable name for a batched sample step.

        Uses the spec's `m_<axis>_<name>` form so multi-batch
        nesting stays unambiguous and the name is stable across
        emits of the same program.
        """
        if not plate.batch_dims:
            return None
        first = plate.batch_dims[0]
        return f"m_{first.name}_{sample_name}"

    def _wrap_in_repeat(self, ctx: RenderContext, inner_value: str, dim: Dim) -> str:
        """`repeat(<|dim|>, function () { return <inner_value>; })`.

        The iid-replication idiom, shared by the batch-axis wrapper
        and the residual-event-axis wrapper.
        """
        size_vid = self._dim_size_value(ctx, dim)
        body = self._fresh(ctx, "rbody")
        ctx.sb.vertex(body, "statement_block")
        self._emit_return_statement(ctx, body, inner_value)
        lam = self._function_expression(ctx, (), body)
        return self._call(ctx, self._ident(ctx, "repeat"), (size_vid, lam))

    def _zero_array(self, ctx: RenderContext, size_vid: str) -> str:
        """`repeat(<size>, function () { return 0; })`.

        The array an index-carrying `mapIndexed` walks when the
        elements themselves are unread: `mapIndexed` takes an array,
        not a count, and WebPPL's `repeat` builds one by calling its
        second argument, which must thus be a function rather
        than the constant it returns.
        """
        body = self._fresh(ctx, "zbody")
        ctx.sb.vertex(body, "statement_block")
        self._emit_return_statement(ctx, body, self._number_literal(ctx, 0))
        return self._call(
            ctx,
            self._ident(ctx, "repeat"),
            (size_vid, self._function_expression(ctx, (), body)),
        )

    def _wrap_for_batch(
        self,
        ctx: RenderContext,
        inner_value: str,
        plate: Plate,
        loop_name: str | None,
        index_dependent: bool,
    ) -> str:
        """Wrap `inner_value` in `repeat(N, function() { return <v>;
        })` per batch dim, or
        `mapIndexed(function(m, _) { return <v>; }, <length-N array>)`
        when an arg uses the loop index.

        `mapIndexed` is the only WebPPL combinator that hands its
        callback the position, and it walks an array rather than a
        count, so the index-dependent form needs a length-`N` array
        to walk: `_zero_array`
        builds one.

        The unbatched case returns `inner_value` unchanged.
        """
        batch_dims_list: list[Dim] = list(plate.batch_dims)
        if not batch_dims_list:
            return inner_value
        if len(batch_dims_list) > 1:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"batch-rank:{len(batch_dims_list)}: WebPPL "
                    f"emits one batch axis per sample; multi-axis "
                    f"plates not implemented"
                ],
            )
        dim = batch_dims_list[0]
        size_vid = self._dim_size_value(ctx, dim)
        body = self._fresh(ctx, "rbody")
        ctx.sb.vertex(body, "statement_block")
        self._emit_return_statement(ctx, body, inner_value)
        if index_dependent and loop_name is not None:
            # mapIndexed(function(m, _) { return <inner>; },
            #            repeat(N, function() { return 0; }))
            lam = self._function_expression(ctx, (loop_name, "_"), body)
            return self._call(
                ctx,
                self._ident(ctx, "mapIndexed"),
                (lam, self._zero_array(ctx, size_vid)),
            )
        # repeat(N, function() { return <inner>; })
        lam = self._function_expression(ctx, (), body)
        return self._call(
            ctx,
            self._ident(ctx, "repeat"),
            (size_vid, lam),
        )

    # ------------------------------------------------------------------
    # Render one IRArg to a JS expression vertex.
    # ------------------------------------------------------------------

    def _render_arg(
        self,
        ctx: RenderContext,
        arg: IRArg,
    ) -> SchemaFragment:
        """Render any [`IRArg`][quivers.transpile.ir.IRArg] to a JS
        expression vertex."""
        if isinstance(arg, IRArgNumber):
            return self._render_number(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            return self._render_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(ctx, arg)
        raise UnsupportedConstruct(
            "qvr-webppl",
            [f"arg:unknown:{type(arg).__name__}"],
        )

    def _render_reciprocal(self, ctx: RenderContext, arg: IRArg) -> SchemaFragment:
        """Render ``1 / <arg>`` as a JS `binary_expression`.

        WebPPL's `Gamma({shape, scale})` is scale-parameterised, but
        QVR / torch carry the Gamma rate; the reciprocal converts the
        rate into the scale WebPPL expects. The inner operand is
        parenthesised when it is itself a binary / unary expression so
        the division binds correctly.
        """
        inner_vid = self._render_arg(ctx, arg)
        inner_kind = self._js_kind_of(arg)
        if inner_kind in ("binary_expression", "unary_expression"):
            inner_vid = self._paren(ctx, inner_vid, inner_kind)
            inner_kind = "parenthesized_expression"
        one = self._number_literal(ctx, 1)
        be = self._fresh(ctx, "bin")
        ctx.sb.vertex(be, "binary_expression")
        ctx.sb.constraint(be, "field:operator", "/")
        ctx.sb.constraint(be, "chose-alt-fingerprint", "/")
        ctx.sb.constraint(be, "chose-alt-child-kinds", f"number {inner_kind}")
        ctx.sb.edge(be, one, "left")
        ctx.sb.edge(be, inner_vid, "right")
        return be

    def _paren(self, ctx: RenderContext, inner_vid: str, inner_kind: str) -> str:
        """Wrap `inner_vid` in a `parenthesized_expression` vertex."""
        paren = self._fresh(ctx, "paren")
        ctx.sb.vertex(paren, "parenthesized_expression")
        ctx.sb.constraint(paren, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(paren, "chose-alt-child-kinds", inner_kind)
        ctx.sb.edge(paren, inner_vid, "child_of")
        return paren

    def _js_kind_of(self, arg: IRArg) -> str:
        """Return the JS vertex kind
        `_render_arg`
        produces for ``arg``.

        Used to populate the ``chose-alt-child-kinds`` constraint of a
        parent expression that wraps a rendered arg (e.g. the
        reciprocal `binary_expression`)."""
        if isinstance(arg, IRArgNumber):
            return "number"
        if isinstance(arg, IRArgRef):
            return "subscript_expression" if arg.indices else "identifier"
        if isinstance(arg, IRArgBroadcast):
            return "call_expression"
        if isinstance(arg, (IRArgList, IRArgMatrix)):
            return "array"
        if isinstance(arg, IRArgFamilyRef):
            return "call_expression"
        raise UnsupportedConstruct(
            "qvr-webppl",
            [f"reciprocal:arg-kind:{type(arg).__name__}"],
        )

    def _render_number(self, ctx: RenderContext, value: float) -> str:
        return self._number_literal(ctx, value)

    def _render_ref(self, ctx: RenderContext, arg: IRArgRef) -> SchemaFragment:
        """Render an IRArgRef. Bare-name refs emit an identifier;
        indexed refs build `subscript_expression` chains."""
        base = self._ident(ctx, arg.name)
        if not arg.indices:
            return base
        current = base
        for idx in arg.indices:
            idx_vid = self._render_arg(ctx, idx)
            current = self._subscript(ctx, current, idx_vid)
        return current

    def _render_family_ref(
        self,
        ctx: RenderContext,
        arg: IRArgFamilyRef,
    ) -> SchemaFragment:
        """Resolve an
        [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] and
        emit the inner distribution call inline.

        WebPPL has no generic truncation idiom; the renderer reads
        the referenced morphism's `init_family` clause from
        `ctx.morphisms` and emits the inner distribution call. The
        outer wrapper family's renderer is responsible for shaping
        the wrapped call into the right WebPPL form.
        """
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"arg:family-ref:{arg.name}: no morphism with "
                    f"`~ Family(...)` init clause in scope"
                ],
            )
        init = decl.init_family
        inner_meta = FAMILY_META.get(init.family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:unknown:{init.family}"],
            )
        inner_name = inner_meta.target_names.get("webppl")
        if inner_name is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:no-webppl-target:{init.family}"],
            )
        # Best-effort: emit the inner call with positional args. The
        # init_family clause carries raw DrawArg-shaped values; we
        # translate to expression vertices and pack them into an
        # object literal using the inner family's `arg_names` from
        # its arg_constraints.
        alias_map = inner_meta.arg_aliases.get("webppl", {})
        raw_args = bind_family_arguments(init.family, init.args or ())
        entries: list[tuple[str, str]] = []
        for raw_key, raw in raw_args:
            keyword = str(alias_map.get(raw_key, raw_key))
            vid = self._render_init_family_arg(ctx, raw)
            entries.append((keyword, vid))
        dist_obj = self._object_literal(ctx, tuple(entries))
        return self._call(
            ctx,
            self._ident(ctx, inner_name),
            (dist_obj,),
        )

    def _render_init_family_arg(
        self,
        ctx: RenderContext,
        raw: object,
    ) -> SchemaFragment:
        """Render an `init_family` raw arg.

        The DSL surface gives us wire-form strings / floats /
        structured DrawArg variants; translate them to JS
        expression vertices.
        """
        if not isinstance(raw, (int, float, str)):
            raw = atom_value(raw)
        if isinstance(raw, (int, float)):
            return self._number_literal(ctx, float(raw))
        if isinstance(raw, str):
            stripped = raw.strip()
            try:
                value = float(stripped)
            except ValueError:
                return self._ident(ctx, stripped)
            return self._number_literal(ctx, value)
        raise UnsupportedConstruct(
            "qvr-webppl",
            [
                f"arg:family-ref:init-arg:{type(raw).__name__}: "
                f"unsupported raw arg shape"
            ],
        )

    # ------------------------------------------------------------------
    # Low-level JS schema builders.
    # ------------------------------------------------------------------

    def _ident(self, ctx: RenderContext, text: str) -> str:
        vid = self._fresh(ctx, "id")
        ctx.sb.vertex(vid, "identifier")
        ctx.sb.constraint(vid, "literal-value", text)
        return vid

    def _prop_ident(self, ctx: RenderContext, text: str) -> str:
        vid = self._fresh(ctx, "pid")
        ctx.sb.vertex(vid, "property_identifier")
        ctx.sb.constraint(vid, "literal-value", text)
        return vid

    def _string_literal(self, ctx: RenderContext, text: str) -> str:
        """Build a double-quoted JS `string` wrapping a
        `string_fragment` child."""
        vid = self._fresh(ctx, "str")
        ctx.sb.vertex(vid, "string")
        ctx.sb.constraint(vid, "chose-alt-fingerprint", '" "')
        ctx.sb.constraint(vid, "chose-alt-child-kinds", "string_fragment")
        frag = self._fresh(ctx, "sfrag")
        ctx.sb.vertex(frag, "string_fragment")
        ctx.sb.constraint(frag, "literal-value", text)
        ctx.sb.constraint(frag, "chose-alt-fingerprint", text)
        ctx.sb.edge(vid, frag, "child_of")
        return vid

    def _number_literal(self, ctx: RenderContext, value: int | float) -> str:
        vid = self._fresh(ctx, "num")
        ctx.sb.vertex(vid, "number")
        text = str(int(value)) if float(value).is_integer() else repr(float(value))
        ctx.sb.constraint(vid, "literal-value", text)
        return vid

    def _object_literal(
        self,
        ctx: RenderContext,
        entries: tuple[tuple[str, str], ...],
    ) -> str:
        """Build a JS object literal `{k1: v1, k2: v2, ...}`."""
        obj = self._fresh(ctx, "obj")
        ctx.sb.vertex(obj, "object")
        for key, value_vid in entries:
            pair = self._fresh(ctx, "pair")
            ctx.sb.vertex(pair, "pair")
            ctx.sb.edge(pair, self._prop_ident(ctx, key), "key")
            ctx.sb.edge(pair, value_vid, "value")
            ctx.sb.edge(obj, pair, "child_of")
        return obj

    def _call(
        self,
        ctx: RenderContext,
        callee_vid: str,
        positional: tuple[str, ...],
    ) -> str:
        """Build a JS call expression `callee(arg1, arg2, ...)`."""
        call = self._fresh(ctx, "call")
        ctx.sb.vertex(call, "call_expression")
        args = self._fresh(ctx, "args")
        ctx.sb.vertex(args, "arguments")
        ctx.sb.edge(call, callee_vid, "function")
        ctx.sb.edge(call, args, "arguments")
        for pid in positional:
            ctx.sb.edge(args, pid, "child_of")
        return call

    def _function_expression(
        self,
        ctx: RenderContext,
        params: tuple[str, ...],
        body_vid: str,
    ) -> str:
        """Build a JS `function(<params>) { <body> }` expression."""
        fn = self._fresh(ctx, "fn")
        ctx.sb.vertex(fn, "function_expression")
        ps = self._fresh(ctx, "ps")
        ctx.sb.vertex(ps, "formal_parameters")
        for name in params:
            ctx.sb.edge(ps, self._ident(ctx, name), "child_of")
        ctx.sb.edge(fn, ps, "parameters")
        ctx.sb.edge(fn, body_vid, "body")
        return fn

    def _subscript(
        self,
        ctx: RenderContext,
        object_vid: str,
        index_vid: str,
    ) -> str:
        """Build a JS subscript expression `<object>[<index>]`."""
        sub = self._fresh(ctx, "sub")
        ctx.sb.vertex(sub, "subscript_expression")
        ctx.sb.edge(sub, object_vid, "object")
        ctx.sb.edge(sub, index_vid, "index")
        return sub

    def _emit_var_decl(
        self,
        ctx: RenderContext,
        parent_vid: str,
        name: str,
        value_vid: str,
    ) -> str:
        """Emit `var <name> = <value>;` into `parent_vid`."""
        vd = self._fresh(ctx, "vd")
        ctx.sb.vertex(vd, "variable_declaration")
        dr = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr, "variable_declarator")
        ctx.sb.edge(dr, self._ident(ctx, name), "name")
        ctx.sb.edge(dr, value_vid, "value")
        ctx.sb.edge(vd, dr, "child_of")
        ctx.sb.edge(parent_vid, vd, "child_of")
        return vd

    def _emit_expression_statement(
        self,
        ctx: RenderContext,
        parent_vid: str,
        expr_vid: str,
    ) -> str:
        """Emit `<expr>;` into `parent_vid`."""
        es = self._fresh(ctx, "es")
        ctx.sb.vertex(es, "expression_statement")
        ctx.sb.edge(es, expr_vid, "child_of")
        ctx.sb.edge(parent_vid, es, "child_of")
        return es

    def _emit_return_statement(
        self,
        ctx: RenderContext,
        parent_vid: str,
        value_vid: str,
    ) -> str:
        """Emit `return <value>;` into `parent_vid`."""
        rs = self._fresh(ctx, "rs")
        ctx.sb.vertex(rs, "return_statement")
        ctx.sb.edge(rs, value_vid, "child_of")
        ctx.sb.edge(parent_vid, rs, "child_of")
        return rs

    def _dim_size_value(self, ctx: RenderContext, dim: Dim) -> str:
        """Render a plate dim's size as a JS expression.

        Static dims emit an integer literal; dynamic dims emit the
        `size_name` identifier so the surrounding scope (typically
        a function parameter) supplies the runtime value.
        """
        if isinstance(dim, DimStatic):
            return self._number_literal(ctx, dim.size)
        if isinstance(dim, DimDynamic):
            return self._ident(ctx, dim.size_name)
        raise UnsupportedConstruct(
            "qvr-webppl",
            [f"dim:unknown:{type(dim).__name__}"],
        )

    def _fresh(self, ctx: RenderContext, prefix: str) -> str:
        """Return a fresh vertex id with `prefix`.

        Renderer-internal counter; the base's
        `RenderContext.fresh_counter` is left untouched so per-walk
        node IDs stay stable across renderer instances.
        """
        del ctx
        self._fresh_n += 1
        return f"{prefix}_{self._fresh_n}"

    # ------------------------------------------------------------------
    # Morphism / let resolution for IRArgFamilyRef.
    # ------------------------------------------------------------------

    def _resolve_morphisms_and_lets(
        self,
    ) -> tuple[dict[str, MorphismDecl], dict[str, Expr]]:
        """Build the morphism / let tables from the source module.

        Used by `IRArgFamilyRef` resolution. Empty when the
        renderer was constructed without a source module; in that
        case `IRArgFamilyRef` rendering raises with a precise
        kind tag.
        """
        if self._source_module is None:
            return {}, {}
        morphisms: dict[str, MorphismDecl] = {}
        lets: dict[str, Expr] = {}
        for stmt in self._source_module.statements:
            if isinstance(stmt, MorphismDecl):
                for name in stmt.names:
                    morphisms[name] = stmt
            elif isinstance(stmt, DefineDecl):
                lets[stmt.name] = stmt.expr
        return morphisms, lets


def _free_names_in_let_expr(expr: LetExprNode) -> tuple[str, ...]:
    """Return the variable names referenced anywhere in ``expr``, in
    left-to-right pre-order.

    Reads bare-name `LetExprVar` nodes plus the function-name of
    `LetExprCall`. Lambda parameters bound inside the expression do
    not occur as free names by construction here (the renderer's
    upstream let-binding semantics rebind every name through
    `IRDeterministic`).
    """
    out: list[str] = []
    _collect_free_names(expr, out)
    return tuple(out)


def _collect_free_names(expr: LetExprNode, out: list[str]) -> None:
    """Walk ``expr`` adding each `LetExprVar` name to ``out``."""
    if isinstance(expr, LetExprVar):
        out.append(expr.name)
        return
    if isinstance(expr, LetExprLiteral):
        return
    if isinstance(expr, LetExprString):
        return
    if isinstance(expr, LetExprBinOp):
        _collect_free_names(expr.left, out)
        _collect_free_names(expr.right, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_free_names(expr.operand, out)
        return
    if isinstance(expr, LetExprCall):
        for a in expr.args:
            _collect_free_names(a, out)
        return
    if isinstance(expr, LetExprIndex):
        _collect_free_names(expr.array, out)
        for idx in expr.indices:
            _collect_free_names(idx, out)
        return
    if isinstance(expr, LetExprList):
        for item in expr.items:
            _collect_free_names(item, out)
        return
    if isinstance(expr, LetExprLambda):
        _collect_free_names(expr.body, out)
        return
    if isinstance(expr, LetExprMethodCall):
        _collect_free_names(expr.receiver, out)
        for a in expr.args:
            _collect_free_names(a, out)
        return
    if isinstance(expr, LetExprFactor):
        if expr.body is not None:
            _collect_free_names(expr.body, out)
        for case in expr.cases:
            _collect_free_names(case.value, out)
        return


# ---------------------------------------------------------------------------
# Vectorised arithmetic for array-valued deterministic bindings.
#
# WebPPL's `+`, `-`, `*`, `/` operators are scalar-only. A deterministic
# `let` that combines an array-valued prior (a vector sample, or another
# array-valued let) under a scalar operator would coerce the array to
# `NaN` (or string-concatenate it). The renderer rewrites every binary /
# unary operator whose operands are array-valued into a
# [`_qvr_bcast`][quivers.transpile.runtime_webppl] call so the arithmetic
# stays elementwise. Reduction calls (`sum`, ...) collapse their argument
# to a scalar, so an operator applied to a reduction result stays scalar.
# ---------------------------------------------------------------------------


#: QVR let-expression call names that reduce an array argument to a
#: scalar; an operator applied to their result does not need
#: broadcasting.
_REDUCTION_FUNCS: frozenset[str] = frozenset(
    {
        "sum",
        "prod",
        "mean",
        "max",
        "min",
        "logsumexp",
        "norm",
        "dot",
    }
)


#: QVR reduction primitives paired with the runtime helper that
#: collapses only the innermost axis. WebPPL's own `sum` flattens a
#: nested array to a single scalar, which is a different quantity from
#: the per-row inner product an event-axis reduction denotes.
_LAST_AXIS_REDUCERS: dict[str, str] = {
    "sum": "_qvr_sum_last",
    "mean": "_qvr_mean_last",
    "prod": "_qvr_prod_last",
    "max": "_qvr_max_last",
    "min": "_qvr_min_last",
}


def _js_event_rank(expr: LetExprNode, ranks: dict[str, int]) -> int:
    """Infer the event rank of a let-expression at emit time.

    Leaf variables read their rank from ``ranks``; compound
    expressions propagate it structurally, mirroring the walk the
    Python and Julia helpers perform: operators broadcast to the wider
    operand, elementwise math preserves the rank, a reducing primitive
    collapses to 0, and each index a subscript supplies consumes one
    axis.
    """
    if isinstance(expr, (LetExprLiteral, LetExprString)):
        return 0
    if isinstance(expr, LetExprVar):
        return ranks.get(expr.name, 0)
    if isinstance(expr, LetExprBinOp):
        return max(
            _js_event_rank(expr.left, ranks),
            _js_event_rank(expr.right, ranks),
        )
    if isinstance(expr, LetExprUnaryOp):
        return _js_event_rank(expr.operand, ranks)
    if isinstance(expr, LetExprCall):
        if expr.func in _REDUCTION_FUNCS and len(expr.args) == 1:
            return 0
        return max((_js_event_rank(a, ranks) for a in expr.args), default=0)
    if isinstance(expr, LetExprIndex):
        return max(0, _js_event_rank(expr.array, ranks) - len(expr.indices))
    if isinstance(expr, LetExprList):
        return max((_js_event_rank(i, ranks) for i in expr.items), default=0)
    return 0


def _needs_last_axis(
    expr: LetExprNode, ranks: dict[str, int], full_ranks: dict[str, int]
) -> bool:
    """Whether a reduction's argument needs the last-axis runtime helper.

    Parameters
    ----------
    expr : LetExprNode
        The reduction's argument.
    ranks : dict[str, int]
        Each name's event rank.
    full_ranks : dict[str, int]
        Each name's batch and event axes together.

    Returns
    -------
    bool
        True when the argument has an event axis or two or more axes of
        any kind: WebPPL's own builtin flattens every axis, while the
        reduction collapses only the last.
    """
    return _js_event_rank(expr, ranks) > 0 or let_expr_rank(expr, full_ranks) >= 2


def _reduce_last_axis(
    expr: LetExprNode, ranks: dict[str, int], full_ranks: dict[str, int]
) -> LetExprNode:
    """Rewrite every reduction over a positive-rank argument into the
    matching ``_qvr_<f>_last`` runtime-helper call.

    A reduction whose argument is a flat array keeps WebPPL's own
    builtin: there is only one axis to collapse and the builtin
    already collapses it. A reduction with no runtime-helper
    counterpart over a positive-rank argument raises rather than
    emitting the flattening builtin, which would silently score a
    scalar where the model asks for one value per row.
    """
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_reduce_last_axis(expr.left, ranks, full_ranks),
            right=_reduce_last_axis(expr.right, ranks, full_ranks),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_reduce_last_axis(expr.operand, ranks, full_ranks)
        )
    if isinstance(expr, LetExprCall):
        args = tuple(_reduce_last_axis(a, ranks, full_ranks) for a in expr.args)
        if (
            expr.func in _REDUCTION_FUNCS
            and len(expr.args) == 1
            and _needs_last_axis(expr.args[0], ranks, full_ranks)
        ):
            helper = _LAST_AXIS_REDUCERS.get(expr.func)
            if helper is None:
                raise UnsupportedConstruct(
                    "qvr-webppl",
                    [
                        f"let-expr:axis-reduction:{expr.func}: no "
                        f"WebPPL runtime helper reduces the innermost "
                        f"axis for this primitive, and the builtin "
                        f"flattens every axis instead"
                    ],
                )
            return LetExprCall(func=helper, args=args)
        return LetExprCall(func=expr.func, args=args)
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_reduce_last_axis(expr.array, ranks, full_ranks),
            indices=tuple(
                _reduce_last_axis(i, ranks, full_ranks) for i in expr.indices
            ),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(_reduce_last_axis(i, ranks, full_ranks) for i in expr.items)
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_reduce_last_axis(expr.receiver, ranks, full_ranks),
            method=expr.method,
            args=tuple(_reduce_last_axis(a, ranks, full_ranks) for a in expr.args),
        )
    if isinstance(expr, LetExprLambda):
        return LetExprLambda(
            param=expr.param,
            body=_reduce_last_axis(expr.body, ranks, full_ranks),
        )
    if isinstance(expr, LetExprFactor):
        return LetExprFactor(
            binders=expr.binders,
            body=(
                _reduce_last_axis(expr.body, ranks, full_ranks)
                if expr.body is not None
                else None
            ),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_reduce_last_axis(c.value, ranks, full_ranks),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    return expr


def _ir_has_marginalize(body: tuple[IRNode, ...]) -> bool:
    """True iff the program integrates a latent anywhere, so the emit
    references the `_qvr_logsumexp` / `_qvr_total` runtime helpers the
    atom reduction is written in."""
    return any(isinstance(node, IRMarginalize) for node in body)


def _ir_has_affine_map(body: tuple[IRNode, ...]) -> bool:
    """True iff any deterministic binds an affine parameter map, whose
    contraction the emit spells as the runtime's `_qvr_affine`."""
    for node in body:
        if isinstance(node, IRDeterministic) and isinstance(
            node.expr, LetExprAffineMap
        ):
            return True
        if isinstance(node, IRMarginalize) and _ir_has_affine_map(node.scope):
            return True
    return False


def _ir_reduces_event_axis(
    body: tuple[IRNode, ...], ranks: dict[str, int], full_ranks: dict[str, int]
) -> bool:
    """True iff any deterministic / score body reduces an event axis,
    so the emit references a `_qvr_<f>_last` runtime helper."""
    for node in body:
        if isinstance(node, (IRDeterministic, IRScore)):
            if _let_expr_reduces_event_axis(node.expr, ranks, full_ranks):
                return True
        elif isinstance(node, IRMarginalize):
            if _ir_reduces_event_axis(node.scope, ranks, full_ranks):
                return True
    return False


def _let_expr_reduces_event_axis(
    expr: LetExprNode, ranks: dict[str, int], full_ranks: dict[str, int]
) -> bool:
    """True iff ``expr`` contains a reduction over a positive-rank
    argument anywhere in its tree."""
    if isinstance(expr, LetExprCall):
        if (
            expr.func in _REDUCTION_FUNCS
            and len(expr.args) == 1
            and _needs_last_axis(expr.args[0], ranks, full_ranks)
        ):
            return True
        return any(
            _let_expr_reduces_event_axis(a, ranks, full_ranks) for a in expr.args
        )
    if isinstance(expr, LetExprBinOp):
        return _let_expr_reduces_event_axis(
            expr.left, ranks, full_ranks
        ) or _let_expr_reduces_event_axis(expr.right, ranks, full_ranks)
    if isinstance(expr, LetExprUnaryOp):
        return _let_expr_reduces_event_axis(expr.operand, ranks, full_ranks)
    if isinstance(expr, LetExprIndex):
        return _let_expr_reduces_event_axis(expr.array, ranks, full_ranks) or any(
            _let_expr_reduces_event_axis(i, ranks, full_ranks) for i in expr.indices
        )
    if isinstance(expr, LetExprList):
        return any(
            _let_expr_reduces_event_axis(i, ranks, full_ranks) for i in expr.items
        )
    if isinstance(expr, LetExprLambda):
        return _let_expr_reduces_event_axis(expr.body, ranks, full_ranks)
    if isinstance(expr, LetExprMethodCall):
        return _let_expr_reduces_event_axis(expr.receiver, ranks, full_ranks) or any(
            _let_expr_reduces_event_axis(a, ranks, full_ranks) for a in expr.args
        )
    if isinstance(expr, LetExprFactor):
        if expr.body is not None and _let_expr_reduces_event_axis(
            expr.body, ranks, full_ranks
        ):
            return True
        return any(
            _let_expr_reduces_event_axis(c.value, ranks, full_ranks) for c in expr.cases
        )
    return False


def _vectorize_let_expr(expr: LetExprNode, ranks: Mapping[str, int]) -> LetExprNode:
    """Rewrite every operator in ``expr`` over an array operand into a
    [`_qvr_bcast`][quivers.transpile.runtime_webppl] call.

    An operand is an array when its rank is positive: a name of positive
    rank read at fewer subscripts than its rank, or any expression over
    one. Binary operators become ``_qvr_bcast("<op>", left, right)``;
    the unary minus becomes ``_qvr_bcast("-", 0, operand)``. Operators
    over scalar operands keep their native form, so purely scalar
    bindings emit unchanged.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    ranks : Mapping[str, int]
        The rank of each name the expression may read.

    Returns
    -------
    LetExprNode
        The expression with every array operator broadcast.
    """
    if isinstance(expr, LetExprBinOp):
        left = _vectorize_let_expr(expr.left, ranks)
        right = _vectorize_let_expr(expr.right, ranks)
        if let_expr_rank(expr.left, ranks) > 0 or let_expr_rank(expr.right, ranks) > 0:
            return LetExprCall(
                func="_qvr_bcast",
                args=(LetExprString(value=expr.op), left, right),
            )
        return LetExprBinOp(op=expr.op, left=left, right=right)
    if isinstance(expr, LetExprUnaryOp):
        operand = _vectorize_let_expr(expr.operand, ranks)
        if let_expr_rank(expr.operand, ranks) > 0:
            return LetExprCall(
                func="_qvr_bcast",
                args=(
                    LetExprString(value="-"),
                    LetExprLiteral(value=0.0),
                    operand,
                ),
            )
        return LetExprUnaryOp(operand=operand)
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(_vectorize_let_expr(a, ranks) for a in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_vectorize_let_expr(expr.array, ranks),
            indices=tuple(_vectorize_let_expr(i, ranks) for i in expr.indices),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(_vectorize_let_expr(i, ranks) for i in expr.items)
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_vectorize_let_expr(expr.receiver, ranks),
            method=expr.method,
            args=tuple(_vectorize_let_expr(a, ranks) for a in expr.args),
        )
    return expr


def _ir_emits_qvr_bcast(ir: IRProgram, ranks: Mapping[str, int]) -> bool:
    """True iff a deterministic binding or score in ``ir`` applies an
    operator to an array operand, which the renderer emits as a
    [`_qvr_bcast`][quivers.transpile.runtime_webppl] call.

    Parameters
    ----------
    ir : IRProgram
        The plan.
    ranks : Mapping[str, int]
        The rank of each name the plan binds.

    Returns
    -------
    bool
        Whether the runtime helper must be grafted.
    """
    return _body_emits_qvr_bcast(ir.body, ranks)


#: QVR math primitives the WebPPL runtime supplies as elementwise
#: globals (`Math` has no bare `sigmoid` / `exp` / `log` / `sqrt` /
#: `abs`, and WebPPL no `max` / `min` / `pow`, whose two-argument forms
#: are pointwise, nor the truncating `quot` / `rem`).
_WEBPPL_MATH_HELPERS: frozenset[str] = frozenset(
    {"sigmoid", "exp", "log", "sqrt", "abs", "max", "min", "pow", "quot", "rem"}
)


def _ir_uses_webppl_math(body: tuple[IRNode, ...]) -> bool:
    """True iff any deterministic / score binding (including nested
    marginalize scopes) calls a runtime math primitive that the WebPPL
    stdlib lacks as a global.

    The graft prepends the runtime carrying `sigmoid` / `exp` /
    `log` / `sqrt` / `abs` whenever one of them is called,
    independent of whether the binding also routes through
    `_qvr_bcast`.
    """
    for node in body:
        if isinstance(node, (IRDeterministic, IRScore)) and (
            _let_expr_calls_any(node.expr, _WEBPPL_MATH_HELPERS)
        ):
            return True
        if isinstance(node, IRMarginalize) and _ir_uses_webppl_math(node.scope):
            return True
    return False


def _let_expr_calls_any(expr: LetExprNode, names: frozenset[str]) -> bool:
    """True iff the let-expression tree contains a `LetExprCall` whose
    callee name is in ``names``."""
    if isinstance(expr, LetExprCall):
        if expr.func in names:
            return True
        return any(_let_expr_calls_any(a, names) for a in expr.args)
    if isinstance(expr, LetExprBinOp):
        return _let_expr_calls_any(expr.left, names) or _let_expr_calls_any(
            expr.right, names
        )
    if isinstance(expr, LetExprUnaryOp):
        return _let_expr_calls_any(expr.operand, names)
    if isinstance(expr, LetExprIndex):
        return _let_expr_calls_any(expr.array, names) or any(
            _let_expr_calls_any(i, names) for i in expr.indices
        )
    if isinstance(expr, LetExprList):
        return any(_let_expr_calls_any(i, names) for i in expr.items)
    if isinstance(expr, LetExprMethodCall):
        return _let_expr_calls_any(expr.receiver, names) or any(
            _let_expr_calls_any(a, names) for a in expr.args
        )
    if isinstance(expr, LetExprLambda):
        return _let_expr_calls_any(expr.body, names)
    if isinstance(expr, LetExprFactor):
        if expr.body is not None and _let_expr_calls_any(expr.body, names):
            return True
        return any(_let_expr_calls_any(case.value, names) for case in expr.cases)
    return False


def _body_emits_qvr_bcast(body: tuple[IRNode, ...], ranks: Mapping[str, int]) -> bool:
    """Recursive worker for `_ir_emits_qvr_bcast`, descending into
    marginalize scopes.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The statements.
    ranks : Mapping[str, int]
        The rank of each name the plan binds.

    Returns
    -------
    bool
        Whether any statement's expression has an array operator.
    """
    for node in body:
        if isinstance(node, IRDeterministic | IRScore) and _let_expr_calls_any(
            _vectorize_let_expr(node.expr, ranks), frozenset({"_qvr_bcast"})
        ):
            return True
        if isinstance(node, IRMarginalize) and _body_emits_qvr_bcast(node.scope, ranks):
            return True
    return False


# WebPPL-side argument injection for QVR families whose torch
# distribution carries fewer parameters than WebPPL's same-named
# distribution. `HalfNormal(scale)` maps to WebPPL's
# `Gaussian({mu: 0, sigma: scale})`; `HalfCauchy(scale)` maps to
# `Cauchy({location: 0, scale: scale})`. The renderer prepends an
# explicit zero-valued location argument so the WebPPL distribution
# constructor sees both parameters. The resulting log-density
# differs from QVR's half-distribution log-density by the constant
# ``+log(2) * N_observations``; the constant-spread equivalence
# check in
# [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
# tolerates this offset.
_PREPEND_MU_ZERO: frozenset[str] = frozenset(
    {
        "HalfNormal",
        "HalfCauchy",
    }
)


#: Per-family set of QVR arg names whose value the WebPPL renderer
#: reciprocates (``1 / x``) before emission. WebPPL's
#: ``Gamma({shape, scale})`` is scale-parameterised (``scale =
#: 1/rate``), so the torch Gamma rate must be inverted; without this
#: the emitted density is wrong whenever ``rate != 1``.
_WEBPPL_ARG_RECIPROCAL: dict[str, frozenset[str]] = {
    "Gamma": frozenset({"rate"}),
}


#: Argument slots whose WebPPL parameter type is a tensor rather than a
#: plain JavaScript array, keyed by QVR family name and QVR argument
#: name. WebPPL's `Dirichlet` declares its concentration
#: `positiveVector` and rejects an array outright, so the rendered
#: value is wrapped in the language's own `Vector(...)` constructor.
#: Slots WebPPL declares as vector-or-array (`Discrete`'s `ps`,
#: `Mixture`'s `ps`) stay unwrapped: the array form is accepted there
#: and carries the same measure.
_WEBPPL_TENSOR_ARGS: dict[str, frozenset[str]] = {
    "Dirichlet": frozenset({"concentration"}),
    "MultivariateNormal": frozenset({"loc"}),
}

#: Argument slots whose WebPPL parameter type is a matrix tensor, keyed
#: as `_WEBPPL_TENSOR_ARGS` is: `MultivariateGaussian` declares its
#: covariance a positive-definite matrix and rejects nested arrays, so
#: the rendered value is wrapped in WebPPL's `Matrix(...)` constructor.
_WEBPPL_MATRIX_ARGS: dict[str, frozenset[str]] = {
    "MultivariateNormal": frozenset({"covariance_matrix"}),
}


def _inject_webppl_specific_args(
    family: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Inject the canonical extra arguments required by WebPPL for
    families whose torch shape is narrower than WebPPL's call shape.

    Returns the possibly-augmented (args, arg_names) tuple in
    parallel order. The injected keyword name uses the QVR-side
    name (e.g. ``loc``) so the family's ``arg_aliases["webppl"]`` map
    can rename it to the WebPPL-side keyword (``mu``) during
    rendering.
    """
    if family in _PREPEND_MU_ZERO:
        return (
            (IRArgNumber(value=0.0), *args),
            ("loc", *arg_names),
        )
    return args, arg_names


# ---------------------------------------------------------------------------
# Runtime-helper graft: `Logistic`, `BetaBinomial`, `HalfStudentT`,
# `Kumaraswamy`, `LKJCholesky`, and `ContinuousBernoulli` as plain
# JavaScript distribution constructors.
#
# WebPPL's `dists` module ships `Gaussian`, `Beta`, `Categorical`,
# `Dirichlet`, ... as built-in distributions but lacks these six.
# The transpile-time graft parses the hand-written helper at
# [`runtime_webppl.js`][quivers.transpile.runtime_webppl] once at
# module-load through panproto's JavaScript tree-sitter grammar;
# per-render, it copies every grafted vertex / constraint / edge into
# the per-render schema (with fresh vertex ids) and attaches the
# runtime's top-level statements as `child_of` of the emitted
# `program` vertex above the `var model = function (...) {...};`
# declaration.
#
# The emit is structurally a normal JavaScript module: a few
# numeric utility functions (`_lgamma`, `_lbeta`, `_gaussian_sample`,
# `_gamma_sample`, `_beta_sample`, `_binomial_sample`) and the six
# distribution constructors. Subsequent
# `sample(Logistic({loc, scale}))` etc. call sites in the model body
# then resolve to the grafted constructors via normal JS name lookup.
# ---------------------------------------------------------------------------


_RUNTIME_WEBPPL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_webppl.js"
)


#: Families whose WebPPL constructor rejects a parameter value the
#: reference measure admits, paired with the runtime helper that
#: scores the parameter object directly. WebPPL's `Poisson` refuses a
#: rate of exactly 0, which is the boundary a zero-inflation
#: indicator pinned to 0 gates its rate to; there the distribution is
#: the point mass at 0 and the reference scores it as such.
_WEBPPL_BOUNDARY_SAFE_SCORERS: dict[str, str] = {
    "Poisson": "_qvr_poisson_score",
}


#: Families whose WebPPL emit relies on the
#: [`runtime_webppl.js`][quivers.transpile.runtime_webppl] helper subtree.
#: WebPPL's `dists` module ships `Gaussian`, `Beta`, `Categorical`, etc.
#: as built-in distributions but lacks these; the renderer grafts the
#: helper when the IR samples or observes from any of them.
_WEBPPL_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset(
    {
        "Logistic",
        "BetaBinomial",
        "HalfStudentT",
        "Kumaraswamy",
        "LKJCholesky",
        "ContinuousBernoulli",
        "GP",
        "MatrixNormal",
        "LogNormal",
        "StudentT",
        "Weibull",
        "NegativeBinomial",
    }
)


def _load_runtime_webppl_schema() -> tuple[panproto.Schema, str, tuple[str, ...]]:
    """Parse [`runtime_webppl.js`][quivers.transpile.runtime_webppl]
    through panproto's JavaScript tree-sitter grammar at module-load
    time.

    Returns the parsed schema, the parsed `program` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `program` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "javascript",
        _RUNTIME_WEBPPL_PATH.read_bytes(),
        str(_RUNTIME_WEBPPL_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "program"),
        None,
    )
    if src_id is None:
        raise RuntimeError(f"`program` not found in parse of {_RUNTIME_WEBPPL_PATH}")
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


_RUNTIME_WEBPPL_SCHEMA, _RUNTIME_WEBPPL_PROGRAM_ID, _RUNTIME_WEBPPL_TOP_LEVEL = (
    _load_runtime_webppl_schema()
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


_RUNTIME_WEBPPL_SUBTREE = _subtree_vertex_ids(
    _RUNTIME_WEBPPL_SCHEMA, _RUNTIME_WEBPPL_TOP_LEVEL
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


def _graft_runtime_webppl_helper(
    sb: panproto.SchemaBuilder,
    renderer: WebPPLRenderer,
    program_vid: str,
) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_webppl.js` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each
    top-level child as a `child_of` of `program_vid` in source order.
    The grafted top-level children appear above the `var model =
    function(...)` declaration in the emit.
    """
    src_schema = _RUNTIME_WEBPPL_SCHEMA
    subtree = _RUNTIME_WEBPPL_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        renderer._fresh_n += 1
        new = f"rw_{renderer._fresh_n}"
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_WEBPPL_TOP_LEVEL:
        sb.edge(program_vid, id_map[child_old], "child_of")


__all__ = ["WebPPLRenderer"]
