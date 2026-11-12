"""Render transpilation IR as a Pyro model function.

Samples and observations use ``pyro.sample`` inside nested plates.
Finite marginalizations score each atom and add their ``logsumexp``
with ``pyro.factor``. Family names and argument aliases come from
``FAMILY_META``.
"""

from __future__ import annotations

import pathlib

import panproto

from quivers.dsl.ast_nodes import MorphismInitFamily
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.renderers._ranks import carries_event_dims, name_rank_map
from quivers.transpile.renderers._python_helpers import (
    MarginalizeBody,
    PyCtx,
    assignment,
    attribute,
    call,
    factor_tower_names,
    float_literal,
    identifier,
    marginal_support_size,
    marginal_weight_probs,
    marginalize_body,
    marginalize_fibration,
    name_event_rank_map,
    name_plate_map,
    number_literal,
    python_binary_op as _python_binary_op,
    python_list,
    python_method_call as _python_method_call,
    python_paren as _python_paren,
    python_unary_minus as _python_unary_minus,
    render_deterministic_python,
    render_let_expr_python,
    shape_tuple,
    string_literal,
    with_statement,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile.ir import (
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
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRCall,
    IRScore,
    Plate,
    StructuredDataArg,
)
from quivers.transpile.renderers._base import (
    marginalize_row_rank,
    BlockKind,
    IRMarginalAtom,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dropped_param_map,
    host_integer_input_names,
    mixture_normal_components,
)
from quivers.transpile.qiec_ir import IRQiecModule
from quivers.transpile.renderers._qiec import (
    emit_call_python,
    graft_python_statements,
    qiec_helper_roots,
    render_computations_dynamic,
)


_TARGET = "pyro"


class PyroRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    Python source `panproto.Schema` under the
    Pyro PPL idiom.
    """

    target: str = _TARGET

    # ----- panproto plumbing -----

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("python")

    # ----- top-level render -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a Pyro `def model(...): ...` function.

        Overrides the inherited walk so the function header (with
        `<observed>=None` default parameters drawn from the IR's
        `IRObserve` nodes) and the body block are constructed up
        front; the inherited dispatch then routes each node into
        the body.

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
        ctx = RenderContext(sb=sb, morphisms={}, defines={})
        pctx = _PyroCtx(
            sb=sb,
            cards=dict(ir.cards),
            name_event_rank=name_event_rank_map(ir),
            name_rank=name_rank_map(ir),
            factor_towers=factor_tower_names(ir),
            name_plates=name_plate_map(ir),
        )

        # Resolve module-level morphism / let tables for IRArgFamilyRef
        # lookup. The IR carries family names directly for atomic
        # families; wrappers like `Truncated(base, ...)` carry the
        # `base` reference via IRArgFamilyRef and the renderer looks
        # the wrapped family up via the resolved init_family.
        # (The IR program does not carry the module; renderers either
        # receive an externally-built morphism table or rebuild from
        # the IR. We rebuild from `IRSample`s the IR exposes; opaque
        # wrappers requiring deeper resolution surface via
        # IRArgFamilyRef and we error if the wrapped family cannot be
        # resolved at emit time.)
        pctx.morphisms = ctx.morphisms

        # Gather observed names so the function signature carries
        # `<obs>=None` defaults.
        observed_names = _observed_names(ir.body)
        # Function header: positional params are every IRDataInput
        # that is not observed; default params are the observed ones.
        param_names = tuple(
            inp.name for inp in ir.inputs if inp.name not in observed_names
        )
        default_params = tuple(
            inp.name for inp in ir.inputs if inp.name in observed_names
        )

        pctx.v("mod", "module")
        # Pyro lacks several QVR families as built-ins (`TruncatedNormal`,
        # `LogitNormal`, `HalfStudentT`, `MatrixNormal`,
        # `InverseWishart`); their runtime helpers live at
        # [`quivers.transpile.runtime_pyro`][quivers.transpile.runtime_pyro].
        # When the IR samples or observes from such a family, graft the
        # parsed helper `class` subtree onto the module above `model`
        # so a reader sees the helper classes first (the natural Python
        # idiom: define classes before consumers).
        for helper_name in sorted(
            _ir_helper_classes_used(ir.body) | qiec_helper_roots(ir, _TARGET)
        ):
            _emit_runtime_helper(pctx, helper_name)
        body = pctx.v(pctx.fresh("body"), "block")
        func = _function_def_split(
            pctx,
            name="model",
            positional=param_names,
            defaults=default_params,
            body_vid=body,
        )
        pctx.e("mod", func, "child_of")

        pctx.body = body
        pctx.observed = frozenset(observed_names)
        pctx.module = ir.module
        if not ir.body:
            noop = pctx.v(pctx.fresh("pass"), "pass_statement")
            pctx.e(body, noop, "child_of")

        # torch's advanced indexing requires integer index tensors, so
        # every host-integer input the program subscripts with is
        # coerced to `long` at the top of the body. QVR declares these
        # gather covariates as integers; the coercion makes the model
        # self-contained regardless of the caller's supplied dtype.
        for idx_name in sorted(_integer_index_input_names(ir)):
            coerce = assignment(
                pctx,
                lhs_name=idx_name,
                rhs=_python_method_call(
                    pctx,
                    identifier(pctx, idx_name),
                    "long",
                    (),
                ),
            )
            pctx.e(pctx.body, coerce, "child_of")

        # Hoist each axis that backs more than one sample / observe site
        # to a single reused `pyro.plate(...)` object.
        for dim in _shared_plate_dims(ir.body):
            axis = str(dim.name)
            var_name = f"{axis}_plate"
            pctx.shared_plate_vars[axis] = var_name
            pctx.e(
                pctx.body,
                assignment(
                    pctx,
                    lhs_name=var_name,
                    rhs=self._plate_object_call(pctx, dim),
                ),
                "child_of",
            )

        for node in ir.body:
            self._dispatch_pyro_node(pctx, ctx, node)

        render_computations_dynamic(sb, ir, target=self.target, root="mod")
        return sb.build()

    # ----- per-node dispatch driving pctx body emission -----

    def _dispatch_pyro_node(
        self,
        pctx: _PyroCtx,
        ctx: RenderContext,
        node: IRNode,
    ) -> None:
        if isinstance(node, IRDataInput):
            # Function signature carries the data input; nothing in
            # the body.
            self.declare(ctx, node.name, node.constraint, node.plate, block="data")
            return
        if isinstance(node, IRSample):
            self._emit_sample_or_observe(
                pctx,
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=False,
            )
            return
        if isinstance(node, IRObserve):
            self._emit_sample_or_observe(
                pctx,
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=True,
            )
            return
        if isinstance(node, IRDeterministic):
            asn = pctx.v(pctx.fresh("asn"), "assignment")
            lhs = identifier(pctx, node.name)
            pctx.e(asn, lhs, "left")
            pctx.e(asn, render_deterministic_python(pctx, node), "right")
            pctx.e(pctx.body, asn, "child_of")
            return
        if isinstance(node, IRScore):
            self._emit_score_pyro(pctx, node)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize_pyro(pctx, node)
            return
        if isinstance(node, IRCall):
            self._emit_call_pyro(pctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_return_pyro(pctx, node.names)
            return
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    def _emit_call_pyro(self, pctx: _PyroCtx, node: IRCall) -> None:
        """``<name> = qiec_<callee>(<args>, qiec_operations=<table>)``.

        Parameters
        ----------
        pctx : _PyroCtx
            The emission context.
        node : IRCall
            The call.
        """
        assert pctx.module is not None
        emit_call_python(pctx, pctx.body, node, pctx.module, pctx.operations_bound)

    # ----- declare: no-op outside `"function_body"` -----

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op outside `"function_body"`: Pyro picks data inputs up
        in the model function signature, not via a declaration block.

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

    # ----- sample / observe -----

    def sample(
        self,
        ctx: RenderContext,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint,
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        """Build the `pyro.sample(...)` call; the caller threads the
        return vertex into the enclosing `with` / block.

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
        del constraint, plate
        return self._build_sample_call(
            ctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            observed=observed,
        )

    def _emit_gp_block(
        self,
        pctx: _PyroCtx,
        *,
        name: str,
        args: tuple[IRArg, ...],
        observed: bool,
    ) -> None:
        """Emit a Gaussian-process sample as a triple of Python
        statements: ``__gp_mean_<name> = torch.zeros(N)``,
        ``__gp_cov_<name> = torch.exp(...) + jitter * torch.eye(N)``,
        ``<name> = pyro.sample("<name>", pyro.distributions.MultivariateNormal(
            loc=__gp_mean_<name>, covariance_matrix=__gp_cov_<name>))``.

        Mirrors the NumPyro emission with `torch` substituted for
        `jnp`. The kernel-cov expression is built using the Python
        helper API plus explicit
        `parenthesized_expression`
        wrappers so operator precedence around the squared diff and
        the squared length scale survives the printer's drop-paren
        default.
        """
        del observed  # GP samples are never observed in QVR.
        if len(args) != 2 or not isinstance(args[1], IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-pyro",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-pyro",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{name}"
        cov_name = f"__gp_cov_{name}"
        # __gp_mean_<name> = torch.zeros(N)
        mean_rhs = call(
            pctx,
            attribute(pctx, ("torch", "zeros")),
            positional=(number_literal(pctx, n),),
        )
        pctx.e(
            pctx.body,
            assignment(pctx, lhs_name=mean_name, rhs=mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = torch.exp(-0.5 * (diff * diff) / (ls * ls))
        #                    + jitter * torch.eye(N)
        x_col = _python_method_call(
            pctx,
            identifier(pctx, x),
            "reshape",
            (
                _python_unary_minus(pctx, number_literal(pctx, 1)),
                number_literal(pctx, 1),
            ),
        )
        x_row = _python_method_call(
            pctx,
            identifier(pctx, x),
            "reshape",
            (
                number_literal(pctx, 1),
                _python_unary_minus(pctx, number_literal(pctx, 1)),
            ),
        )
        diff = _python_paren(
            pctx,
            _python_binary_op(pctx, "-", x_col, x_row),
        )
        diff_sq = _python_paren(
            pctx,
            _python_binary_op(pctx, "*", diff, diff),
        )
        ls_sq = _python_paren(
            pctx,
            _python_binary_op(
                pctx,
                "*",
                number_literal(pctx, ls),
                number_literal(pctx, ls),
            ),
        )
        quotient = _python_binary_op(pctx, "/", diff_sq, ls_sq)
        neg_half = _python_unary_minus(pctx, number_literal(pctx, 0.5))
        exponent = _python_binary_op(pctx, "*", neg_half, quotient)
        kernel_call = call(
            pctx,
            attribute(pctx, ("torch", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            pctx,
            attribute(pctx, ("torch", "eye")),
            positional=(number_literal(pctx, n),),
        )
        jitter_term = _python_binary_op(
            pctx,
            "*",
            number_literal(pctx, jitter),
            eye_call,
        )
        cov_rhs = _python_binary_op(
            pctx,
            "+",
            kernel_call,
            jitter_term,
        )
        pctx.e(
            pctx.body,
            assignment(pctx, lhs_name=cov_name, rhs=cov_rhs),
            "child_of",
        )
        # <name> = pyro.sample("<name>", pyro.distributions.MultivariateNormal(
        #     loc=__gp_mean_<name>, covariance_matrix=__gp_cov_<name>))
        mvn_call = call(
            pctx,
            attribute(
                pctx,
                ("pyro", "distributions", "MultivariateNormal"),
            ),
            keyword=(
                ("loc", identifier(pctx, mean_name)),
                ("covariance_matrix", identifier(pctx, cov_name)),
            ),
        )
        sample_call = call(
            pctx,
            attribute(pctx, ("pyro", "sample")),
            positional=(string_literal(pctx, name), mvn_call),
        )
        sample_asn = assignment(
            pctx,
            lhs_name=name,
            rhs=sample_call,
        )
        pctx.e(pctx.body, sample_asn, "child_of")

    def _emit_sample_or_observe(
        self,
        pctx: _PyroCtx,
        ctx: RenderContext,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        observed: bool,
    ) -> None:
        """Emit `with pyro.plate(<dim>, <size>):` per batch dim,
        wrapping a `pyro.sample(...)` call (optionally assigned to
        `name` for latents)."""
        if family == "GP":
            self._emit_gp_block(
                pctx,
                name=name,
                args=args,
                observed=observed,
            )
            return
        sample_call = self._build_sample_call(
            ctx,
            pctx=pctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            observed=observed,
            plate=plate,
        )
        if observed:
            inner_stmt = sample_call
        else:
            inner_stmt = assignment(pctx, lhs_name=name, rhs=sample_call)

        # Nest one `with pyro.plate(...)` per batch dim. If no batch
        # dims, the call lives directly inside the function body.
        target_block = pctx.body
        if not plate.batch_dims:
            pctx.e(target_block, inner_stmt, "child_of")
            return
        # Build innermost-first by walking the dims in reverse: the
        # innermost block holds the sample, the outer blocks own each
        # inner `with` as their sole child. Pyro gives a plate with no
        # stated `dim` the rightmost free batch axis, so the outermost
        # of several nested plates would take the last axis and the
        # draw would come out transposed; each plate of a nest of
        # several states its axis, outermost leftmost.
        current_stmt = inner_stmt
        count = len(plate.batch_dims)
        for position in reversed(range(count)):
            dim = plate.batch_dims[position]
            inner_block = pctx.v(pctx.fresh("blk"), "block")
            pctx.e(inner_block, current_stmt, "child_of")
            plate_call = (
                self._plate_call(pctx, dim, name=name, observed=observed)
                if count == 1
                else self._plate_object_call(pctx, dim, axis=position - count)
            )
            current_stmt = with_statement(
                pctx,
                expression=plate_call,
                alias=None,
                body_vid=inner_block,
            )
        pctx.e(target_block, current_stmt, "child_of")

    def _plate_call(
        self,
        pctx: _PyroCtx,
        dim: Dim,
        *,
        name: str,
        observed: bool,
    ) -> str:
        """Build the `with`-expression for one batch dim.

        A shared axis (used by more than one site) resolves to the
        hoisted `<axis>_plate` variable so every `with` reuses the one
        plate object; every other axis emits an inline
        `pyro.plate(<name>, <size>)`.
        """
        del observed, name
        shared_var = pctx.shared_plate_vars.get(str(dim.name))
        if shared_var is not None:
            return identifier(pctx, shared_var)
        return self._plate_object_call(pctx, dim)

    def _plate_object_call(
        self, pctx: _PyroCtx, dim: Dim, *, axis: int | None = None
    ) -> str:
        """Build the `pyro.plate(<name>, <size>)` constructor call.

        Parameters
        ----------
        pctx : _PyroCtx
            The render state.
        dim : Dim
            The plate's axis.
        axis : int | None
            The negative batch axis the plate occupies, stated as its
            `dim`; `None` leaves Pyro to allocate it.

        Returns
        -------
        str
            The call's vertex.

        Raises
        ------
        UnsupportedConstruct
            If the axis is neither static nor dynamic.
        """
        plate_callee = attribute(pctx, ("pyro", "plate"))
        if isinstance(dim, DimStatic):
            plate_name = string_literal(pctx, dim.name)
            size_arg = number_literal(pctx, float(dim.size))
        elif isinstance(dim, DimDynamic):
            plate_name = string_literal(pctx, dim.name)
            # Dynamic plate size: `<size_name>` is the canonical
            # data-bound size identifier (e.g. `N_w`). Renderers that
            # want `<observed>.shape[0]` can post-process; we emit the
            # documented bound name from the IR.
            size_arg = identifier(pctx, dim.size_name)
        else:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"dim-kind:{type(dim).__name__}"],
            )
        keyword: tuple[tuple[str, str], ...] = ()
        if axis is not None:
            keyword = (
                ("dim", _python_unary_minus(pctx, number_literal(pctx, float(-axis)))),
            )
        return call(
            pctx,
            plate_callee,
            positional=(plate_name, size_arg),
            keyword=keyword,
        )

    def _build_sample_call(
        self,
        ctx: RenderContext,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        observed: bool,
        plate: Plate | None = None,
        pctx: _PyroCtx | None = None,
    ) -> str:
        """Build `pyro.sample("<name>", pyro.distributions.<Family>(
        <args>)[, obs=<name>])`."""
        if pctx is None:
            # Allow `sample` to run standalone (Protocol contract):
            # build a throwaway PyCtx-compatible view over the same
            # SchemaBuilder. The renderer always drives through
            # `_emit_sample_or_observe` for real emission.
            pctx = _PyroCtx(sb=ctx.sb)
        dist_call = self._build_distribution_call(
            pctx,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
        )
        sample_callee = attribute(pctx, ("pyro", "sample"))
        positional = (string_literal(pctx, name), dist_call)
        keyword: tuple[tuple[str, str], ...] = ()
        if observed:
            keyword = (("obs", identifier(pctx, name)),)
        return call(
            pctx,
            sample_callee,
            positional=positional,
            keyword=keyword,
        )

    def _build_distribution_call(
        self,
        pctx: _PyroCtx,
        *,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate | None = None,
    ) -> str:
        """Build the bare `pyro.distributions.<Family>(<args>)` call,
        with the residual user-declared event dims already lifted."""
        meta = self._family_meta(family)
        dist_class = meta.target_names.get(_TARGET)
        if dist_class is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}"],
            )
        if family == "MixtureNormal":
            return self._mixture_same_family_call(pctx, dist_class, args, arg_names)
        aliases = {
            **meta.arg_aliases.get(_TARGET, {}),
            **_PYRO_KEYWORD_BINDINGS.get(family, {}),
        }
        # Build the distribution call. Pyro lacks several QVR families
        # as built-ins (`TruncatedNormal`, `LogitNormal`,
        # `HalfStudentT`, `MatrixNormal`, `InverseWishart`); the
        # renderer grafts a helper
        # class from `quivers.transpile.runtime_pyro` (see
        # `_emit_runtime_helper`)
        # and dispatches the call to that bare identifier rather than
        # to `pyro.distributions.<name>`.
        if dist_class in _RUNTIME_PYRO_HELPER_ROOTS:
            dist_callee = identifier(pctx, dist_class)
        else:
            dist_callee = attribute(pctx, ("pyro", "distributions", dist_class))
        dist_args = self._build_dist_args(
            pctx,
            meta=meta,
            args=args,
            arg_names=arg_names,
            aliases=aliases,
            plate=plate,
        )
        positional = dist_args.positional
        # The Cholesky-LKJ families take the correlation-matrix
        # dimension as a mandatory leading positional argument
        # (`LKJCorrCholesky(d, eta)`, `LKJ(d, eta)`); it is carried on
        # the sample's event axis, not in the QVR init clause.
        if family in _PYRO_LEADING_DIM_FAMILIES:
            positional = (
                self._leading_dim_arg(pctx, family, plate),
                *positional,
            )
        # A family whose target class fixes a leading parameter the QVR
        # call site never writes gets that value prepended under the
        # target's own parameter name.
        fixed_leading = _PYRO_FIXED_LEADING_ARGS.get(family)
        if fixed_leading is not None:
            positional = (
                number_literal(pctx, fixed_leading),
                *positional,
            )
        dist_call = call(
            pctx,
            dist_callee,
            positional=positional,
            keyword=dist_args.keyword,
        )
        event_dims = plate.event_dims if plate is not None else ()
        return self._wrap_event_dims(pctx, dist_call, family, event_dims, args)

    def _mixture_same_family_call(
        self,
        pctx: _PyroCtx,
        dist_class: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """Emit `MixtureSameFamily(Categorical(probs), Normal(loc, scale))`.

        Pyro ships no single-name Gaussian-mixture family, but it ships
        the mixture *construction*: a categorical mixing distribution
        over a batch of components whose batch axis is the component
        axis. `MixtureNormal(w, mu, sigma)` maps onto it exactly, so the
        emitted distribution's `log_prob` is the same log-sum-exp the
        QVR likelihood scores rather than an approximation of it.

        The component axis is the mixture's own, never a plate axis, so
        the call takes no `.expand(...).to_event(...)` lift.
        """
        weights, loc, scale = mixture_normal_components(_TARGET, args, arg_names)
        meta = FAMILY_META["MixtureNormal"]
        mixing = call(
            pctx,
            attribute(pctx, ("pyro", "distributions", "Categorical")),
            positional=(self._lower_arg(pctx, weights, meta=meta, arg_name="weights"),),
        )
        component = call(
            pctx,
            attribute(pctx, ("pyro", "distributions", "Normal")),
            positional=(
                self._lower_arg(pctx, loc, meta=meta, arg_name="loc"),
                self._lower_arg(pctx, scale, meta=meta, arg_name="scale"),
            ),
        )
        return call(
            pctx,
            attribute(pctx, ("pyro", "distributions", dist_class)),
            positional=(mixing, component),
        )

    def _wrap_event_dims(
        self,
        pctx: _PyroCtx,
        dist_call: str,
        family: str,
        event_dims: tuple[Dim, ...],
        args: tuple[IRArg, ...],
    ) -> str:
        """Lift the residual user-declared event dims into the
        distribution via ``.expand([sizes]).to_event(n)``.

        Reads
        [`FamilyMeta.event_rank`][quivers.transpile.family_meta.FamilyMeta]
        (0 scalar, 1 vector, 2 matrix) and lifts only the dims beyond
        the family's natural rank. A matrix-plate sample of a scalar
        family (`Normal(0, 1) [over=LatentDim, iid_over=Item]`) lifts
        the `LatentDim` axis into the distribution; a vector family
        whose `[over=Axis]` matches its natural event dim gets no lift.
        An argument that already carries the lifted axes (a gathered
        matrix row) gives the family those axes as batch, so only
        `.to_event(n)` is applied.

        Parameters
        ----------
        pctx : _PyroCtx
            The render state.
        dist_call : str
            The family call's vertex.
        family : str
            The family.
        event_dims : tuple[Dim, ...]
            The site's event axes.
        args : tuple[IRArg, ...]
            The family's arguments.

        Returns
        -------
        str
            The lifted call's vertex.
        """
        if not event_dims:
            return dist_call
        meta = FAMILY_META.get(family)
        natural = meta.event_rank if meta is not None else 0
        residual = event_dims[: len(event_dims) - natural] if natural else event_dims
        if not residual:
            return dist_call
        if carries_event_dims(args, residual, pctx.name_plates):
            # An argument already carries the lifted axes (a gathered
            # row of a matrix, say): the family's batch holds them, and
            # only their reading as event axes remains.
            return _python_method_call(
                pctx, dist_call, "to_event", (number_literal(pctx, len(residual)),)
            )
        list_vid = pctx.v(pctx.fresh("list"), "list")
        for dim in residual:
            pctx.e(list_vid, self._dim_size_vid(pctx, dim), "child_of")
        expand_vid = _python_method_call(pctx, dist_call, "expand", (list_vid,))
        return _python_method_call(
            pctx,
            expand_vid,
            "to_event",
            (number_literal(pctx, len(residual)),),
        )

    def _dim_size_vid(self, pctx: _PyroCtx, dim: Dim) -> str:
        """Render `dim.size` as a Pyro source-token vid."""
        if isinstance(dim, DimStatic):
            return number_literal(pctx, dim.size)
        if isinstance(dim, DimDynamic):
            return identifier(pctx, dim.size_name)
        raise UnsupportedConstruct(
            f"qvr-{_TARGET}",
            [f"plate:dim-kind:{type(dim).__name__}"],
        )

    # ----- marginalize: the integrated-density lowering -----

    def marginalize(self, ctx: RenderContext, node: IRMarginalize) -> SchemaFragment:
        """Emit nothing: Pyro integrates latents from its own walk.

        [`render`][quivers.transpile.renderers.PyroRenderer.render] routes
        each marginalize block through the Pyro walk, which carries the
        function-body state this dispatch point does not receive, so the
        shared dispatch point contributes no fragment of its own.

        Parameters
        ----------
        ctx
            The render call's mutable state.
        node
            The marginalize block.

        Returns
        -------
        SchemaFragment
            The empty fragment.
        """
        del ctx, node
        return ""

    def _emit_marginalize_pyro(self, pctx: _PyroCtx, node: IRMarginalize) -> None:
        """Integrate the latent out and add the reduced density to the
        model's log-density with ``pyro.factor``.

        One scored copy of the scope is emitted per atom of the
        latent's finite support, the atom's own deterministic bindings
        re-emitted under their own names immediately before it, and the
        per-atom log-densities reduced across a fresh trailing axis:

        ```python
        __marg_z_w = torch.stack([torch.log1p(-pi_z), torch.log(pi_z)], dim=-1)
        __marg_z = torch.stack([__marg_z_0, __marg_z_1], dim=-1)
        pyro.factor("z", torch.sum(torch.logsumexp(__marg_z_w + __marg_z, dim=-1)))
        ```

        No site is declared for the latent: the atoms replace it, so
        the emitted program denotes the measure the QVR reference
        integrates rather than the larger product measure a live draw
        would denote.
        """
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        prefix = f"__marg_{node.latent}"
        term_names: list[str] = []
        for position, atom in enumerate(atoms):
            scored = marginalize_body(
                atom.scope, latent=node.latent, target=self.target
            )
            for det in scored.deterministics:
                asn = assignment(
                    pctx,
                    lhs_name=det.name,
                    rhs=render_deterministic_python(pctx, det),
                )
                pctx.e(pctx.body, asn, "child_of")
            dist_call = self._build_distribution_call(
                pctx,
                family=scored.observe.family,
                args=scored.observe.args,
                arg_names=scored.observe.arg_names,
                plate=scored.observe.plate,
            )
            term = f"{prefix}_{position}"
            pctx.e(
                pctx.body,
                assignment(
                    pctx,
                    lhs_name=term,
                    rhs=_python_method_call(
                        pctx,
                        dist_call,
                        "log_prob",
                        (identifier(pctx, scored.observe.name),),
                    ),
                ),
                "child_of",
            )
            term_names.append(term)
        pctx.e(
            pctx.body,
            assignment(
                pctx,
                lhs_name=f"{prefix}_w",
                rhs=self._marginal_log_weights(pctx, node, raw, atoms),
            ),
            "child_of",
        )
        pctx.e(
            pctx.body,
            assignment(
                pctx,
                lhs_name=prefix,
                rhs=_stack_last_dim(
                    pctx,
                    tuple(identifier(pctx, name) for name in term_names),
                ),
            ),
            "child_of",
        )
        # A grouped block keys its accumulator by group: the rows the
        # fibration sends to one group are summed before the reduction.
        fibration = marginalize_fibration(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=pctx.name_plates,
            target=self.target,
        )
        if fibration is not None:
            via, group = fibration
            extent = (
                str(group.size) if isinstance(group, DimStatic) else group.size_name
            )
            graft_python_statements(
                pctx.builder,
                f"{prefix} = torch.zeros(({extent}, {prefix}.shape[-1]), "
                f"dtype={prefix}.dtype).index_add(0, {via}, {prefix})\n",
                pctx.body,
                f"marg_{node.latent}_group",
            )
        # An ungrouped block shares one latent across the body's rows,
        # so their per-class log-likelihoods are accumulated before the
        # reduction rather than each row reducing on its own.
        atoms = identifier(pctx, prefix)
        for _ in range(marginalize_row_rank(node)):
            atoms = call(
                pctx,
                attribute(pctx, ("torch", "sum")),
                positional=(atoms,),
                keyword=(("dim", number_literal(pctx, 0)),),
            )
        reduced = call(
            pctx,
            attribute(pctx, ("torch", "logsumexp")),
            positional=(
                _python_binary_op(
                    pctx,
                    "+",
                    identifier(pctx, f"{prefix}_w"),
                    atoms,
                ),
            ),
            keyword=(("dim", _minus_one(pctx)),),
        )
        total = call(
            pctx,
            attribute(pctx, ("torch", "sum")),
            positional=(reduced,),
        )
        factor_call = call(
            pctx,
            attribute(pctx, ("pyro", "factor")),
            positional=(string_literal(pctx, node.latent), total),
        )
        es = pctx.v(pctx.fresh("es"), "expression_statement")
        pctx.e(es, factor_call, "child_of")
        pctx.e(pctx.body, es, "child_of")

    def _marginal_log_weights(
        self,
        pctx: _PyroCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
    ) -> str:
        """Log-weight tensor whose trailing axis runs over the atoms.

        A `Categorical` atom set weights atom ``k`` by ``log p[k]``, so
        the probability tensor's own trailing axis is already the atom
        axis. A `Bernoulli` atom set weights the atoms 0 and 1 by
        ``log1p(-p)`` and ``log(p)``, which stack into a fresh trailing
        axis.
        """
        probs = marginal_weight_probs(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=pctx.name_plates,
            target=self.target,
        )
        probs_vid = self._arg_to_vid(pctx, probs)
        family = atoms[0].weight_family
        if family == "Categorical":
            return call(
                pctx,
                attribute(pctx, ("torch", "log")),
                positional=(probs_vid,),
            )
        if family == "Bernoulli":
            complement = call(
                pctx,
                attribute(pctx, ("torch", "log1p")),
                positional=(_python_unary_minus(pctx, self._arg_to_vid(pctx, probs)),),
            )
            positive = call(
                pctx,
                attribute(pctx, ("torch", "log")),
                positional=(probs_vid,),
            )
            return _stack_last_dim(pctx, (complement, positive))
        raise UnsupportedConstruct(
            f"qvr-{_TARGET}",
            [f"marginalize:weight-family:{family}"],
        )

    # ----- broadcast / list / matrix -----

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit `torch.full((K,), <value>)` for rank-1, `torch.full(
        (R, C), <value>)` for rank-2.

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
        pctx = _PyroCtx(sb=ctx.sb)
        return self._broadcast(pctx, value, target_shape)

    def _broadcast(
        self,
        pctx: _PyroCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> str:
        if len(target_shape) not in (1, 2):
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"broadcast:rank-{len(target_shape)}"],
            )
        shape_vid = shape_tuple(pctx, target_shape)
        # The fill value types the tensor: an integer literal here
        # builds an integer concentration / rate / scale, which the
        # real-valued family rejects.
        value_vid = (
            float_literal(pctx, value.value)
            if isinstance(value, IRArgNumber)
            else self._arg_to_vid(pctx, value)
        )
        full_callee = attribute(pctx, ("torch", "full"))
        return call(
            pctx,
            full_callee,
            positional=(shape_vid, value_vid),
        )

    def _render_list(
        self,
        pctx: _PyroCtx,
        arg: IRArgList,
    ) -> str:
        """Emit `torch.tensor([e0, e1, ...])` for an
        [`IRArgList`][quivers.transpile.ir.IRArgList]."""
        list_vid = pctx.v(pctx.fresh("lst"), "list")
        for elem in arg.elements:
            pctx.e(list_vid, self._arg_to_vid(pctx, elem), "child_of")
        tensor_callee = attribute(pctx, ("torch", "tensor"))
        return call(pctx, tensor_callee, positional=(list_vid,))

    def _render_matrix(
        self,
        pctx: _PyroCtx,
        arg: IRArgMatrix,
    ) -> str:
        """Emit `torch.tensor([[...], [...]])` for an
        [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix]."""
        outer = pctx.v(pctx.fresh("mat"), "list")
        for row in arg.rows:
            row_vid = pctx.v(pctx.fresh("row"), "list")
            for elem in row.elements:
                pctx.e(row_vid, self._arg_to_vid(pctx, elem), "child_of")
            pctx.e(outer, row_vid, "child_of")
        tensor_callee = attribute(pctx, ("torch", "tensor"))
        return call(pctx, tensor_callee, positional=(outer,))

    # ----- arg lowering -----

    def _build_dist_args(
        self,
        pctx: _PyroCtx,
        *,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        aliases: dict[str, str],
        plate: Plate | None = None,
    ) -> _DistArgs:
        """Build the (positional, keyword) split for the distribution
        call.

        Positional args carry every IR arg that needs neither alias
        renaming nor name disambiguation; keyword args carry every
        position whose `arg_name` has been renamed by `arg_aliases`.
        Scalar arguments to vector-typed slots are auto-broadcast via
        `torch.full` when `plate.event_dims` supplies the target
        shape; this realises the Pyro idiom of `torch.full((K,),
        alpha)` for a scalar `alpha` passed to `Dirichlet`.
        """
        positional: list[str] = []
        keyword: list[tuple[str, str]] = []
        if not arg_names or len(arg_names) != len(args):
            torch_constraints = getattr(
                meta.distribution_class, "arg_constraints", None
            )
            if isinstance(torch_constraints, dict):
                arg_names = tuple(torch_constraints.keys())
            else:
                arg_names = ()
        for ir_arg, arg_name in zip(args, arg_names, strict=False):
            target = None
            if plate is not None:
                target = _scalar_broadcast_for_arg(
                    meta, ir_arg, arg_name, plate, pctx.name_rank
                )
            if target is not None:
                vid = self._broadcast(pctx, ir_arg, target)
            else:
                vid = self._lower_arg(pctx, ir_arg, meta=meta, arg_name=arg_name)
            if arg_name in aliases:
                keyword.append((aliases[arg_name], vid))
            else:
                positional.append(vid)
        return _DistArgs(
            positional=tuple(positional),
            keyword=tuple(keyword),
        )

    def _lower_arg(
        self,
        pctx: _PyroCtx,
        arg: IRArg,
        *,
        meta: FamilyMeta,
        arg_name: str,
    ) -> str:
        """Lower one IR arg to a single Python expression vertex.

        Scalar-to-vector broadcasting is the caller's responsibility
        (see [`_build_dist_args`][PyroRenderer._build_dist_args]); this
        helper handles per-arg structural dispatch only.
        """
        if isinstance(arg, IRArgNumber):
            literal = number_literal(pctx, arg.value)
            if (meta.qvr_name, arg_name) in _PYRO_TENSOR_SCALARS:
                return call(
                    pctx,
                    attribute(pctx, ("torch", "tensor")),
                    positional=(literal,),
                )
            return literal
        if isinstance(arg, IRArgRef):
            return _arg_ref_vid(pctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self._broadcast(pctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(pctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(pctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._resolve_family_ref(pctx, arg)
        raise UnsupportedConstruct(
            f"qvr-{_TARGET}",
            [f"arg-kind:{type(arg).__name__}"],
        )

    def _arg_to_vid(self, pctx: _PyroCtx, arg: IRArg) -> str:
        """Convenience: lower an arg without `meta` context (used in
        list / matrix / broadcast inner positions)."""
        if isinstance(arg, IRArgNumber):
            return number_literal(pctx, arg.value)
        if isinstance(arg, IRArgRef):
            return _arg_ref_vid(pctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self._broadcast(pctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(pctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(pctx, arg)
        raise UnsupportedConstruct(
            f"qvr-{_TARGET}",
            [f"inner-arg-kind:{type(arg).__name__}"],
        )

    def _resolve_family_ref(
        self,
        pctx: _PyroCtx,
        arg: IRArgFamilyRef,
    ) -> str:
        """Resolve an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        to a `pyro.distributions.<Family>(...)` call.

        The referenced morphism's `~ Family(...)` clause names the
        inner family; the Pyro renderer dispatches the wrapped form
        directly (e.g. `Truncated(base, ...)` with
        `morphism base ~ Normal(0, 1)` emits
        `pyro.distributions.TruncatedNormal(loc=0, scale=1, low=-2,
        high=2)`).
        """
        morph = pctx.morphisms.get(arg.name)
        if morph is None or not isinstance(morph.init_family, MorphismInitFamily):
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family-ref:unresolved:{arg.name}"],
            )
        inner_family = morph.init_family.family
        # For the Truncated wrapper, the canonical Pyro emission uses
        # the truncated specialisation of the inner family (e.g.
        # TruncatedNormal). The dispatch picks that specialisation by
        # name through FAMILY_META.
        truncated_name = f"Truncated{inner_family}"
        meta = FAMILY_META.get(truncated_name) or FAMILY_META.get(inner_family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family-ref:unknown-family:{inner_family}"],
            )
        dist_class = meta.target_names.get(_TARGET, meta.qvr_name)
        return attribute(pctx, ("pyro", "distributions", dist_class))

    def _leading_dim_arg(
        self,
        pctx: _PyroCtx,
        family: str,
        plate: Plate | None,
    ) -> str:
        """Emit the correlation-matrix dimension for a Cholesky-LKJ
        family, read off the sample's first event axis.

        Pyro's `LKJCorrCholesky` / `LKJ` require the matrix dimension
        `d` as the leading positional argument; the QVR init clause
        supplies only the concentration, so the dimension is recovered
        from the sample site's event axis (e.g. `sample chol : Dim`
        with `Dim : FinSet 4` yields `4`).
        """
        if plate is None or not plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}:missing-dimension-axis"],
            )
        dim = plate.event_dims[0]
        if not isinstance(dim, DimStatic):
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}:dynamic-dimension-axis"],
            )
        return number_literal(pctx, float(dim.size))

    def _family_meta(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}"],
            )
        return meta

    # ----- score / return -----

    def _emit_score_pyro(self, pctx: _PyroCtx, node: IRScore) -> None:
        """`<name> = <expr>; pyro.factor("<name>", <name>)`.

        Pyro-local emitter; never goes through the
        [`RendererBase`][quivers.transpile.renderers.RendererBase]
        dispatch (the Pyro renderer routes through
        `_dispatch_pyro_node`
        instead, with its own
        `_PyroCtx`
        rather than the base `RenderContext`).
        """
        asn = pctx.v(pctx.fresh("asn"), "assignment")
        lhs = identifier(pctx, node.name)
        pctx.e(asn, lhs, "left")
        pctx.e(asn, render_let_expr_python(pctx, node.expr), "right")
        pctx.e(pctx.body, asn, "child_of")
        factor_call = call(
            pctx,
            attribute(pctx, ("pyro", "factor")),
            positional=(
                string_literal(pctx, node.name),
                identifier(pctx, node.name),
            ),
        )
        pctx.e(pctx.body, factor_call, "child_of")

    def _emit_return_pyro(self, pctx: _PyroCtx, names: tuple[str, ...]) -> None:
        """Emit `return <var>` / `return <a>, <b>, ...`.

        Pyro-local emitter; see
        `_emit_score_pyro`
        for why this does not override the base method.
        """
        if not names:
            return
        rs = pctx.v(pctx.fresh("ret"), "return_statement")
        if len(names) == 1:
            pctx.e(rs, identifier(pctx, names[0]), "child_of")
        else:
            elist = pctx.v(pctx.fresh("elist"), "expression_list")
            for var in names:
                pctx.e(elist, identifier(pctx, var), "child_of")
            pctx.e(rs, elist, "child_of")
        pctx.e(pctx.body, rs, "child_of")


# ---------------------------------------------------------------------------
# Helpers / data carriers (renderer-local).
# ---------------------------------------------------------------------------


class _PyroCtx(PyCtx):
    """A `PyCtx`
    enriched with the function body block id, the set of observed
    names, and the resolved morphism table.

    Used by the renderer to thread emission state through the IR walk
    without rebuilding the panproto builder on every call.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        cards: dict[str, int] | None = None,
        name_event_rank: dict[str, int] | None = None,
        factor_towers: frozenset[str] | None = None,
        name_plates: dict[str, Plate] | None = None,
        name_rank: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            sb,
            cards=cards,
            target="pyro",
            name_event_rank=name_event_rank,
            factor_towers=factor_towers,
            name_plates=name_plates,
            name_rank=name_rank,
        )
        self.body: str = ""
        self.observed: frozenset[str] = frozenset()
        self.morphisms: dict = {}
        self.module: IRQiecModule | None = None
        self.operations_bound: set[str] = set()
        # Plate-axis name -> the local variable holding a single reused
        # `pyro.plate(...)` object. Pyro registers each `pyro.plate`
        # context as a site named after the axis, so two inline plates
        # of the same name collide ("Multiple sample sites named ...").
        # Axes that back more than one sample / observe site are hoisted
        # to one shared object and every `with` reuses it.
        self.shared_plate_vars: dict[str, str] = {}


def _minus_one(pctx: _PyroCtx) -> str:
    """Emit the literal ``-1``."""
    return _python_unary_minus(pctx, number_literal(pctx, 1))


def _stack_last_dim(pctx: _PyroCtx, items: tuple[str, ...]) -> str:
    """Emit ``torch.stack([...], dim=-1)``."""
    return call(
        pctx,
        attribute(pctx, ("torch", "stack")),
        positional=(python_list(pctx, items),),
        keyword=(("dim", _minus_one(pctx)),),
    )


class _DistArgs:
    """Split of distribution-call args into positional / keyword."""

    __slots__ = ("positional", "keyword")

    def __init__(
        self,
        *,
        positional: tuple[str, ...],
        keyword: tuple[tuple[str, str], ...],
    ) -> None:
        self.positional = positional
        self.keyword = keyword


def _observed_names(body: tuple[IRNode, ...]) -> set[str]:
    """Recursively collect every name bound by an
    [`IRObserve`][quivers.transpile.ir.IRObserve] in the IR body.
    """
    out: set[str] = set()
    for node in body:
        if isinstance(node, IRObserve):
            out.add(node.name)
        elif isinstance(node, IRMarginalize):
            out.update(_observed_names(node.scope))
    return out


def _shared_plate_dims(body: tuple[IRNode, ...]) -> tuple[Dim, ...]:
    """Batch-plate dims whose axis name backs more than one sample /
    observe site, in first-appearance order.

    Pyro registers each `pyro.plate(<name>, ...)` context as a site
    named after the axis, so two inline plates of the same name raise
    "Multiple sample sites named ...". The renderer hoists each such
    axis to one reused plate object.
    """
    counts: dict[str, int] = {}
    first: dict[str, Dim] = {}
    _count_plate_dims(body, counts, first)
    return tuple(dim for axis, dim in first.items() if counts[axis] > 1)


def _count_plate_dims(
    body: tuple[IRNode, ...],
    counts: dict[str, int],
    first: dict[str, Dim],
) -> None:
    for node in body:
        if isinstance(node, (IRSample, IRObserve)):
            for dim in node.plate.batch_dims:
                axis = str(dim.name)
                counts[axis] = counts.get(axis, 0) + 1
                first.setdefault(axis, dim)
        elif isinstance(node, IRMarginalize):
            _count_plate_dims(node.scope, counts, first)


def _integer_index_input_names(ir: IRProgram) -> set[str]:
    """Host-integer inputs the program uses as a subscript index.

    torch's advanced indexing rejects a float index tensor, so every
    such name is coerced to `long` at the top of the emitted model.
    Walks each [`IRDeterministic`][quivers.transpile.ir.IRDeterministic]
    let-expression for `LetExprIndex` operands that are bare
    [`LetExprVar`][quivers.transpile.ir.LetExprVar] naming a
    host-integer input; a QVR `LetExprVar` in subscript position is
    always a runtime gather covariate.
    """
    host_ints = host_integer_input_names(ir)
    found: set[str] = set()
    _collect_integer_index_names(ir.body, host_ints, found)
    return found


def _collect_integer_index_names(
    body: tuple[IRNode, ...],
    host_ints: frozenset[str],
    found: set[str],
) -> None:
    for node in body:
        if isinstance(node, IRDeterministic):
            _walk_let_expr_for_indices(node.expr, host_ints, found)
        elif isinstance(node, (IRSample, IRObserve)):
            _walk_args_for_indices(node.args, host_ints, found)
            if isinstance(node, IRObserve) and node.via in host_ints:
                # The fibration gathers the marginalize weights at the
                # observation's own row, so `via` reaches a subscript.
                found.add(node.via)
        elif isinstance(node, IRMarginalize):
            _walk_args_for_indices(node.args, host_ints, found)
            _collect_integer_index_names(node.scope, host_ints, found)


def _walk_args_for_indices(
    args: tuple[IRArg, ...],
    host_ints: frozenset[str],
    found: set[str],
) -> None:
    """Collect host-integer inputs that appear in an
    [`IRArgRef`][quivers.transpile.ir.IRArgRef] subscript position."""
    for arg in args:
        if isinstance(arg, IRArgRef):
            for idx in arg.indices:
                if isinstance(idx, IRArgRef) and idx.name in host_ints:
                    found.add(idx.name)
            _walk_args_for_indices(arg.indices, host_ints, found)
        elif isinstance(arg, IRArgBroadcast):
            _walk_args_for_indices((arg.value,), host_ints, found)
        elif isinstance(arg, IRArgList):
            _walk_args_for_indices(arg.elements, host_ints, found)
        elif isinstance(arg, IRArgMatrix):
            for row in arg.rows:
                _walk_args_for_indices(row.elements, host_ints, found)


def _walk_let_expr_for_indices(
    expr: LetExprNode,
    host_ints: frozenset[str],
    found: set[str],
) -> None:
    if isinstance(expr, LetExprIndex):
        for idx in expr.indices:
            # An index is the input itself (`a[idx]`) or an entry of it
            # (`M[r_idx[n], c_idx[n]]`, a pairwise subscript unrolled).
            root = idx
            while isinstance(root, LetExprIndex):
                root = root.array
            if isinstance(root, LetExprVar) and root.name in host_ints:
                found.add(root.name)
            _walk_let_expr_for_indices(idx, host_ints, found)
        _walk_let_expr_for_indices(expr.array, host_ints, found)
        return
    if isinstance(expr, LetExprFactor):
        if expr.body is not None:
            _walk_let_expr_for_indices(expr.body, host_ints, found)
        for case in expr.cases:
            _walk_let_expr_for_indices(case.value, host_ints, found)
        return
    if isinstance(expr, LetExprBinOp):
        _walk_let_expr_for_indices(expr.left, host_ints, found)
        _walk_let_expr_for_indices(expr.right, host_ints, found)
        return
    if isinstance(expr, LetExprUnaryOp):
        _walk_let_expr_for_indices(expr.operand, host_ints, found)
        return
    if isinstance(expr, LetExprCall):
        for a in expr.args:
            _walk_let_expr_for_indices(a, host_ints, found)
        return
    if isinstance(expr, LetExprList):
        for item in expr.items:
            _walk_let_expr_for_indices(item, host_ints, found)


def _arg_ref_vid(pctx: _PyroCtx, arg: IRArgRef) -> str:
    """Build `<name>[<idx0>][<idx1>]...` as a chain of `subscript`
    vertices."""
    base = identifier(pctx, arg.name)
    for idx in arg.indices:
        s = pctx.v(pctx.fresh("subs"), "subscript")
        pctx.e(s, base, "value")
        pctx.e(
            s,
            _arg_ref_vid(pctx, idx)
            if isinstance(idx, IRArgRef)
            else _inner_index_vid(pctx, idx),
            "subscript",
        )
        base = s
    return base


def _inner_index_vid(pctx: _PyroCtx, arg: IRArg) -> str:
    """Inner subscript expression: ref or number; lists/broadcast not
    supported as index expressions."""
    if isinstance(arg, IRArgRef):
        return _arg_ref_vid(pctx, arg)
    if isinstance(arg, IRArgNumber):
        return number_literal(pctx, arg.value)
    raise UnsupportedConstruct(
        f"qvr-{_TARGET}",
        [f"subscript-arg:{type(arg).__name__}"],
    )


def _function_def_split(
    pctx: _PyroCtx,
    *,
    name: str,
    positional: tuple[str, ...],
    defaults: tuple[str, ...],
    body_vid: str,
) -> str:
    """Build `def <name>(<pos0>, <pos1>, ..., <def0>=None, ...): <body>`.

    `function_def`
    emits every param as `<name>=None`; this variant carries the Pyro
    idiom of positional model params followed by `<obs>=None` for
    every observation.
    """
    func = pctx.v(pctx.fresh("fn"), "function_definition")
    fname = identifier(pctx, name)
    params = pctx.v(pctx.fresh("ps"), "parameters")
    pctx.e(func, fname, "name")
    pctx.e(func, params, "parameters")
    pctx.e(func, body_vid, "body")
    for pname in positional:
        pctx.e(params, identifier(pctx, pname), "child_of")
    for pname in defaults:
        dp = pctx.v(pctx.fresh("dp"), "default_parameter")
        dp_name = identifier(pctx, pname)
        dp_val = pctx.v(pctx.fresh("none"), "none")
        pctx.literal(dp_val, "None")
        pctx.e(dp, dp_name, "name")
        pctx.e(dp, dp_val, "value")
        pctx.e(params, dp, "child_of")
    return func


# ---------------------------------------------------------------------------
# Plate-aware scalar broadcast: the per-call path used by
# `_emit_sample_or_observe`.
# ---------------------------------------------------------------------------


def _scalar_broadcast_for_arg(
    meta: FamilyMeta,
    arg: IRArg,
    arg_name: str,
    plate: Plate,
    name_rank: dict[str, int],
) -> tuple[int, ...] | None:
    """If `arg` is a bare scalar reference / literal and `arg_name`'s
    slot is a rank-`n` array, return the target shape derived from
    `plate.event_dims`; otherwise `None`.

    A reference is scalar only when the plan binds its name to a value
    of no axes: a reference to a vector (a sampled `loc` or
    `concentration`) already carries the slot's shape and is passed
    through as it stands.

    When the family declares a
    [`StructuredSampleLowering`][quivers.transpile.ir.StructuredSampleLowering],
    the per-argument shape is read off the matching
    [`StructuredDataArg`][quivers.transpile.ir.StructuredDataArg]'s
    `axis_indices` so a Kronecker-factored family (e.g. `MatrixNormal`)
    fills its row covariance from event axis 0 (`(0, 0)`) and its
    column covariance from event axis 1 (`(1, 1)`), rather than
    filling every covariance to the full `loc` shape. Absent a
    structured lowering, the target shape falls back to the leading
    `event_dim` axes implied by the torch arg constraint (the
    `Dirichlet` idiom of `torch.full((K,), alpha)`).

    Parameters
    ----------
    meta : FamilyMeta
        The family's metadata.
    arg : IRArg
        The argument.
    arg_name : str
        The family parameter the argument binds.
    plate : Plate
        The site's plate, whose event axes size the fill.
    name_rank : dict[str, int]
        The number of axes of each name the plan binds.

    Returns
    -------
    tuple[int, ...] | None
        The shape to fill the scalar to, or `None` when the argument is
        passed as it stands.
    """
    if not isinstance(arg, (IRArgRef, IRArgNumber)):
        return None
    if isinstance(arg, IRArgRef) and (arg.indices or name_rank.get(arg.name, 0)):
        return None
    axis_indices = _structured_axis_indices(meta, arg_name)
    if axis_indices is not None:
        if not axis_indices:
            return None
        return _shape_from_axis_indices(axis_indices, plate)
    constraints = getattr(meta.distribution_class, "arg_constraints", {})
    expected = constraints.get(arg_name) if isinstance(constraints, dict) else None
    if expected is None:
        return None
    event_dim = int(getattr(expected, "event_dim", 0))
    if event_dim < 1:
        return None
    return _shape_from_axis_indices(tuple(range(event_dim)), plate)


def _structured_axis_indices(meta: FamilyMeta, arg_name: str) -> tuple[int, ...] | None:
    """The `axis_indices` of the family's structured-lowering
    [`StructuredDataArg`][quivers.transpile.ir.StructuredDataArg] whose
    `arg_name` matches, or `None` when the family declares no
    structured lowering (or no data arg of that name)."""
    lowering = meta.structured_lowering
    if lowering is None:
        return None
    for spec in lowering.args:
        if isinstance(spec, StructuredDataArg) and spec.arg_name == arg_name:
            return spec.axis_indices
    return None


def _shape_from_axis_indices(
    axis_indices: tuple[int, ...], plate: Plate
) -> tuple[int, ...] | None:
    """Resolve `axis_indices` against `plate.event_dims`, returning the
    concrete static shape or `None` if any indexed axis is out of range
    or dynamic."""
    sizes: list[int] = []
    for i in axis_indices:
        if i >= len(plate.event_dims):
            return None
        dim = plate.event_dims[i]
        if not isinstance(dim, DimStatic):
            return None
        sizes.append(dim.size)
    return tuple(sizes)


#: The Cholesky-LKJ families whose Pyro distribution class takes the
#: correlation-matrix dimension `d` as a mandatory leading positional
#: argument (`LKJCorrCholesky(d, eta)`, `LKJ(d, eta)`). The QVR init
#: clause carries only the concentration, so the renderer prepends the
#: dimension read off the sample's first event axis.
_PYRO_LEADING_DIM_FAMILIES = frozenset(
    {
        "LKJCholesky",
        "LKJCorrelationFactor",
    }
)


#: Families whose Pyro constructor binds the QVR positional slots in a
#: different order than the QVR call site writes them. A QVR argument
#: list is positional against
#: [`FamilyMeta.distribution_class`][quivers.transpile.family_meta.FamilyMeta]'s
#: own constructor, so `BetaBinomial(n, a, b)` in QVR names
#: `(total_count, concentration1, concentration0)` while
#: `pyro.distributions.BetaBinomial` is constructed
#: `(concentration1, concentration0, total_count)`. Emitting those
#: three positionally transposes the trial count with a concentration
#: and silently scores a different density, so every slot of such a
#: family is emitted as a keyword argument, which is order-free. The
#: map is `qvr arg name -> pyro keyword`; identity entries are kept
#: explicit so a reader can check the whole signature in one place.
#: Scalar parameters a Pyro class reads as a tensor: the relaxed
#: families scale their log density by ``temperature`` through
#: ``torch.full_like``, which takes no Python number.
_PYRO_TENSOR_SCALARS: frozenset[tuple[str, str]] = frozenset(
    {
        ("RelaxedBernoulli", "temperature"),
        ("RelaxedOneHotCategorical", "temperature"),
    }
)


_PYRO_KEYWORD_BINDINGS: dict[str, dict[str, str]] = {
    "Categorical": {
        "logits": "logits",
    },
    **{
        family: {"logits": "logits"}
        for family in ("Bernoulli", "Binomial", "Geometric", "NegativeBinomial")
    },
    **{
        family: {"precision_matrix": "precision_matrix"}
        for family in ("MultivariateNormal", "Wishart")
    },
    "BetaBinomial": {
        "total_count": "total_count",
        "concentration1": "concentration1",
        "concentration0": "concentration0",
    },
}


#: Families whose Pyro target class carries a leading parameter that
#: the QVR call site never writes because the family fixes it, mapped
#: to the value that fills it; emitting such a family's arguments
#: positionally would bind the first to the fixed parameter, so the
#: renderer prepends the fixed value instead.
_PYRO_FIXED_LEADING_ARGS: dict[str, float] = {}


_RUNTIME_PYRO_PATH = pathlib.Path(__file__).resolve().parent.parent / "runtime_pyro.py"


def _class_definition_name(schema: panproto.Schema, class_vid: str) -> str | None:
    """The literal name of a `class_definition` vertex, or `None`."""
    for edge in schema.edges:
        if edge.src == class_vid and edge.kind == "name":
            name_v = next(
                (vv for vv in schema.vertices if vv.id == edge.tgt),
                None,
            )
            if name_v is None:
                continue
            return next(
                (
                    c.value
                    for c in schema.constraints_for(name_v.id)
                    if c.sort == "literal-value"
                ),
                None,
            )
    return None


def _subtree_vertex_ids(schema: panproto.Schema, root: str) -> set[str]:
    """Return every vertex id reachable from `root` via outgoing
    edges of `schema`."""
    seen: set[str] = {root}
    frontier: list[str] = [root]
    while frontier:
        src = frontier.pop()
        for edge in schema.edges:
            if edge.src == src and edge.tgt not in seen:
                seen.add(edge.tgt)
                frontier.append(edge.tgt)
    return seen


def _load_runtime_pyro_helpers() -> tuple[
    panproto.Schema, dict[str, str], dict[str, set[str]]
]:
    """Parse [`runtime_pyro.py`][quivers.transpile.runtime_pyro]
    through panproto's Python tree-sitter grammar at module-load time
    and index every helper `class_definition` subtree by class name.

    Returns the parsed schema, a map from class name to the
    class-definition vertex id, and a map from class name to the set
    of vertex ids in that class's subtree. The renderer's
    `_emit_runtime_helper`
    grafts a class subtree (vertex + all descendants + their
    constraints + edges) into the per-render schema as a `child_of`
    of the emitted module. The emit is structurally a real Python
    class definition, with no string literal, no `exec`, and no
    runtime self-injection.
    """
    schema = parser_registry().parse_with_protocol(
        "python",
        _RUNTIME_PYRO_PATH.read_bytes(),
        str(_RUNTIME_PYRO_PATH),
    )
    roots: dict[str, str] = {}
    for v in schema.vertices:
        if v.kind == "class_definition":
            name = _class_definition_name(schema, v.id)
            if name is not None:
                roots[name] = v.id
    if not roots:
        raise RuntimeError(
            f"no helper `class` definitions found in {_RUNTIME_PYRO_PATH}; "
            "the renderer expects it as the source of truth for the "
            "embedded Pyro runtime helpers."
        )
    subtrees = {name: _subtree_vertex_ids(schema, root) for name, root in roots.items()}
    return schema, roots, subtrees


(
    _RUNTIME_PYRO_HELPER_SCHEMA,
    _RUNTIME_PYRO_HELPER_ROOTS,
    _RUNTIME_PYRO_HELPER_SUBTREES,
) = _load_runtime_pyro_helpers()


def _ir_helper_classes_used(body: tuple[IRNode, ...]) -> set[str]:
    """The set of runtime-helper class names the IR body needs grafted.

    For each [`IRSample`][quivers.transpile.ir.IRSample] /
    [`IRObserve`][quivers.transpile.ir.IRObserve] (including nested
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes),
    a helper is needed when the family's Pyro target name matches a
    `class` defined in
    [`runtime_pyro.py`][quivers.transpile.runtime_pyro]."""
    used: set[str] = set()
    for node in body:
        if isinstance(node, (IRSample, IRObserve)):
            meta = FAMILY_META.get(node.family)
            if meta is not None:
                target = meta.target_names.get(_TARGET)
                if target in _RUNTIME_PYRO_HELPER_ROOTS:
                    used.add(target)
        elif isinstance(node, IRMarginalize):
            used |= _ir_helper_classes_used(node.scope)
    return used


def _emit_runtime_helper(pctx: _PyroCtx, class_name: str) -> None:
    """Graft the named helper `class` subtree from
    [`runtime_pyro.py`][quivers.transpile.runtime_pyro] into the
    per-render schema as a top-level child of `mod`.

    The class definition is a real
    `class_definition` panproto
    subtree (parsed once at module load via panproto's Python
    tree-sitter grammar). The renderer copies every vertex, every
    constraint, and every internal edge of the subtree into the
    per-render `SchemaBuilder`, then attaches the class root as a
    `child_of` of `mod`. Subsequent call sites (e.g.
    ``TruncatedNormal(loc, scale, low, high)`` or
    ``MatrixNormal(loc, row_covariance, col_covariance)``) in the
    rendered model body resolve to that class through normal Python
    name lookup. Vertex ids are rewritten through `pctx.fresh` so the
    graft does not collide with builder-allocated ids.
    """
    src_schema = _RUNTIME_PYRO_HELPER_SCHEMA
    subtree = _RUNTIME_PYRO_HELPER_SUBTREES[class_name]
    root = _RUNTIME_PYRO_HELPER_ROOTS[class_name]
    id_map: dict[str, str] = {}

    for old in subtree:
        new = pctx.fresh("rh")
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        pctx.v(new, kind)
        for cstr in src_schema.constraints_for(old):
            pctx.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            pctx.e(id_map[edge.src], id_map[edge.tgt], edge.kind)
    pctx.e("mod", id_map[root], "child_of")


__all__ = ["PyroRenderer"]
