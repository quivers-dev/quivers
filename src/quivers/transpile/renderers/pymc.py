"""Render transpilation IR as a PyMC model builder.

Samples and observations become PyMC distribution constructors inside
a ``Model`` context. Finite marginalizations become ``pymc.Mixture``
observations, and exported values become ``pymc.Deterministic``
variables.
"""

from __future__ import annotations

import pathlib

import torch.distributions.constraints as c
import panproto

from quivers.dsl.ast_nodes import (
    Expr,
    MorphismDecl,
)
from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import (
    parser_registry,
    target_protocol,
)
from quivers.transpile.renderers._ranks import name_rank_map
from quivers.transpile.renderers._python_helpers import (
    MarginalizeBody,
    PyCtx,
    arg_expr,
    assignment,
    attribute,
    call,
    factor_tower_names,
    float_literal,
    function_def,
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
    string_literal,
    subscript,
    with_statement,
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
)
from quivers.transpile.renderers._base import (
    refuse_ungrouped_row_marginalize,
    BlockKind,
    IRArgTransform,
    IRMarginalAtom,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dropped_param_map,
    assert_no_dangling_refs,
)
from quivers.transpile.renderers._qiec import (
    emit_call_python,
    graft_python_statements,
    qiec_helper_families_used,
    render_computations_dynamic,
)


class PyMCRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    PyMC `build_model` function under the `python` tree-sitter grammar.

    The output has the shape:

    ```python
    def build_model(<inputs>=None):
        with pymc.Model(coords={...}) as model:
            theta = pymc.Dirichlet("theta", a=..., dims=(...))
            phi = pymc.Dirichlet("phi", a=..., dims=(...))
            pymc.Categorical("w", p=phi[...], observed=w, dims=(...))
        return model
    ```
    """

    target: str = "qvr-pymc"

    # ----- protocol -----

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
        """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
        PyMC schema. Subclasses normally call
        [`render_with_tables`][quivers.transpile.renderers.pymc.PyMCRenderer.render_with_tables]
        when morphism / let tables are available.

        Parameters
        ----------
        ir
            The lowered program.

        Returns
        -------
        panproto.Schema
            The target program, in the target grammar's theory.
        """
        return self.render_with_tables(ir, morphisms={}, lets={})

    def render_with_tables(
        self,
        ir: IRProgram,
        *,
        morphisms: dict[str, MorphismDecl],
        lets: dict[str, Expr],
    ) -> panproto.Schema:
        """Same as `render` but pre-populates `ctx.morphisms` / `ctx.defines`
        so [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        rendering can read the referenced morphism's `init_family`
        clause.

        Parameters
        ----------
        ir
            The lowered program.
        morphisms
            The module's morphism declarations, by name.
        lets
            The module's define expressions, by name.

        Returns
        -------
        panproto.Schema
            The target program, in the target grammar's theory.
        """
        assert_no_dangling_refs(ir)
        assert_no_dropped_param_map(ir, self.target)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = RenderContext(sb=sb, morphisms=dict(morphisms), defines=dict(lets))
        py = PyCtx(
            sb,
            cards=dict(ir.cards),
            target="pymc",
            name_event_rank=name_event_rank_map(ir),
            name_rank=name_rank_map(ir),
            factor_towers=factor_tower_names(ir),
            name_plates=name_plate_map(ir),
        )
        bag = _PyMCCtx(ctx=ctx, py=py, ir=ir)

        # `module` root.
        py.v("mod", "module")

        # Function body block.
        fn_body = py.v(py.fresh("fbody"), "block")

        # `with pymc.Model(coords={...}) as model:` header.
        coords_dict = self._build_coords_dict(py, ir)
        model_call = call(
            py,
            attribute(py, ("pymc", "Model")),
            keyword=(("coords", coords_dict),),
        )
        with_body = py.v(py.fresh("wbody"), "block")
        ws = with_statement(
            py, expression=model_call, alias="model", body_vid=with_body
        )
        py.e(fn_body, ws, "child_of")

        # Walk the body, dispatching each node into the with-body block.
        bag.with_body = with_body
        if not ir.body:
            noop = py.v(py.fresh("pass"), "pass_statement")
            py.e(with_body, noop, "child_of")
        for node in ir.body:
            self._dispatch_pymc(bag, node)

        # Graft runtime helpers for families PyMC does not ship, once,
        # as top-level definitions preceding `build_model`.
        if any(
            _ir_uses_family(ir.body, family) for family in _PYMC_RUNTIME_HELPER_FAMILIES
        ) or qiec_helper_families_used(ir, self.target):
            _graft_runtime_pymc_helpers(py)

        # Trailing `return model` inside the function body (outside
        # the with-block). PyMC's idiom is for `build_model` to return
        # the constructed model object.
        ret = py.v(py.fresh("ret"), "return_statement")
        py.e(ret, identifier(py, "model"), "child_of")
        py.e(fn_body, ret, "child_of")

        # `def build_model(<data_inputs>=None): <fn_body>`.
        param_names = tuple(inp.name for inp in ir.inputs)
        fn = function_def(
            py,
            name="build_model",
            default_params=param_names,
            body_vid=fn_body,
        )
        py.e("mod", fn, "child_of")

        render_computations_dynamic(sb, ir, target=self.target, root="mod")
        return sb.build()

    # ----- coord dict construction -----

    def _build_coords_dict(self, py: PyCtx, ir: IRProgram) -> str:
        """Build the `coords={"axis": np.arange(N), ...}` dict.

        Walks every plate in the IR (inputs + body), collecting each
        unique [`Dim`][quivers.transpile.ir.Dim] axis name. Static
        dims use `np.arange(<size>)`; dynamic dims use
        `np.arange(<size_name>)` (the size variable arrives via the
        function signature).
        """
        seen: dict[str, Dim] = {}
        for inp in ir.inputs:
            self._collect_dims(inp.plate, seen)
        self._collect_dims_in_body(ir.body, seen)
        # An open extent is the length of an input axis it sizes, which
        # the input's own array carries.
        extents: dict[str, tuple[str, int]] = {}
        for inp in ir.inputs:
            dims = (*inp.plate.batch_dims, *inp.plate.event_dims)
            for position, dim in enumerate(dims):
                if isinstance(dim, DimDynamic):
                    extents.setdefault(dim.size_name, (inp.name, position))

        d = py.v(py.fresh("dict"), "dictionary")
        for name, dim in seen.items():
            key = string_literal(py, name)
            source = extents.get(dim.size_name) if isinstance(dim, DimDynamic) else None
            value = (
                self._np_arange_of_axis(py, *source)
                if source is not None
                else self._np_arange(py, dim)
            )
            pair = py.v(py.fresh("pair"), "pair")
            py.e(pair, key, "key")
            py.e(pair, value, "value")
            py.e(d, pair, "child_of")
        return d

    def _collect_dims(self, plate: Plate, seen: dict[str, Dim]) -> None:
        for dim in (*plate.batch_dims, *plate.event_dims):
            name = _dim_name(dim)
            seen.setdefault(name, dim)

    def _collect_dims_in_body(
        self, body: tuple[IRNode, ...], seen: dict[str, Dim]
    ) -> None:
        for node in body:
            if isinstance(node, (IRSample, IRObserve, IRDeterministic, IRDataInput)):
                self._collect_dims(node.plate, seen)
            elif isinstance(node, IRMarginalize):
                self._collect_dims(node.plate, seen)
                self._collect_dims_in_body(node.scope, seen)

    def _np_arange_of_axis(self, py: PyCtx, name: str, position: int) -> str:
        """Emit `np.arange(np.shape(<name>)[<position>])`.

        Parameters
        ----------
        py : PyCtx
            The render state.
        name : str
            The input whose axis gives the length.
        position : int
            The axis.

        Returns
        -------
        str
            The vertex of the call.
        """
        shape = call(
            py, attribute(py, ("np", "shape")), positional=(identifier(py, name),)
        )
        length = subscript(py, shape, (number_literal(py, position),))
        return call(py, attribute(py, ("np", "arange")), positional=(length,))

    def _np_arange(self, py: PyCtx, dim: Dim) -> str:
        """Emit `np.arange(<size>)` for a static dim, or
        `np.arange(<size_name>)` for a dynamic dim."""
        callee = attribute(py, ("np", "arange"))
        if isinstance(dim, DimStatic):
            return call(
                py,
                callee,
                positional=(number_literal(py, float(dim.size)),),
            )
        if isinstance(dim, DimDynamic):
            return call(
                py,
                callee,
                positional=(identifier(py, dim.size_name),),
            )
        msg = f"unknown Dim kind: {type(dim).__name__}"
        raise UnsupportedConstruct(self.target, [msg])

    # ----- per-node dispatch (PyMC walks over with-body) ----

    def _dispatch_pymc(self, ctx: _PyMCCtx, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            # Data inputs are function parameters, declared in
            # `function_def`. No additional emit inside the with-body.
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            self._emit_sample(ctx, node, observed_name=None)
            return
        if isinstance(node, IRObserve):
            self._emit_sample(ctx, node, observed_name=node.name)
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score_step(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize(ctx, node)
            return
        if isinstance(node, IRCall):
            emit_call_python(
                ctx.py, ctx.with_body, node, ctx.ir.module, ctx.operations_bound
            )
            return
        if isinstance(node, IRReturn):
            self._emit_export(ctx, node.names)
            return
        raise UnsupportedConstruct(self.target, [f"node:{type(node).__name__}"])

    def _emit_export(self, ctx: _PyMCCtx, names: tuple[str, ...]) -> None:
        """Expose returned names as `pymc.Deterministic` values.

        `build_model` returns the model itself, so exports are registered as
        named deterministic variables. The ``_value`` suffix avoids collisions
        with sampled, observed, or let-bound names and matches the Stan export
        spelling. `pymc.math.as_tensor` converts raw observed arrays while
        leaving symbolic tensor expressions usable by `pymc.Deterministic`.
        """
        py = ctx.py
        for name in names:
            det = call(
                py,
                attribute(py, ("pymc", "Deterministic")),
                positional=(
                    string_literal(py, f"{name}_value"),
                    call(
                        py,
                        attribute(py, ("pymc", "math", "as_tensor")),
                        positional=(identifier(py, name),),
                    ),
                ),
            )
            stmt = py.v(py.fresh("es"), "expression_statement")
            py.e(stmt, det, "child_of")
            py.e(ctx.with_body, stmt, "child_of")

    # ----- declare: no-op for PyMC (declarations are constructor calls) ----

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """PyMC declarations ARE the constructor calls (emitted in
        `sample`). `declare` is thus a no-op; the caller's emit
        handles both declaration and assignment in one node.

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

    def _emit_gp_block(
        self,
        ctx: _PyMCCtx,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as a triple of PyMC
        statements inside the with-block:

            __gp_mean_<name> = pt.zeros(N)
            __gp_cov_<name>  = pt.exp(-0.5 * (diff)*(diff) / (ls*ls))
                                + jitter * pt.eye(N)
            <name> = pymc.MvNormal("<name>", mu=__gp_mean_<name>,
                                    cov=__gp_cov_<name>)

        PyMC's `pt` namespace is PyTensor (the array backend
        sym-differentiable layer); `pt.exp`, `pt.eye`, `pt.zeros`
        and `pt.reshape` are the canonical math primitives. Parens
        wrap the diff and squared-length-scale subexpressions so the
        Python pretty-printer keeps precedence intact around the
        nested binary_operator children.
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                self.target,
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                self.target,
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        py = ctx.py
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = pt.zeros(N)
        mean_rhs = call(
            py,
            attribute(py, ("pt", "zeros")),
            positional=(number_literal(py, n),),
        )
        py.e(
            ctx.with_body,
            assignment(py, lhs_name=mean_name, rhs=mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = pt.exp(...) + jitter * pt.eye(N)
        # x_col = pt.reshape(x, (-1, 1)), x_row = pt.reshape(x, (1, -1))
        # PyTensor accepts `x.reshape((-1, 1))` syntax.
        x_col = _python_method_call(
            py,
            identifier(py, x),
            "reshape",
            (
                _python_unary_minus(py, number_literal(py, 1)),
                number_literal(py, 1),
            ),
        )
        x_row = _python_method_call(
            py,
            identifier(py, x),
            "reshape",
            (
                number_literal(py, 1),
                _python_unary_minus(py, number_literal(py, 1)),
            ),
        )
        diff = _python_paren(
            py,
            _python_binary_op(py, "-", x_col, x_row),
        )
        diff_sq = _python_paren(
            py,
            _python_binary_op(py, "*", diff, diff),
        )
        ls_sq = _python_paren(
            py,
            _python_binary_op(
                py,
                "*",
                number_literal(py, ls),
                number_literal(py, ls),
            ),
        )
        quotient = _python_binary_op(py, "/", diff_sq, ls_sq)
        neg_half = _python_unary_minus(py, number_literal(py, 0.5))
        exponent = _python_binary_op(py, "*", neg_half, quotient)
        kernel_call = call(
            py,
            attribute(py, ("pt", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            py,
            attribute(py, ("pt", "eye")),
            positional=(number_literal(py, n),),
        )
        jitter_term = _python_binary_op(
            py,
            "*",
            number_literal(py, jitter),
            eye_call,
        )
        cov_rhs = _python_binary_op(
            py,
            "+",
            kernel_call,
            jitter_term,
        )
        py.e(
            ctx.with_body,
            assignment(py, lhs_name=cov_name, rhs=cov_rhs),
            "child_of",
        )
        # <name> = pymc.MvNormal("<name>", mu=__gp_mean_<name>,
        #                         cov=__gp_cov_<name>)
        mvn_call = call(
            py,
            attribute(py, ("pymc", "MvNormal")),
            positional=(string_literal(py, node.name),),
            keyword=(
                ("mu", identifier(py, mean_name)),
                ("cov", identifier(py, cov_name)),
            ),
        )
        py.e(
            ctx.with_body,
            assignment(py, lhs_name=node.name, rhs=mvn_call),
            "child_of",
        )

    def _emit_sample(
        self,
        ctx: _PyMCCtx,
        node: IRSample | IRObserve,
        observed_name: str | None,
    ) -> None:
        """Build `<name> = pymc.<Family>("<name>", **args, dims=(...),
        observed=<observed>)` and append to the with-body block."""
        if node.family == "Geometric":
            self._emit_geometric(ctx, node, observed_name=observed_name)
            return
        if node.family == "LKJCholesky":
            self._emit_lkj_cholesky(
                ctx,
                node,
                observed_name=observed_name,
            )
            return
        meta = _resolve_meta(node.family, self.target)
        dist_class = meta.target_names.get("pymc")
        if dist_class is None:
            raise UnsupportedConstruct(self.target, [f"family:{node.family}"])

        # A family PyMC does not ship (`ContinuousBernoulli`) resolves
        # to a grafted runtime helper called by bare name; the graft
        # is emitted once in `render_with_tables`.
        if node.family in _PYMC_RUNTIME_HELPER_FAMILIES:
            callee = identifier(ctx.py, _PYMC_RUNTIME_HELPER_FAMILIES[node.family])
        else:
            callee = attribute(ctx.py, ("pymc", dist_class))
        positional = (string_literal(ctx.py, node.name),)

        complement_args = _PYMC_COMPLEMENT_ARGS.get(node.family, frozenset())

        # arg_constraints for this family (read class-level only;
        # property-form parameterisations land in a future pass).
        cls_constraints = getattr(
            meta.distribution_class,
            "arg_constraints",
            {},
        )
        if not isinstance(cls_constraints, dict):
            cls_constraints = {}

        # Build the keyword list, applying alias renaming and
        # broadcasting for scalar refs in vector-expected slots.
        aliases = _merged_aliases(meta)
        keyword: list[tuple[str, str]] = []
        for arg, arg_name in zip(node.args, node.arg_names, strict=True):
            renamed = aliases.get(arg_name, arg_name)
            arg_to_emit = self._maybe_broadcast_ref(
                arg,
                arg_name=arg_name,
                cls_constraints=cls_constraints,
                plate=node.plate,
                ir=ctx.ir,
            )
            rendered = self._render_arg(ctx, arg_to_emit)
            if arg_name in complement_args:
                rendered = _python_paren(
                    ctx.py,
                    _python_binary_op(
                        ctx.py,
                        "-",
                        number_literal(ctx.py, 1.0),
                        rendered,
                    ),
                )
            if arg_name in _PYMC_FACTOR_SCALE_ARGS.get(node.family, frozenset()):
                rendered, renamed = self._scale_from_factor(ctx, arg_to_emit, rendered)
            keyword.append((renamed, rendered))

        # `dims=(<batch+event names>,)`.
        dims_tuple = self._dims_tuple(ctx.py, node.plate)
        if dims_tuple is not None:
            keyword.append(("dims", dims_tuple))

        # `observed=<observed>` for IRObserve.
        if observed_name is not None:
            keyword.append(("observed", identifier(ctx.py, observed_name)))

        rhs = call(
            ctx.py,
            callee,
            positional=positional,
            keyword=tuple(keyword),
        )

        if observed_name is None:
            stmt = assignment(ctx.py, lhs_name=node.name, rhs=rhs)
        else:
            stmt = ctx.py.v(ctx.py.fresh("es"), "expression_statement")
            ctx.py.e(stmt, rhs, "child_of")
        ctx.py.e(ctx.with_body, stmt, "child_of")

    def _emit_geometric(
        self,
        ctx: _PyMCCtx,
        node: IRSample | IRObserve,
        *,
        observed_name: str | None,
    ) -> None:
        """Emit a support-corrected PyMC `Geometric`.

        torch `Geometric(probs)` counts failures on `{0, 1, 2, ...}`,
        while PyMC `Geometric(p)` counts trials on `{1, 2, 3, ...}`.
        The two share the success probability `p == probs` but differ
        by a unit shift, so:

        * a latent sample binds the QVR name to a
          [`pymc.Deterministic`][pymc.Deterministic] holding the PyMC
          draw minus one, keeping the density on the shifted RV while
          exposing the torch-convention value; and
        * an observation feeds `observed + 1` so the shifted data land
          in PyMC's support.
        """
        py = ctx.py
        meta = _resolve_meta("Geometric", self.target)
        aliases = _merged_aliases(meta)

        keyword: list[tuple[str, str]] = []
        for arg, arg_name in zip(node.args, node.arg_names, strict=True):
            renamed = aliases.get(arg_name, arg_name)
            keyword.append((renamed, self._render_arg(ctx, arg)))

        dims_tuple = self._dims_tuple(py, node.plate)
        callee = attribute(py, ("pymc", "Geometric"))

        if observed_name is None:
            geom_keyword = list(keyword)
            if dims_tuple is not None:
                geom_keyword.append(("dims", dims_tuple))
            geom = call(
                py,
                callee,
                positional=(string_literal(py, f"{node.name}__geom"),),
                keyword=tuple(geom_keyword),
            )
            # The shift is a positional argument to `Deterministic`,
            # already delimited by the argument boundary, so it needs
            # no parenthesis (a parenthesized_expression renders empty
            # in positional position under the Python pretty printer).
            shifted = _python_binary_op(
                py,
                "-",
                geom,
                number_literal(py, 1.0),
            )
            det_keyword: list[tuple[str, str]] = []
            det_dims = self._dims_tuple(py, node.plate)
            if det_dims is not None:
                det_keyword.append(("dims", det_dims))
            det = call(
                py,
                attribute(py, ("pymc", "Deterministic")),
                positional=(string_literal(py, node.name), shifted),
                keyword=tuple(det_keyword),
            )
            stmt = assignment(py, lhs_name=node.name, rhs=det)
            py.e(ctx.with_body, stmt, "child_of")
            return

        obs_keyword = list(keyword)
        if dims_tuple is not None:
            obs_keyword.append(("dims", dims_tuple))
        observed_expr = _python_paren(
            py,
            _python_binary_op(
                py,
                "+",
                identifier(py, observed_name),
                number_literal(py, 1.0),
            ),
        )
        obs_keyword.append(("observed", observed_expr))
        rhs = call(
            py,
            callee,
            positional=(string_literal(py, node.name),),
            keyword=tuple(obs_keyword),
        )
        stmt = py.v(py.fresh("es"), "expression_statement")
        py.e(stmt, rhs, "child_of")
        py.e(ctx.with_body, stmt, "child_of")

    def _emit_lkj_cholesky(
        self,
        ctx: _PyMCCtx,
        node: IRSample | IRObserve,
        *,
        observed_name: str | None,
    ) -> None:
        """Emit `LKJCholesky("<name>", n=<dim>, eta=<concentration>)`,
        the grafted runtime helper from
        [`runtime_pymc.py`][quivers.transpile.runtime_pymc].

        PyMC ships no distribution over correlation Cholesky factors
        alone, so the helper supplies the exact factor density. The
        correlation-matrix dimension is a constructor argument rather
        than a `dims` entry: the variable is square over one QVR event
        axis, which a single-name `dims` tuple cannot express."""
        py = ctx.py
        meta = _resolve_meta("LKJCholesky", self.target)
        aliases = _merged_aliases(meta)

        keyword: list[tuple[str, str]] = [
            ("n", number_literal(py, self._lkj_dimension(node.plate))),
        ]
        for arg, arg_name in zip(node.args, node.arg_names, strict=True):
            keyword.append(
                (
                    aliases.get(arg_name, arg_name),
                    self._render_arg(ctx, arg),
                )
            )
        if observed_name is not None:
            keyword.append(("observed", identifier(py, observed_name)))

        rhs = call(
            py,
            identifier(py, _PYMC_RUNTIME_HELPER_FAMILIES["LKJCholesky"]),
            positional=(string_literal(py, node.name),),
            keyword=tuple(keyword),
        )
        if observed_name is None:
            stmt = assignment(py, lhs_name=node.name, rhs=rhs)
        else:
            stmt = py.v(py.fresh("es"), "expression_statement")
            py.e(stmt, rhs, "child_of")
        py.e(ctx.with_body, stmt, "child_of")

    def _lkj_dimension(self, plate: Plate) -> int:
        """The square-matrix dimension of an `LKJCholesky` draw, read
        from the first event dim of the sample's plate."""
        if not plate.event_dims:
            raise UnsupportedConstruct(
                self.target,
                ["family:LKJCholesky:missing-event-dimension"],
            )
        first = plate.event_dims[0]
        if not isinstance(first, DimStatic):
            raise UnsupportedConstruct(
                self.target,
                ["family:LKJCholesky:non-static-event-dimension"],
            )
        return int(first.size)

    def _maybe_broadcast_ref(
        self,
        arg: IRArg,
        *,
        arg_name: str,
        cls_constraints: dict[str, c.Constraint],
        plate: Plate,
        ir: IRProgram,
    ) -> IRArg:
        """Wrap a scalar [`IRArgRef`][quivers.transpile.ir.IRArgRef] in
        [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] when
        the family's `arg_constraints[arg_name]` expects a vector /
        matrix.

        `Lower` skips broadcasting for [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        on the assumption the referenced binding is itself the right
        shape. PyMC's `Dirichlet(a=...)` wants the concentration as a
        vector; when the user passes a scalar program parameter
        (`alpha : Real`), this method injects an `np.full(...)` call so
        the emitted source remains shape-correct."""
        if not isinstance(arg, IRArgRef):
            return arg
        expected = cls_constraints.get(arg_name)
        if expected is None:
            return arg
        if not isinstance(expected, c._IndependentConstraint):
            return arg
        # Only broadcast when the referenced name is a scalar program
        # parameter (an input with no batch or event axes).
        bound = self._input_for(arg.name, ir)
        if bound is None:
            # Not a known input; assume the binding carries the right
            # shape (e.g. a let-bound vector).
            return arg
        if bound.plate.batch_dims or bound.plate.event_dims:
            return arg
        # Build the target shape from the surrounding plate's
        # event_dims (the broadcast target for a vector/matrix arg).
        event = tuple(
            dim.size for dim in plate.event_dims if isinstance(dim, DimStatic)
        )
        if not event or len(event) < expected.event_dim:
            return arg
        target_shape = event[: expected.event_dim]
        return IRArgBroadcast(value=arg, target_shape=target_shape)

    def _input_for(self, name: str, ir: IRProgram) -> IRDataInput | None:
        for inp in ir.inputs:
            if inp.name == name:
                return inp
        return None

    def _dims_tuple(self, py: PyCtx, plate: Plate) -> str | None:
        """Build the `dims=(<batch>..., <event>...)` tuple.

        PyMC's `dims` carries every axis the RV broadcasts over:
        first the batch (replication) axes, then the event axes.
        Both are passed as string axis names matching the coord
        dict's keys."""
        all_dims = (*plate.batch_dims, *plate.event_dims)
        if not all_dims:
            return None
        t = py.v(py.fresh("tup"), "tuple")
        names = [_dim_name(dim) for dim in all_dims]
        for name in names:
            py.e(t, string_literal(py, name), "child_of")
        # Force trailing comma on single-element tuples so pymc parses
        # `("Doc",)` as a tuple rather than a parenthesised string.
        if len(names) == 1:
            py.constraint(t, "ptrace-0", "T(")
            py.constraint(t, "ptrace-1", "Cstring")
            py.constraint(t, "ptrace-2", "T,")
            py.constraint(t, "ptrace-3", "T)")
        return t

    # ----- marginalize: the integrated-density lowering -----

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """No-op at the protocol dispatch point: the renderer drives
        the enumeration through `_emit_marginalize`, which needs the
        PyMC-specific with-body context.

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
        del ctx, node
        return ""

    def _emit_marginalize(self, ctx: _PyMCCtx, node: IRMarginalize) -> None:
        """Integrate the latent out into a [`pymc.Mixture`][pymc.Mixture]
        over the atoms of its finite support.

        One component distribution is built per atom, each preceded by
        that atom's own deterministic bindings, and the atoms are
        weighted by the latent's prior probabilities:

        ```python
        gated_rate = 0.0 * rate
        __marg_z_0 = pymc.Poisson.dist(mu=gated_rate)
        gated_rate = 1.0 * rate
        __marg_z_1 = pymc.Poisson.dist(mu=gated_rate)
        pymc.Mixture(
            "y",
            w=pymc.math.stack([(1 - pi_z), pi_z], axis=-1),
            comp_dists=[__marg_z_0, __marg_z_1],
            observed=y,
        )
        ```

        A mixture is the target-side spelling of the same integral the
        QVR reference computes, and unlike a
        [`pymc.Potential`][pymc.Potential] it is a genuine observed
        random variable, so it carries the block's whole contribution
        to the model's log-density.
        """
        refuse_ungrouped_row_marginalize("qvr-pymc", node)
        py = ctx.py
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        prefix = f"__marg_{node.latent}"
        fibration = marginalize_fibration(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=py.name_plates,
            target=self.target,
        )
        if fibration is not None:
            self._emit_grouped_marginalize(ctx, node, raw, atoms, fibration)
            return
        component_names: list[str] = []
        for position, atom in enumerate(atoms):
            scored = marginalize_body(
                atom.scope, latent=node.latent, target=self.target
            )
            for det in scored.deterministics:
                self._emit_deterministic(ctx, det)
            component = f"{prefix}_{position}"
            py.e(
                ctx.with_body,
                assignment(
                    py,
                    lhs_name=component,
                    rhs=self._component_dist(ctx, scored.observe),
                ),
                "child_of",
            )
            component_names.append(component)
        mixture = call(
            py,
            attribute(py, ("pymc", "Mixture")),
            positional=(string_literal(py, raw.observe.name),),
            keyword=(
                ("w", self._marginal_weights(ctx, node, raw, atoms)),
                (
                    "comp_dists",
                    python_list(
                        py,
                        tuple(identifier(py, name) for name in component_names),
                    ),
                ),
                ("observed", identifier(py, raw.observe.name)),
            ),
        )
        stmt = py.v(py.fresh("es"), "expression_statement")
        py.e(stmt, mixture, "child_of")
        py.e(ctx.with_body, stmt, "child_of")

    def _emit_grouped_marginalize(
        self,
        ctx: _PyMCCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
        fibration: tuple[str, Dim],
    ) -> None:
        """Integrate a grouped block's latent out as a potential.

        A block whose rows are fibred into fewer groups than rows, under
        a prior shared across the groups, keys its accumulator by group
        (`docs/semantics/programs.md` §2.7): each atom's per-row
        log-density is scattered into a ``(|G|, K)`` accumulator through
        the fibration, the log prior is added, and the reduction over
        the atoms is summed over the groups into a
        [`pymc.Potential`][pymc.Potential]. A mixture is a per-row
        integral, which is a different measure here.

        Parameters
        ----------
        ctx : _PyMCCtx
            The render context.
        node : IRMarginalize
            The block.
        raw : MarginalizeBody
            The block's scope, split into its bindings and its observe.
        atoms : tuple[IRMarginalAtom, ...]
            The block's atoms.
        fibration : tuple[str, Dim]
            The fibration's name and the grouping plate's axis.
        """
        py = ctx.py
        prefix = f"__marg_{node.latent}"
        term_names: list[str] = []
        for position, atom in enumerate(atoms):
            scored = marginalize_body(
                atom.scope, latent=node.latent, target=self.target
            )
            for det in scored.deterministics:
                self._emit_deterministic(ctx, det)
            term = f"{prefix}_{position}"
            py.e(
                ctx.with_body,
                assignment(
                    py,
                    lhs_name=term,
                    rhs=call(
                        py,
                        attribute(py, ("pymc", "logp")),
                        positional=(
                            self._component_dist(ctx, scored.observe),
                            identifier(py, scored.observe.name),
                        ),
                    ),
                ),
                "child_of",
            )
            term_names.append(term)
        via, group = fibration
        extent = str(group.size) if isinstance(group, DimStatic) else group.size_name
        weights = self._marginal_weights(ctx, node, raw, atoms)
        py.e(
            ctx.with_body,
            assignment(
                py,
                lhs_name=f"{prefix}_w",
                rhs=call(
                    py, attribute(py, ("pymc", "math", "log")), positional=(weights,)
                ),
            ),
            "child_of",
        )
        stacked = ", ".join(term_names)
        graft_python_statements(
            py.builder,
            f"{prefix} = pymc.math.stack([{stacked}], axis=-1)\n"
            f"{prefix} = pymc.math.dot(pymc.math.eye({extent})[{via}].T, {prefix})\n"
            f'pymc.Potential("{node.latent}", pymc.math.sum('
            f"pymc.math.logsumexp({prefix}_w + {prefix}, axis=-1)))\n",
            ctx.with_body,
            f"marg_{node.latent}_group",
        )

    def _component_dist(self, ctx: _PyMCCtx, observe: IRObserve) -> str:
        """Build the unregistered ``pymc.<Family>.dist(**args)`` an
        atom contributes to the mixture."""
        meta = _resolve_meta(observe.family, self.target)
        dist_class = meta.target_names.get("pymc")
        if dist_class is None:
            raise UnsupportedConstruct(self.target, [f"family:{observe.family}"])
        helper = _PYMC_RUNTIME_HELPER_FAMILIES.get(observe.family)
        chain = (helper, "dist") if helper else ("pymc", dist_class, "dist")
        complement_args = _PYMC_COMPLEMENT_ARGS.get(observe.family, frozenset())
        cls_constraints = getattr(
            meta.distribution_class,
            "arg_constraints",
            {},
        )
        if not isinstance(cls_constraints, dict):
            cls_constraints = {}
        aliases = _merged_aliases(meta)
        keyword: list[tuple[str, str]] = []
        for arg, arg_name in zip(observe.args, observe.arg_names, strict=True):
            arg_to_emit = self._maybe_broadcast_ref(
                arg,
                arg_name=arg_name,
                cls_constraints=cls_constraints,
                plate=observe.plate,
                ir=ctx.ir,
            )
            rendered = self._render_arg(ctx, arg_to_emit)
            renamed = aliases.get(arg_name, arg_name)
            if arg_name in complement_args:
                rendered = _python_paren(
                    ctx.py,
                    _python_binary_op(
                        ctx.py,
                        "-",
                        number_literal(ctx.py, 1.0),
                        rendered,
                    ),
                )
            if arg_name in _PYMC_FACTOR_SCALE_ARGS.get(observe.family, frozenset()):
                rendered, renamed = self._scale_from_factor(ctx, arg_to_emit, rendered)
            keyword.append((renamed, rendered))
        return call(
            ctx.py,
            attribute(ctx.py, chain),
            keyword=tuple(keyword),
        )

    def _marginal_weights(
        self,
        ctx: _PyMCCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
    ) -> str:
        """Mixture weight tensor whose trailing axis runs over the
        atoms.

        A `Categorical` atom set weights atom ``k`` by ``p[k]``, so the
        probability tensor's own trailing axis is already the atom
        axis. A `Bernoulli` atom set weights the atoms 0 and 1 by
        ``1 - p`` and ``p``, which stack into a fresh trailing axis.
        """
        py = ctx.py
        probs = marginal_weight_probs(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=py.name_plates,
            target=self.target,
        )
        probs_vid = self._render_arg(ctx, probs)
        family = atoms[0].weight_family
        if family == "Categorical":
            return probs_vid
        if family == "Bernoulli":
            complement = _python_paren(
                py,
                _python_binary_op(
                    py,
                    "-",
                    number_literal(py, 1.0),
                    self._render_arg(ctx, probs),
                ),
            )
            return call(
                py,
                attribute(py, ("pymc", "math", "stack")),
                positional=(python_list(py, (complement, probs_vid)),),
                keyword=(
                    (
                        "axis",
                        _python_unary_minus(py, number_literal(py, 1)),
                    ),
                ),
            )
        raise UnsupportedConstruct(
            self.target,
            [f"marginalize:weight-family:{family}"],
        )

    # ----- broadcast: np.full((K,), x) for 1D / np.full((R, C), x) for 2D --

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit `np.full((K,), <value>)` for 1D, `np.full((R, C),
        <value>)` for 2D target shapes.

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
        msg = (
            "PyMCRenderer.broadcast: standalone broadcast emission "
            "is internal; use `_render_arg` on an `IRArgBroadcast` "
            "instead."
        )
        raise RuntimeError(msg)

    def _render_broadcast(
        self,
        bag: _PyMCCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> str:
        if len(target_shape) not in (1, 2):
            raise UnsupportedConstruct(
                self.target,
                [f"broadcast:rank-{len(target_shape)}"],
            )
        shape_tuple = bag.py.v(bag.py.fresh("tup"), "tuple")
        for n in target_shape:
            bag.py.e(
                shape_tuple,
                number_literal(bag.py, float(n)),
                "child_of",
            )
        # Trailing-comma constraint for 1-tuples.
        if len(target_shape) == 1:
            bag.py.constraint(shape_tuple, "ptrace-0", "T(")
            bag.py.constraint(shape_tuple, "ptrace-1", "Cinteger")
            bag.py.constraint(shape_tuple, "ptrace-2", "T,")
            bag.py.constraint(shape_tuple, "ptrace-3", "T)")
        callee = attribute(bag.py, ("np", "full"))
        # The fill value types the tensor: an integer literal here
        # builds an integer concentration / rate / scale, which the
        # real-valued family rejects.
        fill = (
            float_literal(bag.py, value.value)
            if isinstance(value, IRArgNumber)
            else self._render_arg(bag, value)
        )
        return call(
            bag.py,
            callee,
            positional=(shape_tuple, fill),
        )

    # ----- arg rendering: dispatch on IRArg variant -----

    def _render_arg(self, ctx: _PyMCCtx, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return number_literal(ctx.py, arg.value)
        if isinstance(arg, IRArgRef):
            return self._render_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self._render_broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(ctx, arg)
        if isinstance(arg, IRArgTransform):
            return self._render_transform(ctx, arg)
        raise UnsupportedConstruct(self.target, [f"arg:{type(arg).__name__}"])

    def _render_ref(self, ctx: _PyMCCtx, ref: IRArgRef) -> str:
        """Emit `name` or `name[i0][i1]...` as a subscript chain."""
        current = identifier(ctx.py, ref.name)
        for idx in ref.indices:
            s = ctx.py.v(ctx.py.fresh("subs"), "subscript")
            ctx.py.e(s, current, "value")
            ctx.py.e(s, self._render_arg(ctx, idx), "subscript")
            current = s
        return current

    def _render_list(self, ctx: _PyMCCtx, lst: IRArgList) -> str:
        """Render a list literal arg as `np.array([e0, e1, ...])`."""
        list_v = ctx.py.v(ctx.py.fresh("list"), "list")
        for e in lst.elements:
            ctx.py.e(list_v, self._render_arg(ctx, e), "child_of")
        return call(
            ctx.py,
            attribute(ctx.py, ("np", "array")),
            positional=(list_v,),
        )

    def _render_matrix(self, ctx: _PyMCCtx, mat: IRArgMatrix) -> str:
        """Render a matrix literal arg as `np.array([[...], [...]])`."""
        outer = ctx.py.v(ctx.py.fresh("list"), "list")
        for row in mat.rows:
            inner = ctx.py.v(ctx.py.fresh("list"), "list")
            for e in row.elements:
                ctx.py.e(inner, self._render_arg(ctx, e), "child_of")
            ctx.py.e(outer, inner, "child_of")
        return call(
            ctx.py,
            attribute(ctx.py, ("np", "array")),
            positional=(outer,),
        )

    def _render_family_ref(self, ctx: _PyMCCtx, ref: IRArgFamilyRef) -> str:
        """Emit `pymc.<base>.dist(<base_args>)` for a referenced
        morphism whose `init_family` clause names a base distribution.

        Used by wrappers (`Truncated`, `Mixture`, etc.): the wrapper
        renderer emits the wrapped distribution as a frozen `.dist()`
        call so PyMC's wrapper can consume it."""
        morph = ctx.ctx.morphisms.get(ref.name)
        if morph is None:
            raise UnsupportedConstruct(
                self.target,
                [f"family_ref:{ref.name}: morphism not in context"],
            )
        init = morph.init_family
        if init is None:
            raise UnsupportedConstruct(
                self.target,
                [f"family_ref:{ref.name}: morphism has no `~ Family(...)` init clause"],
            )
        base_meta = _resolve_meta(init.family, self.target)
        base_dist_class = base_meta.target_names.get("pymc")
        if base_dist_class is None:
            raise UnsupportedConstruct(
                self.target,
                [f"family_ref:{init.family}: pymc target name missing"],
            )
        dist_attr = attribute(ctx.py, ("pymc", base_dist_class, "dist"))
        positional = tuple(
            arg_expr(ctx.py, _draw_arg_to_wire(value))
            for _, value in bind_family_arguments(init.family, init.args or ())
        )
        return call(ctx.py, dist_attr, positional=positional)

    def _scale_from_factor(
        self, ctx: _PyMCCtx, factor: IRArg, rendered: str
    ) -> tuple[str, str]:
        """A Wishart's scale ``L @ L.T`` from its Cholesky factor's name.

        PyMC's `Wishart` takes the scale matrix ``V`` alone, so a factor
        reaches it multiplied out.

        Parameters
        ----------
        ctx : _PyMCCtx
            The render context.
        factor : IRArg
            The factor, a whole named matrix.
        rendered : str
            The factor's rendered vertex, the product's left operand.

        Returns
        -------
        tuple[str, str]
            The rendered product and the keyword it fills, ``V``.

        Raises
        ------
        UnsupportedConstruct
            If the factor is not a whole named matrix.
        """
        if not isinstance(factor, IRArgRef) or factor.indices:
            raise UnsupportedConstruct(
                self.target,
                [
                    "family:Wishart:scale_tril: PyMC's Wishart takes its scale "
                    "multiplied out from a whole named Cholesky factor, and "
                    f"this slot carries a {type(factor).__name__}"
                ],
            )
        product = _python_binary_op(
            ctx.py,
            "@",
            rendered,
            attribute(ctx.py, (factor.name, "T")),
        )
        return product, "V"

    def _render_transform(self, ctx: _PyMCCtx, t: IRArgTransform) -> str:
        """Renderer-applied arithmetic transform on an arg.

        PyMC's [`Potential`][pymc.Potential] does not currently use
        `_ALIAS_TRANSFORMS`, but the renderer ships the mechanism so
        future parameterisation renames plug in without a separate
        dispatch surface."""
        inner = self._render_arg(ctx, t.inner)
        if t.transform == "inv":
            one = number_literal(ctx.py, 1.0)
            return _bin_op(ctx.py, one, "/", inner)
        if t.transform == "neg":
            return _unary_op(ctx.py, "-", inner)
        if t.transform == "inv_square":
            square = _bin_op(ctx.py, inner, "*", inner)
            one = number_literal(ctx.py, 1.0)
            return _bin_op(ctx.py, one, "/", square)
        if t.transform == "log":
            return call(
                ctx.py,
                attribute(ctx.py, ("np", "log")),
                positional=(inner,),
            )
        if t.transform == "exp":
            return call(
                ctx.py,
                attribute(ctx.py, ("np", "exp")),
                positional=(inner,),
            )
        raise UnsupportedConstruct(self.target, [f"transform:{t.transform}"])

    # ----- deterministic / score -----

    def _emit_deterministic(self, ctx: _PyMCCtx, node: IRDeterministic) -> None:
        """Emit `<name> = <expr>` for a let-bound deterministic
        computation."""
        rhs = render_deterministic_python(ctx.py, node)
        stmt = assignment(ctx.py, lhs_name=node.name, rhs=rhs)
        ctx.py.e(ctx.with_body, stmt, "child_of")

    def _emit_score_step(self, ctx: _PyMCCtx, node: IRScore) -> None:
        """Emit `<name> = <expr>` then `pymc.Potential("<name>",
        <name>)` for a score step.

        PyMC's [`Potential`][pymc.Potential] adds an arbitrary scalar
        log-density factor to the model's joint, mirroring NumPyro's
        `factor` primitive."""
        rhs = render_let_expr_python(ctx.py, node.expr)
        stmt = assignment(ctx.py, lhs_name=node.name, rhs=rhs)
        ctx.py.e(ctx.with_body, stmt, "child_of")
        factor_call = call(
            ctx.py,
            attribute(ctx.py, ("pymc", "Potential")),
            positional=(
                string_literal(ctx.py, node.name),
                identifier(ctx.py, node.name),
            ),
        )
        fstmt = ctx.py.v(ctx.py.fresh("es"), "expression_statement")
        ctx.py.e(fstmt, factor_call, "child_of")
        ctx.py.e(ctx.with_body, fstmt, "child_of")

    # ----- Renderer protocol's `sample` (proxies into _emit_sample) -----

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
        """`Renderer` protocol entry: PyMC's emit is structurally
        joined with declare (the constructor call IS the declaration).
        Not called by the PyMC walker (which calls `_emit_sample`
        directly); kept here so external callers using the abstract
        protocol can drive the renderer.

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
        del ctx, name, family, args, arg_names, constraint, plate, observed
        msg = (
            "PyMCRenderer.sample is not called directly; PyMC dispatch "
            "goes through `_emit_sample` to thread the per-walk bag."
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PyMCCtx:
    """Renderer-internal mutable bag threaded through the dispatch
    points: shared [`RenderContext`][quivers.transpile.renderers.RenderContext],
    the `PyCtx` adapter,
    and per-walk state (`with`-body block, parent IR for input
    lookups)."""

    def __init__(
        self,
        *,
        ctx: RenderContext,
        py: PyCtx,
        ir: IRProgram,
    ) -> None:
        self.ctx = ctx
        self.py = py
        self.ir = ir
        self.with_body: str = ""
        self.operations_bound: set[str] = set()


def _dim_name(dim: Dim) -> str:
    """Return a [`Dim`][quivers.transpile.ir.Dim]'s axis name as a
    plain `str`.

    `Dim` is a [`dx.TaggedUnion`][didactic.api.TaggedUnion] base whose
    `.name` slot types as `FieldValue`; narrowing the variant lets the
    type checker prove the slot is a `str`."""
    if isinstance(dim, (DimStatic, DimDynamic)):
        return dim.name
    msg = f"unknown Dim kind: {type(dim).__name__}"
    raise UnsupportedConstruct("qvr-pymc", [msg])


def _resolve_meta(family: str, target: str) -> FamilyMeta:
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(target, [f"family:{family}"])
    return meta


#: Families PyMC does not ship, resolved to a grafted runtime helper
#: called by bare name. The value is both the emitted call name and
#: the top-level function in
#: [`runtime_pymc.py`][quivers.transpile.runtime_pymc] that the
#: renderer grafts into the module.
_PYMC_RUNTIME_HELPER_FAMILIES: dict[str, str] = {
    "ContinuousBernoulli": "ContinuousBernoulli",
    "LKJCholesky": "LKJCholesky",
}


#: Families whose PyMC arg carries the complement of the torch
#: parameter: torch `NegativeBinomial(total_count, probs)` has
#: pmf proportional to `(1 - probs)**total_count * probs**k`, while
#: PyMC `NegativeBinomial(n, p)` uses `p**n * (1 - p)**k`, so PyMC's
#: `p` is `1 - probs`.
_PYMC_COMPLEMENT_ARGS: dict[str, frozenset[str]] = {
    "NegativeBinomial": frozenset({"probs"}),
}

#: Families whose PyMC scale is the product ``L @ L.T`` of the Cholesky
#: factor the call supplies: PyMC's `Wishart` has no factor parameter.
_PYMC_FACTOR_SCALE_ARGS: dict[str, frozenset[str]] = {
    "Wishart": frozenset({"scale_tril"}),
}


def _merged_aliases(meta: FamilyMeta) -> dict[str, str]:
    """Return the PyMC alias map for `meta`: the per-family arg
    renames keyed under `"pymc"` in
    [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META], which
    map each torch canonical argument name to the keyword PyMC's
    constructor expects."""
    return dict(meta.arg_aliases.get("pymc", {}))


def _draw_arg_to_wire(a: object) -> str | float:
    """Coerce a referenced morphism's init-family argument to the
    `arg_expr`-acceptable wire form."""
    if isinstance(a, (int, float)):
        return float(a)
    if isinstance(a, str):
        return a
    return atom_value(a)


def _bin_op(py: PyCtx, lhs: str, op: str, rhs: str) -> str:
    """Build a `binary_operator` vertex with `left`/`operator`/`right`."""
    b = py.v(py.fresh("bop"), "binary_operator")
    py.e(b, lhs, "left")
    op_vid = py.v(py.fresh("op"), "operator")
    py.literal(op_vid, op)
    py.e(b, op_vid, "operator")
    py.e(b, rhs, "right")
    return b


def _unary_op(py: PyCtx, op: str, operand: str) -> str:
    """Build a `unary_operator` vertex with `operator`/`argument`."""
    u = py.v(py.fresh("uop"), "unary_operator")
    op_vid = py.v(py.fresh("op"), "operator")
    py.literal(op_vid, op)
    py.e(u, op_vid, "operator")
    py.e(u, operand, "argument")
    return u


# ---------------------------------------------------------------------------
# Runtime-helper graft: embed distributions PyMC does not ship.
# ---------------------------------------------------------------------------


_RUNTIME_PYMC_PATH = pathlib.Path(__file__).resolve().parent.parent / "runtime_pymc.py"


def _load_runtime_pymc_helpers() -> tuple[panproto.Schema, tuple[str, ...]]:
    """Parse [`runtime_pymc.py`][quivers.transpile.runtime_pymc]
    through panproto's Python tree-sitter grammar at module-load time
    and return the resulting schema plus the vertex ids of its
    top-level `function_definition` roots.

    The renderer grafts these subtrees into the per-render schema so
    the emitted module carries real function definitions (no source
    string, no `exec`). Only the function definitions are copied; the
    helper module's own imports stay behind, matching the emitted
    program's convention of resolving `pymc` / `np` / `pt` as free
    names in the run namespace."""
    schema = parser_registry().parse_with_protocol(
        "python",
        _RUNTIME_PYMC_PATH.read_bytes(),
        str(_RUNTIME_PYMC_PATH),
    )
    module_id = next((v.id for v in schema.vertices if v.kind == "module"), None)
    if module_id is None:
        raise RuntimeError(
            f"no module root parsed from {_RUNTIME_PYMC_PATH}; the "
            "renderer expects a valid Python module of grafted helpers."
        )
    roots = tuple(
        v.id
        for v in schema.vertices
        if v.kind == "function_definition"
        and any(
            e.src == module_id and e.tgt == v.id and e.kind == "child_of"
            for e in schema.edges
        )
    )
    if not roots:
        raise RuntimeError(
            f"no top-level function definitions found in "
            f"{_RUNTIME_PYMC_PATH}; the renderer expects it as the "
            "source of truth for grafted PyMC runtime helpers."
        )
    return schema, roots


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


_RUNTIME_PYMC_SCHEMA, _RUNTIME_PYMC_ROOTS = _load_runtime_pymc_helpers()
_RUNTIME_PYMC_REACHABLE: set[str] = set()
for _root in _RUNTIME_PYMC_ROOTS:
    _RUNTIME_PYMC_REACHABLE |= _subtree_vertex_ids(_RUNTIME_PYMC_SCHEMA, _root)

#: The grafted vertices in the parsed schema's own vertex order, so a
#: render assigns fresh ids, and the pretty printer emits the helper
#: definitions, in a fixed sequence.
_RUNTIME_PYMC_SUBTREE: tuple[str, ...] = tuple(
    v.id for v in _RUNTIME_PYMC_SCHEMA.vertices if v.id in _RUNTIME_PYMC_REACHABLE
)


def _ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any [`IRSample`][quivers.transpile.ir.IRSample] or
    [`IRObserve`][quivers.transpile.ir.IRObserve] in `body` (including
    nested [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
    scopes) draws from `family`."""
    for node in body:
        if isinstance(node, (IRSample, IRObserve)) and node.family == family:
            return True
        if isinstance(node, IRMarginalize) and _ir_uses_family(node.scope, family):
            return True
    return False


def _graft_runtime_pymc_helpers(py: PyCtx) -> None:
    """Graft every top-level `function_definition` subtree from
    [`runtime_pymc.py`][quivers.transpile.runtime_pymc] into the
    per-render schema as a `child_of` of the `mod` module root.

    Each vertex, constraint, and internal edge of the subtrees is
    copied under fresh ids, so a render that grafts more than once
    keeps the schema's vertex-id invariant."""
    src_schema = _RUNTIME_PYMC_SCHEMA
    id_map: dict[str, str] = {}
    for old in _RUNTIME_PYMC_SUBTREE:
        new = py.fresh("rt")
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        py.v(new, kind)
        for cstr in src_schema.constraints_for(old):
            py.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            py.e(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for root in _RUNTIME_PYMC_ROOTS:
        py.e("mod", id_map[root], "child_of")


__all__ = ["PyMCRenderer"]
