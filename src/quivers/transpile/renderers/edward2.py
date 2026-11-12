"""Edward2 renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to
TFP-Edward2 Python source under the `python` tree-sitter grammar.

Edward2's idiom is a plain ``def model(<inputs>=None, ...): ...`` whose
body constructs random variables via
``edward2.<Family>(args, sample_shape=[B0, B1, ...], name="<name>")``.
There is no separate data block; every exogenous identifier (program
parameters, observed values, index maps) becomes a function parameter
with a ``=None`` default. Discrete-latent integration scopes lower to
explicit ``IRSample(latent)`` plus the inlined scope body (Edward2
samples discrete latents natively; ``log_sum_exp`` enumeration is the
Stan-specific path).

Scalar-to-vector broadcast for a per-axis parameter (the ``alpha`` in
``Dirichlet(alpha)`` against the Topic plate) is emitted as
``tf.fill([K], <value>)``; for 2D targets as ``tf.fill([R, C], <value>)``.
``IRArgList`` literals render as plain Python lists ``[e0, e1, ...]``
(TF accepts those for vector-arg positions); ``IRArgMatrix`` as nested
lists ``[[...], [...]]``.

The renderer reads target distribution names from
``FAMILY_META[family].target_names["edward2"]`` and applies per-arg
renames from ``arg_aliases["edward2"]``; no per-family branching
appears here.
"""

from __future__ import annotations

import pathlib

import panproto

from quivers.dsl.ast_nodes import Expr, MorphismDecl
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import EmitPretty, target_protocol
from quivers.transpile.renderers._ranks import name_rank_map
from quivers.transpile.renderers._python_helpers import (
    MarginalizeBody,
    PyCtx,
    assignment,
    attribute,
    call,
    factor_tower_names,
    float_literal,
    function_def,
    identifier,
    marginal_atom_axis,
    marginal_support_size,
    marginal_weight_probs,
    marginalize_body,
    marginalize_index_map,
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
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    CSBoolean,
    CSIntegerInterval,
    CSInterval,
    CSNonnegativeInteger,
    CSPositive,
    CSPositiveInteger,
    CSReal,
    CSUnitInterval,
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
    is_real_cov_matrix,
    is_real_matrix,
    is_real_simplex,
    is_real_vector,
)
from quivers.transpile.renderers._base import (
    refuse_ungrouped_row_marginalize,
    BlockKind,
    IRMarginalAtom,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    ir_uses_family,
    logit_weighted,
    mixture_normal_components,
)
from quivers.transpile.renderers._qiec import (
    emit_call_python,
    graft_python_statements,
    qiec_families_used,
    render_computations_dynamic,
)
from quivers.transpile.qiec_ir import IRQiecModule


_TARGET = "edward2"
_BACKEND_KEY = f"qvr-{_TARGET}"

#: The factor helper grafted into a module whose plan scores or calls:
#: a one-point distribution whose log density is the scored weight.
_FACTOR_HELPER = pathlib.Path(__file__).parent.parent / "runtime_factor_edward2.py"


class _Edward2Ctx(RenderContext):
    """The render context with the module the plan's calls read.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema builder the renderer emits into.
    morphisms : dict[str, MorphismDecl]
        The module's morphism declarations, by name.
    defines : dict[str, Expr]
        The module's define expressions, by name.
    module : IRQiecModule
        The checked module the plan was derived from.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        morphisms: dict[str, MorphismDecl],
        defines: dict[str, Expr],
        *,
        module: IRQiecModule,
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=defines)
        self.module = module
        #: The bodies whose native operation table is already bound.
        self.operations_bound: set[str] = set()


#: Edward2-side argument injection for QVR families whose underlying
#: torch distribution carries fewer parameters than TFP's same-named
#: distribution. ``HalfCauchy(scale)`` maps to TFP's
#: ``HalfCauchy(loc, scale)`` (loc is required); the renderer prepends
#: ``IRArgNumber(0)`` under the ``"loc"`` arg name so TFP sees a
#: complete (loc, scale) pair. The injected ``loc=0`` is the standard
#: half-distribution origin and matches the QVR-side semantics for
#: HalfCauchy / HalfNormal.
_PREPEND_LOC_ZERO: frozenset[str] = frozenset({"HalfCauchy"})


#: Edward2-local argument renames layered on top of
#: ``FAMILY_META[family].arg_aliases["edward2"]``. TFP's ``Pareto``
#: names its shape parameter ``concentration`` where torch / QVR name
#: it ``alpha``; without the rename the emitted keyword ``alpha=`` is
#: not a valid TFP ``Pareto`` constructor argument and construction
#: raises ``TypeError``. The value is unchanged: torch's ``alpha`` and
#: TFP's ``concentration`` are the same Pareto shape parameter.
_EDWARD2_ARG_ALIASES: dict[str, dict[str, str]] = {
    "Pareto": {"alpha": "concentration"},
}


#: QVR arguments that carry an integer support on the torch side but
#: that TFP requires in floating point, keyed by family.
#:
#: `Binomial` and `BetaBinomial` are the only two families in
#: `FAMILY_META` whose `distribution_class.arg_constraints` marks a
#: parameter as integer-valued while the remaining parameters are
#: continuous, and both TFP constructors unify every parameter dtype
#: through `dtype_util.common_dtype`. A trial count supplied as an
#: `int32` tensor beside `float32` concentrations therefore raises
#: `TypeError: Found incompatible dtypes` before anything is scored,
#: so the renderer casts the count to `tf.float32` at the call site.
#: The cast is exact: a trial count is a small non-negative integer
#: and `float32` represents every such value below 2**24 without
#: rounding, so the emitted density is the one the QVR measure names.
_EDWARD2_FLOAT_COUNT_ARGS: dict[str, frozenset[str]] = {
    "Binomial": frozenset({"total_count"}),
    "BetaBinomial": frozenset({"total_count"}),
}


#: Families whose TFP constructor takes a shape argument the generic
#: keyword path does not supply (``MatrixNormal`` needs Cholesky
#: ``LinearOperator``s, ``LKJCholesky`` a leading matrix dimension,
#: ``GP`` a kernel). Each has a dedicated builder for the sample /
#: observe path; none has one for a mixture component, so a
#: `marginalize` over such a family raises rather than emit a call
#: missing a required argument.
_NO_GENERIC_COMPONENT_FAMILIES: frozenset[str] = frozenset(
    {"MatrixNormal", "LKJCholesky", "GP"}
)


class Edward2Renderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to TFP
    Edward2 Python source.

    Output shape (single function, all inputs defaulted to ``None``):

    ```python
    def model(alpha=None, beta=None, word_idx=None, w=None):
        theta = edward2.Dirichlet(tf.fill([3], alpha),
                                  sample_shape=[20], name="theta")
        phi = edward2.Dirichlet(tf.fill([200], beta),
                                sample_shape=[3], name="phi")
        z = edward2.Categorical(probs=theta,
                                sample_shape=[20], name="z")
        w = edward2.Categorical(probs=phi[z[word_idx]], name="w")
        return theta
    ```

    The walk overrides
    [`RendererBase.render`][quivers.transpile.renderers.RendererBase.render]
    because Edward2's program shape (single function, no block
    structure) does not map onto the inherited block-by-block default
    walk.
    """

    target: str = _TARGET

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("python")

    # ------------------------------------------------------------------
    # Top-level render
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
        py = PyCtx(
            sb,
            cards=dict(ir.cards),
            target="edward2",
            name_event_rank=name_event_rank_map(ir),
            name_rank=name_rank_map(ir),
            gather_symbol=("tf", "gather"),
            factor_towers=factor_tower_names(ir),
            name_plates=name_plate_map(ir),
        )
        ctx = _Edward2Ctx(sb=sb, morphisms={}, defines={}, module=ir.module)

        sb.vertex("mod", "module")
        # A marginalize block reduces to a `MixtureSameFamily` whose
        # two component arguments are bare TFP distributions rather
        # than Edward2 random variables (an Edward2 constructor would
        # register a second, unobserved site on the trace). TFP is a
        # hard dependency of Edward2, so the emitted module imports it
        # directly when a marginalize is present. A score or a call
        # needs the factor helper, a TFP distribution.
        factors = _ir_has_factor(ir.body)
        if (
            _ir_has_marginalize(ir.body)
            or ir_uses_family(ir.body, "MixtureNormal")
            or qiec_families_used(ir)
            or factors
        ):
            self._emit_tfp_import(py)
        if factors:
            graft_python_statements(
                sb, _FACTOR_HELPER.read_text(), "mod", "qiec_factor"
            )
        body_vid = py.v(py.fresh("body"), "block")
        if not ir.body:
            noop = py.v(py.fresh("pass"), "pass_statement")
            py.e(body_vid, noop, "child_of")
        param_names = tuple(inp.name for inp in ir.inputs)
        fn = function_def(
            py, name="model", default_params=param_names, body_vid=body_vid
        )
        sb.edge("mod", fn, "child_of")

        # Track input constraint info so the renderer can decide whether
        # to wrap a scalar `IRArgRef` into a `tf.fill(...)` broadcast.
        input_specs: dict[str, ConstraintSpec] = {
            inp.name: inp.constraint for inp in ir.inputs
        }
        bindings: dict[str, _Binding] = {
            inp.name: _Binding(constraint=inp.constraint, plate=inp.plate)
            for inp in ir.inputs
        }

        # TensorFlow's ops are dtype-strict: an integer-valued tensor
        # supplied for a `Real`-declared covariate makes every mixed
        # expression it reaches (`20.0 * (t - tau)`) raise rather than
        # promote, the way JAX / PyTorch / PyTensor would. Casting each
        # real-declared input once at the top of the body makes the
        # emitted model self-contained whatever dtype the caller
        # supplies, mirroring the integer-index coercion the PyTorch
        # renderer emits for the mirror-image case.
        for name in _real_input_names(ir):
            py.e(
                body_vid,
                assignment(
                    py,
                    lhs_name=name,
                    rhs=_cast_float32(py, identifier(py, name)),
                ),
                "child_of",
            )

        for node in ir.body:
            self._emit_node(py, ctx, body_vid, node, input_specs, bindings)

        render_computations_dynamic(sb, ir, target=self.target, root="mod")
        return sb.build()

    def emit_bytes(self, ir: IRProgram) -> bytes:
        """Convenience: render `ir` and run `emit_pretty` to bytes.

        Parameters
        ----------
        ir
            The lowered program.

        Returns
        -------
        bytes
            The target program's source.
        """
        schema = self.render(ir)
        return EmitPretty("python")(schema)

    # ------------------------------------------------------------------
    # IRNode dispatch
    # ------------------------------------------------------------------

    def _emit_node(
        self,
        py: PyCtx,
        ctx: _Edward2Ctx,
        body_vid: str,
        node: IRNode,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> None:
        if isinstance(node, IRDataInput):
            # Inputs are surfaced as function parameters; nothing
            # to emit in the body.
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(py, body_vid, node)
                bindings[node.name] = _Binding(
                    constraint=node.constraint, plate=node.plate
                )
                return
            rhs = self._dist_call(
                py,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                input_specs=input_specs,
                bindings=bindings,
                observed_name=None,
            )
            py.e(
                body_vid,
                assignment(py, lhs_name=node.name, rhs=rhs),
                "child_of",
            )
            bindings[node.name] = _Binding(constraint=node.constraint, plate=node.plate)
            return
        if isinstance(node, IRObserve):
            rhs = self._dist_call(
                py,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                # An observation is a random variable over its plate,
                # whose value is the observed data: a later binding
                # that reads the name reads the data, and a run with no
                # data draws the whole plate.
                plate=node.plate,
                input_specs=input_specs,
                bindings=bindings,
                observed_name=node.name,
            )
            py.e(
                body_vid,
                assignment(py, lhs_name=node.name, rhs=rhs),
                "child_of",
            )
            bindings[node.name] = _Binding(constraint=node.constraint, plate=node.plate)
            return
        if isinstance(node, IRDeterministic):
            asn = py.v(py.fresh("asn"), "assignment")
            lhs = identifier(py, node.name)
            py.e(asn, lhs, "left")
            py.e(asn, render_deterministic_python(py, node), "right")
            py.e(body_vid, asn, "child_of")
            bindings[node.name] = _Binding(constraint=node.constraint, plate=node.plate)
            return
        if isinstance(node, IRScore):
            self._emit_score_node(py, body_vid, node)
            return
        if isinstance(node, IRCall):
            emit_call_python(py, body_vid, node, ctx.module, ctx.operations_bound)
            # A call's result is laid out over the plate its binding
            # declares, so a site reading it knows its broadcast shape.
            bindings[node.name] = _Binding(constraint=CSReal(), plate=node.plate)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize(py, body_vid, node, input_specs, bindings)
            return
        if isinstance(node, IRReturn):
            self._emit_return_statement(py, body_vid, node.names)
            return
        raise UnsupportedConstruct(_BACKEND_KEY, [f"node:{type(node).__name__}"])

    def _emit_gp_block(
        self,
        py: PyCtx,
        body_vid: str,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as three Edward2 statements:

            __gp_mean_<name> = tf.zeros([N])
            __gp_cov_<name>  = tf.exp(-0.5 * (diff)*(diff) / (ls*ls))
                                + jitter * tf.eye(N)
            <name> = edward2.MultivariateNormalFullCovariance(
                        loc=__gp_mean_<name>,
                        covariance_matrix=__gp_cov_<name>, name="<name>")

        Uses tf.zeros / tf.exp / tf.eye math. The kernel matrix is
        the squared-exponential covariance plus diagonal jitter.
        Parens wrap the diff and squared-length-scale subexpressions
        so the Python pretty-printer keeps operator precedence.
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = tf.zeros([N])
        mean_rhs = call(
            py,
            attribute(py, ("tf", "zeros")),
            positional=(number_literal(py, n),),
        )
        py.e(
            body_vid,
            assignment(py, lhs_name=mean_name, rhs=mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = tf.exp(-0.5 * (diff)*(diff) / (ls*ls))
        #                    + jitter * tf.eye(N)
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
            attribute(py, ("tf", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            py,
            attribute(py, ("tf", "eye")),
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
            body_vid,
            assignment(py, lhs_name=cov_name, rhs=cov_rhs),
            "child_of",
        )
        # <name> = edward2.MultivariateNormalFullCovariance(
        #     loc=__gp_mean_<name>, covariance_matrix=__gp_cov_<name>,
        #     name="<name>")
        mvn_call = call(
            py,
            attribute(
                py,
                ("edward2", "MultivariateNormalFullCovariance"),
            ),
            keyword=(
                ("loc", identifier(py, mean_name)),
                ("covariance_matrix", identifier(py, cov_name)),
                ("name", string_literal(py, node.name)),
            ),
        )
        py.e(
            body_vid,
            assignment(py, lhs_name=node.name, rhs=mvn_call),
            "child_of",
        )

    def _emit_score_node(self, py: PyCtx, body_vid: str, node: IRScore) -> None:
        """``<name> = <expr>; _qvr_qiec_factor("<name>", <name>)``.

        Edward2 reads a joint off its tape as the sum of each traced
        variable's log density, so the weight is traced as a factor
        variable, a one-point distribution whose log density is the
        weight.
        """
        asn = py.v(py.fresh("asn"), "assignment")
        lhs = identifier(py, node.name)
        py.e(asn, lhs, "left")
        py.e(asn, render_let_expr_python(py, node.expr), "right")
        py.e(body_vid, asn, "child_of")
        factor_call = call(
            py,
            identifier(py, "_qvr_qiec_factor"),
            positional=(
                string_literal(py, node.name),
                identifier(py, node.name),
            ),
        )
        py.e(body_vid, factor_call, "child_of")

    def _emit_return_statement(
        self, py: PyCtx, body_vid: str, names: tuple[str, ...]
    ) -> None:
        """Emit ``return <name>`` / ``return <a>, <b>`` at the end of
        the model function body.

        Edward2 dispatches its own IR walk through
        `_emit_node`
        rather than the base walker, so this carries the schema
        context the walk threads (`py`, `body_vid`) instead of the
        `RenderContext` signature the base class declares.
        """
        if not names:
            return
        rs = py.v(py.fresh("ret"), "return_statement")
        if len(names) == 1:
            py.e(rs, identifier(py, names[0]), "child_of")
        else:
            elist = py.v(py.fresh("elist"), "expression_list")
            for var in names:
                py.e(elist, identifier(py, var), "child_of")
            py.e(rs, elist, "child_of")
        py.e(body_vid, rs, "child_of")

    # ------------------------------------------------------------------
    # Dispatch points (Renderer protocol)
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
        """No-op: Edward2 has no separate declarations.

        Data inputs are emitted as function parameters by ``render``;
        sample / observe declarations are subsumed into the
        ``edward2.<Family>(...)`` call assigned to the bound name.

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
        """Build the ``edward2.<Family>(...)`` call vertex.

        The IR-walk in ``render`` calls this from a fresh `PyCtx`
        wrapping ``ctx.sb``. Edward2 has no native ``obs=`` keyword;
        observations are surfaced by carrying them as default-None
        function parameters and conditioning via the caller's
        interceptor mechanism, so ``observed`` does not change the
        emitted call.

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
        del constraint, observed
        py = PyCtx(ctx.sb, target="edward2")
        return self._dist_call(
            py,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            input_specs={},
            bindings={},
            observed_name=None,
        )

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """No-op at the protocol dispatch point: the renderer drives
        the enumeration through `_emit_marginalize`, which needs the
        binding tables the top-level walk threads.

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

    def _emit_marginalize(
        self,
        py: PyCtx,
        body_vid: str,
        node: IRMarginalize,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> None:
        """Integrate the latent out into a single
        ``edward2.MixtureSameFamily`` observed variable.

        Each atom of the latent's support contributes one row of the
        component distribution's trailing batch axis, so the atoms are
        materialised argument-by-argument and stacked:

        ```python
        gated_rate = 0.0 * rate
        __marg_z_a0_0 = gated_rate
        gated_rate = 1.0 * rate
        __marg_z_a0_1 = gated_rate
        __marg_z_a0 = tf.stack([__marg_z_a0_0, __marg_z_a0_1], axis=-1)
        y = edward2.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=tf.stack([(1 - pi_z), pi_z], axis=-1)
            ),
            components_distribution=tfp.distributions.Poisson(
                rate=__marg_z_a0
            ),
            name="y",
        )
        ```

        The components and the mixing distribution are bare TFP
        distributions: an Edward2 constructor would register each as
        its own random variable on the trace, and the block denotes
        one observed variable, not three.
        """
        refuse_ungrouped_row_marginalize("qvr-edward2", node)
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        observe = raw.observe
        if observe.family in _NO_GENERIC_COMPONENT_FAMILIES:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [
                    f"marginalize:component-family:{observe.family}: the "
                    f"family's TFP constructor takes a shape argument "
                    f"the generic component builder does not supply"
                ],
            )
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        meta = self._lookup_meta(observe.family)
        expected_event = _event_shape(
            Plate(event_dims=observe.plate.event_dims, batch_dims=())
        )
        prefix = f"__marg_{node.latent}"
        index_map = marginalize_index_map(
            node,
            observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=py.name_plates,
            target=self.target,
        )
        if index_map is not None:
            self._emit_grouped_marginalize(
                py, body_vid, node, raw, atoms, index_map, input_specs, bindings
            )
            return
        per_arg: list[list[str]] = [[] for _ in observe.arg_names]
        for position, atom in enumerate(atoms):
            scored = marginalize_body(
                atom.scope, latent=node.latent, target=self.target
            )
            for det in scored.deterministics:
                asn = py.v(py.fresh("asn"), "assignment")
                py.e(asn, identifier(py, det.name), "left")
                py.e(asn, render_deterministic_python(py, det), "right")
                py.e(body_vid, asn, "child_of")
                bindings[det.name] = _Binding(
                    constraint=det.constraint, plate=det.plate
                )
            for slot, arg in enumerate(scored.observe.args):
                held = f"{prefix}_a{slot}_{position}"
                py.e(
                    body_vid,
                    assignment(
                        py,
                        lhs_name=held,
                        rhs=self._render_arg(
                            py,
                            arg,
                            expected_arg_event=expected_event,
                            arg_position=slot,
                            meta=meta,
                            input_specs=input_specs,
                            bindings=bindings,
                        ),
                    ),
                    "child_of",
                )
                per_arg[slot].append(held)
        axis = marginal_atom_axis(observe.family, observe.arg_names, target=self.target)
        stacked: list[IRArg] = []
        for slot, held_names in enumerate(per_arg):
            stacked_name = f"{prefix}_a{slot}"
            py.e(
                body_vid,
                assignment(
                    py,
                    lhs_name=stacked_name,
                    rhs=_tf_stack(
                        py,
                        tuple(identifier(py, n) for n in held_names),
                        axis,
                    ),
                ),
                "child_of",
            )
            stacked.append(IRArgRef(name=stacked_name))
        components = self._dist_call(
            py,
            name=observe.name,
            family=observe.family,
            args=tuple(stacked),
            arg_names=observe.arg_names,
            plate=Plate(event_dims=(), batch_dims=()),
            input_specs=input_specs,
            bindings=bindings,
            observed_name=None,
            callee_chain=("tfp", "distributions"),
        )
        mixing = call(
            py,
            attribute(py, ("tfp", "distributions", "Categorical")),
            keyword=(
                ("logits", self._marginal_logit_weights(py, node, raw, atoms))
                if logit_weighted(atoms[0])
                else ("probs", self._marginal_weights(py, node, raw, atoms)),
            ),
        )
        py.e(
            body_vid,
            assignment(
                py,
                lhs_name=observe.name,
                rhs=call(
                    py,
                    attribute(py, ("edward2", "MixtureSameFamily")),
                    keyword=(
                        ("mixture_distribution", mixing),
                        ("components_distribution", components),
                        ("name", string_literal(py, observe.name)),
                    ),
                ),
            ),
            "child_of",
        )
        bindings[observe.name] = _Binding(
            constraint=observe.constraint, plate=observe.plate
        )

    def _marginal_weights(
        self,
        py: PyCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
    ) -> str:
        """Mixing-probability tensor whose trailing axis runs over the
        atoms.

        A `Categorical` atom set weights atom ``k`` by ``p[k]``, so the
        probability tensor's own trailing axis is already the atom
        axis. A `Bernoulli` atom set weights the atoms 0 and 1 by
        ``1 - p`` and ``p``, which stack into a fresh trailing axis.
        """
        probs = marginal_weight_probs(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=py.name_plates,
            target=self.target,
        )
        probs_vid = self._weight_ref_expr(py, probs)
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
                    self._weight_ref_expr(py, probs),
                ),
            )
            return _tf_stack(py, (complement, probs_vid), -1)
        raise UnsupportedConstruct(
            _BACKEND_KEY,
            [f"marginalize:weight-family:{family}"],
        )

    def _marginal_logit_weights(
        self,
        py: PyCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
    ) -> str:
        """Log-weight tensor of a Bernoulli atom set given by its logits.

        The atoms 0 and 1 are weighted by ``log_sigmoid(-l)`` and
        ``log_sigmoid(l)``, computed in log space and stacked into a
        fresh trailing axis; they are normalized, so they serve both as
        a `Categorical`'s logits and as log weights.

        Parameters
        ----------
        py : PyCtx
            The emission context.
        node : IRMarginalize
            The block.
        raw : MarginalizeBody
            The block's scored body.
        atoms : tuple[IRMarginalAtom, ...]
            The atoms, weighted from the logits.

        Returns
        -------
        str
            The stacked log weights.
        """
        logits = marginal_weight_probs(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=py.name_plates,
            target=self.target,
        )
        return _tf_stack(
            py,
            (
                call(
                    py,
                    attribute(py, ("tf", "math", "log_sigmoid")),
                    positional=(
                        _python_unary_minus(py, self._weight_ref_expr(py, logits)),
                    ),
                ),
                call(
                    py,
                    attribute(py, ("tf", "math", "log_sigmoid")),
                    positional=(self._weight_ref_expr(py, logits),),
                ),
            ),
            -1,
        )

    def _weight_ref_expr(self, py: PyCtx, probs: IRArg) -> str:
        """Render the atom-weight probability tensor.

        A weight gathered through an observe's `via` index map carries
        a whole plate vector as its index, which TensorFlow's
        ``__getitem__`` rejects, so the gather routes through
        ``tf.gather`` the way every other plate-vector lookup does.
        """
        if (
            isinstance(probs, IRArgRef)
            and len(probs.indices) == 1
            and isinstance(probs.indices[0], IRArgRef)
            and not probs.indices[0].indices
        ):
            return call(
                py,
                attribute(py, ("tf", "gather")),
                positional=(
                    identifier(py, probs.name),
                    identifier(py, probs.indices[0].name),
                ),
            )
        if isinstance(probs, IRArgRef):
            return self._ref_expr(py, probs)
        raise UnsupportedConstruct(
            _BACKEND_KEY,
            [
                f"marginalize:weight-arg:{type(probs).__name__}: the "
                f"atom weights are not a named tensor"
            ],
        )

    def _emit_tfp_import(self, py: PyCtx) -> None:
        """Emit ``import tensorflow_probability as tfp`` at module
        scope."""
        stmt = py.v(py.fresh("imp"), "import_statement")
        aliased = py.v(py.fresh("alias"), "aliased_import")
        dotted = py.v(py.fresh("dn"), "dotted_name")
        py.e(dotted, identifier(py, "tensorflow_probability"), "child_of")
        py.e(aliased, dotted, "name")
        py.e(aliased, identifier(py, "tfp"), "alias")
        py.e(stmt, aliased, "name")
        py.e("mod", stmt, "child_of")

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit a ``tf.fill([...], <value>)`` call.

        1D targets render as ``tf.fill([K], <value>)``; 2D as
        ``tf.fill([R, C], <value>)``. The shape literal is a plain
        Python list (Edward2 / TF accept it).

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
        py = PyCtx(ctx.sb, target="edward2")
        return self._broadcast(py, value, target_shape, {}, {})

    # ------------------------------------------------------------------
    # Distribution-call construction
    # ------------------------------------------------------------------

    def _emit_grouped_marginalize(
        self,
        py: PyCtx,
        body_vid: str,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
        index_map: tuple[str, Dim],
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> None:
        """Integrate a grouped block's latent out as a traced factor.

        A block whose rows are grouped into fewer groups than rows, under
        a prior shared across the groups, keys its accumulator by group
        (`docs/semantics/programs.md` §2.7): each atom's per-row
        log-density is scattered into a ``(|G|, K)`` accumulator through
        the index map with ``tf.math.unsorted_segment_sum``, the log
        prior is added, and the reduction over the atoms summed over the
        groups is traced as a factor variable. A mixture is a per-row
        integral, which is a different measure here.

        Parameters
        ----------
        py : PyCtx
            The emission context.
        body_vid : str
            The model body.
        node : IRMarginalize
            The block.
        raw : MarginalizeBody
            The block's scope, split into its bindings and its observe.
        atoms : tuple[IRMarginalAtom, ...]
            The block's atoms.
        index map : tuple[str, Dim]
            The index map's name and the grouping plate's axis.
        input_specs : dict[str, ConstraintSpec]
            The declared inputs.
        bindings : dict[str, _Binding]
            The bindings in scope.
        """
        observe = raw.observe
        prefix = f"__marg_{node.latent}"
        term_names: list[str] = []
        for position, atom in enumerate(atoms):
            scored = marginalize_body(
                atom.scope, latent=node.latent, target=self.target
            )
            for det in scored.deterministics:
                asn = py.v(py.fresh("asn"), "assignment")
                py.e(asn, identifier(py, det.name), "left")
                py.e(asn, render_deterministic_python(py, det), "right")
                py.e(body_vid, asn, "child_of")
                bindings[det.name] = _Binding(
                    constraint=det.constraint, plate=det.plate
                )
            component = self._dist_call(
                py,
                name=observe.name,
                family=scored.observe.family,
                args=scored.observe.args,
                arg_names=scored.observe.arg_names,
                plate=Plate(event_dims=(), batch_dims=()),
                input_specs=input_specs,
                bindings=bindings,
                observed_name=None,
                callee_chain=("tfp", "distributions"),
            )
            term = f"{prefix}_{position}"
            py.e(
                body_vid,
                assignment(
                    py,
                    lhs_name=term,
                    rhs=call(
                        py,
                        attribute(py, ("tf", "cast")),
                        positional=(
                            _python_method_call(
                                py,
                                component,
                                "log_prob",
                                (identifier(py, observe.name),),
                            ),
                            attribute(py, ("tf", "float32")),
                        ),
                    ),
                ),
                "child_of",
            )
            term_names.append(term)
        py.e(
            body_vid,
            assignment(
                py,
                lhs_name=f"{prefix}_w",
                rhs=self._marginal_logit_weights(py, node, raw, atoms)
                if logit_weighted(atoms[0])
                else call(
                    py,
                    attribute(py, ("tf", "math", "log")),
                    positional=(self._marginal_weights(py, node, raw, atoms),),
                ),
            ),
            "child_of",
        )
        via, group = index_map
        extent = str(group.size) if isinstance(group, DimStatic) else group.size_name
        stacked = ", ".join(term_names)
        graft_python_statements(
            py.builder,
            f"{prefix} = tf.stack([{stacked}], axis=-1)\n"
            f"{prefix} = tf.math.unsorted_segment_sum({prefix}, "
            f"tf.cast({via}, tf.int32), {extent})\n"
            f"{prefix}_total = tf.reduce_sum(tf.reduce_logsumexp("
            f"tf.cast({prefix}_w, {prefix}.dtype) + {prefix}, axis=-1))\n"
            f'_qvr_qiec_factor("{node.latent}", {prefix}_total)\n',
            body_vid,
            f"marg_{node.latent}_group",
        )

    def _dist_call(
        self,
        py: PyCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
        observed_name: str | None,
        callee_chain: tuple[str, ...] = ("edward2",),
    ) -> str:
        """Build ``edward2.<Family>(positional|kwargs,
        sample_shape=[..], name="<name>")``.

        Args are emitted positionally for single-arg distributions like
        ``Dirichlet(concentration)`` and as keyword args for
        multi-positional families to match Edward2's `tfp` conventions.
        ``arg_aliases["edward2"]`` rewrites argument names.

        An observed site (`observed_name` set) also carries
        ``value=<observed_name>``, so its random variable holds the
        observation over the whole plate.

        `callee_chain` names the module the distribution class is read
        from. It defaults to Edward2, whose constructors register a
        random variable on the trace; the marginalize enumeration
        passes ``("tfp", "distributions")`` for the mixture
        components, which are plain distributions rather than sites.
        """
        meta = self._lookup_meta(family)
        dist_class = meta.target_names.get(_TARGET)
        if dist_class is None:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [f"family:{family}:no-edward2-target"],
            )

        # Families whose TFP constructor shape diverges from the
        # generic (rename-only) path get a dedicated builder.
        if family == "MatrixNormal":
            return self._matrix_normal_call(
                py,
                name=name,
                observed_name=observed_name,
                dist_class=dist_class,
                args=args,
                plate=plate,
                input_specs=input_specs,
                bindings=bindings,
            )
        if family == "LKJCholesky":
            return self._lkj_cholesky_call(
                py,
                name=name,
                observed_name=observed_name,
                dist_class=dist_class,
                args=args,
                plate=plate,
                input_specs=input_specs,
                bindings=bindings,
            )
        if family == "MixtureNormal":
            return self._mixture_normal_call(
                py,
                name=name,
                observed_name=observed_name,
                dist_class=dist_class,
                args=args,
                arg_names=arg_names,
                plate=plate,
                meta=meta,
                input_specs=input_specs,
                bindings=bindings,
                callee_chain=callee_chain,
            )

        # ``HalfStudentT`` has no TFP class. A half-Student-t with
        # ``(df, scale)`` is the fold of ``StudentT(df, 0, scale)`` onto
        # the nonnegative half-line; its density there is
        # ``StudentT(df, 0, scale)`` scaled by 2, so the two agree up to
        # the additive constant ``log 2`` on the family's positive
        # support. The renderer emits the location-scale StudentT and
        # relies on the fold being an additive-constant shift.
        if family == "HalfStudentT":
            dist_class = "StudentT"
            args = (args[0], IRArgNumber(value=0.0), *args[1:])
            arg_names = (arg_names[0], "loc", *arg_names[1:])
            meta = self._lookup_meta("StudentT")

        callee = attribute(py, (*callee_chain, dist_class))
        aliases = {
            **meta.arg_aliases.get(_TARGET, {}),
            **_EDWARD2_ARG_ALIASES.get(family, {}),
        }

        # Inject the canonical zero-loc argument for families whose
        # torch distribution carries fewer parameters than TFP's
        # same-named distribution (``HalfCauchy(scale)`` ->
        # ``HalfCauchy(loc, scale)``). The injected name is the QVR
        # side ``"loc"``; ``arg_aliases["edward2"]`` may rename it
        # before the keyword form is emitted.
        if family in _PREPEND_LOC_ZERO:
            args = (IRArgNumber(value=0.0), *args)
            arg_names = ("loc", *arg_names)

        # Build per-arg expression vertices, lifting scalar refs to
        # broadcasts where the family's event_dims demand it.
        target_event_shape = _event_shape(plate)
        rendered_args = tuple(
            self._render_arg(
                py,
                a,
                expected_arg_event=target_event_shape,
                arg_position=i,
                meta=meta,
                input_specs=input_specs,
                bindings=bindings,
            )
            for i, a in enumerate(args)
        )

        # Lift the integer-supported count parameters TFP unifies
        # against continuous ones into `float32` (see
        # `_EDWARD2_FLOAT_COUNT_ARGS`).
        float_count_args = _EDWARD2_FLOAT_COUNT_ARGS.get(family, frozenset())
        if float_count_args:
            rendered_args = tuple(
                _cast_float32(py, rendered)
                if i < len(arg_names) and arg_names[i] in float_count_args
                else rendered
                for i, rendered in enumerate(rendered_args)
            )

        # TFP / Edward2 distribution constructors accept every arg
        # by keyword; some (Categorical, Bernoulli) require the
        # keyword form because `logits` precedes `probs` in their
        # positional signature. The renderer emits the keyword form
        # for arguments whose target name is not the family's first
        # positional, and the positional form otherwise; this matches
        # the canonical Edward2 idiom (`Dirichlet(tf.fill([K], x))`
        # positional; `Categorical(probs=x)` keyword).
        keyword: list[tuple[str, str]] = []
        positional: list[str] = []
        for arg_name, rendered in zip(arg_names, rendered_args, strict=False):
            target_name = aliases.get(arg_name, arg_name)
            if _is_positional_arg(meta, target_name) and not positional:
                positional.append(rendered)
            else:
                keyword.append((target_name, rendered))
        for rendered in rendered_args[len(arg_names) :]:
            positional.append(rendered)

        sample_shape = self._sample_shape(
            py,
            plate,
            family,
            carried=self._carried_batch_keys(args, meta, bindings),
            observed=observed_name is not None,
        )
        if sample_shape is not None:
            keyword.append(("sample_shape", sample_shape))
        keyword.extend(_site_keywords(py, name, observed_name))

        return call(
            py,
            callee,
            positional=tuple(positional),
            keyword=tuple(keyword),
        )

    def _mixture_normal_call(
        self,
        py: PyCtx,
        *,
        name: str,
        observed_name: str | None,
        dist_class: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
        callee_chain: tuple[str, ...],
    ) -> str:
        """Build ``MixtureSameFamily(Categorical(probs), Normal(loc,
        scale))`` for a `MixtureNormal(w, mu, sigma)` site.

        TFP ships no single-name Gaussian-mixture class, but it ships
        the mixture *construction*, and the two agree exactly: a
        categorical mixing distribution over a batch of `K` components
        scores `logsumexp_k [log w_k + log N(y; mu_k, sigma_k)]`, which
        is the QVR likelihood's own closed form rather than an
        approximation of it. The mixing and component distributions are
        bare TFP distributions; an Edward2 constructor would register
        each as its own random variable and the site denotes one
        observed variable, not three.

        The component axis belongs to the mixture, not to the plate:
        all three parameters carry it and the resulting mixture has an
        empty batch shape, so the whole plate goes into
        ``sample_shape`` rather than being split against a carried
        argument axis.
        """
        weights, loc, scale = mixture_normal_components(_BACKEND_KEY, args, arg_names)
        expected_event = _event_shape(plate)

        def rendered(arg: IRArg, position: int) -> str:
            return self._render_arg(
                py,
                arg,
                expected_arg_event=expected_event,
                arg_position=position,
                meta=meta,
                input_specs=input_specs,
                bindings=bindings,
            )

        mixing = call(
            py,
            attribute(py, ("tfp", "distributions", "Categorical")),
            keyword=(("probs", rendered(weights, 0)),),
        )
        components = call(
            py,
            attribute(py, ("tfp", "distributions", "Normal")),
            keyword=(
                ("loc", rendered(loc, 1)),
                ("scale", rendered(scale, 2)),
            ),
        )
        keyword: list[tuple[str, str]] = [
            ("mixture_distribution", mixing),
            ("components_distribution", components),
        ]
        dims = (*plate.batch_dims, *plate.event_dims)
        if dims:
            shape_list = py.v(py.fresh("list"), "list")
            for dim in dims:
                py.e(shape_list, _dim_expr(py, dim), "child_of")
            keyword.append(("sample_shape", shape_list))
        keyword.extend(_site_keywords(py, name, observed_name))
        return call(
            py,
            attribute(py, (*callee_chain, dist_class)),
            keyword=tuple(keyword),
        )

    def _lkj_cholesky_call(
        self,
        py: PyCtx,
        *,
        name: str,
        observed_name: str | None,
        dist_class: str,
        args: tuple[IRArg, ...],
        plate: Plate,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Build ``edward2.LKJ(<d>, <concentration>,
        input_output_cholesky=True, name="<name>")``.

        TFP's ``LKJ`` takes the correlation-matrix dimension ``d`` as
        its first positional argument; the generic path emits only the
        concentration, so ``concentration`` binds to ``dimension`` and
        the true concentration is dropped. The dimension is read from
        the sample's event axis. ``input_output_cholesky=True`` makes
        TFP sample / score Cholesky factors, matching the QVR
        ``LKJCholesky`` support (a bare ``LKJ`` ranges over full
        correlation matrices, a different support and density).
        """
        if not plate.event_dims:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                ["family:LKJCholesky:missing matrix-dimension event axis"],
            )
        callee = attribute(py, ("edward2", dist_class))
        dimension = _dim_expr(py, plate.event_dims[-1])
        concentration = self._render_arg(
            py,
            args[0],
            expected_arg_event=(),
            arg_position=0,
            meta=self._lookup_meta("LKJCholesky"),
            input_specs=input_specs,
            bindings=bindings,
        )
        keyword: list[tuple[str, str]] = [
            ("input_output_cholesky", _true_literal(py)),
        ]
        sample_shape = self._sample_shape(
            py,
            plate,
            "LKJCholesky",
            carried=self._carried_batch_keys(
                args, self._lookup_meta("LKJCholesky"), bindings
            ),
            observed=observed_name is not None,
        )
        if sample_shape is not None:
            keyword.append(("sample_shape", sample_shape))
        keyword.extend(_site_keywords(py, name, observed_name))
        return call(
            py,
            callee,
            positional=(dimension, concentration),
            keyword=tuple(keyword),
        )

    def _matrix_normal_call(
        self,
        py: PyCtx,
        *,
        name: str,
        observed_name: str | None,
        dist_class: str,
        args: tuple[IRArg, ...],
        plate: Plate,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Build ``edward2.MatrixNormalLinearOperator(loc=...,
        scale_row=..., scale_column=..., name="<name>")``.

        TFP parameterizes the matrix-normal by the Cholesky factors of
        the row and column covariances, wrapped as ``LinearOperator``s,
        under the keywords ``scale_row`` / ``scale_column``. The generic
        path forwards the QVR ``row_covariance`` / ``col_covariance``
        keywords with the covariance matrices themselves, neither of
        which TFP accepts. Row / column covariance ``C`` becomes
        ``tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(C))``;
        since ``C = L Lᵀ`` this leaves the distribution (and its
        log-density) unchanged.
        """
        if len(args) != 3:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                ["family:MatrixNormal:expected (loc, row_covariance, col_covariance)"],
            )
        callee = attribute(py, ("edward2", dist_class))
        meta = self._lookup_meta("MatrixNormal")
        event = _event_shape(plate)
        rendered = tuple(
            self._render_arg(
                py,
                arg,
                expected_arg_event=event,
                arg_position=i,
                meta=meta,
                input_specs=input_specs,
                bindings=bindings,
            )
            for i, arg in enumerate(args)
        )
        keyword: list[tuple[str, str]] = [
            ("loc", rendered[0]),
            ("scale_row", _linop_cholesky(py, rendered[1])),
            ("scale_column", _linop_cholesky(py, rendered[2])),
        ]
        sample_shape = self._sample_shape(
            py,
            plate,
            "MatrixNormal",
            carried=self._carried_batch_keys(args, meta, bindings),
            observed=observed_name is not None,
        )
        if sample_shape is not None:
            keyword.append(("sample_shape", sample_shape))
        keyword.extend(_site_keywords(py, name, observed_name))
        return call(py, callee, keyword=tuple(keyword))

    def _render_arg(
        self,
        py: PyCtx,
        arg: IRArg,
        *,
        expected_arg_event: tuple[int, ...],
        arg_position: int,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render one [`IRArg`][quivers.transpile.ir.IRArg] to a
        Python expression vertex.

        Scalar references against a vector / matrix arg position are
        lifted into a ``tf.fill(...)`` broadcast. The expected
        argument shape is read from the family's
        ``arg_constraints[arg_name]``; when that constraint is an
        ``IndependentConstraint`` of rank ``>=1`` the surrounding
        plate's ``event_dims`` give the broadcast target shape.
        """
        if isinstance(arg, IRArgNumber):
            return self._maybe_broadcast_scalar(
                py,
                arg,
                expected_arg_event,
                arg_position,
                meta,
                input_specs,
                bindings,
            )
        if isinstance(arg, IRArgRef):
            return self._maybe_broadcast_ref(
                py,
                arg,
                expected_arg_event,
                arg_position,
                meta,
                input_specs,
                bindings,
            )
        if isinstance(arg, IRArgBroadcast):
            return self._broadcast(
                py, arg.value, arg.target_shape, input_specs, bindings
            )
        if isinstance(arg, IRArgList):
            return self._render_list(py, arg, input_specs, bindings)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(py, arg, input_specs, bindings)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(py, arg, input_specs, bindings)
        raise UnsupportedConstruct(_BACKEND_KEY, [f"arg:{type(arg).__name__}"])

    def _maybe_broadcast_scalar(
        self,
        py: PyCtx,
        arg: IRArgNumber,
        expected_arg_event: tuple[int, ...],
        arg_position: int,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        target_shape = self._broadcast_target_for(
            arg_position, meta, expected_arg_event
        )
        if target_shape is None:
            return number_literal(py, arg.value)
        return self._broadcast(py, arg, target_shape, input_specs, bindings)

    def _maybe_broadcast_ref(
        self,
        py: PyCtx,
        arg: IRArgRef,
        expected_arg_event: tuple[int, ...],
        arg_position: int,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        target_shape = self._broadcast_target_for(
            arg_position, meta, expected_arg_event
        )
        if target_shape is None:
            return self._ref_expr(py, arg)
        # Broadcast only when the referenced value is scalar (an
        # exogenous Real / Nat input or a previously-bound scalar
        # binding). Tensor-shaped refs flow through unchanged.
        if not self._is_scalar_ref(arg, input_specs, bindings):
            return self._ref_expr(py, arg)
        return self._broadcast(py, arg, target_shape, input_specs, bindings)

    def _ref_expr(self, py: PyCtx, arg: IRArgRef) -> str:
        """Build a Python expression for an `IRArgRef`.

        A number or a single-index name subscripts the tensor. An index
        that is itself a tensor (a plated class draw, ``mu[z]``) gathers
        through ``tf.gather``, since TensorFlow's ``__getitem__`` takes
        no tensor index; the gather reads the outermost axis, so only
        the first index may be a tensor.

        Parameters
        ----------
        py : PyCtx
            The render state.
        arg : IRArgRef
            The reference.

        Returns
        -------
        str
            The expression's vertex.

        Raises
        ------
        UnsupportedConstruct
            If a tensor index follows the first index.
        """
        base = identifier(py, arg.name)
        current = base
        for position, idx in enumerate(arg.indices):
            if isinstance(idx, IRArgRef) and py.name_rank.get(idx.name, 0) > 0:
                if position > 0:
                    raise UnsupportedConstruct(
                        _BACKEND_KEY,
                        [
                            f"index:tensor-after-first:{arg.name}: a tensor "
                            f"index after the first has no `tf.gather` reading"
                        ],
                    )
                current = call(
                    py,
                    attribute(py, ("tf", "gather")),
                    positional=(current, self._render_index(py, idx)),
                )
                continue
            sub = py.v(py.fresh("subs"), "subscript")
            py.e(sub, current, "value")
            py.e(sub, self._render_index(py, idx), "subscript")
            current = sub
        return current

    def _render_index(self, py: PyCtx, idx: IRArg) -> str:
        """Render an index expression: refs become identifiers,
        nested refs preserve their own bracket chain, numbers become
        integer literals."""
        if isinstance(idx, IRArgNumber):
            return number_literal(py, idx.value)
        if isinstance(idx, IRArgRef):
            return self._ref_expr(py, idx)
        raise UnsupportedConstruct(_BACKEND_KEY, [f"index:{type(idx).__name__}"])

    def _render_list(
        self,
        py: PyCtx,
        arg: IRArgList,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render ``[e0, e1, ...]`` as a Python list (TF accepts it)."""
        lst = py.v(py.fresh("list"), "list")
        for el in arg.elements:
            py.e(lst, self._render_arg_atom(py, el, input_specs, bindings), "child_of")
        return lst

    def _render_matrix(
        self,
        py: PyCtx,
        arg: IRArgMatrix,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render ``[[...], [...]]`` as nested Python lists."""
        outer = py.v(py.fresh("list"), "list")
        for row in arg.rows:
            py.e(outer, self._render_list(py, row, input_specs, bindings), "child_of")
        return outer

    def _render_arg_atom(
        self,
        py: PyCtx,
        el: IRArg,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render one element inside an `IRArgList` (no broadcast
        lifting; lists carry concrete element expressions)."""
        if isinstance(el, IRArgNumber):
            return number_literal(py, el.value)
        if isinstance(el, IRArgRef):
            return self._ref_expr(py, el)
        if isinstance(el, IRArgList):
            return self._render_list(py, el, input_specs, bindings)
        if isinstance(el, IRArgMatrix):
            return self._render_matrix(py, el, input_specs, bindings)
        if isinstance(el, IRArgBroadcast):
            return self._broadcast(py, el.value, el.target_shape, input_specs, bindings)
        raise UnsupportedConstruct(_BACKEND_KEY, [f"arg-atom:{type(el).__name__}"])

    def _render_family_ref(
        self,
        py: PyCtx,
        arg: IRArgFamilyRef,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render a wrapped-distribution reference as an identifier.

        Wrapper families (`Truncated`, `Mixture`, ...) reference a
        morphism by name; the morphism's declaration is consumed by the
        wrapper's call shape. Without the morphism table in this
        renderer's `ctx` (renderers consume the IR, not the source
        Module), the safe rendering is the wrapped morphism's name as
        an identifier; downstream wrapper handling is per-family
        future work tracked in the spec's §10.10 wrapper rendering
        clause.
        """
        del input_specs, bindings
        return identifier(py, arg.name)

    def _broadcast(
        self,
        py: PyCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Emit ``tf.fill([<shape>], <value>)``.

        1D and 2D targets are both supported; the shape literal is a
        Python list. Higher ranks raise
        [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
        because tree-sitter Python's list shape would render
        identically but TF semantics for rank-3 broadcasts vary by
        family.
        """
        if len(target_shape) not in (1, 2):
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [f"broadcast:rank-{len(target_shape)}"],
            )
        shape_list = py.v(py.fresh("list"), "list")
        for k in target_shape:
            py.e(shape_list, number_literal(py, float(k)), "child_of")
        if isinstance(value, IRArgNumber):
            # The fill value types the tensor: an integer literal here
            # builds an integer concentration / rate / scale, which the
            # real-valued family rejects.
            value_vid = float_literal(py, value.value)
        elif isinstance(value, IRArgRef):
            value_vid = self._ref_expr(py, value)
        else:
            value_vid = self._render_arg_atom(py, value, input_specs, bindings)
        return call(
            py,
            attribute(py, ("tf", "fill")),
            positional=(shape_list, value_vid),
        )

    # ------------------------------------------------------------------
    # Sample-shape and scalar-detection helpers
    # ------------------------------------------------------------------

    def _sample_shape(
        self,
        py: PyCtx,
        plate: Plate,
        family: str,
        *,
        carried: tuple[str, ...] | None,
        observed: bool = False,
    ) -> str | None:
        """Build the ``sample_shape=[D0, D1, ...]`` argument for Edward2.

        The sample shape combines batch dimensions with event dimensions beyond
        the family's natural event rank. Axes already carried by distribution
        arguments are removed because TensorFlow Probability includes their
        broadcast batch shape in the random variable's value shape.

        Returns `None` when no dimensions remain. An observed site carries
        its data as the random variable's ``value``, which fixes its shape,
        so when its arguments' broadcast shape is not statically known it
        takes no ``sample_shape`` and the data's shape stands. A latent
        site raises when carried axes are absent from the declared plate
        or cannot be resolved.
        """
        meta = FAMILY_META.get(family)
        natural = meta.event_rank if meta is not None else 0
        residual_event = (
            plate.event_dims[: len(plate.event_dims) - natural]
            if natural
            else plate.event_dims
        )
        dims = (*plate.batch_dims, *residual_event)
        if not dims:
            return None
        if carried is None and observed:
            return None
        if carried is None:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [
                    f"family:{family}: plate axes over an argument whose "
                    "broadcast shape is not statically determinable"
                ],
            )
        split = len(dims) - len(carried)
        if split < 0 or tuple(_dim_key(d) for d in dims[split:]) != carried:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [
                    f"family:{family}: argument batch axes {carried} are "
                    "not a trailing run of the plate axes "
                    f"{tuple(_dim_key(d) for d in dims)}"
                ],
            )
        dims = dims[:split]
        if not dims:
            return None
        lst = py.v(py.fresh("list"), "list")
        for dim in dims:
            py.e(lst, _dim_expr(py, dim), "child_of")
        return lst

    def _carried_batch_keys(
        self,
        args: tuple[IRArg, ...],
        meta: FamilyMeta,
        bindings: dict[str, _Binding],
    ) -> tuple[str, ...] | None:
        """Return the batch axes the rendered arguments already
        broadcast to, or ``None`` when any argument's shape is not
        statically determinable.

        TFP broadcasts the constructor's arguments against one
        another, so the distribution's batch shape is the widest of
        the per-argument batch shapes; the widest tuple is returned.
        """
        carried: tuple[str, ...] = ()
        for position, arg in enumerate(args):
            keys = self._arg_batch_keys(
                arg,
                arg_event_rank=_arg_event_rank(meta, position),
                bindings=bindings,
            )
            if keys is None:
                return None
            if len(keys) > len(carried):
                carried = keys
        return carried

    def _arg_batch_keys(
        self,
        arg: IRArg,
        *,
        arg_event_rank: int,
        bindings: dict[str, _Binding],
    ) -> tuple[str, ...] | None:
        """Return the leading axes the rendered `arg` carries beyond
        the `arg_event_rank` axes its constructor slot consumes, keyed
        by [`_dim_key`][], or ``None`` when the shape is not
        statically determinable.

        Numeric literals are scalars. A `tf.fill` broadcast targets
        exactly the slot's event shape, so it contributes no batch
        axis. A list / matrix literal contributes its own extents. A
        bare reference contributes the axes of the plate its binding
        was declared under; a bracket-indexed reference lowers to
        ``tf.gather`` or a subscript whose result shape the IR does
        not pin down.
        """
        if isinstance(arg, IRArgNumber):
            return ()
        if isinstance(arg, IRArgBroadcast | IRArgFamilyRef | IRArgKernel):
            return ()
        if isinstance(arg, IRArgMatrix):
            columns = len(arg.rows[0].elements) if arg.rows else 0
            extents = (_static_key(len(arg.rows)), _static_key(columns))
            return _drop_event_axes(extents, arg_event_rank)
        if isinstance(arg, IRArgList):
            return _drop_event_axes((_static_key(len(arg.elements)),), arg_event_rank)
        if isinstance(arg, IRArgRef):
            if arg.indices:
                return None
            binding = bindings.get(arg.name)
            if binding is None:
                return None
            extents = tuple(
                _dim_key(dim)
                for dim in (
                    *binding.plate.batch_dims,
                    *binding.plate.event_dims,
                )
            )
            return _drop_event_axes(extents, arg_event_rank)
        return None

    def _broadcast_target_for(
        self,
        arg_position: int,
        meta: FamilyMeta,
        plate_event: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """Return the broadcast target shape for the arg at
        `arg_position` of `meta`, or ``None`` when the arg position
        does not require a broadcast.

        Reads `meta.distribution_class.arg_constraints` to detect
        rank-`n>=1` independent constraints, simplex constraints,
        positive-definite matrices, and the plain real-vector /
        real-matrix forms; combines them with the plate's
        `event_dims` to compute the target shape.
        """
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return None
        items = list(cls_attr.items())
        if arg_position >= len(items):
            return None
        _, constraint = items[arg_position]
        if is_real_simplex(constraint):
            return plate_event[-1:] if plate_event else None
        if is_real_vector(constraint):
            return plate_event[-1:] if plate_event else None
        if is_real_cov_matrix(constraint) or is_real_matrix(constraint):
            if len(plate_event) >= 2:
                return plate_event[-2:]
            if len(plate_event) == 1:
                return (plate_event[0], plate_event[0])
            return None
        # `IndependentConstraint(base, n>=1)` not covered above.
        rank = int(getattr(constraint, "event_dim", 0))
        if rank == 0:
            return None
        if rank == 1 and plate_event:
            return plate_event[-1:]
        if rank == 2 and len(plate_event) >= 2:
            return plate_event[-2:]
        return None

    def _is_scalar_ref(
        self,
        arg: IRArgRef,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> bool:
        """True iff the referenced name carries a scalar constraint
        with no bracket indexing peeling already applied.

        Exogenous inputs typed `Real` / `Nat` are scalar; sample /
        observe / deterministic bindings inherit their family's
        event_dims plus any batch_dims from the surrounding plate.
        """
        if arg.indices:
            return False
        spec = input_specs.get(arg.name)
        if spec is not None and _is_scalar_constraint(spec):
            binding = bindings.get(arg.name)
            if binding is None:
                return True
            return _is_scalar_plate(binding.plate)
        binding = bindings.get(arg.name)
        if binding is None:
            return False
        return _is_scalar_constraint(binding.constraint) and _is_scalar_plate(
            binding.plate
        )

    def _lookup_meta(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                _BACKEND_KEY, [f"family:{family}: not in FAMILY_META"]
            )
        return meta


# ----------------------------------------------------------------------
# Small structural helpers (no per-family branching).
# ----------------------------------------------------------------------


class _Binding:
    """One previously-bound name's constraint + plate, used by the
    scalar-ref test for broadcast lifting decisions."""

    __slots__ = ("constraint", "plate")

    def __init__(self, constraint: ConstraintSpec, plate: Plate) -> None:
        self.constraint = constraint
        self.plate = plate


def _event_shape(plate: Plate) -> tuple[int, ...]:
    """Return the integer event shape implied by `plate.event_dims`.

    Dynamic dims contribute `0` (a placeholder; the renderer never
    embeds it into the source: it lives only as a marker that the dim
    is dynamic). The `_broadcast_target_for` caller treats `0`-valued
    entries by falling back to whatever the source plate already
    encodes.
    """
    out: list[int] = []
    for dim in plate.event_dims:
        if isinstance(dim, DimStatic):
            out.append(dim.size)
        else:
            out.append(0)
    return tuple(out)


def _cast_float32(py: PyCtx, expr: str) -> str:
    """Wrap `expr` in ``tf.cast(<expr>, tf.float32)``."""
    return call(
        py,
        attribute(py, ("tf", "cast")),
        positional=(expr, attribute(py, ("tf", "float32"))),
    )


def _true_literal(py: PyCtx) -> str:
    """Emit a Python ``True`` literal vertex."""
    vid = py.v(py.fresh("true"), "true")
    py.literal(vid, "True")
    return vid


def _linop_cholesky(py: PyCtx, cov_expr: str) -> str:
    """Wrap a covariance expression ``C`` as the ``LinearOperator``
    ``tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(C))``,
    the Cholesky-factor scale TFP's ``MatrixNormalLinearOperator``
    expects for ``scale_row`` / ``scale_column``.
    """
    chol = call(
        py,
        attribute(py, ("tf", "linalg", "cholesky")),
        positional=(cov_expr,),
    )
    return call(
        py,
        attribute(py, ("tf", "linalg", "LinearOperatorLowerTriangular")),
        positional=(chol,),
    )


def _dim_key(dim: Dim) -> str:
    """Structural key for a plate dim's extent.

    Two dims share a key exactly when they denote the same axis
    length: a static cardinality matches on its size, a runtime length
    on the name of the input carrying it.
    """
    if isinstance(dim, DimStatic):
        return _static_key(dim.size)
    if isinstance(dim, DimDynamic):
        return f"dynamic:{dim.size_name}"
    raise UnsupportedConstruct(_BACKEND_KEY, [f"dim:{type(dim).__name__}"])


def _static_key(size: int) -> str:
    """Key for a statically known extent, matching `_dim_key`."""
    return f"static:{size}"


def _drop_event_axes(extents: tuple[str, ...], event_rank: int) -> tuple[str, ...]:
    """Strip the trailing `event_rank` axes from `extents`, leaving the
    batch axes the argument broadcasts over."""
    return extents[: max(0, len(extents) - event_rank)]


def _arg_event_rank(meta: FamilyMeta, arg_position: int) -> int:
    """Return the number of event axes the family's constructor slot at
    `arg_position` consumes, read off the torch distribution's
    ``arg_constraints``.

    A simplex or real-vector slot consumes one axis, a covariance or
    real-matrix slot two, an ``IndependentConstraint(base, n)`` slot
    ``n``; every other constraint is scalar-valued and consumes none.
    """
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return 0
    items = list(cls_attr.items())
    if arg_position >= len(items):
        return 0
    _, constraint = items[arg_position]
    if is_real_simplex(constraint) or is_real_vector(constraint):
        return 1
    if is_real_cov_matrix(constraint) or is_real_matrix(constraint):
        return 2
    return int(getattr(constraint, "event_dim", 0))


def _dim_expr(py: PyCtx, dim: Dim) -> str:
    """Render a single plate dim into the expression that should sit
    inside ``sample_shape=[...]``.

    `DimStatic` becomes an integer literal; `DimDynamic` becomes the
    runtime size identifier (``N_Doc`` etc.).
    """
    if isinstance(dim, DimStatic):
        return number_literal(py, float(dim.size))
    if isinstance(dim, DimDynamic):
        return identifier(py, dim.size_name)
    raise UnsupportedConstruct(_BACKEND_KEY, [f"dim:{type(dim).__name__}"])


_SCALAR_CONSTRAINT_CLASSES: tuple[type, ...] = (
    CSReal,
    CSPositive,
    CSUnitInterval,
    CSInterval,
    CSBoolean,
    CSIntegerInterval,
    CSNonnegativeInteger,
    CSPositiveInteger,
)


def _is_scalar_constraint(spec: ConstraintSpec) -> bool:
    """True iff `spec` denotes a 0-rank support.

    Scalar inputs (`CSReal`, the `Real`-typed program parameters) are
    eligible for broadcast lifting; vector / matrix constraints are
    already shaped and flow through unchanged.
    """
    return isinstance(spec, _SCALAR_CONSTRAINT_CLASSES)


def _is_scalar_plate(plate: Plate) -> bool:
    """True iff the plate has neither event nor batch dims."""
    return not plate.event_dims and not plate.batch_dims


def _is_positional_arg(meta: FamilyMeta, target_name: str) -> bool:
    """True iff the given arg should ride positionally in the
    ``edward2.<Family>(...)`` call.

    The rule: distributions with exactly one declared arg
    (`Dirichlet(concentration)`, `Exponential(rate)`, ...) render
    that arg positionally; distributions whose torch
    ``arg_constraints`` advertises multiple parameterisations
    (`Categorical(probs, logits)`, `Bernoulli(probs, logits)`, ...)
    render every arg by keyword so the caller doesn't accidentally
    bind to ``logits=`` when TFP's positional signature differs from
    torch's. ``target_name`` is unused today: the predicate is
    structural over the family's arg_constraints shape.
    """
    del target_name
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return False
    return len(cls_attr) == 1


def _real_input_names(ir: IRProgram) -> tuple[str, ...]:
    """Exogenous data inputs the program declares on the unbounded
    reals, in signature order.

    These are the continuous covariates (`Real` domains and design
    columns). Every other constraint either names an integer support
    (an index covariate, a count observation) or a bounded continuous
    one whose caller-side dtype is float by construction. Observed
    names are excluded: each is rebound by its own random variable and
    reaches the model only through that variable's ``log_prob``, so it
    carries a ``None`` default that a cast would reject.
    """
    observed = _observed_names(ir.body)
    return tuple(
        inp.name
        for inp in ir.inputs
        if isinstance(inp.constraint, CSReal) and inp.name not in observed
    )


def _site_keywords(
    py: PyCtx, name: str, observed_name: str | None
) -> tuple[tuple[str, str], ...]:
    """The keywords that name a random variable and, if observed, fix its value.

    Parameters
    ----------
    py : PyCtx
        The emit context.
    name : str
        The site's name.
    observed_name : str | None
        The model argument holding the observation, or ``None`` for a
        latent site.

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``value=<observed_name>`` for an observation, whose random
        variable then holds the data (or a draw when the argument is
        ``None``), followed by ``name="<name>"``.
    """
    named = (("name", string_literal(py, name)),)
    if observed_name is None:
        return named
    return (("value", identifier(py, observed_name)), *named)


def _observed_names(body: tuple[IRNode, ...]) -> frozenset[str]:
    """Every name bound by an
    [`IRObserve`][quivers.transpile.ir.IRObserve] in `body`."""
    out: set[str] = set()
    for node in body:
        if isinstance(node, IRObserve):
            out.add(node.name)
        elif isinstance(node, IRMarginalize):
            out |= _observed_names(node.scope)
    return frozenset(out)


def _ir_has_marginalize(body: tuple[IRNode, ...]) -> bool:
    """True iff `body` carries an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] anywhere."""
    return any(isinstance(node, IRMarginalize) for node in body)


def _ir_has_factor(body: tuple[IRNode, ...]) -> bool:
    """True iff `body` scores or calls, either of which traces a factor.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The plan body.

    Returns
    -------
    bool
        Whether the module needs the factor helper.
    """
    return any(
        isinstance(node, (IRScore, IRCall))
        or (isinstance(node, IRMarginalize) and _marginalize_is_grouped(node))
        for node in body
    )


def _marginalize_is_grouped(node: IRMarginalize) -> bool:
    """Whether a block scatters its rows through an index map.

    Parameters
    ----------
    node : IRMarginalize
        The block.

    Returns
    -------
    bool
        ``True`` when the block carries a grouping plate finer than
        its observe's plate and an observe with a ``via`` index map,
        which the renderer traces as a factor.
    """
    if not node.plate.batch_dims:
        return False
    return any(
        isinstance(inner, IRObserve)
        and inner.via is not None
        and inner.plate.batch_dims != node.plate.batch_dims
        for inner in node.scope
    )


def _tf_stack(py: PyCtx, items: tuple[str, ...], axis: int) -> str:
    """Emit ``tf.stack([...], axis=<axis>)``."""
    axis_vid = (
        _python_unary_minus(py, number_literal(py, -axis))
        if axis < 0
        else number_literal(py, axis)
    )
    return call(
        py,
        attribute(py, ("tf", "stack")),
        positional=(python_list(py, items),),
        keyword=(("axis", axis_vid),),
    )


__all__ = ["Edward2Renderer"]
