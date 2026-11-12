"""Render transpilation IR as a NumPyro model function.

Samples and observations use ``numpyro.sample`` inside nested plates.
Finite marginalizations score each atom and add their ``logsumexp``
with ``numpyro.factor``. Array broadcasts use ``jax.numpy.full``.
"""

from __future__ import annotations

import pathlib

import panproto
import torch.distributions.constraints as c

from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.renderers._ranks import carries_event_dims, name_rank_map
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    arg_expr,
    attribute,
    call,
    MarginalizeBody,
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
    string_literal,
    with_statement,
)
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprFactor,
    LetExprList,
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
    event_dim_of,
)
from quivers.transpile.renderers._base import (
    marginalize_row_rank,
    BlockKind,
    IRMarginalAtom,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dropped_param_map,
    logit_weighted,
    mixture_normal_components,
)
from quivers.transpile.qiec_ir import IRQiecModule
from quivers.transpile.renderers._qiec import (
    emit_call_python,
    graft_python_statements,
    qiec_helper_roots,
    render_computations_dynamic,
)


#: The backend key used to look up `target_names` / `arg_aliases` in
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META].
_BACKEND: str = "numpyro"


#: ``import jax.scipy.special as jsp``, which the marginalize
#: reduction's ``logsumexp`` reads. JAX keeps the stable reduction in
#: `jax.scipy.special`, not on the `jnp` namespace.
_MARGINALIZE_IMPORT: tuple[tuple[str, ...], str] = (
    ("jax", "scipy", "special"),
    "jsp",
)


#: ``import jax.nn as jnn``, whose ``log_sigmoid`` weights the atoms of
#: a Bernoulli-family latent given by its logits.
_LOG_SIGMOID_IMPORT: tuple[tuple[str, ...], str] = (("jax", "nn"), "jnn")


#: Inner-family-keyed lookup table for the specialised truncated
#: distribution classes NumPyro publishes (`TruncatedNormal`,
#: `TruncatedCauchy`, etc.). Inner families absent from this table use
#: the generic `TruncatedDistribution(<inner>, low, high)` wrapper.
_NUMPYRO_TRUNCATED_SPECIALISATION: dict[str, str] = {
    "Normal": "TruncatedNormal",
    "Cauchy": "TruncatedCauchy",
}


#: QVR families for which NumPyro ships no distribution class. Each maps
#: to a helper class grafted from
#: [`runtime_numpyro`][quivers.transpile.runtime_numpyro] whose name is
#: emitted as a bare identifier (not ``numpyro.distributions.<Name>``)
#: at the call site. The helper's ``log_prob`` matches the QVR family up
#: to the shared additive constant.
_NUMPYRO_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset(
    {
        "LogitNormal",
        "HalfStudentT",
        "ContinuousBernoulli",
        "FisherSnedecor",
        "LogisticNormal",
        "OneHotCategorical",
        "OrderedProbit",
        "LogGamma",
        "LogitBeta",
    }
)


#: Extra aliased imports each grafted helper class needs in the emitted
#: module beyond the always-present ``jnp`` / ``numpyro`` /
#: ``numpyro.distributions``. Keyed by the grafted class name.
_NUMPYRO_HELPER_IMPORTS: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "ContinuousBernoulli": ((("jax", "scipy", "special"), "jss"),),
    "FisherSnedecor": ((("jax", "scipy", "special"), "jss"),),
    "OneHotCategorical": ((("jax", "scipy", "special"), "jss"),),
    "OrderedProbit": ((("jax", "scipy", "special"), "jss"),),
    "LogGamma": ((("jax", "scipy", "special"), "jss"), (("jax", "random"), "jrandom")),
    "LogitBeta": (
        (("jax", "scipy", "special"), "jss"),
        (("jax", "random"), "jrandom"),
    ),
}


#: Families that lower with a leading matrix-dimension positional in
#: NumPyro (``LKJCholesky(dimension, concentration)`` /
#: ``LKJ(dimension, concentration)``). The dimension comes from the
#: sample's event/codomain axis, which QVR carries on
#: [`Plate.event_dims`][quivers.transpile.ir.Plate].
_NUMPYRO_MATRIX_DIMENSION_FAMILIES: frozenset[str] = frozenset(
    {"LKJCholesky", "LKJCorrelationFactor"}
)


class NumPyroRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as NumPyro
    Python source.

    Subclasses [`RendererBase`][quivers.transpile.renderers.RendererBase]
    and supplies the four required dispatch points
    (`declare`, `sample`, `marginalize`, `broadcast`) plus a `render`
    override that wraps the IR walk with the ``def model(...): ...``
    function header.
    """

    target: str = "numpyro"

    # ------------------------------------------------------------------
    # `RendererBase` dispatch points
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Use the auto-derived Python tree-sitter protocol.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("python")

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a complete NumPyro module.

        Overrides the base IR walk because NumPyro programs need a
        function header that lists every
        [`IRDataInput`][quivers.transpile.ir.IRDataInput] in the
        signature (observed values get a ``=None`` default).

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
        py = PyCtx(
            sb,
            cards=dict(ir.cards),
            target="numpyro",
            name_event_rank=name_event_rank_map(ir),
            name_rank=name_rank_map(ir),
            factor_towers=factor_tower_names(ir),
            name_plates=name_plate_map(ir),
        )
        scalar_refs, bound_refs = _classify_bindings(ir)
        ctx = _NumPyroCtx(
            sb=sb,
            morphisms={},
            lets={},
            py=py,
            observed_names=self._collect_observed(ir),
            scalar_refs=scalar_refs,
            bound_refs=bound_refs,
            module=ir.module,
        )

        py.v("mod", "module")
        body = py.v(py.fresh("body"), "block")
        params = self._function_params(ir)
        func = self._build_function(py, body, params)
        if not ir.body:
            noop = py.v(py.fresh("pass"), "pass_statement")
            py.e(body, noop, "child_of")

        # Dispatch the body first so any ``LetExprCall`` records the
        # imports its symbol needs (jax.scipy.special / jax.nn), then emit
        # the import block, graft any runtime-helper classes the body
        # uses, and wire the function last so a reader sees imports, then
        # helper classes, then ``def model``.
        self._dispatch_body(ctx, body, ir.body)
        used_helpers = _ir_helper_classes_used(ir.body) | qiec_helper_roots(
            ir, _BACKEND
        )
        for cls_name in used_helpers:
            for extra_import in _NUMPYRO_HELPER_IMPORTS.get(cls_name, ()):
                py.required_imports.add(extra_import)
        self._emit_imports(ctx)
        for cls_name in sorted(used_helpers):
            _emit_runtime_helper(py, cls_name)
        py.e("mod", func, "child_of")

        render_computations_dynamic(sb, ir, target=self.target, root="mod")
        return sb.build()

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op outside ``"function_body"``.

        NumPyro declarations ARE sample calls; the function-header
        parameter list is built up front by `render` from
        [`IRDataInput`][quivers.transpile.ir.IRDataInput] entries.
        Function-body declarations land at the
        [`sample`][quivers.transpile.renderers.numpyro.NumPyroRenderer.sample]
        dispatch point when the IR walker reaches the binding step.

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
        """Emit a ``numpyro.sample`` call inside the surrounding plate
        stack.

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
        del constraint
        npctx = _as_numpyro_ctx(ctx)
        return self._render_sample(
            npctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            observed=observed,
        )

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Integrate an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        latent out, adding the reduced density to the model's
        log-density with ``numpyro.factor``.

        The block emits one scored copy of the scope per atom of the
        latent's finite support, then reduces across atoms:

        ```python
        gated_rate = 0.0 * rate
        __marg_z_0 = numpyro.distributions.Poisson(gated_rate).log_prob(y)
        gated_rate = 1.0 * rate
        __marg_z_1 = numpyro.distributions.Poisson(gated_rate).log_prob(y)
        __marg_z_w = jnp.stack([jnp.log1p(-pi_z), jnp.log(pi_z)], axis=-1)
        __marg_z = jnp.stack([__marg_z_0, __marg_z_1], axis=-1)
        numpyro.factor(
            "z", jnp.sum(jsp.logsumexp(__marg_z_w + __marg_z, axis=-1))
        )
        ```

        Each atom's deterministic bindings are re-emitted under their
        own names immediately before that atom is scored, so the
        scoring call reads the values that atom pins. No site is
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
        npctx = _as_numpyro_ctx(ctx)
        body_vid = npctx.current_body
        if body_vid is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["marginalize:no-enclosing-body"],
            )
        py = npctx.py
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
                py.e(
                    body_vid,
                    self._deterministic_statement(npctx, det),
                    "child_of",
                )
            dist_call = self._distribution_call(
                npctx,
                scored.observe.family,
                scored.observe.args,
                scored.observe.arg_names,
                scored.observe.plate,
            )
            dist_call = self._wrap_event_dims(
                npctx,
                dist_call,
                scored.observe.family,
                scored.observe.plate.event_dims,
                scored.observe.args,
            )
            term = f"{prefix}_{position}"
            py.e(
                body_vid,
                self._assignment_statement(
                    py,
                    term,
                    _python_method_call(
                        py,
                        dist_call,
                        "log_prob",
                        (identifier(py, scored.observe.name),),
                    ),
                ),
                "child_of",
            )
            term_names.append(term)
        weight_vid = self._marginal_log_weights(npctx, node, raw, atoms)
        py.e(
            body_vid,
            self._assignment_statement(
                py,
                f"{prefix}_w",
                weight_vid,
            ),
            "child_of",
        )
        py.e(
            body_vid,
            self._assignment_statement(
                py,
                prefix,
                self._stack_last_axis(
                    py,
                    tuple(identifier(py, name) for name in term_names),
                ),
            ),
            "child_of",
        )
        py.required_imports.add(_MARGINALIZE_IMPORT)
        # A grouped block keys its accumulator by group: the rows the
        # fibration sends to one group are summed before the reduction.
        fibration = marginalize_fibration(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=py.name_plates,
            target=self.target,
        )
        if fibration is not None:
            via, group = fibration
            extent = (
                str(group.size) if isinstance(group, DimStatic) else group.size_name
            )
            graft_python_statements(
                py.builder,
                f"{prefix} = jnp.zeros(({extent}, {prefix}.shape[-1]), "
                f"dtype={prefix}.dtype).at[{via}].add({prefix})\n",
                body_vid,
                f"marg_{node.latent}_group",
            )
        # An ungrouped block shares one latent across the body's rows,
        # so their per-class log-likelihoods are accumulated before the
        # reduction rather than each row reducing on its own.
        atoms = identifier(py, prefix)
        for _ in range(marginalize_row_rank(node)):
            atoms = call(
                py,
                attribute(py, ("jnp", "sum")),
                positional=(atoms,),
                keyword=(("axis", self._zero(py)),),
            )
        reduced = call(
            py,
            attribute(py, ("jsp", "logsumexp")),
            positional=(
                _python_binary_op(
                    py,
                    "+",
                    identifier(py, f"{prefix}_w"),
                    atoms,
                ),
            ),
            keyword=(("axis", self._minus_one(py)),),
        )
        total = call(
            py,
            attribute(py, ("jnp", "sum")),
            positional=(reduced,),
        )
        py.e(
            body_vid,
            self._expression_statement(
                py,
                call(
                    py,
                    attribute(py, ("numpyro", "factor")),
                    positional=(string_literal(py, node.latent), total),
                ),
            ),
            "child_of",
        )
        return ""

    def _marginal_log_weights(
        self,
        ctx: _NumPyroCtx,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
    ) -> str:
        """Log-weight tensor whose trailing axis runs over the atoms.

        A `Categorical` atom set weights atom ``k`` by ``log p[k]``, so
        the probability tensor's own trailing axis is already the atom
        axis. A `Bernoulli` atom set weights the atoms 0 and 1 by
        ``log1p(-p)`` and ``log(p)``, which stack into a fresh trailing
        axis; from its logits ``l`` the weights are ``log_sigmoid(-l)``
        and ``log_sigmoid(l)``, computed in log space.
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
        if logit_weighted(atoms[0]):
            py.required_imports.add(_LOG_SIGMOID_IMPORT)
            log_sigmoid = attribute(py, ("jnn", "log_sigmoid"))
            return self._stack_last_axis(
                py,
                (
                    call(
                        py,
                        log_sigmoid,
                        positional=(
                            _python_unary_minus(py, self._render_arg(ctx, probs)),
                        ),
                    ),
                    call(
                        py,
                        attribute(py, ("jnn", "log_sigmoid")),
                        positional=(probs_vid,),
                    ),
                ),
            )
        if family == "Categorical":
            return call(py, attribute(py, ("jnp", "log")), positional=(probs_vid,))
        if family == "Bernoulli":
            complement = call(
                py,
                attribute(py, ("jnp", "log1p")),
                positional=(_python_unary_minus(py, self._render_arg(ctx, probs)),),
            )
            positive = call(py, attribute(py, ("jnp", "log")), positional=(probs_vid,))
            return self._stack_last_axis(py, (complement, positive))
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [f"marginalize:weight-family:{family}"],
        )

    def _stack_last_axis(self, py: PyCtx, items: tuple[str, ...]) -> str:
        """Emit ``jnp.stack([...], axis=-1)``."""
        return call(
            py,
            attribute(py, ("jnp", "stack")),
            positional=(python_list(py, items),),
            keyword=(("axis", self._minus_one(py)),),
        )

    def _minus_one(self, py: PyCtx) -> str:
        """Emit the literal ``-1``."""
        return _python_unary_minus(py, number_literal(py, 1))

    def _zero(self, py: PyCtx) -> str:
        """Emit the literal ``0``."""
        return number_literal(py, 0)

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit ``jnp.full((K,), value)`` for 1D / ``jnp.full((R, C),
        value)`` for 2D.

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
        npctx = _as_numpyro_ctx(ctx)
        py = npctx.py
        shape_tuple = self._render_shape_tuple(py, target_shape)
        # The fill value types the tensor: an integer literal here
        # builds an integer concentration / rate / scale, which the
        # real-valued family rejects.
        value_vid = (
            float_literal(py, value.value)
            if isinstance(value, IRArgNumber)
            else self._render_arg(npctx, value)
        )
        return call(
            py,
            attribute(py, ("jnp", "full")),
            positional=(shape_tuple, value_vid),
        )

    # ------------------------------------------------------------------
    # Per-renderer rendering helpers
    # ------------------------------------------------------------------

    def _render_list(self, ctx: _NumPyroCtx, arg: IRArgList) -> SchemaFragment:
        """Emit ``jnp.array([e0, e1, ...])`` for an
        [`IRArgList`][quivers.transpile.ir.IRArgList]."""
        py = ctx.py
        lst = py.v(py.fresh("list"), "list")
        for elem in arg.elements:
            py.e(lst, self._render_arg(ctx, elem), "child_of")
        return call(
            py,
            attribute(py, ("jnp", "array")),
            positional=(lst,),
        )

    def _render_matrix(self, ctx: _NumPyroCtx, arg: IRArgMatrix) -> SchemaFragment:
        """Emit ``jnp.array([[...], [...]])`` for an
        [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix]."""
        py = ctx.py
        outer = py.v(py.fresh("list"), "list")
        for row in arg.rows:
            inner = py.v(py.fresh("list"), "list")
            for elem in row.elements:
                py.e(inner, self._render_arg(ctx, elem), "child_of")
            py.e(outer, inner, "child_of")
        return call(
            py,
            attribute(py, ("jnp", "array")),
            positional=(outer,),
        )

    # ------------------------------------------------------------------
    # Internal walk / emission
    # ------------------------------------------------------------------

    def _emit_imports(self, ctx: _NumPyroCtx) -> None:
        """Emit ``import jax.numpy as jnp``, ``import numpyro``,
        ``import numpyro.distributions``, plus any aliased imports a
        lowered ``LetExprCall`` symbol recorded (``jax.scipy.special`` /
        ``jax.nn``)."""
        py = ctx.py
        self._emit_aliased_import(py, ("jax", "numpy"), "jnp")
        self._emit_plain_import(py, ("numpyro",))
        self._emit_plain_import(py, ("numpyro", "distributions"))
        for chain, alias in sorted(py.required_imports):
            self._emit_aliased_import(py, chain, alias)

    def _emit_plain_import(self, py: PyCtx, chain: tuple[str, ...]) -> None:
        """Emit ``import <a>.<b>.<c>``."""
        stmt = py.v(py.fresh("imp"), "import_statement")
        name = self._dotted_name(py, chain)
        py.e(stmt, name, "name")
        py.e("mod", stmt, "child_of")

    def _emit_aliased_import(
        self, py: PyCtx, chain: tuple[str, ...], alias: str
    ) -> None:
        """Emit ``import <a>.<b> as <alias>``."""
        stmt = py.v(py.fresh("imp"), "import_statement")
        aliased = py.v(py.fresh("alias"), "aliased_import")
        py.e(aliased, self._dotted_name(py, chain), "name")
        py.e(aliased, identifier(py, alias), "alias")
        py.e(stmt, aliased, "name")
        py.e("mod", stmt, "child_of")

    def _dotted_name(self, py: PyCtx, chain: tuple[str, ...]) -> str:
        """Build a ``dotted_name`` vertex from the chain segments."""
        dn = py.v(py.fresh("dn"), "dotted_name")
        for seg in chain:
            py.e(dn, identifier(py, seg), "child_of")
        return dn

    def _function_params(self, ir: IRProgram) -> _Params:
        """Split function parameters into (positional, defaulted).

        Observed inputs (variables also bound as
        [`IRObserve`][quivers.transpile.ir.IRObserve]) get the
        ``=None`` default; all other inputs (scalar program params,
        via fibrations, free names) are positional.
        """
        observed = self._collect_observed(ir)
        positional: list[str] = []
        defaulted: list[str] = []
        for inp in ir.inputs:
            if inp.name in observed:
                defaulted.append(inp.name)
            else:
                positional.append(inp.name)
        return _Params(
            positional=tuple(positional),
            defaulted=tuple(defaulted),
        )

    def _build_function(
        self,
        py: PyCtx,
        body_vid: str,
        params: _Params,
    ) -> str:
        """Build ``def model(a, b, c=None, d=None): <body>`` with both
        positional and defaulted parameter slots.

        Mirrors `function_def` from the shared helpers but emits a
        mixed parameter list: positional identifiers first, then
        default-parameter entries with ``=None`` defaults. The grammar
        permits the mixed form natively.
        """
        func = py.v(py.fresh("fn"), "function_definition")
        fname = identifier(py, "model")
        ps = py.v(py.fresh("ps"), "parameters")
        py.e(func, fname, "name")
        py.e(func, ps, "parameters")
        py.e(func, body_vid, "body")
        for name in params.positional:
            py.e(ps, identifier(py, name), "child_of")
        for name in params.defaulted:
            dp = py.v(py.fresh("dp"), "default_parameter")
            dp_name = identifier(py, name)
            dp_val = py.v(py.fresh("none"), "none")
            py.literal(dp_val, "None")
            py.e(dp, dp_name, "name")
            py.e(dp, dp_val, "value")
            py.e(ps, dp, "child_of")
        return func

    def _collect_observed(self, ir: IRProgram) -> set[str]:
        """Return the set of names bound as
        [`IRObserve`][quivers.transpile.ir.IRObserve] anywhere in the
        body (recursive)."""
        observed: set[str] = set()
        self._collect_observed_in(ir.body, observed)
        return observed

    def _collect_observed_in(
        self,
        body: tuple,
        observed: set[str],
    ) -> None:
        for node in body:
            if isinstance(node, IRObserve):
                observed.add(node.name)
            elif isinstance(node, IRMarginalize):
                self._collect_observed_in(node.scope, observed)

    def _dispatch_body(
        self,
        ctx: _NumPyroCtx,
        body_vid: str,
        nodes: tuple,
    ) -> None:
        """Walk the IR nodes attached to ``body_vid`` (a Python
        ``block`` vertex)."""
        previous_body = ctx.current_body
        ctx.current_body = body_vid
        try:
            for node in nodes:
                self._dispatch_one(ctx, body_vid, node)
        finally:
            ctx.current_body = previous_body

    def _dispatch_one(
        self,
        ctx: _NumPyroCtx,
        body_vid: str,
        node,
    ) -> None:
        if isinstance(node, IRDataInput):
            # NumPyro: data inputs ride on the function signature; no
            # body emission. `render` already handled the header.
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, body_vid, node, observed=False)
                return
            stmt = self._sample_statement(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=False,
            )
            ctx.py.e(body_vid, stmt, "child_of")
            return
        if isinstance(node, IRObserve):
            stmt = self._sample_statement(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=True,
            )
            ctx.py.e(body_vid, stmt, "child_of")
            return
        if isinstance(node, IRDeterministic):
            stmt = self._deterministic_statement(ctx, node)
            ctx.py.e(body_vid, stmt, "child_of")
            return
        if isinstance(node, IRScore):
            self._score_statement(ctx, body_vid, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRCall):
            emit_call_python(ctx.py, body_vid, node, ctx.module, ctx.operations_bound)
            return
        if isinstance(node, IRReturn):
            self._return_statement(ctx, body_vid, node.names)
            return
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [f"node:{type(node).__name__}"],
        )

    def _emit_gp_block(
        self,
        ctx: _NumPyroCtx,
        body_vid: str,
        node: IRSample,
        *,
        observed: bool,
    ) -> None:
        """Emit a Gaussian-process sample as a triple:
        ``__gp_mean_<name> = jnp.zeros(N)``,
        ``__gp_cov_<name>  = jnp.exp(...) + jitter * jnp.eye(N)``,
        ``<name> = numpyro.sample("<name>",
        numpyro.distributions.MultivariateNormal(loc=__gp_mean_<name>,
        covariance_matrix=__gp_cov_<name>))``.

        The kernel-covariance expression is built by hand using the
        Python helper API plus explicit
        `parenthesized_expression`
        nodes, since the let-expression renderer drops
        parenthesisation around nested
        [`LetExprBinOp`][quivers.transpile.ir.LetExprBinOp] children
        and would otherwise mis-group the exponent.
        """
        del observed  # GP samples are never observed in QVR.
        py = ctx.py
        _, kernel_arg = node.args
        if not isinstance(kernel_arg, IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf supported"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = jnp.zeros(N)
        mean_rhs = call(
            py,
            attribute(py, ("jnp", "zeros")),
            positional=(number_literal(py, n),),
        )
        mean_asn = self._assignment_statement(py, mean_name, mean_rhs)
        py.e(body_vid, mean_asn, "child_of")
        # __gp_cov_<name> = jnp.exp(-0.5 * (diff * diff) / (ls * ls)) + jitter * jnp.eye(N)
        # where diff = x.reshape(-1, 1) - x.reshape(1, -1)
        x_col = _python_method_call(
            py,
            identifier(py, x),
            "reshape",
            (_python_unary_minus(py, number_literal(py, 1)), number_literal(py, 1)),
        )
        x_row = _python_method_call(
            py,
            identifier(py, x),
            "reshape",
            (number_literal(py, 1), _python_unary_minus(py, number_literal(py, 1))),
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
            attribute(py, ("jnp", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            py,
            attribute(py, ("jnp", "eye")),
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
        cov_asn = self._assignment_statement(py, cov_name, cov_rhs)
        py.e(body_vid, cov_asn, "child_of")
        # sample call
        mvn_call = call(
            py,
            attribute(
                py,
                ("numpyro", "distributions", "MultivariateNormal"),
            ),
            keyword=(
                ("loc", identifier(py, mean_name)),
                ("covariance_matrix", identifier(py, cov_name)),
            ),
        )
        sample_call = call(
            py,
            attribute(py, ("numpyro", "sample")),
            positional=(string_literal(py, node.name), mvn_call),
        )
        sample_stmt = self._assignment_statement(
            py,
            node.name,
            sample_call,
        )
        py.e(body_vid, sample_stmt, "child_of")

    def _render_sample(
        self,
        ctx: _NumPyroCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        """Build the sample / observe statement and attach it to the
        active body."""
        body_vid = ctx.current_body
        if body_vid is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["sample:no-enclosing-body"],
            )
        stmt = self._sample_statement(
            ctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            observed=observed,
        )
        ctx.py.e(body_vid, stmt, "child_of")
        return stmt

    def _sample_statement(
        self,
        ctx: _NumPyroCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        observed: bool,
    ) -> str:
        """Build ``with numpyro.plate(...): <name> = numpyro.sample(...)``
        (or a bare ``numpyro.sample(...)`` for observes)."""
        py = ctx.py
        # Innermost sample call:
        sample_call = self._numpyro_sample_call(
            ctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            obs_name=name if observed else None,
        )
        if observed:
            inner_stmt = self._expression_statement(py, sample_call)
        else:
            inner_stmt = self._assignment_statement(py, name, sample_call)

        # Wrap in nested `with numpyro.plate("<axis>", <size>):` per
        # batch_dim (outermost = first batch_dim, preserving source
        # `iid_over` order). Inner-most contains the sample statement.
        return self._wrap_in_plates(ctx, plate.batch_dims, inner_stmt)

    def _wrap_in_plates(
        self,
        ctx: _NumPyroCtx,
        batch_dims: tuple,
        inner_stmt: str,
    ) -> str:
        """Nest ``with numpyro.plate(name, size):`` blocks around
        ``inner_stmt``, one per batch dim.

        With no batch dims, returns ``inner_stmt`` directly. With one
        dim, returns a single ``with_statement``; with more dims,
        nests outermost-first so the source order of `iid_over`
        becomes the outermost plate.
        """
        if not batch_dims:
            return inner_stmt
        # Build innermost first, then wrap. NumPyro gives a plate with no
        # stated `dim` the rightmost free batch axis, so the outermost of
        # several nested plates would take the last axis and the draw
        # would come out transposed; each plate of a nest of several
        # states its axis, outermost leftmost.
        current = inner_stmt
        count = len(batch_dims)
        for position in reversed(range(count)):
            dim = batch_dims[position]
            plate_call = self._plate_call(
                ctx, dim, axis=None if count == 1 else position - count
            )
            block = ctx.py.v(ctx.py.fresh("blk"), "block")
            ctx.py.e(block, current, "child_of")
            current = with_statement(
                ctx.py,
                expression=plate_call,
                alias=None,
                body_vid=block,
            )
        return current

    def _plate_call(
        self, ctx: _NumPyroCtx, dim: Dim, *, axis: int | None = None
    ) -> str:
        """Build ``numpyro.plate("<name>", <size>)`` for one
        [`Dim`][quivers.transpile.ir.Dim].

        Parameters
        ----------
        ctx : _NumPyroCtx
            The render state.
        dim : Dim
            The plate's axis.
        axis : int | None
            The negative batch axis the plate occupies, stated as its
            `dim`; `None` leaves NumPyro to allocate it.

        Returns
        -------
        str
            The call's vertex.

        Raises
        ------
        UnsupportedConstruct
            If the axis is neither static nor dynamic.
        """
        py = ctx.py
        if isinstance(dim, DimStatic):
            size_vid = number_literal(py, dim.size)
        elif isinstance(dim, DimDynamic):
            size_vid = self._dynamic_size_expr(py, dim.size_name)
        else:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"plate:dim-kind:{type(dim).__name__}"],
            )
        ctx.emitted_plate_names.add(dim.name)
        keyword: tuple[tuple[str, str], ...] = ()
        if axis is not None:
            keyword = (
                ("dim", _python_unary_minus(py, number_literal(py, float(-axis)))),
            )
        return call(
            py,
            attribute(py, ("numpyro", "plate")),
            positional=(string_literal(py, dim.name), size_vid),
            keyword=keyword,
        )

    def _dynamic_size_expr(self, py: PyCtx, size_name: str) -> str:
        """Render the size expression for a
        [`DimDynamic`][quivers.transpile.ir.DimDynamic].

        Dynamic-axis sizes come from the data: the canonical pattern is
        ``<obs>.shape[0]``. Here we use the bare identifier
        (`size_name`) when it begins with ``N_``, since `Lower`
        synthesises ``N_<axis>`` for dynamic axes; otherwise we
        defer to ``<obs>.shape[0]`` via the first observed name.
        """
        # The canonical LDA emit reads `w.shape[0]` for the dynamic
        # word-observation axis. We match that idiom: prefer
        # `<observed>.shape[0]` when `size_name` is the synthesised
        # `N_<axis>` form.
        if size_name.startswith("N_"):
            axis = size_name[len("N_") :]
            observed_for_axis = self._observed_for_axis(axis)
            if observed_for_axis is not None:
                return self._shape_zero(py, observed_for_axis)
        return identifier(py, size_name)

    def _observed_for_axis(self, axis: str) -> str | None:
        """Heuristic for the canonical NumPyro idiom:
        ``Word_obs`` axis maps to the observed variable ``w``.

        Returns ``None`` when no mapping is known; the caller then
        falls back to the bare ``N_<axis>`` identifier."""
        # The canonical LDA emit pairs `Word_obs` with `w`; the broader
        # convention is to map an axis named `<X>_obs` to the lowercase
        # first character.
        axis_to_obs: dict[str, str] = {
            "Word_obs": "w",
        }
        return axis_to_obs.get(axis)

    def _shape_zero(self, py: PyCtx, var_name: str) -> str:
        """Emit ``<var>.shape[0]``."""
        attr = py.v(py.fresh("attr"), "attribute")
        py.e(attr, identifier(py, var_name), "object")
        py.e(attr, identifier(py, "shape"), "attribute")
        subs = py.v(py.fresh("subs"), "subscript")
        py.e(subs, attr, "value")
        py.e(subs, number_literal(py, 0), "subscript")
        return subs

    def _numpyro_sample_call(
        self,
        ctx: _NumPyroCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        obs_name: str | None,
    ) -> str:
        """Build ``numpyro.sample("<name>", <dist>, [obs=<obs>])``."""
        py = ctx.py
        dist_call = self._distribution_call(ctx, family, args, arg_names, plate)
        dist_call = self._wrap_event_dims(
            ctx, dist_call, family, plate.event_dims, args
        )
        sample_callee = attribute(py, ("numpyro", "sample"))
        positional = (string_literal(py, name), dist_call)
        keyword: tuple[tuple[str, str], ...] = ()
        if obs_name is not None:
            keyword = (("obs", identifier(py, obs_name)),)
        return call(
            py,
            sample_callee,
            positional=positional,
            keyword=keyword,
        )

    def _wrap_event_dims(
        self,
        ctx: _NumPyroCtx,
        dist_call: str,
        family: str,
        event_dims: tuple[Dim, ...],
        args: tuple[IRArg, ...],
    ) -> str:
        """Lift the residual user-declared event dims into the
        distribution via ``.expand([s0, ..., sn]).to_event(n)``.

        The family's natural event rank comes from
        [`FamilyMeta.event_rank`][quivers.transpile.family_meta.FamilyMeta].
        A scalar family (`Normal`, `Beta`, ...; `event_rank=0`) lifts
        every plate event dim, so `Normal(0, 1) [over=LatentDim,
        iid_over=Item]` renders as
        `Normal(0, 1).expand([2]).to_event(1)` inside the `Item` plate
        and draws the `(Item, LatentDim)` shape the QVR sample denotes.
        A vector family (`event_rank=1`) lifts only the dims beyond the
        last; a matrix family (`event_rank=2`) only those beyond the
        last two, so the distribution never re-absorbs an axis it emits
        natively. An argument that already carries the lifted axes (a
        gathered matrix row) gives the family those axes as batch, so
        only `.to_event(n)` is applied.

        Parameters
        ----------
        ctx : _NumPyroCtx
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
        py = ctx.py
        if carries_event_dims(args, residual, ctx.py.name_plates):
            # An argument already carries the lifted axes (a gathered
            # row of a matrix, say): the family's batch holds them, and
            # only their reading as event axes remains.
            return _python_method_call(
                py, dist_call, "to_event", (number_literal(py, len(residual)),)
            )
        list_vid = py.v(py.fresh("list"), "list")
        for dim in residual:
            py.e(list_vid, self._dim_size_vid(ctx, dim), "child_of")
        expand_vid = _python_method_call(py, dist_call, "expand", (list_vid,))
        return _python_method_call(
            py, expand_vid, "to_event", (number_literal(py, len(residual)),)
        )

    def _dim_size_vid(self, ctx: _NumPyroCtx, dim: Dim) -> str:
        """Render `dim.size` as a NumPyro source-token vid."""
        if isinstance(dim, DimStatic):
            return number_literal(ctx.py, dim.size)
        if isinstance(dim, DimDynamic):
            return self._dynamic_size_expr(ctx.py, dim.size_name)
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [f"plate:dim-kind:{type(dim).__name__}"],
        )

    def _distribution_call(
        self,
        ctx: _NumPyroCtx,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> str:
        """Build ``numpyro.distributions.<TargetName>(arg=value, ...)``.

        Looks up the target distribution name in `FAMILY_META`, applies
        any `arg_aliases` rename, and emits keyword arguments using
        each family's torch ``arg_constraints`` parameter names.

        Several families need a NumPyro-specific reparameterisation
        because the target distribution's parameters differ from the
        QVR/torch canonical ones. These are handled before the generic
        keyword emission:

        - `NegativeBinomial` -> `NegativeBinomial2(mean, concentration)`,
          converting ``(total_count, probs)``;
        - `MatrixNormal` -> Cholesky ``scale_tril_row`` /
          ``scale_tril_column`` factors of the row/column covariances;
        - `LKJCholesky` / `LKJCorrelationFactor` -> a leading
          matrix-dimension positional drawn from the event axis;
        - families NumPyro does not ship (`LogitNormal`, `HalfStudentT`,
          `ContinuousBernoulli`, ...) -> a bare call to the grafted
          runtime-helper class.

        Wrapper-family handling: when ``family == "Truncated"`` and the
        first arg is an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        pointing at a morphism with ``~ Normal(loc, scale)``, the
        emission collapses to
        ``numpyro.distributions.TruncatedNormal(loc=..., scale=...,
        low=lo, high=hi)`` per §10.10 of the spec.
        """
        py = ctx.py
        if family not in FAMILY_META:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:unknown:{family}"],
            )
        meta = FAMILY_META[family]
        if _BACKEND not in meta.target_names:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:no-target-name:{family}"],
            )

        # Wrapper-family inline emission for Truncated(base, lo, hi).
        wrapper_call = self._maybe_wrapper_call(ctx, family, args, arg_names)
        if wrapper_call is not None:
            return wrapper_call

        if family == "NegativeBinomial":
            return self._negative_binomial2_call(ctx, args, arg_names)
        if family == "MatrixNormal":
            return self._matrix_normal_call(ctx, meta, args, arg_names)
        if family in _NUMPYRO_MATRIX_DIMENSION_FAMILIES:
            return self._matrix_dimension_call(ctx, meta, args, arg_names, plate)
        if family in _NUMPYRO_RUNTIME_HELPER_FAMILIES:
            return self._runtime_helper_call(ctx, meta, args, arg_names)
        if family == "MixtureNormal":
            return self._mixture_same_family_call(ctx, meta, args, arg_names)

        target_name = meta.target_names[_BACKEND]
        callee = attribute(py, ("numpyro", "distributions", target_name))
        aliases = meta.arg_aliases.get(_BACKEND, {})
        cls_constraints = getattr(
            meta.distribution_class,
            "arg_constraints",
            {},
        )
        if not isinstance(cls_constraints, dict):
            cls_constraints = {}
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            emitted_name = aliases.get(name, name)
            value_vid = self._render_arg(
                ctx,
                self._maybe_broadcast_ref(
                    ctx,
                    arg,
                    arg_name=name,
                    cls_constraints=cls_constraints,
                    plate=plate,
                ),
            )
            keyword.append((emitted_name, value_vid))
        return call(py, callee, keyword=tuple(keyword))

    def _maybe_broadcast_ref(
        self,
        ctx: _NumPyroCtx,
        arg: IRArg,
        *,
        arg_name: str,
        cls_constraints: dict[str, c.Constraint],
        plate: Plate,
    ) -> IRArg:
        """Wrap a scalar [`IRArgRef`][quivers.transpile.ir.IRArgRef] in
        [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] when the
        family's ``arg_constraints[arg_name]`` expects a vector or a
        matrix.

        `Lower` leaves an [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        unwrapped on the assumption the referenced binding already has
        the right shape. That holds for a genuinely vector-valued
        binding but not for a scalar one, whether a free scalar data
        input or a let-bound scalar literal / expression:
        ``Dirichlet(concentration=a)`` with a scalar ``a`` is a rank
        error in NumPyro. Broadcasting to the surrounding plate's event
        axes recovers the declared shape, and a constant concentration
        vector is exactly what the QVR ``[over=...]`` clause denotes.

        Scalar-ness is read from the IR classification carried on `ctx`
        (`scalar_refs` / `bound_refs`), never from `IRProgram.inputs`
        alone, so a let-bound scalar is repaired identically to a free
        one. A reference the classification cannot resolve raises rather
        than emitting an unverified rank; a dynamic or under-ranked
        event axis likewise raises rather than silently passing through.
        """
        expected = cls_constraints.get(arg_name)
        expected_event_dim = (
            expected.event_dim if isinstance(expected, c._IndependentConstraint) else 0
        )
        if expected_event_dim < 1:
            # Scalar-parameter slot (loc, scale, rate, probs-as-simplex,
            # ...): a scalar reference already satisfies it and a
            # vector-valued one is emitted verbatim. Nothing to lift.
            return arg
        if not isinstance(arg, IRArgRef):
            # `IRArgList` / `IRArgMatrix` are literal vectors / matrices,
            # `IRArgBroadcast` is a scalar `Lower` already broadcast, and
            # `IRArgFamilyRef` names a component distribution: each
            # reaches a vector / matrix slot already correctly shaped.
            return arg
        if arg.indices:
            # An indexed reference (`mu[g]`) selects along a batch axis,
            # so its rank is fixed by the indexing rather than by this
            # slot; the referenced binding already carries the event
            # shape at the selected position.
            return arg
        if arg.name in ctx.scalar_refs:
            target = self._static_event_shape(plate, expected_event_dim, arg_name)
            return IRArgBroadcast(value=arg, target_shape=target)
        if arg.name in ctx.bound_refs:
            # The referenced binding already carries a vector / matrix
            # shape (a sampled simplex, a declared vector input, a
            # let-bound list literal); emit it unchanged.
            return arg
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [
                f"broadcast:{arg_name}:unresolved-ref:{arg.name}: cannot "
                "classify the referenced binding's shape to decide "
                "whether a scalar-to-event broadcast is required"
            ],
        )

    def _static_event_shape(
        self, plate: Plate, event_dim: int, arg_name: str
    ) -> tuple[int, ...]:
        """The static event shape a scalar arg must broadcast to.

        Takes the first `event_dim` axes off the sample's
        [`Plate.event_dims`][quivers.transpile.ir.Plate]. A plate that
        carries fewer event axes than the constraint requires, or an
        axis whose length is only known at run time
        ([`DimDynamic`][quivers.transpile.ir.DimDynamic]), raises:
        NumPyro's ``jnp.full`` broadcast needs a compile-time shape, and
        silently dropping the axis would regress a dynamic event axis to
        a rank error.
        """
        dims = plate.event_dims
        if len(dims) < event_dim:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [
                    f"broadcast:{arg_name}:event-rank:{len(dims)}<"
                    f"{event_dim}: the sample's plate carries fewer event "
                    "axes than the family constraint requires"
                ],
            )
        sizes: list[int] = []
        for dim in dims[:event_dim]:
            if isinstance(dim, DimStatic):
                sizes.append(dim.size)
            elif isinstance(dim, DimDynamic):
                raise UnsupportedConstruct(
                    "qvr-numpyro",
                    [
                        f"broadcast:{arg_name}:dynamic-event-axis:"
                        f"{dim.name}: NumPyro's jnp.full broadcast needs "
                        "a static event size"
                    ],
                )
            else:
                raise UnsupportedConstruct(
                    "qvr-numpyro",
                    [f"broadcast:{arg_name}:event-dim-kind:{type(dim).__name__}"],
                )
        return tuple(sizes)

    def _negative_binomial2_call(
        self,
        ctx: _NumPyroCtx,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """``NegativeBinomial2(mean=r*p/(1-p), concentration=r)``.

        NumPyro's ``NegativeBinomial2`` is mean / dispersion
        parameterised, whereas QVR/torch ``NegativeBinomial(total_count
        r, probs p)`` counts failures with per-trial success ``p``. The
        NB2 concentration equals ``r`` and the mean equals
        ``r * p / (1 - p)``, which reproduces the torch pmf and both
        moments exactly.
        """
        py = ctx.py
        by_name = dict(zip(arg_names, args, strict=False))
        total = by_name.get("total_count")
        probs = by_name.get("probs")
        if total is None or probs is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["family:NegativeBinomial:expected (total_count, probs)"],
            )
        total_vid = self._render_arg(ctx, total)
        probs_vid = self._render_arg(ctx, probs)
        one = number_literal(py, 1)
        # mean = total_count * probs / (1 - probs)
        one_minus_p = _python_paren(py, _python_binary_op(py, "-", one, probs_vid))
        numerator = _python_paren(py, _python_binary_op(py, "*", total_vid, probs_vid))
        mean_expr = _python_binary_op(py, "/", numerator, one_minus_p)
        callee = attribute(py, ("numpyro", "distributions", "NegativeBinomial2"))
        return call(
            py,
            callee,
            keyword=(
                ("mean", mean_expr),
                ("concentration", self._render_arg(ctx, total)),
            ),
        )

    def _matrix_normal_call(
        self,
        ctx: _NumPyroCtx,
        meta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """``MatrixNormal(loc, scale_tril_row, scale_tril_column)``.

        NumPyro's ``MatrixNormal`` takes Cholesky factors of the row and
        column covariances, not the covariances themselves. The row /
        column covariance args are wrapped in ``jnp.linalg.cholesky``;
        the factors reconstruct the Kronecker covariance exactly, so the
        matrix-normal density is unchanged.
        """
        py = ctx.py
        rename: dict[str, str] = {
            "row_covariance": "scale_tril_row",
            "col_covariance": "scale_tril_column",
        }
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            value_vid = self._render_arg(ctx, arg)
            if name in ("row_covariance", "col_covariance"):
                value_vid = call(
                    py,
                    attribute(py, ("jnp", "linalg", "cholesky")),
                    positional=(value_vid,),
                )
            keyword.append((rename.get(name, name), value_vid))
        callee = attribute(
            py, ("numpyro", "distributions", meta.target_names[_BACKEND])
        )
        return call(py, callee, keyword=tuple(keyword))

    def _matrix_dimension_call(
        self,
        ctx: _NumPyroCtx,
        meta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> str:
        """``LKJCholesky(<dim>, concentration=...)`` /
        ``LKJ(<dim>, concentration=...)``.

        NumPyro's LKJ families require the correlation-matrix dimension
        as a leading positional argument. QVR carries it on the sample's
        event axis; the first event dim's size is emitted before the
        concentration keyword.
        """
        py = ctx.py
        dimension = self._event_matrix_dimension(plate, meta)
        callee = attribute(
            py, ("numpyro", "distributions", meta.target_names[_BACKEND])
        )
        aliases = meta.arg_aliases.get(_BACKEND, {})
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            keyword.append((aliases.get(name, name), self._render_arg(ctx, arg)))
        return call(
            py,
            callee,
            positional=(number_literal(py, dimension),),
            keyword=tuple(keyword),
        )

    def _event_matrix_dimension(self, plate: Plate, meta) -> int:
        """The square-matrix dimension for an LKJ family, read from the
        first event dim of the sample's plate."""
        if not plate.event_dims:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:{meta.qvr_name}:missing-event-dimension"],
            )
        first = plate.event_dims[0]
        if not isinstance(first, DimStatic):
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:{meta.qvr_name}:non-static-event-dimension"],
            )
        return int(first.size)

    def _runtime_helper_call(
        self,
        ctx: _NumPyroCtx,
        meta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """Bare ``<HelperClass>(arg=value, ...)`` for a family NumPyro
        does not ship, resolved by the grafted runtime-helper class of
        the same name."""
        py = ctx.py
        target_name = meta.target_names[_BACKEND]
        callee = identifier(py, target_name)
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            keyword.append((name, self._render_arg(ctx, arg)))
        return call(py, callee, keyword=tuple(keyword))

    def _mixture_same_family_call(
        self,
        ctx: _NumPyroCtx,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """Emit `MixtureSameFamily(Categorical(probs), Normal(loc, scale))`.

        NumPyro ships no single-name Gaussian-mixture family, but it
        ships the mixture *construction*: a categorical mixing
        distribution over a batch of components whose batch axis is the
        component axis. `MixtureNormal(w, mu, sigma)` maps onto it
        exactly, with `Categorical(probs=w)` supplying the weights and
        `Normal(loc=mu, scale=sigma)` the K components, so the emitted
        distribution's `log_prob` is the same log-sum-exp the QVR
        likelihood scores rather than an approximation of it.
        """
        py = ctx.py
        weights, loc, scale = mixture_normal_components(_BACKEND, args, arg_names)
        mixing = call(
            py,
            attribute(py, ("numpyro", "distributions", "Categorical")),
            keyword=(("probs", self._render_arg(ctx, weights)),),
        )
        component = call(
            py,
            attribute(py, ("numpyro", "distributions", "Normal")),
            keyword=(
                ("loc", self._render_arg(ctx, loc)),
                ("scale", self._render_arg(ctx, scale)),
            ),
        )
        return call(
            py,
            attribute(
                py,
                ("numpyro", "distributions", meta.target_names[_BACKEND]),
            ),
            positional=(mixing, component),
        )

    def _maybe_wrapper_call(
        self,
        ctx: _NumPyroCtx,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str | None:
        """Inline-render a `Truncated(base, lo, hi)` call as
        ``TruncatedNormal(loc=..., scale=..., low=lo, high=hi)``
        when ``base`` resolves to a `Normal(...)` morphism.

        Falls through (returns ``None``) when the family is not a
        wrapper or the base resolution does not match a known
        truncation specialisation; the general path emits the wrapper
        via `TruncatedDistribution`.
        """
        if family != "Truncated":
            return None
        if not args:
            return None
        base = args[0]
        if not isinstance(base, IRArgFamilyRef):
            return None
        py = ctx.py
        decl = ctx.morphisms.get(base.name)
        if decl is None or decl.init_family is None:
            # Fall back to TruncatedDistribution(<base_ref>, low, high).
            return self._truncated_distribution_call(
                ctx,
                base_ref=identifier(py, base.name),
                args=args[1:],
                arg_names=arg_names[1:],
            )
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META.get(inner_family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:wrapper-inner-unknown:{inner_family}"],
            )
        # Truncated-specialisation registry: when NumPyro publishes a
        # dedicated `Truncated<Inner>` class for the inner family, prefer
        # that emission over the generic `TruncatedDistribution(<inner>,
        # ...)` wrapper.
        specialised = _NUMPYRO_TRUNCATED_SPECIALISATION.get(inner_family)
        if specialised is not None:
            return self._truncated_specialised_call(
                ctx,
                target=specialised,
                base_decl=decl,
                rest_args=args[1:],
                rest_names=arg_names[1:],
            )
        # General wrapper: build the inner distribution call and pass
        # it as the first positional arg to TruncatedDistribution.
        inner_call = self._inner_morphism_call(ctx, decl)
        return self._truncated_distribution_call(
            ctx,
            base_ref=inner_call,
            args=args[1:],
            arg_names=arg_names[1:],
        )

    def _truncated_specialised_call(
        self,
        ctx: _NumPyroCtx,
        *,
        target: str,
        base_decl,
        rest_args: tuple[IRArg, ...],
        rest_names: tuple[str, ...],
    ) -> str:
        """``numpyro.distributions.<TruncatedInner>(<inner_args>,
        low=lo, high=hi)`` for inner families that NumPyro publishes
        a specialised truncated wrapper for."""
        py = ctx.py
        callee = attribute(py, ("numpyro", "distributions", target))
        base_args = bind_family_arguments(
            base_decl.init_family.family, base_decl.init_family.args or ()
        )
        keyword: list[tuple[str, str]] = []
        for name, arg in base_args:
            keyword.append((name, arg_expr(py, atom_value(arg))))
        for arg, name in zip(rest_args, rest_names, strict=False):
            keyword.append((name, self._render_arg(ctx, arg)))
        return call(py, callee, keyword=tuple(keyword))

    def _inner_morphism_call(self, ctx: _NumPyroCtx, decl) -> str:
        """Build a `numpyro.distributions.<Inner>(...)` call from the
        morphism's `~ Family(...)` init clause."""
        py = ctx.py
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META[inner_family]
        target = inner_meta.target_names[_BACKEND]
        callee = attribute(py, ("numpyro", "distributions", target))
        keyword: list[tuple[str, str]] = []
        for name, arg in bind_family_arguments(
            inner_family, decl.init_family.args or ()
        ):
            keyword.append((name, arg_expr(py, atom_value(arg))))
        return call(py, callee, keyword=tuple(keyword))

    def _truncated_distribution_call(
        self,
        ctx: _NumPyroCtx,
        *,
        base_ref: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """``numpyro.distributions.TruncatedDistribution(<base>, low=lo,
        high=hi)``."""
        py = ctx.py
        callee = attribute(py, ("numpyro", "distributions", "TruncatedDistribution"))
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            keyword.append((name, self._render_arg(ctx, arg)))
        return call(
            py,
            callee,
            positional=(base_ref,),
            keyword=tuple(keyword),
        )

    def _render_arg(self, ctx: _NumPyroCtx, arg: IRArg) -> str:
        """Render one IR arg to a Python expression vertex."""
        py = ctx.py
        if isinstance(arg, IRArgNumber):
            return number_literal(py, arg.value)
        if isinstance(arg, IRArgRef):
            if not arg.indices:
                return identifier(py, arg.name)
            return self._render_indexed_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            decl = ctx.morphisms.get(arg.name)
            if decl is None or decl.init_family is None:
                return identifier(py, arg.name)
            return self._inner_morphism_call(ctx, decl)
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [f"arg:unknown:{type(arg).__name__}"],
        )

    def _render_indexed_ref(self, ctx: _NumPyroCtx, arg: IRArgRef) -> str:
        """Render ``name[i0][i1]...`` as nested ``subscript`` vertices."""
        py = ctx.py
        current = identifier(py, arg.name)
        for idx in arg.indices:
            subs = py.v(py.fresh("subs"), "subscript")
            py.e(subs, current, "value")
            py.e(subs, self._render_arg(ctx, idx), "subscript")
            current = subs
        return current

    def _render_shape_tuple(self, py: PyCtx, shape: tuple[int, ...]) -> str:
        """Emit ``(K,)`` for a 1-tuple or ``(R, C)`` for a 2-tuple.

        Tree-sitter Python's `tuple` node carries the punctuation via
        `chose-alt-fingerprint` / `chose-alt-child-kinds` / `ptrace-*`
        constraints; without them the emitter drops the trailing
        comma on a 1-tuple and produces a `parenthesized_expression`
        rather than a `tuple`. The constraints below mirror what the
        Python tree-sitter parser produces when reading
        ``(<n>,)`` / ``(<r>, <c>)`` literally.
        """
        tup = py.v(py.fresh("tup"), "tuple")
        n = len(shape)
        if n == 0:
            py.constraint(tup, "chose-alt-fingerprint", "()")
            py.constraint(tup, "ptrace-0", "T(")
            py.constraint(tup, "ptrace-1", "T)")
            return tup
        kind_list = " ".join("integer" for _ in range(n))
        if n == 1:
            py.constraint(tup, "chose-alt-fingerprint", "( ,)")
            py.constraint(tup, "ptrace-0", "T(")
            py.constraint(tup, "ptrace-1", "Cinteger")
            py.constraint(tup, "ptrace-2", "T,")
            py.constraint(tup, "ptrace-3", "T)")
        else:
            fingerprint = "( " + " ".join("," for _ in range(n - 1)) + " )"
            py.constraint(tup, "chose-alt-fingerprint", fingerprint)
            py.constraint(tup, "ptrace-0", "T(")
            slot = 1
            for i in range(n):
                py.constraint(tup, f"ptrace-{slot}", "Cinteger")
                slot += 1
                if i < n - 1:
                    py.constraint(tup, f"ptrace-{slot}", "T,")
                    slot += 1
            py.constraint(tup, f"ptrace-{slot}", "T)")
        py.constraint(tup, "chose-alt-child-kinds", kind_list)
        for size in shape:
            py.e(tup, number_literal(py, size), "child_of")
        return tup

    def _expression_statement(self, py: PyCtx, expr: str) -> str:
        """Wrap an expression in an ``expression_statement`` vertex."""
        es = py.v(py.fresh("es"), "expression_statement")
        py.e(es, expr, "child_of")
        return es

    def _assignment_statement(self, py: PyCtx, lhs_name: str, rhs: str) -> str:
        """``<lhs_name> = <rhs>``."""
        asn = py.v(py.fresh("asn"), "assignment")
        py.e(asn, identifier(py, lhs_name), "left")
        py.e(asn, rhs, "right")
        return asn

    def _deterministic_statement(self, ctx: _NumPyroCtx, node: IRDeterministic) -> str:
        """``<name> = <expr>`` for an
        [`IRDeterministic`][quivers.transpile.ir.IRDeterministic]
        let-binding."""
        py = ctx.py
        rhs = render_deterministic_python(py, node)
        return self._assignment_statement(py, node.name, rhs)

    def _score_statement(self, ctx: _NumPyroCtx, body_vid: str, node: IRScore) -> None:
        """``<name> = <expr>``; ``numpyro.factor("<name>", <name>)``."""
        py = ctx.py
        rhs = render_let_expr_python(py, node.expr)
        py.e(
            body_vid,
            self._assignment_statement(py, node.name, rhs),
            "child_of",
        )
        factor_call = call(
            py,
            attribute(py, ("numpyro", "factor")),
            positional=(
                string_literal(py, node.name),
                identifier(py, node.name),
            ),
        )
        py.e(
            body_vid,
            self._expression_statement(py, factor_call),
            "child_of",
        )

    def _return_statement(
        self, ctx: _NumPyroCtx, body_vid: str, names: tuple[str, ...]
    ) -> None:
        """``return <var>`` (single) or ``return (<a>, <b>, ...)`` (tuple)."""
        if not names:
            return
        py = ctx.py
        rs = py.v(py.fresh("ret"), "return_statement")
        if len(names) == 1:
            py.e(rs, identifier(py, names[0]), "child_of")
        else:
            elist = py.v(py.fresh("elist"), "expression_list")
            for name in names:
                py.e(elist, identifier(py, name), "child_of")
            py.e(rs, elist, "child_of")
        py.e(body_vid, rs, "child_of")


# ---------------------------------------------------------------------------
# Renderer-local context
# ---------------------------------------------------------------------------


class _NumPyroCtx(RenderContext):
    """NumPyro-specific extension of
    [`RenderContext`][quivers.transpile.renderers.RenderContext]
    carrying the Python-helpers context and the currently-active body
    vertex used by nested-plate emission."""

    def __init__(
        self,
        *,
        sb: panproto.SchemaBuilder,
        morphisms: dict,
        lets: dict,
        py: PyCtx,
        observed_names: set[str],
        scalar_refs: frozenset[str],
        bound_refs: frozenset[str],
        module: IRQiecModule,
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=lets)
        self.py = py
        self.observed_names = observed_names
        #: Names of scalar-shaped bindings (empty plate, scalar support
        #: for inputs / samples / observes / marginalize latents;
        #: non-list / non-factor let-bindings). Broadcast candidates.
        self.scalar_refs = scalar_refs
        #: Every bindable name in the program (inputs plus body,
        #: descending into marginalize scopes). A vector-slot reference
        #: outside this set is unresolvable and raises.
        self.bound_refs = bound_refs
        self.current_body: str | None = None
        self.emitted_plate_names: set[str] = set()
        self.module: IRQiecModule = module
        self.operations_bound: set[str] = set()


def _as_numpyro_ctx(ctx: RenderContext) -> _NumPyroCtx:
    """Narrow a base `RenderContext` to the NumPyro extension. The
    renderer always constructs `_NumPyroCtx` instances, so this is a
    safe assertion at the boundary."""
    if not isinstance(ctx, _NumPyroCtx):
        raise UnsupportedConstruct(
            "qvr-numpyro",
            ["ctx:type-mismatch"],
        )
    return ctx


def _is_scalar_shape(plate: Plate) -> bool:
    """A binding whose plate carries neither batch nor event axes is
    rank-0 in every replicated / joint dimension."""
    return not plate.batch_dims and not plate.event_dims


def _classify_bindings(
    ir: IRProgram,
) -> tuple[frozenset[str], frozenset[str]]:
    """Classify every bindable name in `ir` as scalar and / or bound.

    Returns ``(scalar_refs, bound_refs)``. A name is *bound* when the
    program introduces it (a data input, sample, observe, deterministic
    let, or marginalize latent). A bound name is *scalar* when it is
    rank-0: an empty plate plus a scalar-support constraint for the
    stochastic / input bindings, and an empty plate whose let-expression
    is not a vector-producing list / factor construct for the
    deterministic ones. A let-bound scalar literal thus lands in
    `scalar_refs` exactly like a free scalar input, so the NumPyro
    broadcast fires on both.
    """
    scalar: set[str] = set()
    bound: set[str] = set()

    def _record_stochastic(name: str, plate: Plate, constraint: ConstraintSpec) -> None:
        bound.add(name)
        if _is_scalar_shape(plate) and (event_dim_of(constraint.to_constraint()) == 0):
            scalar.add(name)

    def _visit(nodes: tuple[IRNode, ...]) -> None:
        for node in nodes:
            if isinstance(node, (IRDataInput, IRSample, IRObserve)):
                _record_stochastic(node.name, node.plate, node.constraint)
            elif isinstance(node, IRDeterministic):
                bound.add(node.name)
                if _is_scalar_shape(node.plate) and not isinstance(
                    node.expr, (LetExprList, LetExprFactor)
                ):
                    scalar.add(node.name)
            elif isinstance(node, IRMarginalize):
                _record_stochastic(node.latent, node.plate, node.constraint)
                _visit(node.scope)

    _visit(ir.inputs)
    _visit(ir.body)
    return frozenset(scalar), frozenset(bound)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Params:
    """Split function parameters into positional vs. defaulted lists."""

    def __init__(
        self,
        *,
        positional: tuple[str, ...],
        defaulted: tuple[str, ...],
    ) -> None:
        self.positional = positional
        self.defaulted = defaulted


# ---------------------------------------------------------------------------
# Runtime-helper grafting
# ---------------------------------------------------------------------------


_RUNTIME_NUMPYRO_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_numpyro.py"
)


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
    """Every vertex id reachable from `root` via outgoing edges."""
    seen: set[str] = {root}
    frontier: list[str] = [root]
    while frontier:
        src = frontier.pop()
        for edge in schema.edges:
            if edge.src == src and edge.tgt not in seen:
                seen.add(edge.tgt)
                frontier.append(edge.tgt)
    return seen


def _load_runtime_numpyro_helpers() -> tuple[
    panproto.Schema, dict[str, str], dict[str, set[str]]
]:
    """Parse [`runtime_numpyro.py`][quivers.transpile.runtime_numpyro]
    through panproto's Python tree-sitter grammar at module-load time
    and index every helper `class_definition` subtree by class name.

    Returns the parsed schema, a map from class name to the
    class-definition vertex id, and a map from class name to the set of
    vertex ids in that class's subtree. The renderer's
    `_emit_runtime_helper`
    grafts a class subtree (vertex + all descendants + their
    constraints + edges) into the per-render schema as a `child_of`
    of the emitted module. The emit is a real Python class definition,
    with no string literal, no `exec`, and no runtime self-injection.
    """
    schema = parser_registry().parse_with_protocol(
        "python",
        _RUNTIME_NUMPYRO_PATH.read_bytes(),
        str(_RUNTIME_NUMPYRO_PATH),
    )
    roots: dict[str, str] = {}
    for v in schema.vertices:
        if v.kind == "class_definition":
            name = _class_definition_name(schema, v.id)
            if name is not None:
                roots[name] = v.id
    if not roots:
        raise RuntimeError(
            f"no helper `class` definitions found in {_RUNTIME_NUMPYRO_PATH}; "
            "the renderer expects it as the source of truth for the "
            "embedded NumPyro runtime helpers."
        )
    subtrees = {name: _subtree_vertex_ids(schema, root) for name, root in roots.items()}
    return schema, roots, subtrees


(
    _RUNTIME_NUMPYRO_HELPER_SCHEMA,
    _RUNTIME_NUMPYRO_HELPER_ROOTS,
    _RUNTIME_NUMPYRO_HELPER_SUBTREES,
) = _load_runtime_numpyro_helpers()


def _ir_helper_classes_used(body: tuple[IRNode, ...]) -> set[str]:
    """The set of runtime-helper class names the IR body needs grafted.

    A helper is needed when an [`IRSample`][quivers.transpile.ir.IRSample]
    / [`IRObserve`][quivers.transpile.ir.IRObserve] (including nested
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes) uses a
    family in
    `_NUMPYRO_RUNTIME_HELPER_FAMILIES`
    whose NumPyro target name matches a `class` in
    [`runtime_numpyro.py`][quivers.transpile.runtime_numpyro].
    """
    used: set[str] = set()
    for node in body:
        if isinstance(node, (IRSample, IRObserve)):
            if node.family in _NUMPYRO_RUNTIME_HELPER_FAMILIES:
                meta = FAMILY_META.get(node.family)
                if meta is not None:
                    target = meta.target_names.get(_BACKEND)
                    if target in _RUNTIME_NUMPYRO_HELPER_ROOTS:
                        used.add(target)
        elif isinstance(node, IRMarginalize):
            used |= _ir_helper_classes_used(node.scope)
    return used


def _emit_runtime_helper(py: PyCtx, class_name: str) -> None:
    """Graft the named helper `class` subtree from
    [`runtime_numpyro.py`][quivers.transpile.runtime_numpyro] into the
    per-render schema as a top-level child of `mod`.

    The renderer copies every vertex, constraint, and internal edge of
    the parsed class subtree into the per-render builder, then attaches
    the class root as a `child_of` of `mod`. Subsequent bare call sites
    (e.g. ``LogitNormal(loc=..., scale=...)``) in the rendered model
    body resolve to that class through normal Python name lookup. Vertex
    ids are rewritten through `py.fresh` so the graft does not collide
    with builder-allocated ids.
    """
    src_schema = _RUNTIME_NUMPYRO_HELPER_SCHEMA
    subtree = _RUNTIME_NUMPYRO_HELPER_SUBTREES[class_name]
    root = _RUNTIME_NUMPYRO_HELPER_ROOTS[class_name]
    id_map: dict[str, str] = {}
    for old in subtree:
        new = py.fresh("rh")
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        py.v(new, kind)
        for cstr in src_schema.constraints_for(old):
            py.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            py.e(id_map[edge.src], id_map[edge.tgt], edge.kind)
    py.e("mod", id_map[root], "child_of")


__all__ = ["NumPyroRenderer"]
