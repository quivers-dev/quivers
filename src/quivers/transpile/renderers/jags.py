"""Render transpilation IR as JAGS source.

Samples and observations become relations inside plate loops. Finite
marginalizations are emitted as weighted density sums through the
zeros trick. Family names and argument conversions come from
``FAMILY_META``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import panproto
import torch.distributions.constraints as _torch_constraints

from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    CSPositiveDefinite,
    CSReal,
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
    event_dim_of,
)
from quivers.transpile.renderers._one_based import is_class_code_family
from quivers.transpile.renderers._bugs_helpers import (
    SQRT_TWO_PI,
    TRUNCATION_FINGERPRINT,
    beta_binomial_log_pmf,
    build_decl_plates,
    continuous_bernoulli_log_pdf,
    factor_axis_sizes,
    factor_cells,
    half_support_truncation,
    hoist_list_operands,
    event_loop_view,
    index_letexpr_refs,
    inline_letexpr,
    irarg_letexpr,
    kumaraswamy_log_pdf,
    list_cells,
    list_sizes,
    marginal_scope_density,
    push_scalar_dets_into_loops,
    render_let_expr_bugs,
    reorder_binomial_dbin,
    reorder_half_studentt_dt,
    reorder_pareto_dpar,
    split_event_dims,
    subscript_letexpr,
)
from quivers.transpile.renderers._python_helpers import (
    marginal_support_size,
    marginal_weight_probs,
    marginalize_body,
    marginalize_fibration,
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
    ir_uses_family,
    logit_weighted,
    mixture_component_count,
    mixture_normal_components,
    reorder_negbin_args,
    reorder_weibull_args,
)
from quivers.transpile.renderers._qiec import render_computations_static
from quivers.transpile.renderers._rows import scalar_parameter, visited_position


#: The backend key consulted in
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] for
#: ``target_names`` and ``arg_aliases`` lookups.
_BACKEND: str = "jags"

#: Renderer-internal table: when an
#: [`arg_aliases`][quivers.transpile.family_meta.FamilyMeta.arg_aliases]
#: entry renames an arg to one of these target names, wrap the
#: source value in the matching arithmetic transform before emitting.
#: Normal's ``scale`` -> ``tau`` is the canonical case: JAGS
#: parameterises Normal by precision, so the renderer emits
#: ``1/(scale*scale)``.
_TransformKind = Literal["inv_square", "inv", "neg", "log", "exp"]

_ALIAS_TRANSFORMS: dict[str, _TransformKind] = {
    "tau": "inv_square",
}

#: Per-family override of the arg-alias arithmetic transform. The
#: shared ``tau`` alias assumes a precision (``1/scale^2``)
#: parameterisation, which is right for JAGS' ``dnorm`` / ``dt`` but
#: wrong for ``ddexp``. JAGS' ``ddexp(mu, tau)`` is rate-parameterised
#: (density ``(tau/2) * exp(-tau*|x-mu|)``), so ``Laplace``'s scale
#: maps to the rate ``tau = 1/scale`` rather than ``1/scale^2``.
_FAMILY_ALIAS_TRANSFORM_OVERRIDE: dict[str, dict[str, _TransformKind]] = {
    "Laplace": {"tau": "inv"},
    "Logistic": {"tau": "inv"},
}

#: Renderer-supplied arg renames for families whose
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] entry
#: records no ``jags`` alias even though the JAGS distribution is
#: parameterised differently from the torch one. ``Logistic`` is the
#: case: torch carries ``(loc, scale)`` while JAGS' ``dlogis(mu,
#: tau)`` has density ``tau * e^{-tau(x-mu)} / (1 +
#: e^{-tau(x-mu)})^2``, i.e. the second slot is the rate ``tau =
#: 1/scale``. Passing the scale through unchanged scores a Logistic
#: of scale ``1/s``, which is a different density at every point
#: rather than a constant offset. The rename feeds the same transform
#: pipeline the ``FAMILY_META`` aliases do, with the ``inv``
#: transform supplied by
#: `_FAMILY_ALIAS_TRANSFORM_OVERRIDE`.
_FAMILY_ALIAS_OVERRIDE: dict[str, dict[str, str]] = {
    "Logistic": {"scale": "tau"},
    "LogNormal": {"scale": "tau"},
}


def _alias_transform_for(family: str, emitted_name: str) -> _TransformKind | None:
    """Resolve the arithmetic transform for an aliased arg, honouring
    the per-family override in
    `_FAMILY_ALIAS_TRANSFORM_OVERRIDE`
    before falling back to the shared
    `_ALIAS_TRANSFORMS`
    table."""
    override = _FAMILY_ALIAS_TRANSFORM_OVERRIDE.get(family)
    if override is not None and emitted_name in override:
        return override[emitted_name]
    return _ALIAS_TRANSFORMS.get(emitted_name)


def _reorder_studentt_dt(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape a location-scale ``StudentT(df, loc, scale)`` into JAGS'
    ``dt(mu, tau, k)`` argument order.

    JAGS' Student-t is parameterised by location, precision, and
    degrees of freedom, in that order; torch's ``StudentT`` carries
    ``(df, loc, scale)``. This reorders the three arguments to
    ``(loc, scale, df)`` and pre-wraps the scale in the
    precision transform ``tau = 1/(scale*scale)`` so the emitted call
    is ``dt(loc, 1/(scale*scale), df)``.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    return (
        (
            by_name["loc"],
            IRArgTransform(inner=by_name["scale"], transform="inv_square"),
            by_name["df"],
        ),
        ("loc", "tau", "df"),
    )


#: JAGS-side argument injection for QVR families whose underlying
#: torch distribution carries fewer parameters than the JAGS
#: distribution it maps to. ``HalfNormal(scale)`` maps to JAGS'
#: ``dnorm(0, tau)``; the renderer prepends ``IRArgNumber(0)`` under
#: the loc-position arg name so the alias-transform pipeline still
#: rewrites the scale into ``tau = 1/(scale*scale)``. The symmetric
#: base distribution is restricted back to the family's support by the
#: one-sided truncation suffix
#: `half_support_truncation`
#: supplies.
_PREPEND_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

#: JAGS-side argument injection for QVR families that map to JAGS'
#: ``dt(mu, tau, k)`` distribution. JAGS Student-t requires three
#: parameters (location, precision, degrees of freedom); Cauchy is
#: the special case ``k = 1``. The renderer appends ``IRArgNumber(1)``
#: as a trailing ``df`` argument after the alias-renaming pipeline so
#: the emitted call is ``dt(mu, tau, 1)``.
_APPEND_DF_ONE: frozenset[str] = frozenset({"Cauchy", "HalfCauchy"})

#: Sentinel name prefix used by the JAGS renderer to encode a `1:N`
#: range as an IRArgRef. The arg-rendering path inspects the name
#: prefix and emits a `range` vertex instead of an
#: `indexed_variable`.
_RANGE_SENTINEL_PREFIX: str = "__jags_range__:"

#: JAGS zeros-trick constant offset. JAGS has no native target
#: statement; the canonical idiom is to add ``<expr>`` to the joint
#: log-likelihood via ``zero_<name> ~ dpois(C - <expr>)`` with a
#: host-bound ``zero_<name> = 0``. The Poisson PMF satisfies
#: ``log P(X = 0; lambda) = -lambda``, so the stochastic relation
#: contributes ``<expr> - C`` to the log-density; the additive
#: constant absorbs into the normalising constant and does not
#: affect inference. ``C`` must stay strictly larger than ``<expr>``
#: for all parameter values so the Poisson rate remains positive;
#: ``1.0e6`` is the conventional safe default for typical fixtures.
_ZEROS_TRICK_OFFSET: float = 1.0e6

#: The families JAGS has no distribution for and whose density the
#: renderer therefore writes out in closed form, adding it to the
#: joint through the zeros trick. Each needs the ``data { ... }``
#: block that binds the trick's constant-zero carrier.
#:
#: The trick contributes a density term without declaring a node the
#: engine can sample, so at an observed site it is the whole emission,
#: while a *latent* draw needs a node declaration as well; the
#: families that can supply one are the entries of
#: `_ZEROS_TRICK_LATENT_CARRIER`.
_ZEROS_TRICK_FAMILIES: frozenset[str] = frozenset(
    {
        "MixtureNormal",
        "BetaBinomial",
        "ContinuousBernoulli",
        "Kumaraswamy",
    }
)

#: Zeros-trick families a *latent* site can carry, mapped to the QVR
#: family whose JAGS distribution declares the drawn node's support.
#:
#: A latent site has to leave the engine a node it can sample, which
#: the zeros trick alone never does. Pairing the trick with a draw
#: from the *uniform measure on the family's own support* supplies
#: one: `z[n] ~ dunif(0, 1)` declares `z` over `(0, 1)` and adds
#: `-log(1 - 0) = 0` to the joint, so the closed-form term the trick
#: carries is the entire contribution and the emitted program scores
#: the family's own density rather than a tilted version of it. The
#: carrier is exact rather than approximate: the uniform's log density
#: is identically zero over the unit interval, not merely small.
#:
#: Both entries are supported on `(0, 1)`, which is what makes the
#: unit uniform the right carrier. `MixtureNormal` and `BetaBinomial`
#: are deliberately absent. A mixture of normals is supported on all
#: of `R`, whose uniform measure is improper: JAGS spells it
#: `dflat()`, which is not a distribution the sampler can initialise a
#: node from, so a latent mixture draw would emit a model that
#: compiles and then fails to run. A beta-binomial is supported on the
#: integers `0..n`, and its carrier would have to be a categorical
#: over a support whose width is a model quantity rather than a
#: compile-time constant.
_ZEROS_TRICK_LATENT_CARRIER: dict[str, str] = {
    "ContinuousBernoulli": "Uniform",
    "Kumaraswamy": "Uniform",
}

#: The `(low, high)` pair the unit-interval carrier draw supplies,
#: with the argument names
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] gives
#: `Uniform`.
_UNIT_INTERVAL_CARRIER_ARGS: tuple[IRArg, ...] = (
    IRArgNumber(value=0.0),
    IRArgNumber(value=1.0),
)
_UNIT_INTERVAL_CARRIER_ARG_NAMES: tuple[str, ...] = ("low", "high")

#: The zeros-trick families whose closed form is a *density* and so
#: may exceed zero, which is what obliges the emit to lift the Poisson
#: rate by `_ZEROS_TRICK_OFFSET`
#: to keep it in support. ``BetaBinomial`` is deliberately absent: a
#: mass function is at most one, so its negated log form is already
#: non-negative and
#: `_emit_beta_binomial`
#: emits it with no lift, which keeps that family's emission equal to
#: the reference measure on the nose rather than up to a constant.
#: ``ContinuousBernoulli`` is present for the same reason
#: ``Kumaraswamy`` is: it is a density on ``(0, 1)`` whose value
#: exceeds one wherever the tilt concentrates the mass at an endpoint,
#: so its log form is positive there and an unlifted ``-log f(z)``
#: would hand ``dpois`` a negative rate.
_ZEROS_TRICK_LIFTED_FAMILIES: frozenset[str] = frozenset(
    {
        "MixtureNormal",
        "ContinuousBernoulli",
        "Kumaraswamy",
    }
)


def _ir_has_score(body: tuple[IRNode, ...]) -> bool:
    """Whether a plan carries a `score` step, marginalize scopes included.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The plan body.

    Returns
    -------
    bool
        ``True`` when any statement is an `IRScore`.
    """
    for node in body:
        if isinstance(node, IRScore):
            return True
        if isinstance(node, IRMarginalize) and _ir_has_score(node.scope):
            return True
    return False


def _ir_observes_class_code(body: tuple[IRNode, ...]) -> bool:
    """Whether a program body observes a class code outside a
    marginalize block.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The plan body.

    Returns
    -------
    bool
        True when an observation scores a Categorical-like code, whose
        one-based form the `data { ... }` block binds.
    """
    return any(
        isinstance(node, IRObserve) and is_class_code_family(node.family)
        for node in body
    )


def _ir_has_marginalize(body: tuple[IRNode, ...]) -> bool:
    """True iff `body` carries an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize].

    Every marginalize emits its integrated density through the zeros
    trick, so the render pre-pass opens the ``data { ... }`` block that
    binds the trick's carrier whenever one is present. Scanning the
    top level is enough: a nested block sits inside an outer one, and
    the outer one is already the answer.
    """
    for node in body:
        if isinstance(node, IRMarginalize):
            return True
    return False


class JAGSRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as a
    JAGS model source.

    Subclasses [`RendererBase`][quivers.transpile.renderers.RendererBase]:
    overrides `render` to wrap the IR walk in a single top-level
    ``model { ... }`` block. The four dispatch points (`declare`,
    `sample`, `marginalize`, `broadcast`) follow the JAGS surface
    conventions described in section 5.1 of the design spec.
    """

    target: str = _BACKEND

    # ------------------------------------------------------------------
    # `RendererBase` overrides
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Use the auto-derived ``jags`` tree-sitter protocol.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("jags")

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a complete JAGS module.

        Overrides the base IR walk because JAGS programs need a
        ``model { ... }`` wrapper around every body statement; the
        wrapper is a single ``model_block`` vertex whose children are
        the statements emitted by `_dispatch_jags_node`.

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
        # JAGS has no scalar-to-vector broadcast; lift empty-plate
        # IRDeterministic nodes whose expressions reference plate-less
        # free data inputs into the plate of the first downstream
        # consumer, then re-index those references at emit time.
        ir = push_scalar_dets_into_loops(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        jctx = _JAGSCtx(sb=sb, morphisms={}, lets={})
        jctx.scalar_refs, jctx.bound_refs = _classify_bindings(ir)
        self._cards = dict(ir.cards)
        # Cache decl_plates so the deterministic emitter can re-index
        # let-expression refs by their declared batch_dims.
        jctx.decl_plates = build_decl_plates(ir)

        _vertex(jctx, "src", "source_file")
        # A site whose family JAGS cannot name, every `marginalize`
        # block, and every `score` step scores through the zeros trick,
        # whose constant-zero carrier is a node JAGS has to see as
        # data. The language's `data { ... }` transformation block
        # binds exactly such nodes from inside the model source, so the
        # emit declares one when, and only when, a site needs it.
        if (
            _ir_has_marginalize(ir.body)
            or _ir_has_score(ir.body)
            or any(ir_uses_family(ir.body, family) for family in _ZEROS_TRICK_FAMILIES)
            or _ir_observes_class_code(ir.body)
        ):
            jctx.sb.constraint("src", "ptrace-0", "Cdata_block")
            jctx.sb.constraint("src", "ptrace-1", "Cmodel_block")
            jctx.sb.constraint("src", "chose-alt-child-kinds", "data_block model_block")
            db = _fresh(jctx, "db", "data_block")
            jctx.sb.edge("src", db, "child_of")
            jctx.data_block = db
        else:
            jctx.sb.constraint("src", "ptrace-0", "Cmodel_block")
            jctx.sb.constraint("src", "chose-alt-child-kinds", "model_block")

        mb = _fresh(jctx, "mb", "model_block")
        jctx.sb.edge("src", mb, "child_of")
        jctx.current_block = mb
        jctx.model_block = mb

        for node in ir.body:
            self._dispatch_jags_node(jctx, node)

        self._finalise_model_block(jctx)
        self._finalise_data_block(jctx)
        render_computations_static(sb, ir, target=self.target, destination=mb)
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
        """JAGS variables are declared implicitly by their first ``~``
        or ``<-`` binding; data inputs ride on the host's external
        ``.data`` file. This dispatch is a no-op for every block.

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
        """Emit a per-batch-axis ``for (m_<axis> in 1:N_<axis>) { <lhs>
        ~ d<family>(args) }`` form.

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
        jctx = _as_jags_ctx(ctx)
        return self._emit_sample(
            jctx,
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
        """Lower an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to the weighted sum over its latent's atoms.

        QVR's `marginalize` denotes the *integral* over the latent of
        the measure the scope carries, so the emitted program has to
        score that integral and declare no latent site: a program that
        draws the latent instead denotes a measure on the product of
        the latent's support with the scope's, which differs from the
        integral by an amount that moves with the data.

        JAGS has no reduction over a parameterised family, but it has
        every piece the integral needs. The support is finite, so the
        sum

            p(y_n) = sum_a w_a * f_a(y_n)

        is an ordinary arithmetic expression once the atom count is
        known, and the zeros trick adds its logarithm to the joint.
        `_emit_marginal_reduction`
        writes it.

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
        refuse_ungrouped_row_marginalize("qvr-jags", node)
        jctx = _as_jags_ctx(ctx)
        self._emit_marginal_reduction(jctx, node)
        return ""

    def _emit_marginal_reduction(self, ctx: _JAGSCtx, node: IRMarginalize) -> None:
        """Emit one marginalized latent as a rowwise log weighted sum.

        `marginalize_body` separates deterministic bindings from the observed
        site, and `marginal_atoms` pins the latent at each support atom.
        Deterministic bindings are inlined so each atom can use its own copy.
        `marginal_scope_density` supplies the observed density, while
        `marginal_weight_probs` gathers grouped weights through the observation
        fibration.

        Rows follow the observation plate. When all atom densities are mass
        functions, the mixture is at most one and its negative log can be used
        as the zeros-trick Poisson rate without an additive lift.
        """
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        observe = raw.observe
        if observe.plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"marginalize:event-axis:{node.latent}: the "
                    f"scalar closed form carries no event shape, but "
                    f"the scope's observed site declares "
                    f"{[_dim_name(d) for d in observe.plate.event_dims]!r}"
                ],
            )
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        fibration = marginalize_fibration(
            node,
            observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=ctx.decl_plates,
            target=self.target,
        )
        if fibration is not None:
            via, group = fibration
            self._emit_grouped_marginal_reduction(
                ctx, node, atoms, observe, via=via, group=group
            )
            return
        total: LetExprNode | None = None
        lifted = False
        for atom in atoms:
            scored = marginalize_body(
                atom.scope, latent=node.latent, target=self.target
            )
            bindings: dict[str, LetExprNode] = {}
            for det in scored.deterministics:
                bindings[det.name] = inline_letexpr(det.expr, bindings)
            density = marginal_scope_density(
                _BACKEND,
                family=scored.observe.family,
                variate=scored.observe.name,
                args=tuple(
                    irarg_letexpr(_BACKEND, arg, bindings)
                    for arg in scored.observe.args
                ),
                arg_names=scored.observe.arg_names,
            )
            lifted = lifted or not density.mass
            term = LetExprBinOp(
                op="*",
                left=self._marginal_atom_weight(ctx, node, observe, atom),
                right=density.expr,
            )
            total = (
                term if total is None else LetExprBinOp(op="+", left=total, right=term)
            )
        if total is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"marginalize:empty-support:{node.latent}: a "
                    f"latent with no atoms has no integrated density"
                ],
            )
        self._emit_zeros_trick_row(
            ctx,
            name=observe.name,
            family=node.family,
            log_density=LetExprCall(func="log", args=(total,)),
            row_plate=observe.plate,
            lifted=lifted,
        )

    def _emit_grouped_marginal_reduction(
        self,
        ctx: _JAGSCtx,
        node: IRMarginalize,
        atoms: tuple[IRMarginalAtom, ...],
        observe: IRObserve,
        *,
        via: str,
        group: Dim,
    ) -> None:
        """Emit a grouped block whose rows fibre into its groups.

        The block keys its accumulator by group: the rows the
        fibration sends to one group are summed per atom before the
        reduction over the atoms, so the integrated density is one
        weighted sum per group rather than per row:

        ```
        for (m_Item in 1:3) { for (m_Item_of in 1:3) {
          ident_z[m_Item, m_Item_of] <- equals(m_Item, m_Item_of) } }
        for (m_Resp in 1:6) { ld_z_0[m_Resp] <- log(<density of atom 0>) }
        for (m_Item in 1:3) { for (m_Resp in 1:6) {
          sel_z_0[m_Item, m_Resp] <- ident_z[idx[m_Resp], m_Item] * ld_z_0[m_Resp] } }
        for (m_Item in 1:3) { lp_z_0[m_Item] <- sum(sel_z_0[m_Item, 1:6]) }
        ...
        for (m_Item in 1:3) { mx_z[m_Item] <- max(lp_z_0[m_Item], lp_z_1[m_Item]) }
        for (m_Item in 1:3) { phi_z[m_Item] <- 1000000 - (mx_z[m_Item] + log(
          probs[1] * exp(lp_z_0[m_Item] - mx_z[m_Item]) + ...)) }
        for (m_Item in 1:3) { zeros_z[m_Item] ~ dpois(phi_z[m_Item]) }
        ```

        The rows of a group are selected through an identity matrix
        subscripted by the fibration, which is how a JAGS model states
        that an integer input is a one-based index, and the per-group
        sums are shifted by their maximum before the exponentials so
        the weighted sum stays in range.

        Parameters
        ----------
        ctx : _JAGSCtx
            The render context.
        node : IRMarginalize
            The block.
        atoms : tuple[IRMarginalAtom, ...]
            The atoms, with the latent pinned in each scope.
        observe : IRObserve
            The scope's observe, with the latent still free.
        via : str
            The fibration sending rows to groups.
        group : Dim
            The grouping plate's axis.

        Raises
        ------
        UnsupportedConstruct
            If the scope's rows carry more than one batch axis, which
            the per-group sum has no slice for.
        """
        if len(observe.plate.batch_dims) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"marginalize:group-rows-rank:{node.latent}: the "
                    f"per-group sum slices one row axis, but the scope's "
                    f"observed site replicates over "
                    f"{[_dim_name(d) for d in observe.plate.batch_dims]!r}"
                ],
            )
        latent = node.latent
        group_plate = Plate(event_dims=(), batch_dims=(group,))
        group_loop = f"m_{_dim_name(group)}"
        mirror = _mirror_dim(group)
        ident = self._fresh_helper_name(ctx, f"ident_{latent}")
        ident_plate = Plate(event_dims=(), batch_dims=(group, mirror))
        ctx.decl_plates[ident] = ident_plate
        self._emit_deterministic(
            ctx,
            IRDeterministic(
                name=ident,
                expr=LetExprCall(
                    func="equals",
                    args=(
                        LetExprVar(name=group_loop),
                        LetExprVar(name=f"m_{_dim_name(mirror)}"),
                    ),
                ),
                constraint=CSReal(),
                plate=ident_plate,
            ),
        )
        sums: list[str] = []
        weights: list[LetExprNode] = []
        lifted = False
        for position, atom in enumerate(atoms):
            scored = marginalize_body(atom.scope, latent=latent, target=self.target)
            bindings: dict[str, LetExprNode] = {}
            for det in scored.deterministics:
                bindings[det.name] = inline_letexpr(det.expr, bindings)
            density = marginal_scope_density(
                _BACKEND,
                family=scored.observe.family,
                variate=scored.observe.name,
                args=tuple(
                    irarg_letexpr(_BACKEND, arg, bindings)
                    for arg in scored.observe.args
                ),
                arg_names=scored.observe.arg_names,
            )
            lifted = lifted or not density.mass
            rows = self._fresh_helper_name(ctx, f"ld_{latent}_{position}")
            ctx.decl_plates[rows] = observe.plate
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=rows,
                    expr=LetExprCall(func="log", args=(density.expr,)),
                    constraint=CSReal(),
                    plate=observe.plate,
                ),
            )
            selected = self._fresh_helper_name(ctx, f"sel_{latent}_{position}")
            selected_plate = Plate(
                event_dims=(), batch_dims=(group, *observe.plate.batch_dims)
            )
            ctx.decl_plates[selected] = selected_plate
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=selected,
                    expr=LetExprBinOp(
                        op="*",
                        left=LetExprIndex(
                            array=LetExprVar(name=ident),
                            indices=(
                                LetExprVar(name=via),
                                LetExprVar(name=group_loop),
                            ),
                        ),
                        right=LetExprVar(name=rows),
                    ),
                    constraint=CSReal(),
                    plate=selected_plate,
                ),
            )
            # Under the group loop the selection's row axis is the
            # slice the sum contracts.
            ctx.decl_plates[selected] = Plate(
                event_dims=observe.plate.batch_dims, batch_dims=(group,)
            )
            total = self._fresh_helper_name(ctx, f"lp_{latent}_{position}")
            ctx.decl_plates[total] = group_plate
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=total,
                    expr=LetExprCall(func="sum", args=(LetExprVar(name=selected),)),
                    constraint=CSReal(),
                    plate=group_plate,
                ),
            )
            sums.append(total)
            weights.append(self._marginal_atom_weight(ctx, node, observe, atom))
        shift = self._fresh_helper_name(ctx, f"mx_{latent}")
        ctx.decl_plates[shift] = group_plate
        maximum: LetExprNode = LetExprVar(name=sums[0])
        for name in sums[1:]:
            maximum = LetExprCall(func="max", args=(maximum, LetExprVar(name=name)))
        self._emit_deterministic(
            ctx,
            IRDeterministic(
                name=shift, expr=maximum, constraint=CSReal(), plate=group_plate
            ),
        )
        weighted: LetExprNode | None = None
        for weight, name in zip(weights, sums, strict=True):
            term = LetExprBinOp(
                op="*",
                left=weight,
                right=LetExprCall(
                    func="exp",
                    args=(
                        LetExprBinOp(
                            op="-",
                            left=LetExprVar(name=name),
                            right=LetExprVar(name=shift),
                        ),
                    ),
                ),
            )
            weighted = (
                term
                if weighted is None
                else LetExprBinOp(op="+", left=weighted, right=term)
            )
        if weighted is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"marginalize:empty-support:{latent}: a "
                    f"latent with no atoms has no integrated density"
                ],
            )
        self._emit_zeros_trick_row(
            ctx,
            name=latent,
            family=node.family,
            log_density=LetExprBinOp(
                op="+",
                left=LetExprVar(name=shift),
                right=LetExprCall(func="log", args=(weighted,)),
            ),
            row_plate=group_plate,
            lifted=lifted,
        )

    def _marginal_atom_weight(
        self,
        ctx: _JAGSCtx,
        node: IRMarginalize,
        observe: IRObserve,
        atom: IRMarginalAtom,
    ) -> LetExprNode:
        """The mixing weight one atom carries, as a let-expression.

        Two atom sets reach here and they read their weights
        differently. A `class_index` set weights atom `k` by the
        `k`-th entry of the latent's own probability vector, so the
        weight is that vector subscripted by the atom's value. A
        `binary` set weights the two atoms by a *discrete* Bernoulli
        on the same probability, so atom one reads the probability
        itself and atom zero reads its complement; from the logits,
        atom one reads their logistic function and atom zero that of
        their negation.

        The probability tensor arrives from
        `marginal_weight_probs`
        already gathered through the observation's `via` fibration
        where one is needed, and stays a named reference so the
        deterministic path re-indexes it against the row loop.
        """
        probs = irarg_letexpr(
            _BACKEND,
            marginal_weight_probs(
                node,
                observe,
                atom.weight_args,
                atom.weight_arg_names,
                name_plates=ctx.decl_plates,
                target=self.target,
            ),
            {},
        )
        if logit_weighted(atom):
            # From the logits `l` atom one weighs `ilogit(l)` and atom
            # zero `ilogit(-l)`: each is the logistic function of its
            # own argument, with no complement that could cancel.
            if atom.value.value == 1.0:
                return LetExprCall(func="sigmoid", args=(probs,))
            return LetExprCall(
                func="sigmoid", args=(LetExprUnaryOp(op="-", operand=probs),)
            )
        if atom.weight_family == "Bernoulli":
            if atom.value.value == 1.0:
                return probs
            return LetExprBinOp(op="-", left=LetExprLiteral(value=1.0), right=probs)
        return subscript_letexpr(
            _BACKEND, probs, LetExprLiteral(value=atom.value.value)
        )

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Render a scalar-to-vector broadcast as JAGS' ``rep`` builtin.

        JAGS has no scalar-to-vector broadcast operator, but its base
        function library provides ``rep(x, times)``, which returns a
        length-``times`` vector filled with ``x``. A scalar
        concentration over a ``K``-atom Dirichlet event axis thus
        emits ``rep(<scalar>, K)``: a valid vector parent whose repeated
        entries reproduce the symmetric-Dirichlet measure the QVR
        ``[over=K]`` clause denotes.

        Only a scalar reference or numeric literal can be repeated; a
        rank other than one has no ``rep`` form and raises rather than
        emitting an invalid parent.

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
        jctx = _as_jags_ctx(ctx)
        if len(target_shape) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"arg:broadcast:rank-{len(target_shape)}: JAGS `rep` "
                    "builds a 1-D vector only"
                ],
            )
        if not isinstance(value, (IRArgRef, IRArgNumber)):
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"arg:broadcast:value:{type(value).__name__}: only a "
                    "scalar reference or numeric literal can be repeated"
                ],
            )
        return self._rep_call(jctx, value, target_shape[0])

    def _rep_call(self, ctx: _JAGSCtx, value: IRArg, size: int) -> str:
        """Build ``rep(<value>, <size>)`` as a JAGS ``function_call``.

        The value expression is converted to a
        [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] and rendered
        through the shared BUGS / JAGS expression emitter so the
        multi-argument call structure (``name`` identifier plus an
        ``argument_list``) matches what the pretty-printer expects.
        """
        rep_expr = LetExprCall(
            func="rep",
            args=(
                _ir_arg_to_let_expr(value),
                LetExprLiteral(value=float(size)),
            ),
        )
        let_ctx = _jags_let_ctx(ctx, self._cards)
        return render_let_expr_bugs(
            let_ctx,
            rep_expr,
            decl_plates=ctx.decl_plates,
        )

    def _broadcast_scalar_args(
        self,
        ctx: _JAGSCtx,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        meta: FamilyMeta,
        plate: Plate,
    ) -> tuple[IRArg, ...]:
        """Wrap each scalar reference bound to a vector / matrix slot in
        [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast].

        `Lower` already wraps literal scalars whose slot constraint is an
        ``IndependentConstraint(base, n>=1)``, but it leaves an
        [`IRArgRef`][quivers.transpile.ir.IRArgRef] unwrapped on the
        assumption the referenced binding is already vector-shaped. A
        scalar reference (free input or let-bound scalar) at such a slot
        is a rank error in JAGS; wrapping it routes emission through
        [`broadcast`][quivers.transpile.renderers.jags.JAGSRenderer.broadcast]
        and the ``rep(<scalar>, K)`` form. A genuinely vector-shaped
        binding stays untouched.
        """
        cls_constraints = meta.distribution_class.arg_constraints
        if not isinstance(cls_constraints, dict):
            return args
        out: list[IRArg] = []
        for arg_name, arg in zip(arg_names, args, strict=False):
            expected = cls_constraints.get(arg_name)
            if (
                isinstance(arg, IRArgRef)
                and not arg.indices
                and isinstance(
                    expected,
                    _torch_constraints._IndependentConstraint,
                )
                and expected.event_dim >= 1
                and arg.name in ctx.scalar_refs
            ):
                target = self._static_event_shape(plate, expected.event_dim, arg_name)
                out.append(IRArgBroadcast(value=arg, target_shape=target))
            else:
                out.append(arg)
        return tuple(out)

    def _static_event_shape(
        self, plate: Plate, event_dim: int, arg_name: str
    ) -> tuple[int, ...]:
        """The static event shape a scalar arg must be repeated to.

        Takes the first `event_dim` axes off the sample's
        [`Plate.event_dims`][quivers.transpile.ir.Plate]. A plate with
        fewer event axes than the constraint requires, or an axis whose
        length is known only at run time
        ([`DimDynamic`][quivers.transpile.ir.DimDynamic]), raises: JAGS'
        ``rep`` needs a compile-time count, and silently dropping the
        axis would regress a dynamic event axis to a scalar parent.
        """
        dims = plate.event_dims
        if len(dims) < event_dim:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"broadcast:{arg_name}:event-rank:{len(dims)}<"
                    f"{event_dim}: the sample's plate carries fewer event "
                    "axes than the family constraint requires"
                ],
            )
        sizes: list[int] = []
        for dim in dims[:event_dim]:
            if not isinstance(dim, DimStatic):
                raise UnsupportedConstruct(
                    f"qvr-{_BACKEND}",
                    [
                        f"broadcast:{arg_name}:dynamic-event-axis:"
                        f"{_dim_name(dim)}: JAGS `rep` needs a static "
                        "event size"
                    ],
                )
            sizes.append(dim.size)
        return tuple(sizes)

    def _render_list(self, ctx: _JAGSCtx, arg: IRArgList) -> SchemaFragment:
        """JAGS does not parse list literals in argument position;
        callers must pre-bind collections via let-decl."""
        del ctx, arg
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            ["arg:list-literal"],
        )

    def _render_matrix(self, ctx: _JAGSCtx, arg: IRArgMatrix) -> SchemaFragment:
        """JAGS does not parse matrix literals in argument position;
        callers must pre-bind matrices via let-decl."""
        del ctx, arg
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            ["arg:matrix-literal"],
        )

    # ------------------------------------------------------------------
    # IR node dispatch
    # ------------------------------------------------------------------

    def _dispatch_jags_node(self, ctx: _JAGSCtx, node: IRNode) -> None:
        """Walk one IR node and emit its JAGS form into the active
        block."""
        if isinstance(node, IRDataInput):
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            if node.family in _ZEROS_TRICK_FAMILIES:
                self._emit_zeros_trick_latent(ctx, node)
                return
            if is_class_code_family(node.family):
                self._emit_class_code_latent(ctx, node)
                return
            self._emit_sample(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
            )
            return
        if isinstance(node, IRObserve):
            if node.family in _ZEROS_TRICK_FAMILIES:
                self._emit_closed_form_density(
                    ctx,
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    plate=node.plate,
                )
                return
            if is_class_code_family(node.family):
                self._emit_class_code_observation(ctx, node)
                return
            self._emit_sample(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                via=node.via,
                observed=True,
            )
            return
        if isinstance(node, IRDeterministic):
            if isinstance(node.expr, LetExprFactor):
                self._emit_factor_deterministic(ctx, node)
                return
            if isinstance(node.expr, LetExprList) and node.plate.batch_dims:
                self._check_factor_plate(node, list_sizes(node.expr))
                self._emit_cells(ctx, node, list_cells(node.expr, ()))
                return
            literals, node = hoist_list_operands(node, frozenset(ctx.decl_plates))
            for literal in literals:
                cells = literal.expr
                assert isinstance(cells, LetExprList)
                ctx.decl_plates[literal.name] = literal.plate
                self._check_factor_plate(literal, list_sizes(cells))
                self._emit_cells(ctx, literal, list_cells(cells, ()))
            self._emit_deterministic(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_export(ctx, node.names)
            return
        if isinstance(node, IRCall):
            # A graph language relates variables; it has no statement
            # that runs a computation, and the plan has already inlined
            # every pure call it could.
            raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"call:graph:{node.callee}"])
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            [f"node:{type(node).__name__}"],
        )

    def _emit_export(self, ctx: _JAGSCtx, names: tuple[str, ...]) -> None:
        """Expose each returned name as a deterministic relation.

        The BUGS language has no `return`: a model block declares
        relations and the inference engine reports whatever the caller
        monitors. The construct that carries "this quantity is part of
        what the model reports" is thus a deterministic relation
        under a name of its own, `<name>_value <- <name>`, which is
        the same idiom the Stan renderer uses for its
        `generated quantities` alias and the PyMC renderer for its
        `pymc.Deterministic`. A relation adds no term to the joint
        (a deterministic node contributes nothing to the deviance),
        so the export rides alongside the density rather than into it.

        The alias reuses the deterministic emitter, which supplies the
        `for (m_<axis> in 1:N)` nest and the per-iteration indexing on
        both sides when the exported name is plated.
        """
        for name in names:
            plate = ctx.decl_plates.get(name)
            if plate is None:
                raise UnsupportedConstruct(
                    f"qvr-{_BACKEND}",
                    [
                        f"return:unbound:{name}: the program returns a "
                        f"name no sample, observe, let, or data input "
                        f"binds, so the emit has no relation to alias"
                    ],
                )
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=f"{name}_value",
                    expr=LetExprVar(name=name),
                    constraint=CSReal(),
                    plate=plate,
                ),
            )

    # ------------------------------------------------------------------
    # Sample / observe emission
    # ------------------------------------------------------------------

    def _emit_gp_block(
        self,
        ctx: _JAGSCtx,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as a sequence of JAGS
        deterministic relations plus a multivariate-normal stochastic
        relation:

            for (i in 1:N) {
              __gp_zeros_<name>[i] <- 0
              for (j in 1:N) {
                __gp_K_<name>[i, j] <- exp(-0.5 *
                    pow(x[i] - x[j], 2) / pow(length_scale, 2)) +
                    ifelse(equals(i, j), jitter, 0)
              }
            }
            __gp_tau_<name> <- inverse(__gp_K_<name>)
            <name> ~ dmnorm(__gp_zeros_<name>, __gp_tau_<name>)

        JAGS's ``dmnorm`` parameterises by precision (the inverse of
        the covariance), so the RBF kernel matrix is constructed in
        a double loop and then inverted before being passed as the
        precision argument.
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                ["family:GP: expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        # JAGS rejects identifiers starting with underscore; use a
        # `gp_` prefix instead of `__gp_` for the synthesized names.
        kmat_name = f"gp_K_{node.name}"
        zeros_name = f"gp_zeros_{node.name}"
        tau_name = f"gp_tau_{node.name}"
        # Build the RBF entry expression as a LetExprNode using
        # loop-var references "i" / "j" (JAGS loop variables) plus
        # `x[i]`, `x[j]` lookups.
        i_var = LetExprVar(name="i")
        j_var = LetExprVar(name="j")
        x_i = LetExprIndex(
            array=LetExprVar(name=x),
            indices=(i_var,),
        )
        x_j = LetExprIndex(
            array=LetExprVar(name=x),
            indices=(j_var,),
        )
        diff = LetExprBinOp(op="-", left=x_i, right=x_j)
        diff_sq = LetExprCall(
            func="pow",
            args=(diff, LetExprLiteral(value=2.0)),
        )
        ls_sq = LetExprCall(
            func="pow",
            args=(
                LetExprLiteral(value=ls),
                LetExprLiteral(value=2.0),
            ),
        )
        half = LetExprLiteral(value=0.5)
        neg_half = LetExprUnaryOp(operand=half)
        scaled = LetExprBinOp(
            op="/",
            left=LetExprBinOp(op="*", left=neg_half, right=diff_sq),
            right=ls_sq,
        )
        exp_call = LetExprCall(func="exp", args=(scaled,))
        eq_call = LetExprCall(func="equals", args=(i_var, j_var))
        ifelse_call = LetExprCall(
            func="ifelse",
            args=(
                eq_call,
                LetExprLiteral(value=jitter),
                LetExprLiteral(value=0.0),
            ),
        )
        rbf_entry_expr = LetExprBinOp(
            op="+",
            left=exp_call,
            right=ifelse_call,
        )
        # Build the K-matrix relation: K[i, j] <- <entry>, wrapped in
        # for (j) inside for (i).
        kij = _fresh(ctx, "kij", "deterministic_relation")
        ctx.sb.constraint(kij, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(kij, "ptrace-0", "Cindexed_variable")
        ctx.sb.constraint(kij, "ptrace-1", "T<-")
        lhs_kij = self._indexed_variable(
            ctx,
            kmat_name,
            ("i", "j"),
            (),
        )
        ctx.sb.edge(kij, lhs_kij, "variable")
        let_ctx = _jags_let_ctx(ctx, self._cards)
        kij_rhs = render_let_expr_bugs(
            let_ctx,
            rbf_entry_expr,
            decl_plates=ctx.decl_plates,
        )
        ctx.sb.edge(kij, kij_rhs, "value")
        # for (j in 1:N) { K[i, j] <- ... }
        inner_dim = DimStatic(size=n, name="j")
        inner_loop = self._wrap_in_for_loops(
            ctx,
            kij,
            (inner_dim,),
            override_var="j",
        )
        # zeros[i] <- 0 (one deterministic per i)
        zi = _fresh(ctx, "zi", "deterministic_relation")
        ctx.sb.constraint(zi, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(zi, "ptrace-0", "Cindexed_variable")
        ctx.sb.constraint(zi, "ptrace-1", "T<-")
        lhs_zi = self._indexed_variable(ctx, zeros_name, ("i",), ())
        ctx.sb.edge(zi, lhs_zi, "variable")
        zi_rhs = render_let_expr_bugs(
            let_ctx,
            LetExprLiteral(value=0.0),
            decl_plates=ctx.decl_plates,
        )
        ctx.sb.edge(zi, zi_rhs, "value")
        # Build the outer i-block: { zeros[i] <- 0; <inner_loop> }
        outer_block = _fresh(ctx, "blk", "block")
        ctx.sb.constraint(outer_block, "chose-alt-fingerprint", "{ }")
        ctx.sb.constraint(
            outer_block,
            "chose-alt-child-kinds",
            "deterministic_relation for_loop",
        )
        ctx.sb.constraint(outer_block, "ptrace-0", "T{")
        ctx.sb.constraint(
            outer_block,
            "ptrace-1",
            "Cdeterministic_relation",
        )
        ctx.sb.constraint(outer_block, "ptrace-2", "Cfor_loop")
        ctx.sb.constraint(outer_block, "ptrace-3", "T}")
        ctx.sb.edge(outer_block, zi, "child_of")
        ctx.sb.edge(outer_block, inner_loop, "child_of")
        # for (i in 1:N) { zeros[i] <- 0 ; for (j in 1:N) {...} }
        outer_loop = self._build_for_loop(
            ctx,
            "i",
            n,
            outer_block,
        )
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, outer_loop, "child_of")
            ctx.block_children.setdefault(
                ctx.current_block,
                [],
            ).append(_block_child_kind(ctx, outer_loop))
        # tau <- inverse(K)
        tau_dr = _fresh(ctx, "tdr", "deterministic_relation")
        ctx.sb.constraint(tau_dr, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(tau_dr, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(tau_dr, "ptrace-1", "T<-")
        ctx.sb.edge(tau_dr, _identifier(ctx, tau_name), "variable")
        tau_rhs = render_let_expr_bugs(
            let_ctx,
            LetExprCall(
                func="inverse",
                args=(LetExprVar(name=kmat_name),),
            ),
            decl_plates=ctx.decl_plates,
        )
        ctx.sb.edge(tau_dr, tau_rhs, "value")
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, tau_dr, "child_of")
            ctx.block_children.setdefault(
                ctx.current_block,
                [],
            ).append(_block_child_kind(ctx, tau_dr))
        # f ~ dmnorm(zeros, tau)
        sr = _fresh(ctx, "sr", "stochastic_relation")
        ctx.sb.constraint(sr, "chose-alt-fingerprint", "~")
        ctx.sb.constraint(sr, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(sr, "ptrace-1", "T~")
        ctx.sb.constraint(sr, "ptrace-2", "Cdistribution_call")
        ctx.sb.edge(sr, _identifier(ctx, node.name), "variable")
        dist_vid = self._build_dmnorm_dist(
            ctx,
            zeros_name,
            tau_name,
        )
        ctx.sb.edge(sr, dist_vid, "distribution")
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, sr, "child_of")
            ctx.block_children.setdefault(
                ctx.current_block,
                [],
            ).append(_block_child_kind(ctx, sr))

    def _build_for_loop(
        self,
        ctx: _JAGSCtx,
        loop_var: str,
        n: int,
        body_block: str,
    ) -> str:
        """Build a single ``for (<loop_var> in 1:<n>) <body_block>``
        for-loop wrapping ``body_block``."""
        fl = _fresh(ctx, "fl", "for_loop")
        ctx.sb.constraint(fl, "chose-alt-fingerprint", "for ( in )")
        ctx.sb.constraint(
            fl,
            "chose-alt-child-kinds",
            "identifier range block",
        )
        ctx.sb.constraint(fl, "ptrace-0", "Tfor")
        ctx.sb.constraint(fl, "ptrace-1", "T(")
        ctx.sb.constraint(fl, "ptrace-2", "Cidentifier")
        ctx.sb.constraint(fl, "ptrace-3", "Tin")
        ctx.sb.constraint(fl, "ptrace-4", "Crange")
        ctx.sb.constraint(fl, "ptrace-5", "T)")
        ctx.sb.constraint(fl, "ptrace-6", "Cblock")
        ctx.sb.edge(fl, _identifier(ctx, loop_var), "variable")
        rng, _ = self._range_static(ctx, n), "range"
        ctx.sb.edge(fl, rng, "range")
        ctx.sb.edge(fl, body_block, "body")
        return fl

    def _build_dmnorm_dist(
        self,
        ctx: _JAGSCtx,
        zeros_name: str,
        tau_name: str,
    ) -> str:
        """Build ``dmnorm(zeros, tau)`` as a ``distribution_call``
        vertex (the panproto schema kind JAGS uses for stochastic
        relations' RHS)."""
        dc = _fresh(ctx, "dc", "distribution_call")
        ctx.sb.constraint(dc, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(
            dc,
            "chose-alt-child-kinds",
            "identifier argument_list",
        )
        ctx.sb.constraint(dc, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(dc, "ptrace-1", "T(")
        ctx.sb.constraint(dc, "ptrace-2", "Cargument_list")
        ctx.sb.constraint(dc, "ptrace-3", "T)")
        ctx.sb.edge(dc, _identifier(ctx, "dmnorm"), "name")
        al = _fresh(ctx, "al", "argument_list")
        ctx.sb.constraint(al, "chose-alt-fingerprint", ", ")
        ctx.sb.constraint(
            al,
            "chose-alt-child-kinds",
            "identifier identifier",
        )
        ctx.sb.constraint(al, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(al, "ptrace-1", "T,")
        ctx.sb.constraint(al, "ptrace-2", "Cidentifier")
        ctx.sb.edge(al, _identifier(ctx, zeros_name), "child_of")
        ctx.sb.edge(al, _identifier(ctx, tau_name), "child_of")
        ctx.sb.edge(dc, al, "arguments")
        return dc

    def _dmnorm_precision_args(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """Rewrite a ``MultivariateNormal`` site into ``dmnorm``'s
        precision parameterisation.

        JAGS' ``dmnorm(mu, Omega)`` reads its second argument as the
        precision, the inverse of the covariance: with
        ``Omega = [[2, .5], [.5, 1.5]]``, ``mu = (.1, .3)`` and
        ``x = (.4, -.2)`` the engine scores -1.534577, which is
        ``torch.distributions.MultivariateNormal(mu,
        precision_matrix=Omega).log_prob(x)`` and not the
        ``covariance_matrix=Omega`` reading (-2.486405). Emitting the
        QVR ``covariance_matrix`` slot straight into that position
        thus scores a different Gaussian at every point.

        A site that already names ``precision_matrix`` passes through
        untouched. A site naming ``covariance_matrix`` gains a
        ``prec_<name> <- inverse(<covariance>)`` relation ahead of the
        draw, which is the same idiom the GP block uses to invert its
        kernel matrix. A ``scale_tril`` site raises: ``dmnorm`` has no
        Cholesky slot, and reconstructing the precision from the
        factor would need a matrix product this renderer has no
        emission for.
        """
        if family != "MultivariateNormal":
            return args, arg_names
        by_name = dict(zip(arg_names, args, strict=True))
        if "loc" not in by_name:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:MultivariateNormal:missing-arg:loc: "
                    f"`dmnorm(mu, Omega)` needs a mean vector; the "
                    f"site supplies {list(arg_names)}"
                ],
            )
        if "precision_matrix" in by_name:
            return (
                (by_name["loc"], by_name["precision_matrix"]),
                ("loc", "precision_matrix"),
            )
        covariance = by_name.get("covariance_matrix")
        if covariance is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:MultivariateNormal:no-scale-arg: "
                    f"`dmnorm(mu, Omega)` takes a precision matrix, "
                    f"which this renderer builds from a covariance or "
                    f"reads directly from a precision; the site "
                    f"supplies {list(arg_names)}"
                ],
            )
        if not isinstance(covariance, IRArgRef) or covariance.indices:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:MultivariateNormal:non-ref-covariance: "
                    f"the `inverse(...)` relation that turns the "
                    f"covariance into `dmnorm`'s precision needs a "
                    f"whole named matrix, and this slot carries a "
                    f"{type(covariance).__name__}"
                ],
            )
        precision_name = f"prec_{name}"
        self._emit_deterministic(
            ctx,
            IRDeterministic(
                name=precision_name,
                expr=LetExprCall(
                    func="inverse",
                    args=(LetExprVar(name=covariance.name),),
                ),
                constraint=CSPositiveDefinite(),
                plate=Plate(event_dims=(), batch_dims=()),
            ),
        )
        return (
            (by_name["loc"], IRArgRef(name=precision_name)),
            ("loc", "precision_matrix"),
        )

    def _emit_sample(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        via: str | None = None,
        observed: bool = False,
    ) -> SchemaFragment:
        """Build ``for (...) { name[m] ~ d<family>(args) }`` per batch
        dim and attach it to the surrounding block."""
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"family:unknown:{family}"])
        target_name = meta.target_names.get(_BACKEND)
        if target_name is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}", [f"family:no-target-name:{family}"]
            )

        # A scalar reference in a vector / matrix distribution slot
        # (e.g. a symmetric-Dirichlet concentration) has no valid JAGS
        # form as a bare scalar; wrap it so emission repeats it into a
        # `rep(<scalar>, K)` vector parent.
        args = self._broadcast_scalar_args(ctx, args, arg_names, meta, plate)

        if family in _PREPEND_ZERO:
            args = (IRArgNumber(value=0.0), *args)
            arg_names = ("loc", *arg_names)

        args, arg_names = self._dmnorm_precision_args(
            ctx, name=name, family=family, args=args, arg_names=arg_names
        )

        # TruncatedNormal lowers to `dnorm(loc, tau) T(low, high)`.
        # Peel off the last two args (low, high) before the
        # alias-renaming pipeline so they don't try to map through
        # the family's arg_aliases (which only renames the
        # distribution-call args). The peeled bounds are reattached
        # below as a `truncation` child of the stochastic_relation.
        truncation_bounds: tuple[IRArg, ...] | None = half_support_truncation(
            family, observed=observed
        )
        if family == "TruncatedNormal":
            if len(args) != 4:
                raise UnsupportedConstruct(
                    f"qvr-{_BACKEND}",
                    [
                        f"family:TruncatedNormal: expected 4 args "
                        f"(loc, scale, low, high), got {len(args)}"
                    ],
                )
            truncation_bounds = args[2:]
            args = args[:2]
            arg_names = arg_names[:2]

        # StudentT: torch (df, loc, scale) -> JAGS dt(mu, tau, k) =
        # (loc, 1/scale^2, df). Reorder and precision-transform before
        # the alias pipeline; the pre-wrapped scale carries its own
        # transform so the loop below leaves it untouched.
        if family == "StudentT":
            args, arg_names = _reorder_studentt_dt(args, arg_names)

        # HalfStudentT: QVR (df, scale) -> JAGS dt(mu, tau, k) =
        # (0, 1/scale^2, df), paired with the `T (0 ,)` suffix
        # `half_support_truncation` supplies for the family. `dt` has
        # no half-support variant and takes three arguments, so the
        # site's two go through the same pre-wrapped reshape StudentT
        # uses rather than through `_PREPEND_ZERO` / the alias
        # pipeline, which would leave `(0, df, scale)`.
        if family == "HalfStudentT":
            args, arg_names = reorder_half_studentt_dt(args, arg_names)

        # NegativeBinomial / Weibull carry a target-specific argument
        # order and reparameterisation that JAGS' `dnegbin(prob, size)`
        # / `dweib(v, lambda)` calls require; reorder before the alias
        # pipeline so the reshaped args (which carry their own
        # transforms) pass through the loop below untouched.
        if family == "NegativeBinomial":
            args, arg_names = reorder_negbin_args(args, arg_names)
        elif family == "Weibull":
            args, arg_names = reorder_weibull_args(args, arg_names)
        elif family == "Binomial":
            args, arg_names = reorder_binomial_dbin(args, arg_names)
        elif family == "Pareto":
            args, arg_names = reorder_pareto_dpar(args, arg_names)

        aliases = {
            **meta.arg_aliases.get(_BACKEND, {}),
            **_FAMILY_ALIAS_OVERRIDE.get(family, {}),
        }
        renamed_pairs: list[tuple[str, IRArg]] = []
        scalar_slots: list[bool] = []
        for arg_name, arg in zip(arg_names, args, strict=False):
            emitted_name = aliases.get(arg_name, arg_name)
            transform = _alias_transform_for(family, emitted_name)
            if transform is not None and emitted_name != arg_name:
                arg = IRArgTransform(inner=arg, transform=transform)
            renamed_pairs.append((emitted_name, arg))
            scalar_slots.append(scalar_parameter(family, arg_name))

        # Append the trailing `df` argument for families whose JAGS
        # target is `dt(mu, tau, k)`. Cauchy / HalfCauchy fix k = 1.
        if family in _APPEND_DF_ONE:
            renamed_pairs.append(("df", IRArgNumber(value=1.0)))

        # Split the site's event dims into the family's own event
        # shape and the residual axes that merely replicate it. A
        # scalar family carrying an `over=` axis is all residual, and
        # JAGS has no vector form for a scalar family, so each
        # residual axis becomes an extra innermost loop rather than a
        # slice on the left-hand side.
        native_event, residual_event = split_event_dims(
            plate.event_dims, meta.event_rank
        )

        # Compute loop var names. The observation-plate convention is to
        # use the canonical `n` when `via` is set and the plate has a
        # single dynamic dim.
        if via is not None and len(plate.batch_dims) == 1 and not residual_event:
            batch_loop_names: tuple[str, ...] = ("n",)
        else:
            batch_loop_names = tuple(f"m_{_dim_name(dim)}" for dim in plate.batch_dims)
        residual_loop_names = _fresh_loop_names(batch_loop_names, residual_event)
        loop_var_names: tuple[str, ...] = (*batch_loop_names, *residual_loop_names)
        loop_dims: tuple[Dim, ...] = (*plate.batch_dims, *residual_event)
        lhs_plate = Plate(event_dims=native_event, batch_dims=loop_dims)

        # Build the axis-to-loop-var map for this sample's surrounding
        # plate. `_rewrite_arg` uses this to choose the right loop var
        # when a latent's recorded axis appears in the current plate.
        axis_to_lv: dict[str, str] = {
            _dim_name(dim): lv
            for dim, lv in zip(loop_dims, loop_var_names, strict=True)
        }

        # Pre-rewrite every arg ONCE: thread latent loop vars + event
        # ranges + via fibration through any ref to a previously-bound
        # latent. The rewrite is idempotent on its output by construction
        # (the appended event-range sentinels are not themselves latents).
        rewritten_args = tuple(
            (
                n,
                _rewrite_arg(
                    ctx,
                    a,
                    batch_loop_names,
                    axis_to_lv,
                    via,
                    scalar_slot=position < len(scalar_slots) and scalar_slots[position],
                ),
            )
            for position, (n, a) in enumerate(renamed_pairs)
        )

        sr = self._build_stochastic_relation(
            ctx,
            lhs_name=name,
            target_dist=target_name,
            renamed_pairs=rewritten_args,
            plate=lhs_plate,
            loop_vars=loop_var_names,
            truncation_bounds=truncation_bounds,
            axis_to_lv=axis_to_lv,
            via=via,
            loop_var_names=batch_loop_names,
        )

        for dim in plate.batch_dims:
            ctx.emitted_plate_names.add(_dim_name(dim))

        # Record this sample's plate info so subsequent refs to `name`
        # thread the loop var + event ranges through. We record both the
        # axis name (so a reference can use the *current* surrounding
        # plate's loop var when the axes match) and a fallback loop var
        # for when no surrounding plate covers the axis.
        if plate.batch_dims:
            fallback_lv = (
                loop_var_names[-1]
                if loop_var_names
                else f"m_{_dim_name(plate.batch_dims[-1])}"
            )
            axes = tuple(_dim_name(d) for d in plate.batch_dims)
            ctx.latent_plate_info[name] = (
                fallback_lv,
                plate.event_dims,
                axes,
            )

        override_var = (
            "n"
            if via is not None and len(plate.batch_dims) == 1 and not residual_event
            else None
        )
        wrapped = self._wrap_in_for_loops(ctx, sr, loop_dims, override_var=override_var)
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, wrapped, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                _block_child_kind(ctx, wrapped)
            )
        return wrapped

    def _build_stochastic_relation(
        self,
        ctx: _JAGSCtx,
        *,
        lhs_name: str,
        target_dist: str,
        renamed_pairs: tuple[tuple[str, IRArg], ...],
        plate: Plate,
        loop_vars: tuple[str, ...],
        truncation_bounds: tuple[IRArg, ...] | None = None,
        axis_to_lv: dict[str, str] | None = None,
        via: str | None = None,
        loop_var_names: tuple[str, ...] = (),
    ) -> str:
        """Build ``<lhs> ~ <dist>(...) [T(low, high)]`` as a
        ``stochastic_relation``.

        When `truncation_bounds` is supplied (used for the
        ``TruncatedNormal`` family, lowered to
        ``dnorm(loc, tau) T(low, high)``), a `truncation` vertex
        carrying the two rendered bound expressions is attached as a
        `child_of` of the stochastic_relation. The bounds go through
        the same `_rewrite_arg` pipeline as the family-call args so
        loop-var / latent-ref rewriting threads through them too.
        """
        sr = _fresh(ctx, "sr", "stochastic_relation")
        has_trunc = truncation_bounds is not None
        if has_trunc:
            ctx.sb.constraint(sr, "chose-alt-fingerprint", "~")
            ctx.sb.constraint(
                sr,
                "chose-alt-child-kinds",
                f"{_lhs_kind(plate, loop_vars)} distribution_call truncation",
            )
            ctx.sb.constraint(sr, "ptrace-0", f"C{_lhs_kind(plate, loop_vars)}")
            ctx.sb.constraint(sr, "ptrace-1", "T~")
            ctx.sb.constraint(sr, "ptrace-2", "Cdistribution_call")
            ctx.sb.constraint(sr, "ptrace-3", "Ctruncation")
        else:
            ctx.sb.constraint(sr, "chose-alt-fingerprint", "~")
            ctx.sb.constraint(
                sr,
                "chose-alt-child-kinds",
                f"{_lhs_kind(plate, loop_vars)} distribution_call",
            )
            ctx.sb.constraint(sr, "ptrace-0", f"C{_lhs_kind(plate, loop_vars)}")
            ctx.sb.constraint(sr, "ptrace-1", "T~")
            ctx.sb.constraint(sr, "ptrace-2", "Cdistribution_call")

        lhs = self._build_lhs(ctx, lhs_name, plate, loop_vars)
        ctx.sb.edge(sr, lhs, "variable")

        dc = self._build_distribution_call(ctx, target_dist, renamed_pairs)
        ctx.sb.edge(sr, dc, "distribution")

        if has_trunc:
            assert truncation_bounds is not None
            trunc = self._build_truncation(
                ctx,
                bounds=truncation_bounds,
                axis_to_lv=axis_to_lv or {},
                via=via,
                loop_var_names=loop_var_names,
            )
            ctx.sb.edge(sr, trunc, "child_of")
        return sr

    def _build_truncation(
        self,
        ctx: _JAGSCtx,
        *,
        bounds: tuple[IRArg, ...],
        axis_to_lv: dict[str, str],
        via: str | None,
        loop_var_names: tuple[str, ...],
    ) -> str:
        """Build a JAGS `truncation` vertex carrying the lo/hi bound
        expressions for a ``T(low, high)`` suffix on a stochastic
        relation.

        Each bound goes through the standard `_rewrite_arg` pipeline
        so loop-var / latent-ref substitutions apply uniformly with
        the surrounding distribution-call args.
        """
        trunc = _fresh(ctx, "tnc", "truncation")
        kinds: list[str] = []
        for bound in bounds:
            rewritten = _rewrite_arg(
                ctx,
                bound,
                loop_var_names,
                axis_to_lv,
                via,
                scalar_slot=True,
            )
            vid, kind = self._render_arg_with_kind(ctx, rewritten)
            ctx.sb.edge(trunc, vid, "child_of")
            kinds.append(kind)
        ctx.sb.constraint(
            trunc, "chose-alt-fingerprint", TRUNCATION_FINGERPRINT[_BACKEND]
        )
        ctx.sb.constraint(
            trunc,
            "chose-alt-child-kinds",
            " ".join(kinds),
        )
        return trunc

    def _build_lhs(
        self,
        ctx: _JAGSCtx,
        name: str,
        plate: Plate,
        loop_vars: tuple[str, ...],
    ) -> str:
        """Build the LHS of a stochastic relation.

        With zero batch dims and no event dim, the LHS is a bare
        identifier. With batch dims, it is an
        ``indexed_variable``: ``name[m_0, m_1, ...]`` when the event
        is scalar, or ``name[m_0, ..., 1:E]`` when the event has a
        vector shape (e.g. Dirichlet draws on a Topic axis)."""
        if not plate.batch_dims and not plate.event_dims:
            return _identifier(ctx, name)
        return self._indexed_variable(ctx, name, loop_vars, plate.event_dims)

    def _indexed_variable(
        self,
        ctx: _JAGSCtx,
        name: str,
        loop_vars: tuple[str, ...],
        event_dims: tuple[object, ...],
    ) -> str:
        """``name[m_0, m_1, ..., 1:E0, 1:E1, ...]``."""
        iv = _fresh(ctx, "iv", "indexed_variable")
        ctx.sb.constraint(iv, "chose-alt-fingerprint", "[ ]")
        ctx.sb.constraint(iv, "chose-alt-child-kinds", "identifier index_list")
        ctx.sb.constraint(iv, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(iv, "ptrace-1", "T[")
        ctx.sb.constraint(iv, "ptrace-2", "Cindex_list")
        ctx.sb.constraint(iv, "ptrace-3", "T]")

        ident = _identifier(ctx, name)
        ctx.sb.edge(iv, ident, "name")

        idx_list = _fresh(ctx, "il", "index_list")
        index_kinds: list[str] = []
        children: list[tuple[str, str]] = []
        for lv in loop_vars:
            child_id = _identifier(ctx, lv)
            children.append((child_id, "identifier"))
            index_kinds.append("identifier")
        for ed in event_dims:
            size_form, kind = self._event_range_form(ctx, ed)
            children.append((size_form, kind))
            index_kinds.append(kind)

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        for i, (_, kind) in enumerate(children):
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            if i < len(children) - 1:
                ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            idx_list,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(idx_list, "chose-alt-child-kinds", " ".join(index_kinds))

        for child_id, _kind in children:
            ctx.sb.edge(idx_list, child_id, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv

    def _event_range_form(self, ctx: _JAGSCtx, dim: object) -> tuple[str, str]:
        """`1:E` range form for one event dim. Returns (vid, kind)."""
        if isinstance(dim, DimStatic):
            return self._range_static(ctx, dim.size), "range"
        if isinstance(dim, DimDynamic):
            return self._range_dynamic(ctx, dim.size_name), "range"
        raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"])

    def _range_static(self, ctx: _JAGSCtx, upper: int) -> str:
        rng = _fresh(ctx, "rng", "range")
        ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
        ctx.sb.constraint(rng, "chose-alt-child-kinds", "number number")
        ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
        ctx.sb.constraint(rng, "ptrace-1", "T:")
        ctx.sb.constraint(rng, "ptrace-2", "Cnumber")
        lo = _number(ctx, 1)
        hi = _number(ctx, upper)
        ctx.sb.edge(rng, lo, "lower")
        ctx.sb.edge(rng, hi, "upper")
        return rng

    def _range_dynamic(self, ctx: _JAGSCtx, upper_name: str) -> str:
        rng = _fresh(ctx, "rng", "range")
        ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
        ctx.sb.constraint(rng, "chose-alt-child-kinds", "number identifier")
        ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
        ctx.sb.constraint(rng, "ptrace-1", "T:")
        ctx.sb.constraint(rng, "ptrace-2", "Cidentifier")
        lo = _number(ctx, 1)
        hi = _identifier(ctx, upper_name)
        ctx.sb.edge(rng, lo, "lower")
        ctx.sb.edge(rng, hi, "upper")
        return rng

    def _build_distribution_call(
        self,
        ctx: _JAGSCtx,
        target_dist: str,
        renamed_pairs: tuple[tuple[str, IRArg], ...],
    ) -> str:
        """Build ``<dist>(<arg0>, <arg1>, ...)``.

        JAGS positional-only calling convention: emit args in the
        order received.
        """
        dc = _fresh(ctx, "dc", "distribution_call")
        ctx.sb.constraint(dc, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(dc, "chose-alt-child-kinds", "identifier argument_list")
        ctx.sb.constraint(dc, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(dc, "ptrace-1", "T(")
        ctx.sb.constraint(dc, "ptrace-2", "Cargument_list")
        ctx.sb.constraint(dc, "ptrace-3", "T)")

        name = _identifier(ctx, target_dist)
        ctx.sb.edge(dc, name, "name")

        al = self._build_argument_list(
            ctx,
            tuple(arg for _name, arg in renamed_pairs),
        )
        ctx.sb.edge(dc, al, "arguments")
        return dc

    def _build_argument_list(
        self,
        ctx: _JAGSCtx,
        args: tuple[IRArg, ...],
    ) -> str:
        """``arg0, arg1, ...`` as an ``argument_list``."""
        al = _fresh(ctx, "al", "argument_list")
        child_pairs: list[tuple[str, str]] = []
        for arg in args:
            vid, kind = self._render_arg_with_kind(ctx, arg)
            child_pairs.append((vid, kind))

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, (_, kind) in enumerate(child_pairs):
            ctx.sb.constraint(al, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            kinds.append(kind)
            if i < len(child_pairs) - 1:
                ctx.sb.constraint(al, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            al,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(al, "chose-alt-child-kinds", " ".join(kinds))

        for vid, _kind in child_pairs:
            ctx.sb.edge(al, vid, "child_of")
        return al

    def _render_arg_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArg,
    ) -> tuple[str, str]:
        """Render one arg; return (vid, kind) so the parent can build
        ptrace constraints."""
        if isinstance(arg, IRArgNumber):
            return _number(ctx, arg.value), "number"
        if isinstance(arg, IRArgRef):
            return self._render_ref_with_kind(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return (
                self.broadcast(ctx, arg.value, arg.target_shape),
                "function_call",
            )
        if isinstance(arg, IRArgList):
            return self._render_list(ctx, arg), "indexed_variable"
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(ctx, arg), "indexed_variable"
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref_with_kind(ctx, arg)
        if isinstance(arg, IRArgTransform):
            return self._render_transform_with_kind(ctx, arg)
        raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"arg:{type(arg).__name__}"])

    def _render_ref_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArgRef,
    ) -> tuple[str, str]:
        """Render a bound-name reference. Bare names emit as
        ``identifier``; indexed references emit as ``indexed_variable``
        with the index expressions as children.

        The arg has already been rewritten by `_rewrite_arg` at sample
        time; no further latent-plate threading happens here.
        """
        # Sentinel: encode a JAGS `1:N` range.
        if arg.name.startswith(_RANGE_SENTINEL_PREFIX):
            upper = arg.name[len(_RANGE_SENTINEL_PREFIX) :]
            try:
                return self._range_static(ctx, int(upper)), "range"
            except ValueError:
                return self._range_dynamic(ctx, upper), "range"
        if not arg.indices:
            return _identifier(ctx, arg.name), "identifier"
        iv = _fresh(ctx, "iv", "indexed_variable")
        ctx.sb.constraint(iv, "chose-alt-fingerprint", "[ ]")
        ctx.sb.constraint(iv, "chose-alt-child-kinds", "identifier index_list")
        ctx.sb.constraint(iv, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(iv, "ptrace-1", "T[")
        ctx.sb.constraint(iv, "ptrace-2", "Cindex_list")
        ctx.sb.constraint(iv, "ptrace-3", "T]")

        name = _identifier(ctx, arg.name)
        ctx.sb.edge(iv, name, "name")

        idx_list = _fresh(ctx, "il", "index_list")
        child_pairs: list[tuple[str, str]] = []
        for idx in arg.indices:
            if isinstance(idx, IRArgRef) and idx.name in ctx.decl_plates:
                # A program value (an index covariate, a drawn class)
                # holds a zero-based QVR code, read one higher.
                shifted = LetExprBinOp(
                    op="+",
                    left=_ir_arg_to_let_expr(idx),
                    right=LetExprLiteral(value=1.0, integral=True),
                )
                vid = render_let_expr_bugs(
                    _jags_let_ctx(ctx, self._cards),
                    shifted,
                    decl_plates=ctx.decl_plates,
                )
                child_pairs.append((vid, "binary_expression"))
                continue
            vid, kind = self._render_arg_with_kind(ctx, idx)
            child_pairs.append((vid, kind))

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, (_, kind) in enumerate(child_pairs):
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            kinds.append(kind)
            if i < len(child_pairs) - 1:
                ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            idx_list,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(idx_list, "chose-alt-child-kinds", " ".join(kinds))
        for vid, _ in child_pairs:
            ctx.sb.edge(idx_list, vid, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv, "indexed_variable"

    def _render_family_ref_with_kind(
        self, ctx: _JAGSCtx, arg: IRArgFamilyRef
    ) -> tuple[str, str]:
        """Emit ``d<family>(args) T(L, U)`` for a `Truncated` wrapper.

        For non-truncated wrappers and family-refs without a morphism
        declaration in `ctx.morphisms`, fall back to a bare identifier
        of the morphism's name (the host has already bound it via a
        let-decl)."""
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            return _identifier(ctx, arg.name), "identifier"
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META.get(inner_family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:wrapper-inner-unknown:{inner_family}"],
            )
        target_inner = inner_meta.target_names.get(_BACKEND)
        if target_inner is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:no-target-name:{inner_family}"],
            )
        bound = bind_family_arguments(inner_family, decl.init_family.args or ())
        inner_args = tuple(_coerce_to_ir_arg(value) for _, value in bound)
        inner_arg_names = tuple(name for name, _ in bound)
        renamed_pairs = tuple(
            (n, a) for n, a in zip(inner_arg_names, inner_args, strict=False)
        )
        dc = self._build_distribution_call(ctx, target_inner, renamed_pairs)
        return dc, "distribution_call"

    def _render_transform_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArgTransform,
    ) -> tuple[str, str]:
        """Render an
        [`IRArgTransform`][quivers.transpile.renderers.IRArgTransform]:
        ``inv_square(x) -> 1/(x*x)`` etc.

        Emits a ``binary_expression`` tree using the JAGS grammar's
        arithmetic operators.
        """
        inner_vid, inner_kind = self._render_arg_with_kind(ctx, arg.inner)
        if arg.transform == "inv_square":
            sq = self._binary_expr(
                ctx, "*", inner_vid, inner_kind, inner_vid, inner_kind
            )
            paren = self._parenthesized(ctx, sq, "binary_expression")
            one = _number(ctx, 1)
            div = self._binary_expr(
                ctx, "/", one, "number", paren, "parenthesized_expression"
            )
            return div, "binary_expression"
        if arg.transform == "inv":
            one = _number(ctx, 1)
            div = self._binary_expr(ctx, "/", one, "number", inner_vid, inner_kind)
            return div, "binary_expression"
        if arg.transform == "neg":
            ue = _fresh(ctx, "ue", "unary_expression")
            ctx.sb.constraint(ue, "field:operator", "-")
            ctx.sb.constraint(ue, "chose-alt-fingerprint", "-")
            ctx.sb.edge(ue, inner_vid, "argument")
            return ue, "unary_expression"
        if arg.transform in ("log", "exp"):
            fc = _fresh(ctx, "fc", "function_call")
            ctx.sb.constraint(fc, "chose-alt-fingerprint", "( )")
            ctx.sb.constraint(fc, "chose-alt-child-kinds", f"identifier {inner_kind}")
            ctx.sb.constraint(fc, "ptrace-0", "Cidentifier")
            ctx.sb.constraint(fc, "ptrace-1", "T(")
            ctx.sb.constraint(fc, "ptrace-2", f"C{inner_kind}")
            ctx.sb.constraint(fc, "ptrace-3", "T)")
            fn = _identifier(ctx, arg.transform)
            ctx.sb.edge(fc, fn, "name")
            ctx.sb.edge(fc, inner_vid, "child_of")
            return fc, "function_call"
        if arg.transform == "one_minus":
            one = _number(ctx, 1)
            diff = self._binary_expr(ctx, "-", one, "number", inner_vid, inner_kind)
            return diff, "binary_expression"
        if arg.transform == "pow_neg":
            if arg.operand is None:
                raise UnsupportedConstruct(
                    f"qvr-{_BACKEND}",
                    ["transform:pow_neg: missing exponent operand"],
                )
            # pow(inner, -operand): JAGS' two-argument power builtin.
            exp_vid, exp_kind = self._render_arg_with_kind(ctx, arg.operand)
            neg = _fresh(ctx, "ue", "unary_expression")
            ctx.sb.constraint(neg, "field:operator", "-")
            ctx.sb.constraint(neg, "chose-alt-fingerprint", "-")
            ctx.sb.constraint(neg, "chose-alt-child-kinds", exp_kind)
            ctx.sb.edge(neg, exp_vid, "operand")
            call = self._function_call_from_vids(
                ctx,
                "pow",
                ((inner_vid, inner_kind), (neg, "unary_expression")),
            )
            return call, "function_call"
        raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"transform:{arg.transform}"])

    def _function_call_from_vids(
        self,
        ctx: _JAGSCtx,
        fn_name: str,
        pairs: tuple[tuple[str, str], ...],
    ) -> str:
        """Emit ``<fn_name>(<a0>, <a1>, ...)`` as a ``function_call``
        whose arguments ride an ``argument_list`` child, from
        already-rendered ``(vertex, kind)`` pairs."""
        fc = _fresh(ctx, "fc", "function_call")
        ctx.sb.constraint(fc, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(fc, "chose-alt-child-kinds", "identifier argument_list")
        ctx.sb.constraint(fc, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(fc, "ptrace-1", "T(")
        ctx.sb.constraint(fc, "ptrace-2", "Cargument_list")
        ctx.sb.constraint(fc, "ptrace-3", "T)")
        ctx.sb.edge(fc, _identifier(ctx, fn_name), "name")
        al = _fresh(ctx, "al", "argument_list")
        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, (_, kind) in enumerate(pairs):
            ctx.sb.constraint(al, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            kinds.append(kind)
            if i < len(pairs) - 1:
                ctx.sb.constraint(al, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            al,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(al, "chose-alt-child-kinds", " ".join(kinds))
        for vid, _kind in pairs:
            ctx.sb.edge(al, vid, "child_of")
        ctx.sb.edge(fc, al, "arguments")
        return fc

    def _binary_expr(
        self,
        ctx: _JAGSCtx,
        op: str,
        left_vid: str,
        left_kind: str,
        right_vid: str,
        right_kind: str,
    ) -> str:
        """Build ``<left> <op> <right>`` as a ``binary_expression``."""
        be = _fresh(ctx, "be", "binary_expression")
        ctx.sb.constraint(be, "field:operator", op)
        ctx.sb.constraint(be, "chose-alt-fingerprint", op)
        ctx.sb.constraint(be, "chose-alt-child-kinds", f"{left_kind} {right_kind}")
        ctx.sb.edge(be, left_vid, "left")
        ctx.sb.edge(be, right_vid, "right")
        return be

    def _parenthesized(self, ctx: _JAGSCtx, inner_vid: str, inner_kind: str) -> str:
        """Build ``( <inner> )`` as a ``parenthesized_expression``."""
        pe = _fresh(ctx, "pe", "parenthesized_expression")
        ctx.sb.constraint(pe, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(pe, "chose-alt-child-kinds", inner_kind)
        ctx.sb.constraint(pe, "ptrace-0", "T(")
        ctx.sb.constraint(pe, "ptrace-1", f"C{inner_kind}")
        ctx.sb.constraint(pe, "ptrace-2", "T)")
        ctx.sb.edge(pe, inner_vid, "child_of")
        return pe

    # ------------------------------------------------------------------
    # For loop wrapping
    # ------------------------------------------------------------------

    def _wrap_in_for_loops(
        self,
        ctx: _JAGSCtx,
        inner_vid: str,
        batch_dims: tuple[object, ...],
        *,
        override_var: str | None = None,
    ) -> str:
        """Wrap `inner_vid` in nested ``for (m_<axis> in 1:N) { ... }``
        loops, outermost-first by source-axis order. With no batch
        dims, returns the inner vertex directly."""
        if not batch_dims:
            return inner_vid
        current = inner_vid
        current_kind = "stochastic_relation"
        for dim in reversed(batch_dims):
            blk = _fresh(ctx, "blk", "block")
            ctx.sb.constraint(blk, "chose-alt-fingerprint", "{ }")
            ctx.sb.constraint(blk, "chose-alt-child-kinds", current_kind)
            ctx.sb.constraint(blk, "ptrace-0", "T{")
            ctx.sb.constraint(blk, "ptrace-1", f"C{current_kind}")
            ctx.sb.constraint(blk, "ptrace-2", "T}")
            ctx.sb.edge(blk, current, "child_of")

            fl = _fresh(ctx, "fl", "for_loop")
            ctx.sb.constraint(fl, "chose-alt-fingerprint", "for ( in )")
            ctx.sb.constraint(fl, "chose-alt-child-kinds", "identifier range block")
            ctx.sb.constraint(fl, "ptrace-0", "Tfor")
            ctx.sb.constraint(fl, "ptrace-1", "T(")
            ctx.sb.constraint(fl, "ptrace-2", "Cidentifier")
            ctx.sb.constraint(fl, "ptrace-3", "Tin")
            ctx.sb.constraint(fl, "ptrace-4", "Crange")
            ctx.sb.constraint(fl, "ptrace-5", "T)")
            ctx.sb.constraint(fl, "ptrace-6", "Cblock")

            loop_var_text = (
                override_var
                if override_var is not None and len(batch_dims) == 1
                else f"m_{_dim_name(dim)}"
            )
            lv = _identifier(ctx, loop_var_text)
            ctx.sb.edge(fl, lv, "variable")

            rng_vid, _ = self._dim_size_range(ctx, dim)
            ctx.sb.edge(fl, rng_vid, "range")
            ctx.sb.edge(fl, blk, "body")

            current = fl
            current_kind = "for_loop"
        return current

    def _dim_size_range(self, ctx: _JAGSCtx, dim: object) -> tuple[str, str]:
        """Build a ``1:N`` range vertex for one batch dim. Returns
        (vid, kind)."""
        if isinstance(dim, DimStatic):
            return self._range_static(ctx, dim.size), "range"
        if isinstance(dim, DimDynamic):
            return self._range_dynamic(ctx, dim.size_name), "range"
        raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"])

    # ------------------------------------------------------------------
    # Deterministic / score emission
    # ------------------------------------------------------------------

    def _emit_factor_deterministic(self, ctx: _JAGSCtx, node: IRDeterministic) -> None:
        """Emit a `factor` binding as one relation per cell.

        A rank-`n` factor denotes a rank-`n` tensor of scalar cells.
        JAGS has no array literal beyond the flat `c(...)` combine and
        no reshape to give one a rank, so the tensor is written out
        cell by cell: `<name>[i_1, ..., i_n] <- <body>` with every
        binder substituted by its coordinate. The cells are already
        enumerated, so no surrounding loop is emitted.
        """
        expr = node.expr
        if not isinstance(expr, LetExprFactor):
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"let-expr:factor:expected-factor:{node.name}"],
            )
        let_ctx = _jags_let_ctx(ctx, self._cards)
        self._check_factor_plate(node, factor_axis_sizes(let_ctx, expr))
        self._emit_cells(ctx, node, factor_cells(let_ctx, expr))

    def _emit_cells(
        self,
        ctx: _JAGSCtx,
        node: IRDeterministic,
        cells: Iterable[tuple[tuple[int, ...], LetExprNode]],
    ) -> None:
        """Emit one relation per enumerated cell of a tensor binding.

        A plated list literal is written out the way a factor is:
        `<name>[i_1, ..., i_n] <- <item>`, the nesting of the literal
        supplying the coordinates.

        Parameters
        ----------
        ctx : _JAGSCtx
            The render context.
        node : IRDeterministic
            The binding the cells belong to.
        cells : Iterable[tuple[tuple[int, ...], LetExprNode]]
            The cells' zero-based coordinates and scalar expressions.
        """
        let_ctx = _jags_let_ctx(ctx, self._cards)
        for indices, body in cells:
            dr = _fresh(ctx, "dr", "deterministic_relation")
            ctx.sb.constraint(dr, "chose-alt-fingerprint", "<-")
            ctx.sb.constraint(dr, "ptrace-0", "Cindexed_variable")
            ctx.sb.constraint(dr, "ptrace-1", "T<-")
            lhs = self._literal_indexed_variable(ctx, node.name, indices)
            ctx.sb.edge(dr, lhs, "variable")
            ctx.sb.edge(
                dr,
                render_let_expr_bugs(
                    let_ctx,
                    body,
                    decl_plates=ctx.decl_plates,
                ),
                "value",
            )
            if ctx.current_block is not None:
                ctx.sb.edge(ctx.current_block, dr, "child_of")
                ctx.block_children.setdefault(ctx.current_block, []).append(
                    _block_child_kind(ctx, dr)
                )
        self._register_deterministic_plate(ctx, node)

    def _check_factor_plate(
        self, node: IRDeterministic, sizes: tuple[int, ...]
    ) -> None:
        """Assert the factor's binder axes are exactly the binding's
        plate, so the per-cell subscripts address the whole node."""
        if node.plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"let-expr:LetExprFactor:{node.name}: a factor "
                    f"binds a tensor of scalar cells, so its plate "
                    f"carries no event axis, but this one declares "
                    f"{len(node.plate.event_dims)}"
                ],
            )
        declared = tuple(
            dim.size if isinstance(dim, DimStatic) else None
            for dim in node.plate.batch_dims
        )
        if declared != sizes:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"let-expr:LetExprFactor:{node.name}: binder axes "
                    f"{sizes} do not match the binding's declared "
                    f"plate {declared}"
                ],
            )

    def _literal_indexed_variable(
        self, ctx: _JAGSCtx, name: str, indices: tuple[int, ...]
    ) -> str:
        """``name[i_0 + 1, i_1 + 1, ...]`` from zero-based coordinates."""
        iv = _fresh(ctx, "iv", "indexed_variable")
        ctx.sb.constraint(iv, "chose-alt-fingerprint", "[ ]")
        ctx.sb.constraint(iv, "chose-alt-child-kinds", "identifier index_list")
        ctx.sb.constraint(iv, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(iv, "ptrace-1", "T[")
        ctx.sb.constraint(iv, "ptrace-2", "Cindex_list")
        ctx.sb.constraint(iv, "ptrace-3", "T]")
        ctx.sb.edge(iv, _identifier(ctx, name), "name")
        idx_list = _fresh(ctx, "il", "index_list")
        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        children: list[str] = []
        for position, value in enumerate(indices):
            children.append(_number(ctx, float(value + 1)))
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "Cnumber")
            ptrace_idx += 1
            if position < len(indices) - 1:
                ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            idx_list,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(
            idx_list,
            "chose-alt-child-kinds",
            " ".join("number" for _ in indices),
        )
        for child in children:
            ctx.sb.edge(idx_list, child, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv

    def _register_deterministic_plate(
        self, ctx: _JAGSCtx, node: IRDeterministic
    ) -> None:
        """Record a deterministic binding's plate so downstream
        `IRArgRef` references thread the right loop variable."""
        if not node.plate.batch_dims:
            return
        axes = tuple(_dim_name(d) for d in node.plate.batch_dims)
        ctx.latent_plate_info[node.name] = (
            f"m_{axes[-1]}",
            node.plate.event_dims,
            axes,
        )
        for dim in node.plate.batch_dims:
            ctx.emitted_plate_names.add(_dim_name(dim))

    def _emit_deterministic(self, ctx: _JAGSCtx, node: IRDeterministic) -> None:
        """JAGS deterministic relation ``<name> <- <expr>``.

        The RHS goes through
        `render_let_expr_bugs`
        (BUGS / JAGS share an expression grammar), with a thin
        ctx shim adapting `_JAGSCtx`'s `_fresh` /
        `panproto.SchemaBuilder` to the helper's protocol.

        When ``node.plate`` carries batch dims, the relation is
        wrapped in matching ``for (m_<axis> in 1:N) { ... }`` loops
        with the LHS indexed by the loop variables. Each let-expr
        reference to a plated binding is re-indexed against the
        same loop variables so the emitted RHS pulls per-iteration
        values out of the surrounding vectors.
        """
        plate, decl_plates = event_loop_view(node.plate, node.expr, ctx.decl_plates)
        loop_var_names = tuple(f"m_{_dim_name(dim)}" for dim in plate.batch_dims)
        # Open the deterministic relation. When plated, the LHS is
        # indexed by every loop variable; otherwise it is a bare name.
        dr = _fresh(ctx, "dr", "deterministic_relation")
        ctx.sb.constraint(dr, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(dr, "ptrace-1", "T<-")
        if loop_var_names or plate.event_dims:
            ctx.sb.constraint(dr, "ptrace-0", "Cindexed_variable")
            lhs = self._indexed_variable(
                ctx, node.name, loop_var_names, plate.event_dims
            )
        else:
            ctx.sb.constraint(dr, "ptrace-0", "Cidentifier")
            lhs = _identifier(ctx, node.name)
        ctx.sb.edge(dr, lhs, "variable")
        rewritten = index_letexpr_refs(node.expr, decl_plates, plate, loop_var_names)
        let_ctx = _jags_let_ctx(ctx, self._cards)
        val = render_let_expr_bugs(
            let_ctx,
            rewritten,
            decl_plates=decl_plates,
            row_index=loop_var_names[0] if loop_var_names else None,
        )
        ctx.sb.edge(dr, val, "value")
        wrapped = self._wrap_in_for_loops(ctx, dr, plate.batch_dims, override_var=None)
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, wrapped, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                _block_child_kind(ctx, wrapped)
            )
        # Record the deterministic's plate so downstream IRArgRef
        # references to `node.name` thread the right loop variable
        # through. Mirrors the IRSample registration in `_emit_sample`.
        if node.plate.batch_dims:
            fallback_lv = (
                loop_var_names[-1]
                if loop_var_names
                else f"m_{_dim_name(node.plate.batch_dims[-1])}"
            )
            axes = tuple(_dim_name(d) for d in node.plate.batch_dims)
            ctx.latent_plate_info[node.name] = (
                fallback_lv,
                node.plate.event_dims,
                axes,
            )
            for dim in node.plate.batch_dims:
                ctx.emitted_plate_names.add(_dim_name(dim))

    def _emit_closed_form_density(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> None:
        """Route one
        `_ZEROS_TRICK_FAMILIES`
        site to the emitter that writes its density out.

        Both site kinds reach here: an observed draw, whose whole
        emission is the density term, and the density half of a latent
        draw, whose node declaration
        `_emit_zeros_trick_latent`
        emits first.
        """
        if family == "MixtureNormal":
            self._emit_mixture_normal(
                ctx,
                name=name,
                args=args,
                arg_names=arg_names,
                plate=plate,
            )
            return
        if family == "BetaBinomial":
            self._emit_beta_binomial(
                ctx,
                name=name,
                args=args,
                arg_names=arg_names,
                plate=plate,
            )
            return
        if family == "ContinuousBernoulli":
            self._emit_continuous_bernoulli(
                ctx,
                name=name,
                args=args,
                arg_names=arg_names,
                plate=plate,
            )
            return
        if family == "Kumaraswamy":
            self._emit_kumaraswamy(
                ctx,
                name=name,
                args=args,
                arg_names=arg_names,
                plate=plate,
            )
            return
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            [
                f"family:{family}:no-closed-form:{name}: the family is "
                f"registered as scoring through the zeros trick, but "
                f"no emitter writes its density out"
            ],
        )

    def _emit_zeros_trick_latent(self, ctx: _JAGSCtx, node: IRSample) -> None:
        """Emit a latent from a family without a native JAGS distribution.

        The latent is drawn from a uniform carrier on its support. A zeros-trick
        relation then adds the requested log density. For unit-interval families
        the carrier is ``dunif(0, 1)``, whose log density is zero on the support.
        Both relations follow the latent's declared plate so site arguments and
        the sampled value use the same indices.
        """
        carrier = _ZEROS_TRICK_LATENT_CARRIER.get(node.family)
        if carrier is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:{node.family}:latent-site:{node.name}: "
                    f"the family has no JAGS distribution and scores "
                    f"through the zeros trick, which adds a density "
                    f"term without declaring a node the engine can "
                    f"sample; declaring one needs a proper uniform "
                    f"measure on the family's support, and this "
                    f"family's support carries none JAGS can name"
                ],
            )
        row_plate = ctx.decl_plates.get(node.name, node.plate)
        self._emit_sample(
            ctx,
            name=node.name,
            family=carrier,
            args=_UNIT_INTERVAL_CARRIER_ARGS,
            arg_names=_UNIT_INTERVAL_CARRIER_ARG_NAMES,
            plate=row_plate,
        )
        self._emit_closed_form_density(
            ctx,
            name=node.name,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=row_plate,
        )

    def _emit_continuous_bernoulli(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> None:
        """Emit a `ContinuousBernoulli` site through the zeros trick.

        JAGS names no continuous Bernoulli in any module a stock
        engine loads, and no reparameterisation reaches it: it is the
        exponentially-tilted uniform on `(0, 1)`, and the tilt's
        normaliser is a transcendental function of the tilt rather
        than a constant a named family absorbs.
        `continuous_bernoulli_log_pdf`
        writes the density out in `log` and `abs` alone and
        `_emit_zeros_trick_row`
        adds it to the joint.

        A residual event axis on the site would ask each row to carry
        a vector-valued variate, which the scalar closed form cannot
        express, so it raises rather than emitting a
        differently-shaped density.
        """
        if plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:ContinuousBernoulli:event-axis:{name}: "
                    f"the scalar closed form carries no event shape, "
                    f"but the site declares "
                    f"{[_dim_name(d) for d in plate.event_dims]!r}"
                ],
            )
        self._emit_zeros_trick_row(
            ctx,
            name=name,
            family="ContinuousBernoulli",
            log_density=continuous_bernoulli_log_pdf(
                _BACKEND,
                variate=name,
                args=args,
                arg_names=arg_names,
            ),
            row_plate=Plate(event_dims=(), batch_dims=plate.batch_dims),
            lifted="ContinuousBernoulli" in _ZEROS_TRICK_LIFTED_FAMILIES,
        )

    def _emit_mixture_normal(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> None:
        """Emit a `MixtureNormal` observation through the zeros trick.

        For each row, the renderer computes the weighted sum of Normal component
        densities. An observed zero from ``dpois(C - log(p))`` contributes
        ``log(p) - C`` to the joint. The zero carrier is declared in JAGS's data
        transformation block. Residual event axes raise because this emission
        supports only scalar mixtures.
        """
        if plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:MixtureNormal:event-axis:{name}: the "
                    f"scalar closed form carries no event shape, but "
                    f"the site declares "
                    f"{[_dim_name(d) for d in plate.event_dims]!r}"
                ],
            )
        weights, loc, scale = mixture_normal_components(_BACKEND, args, arg_names)
        components = mixture_component_count(
            _BACKEND,
            weights,
            ctx.decl_plates.get(weights.name if isinstance(weights, IRArgRef) else ""),
        )
        self._emit_zeros_trick_row(
            ctx,
            name=name,
            family="MixtureNormal",
            log_density=LetExprCall(
                func="log",
                args=(
                    self._mixture_density_expr(name, weights, loc, scale, components),
                ),
            ),
            row_plate=Plate(event_dims=(), batch_dims=plate.batch_dims),
            lifted="MixtureNormal" in _ZEROS_TRICK_LIFTED_FAMILIES,
        )

    def _emit_beta_binomial(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> None:
        """Emit a `BetaBinomial` observation through the zeros trick.

        Stock JAGS does not load the optional ``mix`` module, so
        `beta_binomial_log_pmf` emits the mass function with base-library
        ``loggam`` and ``logfact`` functions. Because a log mass is at most zero,
        its negation is a valid Poisson rate and requires no additive lift. The
        emitted latent conversion rate is integrated out analytically. Residual
        event axes raise because this path scores scalar counts.
        """
        if plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:BetaBinomial:event-axis:{name}: the "
                    f"scalar closed form carries no event shape, but "
                    f"the site declares "
                    f"{[_dim_name(d) for d in plate.event_dims]!r}"
                ],
            )
        self._emit_zeros_trick_row(
            ctx,
            name=name,
            family="BetaBinomial",
            log_density=beta_binomial_log_pmf(
                _BACKEND,
                variate=name,
                args=args,
                arg_names=arg_names,
            ),
            row_plate=Plate(event_dims=(), batch_dims=plate.batch_dims),
            lifted="BetaBinomial" in _ZEROS_TRICK_LIFTED_FAMILIES,
        )

    def _emit_kumaraswamy(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> None:
        """Emit a `Kumaraswamy` observation through the zeros trick.

        `kumaraswamy_log_pdf` supplies the closed-form density because stock JAGS
        has no corresponding distribution. The Poisson rate includes an
        additive lift because a continuous log density may exceed zero.
        Residual event axes raise because this path scores scalar responses.
        """
        if plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:Kumaraswamy:event-axis:{name}: the "
                    f"scalar closed form carries no event shape, but "
                    f"the site declares "
                    f"{[_dim_name(d) for d in plate.event_dims]!r}"
                ],
            )
        self._emit_zeros_trick_row(
            ctx,
            name=name,
            family="Kumaraswamy",
            log_density=kumaraswamy_log_pdf(
                _BACKEND,
                variate=name,
                args=args,
                arg_names=arg_names,
            ),
            row_plate=Plate(event_dims=(), batch_dims=plate.batch_dims),
            lifted="Kumaraswamy" in _ZEROS_TRICK_LIFTED_FAMILIES,
        )

    def _fresh_helper_name(self, ctx: _JAGSCtx, stem: str) -> str:
        """Return `stem`, or the first `<stem>_<k>` no declaration has
        taken.

        Every name the renderer synthesises is registered in
        `ctx.decl_plates` as it is emitted, so that table is the
        complete record of what is already bound.
        """
        if stem not in ctx.decl_plates:
            return stem
        index = 1
        while f"{stem}_{index}" in ctx.decl_plates:
            index += 1
        return f"{stem}_{index}"

    def _emit_zeros_trick_row(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        family: str,
        log_density: LetExprNode,
        row_plate: Plate,
        lifted: bool,
    ) -> None:
        """Add a rowwise log density to the joint with the zeros trick.

        An observed zero from ``dpois(phi)`` contributes ``-phi``. For lifted
        densities, ``phi = C - log_density``; otherwise
        ``phi = -log_density``. Helper names are fresh relative to source
        declarations to avoid duplicate relations.
        """
        zeros_name = self._fresh_helper_name(ctx, f"zeros_{name}")
        phi_name = self._fresh_helper_name(ctx, f"phi_{name}")
        ctx.decl_plates[zeros_name] = row_plate
        ctx.decl_plates[phi_name] = row_plate
        self._emit_zeros_carrier(ctx, family, zeros_name, row_plate)
        rate: LetExprNode
        if lifted:
            rate = LetExprBinOp(
                op="-",
                left=LetExprLiteral(value=_ZEROS_TRICK_OFFSET),
                right=log_density,
            )
        else:
            rate = LetExprUnaryOp(operand=log_density)
        self._emit_deterministic(
            ctx,
            IRDeterministic(
                name=phi_name,
                expr=rate,
                constraint=CSReal(),
                plate=row_plate,
            ),
        )
        self._emit_sample(
            ctx,
            name=zeros_name,
            family="Poisson",
            args=(IRArgRef(name=phi_name),),
            arg_names=("rate",),
            plate=row_plate,
            observed=True,
        )

    def _mixture_density_expr(
        self,
        variate: str,
        weights: IRArg,
        loc: IRArg,
        scale: IRArg,
        components: int,
    ) -> LetExprNode:
        """Build `sum_k w[k] * N(y; mu[k], sigma[k])` in closed form.

        The component axis is unrolled because JAGS has no reduction
        over a parameterised family; each term spells the Gaussian
        density directly, so the sum is the mixture's own density
        rather than a surrogate for it.
        """
        total: LetExprNode | None = None
        for position in range(components):
            term = self._mixture_component_density(
                variate, weights, loc, scale, position
            )
            total = (
                term if total is None else LetExprBinOp(op="+", left=total, right=term)
            )
        if total is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    "family:MixtureNormal:empty-support: a mixture "
                    "with no components has no density"
                ],
            )
        return total

    def _mixture_component_density(
        self,
        variate: str,
        weights: IRArg,
        loc: IRArg,
        scale: IRArg,
        position: int,
    ) -> LetExprNode:
        """One weighted Gaussian component,
        `w[k] * exp(-0.5 * z * z) / (sigma[k] * sqrt(2 pi))` with
        `z = (y - mu[k]) / sigma[k]`."""
        # `LetExprIndex` subscripts count from zero: the BUGS / JAGS
        # helper rebases a literal to the one-based target origin as it
        # emits, so the component position goes in unshifted.
        index = LetExprLiteral(value=float(position))
        weight_k = self._mixture_component_ref(weights, index)
        loc_k = self._mixture_component_ref(loc, index)
        scale_k = self._mixture_component_ref(scale, index)
        standardised = LetExprBinOp(
            op="/",
            left=LetExprBinOp(op="-", left=LetExprVar(name=variate), right=loc_k),
            right=scale_k,
        )
        kernel = LetExprCall(
            func="exp",
            args=(
                LetExprBinOp(
                    op="*",
                    left=LetExprLiteral(value=-0.5),
                    right=LetExprBinOp(op="*", left=standardised, right=standardised),
                ),
            ),
        )
        return LetExprBinOp(
            op="/",
            left=LetExprBinOp(op="*", left=weight_k, right=kernel),
            right=LetExprBinOp(
                op="*",
                left=scale_k,
                right=LetExprLiteral(value=SQRT_TWO_PI),
            ),
        )

    def _mixture_component_ref(self, arg: IRArg, index: LetExprLiteral) -> LetExprNode:
        """Render `<arg>[k]` for one of the three per-component
        vectors a `MixtureNormal` call supplies."""
        if not isinstance(arg, IRArgRef) or arg.indices:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:MixtureNormal:component-arg:"
                    f"{type(arg).__name__}: each of the weight, "
                    f"location and scale arguments must be a bare "
                    f"reference to a per-component vector"
                ],
            )
        return LetExprIndex(array=LetExprVar(name=arg.name), indices=(index,))

    def _emit_class_code_observation(self, ctx: _JAGSCtx, node: IRObserve) -> None:
        """Score an observed class code with `dcat` on its one-based form.

        The observed data holds QVR's zero-based class codes, and `dcat`
        counts classes from one. The `data { ... }` block binds the
        one-based code `<name>_code <- <name> + 1` from the data before
        the model compiles, and the model observes that node, so the
        engine scores the code the data holds.

        Parameters
        ----------
        ctx : _JAGSCtx
            The render context.
        node : IRObserve
            The class-code observation.
        """
        code = self._fresh_helper_name(ctx, f"{node.name}_code")
        ctx.decl_plates[code] = node.plate
        previous = ctx.current_block
        ctx.current_block = ctx.data_block
        try:
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=code,
                    expr=LetExprBinOp(
                        op="+",
                        left=LetExprVar(name=node.name),
                        right=LetExprLiteral(value=1.0, integral=True),
                    ),
                    constraint=node.constraint,
                    plate=node.plate,
                ),
            )
        finally:
            ctx.current_block = previous
        self._emit_sample(
            ctx,
            name=code,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=node.plate,
            via=node.via,
            observed=True,
        )

    def _emit_class_code_latent(self, ctx: _JAGSCtx, node: IRSample) -> None:
        """Draw a class with `dcat` and bind its zero-based QVR code.

        `dcat` draws a class counting from one; the program reads the
        drawn class as QVR's code, `<name> <- <name>_code - 1`, so a
        subscript reads it one higher and a value reads it as QVR does.

        Parameters
        ----------
        ctx : _JAGSCtx
            The render context.
        node : IRSample
            The class-code draw.
        """
        code = self._fresh_helper_name(ctx, f"{node.name}_code")
        ctx.decl_plates[code] = node.plate
        self._emit_sample(
            ctx,
            name=code,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=node.plate,
        )
        self._emit_deterministic(
            ctx,
            IRDeterministic(
                name=node.name,
                expr=LetExprBinOp(
                    op="-",
                    left=LetExprVar(name=code),
                    right=LetExprLiteral(value=1.0, integral=True),
                ),
                constraint=node.constraint,
                plate=node.plate,
            ),
        )

    def _emit_zeros_carrier(
        self,
        ctx: _JAGSCtx,
        family: str,
        zeros_name: str,
        row_plate: Plate,
    ) -> None:
        """Bind `zeros_<name>[n] <- 0` inside the `data { ... }` block.

        JAGS evaluates the data block once before compiling the model
        and treats every node it binds as observed, which is exactly
        what the zeros trick's carrier has to be.
        """
        if ctx.data_block is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [
                    f"family:{family}:no-data-block:{zeros_name}: "
                    f"the emit reached a zeros-trick site the render "
                    f"pre-pass did not see"
                ],
            )
        previous = ctx.current_block
        ctx.current_block = ctx.data_block
        try:
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=zeros_name,
                    expr=LetExprLiteral(value=0.0),
                    constraint=CSReal(),
                    plate=row_plate,
                ),
            )
        finally:
            ctx.current_block = previous

    def _emit_score(self, ctx: _JAGSCtx, node: IRScore) -> None:
        """Emit ``score <name> = <expr>`` via the JAGS zeros trick.

        JAGS has no native ``target +=`` statement; the canonical
        idiom for adding ``<expr>`` to the joint log-likelihood is the
        zeros trick, which exploits ``log P(0; lambda) = -lambda`` for
        the Poisson distribution. Concretely the renderer emits:

        ```
        C_<name> <- 1.0e6 - (<expr>)
        zero_<name> ~ dpois(C_<name>)
        ```

        The carrier ``zero_<name> <- 0`` is bound in the ``data { ... }``
        block, as every zeros-trick carrier is, so the emitted source
        is self-contained. The stochastic relation contributes
        ``-(1.0e6 - <expr>) = <expr> - 1.0e6`` to the log-density;
        the additive constant absorbs into the normalising constant.

        The score expression is wrapped in a ``parenthesized_expression``
        so the emitted source associates the subtraction correctly:
        without the parens an inner additive expression
        (``x*x + y*y``) would re-parse as
        ``1e6 - x*x + y*y = 1e6 - x^2 + y^2`` instead of the intended
        ``1e6 - (x^2 + y^2)``.
        """
        c_name = f"C_{node.name}"
        zero_name = f"zero_{node.name}"
        empty_plate = Plate(event_dims=(), batch_dims=())
        # Open the deterministic relation `C_<name> <- 1.0e6 - (<expr>)`.
        # IRScore carries no plate, so the relation lives at the
        # model-block top level with a bare-identifier LHS.
        dr = _fresh(ctx, "dr", "deterministic_relation")
        ctx.sb.constraint(dr, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(dr, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(dr, "ptrace-1", "T<-")
        lhs = _identifier(ctx, c_name)
        ctx.sb.edge(dr, lhs, "variable")
        # Build the RHS as `1.0e6 - (<expr>)`: an outer
        # `binary_expression` (`-`) with the offset on the left and a
        # `parenthesized_expression` wrapping the score expression on
        # the right. The parens force right-side grouping when the
        # score expression itself contains a `+` / `-` operator.
        offset_id = _number(ctx, _ZEROS_TRICK_OFFSET)
        let_ctx = _jags_let_ctx(ctx, self._cards)
        inner_expr_id = render_let_expr_bugs(
            let_ctx,
            node.expr,
            decl_plates=ctx.decl_plates,
        )
        # JAGS `_parenthesized` requires the child's grammar kind; the
        # score expression's outer kind is the BUGS-helper's emit, which
        # is `binary_expression` for a binop RHS and `identifier` for a
        # bare-var RHS. Use `_letexpr_outer_kind` to look it up.
        inner_kind = _letexpr_outer_kind(node.expr)
        paren_id = self._parenthesized(ctx, inner_expr_id, inner_kind)
        sub_id = self._binary_expr(
            ctx,
            "-",
            offset_id,
            "number",
            paren_id,
            "parenthesized_expression",
        )
        ctx.sb.edge(dr, sub_id, "value")
        # Attach the deterministic relation under the model block.
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, dr, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                _block_child_kind(ctx, dr)
            )
        # Record the carrier's plate so the subsequent `dpois(C_<name>)`
        # call renders `C_<name>` as a bare identifier (no auto-leading
        # indices). `_emit_sample`'s ref-emission path consults
        # `latent_plate_info` and `decl_plates` for the rate-arg
        # IRArgRef.
        ctx.decl_plates[c_name] = empty_plate
        ctx.decl_plates[zero_name] = empty_plate
        self._emit_zeros_carrier(ctx, "Poisson", zero_name, empty_plate)
        # Emit the stochastic relation `zero_<name> ~ dpois(C_<name>)`.
        self._emit_sample(
            ctx,
            name=zero_name,
            family="Poisson",
            args=(IRArgRef(name=c_name),),
            arg_names=("rate",),
            plate=empty_plate,
            observed=True,
        )

    # ------------------------------------------------------------------
    # Plate-name disambiguation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # model_block finalisation
    # ------------------------------------------------------------------

    def _finalise_model_block(self, ctx: _JAGSCtx) -> None:
        """Pin the ``model { ... }`` alternative on the model_block so
        the pretty-printer emits the keyword-prefixed form.

        The auto-derived theory supplies the inter-child layout from the
        grammar's whitespace conventions (newline + two-space indent
        between sibling statements). We only need to disambiguate the
        ``{ ... }`` vs. ``model { ... }`` alternative."""
        mb = ctx.model_block
        children = ctx.block_children.get(mb, [])
        ctx.sb.constraint(mb, "chose-alt-fingerprint", "model { }")
        ctx.sb.constraint(
            mb, "chose-alt-child-kinds", " ".join(children) if children else ""
        )
        ctx.sb.constraint(mb, "ptrace-0", "Tmodel")
        ctx.sb.constraint(mb, "ptrace-1", "T{")
        for i, kind in enumerate(children):
            ctx.sb.constraint(mb, f"ptrace-{2 + i}", f"C{kind}")
        ctx.sb.constraint(mb, f"ptrace-{2 + len(children)}", "T}")

    def _finalise_data_block(self, ctx: _JAGSCtx) -> None:
        """Pin the ``data { ... }`` alternative and its child list.

        Mirrors
        `_finalise_model_block`:
        the auto-derived theory supplies the inter-child layout, and
        the emit only has to name which alternative and which children
        the block carries.
        """
        db = ctx.data_block
        if db is None:
            return
        children = ctx.block_children.get(db, [])
        ctx.sb.constraint(db, "chose-alt-fingerprint", "data { }")
        ctx.sb.constraint(
            db, "chose-alt-child-kinds", " ".join(children) if children else ""
        )
        ctx.sb.constraint(db, "ptrace-0", "Tdata")
        ctx.sb.constraint(db, "ptrace-1", "T{")
        for i, kind in enumerate(children):
            ctx.sb.constraint(db, f"ptrace-{2 + i}", f"C{kind}")
        ctx.sb.constraint(db, f"ptrace-{2 + len(children)}", "T}")


# ---------------------------------------------------------------------------
# Arg pre-rewrite (idempotent thread-once helper).
# ---------------------------------------------------------------------------


def _rewrite_arg(
    ctx: _JAGSCtx,
    arg: IRArg,
    loop_vars: tuple[str, ...],
    axis_to_lv: dict[str, str],
    via: str | None,
    *,
    scalar_slot: bool = False,
) -> IRArg:
    """Thread latent loop-vars + event ranges + via fibration through
    every ref nested inside `arg`.

    The rewrite is a SINGLE PASS: it inspects every IRArgRef in `arg`
    and rewrites it once. The output is idempotent on subsequent passes
    by construction because the appended indices for an already-rewritten
    latent reference are detected via the ``__jags_range__:`` sentinel
    prefix on the trailing event-range entries.

    `axis_to_lv` maps each axis name in the surrounding plate to the
    loop var that iterates it. A latent ref whose recorded axis appears
    in this map indexes through the *current* loop var (so two samples
    on the same axis share the same loop body); one whose axis the
    surrounding plate does not iterate indexes through the caller's
    `via` fibration where there is one, and through the latent's own
    fallback loop var otherwise.

    An unindexed ref to any other declared value whose every axis the
    loops visit, filling a rank-0 family parameter (`scalar_slot`),
    is read at the loops' position: a vector `z` over `Feat` filling
    a `Normal`'s location under a loop over `Feat` is `z[m_Feat]`.

    Parameters
    ----------
    ctx : _JAGSCtx
        The render context.
    arg : IRArg
        The argument.
    loop_vars : tuple[str, ...]
        The site's batch loop variables.
    axis_to_lv : dict[str, str]
        Each visited axis name mapped to its loop variable.
    via : str | None
        The fibration the site reads its groups through, if any.
    scalar_slot : bool
        Whether the argument fills a rank-0 family parameter.

    Returns
    -------
    IRArg
        The argument with every reference indexed at the loops'
        position.
    """
    if isinstance(arg, IRArgRef):
        new_indices = tuple(
            _rewrite_arg(ctx, idx, loop_vars, axis_to_lv, via) for idx in arg.indices
        )
        info = ctx.latent_plate_info.get(arg.name)
        if info is None:
            visited = visited_position(ctx.decl_plates.get(arg.name), axis_to_lv)
            if scalar_slot and visited and not arg.indices:
                return IRArgRef(
                    name=arg.name, indices=tuple(IRArgRef(name=lv) for lv in visited)
                )
            if new_indices == arg.indices:
                return arg
            return IRArgRef(name=arg.name, indices=new_indices)
        fallback_lv, event_dims, axes = info
        declared = ctx.decl_plates.get(arg.name)
        if not arg.indices and declared is not None and len(axes) > 1:
            # A table over several axes: the axes the surrounding loop
            # iterates are read at its row, and the trailing ones it
            # does not are read whole, as the event the consumer reads
            # (a gathered table's class axis under a Categorical).
            led = tuple(
                IRArgRef(name=axis_to_lv[str(dim.name)])
                for dim in declared.batch_dims
                if str(dim.name) in axis_to_lv
            )
            rest = declared.batch_dims[len(led) :]
            if led and all(str(dim.name) not in axis_to_lv for dim in rest):
                return IRArgRef(
                    name=arg.name,
                    indices=(
                        *led,
                        *(_event_range_ir(dim) for dim in rest),
                        *(_event_range_ir(ed) for ed in event_dims),
                    ),
                )
        # Choose the loop-var index expression.
        if axes and axes[-1] in axis_to_lv:
            # Surrounding plate iterates the latent's axis directly:
            # reuse the current loop var (`theta[m_Doc, ...]`).
            lv_idx: IRArg = IRArgRef(name=axis_to_lv[axes[-1]])
        elif via is not None and loop_vars:
            # Caller-set via overrides (passed via fibration).
            lv_idx = IRArgRef(
                name=via,
                indices=(IRArgRef(name=loop_vars[0]),),
            )
        else:
            lv_idx = IRArgRef(name=fallback_lv)

        event_indices = tuple(_event_range_ir(ed) for ed in event_dims)
        # IDEMPOTENCE GUARD: if `arg.indices` already ends with the
        # event_indices we'd append, leave them alone.
        if arg.indices and _has_trailing(arg.indices, event_indices):
            return IRArgRef(name=arg.name, indices=new_indices)
        if arg.indices:
            return IRArgRef(
                name=arg.name,
                indices=new_indices + event_indices,
            )
        return IRArgRef(
            name=arg.name,
            indices=(lv_idx,) + event_indices,
        )
    if isinstance(arg, IRArgBroadcast):
        inner = _rewrite_arg(ctx, arg.value, loop_vars, axis_to_lv, via)
        if inner is arg.value:
            return arg
        return IRArgBroadcast(value=inner, target_shape=arg.target_shape)
    if isinstance(arg, IRArgList):
        elements = tuple(
            _rewrite_arg(ctx, e, loop_vars, axis_to_lv, via) for e in arg.elements
        )
        if elements == arg.elements:
            return arg
        return IRArgList(elements=elements)
    if isinstance(arg, IRArgMatrix):
        rows = tuple(
            IRArgList(
                elements=tuple(
                    _rewrite_arg(ctx, e, loop_vars, axis_to_lv, via)
                    for e in row.elements
                )
            )
            for row in arg.rows
        )
        return IRArgMatrix(rows=rows)
    if isinstance(arg, IRArgTransform):
        inner = _rewrite_arg(
            ctx, arg.inner, loop_vars, axis_to_lv, via, scalar_slot=scalar_slot
        )
        operand = (
            None
            if arg.operand is None
            else _rewrite_arg(
                ctx, arg.operand, loop_vars, axis_to_lv, via, scalar_slot=scalar_slot
            )
        )
        if inner is arg.inner and operand is arg.operand:
            return arg
        return IRArgTransform(inner=inner, transform=arg.transform, operand=operand)
    return arg


def _has_trailing(indices: tuple[IRArg, ...], suffix: tuple[IRArg, ...]) -> bool:
    """Return True iff `indices` ends with `suffix` (by name equality).

    Used by `_rewrite_arg` for idempotence: if a latent ref already has
    the event-range sentinel indices appended, do not re-append them on
    the next rewrite pass.
    """
    if not suffix:
        return False
    if len(indices) < len(suffix):
        return False
    tail = indices[-len(suffix) :]
    for a, b in zip(tail, suffix, strict=False):
        if not (isinstance(a, IRArgRef) and isinstance(b, IRArgRef)):
            return False
        if a.name != b.name:
            return False
    return True


# ---------------------------------------------------------------------------
# Renderer-local context
# ---------------------------------------------------------------------------


class _JAGSCtx(RenderContext):
    """JAGS-specific extension of
    [`RenderContext`][quivers.transpile.renderers.RenderContext]
    carrying the active block vertex, emitted plate names, and the
    per-block child-kind list used to assemble `chose-alt-child-kinds`
    constraints after the IR walk."""

    def __init__(
        self,
        *,
        sb: panproto.SchemaBuilder,
        morphisms: dict,
        lets: dict,
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=lets)
        self.current_block: str | None = None
        self.model_block: str = ""
        #: The `data { ... }` transformation block, created only when
        #: the program needs one. JAGS evaluates it once before the
        #: model, and every node it binds becomes observed data.
        self.data_block: str | None = None
        #: Names of scalar-shaped bindings, broadcast candidates for a
        #: vector / matrix distribution slot. Populated in `render` from
        #: the lowered IR.
        self.scalar_refs: frozenset[str] = frozenset()
        #: Every bindable name in the program. A vector-slot reference
        #: outside this set is unresolvable and raises.
        self.bound_refs: frozenset[str] = frozenset()
        self.emitted_plate_names: set[str] = set()
        self.block_children: dict[str, list[str]] = {}
        #: Declared plate for every named binding (inputs, samples,
        #: observes, deterministics, marginalize-latents); read by
        #: `_emit_deterministic` to re-index plated var references
        #: inside lifted let-expressions.
        self.decl_plates: dict[str, Plate] = {}
        #: For every previously-bound latent name, record the
        #: (fallback_loop_var, event_dims, axis_names) used at its
        #: sample site. The fallback loop var is used when no
        #: surrounding plate covers any of the latent's axes; otherwise
        #: the rewriter looks up the axis in the current sample's
        #: axis-to-loop-var map.
        self.latent_plate_info: dict[
            str, tuple[str, tuple[object, ...], tuple[str, ...]]
        ] = {}


def _as_jags_ctx(ctx: RenderContext) -> _JAGSCtx:
    """Narrow a base `RenderContext` to the JAGS extension."""
    if not isinstance(ctx, _JAGSCtx):
        raise UnsupportedConstruct(f"qvr-{_BACKEND}", ["ctx:type-mismatch"])
    return ctx


# ---------------------------------------------------------------------------
# Schema-builder helpers (vertex / identifier / number constructors).
# ---------------------------------------------------------------------------


def _fresh(ctx: _JAGSCtx, prefix: str, kind: str) -> str:
    """Allocate a fresh vertex with `prefix`-prefixed id of kind
    `kind`."""
    ctx.fresh_counter += 1
    vid = f"{prefix}_{ctx.fresh_counter}"
    ctx.sb.vertex(vid, kind)
    return vid


def _vertex(ctx: _JAGSCtx, vid: str, kind: str) -> str:
    """Register a vertex with an explicit id (used for the
    `source_file` root vertex)."""
    ctx.sb.vertex(vid, kind)
    return vid


class _JagsLetCtx:
    """Adapter exposing the protocol
    `render_let_expr_bugs`
    expects (``v``, ``e``, ``lit``, ``fresh``, ``constraint``,
    ``target``, ``cards``) on top of a `_JAGSCtx`."""

    def __init__(self, ctx: _JAGSCtx, cards: dict[str, int]) -> None:
        self._ctx = ctx
        self.cards = cards
        self.target = "jags"

    def fresh(self, prefix: str) -> str:
        self._ctx.fresh_counter += 1
        return f"{prefix}_{self._ctx.fresh_counter}"

    def v(self, vid: str, kind: str) -> str:
        self._ctx.sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._ctx.sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._ctx.sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._ctx.sb.constraint(vid, sort, value)

    def range_1_to(self, upper: str) -> str:
        """Build the `1:<upper>` range vertex the JAGS grammar wants."""
        return self.range_between("1", upper)

    def range_between(self, lower: str, upper: str) -> str:
        """Build the `<lower>:<upper>` range vertex the JAGS grammar
        wants.

        Either bound is a `number` when it reads as an integer and an
        `identifier` otherwise, which is how a dynamic axis extent
        reaches the range.
        """
        rng = _fresh(self._ctx, "rng", "range")
        self._ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
        kinds: list[str] = []
        bounds: list[tuple[str, str]] = []
        for text, edge in ((lower, "lower"), (upper, "upper")):
            is_static = text.lstrip("-").isdigit()
            kinds.append("number" if is_static else "identifier")
            bounds.append(
                (
                    _number(self._ctx, float(text))
                    if is_static
                    else _identifier(self._ctx, text),
                    edge,
                )
            )
        self._ctx.sb.constraint(rng, "chose-alt-child-kinds", " ".join(kinds))
        self._ctx.sb.constraint(rng, "ptrace-0", f"C{kinds[0]}")
        self._ctx.sb.constraint(rng, "ptrace-1", "T:")
        self._ctx.sb.constraint(rng, "ptrace-2", f"C{kinds[1]}")
        for vid, edge in bounds:
            self._ctx.sb.edge(rng, vid, edge)
        return rng


def _jags_let_ctx(ctx: _JAGSCtx, cards: dict[str, int]) -> _JagsLetCtx:
    """Construct a `_JagsLetCtx` bound to `ctx` and `cards`."""
    return _JagsLetCtx(ctx, cards)


def _identifier(ctx: _JAGSCtx, text: str) -> str:
    """Build an `identifier` vertex with literal-value and
    fingerprint constraints."""
    vid = _fresh(ctx, "id", "identifier")
    ctx.sb.constraint(vid, "literal-value", text)
    ctx.sb.constraint(vid, "chose-alt-fingerprint", text)
    return vid


def _number(ctx: _JAGSCtx, value: float) -> str:
    """Build a `number` vertex carrying `value` as its
    literal-value."""
    vid = _fresh(ctx, "num", "number")
    text = str(int(value)) if value == int(value) else repr(value)
    ctx.sb.constraint(vid, "literal-value", text)
    ctx.sb.constraint(vid, "chose-alt-fingerprint", text)
    return vid


def _letexpr_outer_kind(expr: LetExprNode) -> str:
    """Return the JAGS grammar kind of the vertex produced by
    `render_let_expr_bugs`
    for `expr`.

    Used by the score-emission path to wire a
    ``parenthesized_expression`` around the rendered score expression:
    the parens vertex carries a ``chose-alt-child-kinds`` constraint
    whose value is the inner child's grammar kind, so the caller must
    know that kind before emission.
    """
    if isinstance(expr, LetExprLiteral):
        return "number"
    if isinstance(expr, LetExprVar):
        return "identifier"
    if isinstance(expr, LetExprBinOp):
        return "binary_expression"
    if isinstance(expr, LetExprUnaryOp):
        return "unary_expression"
    if isinstance(expr, LetExprCall):
        return "function_call"
    if isinstance(expr, LetExprIndex):
        return "indexed_variable"
    if isinstance(expr, (LetExprList, LetExprFactor)):
        # Both lower to a `c(...)` function_call in the shared
        # BUGS / JAGS expression helper.
        return "function_call"
    raise UnsupportedConstruct(
        f"qvr-{_BACKEND}",
        [f"let-expr:outer-kind:{type(expr).__name__}: unhandled"],
    )


def _event_range_ir(dim: object) -> IRArg:
    """Encode a `1:E` range as a sentinel IRArgRef whose name carries
    the upper bound. The renderer recognises the prefix and emits a
    JAGS `range` vertex (`1:E`)."""
    if isinstance(dim, DimStatic):
        return IRArgRef(name=f"{_RANGE_SENTINEL_PREFIX}{dim.size}")
    if isinstance(dim, DimDynamic):
        return IRArgRef(name=f"{_RANGE_SENTINEL_PREFIX}{dim.size_name}")
    raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"])


def _fresh_loop_names(taken: tuple[str, ...], dims: tuple[Dim, ...]) -> tuple[str, ...]:
    """Name one loop variable per dim, avoiding every name in `taken`.

    A residual event axis can repeat an axis already iterated by the
    site's batch plate (a square row-stochastic matrix names the same
    object on both sides), and two `for` loops over the same variable
    name would silently alias, so a repeat gets a numeric suffix.
    """
    used = set(taken)
    out: list[str] = []
    for dim in dims:
        base = f"m_{_dim_name(dim)}"
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        out.append(candidate)
    return tuple(out)


def _dim_name(dim: object) -> str:
    """Return the source-axis name carried by a Dim."""
    if isinstance(dim, (DimStatic, DimDynamic)):
        return dim.name
    raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"])


def _mirror_dim(dim: Dim) -> Dim:
    """A second axis of the same extent as `dim`, named apart from it.

    A square helper over one plate loops over the plate twice, and
    the loop variables are named after the axes, so the second copy
    carries its own name.

    Parameters
    ----------
    dim : Dim
        The axis.

    Returns
    -------
    Dim
        The copy, named ``<name>_of``.
    """
    if isinstance(dim, DimStatic):
        return DimStatic(size=dim.size, name=f"{dim.name}_of")
    if isinstance(dim, DimDynamic):
        return DimDynamic(size_name=dim.size_name, name=f"{dim.name}_of")
    raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"])


def _lhs_kind(plate: Plate, loop_vars: tuple[str, ...]) -> str:
    """Return the grammar kind of the LHS of a stochastic_relation.

    Bare names (no batch / event dims) emit as ``identifier``;
    every other shape lands as ``indexed_variable``."""
    if not plate.batch_dims and not plate.event_dims:
        return "identifier"
    if not loop_vars and not plate.event_dims:
        return "identifier"
    return "indexed_variable"


def _block_child_kind(ctx: _JAGSCtx, vid: str) -> str:
    """Look up the grammar kind of `vid` via the fresh-id naming
    convention."""
    del ctx
    if vid.startswith("fl_"):
        return "for_loop"
    if vid.startswith("sr_"):
        return "stochastic_relation"
    if vid.startswith("dr_"):
        return "deterministic_relation"
    return "for_loop"


def _coerce_to_ir_arg(raw: object) -> IRArg:
    """Coerce a morphism's init_family arg (which may be a raw float
    or a name string) to an IRArg variant."""
    if not isinstance(raw, (IRArg, int, float, str)):
        raw = atom_value(raw)
    if isinstance(raw, IRArg):
        return raw
    if isinstance(raw, (int, float)):
        return IRArgNumber(value=float(raw))
    if isinstance(raw, str):
        return IRArgRef(name=raw)
    raise UnsupportedConstruct(f"qvr-{_BACKEND}", [f"arg:coerce:{type(raw).__name__}"])


def _ir_arg_to_let_expr(arg: IRArg) -> LetExprNode:
    """Convert a scalar [`IRArg`][quivers.transpile.ir.IRArg] to the
    let-expression tree the shared BUGS / JAGS emitter consumes.

    Only the scalar arg variants a broadcast can repeat are handled:
    a numeric literal, a bare reference, or an indexed reference. Any
    other variant raises rather than emitting an unrepresentable
    ``rep`` argument.
    """
    if isinstance(arg, IRArgNumber):
        return LetExprLiteral(value=arg.value)
    if isinstance(arg, IRArgRef):
        if not arg.indices:
            return LetExprVar(name=arg.name)
        return LetExprIndex(
            array=LetExprVar(name=arg.name),
            indices=tuple(_ir_arg_to_let_expr(i) for i in arg.indices),
        )
    raise UnsupportedConstruct(
        f"qvr-{_BACKEND}",
        [f"broadcast:value:{type(arg).__name__}"],
    )


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
    deterministic ones. A let-bound scalar thus lands in
    `scalar_refs` exactly like a free scalar input, so the JAGS
    ``rep(<scalar>, K)`` broadcast fires on both.
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


__all__ = ["JAGSRenderer"]
