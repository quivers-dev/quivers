"""Render transpilation IR as a BUGS model block.

Variables are declared by stochastic or deterministic relations, with
batch axes emitted as nested loops. The renderer collapses supported
categorical mixtures to ``dcat`` and refuses every other
marginalization scope, since the language has no statement that adds
a free log-density term to the joint. Family names and argument
conversions come from ``FAMILY_META``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Callable, Literal

import panproto

from quivers.dsl.ast_nodes import (
    Expr,
    MorphismDecl,
    MorphismInitFamily,
)
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
from quivers.transpile.renderers._base import (
    refuse_ungrouped_row_marginalize,
    BlockKind,
    IRArgTransform,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    reorder_negbin_args,
    reorder_weibull_args,
)
from quivers.transpile.renderers._qiec import render_computations_static
from quivers.transpile.renderers._python_helpers import (
    marginal_support_size,
    marginalize_body,
    marginalize_fibration,
)
from quivers.transpile.renderers._one_based import is_class_code_family
from quivers.transpile.renderers._bugs_helpers import (
    TRUNCATION_FINGERPRINT,
    CategoricalMixture,
    categorical_mixture,
    factor_axis_sizes,
    factor_cells,
    half_support_truncation,
    hoist_list_operands,
    index_letexpr_refs,
    list_cells,
    list_sizes,
    push_scalar_dets_into_loops,
    render_let_expr_bugs,
    reorder_binomial_dbin,
    reorder_half_studentt_dt,
    reorder_pareto_dpar,
    split_event_dims,
)


class _BugsLetCtx:
    """Bridge ``_BugsCtx.sb`` to the
    `render_let_expr_bugs`
    helper's ctx protocol (``v``, ``e``, ``lit``, ``fresh``,
    ``constraint``) and carry the per-render ``cards`` map (for
    factor unrolling) plus the ``target`` discriminator the helper
    stamps onto its error tags."""

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        fresh: Callable[[str], str],
        cards: dict[str, int],
        target: str,
    ) -> None:
        self._sb = sb
        self._fresh_fn = fresh
        self.cards = cards
        self.target = target

    def fresh(self, prefix: str) -> str:
        return self._fresh_fn(prefix)

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)

    def range_1_to(self, upper: str) -> str:
        """Build the `1:<upper>` range vertex the BUGS grammar wants."""
        return self.range_between("1", upper)

    def range_between(self, lower: str, upper: str) -> str:
        """Build the `<lower>:<upper>` range vertex the BUGS grammar
        wants.

        Either bound is a `number` when it reads as an integer and an
        `identifier` otherwise, which is how a dynamic axis extent
        reaches the range.
        """
        rng_id = self._fresh_fn("rng")
        self._sb.vertex(rng_id, "range")
        for text, edge in ((lower, "lower"), (upper, "upper")):
            bound_id = self._fresh_fn("num")
            self._sb.vertex(
                bound_id,
                "number" if text.lstrip("-").isdigit() else "identifier",
            )
            self._sb.constraint(bound_id, "literal-value", text)
            self._sb.edge(rng_id, bound_id, edge)
        return rng_id


#: Per-renderer alias-transform table. Keys are the *renamed* arg
#: names from `FAMILY_META[family].arg_aliases["bugs"]`; values are
#: the arithmetic transform the renderer applies before emission.
#: BUGS uses precision (`tau = 1/(scale * scale)`) where torch's
#: `Normal` uses scale, so `arg_aliases["bugs"]["scale"] = "tau"`
#: pairs with `_ALIAS_TRANSFORMS["tau"] = "inv_square"`.
_ALIAS_TRANSFORMS: dict[str, str] = {
    "tau": "inv_square",
}

#: Per-family override of the arg-alias arithmetic transform. The
#: shared ``tau`` alias assumes a precision (``1/scale^2``)
#: parameterisation, right for ``dnorm`` / ``dt`` but wrong for
#: ``ddexp``. BUGS' ``ddexp(mu, tau)`` is rate-parameterised (density
#: ``(tau/2) * exp(-tau*|x-mu|)``), so ``Laplace``'s scale maps to the
#: rate ``tau = 1/scale`` rather than ``1/scale^2``.
_FAMILY_ALIAS_TRANSFORM_OVERRIDE: dict[str, dict[str, str]] = {
    "Laplace": {"tau": "inv"},
    "Logistic": {"tau": "inv"},
}

#: Renderer-supplied arg renames for families whose
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] entry
#: records no ``bugs`` alias even though the BUGS distribution is
#: parameterised differently from the torch one. ``Logistic`` is the
#: case: torch carries ``(loc, scale)`` while ``dlogis(mu, tau)`` has
#: density ``tau * e^{-tau(x-mu)} / (1 + e^{-tau(x-mu)})^2``, i.e.
#: the second slot is the rate ``tau = 1/scale``. Passing the scale
#: through unchanged scores a Logistic of scale ``1/s``, which is a
#: different density at every point rather than a constant offset.
#: The rename feeds the same transform pipeline the ``FAMILY_META``
#: aliases do, with the ``inv`` transform supplied by
#: `_FAMILY_ALIAS_TRANSFORM_OVERRIDE`.
_FAMILY_ALIAS_OVERRIDE: dict[str, dict[str, str]] = {
    "Logistic": {"scale": "tau"},
    "LogNormal": {"scale": "tau"},
}


#: BUGS-side argument injection for QVR families whose underlying
#: torch distribution carries fewer parameters than the BUGS
#: distribution it maps to. ``HalfNormal(scale)`` maps to BUGS'
#: ``dnorm(0, tau)``; the renderer prepends an ``IRArgNumber(0)``
#: under the loc-position arg name so the alias-transform pipeline
#: still rewrites the scale into ``tau = 1/(scale*scale)``.
#: The constant ``log(2)`` offset that distinguishes HalfNormal from
#: the full Normal is absorbed by the constant-spread tolerance in
#: [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match].
_PREPEND_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

#: BUGS-side argument injection for QVR families that map to BUGS'
#: ``dt(mu, tau, k)`` distribution. BUGS Student-t requires three
#: parameters (location, precision, degrees of freedom); Cauchy is
#: the special case ``k = 1``. The renderer appends ``IRArgNumber(1)``
#: as a trailing ``df`` argument after the alias-renaming pipeline so
#: the emitted call is ``dt(mu, tau, 1)``.
_APPEND_DF_ONE: frozenset[str] = frozenset({"Cauchy", "HalfCauchy"})


#: BUGS / JAGS zeros-trick constant offset. The Poisson PMF satisfies
#: ``log P(X = 0; lambda) = -lambda``, so emitting
#: ``zero_<name> ~ dpois(C - <expr>)`` with a host-bound
#: ``zero_<name> = 0`` adds ``<expr>`` to the joint log-likelihood up
#: to the additive constant ``-C`` (which absorbs into the
#: normalising constant and does not affect inference). ``C`` must
#: stay strictly larger than ``<expr>`` over the entire parameter
#: support; ``1.0e6`` is the conventional safe default for typical
#: BUGS / JAGS fixtures and matches the offset shipped in WinBUGS /
#: OpenBUGS examples.
_ZEROS_TRICK_OFFSET: float = 1.0e6


#: Families the BUGS language can neither name nor reach through a
#: written-out density, mapped to what the second route runs into.
#:
#: Each of these three is written out in closed form by the JAGS
#: renderer, so the density itself is not the obstacle: a finite
#: normal mixture, a Kumaraswamy, and a continuous Bernoulli are all
#: ordinary arithmetic in ``log``, ``pow`` and ``abs``, every one of
#: which the BUGS function library carries. What BUGS lacks is a
#: *statement that adds an arithmetic expression to the joint
#: log-density*. It has no ``target +=``, and the zeros / ones trick
#: that stands in for one needs a carrier node the data binds: the
#: language has no ``data { ... }`` block (data arrive from a separate
#: host-side list the model source cannot declare), a node bound by
#: ``zeros[n] <- 0`` inside ``model`` is logical and so cannot also
#: carry a distribution, and an already-observed node cannot stand in
#: because the density reads that node, so scoring it through its own
#: relation would cycle the graph.
#:
#: The route round it that BUGS does express is a different model
#: rather than the same one. Writing a mixture as
#: ``T[n] ~ dcat(w); y[n] ~ dnorm(mu[T[n]], tau[T[n]])`` allocates a
#: discrete latent per row, so the joint it scores lives on a strictly
#: larger space than the marginal QVR names and the two densities
#: agree at no point rather than up to an additive constant. The same
#: holds for any auxiliary-variable construction of the other two.
#:
#: The boundary is therefore the BUGS *language*, not the engine: the
#: ``panproto-test-bugs`` image runs JAGS, which is why every sibling
#: ``jags`` cell scores the same model.
_NO_BUGS_FREE_DENSITY_TERM: dict[str, str] = {
    "MixtureNormal": (
        "a finite mixture is an explicit weighted density in the "
        "BUGS function library, but adding one to the joint needs a "
        "free log-density term the language has no statement for"
    ),
    "Kumaraswamy": (
        "the density is elementary in `log` and `pow`, but adding a "
        "written-out density to the joint needs a free log-density "
        "term the language has no statement for"
    ),
    "ContinuousBernoulli": (
        "the density is elementary in `log` and `abs`, but adding a "
        "written-out density to the joint needs a free log-density "
        "term the language has no statement for"
    ),
}


#: Families whose [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]
#: ``bugs`` entry names a distribution no BUGS engine implements,
#: mapped to the reason the boundary stands.
#:
#: ``BetaBinomial`` is the whole registry. The BUGS distribution
#: catalogue (Lunn et al. 2012, appendix I) carries ``dbin``,
#: ``dbeta``, ``dnegbin`` and ``dhyper`` but no beta-binomial, and the
#: ``dbetabin`` spelling the registry records resolves in no engine
#: this renderer's source can reach: WinBUGS and OpenBUGS have no such
#: name, and JAGS carries ``dbetabin`` only in the optional ``mix``
#: module, which a stock engine (``basemod``, ``bugs``, ``dic``) never
#: loads. Emitting the call anyway produces source that parses and
#: then fails to compile with ``Unknown distribution``.
#:
#: Neither of the two ways round it survives inspection.
#:
#: (i) The compound ``p[n] ~ dbeta(a, b); y[n] ~ dbin(p[n], N)`` is a
#: different model: it allocates a latent rate per row, so the joint
#: it scores lives on a larger space than the beta-binomial marginal
#: QVR names, and the two densities agree at no point rather than up
#: to an additive constant.
#:
#: (ii) The zeros trick writes the marginal out in closed form (see
#: `beta_binomial_log_pmf`)
#: and adds it through ``zeros[n] ~ dpois(C - log p(y_n))``, which is
#: what the JAGS renderer does. It needs ``zeros[n]`` to be *data*,
#: and JAGS binds it in the ``data { ... }`` block the JAGS renderer
#: emits ahead of the model. The BUGS language has no such block: its
#: data arrive from a separate host-side list the model source cannot
#: declare, and a node bound by ``zeros[n] <- 0`` inside ``model``
#: is logical, so giving it a distribution as well is a duplicate
#: definition. Nor can an already-observed node stand in as the
#: carrier: the closed form reads ``y[n]``, so scoring it through
#: ``y[n]``'s own relation would put ``y[n]`` on both sides of a
#: single edge and cycle the graph.
#:
#: `_emit_score_node`
#: does emit a host-bound ``zero_<name>``, and the difference is what
#: the two constructs promise. ``score`` is an explicit request for an
#: extra log-density term, and the carrier it needs is part of that
#: construct's documented host contract; an ``observe`` site promises
#: that the emitted model scores the named likelihood on its own. A
#: caller who ran the emitted source without knowing to add a zeros
#: vector would get a model that compiles, samples the carrier as a
#: free Poisson node, and drops the observation's likelihood term
#: entirely, which is a silently different posterior rather than a
#: reported gap.
#:
#: This mirrors the boundary ``MixtureNormal`` already stands on,
#: where JAGS scores through the zeros trick and BUGS rejects.
_NO_BUGS_DISTRIBUTION: dict[str, str] = {
    "BetaBinomial": (
        "the BUGS distribution catalogue has no beta-binomial, and "
        "the zeros trick that would write its closed-form marginal "
        "into the joint needs a data-bound carrier the BUGS language "
        "cannot declare (it has no `data { ... }` block)"
    ),
}


class _BugsCtx(RenderContext):
    """BUGS-renderer-internal carrier extending `RenderContext`.

    Adds the declaration-plate table (axis-name lookup for index
    emission), the current `via` fibration in scope (the active
    scope's [`IRObserve.via`][quivers.transpile.ir.IRObserve.via]
    when rendering arg indices), and the enclosing plate / loop-
    name pair so an arg's index expansion can resolve loop
    variables by axis name.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema builder the renderer emits into.
    morphisms : dict[str, MorphismDecl]
        The module's morphism declarations, by name.
    defines : dict[str, Expr]
        The module's define expressions, by name.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        morphisms: dict[str, MorphismDecl],
        defines: dict[str, Expr],
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=defines)
        self.decl_plates: dict[str, Plate] = {}
        self.via: str | None = None
        self.enclosing_plate: Plate | None = None
        self.enclosing_loop_names: tuple[str, ...] = ()
        self.block_id: str = ""
        self.broadcast_helpers: dict[tuple[str, int], str] = {}


class BUGSRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to BUGS.

    Implements the four
    [`RendererBase`][quivers.transpile.renderers.RendererBase]
    dispatch points (`declare`, `sample`, `marginalize`, `broadcast`)
    plus `target_protocol` and overrides `render` to install the
    `model { ... }` block prologue and to wire the BUGS-specific
    declaration-plate table that drives slice-emission for vector
    arg positions.
    """

    target: str = "bugs"

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol for the BUGS tree-sitter grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("bugs")

    # ------------------------------------------------------------------
    # Top-level render.
    # ------------------------------------------------------------------

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Emit the BUGS source schema for `ir`.

        Wraps the IR walk in a `source_file -> model_block` shell so
        the emitted bytes are a complete BUGS file. The model_block's
        `ptrace-0 = Tmodel` constraint forces the emitter to render
        the optional `model` keyword.

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
        self._reject_list_args(ir)
        # BUGS / JAGS have no scalar-to-vector broadcast: an
        # IRDeterministic with empty plate whose expression references
        # a plate-less free data input that ends up consumed inside a
        # non-empty-plate observe must be lifted into the consumer's
        # loop, with the references re-indexed by the loop variable.
        ir = push_scalar_dets_into_loops(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _BugsCtx(sb=sb, morphisms={}, defines={})
        self._cards = dict(ir.cards)
        # Populate decl_plates for every IRDataInput and IRSample /
        # IRObserve / IRDeterministic / IRMarginalize-latent.
        self._populate_decl_plates(ir, ctx)
        src_id = self._fresh(ctx, "src")
        sb.vertex(src_id, "source_file")
        mb_id = self._fresh(ctx, "mb")
        sb.vertex(mb_id, "model_block")
        sb.constraint(mb_id, "ptrace-0", "Tmodel")
        sb.edge(src_id, mb_id, "model_block")
        ctx.block_id = mb_id
        for node in ir.body:
            self._dispatch_bugs_node(ctx, node)
        render_computations_static(sb, ir, target=self.target, destination=mb_id)
        return sb.build()

    def _populate_decl_plates(self, ir: IRProgram, ctx: _BugsCtx) -> None:
        for inp in ir.inputs:
            ctx.decl_plates[inp.name] = inp.plate
        for node in self._all_nodes(ir):
            if isinstance(node, IRSample):
                ctx.decl_plates[node.name] = node.plate
            elif isinstance(node, IRObserve):
                ctx.decl_plates[node.name] = node.plate
            elif isinstance(node, IRDeterministic):
                ctx.decl_plates[node.name] = node.plate
            elif isinstance(node, IRMarginalize):
                ctx.decl_plates[node.latent] = node.plate

    def _all_nodes(self, ir: IRProgram) -> list[IRNode]:
        out: list[IRNode] = []
        for node in ir.body:
            self._collect_nodes(node, out)
        return out

    def _collect_nodes(self, node: IRNode, out: list[IRNode]) -> None:
        out.append(node)
        if isinstance(node, IRMarginalize):
            for inner in node.scope:
                self._collect_nodes(inner, out)

    # ------------------------------------------------------------------
    # Dispatch.
    # ------------------------------------------------------------------

    def _dispatch_bugs_node(self, ctx: _BugsCtx, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            # BUGS has no data block syntax; data inputs flow through
            # the caller's data list. No emission.
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            self._emit_sample_node(ctx, node, loop_suffix="")
            return
        if isinstance(node, IRObserve):
            self._emit_observe_node(ctx, node)
            return
        if isinstance(node, IRDeterministic):
            if isinstance(node.expr, LetExprFactor):
                self._emit_factor_deterministic_node(ctx, node)
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
            self._emit_deterministic_node(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score_node(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize_node(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_export(ctx, node.names)
            return
        if isinstance(node, IRCall):
            # A graph language relates variables; it has no statement
            # that runs a computation, and the plan has already inlined
            # every pure call it could.
            raise UnsupportedConstruct(
                f"qvr-{self.target}", [f"call:graph:{node.callee}"]
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    def _emit_export(self, ctx: _BugsCtx, names: tuple[str, ...]) -> None:
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
                    f"qvr-{self.target}",
                    [
                        f"return:unbound:{name}: the program returns a "
                        f"name no sample, observe, let, or data input "
                        f"binds, so the emit has no relation to alias"
                    ],
                )
            self._emit_deterministic_node(
                ctx,
                IRDeterministic(
                    name=f"{name}_value",
                    expr=LetExprVar(name=name),
                    constraint=CSReal(),
                    plate=plate,
                ),
            )

    # ------------------------------------------------------------------
    # Required RendererBase abstract methods (the dispatch points the
    # `Renderer` Protocol mandates). The BUGS renderer routes through
    # its `_emit_*_node` helpers internally; these forward.
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
        """No-op: BUGS has no declaration syntax distinct from `~` / data.

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
        """Emit a `~` statement with surrounding `for` loops.

        BUGS treats observed and latent draws identically at the
        syntactic level: the caller's data list pins the value of
        observed variables and the same `~` line carries the
        likelihood factor.

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
        if not isinstance(ctx, _BugsCtx):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["ctx:wrong-type"],
            )
        node = IRSample(
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
        )
        self._emit_sample_node(ctx, node, loop_suffix="", observed=observed)
        return ""

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Emit an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        as the collapsed `dcat` row its atoms sum to when the scope is
        a categorical mixture, and refuse it otherwise.

        QVR's `marginalize` denotes the integral over the latent of
        the measure the scope carries. A program that declares the
        latent instead denotes a measure on the product of the
        latent's support with the scope's, which differs from the
        integral by an amount that moves with the data, so it is no
        rendering of the block. BUGS has no statement that adds a
        free log-density term to the joint, which is what a general
        `logsumexp` reduction would need, so every scope the collapse
        does not write is refused.

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
        refuse_ungrouped_row_marginalize("qvr-bugs", node)
        if not isinstance(ctx, _BugsCtx):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["ctx:wrong-type"],
            )
        self._emit_marginalize_node(ctx, node)
        return ""

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Render a scalar-to-vector broadcast as a filled helper array.

        BUGS has neither scalar-to-vector coercion nor a repetition function, so
        the renderer emits an indexed fill loop and references its resulting
        slice. Helpers are reused for identical scalar and extent pairs.

        Only plate-free scalars can be hoisted outside enclosing plate loops.
        Values that depend on loop indices and broadcasts to ranks other than
        one raise `UnsupportedConstruct`.

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
        bctx = _as_bugs_ctx(ctx)
        if len(target_shape) != 1:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"arg:broadcast:rank-{len(target_shape)}: a BUGS "
                    "fill loop builds a 1-D vector only"
                ],
            )
        size = int(target_shape[0])
        key = self._broadcast_key(bctx, value, size)
        name = bctx.broadcast_helpers.get(key)
        if name is None:
            name = self._fresh_broadcast_name(bctx)
            bctx.broadcast_helpers[key] = name
            self._emit_broadcast_fill(bctx, name, value, size)
        return self._emit_indexed_name(bctx, name, (), (str(size),))

    def _broadcast_key(self, ctx: _BugsCtx, value: IRArg, size: int) -> tuple[str, int]:
        """Return the dedup key for a broadcast of `value` to `size`.

        Raises when `value` is not a plate-free scalar: an indexed or
        plated reference cannot be hoisted out of the consumer's loops
        without dropping the index it carries.
        """
        if isinstance(value, IRArgNumber):
            return (f"num:{value.value!r}", size)
        if isinstance(value, IRArgRef) and not value.indices:
            decl = ctx.decl_plates.get(value.name)
            if decl is not None and (decl.batch_dims or decl.event_dims):
                raise UnsupportedConstruct(
                    f"qvr-{self.target}",
                    [
                        f"arg:broadcast:plated-ref:{value.name}: only a "
                        "plate-free scalar can fill a helper vector"
                    ],
                )
            return (f"ref:{value.name}", size)
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [
                f"arg:broadcast:value:{type(value).__name__}: only a "
                "plate-free scalar reference or numeric literal can "
                "fill a helper vector"
            ],
        )

    def _fresh_broadcast_name(self, ctx: _BugsCtx) -> str:
        """Return an unused BUGS identifier for a broadcast helper."""
        index = len(ctx.broadcast_helpers)
        taken = set(ctx.decl_plates) | set(ctx.broadcast_helpers.values())
        while True:
            index += 1
            candidate = f"bcast_{index}"
            if candidate not in taken:
                return candidate

    def _emit_broadcast_fill(
        self, ctx: _BugsCtx, name: str, value: IRArg, size: int
    ) -> None:
        """Emit `for (i_<name> in 1:size) { <name>[i_<name>] <- value }`.

        The relation hangs off the model block rather than the caller's
        innermost loop body, and the enclosing-plate state is cleared
        for the duration so the scalar renders without picking up an
        index from a loop it does not live in.
        """
        loop_var = f"i_{name}"
        for_id = self._fresh(ctx, "for")
        ctx.sb.vertex(for_id, "for_loop")
        ctx.sb.edge(ctx.block_id, for_id, "for_loop")
        var_id = self._emit_bare_identifier(ctx, loop_var)
        ctx.sb.edge(for_id, var_id, "variable")
        rng_id = self._fresh(ctx, "rng")
        ctx.sb.vertex(rng_id, "range")
        ctx.sb.edge(for_id, rng_id, "range")
        lo_id = self._fresh(ctx, "num")
        ctx.sb.vertex(lo_id, "number")
        ctx.sb.constraint(lo_id, "literal-value", "1")
        ctx.sb.edge(rng_id, lo_id, "lower")
        hi_id = self._fresh(ctx, "num")
        self._emit_upper_literal(ctx, hi_id, str(size))
        ctx.sb.edge(rng_id, hi_id, "upper")
        body_id = self._fresh(ctx, "blk")
        ctx.sb.vertex(body_id, "block")
        ctx.sb.edge(for_id, body_id, "body")
        dr_id = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr_id, "deterministic_relation")
        ctx.sb.edge(body_id, dr_id, "deterministic_relation")
        lhs_id = self._emit_indexed_name(ctx, name, (_LoopRef(loop_var),), ())
        ctx.sb.edge(dr_id, lhs_id, "variable")
        prev_plate = ctx.enclosing_plate
        prev_loops = ctx.enclosing_loop_names
        prev_via = ctx.via
        ctx.enclosing_plate = None
        ctx.enclosing_loop_names = ()
        ctx.via = None
        try:
            rhs_id = self._emit_arg(ctx, value)
        finally:
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops
            ctx.via = prev_via
        ctx.sb.edge(dr_id, rhs_id, "value")

    # ------------------------------------------------------------------
    # List / matrix literal args: BUGS has no inline literal form.
    # ------------------------------------------------------------------

    def _render_list(self, arg: IRArgList) -> SchemaFragment:
        """Raise: BUGS does not parse `[a, b, c]` in arg position."""
        del arg
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["arg:list-literal"],
        )

    def _render_matrix(self, arg: IRArgMatrix) -> SchemaFragment:
        """Raise: BUGS does not parse `[[a, b], ...]` in arg position."""
        del arg
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["arg:matrix-literal"],
        )

    def _reject_list_args(self, ir: IRProgram) -> None:
        """Report every independent gap on the first literal argument.

        A call may be unsupported both because BUGS has no spelling for its
        family and because BUGS has no inline list or matrix literal syntax.
        Keeping both kinds prevents the order of validation passes from
        hiding either capability boundary.
        """
        for node in self._all_nodes(ir):
            args = getattr(node, "args", ())
            arg_kinds = [
                kind for argument in args for kind in self._literal_arg_kinds(argument)
            ]
            if not arg_kinds:
                continue
            family_kinds: list[str] = []
            family = getattr(node, "family", None)
            if isinstance(family, str) and not _is_wrapper_family_call(args):
                try:
                    self._lookup_family(family)
                except UnsupportedConstruct as exc:
                    family_kinds.extend(exc.kinds)
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [*family_kinds, *dict.fromkeys(arg_kinds)],
            )

    def _literal_arg_kinds(self, arg: IRArg) -> tuple[str, ...]:
        """Return the BUGS literal-syntax gaps nested inside `arg`."""
        if isinstance(arg, IRArgList):
            return ("arg:list-literal",)
        if isinstance(arg, IRArgMatrix):
            return ("arg:matrix-literal",)
        if isinstance(arg, IRArgBroadcast):
            return self._literal_arg_kinds(arg.value)
        if isinstance(arg, IRArgRef):
            return tuple(
                kind for index in arg.indices for kind in self._literal_arg_kinds(index)
            )
        return ()

    # ------------------------------------------------------------------
    # Family lookup.
    # ------------------------------------------------------------------

    def _lookup_family(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:{family}: not in FAMILY_META"],
            )
        if "bugs" not in meta.target_names:
            # A family with no BUGS spelling is reachable only by
            # writing its density out, which needs a statement the
            # language does not have.
            # `_NO_BUGS_FREE_DENSITY_TERM` records what that costs for
            # each family the gallery reaches, so the raise names the
            # limitation rather than only the missing name.
            blocked = _NO_BUGS_FREE_DENSITY_TERM.get(family)
            if blocked is not None:
                raise UnsupportedConstruct(
                    f"qvr-{self.target}",
                    [f"family:{family}:no-free-density-term: {blocked}"],
                )
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:{family}: no BUGS target name"],
            )
        # A registry entry is a spelling, not a guarantee that an
        # engine implements the name. `_NO_BUGS_DISTRIBUTION` records
        # the families whose recorded spelling resolves nowhere, so
        # the boundary shows up as a raise here rather than as source
        # that parses and then fails to compile.
        unavailable = _NO_BUGS_DISTRIBUTION.get(family)
        if unavailable is not None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:{family}:no-bugs-distribution: {unavailable}"],
            )
        return meta

    # ------------------------------------------------------------------
    # Sample / observe / marginalize emission.
    # ------------------------------------------------------------------

    def _emit_sample_node(
        self,
        ctx: _BugsCtx,
        node: IRSample,
        *,
        loop_suffix: str,
        observed: bool = False,
    ) -> None:
        """Render an [`IRSample`][quivers.transpile.ir.IRSample] node.

        A first-arg [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        signals a wrapper-family encoding (`Truncated`, `Mixture`,
        ...) whose runtime distribution comes from the referenced
        morphism's `init_family`. The BUGS renderer handles the
        Truncated case as the `T(lower, upper)` idiom; other
        wrappers route through the generic path (which will fail
        downstream if the wrapper has no BUGS encoding).
        """
        if _is_wrapper_family_call(node.args):
            self._emit_truncated(ctx, node)
            return
        if node.family == "TruncatedNormal":
            self._emit_truncated_normal_native(ctx, node, loop_suffix)
            return
        if is_class_code_family(node.family) and not observed:
            # `dcat` draws a class counting from one; the program reads
            # the draw as QVR's zero-based code.
            code = f"{node.name}_code"
            while code in ctx.decl_plates:
                code = f"{code}_"
            ctx.decl_plates[code] = node.plate
            self._emit_relation(
                ctx,
                name=code,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                via=None,
                loop_suffix=loop_suffix,
                observed=observed,
            )
            self._emit_deterministic_node(
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
            return
        self._emit_relation(
            ctx,
            name=node.name,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=node.plate,
            via=None,
            loop_suffix=loop_suffix,
            observed=observed,
        )

    def _emit_truncated_normal_native(
        self,
        ctx: _BugsCtx,
        node: IRSample,
        loop_suffix: str,
    ) -> None:
        """Emit ``<lhs> ~ dnorm(loc, tau) T(low, high)`` for
        ``TruncatedNormal(loc, scale, low, high)``.

        Splits the 4-arg call into (loc, scale) for the base ``dnorm``
        plus (low, high) for the truncation suffix; the family's
        ``arg_aliases[bugs]`` renames ``scale -> tau`` and the alias
        pipeline applies the ``inv_square`` transform so the emitted
        precision is ``1/(scale*scale)``.
        """
        if len(node.args) != 4:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"family:TruncatedNormal: expected 4 args "
                    f"(loc, scale, low, high), got {len(node.args)}"
                ],
            )
        family_args = node.args[:2]
        family_arg_names = node.arg_names[:2] if node.arg_names else ("loc", "scale")
        lo, hi = node.args[2], node.args[3]
        self._emit_relation(
            ctx,
            name=node.name,
            family="TruncatedNormal",
            args=family_args,
            arg_names=family_arg_names,
            plate=node.plate,
            via=None,
            loop_suffix=loop_suffix,
            truncation=(lo, hi),
        )

    def _emit_observe_node(self, ctx: _BugsCtx, node: IRObserve) -> None:
        """Render an [`IRObserve`][quivers.transpile.ir.IRObserve] node.

        Observes carry an optional `via` fibration: when present,
        index-references in the args whose target sits on a
        different plate are rewritten as `<name>[via[loop]]`. This
        is the BUGS idiom for "observation `n` maps to latent group
        `via[n]`".

        Raises
        ------
        UnsupportedConstruct
            If the observation is a class code; see
            `_refuse_observed_class_code`.
        """
        _refuse_observed_class_code(self.target, node)
        if _is_wrapper_family_call(node.args):
            self._emit_truncated(
                ctx,
                IRSample(
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    constraint=node.constraint,
                    plate=node.plate,
                ),
            )
            return
        if node.family == "TruncatedNormal":
            self._emit_truncated_normal_native(
                ctx,
                IRSample(
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    constraint=node.constraint,
                    plate=node.plate,
                ),
                "",
            )
            return
        self._emit_relation(
            ctx,
            name=node.name,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=node.plate,
            via=node.via,
            loop_suffix="",
            observed=True,
        )

    def _emit_marginalize_node(self, ctx: _BugsCtx, node: IRMarginalize) -> None:
        """Emit a BUGS marginalization.

        A categorical mixture recognized by `categorical_mixture` is
        collapsed to a `dcat` row, one per observed cell. A block whose
        rows fibre into its groups keys its accumulator by group, which
        the per-cell collapse does not write, and a scope that is no
        categorical mixture has no collapse at all; both are refused,
        because BUGS cannot add a free log-density term without a
        data-bound zeros-trick carrier.

        Parameters
        ----------
        ctx : _BugsCtx
            The render context.
        node : IRMarginalize
            The block.

        Raises
        ------
        UnsupportedConstruct
            If the block's rows fibre into its groups under a shared
            prior, the scope is no categorical mixture, or the scope
            observes a class code (see `_refuse_observed_class_code`).
        """
        refuse_ungrouped_row_marginalize("qvr-bugs", node)
        for inner in node.scope:
            if isinstance(inner, IRObserve):
                _refuse_observed_class_code(self.target, inner)
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        fibration = marginalize_fibration(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=ctx.decl_plates,
            target=self.target,
        )
        if fibration is not None:
            via, group = fibration
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"marginalize:grouped-fibration:{node.latent}: the rows "
                    f"`{via}` sends to each `{group.name}` are summed before "
                    f"the reduction over the atoms, and BUGS has no statement "
                    f"that adds the per-group log-density to the joint"
                ],
            )
        mixture = categorical_mixture(node, ctx.decl_plates)
        if mixture is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"marginalize:no-collapse:{node.latent}: the scope is no "
                    f"categorical mixture, and BUGS has no statement that "
                    f"adds its integrated log-density to the joint"
                ],
            )
        self._emit_collapsed_mixture(ctx, node, mixture)

    def _emit_collapsed_mixture(
        self,
        ctx: _BugsCtx,
        node: IRMarginalize,
        mixture: CategoricalMixture,
    ) -> None:
        """Emit a marginalized categorical latent as a collapsed `dcat` row.

        Under the observation plate, the renderer computes the weighted
        categorical row, indexes that row at the observed outcome for the
        exported density, and scores the outcome with `dcat`. BUGS uses
        one-based categorical outcomes as probability-vector indices. The
        density relation remains separate from the draw to avoid a dependency
        cycle through the observed value.
        """
        observe = mixture.observe
        name = self._fresh_mixture_name(ctx, node.latent)
        loop_names = self._loop_names(observe.plate, "")
        self._emit_mixture_rows(ctx, name, mixture, loop_names)
        ctx.decl_plates[name] = Plate(
            event_dims=(),
            batch_dims=(mixture.alphabet_dim, *observe.plate.batch_dims),
        )
        self._emit_observed_density(ctx, name, node.latent, mixture, loop_names)
        self._emit_collapsed_observe(ctx, name, mixture, loop_names)

    def _emit_mixture_rows(
        self,
        ctx: _BugsCtx,
        name: str,
        mixture: CategoricalMixture,
        loop_names: tuple[str, ...],
    ) -> None:
        """Emit the deterministic relations carrying the collapsed
        mixture row, one row per cell of the observation's plate.

        The relation sits under one loop per batch axis the observation
        replicates over plus one innermost loop over the alphabet axis,
        whose variable indexes the column of the row matrix the
        `inprod` contracts against the weights. The weights render
        through the ordinary reference path in the observation's own
        loop and `via` context, so a weight row declared on a coarser
        axis is gathered by the fibration exactly as it would be at the
        observation itself.
        """
        observe = mixture.observe
        column_loop = f"i_{name}"
        body_id = self._open_loops(ctx, ctx.block_id, observe.plate, loop_names)
        body_id = self._open_loops(
            ctx,
            body_id,
            Plate(event_dims=(), batch_dims=(mixture.alphabet_dim,)),
            (column_loop,),
        )
        dr_id = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr_id, "deterministic_relation")
        ctx.sb.edge(body_id, dr_id, "deterministic_relation")
        lhs_id = self._emit_indexed_name(
            ctx,
            name,
            tuple(_LoopRef(n) for n in (column_loop, *loop_names)),
            (),
        )
        ctx.sb.edge(dr_id, lhs_id, "variable")
        prev_plate = ctx.enclosing_plate
        prev_loops = ctx.enclosing_loop_names
        prev_via = ctx.via
        ctx.enclosing_plate = observe.plate
        ctx.enclosing_loop_names = loop_names
        ctx.via = observe.via
        try:
            weights_id = self._emit_arg(ctx, mixture.weights)
        finally:
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops
            ctx.via = prev_via
        rows_id = self._emit_column_slice(
            ctx,
            mixture.rows_name,
            self._dim_upper_text(mixture.atom_dim),
            column_loop,
        )
        ctx.sb.edge(dr_id, self._emit_inprod(ctx, weights_id, rows_id), "value")

    def _emit_observed_density(
        self,
        ctx: _BugsCtx,
        name: str,
        latent: str,
        mixture: CategoricalMixture,
        loop_names: tuple[str, ...],
    ) -> None:
        """Emit `<dens>[<cell>] <- <mix>[<y>[<cell>], <cell>]`, the
        integrated density at the symbol each observed cell carries.

        This is the quantity the `marginalize` reduction denotes,
        evaluated where the data lands, and it is the relation that
        writes the observation into subscript position, which is how a
        BUGS model states that an integer input is a 1-based index.
        """
        observe = mixture.observe
        dens_name = self._fresh_density_name(ctx, latent)
        dr_id = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr_id, "deterministic_relation")
        body_id = self._open_loops(ctx, ctx.block_id, observe.plate, loop_names)
        ctx.sb.edge(body_id, dr_id, "deterministic_relation")
        lhs_id = self._emit_lhs(ctx, dens_name, observe.plate, loop_names)
        ctx.sb.edge(dr_id, lhs_id, "variable")
        symbol = IRArgRef(
            name=observe.name,
            indices=tuple(IRArgRef(name=n) for n in loop_names),
        )
        rhs_id = self._emit_indexed_name(
            ctx,
            name,
            (symbol, *(_LoopRef(n) for n in loop_names)),
            (),
        )
        ctx.sb.edge(dr_id, rhs_id, "value")
        ctx.decl_plates[dens_name] = observe.plate

    def _emit_collapsed_observe(
        self,
        ctx: _BugsCtx,
        name: str,
        mixture: CategoricalMixture,
        loop_names: tuple[str, ...],
    ) -> None:
        """Emit `<y>[<cell>] ~ dcat(<mix>[1:V, <cell>])`.

        The row is an alphabet-major slice of the mixture array, so
        the argument is the whole alphabet's measure at that cell and
        the scored entry is the one the observed symbol subscripts.
        """
        observe = mixture.observe
        body_id = self._open_loops(ctx, ctx.block_id, observe.plate, loop_names)
        sr_id = self._fresh(ctx, "sr")
        ctx.sb.vertex(sr_id, "stochastic_relation")
        ctx.sb.edge(body_id, sr_id, "stochastic_relation")
        lhs_id = self._emit_lhs(ctx, observe.name, observe.plate, loop_names)
        ctx.sb.edge(sr_id, lhs_id, "variable")
        row_id = self._emit_row_slice(
            ctx,
            name,
            self._dim_upper_text(mixture.alphabet_dim),
            loop_names,
        )
        dc_id = self._fresh(ctx, "dc")
        ctx.sb.vertex(dc_id, "distribution_call")
        name_id = self._fresh(ctx, "id")
        ctx.sb.vertex(name_id, "identifier")
        ctx.sb.constraint(
            name_id,
            "literal-value",
            self._lookup_family(observe.family).target_names[self.target],
        )
        ctx.sb.edge(dc_id, name_id, "name")
        al_id = self._fresh(ctx, "al")
        ctx.sb.vertex(al_id, "argument_list")
        ctx.sb.edge(dc_id, al_id, "arguments")
        ctx.sb.edge(al_id, row_id, "indexed_variable")
        ctx.sb.edge(sr_id, dc_id, "distribution")

    def _emit_row_slice(
        self,
        ctx: _BugsCtx,
        name: str,
        upper: str,
        loop_names: tuple[str, ...],
    ) -> str:
        """Emit `name[1:upper, <loop_names>]`, the alphabet-major row
        of the collapsed mixture at one cell of the observation."""
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        ctx.sb.edge(iv_id, self._emit_bare_identifier(ctx, name), "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        self._emit_one_to(ctx, il_id, upper)
        for loop_name in loop_names:
            ctx.sb.edge(il_id, self._emit_bare_identifier(ctx, loop_name), "identifier")
        return iv_id

    def _fresh_density_name(self, ctx: _BugsCtx, latent: str) -> str:
        """Return an unused BUGS identifier for the integrated density
        of `latent` at the observed symbol."""
        taken = set(ctx.decl_plates) | set(ctx.broadcast_helpers.values())
        candidate = f"dens_{latent}"
        index = 1
        while candidate in taken:
            index += 1
            candidate = f"dens_{latent}_{index}"
        return candidate

    def _fresh_mixture_name(self, ctx: _BugsCtx, latent: str) -> str:
        """Return an unused BUGS identifier for the collapsed mixture
        row of `latent`.

        The name leads with a letter, which is what the BUGS / JAGS
        lexer requires of an identifier, and carries the latent's own
        name so the emitted source reads as the integral of that
        latent rather than as an anonymous helper.
        """
        taken = set(ctx.decl_plates) | set(ctx.broadcast_helpers.values())
        candidate = f"marg_{latent}"
        index = 1
        while candidate in taken:
            index += 1
            candidate = f"marg_{latent}_{index}"
        return candidate

    def _emit_column_slice(
        self,
        ctx: _BugsCtx,
        name: str,
        upper: str,
        column_loop: str,
    ) -> str:
        """Emit `name[1:upper, <column_loop>]`, the atom-indexed column
        of a row-stochastic matrix."""
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        ctx.sb.edge(iv_id, self._emit_bare_identifier(ctx, name), "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        self._emit_one_to(ctx, il_id, upper)
        ctx.sb.edge(il_id, self._emit_bare_identifier(ctx, column_loop), "identifier")
        return iv_id

    def _emit_inprod(self, ctx: _BugsCtx, left: str, right: str) -> str:
        """Emit `inprod(<left>, <right>)`.

        Both operands are indexed variables:
        `categorical_mixture`
        admits only weights declared over an event axis and a row
        matrix declared over a batch and an event axis, so each side
        carries at least one index.
        """
        call_id = self._fresh(ctx, "call")
        ctx.sb.vertex(call_id, "function_call")
        ctx.sb.edge(call_id, self._emit_bare_identifier(ctx, "inprod"), "name")
        al_id = self._fresh(ctx, "al")
        ctx.sb.vertex(al_id, "argument_list")
        ctx.sb.edge(call_id, al_id, "arguments")
        ctx.sb.edge(al_id, left, "indexed_variable")
        ctx.sb.edge(al_id, right, "indexed_variable")
        return call_id

    def _emit_truncated(self, ctx: _BugsCtx, node: IRSample) -> None:
        """Emit `<lhs> ~ d<base>(base_args) T(lo, hi)`."""
        family_ref = node.args[0]
        if not isinstance(family_ref, IRArgFamilyRef):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["truncated:expected IRArgFamilyRef at arg 0"],
            )
        decl = ctx.morphisms.get(family_ref.name)
        if not isinstance(decl, MorphismDecl):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"truncated:base:{family_ref.name}: morphism not bound"],
            )
        init = decl.init
        if not isinstance(init, MorphismInitFamily):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"truncated:base:{family_ref.name}: init is not `~ Family(args)`"],
            )
        base_family = init.family
        bound = bind_family_arguments(base_family, init.args or ())
        base_args = tuple(_draw_arg_to_ir(value) for _, value in bound)
        base_arg_names = tuple(name for name, _ in bound)
        if len(node.args) < 3:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["truncated:missing-bounds"],
            )
        lo, hi = node.args[1], node.args[2]
        self._emit_relation(
            ctx,
            name=node.name,
            family=base_family,
            args=base_args,
            arg_names=base_arg_names,
            plate=node.plate,
            via=None,
            loop_suffix="",
            truncation=(lo, hi),
        )

    def _emit_gp_block(
        self,
        ctx: _BugsCtx,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as BUGS deterministic loops
        plus a multivariate-normal stochastic relation.

        BUGS `dmnorm` parameterises by precision (the inverse of the
        covariance), so the RBF kernel matrix is constructed in a
        double loop and then inverted before being passed as the
        precision argument:

            for (i in 1:N) {
              __gp_zeros_<name>[i] <- 0
              for (j in 1:N) {
                __gp_K_<name>[i, j] <-
                    exp(-0.5 * pow(x[i] - x[j], 2)
                              / pow(length_scale, 2))
                    + equals(i, j) * jitter
              }
            }
            __gp_tau_<name> <- inverse(__gp_K_<name>)
            <name> ~ dmnorm(__gp_zeros_<name>, __gp_tau_<name>)

        BUGS lacks `ifelse`, so the jitter is added via the
        ``equals(i, j) * jitter`` idiom (equals returns 1 if true,
        0 otherwise).
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        # BUGS rejects identifiers starting with underscore; use a
        # `gp_` prefix instead of `__gp_` for the synthesized names.
        kmat_name = f"gp_K_{node.name}"
        zeros_name = f"gp_zeros_{node.name}"
        tau_name = f"gp_tau_{node.name}"
        # Synthesize an IRDeterministic for the zeros: K_zeros[i] <- 0
        # with batch_dim "i" (so the existing emit produces the outer
        # for(i) loop and the LHS index).
        i_dim = DimStatic(size=n, name="i")
        j_dim = DimStatic(size=n, name="j")
        zeros_det = IRDeterministic(
            name=zeros_name,
            expr=LetExprLiteral(value=0.0),
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=(i_dim,)),
        )
        self._emit_deterministic_node(ctx, zeros_det)
        # Synthesize an IRDeterministic for K[i, j] <- exp(...) + ...
        # with batch_dims (i, j). The let-expr references `m_i`,
        # `m_j` as loop variables; index_letexpr_refs would rewrite
        # IRArgRef-ish refs to plate-bound names, but our expression
        # uses raw LetExprVar("m_i") / LetExprVar("m_j") and
        # LetExprIndex on `x` directly so the rewrite step is a no-op.
        i_var = LetExprVar(name="m_i")
        j_var = LetExprVar(name="m_j")
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
        neg_half = LetExprUnaryOp(operand=LetExprLiteral(value=0.5))
        scaled = LetExprBinOp(
            op="/",
            left=LetExprBinOp(op="*", left=neg_half, right=diff_sq),
            right=ls_sq,
        )
        exp_call = LetExprCall(func="exp", args=(scaled,))
        # BUGS lacks `ifelse`; use `equals(i, j) * jitter` (equals
        # returns 1 if its args are equal, 0 otherwise).
        eq_call = LetExprCall(func="equals", args=(i_var, j_var))
        jitter_term = LetExprBinOp(
            op="*",
            left=eq_call,
            right=LetExprLiteral(value=jitter),
        )
        kij_expr = LetExprBinOp(
            op="+",
            left=exp_call,
            right=jitter_term,
        )
        kij_det = IRDeterministic(
            name=kmat_name,
            expr=kij_expr,
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=(i_dim, j_dim)),
        )
        self._emit_deterministic_node(ctx, kij_det)
        # __gp_tau_<name> <- inverse(__gp_K_<name>)
        tau_det = IRDeterministic(
            name=tau_name,
            expr=LetExprCall(
                func="inverse",
                args=(LetExprVar(name=kmat_name),),
            ),
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=()),
        )
        self._emit_deterministic_node(ctx, tau_det)
        # <name> ~ dmnorm(__gp_zeros_<name>, __gp_tau_<name>)
        # Build a sample IR node for MultivariateNormal whose args
        # reference the bound names; the existing _emit_sample_node
        # handles emission via the family's BUGS target_name (dmnorm).
        mvn_sample = IRSample(
            name=node.name,
            family="MultivariateNormal",
            args=(
                IRArgRef(name=zeros_name),
                IRArgRef(name=tau_name),
            ),
            arg_names=("loc", "precision_matrix"),
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=()),
        )
        self._emit_sample_node(ctx, mvn_sample, loop_suffix="")

    def _emit_factor_deterministic_node(
        self, ctx: _BugsCtx, node: IRDeterministic
    ) -> None:
        """Emit a `factor` binding as one relation per cell.

        A rank-`n` factor denotes a rank-`n` tensor of scalar cells.
        BUGS has no array literal beyond the flat `c(...)` combine and
        no reshape to give one a rank, so the tensor is written out
        cell by cell: `<name>[i_1, ..., i_n] <- <body>` with every
        binder substituted by its coordinate. The cells are already
        enumerated, so no surrounding loop is emitted.
        """
        expr = node.expr
        if not isinstance(expr, LetExprFactor):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"let-expr:factor:expected-factor:{node.name}"],
            )
        let_ctx = _BugsLetCtx(
            ctx.sb,
            lambda p: self._fresh(ctx, p),
            self._cards,
            self.target,
        )
        self._check_factor_plate(node, factor_axis_sizes(let_ctx, expr))
        self._emit_cells(ctx, node, factor_cells(let_ctx, expr))

    def _emit_cells(
        self,
        ctx: _BugsCtx,
        node: IRDeterministic,
        cells: Iterable[tuple[tuple[int, ...], LetExprNode]],
    ) -> None:
        """Emit one relation per enumerated cell of a tensor binding.

        A plated list literal is written out the way a factor is:
        `<name>[i_1, ..., i_n] <- <item>`, the nesting of the literal
        supplying the coordinates.

        Parameters
        ----------
        ctx : _BugsCtx
            The render context.
        node : IRDeterministic
            The binding the cells belong to.
        cells : Iterable[tuple[tuple[int, ...], LetExprNode]]
            The cells' zero-based coordinates and scalar expressions.
        """
        let_ctx = _BugsLetCtx(
            ctx.sb,
            lambda p: self._fresh(ctx, p),
            self._cards,
            self.target,
        )
        for indices, body in cells:
            dr_id = self._fresh(ctx, "dr")
            ctx.sb.vertex(dr_id, "deterministic_relation")
            ctx.sb.edge(ctx.block_id, dr_id, "deterministic_relation")
            lhs_id = self._emit_indexed_from_pieces(
                ctx,
                node.name,
                tuple(_IndexPiece.number(str(value + 1)) for value in indices),
                (),
            )
            ctx.sb.edge(dr_id, lhs_id, "variable")
            ctx.sb.edge(
                dr_id,
                render_let_expr_bugs(
                    let_ctx,
                    body,
                    decl_plates=ctx.decl_plates,
                ),
                "value",
            )

    def _check_factor_plate(
        self, node: IRDeterministic, sizes: tuple[int, ...]
    ) -> None:
        """Assert the factor's binder axes are exactly the binding's
        plate, so the per-cell subscripts address the whole node."""
        if node.plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
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
                f"qvr-{self.target}",
                [
                    f"let-expr:LetExprFactor:{node.name}: binder axes "
                    f"{sizes} do not match the binding's declared "
                    f"plate {declared}"
                ],
            )

    def _emit_deterministic_node(self, ctx: _BugsCtx, node: IRDeterministic) -> None:
        """Emit a BUGS deterministic relation ``<name> <- <expr>``.

        Wraps the relation in `for (m_<axis> in 1:N_<axis>)` loops
        when the node's plate carries batch dims (one loop per
        dim, BUGS-style nesting); the LHS is then indexed by each
        loop variable so the relation populates one element per
        iteration. The RHS expression goes through
        `render_let_expr_bugs`,
        which lowers `LetExpr*` nodes to BUGS expression vertices
        (`binary_expression`, `function_call`, `indexed_variable`,
        ...).
        """
        loop_names = self._loop_names(node.plate, "")
        body_id = self._open_loops(ctx, ctx.block_id, node.plate, loop_names)
        prev_plate = ctx.enclosing_plate
        prev_loops = ctx.enclosing_loop_names
        ctx.enclosing_plate = node.plate
        ctx.enclosing_loop_names = loop_names
        try:
            dr_id = self._fresh(ctx, "dr")
            ctx.sb.vertex(dr_id, "deterministic_relation")
            ctx.sb.edge(body_id, dr_id, "deterministic_relation")
            lhs_id = self._emit_lhs(ctx, node.name, node.plate, loop_names)
            ctx.sb.edge(dr_id, lhs_id, "variable")
            # Rewrite the expression so any var reference whose
            # declared plate axes match the surrounding loop axes
            # emits as `name[loop_var, ...]`. Required when an empty-
            # plate det was lifted into a non-empty plate by the
            # `_push_scalar_dets_into_loops` pre-pass: the lifted
            # det's expression still names a free data input
            # (`x_design`) that the lift retagged with the new plate.
            expr = index_letexpr_refs(
                node.expr, ctx.decl_plates, node.plate, loop_names
            )
            let_ctx = _BugsLetCtx(
                ctx.sb,
                lambda p: self._fresh(ctx, p),
                self._cards,
                self.target,
            )
            rhs_id = render_let_expr_bugs(
                let_ctx,
                expr,
                decl_plates=ctx.decl_plates,
                row_index=loop_names[0] if loop_names else None,
            )
            ctx.sb.edge(dr_id, rhs_id, "value")
        finally:
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops

    def _emit_score_node(self, ctx: _BugsCtx, node: IRScore) -> None:
        """Emit ``score <name> = <expr>`` via the BUGS zeros trick.

        BUGS has no native ``target +=`` statement; the canonical idiom
        for adding ``<expr>`` to the joint log-likelihood is the
        zeros trick, which exploits ``log P(0; lambda) = -lambda`` for
        the Poisson distribution. Concretely the renderer emits:

        ```
        C_<name> <- 1.0e6 - (<expr>)
        zero_<name> ~ dpois(C_<name>)
        ```

        The host supplies ``zero_<name> = 0`` in the data list so the
        stochastic relation contributes ``-(1.0e6 - <expr>) = <expr> -
        1.0e6`` to the log-density; the additive constant ``-1.0e6``
        absorbs into the normalising constant and does not affect
        posterior inference.

        The offset ``1.0e6`` must stay strictly larger than ``<expr>``
        over the entire parameter support so the Poisson rate argument
        remains positive; ``1.0e6`` is the canonical safe default for
        typical BUGS / JAGS fixtures.

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
        # IRScore carries no plate, so the relation lives at the model
        # block's top level with a bare-identifier LHS.
        dr_id = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr_id, "deterministic_relation")
        ctx.sb.edge(ctx.block_id, dr_id, "deterministic_relation")
        lhs_id = self._emit_bare_identifier(ctx, c_name)
        ctx.sb.edge(dr_id, lhs_id, "variable")
        # Build the RHS as `1.0e6 - (<expr>)`: an outer
        # `binary_expression` with `field:operator = -`, left child a
        # number for the offset, right child a `parenthesized_expression`
        # wrapping the score expression. The parens force right-side
        # grouping when the score expression itself contains a `+` / `-`
        # operator.
        offset_id = self._emit_number(ctx, _ZEROS_TRICK_OFFSET)
        # Render the inner score expression through the standard
        # let-expression pipeline.
        let_ctx = _BugsLetCtx(
            ctx.sb,
            lambda p: self._fresh(ctx, p),
            self._cards,
            self.target,
        )
        inner_expr_id = render_let_expr_bugs(
            let_ctx,
            node.expr,
            decl_plates=ctx.decl_plates,
        )
        paren_id = self._fresh(ctx, "par")
        ctx.sb.vertex(paren_id, "parenthesized_expression")
        ctx.sb.edge(paren_id, inner_expr_id, "parenthesized_expression")
        sub_id = self._fresh(ctx, "be")
        ctx.sb.vertex(sub_id, "binary_expression")
        ctx.sb.constraint(sub_id, "field:operator", "-")
        ctx.sb.constraint(sub_id, "chose-alt-fingerprint", "-")
        ctx.sb.edge(sub_id, offset_id, "left")
        ctx.sb.edge(sub_id, paren_id, "right")
        ctx.sb.edge(dr_id, sub_id, "value")
        # Record the carrier's plate so the subsequent `dpois(C_<name>)`
        # call renders `C_<name>` as a bare identifier (no auto-leading
        # indices). `_emit_relation`'s ref-emission path consults
        # `decl_plates` for the rate-arg IRArgRef.
        ctx.decl_plates[c_name] = empty_plate
        ctx.decl_plates[zero_name] = empty_plate
        # Emit the stochastic relation `zero_<name> ~ dpois(C_<name>)`.
        # The host supplies `zero_<name> = 0` in the data list; BUGS
        # has no in-model data declaration so the model source itself
        # carries no `zero_<name> <- 0` line.
        self._emit_relation(
            ctx,
            name=zero_name,
            family="Poisson",
            args=(IRArgRef(name=c_name),),
            arg_names=("rate",),
            plate=empty_plate,
            via=None,
            loop_suffix="",
        )

    # ------------------------------------------------------------------
    # Core relation emitter.
    # ------------------------------------------------------------------

    def _inject_bugs_specific_args(
        self,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """Prepend the canonical zero-location argument for QVR
        families whose torch distribution carries fewer parameters
        than the BUGS distribution they map to.

        ``HalfNormal(scale)`` maps to BUGS' ``dnorm(0, tau)``; this
        helper prepends an ``IRArgNumber(0)`` plus the parallel
        ``"loc"`` arg-name entry so the alias-transform pipeline
        still rewrites the scale into ``tau = 1/(scale*scale)``.
        ``Cauchy(loc, scale)`` and ``HalfCauchy(scale)`` map to BUGS'
        ``dt(mu, tau, k)`` (Student-t parameterised by precision and
        degrees of freedom); this helper appends ``IRArgNumber(1)``
        under the ``"df"`` arg-name entry so the emitted call is
        ``dt(mu, tau, 1)``.
        """
        if family in _PREPEND_ZERO:
            args = (IRArgNumber(value=0.0), *args)
            arg_names = ("loc", *arg_names)
        if family in _APPEND_DF_ONE:
            args = (*args, IRArgNumber(value=1.0))
            arg_names = (*arg_names, "df")
        return args, arg_names

    def _dmnorm_precision_args(
        self,
        ctx: _BugsCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """Rewrite a ``MultivariateNormal`` site into ``dmnorm``'s
        precision parameterisation.

        ``dmnorm(mu, Omega)`` reads its second argument as the
        precision, the inverse of the covariance: with
        ``Omega = [[2, .5], [.5, 1.5]]``, ``mu = (.1, .3)`` and
        ``x = (.4, -.2)`` the engine scores -1.534577, which is
        ``torch.distributions.MultivariateNormal(mu,
        precision_matrix=Omega).log_prob(x)`` and not the
        ``covariance_matrix=Omega`` reading (-2.486405). Emitting the
        QVR ``covariance_matrix`` slot straight into that position
        thus scores a different Gaussian at every point.

        A site that already names ``precision_matrix`` passes through
        untouched, which is the door the GP block comes through: it
        inverts its own kernel matrix and hands the result over
        already in precision form. A site naming ``covariance_matrix``
        gains a ``prec_<name> <- inverse(<covariance>)`` relation
        ahead of the draw. A ``scale_tril`` site raises: ``dmnorm``
        has no Cholesky slot, and rebuilding the precision from the
        factor would need a matrix product this renderer has no
        emission for.
        """
        if family != "MultivariateNormal":
            return args, arg_names
        by_name = dict(zip(arg_names, args, strict=True))
        if "loc" not in by_name:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
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
                f"qvr-{self.target}",
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
                f"qvr-{self.target}",
                [
                    f"family:MultivariateNormal:non-ref-covariance: "
                    f"the `inverse(...)` relation that turns the "
                    f"covariance into `dmnorm`'s precision needs a "
                    f"whole named matrix, and this slot carries a "
                    f"{type(covariance).__name__}"
                ],
            )
        precision_name = f"prec_{name}"
        self._emit_deterministic_node(
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

    def _emit_relation(
        self,
        ctx: _BugsCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        via: str | None,
        loop_suffix: str,
        truncation: tuple[IRArg, ...] | None = None,
        observed: bool = False,
    ) -> None:
        """Open the for-loops over `plate.batch_dims`, then emit the
        `<lhs> ~ <dist>(args)` line."""
        if truncation is None:
            truncation = half_support_truncation(family, observed=observed)
        meta = self._lookup_family(family)
        args, arg_names = self._inject_bugs_specific_args(family, args, arg_names)
        args, arg_names = self._dmnorm_precision_args(
            ctx, name=name, family=family, args=args, arg_names=arg_names
        )
        # Split the site's event dims into the family's own event
        # shape and the residual axes that merely replicate it. BUGS
        # has no vector form for a scalar family, so each residual
        # axis becomes an extra innermost loop rather than a slice on
        # the left-hand side.
        native_event, residual_event = split_event_dims(
            plate.event_dims, meta.event_rank
        )
        loop_plate = Plate(
            event_dims=native_event,
            batch_dims=(*plate.batch_dims, *residual_event),
        )
        loop_names = self._loop_names(loop_plate, loop_suffix)
        body_id = self._open_loops(ctx, ctx.block_id, loop_plate, loop_names)
        plate = loop_plate
        # Stash enclosing-plate + via for arg emission.
        prev_via = ctx.via
        prev_plate = ctx.enclosing_plate
        prev_loops = ctx.enclosing_loop_names
        ctx.via = via
        ctx.enclosing_plate = plate
        ctx.enclosing_loop_names = loop_names
        try:
            sr_id = self._fresh(ctx, "sr")
            ctx.sb.vertex(sr_id, "stochastic_relation")
            ctx.sb.edge(body_id, sr_id, "stochastic_relation")
            # When a truncation suffix is present, the
            # stochastic_relation's child-kind alternative needs to
            # advertise it so the panproto pretty-printer picks the
            # `~ ... T( , )` alternative rather than the no-suffix
            # alternative. Both the `jags` and the `bugs` backend
            # emit the JAGS `T( , )` spelling.
            if truncation is not None:
                lhs_kind = (
                    "identifier"
                    if not loop_names and not plate.event_dims
                    else "indexed_variable"
                )
                ctx.sb.constraint(sr_id, "chose-alt-fingerprint", "~")
                ctx.sb.constraint(
                    sr_id,
                    "chose-alt-child-kinds",
                    f"{lhs_kind} distribution_call truncation",
                )
            lhs_id = self._emit_lhs(ctx, name, plate, loop_names)
            ctx.sb.edge(sr_id, lhs_id, "variable")
            dc_id = self._emit_distribution_call(ctx, meta, args, arg_names)
            ctx.sb.edge(sr_id, dc_id, "distribution")
            if truncation is not None:
                trunc_id = self._emit_truncation(ctx, truncation)
                ctx.sb.edge(sr_id, trunc_id, "child_of")
        finally:
            ctx.via = prev_via
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops

    # ------------------------------------------------------------------
    # Loop emission.
    # ------------------------------------------------------------------

    def _loop_names(self, plate: Plate, suffix: str) -> tuple[str, ...]:
        """Return the loop-variable name for each `plate.batch_dim`.

        A residual event axis lifted into the batch list can repeat an
        axis the site already iterates (a square row-stochastic matrix
        names the same object on both sides), and two `for` loops over
        the same variable name would silently alias, so a repeat gets
        a numeric suffix.
        """
        used: set[str] = set()
        out: list[str] = []
        for dim in plate.batch_dims:
            base = f"m_{dim.name}{suffix}"
            candidate = base
            index = 1
            while candidate in used:
                index += 1
                candidate = f"{base}_{index}"
            used.add(candidate)
            out.append(candidate)
        return tuple(out)

    def _open_loops(
        self,
        ctx: _BugsCtx,
        parent: str,
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Open one `for_loop` per batch dim. Return the innermost
        block id that the caller hangs the `~` line on."""
        current = parent
        for dim, name in zip(plate.batch_dims, loop_names, strict=True):
            for_id = self._fresh(ctx, "for")
            ctx.sb.vertex(for_id, "for_loop")
            ctx.sb.edge(current, for_id, "for_loop")
            var_id = self._fresh(ctx, "id")
            ctx.sb.vertex(var_id, "identifier")
            ctx.sb.constraint(var_id, "literal-value", name)
            ctx.sb.edge(for_id, var_id, "variable")
            rng_id = self._fresh(ctx, "rng")
            ctx.sb.vertex(rng_id, "range")
            ctx.sb.edge(for_id, rng_id, "range")
            lo_id = self._fresh(ctx, "num")
            ctx.sb.vertex(lo_id, "number")
            ctx.sb.constraint(lo_id, "literal-value", "1")
            ctx.sb.edge(rng_id, lo_id, "lower")
            hi_id = self._fresh(ctx, "num")
            upper_text = self._dim_upper_text(dim)
            self._emit_upper_literal(ctx, hi_id, upper_text)
            ctx.sb.edge(rng_id, hi_id, "upper")
            body_id = self._fresh(ctx, "blk")
            ctx.sb.vertex(body_id, "block")
            ctx.sb.edge(for_id, body_id, "body")
            current = body_id
        return current

    def _emit_upper_literal(self, ctx: _BugsCtx, vid: str, text: str) -> None:
        """Emit either a number or an identifier vertex for the loop
        upper bound, depending on whether `text` is a digit string."""
        if text.lstrip("-").isdigit():
            ctx.sb.vertex(vid, "number")
        else:
            ctx.sb.vertex(vid, "identifier")
        ctx.sb.constraint(vid, "literal-value", text)

    def _dim_upper_text(self, dim: Dim) -> str:
        """Return the upper bound text for the for-loop range:
        either the static cardinality or the dynamic size name."""
        size = getattr(dim, "size", None)
        if size is not None:
            return str(int(size))
        size_name = getattr(dim, "size_name", None)
        if size_name is not None:
            return str(size_name)
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"dim:{type(dim).__name__}: unknown shape"],
        )

    # ------------------------------------------------------------------
    # LHS emission.
    # ------------------------------------------------------------------

    def _emit_lhs(
        self,
        ctx: _BugsCtx,
        name: str,
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Emit the LHS variable, indexed by batch loop names plus
        event-axis slices."""
        event_slices = tuple(self._dim_upper_text(d) for d in plate.event_dims)
        if not loop_names and not event_slices:
            return self._emit_bare_identifier(ctx, name)
        return self._emit_indexed_name(
            ctx,
            name,
            tuple(_LoopRef(n) for n in loop_names),
            event_slices,
        )

    # ------------------------------------------------------------------
    # Distribution-call emission.
    # ------------------------------------------------------------------

    def _emit_distribution_call(
        self,
        ctx: _BugsCtx,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """Emit the `<dist_name>(<args>)` distribution_call."""
        dc_id = self._fresh(ctx, "dc")
        ctx.sb.vertex(dc_id, "distribution_call")
        dist_name = meta.target_names["bugs"]
        name_id = self._fresh(ctx, "id")
        ctx.sb.vertex(name_id, "identifier")
        ctx.sb.constraint(name_id, "literal-value", dist_name)
        ctx.sb.edge(dc_id, name_id, "name")
        if not args:
            return dc_id
        al_id = self._fresh(ctx, "al")
        ctx.sb.vertex(al_id, "argument_list")
        ctx.sb.edge(dc_id, al_id, "arguments")
        if meta.qvr_name == "StudentT":
            args, arg_names = _reorder_studentt_dt(args, arg_names)
        elif meta.qvr_name == "HalfStudentT":
            args, arg_names = reorder_half_studentt_dt(args, arg_names)
        elif meta.qvr_name == "NegativeBinomial":
            args, arg_names = reorder_negbin_args(args, arg_names)
        elif meta.qvr_name == "Weibull":
            args, arg_names = reorder_weibull_args(args, arg_names)
        elif meta.qvr_name == "Binomial":
            args, arg_names = reorder_binomial_dbin(args, arg_names)
        elif meta.qvr_name == "Pareto":
            args, arg_names = reorder_pareto_dpar(args, arg_names)
        renames = {
            **meta.arg_aliases.get("bugs", {}),
            **_FAMILY_ALIAS_OVERRIDE.get(meta.qvr_name, {}),
        }
        for arg, aname in zip(args, arg_names, strict=True):
            wrapped = self._apply_alias_transform(arg, aname, renames, meta.qvr_name)
            receiver_event_rank = self._receiver_event_rank(meta, aname)
            child_id = self._emit_arg(
                ctx, wrapped, receiver_event_rank=receiver_event_rank
            )
            ctx.sb.edge(al_id, child_id, self._arg_edge_kind(wrapped))
        return dc_id

    def _receiver_event_rank(self, meta: FamilyMeta, arg_name: str) -> int:
        """Return the receiver arg-constraint's event rank.

        Reads from `meta.distribution_class.arg_constraints[arg_name]`
        when the table is class-level. Property-form
        `arg_constraints` returns rank 0 (the renderer has no
        sentinel; lack of slice in the emit is conservative)."""
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return 0
        expected = cls_attr.get(arg_name)
        if expected is None:
            return 0
        return int(getattr(expected, "event_dim", 0))

    def _apply_alias_transform(
        self,
        arg: IRArg,
        arg_name: str,
        renames: dict[str, str],
        family: str,
    ) -> IRArg:
        """If `renames[arg_name]` targets a name with an arithmetic
        transform in `_ALIAS_TRANSFORMS`
        (or the per-family override in
        `_FAMILY_ALIAS_TRANSFORM_OVERRIDE`),
        wrap `arg` in
        [`IRArgTransform`][quivers.transpile.renderers.IRArgTransform];
        otherwise return `arg` unchanged."""
        target_name = renames.get(arg_name)
        if target_name is None:
            return arg
        override = _FAMILY_ALIAS_TRANSFORM_OVERRIDE.get(family)
        if override is not None and target_name in override:
            transform: str | None = override[target_name]
        else:
            transform = _ALIAS_TRANSFORMS.get(target_name)
        if transform is None:
            return arg
        return IRArgTransform(inner=arg, transform=_as_transform(transform))

    def _arg_edge_kind(self, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return "number"
        if isinstance(arg, IRArgBroadcast):
            return "indexed_variable"
        if isinstance(arg, IRArgTransform):
            if arg.transform in ("log", "exp", "pow_neg"):
                return "function_call"
            return "binary_expression"
        if isinstance(arg, IRArgRef):
            decl_plate = self._lookup_decl_plate_if_known(arg)
            if decl_plate is None:
                return "indexed_variable" if arg.indices else "identifier"
            has_loop = bool(decl_plate.batch_dims)
            has_event = bool(decl_plate.event_dims)
            return (
                "indexed_variable"
                if arg.indices or has_loop or has_event
                else "identifier"
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"arg:{type(arg).__name__}"],
        )

    def _lookup_decl_plate_if_known(self, arg: IRArgRef) -> Plate | None:
        """Edge-kind hook reserved for callers that need to discriminate
        on the arg's declaration plate. The class-method form returns
        `None` so the renderer falls through to no-slicing; the real
        lookup runs inside `_emit_arg` against the per-render context.
        """
        del arg
        return None

    # ------------------------------------------------------------------
    # Arg emission.
    # ------------------------------------------------------------------

    def _emit_arg(
        self,
        ctx: _BugsCtx,
        arg: IRArg,
        *,
        receiver_event_rank: int = 0,
    ) -> str:
        if isinstance(arg, IRArgNumber):
            return self._emit_number(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            return self._emit_ref(ctx, arg, receiver_event_rank=receiver_event_rank)
        if isinstance(arg, IRArgTransform):
            return self._emit_transform(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(arg)
        if isinstance(arg, IRArgFamilyRef):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    "arg:family-ref: BUGS wrappers inline via "
                    "truncation idiom; not legal as a free arg"
                ],
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"arg:{type(arg).__name__}"],
        )

    def _emit_number(self, ctx: _BugsCtx, value: float) -> str:
        nid = self._fresh(ctx, "num")
        ctx.sb.vertex(nid, "number")
        text = str(int(value)) if float(value).is_integer() else repr(value)
        ctx.sb.constraint(nid, "literal-value", text)
        return nid

    def _emit_bare_identifier(self, ctx: _BugsCtx, name: str) -> str:
        nid = self._fresh(ctx, "id")
        ctx.sb.vertex(nid, "identifier")
        ctx.sb.constraint(nid, "literal-value", name)
        return nid

    def _emit_ref(
        self,
        ctx: _BugsCtx,
        ref: IRArgRef,
        *,
        receiver_event_rank: int = 0,
    ) -> str:
        """Emit a name reference, with auto-derived indices.

        The full BUGS index list for a referenced variable is:

        1. The user-supplied bracket indices (`ref.indices`),
           recursively rendered. They fill the leftmost batch_dim
           positions of the referenced variable.
        2. Auto-leading: one entry per UNFILLED batch_dim of the
           referenced variable's declaration plate, drawn from the
           surrounding loop variables by axis-name lookup. When the
           axis does not match any enclosing loop and the enclosing
           observe has a `via` fibration in scope, the axis renders
           as `via[enclosing_loop]`.
        3. One `1:E` slice per event dim of the referenced
           variable's declaration plate. When the referenced
           variable has no event dims but the *receiver* arg
           position expects a vector (`receiver_event_rank > 0`),
           the renderer emits one slice per receiver event rank
           using the receiver's enclosing-plate event-dim sizes
           (e.g. `alpha` referenced in a Dirichlet position emits
           `alpha[1:K]` where K is the surrounding Dirichlet's
           event dim).
        """
        decl_plate = ctx.decl_plates.get(ref.name)
        # Decide whether the reference is "scalar-into-vector" (the
        # referenced var has no declared event_dims AND no
        # batch_dims; the receiver expects a vector). Emit
        # receiver-driven slice in that case.
        if (
            decl_plate is not None
            and not decl_plate.event_dims
            and not decl_plate.batch_dims
            and receiver_event_rank > 0
            and not ref.indices
        ):
            slice_uppers = self._receiver_slice_uppers(ctx, receiver_event_rank)
            if slice_uppers:
                return self._emit_indexed_from_pieces(ctx, ref.name, (), slice_uppers)
        if decl_plate is None:
            if not ref.indices:
                return self._emit_bare_identifier(ctx, ref.name)
            indices = tuple(self._emit_index_child(ctx, ix) for ix in ref.indices)
            kinds = tuple(self._index_child_kind(ix) for ix in ref.indices)
            return self._emit_indexed_with_pre_emitted(
                ctx, ref.name, indices, kinds, ()
            )
        # User indices fill the leftmost batch_dim positions.
        user_pieces = tuple(self._user_index_piece(ctx, ix) for ix in ref.indices)
        # Auto-leading covers the remaining batch dims.
        auto_leading: list[_IndexPiece] = []
        for dim in decl_plate.batch_dims[len(user_pieces) :]:
            axis = str(dim.name)
            loop_var = self._loop_for_axis(ctx, axis)
            if loop_var is not None:
                auto_leading.append(_IndexPiece.ident(loop_var))
                continue
            # No surrounding loop on this axis. If the enclosing
            # observe has a via fibration, apply it; the fibration
            # holds zero-based QVR group codes.
            if ctx.via is not None and ctx.enclosing_loop_names:
                via_inner = ctx.enclosing_loop_names[-1]
                auto_leading.append(
                    _IndexPiece.shifted(
                        _IndexPiece.indexed(ctx.via, (_LoopRef(via_inner),))
                    )
                )
                continue
            # Otherwise the reference is degenerate; emit the axis
            # name as a free identifier (will probably fail in BUGS
            # but is a signal of an under-specified IR).
            auto_leading.append(_IndexPiece.ident(axis))
        # Event dims the receiver does not read as its own event, each
        # visited by an enclosing loop, are read at the loop's
        # position: a vector `z` over `Feat` filling a scalar family
        # replicated over `Feat` is `z[m_Feat]`, one entry per draw.
        event_dims = decl_plate.event_dims
        visited = len(event_dims) - receiver_event_rank
        if visited > 0 and len(user_pieces) <= len(decl_plate.batch_dims):
            loops = [
                self._loop_for_axis(ctx, str(dim.name)) for dim in event_dims[:visited]
            ]
            named = [loop for loop in loops if loop is not None]
            if len(named) == len(loops):
                auto_leading.extend(_IndexPiece.ident(loop) for loop in named)
                event_dims = event_dims[visited:]
        # Event-dim slices: prefer the referenced var's own event
        # dims; fall back to the receiver's expected event rank
        # when the referenced var has none but the receiver does.
        slice_uppers = tuple(self._dim_upper_text(d) for d in event_dims)
        if not slice_uppers and receiver_event_rank > len(user_pieces):
            slice_uppers = self._receiver_slice_uppers(ctx, receiver_event_rank)
        if not (auto_leading or user_pieces or slice_uppers):
            return self._emit_bare_identifier(ctx, ref.name)
        all_pieces = tuple(list(user_pieces) + auto_leading)
        return self._emit_indexed_from_pieces(ctx, ref.name, all_pieces, slice_uppers)

    def _receiver_slice_uppers(
        self, ctx: _BugsCtx, receiver_event_rank: int
    ) -> tuple[str, ...]:
        """Return slice upper-bound texts derived from the receiver
        family's enclosing-plate event dims.

        Used when a scalar-input (e.g. `alpha`) is referenced in a
        receiver position that expects a vector (e.g. Dirichlet's
        `concentration`). The receiver's enclosing plate's
        event_dims supply the per-axis size.
        """
        if ctx.enclosing_plate is None:
            return ()
        ev = ctx.enclosing_plate.event_dims
        if len(ev) < receiver_event_rank:
            return ()
        return tuple(self._dim_upper_text(d) for d in ev[:receiver_event_rank])

    def _loop_for_axis(self, ctx: _BugsCtx, axis: str) -> str | None:
        """Return the enclosing loop variable for `axis`, or None.

        Searches the enclosing plate's batch dims for a dim whose
        name matches `axis`; the parallel entry in
        `enclosing_loop_names` is the loop variable.
        """
        if ctx.enclosing_plate is None:
            return None
        for dim, ln in zip(
            ctx.enclosing_plate.batch_dims,
            ctx.enclosing_loop_names,
            strict=True,
        ):
            if dim.name == axis:
                return ln
        return None

    def _user_index_piece(self, ctx: _BugsCtx, arg: IRArg) -> _IndexPiece:
        """Build an `_IndexPiece` for a user-supplied bracket index.

        Numeric literals become number-pieces. Bare identifier refs
        whose name happens to be a declared variable (e.g. `z` in
        `phi[z]`) get the full per-batch-dim index expansion so
        `phi[z]` over a Doc-batched z under a Word-batched observe
        with via=word_idx renders as `z[word_idx[m_Word]]`.
        """
        if isinstance(arg, IRArgNumber):
            return _IndexPiece.number(
                str(int(arg.value)) if arg.value.is_integer() else repr(arg.value)
            )
        if isinstance(arg, IRArgRef):
            inner_plate = ctx.decl_plates.get(arg.name)
            if inner_plate is None or not inner_plate.batch_dims:
                if not arg.indices:
                    piece = _IndexPiece.ident(arg.name)
                else:
                    children = tuple(
                        self._user_index_piece(ctx, ix) for ix in arg.indices
                    )
                    piece = _IndexPiece.indexed_pieces(arg.name, children)
                # A program value holds a zero-based QVR code.
                return piece if inner_plate is None else _IndexPiece.shifted(piece)
            # arg has declared batch dims: build nested indexing.
            sub: list[_IndexPiece] = []
            for dim in inner_plate.batch_dims:
                axis = str(dim.name)
                loop_var = self._loop_for_axis(ctx, axis)
                if loop_var is not None:
                    sub.append(_IndexPiece.ident(loop_var))
                    continue
                if ctx.via is not None and ctx.enclosing_loop_names:
                    via_inner = ctx.enclosing_loop_names[-1]
                    sub.append(
                        _IndexPiece.shifted(
                            _IndexPiece.indexed(ctx.via, (_LoopRef(via_inner),))
                        )
                    )
                    continue
                sub.append(_IndexPiece.ident(axis))
            for ix in arg.indices:
                sub.append(self._user_index_piece(ctx, ix))
            return _IndexPiece.shifted(_IndexPiece.indexed_pieces(arg.name, tuple(sub)))
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"user-index:{type(arg).__name__}"],
        )

    # ------------------------------------------------------------------
    # Indexed-variable emission helpers.
    # ------------------------------------------------------------------

    def _emit_indexed_name(
        self,
        ctx: _BugsCtx,
        name: str,
        index_args: tuple[_LoopRef | IRArg, ...],
        slice_uppers: tuple[str, ...],
    ) -> str:
        """Emit `name[<index_args>, <1:slice_uppers>...]`.

        `index_args` carries already-prepared pieces (loop-ref or
        IRArg); each prepared piece becomes one index_list child.
        """
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        nid = self._emit_bare_identifier(ctx, name)
        ctx.sb.edge(iv_id, nid, "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for piece in index_args:
            if isinstance(piece, _LoopRef):
                pid = self._emit_bare_identifier(ctx, piece.name)
                ctx.sb.edge(il_id, pid, "identifier")
                continue
            child_id = self._emit_index_child(ctx, piece)
            ctx.sb.edge(il_id, child_id, self._index_child_kind(piece))
        for upper in slice_uppers:
            self._emit_one_to(ctx, il_id, upper)
        return iv_id

    def _emit_indexed_from_pieces(
        self,
        ctx: _BugsCtx,
        name: str,
        pieces: tuple[_IndexPiece, ...],
        slice_uppers: tuple[str, ...],
    ) -> str:
        """Emit `name[<piece_1>, <piece_2>, ..., <1:slice_uppers>]`."""
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        nid = self._emit_bare_identifier(ctx, name)
        ctx.sb.edge(iv_id, nid, "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for piece in pieces:
            child_id, kind = piece.emit(ctx, self)
            ctx.sb.edge(il_id, child_id, kind)
        for upper in slice_uppers:
            self._emit_one_to(ctx, il_id, upper)
        return iv_id

    def _emit_indexed_with_pre_emitted(
        self,
        ctx: _BugsCtx,
        name: str,
        index_child_ids: tuple[str, ...],
        index_child_kinds: tuple[str, ...],
        slice_uppers: tuple[str, ...],
    ) -> str:
        """Emit `name[<pre_emitted_indices>..., <slices>...]` from
        pre-rendered index_list children."""
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        nid = self._emit_bare_identifier(ctx, name)
        ctx.sb.edge(iv_id, nid, "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for cid, kind in zip(index_child_ids, index_child_kinds, strict=True):
            ctx.sb.edge(il_id, cid, kind)
        for upper in slice_uppers:
            self._emit_one_to(ctx, il_id, upper)
        return iv_id

    def _emit_index_child(self, ctx: _BugsCtx, arg: IRArg) -> str:
        """Emit one index_list child for a user-supplied index."""
        if isinstance(arg, IRArgNumber):
            return self._emit_number(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            if not arg.indices:
                return self._emit_bare_identifier(ctx, arg.name)
            children = tuple(self._emit_index_child(ctx, ix) for ix in arg.indices)
            kinds = tuple(self._index_child_kind(ix) for ix in arg.indices)
            return self._emit_indexed_with_pre_emitted(
                ctx, arg.name, children, kinds, ()
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"index:{type(arg).__name__}"],
        )

    def _index_child_kind(self, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return "number"
        if isinstance(arg, IRArgRef):
            return "indexed_variable" if arg.indices else "identifier"
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"index:{type(arg).__name__}"],
        )

    def _emit_one_to(self, ctx: _BugsCtx, parent_index_list: str, upper: str) -> None:
        """Emit a `1:upper` range as a child of `parent_index_list`.

        `upper` is the raw upper-bound text (digit string for a
        static dim, identifier text for a dynamic dim)."""
        rng_id = self._fresh(ctx, "rng")
        ctx.sb.vertex(rng_id, "range")
        ctx.sb.edge(parent_index_list, rng_id, "range")
        lo_id = self._fresh(ctx, "num")
        ctx.sb.vertex(lo_id, "number")
        ctx.sb.constraint(lo_id, "literal-value", "1")
        ctx.sb.edge(rng_id, lo_id, "lower")
        hi_id = self._fresh(ctx, "num")
        self._emit_upper_literal(ctx, hi_id, upper)
        ctx.sb.edge(rng_id, hi_id, "upper")

    # ------------------------------------------------------------------
    # Transform emission (1 / (x*x), 1/x, -x, log(x), exp(x)).
    # ------------------------------------------------------------------

    def _emit_transform(self, ctx: _BugsCtx, wrapped: IRArgTransform) -> str:
        """Emit the arithmetic transform's expression schema.

        BUGS infix arithmetic. The root of the emitted fragment is
        a `binary_expression` (or `function_call` for log / exp).
        """
        first_inner = self._emit_arg(ctx, wrapped.inner)
        if wrapped.transform == "inv_square":
            second_inner = self._emit_arg(ctx, wrapped.inner)
            return self._emit_inv_square(ctx, first_inner, second_inner)
        if wrapped.transform == "inv":
            return self._emit_inv(ctx, first_inner)
        if wrapped.transform == "neg":
            return self._emit_neg(ctx, first_inner)
        if wrapped.transform == "log":
            return self._emit_unary_call(ctx, "log", first_inner)
        if wrapped.transform == "exp":
            return self._emit_unary_call(ctx, "exp", first_inner)
        if wrapped.transform == "one_minus":
            return self._emit_one_minus(ctx, first_inner)
        if wrapped.transform == "pow_neg":
            if wrapped.operand is None:
                raise UnsupportedConstruct(
                    f"qvr-{self.target}",
                    ["transform:pow_neg: missing exponent operand"],
                )
            operand_id = self._emit_arg(ctx, wrapped.operand)
            return self._emit_pow_neg(
                ctx,
                first_inner,
                self._arg_edge_kind(wrapped.inner),
                operand_id,
                self._arg_edge_kind(wrapped.operand),
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"transform:{wrapped.transform}"],
        )

    def _emit_one_minus(self, ctx: _BugsCtx, inner_id: str) -> str:
        """Emit `1 - <inner>`."""
        one = self._emit_number(ctx, 1.0)
        diff = self._fresh(ctx, "be")
        ctx.sb.vertex(diff, "binary_expression")
        ctx.sb.constraint(diff, "field:operator", "-")
        ctx.sb.constraint(diff, "chose-alt-fingerprint", "-")
        ctx.sb.edge(diff, one, "left")
        ctx.sb.edge(diff, inner_id, "right")
        return diff

    def _emit_pow_neg(
        self,
        ctx: _BugsCtx,
        inner_id: str,
        inner_kind: str,
        operand_id: str,
        operand_kind: str,
    ) -> str:
        """Emit `pow(<inner>, -<operand>)` as a two-argument
        `function_call`."""
        neg = self._fresh(ctx, "ue")
        ctx.sb.vertex(neg, "unary_expression")
        ctx.sb.constraint(neg, "field:operator", "-")
        ctx.sb.constraint(neg, "chose-alt-fingerprint", "-")
        ctx.sb.constraint(neg, "chose-alt-child-kinds", operand_kind)
        ctx.sb.edge(neg, operand_id, "operand")
        call = self._fresh(ctx, "call")
        ctx.sb.vertex(call, "function_call")
        fn = self._emit_bare_identifier(ctx, "pow")
        ctx.sb.edge(call, fn, "name")
        al = self._fresh(ctx, "al")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(call, al, "arguments")
        ctx.sb.edge(al, inner_id, inner_kind)
        ctx.sb.edge(al, neg, "unary_expression")
        return call

    def _emit_inv_square(
        self,
        ctx: _BugsCtx,
        first_inner: str,
        second_inner: str,
    ) -> str:
        """Emit `1 / (<inner> * <inner>)`."""
        mul = self._fresh(ctx, "be")
        ctx.sb.vertex(mul, "binary_expression")
        ctx.sb.constraint(mul, "field:operator", "*")
        ctx.sb.constraint(mul, "chose-alt-fingerprint", "*")
        ctx.sb.edge(mul, first_inner, "left")
        ctx.sb.edge(mul, second_inner, "right")
        paren = self._fresh(ctx, "par")
        ctx.sb.vertex(paren, "parenthesized_expression")
        ctx.sb.edge(paren, mul, "parenthesized_expression")
        one = self._emit_number(ctx, 1.0)
        div = self._fresh(ctx, "be")
        ctx.sb.vertex(div, "binary_expression")
        ctx.sb.constraint(div, "field:operator", "/")
        ctx.sb.constraint(div, "chose-alt-fingerprint", "/")
        ctx.sb.edge(div, one, "left")
        ctx.sb.edge(div, paren, "right")
        return div

    def _emit_inv(self, ctx: _BugsCtx, inner_id: str) -> str:
        """Emit `1 / <inner>`."""
        one = self._emit_number(ctx, 1.0)
        div = self._fresh(ctx, "be")
        ctx.sb.vertex(div, "binary_expression")
        ctx.sb.constraint(div, "field:operator", "/")
        ctx.sb.constraint(div, "chose-alt-fingerprint", "/")
        ctx.sb.edge(div, one, "left")
        ctx.sb.edge(div, inner_id, "right")
        return div

    def _emit_neg(self, ctx: _BugsCtx, inner_id: str) -> str:
        """Emit `-<inner>` via unary_expression."""
        u = self._fresh(ctx, "ue")
        ctx.sb.vertex(u, "unary_expression")
        ctx.sb.constraint(u, "field:operator", "-")
        ctx.sb.constraint(u, "chose-alt-fingerprint", "-")
        ctx.sb.edge(u, inner_id, "operand")
        return u

    def _emit_unary_call(self, ctx: _BugsCtx, fn_name: str, inner_id: str) -> str:
        """Emit `<fn_name>(<inner>)` as a function_call."""
        call = self._fresh(ctx, "call")
        ctx.sb.vertex(call, "function_call")
        fn = self._emit_bare_identifier(ctx, fn_name)
        ctx.sb.edge(call, fn, "name")
        al = self._fresh(ctx, "al")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(call, al, "arguments")
        ctx.sb.edge(al, inner_id, "identifier")
        return call

    # ------------------------------------------------------------------
    # Truncation: T(lower, upper).
    # ------------------------------------------------------------------

    def _emit_truncation(
        self,
        ctx: _BugsCtx,
        bounds: tuple[IRArg, ...],
    ) -> str:
        """Emit a `truncation` node carrying `T(lower, upper)`, or
        `T(lower,)` when `bounds` holds a single lower bound.

        The `bugs` backend executes through the JAGS engine, so it
        emits JAGS's renormalized `T( , )` suffix, the same spelling
        the `jags` backend uses. The grammar's truncation rule offers
        `T( , )` and `I( , )` as alternatives keyed by the
        `chose-alt-fingerprint` constraint; the renderer selects
        `T( , )` for both backends because `I( , )` is JAGS interval
        censoring, which JAGS rejects on any latent-parent node.
        """
        if not bounds or len(bounds) > 2:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"truncation:expected 1 or 2 bounds, got {len(bounds)}"],
            )
        tr = self._fresh(ctx, "tr")
        ctx.sb.vertex(tr, "truncation")
        ctx.sb.constraint(
            tr, "chose-alt-fingerprint", TRUNCATION_FINGERPRINT[self.target]
        )
        for bound in bounds:
            ctx.sb.edge(tr, self._emit_arg(ctx, bound), "child_of")
        return tr

    # ------------------------------------------------------------------
    # Misc helpers.
    # ------------------------------------------------------------------

    def _infer_arg_names(
        self, meta: FamilyMeta, args: tuple[IRArg, ...]
    ) -> tuple[str, ...]:
        """Return the arg-name tuple for `args` against `meta`."""
        cls_attr = meta.distribution_class.arg_constraints
        if isinstance(cls_attr, dict):
            names = tuple(cls_attr.keys())
        else:
            names = ()
        return names[: len(args)]

    def _fresh(self, ctx: _BugsCtx, prefix: str) -> str:
        ctx.fresh_counter += 1
        return f"{prefix}_{ctx.fresh_counter}"


# ----------------------------------------------------------------------
# Helpers outside the class.
# ----------------------------------------------------------------------


def _as_bugs_ctx(ctx: RenderContext) -> _BugsCtx:
    """Narrow a base `RenderContext` to the BUGS extension."""
    if not isinstance(ctx, _BugsCtx):
        raise UnsupportedConstruct("qvr-bugs", ["ctx:type-mismatch"])
    return ctx


@dataclasses.dataclass(frozen=True)
class _LoopRef:
    """A bare-identifier reference to a for-loop variable."""

    name: str


@dataclasses.dataclass(frozen=True)
class _IndexPiece:
    """One piece of an index list, post-substitution.

    Either a bare identifier (`ident`), a number literal, or an
    indexed-variable form (`indexed`) with nested pieces.
    """

    kind: Literal["ident", "number", "indexed", "shifted"]
    text: str
    children: tuple[_IndexPiece | _LoopRef, ...] = ()

    @classmethod
    def ident(cls, name: str) -> _IndexPiece:
        return cls(kind="ident", text=name)

    @classmethod
    def number(cls, text: str) -> _IndexPiece:
        return cls(kind="number", text=text)

    @classmethod
    def indexed(
        cls, name: str, children: tuple[_IndexPiece | _LoopRef, ...]
    ) -> _IndexPiece:
        return cls(kind="indexed", text=name, children=children)

    @classmethod
    def shifted(cls, inner: _IndexPiece) -> _IndexPiece:
        """The subscript `<inner> + 1`: a zero-based QVR code read from one.

        Parameters
        ----------
        inner : _IndexPiece
            The piece holding the code.

        Returns
        -------
        _IndexPiece
            The shifted piece.
        """
        return cls(kind="shifted", text="+", children=(inner,))

    @classmethod
    def indexed_pieces(
        cls, name: str, children: tuple[_IndexPiece, ...]
    ) -> _IndexPiece:
        return cls(kind="indexed", text=name, children=tuple(children))

    def emit(self, ctx: _BugsCtx, renderer: BUGSRenderer) -> tuple[str, str]:
        """Emit this piece as a schema fragment; return `(id, kind)`."""
        if self.kind == "ident":
            return renderer._emit_bare_identifier(ctx, self.text), "identifier"
        if self.kind == "number":
            nid = renderer._fresh(ctx, "num")
            ctx.sb.vertex(nid, "number")
            ctx.sb.constraint(nid, "literal-value", self.text)
            return nid, "number"
        if self.kind == "shifted":
            (inner,) = self.children
            assert isinstance(inner, _IndexPiece)
            inner_id, _inner_kind = inner.emit(ctx, renderer)
            one_id, _one_kind = _IndexPiece.number("1").emit(ctx, renderer)
            be_id = renderer._fresh(ctx, "be")
            ctx.sb.vertex(be_id, "binary_expression")
            ctx.sb.constraint(be_id, "field:operator", "+")
            ctx.sb.constraint(be_id, "chose-alt-fingerprint", "+")
            ctx.sb.edge(be_id, inner_id, "left")
            ctx.sb.edge(be_id, one_id, "right")
            return be_id, "binary_expression"
        # indexed:
        iv_id = renderer._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        name_id = renderer._emit_bare_identifier(ctx, self.text)
        ctx.sb.edge(iv_id, name_id, "name")
        il_id = renderer._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for child in self.children:
            if isinstance(child, _LoopRef):
                cid = renderer._emit_bare_identifier(ctx, child.name)
                ctx.sb.edge(il_id, cid, "identifier")
                continue
            cid, kind = child.emit(ctx, renderer)
            ctx.sb.edge(il_id, cid, kind)
        return iv_id, "indexed_variable"


def _refuse_observed_class_code(target: str, node: IRObserve) -> None:
    """Refuse an observed class code, which BUGS cannot score as QVR codes it.

    The observed data holds QVR's zero-based class codes, and `dcat`
    counts classes from one. JAGS binds the one-based code in its
    `data { ... }` block; the BUGS language has no such block, cannot
    give a logical node computed from data a distribution of its own,
    and cannot add the written-out log-probability to the joint without
    a data-bound zeros-trick carrier (see `_NO_BUGS_FREE_DENSITY_TERM`),
    so no emitted relation scores the codes the data holds.

    Parameters
    ----------
    target : str
        The target being rendered.
    node : IRObserve
        The observation.

    Raises
    ------
    UnsupportedConstruct
        If the observation's family is a class-index family.
    """
    if not is_class_code_family(node.family):
        return
    raise UnsupportedConstruct(
        f"qvr-{target}",
        [
            f"observe:class-code:{node.name}: the data codes {node.family} "
            f"classes from zero and `dcat` reads them from one; BUGS has no "
            f"data block to shift the observed codes and no free "
            f"log-density statement to score them otherwise"
        ],
    )


def _is_wrapper_family_call(args: tuple[IRArg, ...]) -> bool:
    """Return True iff the first arg is an
    [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef].

    The first-arg [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
    is the structural signal Lower emits for wrapper-family calls
    (`Truncated(base, lo, hi)`, `Mixture(weights, component)`, ...);
    the renderer dispatches on this shape rather than on the family
    name string."""
    return bool(args) and isinstance(args[0], IRArgFamilyRef)


def _draw_arg_to_ir(a: str | float) -> IRArg:
    """Convert a morphism ``~ Family(args)`` init arg into the
    corresponding [`IRArg`][quivers.transpile.ir.IRArg].

    Init-family args arrive in wire form: a ``float`` literal or a
    ``str`` identifier. Used by the wrapper-family (Truncated)
    handler to lift the referenced morphism's init clause into IR
    form for re-emission as the truncated call's args.
    """
    if not isinstance(a, (int, float, str)):
        a = atom_value(a)
    if isinstance(a, (int, float)):
        return IRArgNumber(value=float(a))
    stripped = a.strip()
    try:
        return IRArgNumber(value=float(stripped))
    except ValueError:
        return IRArgRef(name=stripped)


def _reorder_studentt_dt(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape a location-scale ``StudentT(df, loc, scale)`` into BUGS'
    ``dt(mu, tau, k)`` argument order.

    BUGS' Student-t is parameterised by location, precision, and
    degrees of freedom, in that order; torch's ``StudentT`` carries
    ``(df, loc, scale)``. This reorders to ``(loc, scale, df)`` and
    pre-wraps the scale in the precision transform
    ``tau = 1/(scale*scale)`` so the emitted call is
    ``dt(loc, 1/(scale*scale), df)``.
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


def _as_transform(
    t: str,
) -> Literal["inv_square", "inv", "neg", "log", "exp"]:
    """Coerce a string to the `IRArgTransform.transform` literal type."""
    if t == "inv_square":
        return "inv_square"
    if t == "inv":
        return "inv"
    if t == "neg":
        return "neg"
    if t == "log":
        return "log"
    if t == "exp":
        return "exp"
    raise UnsupportedConstruct(
        "qvr-bugs",
        [f"transform:{t}"],
    )


__all__ = ["BUGSRenderer"]
