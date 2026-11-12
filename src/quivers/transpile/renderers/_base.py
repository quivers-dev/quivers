"""Shared machinery for transpilation renderers.

``RendererBase`` dispatches IR nodes, substitutes plate indices, lowers
finite marginalizations to weighted atom reductions, and checks
structural invariants before emission. Concrete renderers implement
declaration, sampling, marginalization, and broadcast forms.
"""

from __future__ import annotations

import abc
from typing import Literal, Protocol, runtime_checkable

import didactic.api as dx
import panproto
from torch.distributions.constraints import Constraint, simplex

from quivers.dsl.ast_nodes import Expr, MorphismDecl
from quivers.dsl.ast_nodes.let_expressions import (
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
    LetFactorCase,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.family_meta import FAMILY_META, marginalize_support
from quivers.transpile.ir import (
    ConstraintSpec,
    LetAffineSource,
    LetExprAffineMap,
    CSIntegerInterval,
    CSNonnegativeInteger,
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
    IRExpr,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
    Plate,
)


type BlockKind = Literal[
    "data",
    "parameters",
    "transformed_parameters",
    "model",
    "generated_quantities",
    "function_body",
]
"""Where a declaration lands in the target's program structure.

Each backend interprets the kind per its own program layout: Stan has
actual blocks, while NumPyro's block is the function body.
"""


type SchemaFragment = str
"""One panproto schema fragment.

Either an opaque vertex id, or the empty string when the dispatch point
emits nothing of its own.
"""


#: The class of `torch.distributions.constraints.simplex`. Torch
#: exports the constraint only as that singleton instance, so its
#: type is how a family's `arg_constraints` entry is recognised as a
#: mixing-weight slot.
_SIMPLEX_CONSTRAINT: type[Constraint] = type(simplex)


class RenderContext:
    """The mutable state of one `render` call.

    A renderer creates one per call and threads it through the IR-walk
    dispatch and its helpers as the first positional argument. It holds
    the panproto schema builder the target program is assembled in, the
    resolved morphism and define tables, a counter for fresh vertex
    names, and the cardinality table. Unlike the IR it is mutable and
    does not round-trip.

    Parameters
    ----------
    sb
        The schema builder the renderer emits into.
    morphisms
        The module's morphism declarations, by name.
    defines
        The module's define expressions, by name.
    fresh_counter
        The next fresh-name ordinal.
    cards
        The cardinality of each finite object, by name; empty when not
        supplied.

    Attributes
    ----------
    sb
        The schema builder the renderer emits into.
    morphisms
        The module's morphism declarations, by name.
    defines
        The module's define expressions, by name.
    fresh_counter
        The next fresh-name ordinal; helpers increment it as they mint
        names.
    cards
        The cardinality of each finite object, by name.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        morphisms: dict[str, MorphismDecl],
        defines: dict[str, Expr],
        fresh_counter: int = 0,
        cards: dict[str, int] | None = None,
    ) -> None:
        self.sb = sb
        self.morphisms = morphisms
        self.defines = defines
        self.fresh_counter = fresh_counter
        self.cards: dict[str, int] = {} if cards is None else cards


class IRArgTransform(IRArg):
    """A renderer-applied transform on an IR arg.

    The renderer constructs this during emission when an
    `arg_aliases` rename targets a parameterisation that requires
    arithmetic conversion (e.g. BUGS `tau = 1/(scale*scale)` for
    Normal). Per-renderer `_ALIAS_TRANSFORMS` tables key the
    transform on the renamed target name; `RendererBase` provides
    the constructor.

    Most transforms act on `inner` alone (``inv``, ``neg``,
    ``one_minus`` = ``1 - inner``, ...). The two-operand ``pow_neg``
    additionally reads `operand`, emitting ``pow(inner, -operand)``:
    the JAGS / BUGS Weibull rate parameterisation needs the
    concentration argument as the exponent, so the reorder helper
    threads it in as `operand`.

    `IRArgTransform` is a renderer-side IR extension. `Lower`
    never constructs it.

    Parameters
    ----------
    inner
        The argument transformed.
    transform
        The transform applied.
    operand
        The second operand of ``pow_neg``; ``None`` otherwise.
    kind
        The discriminator; always ``"transform"``.
    """

    inner: IRArg
    transform: Literal["inv_square", "inv", "neg", "log", "exp", "one_minus", "pow_neg"]
    operand: IRArg | None = None
    kind: Literal["transform"] = "transform"


class IRMarginalAtom(dx.Model):
    """One atom of a marginalized latent's finite support.

    [`RendererBase.marginal_atoms`][quivers.transpile.renderers.RendererBase.marginal_atoms]
    returns one of these per support point of an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]. Together
    they carry the whole integrated density: writing `L_a` for the
    log-density the target accumulates while emitting `scope`, and
    `w_a` for the log-density of `weight_family(weight_args)` at
    `value`, the block contributes

        logsumexp_a (w_a + L_a)

    to the program's log-density, elementwise over whatever rows the
    scope's own plates carry. No latent variable is declared: the
    atoms replace it.

    `weight_family` is the family the latent's marginalize support
    names. For a `Bernoulli`, `weight_args` carries the probability
    argument, or the logits when the call site names them
    (`weight_arg_names` reads ``("logits",)``), which a renderer
    weights by ``log sigmoid(-l)`` and ``log sigmoid(l)`` in log space.

    `IRMarginalAtom` is a renderer-side IR extension. `Lower`
    never constructs it.

    Parameters
    ----------
    value
        The support point.
    weight_family
        The family whose log-density at ``value`` is the atom's log-prior
        weight.
    weight_args
        The arguments ``weight_family`` is evaluated with.
    weight_arg_names
        The parameter name of each of ``weight_args``.
    scope
        The block's body with the latent replaced by ``value``.
    """

    value: IRArgNumber
    weight_family: str
    weight_args: tuple[IRArg, ...]
    weight_arg_names: tuple[str, ...]
    scope: tuple[IRNode, ...]


@runtime_checkable
class Renderer(Protocol):
    """The protocol every backend renderer satisfies.

    [`transpile`][quivers.transpile.transpile] calls `render` on a lowered
    [`IRProgram`][quivers.transpile.IRProgram]; the four dispatch points
    are what the shared IR walk of
    [`RendererBase`][quivers.transpile.renderers.RendererBase] calls per
    node.
    """

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
            The site's name.
        family
            The distribution family's QVR name.
        args
            The family's arguments, in family order.
        arg_names
            The parameter name of each argument.
        constraint
            The site's support.
        plate
            The site's event and batch dimensions.
        observed
            Whether the site is conditioned on data.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        ...

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
        ...


def marginalize_row_rank(node: IRMarginalize) -> int:
    """How many axes of the block's body are accumulated before the
    reduction over the latent.

    `docs/semantics/programs.md` §2.7 reduces per group `g`:
    ``aggr_k [log pi(g, k) + sum_{n : iota(n) = g} l(n, k)]``. A block
    that carries a grouping plate has already keyed its accumulator by
    `g`, and each group's rows are gathered into it by the observes'
    `via` fibrations, so nothing further is accumulated here.

    A block with no grouping plate declares one latent for its whole
    body (§2.6 reduces "the accumulated log-likelihood"), which is the
    same formula with a single group: every row of the enclosed
    observe is conditioned on that one draw, so all of them are summed
    before the reduction rather than each reducing on its own. The
    two orders differ whenever the body is plated, and reducing per
    row silently gives each row a draw the source never declared.
    """
    if node.plate.batch_dims:
        return 0
    ranks = {
        len(inner.plate.batch_dims)
        for inner in node.scope
        if isinstance(inner, IRObserve)
    }
    return max(ranks, default=0)


def refuse_ungrouped_row_marginalize(target: str, node: IRMarginalize) -> None:
    """Refuse a block this renderer would score with the wrong order.

    An ungrouped `marginalize` over a plated `observe` shares one
    latent across the body's rows, so its density accumulates the rows
    before reducing over the latent
    (`docs/semantics/programs.md` §2.6, and §2.7 with a single group).
    A renderer that reduces each row on its own instead scores a
    measure in which every row carries its own draw, which differs
    from the program's by an amount that moves with the data and so
    survives Theorem 4.1's quotient by a constant.

    Call this from a renderer that has not been taught the accumulated
    order. Refusing is the honest outcome while that is true: a wrong
    number that no comparison catches is worse than no number, and the
    message says which order the target owes and where the ones that
    already emit it can be read.
    """
    if marginalize_row_rank(node) == 0:
        return
    raise UnsupportedConstruct(
        target,
        [f"marginalize:ungrouped-over-plate:{node.latent}"],
    )


class RendererBase(abc.ABC):
    """Shared IR-walk dispatch and helpers for backend renderers.

    Subclasses implement `target_protocol`, `declare`, `sample`,
    `marginalize`, `broadcast`, and may override `render` to wrap
    the inherited walk with per-backend pre / post processing
    (block emission, function header, etc.).
    """

    target: str

    # ----- abstract dispatch points -----

    @abc.abstractmethod
    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol for the renderer's target language.

        Each renderer builds its schema from this protocol's
        `schema()` builder.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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
            The site's name.
        family
            The distribution family's QVR name.
        args
            The family's arguments, in family order.
        arg_names
            The parameter name of each argument.
        constraint
            The site's support.
        plate
            The site's event and batch dimensions.
        observed
            Whether the site is conditioned on data.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    # ----- IR walk -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and route each node to the right dispatch.

        The default walk dispatches on the IRNode discriminator:

        | node | dispatch |
        | --- | --- |
        | `IRDataInput` | `declare(name, ..., block="data")` |
        | `IRSample` (latent) | `declare + sample(observed=False)` |
        | `IRObserve` | `declare + sample(observed=True)` |
        | `IRDeterministic` | `declare + sample(observed=True)` (deterministic) |
        | `IRScore` | renderer-specific (`target +=`) |
        | `IRMarginalize` | `marginalize(node)` |
        | `IRReturn` | `_emit_return(names)` |

        Subclasses may override `render` to wrap the walk with
        their own block prologue / epilogue.

        Parameters
        ----------
        ir
            The lowered program.

        Returns
        -------
        panproto.Schema
            The target program, in the target grammar's theory.
        """
        assert_no_lists(ir)
        assert_no_dropped_param_map(ir, self.target)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = RenderContext(sb=sb, morphisms={}, defines={})
        self._walk(ctx, ir)
        return sb.build()

    def _walk(self, ctx: RenderContext, ir: IRProgram) -> None:
        for inp in ir.inputs:
            self.declare(ctx, inp.name, inp.constraint, inp.plate, block="data")
        for node in ir.body:
            self._dispatch_node(ctx, node)

    def _dispatch_node(self, ctx: RenderContext, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            self.declare(ctx, node.name, node.constraint, node.plate, block="data")
            return
        if isinstance(node, IRSample):
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
            return
        if isinstance(node, IRDeterministic):
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="transformed_parameters",
            )
            return
        if isinstance(node, IRScore):
            # Per-renderer: BUGS / Stan handle `target +=` natively;
            # NumPyro / Pyro use `factor(...)`. Subclasses override
            # `_emit_score` for the target idiom.
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_return(ctx, node.names)
            return
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    # ----- index-substitution helpers -----

    def index_for(self, ctx: RenderContext, plate: Plate) -> tuple[str, ...]:
        """Return loop-index names for the plate's batch_dims.

        Returns one name per batch_dim, generated via
        `ctx.fresh_counter`. Renderers pair these with each batch
        dim's size to emit nested `for (m_i in 1:B_i)` loops.

        Parameters
        ----------
        ctx
            The render call's mutable state.
        plate
            The plate whose batch dimensions are looped over.

        Returns
        -------
        tuple[str, ...]
            One index name per batch dimension, outermost first.
        """
        del ctx
        return tuple(f"m_{dim.name}" for dim in plate.batch_dims)

    def substitute_indices(self, arg: IRArg, names: tuple[str, ...]) -> IRArg:
        """Apply the loop-index names from `names` to the arg's
        bracket indices.

        Used by sample / observe rendering to thread the
        surrounding plate's loop variables through an
        [`IRArgRef`][quivers.transpile.ir.IRArgRef]'s index list so
        the emitted call form is `name[m_0, m_1, ...]`.

        Parameters
        ----------
        arg
            The argument whose index list is rewritten.
        names
            The surrounding plate's loop-index names.

        Returns
        -------
        IRArg
            The argument with its nested indices substituted; any other
            argument unchanged.
        """
        if isinstance(arg, IRArgRef) and arg.indices:
            new_indices = tuple(
                self.substitute_indices(idx, names) for idx in arg.indices
            )
            return IRArgRef(name=arg.name, indices=new_indices)
        return arg

    # ----- marginalize: the integrated-density lowering -----

    def marginal_atoms(
        self,
        node: IRMarginalize,
        *,
        support_size: int | None = None,
    ) -> tuple[IRMarginalAtom, ...]:
        """Expand an `IRMarginalize` node into weighted atoms.

        Each returned `IRMarginalAtom` pins the latent within a copy of
        `node.scope`. The renderer combines each atom's weight with its scope
        log density and reduces the result with `node.reduction`.

        `support_size` gives the class count for a ``"class_index"`` atom set;
        binary atom sets ignore it.

        Parameters
        ----------
        node
            The marginalize block.
        support_size
            The class count of a ``"class_index"`` atom set, when the family
            does not fix it.

        Returns
        -------
        tuple[IRMarginalAtom, ...]
            One atom per support point, in support order.

        Raises
        ------
        UnsupportedConstruct
            If the family is unknown, has no supported marginal, or its
            class count is unresolved.
        """
        meta = FAMILY_META.get(node.family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:unknown:{node.family}"],
            )
        support = marginalize_support(meta)
        if support is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"marginalize:non-finite-support:{node.family}"],
            )
        size = support.size if support.size is not None else support_size
        if size is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"marginalize:unknown-cardinality:{node.family}"],
            )
        if support.atoms == "binary":
            weight_name = (
                support.logit_arg
                if support.logit_arg is not None and support.logit_arg in node.arg_names
                else support.weight_arg
            )
            weight_args: tuple[IRArg, ...] = (
                self._marginal_weight_arg(node, weight_name),
            )
            weight_arg_names: tuple[str, ...] = (weight_name,)
        else:
            weight_args = node.args
            weight_arg_names = node.arg_names
        return tuple(
            IRMarginalAtom(
                value=IRArgNumber(value=float(k)),
                weight_family=support.weight_family,
                weight_args=weight_args,
                weight_arg_names=weight_arg_names,
                scope=substitute_latent(
                    node.scope, node.latent, IRArgNumber(value=float(k))
                ),
            )
            for k in range(size)
        )

    def _marginal_weight_arg(self, node: IRMarginalize, arg_name: str) -> IRArg:
        """Return the probability argument the atom weights read."""
        for name, arg in zip(node.arg_names, node.args, strict=True):
            if name == arg_name:
                return arg
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [
                f"marginalize:missing-probability-argument:"
                f"{node.family}: the call site names no {arg_name!r} "
                f"argument to weight the atoms with"
            ],
        )

    # ----- score / return defaults -----

    def _emit_score(self, ctx: RenderContext, node: IRScore) -> None:
        """Default: subclasses override to emit `target +=` /
        `numpyro.factor` / similar idiom for the renderer's target.
        """
        del ctx, node
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["node:IRScore: renderer does not implement score"],
        )

    def _emit_return(self, ctx: RenderContext, names: tuple[str, ...]) -> None:
        """Default: subclasses override to emit `return ...` / Stan
        generated-quantities aliasing / similar.
        """
        del ctx, names


# ---------------------------------------------------------------------------
# Latent substitution: pin a marginalized latent to one atom of its
# support throughout a scope.
# ---------------------------------------------------------------------------


def logit_weighted(atom: IRMarginalAtom) -> bool:
    """Whether an atom set is weighted from the logits of a Bernoulli.

    Such atoms are weighted by ``log sigmoid(-l)`` for 0 and
    ``log sigmoid(l)`` for 1, which a renderer computes in log space
    rather than forming the probability, whose complement rounds to
    zero where the logit is large.

    Parameters
    ----------
    atom : IRMarginalAtom
        Any atom of the set; every atom carries the same weights.

    Returns
    -------
    bool
        ``True`` when the atoms carry the Bernoulli's logits.
    """
    return atom.weight_family == "Bernoulli" and atom.weight_arg_names == ("logits",)


def substitute_latent(
    body: tuple[IRNode, ...], latent: str, value: IRArgNumber
) -> tuple[IRNode, ...]:
    """Rewrite `body` with every reference to `latent` replaced by the
    constant `value`.

    This is the enumeration step of the marginalize lowering: one
    rewritten copy of the scope per atom of the latent's support. Both
    reference languages are covered, distribution arguments
    (`phi[z]` becomes `phi[k]`) and let-expressions (`z * rate`
    becomes `k * rate`), so the rewritten scope names the latent
    nowhere and needs no latent declaration.

    A nested [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
    that rebinds the same name shadows the outer one, and its scope is
    left alone.
    """
    return tuple(_substitute_latent_node(node, latent, value) for node in body)


def _substitute_latent_node(node: IRNode, latent: str, value: IRArgNumber) -> IRNode:
    if isinstance(node, IRSample):
        return IRSample(
            name=node.name,
            family=node.family,
            args=_substitute_latent_args(node.args, latent, value),
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
        )
    if isinstance(node, IRObserve):
        return IRObserve(
            name=node.name,
            family=node.family,
            args=_substitute_latent_args(node.args, latent, value),
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
            via=node.via,
        )
    if isinstance(node, IRDeterministic):
        return IRDeterministic(
            name=node.name,
            expr=_substitute_latent_expr(node.expr, latent, value),
            constraint=node.constraint,
            plate=node.plate,
        )
    if isinstance(node, IRScore):
        return IRScore(
            name=node.name,
            expr=_substitute_latent_expr(node.expr, latent, value),
        )
    if isinstance(node, IRMarginalize):
        inner_scope = (
            node.scope
            if node.latent == latent
            else substitute_latent(node.scope, latent, value)
        )
        return IRMarginalize(
            latent=node.latent,
            family=node.family,
            args=_substitute_latent_args(node.args, latent, value),
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
            reduction=node.reduction,
            scope=inner_scope,
        )
    if isinstance(node, (IRDataInput, IRReturn)):
        # Neither binds nor reads a latent: an input is exogenous and
        # a return names only program-level results.
        return node
    raise UnsupportedConstruct(
        "qvr-renderer",
        [f"marginalize:scope:{type(node).__name__}"],
    )


def _substitute_latent_args(
    args: tuple[IRArg, ...], latent: str, value: IRArgNumber
) -> tuple[IRArg, ...]:
    return tuple(_substitute_latent_arg(arg, latent, value) for arg in args)


def _substitute_latent_arg(arg: IRArg, latent: str, value: IRArgNumber) -> IRArg:
    if isinstance(arg, IRArgRef):
        if arg.name == latent:
            if arg.indices:
                raise UnsupportedConstruct(
                    "qvr-renderer",
                    [
                        f"marginalize:indexed-latent:{latent}: the "
                        f"latent is subscripted, so no single atom of "
                        f"its support stands for the reference"
                    ],
                )
            return value
        return IRArgRef(
            name=arg.name,
            indices=_substitute_latent_args(arg.indices, latent, value),
        )
    if isinstance(arg, IRArgBroadcast):
        return IRArgBroadcast(
            value=_substitute_latent_arg(arg.value, latent, value),
            target_shape=arg.target_shape,
        )
    if isinstance(arg, IRArgList):
        return IRArgList(elements=_substitute_latent_args(arg.elements, latent, value))
    if isinstance(arg, IRArgMatrix):
        return IRArgMatrix(
            rows=tuple(
                IRArgList(elements=_substitute_latent_args(row.elements, latent, value))
                for row in arg.rows
            )
        )
    if isinstance(arg, IRArgTransform):
        return IRArgTransform(
            inner=_substitute_latent_arg(arg.inner, latent, value),
            transform=arg.transform,
            operand=(
                None
                if arg.operand is None
                else _substitute_latent_arg(arg.operand, latent, value)
            ),
        )
    if isinstance(arg, (IRArgNumber, IRArgFamilyRef, IRArgKernel)):
        # A literal binds no name; a family ref names a morphism and a
        # kernel arg an exogenous input, neither of which a
        # marginalized latent can shadow.
        return arg
    raise UnsupportedConstruct(
        "qvr-renderer",
        [f"marginalize:arg:{type(arg).__name__}"],
    )


def _substitute_latent_expr(expr: IRExpr, latent: str, value: IRArgNumber) -> IRExpr:
    if isinstance(expr, LetExprVar):
        if expr.name == latent:
            return LetExprLiteral(value=value.value)
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute_latent_expr(expr.left, latent, value),
            right=_substitute_latent_expr(expr.right, latent, value),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_substitute_latent_expr(expr.operand, latent, value)
        )
    if isinstance(expr, LetExprCall):
        if (
            expr.func == "select"
            and len(expr.args) == 3
            and isinstance(expr.args[0], LetExprVar)
            and expr.args[0].name == latent
        ):
            # A choice conditioned on a Boolean latent (`real(z)` is
            # `select(z, 1.0, 0.0)`) is decided at each atom: the atom 1
            # is `true` and selects the first branch, the atom 0 is
            # `false` and selects the second. Substituting the number
            # into the condition would hand a target a numeric
            # condition, which a vectorized `where` rejects or reads as
            # truthy.
            chosen = expr.args[1] if value.value != 0.0 else expr.args[2]
            return _substitute_latent_expr(chosen, latent, value)
        return LetExprCall(
            func=expr.func,
            args=_substitute_latent_exprs(expr.args, latent, value),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute_latent_expr(expr.array, latent, value),
            indices=_substitute_latent_exprs(expr.indices, latent, value),
        )
    if isinstance(expr, LetExprAffineMap):
        return LetExprAffineMap(
            weight=_substitute_latent_expr(expr.weight, latent, value),
            bias=_substitute_latent_expr(expr.bias, latent, value),
            sources=tuple(
                LetAffineSource(
                    value=_substitute_latent_expr(source.value, latent, value),
                    width=source.width,
                )
                for source in expr.sources
            ),
            row_offset=expr.row_offset,
            rows=expr.rows,
            transform=expr.transform,
        )
    if isinstance(expr, LetExprList):
        return LetExprList(items=_substitute_latent_exprs(expr.items, latent, value))
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute_latent_expr(expr.receiver, latent, value),
            method=expr.method,
            args=_substitute_latent_exprs(expr.args, latent, value),
        )
    if isinstance(expr, LetExprLambda):
        if expr.param == latent:
            return expr
        return LetExprLambda(
            param=expr.param,
            body=_substitute_latent_expr(expr.body, latent, value),
        )
    if isinstance(expr, LetExprFactor):
        if any(binder.var == latent for binder in expr.binders):
            return expr
        return LetExprFactor(
            binders=expr.binders,
            body=(
                None
                if expr.body is None
                else _substitute_latent_expr(expr.body, latent, value)
            ),
            cases=tuple(
                LetFactorCase(
                    label=case.label,
                    value=_substitute_latent_expr(case.value, latent, value),
                    line=case.line,
                    col=case.col,
                )
                for case in expr.cases
            ),
        )
    if isinstance(expr, (LetExprLiteral, LetExprString)):
        # Leaf constants: no name to rewrite.
        return expr
    raise UnsupportedConstruct(
        "qvr-renderer",
        [f"marginalize:expr:{type(expr).__name__}"],
    )


def _substitute_latent_exprs(
    exprs: tuple[LetExprNode, ...], latent: str, value: IRArgNumber
) -> tuple[LetExprNode, ...]:
    return tuple(_substitute_latent_expr(expr, latent, value) for expr in exprs)


# ---------------------------------------------------------------------------
# Runtime-helper graft predicate.
# ---------------------------------------------------------------------------


def ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any draw in `body` reads `family`.

    Covers [`IRSample`][quivers.transpile.ir.IRSample],
    [`IRObserve`][quivers.transpile.ir.IRObserve], and the latent draw
    of an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] itself
    before descending into that block's scope.

    Renderers graft a runtime helper for a family only when this
    predicate fires, so a marginalize whose *latent* needs the helper
    (`marginalize z <- ContinuousBernoulli(pi)` on a target that ships
    no continuous Bernoulli) must be visible here; testing the scope
    alone leaves the emitted program calling a name it never defines.
    """
    for node in body:
        if isinstance(node, (IRSample, IRObserve)) and node.family == family:
            return True
        if isinstance(node, IRMarginalize) and (
            node.family == family or ir_uses_family(node.scope, family)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Host integer inputs (used by index-aware renderers to identify the
# integer-typed data inputs a program subscripts a plate with).
# ---------------------------------------------------------------------------


def host_integer_input_names(ir: IRProgram) -> frozenset[str]:
    """Return the names of every :class:`IRDataInput` whose constraint
    is integer-typed (:class:`CSNonnegativeInteger` or
    :class:`CSIntegerInterval`).

    These are the exogenous covariates a program subscripts a plate
    with (``item_idx``, ``cat_idx``, ``out_idx``). Renderers consult
    the set once per render to discriminate index covariates from
    ordinary integer observations.
    """
    integer_kinds = (CSNonnegativeInteger, CSIntegerInterval)
    return frozenset(
        inp.name for inp in ir.inputs if isinstance(inp.constraint, integer_kinds)
    )


# ---------------------------------------------------------------------------
# Per-family argument reorder / reparameterisation for the 1-based
# BUGS-family targets (JAGS, BUGS) whose distribution call convention
# differs from the QVR / torch parameterisation.
# ---------------------------------------------------------------------------


def reorder_negbin_args(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape ``NegativeBinomial(total_count, probs)`` into JAGS /
    BUGS' ``dnegbin(prob, size)`` argument order.

    torch's ``NegativeBinomial(total_count=r, probs=p)`` scores
    ``x`` failures before ``r`` successes with per-trial success
    probability ``p``, which equals ``dnbinom(x; size = r,
    prob = 1 - p)`` in the BUGS / JAGS ``dnegbin(prob, size)``
    convention. This swaps the two arguments and complements the
    probability (``one_minus``) so the emitted call is
    ``dnegbin(1 - probs, total_count)``.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    return (
        (
            IRArgTransform(inner=by_name["probs"], transform="one_minus"),
            by_name["total_count"],
        ),
        ("prob", "size"),
    )


def reorder_weibull_args(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape ``Weibull(scale, concentration)`` into JAGS / BUGS'
    ``dweib(v, lambda)`` argument order.

    torch's ``Weibull(scale = s, concentration = k)`` has density
    ``(k / s) (t / s)^(k-1) exp(-(t / s)^k)``; JAGS / BUGS'
    ``dweib(v, lambda)`` has density
    ``v lambda t^(v-1) exp(-lambda t^v)``. Matching the two gives
    ``v = k`` and ``lambda = s^(-k)``. This reorders to
    ``(concentration, scale)`` and wraps the scale in the
    ``pow_neg`` transform (``pow(scale, -concentration)``) so the
    emitted call is ``dweib(concentration, pow(scale,
    -concentration))``.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    return (
        (
            by_name["concentration"],
            IRArgTransform(
                inner=by_name["scale"],
                transform="pow_neg",
                operand=by_name["concentration"],
            ),
        ),
        ("shape", "rate"),
    )


def mixture_normal_components(
    target: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> tuple[IRArg, IRArg, IRArg]:
    """Return the `(weights, loc, scale)` args of a `MixtureNormal` call.

    `MixtureNormal(weights, loc, scale)` denotes the K-component
    Gaussian mixture whose per-row density is

        p(y) = sum_k weights[k] * Normal(y; loc[k], scale[k]),

    with `K = len(weights)` and the component axis last on all three
    parameters. Every target renderer needs the same three arguments in
    the same roles, whether it spells the mixture as a native
    mixture distribution or as an explicit log-sum-exp, so the
    extraction and its shape contract live here rather than once per
    renderer.

    Raises when the call does not carry exactly those three names, so a
    parameterisation drift surfaces as a precise transpile gap rather
    than as a silently mis-ordered emission.
    """
    expected = ("weights", "loc", "scale")
    by_name = dict(zip(arg_names, args, strict=False))
    if len(args) != len(expected) or tuple(arg_names) != expected:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                f"family:MixtureNormal:arity: expected args "
                f"{expected}, got {tuple(arg_names)}"
            ],
        )
    return (by_name["weights"], by_name["loc"], by_name["scale"])


def mixture_component_count(target: str, weights: IRArg, declared: Plate | None) -> int:
    """Return `K`, the number of components a `MixtureNormal` mixes.

    Read off the declared shape of the weight vector, which is the only
    place the count is available at transpile time. Targets that unroll
    the mixture into an explicit sum need `K` as a compile-time integer,
    and a guessed count would score a different number of components
    than the source names, so a weight argument that is not a bare
    reference to a single statically-sized axis raises.
    """
    if isinstance(weights, IRArgRef) and not weights.indices:
        if declared is not None:
            dims = declared.event_dims or declared.batch_dims
            if len(dims) == 1 and isinstance(dims[0], DimStatic):
                return dims[0].size
    raise UnsupportedConstruct(
        f"qvr-{target}",
        [
            "family:MixtureNormal:unknown-component-count: the weight "
            "argument's declared shape does not name a single static "
            "component axis"
        ],
    )


# ---------------------------------------------------------------------------
# Structural invariants every renderer checks before emission.
# ---------------------------------------------------------------------------


def assert_no_dangling_refs(ir: IRProgram) -> None:
    """Raise if the IR contains an `IRArgRef` whose name is not bound
    by a declaration, a previous step, or an input."""
    declared: set[str] = {inp.name for inp in ir.inputs}
    # ``__row_var__`` is the sentinel Lower writes into observe arg
    # threads to mark "the renderer's per-row loop variable; name
    # bound at render time". Each renderer substitutes the sentinel
    # for its actual loop variable (Stan ``n``, BUGS ``n``, NumPyro's
    # implicit plate index, ...). It is a structural marker, not a
    # true free name, so it does not count as dangling.
    declared.add("__row_var__")
    _walk_for_refs(ir.body, declared, ir.name)


def _walk_for_refs(
    body: tuple[IRNode, ...], declared: set[str], program_name: str
) -> None:
    for node in body:
        if isinstance(node, (IRSample, IRObserve, IRMarginalize)):
            for arg in node.args:
                _check_arg_refs(arg, declared, program_name)
        if isinstance(node, (IRSample, IRObserve, IRDeterministic, IRCall)):
            declared.add(node.name)
        elif isinstance(node, IRMarginalize):
            declared.add(node.latent)
            _walk_for_refs(node.scope, declared, program_name)
        elif isinstance(node, IRScore):
            declared.add(node.name)


def _check_arg_refs(arg: IRArg, declared: set[str], program_name: str) -> None:
    if isinstance(arg, IRArgRef):
        if arg.name not in declared:
            raise UnsupportedConstruct(
                "qvr-renderer",
                [
                    f"program:{program_name}: arg references "
                    f"undeclared name {arg.name!r}"
                ],
            )
        for idx in arg.indices:
            _check_arg_refs(idx, declared, program_name)
        return
    if isinstance(arg, IRArgBroadcast):
        _check_arg_refs(arg.value, declared, program_name)
        return
    if isinstance(arg, IRArgList):
        for e in arg.elements:
            _check_arg_refs(e, declared, program_name)
        return
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for e in row.elements:
                _check_arg_refs(e, declared, program_name)
        return
    if isinstance(arg, IRArgFamilyRef):
        if arg.name not in declared:
            # Family-ref names resolve via the morphism table, not
            # the bound-name set; renderers tolerate this and
            # consult ctx.morphisms during emission.
            return
    if isinstance(arg, IRArgKernel):
        if arg.x_name not in declared:
            raise UnsupportedConstruct(
                "qvr-renderer",
                [
                    f"program:{program_name}: GP kernel input "
                    f"references undeclared name {arg.x_name!r}"
                ],
            )
        return


def _static_extent(plate: Plate) -> int | None:
    """Number of elements `plate` declares, or None when any axis is
    dynamic.

    Event and batch axes multiply together: what the check below
    compares is how many numbers a binding carries, and a value's
    element count does not care which axes the family treats as its
    event and which as replication.
    """
    total = 1
    for dim in (*plate.event_dims, *plate.batch_dims):
        if not isinstance(dim, DimStatic):
            return None
        total *= dim.size
    return total


def _family_arg_constraints(family: str) -> dict[str, Constraint]:
    """The declared per-argument constraints of `family`.

    Read from the distribution class, never from an instance. Two
    kinds of family have no class-level dict to read: one absent from
    [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META], and
    one whose `arg_constraints` torch declares as a property because
    the support depends on the instance (`Uniform`, `Wishart`). Both
    come back as an empty map, and the callers below read that as
    "the argument ranks of this family are not statically known" and
    decline to judge the call's arguments rather than guess a rank.
    """
    meta = FAMILY_META.get(family)
    if meta is None:
        return {}
    declared = getattr(meta.distribution_class, "arg_constraints", None)
    if not isinstance(declared, dict):
        return {}
    return {
        name: constraint
        for name, constraint in declared.items()
        if isinstance(name, str) and isinstance(constraint, Constraint)
    }


def _mixes_over_components(family: str, arg_names: tuple[str, ...]) -> bool:
    """True when the call carries a simplex-constrained mixing weight.

    A mixture family (`MixtureNormal(weights, loc, scale)`) indexes
    every one of its component parameters by the component axis, so a
    scalar-constrained parameter of such a call is a `K`-wide vector
    whose width answers to the component count rather than to the
    site's own width. The extent agreement the check below asserts
    does not hold for those parameters and is not meant to.
    """
    constraints = _family_arg_constraints(family)
    return any(
        isinstance(constraints[name], _SIMPLEX_CONSTRAINT)
        for name in arg_names
        if name in constraints
    )


def _scalar_valued_arg(family: str, arg_name: str) -> bool:
    """True when `family` takes one number per element in `arg_name`.

    Read off the family's declared `arg_constraints`: a rank-0
    constraint (`Normal.loc` is `Real()`) means the argument holds one
    number per scored element, so its width has to be the site's own
    width. A rank-1 or rank-2 constraint (`Categorical.probs` is
    `Simplex()`) describes a whole event and carries its own axis.
    """
    constraints = _family_arg_constraints(family)
    constraint = constraints.get(arg_name)
    if constraint is None:
        return False
    return int(constraint.event_dim) == 0


class _BindingExtents(dx.Model):
    """Static element counts of the names a program body binds.

    `by_name` maps a bound name to its element count (None when the
    binding's plate carries a dynamic axis). `opaque` names bindings
    whose declared plate does not describe the value a reference to
    them carries: the latent of an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] is pinned to
    one atom per branch inside its scope and is unbound outside it, so
    its width is not the width the reference reads.

    Every field read on a `dx.Model` hands back a copy, so the two
    maps are grown with `bind` / `hide`, which return a fresh
    `_BindingExtents`. Threading the returned value is what makes a
    binding visible to the nodes that follow it.
    """

    by_name: dict[str, int | None] = dx.Field(default_factory=dict)
    opaque: frozenset[str] = dx.Field(default_factory=frozenset)

    def bind(self, name: str, extent: int | None) -> _BindingExtents:
        """This map with `name` bound to `extent`."""
        return _BindingExtents(
            by_name={**self.by_name, name: extent},
            opaque=self.opaque,
        )

    def hide(self, name: str) -> _BindingExtents:
        """This map with `name` marked opaque: bound, but carrying a
        value whose width its declared plate does not describe."""
        return _BindingExtents(
            by_name=self.by_name,
            opaque=self.opaque | {name},
        )


def assert_no_dropped_param_map(ir: IRProgram, target: str) -> None:
    """Reject scalar family arguments whose reference width cannot match the site.

    A declared conditional kernel may carry a learned parameter map from its
    domain to its family heads. Those learned weights do not occur in emitted
    source or sample sites. The IR must thus not reduce such a map to a bare
    reference with a different static width.

    Scalar arguments may broadcast from width one. Indexed references, mixture
    arguments, and values gathered through an observation fibration are
    excluded because their effective widths follow other rules. Families whose
    argument ranks are unavailable are also skipped.
    """
    extents = _BindingExtents()
    for inp in ir.inputs:
        extents = extents.bind(inp.name, _static_extent(inp.plate))
    _walk_for_param_maps(ir.body, extents, ir.name, target)


def _walk_for_param_maps(
    body: tuple[IRNode, ...],
    extents: _BindingExtents,
    program_name: str,
    target: str,
) -> None:
    """Check every site in `body`, growing `extents` as the walk
    passes each binder.

    A marginalize scope is walked under its own extended map: the
    latent it pins is in scope for the branch and out of scope after
    it, so the scope's bindings do not leak into the nodes that
    follow the marginalize.
    """
    for node in body:
        if isinstance(node, (IRSample, IRObserve)):
            _check_node_param_widths(node, extents, program_name, target)
            extents = extents.bind(node.name, _static_extent(node.plate))
        elif isinstance(node, IRDeterministic):
            extents = extents.bind(node.name, _static_extent(node.plate))
        elif isinstance(node, IRMarginalize):
            _walk_for_param_maps(
                node.scope,
                extents.bind(node.latent, _static_extent(node.plate)).hide(node.latent),
                program_name,
                target,
            )
        elif isinstance(node, IRScore):
            extents = extents.bind(node.name, 1)


def _check_node_param_widths(
    node: IRSample | IRObserve,
    extents: _BindingExtents,
    program_name: str,
    target: str,
) -> None:
    """Assert every scalar-valued argument of one site is as wide as
    the site, or raise the dropped-parameter-map diagnostic naming the
    two widths."""
    if isinstance(node, IRObserve) and node.via is not None:
        return
    if _mixes_over_components(node.family, node.arg_names):
        return
    site_extent = _static_extent(node.plate)
    if site_extent is None:
        return
    for arg, arg_name in zip(node.args, node.arg_names, strict=True):
        if not isinstance(arg, IRArgRef) or arg.indices:
            continue
        if not _scalar_valued_arg(node.family, arg_name):
            continue
        if arg.name in extents.opaque:
            continue
        ref_extent = extents.by_name.get(arg.name)
        if ref_extent is None:
            continue
        if ref_extent == 1 or site_extent == 1:
            continue
        if ref_extent == site_extent:
            continue
        raise UnsupportedConstruct(
            target,
            [
                f"param-source:linear:width-mismatch:"
                f"{program_name}:{node.name}:{node.family}:"
                f"{arg_name}:{ref_extent}:{site_extent}",
            ],
        )


def assert_no_lists(ir: IRProgram) -> None:
    """Raise if the IR contains an `IRArgList` /
    [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix] in a position a
    backend does not support.

    The default implementation is permissive: list / matrix args are
    legal in the IR; only renderers that lack a native list / matrix
    literal raise. Per-renderer `assert_no_lists` overrides plug in
    the per-target rejection rule.
    """
    del ir


__all__ = [
    "BlockKind",
    "IRArgTransform",
    "IRMarginalAtom",
    "RenderContext",
    "Renderer",
    "RendererBase",
    "SchemaFragment",
    "assert_no_dangling_refs",
    "assert_no_dropped_param_map",
    "assert_no_lists",
    "ir_uses_family",
    "logit_weighted",
    "mixture_component_count",
    "mixture_normal_components",
    "substitute_latent",
]
