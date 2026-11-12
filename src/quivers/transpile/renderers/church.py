"""Church renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to a
Scheme `panproto.Schema` under the ``scheme``
tree-sitter grammar.

The Church idiom for a probabilistic program is a top-level
``(define (model ...) ...)``: every sample step is a
``(define <name> (map (lambda (m_<axis>) (sample <dist>)) (iota N)))``
form for each batch axis; observed steps are
``(for-each (lambda (m_<axis>) (observe <dist> (list-ref <obs>
m_<axis>))) (iota N))``, bound under a name nothing reads so a
definition may follow them in the body. The IR's
[`IRDataInput`][quivers.transpile.ir.IRDataInput] entries become the
model's formal parameter list (Scheme has no separate declaration
block; the function header carries the inputs).

Church has no single canonical interpreter, so every emit grafts the
self-contained reference runtime at
[`runtime_church.scm`][quivers.transpile.runtime_church] above the
model form: it defines ``sample`` / ``observe`` / ``factor`` on a
distribution-object protocol plus one correctly parameterised
distribution constructor per family, making each emitted program a
complete Scheme module.

The renderer reads
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] for each
family's Church distribution name (`target_names["church"]`), reuses
[`RendererBase`][quivers.transpile.renderers.RendererBase] for
the IR walk and the atom enumeration that integrates an
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] latent out
through the runtime's ``log-sum-exp`` and ``factor``, and dispatches
the four primitives (`declare`, `sample`, `marginalize`, `broadcast`)
per `Renderer`. Broadcast scalars emit ``(make-list K <value>)``; list
literals emit ``(list e0 e1 ...)``; matrix literals raise
`UnsupportedConstruct`
with ``arg:matrix-literal`` since Scheme has no canonical matrix
form.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

import panproto

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.renderers._ranks import (
    carried_dims,
    let_expr_rank,
    several_axis_reductions,
)
from quivers.transpile.renderers._rows import (
    scalar_parameter,
    take_row,
    visited_position,
)
from quivers.transpile.renderers._scheme_helpers import render_let_expr_scheme
from quivers.transpile.family_meta import FAMILY_META
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprFactor,
    LetExprNode,
    LetExprVar,
)
from quivers.transpile.ir import (
    ConstraintSpec,
    CSCorrCholesky,
    CSLowerCholesky,
    CSOneHot,
    CSPositiveDefinite,
    CSRealMatrix,
    CSRealVector,
    CSSimplex,
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
    IRQiecValueExpr,
    IRScore,
    LetExprAffineMap,
    Plate,
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
    marginalize_row_rank,
)
from quivers.transpile.renderers._python_helpers import (
    MarginalizeBody,
    marginal_support_size,
    marginal_weight_probs,
    marginalize_body,
    marginalize_index_map,
)
from quivers.transpile.renderers._qiec import (
    graft_scheme_forms,
    render_computations_dynamic,
    scheme_call_source,
    value_argument_source,
    scheme_operations_source,
)


#: The model-body name of the native operation table.
_OPERATIONS = "_qvr-qiec-operations"


_TARGET = "qvr-church"


#: Half-support families whose Church emit injects a ``loc=0``
#: argument and folds the symmetric base distribution onto the
#: nonnegative reals through the runtime ``half`` wrapper.
_HALF_FAMILIES: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

#: The runtime positivity-fold wrapper name.
_HALF_WRAP: str = "half"

#: IR constraints whose support is a vector or matrix; a family with
#: one of these draws its event structure natively, so a sample
#: carrying event axes needs no inner per-event map.
_VECTOR_CONSTRAINTS: tuple[type[ConstraintSpec], ...] = (
    CSRealVector,
    CSSimplex,
    CSRealMatrix,
    CSPositiveDefinite,
    CSCorrCholesky,
    CSLowerCholesky,
    CSOneHot,
)


def _index_event_axes(
    arg: IRArg, event_dims: tuple[Dim, ...], plates: Mapping[str, Plate]
) -> IRArg:
    """Read a family argument at the event-axis loops it carries.

    A reference whose value carries one of an observe's event axes (a
    gathered row ``mu[z]``, or a vector over that axis) is indexed at
    the loop over each such axis, in the order the value carries them,
    so the scalar family scores one coordinate per iteration.

    Parameters
    ----------
    arg : IRArg
        The argument.
    event_dims : tuple[Dim, ...]
        The observe's event axes, each looped as ``m_<axis>``.
    plates : Mapping[str, Plate]
        The plate of each name bound so far.

    Returns
    -------
    IRArg
        The argument, with one trailing index per event axis it carries.
    """
    if not isinstance(arg, IRArgRef):
        return arg
    looped = {str(dim.name) for dim in event_dims}
    extra = tuple(
        IRArgRef(name=f"m_{dim.name}")
        for dim in carried_dims(arg, plates)
        if str(dim.name) in looped
    )
    if not extra:
        return arg
    return IRArgRef(name=arg.name, indices=(*arg.indices, *extra))


def _read_event_at_position(
    arg: IRArg, plates: Mapping[str, Plate], loops: Mapping[str, str]
) -> IRArg:
    """Read a value of event axes only at the batch loops visiting them.

    A reference bound on batch axes is threaded through the plate's
    row index by the alignment pass; one bound as a single event (a
    vector drawn ``[over=Feat]``) has no batch axis to align, and is
    read here at the loop over each of its axes.

    Parameters
    ----------
    arg : IRArg
        The argument.
    plates : Mapping[str, Plate]
        The plate of each name bound so far.
    loops : Mapping[str, str]
        Each of the site's batch axes mapped to its loop variable.

    Returns
    -------
    IRArg
        The argument indexed at the loops when it is an unindexed
        reference to a value of event axes only, each a visited axis;
        else `arg` unchanged.
    """
    if not isinstance(arg, IRArgRef) or arg.indices:
        return arg
    plate = plates.get(arg.name)
    if plate is None or plate.batch_dims:
        return arg
    visited = visited_position(plate, loops)
    if not visited:
        return arg
    return IRArgRef(name=arg.name, indices=tuple(IRArgRef(name=lv) for lv in visited))


#: The runtime constructor of each family call written through an
#: alternative parameterization: a Bernoulli given by its logits scores
#: each outcome by the log-sigmoid of the signed logits.
_LOGIT_SYMBOLS: dict[tuple[str, tuple[str, ...]], str] = {
    ("Bernoulli", ("logits",)): "flip-logit",
}


def _is_scalar_constraint(constraint: ConstraintSpec) -> bool:
    """True iff `constraint` describes a scalar-valued support.

    A scalar family stamped with event axes needs an explicit inner
    map to realise the declared vector; a vector / matrix family does
    not.
    """
    return not isinstance(constraint, _VECTOR_CONSTRAINTS)


def _loop_name(plate: Plate) -> str | None:
    """The per-row loop-variable name for a batched plate.

    Returns ``m_<first-axis>`` (matching the outermost map layer's
    loop variable) or ``None`` for an unbatched plate.
    """
    if not plate.batch_dims:
        return None
    return f"m_{plate.batch_dims[0].name}"


def _denotes_whole_row(expr: LetExprNode, plate: Plate) -> bool:
    """True iff `expr` evaluates to the binding's whole plate row
    rather than to one element of it.

    Two expression kinds do. A
    [`LetExprAffineMap`][quivers.transpile.ir.LetExprAffineMap] is
    the contraction of a whole row block, and a
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] over a
    single binder enumerates the axis itself: the Scheme helper
    renders each as a length-`N` list. Lifting either through the
    per-row map would bind an `N`-by-`N` table under a name every
    reader indexes once.

    The predicate is deliberately narrow. It answers only for a
    rank-1 batch plate, because that is the rank at which a Scheme
    list and a plate row are the same shape; a higher-rank plate
    keeps the per-row lift.
    """
    if len(plate.batch_dims) != 1 or plate.event_dims:
        return False
    if isinstance(expr, LetExprAffineMap):
        return True
    return isinstance(expr, LetExprFactor) and len(expr.binders) == 1


def _plates_align(decl: Plate, surrounding: Plate) -> bool:
    """True iff `decl`'s batch axes and `surrounding`'s agree on their
    common leading axes.

    Purely name-based on `Dim.name`: a ref bound on ``(A,)`` read from
    a site on ``(A, B)`` aligns on ``A``, and so does a ref bound on
    ``(A, B)`` read from a site on ``(A,)``, whose row is then the
    rest of its table (the event a Categorical's `probs` reads).
    """
    if not decl.batch_dims or not surrounding.batch_dims:
        return False
    common = min(len(decl.batch_dims), len(surrounding.batch_dims))
    decl_names = tuple(d.name for d in decl.batch_dims[:common])
    surr_names = tuple(d.name for d in surrounding.batch_dims[:common])
    return decl_names == surr_names


class ChurchRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    Scheme schema in the Church probabilistic-programming idiom.

    Subclass of [`RendererBase`][quivers.transpile.renderers.RendererBase]:
    overrides `render` to wrap the IR walk in a top-level
    ``(define (model <inputs...>) <body...>)`` form, then dispatches
    each [`IRNode`][quivers.transpile.ir.IRNode] through `declare` /
    `sample` / `marginalize` / `broadcast`.
    """

    target: str = _TARGET

    # ----- top-level wrapper -----

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("scheme")

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
        # Snapshot `IRProgram.cards` for the renderer-local cards map
        # threaded through every `_LetExprCtx`; the Scheme let-expr
        # helper consults `ctx.cards` when a factor binder's index
        # references a named QVR object whose cardinality must be
        # resolved at unroll time.
        self._cards: dict[str, int] = dict(ir.cards)
        # Name -> binding plate for every input, sample, observe,
        # deterministic, and marginalize latent. Consulted by the
        # per-row indexing pass so a batched reference emits a
        # `(list-ref name m_<axis>)` access aligned with the
        # surrounding plate.
        self._binding_plates: dict[str, Plate] = {}
        self._input_names: set[str] = {inp.name for inp in ir.inputs}
        for inp in ir.inputs:
            self._binding_plates[inp.name] = inp.plate
        self._collect_binding_plates(ir.body)
        # The active marginalize group plate's batch axes (empty
        # outside a marginalize scope). Refs bound on this plate that
        # are read from an observe on a different plate thread through
        # the observe's `via` index map.
        self._group_plate_axes: tuple[str, ...] = ()
        # The active observe's `via` index map name (None otherwise).
        self._current_via: str | None = None
        self._module = ir.module
        # Whether the model body has bound its native operation table.
        self._operations_bound = False
        # The names checked value arguments have bound in the model body.
        self._value_bindings: set[str] = set()
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = RenderContext(sb=sb, morphisms={}, defines={}, cards=self._cards)
        prog_id = _v(ctx, "prog", "program")
        # Every emitted program uses `sample` / `observe` / `factor`
        # and one distribution constructor per family, all defined in
        # the self-contained Church runtime at
        # [`runtime_church.scm`][quivers.transpile.runtime_church].
        # Graft the whole runtime subtree above the model `(define
        # ...)` form so every referenced identifier resolves through
        # Scheme's top-level binding lookup.
        _graft_runtime_church_helper(ctx, prog_id)

        # `(model <param1> <param2> ...)` -- function signature.
        signature_children: list[str] = [_sym(ctx, "model")]
        for inp in ir.inputs:
            signature_children.append(_sym(ctx, inp.name))
        signature = _list(ctx, tuple(signature_children))

        # Body forms: one or more forms per IR node, plus a trailing
        # return symbol. Most nodes emit one form; `IRMarginalize`
        # expands inline to the latent sample plus each scope step,
        # yielding several sibling forms at the model-body level.
        body_forms: list[str] = []
        for node in ir.body:
            body_forms.extend(self._render_body_forms(ctx, node))
        # Trailing return: Church returns the last expression of the
        # body. `_emit_return` appends one bare symbol per returned
        # name.
        return_form = self._return_form(ctx, ir.body)
        if return_form is not None:
            body_forms.append(return_form)
        if not body_forms:
            # Chez rejects an empty procedure body. QIEC-only and
            # declaration-only modules retain the existing `model` ABI with a
            # harmless scalar result before their named QIEC entry points.
            body_forms.append(_num(ctx, 0.0))

        top_define = _list(ctx, (_sym(ctx, "define"), signature, *body_forms))
        _e(ctx, prog_id, top_define)
        render_computations_dynamic(sb, ir, target=self.target, root=prog_id)
        return sb.build()

    # ----- IR-node dispatch -----

    def _render_body_forms(
        self, ctx: RenderContext, node: IRNode
    ) -> tuple[SchemaFragment, ...]:
        """Render one IR node into one or more sibling body forms.

        Most node kinds emit a single form;
        [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] expands
        to the latent sample plus the scope body inline, returning
        multiple forms so they sit at the same nesting level as the
        surrounding ``(define (model ...) ...)`` body.
        """
        if isinstance(node, IRDataInput):
            # Inputs land in the model signature; no body form.
            return ()
        if isinstance(node, IRSample):
            if node.family == "GP":
                return self._render_gp_forms(ctx, node)
            return (
                self.sample(
                    ctx,
                    node.name,
                    node.family,
                    node.args,
                    node.arg_names,
                    node.constraint,
                    node.plate,
                    observed=False,
                ),
            )
        if isinstance(node, IRObserve):
            return (
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
                ),
            )
        if isinstance(node, IRDeterministic):
            return (self._render_deterministic_form(ctx, node),)
        if isinstance(node, IRScore):
            # Church's score primitive: `(define <name> <expr>)` then
            # `(factor <name>)`, so a later step reads the weight.
            expr_id = render_let_expr_scheme(
                _LetExprCtx(ctx.sb, ctx, self._cards), node.expr
            )
            return (
                _define(ctx, node.name, expr_id),
                _statement(
                    ctx, _list(ctx, (_sym(ctx, "factor"), _sym(ctx, node.name)))
                ),
            )
        if isinstance(node, IRCall):
            return self._render_call_forms(ctx, node)
        if isinstance(node, IRMarginalize):
            return self._render_marginalize_forms(ctx, node)
        if isinstance(node, IRReturn):
            # Handled by `_return_form`; nothing to emit inline.
            return ()
        raise UnsupportedConstruct(_TARGET, [f"node:{type(node).__name__}"])

    def _render_call_forms(
        self, ctx: RenderContext, node: IRCall
    ) -> tuple[SchemaFragment, ...]:
        """The body forms calling a module computation.

        The first call binds the native operation table, under which
        the callee's draws and scores go through the runtime's own
        `sample` and `factor`; an argument that is not a bare name is
        bound first, so the call reads names alone.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        node : IRCall
            The call.

        Returns
        -------
        tuple[SchemaFragment, ...]
            The `define` forms, in order.
        """
        forms: list[SchemaFragment] = []
        names: list[str] = []
        source = ""
        for position, argument in enumerate(node.arguments):
            if isinstance(argument, LetExprVar):
                names.append(argument.name)
                continue
            bound = f"{node.name}_arg{position}"
            if isinstance(argument, IRQiecValueExpr):
                source += value_argument_source(
                    "scheme",
                    "church",
                    bound,
                    argument.value,
                    lambda local: local,
                    body="model",
                    defined=self._value_bindings,
                )
                names.append(bound)
                continue
            expr_id = render_let_expr_scheme(
                _LetExprCtx(ctx.sb, ctx, self._cards), argument
            )
            forms.append(_list(ctx, (_sym(ctx, "define"), _sym(ctx, bound), expr_id)))
            names.append(bound)
        if not self._operations_bound:
            self._operations_bound = True
            source += scheme_operations_source(node, self._module, _OPERATIONS)
        source += scheme_call_source(node, self._module, tuple(names), _OPERATIONS)
        forms.extend(graft_scheme_forms(ctx.sb, source, f"qiec_call_{node.name}"))
        return tuple(forms)

    def _collect_binding_plates(self, body: tuple[IRNode, ...]) -> None:
        """Record the binding plate of every named node in `body`.

        Walks samples, observes, deterministics, and marginalize
        latents (recursing into marginalize scopes) so the per-row
        indexing pass can resolve any referenced name's batch axes.
        """
        for node in body:
            if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
                self._binding_plates[node.name] = node.plate
            elif isinstance(node, IRMarginalize):
                self._binding_plates[node.latent] = node.plate
                self._collect_binding_plates(node.scope)

    def _render_deterministic_form(
        self, ctx: RenderContext, node: IRDeterministic
    ) -> SchemaFragment:
        """Emit ``(define <name> <expr>)`` for a deterministic let.

        When the binding carries a batch plate, the body is lifted
        into a per-row ``(map (lambda (m_<axis>) <body>) (iota N))``
        over each batch dim and every reference bound on the same
        plate is indexed by the loop variable, so the arithmetic runs
        elementwise rather than collapsing the batch axis to a single
        shared value.

        A binding whose expression already denotes the whole row is
        the exception: the per-row lift would wrap a length-`N` list
        in a length-`N` map and bind an `N`-by-`N` table under a name
        every reader indexes once. See
        `_denotes_whole_row`.
        """
        expr = node.expr
        lift = bool(node.plate.batch_dims) and not _denotes_whole_row(expr, node.plate)
        if lift:
            expr = self._take_loop_rows(expr, node.plate)
        ranks = {
            name: len(plate.batch_dims) + len(plate.event_dims)
            for name, plate in self._binding_plates.items()
        }
        found = several_axis_reductions(expr, ranks)
        if found:
            # The runtime's `sum` and its kin fold a flat list, so an
            # argument that still has several axes would not reduce
            # along its last one alone.
            raise UnsupportedConstruct(
                _TARGET,
                [
                    f"let-expr:axis-reduction:church: `{found[0].func}(...)` in "
                    f"{node.name!r} reduces only the last axis of an argument "
                    "with several axes, which the list fold has no form for"
                ],
            )
        expr_id = render_let_expr_scheme(_LetExprCtx(ctx.sb, ctx, self._cards), expr)
        if lift:
            expr_id = self._wrap_in_maps(
                ctx, expr_id, node.plate.batch_dims, axis_suffix=""
            )
        return _list(ctx, (_sym(ctx, "define"), _sym(ctx, node.name), expr_id))

    def _take_loop_rows(self, expr: LetExprNode, plate: Plate) -> LetExprNode:
        """Read a binding's expression at one iteration of its maps.

        The binding is built by one ``(map (lambda (m_<axis>) ...)
        (iota N))`` layer per batch axis, outermost first, and each
        layer's variable is the zero-based position on its axis. Each
        operand carrying the binding's leading remaining axis is read
        at that position through
        [`take_row`][quivers.transpile.renderers._rows.take_row]; an
        expression with fewer axes than remain is the same at every
        position (QVR aligns trailing axes), so that layer selects
        nothing from it.

        Parameters
        ----------
        expr : LetExprNode
            The binding's expression.
        plate : Plate
            The binding's plate.

        Returns
        -------
        LetExprNode
            The expression for the entry the maps reach.
        """
        ranks = {
            name: len(bound.batch_dims) + len(bound.event_dims)
            for name, bound in self._binding_plates.items()
        }
        remaining = len(plate.batch_dims) + len(plate.event_dims)
        for loop_var in _loop_variables(plate.batch_dims, ""):
            if let_expr_rank(expr, ranks) == remaining:
                expr = take_row(expr, LetExprVar(name=loop_var), ranks, "church")
            remaining -= 1
        return expr

    def _render_gp_forms(
        self,
        ctx: RenderContext,
        node: IRSample,
    ) -> tuple[SchemaFragment, ...]:
        """Emit three Scheme forms for a Gaussian-process sample:

        ``(define __gp_mean_<name> (repeat <N> (lambda () 0)))``
        ``(define __gp_cov_<name> <inline-rbf-matrix>)``
        ``(define <name> (multivariate-gaussian __gp_mean_<name>
                                                  __gp_cov_<name>))``

        The inline-rbf-matrix is a nested `map` over the grid index
        producing the N-by-N covariance matrix:
        ``K[i,j] = exp(-0.5 * (x[i]-x[j])^2 / length_scale^2)``
        plus diagonal jitter via `(if (= i j) <jitter> 0)`.
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                _TARGET,
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                _TARGET,
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # (define __gp_mean_<name> (repeat N (lambda () 0)))
        zero_lambda = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, ()),
                _num(ctx, 0),
            ),
        )
        mean_form = _list(
            ctx,
            (
                _sym(ctx, "define"),
                _sym(ctx, mean_name),
                _list(
                    ctx,
                    (_sym(ctx, "repeat"), _num(ctx, n), zero_lambda),
                ),
            ),
        )

        # (define __gp_cov_<name>
        #   (map (lambda (i)
        #          (map (lambda (j)
        #                 (+ (exp (- (* 0.5 (* (- (list-ref x i) (list-ref x j))
        #                                       (- (list-ref x i) (list-ref x j))))
        #                            (* ls ls)))
        #                    (if (= i j) jitter 0)))
        #               (iota N)))
        #        (iota N)))
        def diff_form() -> str:
            return _list(
                ctx,
                (
                    _sym(ctx, "-"),
                    _list(ctx, (_sym(ctx, "list-ref"), _sym(ctx, x), _sym(ctx, "i"))),
                    _list(ctx, (_sym(ctx, "list-ref"), _sym(ctx, x), _sym(ctx, "j"))),
                ),
            )

        diff_sq = _list(
            ctx,
            (_sym(ctx, "*"), diff_form(), diff_form()),
        )
        # arg to exp: (- (/ (* 0.5 diff_sq) (* ls ls)))
        ls_sq = _list(
            ctx,
            (_sym(ctx, "*"), _num(ctx, ls), _num(ctx, ls)),
        )
        scaled = _list(
            ctx,
            (
                _sym(ctx, "/"),
                _list(
                    ctx,
                    (_sym(ctx, "*"), _num(ctx, 0.5), diff_sq),
                ),
                ls_sq,
            ),
        )
        neg_scaled = _list(
            ctx,
            (_sym(ctx, "-"), scaled),
        )
        exp_call = _list(
            ctx,
            (_sym(ctx, "exp"), neg_scaled),
        )
        jitter_if = _list(
            ctx,
            (
                _sym(ctx, "if"),
                _list(
                    ctx,
                    (_sym(ctx, "="), _sym(ctx, "i"), _sym(ctx, "j")),
                ),
                _num(ctx, jitter),
                _num(ctx, 0),
            ),
        )
        entry = _list(
            ctx,
            (_sym(ctx, "+"), exp_call, jitter_if),
        )
        inner_lambda = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, (_sym(ctx, "j"),)),
                entry,
            ),
        )
        iota_n_inner = _list(
            ctx,
            (_sym(ctx, "iota"), _num(ctx, n)),
        )
        inner_map = _list(
            ctx,
            (_sym(ctx, "map"), inner_lambda, iota_n_inner),
        )
        outer_lambda = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, (_sym(ctx, "i"),)),
                inner_map,
            ),
        )
        iota_n_outer = _list(
            ctx,
            (_sym(ctx, "iota"), _num(ctx, n)),
        )
        outer_map = _list(
            ctx,
            (_sym(ctx, "map"), outer_lambda, iota_n_outer),
        )
        cov_form = _list(
            ctx,
            (
                _sym(ctx, "define"),
                _sym(ctx, cov_name),
                outer_map,
            ),
        )
        # (define <name>
        #   (map (lambda (m_<axis>) ...)
        #        (sample (multivariate-gaussian __gp_mean_<name>
        #                                        __gp_cov_<name>))))
        # The GP realisation is a genuine random site, so the draw is
        # routed through `sample`; any plate batch dims wrap it in a
        # per-row map.
        draw_form = _list(
            ctx,
            (
                _sym(ctx, "sample"),
                _list(
                    ctx,
                    (
                        _sym(ctx, "multivariate-gaussian"),
                        _sym(ctx, mean_name),
                        _sym(ctx, cov_name),
                    ),
                ),
            ),
        )
        body = self._wrap_in_maps(ctx, draw_form, node.plate.batch_dims, axis_suffix="")
        sample_form = _list(ctx, (_sym(ctx, "define"), _sym(ctx, node.name), body))
        return (mean_form, cov_form, sample_form)

    def _render_marginalize_forms(
        self, ctx: RenderContext, node: IRMarginalize
    ) -> tuple[SchemaFragment, ...]:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to the sibling forms that integrate its latent out.

        One scored copy of the scope per atom of the latent's finite
        support, then a reduction across the atoms whose result the
        runtime `factor` adds to the log-weight:

        ```scheme
        (define (__marg_z_atom_0)
          (map (lambda (m_Resp)
                 (dist-score (gaussian (list-ref mu 0) 1) (list-ref y m_Resp)))
               (iota 6)))
        (define __marg_z_0 (__marg_z_atom_0))
        ...
        (define __marg_z_w (broadcast1 log probs))
        (define __marg_z_w_0 (take-last __marg_z_w 0))
        (define __marg_z_t_0 (+ __marg_z_w_0 (group-sums __marg_z_0 idx 3)))
        ...
        (define __marg_z (log-sum-exp (list __marg_z_t_0 __marg_z_t_1)))
        (factor (sum-leaves __marg_z))
        ```

        Each atom's deterministic bindings live inside that atom's own
        thunk, which gives them a scope: a Scheme body admits one
        definition per name. A grouped block keys its accumulator by
        group, so the rows the index map sends to one group are
        summed before the reduction; an ungrouped block shares one
        latent across the body's rows, so their log-likelihoods are
        summed to a scalar first. No site is declared for the latent:
        the atoms replace it, and the emitted program denotes the same
        measure the QVR reference integrates.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        node : IRMarginalize
            The block.

        Returns
        -------
        tuple[SchemaFragment, ...]
            The sibling forms, in order.
        """
        refuse_ungrouped_row_marginalize(_TARGET, node)
        plates = dict(self._binding_plates)
        atoms = self.marginal_atoms(
            node,
            support_size=marginal_support_size(node),
        )
        raw = marginalize_body(node.scope, latent=node.latent, target=self.target)
        prefix = f"__marg_{node.latent}"
        prev_group = self._group_plate_axes
        self._group_plate_axes = tuple(str(d.name) for d in node.plate.batch_dims)
        out: list[SchemaFragment] = []
        term_names: list[str] = []
        try:
            for position, atom in enumerate(atoms):
                out.extend(self._atom_forms(ctx, node, atom, prefix, position))
                term_names.append(f"{prefix}_{position}")
            weight_names, weight_forms = self._atom_weight_forms(
                ctx, node, raw, atoms, prefix, plates
            )
            out.extend(weight_forms)
        finally:
            self._group_plate_axes = prev_group
        index_map = marginalize_index_map(
            node,
            raw.observe,
            atoms[0].weight_args,
            atoms[0].weight_arg_names,
            name_plates=plates,
            target=self.target,
        )
        accumulate_rows = marginalize_row_rank(node) > 0
        shifted: list[str] = []
        for position, (weight, term) in enumerate(
            zip(weight_names, term_names, strict=True)
        ):
            term_form: SchemaFragment = _sym(ctx, term)
            if index_map is not None:
                via, group = index_map
                term_form = _list(
                    ctx,
                    (
                        _sym(ctx, "group-sums"),
                        term_form,
                        _sym(ctx, via),
                        self._dim_size_form(ctx, group),
                    ),
                )
            if accumulate_rows:
                term_form = _list(ctx, (_sym(ctx, "sum-leaves"), term_form))
            name = f"{prefix}_t_{position}"
            out.append(
                _define(
                    ctx,
                    name,
                    _list(ctx, (_sym(ctx, "+"), _sym(ctx, weight), term_form)),
                )
            )
            shifted.append(name)
        out.append(
            _define(
                ctx,
                prefix,
                _list(
                    ctx,
                    (
                        _sym(ctx, "log-sum-exp"),
                        _list(
                            ctx, (_sym(ctx, "list"), *(_sym(ctx, n) for n in shifted))
                        ),
                    ),
                ),
            )
        )
        out.append(
            _statement(
                ctx,
                _list(
                    ctx,
                    (
                        _sym(ctx, "factor"),
                        _list(ctx, (_sym(ctx, "sum-leaves"), _sym(ctx, prefix))),
                    ),
                ),
            )
        )
        return tuple(out)

    def _atom_forms(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
        atom: IRMarginalAtom,
        prefix: str,
        position: int,
    ) -> tuple[SchemaFragment, SchemaFragment]:
        """One atom's scope log-density as a thunk and its binding.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        node : IRMarginalize
            The block.
        atom : IRMarginalAtom
            The atom, with the latent pinned in its scope.
        prefix : str
            The block's naming prefix.
        position : int
            The atom's position in the support.

        Returns
        -------
        tuple[SchemaFragment, SchemaFragment]
            The thunk's definition and the binding of its value.
        """
        scored = marginalize_body(atom.scope, latent=node.latent, target=self.target)
        body: list[SchemaFragment] = []
        for det in scored.deterministics:
            body.extend(self._render_body_forms(ctx, det))
        observe = scored.observe
        dist_form = self._dist_form(
            ctx,
            observe.family,
            observe.args,
            observe.plate,
            observe.via,
            arg_names=observe.arg_names,
        )
        body.append(self._wrap_score_map(ctx, observe.name, dist_form, observe.plate))
        thunk = f"{prefix}_atom_{position}"
        definition = _list(
            ctx,
            (_sym(ctx, "define"), _list(ctx, (_sym(ctx, thunk),)), *body),
        )
        binding = _define(ctx, f"{prefix}_{position}", _list(ctx, (_sym(ctx, thunk),)))
        return definition, binding

    def _atom_weight_forms(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
        raw: MarginalizeBody,
        atoms: tuple[IRMarginalAtom, ...],
        prefix: str,
        plates: dict[str, Plate],
    ) -> tuple[tuple[str, ...], tuple[SchemaFragment, ...]]:
        """Bind one log-weight name per atom.

        A `Bernoulli` atom set weights the atoms 0 and 1 by
        `log(1 - p)` and `log(p)`, both shaped like the probability
        itself. A `Categorical` atom set reads the class axis of the
        probability tensor, which `take-last` slices however many
        grouping axes sit above it; a prior read through an index map
        is gathered one row per observation first.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        node : IRMarginalize
            The block.
        raw : MarginalizeBody
            The block's scope with its latent still free.
        atoms : tuple[IRMarginalAtom, ...]
            The atoms.
        prefix : str
            The block's naming prefix.
        plates : dict[str, Plate]
            The plate of every bound name.

        Returns
        -------
        tuple[tuple[str, ...], tuple[SchemaFragment, ...]]
            The weight names, one per atom, and the forms binding them.

        Raises
        ------
        UnsupportedConstruct
            If the atom set's weight family has no log-weight form.
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
        log_form = _sym(ctx, "log")
        if logit_weighted(atoms[0]):
            # From the logits `l` the atoms weigh `log sigmoid(-l)` and
            # `log sigmoid(l)`, which the runtime's `qvr-log-sigmoid`
            # computes in log space.
            zero = f"{prefix}_w_0"
            one = f"{prefix}_w_1"
            negated = _list(ctx, (_sym(ctx, "-"), self._render_arg(ctx, probs)))
            return (zero, one), (
                _define(
                    ctx,
                    zero,
                    _list(ctx, (_sym(ctx, "qvr-log-sigmoid"), negated)),
                ),
                _define(
                    ctx,
                    one,
                    _list(
                        ctx,
                        (_sym(ctx, "qvr-log-sigmoid"), self._render_arg(ctx, probs)),
                    ),
                ),
            )
        if family == "Bernoulli":
            zero = f"{prefix}_w_0"
            one = f"{prefix}_w_1"
            complement = _list(
                ctx, (_sym(ctx, "-"), _num(ctx, 1.0), self._render_arg(ctx, probs))
            )
            bit_forms = (
                _define(
                    ctx,
                    zero,
                    _list(ctx, (_sym(ctx, "broadcast1"), log_form, complement)),
                ),
                _define(
                    ctx,
                    one,
                    _list(
                        ctx,
                        (
                            _sym(ctx, "broadcast1"),
                            _sym(ctx, "log"),
                            self._render_arg(ctx, probs),
                        ),
                    ),
                ),
            )
            return (zero, one), bit_forms
        if family != "Categorical":
            raise UnsupportedConstruct(
                _TARGET,
                [
                    f"marginalize:weight-family:{family}: no Church "
                    f"log-weight form for this atom set"
                ],
            )
        log_probs = f"{prefix}_w"
        if isinstance(probs, IRArgRef) and probs.indices:
            gathered = _list(
                ctx,
                (
                    _sym(ctx, "map"),
                    _list(
                        ctx,
                        (
                            _sym(ctx, "lambda"),
                            _list(ctx, (_sym(ctx, "i"),)),
                            _list(
                                ctx,
                                (
                                    _sym(ctx, "list-ref"),
                                    _sym(ctx, probs.name),
                                    _sym(ctx, "i"),
                                ),
                            ),
                        ),
                    ),
                    self._render_arg(ctx, probs.indices[0]),
                ),
            )
            weight_form = _list(ctx, (_sym(ctx, "broadcast1"), log_form, gathered))
        else:
            weight_form = _list(
                ctx, (_sym(ctx, "broadcast1"), log_form, self._render_arg(ctx, probs))
            )
        forms: list[SchemaFragment] = [_define(ctx, log_probs, weight_form)]
        names: list[str] = []
        for position in range(len(atoms)):
            name = f"{prefix}_w_{position}"
            forms.append(
                _define(
                    ctx,
                    name,
                    _list(
                        ctx,
                        (
                            _sym(ctx, "take-last"),
                            _sym(ctx, log_probs),
                            _num(ctx, float(position)),
                        ),
                    ),
                )
            )
            names.append(name)
        return tuple(names), tuple(forms)

    def _return_form(
        self, ctx: RenderContext, body: tuple[IRNode, ...]
    ) -> SchemaFragment | None:
        for node in body:
            if isinstance(node, IRReturn):
                names = node.names
                if not names:
                    return None
                if len(names) == 1:
                    return _sym(ctx, names[0])
                return _list(
                    ctx,
                    (_sym(ctx, "list"), *(_sym(ctx, n) for n in names)),
                )
        return None

    # ----- the four dispatch points -----

    def declare(
        self,
        ctx: RenderContext,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """Church has no separate declaration block.

        Sample / observe bind their result names directly via
        ``(define name (map ...))``; data inputs land in the model
        signature in `render`. This dispatch is a no-op for every
        block.

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
        via: str | None = None,
    ) -> SchemaFragment:
        """Emit the per-batch-axis sample / observe form.

        For a non-observed step with N batch dims:

        ``(define <name>
            (map (lambda (m_<a0>)
                   (map (lambda (m_<a1>) (sample <dist>))
                        (iota <B1>)))
                 (iota <B0>)))``

        For an observed step: a ``for-each`` over the observation
        plate, with ``(observe <dist> (list-ref <obs> m))`` inside
        each iteration.

        Half-support families (`HalfNormal`, `HalfCauchy`) inject a
        ``loc=0`` argument and wrap the base distribution in the
        runtime ``half`` fold; references bound on an aligned plate
        are indexed by the per-row loop variable; and a scalar family
        stamped with event axes (via ``over=``) wraps an inner map
        over the event dims so each draw is the declared vector.

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
            The index map an observe gathers its rows through, or ``None``.

        Returns
        -------
        SchemaFragment
            The emitted vertex id, or ``""`` when nothing is emitted.
        """
        # A scalar family observed over event axes scores each event
        # coordinate on its own: the observe loops those axes too, and
        # every argument carrying one of them reads it at that loop.
        event_loops = (
            plate.event_dims
            if observed and plate.event_dims and _is_scalar_constraint(constraint)
            else ()
        )
        if event_loops:
            args = tuple(
                _index_event_axes(arg, event_loops, self._binding_plates)
                for arg in args
            )
        # A rank-0 parameter filled by one vector over the site's batch
        # axes (a `z` drawn over `Feat` as the location of a `Normal`
        # replicated over `Feat`) reads its entry at each position.
        loops = _loop_variables(plate.batch_dims, "")
        if len(set(loops)) == len(loops):
            visiting = {
                str(dim.name): loop
                for dim, loop in zip(plate.batch_dims, loops, strict=True)
            }
            args = tuple(
                _read_event_at_position(arg, self._binding_plates, visiting)
                if scalar_parameter(family, arg_name)
                else arg
                for arg, arg_name in zip(args, arg_names, strict=False)
            )
        dist_form = self._dist_form(ctx, family, args, plate, via, arg_names=arg_names)
        if observed:
            return _statement(
                ctx, self._wrap_observe(ctx, name, dist_form, plate, event_loops)
            )
        return self._wrap_sample_define(ctx, name, dist_form, plate, constraint)

    def _dist_form(
        self,
        ctx: RenderContext,
        family: str,
        args: tuple[IRArg, ...],
        plate: Plate,
        via: str | None,
        *,
        arg_names: tuple[str, ...] = (),
    ) -> SchemaFragment:
        """The distribution form of one site, its arguments indexed
        against the site's plate.

        Half-support families (`HalfNormal`, `HalfCauchy`) inject a
        ``loc=0`` argument and wrap the base distribution in the
        runtime ``half`` fold; references bound on an aligned plate
        are indexed by the per-row loop variable, and references bound
        on the group plate of an enclosing block are gathered through
        `via`.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        family : str
            The distribution family.
        args : tuple[IRArg, ...]
            The family's arguments.
        plate : Plate
            The site's plate.
        via : str | None
            The index map an observe gathers group-plate values through.

        Returns
        -------
        SchemaFragment
            The distribution form.

        Raises
        ------
        UnsupportedConstruct
            If the family has no Church constructor.
        """
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(_TARGET, [f"family:{family}"])
        target_symbol = meta.target_names.get("church")
        if target_symbol is None:
            raise UnsupportedConstruct(_TARGET, [f"family:{family}:no-church-target"])
        target_symbol = _LOGIT_SYMBOLS.get((family, arg_names), target_symbol)

        if family in _HALF_FAMILIES:
            # `HalfNormal(scale)` -> `(half (gaussian 0 scale))`;
            # `HalfCauchy(scale)` -> `(half (cauchy 0 scale))`.
            args = (IRArgNumber(value=0.0), *args)

        loop_name = _loop_name(plate)
        if loop_name is not None:
            prev_via = self._current_via
            self._current_via = via
            try:
                args = tuple(
                    self._substitute_ref_indexing(a, plate, loop_name) for a in args
                )
            finally:
                self._current_via = prev_via

        dist_form = self._build_dist_call(ctx, target_symbol, args)
        if family in _HALF_FAMILIES:
            dist_form = _list(ctx, (_sym(ctx, _HALF_WRAP), dist_form))
        return dist_form

    def _substitute_ref_indexing(
        self, arg: IRArg, plate: Plate, loop_name: str
    ) -> IRArg:
        """Thread the surrounding plate's row index into any ref whose
        binding plate aligns with `plate`, or through the active
        observe's `via` index map when the ref is bound on the group
        plate.

        Returns a rewritten IR-arg tree; the rewrite prepends a loop
        index to the aligned ref's index list so the emitted
        ``(list-ref name m_<axis>)`` selects the row, then defers to
        the ordinary ref renderer.
        """
        if isinstance(arg, IRArgRef):
            new_indices = tuple(
                self._substitute_ref_indexing(i, plate, loop_name) for i in arg.indices
            )
            decl = self._binding_plates.get(arg.name)
            if decl is not None and _plates_align(decl, plate):
                # Every leading axis the binding shares with the site's
                # plate is read at the site's loop over it; the site's
                # loops are `m_<axis>` (see `_wrap_observe` and
                # `_wrap_in_maps`), the first one `loop_name`.
                shared = [IRArgRef(name=loop_name)]
                site_names = [str(dim.name) for dim in plate.batch_dims]
                if loop_name == f"m_{site_names[0]}" and len(set(site_names)) == len(
                    site_names
                ):
                    for declared, site in zip(
                        decl.batch_dims[1:], plate.batch_dims[1:], strict=False
                    ):
                        if declared.name != site.name:
                            break
                        shared.append(IRArgRef(name=f"m_{site.name}"))
                return IRArgRef(name=arg.name, indices=(*shared, *new_indices))
            if (
                self._current_via is not None
                and decl is not None
                and tuple(str(d.name) for d in decl.batch_dims)
                == self._group_plate_axes
                and self._group_plate_axes
                and arg.name not in self._input_names
            ):
                via_ref = IRArgRef(
                    name=self._current_via,
                    indices=(IRArgRef(name=loop_name),),
                )
                return IRArgRef(name=arg.name, indices=(via_ref, *new_indices))
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

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to the forms that integrate its latent out.

        The top-level
        [`render`][quivers.transpile.renderers.church.ChurchRenderer.render]
        consumes the multi-form expansion via
        ``_render_marginalize_forms`` so the latent sample and scope
        observes sit as siblings of the surrounding ``(define (model
        ...) ...)`` body. When `marginalize` is invoked as a single
        dispatch point (per the Renderer protocol contract), the
        forms collapse into a ``(begin ...)`` block.

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
        forms = self._render_marginalize_forms(ctx, node)
        if not forms:
            return ""
        if len(forms) == 1:
            return forms[0]
        return _list(ctx, (_sym(ctx, "begin"), *forms))

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """`(make-list K <value>)` for 1-D broadcasts.

        Higher-rank broadcasts raise
        `UnsupportedConstruct`
        with ``arg:broadcast-rank-2+`` (Church has no canonical
        matrix form, so a rank-2 broadcast has no idiomatic
        rendering).

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
        if len(target_shape) != 1:
            raise UnsupportedConstruct(
                _TARGET,
                [f"arg:broadcast-rank-{len(target_shape)}"],
            )
        (k,) = target_shape
        value_form = self._render_arg(ctx, value)
        return _list(
            ctx,
            (_sym(ctx, "make-list"), _num(ctx, float(k)), value_form),
        )

    # ----- helpers -----

    def _build_dist_call(
        self,
        ctx: RenderContext,
        symbol: str,
        args: tuple[IRArg, ...],
    ) -> SchemaFragment:
        """`(<symbol> <arg0> <arg1> ...)`."""
        children = [_sym(ctx, symbol)]
        for arg in args:
            children.append(self._render_arg(ctx, arg))
        return _list(ctx, tuple(children))

    def _render_arg(self, ctx: RenderContext, arg: IRArg) -> SchemaFragment:
        """Render one [`IRArg`][quivers.transpile.ir.IRArg] to a
        Scheme schema fragment.

        Dispatches on the IR variant; consults
        [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] for
        `IRArgFamilyRef` lookups so the wrapper family's inner
        distribution renders inline.
        """
        if isinstance(arg, IRArgNumber):
            return _num(ctx, arg.value)
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
        raise UnsupportedConstruct(_TARGET, [f"arg:{type(arg).__name__}"])

    def _render_ref(self, ctx: RenderContext, arg: IRArgRef) -> SchemaFragment:
        """`x` for a bare reference; `(list-ref x idx)` for one index;
        nested `list-ref` for higher-rank indexing."""
        if not arg.indices:
            return _sym(ctx, arg.name)
        current: SchemaFragment = _sym(ctx, arg.name)
        for idx in arg.indices:
            idx_form = self._render_arg(ctx, idx)
            current = _list(ctx, (_sym(ctx, "list-ref"), current, idx_form))
        return current

    def _render_list(self, ctx: RenderContext, arg: IRArgList) -> SchemaFragment:
        children = [_sym(ctx, "list")]
        for elem in arg.elements:
            children.append(self._render_arg(ctx, elem))
        return _list(ctx, tuple(children))

    def _render_matrix(self, ctx: RenderContext, arg: IRArgMatrix) -> SchemaFragment:
        del ctx, arg
        raise UnsupportedConstruct(_TARGET, ["arg:matrix-literal"])

    def _render_family_ref(
        self, ctx: RenderContext, arg: IRArgFamilyRef
    ) -> SchemaFragment:
        """Resolve an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        through the morphism table and emit the inner distribution
        call inline.

        Wrapper families (`Truncated`, `Mixture`, `Independent`,
        `Transformed`, `LKJCorrelationFactor`) carry their wrapped
        distribution as a morphism name; the morphism table maps that
        name to its `~ Family(...)` clause. Church does not expose
        any of those wrappers natively, so the renderer raises
        `UnsupportedConstruct`
        on absent morphism data; callers wanting truncated /
        composition / rejection-sampling semantics fold them into the
        outer distribution call before lowering.
        """
        del ctx
        raise UnsupportedConstruct(_TARGET, [f"arg:family_ref:{arg.name}"])

    def _wrap_sample_define(
        self,
        ctx: RenderContext,
        name: str,
        dist_form: SchemaFragment,
        plate: Plate,
        constraint: ConstraintSpec,
    ) -> SchemaFragment:
        """`(define <name> <body>)` where `<body>` is the (possibly
        nested) `(map (lambda (m_<axis>) (sample <dist>)) (iota N))`
        wrapping over the plate's batch dims.

        When the family draws a scalar (its IR constraint is a scalar
        support) but the sample carries event axes from an ``over=``
        declaration, an inner map over the event dims wraps the draw
        so each batch element is the declared-length vector rather
        than a single scalar. Families that draw a vector natively
        (Dirichlet, MultivariateNormal, ...) carry the event axes
        intrinsically and need no inner map.
        """
        inner: SchemaFragment = _list(ctx, (_sym(ctx, "sample"), dist_form))
        if plate.event_dims and _is_scalar_constraint(constraint):
            inner = self._wrap_in_maps(ctx, inner, plate.event_dims, axis_suffix="")
        wrapped = self._wrap_in_maps(ctx, inner, plate.batch_dims, axis_suffix="")
        return _list(ctx, (_sym(ctx, "define"), _sym(ctx, name), wrapped))

    def _wrap_in_maps(
        self,
        ctx: RenderContext,
        inner: SchemaFragment,
        batch_dims: tuple[object, ...],
        *,
        axis_suffix: str,
    ) -> SchemaFragment:
        """Wrap `inner` in one `(map (lambda (m_<axis>) ...) (iota N))`
        per batch dim, outermost-first.

        ``axis_suffix`` distinguishes the loop variable across nested
        contexts (e.g. the marginalized latent's per-axis index uses
        ``"_z"`` so the scope's inner observation can rebind without
        shadowing).
        """
        current = inner
        names = _loop_variables(batch_dims, axis_suffix)
        for dim, loop_var in zip(reversed(batch_dims), reversed(names), strict=True):
            current = self._one_map_layer(ctx, current, dim, loop_var)
        return current

    def _one_map_layer(
        self,
        ctx: RenderContext,
        inner: SchemaFragment,
        dim: object,
        loop_var: str,
    ) -> SchemaFragment:
        size_form = self._dim_size_form(ctx, dim)
        lambda_form = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, (_sym(ctx, loop_var),)),
                inner,
            ),
        )
        iota_form = _list(ctx, (_sym(ctx, "iota"), size_form))
        return _list(
            ctx,
            (_sym(ctx, "map"), lambda_form, iota_form),
        )

    def _dim_size_form(self, ctx: RenderContext, dim: object) -> SchemaFragment:
        """Emit the size form for one plate dim.

        Static dims render as their integer cardinality; dynamic dims
        emit ``(length <size_name>)`` so the host can supply runtime
        data without recompilation.
        """
        if isinstance(dim, DimStatic):
            return _num(ctx, float(dim.size))
        if isinstance(dim, DimDynamic):
            return _list(ctx, (_sym(ctx, "length"), _sym(ctx, dim.size_name)))
        raise UnsupportedConstruct(_TARGET, [f"dim:{type(dim).__name__}"])

    def _wrap_observe(
        self,
        ctx: RenderContext,
        obs_name: str,
        dist_form: SchemaFragment,
        plate: Plate,
        event_loops: tuple[Dim, ...] = (),
    ) -> SchemaFragment:
        """`(for-each (lambda (n) (observe <dist> (list-ref <obs> n)))
        (iota (length <obs>)))` over the observation plate.

        For higher-rank batch dims the function nests `for-each` from
        outermost to innermost, followed by one per event axis a scalar
        family scores coordinate by coordinate. For an empty plate the
        form is a single ``(observe <dist> <obs>)`` (no iteration).

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        obs_name : str
            The observed value's name.
        dist_form : SchemaFragment
            The distribution form.
        plate : Plate
            The observe's plate.
        event_loops : tuple[Dim, ...]
            The event axes looped inside the batch axes.

        Returns
        -------
        SchemaFragment
            The observe form.
        """
        loop_dims = (*plate.batch_dims, *event_loops)
        if not loop_dims:
            return _list(
                ctx,
                (
                    _sym(ctx, "observe"),
                    dist_form,
                    _sym(ctx, obs_name),
                ),
            )
        # One `m_<axis>` loop variable per batch dim, matching the
        # loop-variable names the per-row indexing pass threaded into
        # the distribution arguments.
        loop_vars = tuple(f"m_{dim.name}" for dim in loop_dims)
        # Build the indexed reference `(list-ref ... obs)` nested per
        # loop, innermost-first to match the lambda nesting.
        obs_ref: SchemaFragment = _sym(ctx, obs_name)
        for lv in loop_vars:
            obs_ref = _list(ctx, (_sym(ctx, "list-ref"), obs_ref, _sym(ctx, lv)))
        innermost = _list(ctx, (_sym(ctx, "observe"), dist_form, obs_ref))
        # Wrap in nested `for-each` from inner to outer.
        current = innermost
        for lv, dim in zip(
            reversed(loop_vars),
            reversed(loop_dims),
            strict=True,
        ):
            size_form = self._dim_size_form(ctx, dim)
            lambda_form = _list(
                ctx,
                (
                    _sym(ctx, "lambda"),
                    _list(ctx, (_sym(ctx, lv),)),
                    current,
                ),
            )
            iota_form = _list(ctx, (_sym(ctx, "iota"), size_form))
            current = _list(
                ctx,
                (_sym(ctx, "for-each"), lambda_form, iota_form),
            )
        return current

    def _wrap_score_map(
        self,
        ctx: RenderContext,
        obs_name: str,
        dist_form: SchemaFragment,
        plate: Plate,
    ) -> SchemaFragment:
        """`(map (lambda (n) (dist-score <dist> (list-ref <obs> n)))
        (iota <N>))` over the observation plate.

        The per-row log-densities of one marginalize atom, shaped like
        the plate: nested `map`s for higher-rank batch dims, and a
        single ``(dist-score <dist> <obs>)`` for an empty plate.

        Parameters
        ----------
        ctx : RenderContext
            The render context.
        obs_name : str
            The observed data's name.
        dist_form : SchemaFragment
            The distribution form, its arguments indexed per row.
        plate : Plate
            The observation plate.

        Returns
        -------
        SchemaFragment
            The scoring form.
        """
        if not plate.batch_dims:
            return _list(ctx, (_sym(ctx, "dist-score"), dist_form, _sym(ctx, obs_name)))
        loop_vars = tuple(f"m_{dim.name}" for dim in plate.batch_dims)
        obs_ref: SchemaFragment = _sym(ctx, obs_name)
        for lv in loop_vars:
            obs_ref = _list(ctx, (_sym(ctx, "list-ref"), obs_ref, _sym(ctx, lv)))
        current = _list(ctx, (_sym(ctx, "dist-score"), dist_form, obs_ref))
        for lv, dim in zip(
            reversed(loop_vars),
            reversed(plate.batch_dims),
            strict=True,
        ):
            lambda_form = _list(
                ctx,
                (
                    _sym(ctx, "lambda"),
                    _list(ctx, (_sym(ctx, lv),)),
                    current,
                ),
            )
            iota_form = _list(ctx, (_sym(ctx, "iota"), self._dim_size_form(ctx, dim)))
            current = _list(ctx, (_sym(ctx, "map"), lambda_form, iota_form))
        return current


def _loop_variables(dims: tuple[object, ...], suffix: str) -> tuple[str, ...]:
    """One distinct `map` variable per axis, outermost first.

    Two axes of one binding can share a name (a table over `Step` whose
    second axis has the same extent reads as `Step` too), and nested
    lambdas binding one name would let the inner shadow the outer, so a
    repeated name takes a counter.

    Parameters
    ----------
    dims : tuple[object, ...]
        The axes.
    suffix : str
        Appended to every variable, to set a nested context's apart.

    Returns
    -------
    tuple[str, ...]
        The variables.
    """
    seen: dict[str, int] = {}
    names: list[str] = []
    for dim in dims:
        base = f"m_{getattr(dim, 'name', 'i')}{suffix}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return tuple(names)


# ---------------------------------------------------------------------------
# Schema-builder helpers (vertex / edge / literal constructors).
# ---------------------------------------------------------------------------


def _fresh(ctx: RenderContext, prefix: str) -> str:
    ctx.fresh_counter += 1
    return f"{prefix}_{ctx.fresh_counter}"


def _v(ctx: RenderContext, vid: str, kind: str) -> str:
    ctx.sb.vertex(vid, kind)
    return vid


def _e(ctx: RenderContext, src: str, tgt: str, kind: str = "child_of") -> None:
    ctx.sb.edge(src, tgt, kind)


def _sym(ctx: RenderContext, text: str) -> SchemaFragment:
    vid = _v(ctx, _fresh(ctx, "sym"), "symbol")
    ctx.sb.constraint(vid, "literal-value", text)
    return vid


def _num(ctx: RenderContext, value: float) -> SchemaFragment:
    vid = _v(ctx, _fresh(ctx, "num"), "number")
    text = str(int(value)) if value == int(value) else repr(value)
    ctx.sb.constraint(vid, "literal-value", text)
    return vid


def _list(ctx: RenderContext, children: tuple[SchemaFragment, ...]) -> SchemaFragment:
    """A parenthesised Scheme list with `children` in order."""
    lst = _v(ctx, _fresh(ctx, "lst"), "list")
    for child in children:
        if child:
            _e(ctx, lst, child)
    return lst


def _statement(ctx: RenderContext, form: SchemaFragment) -> SchemaFragment:
    """`(define _qvr_stmt_<n> <form>)`: an expression placed among the
    definitions of a body.

    A Scheme body admits definitions before expressions only, so a
    form evaluated for its effect on the log-weight (an `observe`, a
    `factor`) is bound under a name nothing reads, which lets a later
    step still define its own binding.

    Parameters
    ----------
    ctx : RenderContext
        The render context.
    form : SchemaFragment
        The expression form.

    Returns
    -------
    SchemaFragment
        The definition form.
    """
    return _define(ctx, _fresh(ctx, "_qvr_stmt"), form)


def _define(ctx: RenderContext, name: str, value: SchemaFragment) -> SchemaFragment:
    """`(define <name> <value>)`.

    Parameters
    ----------
    ctx : RenderContext
        The render context.
    name : str
        The bound name.
    value : SchemaFragment
        The bound form.

    Returns
    -------
    SchemaFragment
        The definition form.
    """
    return _list(ctx, (_sym(ctx, "define"), _sym(ctx, name), value))


# ---------------------------------------------------------------------------
# Bridge adapter: the
# `render_let_expr_scheme`
# helper consumes a duck-typed context with `.fresh`, `.v`, `.e`,
# `.lit`, `.constraint`, `.cards`, `.target` members. The IR renderer's
# [`RenderContext`][quivers.transpile.renderers.RenderContext] exposes
# only the panproto `SchemaBuilder` and the morphism / let / cards
# tables; this adapter projects the helper's expected surface onto it
# without rebinding the fresh-id counter, so let / score expressions
# render through the same code path as the schema-extraction backend
# and benefit from cards-driven binder-size resolution when the helper
# adds it.
# ---------------------------------------------------------------------------


class _LetExprCtx:
    """Duck-typed adapter exposing the
    `render_let_expr_scheme`
    context protocol (``fresh``, ``v``, ``e``, ``lit``, ``constraint``,
    ``cards``, ``target``) on top of a
    [`RenderContext`][quivers.transpile.renderers.RenderContext]."""

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        owner: RenderContext,
        cards: dict[str, int],
    ) -> None:
        self._sb = sb
        self._owner = owner
        self.cards = cards
        self.target = "church"

    def fresh(self, prefix: str) -> str:
        return _fresh(self._owner, prefix)

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
# Runtime graft: the self-contained Church reference runtime.
#
# Church has no single canonical interpreter, so
# [`runtime_church.scm`][quivers.transpile.runtime_church] defines
# `sample` / `observe` / `factor` on a distribution-object protocol
# plus one correctly parameterised distribution constructor per QVR
# family. The transpile-time graft parses the runtime once at
# module-load through panproto's Scheme tree-sitter grammar; per-render,
# it copies every grafted vertex / constraint / edge into the per-render
# schema (with fresh vertex ids) and attaches the runtime's top-level
# forms as `child_of` of the emitted `program` vertex above the
# `(define (model ...) ...)` form, so every referenced identifier
# resolves through the standard Scheme top-level binding lookup at
# evaluation time.
# ---------------------------------------------------------------------------


_RUNTIME_CHURCH_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_church.scm"
)


def _load_runtime_church_schema() -> tuple[panproto.Schema, str, tuple[str, ...]]:
    """Parse [`runtime_church.scm`][quivers.transpile.runtime_church]
    through panproto's Scheme tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `program` vertex id, and the
    tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `program` so the emit's top-level forms appear in
    the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "scheme",
        _RUNTIME_CHURCH_PATH.read_bytes(),
        str(_RUNTIME_CHURCH_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "program"),
        None,
    )
    if src_id is None:
        raise RuntimeError(f"`program` not found in parse of {_RUNTIME_CHURCH_PATH}")
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


_RUNTIME_CHURCH_SCHEMA, _RUNTIME_CHURCH_PROGRAM_ID, _RUNTIME_CHURCH_TOP_LEVEL = (
    _load_runtime_church_schema()
)


def _church_subtree_vertex_ids(
    schema: panproto.Schema, roots: tuple[str, ...]
) -> set[str]:
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


_RUNTIME_CHURCH_SUBTREE = _church_subtree_vertex_ids(
    _RUNTIME_CHURCH_SCHEMA, _RUNTIME_CHURCH_TOP_LEVEL
)


def _graft_runtime_church_helper(ctx: RenderContext, program_vid: str) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_church.scm` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each top-level
    child as a `child_of` of `program_vid` in source order. The grafted
    forms appear above the model `(define ...)` in the emit so the
    helper bindings are in scope when the model is evaluated.
    """
    src_schema = _RUNTIME_CHURCH_SCHEMA
    subtree = _RUNTIME_CHURCH_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        new = _fresh(ctx, "rc")
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        ctx.sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            ctx.sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            ctx.sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_CHURCH_TOP_LEVEL:
        ctx.sb.edge(program_vid, id_map[child_old], "child_of")


__all__ = ["ChurchRenderer"]
