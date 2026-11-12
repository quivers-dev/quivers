"""[`StanRenderer`][quivers.transpile.renderers.stan.StanRenderer]: IR to Stan source.

The renderer subclasses
[`RendererBase`][quivers.transpile.renderers.RendererBase] and
implements the four dispatch points (`declare`, `sample`,
`marginalize`, `broadcast`) plus the two arg helpers (`_render_list`,
`_render_matrix`). Distribution names live in
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]'s
`target_names["stan"]`; no per-renderer family table. Support
classification dispatches on the predicates exported from
[`ir.py`][quivers.transpile.ir]; the renderer never branches on the
family-name string.

The Stan-specific layout decisions:

* Program block: top-level `program` vertex carrying the five
  standard child blocks (`data`, `parameters`,
  `transformed_parameters`, `model`, `generated_quantities`).
  Blocks are materialised lazily when first written to, so a
  program with no `transformed_parameters` declarations emits no
  empty block.
* Sample / observe steps: an explicit
  `target += <family>_lpdf(<variate> | <args>);` increment rather
  than Stan's `~` sampling notation. `~` drops every term that does
  not depend on a parameter, which makes the program's
  `log_prob(jacobian=False)` differ from the QVR joint by an amount
  that moves with the data; the `_lpdf` / `_lpmf` form keeps all
  normalizing constants, so the emitted program computes the joint
  exactly.
* Sample-step plate loops: every batch dimension on a sample's
  plate becomes a nested `for (m_<axis> in 1:<size>)` loop, with the
  variate and every same-axis-indexed arg rewritten through
  [`substitute_indices`][quivers.transpile.renderers.RendererBase.substitute_indices].
* Truncated families: the retained interval's log-mass is subtracted
  explicitly through `<family>_lcdf` / `<family>_lccdf` and
  `log_diff_exp`, since Stan's `T[low, high]` suffix attaches only to
  a `~` statement.
* Marginalize: per spec, Stan emits
  `log_sum_exp` per-group enumeration. The per-call
  [`finite_enumerable_at_call_site`][quivers.transpile.family_meta.finite_enumerable_at_call_site]
  predicate guards the construct;
  non-finite-support families raise `UnsupportedConstruct` with
  kind `marginalize:non-finite-support:<family>`.
* Broadcast: `rep_vector(<value>, K)` for 1D,
  `rep_matrix(<value>, R, C)` for 2D. The Stan grammar's
  `function_expression` carries the call.
* List / matrix args: `[<e0>, <e1>, ...]'` (with transpose) for
  list literals; `to_matrix({{<row0>}, {<row1>}, ...})` for matrix
  literals.
* IRArgFamilyRef: the referenced morphism's `init_family` is read
  from `RenderContext.morphisms`; Stan-side wrappers emit the inner
  call inline followed by Stan's truncation / wrapper syntax (e.g.
  `normal(0, 1) T[-2, 2]` for `Truncated(base, -2, 2)`).
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable, Mapping

import panproto
import torch.distributions.constraints as _torch_constraints
from torch.distributions.constraints import Constraint

from quivers.dsl.ast_nodes import (
    DrawArg,
    Expr,
    DefineDecl,
    Module,
    MorphismDecl,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprCall,
    LetExprFactor,
    LetExprList,
    LetExprNode,
    LetExprBinOp,
    LetExprLiteral,
    LetExprVar,
)
from quivers.dsl.draw_args import atom_value, bind_family_arguments
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.family_meta import (
    class_index_outcome,
    marginalize_support,
    FAMILY_META,
    FamilyMeta,
)
from quivers.transpile.ir import (
    CSReal,
    ConstraintSpec,
    LetExprAffineMap,
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
    is_int_bit,
    is_int_category,
    is_int_count,
    is_real_corr_chol,
    is_real_cov_matrix,
    is_real_matrix,
    is_real_one_hot,
    is_real_bounded_interval,
    is_real_positive,
    is_real_scalar,
    is_real_simplex,
    is_real_unit_interval,
    is_real_vector,
    real_interval_bounds,
)
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    RenderContext,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    mixture_component_count,
    mixture_normal_components,
)
from quivers.transpile.renderers._qiec import (
    refuse_static_gaps,
    render_computations_static,
    carried_computations,
    has_runtime_computations,
)
from quivers.transpile.renderers._one_based import counter_value, is_class_code_family
from quivers.transpile.renderers._ranks import (
    let_expr_rank,
    name_rank_map,
    several_axis_reductions,
)
from quivers.transpile.renderers._rows import scalar_parameter, take_row
from quivers.transpile.renderers._stan_helpers import (
    _substitute_let_expr,
    render_let_expr_stan,
)


#: Stan spells a distribution's log-density function
#: `<name>_lpdf` when the variate is continuous and `<name>_lpmf`
#: when it is discrete. The keys are the `target_names["stan"]`
#: entries of [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]
#: plus the helper densities grafted from `runtime_stan_functions.stan`
#: (`continuous_bernoulli`, `kumaraswamy`, `log_gamma`, `logit_beta`,
#: `logit_normal`, `matrix_normal`). A Stan sampling name absent from this table has
#: no log-density form, and
#: [`StanRenderer._log_density_name`][quivers.transpile.renderers.stan.StanRenderer]
#: raises rather than emit a `~` statement that would drop constants.
_STAN_LOG_DENSITY_SUFFIX: dict[str, str] = {
    "bernoulli": "lpmf",
    "beta": "lpdf",
    "beta_binomial": "lpmf",
    "binomial": "lpmf",
    "categorical": "lpmf",
    "categorical_logit": "lpmf",
    "bernoulli_logit": "lpmf",
    "binomial_logit": "lpmf",
    "multi_normal_prec": "lpdf",
    "multi_normal_cholesky": "lpdf",
    "cauchy": "lpdf",
    "chi_square": "lpdf",
    "continuous_bernoulli": "lpdf",
    "dirichlet": "lpdf",
    "double_exponential": "lpdf",
    "exponential": "lpdf",
    "gamma": "lpdf",
    "gumbel": "lpdf",
    "inv_gamma": "lpdf",
    "inv_wishart": "lpdf",
    "kumaraswamy": "lpdf",
    "lkj_corr": "lpdf",
    "lkj_corr_cholesky": "lpdf",
    "logistic": "lpdf",
    "log_gamma": "lpdf",
    "logit_beta": "lpdf",
    "logit_normal": "lpdf",
    "lognormal": "lpdf",
    "matrix_normal": "lpdf",
    "multi_normal": "lpdf",
    "neg_binomial_2": "lpmf",
    "normal": "lpdf",
    "ordered_logistic": "lpmf",
    "ordered_probit": "lpmf",
    "pareto": "lpdf",
    "poisson": "lpmf",
    "student_t": "lpdf",
    "uniform": "lpdf",
    "von_mises": "lpdf",
    "weibull": "lpdf",
    "wishart": "lpdf",
}


#: The Stan density that takes each alternative parameterization a family
#: call can name, keyed by family and the parameter that selects it.
_STAN_PARAMETERIZED_NAMES: dict[tuple[str, str], str] = {
    ("Categorical", "logits"): "categorical_logit",
    ("Bernoulli", "logits"): "bernoulli_logit",
    ("Binomial", "logits"): "binomial_logit",
    ("MultivariateNormal", "precision_matrix"): "multi_normal_prec",
    ("MultivariateNormal", "scale_tril"): "multi_normal_cholesky",
}


def _stan_parameterized_name(
    family: str, arg_names: tuple[str, ...], default: str
) -> str:
    """The Stan density for the parameterization a call names.

    Parameters
    ----------
    family : str
        The family.
    arg_names : tuple[str, ...]
        The parameters the call supplies.
    default : str
        The family's Stan density under its canonical parameterization.

    Returns
    -------
    str
        The density that takes the named parameterization.
    """
    for name in arg_names:
        selected = _STAN_PARAMETERIZED_NAMES.get((family, name))
        if selected is not None:
            return selected
    return default


def _is_infinite_bound(bound: IRArg) -> bool:
    """Whether a truncation bound is an infinite literal, i.e. the
    corresponding side of the support is not actually truncated."""
    return isinstance(bound, IRArgNumber) and math.isinf(bound.value)


class _StanLetCtx:
    """Bridge ``RenderContext.sb`` to the
    `render_let_expr_stan`
    helper interface (`vertex`, `edge`, `literal`, `constraint`,
    `fresh`) and carry the object-name -> static-cardinality map
    consulted when unrolling
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor].

    The Stan IR-walk operates over `RenderContext`; the let-expression
    helper expects a small carrier that exposes
    `panproto.SchemaBuilder` operations
    under terse method names. The shim keeps the helper independent
    of any specific renderer class.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        fresh: Callable[[str], str],
        cards: dict[str, int],
        ranks: Mapping[str, int],
        vectors: frozenset[str],
    ) -> None:
        """Initialise the carrier.

        Parameters
        ----------
        sb : panproto.SchemaBuilder
            The schema under construction.
        fresh : Callable[[str], str]
            The renderer's fresh-vertex-id source.
        cards : dict[str, int]
            Each object's static cardinality.
        ranks : Mapping[str, int]
            Each bound name's rank, which decides whether arithmetic
            over it is elementwise.
        vectors : frozenset[str]
            The bindings declared as Stan `vector`s.
        """
        self._sb = sb
        self._fresh_fn = fresh
        self.cards = cards
        self.ranks = ranks
        self.vectors = vectors

    def fresh(self, prefix: str) -> str:
        return self._fresh_fn(prefix)

    def vertex(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def edge(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


class StanRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    Stan `panproto.Schema`.

    Subclasses
    [`RendererBase`][quivers.transpile.renderers.RendererBase]
    and overrides the four dispatch points
    (`declare`, `sample`, `marginalize`, `broadcast`) plus the two
    list / matrix arg helpers per the spec.
    """

    target: str = "stan"

    def __init__(self, *, source_module: Module | None = None) -> None:
        """Initialise the renderer.

        Parameters
        ----------
        source_module
            Optional original [`Module`][quivers.dsl.ast_nodes.Module]
            the IR came from. Carries `MorphismDecl` / `DefineDecl`
            entries the renderer reads when resolving
            [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] args
            (`Truncated(base, ...)` style wrappers). When omitted, the
            renderer raises on `IRArgFamilyRef` rather than guess.
        """
        self._source_module = source_module
        # Block roots are populated lazily during the IR walk.
        self._blocks: dict[BlockKind, str] = {}
        # Marginalize state for nested arg-substitution.
        self._marginalize_stack: tuple[Plate, ...] = ()
        # `lps_<latent>` accumulators per marginalize call site,
        # used to thread per-group log-sums through scope observes.
        self._marginalize_var: str | None = None
        self._marginalize_family: str | None = None
        self._marginalize_atom: float | None = None
        self._marginalize_latent_card: int | None = None
        self._marginalize_group_idx: tuple[str, ...] = ()
        # True while the active marginalize keys its accumulator by
        # the observation rather than by the grouping plate.
        self._marginalize_per_row: bool = False
        # Scope-local let bindings inside the current marginalize
        # block; the per-k observe expands these inline since the
        # let target is never declared as a Stan parameter.
        self._marginalize_let_subs: dict[str, LetExprNode] = {}
        # Bookkeeping for redeclaration avoidance.
        self._declared: dict[BlockKind, set[str]] = {}
        # Fresh counter is renderer-internal; the base's RenderContext
        # counter remains untouched so per-walk node IDs stay stable.
        self._fresh_n = 0
        # Per-render lookup caches; `render()` clears them on every
        # entry so repeated render calls on the same renderer produce
        # identical schemas.
        self._declared_shapes_state: dict[str, tuple[Constraint, Plate]] = {}
        # name -> K for chain variables whose downstream consumer
        # binds the value to a slot whose constraint has
        # `event_dim >= 1` (e.g. Categorical's `probs`). The
        # declaration path promotes the scalar real type to
        # `vector[K]`, and the deterministic / sample emit paths
        # wrap any scalar RHS in `rep_vector(rhs, K)`. See
        # `_compute_vector_promotions`.
        self._vector_promotions_state: dict[str, int] = {}
        self._class_index_widths_state: dict[str, int] = {}
        # name -> number of trailing axes of a let binding that a
        # consumer reads as one event (a `vector` for one axis), and
        # the rank of every bound name; see `_compute_let_events`.
        self._let_events: dict[str, int] = {}
        self._name_ranks: dict[str, int] = {}

    # ------------------------------------------------------------------
    # abstract overrides
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol of the target grammar.

        Returns
        -------
        panproto.Protocol
            The protocol of the target grammar.
        """
        return target_protocol("stan")

    # ----- the full render override -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a Stan schema.

        Override of the base `render` so the program-level layout
        (the `program` vertex with five child blocks) is built once
        per call.

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
        self._blocks = {}
        self._declared = {
            "data": set(),
            "parameters": set(),
            "transformed_parameters": set(),
            "model": set(),
            "generated_quantities": set(),
            "function_body": set(),
        }
        self._marginalize_stack = ()
        self._marginalize_var = None
        self._marginalize_family = None
        self._marginalize_atom = None
        self._marginalize_latent_card = None
        self._marginalize_group_idx = ()
        self._marginalize_per_row = False
        self._fresh_n = 0
        self._cards = dict(ir.cards)
        # Reset the per-render lookup caches so repeated render
        # calls on the same renderer produce identical schemas.
        self._declared_shapes_state.clear()
        self._name_ranks = name_rank_map(ir)
        self._let_events = self._compute_let_events(ir)
        self._vector_promotions_state.clear()
        self._vector_promotions_state.update(self._compute_vector_promotions(ir))
        self._class_index_widths_state.clear()
        self._class_index_widths_state.update(self._compute_class_index_widths(ir))
        # A needed computation the static subset lacks is refused before
        # the body is walked, so its call site is never rendered.
        if has_runtime_computations(ir):
            refuse_static_gaps(ir, self.target)
        # Program root.
        ctx.sb.vertex("prog", "program")
        if carried_computations(ir, self.target):
            self._ensure_block(ctx, "function_body")
        # Stan ships `normal`, `beta`, `gamma`, ... as built-in
        # densities but lacks `kumaraswamy`. When the IR samples or
        # observes from a family whose Stan emit relies on a user-
        # defined `<family>_lpdf` / `<family>_rng`, graft the
        # hand-written helper at
        # [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
        # into the program above the data block so the sampling
        # statement `y ~ kumaraswamy(a, b);` resolves through Stan's
        # `<family>_lpdf` lookup convention.
        if any(_ir_uses_family(ir.body, f) for f in _STAN_RUNTIME_HELPER_FAMILIES):
            _graft_runtime_stan_helper(ctx.sb, self, "prog")
        # Pre-create blocks in canonical Stan order so emit_pretty
        # honours the `data? transformed_data? parameters?
        # transformed_parameters? model? generated_quantities?`
        # grammar production. Without this the IR-walk's lazy
        # `_ensure_block` connects blocks in IR-encounter order
        # (`data` -> `parameters` -> `model` -> `transformed_parameters`),
        # which puts `transformed_parameters` past `model` and
        # makes the pretty-printer drop it.
        for kind in self._needed_blocks(ir):
            self._ensure_block(ctx, kind)
        # An input of an open extent, such as a host table `M` typed
        # `Tensor[Real]([M_extent0, M_extent1])`, is declared at a size
        # Stan reads as data, so each extent is an integer datum first.
        for inp in ir.inputs:
            for dim in (*inp.plate.batch_dims, *inp.plate.event_dims):
                if isinstance(dim, DimDynamic):
                    self._declare_extent(ctx, dim.size_name)
        # IRDataInput entries land in `data`.
        for inp in ir.inputs:
            self.declare(
                ctx,
                inp.name,
                inp.constraint,
                inp.plate,
                block="data",
            )
        # Walk the body.
        for node in ir.body:
            self._dispatch_node(ctx, node)
        if has_runtime_computations(ir):
            render_computations_static(
                ctx.sb,
                ir,
                target=self.target,
                destination=self._blocks.get("function_body", "prog"),
            )
        return ctx.sb.build()

    def _let_ctx(self, ctx: RenderContext) -> _StanLetCtx:
        """The let-expression carrier for this render.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.

        Returns
        -------
        _StanLetCtx
            A carrier over the render's schema builder, carrying the
            object cardinalities, every bound name's rank, and the
            bindings declared as vectors.
        """
        return _StanLetCtx(
            ctx.sb,
            lambda p: self._fresh(ctx, p),
            self._cards,
            self._name_ranks,
            frozenset(name for name, event in self._let_events.items() if event == 1),
        )

    def _compute_let_events(self, ir: IRProgram) -> dict[str, int]:
        """How many trailing axes of each let binding are an event.

        A let binding's checked type gives its axes but not how Stan
        should hold them: `array[] real` and `vector` are both one
        axis of reals. The use decides. A binding read by a family
        argument whose constraint has event rank ``e`` (Categorical's
        `probs`, Dirichlet's `concentration`, a multivariate normal's
        `loc`) is read there as an ``e``-axis event, which Stan types
        as a `vector` (or a `matrix` for two axes), so its trailing
        ``e`` axes are declared that way and its leading ones as array
        dimensions. A binding with fewer axes than the event is a
        scalar the vector-promotion path broadcasts instead.

        Parameters
        ----------
        ir : IRProgram
            The plan.

        Returns
        -------
        dict[str, int]
            Each such binding's event rank.
        """
        lets = {
            node.name: node
            for node in _iter_ir_nodes(ir.body)
            if isinstance(node, IRDeterministic)
        }
        events: dict[str, int] = {}
        for node in _iter_ir_nodes(ir.body):
            if not isinstance(node, (IRSample, IRObserve, IRMarginalize)):
                continue
            meta = FAMILY_META.get(node.family)
            if meta is None:
                continue
            arg_constraints = meta.distribution_class.arg_constraints
            if not isinstance(arg_constraints, dict):
                continue
            for arg, arg_name in zip(node.args, node.arg_names, strict=True):
                if not isinstance(arg, IRArgRef) or arg.name not in lets:
                    continue
                expected = arg_constraints.get(arg_name)
                event = int(getattr(expected, "event_dim", 0))
                if event < 1:
                    continue
                plate = lets[arg.name].plate
                rank = len(plate.batch_dims) + len(plate.event_dims)
                if rank - len(arg.indices) < event:
                    continue
                events[arg.name] = max(events.get(arg.name, 0), event)
        return events

    def _stan_plate(self, name: str, plate: Plate) -> Plate:
        """The plate a let binding is declared over in Stan.

        Parameters
        ----------
        name : str
            The binding.
        plate : Plate
            Its plate in the plan.

        Returns
        -------
        Plate
            The plate with the binding's event axes (see
            `_compute_let_events`) moved from the batch axes to the
            event axes.
        """
        event = self._let_events.get(name, 0)
        if event == 0:
            return plate
        axes = (*plate.batch_dims, *plate.event_dims)
        return Plate(event_dims=axes[-event:], batch_dims=axes[:-event])

    @property
    def _class_index_widths(self) -> dict[str, int]:
        """Per-render map from an observed class-index outcome to the
        width of its family's alphabet.

        Populated by `_compute_class_index_widths` at the top of
        `render()` and consulted by
        [`declare`][quivers.transpile.renderers.stan.StanRenderer.declare]
        so a Categorical observation is declared
        `int<lower=1, upper=K>` rather than on the sentinel `[0, 1]`
        interval the IR carries for every integer support.
        """
        return self._class_index_widths_state

    def _compute_class_index_widths(self, ir: IRProgram) -> dict[str, int]:
        """Scan the IR for observations under a class-index family and
        record each outcome's alphabet width.

        The width is the trailing event extent of the family's
        probability argument: a `Categorical(probs=phi[z])` whose `phi` is
        declared `array[K] simplex[V]` scores a value on `1:V`.
        Raises when the family is class-index and the width cannot be
        resolved statically, because the declaration would otherwise
        state a support the data does not live in.
        """
        plates = self._declared_plates(ir)
        out: dict[str, int] = {}
        for node in _iter_ir_nodes(ir.body):
            if not isinstance(node, IRObserve):
                continue
            if not is_class_code_family(node.family):
                continue
            width = self._class_alphabet_width(node, plates)
            if width is None:
                raise UnsupportedConstruct(
                    "qvr-stan",
                    [
                        f"declare:class-index-width:{node.name}: the "
                        f"{node.family} observation's probability "
                        f"argument has no statically resolvable "
                        f"alphabet width, so the outcome's Stan "
                        f"support cannot be declared"
                    ],
                )
            out[node.name] = width
        return out

    def _declared_plates(self, ir: IRProgram) -> dict[str, Plate]:
        """Every bound name's declared plate, inputs and body alike."""
        out: dict[str, Plate] = {inp.name: inp.plate for inp in ir.inputs}
        for node in _iter_ir_nodes(ir.body):
            if isinstance(node, IRDeterministic):
                out[node.name] = self._stan_plate(node.name, node.plate)
            elif isinstance(node, (IRSample, IRObserve)):
                out[node.name] = node.plate
            elif isinstance(node, IRMarginalize):
                out[node.latent] = node.plate
        return out

    def _class_alphabet_width(
        self, node: IRObserve, plates: dict[str, Plate]
    ) -> int | None:
        """The number of classes a class-index observation codes.

        The alphabet is the trailing axis of the family's alphabet
        argument (Categorical's `probs` or `logits`, the ordinal
        families' `cutpoints`, which bound one class fewer than there
        are), read after the reference's own subscripts.

        Parameters
        ----------
        node : IRObserve
            The observation.
        plates : dict[str, Plate]
            Every bound name's Stan plate.

        Returns
        -------
        int | None
            The class count, or `None` when it is not static.
        """
        meta = FAMILY_META.get(node.family)
        outcome = class_index_outcome(meta) if meta is not None else None
        if outcome is None:
            return None
        arg: IRArg | None = next(
            (
                value
                for value, name in zip(node.args, node.arg_names, strict=True)
                if name in outcome.alphabet_args
            ),
            None,
        )
        extent: int | None = None
        if isinstance(arg, IRArgList):
            extent = len(arg.elements)
        elif isinstance(arg, IRArgBroadcast) and arg.target_shape:
            extent = int(arg.target_shape[-1])
        elif isinstance(arg, IRArgRef):
            plate = plates.get(arg.name)
            axes = (
                (*plate.batch_dims, *plate.event_dims)[len(arg.indices) :]
                if plate is not None
                else ()
            )
            if axes and isinstance(axes[-1], DimStatic):
                extent = int(axes[-1].size)
            elif not axes:
                # A number feeding the alphabet slot is broadcast to
                # the promoted `vector[K]`, whose K is the alphabet.
                extent = self._vector_promotions.get(arg.name)
        if extent is None:
            return None
        return extent + outcome.extent_offset

    @property
    def _vector_promotions(self) -> dict[str, int]:
        """Per-render map from chain-tail name to the K required by
        its downstream Categorical (or other event-dim>=1) consumer.

        Populated by `_compute_vector_promotions` at the top of
        `render()`. The
        [`declare`][quivers.transpile.renderers.stan.StanRenderer.declare]
        and
        `_emit_deterministic`
        paths consult this map to promote scalar real declarations to
        `vector[K]` and to wrap scalar RHS expressions in
        `rep_vector(rhs, K)`.
        """
        return self._vector_promotions_state

    def _compute_vector_promotions(self, ir: IRProgram) -> dict[str, int]:
        """Scan `ir.body` for consumers whose argument slot has
        `event_dim >= 1` and record the producer name plus required K.

        For each [`IRObserve`][quivers.transpile.ir.IRObserve] or
        [`IRSample`][quivers.transpile.ir.IRSample] node whose
        family's positional arg constraint is an
        `IndependentConstraint(base, n>=1)`, walk its `args`. When the
        matching arg is an [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        to a name that the program declares with scalar real support
        (no event dims), record `{name: K}` where K is the consumer's
        event-axis cardinality.

        K is sourced from the consumer node's plate's last batch dim
        (the Categorical's output type axis, per the language-model
        gallery's `morphism lm_head : Hidden -> Token ~ Categorical`
        idiom). When the consumer's batch dim does not exist or its
        size is dynamic, the promotion is skipped and the original
        declaration shape stands.
        """
        producers: dict[str, ConstraintSpec] = {}
        producer_plates: dict[str, Plate] = {}
        for inp in ir.inputs:
            producers[inp.name] = inp.constraint
            producer_plates[inp.name] = inp.plate
        promotions: dict[str, int] = {}
        for node in ir.body:
            if isinstance(node, (IRSample, IRObserve)):
                meta = FAMILY_META.get(node.family)
                if meta is not None:
                    self._record_vector_promotions(
                        promotions,
                        node,
                        meta,
                        producers,
                        producer_plates,
                    )
            if isinstance(node, (IRSample, IRDeterministic, IRObserve)):
                producers[node.name] = node.constraint
                producer_plates[node.name] = node.plate
        return promotions

    def _record_vector_promotions(
        self,
        promotions: dict[str, int],
        node: IRSample | IRObserve,
        meta: FamilyMeta,
        producers: dict[str, ConstraintSpec],
        producer_plates: dict[str, Plate],
    ) -> None:
        """For each arg-position whose family constraint has
        `event_dim >= 1` and whose arg is a ref to a scalar producer,
        record the producer name with the required K.

        Parameters
        ----------
        promotions : dict[str, int]
            The promotions found so far, extended in place.
        node : IRSample | IRObserve
            The consumer.
        meta : FamilyMeta
            The consumer's family.
        producers : dict[str, ConstraintSpec]
            Each name bound so far and its support.
        producer_plates : dict[str, Plate]
            Each name bound so far and its plate.
        """
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return
        constraints = tuple(cls_attr.values())
        K = self._consumer_event_K(node)
        if K is None:
            return
        for i, arg in enumerate(node.args):
            if i >= len(constraints):
                continue
            expected = constraints[i]
            event_dim = int(getattr(expected, "event_dim", 0))
            if event_dim < 1:
                continue
            if not isinstance(arg, IRArgRef) or arg.indices:
                continue
            if arg.name in self._let_events:
                # The binding carries the event's axes itself.
                continue
            producer_cs = producers.get(arg.name)
            if producer_cs is None:
                continue
            producer_plate = producer_plates.get(arg.name)
            if producer_plate is not None and (
                producer_plate.event_dims or producer_plate.batch_dims
            ):
                # A producer with axes of its own carries the event
                # (a sampled vector over the class axis is read with
                # `to_vector`); only a number is promoted.
                continue
            producer_sup = producer_cs.to_constraint()
            if not (
                is_real_scalar(producer_sup)
                or is_real_positive(producer_sup)
                or is_real_unit_interval(producer_sup)
                or is_real_bounded_interval(producer_sup)
            ):
                continue
            existing = promotions.get(arg.name)
            if existing is not None and existing != K:
                # Conflicting K from two different consumers; skip
                # promotion rather than guess.
                continue
            promotions[arg.name] = K

    def _consumer_event_K(self, node: IRSample | IRObserve) -> int | None:
        """Derive the event-axis cardinality K for a consumer whose
        slot expects an event_dim>=1 arg.

        For the language-model idiom `observe target : Token <-
        Categorical(logits=h)`, the Token axis is the consumer node's plate
        batch_dim. The cardinality is the static size of that batch
        dim. When the batch dim is dynamic or absent, returns None
        and the promotion path skips this consumer.
        """
        if not node.plate.batch_dims:
            return None
        last = node.plate.batch_dims[-1]
        if isinstance(last, DimStatic):
            return last.size
        return None

    _CANONICAL_BLOCK_ORDER: tuple[BlockKind, ...] = (
        "data",
        "parameters",
        "transformed_parameters",
        "model",
        "generated_quantities",
    )

    def _needed_blocks(self, ir: IRProgram) -> list[BlockKind]:
        """Return the Stan blocks the renderer must materialise for
        `ir`, in canonical grammar order. A block is needed iff some
        IR node will route emit into it; pre-allocating in this order
        ensures the pretty-printer never sees a block out of grammar
        position."""
        needed: set[BlockKind] = set()
        if ir.inputs:
            needed.add("data")
        for node in ir.body:
            self._collect_needed_blocks(node, needed)
        return [k for k in self._CANONICAL_BLOCK_ORDER if k in needed]

    def _collect_needed_blocks(self, node: IRNode, needed: set[BlockKind]) -> None:
        """Walk a single IR node, recording the blocks it touches.

        Descends into [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        scopes so a continuous-latent marginalize whose scope contains
        an observe contributes `data` / `model`, and a scope
        deterministic contributes `transformed_parameters`.
        """
        if isinstance(node, IRSample):
            needed.add("parameters")
            needed.add("model")
        elif isinstance(node, IRObserve):
            needed.add("data")
            needed.add("model")
        elif isinstance(node, IRDataInput):
            needed.add("data")
        elif isinstance(node, (IRDeterministic, IRCall)):
            needed.add("transformed_parameters")
        elif isinstance(node, IRScore):
            needed.add("model")
        elif isinstance(node, IRMarginalize):
            # The discrete (logsumexp) path emits into model only;
            # the continuous path emits the latent into parameters
            # plus the latent's log-density increment into model,
            # then dispatches the
            # scope through the normal IR-walk. Pre-allocate both so
            # either path finds its blocks in canonical order.
            needed.add("parameters")
            needed.add("model")
            latent_sup = node.constraint.to_constraint()
            if _is_continuous_support(latent_sup):
                for child in node.scope:
                    self._collect_needed_blocks(child, needed)
        elif isinstance(node, IRReturn):
            needed.add("generated_quantities")

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
        """Emit a declaration for `name` in `block`.

        Dispatches purely on the support predicates of §2.2 of the
        spec, threaded through
        [`ConstraintSpec.to_constraint`][quivers.transpile.ir.ConstraintSpec.to_constraint].
        The table is the §5 Stan table.

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
        if name in self._declared[block]:
            return ""
        self._declared[block].add(name)
        sup = constraint.to_constraint()
        # Track for later GQ aliasing emission.
        self._declared_shapes[name] = (sup, plate)
        parent = self._ensure_block(ctx, block)
        decl = self._fresh(ctx, "decl")
        ctx.sb.vertex(decl, "top_var_decl_no_assign")
        ctx.sb.edge(parent, decl, "child_of")
        # arr_dims when there are batch dims.
        if plate.batch_dims:
            arr = self._fresh(ctx, "arr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in plate.batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(decl, arr, "child_of")
        # top_var_type, then the appropriate inner type per predicate.
        tvt = self._fresh(ctx, "tvt")
        ctx.sb.vertex(tvt, "top_var_type")
        promoted_k = self._vector_promotions.get(name)
        class_width = self._class_index_widths.get(name)
        if promoted_k is not None:
            self._emit_vector_type_of_size(ctx, tvt, promoted_k)
        elif class_width is not None:
            # A class-index outcome holds the QVR code of a class of its
            # family's alphabet, `0` to `K - 1`; the log-pmf reads it one
            # higher.
            self._emit_int_type(ctx, tvt, lower=0, upper=class_width - 1)
        else:
            self._emit_type(ctx, tvt, sup, plate.event_dims)
        ctx.sb.edge(decl, tvt, "child_of")
        # name field.
        nm = self._fresh(ctx, "ident")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", name)
        ctx.sb.edge(decl, nm, "name")
        return decl

    def _declare_extent(self, ctx: RenderContext, name: str) -> None:
        """Declare an open extent as a non-negative integer datum.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.
        name : str
            The extent's name, which the declarations of the inputs it
            sizes read.
        """
        if name in self._declared["data"]:
            return
        self._declared["data"].add(name)
        parent = self._ensure_block(ctx, "data")
        decl = self._fresh(ctx, "decl")
        ctx.sb.vertex(decl, "top_var_decl_no_assign")
        ctx.sb.edge(parent, decl, "child_of")
        tvt = self._fresh(ctx, "tvt")
        ctx.sb.vertex(tvt, "top_var_type")
        self._emit_int_type(ctx, tvt, lower=0, upper=None)
        ctx.sb.edge(decl, tvt, "child_of")
        nm = self._fresh(ctx, "ident")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", name)
        ctx.sb.edge(decl, nm, "name")

    def _emit_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        sup: Constraint,
        event_dims: tuple[Dim, ...],
    ) -> None:
        """Materialise the appropriate top_var_type child per the
        renderer's support dispatch table."""
        # A real-valued family with non-empty event dims is logically
        # a vector / matrix even though the per-element support is
        # `CSReal`; defer to the event-shape-aware emitters so the
        # declaration is `vector[K]` rather than scalar `real`.
        if event_dims and (
            is_real_scalar(sup)
            or is_real_positive(sup)
            or is_real_unit_interval(sup)
            or is_real_bounded_interval(sup)
        ):
            if len(event_dims) == 1:
                self._emit_vector_type(ctx, tvt_vid, event_dims)
                return
            if len(event_dims) == 2:
                self._emit_matrix_type(ctx, tvt_vid, event_dims)
                return
        # Scalar real family.
        if is_real_scalar(sup):
            self._emit_real_type(ctx, tvt_vid, lower=None, upper=None)
            return
        if is_real_positive(sup):
            self._emit_real_type(ctx, tvt_vid, lower=0, upper=None)
            return
        if is_real_unit_interval(sup):
            self._emit_real_type(ctx, tvt_vid, lower=0, upper=1)
            return
        if is_real_bounded_interval(sup):
            lo, hi = real_interval_bounds(sup)
            self._emit_real_type(ctx, tvt_vid, lower=lo, upper=hi)
            return
        if is_real_vector(sup):
            self._emit_vector_type(ctx, tvt_vid, event_dims)
            return
        if is_real_simplex(sup) or is_real_one_hot(sup):
            self._emit_simplex_type(ctx, tvt_vid, event_dims)
            return
        if is_real_cov_matrix(sup):
            self._emit_cov_matrix_type(ctx, tvt_vid, event_dims)
            return
        if is_real_corr_chol(sup):
            self._emit_corr_chol_type(ctx, tvt_vid, event_dims)
            return
        if is_real_matrix(sup):
            self._emit_matrix_type(ctx, tvt_vid, event_dims)
            return
        if is_int_bit(sup):
            self._emit_int_type(ctx, tvt_vid, lower=0, upper=1)
            return
        if is_int_category(sup):
            # An integer range holds QVR's own values (an index
            # covariate counts from zero); a subscript reads it one
            # higher.
            self._emit_int_type(
                ctx,
                tvt_vid,
                lower=int(getattr(sup, "lower_bound", 0)),
                upper=int(getattr(sup, "upper_bound", 0)),
            )
            return
        if is_int_count(sup):
            self._emit_int_type(ctx, tvt_vid, lower=0, upper=None)
            return
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"declare:unsupported-support:{type(sup).__name__}"],
        )

    def _emit_real_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        *,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        rt = self._fresh(ctx, "rt")
        ctx.sb.vertex(rt, "real_type")
        ctx.sb.constraint(rt, "literal-value", "real")
        self._maybe_emit_range_constraint(ctx, rt, lower, upper)
        ctx.sb.edge(tvt_vid, rt, "child_of")

    def _emit_int_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        *,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        it = self._fresh(ctx, "it")
        ctx.sb.vertex(it, "int_type")
        ctx.sb.constraint(it, "literal-value", "int")
        self._maybe_emit_int_range(ctx, it, lower, upper)
        ctx.sb.edge(tvt_vid, it, "child_of")

    def _emit_vector_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:vector:event-rank:{len(event_dims)}: "
                    f"Stan vector requires exactly one event dim"
                ],
            )
        vt = self._fresh(ctx, "vt")
        ctx.sb.vertex(vt, "vector_type")
        size_vid = self._dim_size_vertex(ctx, event_dims[0])
        ctx.sb.edge(vt, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, vt, "child_of")

    def _emit_vector_type_of_size(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        size: int,
    ) -> None:
        """Emit a `vector[K]` type child under `tvt_vid` with a
        literal integer size.

        Used by the vector-promotion path: chain-tail variables whose
        downstream consumer binds them to an event_dim>=1 slot are
        declared as `vector[K]` (wrapped in `array[batch]` when the
        producer had batch dims) rather than scalar `real`.
        """
        vt = self._fresh(ctx, "vt")
        ctx.sb.vertex(vt, "vector_type")
        size_vid = self._int_literal(ctx, size)
        ctx.sb.edge(vt, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, vt, "child_of")

    def _emit_simplex_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:simplex:event-rank:{len(event_dims)}: "
                    f"Stan simplex requires exactly one event dim"
                ],
            )
        st = self._fresh(ctx, "st")
        ctx.sb.vertex(st, "simplex_type")
        size_vid = self._dim_size_vertex(ctx, event_dims[0])
        ctx.sb.edge(st, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, st, "child_of")

    def _emit_cov_matrix_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        # Accept either a single event dim (e.g. `(Dim,)` from an
        # explicit `over=[Dim]`) or two equal-size event dims (e.g.
        # `(Dim, Dim)` from a matrix-valued morphism with two
        # identical axes). Stan's `cov_matrix[N]` is square by
        # definition; both representations carry the same `N`.
        if len(event_dims) == 2:
            d0, d1 = event_dims
            size0 = getattr(d0, "size", None)
            size1 = getattr(d1, "size", None)
            if size0 is None or size1 is None or size0 != size1:
                raise UnsupportedConstruct(
                    "qvr-stan",
                    [
                        f"declare:cov_matrix:event-rank:2-non-square: "
                        f"sizes {size0} vs {size1}"
                    ],
                )
            chosen = d0
        elif len(event_dims) == 1:
            chosen = event_dims[0]
        else:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:cov_matrix:event-rank:{len(event_dims)}: "
                    f"Stan cov_matrix requires a square event dim"
                ],
            )
        ct = self._fresh(ctx, "ct")
        ctx.sb.vertex(ct, "cov_matrix_type")
        size_vid = self._dim_size_vertex(ctx, chosen)
        ctx.sb.edge(ct, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, ct, "child_of")

    def _emit_corr_chol_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:cholesky:event-rank:{len(event_dims)}: "
                    f"Stan cholesky_factor_corr requires one event dim"
                ],
            )
        ct = self._fresh(ctx, "cct")
        ctx.sb.vertex(ct, "cholesky_factor_corr_type")
        size_vid = self._dim_size_vertex(ctx, event_dims[0])
        ctx.sb.edge(ct, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, ct, "child_of")

    def _emit_matrix_type(
        self,
        ctx: RenderContext,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 2:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:matrix:event-rank:{len(event_dims)}: "
                    f"Stan matrix requires two event dims"
                ],
            )
        mt = self._fresh(ctx, "mt")
        ctx.sb.vertex(mt, "matrix_type")
        row_vid = self._dim_size_vertex(ctx, event_dims[0])
        col_vid = self._dim_size_vertex(ctx, event_dims[1])
        ctx.sb.edge(mt, row_vid, "child_of")
        ctx.sb.edge(mt, col_vid, "child_of")
        ctx.sb.edge(tvt_vid, mt, "child_of")

    def _maybe_emit_range_constraint(
        self,
        ctx: RenderContext,
        parent: str,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        """Attach a `type_constraint` with `<lower=..., upper=...>` to
        a real-typed declaration."""
        if lower is None and upper is None:
            return
        tc = self._fresh(ctx, "tc")
        ctx.sb.vertex(tc, "type_constraint")
        range_vid = self._build_range_node(ctx, lower, upper)
        ctx.sb.edge(tc, range_vid, "child_of")
        ctx.sb.edge(parent, tc, "child_of")

    def _maybe_emit_int_range(
        self,
        ctx: RenderContext,
        int_type_vid: str,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        """Attach a `range_*` directly to an `int_type`. Stan's int
        type accepts the range as a direct child."""
        if lower is None and upper is None:
            return
        range_vid = self._build_range_node(ctx, lower, upper)
        ctx.sb.edge(int_type_vid, range_vid, "child_of")

    def _build_range_node(
        self,
        ctx: RenderContext,
        lower: int | float | None,
        upper: int | float | None,
    ) -> str:
        if lower is not None and upper is not None:
            rng = self._fresh(ctx, "rng")
            ctx.sb.vertex(rng, "range_lower_upper")
            lo = self._int_literal(ctx, lower)
            hi = self._int_literal(ctx, upper)
            ctx.sb.edge(rng, lo, "child_of")
            ctx.sb.edge(rng, hi, "child_of")
            return rng
        if lower is not None:
            rng = self._fresh(ctx, "rng")
            ctx.sb.vertex(rng, "range_lower")
            lo = self._int_literal(ctx, lower)
            ctx.sb.edge(rng, lo, "child_of")
            return rng
        rng = self._fresh(ctx, "rng")
        ctx.sb.vertex(rng, "range_upper")
        hi = self._int_literal(ctx, upper if upper is not None else 0)
        ctx.sb.edge(rng, hi, "child_of")
        return rng

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
        """Emit a `target += <family>_lpdf(...)` statement for a
        sample / observe step.

        Stan's `~` sampling notation drops every term that does not
        depend on a parameter, so `model.log_prob(jacobian=False)`
        on a `~`-built program omits data-only contributions and the
        omitted amount moves with the data. The explicit
        `target += <family>_lpdf(<lhs> | <args>)` form keeps every
        normalizing constant, which is what the constant-spread
        equivalence contract requires: the emitted program computes
        the joint exactly, not up to a data-dependent offset.

        Wraps the increment in nested `for (m_<axis> in 1:<size>)`
        loops for each batch dim. Indexes the LHS and every arg whose
        ref-name sits on the plate.

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
        del observed  # The sample emission shape does not differ.
        del constraint  # The constraint shaped the declaration; the
        # log-density suffix comes from the Stan family name.
        # Resolve the per-target family name.
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{family}"],
            )
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{family}"],
            )
        if family == "MixtureNormal":
            return self._emit_mixture_normal(
                ctx, name=name, args=args, arg_names=arg_names, plate=plate
            )
        stan_name = _stan_parameterized_name(family, arg_names, stan_name)
        density_name = self._log_density_name(family, stan_name)
        vector_slots = _vector_argument_slots(meta, arg_names)
        parent = self._ensure_block(ctx, "model")
        # Build loop indices for the batch dims.
        loop_names = self.index_for(ctx, plate)
        # The innermost block we'll attach the target statement to.
        innermost_parent = self._wrap_in_for_loops(
            ctx, parent, plate.batch_dims, loop_names
        )
        # Args + optional truncation correction.
        # TruncatedNormal: split (loc, scale, low, high) -> two args
        # to `normal_lpdf()` plus an explicit log-mass correction.
        if family == "TruncatedNormal":
            if len(args) != 4:
                raise UnsupportedConstruct(
                    "qvr-stan",
                    [
                        f"family:TruncatedNormal: expected 4 args "
                        f"(loc, scale, low, high), got {len(args)}"
                    ],
                )
            family_args = args[:2]
            truncation_args: tuple[IRArg, ...] | None = args[2:]
        else:
            family_args = args
            truncation_args = None
        injected = self._inject_stan_specific_args(family, family_args)
        rewritten = self._broadcast_scalar_refs(injected, meta, plate)
        # `<density_name>(<lhs> | <args>)`.
        de = self._fresh(ctx, "de")
        ctx.sb.vertex(de, "distr_expression")
        dn = self._fresh(ctx, "dnm")
        ctx.sb.vertex(dn, "identifier")
        ctx.sb.constraint(dn, "literal-value", density_name)
        ctx.sb.edge(de, dn, "name")
        dal = self._fresh(ctx, "dal")
        ctx.sb.vertex(dal, "distr_argument_list")
        # Variate: name indexed by loop vars, read one higher when it is
        # a zero-based class code.
        ctx.sb.edge(
            dal,
            self._class_code(ctx, family, self._build_lhs(ctx, name, loop_names)),
            "child_of",
        )
        if family == "NegativeBinomial":
            self._emit_neg_binomial_2_args(ctx, dal, rewritten, plate, loop_names)
        else:
            for position, arg in enumerate(rewritten):
                substituted = self._substitute_for_loops(
                    arg,
                    plate,
                    loop_names,
                    scalar_slot=position < len(arg_names)
                    and scalar_parameter(family, arg_names[position]),
                )
                arg_vid = self._render_arg(ctx, substituted)
                if position in vector_slots and self._reads_real_array(substituted):
                    arg_vid = self._as_event(ctx, arg_vid, 1)
                if (
                    family == "Wishart"
                    and position < len(arg_names)
                    and arg_names[position] == "scale_tril"
                ):
                    # Stan's Wishart takes the scale itself: L L^T.
                    arg_vid = self._stan_call(
                        ctx, "multiply_lower_tri_self_transpose", (arg_vid,)
                    )
                ctx.sb.edge(dal, arg_vid, "child_of")
        ctx.sb.edge(de, dal, "child_of")
        if truncation_args is None:
            rhs = de
        else:
            rhs = self._apply_truncation_correction(
                ctx,
                de,
                stan_name,
                rewritten,
                truncation_args,
                plate,
                loop_names,
            )
        ts = self._fresh(ctx, "sts")
        ctx.sb.vertex(ts, "target_statement")
        ctx.sb.edge(ts, rhs, "child_of")
        ctx.sb.edge(innermost_parent, ts, "child_of")
        return ts

    def _emit_mixture_normal(
        self,
        ctx: RenderContext,
        *,
        name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> SchemaFragment:
        """Emit a `MixtureNormal` site with `log_sum_exp`.

        For each row, the renderer fills a component vector with
        ``log(w[k]) + normal_lpdf(y | mu[k], sigma[k])`` and adds its
        `log_sum_exp` to the target density. Residual event axes raise because
        this path supports scalar Normal components.
        """
        if plate.event_dims:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"family:MixtureNormal:event-axis:{name}: the "
                    f"scalar-component log-sum-exp form carries no "
                    f"event shape, but the site declares "
                    f"{[d.name for d in plate.event_dims]!r}"
                ],
            )
        weights, loc, scale = mixture_normal_components("stan", args, arg_names)
        declared = self._declared_shapes.get(
            weights.name if isinstance(weights, IRArgRef) else ""
        )
        components = mixture_component_count(
            "stan", weights, declared[1] if declared is not None else None
        )
        parent = self._ensure_block(ctx, "model")
        outer_block = self._fresh(ctx, "mxbs")
        ctx.sb.vertex(outer_block, "block_statement")
        ctx.sb.edge(parent, outer_block, "child_of")
        lps_name = f"lps_{name}"
        loop_names = self.index_for(ctx, plate)
        self._declare_lps_array(
            ctx, outer_block, lps_name, plate.batch_dims, components
        )
        current = self._wrap_in_for_loops(
            ctx, outer_block, plate.batch_dims, loop_names
        )
        k_loop = self._fresh(ctx, "mxfs")
        ctx.sb.vertex(k_loop, "for_statement")
        ctx.sb.edge(current, k_loop, "child_of")
        klv = self._fresh(ctx, "mxlv")
        ctx.sb.vertex(klv, "identifier")
        ctx.sb.constraint(klv, "literal-value", "k")
        ctx.sb.edge(k_loop, klv, "loopvar")
        ctx.sb.edge(k_loop, self._int_literal(ctx, 1), "child_of")
        ctx.sb.edge(k_loop, self._int_literal(ctx, components), "child_of")
        kbs = self._fresh(ctx, "mxbs2")
        ctx.sb.vertex(kbs, "block_statement")
        ctx.sb.edge(k_loop, kbs, "child_of")
        asn = self._fresh(ctx, "mxasn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(kbs, asn, "child_of")
        ctx.sb.edge(
            asn,
            self._build_indexed_lhs(ctx, lps_name, (*loop_names, "k")),
            "child_of",
        )
        op = self._fresh(ctx, "mxop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "=")
        ctx.sb.edge(asn, op, "child_of")
        log_weight = self._stan_call(
            ctx,
            "log",
            (self._mixture_component_slice(ctx, weights),),
        )
        de = self._fresh(ctx, "mxde")
        ctx.sb.vertex(de, "distr_expression")
        fn_id = self._fresh(ctx, "mxfid")
        ctx.sb.vertex(fn_id, "identifier")
        ctx.sb.constraint(fn_id, "literal-value", "normal_lpdf")
        ctx.sb.edge(de, fn_id, "name")
        dal = self._fresh(ctx, "mxdal")
        ctx.sb.vertex(dal, "distr_argument_list")
        ctx.sb.edge(dal, self._build_lhs(ctx, name, loop_names), "child_of")
        ctx.sb.edge(dal, self._mixture_component_slice(ctx, loc), "child_of")
        ctx.sb.edge(dal, self._mixture_component_slice(ctx, scale), "child_of")
        ctx.sb.edge(de, dal, "child_of")
        ctx.sb.edge(asn, self._stan_binop(ctx, log_weight, "+", de), "child_of")
        self._emit_lps_accumulate(
            ctx, outer_block, lps_name, plate.batch_dims, loop_names
        )
        return outer_block

    def _mixture_component_slice(self, ctx: RenderContext, arg: IRArg) -> str:
        """Render `<arg>[k]`: the component-`k` entry of one of a
        `MixtureNormal` call's three per-component vectors."""
        if not isinstance(arg, IRArgRef) or arg.indices:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"family:MixtureNormal:component-arg:"
                    f"{type(arg).__name__}: each of the weight, "
                    f"location and scale arguments must be a bare "
                    f"reference to a per-component vector"
                ],
            )
        return self._build_indexed_arg_expression(ctx, arg.name, ("k",))

    def _apply_truncation_correction(
        self,
        ctx: RenderContext,
        density_vid: str,
        stan_name: str,
        family_args: tuple[IRArg, ...],
        bounds: tuple[IRArg, ...],
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Subtract the log-mass of the retained interval from an
        already-rendered log-density expression.

        A truncated draw on `(low, high)` has log-density
        `base_lpdf(y) - log(F(high) - F(low))`. Stan spells the two
        tail quantities `<family>_lcdf` and `<family>_lccdf`, and
        `log_diff_exp` forms the two-sided log-mass without leaving
        the log scale. An infinite bound drops the corresponding
        term: a `(-inf, high)` support corrects by `_lcdf(high)`
        alone, a `(low, inf)` support by `_lccdf(low)` alone, and an
        unbounded support needs no correction at all.
        """
        low, high = bounds
        low_infinite = _is_infinite_bound(low)
        high_infinite = _is_infinite_bound(high)
        if low_infinite and high_infinite:
            return density_vid
        if high_infinite:
            correction = self._build_tail_mass_call(
                ctx,
                f"{stan_name}_lccdf",
                low,
                family_args,
                plate,
                loop_names,
            )
        elif low_infinite:
            correction = self._build_tail_mass_call(
                ctx,
                f"{stan_name}_lcdf",
                high,
                family_args,
                plate,
                loop_names,
            )
        else:
            correction = self._stan_call(
                ctx,
                "log_diff_exp",
                (
                    self._build_tail_mass_call(
                        ctx,
                        f"{stan_name}_lcdf",
                        high,
                        family_args,
                        plate,
                        loop_names,
                    ),
                    self._build_tail_mass_call(
                        ctx,
                        f"{stan_name}_lcdf",
                        low,
                        family_args,
                        plate,
                        loop_names,
                    ),
                ),
            )
        return self._stan_binop(ctx, density_vid, "-", correction)

    def _build_tail_mass_call(
        self,
        ctx: RenderContext,
        function_name: str,
        bound: IRArg,
        family_args: tuple[IRArg, ...],
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Build `<function_name>(<bound> | <family args>)`, the
        `_lcdf` / `_lccdf` companion of a truncated family's
        log-density."""
        de = self._fresh(ctx, "tde")
        ctx.sb.vertex(de, "distr_expression")
        fnid = self._fresh(ctx, "tdeid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", function_name)
        ctx.sb.edge(de, fnid, "name")
        dal = self._fresh(ctx, "tdal")
        ctx.sb.vertex(dal, "distr_argument_list")
        bound_vid = self._render_arg(
            ctx,
            self._substitute_for_loops(bound, plate, loop_names, scalar_slot=True),
        )
        ctx.sb.edge(dal, bound_vid, "child_of")
        for arg in family_args:
            substituted = self._substitute_for_loops(
                arg, plate, loop_names, scalar_slot=True
            )
            ctx.sb.edge(dal, self._render_arg(ctx, substituted), "child_of")
        ctx.sb.edge(de, dal, "child_of")
        return de

    def _stan_call(
        self,
        ctx: RenderContext,
        function_name: str,
        arg_vids: tuple[str, ...],
    ) -> str:
        """Emit a `function_expression` calling `function_name` on
        already-rendered argument vertices."""
        fn = self._fresh(ctx, "cfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "cfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", function_name)
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "cal")
        ctx.sb.vertex(al, "argument_list")
        for arg_vid in arg_vids:
            ctx.sb.edge(al, arg_vid, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _inject_stan_specific_args(
        self,
        family: str,
        args: tuple[IRArg, ...],
    ) -> tuple[IRArg, ...]:
        """Prepend Stan-side parameter placeholders for QVR families
        whose torch distribution carries fewer parameters than Stan's
        same-named distribution.

        `HalfNormal(scale)` maps to Stan's `normal(0, scale)`; the
        renderer prepends `IRArgNumber(0)` before emission. The
        `<lower=0>` parameter constraint on the LHS comes from the
        family's `support`, dispatched by `is_real_positive` in
        `declare`.

        GP families carry an `IRArgNumber(0.0)` mean and an
        `IRArgKernel(...)` covariance; Stan's `multi_normal` expects
        a vector mean of size `N`, so the scalar zero is wrapped in
        `IRArgBroadcast` whose `target_shape` is the kernel's grid
        size.

        `Weibull(scale, concentration)` (torch's positional order)
        maps to Stan's `weibull(alpha, sigma)` which is shape-first
        (`alpha` = concentration, `sigma` = scale); the two positional
        args are swapped so the emitted density matches the QVR model.
        """
        if family == "Weibull" and len(args) == 2:
            return (args[1], args[0])
        if family == "GP" and len(args) == 2:
            mean_arg, kernel_arg = args
            if isinstance(mean_arg, IRArgNumber) and isinstance(
                kernel_arg, IRArgKernel
            ):
                broadcast_mean = IRArgBroadcast(
                    value=mean_arg,
                    target_shape=(kernel_arg.grid_size,),
                )
                return (broadcast_mean, kernel_arg)
        if family in _PREPEND_ZERO:
            return (IRArgNumber(value=0.0), *args)
        pos = _INSERT_ZERO_AT.get(family)
        if pos is not None:
            return (
                *args[:pos],
                IRArgNumber(value=0.0),
                *args[pos:],
            )
        return args

    def _emit_neg_binomial_2_args(
        self,
        ctx: RenderContext,
        stmt: str,
        args: tuple[IRArg, ...],
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> None:
        """Emit the two Stan `neg_binomial_2(mu, phi)` arguments from a
        QVR `NegativeBinomial(total_count, probs)` call.

        Stan's `neg_binomial_2` is mean / dispersion parameterised
        (mean `mu`, `variance = mu + mu^2 / phi`), whereas the QVR /
        torch `NegativeBinomial(total_count, probs)` has
        `pmf` proportional to `(1 - probs)^total_count probs^k` with
        mean `total_count * probs / (1 - probs)`. Matching the two
        gives ``mu = total_count * probs / (1 - probs)`` and
        ``phi = total_count``; the emitted call is
        ``neg_binomial_2(total_count * probs / (1 - probs),
        total_count)``. Each source arg is substituted for the
        surrounding batch loops before rendering, so a plated
        ``probs[m_Resp]`` carries its index into both slots.
        """
        if len(args) != 2:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"family:NegativeBinomial: expected 2 args "
                    f"(total_count, probs), got {len(args)}"
                ],
            )
        total_count = self._substitute_for_loops(
            args[0], plate, loop_names, scalar_slot=True
        )
        probs = self._substitute_for_loops(args[1], plate, loop_names, scalar_slot=True)
        # mu = total_count * probs / (1 - probs)
        product = self._stan_binop(
            ctx,
            self._render_arg(ctx, total_count),
            "*",
            self._render_arg(ctx, probs),
        )
        complement = self._stan_binop(
            ctx,
            self._int_literal(ctx, 1),
            "-",
            self._render_arg(ctx, probs),
        )
        mu = self._stan_binop(ctx, product, "/", self._stan_paren(ctx, complement))
        ctx.sb.edge(stmt, mu, "child_of")
        # phi = total_count
        ctx.sb.edge(stmt, self._render_arg(ctx, total_count), "child_of")

    def _stan_binop(
        self,
        ctx: RenderContext,
        left_vid: str,
        op: str,
        right_vid: str,
    ) -> str:
        """Emit an `infix_op_expression` for the binary operator `op`
        over two already-rendered operand vertices."""
        vid = self._fresh(ctx, "bin")
        ctx.sb.vertex(vid, "infix_op_expression")
        ctx.sb.constraint(vid, "chose-alt-fingerprint", op)
        ctx.sb.edge(vid, left_vid, "child_of")
        ctx.sb.edge(vid, right_vid, "child_of")
        return vid

    def _stan_paren(self, ctx: RenderContext, vid: str) -> str:
        """Wrap an already-rendered expression vertex in a
        `parenthized_expression` so its grouping survives Stan's
        left-to-right printer (e.g. the `(1 - probs)` denominator)."""
        paren = self._fresh(ctx, "paren")
        ctx.sb.vertex(paren, "parenthized_expression")
        ctx.sb.constraint(paren, "chose-alt-fingerprint", "( )")
        ctx.sb.edge(paren, vid, "child_of")
        return paren

    def _broadcast_scalar_refs(
        self,
        args: tuple[IRArg, ...],
        meta: FamilyMeta,
        plate: Plate,
    ) -> tuple[IRArg, ...]:
        """Per-renderer broadcast normalisation.

        Lower leaves [`IRArgRef`][quivers.transpile.ir.IRArgRef] args
        unchanged when the user supplied a name; the family's
        `arg_constraints` may still require a vector / matrix at that
        position. When a ref's declared shape is scalar but the
        constraint is an `IndependentConstraint(base, n>=1)`, wrap it
        in [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] so
        the Stan emit becomes `rep_vector(<name>, K)`.

        Target shape is derived from the family's event-axis size on
        the sample step (the first event_dim, when present).
        """
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return args
        constraints = tuple(cls_attr.values())
        out: list[IRArg] = []
        for i, arg in enumerate(args):
            if i >= len(constraints):
                out.append(arg)
                continue
            expected = constraints[i]
            if not isinstance(arg, IRArgRef) or not isinstance(
                expected, _torch_constraints._IndependentConstraint
            ):
                out.append(arg)
                continue
            declared = self._declared_shapes.get(arg.name)
            if declared is None:
                out.append(arg)
                continue
            declared_sup, declared_plate = declared
            if declared_plate.batch_dims and tuple(declared_plate.batch_dims) == tuple(
                plate.event_dims
            ):
                # An array over the draw's event axis is the vector the
                # slot takes, read as one.
                out.append(arg)
                continue
            if (
                declared_plate.event_dims
                or not is_real_scalar(declared_sup)
                and not is_real_positive(declared_sup)
                and not is_real_unit_interval(declared_sup)
            ):
                out.append(arg)
                continue
            # Derive the target shape: take the first n event dims off
            # the sample's plate (event_dim of the constraint).
            n = int(expected.event_dim)
            if len(plate.event_dims) < n:
                out.append(arg)
                continue
            sizes: list[int] = []
            ok = True
            for dim in plate.event_dims[:n]:
                if isinstance(dim, DimStatic):
                    sizes.append(dim.size)
                else:
                    ok = False
                    break
            if not ok:
                out.append(arg)
                continue
            out.append(IRArgBroadcast(value=arg, target_shape=tuple(sizes)))
        return tuple(out)

    def _wrap_in_for_loops(
        self,
        ctx: RenderContext,
        parent: str,
        batch_dims: tuple[Dim, ...],
        loop_names: tuple[str, ...],
    ) -> str:
        """Emit nested `for (m_<name> in 1:<size>) { ... }` loops, one
        per batch dim. Returns the innermost block vertex id."""
        current = parent
        for dim, lv_name in zip(batch_dims, loop_names, strict=True):
            fs = self._fresh(ctx, "fs")
            ctx.sb.vertex(fs, "for_statement")
            ctx.sb.edge(current, fs, "child_of")
            # loopvar identifier
            lv = self._fresh(ctx, "lv")
            ctx.sb.vertex(lv, "identifier")
            ctx.sb.constraint(lv, "literal-value", lv_name)
            ctx.sb.edge(fs, lv, "loopvar")
            # lower = 1
            lo = self._int_literal(ctx, 1)
            ctx.sb.edge(fs, lo, "child_of")
            # upper = batch_dim size
            hi = self._dim_size_vertex(ctx, dim)
            ctx.sb.edge(fs, hi, "child_of")
            # body block
            bs = self._fresh(ctx, "bs")
            ctx.sb.vertex(bs, "block_statement")
            ctx.sb.edge(fs, bs, "child_of")
            current = bs
        return current

    def _build_lhs(
        self,
        ctx: RenderContext,
        name: str,
        loop_names: tuple[str, ...],
    ) -> str:
        """Build the variate of a log-density call: either a bare
        `variable_expression` or an `indexed_expression`. Lower's
        sample emission for a plated draw lands as
        `<name>[m_0, m_1, ...]` when loop vars exist."""
        base_ve = self._variable_expression(ctx, name)
        if not loop_names:
            return base_ve
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        ctx.sb.edge(ie, base_ve, "child_of")
        for lv_name in loop_names:
            idx = self._fresh(ctx, "idx")
            ctx.sb.vertex(idx, "index")
            idx_ve = self._variable_expression(ctx, lv_name)
            ctx.sb.edge(idx, idx_ve, "child_of")
            ctx.sb.edge(ie, idx, "child_of")
        return ie

    def _substitute_for_loops(
        self,
        arg: IRArg,
        plate: Plate,
        loop_names: tuple[str, ...],
        *,
        scalar_slot: bool = False,
    ) -> IRArg:
        """For an [`IRArgRef`][quivers.transpile.ir.IRArgRef] whose
        referenced name itself sits on the surrounding plate, prepend
        the loop indices.

        Two cases handled:

        * User-written index already present (`phi[z]`, `mu[cls]`):
          left alone; the IR already encodes the per-element lookup.
        * Unindexed ref whose declared plate's batch_dims align (as a
          tuple-equality on `Dim`s) with `plate.batch_dims`, or, when
          the ref fills a rank-0 family parameter (`scalar_slot`),
          whose every dimension, batch then event, does: prepend one
          [`IRArgRef`][quivers.transpile.ir.IRArgRef] index per
          aligned batch dim, in declaration order. The loop indices
          are 1-based identifiers; the Stan
          `indexed_expression`
          emit takes them verbatim.

        This is what the plan's typed plates rely on: a let-bound
        `mu` over `Obs` is declared `array[Obs] real`, and this hook
        makes the surrounding
        `observe y : Obs <- Normal(mu, ...)` index `mu` per-element
        rather than reading the whole array into the scalar Normal
        slot (which Stan rejects with a dimension mismatch at runtime
        data binding). A vector `z` over `Feat`, one event, read by a
        scalar family the plan replicates over `Feat` is read at the
        loop's position too: inside the loop a whole `vector` in the
        location slot would score each entry of the observation
        against every entry of `z`. A vector filling a parameter that
        reads a whole event (`Categorical`'s `probs`) is that event at
        every position, so only a rank-0 slot is read this way.

        Parameters
        ----------
        arg : IRArg
            The argument.
        plate : Plate
            The site's plate.
        loop_names : tuple[str, ...]
            The loop variable of each of the plate's batch dims.
        scalar_slot : bool
            Whether the argument fills a rank-0 family parameter.

        Returns
        -------
        IRArg
            The argument, indexed at the loops' position when it varies
            along the axes they visit.
        """
        if not isinstance(arg, IRArgRef):
            return arg
        if arg.indices:
            return arg
        declared = self._declared_shapes.get(arg.name)
        if declared is None:
            return arg
        decl_plate = declared[1]
        if decl_plate.batch_dims != plate.batch_dims and (
            not scalar_slot
            or (*decl_plate.batch_dims, *decl_plate.event_dims)
            != tuple(plate.batch_dims)
        ):
            return arg
        if not loop_names:
            return arg
        return IRArgRef(
            name=arg.name,
            indices=tuple(IRArgRef(name=lv) for lv in loop_names),
        )

    # ----- marginalize dispatch -----

    def marginalize(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Emit a marginalized latent.

        Families with declared finite marginal support are enumerated with a
        `log_sum_exp` reduction. Families without finite support
        become constrained Stan parameters: the renderer adds their prior
        density and emits the scope with that parameter in context. Stan then
        samples the resulting joint model.

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
        meta = FAMILY_META.get(node.family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{node.family}"],
            )
        latent_sup = node.constraint.to_constraint()
        # `finite_enumerable_at_call_site` answers the plate question,
        # whether the marginalize index sizes the family's own
        # support; `marginalize_support` answers the integration one.
        # A relaxation family replicates along its index and is still
        # integrable over its two atoms, so the latter is what decides
        # between enumerating and declaring a live latent.
        integrable = marginalize_support(meta) is not None
        if not integrable and _is_continuous_support(latent_sup):
            return self._marginalize_continuous(ctx, node, meta)
        if not integrable:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"marginalize:non-finite-support:{node.family}"],
            )
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{node.family}"],
            )
        stan_name = _stan_parameterized_name(node.family, node.arg_names, stan_name)
        # Eligibility checks satisfied; compute the latent cardinality.
        latent_card = self._latent_cardinality(meta, node)
        if latent_card is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"marginalize:unknown-cardinality:{node.family}: "
                    f"cannot determine support size at compile time"
                ],
            )
        parent = self._ensure_block(ctx, "model")
        # Open a fresh `{ ... }` block scope for the lps_<latent> array.
        outer_block = self._fresh(ctx, "mbs")
        ctx.sb.vertex(outer_block, "block_statement")
        ctx.sb.edge(parent, outer_block, "child_of")
        # Choose what the accumulator is keyed by. A per-row prior --
        # the probability argument is itself indexed by the grouping
        # plate, so each observation reads its own group's row --
        # replicates the latent per observation, and the enumeration
        # runs once per observation. A global prior instead shares one
        # latent across each group's rows, so the rows' contributions
        # scatter into the group before the reduction. This is the
        # same discriminator the QVR compiler applies.
        per_row = self._prior_is_group_plated(node)
        if per_row:
            acc_dims = self._marginalize_row_dims(node)
            acc_loop_names = self._observe_scope_loop_names(acc_dims)
        else:
            acc_dims = node.plate.batch_dims
            acc_loop_names = self._marginalize_loop_names(acc_dims)
        # 1. Declare `array[acc_dims] vector[K] lps_<latent>;`.
        lps_name = f"lps_{node.latent}"
        self._declare_lps_array(ctx, outer_block, lps_name, acc_dims, latent_card)
        # 2. Seed each accumulator row with the latent log-pmf:
        #    for each accumulator index, for k in 1:K,
        #    lps[..., k] = lpmf(k | args).
        self._emit_lps_init(
            ctx,
            outer_block,
            lps_name,
            acc_dims,
            acc_loop_names,
            latent_card,
            stan_name,
            node.args,
            node.arg_names,
            meta,
            self._prior_index_args(node, acc_loop_names, per_row),
        )
        # 3. Walk the scope body: each scope IRObserve emits an
        #    inner-loop that accumulates per-k contributions into
        #    `lps[<accumulator index>, k]`.
        prev_marg_var = self._marginalize_var
        prev_marg_family = self._marginalize_family
        prev_marg_card = self._marginalize_latent_card
        prev_group_idx = self._marginalize_group_idx
        prev_per_row = self._marginalize_per_row
        prev_stack = self._marginalize_stack
        prev_lets = self._marginalize_let_subs
        self._marginalize_let_subs = self._resolve_scope_lets(node.scope)
        self._marginalize_var = lps_name
        self._marginalize_family = node.family
        self._marginalize_latent_card = latent_card
        self._marginalize_group_idx = acc_loop_names
        self._marginalize_per_row = per_row
        self._marginalize_stack = (*self._marginalize_stack, node.plate)
        try:
            for scope_node in node.scope:
                self._dispatch_marginalize_scope(ctx, outer_block, scope_node, node)
        finally:
            self._marginalize_var = prev_marg_var
            self._marginalize_family = prev_marg_family
            self._marginalize_latent_card = prev_marg_card
            self._marginalize_group_idx = prev_group_idx
            self._marginalize_per_row = prev_per_row
            self._marginalize_stack = prev_stack
            self._marginalize_let_subs = prev_lets
        # 4. Accumulate the per-row log-sums into `target`.
        self._emit_lps_accumulate(
            ctx,
            outer_block,
            lps_name,
            acc_dims,
            acc_loop_names,
        )
        return outer_block

    def _prior_is_group_plated(self, node: IRMarginalize) -> bool:
        """True when the latent's probability argument carries the
        marginalize's grouping plate.

        A `Categorical(probs=theta)` whose `theta` is declared
        `array[|G|] simplex[K]` gives every row of the group its own
        draw from its group's prior, so the marginal is one
        `log_sum_exp` per row. A bare `simplex[K]` prior instead
        denotes one draw shared across the group's rows.
        """
        if not node.plate.batch_dims:
            return False
        for arg in node.args:
            if not isinstance(arg, IRArgRef) or arg.indices:
                continue
            declared = self._declared_shapes.get(arg.name)
            if declared is None:
                continue
            _, declared_plate = declared
            if len(declared_plate.batch_dims) >= len(node.plate.batch_dims):
                return True
        return False

    def _marginalize_row_dims(self, node: IRMarginalize) -> tuple[Dim, ...]:
        """The plate a per-row accumulator is keyed by: the batch dims
        of the scope's observe steps."""
        plates = {
            tuple(inner.plate.batch_dims)
            for inner in node.scope
            if isinstance(inner, IRObserve)
        }
        if len(plates) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"marginalize:per-row-prior:{node.latent}: a "
                    f"per-row prior needs exactly one observation "
                    f"plate to key the accumulator by, found "
                    f"{len(plates)}"
                ],
            )
        return next(iter(plates))

    def _prior_index_args(
        self,
        node: IRMarginalize,
        acc_loop_names: tuple[str, ...],
        per_row: bool,
    ) -> tuple[IRArg, ...]:
        """Index expressions that select the latent's prior row for
        one accumulator entry.

        In the grouped reading the accumulator is already keyed by the
        grouping plate, so the loop variables index the prior
        directly. In the per-row reading the accumulator is keyed by
        the observation, so the prior row is reached through the
        observe's `via` fibration.
        """
        if not per_row:
            return tuple(IRArgRef(name=name) for name in acc_loop_names)
        via = self._marginalize_scope_via(node)
        if via is None:
            return tuple(IRArgRef(name=name) for name in acc_loop_names)
        if not acc_loop_names:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"marginalize:per-row-prior:{node.latent}: the "
                    f"observe carries a `via` fibration but no "
                    f"observation plate to index it with"
                ],
            )
        row_loop = IRArgRef(name=acc_loop_names[0])
        return tuple(
            IRArgRef(name=via, indices=(row_loop,)) for _ in node.plate.batch_dims
        )

    def _marginalize_scope_via(self, node: IRMarginalize) -> str | None:
        """The `via` fibration the scope's observe steps share, or
        `None` when they carry none."""
        vias = {inner.via for inner in node.scope if isinstance(inner, IRObserve)}
        if len(vias) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"marginalize:per-row-prior:{node.latent}: the "
                    f"scope's observe steps disagree on the `via` "
                    f"fibration into the grouping plate"
                ],
            )
        return next(iter(vias))

    def _marginalize_continuous(
        self,
        ctx: RenderContext,
        node: IRMarginalize,
        meta: FamilyMeta,
    ) -> SchemaFragment:
        """Emit a continuous-latent marginalize.

        Routes the marginalize through Stan's standard parameter +
        log-density-increment path: the latent is declared as a
        constrained parameter, scored against its family with a
        `target +=` statement in the `model` block, and the scope
        body's
        deterministic / observe nodes pass through the normal
        per-construct dispatch with the latent name now in scope as a
        parameter reference. HMC samples the latent jointly with the
        rest of the model; no per-step enumeration loop is emitted.

        Records the latent in `_declared_shapes` so downstream IR
        nodes that reference it (e.g. a scope `IRDeterministic` whose
        let-expression mentions the latent) resolve the ref through
        the same lookup path used for any other declared parameter.
        """
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{node.family}"],
            )
        stan_name = _stan_parameterized_name(node.family, node.arg_names, stan_name)
        # Declare the latent in the parameters block.
        self.declare(
            ctx,
            node.latent,
            node.constraint,
            node.plate,
            block="parameters",
        )
        # Emit the latent's log-density increment in the model block.
        self.sample(
            ctx,
            node.latent,
            node.family,
            node.args,
            node.arg_names,
            node.constraint,
            node.plate,
            observed=False,
        )
        # Dispatch each scope node through the normal IR-walk; the
        # latent is now an in-scope parameter reference for
        # deterministic let-bindings and downstream observes.
        for scope_node in node.scope:
            self._dispatch_node(ctx, scope_node)
        # Return the model-block vertex; the continuous path does not
        # open a dedicated `{ ... }` scope (no `lps` accumulator to
        # localise).
        return self._ensure_block(ctx, "model")

    def _dispatch_marginalize_scope(
        self,
        ctx: RenderContext,
        scope_block: str,
        node: IRNode,
        parent: IRMarginalize,
    ) -> None:
        """Handle one node inside a [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        scope. Inner observe steps contribute to the per-group `lps`
        accumulator; other constructs raise."""
        if isinstance(node, IRObserve):
            self._emit_marginalize_scope_observe(ctx, scope_block, node, parent)
            return
        if isinstance(node, IRSample):
            self._emit_marginalize_scope_sample(ctx, scope_block, node, parent)
            return
        if isinstance(node, IRDeterministic):
            # A scope-local let binding (e.g. `let gated_rate = z *
            # rate; observe y <- Poisson(gated_rate)` inside a
            # marginalize over z) is meaningful only per atom and per
            # row, so `_resolve_scope_lets` has already resolved it and
            # each observe reading it inlines it at its own atom and
            # row (see `_scope_arg_expr`).
            return
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"marginalize:scope:unsupported:{type(node).__name__}"],
        )

    def _emit_marginalize_scope_observe(
        self,
        ctx: RenderContext,
        scope_block: str,
        node: IRObserve,
        parent: IRMarginalize,
    ) -> None:
        """Emit the per-k log-pmf accumulation for a scope observe.

        For an observe `r ~ Normal(mu[cls], sigma[cls])` inside a
        `marginalize cls : Component` scope with parent plate `[Item]`,
        Stan emits:

            for (n in 1:N_r) {
              for (k in 1:K) {
                lps_cls[<via(n)>, k] += normal_lpdf(r[n] | mu[k], sigma[k]);
              }
            }
        """
        meta = FAMILY_META.get(node.family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{node.family}"],
            )
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{node.family}"],
            )
        stan_name = _stan_parameterized_name(node.family, node.arg_names, stan_name)
        lps_name = self._marginalize_var or ""
        if not lps_name:
            raise UnsupportedConstruct(
                "qvr-stan",
                ["marginalize:scope:observe:no-lps-context"],
            )
        # Outer loop over the observe's plate batch dims.
        loop_names = self._observe_scope_loop_names(node.plate.batch_dims)
        current = self._wrap_in_for_loops(
            ctx, scope_block, node.plate.batch_dims, loop_names
        )
        # Inner loop over k in 1:K, or one statement per atom when
        # the atoms are numbers the body computes with. A relaxation
        # family's off atom is a zero, and folding it out of the
        # body's arithmetic is only possible once it is a literal
        # rather than an expression in the loop variable.
        latent_card = self._marginalize_latent_card or 0
        meta = FAMILY_META.get(parent.family)
        declared = marginalize_support(meta) if meta is not None else None
        if declared is not None and declared.atoms == "binary":
            self._emit_unrolled_scope_observe(
                ctx,
                current,
                node,
                parent,
                loop_names,
                latent_card,
            )
            return
        k_name = "k"
        k_loop = self._fresh(ctx, "kfs")
        ctx.sb.vertex(k_loop, "for_statement")
        ctx.sb.edge(current, k_loop, "child_of")
        lv = self._fresh(ctx, "klv")
        ctx.sb.vertex(lv, "identifier")
        ctx.sb.constraint(lv, "literal-value", k_name)
        ctx.sb.edge(k_loop, lv, "loopvar")
        ctx.sb.edge(k_loop, self._int_literal(ctx, 1), "child_of")
        ctx.sb.edge(k_loop, self._int_literal(ctx, latent_card), "child_of")
        kbs = self._fresh(ctx, "kbs")
        ctx.sb.vertex(kbs, "block_statement")
        ctx.sb.edge(k_loop, kbs, "child_of")
        # assignment_statement: lps_<latent>[<group_indices>, k] += lpdf(...)
        asn = self._fresh(ctx, "asn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(kbs, asn, "child_of")
        # LHS: indexed_lhs lps[group_idx..., k]
        group_idx_exprs = self._marginalize_group_index_exprs(node, parent, loop_names)
        lhs_vid = self._build_indexed_lhs(ctx, lps_name, (*group_idx_exprs, k_name))
        ctx.sb.edge(asn, lhs_vid, "child_of")
        # op: +=
        op = self._fresh(ctx, "aop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "+=")
        ctx.sb.edge(asn, op, "child_of")
        # RHS: <family>_lpdf/_lpmf(observed_var[n] | substituted args)
        lpdf_name = self._log_density_name(node.family, stan_name)
        lpdf_call = self._build_lpdf_call(
            ctx,
            lpdf_name,
            node.name,
            node.family,
            loop_names,
            node.args,
            k_name,
            node.arg_names,
        )
        ctx.sb.edge(asn, lpdf_call, "child_of")

    def _emit_unrolled_scope_observe(
        self,
        ctx: RenderContext,
        current: str,
        node: IRObserve,
        parent: IRMarginalize,
        loop_names: tuple[str, ...],
        latent_card: int,
    ) -> None:
        """One accumulation per atom, with the atom as a literal.

        The looped form writes the latent as an expression in the loop
        variable, which leaves the body's arithmetic depending on it:
        `poisson_lpmf(y | (k - 1) * rate)` reads as a rate the
        parameters reach even at the atom where it is zero, so its
        derivative there is `0 * inf` and Stan rejects every initial
        value. Unrolled, the off atom's rate folds to a constant and
        the model samples.
        """
        meta = FAMILY_META.get(node.family)
        stan_name = meta.target_names.get("stan") if meta is not None else None
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{node.family}"],
            )
        stan_name = _stan_parameterized_name(node.family, node.arg_names, stan_name)
        lpdf_name = self._log_density_name(node.family, stan_name)
        lps_name = self._marginalize_var or ""
        group_idx_exprs = self._marginalize_group_index_exprs(node, parent, loop_names)
        previous_atom = self._marginalize_atom
        try:
            for position in range(latent_card):
                self._marginalize_atom = float(position)
                asn = self._fresh(ctx, "uasn")
                ctx.sb.vertex(asn, "assignment_statement")
                ctx.sb.edge(current, asn, "child_of")
                lhs_vid = self._build_indexed_lhs(
                    ctx,
                    lps_name,
                    (*group_idx_exprs, str(position + 1)),
                )
                ctx.sb.edge(asn, lhs_vid, "child_of")
                op = self._fresh(ctx, "uaop")
                ctx.sb.vertex(op, "assignment_op")
                ctx.sb.constraint(op, "literal-value", "+=")
                ctx.sb.edge(asn, op, "child_of")
                ctx.sb.edge(
                    asn,
                    self._build_lpdf_call(
                        ctx,
                        lpdf_name,
                        node.name,
                        node.family,
                        loop_names,
                        node.args,
                        str(position + 1),
                        node.arg_names,
                    ),
                    "child_of",
                )
        finally:
            self._marginalize_atom = previous_atom

    def _emit_marginalize_scope_sample(
        self,
        ctx: RenderContext,
        scope_block: str,
        node: IRSample,
        parent: IRMarginalize,
    ) -> None:
        """A nested sample inside a marginalize scope.

        Stan emits per-k conditional accumulation; an inner sample
        of a continuous latent is uncommon in canonical
        log_sum_exp idiom (the spec's gallery examples don't
        exercise this), so raise rather than emit something
        ambiguous.
        """
        del ctx, scope_block, parent
        raise UnsupportedConstruct(
            "qvr-stan",
            [
                f"marginalize:scope:IRSample:{node.name}: nested "
                f"sample inside marginalize scope not supported by "
                f"the Stan log_sum_exp emit"
            ],
        )

    def _marginalize_group_index_exprs(
        self,
        observe: IRObserve,
        parent: IRMarginalize,
        loop_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Map the outer marginalize plate's batch dims to indexing
        expressions usable inside this observe's loop body.

        When the observe's `via` fibration names a per-row indexer
        into a group axis, the group index is `<via>[<row-loop>]`;
        otherwise the surrounding marginalize loop variables apply
        directly. When the marginalize has no batch dims, returns ().
        """
        del parent  # parent plate already threaded through
        # _marginalize_group_idx by the marginalize() caller.
        group_count = len(self._marginalize_group_idx)
        if group_count == 0:
            return ()
        if self._marginalize_per_row:
            # The accumulator is keyed by the observation itself, so
            # the observe's own loop variables address it directly.
            return loop_names
        if observe.via is not None and loop_names:
            # `<via>[<observe-loop-0>]` per group dim. For
            # single-group marginalize this is exactly the spec's
            # `lps[word_idx[n], k]` form.
            return tuple(
                f"{observe.via}[{loop_names[0]}]+1" for _ in range(group_count)
            )
        # Fall through: use the marginalize group's loop names.
        return self._marginalize_group_idx

    def _build_indexed_lhs(
        self,
        ctx: RenderContext,
        base_name: str,
        index_exprs: tuple[str, ...],
    ) -> str:
        """Build a lhs node for `<base_name>[<idx0>, <idx1>, ...]`.

        Index expressions are bare strings (loop var names like `k`
        or call forms like `word_idx[n]`); they're materialised via
        the Stan parser-facing identifier or indexed_expression form.
        """
        lhs = self._fresh(ctx, "lhs")
        ctx.sb.vertex(lhs, "lhs")
        if not index_exprs:
            ve = self._variable_expression(ctx, base_name)
            ctx.sb.edge(lhs, ve, "child_of")
            return lhs
        ilhs = self._fresh(ctx, "ilhs")
        ctx.sb.vertex(ilhs, "indexed_lhs")
        # Inner lhs is the bare-name variable_expression.
        inner_lhs = self._fresh(ctx, "ilhs_inner")
        ctx.sb.vertex(inner_lhs, "lhs")
        base_ve = self._variable_expression(ctx, base_name)
        ctx.sb.edge(inner_lhs, base_ve, "child_of")
        ctx.sb.edge(ilhs, inner_lhs, "child_of")
        for idx_expr in index_exprs:
            idx_vid = self._build_index_node(ctx, idx_expr)
            ctx.sb.edge(ilhs, idx_vid, "child_of")
        ctx.sb.edge(lhs, ilhs, "child_of")
        return lhs

    def _build_index_node(self, ctx: RenderContext, expr_text: str) -> str:
        """Build an `index` node carrying a parsed expression.

        Recognises the `<name>[<inner>]` shape and emits an
        `indexed_expression`; otherwise treats `expr_text` as a bare
        name and emits a `variable_expression`.
        """
        idx = self._fresh(ctx, "idx")
        ctx.sb.vertex(idx, "index")
        inner = self._build_expr_from_text(ctx, expr_text)
        ctx.sb.edge(idx, inner, "child_of")
        return idx

    def _build_expr_from_text(self, ctx: RenderContext, expr_text: str) -> str:
        """Build an expression vertex for a small text form.

        Supports bare identifiers, integer literals, a single level of
        `<name>[<inner>]` indexing, and a trailing `+1`, the shift of a
        zero-based value read as a Stan subscript.
        """
        text = expr_text.strip()
        if text.endswith("+1"):
            return self._stan_binop(
                ctx,
                self._build_expr_from_text(ctx, text[:-2]),
                "+",
                self._int_literal(ctx, 1),
            )
        if text.lstrip("-").isdigit():
            return self._int_literal(ctx, int(text))
        if "[" in text and text.endswith("]"):
            bracket = text.index("[")
            base = text[:bracket]
            inner = text[bracket + 1 : -1]
            ie = self._fresh(ctx, "ie")
            ctx.sb.vertex(ie, "indexed_expression")
            base_ve = self._variable_expression(ctx, base)
            ctx.sb.edge(ie, base_ve, "child_of")
            inner_idx = self._build_index_node(ctx, inner)
            ctx.sb.edge(ie, inner_idx, "child_of")
            return ie
        return self._variable_expression(ctx, text)

    def _observe_scope_loop_names(self, batch_dims: tuple[Dim, ...]) -> tuple[str, ...]:
        """Generate per-observe loop-variable names. Distinct from
        the outer marginalize group's loop names to keep nesting
        unambiguous."""
        return tuple(f"n_{dim.name}" for dim in batch_dims)

    def _marginalize_loop_names(self, batch_dims: tuple[Dim, ...]) -> tuple[str, ...]:
        """Generate group-loop names for the outer marginalize."""
        return tuple(f"g_{dim.name}" for dim in batch_dims)

    def _build_lpdf_call(
        self,
        ctx: RenderContext,
        lpdf_name: str,
        observed_var: str,
        family: str,
        loop_names: tuple[str, ...],
        args: tuple[IRArg, ...],
        k_name: str,
        arg_names: tuple[str, ...],
    ) -> str:
        """Build `<lpdf_name>(<observed>[<loop_idx>] | <args with z=k>)`.

        Stan requires the `|` syntax for `_lpdf` / `_lpmf` calls:
        emit a `distr_expression` whose `distr_argument_list` carries
        the observed value first, then a `|`, then the remaining
        positional args.

        A reference argument is read at the observation's row and at
        the latent's atom (see `_scope_arg_expr`); any other argument
        has the latent's references replaced by `k_name`, the Stan
        subscript of the atom.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.
        lpdf_name : str
            The Stan log-density function.
        observed_var : str
            The observation.
        family : str
            The observation's family.
        loop_names : tuple[str, ...]
            The observation's row counters.
        args : tuple[IRArg, ...]
            The family's arguments.
        k_name : str
            The Stan subscript of the atom the call scores.
        arg_names : tuple[str, ...]
            The arguments' names.

        Returns
        -------
        str
            The call's vertex.
        """
        de = self._fresh(ctx, "de")
        ctx.sb.vertex(de, "distr_expression")
        fnid = self._fresh(ctx, "defnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", lpdf_name)
        ctx.sb.edge(de, fnid, "name")
        dal = self._fresh(ctx, "dal")
        ctx.sb.vertex(dal, "distr_argument_list")
        # First arg: observed[loop_idx] or bare observed.
        if loop_names:
            obs_vid = self._indexed_expression_text(ctx, observed_var, loop_names[0])
        else:
            obs_vid = self._variable_expression(ctx, observed_var)
        ctx.sb.edge(dal, self._class_code(ctx, family, obs_vid), "child_of")
        meta = FAMILY_META.get(family)
        arg_constraints = (
            meta.distribution_class.arg_constraints if meta is not None else None
        )
        for arg, arg_name in zip(args, arg_names, strict=True):
            expected = (
                arg_constraints.get(arg_name)
                if isinstance(arg_constraints, dict)
                else None
            )
            event = int(getattr(expected, "event_dim", 0))
            expr = self._scope_arg_expr(arg, loop_names, event)
            if expr is None:
                substituted = self._substitute_latent_for_k(arg, k_name)
                arg_vid = self._render_arg(ctx, substituted)
            else:
                arg_vid = render_let_expr_stan(self._let_ctx(ctx), expr)
            ctx.sb.edge(dal, arg_vid, "child_of")
        ctx.sb.edge(de, dal, "child_of")
        return de

    def _resolve_scope_lets(self, scope: tuple[IRNode, ...]) -> dict[str, LetExprNode]:
        """The expression each let of a marginalize body stands for.

        A body let may read the lets before it, so each is resolved in
        order with the earlier ones substituted: every resolved
        expression reads only the latent and names bound outside the
        body, and an observe can inline any of them whatever chain of
        lets it depends on.

        Parameters
        ----------
        scope : tuple[IRNode, ...]
            The marginalize body.

        Returns
        -------
        dict[str, LetExprNode]
            Each body let's resolved expression.
        """
        resolved: dict[str, LetExprNode] = {}
        for node in scope:
            if not isinstance(node, IRDeterministic):
                continue
            expr = node.expr
            for name, value in resolved.items():
                expr = _substitute_let_expr(
                    expr, name, index_value=value, scalar_value=value
                )
            resolved[node.name] = expr
        return resolved

    def _scope_arg_expr(
        self,
        arg: IRArg,
        loop_names: tuple[str, ...],
        event: int,
    ) -> LetExprNode | None:
        """A body observe's reference argument at its row and atom.

        The body runs once per atom of the latent and once per row of
        the observation. A reference to a body let is that let's
        resolved expression; a reference to a binding outside the body
        that carries the observation's rows is the binding itself.
        Either way the latent reads as the atom's QVR value, and every
        operand carrying the observation's rows is read at the row.

        Parameters
        ----------
        arg : IRArg
            The argument.
        loop_names : tuple[str, ...]
            The observation's row counters, outermost first.
        event : int
            The event rank the family's argument slot reads.

        Returns
        -------
        LetExprNode | None
            The argument as an expression at the row and atom, or
            `None` when the argument is not an unindexed reference to
            such a binding and renders as a plan argument.
        """
        if not isinstance(arg, IRArgRef) or arg.indices:
            return None
        if arg.name in self._marginalize_let_subs:
            expr = self._marginalize_let_subs[arg.name]
        elif self._name_ranks.get(arg.name, 0) > event and loop_names:
            expr = LetExprVar(name=arg.name)
        else:
            return None
        latent = self._current_latent_name()
        if latent is not None:
            atom = self._latent_atom_expr()
            expr = _substitute_let_expr(
                expr, latent, index_value=atom, scalar_value=atom
            )
        remaining = len(loop_names) + event
        for name in loop_names:
            if let_expr_rank(expr, self._name_ranks) == remaining:
                expr = take_row(expr, counter_value(name), self._name_ranks, "stan")
            remaining -= 1
        return self._fold_constant_products(expr)

    def _substitute_latent_for_k(self, arg: IRArg, k_name: str) -> IRArg:
        """Rewrite any IR ref whose name is the marginalize latent
        into `IRArgRef(name=k_name)` so the per-k iteration uses the
        loop variable in place of the latent."""
        latent = self._current_latent_name()
        if latent is None:
            return arg
        if isinstance(arg, IRArgRef):
            new_indices = tuple(
                self._substitute_latent_for_k(idx, k_name) for idx in arg.indices
            )
            if arg.name == latent:
                return IRArgRef(name=k_name, indices=new_indices)
            return IRArgRef(name=arg.name, indices=new_indices)
        if isinstance(arg, IRArgBroadcast):
            return IRArgBroadcast(
                value=self._substitute_latent_for_k(arg.value, k_name),
                target_shape=arg.target_shape,
            )
        if isinstance(arg, IRArgList):
            return IRArgList(
                elements=tuple(
                    self._substitute_latent_for_k(e, k_name) for e in arg.elements
                )
            )
        if isinstance(arg, IRArgMatrix):
            new_rows: list[IRArgList] = []
            for row in arg.rows:
                new_rows.append(
                    IRArgList(
                        elements=tuple(
                            self._substitute_latent_for_k(e, k_name)
                            for e in row.elements
                        )
                    )
                )
            return IRArgMatrix(rows=tuple(new_rows))
        return arg

    def _current_latent_name(self) -> str | None:
        """The latent name of the innermost active marginalize scope.

        Encoded into `_marginalize_var` as `lps_<latent>`; strip the
        prefix.
        """
        if self._marginalize_var is None:
            return None
        if self._marginalize_var.startswith("lps_"):
            return self._marginalize_var[4:]
        return None

    def _indexed_expression_text(
        self,
        ctx: RenderContext,
        base_name: str,
        index_text: str,
    ) -> str:
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        base_ve = self._variable_expression(ctx, base_name)
        ctx.sb.edge(ie, base_ve, "child_of")
        idx_node = self._build_index_node(ctx, index_text)
        ctx.sb.edge(ie, idx_node, "child_of")
        return ie

    def _log_density_name(
        self,
        family: str,
        stan_family_name: str,
    ) -> str:
        """The `<family>_lpdf` / `<family>_lpmf` name for a Stan
        distribution.

        The suffix is a property of the distribution, not of the call
        site: Stan names the discrete ones `_lpmf` and the continuous
        ones `_lpdf`, and
        [`_STAN_LOG_DENSITY_SUFFIX`][quivers.transpile.renderers.stan]
        records that split. A family with a Stan sampling name but no
        log-density function cannot be emitted as an explicit target
        increment, and dropping back to `~` would silently discard
        normalizing constants, so it raises instead.
        """
        suffix = _STAN_LOG_DENSITY_SUFFIX.get(stan_family_name)
        if suffix is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"family:no-stan-log-density:{family}: Stan's "
                    f"`{stan_family_name}` has no `_lpdf` / `_lpmf` "
                    f"form"
                ],
            )
        return f"{stan_family_name}_{suffix}"

    def _latent_cardinality(
        self,
        meta: FamilyMeta,
        node: IRMarginalize,
    ) -> int | None:
        """The number of atoms the latent is enumerated over.

        Parameters
        ----------
        meta : FamilyMeta
            The latent's family.
        node : IRMarginalize
            The marginalize block.

        Returns
        -------
        int | None
            The atom count, or `None` when the latent's support is not
            a finite integer range.
        """
        # A family that declares its own marginalize support says how
        # many atoms it has, whatever its draws' constraint looks
        # like. A relaxation family is the case that needs it: its
        # `probs` argument is one probability per row rather than a
        # simplex over the support.
        declared = marginalize_support(meta)
        if declared is not None and declared.atoms == "binary":
            return declared.size
        # Otherwise the plan states the support: the latent's index
        # object, or the width of its probability argument, sizes an
        # integer range (see `_enumerated_support` in the plan).
        sup = node.constraint.to_constraint()
        if is_int_bit(sup):
            return 2
        if is_int_category(sup):
            lower = int(getattr(sup, "lower_bound", 0))
            upper = int(getattr(sup, "upper_bound", 0))
            return upper - lower + 1
        return None

    def _declare_lps_array(
        self,
        ctx: RenderContext,
        scope_block: str,
        lps_name: str,
        batch_dims: tuple[Dim, ...],
        latent_card: int,
    ) -> None:
        """Emit `array[<batch>] vector[K] lps_<latent>;` inside the
        marginalize block. Initialised to zero implicitly by the
        Stan runtime."""
        vd = self._fresh(ctx, "lpsvd")
        ctx.sb.vertex(vd, "var_decl")
        if batch_dims:
            arr = self._fresh(ctx, "lpsarr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(vd, arr, "child_of")
        sbt = self._fresh(ctx, "lpssbt")
        ctx.sb.vertex(sbt, "sized_basic_type")
        # vector[K]
        kv = self._int_literal(ctx, latent_card)
        ctx.sb.edge(sbt, kv, "child_of")
        ctx.sb.edge(vd, sbt, "child_of")
        nm = self._fresh(ctx, "lpsnm")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", lps_name)
        ctx.sb.edge(vd, nm, "name")
        # Initializer: rep_vector(0, K) for a scalar accumulator (no
        # batch_dims) or rep_array(rep_vector(0, K), B0, B1, ...) for
        # the per-group array. The two-argument variant Stan offers
        # for ``rep_array`` requires at least one size, so the scalar
        # case must call ``rep_vector`` directly.
        rep_v = self._fresh(ctx, "lpsrv")
        ctx.sb.vertex(rep_v, "function_expression")
        rep_v_fn = self._fresh(ctx, "lpsrvf")
        ctx.sb.vertex(rep_v_fn, "identifier")
        ctx.sb.constraint(rep_v_fn, "literal-value", "rep_vector")
        ctx.sb.edge(rep_v, rep_v_fn, "name")
        rep_v_al = self._fresh(ctx, "lpsrval")
        ctx.sb.vertex(rep_v_al, "argument_list")
        ctx.sb.edge(rep_v_al, self._int_literal(ctx, 0), "child_of")
        ctx.sb.edge(rep_v_al, self._int_literal(ctx, latent_card), "child_of")
        ctx.sb.edge(rep_v, rep_v_al, "child_of")
        if batch_dims:
            init_call = self._fresh(ctx, "lpsinit")
            ctx.sb.vertex(init_call, "function_expression")
            init_fn = self._fresh(ctx, "lpsfn")
            ctx.sb.vertex(init_fn, "identifier")
            ctx.sb.constraint(init_fn, "literal-value", "rep_array")
            ctx.sb.edge(init_call, init_fn, "name")
            init_al = self._fresh(ctx, "lpsal")
            ctx.sb.vertex(init_al, "argument_list")
            ctx.sb.edge(init_al, rep_v, "child_of")
            for dim in batch_dims:
                ctx.sb.edge(init_al, self._dim_size_vertex(ctx, dim), "child_of")
            ctx.sb.edge(init_call, init_al, "child_of")
            ctx.sb.edge(vd, init_call, "child_of")
        else:
            ctx.sb.edge(vd, rep_v, "child_of")
        ctx.sb.edge(scope_block, vd, "child_of")

    def _emit_lps_init(
        self,
        ctx: RenderContext,
        scope_block: str,
        lps_name: str,
        batch_dims: tuple[Dim, ...],
        group_idx_names: tuple[str, ...],
        latent_card: int,
        stan_name: str,
        latent_args: tuple[IRArg, ...],
        latent_arg_names: tuple[str, ...],
        latent_meta: FamilyMeta,
        prior_index_args: tuple[IRArg, ...],
    ) -> None:
        """Emit `for g in batch_dims, for k in 1:K, lps[g, k] =
        <family>_lpmf(k | <latent_args>);` to seed each accumulator
        row with the latent's own log-pmf.

        `latent_arg_names` names the parameterization the latent's
        call gives, so a relaxation weighted from its logits seeds the
        atoms with the discrete family's logit density
        (`bernoulli_logit_lpmf`), which Stan computes in log space.
        `prior_index_args` selects the prior row each accumulator
        entry reads: the accumulator's own loop variables under the
        grouped reading, the observe's `via` fibration applied to the
        row loop variable under the per-row reading.
        """
        current = self._wrap_in_for_loops(ctx, scope_block, batch_dims, group_idx_names)
        k_loop = self._fresh(ctx, "ifs")
        ctx.sb.vertex(k_loop, "for_statement")
        ctx.sb.edge(current, k_loop, "child_of")
        klv = self._fresh(ctx, "ilv")
        ctx.sb.vertex(klv, "identifier")
        ctx.sb.constraint(klv, "literal-value", "k")
        ctx.sb.edge(k_loop, klv, "loopvar")
        ctx.sb.edge(k_loop, self._int_literal(ctx, 1), "child_of")
        ctx.sb.edge(k_loop, self._int_literal(ctx, latent_card), "child_of")
        kbs = self._fresh(ctx, "ibs")
        ctx.sb.vertex(kbs, "block_statement")
        ctx.sb.edge(k_loop, kbs, "child_of")
        asn = self._fresh(ctx, "iasn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(kbs, asn, "child_of")
        # LHS lps[group..., k]
        lhs_vid = self._build_indexed_lhs(ctx, lps_name, (*group_idx_names, "k"))
        ctx.sb.edge(asn, lhs_vid, "child_of")
        op = self._fresh(ctx, "iop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "=")
        ctx.sb.edge(asn, op, "child_of")
        # RHS: <stan_name>_lpmf(<atom> | latent_args with z-><atom>)
        #
        # A two-atom family weights its atoms by the family its
        # marginalize support names, the Bernoulli's
        # `[log(1 - p), log p]`, and its atoms are 0 and 1 rather than
        # the loop's 1 and 2.
        declared = marginalize_support(latent_meta)
        binary_atoms = declared is not None and declared.atoms == "binary"
        if declared is not None and binary_atoms and declared.weight_family is not None:
            weight_meta = FAMILY_META.get(declared.weight_family)
            weight_name = _stan_parameterized_name(
                declared.weight_family,
                latent_arg_names,
                (
                    weight_meta.target_names.get("stan")
                    if weight_meta is not None
                    else None
                )
                or stan_name,
            )
        else:
            weight_name = stan_name
        de = self._fresh(ctx, "ide")
        ctx.sb.vertex(de, "distr_expression")
        fn_id = self._fresh(ctx, "ifid")
        ctx.sb.vertex(fn_id, "identifier")
        ctx.sb.constraint(fn_id, "literal-value", f"{weight_name}_lpmf")
        ctx.sb.edge(de, fn_id, "name")
        dal = self._fresh(ctx, "idal")
        ctx.sb.vertex(dal, "distr_argument_list")
        # First arg: the atom this iteration scores.
        if binary_atoms:
            k_ve = render_let_expr_stan(self._let_ctx(ctx), counter_value("k"))
        else:
            k_ve = self._variable_expression(ctx, "k")
        ctx.sb.edge(dal, k_ve, "child_of")
        # Subsequent: the latent's args, with refs to per-group
        # parameter arrays indexed by the group loop variable.
        rewritten = self._index_groupplated_refs(
            latent_args, latent_meta, prior_index_args
        )
        for arg in rewritten:
            arg_vid = self._render_arg(ctx, arg)
            ctx.sb.edge(dal, arg_vid, "child_of")
        ctx.sb.edge(de, dal, "child_of")
        ctx.sb.edge(asn, de, "child_of")

    def _index_groupplated_refs(
        self,
        args: tuple[IRArg, ...],
        meta: FamilyMeta,
        prior_index_args: tuple[IRArg, ...],
    ) -> tuple[IRArg, ...]:
        """For each arg that's an [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        to a previously-declared name whose declaration plate carries
        the grouping axes, prepend `prior_index_args` as indices.

        For LDA's `Categorical(probs=theta)` inside a `marginalize ... [over=Doc]`
        scope where `theta : array[20] simplex[3]`, the ref to `theta`
        becomes `theta[word_idx[n_Word]]` under the per-row reading
        and `theta[g_Doc]` under the grouped one.
        """
        del meta
        if not prior_index_args:
            return args
        out: list[IRArg] = []
        for arg in args:
            if not isinstance(arg, IRArgRef) or arg.indices:
                out.append(arg)
                continue
            declared = self._declared_shapes.get(arg.name)
            if declared is None:
                out.append(arg)
                continue
            _, declared_plate = declared
            if len(declared_plate.batch_dims) < len(prior_index_args):
                out.append(arg)
                continue
            out.append(IRArgRef(name=arg.name, indices=prior_index_args))
        return tuple(out)

    def _emit_lps_accumulate(
        self,
        ctx: RenderContext,
        scope_block: str,
        lps_name: str,
        batch_dims: tuple[Dim, ...],
        group_idx_names: tuple[str, ...],
    ) -> None:
        """Emit `for g in batch_dims, target += log_sum_exp(lps[g]);`."""
        current = self._wrap_in_for_loops(ctx, scope_block, batch_dims, group_idx_names)
        ts = self._fresh(ctx, "ats")
        ctx.sb.vertex(ts, "target_statement")
        ctx.sb.edge(current, ts, "child_of")
        # log_sum_exp(lps[group_idx])
        fn = self._fresh(ctx, "lsefn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "lsefnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", "log_sum_exp")
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "lseal")
        ctx.sb.vertex(al, "argument_list")
        lps_idx = self._build_indexed_arg_expression(ctx, lps_name, group_idx_names)
        ctx.sb.edge(al, lps_idx, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        ctx.sb.edge(ts, fn, "child_of")

    def _build_indexed_arg_expression(
        self,
        ctx: RenderContext,
        base_name: str,
        index_names: tuple[str, ...],
    ) -> str:
        """Build `<base>[<idx0>, <idx1>, ...]` as an expression node."""
        if not index_names:
            return self._variable_expression(ctx, base_name)
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        ctx.sb.edge(ie, self._variable_expression(ctx, base_name), "child_of")
        for idx_name in index_names:
            idx_vid = self._build_index_node(ctx, idx_name)
            ctx.sb.edge(ie, idx_vid, "child_of")
        return ie

    # ----- broadcast dispatch -----

    def broadcast(
        self,
        ctx: RenderContext,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit a Stan broadcast call:
        `rep_vector(<value>, K)` for 1D, `rep_matrix(<value>, R, C)`
        for 2D.

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
        if len(target_shape) == 1:
            fn_name = "rep_vector"
        elif len(target_shape) == 2:
            fn_name = "rep_matrix"
        else:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"broadcast:rank:{len(target_shape)}: Stan "
                    f"broadcast supports up to rank 2"
                ],
            )
        fn = self._fresh(ctx, "bfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "bfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", fn_name)
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "bal")
        ctx.sb.vertex(al, "argument_list")
        val_vid = self._render_arg(ctx, value)
        ctx.sb.edge(al, val_vid, "child_of")
        for size in target_shape:
            ctx.sb.edge(al, self._int_literal(ctx, size), "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    # ----- arg rendering helpers -----

    def _render_list(
        self,
        ctx: RenderContext,
        arg: IRArgList,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, Stan list args render as a vector
        literal `[<e0>, <e1>, ...]'`.

        The Stan grammar's `vector_expression` accepts the bracketed
        list; the transpose `'` ensures the literal is a column vector
        when used as an arg to a distribution expecting `vector[N]`.
        """
        ve = self._fresh(ctx, "vex")
        ctx.sb.vertex(ve, "vector_expression")
        for element in arg.elements:
            child_vid = self._render_arg(ctx, element)
            ctx.sb.edge(ve, child_vid, "child_of")
        # Wrap in a postfix_op_expression to render the transpose.
        post = self._fresh(ctx, "poe")
        ctx.sb.vertex(post, "postfix_op_expression")
        ctx.sb.constraint(post, "chose-alt-fingerprint", "'")
        ctx.sb.edge(post, ve, "child_of")
        return post

    def _render_matrix(
        self,
        ctx: RenderContext,
        arg: IRArgMatrix,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, Stan matrix args render as
        `to_matrix({{<row0>}, {<row1>}, ...})`."""
        fn = self._fresh(ctx, "tmfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "tmfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", "to_matrix")
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "tmal")
        ctx.sb.vertex(al, "argument_list")
        # The outer `{ ... }` is a vector_expression of vector_expressions.
        outer_ve = self._fresh(ctx, "tmve")
        ctx.sb.vertex(outer_ve, "vector_expression")
        for row in arg.rows:
            row_ve = self._fresh(ctx, "tmrow")
            ctx.sb.vertex(row_ve, "vector_expression")
            for element in row.elements:
                el_vid = self._render_arg(ctx, element)
                ctx.sb.edge(row_ve, el_vid, "child_of")
            ctx.sb.edge(outer_ve, row_ve, "child_of")
        ctx.sb.edge(al, outer_ve, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _render_arg(
        self,
        ctx: RenderContext,
        arg: IRArg,
    ) -> SchemaFragment:
        """Render any [`IRArg`][quivers.transpile.ir.IRArg] to a Stan
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
        if isinstance(arg, IRArgKernel):
            return self._render_kernel(ctx, arg)
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"arg:unknown:{type(arg).__name__}"],
        )

    def _render_kernel(
        self,
        ctx: RenderContext,
        arg: IRArgKernel,
    ) -> SchemaFragment:
        """Emit ``gp_exp_quad_cov(x, 1.0, length_scale) +
        diag_matrix(rep_vector(jitter, N))`` for an
        [`IRArgKernel`][quivers.transpile.ir.IRArgKernel].

        Stan's built-in ``gp_exp_quad_cov(x, alpha, rho)`` returns the
        N-by-N squared-exponential covariance matrix; the diagonal
        jitter ensures positive-definiteness under the Cholesky
        decomposition `multi_normal` performs internally.
        """
        if arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"arg:kernel:{arg.kernel}: only rbf is implemented"],
            )
        # gp_exp_quad_cov(x, alpha=1.0, length_scale)
        kernel_fn = self._fresh(ctx, "gpfn")
        ctx.sb.vertex(kernel_fn, "function_expression")
        kernel_id = self._fresh(ctx, "gpfid")
        ctx.sb.vertex(kernel_id, "identifier")
        ctx.sb.constraint(kernel_id, "literal-value", "gp_exp_quad_cov")
        ctx.sb.edge(kernel_fn, kernel_id, "name")
        kernel_args = self._fresh(ctx, "gpal")
        ctx.sb.vertex(kernel_args, "argument_list")
        x_ref = self._variable_expression(ctx, arg.x_name)
        ctx.sb.edge(kernel_args, x_ref, "child_of")
        # amplitude = 1.0
        alpha = self._fresh(ctx, "gpal_a")
        ctx.sb.vertex(alpha, "real_literal")
        ctx.sb.constraint(alpha, "literal-value", "1.0")
        ctx.sb.edge(kernel_args, alpha, "child_of")
        # length_scale literal
        ls = self._fresh(ctx, "gpal_l")
        ctx.sb.vertex(ls, "real_literal")
        ctx.sb.constraint(ls, "literal-value", repr(arg.length_scale))
        ctx.sb.edge(kernel_args, ls, "child_of")
        ctx.sb.edge(kernel_fn, kernel_args, "child_of")
        # diag_matrix(rep_vector(jitter, N))
        diag_fn = self._fresh(ctx, "djfn")
        ctx.sb.vertex(diag_fn, "function_expression")
        diag_id = self._fresh(ctx, "djfid")
        ctx.sb.vertex(diag_id, "identifier")
        ctx.sb.constraint(diag_id, "literal-value", "diag_matrix")
        ctx.sb.edge(diag_fn, diag_id, "name")
        diag_args = self._fresh(ctx, "djal")
        ctx.sb.vertex(diag_args, "argument_list")
        rep_fn = self._fresh(ctx, "rvfn")
        ctx.sb.vertex(rep_fn, "function_expression")
        rep_id = self._fresh(ctx, "rvfid")
        ctx.sb.vertex(rep_id, "identifier")
        ctx.sb.constraint(rep_id, "literal-value", "rep_vector")
        ctx.sb.edge(rep_fn, rep_id, "name")
        rep_args = self._fresh(ctx, "rval")
        ctx.sb.vertex(rep_args, "argument_list")
        jit = self._fresh(ctx, "jit")
        ctx.sb.vertex(jit, "real_literal")
        ctx.sb.constraint(jit, "literal-value", repr(arg.jitter))
        ctx.sb.edge(rep_args, jit, "child_of")
        ctx.sb.edge(rep_args, self._int_literal(ctx, arg.grid_size), "child_of")
        ctx.sb.edge(rep_fn, rep_args, "child_of")
        ctx.sb.edge(diag_args, rep_fn, "child_of")
        ctx.sb.edge(diag_fn, diag_args, "child_of")
        # Sum: gp_exp_quad_cov(...) + diag_matrix(rep_vector(...))
        sum_v = self._fresh(ctx, "ksum")
        ctx.sb.vertex(sum_v, "infix_op_expression")
        ctx.sb.constraint(sum_v, "chose-alt-fingerprint", "+")
        ctx.sb.edge(sum_v, kernel_fn, "child_of")
        ctx.sb.edge(sum_v, diag_fn, "child_of")
        return sum_v

    def _render_number(self, ctx: RenderContext, value: float) -> str:
        if float(value).is_integer():
            return self._int_literal(ctx, int(value))
        v = self._fresh(ctx, "rl")
        ctx.sb.vertex(v, "real_literal")
        ctx.sb.constraint(v, "literal-value", repr(float(value)))
        return v

    def _latent_atom_expr(self) -> LetExprNode:
        """The QVR value the latent takes on iteration `k` of the scope.

        The latent's atoms count from zero, as every QVR index does,
        and the enumeration loop counts from one, so on iteration `k`
        the latent is `k - 1`. Where the latent subscripts an array the
        subscript translation folds that back to `k`; where it is a
        number (`real(z + 1)`, a relaxation family's `0` / `1`) it
        keeps its own value.
        """
        if self._marginalize_atom is not None:
            # The scope is unrolled at this atom, so the latent is the
            # number the atom stands for rather than an expression in
            # the loop variable.
            return LetExprLiteral(value=self._marginalize_atom, integral=True)
        return counter_value("k")

    def _fold_constant_products(self, expr: LetExprNode) -> LetExprNode:
        """Fold `0 * x` to `0` and `1 * x` to `x`.

        An unrolled atom substitutes a literal into the scope's `let`,
        and the off atom of a relaxation family makes that literal a
        zero: `gated_rate = z * rate` becomes `0 * rate`. Left
        standing, that reads as a value which is zero but which the
        parameters still reach, so the density's derivative at it is
        `0 * inf` wherever the observed count is positive, and Stan
        rejects every initial value it tries. Folded, the rate is a
        constant the gradient does not run through, which is what the
        model means: the off state of a zero-inflated count carries no
        rate at all.
        """
        if not isinstance(expr, LetExprBinOp):
            return expr
        left = self._fold_constant_products(expr.left)
        right = self._fold_constant_products(expr.right)
        if expr.op == "*":
            for a, b in ((left, right), (right, left)):
                if isinstance(a, LetExprLiteral) and a.value == 0.0:
                    return LetExprLiteral(value=0.0, integral=a.integral)
                if isinstance(a, LetExprLiteral) and a.value == 1.0 and a.integral:
                    # A real `1.0` stays: it may be what makes the
                    # product, and a later quotient, real in Stan.
                    return b
        return LetExprBinOp(op=expr.op, left=left, right=right)

    def _render_ref(self, ctx: RenderContext, arg: IRArgRef) -> SchemaFragment:
        """Render an IRArgRef. Bare-name refs become a
        `variable_expression`; indexed refs nest `indexed_expression`
        nodes.

        A subscript that reads a program value (an index covariate, a
        drawn class) holds a zero-based QVR index and is read one
        higher; a renderer's own loop counter or enumerated atom is a
        Stan subscript already.
        """
        base = self._variable_expression(ctx, arg.name)
        if not arg.indices:
            return base
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        ctx.sb.edge(ie, base, "child_of")
        for idx in arg.indices:
            idx_vid = self._fresh(ctx, "idx")
            ctx.sb.vertex(idx_vid, "index")
            inner_vid = self._render_arg(ctx, idx)
            if isinstance(idx, IRArgRef) and self._is_program_value(idx.name):
                inner_vid = self._stan_binop(
                    ctx, inner_vid, "+", self._int_literal(ctx, 1)
                )
            ctx.sb.edge(idx_vid, inner_vid, "child_of")
            ctx.sb.edge(ie, idx_vid, "child_of")
        return ie

    def _class_code(self, ctx: RenderContext, family: str, variate: str) -> str:
        """A variate as the Stan log-pmf of its family reads it.

        A class-index family (Categorical, OrderedLogistic,
        OrderedProbit) scores a class code, which QVR counts from zero
        and Stan's `categorical_lpmf` and its kin count from one, so the
        code is read one higher; any other variate is read as it is.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.
        family : str
            The family scoring the variate.
        variate : str
            The rendered variate.

        Returns
        -------
        str
            The variate's vertex, shifted for a class code.
        """
        if not is_class_code_family(family):
            return variate
        return self._stan_binop(ctx, variate, "+", self._int_literal(ctx, 1))

    def _reads_real_array(self, arg: IRArg) -> bool:
        """Whether an argument reads a one-axis Stan array of reals.

        Parameters
        ----------
        arg : IRArg
            The argument, after its row subscripts.

        Returns
        -------
        bool
            True for a reference whose declared Stan type, after its
            subscripts, is `array[N] real`: a site drawn over an axis
            with no event of its own, which a vector argument reads
            through `to_vector`.
        """
        if not isinstance(arg, IRArgRef):
            return False
        declared = self._declared_shapes.get(arg.name)
        if declared is None:
            return False
        sup, plate = declared
        if plate.event_dims or not _is_continuous_support(sup):
            return False
        return len(plate.batch_dims) - len(arg.indices) == 1

    def _is_program_value(self, name: str) -> bool:
        """Whether a name is a value of the program rather than a
        renderer's loop counter or enumerated atom.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        bool
            True for an input, a site, or a binding of the program.
        """
        return name in self._name_ranks

    def _render_family_ref(
        self,
        ctx: RenderContext,
        arg: IRArgFamilyRef,
    ) -> SchemaFragment:
        """Render an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]:
        look up the morphism's `init_family` clause and emit the
        Stan inline call form."""
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"arg:family-ref:{arg.name}: no morphism with "
                    f"`~ Family(...)` init clause in scope"
                ],
            )
        init = decl.init_family
        inner_meta = FAMILY_META.get(init.family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{init.family}"],
            )
        inner_stan_name = inner_meta.target_names.get("stan")
        if inner_stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{init.family}"],
            )
        fn = self._fresh(ctx, "frfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "frfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", inner_stan_name)
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "fral")
        ctx.sb.vertex(al, "argument_list")
        for _, raw in bind_family_arguments(init.family, init.args or ()):
            arg_vid = self._render_init_family_arg(ctx, raw)
            ctx.sb.edge(al, arg_vid, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _render_init_family_arg(
        self,
        ctx: RenderContext,
        arg: DrawArg,
    ) -> SchemaFragment:
        """Render an `init_family` argument.

        `init_family` clauses carry their constants as atomic draw
        arguments, whose wire values are ``float`` literals or ``str``
        identifiers (a bracket-indexed reference re-serialises to
        ``name[i]`` text). Structured list / matrix args are not
        admitted in an `init_family` clause, so every arg here is
        atomic.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.
        arg : DrawArg
            The argument.

        Returns
        -------
        SchemaFragment
            The rendered number or name.
        """
        raw = atom_value(arg)
        if isinstance(raw, str):
            return self._render_named_or_number(ctx, raw)
        return self._render_number(ctx, float(raw))

    def _render_named_or_number(self, ctx: RenderContext, raw: str) -> SchemaFragment:
        """Render a wire-form atom spelled as text.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.
        raw : str
            A number's spelling or a name.

        Returns
        -------
        SchemaFragment
            The number, or the name as a variable.
        """
        stripped = raw.strip()
        try:
            value = float(stripped)
        except ValueError:
            return self._variable_expression(ctx, stripped)
        return self._render_number(ctx, value)

    # ----- shared utilities -----

    def _variable_expression(self, ctx: RenderContext, name: str) -> str:
        ve = self._fresh(ctx, "ve")
        ctx.sb.vertex(ve, "variable_expression")
        ident = self._fresh(ctx, "vid")
        ctx.sb.vertex(ident, "identifier")
        ctx.sb.constraint(ident, "literal-value", name)
        ctx.sb.edge(ve, ident, "child_of")
        return ve

    def _int_literal(self, ctx: RenderContext, value: int | float) -> str:
        v = self._fresh(ctx, "il")
        ctx.sb.vertex(v, "integer_literal")
        ctx.sb.constraint(v, "literal-value", str(int(value)))
        return v

    def _ensure_block(self, ctx: RenderContext, kind: BlockKind) -> str:
        """Lazily emit a top-level Stan block of the given kind.

        Blocks are children of the top `program` vertex. Each emit
        happens at most once per render call.
        """
        if kind in self._blocks:
            return self._blocks[kind]
        vid = self._fresh(ctx, kind)
        ctx.sb.vertex(vid, _BLOCK_KIND_MAP[kind])
        ctx.sb.edge("prog", vid, "child_of")
        self._blocks[kind] = vid
        return vid

    def _dim_size_vertex(self, ctx: RenderContext, dim: Dim) -> str:
        """Materialise an integer literal (for `DimStatic`) or a
        variable reference (for `DimDynamic`) representing the dim's
        size."""
        if isinstance(dim, DimStatic):
            return self._int_literal(ctx, dim.size)
        if isinstance(dim, DimDynamic):
            return self._variable_expression(ctx, dim.size_name)
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"dim:unknown:{type(dim).__name__}"],
        )

    def _fresh(self, ctx: RenderContext, prefix: str) -> str:
        """Return a fresh vertex id with `prefix`. Renderer-internal
        counter; doesn't disturb the base's `RenderContext.fresh_counter`."""
        del ctx
        self._fresh_n += 1
        return f"{prefix}_{self._fresh_n}"

    # ----- IR-walk overrides -----

    def _dispatch_node(self, ctx: RenderContext, node: IRNode) -> None:
        """Stan-specific dispatch of one plan node.

        Raises
        ------
        UnsupportedConstruct
            If a sample draws a class code, which Stan has no parameter
            type for, or the node has no Stan form.
        """
        if isinstance(node, IRSample):
            if is_class_code_family(node.family):
                raise UnsupportedConstruct(
                    "qvr-stan",
                    [
                        f"sample:discrete-latent:{node.name}: Stan has no "
                        f"discrete parameters, so a {node.family} draw must be "
                        f"marginalized rather than sampled"
                    ],
                )
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
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="data",
            )
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
        if isinstance(node, IRDataInput):
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="data",
            )
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(ctx, node)
            return
        if isinstance(node, IRCall):
            # The callee is a user-defined function in the `functions`
            # block, so the call binds like a deterministic; whether
            # Stan can define the callee is settled when the block is
            # rendered, which refuses an effectful one.
            self._emit_deterministic(
                ctx,
                IRDeterministic(
                    name=node.name,
                    expr=LetExprCall(
                        func=f"qiec_{node.callee}", args=tuple(node.arguments)
                    ),
                    constraint=CSReal(),
                    plate=node.plate,
                ),
            )
            return
        if isinstance(node, IRScore):
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_return(ctx, node.names)
            return
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"node:{type(node).__name__}"],
        )

    def _emit_deterministic(
        self,
        ctx: RenderContext,
        node: IRDeterministic,
    ) -> None:
        """Emit a deterministic let-binding into the
        ``transformed_parameters`` block.

        Scalar plate (``batch_dims=()``): the binding is a single
        ``<type> <name> = <expr>;`` declaration whose RHS is
        [`node.expr`][quivers.transpile.ir.IRDeterministic.expr]
        rendered through
        `render_let_expr_stan`.

        Non-scalar plate (``batch_dims != ()``): the binding splits
        into a declaration of the array type and a ``for`` loop that
        assigns elementwise. Stan's elementwise arithmetic is defined
        on ``vector``/``matrix`` types but not on
        ``array[N] real``; using a loop sidesteps the type mismatch
        and works for arbitrary expression shapes, including those
        mixing scalar parameters with plated data inputs (the canonical
        ``a + b * x_design`` regression form).

        The loop reads the expression at its row through
        [`take_row`][quivers.transpile.renderers._rows.take_row]: each
        operand carrying the binding's leading axis (``x_design``) is
        read at the row, and an operand of lower rank (``a``, ``b``)
        passes through unchanged.

        A binding a consumer reads as an event (Categorical's
        `probs`) is declared with its trailing axes as that event, a
        `vector`, and its leading axes as array dimensions; see
        `_compute_let_events`.
        """
        parent = self._ensure_block(ctx, "transformed_parameters")
        if node.name in self._declared["transformed_parameters"]:
            return
        self._declared["transformed_parameters"].add(node.name)
        sup = node.constraint.to_constraint()
        event = self._let_events.get(node.name, 0)
        stan_plate = self._stan_plate(node.name, node.plate)
        self._declared_shapes[node.name] = (sup, stan_plate)
        # Normalise the RHS shape for fan / parallel-branch literals.
        # The lower phase aggregates parallel-branch outputs as
        # `LetExprList` regardless of length; for the Stan declaration
        # the literal's length dictates the declared type:
        #   * length 1 -> unwrap to the inner scalar (the wrapping
        #     list serves no Stan purpose and breaks `real X = {a};`).
        #   * length >1 -> declare `array[len] real` so the
        #     `{a, b, ...}` initialiser type-checks.
        body_expr = node.expr
        list_array_size: int | None = None
        if isinstance(body_expr, LetExprList) and not node.plate.batch_dims:
            if len(body_expr.items) == 1:
                body_expr = body_expr.items[0]
            elif len(body_expr.items) > 1 and event == 0:
                list_array_size = len(body_expr.items)
        decl = self._fresh(ctx, "tpvd")
        ctx.sb.vertex(decl, "top_var_decl")
        ctx.sb.edge(parent, decl, "child_of")
        promoted_k = self._vector_promotions.get(node.name)
        if list_array_size is not None and promoted_k is None:
            # `array[N] real` declaration, no batch dims.
            arr = self._fresh(ctx, "tparr")
            ctx.sb.vertex(arr, "arr_dims")
            ctx.sb.edge(
                arr,
                self._int_literal(ctx, list_array_size),
                "child_of",
            )
            ctx.sb.edge(decl, arr, "child_of")
        elif stan_plate.batch_dims:
            arr = self._fresh(ctx, "tparr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in stan_plate.batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(decl, arr, "child_of")
        tvt = self._fresh(ctx, "tpvt")
        ctx.sb.vertex(tvt, "top_var_type")
        if promoted_k is not None:
            self._emit_vector_type_of_size(ctx, tvt, promoted_k)
        else:
            self._emit_type(ctx, tvt, sup, stan_plate.event_dims)
        ctx.sb.edge(decl, tvt, "child_of")
        nm = self._fresh(ctx, "tpnm")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", node.name)
        ctx.sb.edge(decl, nm, "name")
        # Inline-init path: emit `<type> <name> = <expr>;` as a single
        # `top_var_decl` with an init child. Correct when either:
        #   - the binding has no array dimensions; OR
        #   - the binding is an array of reals whose expression is
        #     already shape-aware (a
        #     [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
        #     or [`LetExprList`][quivers.dsl.ast_nodes.LetExprList]
        #     literal produces the full array on the RHS, and a
        #     [`LetExprAffineMap`][quivers.transpile.ir.LetExprAffineMap]
        #     produces it as one matrix-vector product, so Stan accepts
        #     the array-to-array assignment directly).
        if not stan_plate.batch_dims or (
            event == 0
            and isinstance(body_expr, (LetExprAffineMap, LetExprFactor, LetExprList))
        ):
            self._refuse_several_axis_reductions(node.name, body_expr)
            rhs = render_let_expr_stan(self._let_ctx(ctx), body_expr)
            if promoted_k is not None:
                rhs = self._wrap_rep_vector(ctx, rhs, promoted_k)
            rhs = self._as_event(ctx, rhs, event)
            ctx.sb.edge(decl, rhs, "child_of")
            return

        # Array case: emit
        # `for (lv0 in 1:N0) ... { name[lv0]...[lvN-1] = expr; }`
        # with the expression read at the loop's row. The for-loop is
        # required because Stan's elementwise arithmetic is undefined
        # on `array[N] real`; the loop computes one entry at a time,
        # each a Stan `real` (or the event's `vector`).
        loop_names = self.index_for(
            ctx, Plate(event_dims=(), batch_dims=stan_plate.batch_dims)
        )
        inner_parent = self._wrap_in_for_loops(
            ctx, parent, stan_plate.batch_dims, loop_names
        )
        subbed = self._take_loop_rows(node.expr, node.plate, loop_names)
        self._refuse_several_axis_reductions(node.name, subbed)
        # LHS: indexed_lhs <name>[lv0, lv1, ...]
        asn = self._fresh(ctx, "asn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(inner_parent, asn, "child_of")
        lhs_vid = self._build_indexed_lhs(ctx, node.name, loop_names)
        ctx.sb.edge(asn, lhs_vid, "child_of")
        op = self._fresh(ctx, "aop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "=")
        ctx.sb.edge(asn, op, "child_of")
        rhs = render_let_expr_stan(self._let_ctx(ctx), subbed)
        if promoted_k is not None:
            rhs = self._wrap_rep_vector(ctx, rhs, promoted_k)
        rhs = self._as_event(ctx, rhs, event)
        ctx.sb.edge(asn, rhs, "child_of")

    def _as_event(self, ctx: RenderContext, rhs: str, event: int) -> str:
        """Convert a rendered value to the Stan type of its event.

        Parameters
        ----------
        ctx : RenderContext
            The render call's mutable state.
        rhs : str
            The rendered value, an array of reals or already the event
            type.
        event : int
            The number of event axes: none keeps the value, one makes
            it a `vector`, two a `matrix`.

        Returns
        -------
        str
            The converted value's vertex.

        Raises
        ------
        UnsupportedConstruct
            If the event has more than two axes, which Stan has no
            type for.
        """
        if event == 0:
            return rhs
        if event > 2:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"declare:let-event-rank:{event}: Stan has no {event}-axis event"],
            )
        fn = self._fresh(ctx, "evfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "evfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(
            fnid, "literal-value", "to_vector" if event == 1 else "to_matrix"
        )
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "eval")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(al, rhs, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _take_loop_rows(
        self,
        expr: LetExprNode,
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> LetExprNode:
        """Read a binding's expression at one iteration of its loops.

        Each loop runs over one leading axis of the binding, and its
        counter stands for the zero-based position ``counter - 1``. An
        expression with fewer axes than remain is the same at every
        position of the leading one (QVR aligns trailing axes), so
        that loop selects nothing from it.

        Parameters
        ----------
        expr : LetExprNode
            The binding's expression.
        plate : Plate
            The binding's plate in the plan.
        loop_names : tuple[str, ...]
            The loop counters, outermost first.

        Returns
        -------
        LetExprNode
            The expression for the entry the loops reach.
        """
        remaining = len(plate.batch_dims) + len(plate.event_dims)
        out = expr
        for name in loop_names:
            if let_expr_rank(out, self._name_ranks) == remaining:
                out = take_row(out, counter_value(name), self._name_ranks, "stan")
            remaining -= 1
        return out

    def _wrap_rep_vector(
        self,
        ctx: RenderContext,
        inner_vid: str,
        size: int,
    ) -> str:
        """Wrap an existing expression vertex in a
        `rep_vector(<inner>, K)` call.

        Used by the vector-promotion path: when a scalar deterministic
        binding produces a name that is consumed by an event_dim>=1
        slot, the per-element RHS is broadcast to a length-K vector
        so the downstream consumer's shape contract is satisfied.
        """
        fn = self._fresh(ctx, "rvfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "rvfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", "rep_vector")
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "rval")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(al, inner_vid, "child_of")
        ctx.sb.edge(al, self._int_literal(ctx, size), "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _refuse_several_axis_reductions(self, name: str, expr: LetExprNode) -> None:
        """Refuse a reduction whose argument still has several axes.

        A let reduction collapses the last axis of its argument. Inside
        a binding's plate loop each plated name is read one row at a
        time, so the argument usually has at most one axis left, which
        Stan's `sum` and its kin reduce whole. An argument with two or
        more axes left (a matrix-valued event, or a table built over
        two axes) has no single Stan call that reduces only its last
        axis, so the binding is refused rather than reduced whole.

        Parameters
        ----------
        name : str
            The binding.
        expr : LetExprNode
            Its right-hand side, as the loop reads it.

        Raises
        ------
        UnsupportedConstruct
            If the expression holds such a reduction.
        """
        ranks = {
            declared: len(plate.batch_dims) + len(plate.event_dims)
            for declared, (_support, plate) in self._declared_shapes.items()
        }
        found = several_axis_reductions(expr, ranks)
        if found:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"let-expr:axis-reduction:stan: `{found[0].func}(...)` in "
                    f"{name!r} reduces only the last axis of an argument with "
                    f"several axes, and Stan's `{found[0].func}` reduces every "
                    f"axis of an array or matrix"
                ],
            )

    def _emit_score(self, ctx: RenderContext, node: IRScore) -> None:
        """Emit ``target += <expr>;`` to the ``model`` block.

        Renders
        [`node.expr`][quivers.transpile.ir.IRScore.expr] through
        `render_let_expr_stan`
        so the increment is a real Stan expression rather than a
        bare-name reference to an undeclared variable.
        """
        parent = self._ensure_block(ctx, "model")
        ts = self._fresh(ctx, "scts")
        ctx.sb.vertex(ts, "target_statement")
        ctx.sb.edge(parent, ts, "child_of")
        rhs = render_let_expr_stan(
            self._let_ctx(ctx),
            node.expr,
        )
        ctx.sb.edge(ts, rhs, "child_of")

    def _emit_return(
        self,
        ctx: RenderContext,
        names: tuple[str, ...],
    ) -> None:
        """Emit `generated quantities { <type> <name>_value = <name>;
        ... }` for each return variable.

        Stan has no program-level return; the idiom is to expose
        sampled / let-bound variables in `generated quantities`.
        Re-declaring an existing name is illegal in Stan, so each
        return-var emerges as `<var>_value`.
        """
        for var in names:
            self._emit_gq_alias(ctx, var)

    def _emit_gq_alias(self, ctx: RenderContext, var: str) -> None:
        decl_name = f"{var}_value"
        if decl_name in self._declared["generated_quantities"]:
            return
        self._declared["generated_quantities"].add(decl_name)
        parent = self._ensure_block(ctx, "generated_quantities")
        decl = self._fresh(ctx, "gqvd")
        ctx.sb.vertex(decl, "top_var_decl")
        ctx.sb.edge(parent, decl, "child_of")
        # Reuse the declared support / plate by looking up the
        # parameter / data block entry. For LDA the `theta` is an
        # `array[20] simplex[3]`; emit the same type here.
        sup, plate = self._declared_shape(var)
        if plate.batch_dims:
            arr = self._fresh(ctx, "gqarr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in plate.batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(decl, arr, "child_of")
        tvt = self._fresh(ctx, "gqvt")
        ctx.sb.vertex(tvt, "top_var_type")
        class_width = self._class_index_widths.get(var)
        if class_width is not None:
            # The alias holds the outcome as its declaration does, as a
            # zero-based class code.
            self._emit_int_type(ctx, tvt, lower=0, upper=class_width - 1)
        else:
            self._emit_type(ctx, tvt, sup, plate.event_dims)
        ctx.sb.edge(decl, tvt, "child_of")
        nm = self._fresh(ctx, "gqnm")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", decl_name)
        ctx.sb.edge(decl, nm, "name")
        rhs = self._variable_expression(ctx, var)
        ctx.sb.edge(decl, rhs, "child_of")

    def _declared_shape(self, var: str) -> tuple[Constraint, Plate]:
        """Look up a previously-declared name's (support, plate) for
        the generated-quantities aliasing emission.

        Raises `UnsupportedConstruct`
        when the return-var was never declared.
        """
        info = self._declared_shapes.get(var)
        if info is not None:
            return info
        raise UnsupportedConstruct(
            "qvr-stan",
            [
                f"return:undeclared:{var}: return variable was not "
                f"declared in any earlier IR node; cannot determine "
                f"shape for generated-quantities aliasing"
            ],
        )

    @property
    def _declared_shapes(self) -> dict[str, tuple[Constraint, Plate]]:
        """Per-render map from declared name to (support, plate).

        Reset at the top of every `render()` call so repeated render
        invocations on the same renderer produce identical schemas.
        """
        return self._declared_shapes_state

    # ----- morphism / let resolution -----

    def _resolve_morphisms_and_lets(
        self,
    ) -> tuple[dict[str, MorphismDecl], dict[str, Expr]]:
        """Build the morphism / let tables from the source module.

        Used by [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        resolution. Empty when the renderer was constructed without a
        source module; in that case `IRArgFamilyRef` rendering raises.
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


def _vector_argument_slots(
    meta: FamilyMeta, arg_names: tuple[str, ...]
) -> frozenset[int]:
    """The positions of a family call's arguments that read a vector.

    Parameters
    ----------
    meta : FamilyMeta
        The family.
    arg_names : tuple[str, ...]
        The call's argument names, in order.

    Returns
    -------
    frozenset[int]
        Each position whose constraint has one event axis (Categorical's
        `probs` or `logits`, a multivariate normal's `loc`).
    """
    constraints = meta.distribution_class.arg_constraints
    if not isinstance(constraints, dict):
        return frozenset()
    return frozenset(
        position
        for position, name in enumerate(arg_names)
        if int(getattr(constraints.get(name), "event_dim", 0)) == 1
    )


def _is_continuous_support(sup: Constraint) -> bool:
    """True iff the support is real-valued (continuous), rather than
    integer-valued (discrete).

    Used by the marginalize dispatch: discrete latents (Bernoulli,
    Categorical, ...) compile to Stan's `log_sum_exp` enumeration;
    continuous latents (ContinuousBernoulli, Beta, ...) compile to
    a parameter declaration plus an inline `target +=` increment,
    leaning on Stan's HMC to handle the joint over the latent.

    Recognises scalar reals (any of the four bounded variants),
    vectors, simplices, covariance / correlation Choleskys, and
    matrices. Returns False for any integer support
    ([`is_int_bit`][quivers.transpile.ir.is_int_bit],
    [`is_int_category`][quivers.transpile.ir.is_int_category],
    [`is_int_count`][quivers.transpile.ir.is_int_count]).
    """
    return (
        is_real_scalar(sup)
        or is_real_positive(sup)
        or is_real_unit_interval(sup)
        or is_real_bounded_interval(sup)
        or is_real_vector(sup)
        or is_real_simplex(sup)
        or is_real_one_hot(sup)
        or is_real_cov_matrix(sup)
        or is_real_corr_chol(sup)
        or is_real_matrix(sup)
    )


# Stan-side argument injection for QVR families whose underlying
# torch distribution carries fewer parameters than Stan's same-named
# distribution. `HalfNormal(scale)` maps to Stan's `normal(0, scale)`;
# the renderer prepends `IRArgNumber(0)` before emission.
_PREPEND_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

# Per-family insertion of `IRArgNumber(0)` at an intermediate
# position. `HalfStudentT(df, scale)` maps to Stan's
# `student_t(df, 0, scale)`; the renderer injects `mu = 0` between
# the first and second user-supplied args.
_INSERT_ZERO_AT: dict[str, int] = {
    "HalfStudentT": 1,
}


_BLOCK_KIND_MAP: dict[BlockKind, str] = {
    "data": "data",
    "parameters": "parameters",
    "transformed_parameters": "transformed_parameters",
    "model": "model",
    "generated_quantities": "generated_quantities",
    "function_body": "functions",
}


def format_stan(source: bytes) -> bytes:
    """Whitespace-only formatter for Stan bytes.

    Invariants enforced:

    * One statement per line.
    * Two-space indentation.
    * Block headers at column 0.
    * No trailing whitespace.
    * No space-before-comma.
    * Space-after-comma.

    The formatter only touches whitespace and idiomatic spacing
    around `[`, `<`, and `=` inside type constraints; every
    non-whitespace token from the input survives.
    """
    text = source.decode("utf-8")
    # Stage 1: tighten Stan-specific spacing on the raw pretty-print
    # output so subsequent line-splitting matches Stan idioms.
    text = _tighten_stan_spacing(text)
    # Stage 2: split on statement boundaries and brace pairs and
    # re-indent.
    lines = _split_and_indent(text)
    # Stage 3: final whitespace pass.
    out: list[str] = []
    prev_blank = False
    for line in lines:
        s = line.rstrip()
        if not s:
            if not prev_blank and out:
                out.append("")
            prev_blank = True
            continue
        prev_blank = False
        out.append(s)
    return ("\n".join(out) + "\n").encode("utf-8")


def _tighten_stan_spacing(text: str) -> str:
    """Apply Stan idiomatic spacing fixes: tight brackets / angle
    constraints / commas. Whitespace-only inside identifiers and
    literals."""
    out = text
    # Remove space before comma.
    while " ," in out:
        out = out.replace(" ,", ",")
    # No space inside `<...>` type constraints.
    out = (
        out.replace("<lower = ", "<lower=")
        .replace("<upper = ", "<upper=")
        .replace(", upper = ", ", upper=")
        .replace(", lower = ", ", lower=")
        .replace("< lower", "<lower")
        .replace("< upper", "<upper")
    )
    # `<keyword>` + space + `[` -> remove the space (Stan idiom:
    # `vector[N]`, not `vector [N]`).
    for kw in (
        "array",
        "vector",
        "simplex",
        "cov_matrix",
        "cholesky_factor_corr",
        "matrix",
        "row_vector",
    ):
        out = out.replace(f"{kw} [", f"{kw}[")
    # `int <` -> `int<`; `real <` -> `real<`.
    out = out.replace("int <", "int<").replace("real <", "real<")
    # Spacing around `:` in `1 : 20` -> `1:20` (Stan idiom for
    # iteration ranges).
    out = _tighten_colon_ranges(out)
    # Spacing in `target +=` (the pretty printer omits the space
    # after `+=`).
    out = out.replace("target +=", "target += ")
    # Ensure space after comma.
    out = _ensure_space_after_comma(out)
    return out


def _tighten_colon_ranges(text: str) -> str:
    """Replace ` : ` with `:` only when both sides are bare numbers /
    identifiers (the Stan iteration range form). Leaves declarations
    like `complex : x` untouched if any ever arise (Stan has no such
    form today, but the guard keeps the formatter safe)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if i + 2 < n and text[i] == " " and text[i + 1] == ":" and text[i + 2] == " ":
            # Look at neighbouring tokens: previous non-space char and
            # next non-space char both alphanumeric / underscore.
            left = out[-1] if out else ""
            right = text[i + 3] if i + 3 < n else ""
            if (left.isalnum() or left == "_") and (right.isalnum() or right == "_"):
                out.append(":")
                i += 3
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _ensure_space_after_comma(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        out.append(c)
        if (
            c == ","
            and i + 1 < n
            and text[i + 1] not in {" ", "\n", "\t", "}", ")", "]"}
        ):
            out.append(" ")
        i += 1
    return "".join(out)


def _split_and_indent(text: str) -> list[str]:
    """Split the tightened source on `{`, `}`, and `;` boundaries and
    re-indent with two-space steps.

    Strings / comments are not present in the renderer's output, so
    naive tokenisation suffices.
    """
    # Insert newlines around block boundaries.
    spaced: list[str] = []
    for ch in text:
        if ch == "{":
            spaced.append(" {\n")
        elif ch == "}":
            spaced.append("\n}\n")
        elif ch == ";":
            spaced.append(";\n")
        elif ch == "\n":
            spaced.append(" ")
        else:
            spaced.append(ch)
    joined = "".join(spaced)
    # Normalise multiple spaces inside lines.
    raw_lines = joined.split("\n")
    # Re-indent.
    out: list[str] = []
    depth = 0
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        # Collapse internal runs of spaces.
        compact = _collapse_spaces(stripped)
        # Closing brace decrements depth before printing.
        if compact.startswith("}"):
            depth = max(0, depth - 1)
            out.append("  " * depth + compact)
            # Keep the leading `}` then continue with whatever
            # follows on its line.
        else:
            out.append("  " * depth + compact)
        # Increment depth after an opening brace.
        if compact.endswith("{"):
            depth += 1
    return out


def _collapse_spaces(text: str) -> str:
    """Collapse runs of spaces into single spaces, preserving content."""
    out: list[str] = []
    prev_space = False
    for ch in text:
        if ch == " ":
            if not prev_space:
                out.append(ch)
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out)


# ---------------------------------------------------------------------------
# Runtime-helper graft: Stan user-defined `<family>_lpdf` / `_rng`.
#
# Stan ships `normal`, `beta`, `gamma`, ... as built-in densities but
# lacks `kumaraswamy`. The transpile-time graft parses the hand-written
# helper at
# [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
# once at module-load through panproto's Stan tree-sitter grammar;
# per-render, it copies the parsed `functions { ... }` block into the
# per-render schema (with fresh vertex ids) and attaches it as a
# `child_of` of the program above the data block. Subsequent
# `kumaraswamy_lpdf(y | a, b)` increments then resolve against the
# grafted helper by Stan's `<family>_lpdf` naming convention.
# ---------------------------------------------------------------------------


_RUNTIME_STAN_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_stan_functions.stan"
)


#: Families whose Stan emit relies on the
#: [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
#: helper subtree. Stan ships `normal`, `beta`, `gamma`, etc. as
#: built-in densities but lacks `kumaraswamy`; the renderer grafts the
#: helper when the IR samples or observes from any of them.
_STAN_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset(
    {
        "Kumaraswamy",
        "ContinuousBernoulli",
        "MatrixNormal",
        "LogitNormal",
        "LogGamma",
        "LogitBeta",
    }
)


def _load_runtime_stan_schema() -> tuple[panproto.Schema, str, tuple[str, ...]]:
    """Parse
    [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
    through panproto's Stan tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `program` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `program` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "stan",
        _RUNTIME_STAN_PATH.read_bytes(),
        str(_RUNTIME_STAN_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "program"),
        None,
    )
    if src_id is None:
        raise RuntimeError(f"`program` not found in parse of {_RUNTIME_STAN_PATH}")
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


_RUNTIME_STAN_SCHEMA, _RUNTIME_STAN_PROGRAM_ID, _RUNTIME_STAN_TOP_LEVEL = (
    _load_runtime_stan_schema()
)


def _stan_subtree_vertex_ids(
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


_RUNTIME_STAN_SUBTREE = _stan_subtree_vertex_ids(
    _RUNTIME_STAN_SCHEMA, _RUNTIME_STAN_TOP_LEVEL
)


def _iter_ir_nodes(body: tuple[IRNode, ...]) -> list[IRNode]:
    """Flatten `body`, descending into every marginalize scope."""
    out: list[IRNode] = []
    for node in body:
        out.append(node)
        if isinstance(node, IRMarginalize):
            out.extend(_iter_ir_nodes(node.scope))
    return out


def _ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any [`IRSample`][quivers.transpile.ir.IRSample],
    [`IRObserve`][quivers.transpile.ir.IRObserve], or
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] in `body`
    (including nested marginalize scopes) draws from `family`.

    The marginalize check covers the continuous-latent emit path:
    a `marginalize z <- ContinuousBernoulli(p)` block compiles to
    a Stan parameter plus a `continuous_bernoulli_lpdf(z | p)`
    increment, both of which depend on the runtime helper
    `continuous_bernoulli_lpdf`.
    """
    for node in body:
        if isinstance(node, (IRSample, IRObserve)) and node.family == family:
            return True
        if isinstance(node, IRMarginalize):
            if node.family == family:
                return True
            if _ir_uses_family(node.scope, family):
                return True
    return False


def _graft_runtime_stan_helper(
    sb: panproto.SchemaBuilder,
    renderer: StanRenderer,
    program_vid: str,
) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_stan_functions.stan` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each top-level
    child as a `child_of` of `program_vid` in source order. The grafted
    `functions { ... }` block appears above the data block in the
    emit, satisfying Stan's `functions? data? ...` grammar production.
    """
    src_schema = _RUNTIME_STAN_SCHEMA
    subtree = _RUNTIME_STAN_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        renderer._fresh_n += 1
        new = f"rs_{renderer._fresh_n}"
        id_map[old] = new
        kind = next(v.kind for v in src_schema.vertices if v.id == old)
        sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_STAN_TOP_LEVEL:
        sb.edge(program_vid, id_map[child_old], "child_of")


__all__ = ["StanRenderer"]
