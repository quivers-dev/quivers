"""Program-block step AST nodes (). The
parser emits one AST class per surface step keyword. The internal
IR forms (``DrawStep``, ``PlateDrawStep``, ``VectorisedObserveStep``,
``GroupedLatentInitStep``, ``GroupedBodyObserveStep``,
``MarginalizeStep`` (compiler-only),
``GroupedObserveEntry``) remain unchanged: the compiler expands each
surface step into one of these specialized IR shapes at the entry to
``_compile_program``.

Surface AST (what the parser emits):

* `SampleStep` - ``sample v[: A] <- F(args) [options]``
* `ObserveStep` - ``observe v[: A] <- F(args) [options]``
* `MarginalizeStep` - ``marginalize c[: K] <- F(args) [options] : scope``
* `LetStep` - ``let x = expr``
* `ReturnStep` - ``return v`` or ``return (a, b, ...)``

Compiler-only IR (synthesized at compile time):

* `BindStep` - the unified Kleisli-bind shape used internally
  by the compiler when expanding surface steps.
* `DrawStep`, `PlateDrawStep`,
  `VectorisedObserveStep` - specializations.
* `GroupedLatentInitStep`, `GroupedBodyObserveStep`,
  `GroupedObserveEntry` - grouped-marginalize machinery.
"""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import AxisSpec, OptionEntry
from quivers.dsl.ast_nodes.let_expressions import LetExprNode
from quivers.dsl.ast_nodes.objects import ObjectExpr
from quivers.dsl.ast_nodes.qiec import QiecCallComputation


class DrawArg(dx.TaggedUnion, discriminator="kind"):
    """A single argument in a draw step. Variants cover all four
    surface shapes the parser can produce: bare identifier
    (variable reference), numeric literal, distribution-call
    expression (compositional measure algebra), and list of args.

    Sum-type representation rather than a Python union so the
    panproto translation can carry the AST through the grammar
    walker, the migration lens, and the pretty-printer uniformly.
    """


class DrawArgNamed(DrawArg):
    """One keyword argument in a distribution-family construction.

    The parameter name is structural: it wraps the argument value instead
    of being optional metadata on that value. Rewriters must therefore
    preserve or deliberately transform the complete named entry, and cannot
    accidentally erase a parameterization while rebuilding its value.
    """

    parameter: str
    value: DrawArg
    line: int = 0
    col: int = 0
    kind: Literal["named"] = "named"


class DrawArgName(DrawArg):
    """Bare identifier reference in a draw step. Carries only the
    identifier text; bracket-indexed forms (``theta[N]``) parse to
    the dedicated [`DrawArgIndex`][quivers.dsl.ast_nodes.DrawArgIndex]
    variant.
    """

    text: str
    line: int = 0
    col: int = 0
    kind: Literal["name"] = "name"


class DrawArgIndex(DrawArg):
    """Bracket-indexed reference in a draw step. `name` is the base
    tensor identifier; `indices` is the tuple of index-tensor
    identifiers referenced inside the brackets (typically length
    one, e.g. ``theta[N]``). Both the base name and every index
    identifier count as dependencies for plate-graph edge emission.
    """

    name: str
    indices: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["index"] = "index"


class DrawArgScalar(DrawArg):
    """Numeric literal in a draw step."""

    value: float
    line: int = 0
    col: int = 0
    kind: Literal["scalar"] = "scalar"


class DrawArgDist(DrawArg):
    """Nested `Family(...)` call inside the `args` of a draw step.

    Surfaces the compositional measure algebra (`Mixture`,
    `Restrict`, `Pushforward`, `PointMass`, ...) so distributions
    can be expressed as ordinary draw-arg expressions:

        observe y <- Mixture([0.3, 0.7], [PointMass(0), Poisson(rate)])

    The compiler recurses into `args` to build the inner measure
    before passing it to the outer family's builder.
    """

    family: str
    args: tuple[DrawArg, ...]
    line: int = 0
    col: int = 0
    kind: Literal["dist"] = "dist"


class DrawArgList(DrawArg):
    """List-of-draw-args, e.g. `[0.3, 0.7]` for mixture weights or
    `[PointMass(0), Poisson(rate)]` for mixture components. The
    compiler unpacks the list into a vector or list of inner
    distributions depending on the consuming family's argument
    schema.
    """

    items: tuple[DrawArg, ...]
    line: int = 0
    col: int = 0
    kind: Literal["list"] = "list"


def atom_to_draw_arg(value: str | float | int | DrawArg) -> DrawArg:
    """Lift a plain Python value into the `DrawArg` it denotes.

    Parameters
    ----------
    value : str | float | int | DrawArg
        A name, a number, or an argument that is already a `DrawArg`.

    Returns
    -------
    DrawArg
        A `DrawArgName` for a string, a `DrawArgScalar` for a number,
        and ``value`` itself for a `DrawArg`.

    Raises
    ------
    TypeError
        If ``value`` is none of the accepted kinds.
    """
    if isinstance(value, DrawArg):
        return value
    if isinstance(value, str):
        return DrawArgName(text=value)
    if isinstance(value, (int, float)):
        return DrawArgScalar(value=float(value))
    raise TypeError(
        f"atom_to_draw_arg: expected str | float | DrawArg, got {type(value).__name__}"
    )


class ProgramStep(dx.TaggedUnion, discriminator="kind"):
    """Sum of program-block step node kinds."""


# ---------------------------------------------------------------------------
# Surface program steps
# ---------------------------------------------------------------------------


class SampleStep(ProgramStep):
    """``sample vars[: index] <- morphism(args) [options]``.

    Surface forms::

        sample v <- F(args)                         # scalar draw
        sample v : A <- F(args)                     # A-indexed plate
        sample (a, b) <- F(args)                    # destructuring tuple

    Optional ``[options]`` block carries axis-role and other
    family-level config (``[over=[...], iid=[...]]``).
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    index: ObjectExpr | None = None
    axes: AxisSpec | None = None
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["sample_step"] = "sample_step"


class ObserveStep(ProgramStep):
    """``observe vars[: index] <- morphism(args) [options]``.

    Scored Kleisli bind; the bound coordinate is clamped at runtime
    by the ``observations`` dict, making the resulting kernel
    sub-probabilistic. ``vars`` shares the pattern shape with
    `SampleStep` (a bare name or a parenthesized tuple); the compiler
    validates arity against the morphism's codomain.
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    index: ObjectExpr | None = None
    axes: AxisSpec | None = None
    via: str | None = None
    via_axes: tuple[str, ...] | None = None
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["observe_step"] = "observe_step"


class MarginalizeStep(ProgramStep):
    """``marginalize var[: index] <- morphism(args) [options] : scope``.

    Introduces a coordinate, executes the scope's steps with that
    coordinate in trace context, and at the end of the scope pushes
    forward through the projection (logsumexp for discrete, fibrewise
    integration for continuous). When the option block carries
    ``over=...``, the reduction is grouped (grouped marginalize); the
    ``via`` clause on each inner observe names the per-observe
    index map.
    """

    var: str
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    index: ObjectExpr | None = None
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    reduction: str | None = None
    options: tuple[OptionEntry, ...] = ()
    scope: tuple[ProgramStep, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["marginalize_step"] = "marginalize_step"


class LetStep(ProgramStep):
    """``let name = value`` deterministic step.

    The value is a let-arithmetic expression evaluated against the
    surrounding program env.
    """

    name: str
    value: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["let_step"] = "let_step"


class CallStep(ProgramStep):
    """``let name <- computation(args)`` binding step.

    The bound name takes the result of a named computation declared in
    the module; the computation's effects join the program's row.
    """

    name: str
    call: QiecCallComputation
    line: int = 0
    col: int = 0
    kind: Literal["call_step"] = "call_step"


class ScoreStep(ProgramStep):
    """``score name = value`` log-density factor step.

    Like `LetStep` in that ``name`` is bound to the evaluated
    expression's value for downstream reference, but distinct in
    that the value is *additionally* added to the program's
    ``log_joint``. This is the deduction-style analog of
    `ObserveStep`: the chart's `goal_weight` is a
    log-density factor on the corpus, and ``score log_Z = chart.goal_weight()``
    expresses that contribution as a first-class step.
    """

    name: str
    value: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["score_step"] = "score_step"


class ReturnStep(ProgramStep):
    """``return v`` or ``return (a, b)`` or ``return (a: x, b: y)`` step.

    Always the terminal step of a program body. ``labels`` is
    non-``None`` for the labelled-tuple form ``return (a: x, b: y)``.
    """

    vars: tuple[str, ...]
    labels: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["return_step"] = "return_step"


# ---------------------------------------------------------------------------
# Compiler-only IR (synthesized from surface steps at compile time)
# ---------------------------------------------------------------------------


class GroupedObserveEntry(dx.Model):
    """One entry in a `GroupedMarginalizeStep`'s
    ``body_observes`` list.

    Pairs an env slot (where the captured observe writes its
    ``(N_m, K)`` log-likelihood) with the index map that carries
    those rows into the shared grouping plate. An entry with no
    index map is a nested grouped block: the slot holds the inner
    block's per-position aggregates over ``inner_groups``, which the
    enclosing block projects onto its own grouping plate.
    """

    ll_slot: str
    index_map_var: str | None = None
    index_map_axes: tuple[str, ...] | None = None
    inner_groups: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0


class GroupedMarginalizeStep(ProgramStep):
    """Internal compiler IR: a marginalisation pushforward.

    The compiler lowers a surface `MarginalizeStep` into this
    shape after expanding the scope. ``class_size`` is the resolved
    cardinality of the latent index; ``probs_var`` names the env
    slot holding the family's probability or logits tensor, which
    ``probs_indices`` gather rows of, left to right, when the prior is
    written ``probs[latent]`` against an enclosing latent;
    ``parameterization`` records whether that tensor is probabilities or
    logits; ``over_obj``
    / ``over_objs`` carry the grouping object (single or product);
    ``body_ll_var`` names the env slot that the grouped observe
    pushed its (N_m, K) log-likelihood into; ``body_observes`` lists
    the (ll_slot, index map) entries that the runtime callable
    consumes for grouped pushforward.
    """

    var_name: str
    class_size: int
    probs_var: str | None = None
    probs_indices: tuple[str, ...] = ()
    parameterization: Literal["probs", "logits"] = "probs"
    over_obj: str | None = None
    over_objs: tuple[str, ...] | None = None
    body_ll_var: str = ""
    body_observes: tuple[GroupedObserveEntry, ...] | None = None
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["grouped_marginalize_step"] = "grouped_marginalize_step"


class BindStep(ProgramStep):
    """Internal compiler IR: a unified Kleisli bind.

    Synthesized from surface `SampleStep` / `ObserveStep`
    / `MarginalizeStep` during template instantiation. The
    surface AST never carries this shape directly.
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    index: ObjectExpr | None = None
    mode: Literal["sample", "score", "marginal"] = "sample"
    scope: tuple[ProgramStep, ...] | None = None
    axes: AxisSpec | None = None
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    via: str | None = None
    via_axes: tuple[str, ...] | None = None
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["bind_step"] = "bind_step"


class DrawStep(ProgramStep):
    """Internal compiler IR: a scalar sample or score step.

    Synthesised from a `SampleStep`/`ObserveStep` with
    no index annotation; ``is_observed`` distinguishes sample
    (``False``) from score (``True``).
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    is_observed: bool = False
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0
    kind: Literal["draw_step"] = "draw_step"


class PlateDrawStep(ProgramStep):
    """Internal compiler IR: an A-indexed sample step.

    Synthesised from a `SampleStep` with an index annotation;
    realises a Kern-morphism ``A -> Kern(K)`` as a tensor of shape
    ``(|A|, *K.shape)``.
    """

    name: str
    index: ObjectExpr
    codomain: ObjectExpr
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0
    kind: Literal["plate_draw_step"] = "plate_draw_step"


class VectorisedObserveStep(ProgramStep):
    """Internal compiler IR: an A-indexed score step.

    Synthesised from an `ObserveStep` with an index
    annotation; denotes the sub-probabilistic kernel
    ``Phi -> Kern_{<=1}(Phi)`` with score
    ``prod_n p_F(r_obs(n); theta(n, phi))``.
    """

    index_var: str
    index_set: ObjectExpr
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    response_var: str = ""
    index_map_var: str | None = None
    index_map_axes: tuple[str, ...] | None = None
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0
    kind: Literal["vectorized_observe_step"] = "vectorized_observe_step"


class GroupedLatentInitStep(ProgramStep):
    """Internal compiler IR: initialise the latent's env slot to
    ``torch.arange(class_size)`` at the start of a grouped
    marginalize block's body.
    """

    latent_name: str
    class_size: int
    line: int = 0
    col: int = 0
    kind: Literal["grouped_latent_init_step"] = "grouped_latent_init_step"


class GroupedBodyObserveStep(ProgramStep):
    """Internal compiler IR: a captured observe inside a grouped
    marginalize block.
    """

    response_var: str
    morphism: str
    args: tuple[DrawArg, ...] | None = None
    index_set: ObjectExpr | None = None
    index_var: str = ""
    latent_name: str = ""
    class_size: int = 0
    index_map_var: str | None = None
    index_map_axes: tuple[str, ...] | None = None
    ll_slot: str = ""
    line: int = 0
    col: int = 0
    kind: Literal["grouped_body_observe_step"] = "grouped_body_observe_step"


__all__ = [
    "BindStep",
    "CallStep",
    "DrawArg",
    "DrawArgDist",
    "DrawArgIndex",
    "DrawArgList",
    "DrawArgName",
    "DrawArgNamed",
    "DrawArgScalar",
    "DrawStep",
    "GroupedBodyObserveStep",
    "GroupedLatentInitStep",
    "GroupedMarginalizeStep",
    "GroupedObserveEntry",
    "LetStep",
    "MarginalizeStep",
    "ObserveStep",
    "PlateDrawStep",
    "ProgramStep",
    "ReturnStep",
    "SampleStep",
    "ScoreStep",
    "VectorisedObserveStep",
    "atom_to_draw_arg",
]
