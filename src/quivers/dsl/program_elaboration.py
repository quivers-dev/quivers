"""Elaboration of ``program`` declarations into QIEC computations.

A program is domain-specific notation for a named computation: its data,
observations, and index maps are the computation's typed parameters, its
``sample`` steps perform ``Random.sample`` on the module's canonical
``random`` instance, its ``observe`` and ``score`` steps perform
``Score.add`` on the canonical ``score`` instance, its ``let`` steps bind
pure expressions, its ``marginalize`` blocks call a helper computation
that enumerates a finite latent under the prelude's enumeration handler,
and its ``return`` builds the result tuple. The elaboration is a mixin
over the QIEC elaborator, since it shares the value lowering, the
registry, and the diagnostics of every other declaration.

Plates are typed data. A step's ``over`` and ``iid_over`` axes, its
``: Axis`` annotation, and a morphism's ``Real N`` codomain fix a
``PlateShape`` on the step's distribution, whose sample type is the
tensor over those axes; a ``via`` index map re-indexes a grouped latent's
arguments to the observation rows and segments the observation's weights
back to the groups.
"""

from __future__ import annotations

import didactic.api as dx

import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from quivers.dsl.ast_nodes import (
    CallStep,
    ContinuousConstructor,
    DiscreteConstructor,
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    ExprIdent,
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprTuple,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    MarginalizeStep,
    MorphismDecl,
    ObjectDecl,
    ObjectExpr,
    ObjectProduct,
    ObserveStep,
    OptionCall,
    OptionEntry,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ObjectParam,
    MorphismParam,
    ScalarParam,
    ScoreStep,
    TypeEnumSet,
    TypeFromExpr,
    TypeName,
)
from quivers.qiec.canonical import (
    LOG_WEIGHT,
    sampleable_type,
    sampled_element,
    site_type,
    tensor_shape,
    tensor_type,
)
from quivers.qiec.checking import (
    _plated_sample_type,
    CheckContext,
    ComputationSignature,
    KernelError,
    infer_computation,
)
from quivers.qiec.effects import (
    ComputationType,
    EffectRef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
    RowEntry,
    instantiate_effect,
)
from quivers.qiec.families import FAMILIES, DistributionFamily, FamilyParameter
from quivers.qiec.identifiers import (
    ComputationId,
    HandlerId,
    SiteProvenance,
    SourceOrigin,
)
from quivers.qiec.kinds import NAT, IndexBinder, TypeBinder
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.programs import (
    ParameterRole,
    ProgramEntry,
    ProgramParameter,
    ProgramSite,
)
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_operation,
    instantiate_telescope,
    substitute_computation,
    substitute_index,
    substitute_plate,
    substitute_row,
    substitute_type,
    substitute_value,
)
from quivers.qiec.terms import (
    AffineMap,
    Bind,
    Call,
    Comprehension,
    Computation,
    DistributionValue,
    Gather,
    Handle,
    If,
    KernelMatrix,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    PlateAxis,
    PlateShape,
    Reduction,
    Return,
    SegmentSum,
    SiteValue,
    TableMap,
    TensorValue,
    TupleValue,
    Value,
    Var,
    WeightSum,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    UNIT,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeExpr,
    TypeVariable,
    product_type,
    render_static,
)
from quivers.dsl.composite_lets import (
    expand_composite_lets,
    ScanCall,
    is_learned_scan_init,
    positions_let_call,
    scan_let_call,
)
from quivers.dsl.family_schemas import (
    PARAMETER_ALIASES,
    family_parameterizations,
    render_parameterizations,
)
from quivers.dsl.let_expr_traversal import (
    free_let_names,
    let_expr_children,
    map_let_expr,
    walk_let_expr,
)
from quivers.dsl.deduction_elaboration import PARAMS_INSTANCE
from quivers.continuous.deterministic import learned_function_draw_message
from quivers.continuous.param_source import attention_draw_message
from quivers.dsl.draw_args import kernel_argument_refusal
from quivers.dsl.program_templates import (
    TemplateError,
    instantiate_program,
    template_bindings,
)
from quivers.dsl.pure_builtins import (
    _REDUCTIONS,
    _ROWWISE,
    HOST_ONLY_BUILTINS,
    PURE_BUILTINS,
    SHAPE_BUILTINS,
)
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.markov_chains import chain_cells, markov_cell
from quivers.dsl.sequence_functions import (
    _SequenceFunctionElaboration,
    sequence_source_kind,
)
from quivers.dsl.step_resolution import (
    ResolvedDist,
    StepResolutionError,
    _step_value_names,
    build_let_table,
    morphism_table,
    resolve_step_dist,
)

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


#: The name of the handler enumerating a marginalized latent.
ENUMERATE_HANDLER = "enumerate_marginal"
#: The name of the handler enumerating a latent nested in a grouped
#: marginalization, which answers one weight per group position.
ENUMERATE_GROUPED_HANDLER = "enumerate_grouped_marginal"
#: The reductions a marginalization aggregates its latent's values by.
MARGINAL_REDUCTIONS: tuple[str, ...] = ("logsumexp", "sum", "mean")
#: The enumeration handler by whether it answers per group position and
#: by reduction: the plain names reduce by ``logsumexp``, the others carry
#: the reduction as a suffix.
MARGINAL_HANDLERS: dict[tuple[bool, str], str] = {
    (grouped, reduction): (ENUMERATE_GROUPED_HANDLER if grouped else ENUMERATE_HANDLER)
    + ("" if reduction == "logsumexp" else f"_{reduction}")
    for grouped in (False, True)
    for reduction in MARGINAL_REDUCTIONS
}
#: The name of the handler enumerating a marginalized chain's next state,
#: which answers the weighted shots one per support value.
ENUMERATE_TABLE_HANDLER = f"{ENUMERATE_HANDLER}_table"
#: The name of the handler collecting a marginalization scope's weights.
COLLECT_HANDLER = "collect_marginal"
#: The canonical instance names a program's requests address.
RANDOM_INSTANCE = "random"
SCORE_INSTANCE = "score"

#: Operator-algebra spellings that name a registry family under another
#: name: a truncation is a restriction, a pushforward a transformed
#: distribution.
OPERATOR_ALIASES: Mapping[str, str] = MappingProxyType(
    {"Truncate": "Restrict", "Pushforward": "Transformed"}
)

#: Families whose construction is a measure a step normalizes at its
#: boundary rather than a probability measure as built.
MEASURE_FAMILIES: frozenset[str] = frozenset(
    {"Restrict", "Mixture", "Transformed", "Independent"}
)

#: The bijectors of the operator algebra, by their source names, as the
#: transform chains the ``Transformed`` family takes.
BIJECTOR_TRANSFORMS: Mapping[str, str] = MappingProxyType(
    {
        "Identity": "",
        "Exp": "exp",
        "Log": "log",
        "Sigmoid": "sigmoid",
        "Logit": "logit",
        "Softplus": "softplus",
    }
)

#: The diagnostic code of a program construct whose elaboration is not yet
#: defined: a parsing chart, a chart method, or a morphism parameterized by
#: a network. Such a program is reported rather than approximated, and a
#: caller may lower the module without its programs meanwhile.
GAP_CODE = "qiec-program-gap"

#: Let-expression builtins that build or read a deduction chart.
_CHART_BUILTINS = frozenset({"parse", "chart", "chart_fold", "op_apply"})

#: How a kernel morphism's parameter map is read into one family
#: parameter: as it is, exponentiated and clamped at the scale floor,
#: through a softplus lifted off zero, through a softplus shifted by a
#: tenth, or through a sigmoid.
HeadTransform = Literal[
    "identity", "exp_floor", "softplus", "softplus_shifted", "sigmoid"
]

#: The lift a ``softplus`` head adds, so the parameter stays positive.
SOFTPLUS_LIFT = 1e-7

#: The shift a ``softplus_shifted`` head adds.
SOFTPLUS_SHIFT = 0.1

#: Families whose bare ``~ Family`` kernel morphism reads its arguments off
#: a parameter map, with each head's transform, in the torch runtime's
#: parameterization of the same kernel.
_CONDITIONAL_HEADS: dict[str, tuple[tuple[str, HeadTransform], ...]] = {
    "Normal": (("loc", "identity"), ("scale", "exp_floor")),
    "LogitNormal": (("loc", "identity"), ("scale", "exp_floor")),
    "Cauchy": (("loc", "identity"), ("scale", "softplus")),
    "Laplace": (("loc", "identity"), ("scale", "softplus")),
    "Gumbel": (("loc", "identity"), ("scale", "softplus")),
    "LogNormal": (("loc", "identity"), ("scale", "softplus")),
    "StudentT": (
        ("df", "softplus_shifted"),
        ("loc", "identity"),
        ("scale", "softplus"),
    ),
    "Exponential": (("rate", "softplus"),),
    "Gamma": (("concentration", "softplus_shifted"), ("rate", "softplus")),
    "LogGamma": (("concentration", "softplus"), ("rate", "softplus")),
    "LogitBeta": (("concentration1", "softplus"), ("concentration0", "softplus")),
    "Chi2": (("df", "softplus_shifted"),),
    "HalfCauchy": (("scale", "softplus"),),
    "HalfNormal": (("scale", "softplus"),),
    "InverseGamma": (("concentration", "softplus_shifted"), ("rate", "softplus")),
    "Weibull": (("scale", "softplus"), ("concentration", "softplus")),
    "Pareto": (("scale", "softplus"), ("alpha", "softplus")),
    "ContinuousBernoulli": (("logits", "identity"),),
    "Poisson": (("rate", "softplus"),),
    "Geometric": (("probs", "sigmoid"),),
    "NegativeBinomial": (("total_count", "softplus_shifted"), ("probs", "sigmoid")),
    "VonMises": (("loc", "identity"), ("concentration", "softplus")),
    "Beta": (
        ("concentration1", "softplus_shifted"),
        ("concentration0", "softplus_shifted"),
    ),
    "Dirichlet": (("concentration", "softplus_shifted"),),
    "Bernoulli": (("logits", "identity"),),
    "Categorical": (("logits", "identity"),),
}

#: The number of parameter heads of each family whose kernel a network
#: supplies, which sizes a latent-weight network's output layer.
LATENT_WEIGHT_HEADS: dict[str, int] = {
    family: len(heads) for family, heads in _CONDITIONAL_HEADS.items()
}

#: Families whose kernel head is one row of logits over the codomain's
#: elements rather than one coordinate per real dimension.
_LOGIT_HEAD_FAMILIES: frozenset[str] = frozenset({"Bernoulli", "Categorical"})

#: Families whose sample is an index into their own alphabet, with the
#: argument naming the alphabet.
_CLASS_INDEX_FAMILIES: dict[str, str] = {"Categorical": "probs"}

_MATRIX_CONSTRAINTS: frozenset[str] = frozenset({"lower_cholesky", "positive_definite"})
"""Parameter constraints a matrix meets only as a whole: a broadcast scalar
is no Cholesky factor or covariance, so a free name filling such a
parameter is a matrix even beside a plate's event axes."""


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """What the elaborator reads off one ``object`` declaration.

    Parameters
    ----------
    extent
        The object's size as an axis: its cardinality when finite, its
        coordinate count when continuous. An object parameter of a
        program template has an index variable for its size, bound by
        the template's telescope.
    real_width
        The width of a ``Real N`` object, whose values are real vectors.
    finite
        Whether the object is a finite set.
    """

    extent: int | IndexVariable | None = None
    real_width: int | IndexVariable | None = None
    finite: bool = False


@dataclass(slots=True)
class _Scope:
    """Bindings of one program body or marginalization scope.

    Parameters
    ----------
    locals
        Every bound name and its local, in binding order.
    steps
        The binds accumulated so far, each a binder and the computation
        it binds.
    referenced
        Names of enclosing bindings the scope has read, which a helper
        computation takes as parameters.
    group
        The grouping axis of a grouped marginalization scope, else
        ``None``.
    factors
        The axes a product grouping axis flattens, in order; the group
        itself for a single axis.
    weight_instance
        The local ``Weight`` instance a marginalization scope adds its
        weights to, else ``None`` for the program body.
    weight_type
        The type that instance accumulates.
    weights_added
        Whether a step of the scope has added to its ``Weight`` instance.
    outer
        The enclosing scope, else ``None``.
    """

    locals: dict[str, Local] = field(default_factory=dict)
    steps: list[tuple[Local, Computation]] = field(default_factory=list)
    referenced: set[str] = field(default_factory=set)
    group: PlateAxis | None = None
    factors: tuple[PlateAxis, ...] = ()
    weight_instance: NamedEffectInstance | None = None
    weight_type: TypeExpr | None = None
    weights_added: bool = False
    outer: _Scope | None = None

    def lookup(self, name: str) -> Local | None:
        """Find a binding in this scope or an enclosing one.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Local | None
            The innermost binding, or ``None``.
        """
        scope: _Scope | None = self
        while scope is not None:
            local = scope.locals.get(name)
            if local is not None:
                if scope is not self:
                    self.referenced.add(name)
                return local
            scope = scope.outer
        return None

    def context(self) -> CheckContext:
        """The kernel context of every binding visible here.

        Returns
        -------
        CheckContext
            Outer bindings first, then this scope's.
        """
        chain: list[_Scope] = []
        scope: _Scope | None = self
        while scope is not None:
            chain.append(scope)
            scope = scope.outer
        locals_: list[Local] = []
        for item in reversed(chain):
            locals_.extend(item.locals.values())
        return CheckContext(tuple(locals_))


def _scanned_names(steps: tuple[ProgramStep, ...]) -> frozenset[str]:
    """The names a program's let steps scan over.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps, searched through marginalization scopes.

    Returns
    -------
    frozenset[str]
        The sequence names of every ``scan(step, xs)`` let.
    """
    found: set[str] = set()
    for step in steps:
        if isinstance(step, MarginalizeStep):
            found |= _scanned_names(step.scope)
        elif isinstance(step, LetStep):
            scan = scan_let_call(step.value)
            if scan is not None:
                found.add(scan.sequence)
    return frozenset(found)


def _bound_names(step: ProgramStep) -> frozenset[str]:
    """The names a step binds, with those its scope binds.

    Parameters
    ----------
    step : ProgramStep
        The step.

    Returns
    -------
    frozenset[str]
        The variables of a draw or observation, the name of a let, call,
        or score, and a block's latent with every name its scope binds.
    """
    if isinstance(step, SampleStep | ObserveStep):
        return frozenset(step.vars)
    if isinstance(step, LetStep | CallStep | ScoreStep):
        return frozenset({step.name})
    if isinstance(step, MarginalizeStep):
        found = {step.var}
        for inner in step.scope:
            found |= _bound_names(inner)
        return frozenset(found)
    return frozenset()


def _own_site_labels(steps: tuple[ProgramStep, ...]) -> frozenset[str]:
    """The labels of the sites a program body draws or observes itself.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps, searched through marginalization scopes but not
        into the programs they call or scan with.

    Returns
    -------
    frozenset[str]
        The label of every ``sample`` and ``observe`` step and of every
        marginalized latent.
    """
    found: set[str] = set()
    for step in steps:
        if isinstance(step, SampleStep | ObserveStep):
            found.add(",".join(step.vars))
        elif isinstance(step, MarginalizeStep):
            found.add(step.var)
            found |= _own_site_labels(step.scope)
    return frozenset(found)


def reached_site_labels(
    name: str, declarations: Mapping[str, ProgramDecl]
) -> frozenset[str]:
    """The labels of every site a program reaches when it runs.

    A program reaches its own sites, the sites of the programs it calls
    with ``let y <- p(...)``, which keep their labels, and the sites of
    the step programs it scans with, which it reaches once per position
    under the same labels.

    Parameters
    ----------
    name : str
        The program's name.
    declarations : Mapping[str, ProgramDecl]
        The module's program declarations, step programs included.

    Returns
    -------
    frozenset[str]
        The labels; empty for a name that is not a program.
    """
    found: set[str] = set()
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        declaration = declarations.get(current)
        if declaration is None:
            continue
        found |= _own_site_labels(declaration.draws)
        pending.extend(_callees(declaration.draws))
    return frozenset(found)


def _callees(steps: tuple[ProgramStep, ...]) -> tuple[str, ...]:
    """The programs a body calls in place or scans with.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps, searched through marginalization scopes.

    Returns
    -------
    tuple[str, ...]
        The callee of every ``let y <- p(...)`` and the step program
        of every ``let h = scan(step, xs)``.
    """
    found: list[str] = []
    for step in steps:
        if isinstance(step, CallStep):
            found.append(step.call.callee)
        elif isinstance(step, MarginalizeStep):
            found.extend(_callees(step.scope))
        elif isinstance(step, LetStep):
            scan = scan_let_call(step.value)
            if scan is not None:
                found.append(scan.step)
    return tuple(found)


def _zero_of(type_: TypeExpr) -> Value:
    """The zero of a state type.

    Parameters
    ----------
    type_ : TypeExpr
        ``Real``, ``Int``, or a tensor of either with literal extents.

    Returns
    -------
    Value
        The literal zero, or a tensor of zeros.

    Raises
    ------
    ValueError
        If the type has no zero.
    """
    if type_ == REAL:
        return LiteralValue(0.0, REAL)
    if type_ == INT:
        return LiteralValue(0, INT)
    shape = tensor_shape(type_)
    if shape is None or not shape[1] or not isinstance(shape[1][0], IndexLiteral):
        raise ValueError(f"{render_static(type_)} has no zero")
    inner: TypeExpr = (
        shape[0] if len(shape[1]) == 1 else tensor_type(shape[0], shape[1][1:])
    )
    return TensorValue(
        tuple(_zero_of(inner) for _ in range(int(shape[1][0].value))), type_
    )


def _map_transform(transform: HeadTransform) -> Literal["identity", "exp_floor"]:
    """The transform a parameter map carries for a head.

    Parameters
    ----------
    transform : HeadTransform
        The head's transform.

    Returns
    -------
    Literal["identity", "exp_floor"]
        ``exp_floor`` when the map exponentiates the head itself, else
        ``identity``; a softplus or sigmoid applies after the map.
    """
    return "exp_floor" if transform == "exp_floor" else "identity"


def _rows_axis(
    plate: PlateShape, step: SampleStep | ObserveStep | MarginalizeStep
) -> PlateAxis | None:
    """The plate axis a kernel's parameters vary along, if any.

    Parameters
    ----------
    plate : PlateShape
        The distribution's plate.
    step : SampleStep | ObserveStep | MarginalizeStep
        The step.

    Returns
    -------
    PlateAxis | None
        The leading batch axis when the step's ``: Axis`` annotation
        names it, else ``None``.
    """
    if (
        plate.batch
        and isinstance(step.index, TypeName)
        and plate.batch[0].name == step.index.name
    ):
        return plate.batch[0]
    return None


def _per_row(head: Value, width: int, row: Local, rows_axis: PlateAxis) -> Value:
    """A parameter head evaluated at every row of a plate.

    The plate's trailing axis is the codomain's width, so a per-row
    head keeps that axis even at width one.

    Parameters
    ----------
    head : Value
        The head at one row, reading the row through ``row``.
    width : int
        The head's width.
    row : Local
        The ``Int`` local the head reads the row through.
    rows_axis : PlateAxis
        The plate axis of the rows.

    Returns
    -------
    Value
        A ``Tensor[Real]([rows, width])`` comprehension.
    """
    if width == 1:
        head = TensorValue((head,), tensor_type(REAL, (_extent(1),)))
    return Comprehension(
        row,
        rows_axis.size,
        head,
        tensor_type(REAL, (rows_axis.size, _extent(width))),
    )


def _names_support(record: DistributionFamily, axis: PlateAxis) -> bool:
    """Whether a marginalization's annotated axis names the family's support.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    axis : PlateAxis
        The annotated axis.

    Returns
    -------
    bool
        ``True`` for a class-index family, whose alphabet the axis
        names; a family with a fixed set of atoms has nothing for the
        axis to name, so the axis replicates the latent.
    """
    del axis
    return record.name in _CLASS_INDEX_FAMILIES


def _enumerable(record: DistributionFamily) -> bool:
    """Whether a marginalization over a family sums a finite support.

    Parameters
    ----------
    record : DistributionFamily
        The family.

    Returns
    -------
    bool
        ``True`` for a family with a finite support or a class-index
        family. A continuous family, a relaxation of a discrete one
        included, has no atoms to sum: its marginal is an integral.
    """
    return record.finite_support is not None or record.name in _CLASS_INDEX_FAMILIES


def _bare_init_family(morphism: MorphismDecl) -> str | None:
    """The family a morphism's bare ``~ Family`` initializer names.

    Parameters
    ----------
    morphism : MorphismDecl
        The morphism.

    Returns
    -------
    str | None
        The family's name when the initializer is a family with no
        arguments, written as a call or as a bare identifier; ``None``
        otherwise.
    """
    init = morphism.init_family
    if init is not None:
        return init.family if not init.args else None
    if isinstance(morphism.init_expr, ExprIdent):
        return morphism.init_expr.name
    return None


def _option_entry(options: tuple[OptionEntry, ...], key: str) -> OptionValue | None:
    """Read an option's value off an option block.

    Parameters
    ----------
    options : tuple[OptionEntry, ...]
        The block.
    key : str
        The option's key.

    Returns
    -------
    OptionValue | None
        The value of the first entry under the key, or ``None`` when
        absent.
    """
    for entry in options:
        if entry.key == key:
            return entry.value
    return None


def _source_kind(morphism: MorphismDecl) -> str:
    """The network kind a morphism's ``param_source`` option names.

    Parameters
    ----------
    morphism : MorphismDecl
        The morphism.

    Returns
    -------
    str
        The kind (``linear``, ``mlp``, ``identity``, ...), ``linear``
        when the option is absent.
    """
    source = _option_entry(morphism.options, "param_source")
    if isinstance(source, OptionCall):
        return source.func
    if isinstance(source, OptionName):
        return source.value
    return "linear"


def _option_value(options: tuple[OptionEntry, ...], key: str) -> str | float | None:
    """Read a scalar option off an option block.

    Parameters
    ----------
    options : tuple[OptionEntry, ...]
        The block.
    key : str
        The option's key.

    Returns
    -------
    str | float | None
        A name, a string, or a number, or ``None`` when absent or not
        scalar.
    """
    for entry in options:
        if entry.key != key:
            continue
        value = entry.value
        if isinstance(value, OptionName | OptionString):
            return value.value
        if isinstance(value, OptionNumber):
            return value.value
    return None


def _option_axes(options: tuple[OptionEntry, ...], key: str) -> tuple[str, ...]:
    """Read an axis list off an option block.

    Parameters
    ----------
    options : tuple[OptionEntry, ...]
        The block.
    key : str
        The option's key, ``over`` or ``iid_over``.

    Returns
    -------
    tuple[str, ...]
        The axis names; one name for a bare identifier.
    """
    for entry in options:
        if entry.key != key:
            continue
        value = entry.value
        if isinstance(value, OptionName):
            return (value.value,)
        if isinstance(value, OptionList):
            return tuple(
                item.value
                for item in value.items
                if isinstance(item, OptionName | OptionString)
            )
    return ()


def _factors(expr: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """Flatten a product object expression into its factors.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.

    Returns
    -------
    tuple[ObjectExpr, ...]
        The factors in source order; a non-product is its own factor.
    """
    if isinstance(expr, ObjectProduct):
        return tuple(
            item for component in expr.components for item in _factors(component)
        )
    return (expr,)


def _free_let_names(
    expr: LetExprNode, bound: frozenset[str] = frozenset()
) -> list[str]:
    """The variables a let expression reads, in first-use order.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    bound : frozenset[str]
        Names bound inside the expression.

    Returns
    -------
    list[str]
        Free variable names, without repeats.
    """
    return list(free_let_names(expr, bound=bound))


def _is_gap(kinds: Sequence[str]) -> bool:
    """Whether a resolution failure names a construct outside the calculus.

    Parameters
    ----------
    kinds : Sequence[str]
        The failure's structured kinds.

    Returns
    -------
    bool
        ``True`` for a network-parameterized morphism the elaboration
        has no form for, or a ``scan`` recurrence outside the shapes
        it elaborates.
    """
    return any(kind.startswith(("param-source:", "scan:")) for kind in kinds)


def _unknown_call(expr: LetExprNode, macros: Mapping[str, object]) -> str | None:
    """Name the first call to something the pure layer does not define.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    macros : Mapping[str, object]
        The lambda-bound names in scope.

    Returns
    -------
    str | None
        The called name, or ``None`` when every call is a builtin, a
        reduction, a last-axis operation, a family, or a macro.
    """
    for node in walk_let_expr(expr):
        if not isinstance(node, LetExprCall):
            continue
        if (
            node.func not in PURE_BUILTINS
            and node.func not in FAMILIES
            and node.func not in macros
            and node.func not in _CHART_BUILTINS
            and node.func not in ("site", "log_prob")
        ):
            return node.func
    return None


def _chart_construct(expr: LetExprNode) -> str | None:
    """Name a chart construct inside a let expression, if any.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    str | None
        A description of the first method call or chart builtin found,
        or ``None``.
    """
    for node in walk_let_expr(expr):
        if isinstance(node, LetExprMethodCall) and node.method != "goal_weight":
            return f"the method call `.{node.method}(...)`"
        if isinstance(node, LetExprCall) and node.func in _CHART_BUILTINS:
            return f"the builtin `{node.func}(...)`"
    return None


def _index_names(expr: LetExprNode) -> list[str]:
    """The variables an expression uses as tensor indices.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    list[str]
        Names whose elements serve as an index, in order: a name
        written as an index (``E[tokens]``), and a name gathered from in
        an index position (``E[tokens[q], k]``), whose gathered entries
        are the index values.
    """
    found: list[str] = []
    for node in walk_let_expr(expr):
        if not isinstance(node, LetExprIndex):
            continue
        for index in node.indices:
            source = index
            while isinstance(source, LetExprIndex):
                source = source.array
            if isinstance(source, LetExprVar) and source.name not in found:
                found.append(source.name)
    return found


def _gathered_tables(steps: Sequence[ProgramStep]) -> dict[str, int]:
    """The names let expressions only gather from, with the axes they index.

    A name subscripted as ``table[i]`` or ``table[i, j]``, and read no
    other way, is a lookup table: its extents are those of whatever it
    tabulates, not of the plate the gather's result ranges over. A name
    also read whole, as an operand or a draw argument, keeps the type
    that use gives it.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        The program's steps, marginalization scopes included.

    Returns
    -------
    dict[str, int]
        Each gathered name and how many axes one gather indexes; the
        largest such count when it is gathered more than once.
    """
    tables: dict[str, int] = {}
    whole: set[str] = set()

    def visit(body: Sequence[ProgramStep]) -> None:
        """Record the gathers and whole reads of a body's steps.

        Parameters
        ----------
        body : Sequence[ProgramStep]
            The steps.
        """
        for step in body:
            if isinstance(step, MarginalizeStep):
                visit(step.scope)
            if isinstance(step, SampleStep | ObserveStep | MarginalizeStep):
                for argument in step.args or ():
                    whole.update(_argument_names(argument))
            if not isinstance(step, LetStep | ScoreStep):
                continue
            gathered: dict[str, int] = {}
            reads: dict[str, int] = {}
            for node in walk_let_expr(step.value):
                if isinstance(node, LetExprVar):
                    reads[node.name] = reads.get(node.name, 0) + 1
                if not isinstance(node, LetExprIndex):
                    continue
                if isinstance(node.array, LetExprVar):
                    # The innermost subscript of a chain holds the name.
                    name = node.array.name
                    gathered[name] = gathered.get(name, 0) + 1
                rank = len(node.indices)
                array = node.array
                while isinstance(array, LetExprIndex):
                    rank += len(array.indices)
                    array = array.array
                if isinstance(array, LetExprVar):
                    tables[array.name] = max(rank, tables.get(array.name, 0))
            whole.update(
                name for name, count in reads.items() if count > gathered.get(name, 0)
            )

    visit(steps)
    return {name: rank for name, rank in tables.items() if name not in whole}


type _Shape = tuple[IndexTerm, ...]
"""The dimensions of a value, outermost first; empty for a scalar."""


def _tie_extents(
    parameter: TypeExpr, bound: TypeExpr, extents: dict[str, IndexTerm]
) -> None:
    """Tie a callee's open extents to the extents of a value passed to it.

    Parameters
    ----------
    parameter : TypeExpr
        The callee's parameter type, which may mention its index
        variables.
    bound : TypeExpr
        The type of the value the caller passes.
    extents : dict[str, IndexTerm]
        The callee's index variables tied so far, extended in place with
        each variable the parameter's dimensions name, taken from the
        passed value's dimension at the same position.
    """
    expected = tensor_shape(parameter)
    actual = tensor_shape(bound)
    if expected is None or actual is None or len(expected[1]) != len(actual[1]):
        return
    for wanted, given in zip(expected[1], actual[1], strict=True):
        if isinstance(wanted, IndexVariable) and wanted.name not in extents:
            extents[wanted.name] = given


def _specialized_scan_step(
    elaborator: _Elaborator,
    step_name: str,
    scope: _Scope,
    state: _ProgramState,
    node: LetStep,
) -> str:
    """The scan step specialized to the bindings its cell reads.

    A program cell is elaborated on its own, so a name it reads
    without binding it is typed from its uses there, as host data: a
    scalar for a bare use, a table of open extents for a subscripted
    one. When the scanning program binds that name before the scan,
    the cell reads the binding, so it is elaborated again with each
    such name at its binding's type, under a name of its own, and the
    step program calls that specialization.

    Parameters
    ----------
    elaborator : _Elaborator
        The module's elaborator.
    step_name : str
        The step program the scan runs.
    scope : _Scope
        The scanning program's scope at the scan.
    state : _ProgramState
        The scanning program's state, whose helpers gain the
        specialized step and cell.
    node : LetStep
        The scan's step.

    Returns
    -------
    str
        The specialized step program's name, or ``step_name`` when no
        binding of the scanning program differs from the type the
        cell reads its name at.
    """
    step_decl = elaborator._program_declarations.get(step_name)
    if step_decl is None:
        return step_name
    suffix = f"__in_{state.declaration.name}_{node.name}"
    draws = list(step_decl.draws)
    changed = False
    for index, draw in enumerate(step_decl.draws):
        if not isinstance(draw, CallStep):
            continue
        cell_decl = elaborator._program_declarations.get(draw.call.callee)
        cell_entry = next(
            (item for item in elaborator.entries if item.name == draw.call.callee), None
        )
        if cell_decl is None or cell_entry is None:
            continue
        pins: dict[str, TypeExpr] = {}
        for parameter in cell_entry.parameters:
            if parameter.role in ("domain", "observation"):
                continue
            binding = scope.lookup(parameter.name)
            if binding is not None and binding.type != parameter.type:
                pins[parameter.name] = binding.type
        if not pins:
            continue
        cell_name = f"{cell_decl.name}{suffix}"
        state.helpers.extend(
            _elaborate_nested_program(elaborator, cell_decl.with_(name=cell_name), pins)
        )
        draws[index] = draw.with_(call=draw.call.with_(callee=cell_name))
        changed = True
    if not changed:
        return step_name
    specialized = step_decl.with_(name=f"{step_name}{suffix}", draws=tuple(draws))
    state.helpers.extend(_elaborate_nested_program(elaborator, specialized, {}))
    return specialized.name


def _elaborate_nested_program(
    elaborator: _Elaborator, declaration: ProgramDecl, pins: Mapping[str, TypeExpr]
) -> tuple[NamedComputation, ...]:
    """Elaborate a program while another program is being elaborated.

    Parameters
    ----------
    elaborator : _Elaborator
        The module's elaborator.
    declaration : ProgramDecl
        The program.
    pins : Mapping[str, TypeExpr]
        Types fixed for free names the program reads.

    Returns
    -------
    tuple[NamedComputation, ...]
        The program's computation and its helpers.
    """
    saved_state = elaborator._program_state
    saved_macros = elaborator._lambda_macros
    saved_pins = elaborator._pinned_types
    elaborator._lambda_macros = {}
    elaborator._pinned_types = dict(pins)
    elaborator._program_declarations[declaration.name] = declaration
    try:
        return elaborator._elaborate_program(declaration)
    finally:
        elaborator._program_state = saved_state
        elaborator._lambda_macros = saved_macros
        elaborator._pinned_types = saved_pins


def _scan_step_inputs(
    elaborator: _Elaborator,
    step_name: str,
    step_entry: ProgramEntry,
    signature: ComputationSignature,
    extent: IndexTerm,
    scope: _Scope,
    state: _ProgramState,
    node: LetStep,
) -> tuple[list[Local], list[bool], tuple[StaticArgument, ...], StaticSubstitution]:
    """The inputs a scan passes its step after the entry and the state.

    Parameters
    ----------
    elaborator : _Elaborator
        The module's elaborator.
    step_name : str
        The step program the scan runs at every position.
    step_entry : ProgramEntry
        Its entry, whose parameters after the first two are passed.
    signature : ComputationSignature
        Its signature, whose telescope the passed inputs fix.
    extent : IndexTerm
        The number of positions the scan runs over.
    scope : _Scope
        The scanning program's scope at the scan, which gains the
        inputs it declares.
    state : _ProgramState
        The scanning program's state.
    node : LetStep
        The scan's step.

    Returns
    -------
    tuple[list[Local], list[bool], tuple[StaticArgument, ...], StaticSubstitution]
        The scanning program's local passed for each parameter; for
        each, whether it is a sequence of observations read at the
        position; the step's static arguments; and the substitution
        fixing the step's open extents.

    Raises
    ------
    QiecDiagnosticError
        If the scanning program binds a name the step observes at
        every position, or no input fixes one of the step's open
        extents.
    """
    # The step's remaining inputs pass through the recurrence from
    # the scanning program. A name the step reads without binding
    # it is the scanning program's binding of that name before the
    # scan, else an input of the scanning program. An observation
    # the step scores is one observation per position: the scanning
    # program takes the sequence of them, and position ``t`` reads
    # its ``t``-th entry.
    # The step's own open extents (a table it reads as host data) are
    # tied to extents of the scanning program, from the binding the
    # scanning program passes or from a fresh input of its own.
    extents: dict[str, IndexTerm] = {}
    passed: list[Local] = []
    positional: list[bool] = []
    for parameter in step_entry.parameters[2:]:
        existing = scope.lookup(parameter.name)
        if existing is not None:
            _tie_extents(parameter.type, existing.type, extents)
        if parameter.role == "observation":
            if scope.lookup(parameter.name) is not None:
                elaborator._fail(
                    node,
                    f"the scan step {step_name!r} observes "
                    f"{parameter.name!r} at every position, so the "
                    f"scanning program cannot also bind {parameter.name!r}",
                    code="qiec-program",
                )
            observed = elaborator._passed_input_type(
                parameter.name, parameter.type, state, extents
            )
            observed_shape = tensor_shape(observed)
            sequence_type: TypeExpr = (
                tensor_type(observed, (extent,))
                if observed_shape is None
                else tensor_type(observed_shape[0], (extent, *observed_shape[1]))
            )
            passed.append(
                elaborator._declare_parameter(
                    parameter.name,
                    sequence_type,
                    parameter.role,
                    scope,
                    state,
                    node,
                )
            )
            positional.append(True)
            continue
        if existing is None:
            existing = elaborator._declare_parameter(
                parameter.name,
                elaborator._passed_input_type(
                    parameter.name, parameter.type, state, extents
                ),
                parameter.role,
                scope,
                state,
                node,
            )
        passed.append(existing)
        positional.append(False)
    unresolved = [
        binder.name for binder in signature.telescope if binder.name not in extents
    ]
    if unresolved:
        elaborator._fail(
            node,
            f"scan steps with {step_name!r}, whose open extents "
            f"{', '.join(unresolved)} no input of the scanning program fixes",
            code="qiec-program",
        )
    step_statics: tuple[StaticArgument, ...] = tuple(
        extents[binder.name] for binder in signature.telescope
    )
    substitution = StaticSubstitution(
        indices=tuple(
            (binder.name, extents[binder.name]) for binder in signature.telescope
        )
    )
    return passed, positional, step_statics, substitution


def _elaborate_marginal_chain(
    elaborator: _Elaborator,
    scan: ScanCall,
    scope: _Scope,
    state: _ProgramState,
    node: LetStep,
) -> None:
    """Elaborate a marginalized chain to its log marginal likelihood.

    The scan's cell returns the latent it marginalizes, so the chain
    ``z_0 ~ init``, ``z_t ~ cell's transition from z_{t-1}``, with
    each position's observations scored at ``z_t``, has every state
    summed out. The cell elaborates to a row: the program that, at
    an entry and a previous state, binds its block's latent to the
    block's weighted shots, ``log p(z_t = j | z_{t-1}) + log p(obs_t
    | z_t = j)`` for every state ``j``, enumerated under the table
    handler. A recursion over the positions carries the log filtering
    distribution ``log p(z_t | obs_{1..t})`` and adds each position's
    evidence, which is the forward algorithm in log space::

        table_t[j] = logsumexp_i(carried[i] + row(x_t, i)[j])
        evidence_t = logsumexp_j(table_t[j])
        carried'   = table_t - evidence_t

    starting from ``log init``, and the sum of the evidences is the
    log marginal likelihood, bound to the step's name and scored.

    Parameters
    ----------
    elaborator : _Elaborator
        The module's elaborator.
    scan : ScanCall
        The marginalized scan: its cell, its sequence, and the name
        of the first state's distribution.
    scope : _Scope
        The scanning program's scope at the scan.
    state : _ProgramState
        The scanning program's accumulating state.
    node : LetStep
        The scan's step.

    Raises
    ------
    QiecDiagnosticError
        If the cell is not a chain step the forward algorithm sums
        out, its state object has no literal extent, the sequence or
        the first state's distribution is not bound, or the entries
        are not what the cell reads.
    """
    cell_decl = elaborator._program_declarations.get(scan.step)
    if cell_decl is None:
        elaborator._fail(
            node,
            f"the marginalized chain runs {scan.step!r}, which is not a program",
            code="qiec-program",
        )
    try:
        cell = markov_cell(cell_decl)
    except StepResolutionError as error:
        elaborator._fail(node, error.detail, code="qiec-program")
    states = elaborator._object_axis(cell.state_object, node)
    if not isinstance(states.size, IndexLiteral):
        elaborator._fail(
            node,
            f"the marginalized chain over {cell.name!r} sums states of an "
            "object with no literal extent",
            code="qiec-program",
        )
    count = int(states.size.value)
    if scan.init is None or scope.lookup(scan.init) is None:
        elaborator._fail(
            node,
            f"the marginalized chain over {cell.name!r} starts from "
            f"{scan.init!r}, which the program does not bind before the scan",
            code="qiec-program",
        )
    # The cell reads the scanning program's bindings at their types.
    own = {cell.entry, cell.previous, cell.latent}
    for step in cell_decl.draws:
        own |= _bound_names(step)
    reads: set[str] = set()
    for step in cell_decl.draws:
        reads |= _step_value_names(step)
    pins = {
        name: local.type
        for name in sorted(reads - own)
        if (local := scope.lookup(name)) is not None
    }
    row_name = f"{cell.name}__row__in_{state.declaration.name}_{node.name}"
    elaborator._markov_rows[row_name] = (cell.latent, cell.entry)
    try:
        state.helpers.extend(
            _elaborate_nested_program(elaborator, cell_decl.with_(name=row_name), pins)
        )
    finally:
        del elaborator._markov_rows[row_name]
    signature = elaborator.computation_signatures[row_name]
    row_entry = next(item for item in elaborator.entries if item.name == row_name)
    table_type = tensor_type(LOG_WEIGHT, (states.size,))
    if (
        len(signature.parameters) < 2
        or signature.parameters[1] != INT
        or signature.result != table_type
    ):
        elaborator._fail(
            node,
            f"the cell {cell.name!r} of a marginalized chain does not take "
            "an entry and a state and weigh each next state",
            code="qiec-program",
        )
    entry_type = signature.parameters[0]
    local = scope.lookup(scan.sequence)
    if local is None:
        elaborator._fail(
            node,
            f"the marginalized chain runs over {scan.sequence!r}, which is "
            "neither bound nor a program input",
            code="qiec-program",
        )
    shape = tensor_shape(local.type)
    if shape is None or not shape[1]:
        elaborator._fail(
            node,
            f"the marginalized chain runs over {scan.sequence!r}, which is "
            "not a sequence",
            code="qiec-program",
        )
    extent = shape[1][0]
    per_position: TypeExpr = (
        shape[0] if len(shape[1]) == 1 else tensor_type(shape[0], shape[1][1:])
    )
    if per_position != entry_type:
        elaborator._fail(
            node,
            f"the marginalized chain feeds {cell.name!r} entries of "
            f"{render_static(per_position)}, not the "
            f"{render_static(entry_type)} it reads",
            code="qiec-program",
        )
    passed, positional, row_statics, substitution = _scan_step_inputs(
        elaborator, row_name, row_entry, signature, extent, scope, state, node
    )
    name = f"{state.declaration.name}__{node.name}_chain"
    if name in elaborator.computation_signatures:
        elaborator._fail(
            node,
            f"chain helper {name!r} is declared twice",
            code="qiec-program",
        )
    mentioned = _index_variables_in((local.type, *(item.type for item in passed)))
    telescope = tuple(
        binder for name_, binder in state.extents.items() if name_ in mentioned
    )
    statics: tuple[StaticArgument, ...] = tuple(
        IndexVariable(binder.name, binder.sort) for binder in telescope
    )
    filter_type = tensor_type(REAL, (states.size,))
    visited = Local("__chain_t", INT)
    sequence = Local("__chain_xs", local.type)
    carried = Local("__chain_log_filter", filter_type)
    passed_locals = tuple(Local(f"__chain_{item.name}", item.type) for item in passed)
    identity = ComputationId.derive(elaborator.source.module_name, "computation", name)
    helper = ComputationSignature(
        identity,
        name,
        telescope,
        (INT, local.type, filter_type, *(item.type for item in passed_locals)),
        REAL,
        signature.effects,
    )
    try:
        elaborator.registry.register_computation(helper)
    except (KernelError, TypeError, ValueError) as error:
        elaborator._fail_kernel(node, error, fallback="qiec-program")
    elaborator.computation_signatures[name] = helper
    path = ("programs", state.declaration.name, "chains", node.name)
    counter = Local("__chain_k", INT)
    length: Value = Reduction(
        "sum",
        Comprehension(
            counter, extent, LiteralValue(1, INT), tensor_type(INT, (extent,))
        ),
        INT,
    )
    position = Local("__chain_p", INT)
    position_value: Value = (
        elaborator._primitive(
            "sub_int",
            (
                elaborator._primitive(
                    "sub_int", (length, LiteralValue(1, INT)), node, (*path, "last")
                ),
                Var(visited),
            ),
            node,
            (*path, "position"),
        )
        if scan.reverse
        else Var(visited)
    )
    entry = Local("__chain_x", entry_type)
    rows = tuple(Local(f"__chain_row_{index}", table_type) for index in range(count))
    table = Local("__chain_table", filter_type)
    evidence = Local("__chain_evidence", REAL)
    following = Local("__chain_next", filter_type)
    rest = Local("__chain_rest", REAL)
    column = Local("__chain_j", INT)
    # The weight of moving from state i to state j and scoring j's
    # observations, after the filtered weight of i.
    moves = TensorValue(
        tuple(
            elaborator._primitive(
                "add_real",
                (
                    Gather(Var(carried), LiteralValue(index, INT), REAL),
                    elaborator._primitive(
                        "weight_value",
                        (Gather(Var(row), Var(column), LOG_WEIGHT),),
                        node,
                        (*path, "rows", index, "weight"),
                    ),
                ),
                node,
                (*path, "rows", index, "move"),
            )
            for index, row in enumerate(rows)
        ),
        filter_type,
    )
    recur = Call(
        identity,
        name,
        statics,
        (
            elaborator._primitive(
                "add_int",
                (Var(visited), LiteralValue(1, INT)),
                node,
                (*path, "successor"),
            ),
            Var(sequence),
            Var(following),
            *(Var(item) for item in passed_locals),
        ),
        REAL,
        signature.effects,
        _origin_at(elaborator, path, "call", node),
    )
    continuation: Computation = Bind(
        table,
        Return(
            Comprehension(
                column,
                states.size,
                Reduction("logsumexp", moves, REAL),
                filter_type,
            )
        ),
        Bind(
            evidence,
            Return(Reduction("logsumexp", Var(table), REAL)),
            Bind(
                following,
                Return(
                    Comprehension(
                        column,
                        states.size,
                        elaborator._primitive(
                            "sub_real",
                            (
                                Gather(Var(table), Var(column), REAL),
                                Var(evidence),
                            ),
                            node,
                            (*path, "normalize"),
                        ),
                        filter_type,
                    )
                ),
                Bind(
                    rest,
                    recur,
                    Return(
                        elaborator._primitive(
                            "add_real",
                            (Var(evidence), Var(rest)),
                            node,
                            (*path, "accumulate"),
                        )
                    ),
                ),
            ),
        ),
    )
    passed_arguments = tuple(
        Gather(
            Var(item),
            Var(position),
            substitute_type(parameter.type, substitution),
        )
        if per_step
        else Var(item)
        for item, per_step, parameter in zip(
            passed_locals, positional, row_entry.parameters[2:], strict=True
        )
    )
    for index, row in reversed(tuple(enumerate(rows))):
        continuation = Bind(
            row,
            Call(
                signature.id,
                signature.name,
                row_statics,
                (Var(entry), LiteralValue(index, INT), *passed_arguments),
                table_type,
                signature.effects,
                _origin_at(elaborator, (*path, "rows", index), "call", node),
            ),
            continuation,
        )
    body: Computation = If(
        elaborator._primitive(
            "eq_int", (Var(visited), length), node, (*path, "finished")
        ),
        Return(LiteralValue(0.0, REAL)),
        Bind(
            position,
            Return(position_value),
            Bind(
                entry,
                Return(Gather(Var(sequence), Var(position), entry_type)),
                continuation,
            ),
        ),
    )
    state.helpers.append(
        elaborator._checked_computation(
            helper, (visited, sequence, carried, *passed_locals), body, node, path
        )
    )
    # The log filtering distribution before the first position is the
    # log of the first state's distribution, minus infinity where it
    # puts no mass.
    start = elaborator._step_distribution(
        SampleStep(
            vars=(f"{node.name}$start",),
            morphism="Categorical",
            args=(DrawArgNamed(parameter="probs", value=DrawArgName(text=scan.init)),),
            line=node.line,
            col=node.col,
        ),
        scope,
        state,
        f"{node.name}$start",
    )
    initial = Comprehension(
        column,
        states.size,
        elaborator._primitive(
            "weight_value",
            (
                LogDensity(
                    start,
                    Var(column),
                    _origin_at(elaborator, (*path, "start"), "log-density", node),
                ),
            ),
            node,
            (*path, "start"),
        ),
        filter_type,
    )
    marginal = Call(
        identity,
        name,
        statics,
        (
            LiteralValue(0, INT),
            Var(local),
            initial,
            *(Var(item) for item in passed),
        ),
        REAL,
        signature.effects,
        _origin_at(elaborator, path, "call", node),
    )
    binder = elaborator._bind_step(scope, node.name, REAL, marginal, node)
    weight = elaborator._primitive("as_weight", (Var(binder),), node, (*path, "weight"))
    assert state.score is not None
    elaborator._add_score(state.score, weight, scope, state, node)


def _positions_value(
    elaborator: _Elaborator,
    names: tuple[str, ...],
    scope: _Scope,
    node: LetStep,
) -> Comprehension:
    """Pair sequences position by position: ``$positions(a, b, ...)``.

    Parameters
    ----------
    elaborator : _Elaborator
        The elaborator, which reports a refusal.
    names : tuple[str, ...]
        The sequences, each bound to a tensor of reals whose leading
        axis is the positions and whose entries have a literal width
        (or are scalars).
    scope : _Scope
        The scope.
    node : LetStep
        The step.

    Returns
    -------
    Comprehension
        The sequence whose entry at each position lays the entries
        of the sequences there end to end, ``Tensor[Real]([E, w])``
        for the shared positions ``E`` and the summed width ``w``.

    Raises
    ------
    QiecDiagnosticError
        If a name is unbound, is not such a sequence, or the
        sequences do not share their positions.
    """
    extent: IndexTerm | None = None
    position = Local("__positions_t", INT)
    entries: list[Value] = []
    for name in names:
        local = scope.lookup(name)
        shape = None if local is None else tensor_shape(local.type)
        if (
            local is None
            or shape is None
            or shape[0] != REAL
            or not 1 <= len(shape[1]) <= 2
            or (len(shape[1]) == 2 and not isinstance(shape[1][1], IndexLiteral))
        ):
            elaborator._fail(
                node,
                f"the branches paired position by position must each be a "
                f"sequence of real states; {name!r} is not",
                code="qiec-program",
            )
        assert local is not None and shape is not None
        if extent is None:
            extent = shape[1][0]
        elif shape[1][0] != extent:
            elaborator._fail(
                node,
                f"the branches paired position by position run over "
                f"{render_static(extent)} and {render_static(shape[1][0])} "
                f"positions; pair sequences over one input",
                code="qiec-program",
            )
        if len(shape[1]) == 1:
            entries.append(Gather(Var(local), Var(position), REAL))
            continue
        width = shape[1][1]
        assert isinstance(width, IndexLiteral)
        row = Gather(Var(local), Var(position), tensor_type(REAL, (width,)))
        entries.extend(
            Gather(row, LiteralValue(index, INT), REAL)
            for index in range(int(width.value))
        )
    assert extent is not None
    entry_type = tensor_type(REAL, (IndexLiteral(len(entries), NAT),))
    return Comprehension(
        position,
        extent,
        TensorValue(tuple(entries), entry_type),
        tensor_type(REAL, (extent, IndexLiteral(len(entries), NAT))),
    )


def _scan_initial_state(
    elaborator: _Elaborator,
    init: str,
    state_type: TypeExpr,
    scope: _Scope,
    state: _ProgramState,
    node: LetStep,
) -> Value:
    """The state a scan starts from, named by its ``init``.

    Parameters
    ----------
    elaborator : _Elaborator
        The module's elaborator.
    init : str
        The name ``scan(cell, init=...)`` gives: a binding of the
        scanning program made before the scan, or the input a learned
        initial state is read from.
    state_type : TypeExpr
        The scan's state type.
    scope : _Scope
        The scanning program's scope at the scan.
    state : _ProgramState
        The scanning program's state.
    node : LetStep
        The scan's step.

    Returns
    -------
    Value
        The binding, a scalar binding wrapped as the one entry of a
        one-wide state, or the learned initial state's input.

    Raises
    ------
    QiecDiagnosticError
        If the name is neither bound before the scan nor a learned
        initial state, or the binding's type is not the state's.
    """
    bound = scope.lookup(init)
    if bound is None:
        if not is_learned_scan_init(init):
            elaborator._fail(
                node,
                f"the scan starts from {init!r} (init={init}), which the "
                "program does not bind before the scan",
                code="qiec-program",
            )
        learned = elaborator._declare_parameter(
            init, state_type, "weight", scope, state, node
        )
        return Var(learned)
    if bound.type == state_type:
        return Var(bound)
    shape = tensor_shape(state_type)
    if (
        shape is not None
        and shape[1] == (IndexLiteral(1, NAT),)
        and bound.type == shape[0]
    ):
        counter = Local("__scan_init_k", INT)
        return Comprehension(counter, IndexLiteral(1, NAT), Var(bound), state_type)
    elaborator._fail(
        node,
        f"the scan starts from {init!r}, of type {render_static(bound.type)}, "
        f"but its state has type {render_static(state_type)}",
        code="qiec-program",
    )


def _unify_free_shapes(
    expr: LetExprNode,
    free: frozenset[str],
    bound_type: Callable[[str], TypeExpr | None],
    bound_batch: Callable[[str], _Shape],
) -> tuple[dict[str, _Shape], frozenset[str], _Shape]:
    """Infer the shapes of free names from the plated operands they meet.

    A free name combined elementwise, by an arithmetic operator or
    through elementwise calls, with a value ranging over plates takes
    those plates' extents: it holds one entry per row the value has. A
    value's event axes, and the axes of a value no plate shapes, do not
    fix a free name, which a scalar could fill by broadcasting. A free
    name whose value a reduction or a rowwise operation consumes needs
    at least one axis; if nothing in the expression fixes its shape, it
    is reported as ambiguous.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    free : frozenset[str]
        The names no step binds.
    bound_type : Callable[[str], TypeExpr | None]
        The type of a bound name, or ``None`` for a name the expression
        binds itself.
    bound_batch : Callable[[str], _Shape]
        The plate extents a bound name ranges over, outermost first.

    Returns
    -------
    tuple[dict[str, _Shape], frozenset[str], _Shape]
        The plate extents fixed for each free name a plated operand
        meets, the free names a reduction consumes that nothing fixes,
        and the plate extents the expression's value ranges over.
    """
    fixed: dict[str, _Shape] = {}
    reduced: set[str] = set()

    def bare(node: LetExprNode) -> set[str]:
        """Free names reaching ``node``'s value through elementwise steps.

        Parameters
        ----------
        node : LetExprNode
            The operand.

        Returns
        -------
        set[str]
            The names whose shape is the operand's.
        """
        if isinstance(node, LetExprVar):
            return {node.name} if node.name in free else set()
        if isinstance(node, LetExprBinOp):
            return bare(node.left) | bare(node.right)
        if isinstance(node, LetExprUnaryOp):
            return bare(node.operand)
        if (
            isinstance(node, LetExprCall)
            and node.func not in _REDUCTIONS
            and node.func not in SHAPE_BUILTINS
            and node.func in PURE_BUILTINS
        ):
            return set().union(*(bare(argument) for argument in node.args))
        return set()

    def rank(node: LetExprNode) -> int | None:
        """The number of axes of a value, when the expression fixes it.

        Parameters
        ----------
        node : LetExprNode
            The value.

        Returns
        -------
        int | None
            Its rank, or ``None`` when it reads a free name of unknown
            shape or a form whose rank this walk does not follow.
        """
        if isinstance(node, LetExprLiteral | LetExprString):
            return 0
        if isinstance(node, LetExprVar):
            if node.name in free:
                known = fixed.get(node.name)
                return None if known is None else len(known)
            type_ = bound_type(node.name)
            if type_ is None:
                return None
            split = tensor_shape(type_)
            return 0 if split is None else len(split[1])
        if isinstance(node, LetExprUnaryOp):
            return rank(node.operand)
        if isinstance(node, LetExprBinOp | LetExprCall):
            children = (
                (node.left, node.right) if isinstance(node, LetExprBinOp) else node.args
            )
            ranks = [rank(child) for child in children]
            if any(item is None for item in ranks):
                return None
            widest = max((item for item in ranks if item is not None), default=0)
            if (
                isinstance(node, LetExprCall)
                and node.func in _REDUCTIONS
                and len(node.args) == 1
            ):
                return max(0, widest - 1)
            return widest
        return None

    def batch(node: LetExprNode) -> _Shape | None:
        """The plate extents a value ranges over.

        Parameters
        ----------
        node : LetExprNode
            The value.

        Returns
        -------
        _Shape | None
            Its plate extents, outermost first, or ``None`` when it
            reads a free name nothing has fixed.
        """
        if isinstance(node, LetExprVar):
            if node.name in free:
                return fixed.get(node.name)
            return bound_batch(node.name)
        if isinstance(node, LetExprUnaryOp):
            return batch(node.operand)
        if isinstance(node, LetExprBinOp):
            left, right = batch(node.left), batch(node.right)
            if left is None and right is not None:
                meet(node.left, right)
                return right
            if right is None and left is not None:
                meet(node.right, left)
                return left
            if left is None or right is None:
                return None
            return left if len(left) >= len(right) else right
        if isinstance(node, LetExprCall):
            shapes = [batch(argument) for argument in node.args]
            if node.func in _REDUCTIONS and len(node.args) == 1:
                if shapes[0] is None:
                    reduced.update(bare(node.args[0]))
                    return None
                # The last axis is a plate's when the argument has no
                # event axis after its plates.
                if shapes[0] and rank(node.args[0]) == len(shapes[0]):
                    return shapes[0][:-1]
                return shapes[0]
            if node.func in _ROWWISE and len(node.args) == 1:
                if shapes[0] is None:
                    reduced.update(bare(node.args[0]))
                return shapes[0]
            if node.func in SHAPE_BUILTINS:
                # The plates lead every operand and are kept; the last
                # axis is rebuilt, so no free operand takes a shape here.
                tensors = shapes if node.func == "concat" else shapes[:1]
                if any(item is None for item in tensors) or len(set(tensors)) != 1:
                    return None
                return tensors[0]
            if node.func in PURE_BUILTINS:
                known = [item for item in shapes if item is not None]
                if len(known) < len(shapes):
                    return None
                return max(known, key=len, default=())
            return ()
        for child in let_expr_children(node):
            batch(child)
        return ()

    def meet(unknown: LetExprNode, known: _Shape) -> None:
        """Fix the free names of an operand to a plated partner's extents.

        Parameters
        ----------
        unknown : LetExprNode
            The operand of unknown shape.
        known : _Shape
            Its partner's plate extents.
        """
        if not known:
            # A partner over no plate broadcasts a scalar as well as a
            # vector, so it fixes nothing.
            return
        for name in bare(unknown):
            fixed.setdefault(name, known)

    # A shape fixed late in the walk can settle an operand met earlier.
    batch(expr)
    result = batch(expr)
    return fixed, frozenset(reduced - set(fixed)), result or ()


def _argument_names(argument: DrawArg) -> list[str]:
    """The names a draw argument reads.

    Parameters
    ----------
    argument : DrawArg
        The argument.

    Returns
    -------
    list[str]
        Bare names, indexed bases and their indices, and list entries.
    """
    if isinstance(argument, DrawArgNamed):
        return _argument_names(argument.value)
    if isinstance(argument, DrawArgName):
        return [argument.text]
    if isinstance(argument, DrawArgIndex):
        return [argument.name, *argument.indices]
    if isinstance(argument, DrawArgList):
        return [name for item in argument.items for name in _argument_names(item)]
    if isinstance(argument, DrawArgDist):
        return [name for item in argument.args for name in _argument_names(item)]
    return []


def _is_number_text(text: str) -> bool:
    """Whether wire text spells a number.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    bool
        ``True`` when ``float`` accepts it.
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class _Spread:
    """A single literal filling every position of a vector parameter.

    Parameters
    ----------
    value
        The literal.
    dimension
        The vector's length.
    """

    value: float
    dimension: int


def _literal(value: float, integral: bool) -> LiteralValue:
    """A numeric literal at the type a parameter expects.

    Parameters
    ----------
    value : float
        The number.
    integral : bool
        Whether the position takes an ``Int``.

    Returns
    -------
    LiteralValue
        An ``Int`` literal when integral and whole, else a ``Real``.
    """
    if integral and float(value).is_integer():
        return LiteralValue(int(value), INT)
    return LiteralValue(float(value), REAL)


def _element_of(parameter: FamilyParameter | _ScalarParameter) -> TypeApplication:
    """The scalar element type a family parameter is made of.

    Parameters
    ----------
    parameter : FamilyParameter | _ScalarParameter
        The parameter.

    Returns
    -------
    TypeApplication
        ``Bool`` for a Boolean constraint, ``Int`` for an integer one,
        else ``Real``.
    """
    if parameter.constraint == "boolean":
        return BOOL
    if "integer" in parameter.constraint:
        return INT
    return REAL


class _ProgramElaboration(_SequenceFunctionElaboration):
    """Program elaboration, mixed into the QIEC elaborator.

    The methods read and extend the elaborator's declaration tables, so
    they run after the module's instances, handlers, and computation
    signatures are declared and before its computation bodies are
    lowered, which lets a program call a declared computation and a
    computation call a program.
    """

    def _declare_programs(self: _Elaborator) -> tuple[NamedComputation, ...]:
        """Elaborate every non-parametric program in the module.

        Returns
        -------
        tuple[NamedComputation, ...]
            The programs' computations followed by their marginalization
            helpers, in source order.

        Raises
        ------
        QiecDiagnosticError
            If a program uses a form the elaboration does not admit, or
            its body fails to check.
        """
        declarations = [
            statement
            for statement in self.source.syntax.statements
            if isinstance(statement, ProgramDecl) and _is_entry_point(statement)
        ]
        self._program_objects = _object_table(self.source.syntax.statements)
        if not declarations or not self.source.elaborate_programs:
            return ()
        try:
            expanded = expand_composite_lets(self.source.syntax)
            self._program_morphisms = morphism_table(expanded)
        except StepResolutionError as error:
            self._fail(
                declarations[0],
                error.detail,
                code=GAP_CODE if _is_gap(error.kinds) else "qiec-program",
            )
        self._refuse_kernel_arguments()
        self._program_lets = build_let_table(expanded)
        self._program_declarations = {
            declaration.name: declaration
            for declaration in expanded.statements
            if isinstance(declaration, ProgramDecl)
        }
        computations: list[NamedComputation] = []
        # The type of each latent every elaborated program samples, by
        # program, for the posterior blocks that consume them.
        self._program_latent_types: dict[str, dict[str, TypeExpr]] = {}
        # The cell of a marginalized chain returns the latent it
        # marginalizes; it elaborates only as the row of the scan that
        # runs it.
        cells = chain_cells(self.source.syntax)
        for declaration in expanded.statements:
            if (
                isinstance(declaration, ProgramDecl)
                and _is_entry_point(declaration)
                and declaration.name not in cells
            ):
                try:
                    computations.extend(self._elaborate_program(declaration))
                except QiecDiagnosticError as error:
                    error.program = declaration.name
                    raise
        return tuple(computations)

    def _refuse_kernel_arguments(self: _Elaborator) -> None:
        """Refuse a kernel declaration whose family arguments read values.

        A kernel's ``~ Family(...)`` arguments are literals or absent; a
        name or reference among them has nothing to read, since a
        declaration binds no value. The declaration compiler refuses the
        same declarations with the same message.

        Raises
        ------
        QiecDiagnosticError
            At the first such argument, in source order.
        """
        for name, morphism in self._program_morphisms.items():
            init = morphism.init_family
            if init is None or not init.args:
                continue
            role = _option_entry(morphism.options, "role")
            if role is not None and not (
                isinstance(role, OptionName) and role.value == "kernel"
            ):
                continue
            refusal = kernel_argument_refusal(name, init.family, init.args)
            if refusal is None:
                continue
            offending, message = refusal
            node = offending if getattr(offending, "line", 0) else init
            self._fail(node, message, code="qiec-program")

    # ------------------------------------------------------------------
    # module-level scaffolding

    def _declare_program_instances(self: _Elaborator) -> None:
        """Allocate the canonical instances before any signature is read.

        A computation may name ``random`` or ``score`` in its row to call
        a program, so the instances exist as soon as the module has a
        program to elaborate.
        """
        if not self.source.elaborate_programs:
            return
        if not any(
            isinstance(statement, ProgramDecl) and _is_entry_point(statement)
            for statement in self.source.syntax.statements
        ):
            return
        self._program_instance(RANDOM_INSTANCE, "Random")
        self._program_instance(SCORE_INSTANCE, "Score")

    def _program_instance(
        self: _Elaborator, name: str, interface: str
    ) -> NamedEffectInstance:
        """The canonical module instance a program's requests address.

        Parameters
        ----------
        name : str
            The instance's name.
        interface : str
            The prelude interface it must be an instance of.

        Returns
        -------
        NamedEffectInstance
            The module's instance of that name, declared or synthesized.

        Raises
        ------
        QiecDiagnosticError
            If the module declares the name as an instance of another
            interface.
        """
        existing = self.instances.get(name)
        if existing is not None:
            if existing.entry.effect.name != interface:
                self._fail(
                    self.source.syntax.statements[0],
                    f"instance {name!r} is reserved for the program's {interface} "
                    f"requests but is an instance of {existing.entry.effect.name!r}",
                    code="qiec-program",
                )
            return existing
        definition = self._effect(interface)
        assert definition is not None
        entry = instantiate_effect(
            definition.ref,
            module=self.source.module_name,
            lexical_path=("instances", name),
        )
        instance = NamedEffectInstance(
            name,
            entry,
            _origin_at(self, ("instances", name), "effect-instance"),
        )
        self.instances[name] = instance
        return instance

    def _program_handlers(self: _Elaborator) -> dict[str, HandlerDef]:
        """The enumeration and collection handlers marginalization uses.

        Returns
        -------
        dict[str, HandlerDef]
            By name: the ``enumerate_marginal[w]`` handlers over
            ``Random``, one per reduction, whose input is a unit answer
            paired with the collected weight of type ``w`` and whose
            output is the marginal ``LogWeight``; the
            ``enumerate_grouped_marginal[w]`` handlers, alike but
            answering the collected weight's type ``w`` itself, one
            marginal per group position; the ``enumerate_marginal_table[w]``
            handler, whose input is a unit answer paired with a
            ``LogWeight`` and whose output ``w`` is the tensor of the
            weighted shots, one per support value; and the
            ``collect_marginal[w]`` handler over ``Weight[w]``, which
            pairs the scope's unit answer with its total.

        Raises
        ------
        QiecDiagnosticError
            If the module declares a handler under one of the reserved
            names.
        """
        names = (*MARGINAL_HANDLERS.values(), ENUMERATE_TABLE_HANDLER, COLLECT_HANDLER)
        for name in names:
            declared = self.handlers.get(name)
            if declared is not None and declared.implementation != "foreign":
                self._fail(
                    self.source.syntax.statements[0],
                    f"handler {name!r} is reserved for marginalization",
                    code="qiec-program",
                )
        if all(name in self.handlers for name in names):
            return {name: self.handlers[name] for name in names}
        weight_binder = TypeBinder("w")
        weight = TypeVariable("w")
        random = self._effect("Random")
        weight_effect = self._effect("Weight")
        assert random is not None and weight_effect is not None
        sample = next(item for item in random.operations if item.name == "sample")
        add = next(item for item in weight_effect.operations if item.name == "add")
        declared_handlers: dict[str, HandlerDef] = {}
        for (grouped, _reduction), name in MARGINAL_HANDLERS.items():
            declared_handlers[name] = HandlerDef(
                HandlerId.derive(self.source.module_name, "handler", name),
                name,
                random.ref,
                (HandlerClauseDef(sample.id, ResumptionGrade.UNRESTRICTED),),
                product_type(UNIT, weight),
                weight if grouped else LOG_WEIGHT,
                EffectRow(),
                total=True,
                telescope=(weight_binder,),
                implementation="foreign",
            )
        declared_handlers[ENUMERATE_TABLE_HANDLER] = HandlerDef(
            HandlerId.derive(
                self.source.module_name, "handler", ENUMERATE_TABLE_HANDLER
            ),
            ENUMERATE_TABLE_HANDLER,
            random.ref,
            (HandlerClauseDef(sample.id, ResumptionGrade.UNRESTRICTED),),
            product_type(UNIT, LOG_WEIGHT),
            weight,
            EffectRow(),
            total=True,
            telescope=(weight_binder,),
            implementation="foreign",
        )
        declared_handlers[COLLECT_HANDLER] = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", COLLECT_HANDLER),
            COLLECT_HANDLER,
            weight_effect.apply((weight,)),
            (HandlerClauseDef(add.id, ResumptionGrade.LINEAR),),
            UNIT,
            product_type(UNIT, weight),
            EffectRow(),
            total=True,
            telescope=(weight_binder,),
            implementation="foreign",
        )
        for name, definition in declared_handlers.items():
            self.handlers[name] = definition
            self.registry.register_handler(definition)
        return declared_handlers

    # ------------------------------------------------------------------
    # one program

    def _elaborate_program(
        self: _Elaborator, declaration: ProgramDecl
    ) -> tuple[NamedComputation, ...]:
        """Elaborate one program declaration.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.

        Returns
        -------
        tuple[NamedComputation, ...]
            The program's computation and its marginalization helpers.

        Raises
        ------
        QiecDiagnosticError
            If a step cannot be elaborated or the body fails to check.
        """
        if declaration.name in self.computation_signatures:
            self._fail(
                declaration,
                f"program {declaration.name!r} shares its name with a computation",
                code="qiec-program",
            )
        state = _ProgramState(declaration)
        row = self._markov_rows.get(declaration.name)
        if row is not None:
            state.markov_row, state.markov_entry = row
        self._program_state = state
        random = self._program_instance(RANDOM_INSTANCE, "Random")
        score = self._program_instance(SCORE_INSTANCE, "Score")
        state.random = random
        state.score = score
        scope = _Scope()
        self._program_parameters(declaration, scope, state)
        state.input_shapes = self._infer_input_shapes(declaration)
        state.tables = _gathered_tables(declaration.draws)
        state.table_events = self._table_event_ranks(declaration, state.tables)
        self._elaborate_steps(declaration.draws, scope, state)
        self._program_latent_types[declaration.name] = {
            site.name: scope.locals[site.name].type
            for site in state.sites
            if site.kind == "sample" and site.name in scope.locals
        }
        self._unify_result_extents(declaration, scope, state)
        result, result_type = self._program_return(declaration, scope, state)
        body = self._fold_steps(scope, Return(result))
        if state.extent_solution:
            # Steps elaborated before an extent was fixed read it open;
            # the body, its sites, and the telescope take the fixed one.
            substitution = self._extent_substitution(state)
            body = substitute_computation(body, substitution)
            result_type = substitute_type(result_type, substitution)
            for name in state.extent_solution:
                state.extents.pop(name, None)
            state.sites = [
                replace(
                    site,
                    batch=fixed.batch,
                    event=fixed.event,
                )
                for site in state.sites
                for fixed in (
                    substitute_plate(PlateShape(site.batch, site.event), substitution),
                )
            ]
        row_entries: list[RowEntry] = []
        if state.uses_random:
            row_entries.append(random.entry)
        if state.uses_score:
            row_entries.append(score.entry)
        if state.uses_params:
            row_entries.append(self.instances[PARAMS_INSTANCE].entry)
        row = EffectRow(tuple(row_entries))
        parameters = tuple(scope.locals[name] for name, _ in state.parameters)
        telescope = tuple(state.extents.values())
        identity = ComputationId.derive(
            self.source.module_name, "computation", declaration.name
        )
        signature = ComputationSignature(
            identity,
            declaration.name,
            telescope,
            tuple(parameter.type for parameter in parameters),
            result_type,
            row,
        )
        try:
            self.registry.register_computation(signature)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(declaration, error, fallback="qiec-program")
        self.computation_signatures[declaration.name] = signature
        computation = NamedComputation(
            identity,
            declaration.name,
            telescope,
            parameters,
            body,
            ComputationType(row, result_type),
            _origin_at(self, ("programs", declaration.name), "computation"),
        )
        context = CheckContext(parameters)
        try:
            actual = infer_computation(body, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(declaration, error, fallback="qiec-program")
        if actual.result != result_type:
            self._fail(
                declaration,
                f"program body returns {actual.result!r}, expected {result_type!r}",
                code="qiec-program",
            )
        self.entries.append(
            ProgramEntry(
                declaration.name,
                identity,
                tuple(
                    ProgramParameter(
                        name,
                        role,
                        scope.locals[name].type,
                        state.parameter_axes.get(name, ()),
                        per_row=name in state.per_row,
                    )
                    for name, role in state.parameters
                ),
                tuple(declaration.return_vars),
                declaration.return_labels,
                tuple(state.sites),
                random.entry.instance,
                score.entry.instance,
                telescope,
            )
        )
        self._program_state = None
        for name in state.template_objects:
            del self._program_objects[name]
        return (computation, *state.helpers)

    def _program_parameters(
        self: _Elaborator, declaration: ProgramDecl, scope: _Scope, state: _ProgramState
    ) -> None:
        """Declare the parameters the program's signature names.

        A named parameter is typed by its domain factor: a real vector
        for a ``Real N`` factor, an index for a finite one. Without
        names, every ``Real N`` factor becomes a parameter named after
        its object in lowercase. Scalar parameters of the option block
        are reals or naturals.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        scope : _Scope
            The body's scope, which the parameters are bound in.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the parameter count does not fit the domain, or a factor
            has no readable shape.
        """
        self._template_objects(declaration, state)
        factors = _factors(declaration.domain)
        scanned = _scanned_names(declaration.draws)
        if declaration.params is not None:
            if len(declaration.params) != len(factors):
                self._fail(
                    declaration,
                    f"program {declaration.name!r} names {len(declaration.params)} "
                    f"parameter(s) for a domain of {len(factors)} factor(s)",
                    code="qiec-program",
                )
            consumed = self._consumed_latent_types(declaration)
            for name, factor in zip(declaration.params, factors, strict=True):
                if consumed is not None:
                    latent = consumed.get(name)
                    if latent is None:
                        self._fail(
                            declaration,
                            f"posterior block {declaration.name!r} names the "
                            f"parameter {name!r}, which is not a latent the "
                            "model it consumes samples; its latents are "
                            + (", ".join(repr(n) for n in consumed) or "none"),
                            code="qiec-program",
                        )
                    self._declare_parameter(
                        name, latent, "domain", scope, state, declaration
                    )
                    continue
                info = self._object_info(factor, declaration)
                if info.real_width is not None:
                    type_: TypeExpr = tensor_type(REAL, (_extent(info.real_width),))
                elif info.finite:
                    type_ = INT
                    if info.extent is not None:
                        # A class-valued parameter indexes an input by its
                        # object, as a class-valued step does.
                        state.alphabets[name] = self._object_axis(factor, declaration)
                else:
                    type_ = REAL
                if name in scanned:
                    type_ = self._sequence_type(name, type_, state)
                self._declare_parameter(
                    name,
                    type_,
                    "domain",
                    scope,
                    state,
                    declaration,
                    axes=self._domain_axes(factor, type_),
                )
        else:
            for factor in factors:
                if not isinstance(factor, TypeName):
                    continue
                info = self._program_objects.get(factor.name)
                name = factor.name.lower()
                if info is not None and info.real_width is not None:
                    type_ = tensor_type(REAL, (_extent(info.real_width),))
                elif info is not None and info.finite and name in scanned:
                    type_ = INT
                else:
                    continue
                if name in scanned:
                    # A scanned input is a sequence, one entry per position
                    # of an open extent.
                    type_ = self._sequence_type(name, type_, state)
                self._declare_parameter(
                    name,
                    type_,
                    "domain",
                    scope,
                    state,
                    declaration,
                    axes=self._domain_axes(factor, type_),
                )
        if declaration.type_params is not None:
            for parameter in declaration.type_params:
                if isinstance(parameter, ScalarParam):
                    self._declare_parameter(
                        parameter.name,
                        REAL if parameter.scalar_kind == "Real" else INT,
                        "scalar",
                        scope,
                        state,
                        declaration,
                    )

    def _template_objects(
        self: _Elaborator, declaration: ProgramDecl, state: _ProgramState
    ) -> None:
        """Bind a template's object parameters as index binders.

        An object parameter ``K : FinSet`` is an object of the program's
        body whose extent is the index variable ``K`` of the nat sort,
        bound first in the computation's telescope so a call supplies
        it as a static argument; a ``Space`` parameter is a real object
        of that width. The bindings last while the program elaborates.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        state : _ProgramState
            The program's accumulating state, whose extents gain one
            binder per object parameter.

        Raises
        ------
        QiecDiagnosticError
            If an object parameter shadows a declared object.
        """
        if declaration.type_params is None:
            return
        for parameter in declaration.type_params:
            if not isinstance(parameter, ObjectParam):
                continue
            if parameter.name in self._program_objects:
                self._fail(
                    parameter,
                    f"object parameter {parameter.name!r} of program "
                    f"{declaration.name!r} shadows a declared object",
                    code="qiec-program",
                )
            variable = IndexVariable(parameter.name, NAT)
            state.extents[parameter.name] = IndexBinder(parameter.name, NAT)
            if parameter.universe == "Space":
                info = ObjectInfo(extent=variable, real_width=variable)
            else:
                info = ObjectInfo(
                    extent=variable, finite=parameter.universe == "FinSet"
                )
            self._program_objects[parameter.name] = info
            state.template_objects.append(parameter.name)

    def _consumed_latent_types(
        self: _Elaborator, declaration: ProgramDecl
    ) -> dict[str, TypeExpr] | None:
        """The latent types of the model a posterior block consumes.

        A posterior block (``[over=model]``) names the model's latents as
        its parameters, so each parameter has the type of the model's
        site of that name: real for a real draw, a real vector for a
        vector draw, an index for a draw over a finite set.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.

        Returns
        -------
        dict[str, TypeExpr] | None
            The consumed model's sampled latents and their types, or
            ``None`` when the program is not a posterior block.

        Raises
        ------
        QiecDiagnosticError
            If the consumed model is not a program elaborated before the
            block.
        """
        over = _option_entry(declaration.options, "over")
        if not isinstance(over, OptionName):
            return None
        latents = self._program_latent_types.get(over.value)
        if latents is None:
            self._fail(
                declaration,
                f"posterior block {declaration.name!r} consumes {over.value!r}, "
                "which is not a program declared before it",
                code="qiec-program",
            )
        return latents

    def _declare_parameter(
        self: _Elaborator,
        name: str,
        type_: TypeExpr,
        role: ParameterRole,
        scope: _Scope,
        state: _ProgramState,
        node: object,
        axes: tuple[str, ...] | None = None,
    ) -> Local:
        """Bind one program parameter in the body's scope.

        Parameters
        ----------
        name : str
            The parameter's name.
        type_ : TypeExpr
            Its type.
        role : ParameterRole
            How it is supplied.
        scope : _Scope
            The scope it is bound in; a nested scope binds it in the
            program body's scope beneath it.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node the use is reported at.
        axes : tuple[str, ...] | None
            The name of each dimension of a tensor-typed parameter;
            read off the type when absent: an open extent's variable,
            else the one object of the dimension's cardinality, else
            the dimension's position.

        Returns
        -------
        Local
            The parameter's local.

        Raises
        ------
        QiecDiagnosticError
            If the name is already a parameter at another type or a
            bound step.
        """
        root = scope
        while root.outer is not None:
            root = root.outer
        existing = root.locals.get(name)
        if existing is not None:
            if existing.type != type_ and not self._unify_types(
                existing.type, type_, scope, state
            ):
                self._fail(
                    node,
                    f"program input {name!r} is read as {self._render(type_)} "
                    f"here but as {self._render(existing.type)} elsewhere",
                    code="qiec-program",
                )
            if scope is not root:
                scope.referenced.add(name)
            return root.locals[name]
        if scope.lookup(name) is not None:
            self._fail(
                node,
                f"{name!r} is bound by a step and cannot also be a program input",
                code="qiec-program",
            )
        local = Local(name, type_)
        root.locals[name] = local
        state.parameters.append((name, role))
        state.parameter_axes[name] = (
            axes if axes is not None else self._axes_of_type(type_)
        )
        if scope is not root:
            scope.referenced.add(name)
        return local

    def _axes_of_type(self: _Elaborator, type_: TypeExpr) -> tuple[str, ...]:
        """The axis names a tensor type suggests.

        Parameters
        ----------
        type_ : TypeExpr
            The type.

        Returns
        -------
        tuple[str, ...]
            Per dimension: an open extent's variable name, the one
            declared object of the dimension's cardinality when there
            is exactly one, else ``axis<i>``.
        """
        shape = tensor_shape(type_)
        if shape is None:
            return ()
        names: list[str] = []
        for index, size in enumerate(shape[1]):
            if isinstance(size, IndexVariable):
                names.append(size.name)
                continue
            if isinstance(size, IndexLiteral):
                matching = [
                    name
                    for name, info in self._program_objects.items()
                    if info.extent == size.value or info.real_width == size.value
                ]
                if len(matching) == 1:
                    names.append(matching[0])
                    continue
            names.append(f"axis{index}")
        return tuple(names)

    # ------------------------------------------------------------------
    # steps

    def _elaborate_steps(
        self: _Elaborator,
        steps: Sequence[ProgramStep],
        scope: _Scope,
        state: _ProgramState,
    ) -> None:
        """Elaborate a sequence of steps into the scope's binds.

        Parameters
        ----------
        steps : Sequence[ProgramStep]
            The steps, in order.
        scope : _Scope
            The scope they bind into.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If a step is of a kind the elaboration does not admit.
        """
        for step in steps:
            if isinstance(step, SampleStep):
                self._elaborate_sample(step, scope, state)
            elif isinstance(step, ObserveStep):
                self._elaborate_observe(step, scope, state)
            elif isinstance(step, LetStep):
                self._elaborate_let(step, scope, state)
            elif isinstance(step, CallStep):
                self._elaborate_call(step, scope, state)
            elif isinstance(step, ScoreStep):
                self._elaborate_score(step, scope, state)
            elif isinstance(step, MarginalizeStep):
                self._elaborate_marginalize(step, scope, state)
            elif isinstance(step, ReturnStep):
                continue
            else:
                self._fail(
                    step,
                    f"program step {step.kind!r} has no QIEC elaboration",
                    code="qiec-program",
                )

    def _bind_step(
        self: _Elaborator,
        scope: _Scope,
        name: str | None,
        type_: TypeExpr,
        computation: Computation,
        node: object,
    ) -> Local:
        """Append one bind to a scope.

        Parameters
        ----------
        scope : _Scope
            The scope.
        name : str | None
            The bound name, or ``None`` for a step whose value is unused.
        type_ : TypeExpr
            The bound value's type.
        computation : Computation
            The computation bound.
        node : object
            The source node, for diagnostics.

        Returns
        -------
        Local
            The binder.

        Raises
        ------
        QiecDiagnosticError
            If the name is already bound.
        """
        if name is None:
            binder = Local(f"_step{len(scope.steps)}_{id(scope) & 0xFFFF:x}", type_)
        else:
            if scope.lookup(name) is not None:
                self._fail(
                    node,
                    f"variable {name!r} already bound in program",
                    code="qiec-program",
                )
            binder = Local(name, type_)
            scope.locals[name] = binder
        scope.steps.append((binder, computation))
        return binder

    def _fold_steps(self, scope: _Scope, tail: Computation) -> Computation:
        """Nest a scope's binds around a final computation.

        Parameters
        ----------
        scope : _Scope
            The scope.
        tail : Computation
            What runs after the last step.

        Returns
        -------
        Computation
            The nested binds.
        """
        body = tail
        for binder, computation in reversed(scope.steps):
            body = Bind(binder, computation, body)
        return body

    def _elaborate_sample(
        self: _Elaborator, step: SampleStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``sample x <- Family(args)`` to a ``Random.sample``.

        Parameters
        ----------
        step : SampleStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the step destructures a family draw, or its distribution
            cannot be elaborated.
        """
        if step.morphism in self._program_declarations:
            self._elaborate_program_draw(step, scope, state)
            return
        if len(step.vars) != 1:
            self._fail(
                step,
                "a sample step destructures the value of a program; a family "
                "draw binds one name",
                code="qiec-program",
            )
        name = step.vars[0]
        state.current_site = name
        state.pending_alphabet = None
        distribution = self._step_distribution(step, scope, state, name)
        if state.pending_alphabet is not None:
            state.alphabets[name] = state.pending_alphabet
        sampled = self._sampled_type(distribution)
        state.uses_random = True
        request = self._sample_request(
            state.random, name, sampled, distribution, step, state
        )
        state.sites.append(
            ProgramSite(
                name,
                "sample",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        self._bind_step(scope, name, sampled, Perform(request), step)

    def _elaborate_program_draw(
        self: _Elaborator, step: SampleStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``sample x <- sub(args)`` by running ``sub`` in place.

        The program's steps join this program's under names of its own:
        each local ``z`` of ``sub`` becomes ``x$z``, the name ``sub``
        returns becomes ``x`` itself, and a pattern ``(u, v)`` takes the
        returned names positionally, as
        [`instantiate_program`][quivers.dsl.program_templates.instantiate_program]
        states. A template's object parameters take the objects the
        draw names and its scalar parameters the numbers or bound names,
        so every draw of a program contributes its own sites.

        Parameters
        ----------
        step : SampleStep
            The draw.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the arguments do not fit the program's parameters, or the
            pattern's arity is not what it returns.
        """
        self._inline_program(
            step.morphism, step.vars, step.args or (), step, scope, state
        )

    def _inline_program(
        self: _Elaborator,
        name: str,
        binders: tuple[str, ...],
        arguments: tuple[DrawArg, ...],
        node: SampleStep | CallStep,
        scope: _Scope,
        state: _ProgramState,
    ) -> None:
        """Elaborate a program's steps in this program under a pattern.

        Parameters
        ----------
        name : str
            The program drawn from.
        binders : tuple[str, ...]
            The draw's pattern.
        arguments : tuple[DrawArg, ...]
            The draw's arguments, filling the program's template
            parameters.
        node : SampleStep | CallStep
            The source step, for diagnostics.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the arguments do not fit the program's parameters or the
            pattern's arity is not what it returns.
        """
        template = self._program_declarations[name]
        bound = {local: LetExprVar(name=local) for local in scope.locals}
        try:
            bindings = template_bindings(template, arguments, bound)
            steps = instantiate_program(template, binders, bindings)
        except TemplateError as error:
            self._fail(node, error.message, code="qiec-program")
        for parameter, argument in bindings.objects.items():
            if argument not in self._program_objects:
                self._fail(
                    node,
                    f"object {argument!r} passed as {parameter!r} of program "
                    f"{name!r} is not declared",
                    code="qiec-program",
                )
        for parameter, argument in bindings.morphisms.items():
            if (
                argument not in self._program_morphisms
                and argument not in self._program_declarations
            ):
                self._fail(
                    node,
                    f"morphism {argument!r} passed as {parameter!r} of program "
                    f"{name!r} is not declared",
                    code="qiec-program",
                )
        self._elaborate_steps(steps, scope, state)

    def _sample_request(
        self: _Elaborator,
        instance: NamedEffectInstance,
        label: str,
        sampled: TypeExpr,
        distribution: DistributionValue,
        node: object,
        state: _ProgramState,
    ) -> EffectRequest:
        """Build a ``Random.sample`` request on an instance.

        Parameters
        ----------
        instance : NamedEffectInstance
            The ``Random`` instance addressed.
        label : str
            The site's label.
        sampled : TypeExpr
            The sampled value's type.
        distribution : DistributionValue
            The distribution.
        node : object
            The source node.
        state : _ProgramState
            The program's accumulating state, for the site path.

        Returns
        -------
        EffectRequest
            The request, instantiated at the sampled type.
        """
        random = self._effect("Random")
        assert random is not None
        operation = next(item for item in random.operations if item.name == "sample")
        outer = instantiate_telescope(random.telescope, instance.entry.effect.arguments)
        parameter_types, result_type = instantiate_operation(
            operation, (sampled,), outer
        )
        del parameter_types
        path = ("programs", state.declaration.name, "sites", label)
        return EffectRequest(
            instance.entry.instance,
            instance.entry.effect,
            operation.id,
            (sampled,),
            (SiteValue(label, site_type(sampled)), distribution),
            result_type,
            SiteProvenance(_origin_at(self, path, "effect-request", node)),
        )

    def _elaborate_observe(
        self: _Elaborator, step: ObserveStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``observe y <- Family(args)`` to a scored log density.

        Outside a marginalization scope the density's total goes to the
        canonical ``score`` instance. Inside one it goes to the scope's
        weight instance: as the weight vector over the group axis when
        the observation is plated over it, segmented by the ``via``
        index map when the observation rows map to groups, or as its
        total in an ungrouped scope.

        Parameters
        ----------
        step : ObserveStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the step destructures a tuple, uses ``via`` outside a
            grouped scope, or its weights cannot be brought to the
            scope's group axis.
        """
        if len(step.vars) != 1:
            self._fail(
                step,
                "an observe step names one observation; tuple destructuring has "
                "no elaboration",
                code="qiec-program",
            )
        name = step.vars[0]
        group = _enclosing_group(scope)
        via = _index_map_names(step)
        if via is not None and group is None:
            self._fail(
                step,
                f"observation {name!r} names the index map {'*'.join(via)!r} "
                "outside a grouped marginalization",
                code="qiec-program",
            )
        state.current_site = name
        state.pending_alphabet = None
        distribution = self._step_distribution(
            step, scope, state, name, via=via, group=group
        )
        observed_type = self._sampled_type(distribution)
        if state.markov_row is not None and name == state.markov_entry:
            # The row of a marginalized chain's cell scores the entry it
            # reads as the observation its next state emits.
            entry = scope.lookup(name)
            assert entry is not None
            if entry.type != observed_type:
                self._fail(
                    step,
                    f"the cell observes its entry {name!r} of type "
                    f"{render_static(entry.type)} from a family over "
                    f"{render_static(observed_type)}",
                    code="qiec-program",
                )
            observed = entry
        else:
            observed = self._declare_parameter(
                name,
                observed_type,
                "observation",
                scope,
                state,
                step,
                axes=tuple(
                    axis.name
                    for axis in (*distribution.plate.batch, *distribution.plate.event)
                ),
            )
        state.sites.append(
            ProgramSite(
                name,
                "observe",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        origin = _origin_at(
            self,
            ("programs", state.declaration.name, "sites", name),
            "log-density",
            step,
        )
        weight_scope = _weight_scope(scope)
        if weight_scope is None:
            weight: Value = WeightSum(
                LogDensity(
                    distribution, Var(observed), origin, distribution.plate.batch
                )
            )
            self._add_score(state.score, weight, scope, state, step)
            return
        assert weight_scope.weight_type is not None
        if weight_scope.group is None:
            weight = WeightSum(
                LogDensity(
                    distribution, Var(observed), origin, distribution.plate.batch
                )
            )
        else:
            batch = distribution.plate.batch
            density = LogDensity(distribution, Var(observed), origin, batch)
            if via is not None:
                if len(batch) != 1:
                    self._fail(
                        step,
                        f"observation {name!r} is grouped by {'*'.join(via)!r} "
                        "but is not plated over one row axis",
                        code="qiec-program",
                    )
                index_map = self._index_map(
                    via, batch[0].size, batch[0].name, weight_scope, scope, state, step
                )
                weight = SegmentSum(
                    density,
                    index_map,
                    weight_scope.group.size,
                    tensor_type(LOG_WEIGHT, (weight_scope.group.size,)),
                )
            elif len(batch) == 1 and batch[0].size == weight_scope.group.size:
                weight = density
            elif not batch:
                weight = density
            else:
                self._fail(
                    step,
                    f"observation {name!r} inside the marginalization over "
                    f"{weight_scope.group.name!r} is plated over neither that "
                    "axis nor an index map into it",
                    code="qiec-program",
                )
        assert weight_scope.weight_instance is not None
        self._add_weight(weight_scope.weight_instance, weight, scope, state, step)

    def _add_score(
        self: _Elaborator,
        instance: NamedEffectInstance,
        weight: Value,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> None:
        """Append a ``Score.add`` of a weight to a scope.

        Parameters
        ----------
        instance : NamedEffectInstance
            The ``Score`` instance.
        weight : Value
            The ``LogWeight`` added.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        """
        state.uses_score = True
        score = self._effect("Score")
        assert score is not None
        operation = next(item for item in score.operations if item.name == "add")
        request = EffectRequest(
            instance.entry.instance,
            instance.entry.effect,
            operation.id,
            (),
            (weight,),
            UNIT,
            SiteProvenance(
                _origin_at(
                    self,
                    ("programs", state.declaration.name, "scores", len(scope.steps)),
                    "effect-request",
                    node,
                )
            ),
        )
        self._bind_step(scope, None, UNIT, Perform(request), node)

    def _add_weight(
        self: _Elaborator,
        instance: NamedEffectInstance,
        weight: Value,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> None:
        """Append a ``Weight.add`` of a weight to a marginalization scope.

        Parameters
        ----------
        instance : NamedEffectInstance
            The scope's ``Weight`` instance.
        weight : Value
            The weight added, of the instance's weight type.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        """
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        operation = next(
            item for item in weight_effect.operations if item.name == "add"
        )
        outer = instantiate_telescope(
            weight_effect.telescope, instance.entry.effect.arguments
        )
        _parameter_types, result_type = instantiate_operation(operation, (), outer)
        request = EffectRequest(
            instance.entry.instance,
            instance.entry.effect,
            operation.id,
            (),
            (weight,),
            result_type,
            SiteProvenance(
                _origin_at(
                    self,
                    ("programs", state.declaration.name, "weights", len(scope.steps)),
                    "effect-request",
                    node,
                )
            ),
        )
        self._bind_step(scope, None, UNIT, Perform(request), node)
        owner = _weight_scope(scope)
        assert owner is not None
        owner.weights_added = True

    def _elaborate_let(
        self: _Elaborator, step: LetStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``let name = expr`` to a pure binding.

        Parameters
        ----------
        step : LetStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        """
        if isinstance(step.value, LetExprLambda):
            if scope.lookup(step.name) is not None or step.name in self._lambda_macros:
                self._fail(
                    step,
                    f"variable {step.name!r} already bound in program",
                    code="qiec-program",
                )
            self._lambda_macros[step.name] = step.value
            return
        deduction = self._parse_call(step.value, step)
        if deduction is not None:
            call = self._deduction_call(deduction, scope, state, step)
            self._bind_step(scope, step.name, call.result_type, call, step)
            return
        scan = scan_let_call(step.value)
        if scan is not None and scan.marginal:
            _elaborate_marginal_chain(self, scan, scope, state, step)
            return
        if scan is not None:
            call = self._scan_call(scan, scope, state, step)
            self._bind_step(scope, step.name, call.result_type, call, step)
            return
        paired = positions_let_call(step.value)
        if paired is not None:
            value = _positions_value(self, paired, scope, step)
            self._bind_step(scope, step.name, value.result_type, Return(value), step)
            return
        value = self._let_value(step.value, scope, state, step)
        type_ = self._value_type(value, scope.context(), step.value)
        self._bind_step(scope, step.name, type_, Return(value), step)
        self._record_batch(step.name, step.value, scope, state)

    def _parse_call(
        self: _Elaborator, expr: LetExprNode, node: object
    ) -> _ParseCall | None:
        """Read ``parse(D, x[, Plate][, ("A", p)...])``.

        Parameters
        ----------
        expr : LetExprNode
            The expression.
        node : object
            The step, blamed for a malformed call.

        Returns
        -------
        _ParseCall | None
            The call's parts when the expression applies ``parse`` to a
            declared deduction, else ``None``.

        Raises
        ------
        QiecDiagnosticError
            If a call of ``parse`` on a deduction does not name its input
            by a variable, or an argument after the input and the
            optional plate is not a pair of a string label and a value.
        """
        if not isinstance(expr, LetExprCall) or expr.func != "parse":
            return None
        if len(expr.args) < 2 or not isinstance(expr.args[0], LetExprVar):
            return None
        target = expr.args[0]
        if target.name not in getattr(self, "_deductions", {}):
            return None
        source = expr.args[1]
        if not isinstance(source, LetExprVar):
            self._fail(
                node,
                f"parse reads the input of {target.name!r} from a program input "
                "named by a variable",
                code="qiec-program",
            )
        rest = list(expr.args[2:])
        plate: str | None = None
        if rest and isinstance(rest[0], LetExprVar):
            plate = rest[0].name
            rest = rest[1:]
        bindings: list[_ParseBinding] = []
        for argument in rest:
            if not (
                isinstance(argument, LetExprTuple)
                and len(argument.items) == 2
                and isinstance(argument.items[0], LetExprString)
            ):
                self._fail(
                    argument,
                    "parse binds production probabilities as "
                    '("<left-hand side>", <probabilities>) pairs after its input '
                    "and optional plate",
                    code="qiec-program",
                )
            label = argument.items[0]
            assert isinstance(label, LetExprString)
            bindings.append(
                _ParseBinding(
                    label=label.value,
                    value=argument.items[1],
                    line=argument.line,
                    col=argument.col,
                )
            )
        return _ParseCall(
            deduction=target.name,
            source=source.name,
            plate=plate,
            bindings=tuple(bindings),
        )

    def _deduction_call(
        self: _Elaborator,
        parse: _ParseCall,
        scope: _Scope,
        state: _ProgramState,
        node: LetStep,
    ) -> Call:
        """Call a deduction's entry on a sentence or a corpus the program takes.

        The input is a program input of tokens: a sentence, whose length
        is one of the program's open extents, or with a plate, a corpus of
        one sentence per object of the plate, all of one open length,
        whose answer is the semiring product of the sentences' inside
        weights (their joint weight, the sentences being independent).
        Without bindings the deduction reads its learned weights through
        the module's ``params`` instance, which the program's row gains.
        With bindings, the deduction must be normalized per left-hand
        side and each binding gives one left-hand side's production
        probabilities: the call goes to the deduction's weighted variant
        with the table of the probabilities' logarithms.

        Parameters
        ----------
        parse : _ParseCall
            The call's parts.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : LetStep
            The step.

        Returns
        -------
        Call
            The call.

        Raises
        ------
        QiecDiagnosticError
            If the deduction takes its axioms rather than a sentence, the
            input is bound by a step, the plate names no object with a
            static extent, or the bindings do not fit the deduction (see
            `_production_table`).
        """
        name, source = parse.deduction, parse.source
        signature = self.computation_signatures[f"{name}__run"]
        if len(signature.parameters) != 1:
            self._fail(
                node,
                f"deduction {name!r} takes its axioms and their weights, not a "
                "sentence; a program supplies items to it through a computation",
                code="qiec-program",
            )
        if scope.lookup(source) is not None:
            self._fail(
                node,
                f"parse reads {source!r}, which a step binds; the sentence a "
                "deduction parses is a program input",
                code="qiec-program",
            )
        table: TensorValue | None = None
        if parse.bindings:
            table = self._production_table(parse, scope, state, node)
            signature = self.computation_signatures[f"{name}__weighted_run"]
        binder = state.extents.get(f"{source}_extent")
        if binder is None:
            binder = IndexBinder(f"{source}_extent", NAT)
            state.extents[binder.name] = binder
        extent = IndexVariable(binder.name, binder.sort)
        sentence_type = tensor_type(STRING, (extent,))
        rows = None if parse.plate is None else self._axis(parse.plate, node).size
        tokens = self._declare_parameter(
            source,
            sentence_type if rows is None else tensor_type(STRING, (rows, extent)),
            "data",
            scope,
            state,
            node,
        )
        params = self.instances.get(PARAMS_INSTANCE)
        if params is not None and any(
            entry.instance == params.entry.instance
            for entry in signature.effects.entries
        ):
            state.uses_params = True
        origin = cast(
            SourceOrigin,
            _origin_at(
                self,
                ("programs", state.declaration.name, "parses", name),
                "call",
                node,
            ),
        )
        extra = () if table is None else (table,)
        if rows is None:
            return Call(
                signature.id,
                signature.name,
                (extent,),
                (Var(tokens), *extra),
                signature.result,
                signature.effects,
                origin,
            )
        return self._corpus_call(
            signature, tokens, rows, extent, extra, state, node, origin
        )

    def _production_table(
        self: _Elaborator,
        parse: _ParseCall,
        scope: _Scope,
        state: _ProgramState,
        node: LetStep,
    ) -> TensorValue:
        """The table of production log-weights a call's bindings give.

        Each binding names a left-hand side of several productions and a
        probability vector over them, in the order
        `DeductionSystem.production_groups` lists; the table holds each
        entry's log-probability once the vector is renormalized,
        ``log p_q - log sum p``, groups in order. A value drawn at a
        sample site must come from a family whose support is the simplex.

        Parameters
        ----------
        parse : _ParseCall
            The call's parts.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : LetStep
            The step.

        Returns
        -------
        Value
            The table, a vector of reals: log-probabilities.

        Raises
        ------
        QiecDiagnosticError
            If the deduction is not normalized per left-hand side, a
            label names no left-hand side of several productions or is
            bound twice, a left-hand side of several productions is not
            bound, a value is not a vector of reals of its group's size,
            or a sample site's family's support is not the simplex.
        """
        system = self._deductions[parse.deduction]
        if not system.normalized:
            self._fail(
                node,
                f"parse binds production probabilities to {parse.deduction!r}, "
                "whose weights are free log-weights; probabilities replace the "
                "weights of a grammar normalized per left-hand side, so "
                "declare normalize=lhs",
                code="qiec-program",
            )
        bound = {
            group.label: group for group in system.groups if len(group.productions) > 1
        }
        values: dict[str, Value] = {}
        for binding in parse.bindings:
            group = bound.get(binding.label)
            if group is None:
                single = any(item.label == binding.label for item in system.groups)
                self._fail(
                    binding,
                    f"parse binds {binding.label!r}, "
                    + (
                        f"which has one production in {parse.deduction!r}, whose "
                        "probability is one"
                        if single
                        else f"which is no left-hand side of {parse.deduction!r}"
                    )
                    + f"; the left-hand sides of several productions are "
                    f"{', '.join(bound)}",
                    code="qiec-program",
                )
            if binding.label in values:
                self._fail(
                    binding,
                    f"parse binds {binding.label!r} twice",
                    code="qiec-program",
                )
            size = len(group.productions)
            value = self._let_value(binding.value, scope, state, node)
            type_ = self._value_type(value, scope.context(), binding)
            if type_ != tensor_type(REAL, (IndexLiteral(size, NAT),)):
                self._fail(
                    binding,
                    f"parse binds {binding.label!r} to a value of type "
                    f"{render_static(type_)}, but {binding.label!r} has {size} "
                    f"productions ({', '.join(item.label for item in group.productions)}) "
                    f"and takes a probability vector of type "
                    f"{render_static(tensor_type(REAL, (IndexLiteral(size, NAT),)))}",
                    code="qiec-program",
                )
            support = _sampled_support(binding.value, state.declaration)
            if support is not None and support[1] != "simplex":
                self._fail(
                    binding,
                    f"parse binds {binding.label!r} to {support[0]!r}, drawn from a "
                    f"family whose support is {support[1]}; production "
                    "probabilities lie on the simplex, as a Dirichlet draw does",
                    code="qiec-program",
                )
            values[binding.label] = value
        missing = [label for label in bound if label not in values]
        if missing:
            self._fail(
                node,
                f"parse binds no probabilities for {', '.join(missing)} of "
                f"{parse.deduction!r}; every left-hand side of several "
                "productions takes one",
                code="qiec-program",
            )
        entries: list[Value] = []
        path = ("programs", state.declaration.name, "parses", parse.deduction)
        for label, group in bound.items():
            size = len(group.productions)
            probabilities = [
                Gather(values[label], LiteralValue(position, INT), REAL)
                for position in range(size)
            ]
            total: Value = probabilities[0]
            for item in probabilities[1:]:
                total = self._primitive(
                    "add_real", (total, item), node, (*path, "table", label, "total")
                )
            normalizer = self._primitive(
                "log", (total,), node, (*path, "table", label, "normalizer")
            )
            for position, item in enumerate(probabilities):
                entries.append(
                    self._primitive(
                        "sub_real",
                        (
                            self._primitive(
                                "log", (item,), node, (*path, "table", label, position)
                            ),
                            normalizer,
                        ),
                        node,
                        (*path, "table", label, position, "normalized"),
                    )
                )
        return TensorValue(
            tuple(entries), tensor_type(REAL, (IndexLiteral(len(entries), NAT),))
        )

    def _corpus_call(
        self: _Elaborator,
        signature: ComputationSignature,
        tokens: Local,
        rows: IndexTerm,
        extent: IndexVariable,
        extra: tuple[TensorValue, ...],
        state: _ProgramState,
        node: LetStep,
        origin: SourceOrigin,
    ) -> Call:
        """Call a deduction's entry on every sentence of a corpus.

        A helper computation recurs over the corpus's rows, calls the
        entry on each sentence, and multiplies the answers in the
        semiring, starting from its one.

        Parameters
        ----------
        signature : ComputationSignature
            The entry's signature.
        tokens : Local
            The corpus, a table of tokens with one sentence per row.
        rows : IndexTerm
            The number of sentences.
        extent : IndexVariable
            The sentences' length.
        extra : tuple[Value, ...]
            The production table the entry takes after the sentence, if
            any.
        state : _ProgramState
            The program's accumulating state, which gains the helper.
        node : LetStep
            The step.
        origin : SourceOrigin
            The call's origin.

        Returns
        -------
        Call
            The call of the helper at the first row.

        Raises
        ------
        QiecDiagnosticError
            If the helper is declared twice.
        """
        carrier = signature.result
        name = f"{state.declaration.name}__{node.name}_corpus"
        if name in self.computation_signatures:
            self._fail(
                node,
                f"corpus helper {name!r} is declared twice",
                code="qiec-program",
            )
        path = ("programs", state.declaration.name, "corpora", node.name)
        position = Local("__corpus_r", INT)
        corpus = Local("__corpus_tokens", tokens.type)
        total = Local("__corpus_total", carrier)
        carried = tuple(
            Local(f"__corpus_table_{index}", value.result_type)
            for index, value in enumerate(extra)
        )
        telescope = (IndexBinder(extent.name, extent.sort),)
        identity = ComputationId.derive(self.source.module_name, "computation", name)
        helper = ComputationSignature(
            identity,
            name,
            telescope,
            (INT, tokens.type, carrier, *(item.type for item in carried)),
            carrier,
            signature.effects,
        )
        try:
            self.registry.register_computation(helper)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(node, error, fallback="qiec-program")
        self.computation_signatures[name] = helper
        statics: tuple[StaticArgument, ...] = (extent,)
        counter = Local("__corpus_k", INT)
        count: Value = Reduction(
            "sum",
            Comprehension(
                counter, rows, LiteralValue(1, INT), tensor_type(INT, (rows,))
            ),
            INT,
        )
        sentence = Local("__corpus_sentence", tensor_type(STRING, (extent,)))
        weight = Local("__corpus_weight", carrier)
        product = self._semiring_times(Var(total), Var(weight), carrier, node, path)
        recur = Call(
            identity,
            name,
            statics,
            (
                self._primitive(
                    "add_int",
                    (Var(position), LiteralValue(1, INT)),
                    node,
                    (*path, "next"),
                ),
                Var(corpus),
                product,
                *(Var(item) for item in carried),
            ),
            carrier,
            signature.effects,
            origin,
        )
        body: Computation = If(
            self._primitive(
                "eq_int", (Var(position), count), node, (*path, "finished")
            ),
            Return(Var(total)),
            Bind(
                sentence,
                Return(Gather(Var(corpus), Var(position), sentence.type)),
                Bind(
                    weight,
                    Call(
                        signature.id,
                        signature.name,
                        (extent,),
                        (Var(sentence), *(Var(item) for item in carried)),
                        carrier,
                        signature.effects,
                        cast(
                            SourceOrigin,
                            _origin_at(self, (*path, "sentence"), "call", node),
                        ),
                    ),
                    recur,
                ),
            ),
        )
        state.helpers.append(
            self._checked_computation(
                helper, (position, corpus, total, *carried), body, node, path
            )
        )
        return Call(
            identity,
            name,
            statics,
            (
                LiteralValue(0, INT),
                Var(tokens),
                self._semiring_one(carrier, node, path),
                *extra,
            ),
            carrier,
            signature.effects,
            origin,
        )

    def _semiring_one(
        self: _Elaborator, carrier: TypeExpr, node: object, path: tuple[str | int, ...]
    ) -> Value:
        """The multiplicative identity of a deduction's carrier.

        Parameters
        ----------
        carrier : TypeExpr
            ``LogWeight``, ``Bool``, or ``Int``.
        node : object
            The source node.
        path : tuple[str | int, ...]
            The origin path.

        Returns
        -------
        Value
            Log-weight zero, ``true``, or one.
        """
        if carrier == LOG_WEIGHT:
            return self._primitive(
                "as_weight", (LiteralValue(0.0, REAL),), node, (*path, "one")
            )
        if carrier == BOOL:
            return LiteralValue(True, BOOL)
        return LiteralValue(1, INT)

    def _semiring_times(
        self: _Elaborator,
        left: Value,
        right: Value,
        carrier: TypeExpr,
        node: object,
        path: tuple[str | int, ...],
    ) -> Value:
        """The semiring product of two carrier values.

        Parameters
        ----------
        left : Value
            A value of the carrier.
        right : Value
            A value of the carrier.
        carrier : TypeExpr
            ``LogWeight``, ``Bool``, or ``Int``.
        node : object
            The source node.
        path : tuple[str | int, ...]
            The origin path.

        Returns
        -------
        Value
            Their product: the sum of log-weights, the conjunction, or
            the integer product.
        """
        operation = (
            "add_weight"
            if carrier == LOG_WEIGHT
            else "and"
            if carrier == BOOL
            else "mul_int"
        )
        return self._primitive(operation, (left, right), node, (*path, "times"))

    def _check_scan_site_labels(
        self: _Elaborator,
        step_name: str,
        state: _ProgramState,
        node: LetStep,
        written: str,
    ) -> None:
        """Refuse a scan whose step shares a site label with another site.

        A site the step reaches is addressed by its label and its
        occurrence, ``"<label>@<n>"``, counted over the run. A label
        the scanning program also draws or observes under, or that
        another scan or call of the program reaches, would number that
        site among the step's occurrences, so ``"<label>@0"`` would not
        name the step's first position. The labels the expansion of a
        scan synthesizes derive from the program's own bindings and
        cannot collide; a program cell's own labels can.

        Parameters
        ----------
        step_name : str
            The step program's name.
        state : _ProgramState
            The scanning program's accumulating state.
        node : LetStep
            The scan's step.
        written : str
            The step program the source's scan names, which ``step_name``
            specializes when the cell reads bindings of the scanning
            program; it is the same scan, not another one.

        Raises
        ------
        QiecDiagnosticError
            If a label is shared.
        """
        reached = reached_site_labels(step_name, self._program_declarations)
        others: list[tuple[str, frozenset[str]]] = [
            (
                f"the program {state.declaration.name!r}",
                _own_site_labels(state.declaration.draws),
            )
        ]
        others.extend(
            (
                f"the step {callee!r}",
                reached_site_labels(callee, self._program_declarations),
            )
            for callee in _callees(state.declaration.draws)
            if callee not in (step_name, written)
        )
        for owner, labels in others:
            shared = sorted(reached & labels)
            if shared:
                self._fail(
                    node,
                    f"the scan step {step_name!r} reaches site "
                    f"{', '.join(repr(label) for label in shared)} at every "
                    f"position, and {owner} reaches a site under the same "
                    f"label, so the two would share addresses; rename one "
                    f"of them",
                    code="qiec-program",
                )

    def _scan_call(
        self: _Elaborator,
        scan: ScanCall,
        scope: _Scope,
        state: _ProgramState,
        node: LetStep,
    ) -> Call:
        """Elaborate ``let h = scan(step, xs)`` to a call of a recurrence.

        The recurrence is a helper computation over the sequence's
        positions: after visiting ``k`` positions it reads the entry at
        position ``p`` (``k`` itself, or ``length - 1 - k`` for a
        reversed scan), calls the step program with the entry and the
        state, and recurs at ``k + 1`` with the step's result, answering
        the state once ``k`` reaches the sequence's length. A scan with
        ``outputs=all`` also carries the states so far, a tensor over
        the positions whose entry at ``p`` the step's result replaces,
        and answers it: every state, in input order. The initial state
        is zero, or a learned state read as a ``weight`` input when the
        scan names one. A sequence that is a program input was typed
        over an open extent when the program's parameters were
        declared, an entry per position of the domain factor it reads.

        Parameters
        ----------
        scan : ScanCall
            The step program's name, the sequence's name, the initial
            state's name when there is one, and the scan's options.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state, which gains the helper.
        node : LetStep
            The step.

        Returns
        -------
        Call
            The call of the recurrence at position zero.

        Raises
        ------
        QiecDiagnosticError
            If the step program is not a program of two parameters
            returning its state, or the sequence is neither a program
            input nor a tensor with a leading axis of entries the step
            reads.
        """
        written, source, init = scan.step, scan.sequence, scan.init
        step_name = _specialized_scan_step(self, written, scope, state, node)
        signature = self.computation_signatures.get(step_name)
        step_entry = next(
            (item for item in self.entries if item.name == step_name), None
        )
        if (
            signature is None
            or step_entry is None
            or len(signature.parameters) < 2
            or signature.parameters[1] != signature.result
            or [parameter.role for parameter in step_entry.parameters[:2]]
            != ["domain", "domain"]
        ):
            self._fail(
                node,
                f"scan steps with {step_name!r}, which is not a program of an "
                "input and a state returning the state",
                code="qiec-program",
            )
        entry_type, state_type = signature.parameters[:2]
        if _index_variables_in((entry_type, state_type)):
            self._fail(
                node,
                f"scan steps with {step_name!r}, whose input or state has an "
                "open extent",
                code="qiec-program",
            )
        local = scope.lookup(source)
        if local is None:
            self._fail(
                node,
                f"scan runs over {source!r}, which is neither bound nor a "
                "program input",
                code="qiec-program",
            )
        shape = tensor_shape(local.type)
        if shape is None or not shape[1]:
            self._fail(
                node,
                f"scan runs over {source!r}, which is not a sequence",
                code="qiec-program",
            )
        extent = shape[1][0]
        per_position: TypeExpr = (
            shape[0] if len(shape[1]) == 1 else tensor_type(shape[0], shape[1][1:])
        )
        if per_position != entry_type:
            self._fail(
                node,
                f"scan feeds {step_name!r} entries of {render_static(per_position)}, "
                f"not the {render_static(entry_type)} it reads",
                code="qiec-program",
            )
        self._check_scan_site_labels(step_name, state, node, written)
        passed, positional, step_statics, substitution = _scan_step_inputs(
            self, step_name, step_entry, signature, extent, scope, state, node
        )
        name = f"{state.declaration.name}__{node.name}_scan"
        if name in self.computation_signatures:
            self._fail(
                node,
                f"scan helper {name!r} is declared twice",
                code="qiec-program",
            )
        mentioned = _index_variables_in((local.type, *(item.type for item in passed)))
        telescope = tuple(
            binder for name_, binder in state.extents.items() if name_ in mentioned
        )
        statics: tuple[StaticArgument, ...] = tuple(
            IndexVariable(binder.name, binder.sort) for binder in telescope
        )
        visited = Local("__scan_t", INT)
        sequence = Local("__scan_xs", local.type)
        current = Local("__scan_h", state_type)
        gathers = scan.outputs == "all"
        state_shape = tensor_shape(state_type)
        states_type: TypeExpr = (
            tensor_type(state_type, (extent,))
            if state_shape is None
            else tensor_type(state_shape[0], (extent, *state_shape[1]))
        )
        result_type = states_type if gathers else state_type
        states = Local("__scan_states", states_type)
        gathered = (states,) if gathers else ()
        carried = tuple(Local(f"__scan_{item.name}", item.type) for item in passed)
        identity = ComputationId.derive(self.source.module_name, "computation", name)
        helper = ComputationSignature(
            identity,
            name,
            telescope,
            (
                INT,
                local.type,
                state_type,
                *(item.type for item in gathered),
                *(item.type for item in carried),
            ),
            result_type,
            signature.effects,
        )
        try:
            self.registry.register_computation(helper)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(node, error, fallback="qiec-program")
        self.computation_signatures[name] = helper
        path = ("programs", state.declaration.name, "scans", node.name)
        counter = Local("__scan_k", INT)
        length: Value = Reduction(
            "sum",
            Comprehension(
                counter, extent, LiteralValue(1, INT), tensor_type(INT, (extent,))
            ),
            INT,
        )
        # The position the scan visits after ``visited`` others: the same
        # number left to right, that many from the end right to left.
        position = Local("__scan_p", INT)
        position_value: Value = (
            self._primitive(
                "sub_int",
                (
                    self._primitive(
                        "sub_int",
                        (length, LiteralValue(1, INT)),
                        node,
                        (*path, "last"),
                    ),
                    Var(visited),
                ),
                node,
                (*path, "position"),
            )
            if scan.reverse
            else Var(visited)
        )
        entry = Local("__scan_x", entry_type)
        following = Local("__scan_next", state_type)
        following_states = Local("__scan_states_next", states_type)
        origin = _origin_at(self, path, "call", node)
        recur = Call(
            identity,
            name,
            statics,
            (
                self._primitive(
                    "add_int",
                    (Var(visited), LiteralValue(1, INT)),
                    node,
                    (*path, "successor"),
                ),
                Var(sequence),
                Var(following),
                *((Var(following_states),) if gathers else ()),
                *(Var(item) for item in carried),
            ),
            result_type,
            signature.effects,
            origin,
        )
        continuation: Computation = recur
        if gathers:
            # The states after this position replace the states' entry at
            # it: at each index ``j`` the entry is chosen from the pair
            # (old entry, new state) by ``max(0, 1 - |j - p|)``, which is
            # one at ``p`` and zero elsewhere.
            index = Local("__scan_j", INT)
            pair_type = (
                tensor_type(state_type, (IndexLiteral(2, NAT),))
                if state_shape is None
                else tensor_type(
                    state_shape[0], (IndexLiteral(2, NAT), *state_shape[1])
                )
            )
            here = self._primitive(
                "max_int",
                (
                    LiteralValue(0, INT),
                    self._primitive(
                        "sub_int",
                        (
                            LiteralValue(1, INT),
                            self._primitive(
                                "abs_int",
                                (
                                    self._primitive(
                                        "sub_int",
                                        (Var(index), Var(position)),
                                        node,
                                        (*path, "offset"),
                                    ),
                                ),
                                node,
                                (*path, "distance"),
                            ),
                        ),
                        node,
                        (*path, "nearness"),
                    ),
                ),
                node,
                (*path, "here"),
            )
            continuation = Bind(
                following_states,
                Return(
                    Comprehension(
                        index,
                        extent,
                        Gather(
                            TensorValue(
                                (
                                    Gather(Var(states), Var(index), state_type),
                                    Var(following),
                                ),
                                pair_type,
                            ),
                            here,
                            state_type,
                        ),
                        states_type,
                    )
                ),
                recur,
            )
        body: Computation = If(
            self._primitive(
                "eq_int", (Var(visited), length), node, (*path, "finished")
            ),
            Return(Var(states if gathers else current)),
            Bind(
                position,
                Return(position_value),
                Bind(
                    entry,
                    Return(Gather(Var(sequence), Var(position), entry_type)),
                    Bind(
                        following,
                        Call(
                            signature.id,
                            signature.name,
                            step_statics,
                            (
                                Var(entry),
                                Var(current),
                                *(
                                    Gather(
                                        Var(item),
                                        Var(position),
                                        substitute_type(parameter.type, substitution),
                                    )
                                    if per_step
                                    else Var(item)
                                    for item, per_step, parameter in zip(
                                        carried,
                                        positional,
                                        step_entry.parameters[2:],
                                        strict=True,
                                    )
                                ),
                            ),
                            state_type,
                            signature.effects,
                            _origin_at(self, (*path, "step"), "call", node),
                        ),
                        continuation,
                    ),
                ),
            ),
        )
        state.helpers.append(
            self._checked_computation(
                helper,
                (visited, sequence, current, *gathered, *carried),
                body,
                node,
                path,
            )
        )
        params = self.instances.get(PARAMS_INSTANCE)
        assert state.random is not None and state.score is not None
        for row_entry in signature.effects.entries:
            if row_entry.instance == state.random.entry.instance:
                state.uses_random = True
            elif row_entry.instance == state.score.entry.instance:
                state.uses_score = True
            elif params is not None and row_entry.instance == params.entry.instance:
                state.uses_params = True
        initial: Value = _zero_of(state_type)
        if init is not None:
            initial = _scan_initial_state(self, init, state_type, scope, state, node)
        zero_states: tuple[Value, ...] = (
            (
                Comprehension(
                    Local("__scan_zero_j", INT),
                    extent,
                    _zero_of(state_type),
                    states_type,
                ),
            )
            if gathers
            else ()
        )
        return Call(
            identity,
            name,
            statics,
            (
                LiteralValue(0, INT),
                Var(local),
                initial,
                *zero_states,
                *(Var(item) for item in passed),
            ),
            result_type,
            signature.effects,
            origin,
        )

    def _domain_axes(
        self: _Elaborator, factor: ObjectExpr, type_: TypeExpr
    ) -> tuple[str, ...]:
        """The axis names of a domain input.

        Parameters
        ----------
        factor : ObjectExpr
            The domain factor the input carries.
        type_ : TypeExpr
            The input's type.

        Returns
        -------
        tuple[str, ...]
            The factor's object name for its width, after the extent's
            variable name when the input is a sequence.
        """
        shape = tensor_shape(type_)
        if shape is None:
            return ()
        names: list[str] = []
        for size in shape[1]:
            if isinstance(size, IndexVariable):
                names.append(size.name)
            elif isinstance(factor, TypeName):
                names.append(factor.name)
            else:
                names.append(f"axis{len(names)}")
        return tuple(names)

    def _sequence_type(
        self: _Elaborator, source: str, entry_type: TypeExpr, state: _ProgramState
    ) -> TypeExpr:
        """The type of a program input read as a sequence.

        Parameters
        ----------
        source : str
            The input's name.
        entry_type : TypeExpr
            The type of one position's entry.
        state : _ProgramState
            The program's accumulating state, which gains the extent.

        Returns
        -------
        TypeExpr
            A tensor with one entry per position of an open extent
            named ``<source>_extent``.
        """
        binder = state.extents.get(f"{source}_extent")
        if binder is None:
            binder = IndexBinder(f"{source}_extent", NAT)
            state.extents[binder.name] = binder
        extent = IndexVariable(binder.name, binder.sort)
        entry_shape = tensor_shape(entry_type)
        if entry_shape is None:
            return tensor_type(entry_type, (extent,))
        return tensor_type(entry_shape[0], (extent, *entry_shape[1]))

    def _elaborate_call(
        self: _Elaborator, step: CallStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``let name <- computation(args)`` to a bound call.

        Parameters
        ----------
        step : CallStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the callee is unknown or the arguments do not fit, or the
            callee's row names an instance other than the program's.
        """
        path = ("programs", state.declaration.name, "calls", step.name)
        signature = self.computation_signatures.get(step.call.callee)
        extents: dict[str, IndexTerm] = {}
        step = self._with_passed_inputs(step, scope, state, extents)
        for position, argument in enumerate(step.call.arguments):
            for name in _free_let_names(argument):
                if scope.lookup(name) is not None or name in self._lambda_macros:
                    continue
                type_ = self._input_type(name, state)
                if (
                    signature is not None
                    and isinstance(argument, LetExprVar)
                    and position < len(signature.parameters)
                ):
                    # A bare name filling a parameter shaped by the callee's
                    # open extents takes that shape, over extents of the
                    # caller's own; two parameters sharing a callee extent
                    # share the caller's, named after the first name.
                    type_ = self._passed_input_type(
                        name, signature.parameters[position], state, extents
                    )
                self._declare_parameter(name, type_, "data", scope, state, step)
        if (
            signature is not None
            and signature.telescope
            and not step.call.static_arguments
        ):
            call = self._call_with_inferred_extents(step, signature, scope, path)
        else:
            call = self._lower_call(step.call, (), scope.context(), path, None)
        try:
            call_type = infer_computation(call, self.registry, scope.context())
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(step.call, error)
        params = self.instances.get(PARAMS_INSTANCE)
        for entry in call_type.effects.entries:
            if entry.instance == state.random.entry.instance:
                state.uses_random = True
            elif entry.instance == state.score.entry.instance:
                state.uses_score = True
            elif params is not None and entry.instance == params.entry.instance:
                state.uses_params = True
            else:
                self._fail(
                    step,
                    f"call to {step.call.callee!r} performs on instance "
                    f"{entry.instance!r}, which the program's row cannot carry",
                    code="qiec-program",
                )
        self._bind_step(scope, step.name, call_type.result, call, step)

    def _with_passed_inputs(
        self: _Elaborator,
        step: CallStep,
        scope: _Scope,
        state: _ProgramState,
        extents: dict[str, IndexTerm],
    ) -> CallStep:
        """Pass a called program's remaining inputs through the caller.

        A program's computation takes every input the program reads:
        its domain, and then its data, observations, index maps, and
        the numbers of its parameter maps. A call that supplies the
        leading parameters alone passes the rest through: each becomes
        an input of the caller under the callee's name and role, and
        fills the callee's parameter.

        Parameters
        ----------
        step : CallStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        extents : dict[str, IndexTerm]
            The callee's index variables tied to extents of this
            program so far, extended in place.

        Returns
        -------
        CallStep
            The step with the passed inputs appended to its arguments;
            the step itself when the callee is not a program or the
            call supplies every parameter.
        """
        entry = next(
            (item for item in self.entries if item.name == step.call.callee), None
        )
        if entry is None or len(step.call.arguments) >= len(entry.parameters):
            return step
        passed = list(step.call.arguments)
        for parameter in entry.parameters[len(step.call.arguments) :]:
            if scope.lookup(parameter.name) is None:
                type_ = self._passed_input_type(
                    parameter.name, parameter.type, state, extents
                )
                self._declare_parameter(
                    parameter.name, type_, parameter.role, scope, state, step
                )
            passed.append(LetExprVar(name=parameter.name))
        return step.with_(call=step.call.with_(arguments=tuple(passed)))

    def _passed_input_type(
        self: _Elaborator,
        name: str,
        parameter: TypeExpr,
        state: _ProgramState,
        extents: dict[str, IndexTerm],
    ) -> TypeExpr:
        """The type of a free name passed straight to a computation.

        Parameters
        ----------
        name : str
            The name.
        parameter : TypeExpr
            The callee's parameter type.
        state : _ProgramState
            The program's accumulating state, which gains one extent
            binder per index variable the parameter type mentions that
            no earlier argument of the call fixed.
        extents : dict[str, IndexTerm]
            The callee's index variables the call has already tied to
            extents of this program, extended in place.

        Returns
        -------
        TypeExpr
            The parameter type with each of the callee's index variables
            replaced by an extent of this program, named after the first
            name that fixed it; the type unchanged when it mentions none.
        """
        variables = sorted(_index_variables_in((parameter,)))
        if not variables:
            return parameter
        fresh = [variable for variable in variables if variable not in extents]
        for position, variable in enumerate(fresh):
            extent_name = (
                f"{name}_extent" if len(fresh) == 1 else f"{name}_extent{position}"
            )
            binder = state.extents.get(extent_name)
            if binder is None:
                binder = IndexBinder(extent_name, NAT)
                state.extents[extent_name] = binder
            extents[variable] = IndexVariable(extent_name, NAT)
        return substitute_type(
            parameter,
            StaticSubstitution(
                indices=tuple((variable, extents[variable]) for variable in variables)
            ),
        )

    def _call_with_inferred_extents(
        self: _Elaborator,
        step: CallStep,
        signature: ComputationSignature,
        scope: _Scope,
        path: tuple[str | int, ...],
    ) -> Call:
        """Call a computation whose index binders the arguments determine.

        A program with open input extents is a computation with index
        binders; a call step writes no static arguments, so each binder
        is read off the type of the argument filling a parameter the
        binder shapes.

        Parameters
        ----------
        step : CallStep
            The step.
        signature : ComputationSignature
            The callee's signature, with a non-empty telescope.
        scope : _Scope
            The scope.
        path : tuple[str | int, ...]
            The structural path of the call.

        Returns
        -------
        Call
            The call with its static arguments instantiated.

        Raises
        ------
        QiecDiagnosticError
            If the argument count differs from the parameter count, a
            binder is not an index binder or shapes no parameter, or the
            arguments do not instantiate the telescope.
        """
        context = scope.context()
        if len(step.call.arguments) != len(signature.parameters):
            self._fail(
                step,
                f"call to {step.call.callee!r} passes {len(step.call.arguments)} "
                f"argument(s); the declaration takes {len(signature.parameters)}",
                code="qiec-program",
            )
        arguments = tuple(
            self._lower_value(value, (), context, None, (*path, position))
            for position, value in enumerate(step.call.arguments)
        )
        bindings: dict[str, IndexTerm] = {}
        for parameter, argument in zip(signature.parameters, arguments, strict=True):
            _bind_extents(
                parameter, self._value_type(argument, context, step), bindings
            )
        statics: list[StaticArgument] = []
        for binder in signature.telescope:
            bound = (
                bindings.get(binder.name) if isinstance(binder, IndexBinder) else None
            )
            if bound is None:
                self._fail(
                    step,
                    f"call to {step.call.callee!r} cannot fix its static "
                    f"{binder.name!r} from the arguments",
                    code="qiec-program",
                )
            statics.append(bound)
        try:
            substitution = instantiate_telescope(signature.telescope, tuple(statics))
        except (TypeError, ValueError) as error:
            self._fail_kernel(step, error, fallback="qiec-program")
        return Call(
            signature.id,
            step.call.callee,
            tuple(statics),
            arguments,
            substitute_type(signature.result, substitution),
            substitute_row(signature.effects, substitution),
            _origin_at(self, path, "call", step),
        )

    def _elaborate_score(
        self: _Elaborator, step: ScoreStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``score name = expr`` to a binding and a ``Score.add``.

        Parameters
        ----------
        step : ScoreStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the expression is not real-valued, or the step lies in a
            grouped marginalization scope.
        """
        value = self._let_value(step.value, scope, state, step)
        type_ = self._value_type(value, scope.context(), step.value)
        if type_ != REAL:
            self._fail(
                step,
                f"score {step.name!r} has type {self._render(type_)}; a score is a "
                "Real",
                code="qiec-program",
            )
        binder = self._bind_step(scope, step.name, REAL, Return(value), step)
        weight = self._primitive(
            "as_weight",
            (Var(binder),),
            step,
            ("programs", state.declaration.name, "scores", step.name),
        )
        weight_scope = _weight_scope(scope)
        if weight_scope is None:
            self._add_score(state.score, weight, scope, state, step)
            return
        if weight_scope.group is not None:
            self._fail(
                step,
                f"score {step.name!r} lies inside a grouped marginalization; a "
                "scalar factor has no group to belong to",
                code="qiec-program",
            )
        assert weight_scope.weight_instance is not None
        self._add_weight(weight_scope.weight_instance, weight, scope, state, step)

    def _let_value(
        self: _Elaborator,
        expr: LetExprNode,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower a let expression, declaring its free names as inputs.

        Parameters
        ----------
        expr : LetExprNode
            The expression.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The lowered value.
        """
        expr = self._hoist_function_calls(expr, scope, state, node)
        chart = _chart_construct(expr)
        if chart is not None:
            self._fail(
                node,
                f"let:chart; {chart} has no elaboration yet; deduction charts "
                "are a logic-programming construct outside the program calculus",
                code=GAP_CODE,
            )
        host_only = next(
            (
                candidate.func
                for candidate in walk_let_expr(expr)
                if isinstance(candidate, LetExprCall)
                and candidate.func in HOST_ONLY_BUILTINS
            ),
            None,
        )
        if host_only is not None:
            self._fail(
                node,
                f"builtin {host_only!r} is available only to native eager "
                "execution and has no checked QIEC lowering",
                code="qiec-builtin:host-only",
            )
        unknown = _unknown_call(expr, self._lambda_macros)
        if unknown is not None:
            self._fail(
                node,
                f"let:call:unknown:{unknown}; the call `{unknown}(...)` names no "
                "builtin, lambda, or computation; a host function reached by "
                "name has no elaboration",
                code=GAP_CODE,
            )
        free = frozenset(
            name
            for name in _free_let_names(expr)
            if scope.lookup(name) is None and name not in self._lambda_macros
        )
        fixed, reduced, _ = _unify_free_shapes(
            expr,
            free,
            lambda name: (
                local.type if (local := scope.lookup(name)) is not None else None
            ),
            lambda name: self._batch_of(name, state),
        )
        for name in _free_let_names(expr):
            if name not in free or scope.lookup(name) is not None:
                continue
            inferred = state.input_shapes.get(name)
            axis = inferred[1] if inferred is not None else None
            type_ = self._input_type(name, state)
            axes = (
                (axis.name,) if axis is not None and name not in state.tables else None
            )
            if name not in state.tables and name in fixed:
                dimensions = fixed[name]
                if axis is not None and dimensions != (axis.size,):
                    self._fail(
                        node,
                        f"host datum {name!r} is combined here with a value of "
                        f"shape {self._render(tensor_type(REAL, dimensions))}, but "
                        f"the plated step reading it ranges over "
                        f"{axis.name!r}; the two uses disagree on its extent",
                        code="qiec-program",
                    )
                element = inferred[0] if inferred is not None else REAL
                type_ = tensor_type(element, dimensions)
                axes = None
            elif name not in state.tables and name in reduced and axis is None:
                self._fail(
                    node,
                    f"host datum {name!r} is reduced here, but nothing fixes its "
                    "shape: no value of known shape is combined with it "
                    "elementwise and no plated step reads it. State the extent "
                    "it ranges over by combining it elementwise with a value over "
                    f"that object, or by reading it entrywise in a factor, as in "
                    f"`factor i : <Object> in {name}[i]`",
                    code="qiec-program",
                )
            self._declare_parameter(name, type_, "data", scope, state, node, axes=axes)
            split = tensor_shape(type_)
            if split is not None and name not in state.tables:
                state.batch_shapes[name] = tuple(split[1])
        return self._lower_value(
            expr,
            (),
            scope.context(),
            None,
            ("programs", state.declaration.name, "lets"),
        )

    def _function_morphism(self: _Elaborator, name: str) -> MorphismDecl | None:
        """The learned-function declaration a name denotes, if any.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        MorphismDecl | None
            The ``[role=function]`` morphism of that name, else ``None``.
        """
        morphism = self._program_morphisms.get(name)
        if morphism is None or _option_value(morphism.options, "role") != "function":
            return None
        return morphism

    def _hoist_function_calls(
        self: _Elaborator,
        expr: LetExprNode,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> LetExprNode:
        """Bind every learned-function call in a let expression on its own.

        Each call ``f(a, b)`` of a ``[role=function]`` morphism is
        elaborated, innermost first, to its network applied to its
        arguments and bound to a fresh name in the scope, and the
        expression reads that name in its place.

        Parameters
        ----------
        expr : LetExprNode
            The expression.
        scope : _Scope
            The scope the calls bind in.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        LetExprNode
            The expression with every learned-function call replaced.

        Raises
        ------
        QiecDiagnosticError
            If a call lies under a ``factor`` or lambda binder, or its
            arguments do not fit the function's domain.
        """
        calls = [
            candidate
            for candidate in walk_let_expr(expr)
            if isinstance(candidate, LetExprCall)
            and self._function_morphism(candidate.func) is not None
        ]
        if not calls:
            return expr
        for binder in walk_let_expr(expr):
            if not isinstance(binder, LetExprFactor | LetExprLambda):
                continue
            inner = next(
                (
                    candidate.func
                    for candidate in walk_let_expr(binder)
                    if isinstance(candidate, LetExprCall)
                    and self._function_morphism(candidate.func) is not None
                ),
                None,
            )
            if inner is not None:
                self._fail(
                    node,
                    f"the learned function {inner!r} is called inside a factor or "
                    "lambda body; call it in a let of its own and read the result "
                    "there",
                    code="qiec-program",
                )

        def hoist(candidate: LetExprNode) -> LetExprNode:
            if not isinstance(candidate, LetExprCall):
                return candidate
            morphism = self._function_morphism(candidate.func)
            if morphism is None:
                return candidate
            value, type_ = self._function_call(morphism, candidate, scope, state, node)
            serial = state.function_calls
            state.function_calls += 1
            name = f"{candidate.func}__call{serial}"
            self._bind_step(scope, name, type_, Return(value), node)
            return LetExprVar(name=name)

        return map_let_expr(expr, hoist)

    def _function_call(
        self: _Elaborator,
        morphism: MorphismDecl,
        call: LetExprCall,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> tuple[Value, TypeExpr]:
        """Elaborate one call of a learned function to its network.

        The network's layers are affine maps whose weights and biases
        are program inputs named after the function, as a network
        kernel's are: ``<f>_param_layer<k>_weight`` and
        ``<f>_param_layer<k>_bias`` for each hidden layer, under the
        declared activation, then ``<f>_param_weight`` and
        ``<f>_param_bias`` for the output layer, under the declared
        output nonlinearity. An identity network applies only the
        output nonlinearity.

        Parameters
        ----------
        morphism : MorphismDecl
            The ``[role=function]`` declaration.
        call : LetExprCall
            The call, its arguments already free of learned-function
            calls.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        tuple[Value, TypeExpr]
            The image and its type: ``Real`` for a one-wide codomain,
            else ``Tensor[Real]([width])``.

        Raises
        ------
        QiecDiagnosticError
            If the call's arity or an argument's width differs from the
            domain's factors, an argument is not a real scalar or vector,
            or a side has no fixed real width.
        """
        name = call.func
        widths: list[int] = []
        for factor in _factors(morphism.domain):
            info = (
                self._program_objects.get(factor.name)
                if isinstance(factor, TypeName)
                else None
            )
            width = (
                None
                if info is None or info.real_width is None
                else _fixed_extent(info.real_width)
            )
            if width is None:
                self._fail(
                    node,
                    f"the learned function {name!r} has a domain factor with no "
                    "fixed real width",
                    code="qiec-program",
                )
            widths.append(width)
        codomain = morphism.codomain
        info = (
            self._program_objects.get(codomain.name)
            if isinstance(codomain, TypeName)
            else None
        )
        out_width = (
            None
            if info is None or info.real_width is None
            else _fixed_extent(info.real_width)
        )
        if out_width is None:
            self._fail(
                node,
                f"the learned function {name!r} has a codomain with no fixed real "
                "width",
                code="qiec-program",
            )
        if len(call.args) != len(widths):
            self._fail(
                node,
                f"the learned function {name!r} takes {len(widths)} argument(s), "
                f"one per factor of its domain, but is called with "
                f"{len(call.args)}",
                code="qiec-program",
            )
        sequence_kind = sequence_source_kind(morphism)
        if sequence_kind is not None:
            return self._sequence_function_call(
                morphism,
                sequence_kind,
                call,
                widths[0],
                out_width,
                scope,
                state,
                node,
            )
        sources: list[Value] = []
        for position, (argument, width) in enumerate(
            zip(call.args, widths, strict=True)
        ):
            if (
                isinstance(argument, LetExprVar)
                and scope.lookup(argument.name) is None
                and argument.name not in self._lambda_macros
            ):
                # A host datum passed straight to the function takes the
                # type of the domain factor it fills.
                self._declare_parameter(
                    argument.name,
                    REAL if width == 1 else tensor_type(REAL, (_extent(width),)),
                    "data",
                    scope,
                    state,
                    node,
                )
            value = self._let_value(argument, scope, state, node)
            type_ = self._value_type(value, scope.context(), argument)
            shape = tensor_shape(type_)
            if type_ == REAL:
                given: int | None = 1
            elif (
                shape is not None
                and shape[0] == REAL
                and len(shape[1]) == 1
                and isinstance(shape[1][0], IndexLiteral)
            ):
                given = int(shape[1][0].value)
            else:
                given = None
            if given != width:
                self._fail(
                    node,
                    f"argument {position} of the learned function {name!r} has "
                    f"type {self._render(type_)}, but its domain factor is a "
                    f"{width}-wide real vector",
                    code="qiec-program",
                )
            sources.append(value)
        output = _option_value(morphism.options, "output") or "identity"
        result: TypeExpr = (
            REAL if out_width == 1 else tensor_type(REAL, (_extent(out_width),))
        )
        result_shape: tuple[IndexTerm, ...] | None = (
            None if out_width == 1 else (_extent(out_width),)
        )
        path = ("programs", state.declaration.name, "functions", name)
        if _source_kind(morphism) == "identity":
            if len(sources) != 1:
                self._fail(
                    node,
                    f"the identity function {name!r} has a product domain",
                    code="qiec-program",
                )
            image: Value = sources[0]
        else:
            activation = str(_option_value(morphism.options, "activation") or "tanh")
            fan_in = sum(widths)
            for depth, hidden in enumerate(self._hidden_widths(morphism, node)):
                layer_weight = self._declare_parameter(
                    f"{name}_param_layer{depth}_weight",
                    tensor_type(REAL, (_extent(hidden), _extent(fan_in))),
                    "weight",
                    scope,
                    state,
                    node,
                    axes=(f"{name}_layer{depth}_row", f"{name}_layer{depth}_col"),
                )
                layer_bias = self._declare_parameter(
                    f"{name}_param_layer{depth}_bias",
                    tensor_type(REAL, (_extent(hidden),)),
                    "bias",
                    scope,
                    state,
                    node,
                    axes=(f"{name}_layer{depth}_row",),
                )
                hidden_type = tensor_type(REAL, (_extent(hidden),))
                sources = [
                    self._primitive(
                        activation,
                        (
                            AffineMap(
                                Var(layer_weight),
                                Var(layer_bias),
                                tuple(sources),
                                0,
                                hidden,
                                "identity",
                                hidden_type,
                            ),
                        ),
                        node,
                        (*path, "layers", depth),
                        (_extent(hidden),),
                    )
                ]
                fan_in = hidden
            weight = self._declare_parameter(
                f"{name}_param_weight",
                tensor_type(REAL, (_extent(out_width), _extent(fan_in))),
                "weight",
                scope,
                state,
                node,
                axes=(f"{name}_param_row", f"{name}_param_col"),
            )
            bias = self._declare_parameter(
                f"{name}_param_bias",
                tensor_type(REAL, (_extent(out_width),)),
                "bias",
                scope,
                state,
                node,
                axes=(f"{name}_param_row",),
            )
            image = AffineMap(
                Var(weight),
                Var(bias),
                tuple(sources),
                0,
                out_width,
                "identity",
                result,
            )
        if output != "identity":
            image = self._primitive(
                str(output), (image,), node, (*path, "output"), result_shape
            )
        return image, result

    def _batch_of(self: _Elaborator, name: str, state: _ProgramState) -> _Shape:
        """The plate extents a bound name ranges over.

        Parameters
        ----------
        name : str
            The name.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        _Shape
            A site's batch extents, a plated input's or let's recorded
            extents, else none.
        """
        known = state.batch_shapes.get(name)
        if known is not None:
            return known
        for site in state.sites:
            if site.name == name:
                return tuple(axis.size for axis in site.batch)
        return ()

    def _record_batch(
        self: _Elaborator,
        name: str,
        expr: LetExprNode,
        scope: _Scope,
        state: _ProgramState,
    ) -> None:
        """Record the plate extents a let's value ranges over.

        Parameters
        ----------
        name : str
            The let's name.
        expr : LetExprNode
            Its expression, whose free names are already declared.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        """
        _, _, extents = _unify_free_shapes(
            expr,
            frozenset(),
            lambda read: (
                local.type if (local := scope.lookup(read)) is not None else None
            ),
            lambda read: self._batch_of(read, state),
        )
        state.batch_shapes[name] = extents

    def _input_type(self: _Elaborator, name: str, state: _ProgramState) -> TypeExpr:
        """The type a free name read by a let expression is declared at.

        Parameters
        ----------
        name : str
            The name.
        state : _ProgramState
            The program's accumulating state, whose inferred shapes say
            which plate the name's values range over.

        Returns
        -------
        TypeExpr
            For a name some let gathers from, a table: the tensor of its
            element over one open extent per axis a gather indexes, and
            per axis of a row when the gathered rows fill a vector-valued
            family parameter, each a static binder of the program that a
            run reads off the data. Otherwise ``Int`` or ``Real`` when the name reaches no
            plated step, else the tensor of that element over the
            plate's first batch axis. A name pinned for a scan cell
            specialized to its scanning program has the pinned type.
        """
        pinned = self._pinned_types.get(name)
        if pinned is not None:
            return pinned
        element, axis = state.input_shapes.get(name, (REAL, None))
        rank = state.tables.get(name)
        if rank is not None:
            rank += state.table_events.get(name, 0)
            return tensor_type(element, self._open_extents(name, rank, state))
        if axis is None:
            return element
        return tensor_type(element, (axis.size,))

    def _table_event_ranks(
        self: _Elaborator, declaration: ProgramDecl, tables: Mapping[str, int]
    ) -> dict[str, int]:
        """The axes each table's rows carry, read off the draws they fill.

        A let that only gathers rows from a table, as in ``let row_cuts =
        cuts[participant_idx]``, and fills a family parameter of rank r
        with them, gathers vectors (or matrices) of that rank: the table
        has r further axes after the ones a gather indexes.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        tables : Mapping[str, int]
            The gathered tables, with the axes a gather indexes.

        Returns
        -------
        dict[str, int]
            Each table whose rows fill a vector-valued parameter, with the
            rank of one row.
        """
        gathers: dict[str, str] = {}
        steps: list[ProgramStep] = []

        def collect(body: Sequence[ProgramStep]) -> None:
            """Gather a body's steps, marginalization scopes included.

            Parameters
            ----------
            body : Sequence[ProgramStep]
                The steps.
            """
            for step in body:
                steps.append(step)
                if isinstance(step, MarginalizeStep):
                    collect(step.scope)

        collect(declaration.draws)
        for step in steps:
            if isinstance(step, LetStep) and isinstance(step.value, LetExprIndex):
                array = step.value.array
                while isinstance(array, LetExprIndex):
                    array = array.array
                if isinstance(array, LetExprVar) and array.name in tables:
                    gathers[step.name] = array.name
        events: dict[str, int] = {}
        for step in steps:
            if not isinstance(step, SampleStep | ObserveStep) or not step.args:
                continue
            try:
                resolved = resolve_step_dist(
                    step.morphism,
                    step.args,
                    morphisms=self._program_morphisms,
                    lets=self._program_lets,
                    family_registry=frozenset(FAMILIES),
                    target="qvr-qiec",
                )
            except StepResolutionError, TypeError:
                continue
            record = FAMILIES.get(resolved.family)
            if record is None:
                continue
            for argument in step.args:
                if not isinstance(argument, DrawArgNamed):
                    continue
                value = argument.value
                if not isinstance(value, DrawArgName) or value.text not in gathers:
                    continue
                try:
                    parameter = _semantic_family_parameter(
                        record, argument.parameter, step.morphism
                    )
                except KeyError:
                    continue
                if parameter.rank > 0:
                    table = gathers[value.text]
                    events[table] = max(parameter.rank, events.get(table, 0))
        return events

    def _infer_input_shapes(
        self: _Elaborator, declaration: ProgramDecl
    ) -> dict[str, tuple[TypeApplication, PlateAxis | None]]:
        """Infer the shapes of the free names let expressions read.

        A free name is a scalar unless a step plated over a batch axis
        reads a let expression that depends on it, in which case its
        values range over that axis; a name that indexes a tensor is
        integral. The inference walks the steps in order, so the first
        plated consumer of a name fixes its axis.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.

        Returns
        -------
        dict[str, tuple[TypeApplication, PlateAxis | None]]
            Each free name's element type and batch axis, if any.
        """
        bound: set[str] = set()
        if declaration.params is not None:
            bound.update(declaration.params)
        if declaration.type_params is not None:
            bound.update(
                parameter.name
                for parameter in declaration.type_params
                if isinstance(parameter, ScalarParam)
            )
        depends: dict[str, set[str]] = {}
        elements: dict[str, TypeApplication] = {}
        axes: dict[str, PlateAxis] = {}

        def free_of(expr: LetExprNode) -> set[str]:
            """Every free name an expression depends on, through lets.

            Parameters
            ----------
            expr : LetExprNode
                The expression.

            Returns
            -------
            set[str]
                Free names, transitively through let bindings.
            """
            found: set[str] = set()
            for name in _free_let_names(expr):
                if name in depends:
                    found |= depends[name]
                elif name not in bound and name not in self._lambda_macros:
                    found.add(name)
            return found

        def visit(steps: Sequence[ProgramStep], group: PlateAxis | None) -> None:
            """Walk steps, recording dependencies and plated consumers.

            Parameters
            ----------
            steps : Sequence[ProgramStep]
                The steps.
            group : PlateAxis | None
                The grouping axis of the enclosing marginalization.
            """
            for step in steps:
                if isinstance(step, LetStep | ScoreStep):
                    if isinstance(step.value, LetExprLambda):
                        continue
                    for index_name in _index_names(step.value):
                        if index_name not in bound and index_name not in depends:
                            elements[index_name] = INT
                    depends[step.name] = free_of(step.value)
                    bound.add(step.name)
                elif isinstance(step, CallStep):
                    depends[step.name] = set().union(
                        *(free_of(argument) for argument in step.call.arguments)
                    )
                    bound.add(step.name)
                elif (
                    isinstance(step, SampleStep)
                    and step.morphism in self._program_declarations
                ):
                    # A draw from a program runs the program's steps in
                    # place, so the names they read are typed as they
                    # are read there.
                    template = self._program_declarations[step.morphism]
                    try:
                        instantiated = instantiate_program(
                            template,
                            step.vars,
                            template_bindings(
                                template,
                                step.args or (),
                                {name: LetExprVar(name=name) for name in bound},
                            ),
                        )
                    except TemplateError:
                        # The elaboration of the draw reports the misfit
                        # with its location.
                        bound.update(step.vars)
                        continue
                    visit(instantiated, group)
                    bound.update(step.vars)
                elif isinstance(step, SampleStep | ObserveStep | MarginalizeStep):
                    names = (
                        [step.var]
                        if isinstance(step, MarginalizeStep)
                        else list(step.vars)
                    )
                    try:
                        resolved = resolve_step_dist(
                            step.morphism,
                            step.args,
                            morphisms=self._program_morphisms,
                            lets=self._program_lets,
                            family_registry=frozenset(FAMILIES),
                            target="qvr-qiec",
                        )
                    except StepResolutionError, TypeError:
                        # A step the resolver has no wire form for, such
                        # as an operator form over nested distributions,
                        # shapes nothing here; the elaboration reads it
                        # structurally.
                        bound.update(names)
                        continue
                    record = FAMILIES.get(resolved.family)
                    if record is None:
                        bound.update(names)
                        continue
                    morphism = self._program_morphisms.get(step.morphism)
                    # A grouped marginalization's latent is batched over
                    # its own group, so its prior's free names range over
                    # that axis as its scope's do.
                    own_group = group
                    if isinstance(step, MarginalizeStep):
                        if step.over is not None:
                            own_group = self._axis(step.over, step)
                        elif step.over_objs:
                            own_group = self._axis(step.over_objs[0], step)
                    plate = self._step_plate(step, record, morphism, own_group)
                    batch = plate.batch[0] if plate.batch else None
                    if batch is not None:
                        for argument in step.args or ():
                            for name in _argument_names(argument):
                                for free in depends.get(name, set()):
                                    axes.setdefault(free, batch)
                    bound.update(names)
                    if isinstance(step, MarginalizeStep):
                        visit(step.scope, own_group)

        visit(declaration.draws, None)
        return {
            name: (elements.get(name, REAL), axes.get(name))
            for name in set(elements) | set(axes)
        }

    def _unify_result_extents(
        self: _Elaborator, declaration: ProgramDecl, scope: _Scope, state: _ProgramState
    ) -> None:
        """Fix the open extents of a returned value to the program's codomain.

        A program declared ``p : A -> D`` with ``D`` a ``Real N`` object
        returns a value of ``D``: each axis of a returned real vector, or
        of a returned square matrix, whose extent only the data would fix,
        such as a draw whose mean is a host vector, has ``N`` entries.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        scope : _Scope
            The body's scope.
        state : _ProgramState
            The program's accumulating state.
        """
        if len(declaration.return_vars) != 1:
            return
        local = scope.lookup(declaration.return_vars[0])
        info = _object_expr_info(declaration.codomain, self._program_objects)
        if local is None or info is None or info.real_width is None:
            return
        shape = tensor_shape(local.type)
        if shape is None or shape[0] != REAL:
            return
        own = [self._resolved_extent(item, state) for item in shape[1]]
        if len(own) > 2 or (len(own) == 2 and own[0] != own[1]):
            # A ``Real N`` codomain sizes a vector, or a square matrix over
            # it; it says nothing of a matrix of two shapes.
            return
        width = _extent(info.real_width)
        open_sizes = [
            size
            for size in (self._resolved_extent(item, state) for item in shape[1])
            if isinstance(size, IndexVariable) and size.name in state.open_extents
        ]
        self._unify_extents([(size, width) for size in open_sizes], scope, state)

    def _program_return(
        self: _Elaborator, declaration: ProgramDecl, scope: _Scope, state: _ProgramState
    ) -> tuple[Value, TypeExpr]:
        """The program's result value and type.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        scope : _Scope
            The body's scope.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        tuple[Value, TypeExpr]
            The returned value, a tuple for several names, and its type.

        Raises
        ------
        QiecDiagnosticError
            If a returned name is unbound.
        """
        del state
        values: list[Value] = []
        types: list[TypeExpr] = []
        for name in declaration.return_vars:
            local = scope.lookup(name)
            if local is None:
                self._fail(
                    declaration,
                    f"program {declaration.name!r} returns unbound name {name!r}",
                    code="qiec-program",
                )
            values.append(Var(local))
            types.append(local.type)
        if not values:
            return LiteralValue(None, UNIT), UNIT
        if len(values) == 1:
            return values[0], types[0]
        result_type = product_type(*types)
        return TupleValue(tuple(values), result_type), result_type

    # ------------------------------------------------------------------
    # marginalization

    def _elaborate_marginalize(
        self: _Elaborator, step: MarginalizeStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate a marginalization block to a helper computation call.

        The helper allocates a ``Random`` instance for the latent, handles
        it with the enumeration handler, samples the latent, allocates a
        ``Weight`` instance for the scope, handles it with the collecting
        handler, runs the scope's steps, and answers ``unit``; the
        enumeration handler's answer is the block's log marginal, which
        the enclosing scope scores.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        scope : _Scope
            The enclosing scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the reduction is not one of ``logsumexp``, ``sum``, and
            ``mean``, a reduction is written on a latent with no finite
            support, or the block lies inside a grouped marginalization
            without sharing its group's extent.
        """
        reduction = step.reduction or "logsumexp"
        if reduction not in MARGINAL_REDUCTIONS:
            self._fail(
                step,
                f"marginalize reduction {reduction!r} has no elaboration; the "
                f"reductions are {', '.join(MARGINAL_REDUCTIONS)}",
                code="qiec-program",
            )
        handlers = self._program_handlers()
        collect_handler = handlers[COLLECT_HANDLER]
        group: PlateAxis | None = None
        factors: tuple[PlateAxis, ...] = ()
        if step.over is not None:
            group = self._axis(step.over, step)
            factors = (group,)
        elif step.over_objs:
            factors = tuple(self._axis(name, step) for name in step.over_objs)
            group = factors[0] if len(factors) == 1 else _product_axis(factors)
        elif step.index is not None:
            # An annotated axis that is not the family's support
            # replicates the latent: one marginal per element, which is
            # a grouping by the axis with every row its own group.
            record = self._marginal_family(step)
            axis = self._object_axis(step.index, step)
            if (
                record is not None
                and record.event_rank == 0
                and _enumerable(record)
                and not _names_support(record, axis)
            ):
                group = axis
                factors = (axis,)
        inner = _Scope(outer=scope, group=group, factors=factors)
        remaining = self._hoist_draws(step, scope, state)
        latent_name = step.var
        weight_scope = _weight_scope(scope)
        outer_group = weight_scope.group if weight_scope is not None else None
        projection: Value | None = None
        if outer_group is not None:
            if group is None:
                self._fail(
                    step,
                    f"marginalization of {latent_name!r} lies inside a "
                    f"marginalization grouped over {outer_group.name!r} and must "
                    "be grouped itself to add to that group's weights",
                    code="qiec-program",
                )
            if group.size != outer_group.size:
                projection = _group_projection(factors, outer_group)
                if projection is None:
                    self._fail(
                        step,
                        f"marginalization of {latent_name!r} is grouped over "
                        f"{group.name!r} inside a marginalization grouped over "
                        f"{outer_group.name!r}; the inner group must be of the "
                        "outer's extent, identified with it position by "
                        "position, or a product with the outer axis as a factor, "
                        "projected onto it",
                        code="qiec-program",
                    )
        state.current_site = latent_name
        state.pending_alphabet = None
        distribution = self._step_distribution(
            step,
            inner,
            state,
            latent_name,
            group=group,
            refinement=(
                None
                if projection is None or outer_group is None
                else (outer_group, projection)
            ),
        )
        if state.pending_alphabet is not None:
            state.alphabets[latent_name] = state.pending_alphabet
        record = FAMILIES[distribution.name]
        if not _enumerable(record):
            # Nothing finite to sum over: the latent is drawn once per
            # position and the scope runs in the enclosing scope, which
            # is what the runtime does with such a block.
            if step.reduction is not None:
                self._fail(
                    step,
                    f"marginalize reduction {step.reduction!r} aggregates a "
                    f"finite support, which {distribution.name!r} has none of",
                    code="qiec-program",
                )
            self._elaborate_continuous_marginalize(
                step, distribution, scope, state, latent_name
            )
            return
        sampled = self._sampled_type(distribution)
        weight_type: TypeExpr = (
            LOG_WEIGHT if group is None else tensor_type(LOG_WEIGHT, (group.size,))
        )
        # Nested in a grouped marginalization, the block answers one
        # marginal per group position for the enclosing group's weights.
        result_type: TypeExpr = LOG_WEIGHT if outer_group is None else weight_type
        chooser = handlers[MARGINAL_HANDLERS[(outer_group is not None, reduction)]]
        # The row of a marginalized chain's cell keeps the block's weighted
        # shots, one per state, which the forward algorithm reduces.
        table = (
            state.markov_row == latent_name and weight_scope is None and group is None
        )
        if table:
            states = self._object_axis(state.declaration.codomain, step)
            result_type = tensor_type(LOG_WEIGHT, (states.size,))
            chooser = handlers[ENUMERATE_TABLE_HANDLER]
        path = ("programs", state.declaration.name, "marginals", latent_name)
        choice = self._local_instance(
            f"{latent_name}_choice", "Random", (), (*path, "choice")
        )
        weight = self._local_instance(
            f"{latent_name}_weight", "Weight", (weight_type,), (*path, "weight")
        )
        inner.weight_instance = weight
        inner.weight_type = weight_type
        request = self._sample_request(
            choice, latent_name, sampled, distribution, step, state
        )
        state.sites.append(
            ProgramSite(
                latent_name,
                "marginal",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        latent = Local(latent_name, sampled)
        inner.locals[latent_name] = latent
        self._elaborate_steps(remaining, inner, state)
        if not inner.weights_added:
            # A scope adding nothing still performs on its instance, so
            # the collecting handler has the effect it handles: the
            # identity weight at the scope's type.
            self._add_weight(
                weight, self._zero_weight(weight_type, state, step), inner, state, step
            )
        collected = Handle(
            weight.entry.instance,
            collect_handler.id,
            self._fold_steps(inner, Return(LiteralValue(None, UNIT))),
            (weight_type,),
        )
        del self.instances[weight.name]
        del self.instances[choice.name]
        enumerated = Handle(
            choice.entry.instance,
            chooser.id,
            Bind(
                latent,
                Perform(request),
                NewInstance(
                    weight.entry.instance,
                    weight.entry.effect,
                    collected,
                    _origin_at(self, (*path, "weight"), "local-instance", step),
                ),
            ),
            (result_type if table else weight_type,),
        )
        body = NewInstance(
            choice.entry.instance,
            choice.entry.effect,
            enumerated,
            _origin_at(self, (*path, "choice"), "local-instance", step),
        )
        helper = self._marginal_helper(step, inner, body, scope, state, result_type)
        if table:
            self._bind_step(scope, latent_name, result_type, helper, step)
            return
        weight_local = self._bind_step(scope, None, result_type, helper, step)
        if weight_scope is None:
            self._add_score(state.score, Var(weight_local), scope, state, step)
            return
        assert weight_scope.weight_instance is not None
        assert outer_group is not None
        weight: Value = Var(weight_local)
        if projection is not None:
            # The inner group refines the outer: its per-position
            # marginals sum into the outer positions they project to.
            weight = SegmentSum(
                weight,
                projection,
                outer_group.size,
                tensor_type(LOG_WEIGHT, (outer_group.size,)),
            )
        self._add_weight(weight_scope.weight_instance, weight, scope, state, step)

    def _hoist_draws(
        self: _Elaborator,
        step: MarginalizeStep,
        scope: _Scope,
        state: _ProgramState,
    ) -> tuple[ProgramStep, ...]:
        """Draw a block's independent latents once, before the block.

        A ``sample`` inside a marginalization block reads nothing the
        block binds, so it is one draw the block's every shot shares,
        not a draw per value of the latent; the enclosing scope draws it
        before the block, as the runtime does.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        scope : _Scope
            The enclosing scope, which the draws join.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        tuple[ProgramStep, ...]
            The block's steps without the hoisted draws.

        Raises
        ------
        QiecDiagnosticError
            If a draw inside the block reads a name the block binds,
            which would be a draw per value of the latent.
        """
        bound: set[str] = {step.var}
        remaining: list[ProgramStep] = []
        for item in step.scope:
            if isinstance(item, SampleStep):
                names = _draw_dependencies(item)
                if names & bound:
                    self._fail(
                        item,
                        f"draw of {', '.join(item.vars)!r} inside the marginalization "
                        f"of {step.var!r} reads {', '.join(sorted(names & bound))!r}, "
                        "which the block binds; a draw per value of the latent "
                        "has no elaboration",
                        code="qiec-program",
                    )
                self._elaborate_sample(item, scope, state)
                continue
            remaining.append(item)
            bound.update(_step_binders(item))
        return tuple(remaining)

    def _zero_weight(
        self: _Elaborator, weight_type: TypeExpr, state: _ProgramState, node: object
    ) -> Value:
        """The identity weight at a scope's type.

        Parameters
        ----------
        weight_type : TypeExpr
            ``LogWeight`` or a tensor of it with literal dimensions.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            Zero as a weight, or a tensor of zeros of that shape.
        """
        shape = tensor_shape(weight_type)
        if shape is None:
            return self._primitive(
                "as_weight",
                (LiteralValue(0.0, REAL),),
                node,
                ("programs", state.declaration.name, "weights", "identity"),
            )
        dimensions = shape[1]
        assert isinstance(dimensions[0], IndexLiteral)
        inner = (
            tensor_type(shape[0], dimensions[1:]) if len(dimensions) > 1 else shape[0]
        )
        entries = tuple(
            self._zero_weight(inner, state, node) for _ in range(dimensions[0].value)
        )
        return TensorValue(entries, weight_type)

    def _marginal_family(
        self: _Elaborator, step: MarginalizeStep
    ) -> DistributionFamily | None:
        """The family a marginalization's latent is drawn from.

        Parameters
        ----------
        step : MarginalizeStep
            The step.

        Returns
        -------
        DistributionFamily | None
            The registry family the step's morphism slot names, either
            directly or through a morphism's bare initializer; ``None``
            when neither names one.
        """
        name = OPERATOR_ALIASES.get(step.morphism, step.morphism)
        record = FAMILIES.get(name)
        if record is not None:
            return record
        morphism = self._program_morphisms.get(step.morphism)
        if morphism is None:
            return None
        family = _bare_init_family(morphism)
        if family is None and morphism.init_family is not None:
            family = morphism.init_family.family
        return FAMILIES.get(OPERATOR_ALIASES.get(family or "", family or ""))

    def _elaborate_continuous_marginalize(
        self: _Elaborator,
        step: MarginalizeStep,
        distribution: DistributionValue,
        scope: _Scope,
        state: _ProgramState,
        latent_name: str,
    ) -> None:
        """Elaborate a marginalization over a family with no finite support.

        The latent is sampled on the canonical ``random`` instance, plated
        as the block says, and the scope's steps run in the enclosing
        scope with the latent bound.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        distribution : DistributionValue
            The latent's construction.
        scope : _Scope
            The enclosing scope.
        state : _ProgramState
            The program's accumulating state.
        latent_name : str
            The latent's name.
        """
        sampled = self._sampled_type(distribution)
        state.uses_random = True
        request = self._sample_request(
            state.random, latent_name, sampled, distribution, step, state
        )
        state.sites.append(
            ProgramSite(
                latent_name,
                "sample",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        self._bind_step(scope, latent_name, sampled, Perform(request), step)
        self._elaborate_steps(step.scope, scope, state)

    def _local_instance(
        self: _Elaborator,
        name: str,
        interface: str,
        arguments: tuple[TypeExpr, ...],
        path: tuple[str | int, ...],
    ) -> NamedEffectInstance:
        """Allocate a lexical instance for a marginalization block.

        Parameters
        ----------
        name : str
            The instance's name.
        interface : str
            The prelude interface.
        arguments : tuple[TypeExpr, ...]
            The interface's type arguments.
        path : tuple[str | int, ...]
            The lexical path, which the instance's identity derives from.

        Returns
        -------
        NamedEffectInstance
            The instance, registered under its name until the block ends.

        Raises
        ------
        QiecDiagnosticError
            If the name shadows an instance in scope.
        """
        definition = self._effect(interface)
        assert definition is not None
        effect: EffectRef = definition.apply(arguments) if arguments else definition.ref
        if name in self.instances:
            self._fail(
                self.source.syntax.statements[0],
                f"marginalization instance {name!r} shadows an instance in scope",
                code="qiec-program",
            )
        entry = instantiate_effect(
            effect, module=self.source.module_name, lexical_path=path
        )
        instance = NamedEffectInstance(
            name, entry, _origin_at(self, path, "effect-instance")
        )
        self.instances[name] = instance
        return instance

    def _marginal_helper(
        self: _Elaborator,
        step: MarginalizeStep,
        inner: _Scope,
        body: Computation,
        scope: _Scope,
        state: _ProgramState,
        result_type: TypeExpr,
    ) -> Computation:
        """Declare the helper computation holding a marginalization block.

        The helper takes every enclosing binding the block read, so the
        block is a closed computation the enclosing scope calls.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        inner : _Scope
            The block's scope, whose ``referenced`` names are the
            helper's parameters.
        body : Computation
            The block's body.
        scope : _Scope
            The enclosing scope.
        state : _ProgramState
            The program's accumulating state.
        result_type : TypeExpr
            What the block answers with: the marginal ``LogWeight``, or
            one per group position when nested in a grouped block.

        Returns
        -------
        Computation
            The call of the helper from the enclosing scope.

        Raises
        ------
        QiecDiagnosticError
            If the block's body fails to check.
        """
        captured = tuple(
            scope.lookup(name)
            for name in sorted(
                inner.referenced, key=lambda item: _binding_order(scope, item)
            )
        )
        parameters = tuple(local for local in captured if local is not None)
        name = f"{state.declaration.name}__{step.var}_marginal"
        if name in self.computation_signatures:
            self._fail(
                step,
                f"marginalization helper {name!r} is declared twice",
                code="qiec-program",
            )
        # The helper takes the program's open extents its parameters are
        # shaped by, so a captured input keeps its type.
        mentioned = _index_variables_in(
            tuple(parameter.type for parameter in parameters)
        )
        telescope = tuple(
            binder for name_, binder in state.extents.items() if name_ in mentioned
        )
        statics: tuple[StaticArgument, ...] = tuple(
            IndexVariable(binder.name, binder.sort) for binder in telescope
        )
        identity = ComputationId.derive(self.source.module_name, "computation", name)
        signature = ComputationSignature(
            identity,
            name,
            telescope,
            tuple(parameter.type for parameter in parameters),
            result_type,
            EffectRow(),
        )
        try:
            self.registry.register_computation(signature)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(step, error, fallback="qiec-program")
        self.computation_signatures[name] = signature
        context = CheckContext(parameters)
        try:
            actual = infer_computation(body, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(step, error, fallback="qiec-program")
        if actual.result != result_type or actual.effects.entries:
            self._fail(
                step,
                f"marginalization of {step.var!r} does not close to "
                f"{render_static(result_type)}",
                code="qiec-program",
            )
        origin = _origin_at(
            self,
            ("programs", state.declaration.name, "marginals", step.var),
            "computation",
            step,
        )
        state.helpers.append(
            NamedComputation(
                identity,
                name,
                telescope,
                parameters,
                body,
                ComputationType(EffectRow(), result_type),
                origin,
            )
        )
        return Call(
            identity,
            name,
            statics,
            tuple(Var(parameter) for parameter in parameters),
            result_type,
            EffectRow(),
            _origin_at(
                self,
                ("programs", state.declaration.name, "marginals", step.var, "call"),
                "call",
                step,
            ),
        )

    # ------------------------------------------------------------------
    # distributions and plates

    def _step_distribution(
        self: _Elaborator,
        step: SampleStep | ObserveStep | MarginalizeStep,
        scope: _Scope,
        state: _ProgramState,
        site: str,
        *,
        via: tuple[str, ...] | None = None,
        group: PlateAxis | None = None,
        refinement: tuple[PlateAxis, Value] | None = None,
    ) -> DistributionValue:
        """The distribution a step draws from, plated as the step says.

        Parameters
        ----------
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        scope : _Scope
            The scope its arguments are read in.
        state : _ProgramState
            The program's accumulating state.
        site : str
            The site's name, for diagnostics and identities.
        via : str | None
            The index map re-indexing grouped arguments, for an observe.
        group : PlateAxis | None
            The grouping axis of the enclosing or this marginalization.
        refinement : tuple[PlateAxis, Value] | None
            For a marginalization grouped over a product refining the
            enclosing group: that group and the index projecting this
            group's positions onto it, which re-indexes arguments shaped
            by the enclosing group.

        Returns
        -------
        DistributionValue
            The typed construction.

        Raises
        ------
        QiecDiagnosticError
            If the morphism cannot be resolved to a family, or the
            arguments and plate do not type.
        """
        if self._function_morphism(step.morphism) is not None:
            self._fail(
                step, learned_function_draw_message(step.morphism), code="qiec-program"
            )
        if step.morphism in FAMILIES or step.morphism in OPERATOR_ALIASES:
            # A family applied by name resolves to itself; its arguments
            # are read structurally, nested distributions included.
            resolved = ResolvedDist(step.morphism, (), step.morphism)
        else:
            try:
                resolved = resolve_step_dist(
                    step.morphism,
                    step.args,
                    morphisms=self._program_morphisms,
                    lets=self._program_lets,
                    family_registry=frozenset(FAMILIES) | frozenset(OPERATOR_ALIASES),
                    target="qvr-qiec",
                )
            except StepResolutionError as error:
                if any(
                    kind == "param-source:attention"
                    or kind.startswith("param-source:attention:")
                    for kind in error.kinds
                ):
                    self._fail(
                        step,
                        attention_draw_message(step.morphism),
                        code="qiec-program",
                    )
                network = self._network_kernel(step, error.kinds)
                if network is None:
                    self._fail(
                        step,
                        error.detail,
                        code=GAP_CODE if _is_gap(error.kinds) else "qiec-program",
                    )
                resolved = network
        family_name = OPERATOR_ALIASES.get(resolved.family, resolved.family)
        record = FAMILIES.get(family_name)
        if record is None:
            self._fail(
                step,
                f"family:{resolved.family}: not in the semantic family registry",
                code="qiec-program",
            )
        morphism = self._program_morphisms.get(step.morphism)
        plate = self._step_plate(step, record, morphism, group)
        if record.name in MEASURE_FAMILIES:
            # An operator form denotes a measure the step normalizes at its
            # boundary: the construction itself is unplated, and the
            # normalization carries the step's plate.
            inner = self._nested_distribution(
                DrawArgDist(family=resolved.family, args=tuple(step.args)),
                scope,
                state,
                step,
            )
            normalize = FAMILIES["Normalize"]
            path = ("programs", state.declaration.name, "sites", site, "distribution")
            provisional = DistributionValue(
                normalize.id,
                normalize.name,
                (("base", inner),),
                sampleable_type(REAL),
                _origin_at(self, path, "distribution", step),
                plate,
            )
            value = DistributionValue(
                normalize.id,
                normalize.name,
                (("base", inner),),
                self._plated_type(provisional, scope, step),
                provisional.origin,
                plate,
            )
            self._value_type(value, scope.context(), step)
            return value
        if morphism is not None and _structured(record, morphism, step):
            arguments = self._structured_arguments(
                record, morphism, plate, scope, state, step
            )
        elif morphism is not None and self._conditional_map(morphism, record.name):
            arguments = self._mapped_arguments(
                record, morphism, step, plate, scope, state
            )
        else:
            arguments = self._family_arguments(
                record, resolved, step, plate, scope, state, via, group, morphism
            )
            plate = self._annotated_plate(
                record, step, morphism, plate, arguments, scope
            )
            plate = self._broadcast_plate(record, step, plate, arguments, scope)
        if refinement is not None:
            arguments = [
                (name, self._refined(value, refinement, plate, scope, step))
                for name, value in arguments
            ]
        arguments = self._widen_alphabet(record, morphism, arguments, scope, step)
        arguments, plate = self._unify_event_shape(
            record, arguments, plate, scope, state, step
        )
        path = ("programs", state.declaration.name, "sites", site, "distribution")
        value = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            sampleable_type(REAL),
            _origin_at(self, path, "distribution", step),
            plate,
        )
        value = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            self._plated_type(value, scope, step),
            value.origin,
            plate,
        )
        self._value_type(value, scope.context(), step)
        return value

    def _unify_event_shape(
        self: _Elaborator,
        record: DistributionFamily,
        arguments: list[tuple[str, Value]],
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> tuple[list[tuple[str, Value]], PlateShape]:
        """Fix the open extents of a draw's event source to its event axes.

        A family whose plate names its event axes samples over them, so
        the argument its event shape comes from has their extents: an
        open extent of that argument, a host datum's size the data would
        otherwise fix, is fixed to the axis's size. Extents that are not
        open and differ are left for the kernel to report.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        arguments : list[tuple[str, Value]]
            The lowered arguments.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        tuple[list[tuple[str, Value]], PlateShape]
            The arguments and plate at every extent fixed so far.
        """
        supplied = record.event_argument(arguments)
        if (
            supplied is not None
            and record.event_rank
            and len(plate.event) >= record.event_rank
        ):
            shape = tensor_shape(self._value_type(supplied[1], scope.context(), node))
            if shape is not None and len(shape[1]) >= record.event_rank:
                natural = shape[1][len(shape[1]) - record.event_rank :]
                trailing = plate.event[len(plate.event) - record.event_rank :]
                self._unify_extents(
                    [
                        (size, axis.size)
                        for size, axis in zip(natural, trailing, strict=True)
                    ],
                    scope,
                    state,
                )
        if not state.extent_solution:
            return arguments, plate
        substitution = self._extent_substitution(state)
        return (
            [
                (name, substitute_value(value, substitution))
                for name, value in arguments
            ],
            substitute_plate(plate, substitution),
        )

    def _plated_type(
        self: _Elaborator, value: DistributionValue, scope: _Scope, node: object
    ) -> TypeExpr:
        """The ``Sampleable`` type of a plated construction.

        Parameters
        ----------
        value : DistributionValue
            The construction, whose claimed type is provisional.
        scope : _Scope
            The scope.
        node : object
            The source node.

        Returns
        -------
        TypeExpr
            The type the kernel assigns from the family, its arguments,
            and its plate.

        Raises
        ------
        QiecDiagnosticError
            If the kernel rejects the construction.
        """
        record = FAMILIES[value.name]
        context = scope.context()
        argument_types = {
            name: self._value_type(argument, context, node)
            for name, argument in value.arguments
        }
        try:
            return _plated_sample_type(value, record, argument_types)
        except KernelError as error:
            self._fail_kernel(node, error, fallback="qiec-program")

    def _sampled_type(self, distribution: DistributionValue) -> TypeExpr:
        """The type a distribution's draws have.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.

        Returns
        -------
        TypeExpr
            The element of its ``Sampleable`` type.
        """
        assert isinstance(distribution.result_type, TypeApplication)
        return cast(TypeExpr, distribution.result_type.arguments[0])

    def _step_plate(
        self: _Elaborator,
        step: SampleStep | ObserveStep | MarginalizeStep,
        record: DistributionFamily,
        morphism: MorphismDecl | None,
        group: PlateAxis | None,
    ) -> PlateShape:
        """The plate a step's distribution ranges over.

        Event axes come from the step's ``over``, else the morphism's
        ``[over=...]``, else a multivariate family's codomain factors;
        batch axes from ``iid_over``, else the ``: Axis`` annotation and
        a ``Real N`` codomain's width for a scalar family. An annotation
        beside an ``over`` is the plate its event is repeated over. A
        grouped marginalization's latent is batched over its group.

        Parameters
        ----------
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        record : DistributionFamily
            The family.
        morphism : MorphismDecl | None
            The declared morphism the step draws through, if any.
        group : PlateAxis | None
            The grouping axis of this marginalization.

        Returns
        -------
        PlateShape
            The plate.
        """
        if isinstance(step, MarginalizeStep):
            batch: list[PlateAxis] = [group] if group is not None else []
            event: list[PlateAxis] = []
            if step.index is not None:
                axis = self._object_axis(step.index, step)
                if record.event_rank > 0:
                    event.append(axis)
                elif (
                    not _enumerable(record) or not _names_support(record, axis)
                ) and axis not in batch:
                    # The annotation names the support only when the
                    # family's atoms are as many as the axis's elements;
                    # otherwise it replicates the latent, one per element,
                    # which is the grouping when the block groups by it.
                    batch.append(axis)
            return PlateShape(tuple(batch), tuple(event))
        axes = step.axes
        over = (
            tuple(axes.over) if axes is not None else _option_axes(step.options, "over")
        )
        iid_over = (
            tuple(axes.iid_over)
            if axes is not None
            else _option_axes(step.options, "iid_over")
        )
        if over or iid_over:
            batch_axes = tuple(self._axis(name, step) for name in iid_over)
            if (
                not iid_over
                and step.index is not None
                and not (isinstance(step.index, TypeName) and step.index.name in over)
            ):
                # The annotation is the plate the event axes' draws are
                # repeated over: ``sample p : Row <- Dirichlet(1.0)
                # [over=Col]`` draws one simplex point per row. An
                # annotation naming an event axis restates it.
                batch_axes = (self._object_axis(step.index, step),)
            rank = record.event_rank
            if (
                isinstance(step, SampleStep | ObserveStep)
                and rank <= 1
                and len(over) > rank
                and not (rank == 0 and len(over) == 1)
            ):
                # The leading axes beyond the family's event rank index
                # an independent array of variates, sampled or observed:
                # ``Normal`` over
                # ``[B, C]`` is a ``(B, C)`` array of independent
                # normals, ``Dirichlet`` over ``[B, K]`` one ``K``-simplex
                # point per element of ``B``. They are batch axes after
                # the plate, in the order written.
                split = len(over) - rank
                return PlateShape(
                    (*batch_axes, *(self._axis(name, step) for name in over[:split])),
                    tuple(self._axis(name, step) for name in over[split:]),
                )
            return PlateShape(
                batch_axes, tuple(self._axis(name, step) for name in over)
            )
        if morphism is not None:
            morphism_over = _option_axes(morphism.options, "over")
            if morphism_over:
                return PlateShape(
                    (), tuple(self._axis(name, step) for name in morphism_over)
                )
        event_axes: tuple[PlateAxis, ...] = ()
        if record.name == "GP" and morphism is not None:
            return PlateShape((), (self._gp_grid(morphism, step),))
        if record.event_rank > 0 and morphism is not None:
            factors = [
                factor
                for factor in _factors(morphism.codomain)
                if isinstance(factor, TypeName)
            ]
            finite = [
                self._axis(factor.name, step)
                for factor in factors
                if self._program_objects.get(factor.name) is not None
                and self._program_objects[factor.name].extent is not None
            ]
            if len(finite) == record.event_rank:
                event_axes = tuple(finite)
            elif finite and record.event_rank == 1:
                event_axes = (finite[0],)
        if event_axes:
            return PlateShape((), event_axes)
        width: PlateAxis | None = None
        if (
            record.event_rank == 0
            and morphism is not None
            and isinstance(morphism.codomain, TypeName)
        ):
            info = self._program_objects.get(morphism.codomain.name)
            if info is not None and info.real_width is not None:
                width = PlateAxis(morphism.codomain.name, _extent(info.real_width))
        batch = []
        if step.index is not None:
            axis = self._object_axis(step.index, step)
            if record.event_rank > 0:
                # A matrix family draws one square variate whose side is
                # the annotated axis; a vector family one vector over it.
                return PlateShape((), tuple(axis for _ in range(record.event_rank)))
            batch.append(axis)
        if width is not None:
            batch.append(width)
        return PlateShape(tuple(batch), ())

    def _annotated_plate(
        self: _Elaborator,
        record: DistributionFamily,
        step: SampleStep | ObserveStep | MarginalizeStep,
        morphism: MorphismDecl | None,
        plate: PlateShape,
        arguments: list[tuple[str, Value]],
        scope: _Scope,
    ) -> PlateShape:
        """Read a ``: Axis`` annotation on a vector family against its arguments.

        The annotation names the variate's own axis when nothing else
        fixes it: ``sample pi : K <- Dirichlet(1.0)`` draws one point of
        the ``K``-simplex. When the arguments fix a different event
        shape, as ``sample pc : Item <- Dirichlet(1.0, 2.0, 3.0)`` does
        with three entries, the annotation is the plate the draws are
        repeated over instead.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        morphism : MorphismDecl | None
            The declared morphism the step draws through, if any.
        plate : PlateShape
            The plate as the annotation alone reads.
        arguments : list[tuple[str, Value]]
            The lowered arguments.
        scope : _Scope
            The scope.

        Returns
        -------
        PlateShape
            The plate, the annotated axis moved to the batch when the
            arguments fix the event shape otherwise.
        """
        if (
            step.index is None
            or isinstance(step, MarginalizeStep)
            or step.axes is not None
            or record.event_rank == 0
            or not record.event_sources
            or _option_axes(step.options, "over")
            or _option_axes(step.options, "iid_over")
            or (morphism is not None and _option_axes(morphism.options, "over"))
        ):
            return plate
        supplied = record.event_argument(arguments)
        if supplied is None:
            return plate
        shape = tensor_shape(self._value_type(supplied[1], scope.context(), step))
        if shape is None or len(shape[1]) < record.event_rank:
            return plate
        natural = tuple(shape[1][len(shape[1]) - record.event_rank :])
        if not all(isinstance(size, IndexLiteral) for size in natural):
            return plate
        if natural == tuple(axis.size for axis in plate.event):
            return plate
        return PlateShape((self._object_axis(step.index, step),), ())

    def _broadcast_plate(
        self: _Elaborator,
        record: DistributionFamily,
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        arguments: list[tuple[str, Value]],
        scope: _Scope,
    ) -> PlateShape:
        """Read an unplated scalar family's plate off a tensor argument.

        ``Normal(mu, 0.5)`` with ``mu`` a vector draws one coordinate
        per entry of ``mu``, as the torch runtime broadcasts it: the
        argument's axes are the batch, named after the argument when it
        is a binding.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The plate read off the step.
        arguments : list[tuple[str, Value]]
            The lowered arguments.
        scope : _Scope
            The scope.

        Returns
        -------
        PlateShape
            The plate unchanged when the step fixes one or the family
            has an event; else the widest tensor argument's axes.

        Raises
        ------
        QiecDiagnosticError
            If two tensor arguments disagree in shape.
        """
        if plate.batch or plate.event or record.event_rank > 0:
            return plate
        ranks = {parameter.name: parameter.rank for parameter in record.parameters}
        axes: tuple[PlateAxis, ...] | None = None
        for name, value in arguments:
            if ranks.get(name, 0) != 0:
                continue
            shape = tensor_shape(self._value_type(value, scope.context(), step))
            if shape is None or not shape[1]:
                continue
            stem = value.local.name if isinstance(value, Var) else name
            found = tuple(
                PlateAxis(stem if index == 0 else f"{stem}_{index}", size)
                for index, size in enumerate(shape[1])
            )
            if axes is None:
                axes = found
            elif tuple(axis.size for axis in axes) != tuple(
                axis.size for axis in found
            ):
                draw = (
                    step.var
                    if isinstance(step, MarginalizeStep)
                    else ", ".join(step.vars)
                )
                first, second = (
                    " x ".join(render_static(axis.size) for axis in extent)
                    for extent in (axes, found)
                )
                self._fail(
                    step,
                    f"{record.name} draw of {draw!r} reads parameters of "
                    f"{first} and {second} entries, which do not broadcast to "
                    f"one extent",
                    code="qiec-program",
                )
        if axes is None:
            return plate
        return PlateShape(axes, ())

    def _axis(self: _Elaborator, name: str, node: object) -> PlateAxis:
        """A plate axis named by an object with a known extent.

        Parameters
        ----------
        name : str
            The object's name.
        node : object
            The source node.

        Returns
        -------
        PlateAxis
            The axis.

        Raises
        ------
        QiecDiagnosticError
            If the name is no object with a static extent.
        """
        info = self._program_objects.get(name)
        if info is None or info.extent is None:
            self._fail(
                node,
                f"axis {name!r} names no object with a static extent",
                code="qiec-program",
            )
        return PlateAxis(name, _extent(info.extent))

    def _object_axis(self: _Elaborator, expr: ObjectExpr, node: object) -> PlateAxis:
        """A plate axis from an index annotation.

        Parameters
        ----------
        expr : ObjectExpr
            The annotation: an object name, a ``FinSet N``, or a product
            of objects, which flattens to one axis.
        node : object
            The source node.

        Returns
        -------
        PlateAxis
            The axis.

        Raises
        ------
        QiecDiagnosticError
            If the expression has no static extent.
        """
        if isinstance(expr, TypeName):
            if expr.name.isdigit():
                return PlateAxis(expr.name, _extent(int(expr.name)))
            return self._axis(expr.name, node)
        info = self._object_info(expr, node)
        if info.extent is None:
            self._fail(
                node, "an index annotation needs a static extent", code="qiec-program"
            )
        return PlateAxis(_axis_name(expr), _extent(info.extent))

    def _object_info(self: _Elaborator, expr: ObjectExpr, node: object) -> ObjectInfo:
        """The shape an object expression denotes.

        Parameters
        ----------
        expr : ObjectExpr
            The expression.
        node : object
            The source node.

        Returns
        -------
        ObjectInfo
            Its extent, real width, and finiteness.

        Raises
        ------
        QiecDiagnosticError
            If the expression names an unknown object or one with no
            static shape.
        """
        info = _object_expr_info(expr, self._program_objects)
        if info is None:
            self._fail(
                node,
                f"object expression {_axis_name(expr)!r} has no static shape",
                code="qiec-program",
            )
        return info

    # ------------------------------------------------------------------
    # arguments

    def _family_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        resolved: ResolvedDist,
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        via: tuple[str, ...] | None,
        group: PlateAxis | None,
        morphism: MorphismDecl | None,
    ) -> list[tuple[str, Value]]:
        """Lower a step's arguments to the family's named parameters.

        The step's own arguments keep their structure; positions the
        resolver filled from a morphism's declaration or options come in
        wire form. Each is lowered against the parameter it fills, and a
        grouped argument reaching a grouped observation is re-indexed by
        the index map.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        resolved : ResolvedDist
            The resolver's family and wire arguments.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        via : str | None
            The index map of a grouped observation.
        group : PlateAxis | None
            The grouping axis in force.
        morphism : MorphismDecl | None
            The morphism drawn through, whose codomain widens a class
            alphabet.

        Returns
        -------
        list[tuple[str, Value]]
            The named arguments in registry order.

        Raises
        ------
        QiecDiagnosticError
            If more arguments are supplied than the family has
            parameters.
        """
        direct_family = step.morphism in FAMILIES or step.morphism in OPERATOR_ALIASES
        structural: tuple[DrawArg, ...] = (step.args or ()) if direct_family else ()
        if (
            not structural
            and not step.args
            and morphism is not None
            and morphism.init_family is not None
            and OPERATOR_ALIASES.get(
                morphism.init_family.family, morphism.init_family.family
            )
            == record.name
            and morphism.init_family.args
        ):
            init = morphism.init_family
            structural = init.args
        wire = tuple(resolved.args or ())
        raw: list[DrawArg | str | float | _Spread] = (
            list(structural) if structural else list(wire)
        )
        if (
            record.name == "Categorical"
            and not structural
            and not (morphism is not None and step.args)
        ):
            self._fail(
                step,
                "Categorical requires exactly one explicit parameterization; "
                "use probs=... or logits=...",
                code="qiec-program",
            )
        if structural:
            return self._named_family_arguments(
                record,
                resolved.family,
                list(structural),
                step,
                plate,
                scope,
                state,
                via,
                group,
            )

        raw = self._bundle_vector_argument(record, raw, step, plate, state)
        if len(raw) > len(record.parameters):
            self._fail(
                step,
                f"family {record.name!r} takes at most {len(record.parameters)} "
                f"parameters ({', '.join(record.parameter_names)})",
                code="qiec-program",
            )
        return self._lower_family_parameters(
            record,
            list(zip(record.parameters, raw, strict=False)),
            step,
            plate,
            scope,
            state,
            via,
            group,
        )

    def _named_family_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        surface_family: str,
        raw: list[DrawArg | str | float | _Spread],
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        via: tuple[str, ...] | None,
        group: PlateAxis | None,
    ) -> list[tuple[str, Value]]:
        """Bind a keyword-only construction to one complete family schema."""
        schemas = family_parameterizations(surface_family, record.parameter_names)
        schema_text = render_parameterizations(schemas)
        supplied: set[str] = set()
        source: dict[str, DrawArg] = {}
        for argument in raw:
            if not isinstance(argument, DrawArgNamed):
                self._fail(
                    step,
                    f"family {record.name!r} arguments are keyword-only; "
                    f"use one complete schema: {schema_text}",
                    code="qiec-program",
                )
            parameter_name = argument.parameter
            if parameter_name in supplied:
                self._fail(
                    step,
                    f"family {record.name!r} is given parameter "
                    f"{parameter_name!r} twice",
                    code="qiec-program",
                )
            supplied.add(parameter_name)
            source[parameter_name] = argument.value

        selected = next((schema for schema in schemas if set(schema) == supplied), None)
        if selected is None:
            self._fail(
                step,
                f"family {record.name!r} requires exactly one complete parameter "
                f"schema {schema_text}; got ({', '.join(source)})",
                code="qiec-program",
            )

        pairs: list[tuple[FamilyParameter, DrawArg | str | float | _Spread]] = []
        for parameter_name in selected:
            try:
                parameter = _semantic_family_parameter(
                    record, parameter_name, surface_family
                )
            except KeyError:
                self._fail(
                    step,
                    f"family {record.name!r} has no parameter "
                    f"{parameter_name!r}; it takes "
                    f"{', '.join(record.parameter_names)}",
                    code="qiec-program",
                )
            argument: DrawArg | str | float | _Spread = source[parameter_name]
            if len(selected) == 1:
                bundled = self._bundle_vector_argument(
                    record, [argument], step, plate, state
                )
                argument = bundled[0]
            pairs.append((parameter, argument))
        return self._lower_family_parameters(
            record, pairs, step, plate, scope, state, via, group
        )

    def _lower_family_parameters(
        self: _Elaborator,
        record: DistributionFamily,
        pairs: list[tuple[FamilyParameter, DrawArg | str | float | _Spread]],
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        via: tuple[str, ...] | None,
        group: PlateAxis | None,
    ) -> list[tuple[str, Value]]:
        """Lower each argument against its parameter, sizing free names by the draw.

        A free name filling a tensor parameter whose axes the family ties
        to its sample's (a multivariate normal's mean and Cholesky
        factor, a Wishart's scale) takes those axes' extents from the
        draw: the plate's trailing event axes when it names enough, else
        the arguments already typed. An axis nothing in the draw fixes is
        one open extent, shared by every free name the family ties to it.
        Free names are therefore lowered after the arguments that may fix
        their extents.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        pairs : list[tuple[FamilyParameter, DrawArg | str | float | _Spread]]
            Each parameter with the argument filling it, in order.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        via : str | None
            The index map of a grouped observation.
        group : PlateAxis | None
            The grouping axis in force.

        Returns
        -------
        list[tuple[str, Value]]
            The named arguments, in the order of ``pairs``.
        """
        event: dict[int, IndexTerm] = {}
        if (
            record.event_rank
            and len(plate.event) >= record.event_rank
            and not self._provisional_event(record, step)
        ):
            trailing = plate.event[len(plate.event) - record.event_rank :]
            event = {position: axis.size for position, axis in enumerate(trailing)}
        values: dict[int, Value] = {}
        tied: list[int] = []
        for position, (parameter, argument) in enumerate(pairs):
            if self._tied_free_name(parameter, argument, plate, scope) is not None:
                tied.append(position)
                continue
            value = self._draw_argument(argument, parameter, plate, scope, state, step)
            values[position] = value
            if parameter.event_dims:
                self._tie_event_extents(
                    record, parameter, value, event, step, scope, state
                )
        for position in tied:
            parameter, argument = pairs[position]
            name = self._tied_free_name(parameter, argument, plate, scope)
            assert name is not None
            extents: list[IndexTerm] = []
            for axis, dim in enumerate(parameter.event_dims):
                if dim is not None and dim in event:
                    extents.append(event[dim])
                    continue
                extent = self._open_extent(
                    f"{name}_extent" if parameter.rank == 1 else f"{name}_extent{axis}",
                    state,
                )
                if dim is not None:
                    event[dim] = extent
                extents.append(extent)
            values[position] = self._draw_argument(
                argument,
                parameter,
                plate,
                scope,
                state,
                step,
                extents=tuple(extents),
            )
        if state.extent_solution:
            # A free name's typing may fix an extent an argument lowered
            # before it reads.
            substitution = self._extent_substitution(state)
            values = {
                position: substitute_value(value, substitution)
                for position, value in values.items()
            }
        arguments: list[tuple[str, Value]] = []
        for position, (parameter, _) in enumerate(pairs):
            value = values[position]
            if via is not None and group is not None:
                value = self._grouped(value, via, group, plate, scope, state, step)
            arguments.append((parameter.name, value))
        return arguments

    def _provisional_event(
        self: _Elaborator,
        record: DistributionFamily,
        step: SampleStep | ObserveStep | MarginalizeStep,
    ) -> bool:
        """Whether a draw's event axes come from an annotation its arguments may override.

        A vector family's ``: Axis`` annotation, with no axis-role
        clause, names the event axis only when the arguments fix no other
        event size; otherwise it is the plate the draws repeat over.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.

        Returns
        -------
        bool
            Whether the plate's event axes are such an annotation.
        """
        if (
            step.index is None
            or isinstance(step, MarginalizeStep)
            or step.axes is not None
            or record.event_rank == 0
            or not record.event_sources
            or _option_axes(step.options, "over")
            or _option_axes(step.options, "iid_over")
        ):
            return False
        morphism = self._program_morphisms.get(step.morphism)
        return morphism is None or not _option_axes(morphism.options, "over")

    def _tie_event_extents(
        self: _Elaborator,
        record: DistributionFamily,
        parameter: FamilyParameter,
        value: Value,
        event: dict[int, IndexTerm],
        step: SampleStep | ObserveStep | MarginalizeStep,
        scope: _Scope,
        state: _ProgramState,
    ) -> None:
        """Read a typed argument's tied axes against the draw's event extents.

        Each axis the family ties to a sample axis either fixes that
        axis's extent, when nothing has yet, or must equal it: an open
        extent on either side is fixed to the other, and two different
        fixed extents are refused.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        parameter : FamilyParameter
            The parameter the argument fills.
        value : Value
            The lowered argument.
        event : dict[int, IndexTerm]
            The extents of the sample's axes fixed so far, extended in
            place.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step, where a mismatch is reported.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If an axis of the argument has a fixed extent other than the
            sample axis's.
        """
        shape = tensor_shape(self._value_type(value, scope.context(), step))
        if shape is None or len(shape[1]) < parameter.rank:
            return
        own = shape[1][len(shape[1]) - parameter.rank :]
        for size, dim in zip(own, parameter.event_dims, strict=True):
            if dim is None:
                continue
            if dim not in event:
                event[dim] = self._resolved_extent(size, state)
                continue
            if not self._unify_extents([(size, event[dim])], scope, state):
                source = f"{value.local.name!r}, " if isinstance(value, Var) else ""
                self._fail(
                    step,
                    f"family {record.name!r} reads its {parameter.name!r} from "
                    f"{source}an axis of extent "
                    f"{render_static(self._resolved_extent(size, state))}, but "
                    f"the sample's axis it spans has extent "
                    f"{render_static(self._resolved_extent(event[dim], state))}",
                    code="qiec-program",
                )
            event[dim] = self._resolved_extent(event[dim], state)

    def _tied_free_name(
        self: _Elaborator,
        parameter: FamilyParameter,
        argument: DrawArg | str | float | _Spread,
        plate: PlateShape,
        scope: _Scope,
    ) -> str | None:
        """The free name filling a parameter whose axes the draw sizes.

        Parameters
        ----------
        parameter : FamilyParameter
            The parameter.
        argument : DrawArg | str | float | _Spread
            The argument filling it.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.

        Returns
        -------
        str | None
            The name, when the argument is a name nothing binds, the
            family ties the parameter's axes to its sample's, and the
            draw reads the name as a tensor over those axes: with no
            plate, one value per row of a plate with no event axes, or
            beside event axes a matrix, which no scalar broadcast stands
            in for. ``None`` otherwise.
        """
        if not parameter.event_dims:
            return None
        if isinstance(argument, DrawArgNamed):
            argument = argument.value
        if isinstance(argument, DrawArgName):
            name = argument.text
        elif (
            isinstance(argument, str)
            and not _is_number_text(argument)
            and "[" not in argument
        ):
            name = argument
        else:
            return None
        if scope.lookup(name) is not None or name in self._program_morphisms:
            return None
        if (
            plate.event
            and parameter.constraint not in _MATRIX_CONSTRAINTS
            and parameter.rank < 2
        ):
            return None
        return name

    def _bundle_vector_argument(
        self: _Elaborator,
        record: DistributionFamily,
        raw: list[DrawArg | str | float | _Spread],
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        state: _ProgramState,
    ) -> list[DrawArg | str | float | _Spread]:
        """Gather scalar literals filling a family's one vector parameter.

        A family whose only parameter is a vector, such as ``Dirichlet``,
        is written with its entries spread as positional literals:
        ``Dirichlet(1.0, 2.0, 3.0)`` is the three-simplex with that
        concentration. A single literal, ``Dirichlet(1.0)``, is the
        symmetric concentration on the simplex the step's plate or the
        program's declared codomain fixes the dimension of.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        raw : list[DrawArg | str | float | _Spread]
            The arguments as written.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The distribution's plate, whose event axis fixes the vector's
            length when the step annotates one.
        state : _ProgramState
            The program's accumulating state, whose declaration's
            codomain fixes the length otherwise.

        Returns
        -------
        list[DrawArg | str | float | _Spread]
            The arguments, the literals gathered into one list argument
            when the family takes a vector, a single literal spread over
            its dimension; unchanged otherwise.

        Raises
        ------
        QiecDiagnosticError
            If a single literal's vector has no dimension to take: the
            step annotates no axis and the program's codomain has no
            static extent.
        """
        if len(record.parameters) != 1 or record.parameters[0].rank != 1:
            return raw
        if record.parameters[0].constraint in ("sampleable", "transform"):
            return raw
        literals = [
            item
            for item in raw
            if isinstance(item, float | int | DrawArgScalar)
            or (isinstance(item, str) and _is_number_text(item))
        ]
        if not raw or len(literals) != len(raw):
            return raw
        items = tuple(
            item
            if isinstance(item, DrawArgScalar)
            else DrawArgScalar(value=float(item), line=step.line, col=step.col)
            for item in literals
        )
        if len(items) >= 2:
            return [DrawArgList(items=items, line=step.line, col=step.col)]
        if plate.event:
            length = plate.event[-1].size
            if not isinstance(length, IndexLiteral):
                self._fail(
                    step,
                    f"family {record.name!r} takes a vector whose length the "
                    "annotated axis leaves open; write its entries",
                    code="qiec-program",
                )
            dimension = length.value
        else:
            info = self._object_info(state.declaration.codomain, step)
            dimension = None if info.extent is None else _fixed_extent(info.extent)
            if dimension is None:
                self._fail(
                    step,
                    f"family {record.name!r} takes a vector whose length the "
                    "program's codomain leaves open; write its entries",
                    code="qiec-program",
                )
        return [_Spread(items[0].value, dimension)]

    def _draw_argument(
        self: _Elaborator,
        argument: DrawArg | str | float | _Spread,
        parameter: FamilyParameter | _ScalarParameter,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
        extents: tuple[IndexTerm, ...] | None = None,
    ) -> Value:
        """Lower one draw argument against the parameter it fills.

        Parameters
        ----------
        argument : DrawArg | str | float | _Spread
            The argument: a tagged node, wire text or a number, or a
            literal spread over a vector.
        parameter : FamilyParameter | _ScalarParameter
            The family parameter, which fixes the element type and rank.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        extents : tuple[IndexTerm, ...] | None
            The extents the draw fixes for each axis of a free name
            filling the parameter; ``None`` when it fixes none.

        Returns
        -------
        Value
            The lowered value.

        Raises
        ------
        QiecDiagnosticError
            If the argument is a nested distribution where a value is
            expected, or names something unreadable.
        """
        element = _element_of(parameter)
        integral = element == INT
        rank = parameter.rank
        constraint = parameter.constraint
        if constraint == "transform":
            return self._transform_chain(argument, node)
        if isinstance(argument, _Spread):
            # A single literal spread over a vector parameter is the
            # same number at every position of the vector.
            spread = Local(f"__spread_{state.current_site}", INT)
            return Comprehension(
                spread,
                _extent(argument.dimension),
                _literal(argument.value, integral),
                tensor_type(element, (_extent(argument.dimension),)),
            )
        if isinstance(argument, float | int):
            return _literal(float(argument), integral)
        if isinstance(argument, DrawArgScalar):
            return _literal(argument.value, integral)
        if isinstance(argument, DrawArgNamed):
            return self._draw_argument(
                argument.value, parameter, plate, scope, state, node, extents
            )
        elif isinstance(argument, DrawArgName):
            return self._named_argument(
                argument.text,
                element,
                rank,
                constraint,
                plate,
                scope,
                state,
                node,
                extents,
            )
        if isinstance(argument, DrawArgIndex):
            return self._indexed_argument(
                argument.name, argument.indices, element, scope, state, node
            )
        if isinstance(argument, DrawArgList) and constraint == "sampleable":
            return self._distribution_list(argument, scope, state, node)
        if isinstance(argument, DrawArgList):
            return self._list_argument(argument, element, scope, state, node)
        if isinstance(argument, DrawArgDist):
            if constraint != "sampleable":
                self._fail(
                    node,
                    f"a nested distribution {argument.family!r} fills a parameter "
                    "that takes a value",
                    code="qiec-program",
                )
            return self._nested_distribution(argument, scope, state, node)
        text = str(argument)
        if _is_number_text(text):
            return _literal(float(text), integral)
        if "[" in text:
            base, _, rest = text.partition("[")
            indices = tuple(item.rstrip("]") for item in rest.split("["))
            return self._indexed_argument(base, indices, element, scope, state, node)
        return self._named_argument(
            text, element, rank, constraint, plate, scope, state, node, extents
        )

    def _named_argument(
        self: _Elaborator,
        name: str,
        element: TypeApplication,
        rank: int,
        constraint: str,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
        extents: tuple[IndexTerm, ...] | None = None,
    ) -> Value:
        """Lower a bare name in argument position.

        A bound name is read; a morphism with a distribution initializer
        is that distribution when a sampleable is expected; any other
        name is a data input. Filling a parameter of a plated draw whose
        plate names no event axes, it holds one value per row: its type is
        the plate's first batch axis, then the parameter's own axes, and a
        datum without the plate axis is a value every row shares. Filling
        a tensor parameter of a draw with no batch axes, it is one tensor
        over the parameter's own axes. A parameter's own axes are the
        ``extents`` the draw fixes, when given, and otherwise one open
        extent each. Filling any other parameter of a draw whose plate
        names event axes (`[over=...]`), it is one value of the
        parameter's element type, which the plate broadcasts; filling a
        scalar parameter of a draw with no plate, one value of its
        element type.

        Parameters
        ----------
        name : str
            The name.
        element : TypeApplication
            The parameter's element type.
        rank : int
            The parameter's rank.
        constraint : str
            The parameter's constraint.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        extents : tuple[IndexTerm, ...] | None
            The extent of each of the parameter's own axes, as the draw
            fixes them; ``None`` when the draw fixes none.

        Returns
        -------
        Value
            The lowered value.
        """
        local = scope.lookup(name)
        if local is not None:
            return Var(local)
        morphism = self._program_morphisms.get(name)
        if morphism is not None and constraint == "sampleable":
            return self._morphism_distribution(morphism, name, scope, state, node)
        if rank > 0 and not plate.batch and (plate.empty or extents is not None):
            # A draw with no batch axes reads the input as one tensor over
            # the parameter's own axes. An axis the draw does not fix is a
            # static index parameter of the program: the input is typed
            # over a fresh index variable the program's telescope binds,
            # and a run reads its value off the data it is given.
            own = (
                extents
                if extents is not None
                else self._open_extents(name, rank, state)
            )
            return Var(
                self._declare_parameter(
                    name,
                    tensor_type(element, own),
                    "data",
                    scope,
                    state,
                    node,
                )
            )
        if plate.batch and not plate.event:
            # A parameter of a plated draw takes one value per row, as a
            # name a let feeds to the draw does: the host datum ranges over
            # the plate's first batch axis, then over the parameter's own
            # axes. A datum without the plate axis is one value every row
            # shares.
            axis = plate.batch[0]
            own = (
                extents
                if extents is not None
                else self._open_extents(name, rank, state)
            )
            dimensions_per_row: list[IndexTerm] = [axis.size, *own]
            axes_per_row: list[str] = [
                axis.name,
                *self._axes_of_type(tensor_type(element, own))[:rank],
            ]
            local = self._declare_parameter(
                name,
                tensor_type(element, tuple(dimensions_per_row)),
                "data",
                scope,
                state,
                node,
                axes=tuple(axes_per_row),
            )
            state.per_row.add(name)
            state.batch_shapes[name] = (axis.size,)
            return Var(local)
        if rank > 0 and extents is not None:
            # A tensor parameter whose axes the draw's event fixes, such
            # as a Cholesky factor beside an ``[over=...]`` plate, is one
            # tensor every row shares.
            return Var(
                self._declare_parameter(
                    name,
                    tensor_type(element, extents),
                    "data",
                    scope,
                    state,
                    node,
                )
            )
        return Var(self._declare_parameter(name, element, "data", scope, state, node))

    def _open_extents(
        self: _Elaborator, name: str, rank: int, state: _ProgramState
    ) -> tuple[IndexTerm, ...]:
        """Fresh open extents for each axis of a free name's value.

        Parameters
        ----------
        name : str
            The free name.
        rank : int
            How many axes its value has.
        state : _ProgramState
            The program's accumulating state, whose telescope binds each
            extent.

        Returns
        -------
        tuple[IndexTerm, ...]
            One index variable per axis, named after the name and the
            axis's position, or the term a later use fixed it to.
        """
        return tuple(
            self._open_extent(
                f"{name}_extent" if rank == 1 else f"{name}_extent{position}", state
            )
            for position in range(rank)
        )

    def _open_extent(self: _Elaborator, name: str, state: _ProgramState) -> IndexTerm:
        """The open extent of a given name, bound by the program's telescope.

        Parameters
        ----------
        name : str
            The extent's name.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        IndexTerm
            The extent's index variable, or the term a use has fixed it
            to.
        """
        if name in state.extent_solution:
            return state.extent_solution[name]
        if name not in state.extents:
            state.extents[name] = IndexBinder(name, NAT)
            state.open_extents.add(name)
        return IndexVariable(name, NAT)

    @staticmethod
    def _resolved_extent(term: IndexTerm, state: _ProgramState) -> IndexTerm:
        """An extent with every open extent a use has fixed replaced.

        Parameters
        ----------
        term : IndexTerm
            The extent.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        IndexTerm
            The term the extent equals.
        """
        if isinstance(term, IndexVariable) and term.identity is None:
            return state.extent_solution.get(term.name, term)
        return term

    def _unify_extents(
        self: _Elaborator,
        pairs: Sequence[tuple[IndexTerm, IndexTerm]],
        scope: _Scope,
        state: _ProgramState,
    ) -> bool:
        """Fix open extents so that each pair of extents is one extent.

        An open extent a free name's typing introduced equals whatever it
        meets: a declared object's size, a literal, or another extent.
        Two extents neither of which is open, such as two different
        literals, do not unify. Each fix is applied to every name bound
        so far, so later uses read the fixed types.

        Parameters
        ----------
        pairs : Sequence[tuple[IndexTerm, IndexTerm]]
            The extents that must be equal.
        scope : _Scope
            The scope in force, whose bindings and enclosing ones take
            the fixed types.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        bool
            Whether every pair unifies. Fixes found before a pair that
            does not unify are kept.
        """
        changed = False
        unified = True
        for left, right in pairs:
            left = self._resolved_extent(left, state)
            right = self._resolved_extent(right, state)
            if left == right:
                continue
            if isinstance(left, IndexVariable) and left.name in state.open_extents:
                variable, term = left, right
            elif isinstance(right, IndexVariable) and right.name in state.open_extents:
                variable, term = right, left
            else:
                unified = False
                continue
            fixed = StaticSubstitution((), ((variable.name, term),), ())
            state.extent_solution = {
                name: substitute_index(value, fixed)
                for name, value in state.extent_solution.items()
            }
            state.extent_solution[variable.name] = term
            changed = True
        if changed:
            self._apply_extent_solution(scope, state)
        return unified

    def _unify_types(
        self: _Elaborator,
        left: TypeExpr,
        right: TypeExpr,
        scope: _Scope,
        state: _ProgramState,
    ) -> bool:
        """Fix open extents so that two tensor types are one type.

        Parameters
        ----------
        left : TypeExpr
            One type.
        right : TypeExpr
            The other.
        scope : _Scope
            The scope in force.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        bool
            Whether the types are equal once their open extents are
            fixed: tensors of one element and rank whose extents unify.
        """
        left_shape = tensor_shape(left)
        right_shape = tensor_shape(right)
        if (
            left_shape is None
            or right_shape is None
            or left_shape[0] != right_shape[0]
            or len(left_shape[1]) != len(right_shape[1])
        ):
            return False
        return self._unify_extents(
            list(zip(left_shape[1], right_shape[1], strict=True)), scope, state
        )

    def _extent_substitution(
        self: _Elaborator, state: _ProgramState
    ) -> StaticSubstitution:
        """The substitution fixing every open extent a use has fixed.

        Parameters
        ----------
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        StaticSubstitution
            Each fixed extent to its term.
        """
        return StaticSubstitution((), tuple(state.extent_solution.items()), ())

    def _apply_extent_solution(
        self: _Elaborator, scope: _Scope, state: _ProgramState
    ) -> None:
        """Retype every name bound so far at the fixed extents.

        Parameters
        ----------
        scope : _Scope
            The scope in force; it and every enclosing scope are retyped.
        state : _ProgramState
            The program's accumulating state.
        """
        substitution = self._extent_substitution(state)
        current: _Scope | None = scope
        while current is not None:
            for name, local in list(current.locals.items()):
                fixed = substitute_type(local.type, substitution)
                if fixed != local.type:
                    current.locals[name] = Local(local.name, fixed)
            current = current.outer

    def _indexed_argument(
        self: _Elaborator,
        base: str,
        indices: Sequence[str],
        element: TypeApplication,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower ``base[i][j]`` to nested gathers.

        Parameters
        ----------
        base : str
            The indexed name.
        indices : Sequence[str]
            The index names, outermost first.
        element : TypeApplication
            The parameter's element type, which types an unbound base.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The selection.

        Raises
        ------
        QiecDiagnosticError
            If an index is unbound, or the base is unbound and its
            alphabet cannot be read off the index.
        """
        index_values: list[Value] = []
        for index in indices:
            local = scope.lookup(index)
            if local is None:
                if _is_number_text(index):
                    index_values.append(LiteralValue(int(float(index)), INT))
                    continue
                self._fail(
                    node,
                    f"index {index!r} of {base!r} is not bound by a step",
                    code="qiec-program",
                )
            index_values.append(Var(local))
        source = scope.lookup(base)
        pinned = self._pinned_types.get(base) if source is None else None
        if pinned is not None:
            source = self._declare_parameter(base, pinned, "data", scope, state, node)
        if source is None:
            alphabet = state.alphabets.get(indices[0])
            if alphabet is None:
                self._fail(
                    node,
                    f"input {base!r} is indexed by {indices[0]!r}, whose alphabet is "
                    "unknown; index it by a class-valued step",
                    code="qiec-program",
                )
            source = self._declare_parameter(
                base, tensor_type(element, (alphabet.size,)), "data", scope, state, node
            )
        value: Value = Var(source)
        context = scope.context()
        for index_value in index_values:
            source_type = self._value_type(value, context, node)
            shape = tensor_shape(source_type)
            if shape is None:
                self._fail(
                    node,
                    f"{base!r} is indexed but is not a Tensor",
                    code="qiec-program",
                )
            index_type = self._value_type(index_value, context, node)
            rest = shape[1][1:]
            if index_type == INT:
                result: TypeExpr = shape[0] if not rest else tensor_type(shape[0], rest)
            else:
                index_shape = tensor_shape(index_type)
                if index_shape is None or index_shape[0] != INT:
                    self._fail(
                        node,
                        f"index into {base!r} is not an Int or a Tensor of them",
                        code="qiec-program",
                    )
                result = tensor_type(shape[0], (*index_shape[1], *rest))
            value = Gather(value, index_value, result)
        return value

    def _list_argument(
        self: _Elaborator,
        argument: DrawArgList,
        element: TypeApplication,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower a list argument to a tensor literal.

        Parameters
        ----------
        argument : DrawArgList
            The list.
        element : TypeApplication
            The entry type.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The tensor.

        Raises
        ------
        QiecDiagnosticError
            If the list is empty or its entries differ in type.
        """
        if not argument.items:
            self._fail(
                node,
                "an empty list fills no distribution parameter",
                code="qiec-program",
            )
        items = [
            self._list_argument(item, element, scope, state, node)
            if isinstance(item, DrawArgList)
            else self._draw_argument(
                item, _ScalarParameter(element), PlateShape(), scope, state, node
            )
            for item in argument.items
        ]
        context = scope.context()
        types = [self._value_type(item, context, node) for item in items]
        if any(item != types[0] for item in types):
            self._fail(node, "list entries have differing types", code="qiec-program")
        inner = tensor_shape(types[0])
        if inner is None:
            result = tensor_type(types[0], (_extent(len(items)),))
        else:
            result = tensor_type(inner[0], (_extent(len(items)), *inner[1]))
        return TensorValue(tuple(items), result)

    def _transform_chain(self: _Elaborator, argument: object, node: object) -> Value:
        """Lower a transform name in a ``Transformed`` construction.

        Parameters
        ----------
        argument : object
            The argument: a name of a bijector of the operator algebra,
            such as ``Exp``, or a string literal naming a chain.
        node : object
            The source node.

        Returns
        -------
        Value
            The chain as a string literal, in the reference backend's
            vocabulary.

        Raises
        ------
        QiecDiagnosticError
            If the name is not a transform the kernel knows.
        """
        name = argument.text if isinstance(argument, DrawArgName) else str(argument)
        chain = BIJECTOR_TRANSFORMS.get(name)
        if chain is None:
            known = ", ".join(sorted(BIJECTOR_TRANSFORMS))
            self._fail(
                node,
                f"{name!r} is not a transform a pushforward can apply; the "
                f"transforms are {known}",
                code="qiec-program",
            )
        return LiteralValue(chain, STRING)

    def _distribution_list(
        self: _Elaborator,
        argument: DrawArgList,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower a list of distributions to a tensor of sampleables.

        Parameters
        ----------
        argument : DrawArgList
            The list, whose entries are family applications.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            A tensor of sampleables of one element type, with one axis.

        Raises
        ------
        QiecDiagnosticError
            If the list is empty, an entry is not a distribution, or the
            entries sample different types.
        """
        if not argument.items:
            self._fail(
                node, "a mixture needs at least one component", code="qiec-program"
            )
        items: list[Value] = []
        for item in argument.items:
            if not isinstance(item, DrawArgDist):
                self._fail(
                    node,
                    "every component of a mixture must be a distribution",
                    code="qiec-program",
                )
            items.append(self._nested_distribution(item, scope, state, node))
        context = scope.context()
        types = [self._value_type(item, context, node) for item in items]
        if any(item != types[0] for item in types):
            # An atom written as a real literal beside an integer-valued
            # component is an atom of the integers: a count model with a
            # spike at zero writes ``PointMass(0.0)``.
            elements = {sampled_element(item) for item in types}
            if elements == {INT, REAL}:
                items = [
                    self._nested_distribution(item, scope, state, node, element=INT)
                    if isinstance(item, DrawArgDist)
                    and item.family == "PointMass"
                    and sampled_element(self._value_type(lowered, context, node))
                    == REAL
                    else lowered
                    for item, lowered in zip(argument.items, items, strict=True)
                ]
                types = [self._value_type(item, context, node) for item in items]
        if any(item != types[0] for item in types):
            self._fail(
                node, "mixture components sample differing types", code="qiec-program"
            )
        return TensorValue(tuple(items), tensor_type(types[0], (_extent(len(items)),)))

    def _nested_distribution(
        self: _Elaborator,
        argument: DrawArgDist,
        scope: _Scope,
        state: _ProgramState,
        node: object,
        element: TypeApplication | None = None,
    ) -> DistributionValue:
        """Lower a nested family application in argument position.

        Parameters
        ----------
        argument : DrawArgDist
            The application.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        element : TypeApplication | None
            The element type an atom's value is read at, for a
            ``PointMass`` whose siblings fix it; ``None`` reads the
            family's own.

        Returns
        -------
        DistributionValue
            The unplated construction.

        Raises
        ------
        QiecDiagnosticError
            If the family is unknown.
        """
        record = FAMILIES.get(OPERATOR_ALIASES.get(argument.family, argument.family))
        if record is None:
            self._fail(
                node,
                f"family:{argument.family}: not in the semantic family registry",
                code="qiec-program",
            )
        schemas = family_parameterizations(argument.family, record.parameter_names)
        schema_text = render_parameterizations(schemas)
        if any(not isinstance(item, DrawArgNamed) for item in argument.args):
            self._fail(
                node,
                f"family {argument.family!r} arguments are keyword-only; "
                f"use one complete schema: {schema_text}",
                code="qiec-program",
            )
        source = {item.parameter: item.value for item in argument.args}
        if len(source) != len(argument.args):
            self._fail(
                node,
                f"family {argument.family!r} is given a parameter twice",
                code="qiec-program",
            )
        selected = next(
            (schema for schema in schemas if set(schema) == set(source)), None
        )
        if selected is None:
            self._fail(
                node,
                f"family {argument.family!r} requires exactly one complete "
                f"parameter schema {schema_text}; got ({', '.join(source)})",
                code="qiec-program",
            )
        arguments = []
        for parameter_name in selected:
            try:
                parameter = _semantic_family_parameter(
                    record, parameter_name, argument.family
                )
            except KeyError:
                self._fail(
                    node,
                    f"family {record.name!r} has no parameter {parameter_name!r}",
                    code="qiec-program",
                )
            arguments.append(
                (
                    parameter.name,
                    self._draw_argument(
                        source[parameter_name],
                        _ScalarParameter(element)
                        if element is not None and record.name == "PointMass"
                        else parameter,
                        PlateShape(),
                        scope,
                        state,
                        node,
                    ),
                )
            )
        provisional = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            sampleable_type(REAL),
            _origin_at(
                self,
                ("programs", state.declaration.name, "nested", argument.family),
                "distribution",
                node,
            ),
        )
        return DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            self._plated_type(provisional, scope, node),
            provisional.origin,
        )

    def _morphism_distribution(
        self: _Elaborator,
        morphism: MorphismDecl,
        name: str,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> DistributionValue:
        """The distribution a morphism's initializer names, as a value.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.
        name : str
            Its name.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        DistributionValue
            The construction from the initializer's family and arguments.

        Raises
        ------
        QiecDiagnosticError
            If the morphism has no distribution initializer.
        """
        try:
            resolved = resolve_step_dist(
                name,
                None,
                morphisms=self._program_morphisms,
                lets=self._program_lets,
                family_registry=frozenset(FAMILIES),
                target="qvr-qiec",
            )
        except StepResolutionError as error:
            self._fail(node, error.detail, code="qiec-program")
        record = FAMILIES.get(resolved.family)
        if record is None:
            self._fail(
                node,
                f"family:{resolved.family}: not in the semantic family registry",
                code="qiec-program",
            )
        init = morphism.init_family
        if init is not None and init.args:
            schemas = family_parameterizations(init.family, record.parameter_names)
            schema_text = render_parameterizations(schemas)
            if any(not isinstance(item, DrawArgNamed) for item in init.args):
                self._fail(
                    node,
                    f"family {record.name!r} arguments are keyword-only; "
                    f"use one complete schema: {schema_text}",
                    code="qiec-program",
                )
            source = {item.parameter: item.value for item in init.args}
            selected = next(
                (schema for schema in schemas if set(schema) == set(source)), None
            )
            if selected is None or len(source) != len(init.args):
                self._fail(
                    node,
                    f"family {record.name!r} requires exactly one complete "
                    f"parameter schema {schema_text}",
                    code="qiec-program",
                )
            arguments = []
            for parameter_name in selected:
                parameter = _semantic_family_parameter(
                    record, parameter_name, init.family
                )
                arguments.append(
                    (
                        parameter.name,
                        self._draw_argument(
                            source[parameter_name],
                            parameter,
                            PlateShape(),
                            scope,
                            state,
                            node,
                        ),
                    )
                )
        else:
            arguments = [
                (
                    parameter.name,
                    self._draw_argument(
                        item, parameter, PlateShape(), scope, state, node
                    ),
                )
                for parameter, item in zip(
                    record.parameters, resolved.args or (), strict=False
                )
            ]
        provisional = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            sampleable_type(REAL),
            _origin_at(
                self,
                ("programs", state.declaration.name, "morphisms", name),
                "distribution",
                node,
            ),
        )
        return DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            self._plated_type(provisional, scope, node),
            provisional.origin,
        )

    def _grouped(
        self: _Elaborator,
        value: Value,
        via: tuple[str, ...],
        group: PlateAxis,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Re-index a grouped argument to the observation rows.

        Parameters
        ----------
        value : Value
            The argument.
        via : tuple[str, ...]
            The index map's names, one per factor of the group.
        group : PlateAxis
            The grouping axis.
        plate : PlateShape
            The observation's plate, whose one batch axis is the rows.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The argument gathered by the index map when its leading
            dimension is the group axis, else unchanged.

        Raises
        ------
        QiecDiagnosticError
            If the observation is not plated over one row axis.
        """
        shape = tensor_shape(self._value_type(value, scope.context(), node))
        if shape is None or not shape[1] or shape[1][0] != group.size:
            return value
        if len(plate.batch) != 1:
            self._fail(
                node,
                f"index map {'*'.join(via)!r} needs an observation plated over "
                "one row axis",
                code="qiec-program",
            )
        rows = plate.batch[0].size
        grouped = _weight_scope(scope)
        assert grouped is not None
        index_map = self._index_map(
            via, rows, plate.batch[0].name, grouped, scope, state, node
        )
        return Gather(value, index_map, tensor_type(shape[0], (rows, *shape[1][1:])))

    def _refined(
        self: _Elaborator,
        value: Value,
        refinement: tuple[PlateAxis, Value],
        plate: PlateShape,
        scope: _Scope,
        node: object,
    ) -> Value:
        """Re-index an argument shaped by the outer group to a refining group.

        Parameters
        ----------
        value : Value
            The argument.
        refinement : tuple[PlateAxis, Value]
            The outer group and the index projecting the refining group's
            positions onto it.
        plate : PlateShape
            The latent's plate, whose batch axis is the refining group.
        scope : _Scope
            The scope.
        node : object
            The source node.

        Returns
        -------
        Value
            The argument gathered by the projection when its leading
            dimension is the outer group, else unchanged.
        """
        outer, projection = refinement
        shape = tensor_shape(self._value_type(value, scope.context(), node))
        if shape is None or not shape[1] or shape[1][0] != outer.size:
            return value
        assert len(plate.batch) == 1
        rows = plate.batch[0].size
        return Gather(value, projection, tensor_type(shape[0], (rows, *shape[1][1:])))

    def _index_map(
        self: _Elaborator,
        via: tuple[str, ...],
        rows: IndexTerm,
        rows_name: str,
        grouped: _Scope,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """The index of each observation row's group.

        A single index map is the program input of that name; a product
        index map into a product group is the row-major flattening of
        its factors' indices, the last factor varying fastest.

        Parameters
        ----------
        via : tuple[str, ...]
            The index map's names, one per factor of the group.
        rows : IndexTerm
            The observation's row extent.
        rows_name : str
            The row axis's name.
        grouped : _Scope
            The grouped marginalization scope, whose factors the names
            fill.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            A ``Tensor[Int]`` over the rows.

        Raises
        ------
        QiecDiagnosticError
            If the index map names fewer or more factors than the group
            has, or a name is bound by a step at another type.
        """
        factors = grouped.factors
        assert grouped.group is not None
        if len(via) != len(factors):
            self._fail(
                node,
                f"index map {'*'.join(via)!r} names {len(via)} factor(s) but the "
                f"group {grouped.group.name!r} has {len(factors)}",
                code="qiec-program",
            )
        index_type = tensor_type(INT, (rows,))
        flat: Value | None = None
        for name, factor in zip(via, factors, strict=True):
            bound = scope.lookup(name)
            if bound is not None and bound.type != index_type:
                self._fail(
                    node,
                    f"index map {name!r} is bound by a step at "
                    f"{self._render(bound.type)}, not a Tensor[Int] over the rows",
                    code="qiec-program",
                )
            if bound is not None:
                entry: Value = Var(bound)
            else:
                parameter = self._declare_parameter(
                    name, index_type, "index_map", scope, state, node, axes=(rows_name,)
                )
                entry = Var(parameter)
            if flat is None:
                flat = entry
                continue
            assert isinstance(factor.size, IndexLiteral)
            path = ("programs", state.declaration.name, "index_maps", name)
            scaled = self._primitive(
                "mul_int",
                (flat, LiteralValue(factor.size.value, INT)),
                node,
                (*path, "scale"),
                shape=(rows,),
            )
            flat = self._primitive(
                "add_int", (scaled, entry), node, (*path, "offset"), shape=(rows,)
            )
        assert flat is not None
        return flat

    def _widen_alphabet(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl | None,
        arguments: list[tuple[str, Value]],
        scope: _Scope,
        step: object,
    ) -> list[tuple[str, Value]]:
        """Record a class-index draw's alphabet for the steps indexing by it.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl | None
            The morphism drawn through, whose finite codomain names the
            alphabet.
        arguments : list[tuple[str, Value]]
            The lowered arguments, whose probability vector's length
            names it otherwise.
        scope : _Scope
            The scope.
        step : object
            The step.

        Returns
        -------
        list[tuple[str, Value]]
            The arguments, unchanged.
        """
        del step
        alphabet = _CLASS_INDEX_FAMILIES.get(record.name)
        if alphabet is None:
            return arguments
        state = self._program_state
        assert state is not None
        for name, value in arguments:
            if name != alphabet:
                continue
            shape = tensor_shape(self._value_type(value, scope.context(), None))
            if shape is not None and shape[1]:
                state.pending_alphabet = PlateAxis("classes", shape[1][-1])
        if morphism is not None and isinstance(morphism.codomain, TypeName):
            info = self._program_objects.get(morphism.codomain.name)
            if info is not None and info.finite and info.extent is not None:
                state.pending_alphabet = PlateAxis(
                    morphism.codomain.name, _extent(info.extent)
                )
        return arguments

    # ------------------------------------------------------------------
    # structured and mapped morphisms

    def _gp_grid(self: _Elaborator, morphism: MorphismDecl, step: object) -> PlateAxis:
        """The grid axis a GP morphism's draws range over.

        Parameters
        ----------
        morphism : MorphismDecl
            The ``~ GP`` morphism.
        step : object
            The step, for diagnostics.

        Returns
        -------
        PlateAxis
            The first finite factor of the morphism's domain.

        Raises
        ------
        QiecDiagnosticError
            If the domain has no finite factor.
        """
        for factor in _factors(morphism.domain):
            if not isinstance(factor, TypeName):
                continue
            info = self._program_objects.get(factor.name)
            if info is not None and info.finite and info.extent is not None:
                return PlateAxis(factor.name, _extent(info.extent))
        self._fail(
            step,
            f"GP morphism {morphism.names[0]!r} has no finite domain axis for its grid",
            code="qiec-program",
        )

    def _conditional_map(
        self: _Elaborator, morphism: MorphismDecl, family: str
    ) -> bool:
        """Whether a kernel morphism reads its arguments off a parameter map.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.
        family : str
            The family it draws from.

        Returns
        -------
        bool
            ``True`` for a kernel-role morphism with a bare ``~ Family``
            initializer of a family with conditional heads, none of whose
            heads the option block writes.
        """
        role = _option_value(morphism.options, "role")
        if role == "embed":
            return family == "Normal"
        if role is not None and role != "kernel":
            return False
        heads = _CONDITIONAL_HEADS.get(family)
        if heads is None:
            return False
        if _bare_init_family(morphism) != family:
            return False
        names = {head for head, _ in heads}
        return not any(
            entry.key in names and isinstance(entry.value, OptionNumber | OptionString)
            for entry in morphism.options
        )

    def _mapped_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl,
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
    ) -> list[tuple[str, Value]]:
        """The arguments of a draw through a conditional kernel morphism.

        The morphism's weight and bias are program inputs; each head of
        the family reads its row block of the map applied to the
        conditioning row, which is the step's arguments or, absent any,
        the program's domain inputs. A morphism over a finite domain
        reads a table instead, one row per element of the domain, at the
        element the program is applied to.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl
            The morphism.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        list[tuple[str, Value]]
            One affine head per family parameter the map supplies.

        Raises
        ------
        QiecDiagnosticError
            If the codomain has no real width, the conditioning row is
            not made of bindings, or its width differs from the domain's.
        """
        width = self._head_width(record, morphism, step)
        heads = _CONDITIONAL_HEADS[record.name]
        finite = self._finite_domain(morphism)
        if finite is not None and len(step.args or ()) <= 1:
            return self._table_arguments(
                morphism, step, heads, width, finite, plate, scope, state
            )
        declared = 0
        for factor in _factors(morphism.domain):
            factor_info = (
                self._program_objects.get(factor.name)
                if isinstance(factor, TypeName)
                else None
            )
            factor_width = (
                None
                if factor_info is None or factor_info.real_width is None
                else _fixed_extent(factor_info.real_width)
            )
            if factor_width is None:
                self._fail(
                    step,
                    f"morphism {step.morphism!r} has a domain factor with no fixed "
                    "real width",
                    code="qiec-program",
                )
            declared += factor_width
        rows_axis = _rows_axis(plate, step)
        sources: list[Value] = []
        widths: list[int] = []
        batched = False
        row = Local(f"__{step.morphism}_row", INT)
        if step.args:
            for argument in step.args:
                if not isinstance(argument, DrawArgName):
                    self._fail(
                        step,
                        f"morphism {step.morphism!r} conditions on a value, not a {argument.kind} argument",
                        code="qiec-program",
                    )
                local = scope.lookup(argument.text)
                if local is None:
                    if rows_axis is None or len(step.args) != 1:
                        self._fail(
                            step,
                            f"conditioning binding {argument.text!r} is not bound",
                            code="qiec-program",
                        )
                    # An unbound conditioning name under a plate is a
                    # data input with one row of the morphism's domain
                    # per position of the plate.
                    local = self._declare_parameter(
                        argument.text,
                        tensor_type(REAL, (rows_axis.size, _extent(declared))),
                        "data",
                        scope,
                        state,
                        step,
                    )
                shape = tensor_shape(local.type)
                if local.type == REAL:
                    widths.append(1)
                    sources.append(Var(local))
                elif (
                    shape is not None
                    and len(shape[1]) == 1
                    and isinstance(shape[1][0], IndexLiteral)
                ):
                    widths.append(int(shape[1][0].value))
                    sources.append(Var(local))
                elif (
                    shape is not None
                    and rows_axis is not None
                    and len(shape[1]) == 2
                    and shape[1][0] == rows_axis.size
                    and isinstance(shape[1][1], IndexLiteral)
                ):
                    # A row of the binding per position of the plate:
                    # the map applies at each row.
                    batched = True
                    widths.append(int(shape[1][1].value))
                    sources.append(
                        Gather(Var(local), Var(row), tensor_type(REAL, (shape[1][1],)))
                    )
                elif (
                    shape is not None
                    and len(shape[1]) == 2
                    and isinstance(shape[1][0], IndexLiteral)
                    and isinstance(shape[1][1], IndexLiteral)
                ):
                    # A bundle of rows, as a fan or a product leaves
                    # behind, conditions on the rows in order.
                    for position in range(int(shape[1][0].value)):
                        widths.append(int(shape[1][1].value))
                        sources.append(
                            Gather(
                                Var(local),
                                LiteralValue(position, INT),
                                tensor_type(REAL, (shape[1][1],)),
                            )
                        )
                else:
                    self._fail(
                        step,
                        f"conditioning binding {argument.text!r} has no static width",
                        code="qiec-program",
                    )
        else:
            for name, role in state.parameters:
                if role != "domain":
                    continue
                local = scope.lookup(name)
                assert local is not None
                shape = tensor_shape(local.type)
                if (
                    shape is None
                    or len(shape[1]) != 1
                    or not isinstance(shape[1][0], IndexLiteral)
                ):
                    continue
                widths.append(int(shape[1][0].value))
                sources.append(Var(local))
        if declared != sum(widths):
            self._fail(
                step,
                f"morphism {step.morphism!r} reads a {declared}-wide row but is "
                f"conditioned on {sum(widths)} coordinates",
                code="qiec-program",
            )
        # A network's hidden layers each read the previous layer's
        # activations through an affine map followed by a tanh; the
        # family's heads read the last layer.
        fan_in = declared
        for depth, hidden in enumerate(self._hidden_widths(morphism, step)):
            layer_weight = self._declare_parameter(
                f"{step.morphism}_param_layer{depth}_weight",
                tensor_type(REAL, (_extent(hidden), _extent(fan_in))),
                "weight",
                scope,
                state,
                step,
                axes=(
                    f"{step.morphism}_layer{depth}_row",
                    f"{step.morphism}_layer{depth}_col",
                ),
            )
            layer_bias = self._declare_parameter(
                f"{step.morphism}_param_layer{depth}_bias",
                tensor_type(REAL, (_extent(hidden),)),
                "bias",
                scope,
                state,
                step,
                axes=(f"{step.morphism}_layer{depth}_row",),
            )
            hidden_type = tensor_type(REAL, (_extent(hidden),))
            activation = self._primitive(
                "tanh",
                (
                    AffineMap(
                        Var(layer_weight),
                        Var(layer_bias),
                        tuple(sources),
                        0,
                        hidden,
                        "identity",
                        hidden_type,
                    ),
                ),
                step,
                ("programs", state.declaration.name, "layers", step.morphism, depth),
                (_extent(hidden),),
            )
            sources = [activation]
            fan_in = hidden
        rows = width * len(heads)
        weight = self._declare_parameter(
            f"{step.morphism}_param_weight",
            tensor_type(REAL, (_extent(rows), _extent(fan_in))),
            "weight",
            scope,
            state,
            step,
            axes=(f"{step.morphism}_param_row", f"{step.morphism}_param_col"),
        )
        bias = self._declare_parameter(
            f"{step.morphism}_param_bias",
            tensor_type(REAL, (_extent(rows),)),
            "bias",
            scope,
            state,
            step,
            axes=(f"{step.morphism}_param_row",),
        )
        arguments: list[tuple[str, Value]] = []
        for index, (name, transform) in enumerate(heads):
            result: TypeExpr = (
                REAL if width == 1 else tensor_type(REAL, (_extent(width),))
            )
            head: Value = AffineMap(
                Var(weight),
                Var(bias),
                tuple(sources),
                index * width,
                width,
                _map_transform(transform),
                result,
            )
            head = self._transformed_head(head, transform, width, step, state, name)
            if batched:
                assert rows_axis is not None
                head = _per_row(head, width, row, rows_axis)
            arguments.append((name, head))
        return arguments

    def _head_width(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl,
        step: SampleStep | ObserveStep | MarginalizeStep,
    ) -> int:
        """The width of one head of a kernel morphism's parameter map.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl
            The morphism.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.

        Returns
        -------
        int
            The codomain's real width, or, for a family whose head is a
            row of logits, the number of the codomain's elements, one
            for a two-element codomain of ``Bernoulli``.

        Raises
        ------
        QiecDiagnosticError
            If the codomain has no width the head can fill.
        """
        codomain = morphism.codomain
        info = (
            self._program_objects.get(codomain.name)
            if isinstance(codomain, TypeName)
            else None
        )
        if record.name in _LOGIT_HEAD_FAMILIES:
            classes = (
                None
                if info is None or not info.finite or info.extent is None
                else _fixed_extent(info.extent)
            )
            if classes is None:
                self._fail(
                    step,
                    f"morphism {step.morphism!r} draws from {record.name} onto a "
                    "codomain that is not a finite object of fixed size",
                    code="qiec-program",
                )
            if record.name == "Bernoulli":
                if classes != 2:
                    self._fail(
                        step,
                        f"morphism {step.morphism!r} draws from Bernoulli onto a "
                        f"codomain of {classes} elements, not two",
                        code="qiec-program",
                    )
                return 1
            return classes
        width = (
            None
            if info is None or info.real_width is None
            else _fixed_extent(info.real_width)
        )
        if width is None:
            self._fail(
                step,
                f"morphism {step.morphism!r} maps its parameters onto a codomain with no "
                "fixed real width",
                code="qiec-program",
            )
        return width

    def _transformed_head(
        self: _Elaborator,
        head: Value,
        transform: HeadTransform,
        width: int,
        step: SampleStep | ObserveStep | MarginalizeStep,
        state: _ProgramState,
        name: str,
    ) -> Value:
        """Apply a head's transform the map itself does not carry.

        Parameters
        ----------
        head : Value
            The head, read as it is off the map.
        transform : HeadTransform
            The head's transform.
        width : int
            The head's width.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        state : _ProgramState
            The program's accumulating state.
        name : str
            The family parameter the head fills.

        Returns
        -------
        Value
            The head under a softplus lifted or shifted, under a
            sigmoid, or unchanged when the map carries the transform.
        """
        if transform in ("identity", "exp_floor"):
            return head
        path = ("programs", state.declaration.name, "heads", step.morphism, name)
        shape: tuple[IndexTerm, ...] | None = None if width == 1 else (_extent(width),)
        if transform == "sigmoid":
            return self._primitive("sigmoid", (head,), step, path, shape)
        lifted = self._primitive("softplus", (head,), step, (*path, "softplus"), shape)
        offset = SOFTPLUS_LIFT if transform == "softplus" else SOFTPLUS_SHIFT
        return self._primitive(
            "add_real",
            (lifted, LiteralValue(offset, REAL)),
            step,
            (*path, "offset"),
            shape,
        )

    def _hidden_widths(
        self: _Elaborator, morphism: MorphismDecl, node: object
    ) -> tuple[int, ...]:
        """The hidden layer widths a network-parameterized morphism declares.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.
        node : object
            The source node.

        Returns
        -------
        tuple[int, ...]
            Empty for a linear map; for ``[param_source=mlp]`` the
            widths ``mlp(a, b, ...)`` lists, else the ``hidden_dim``
            option's one width or list of widths, else two layers of
            sixty-four.

        Raises
        ------
        QiecDiagnosticError
            If a width is not a positive integer.
        """
        source = _option_entry(morphism.options, "param_source")
        if source is None:
            return ()
        kind = (
            source.func
            if isinstance(source, OptionCall)
            else source.value
            if isinstance(source, OptionName)
            else None
        )
        if kind != "mlp":
            return ()
        raw: list[OptionValue] = []
        if isinstance(source, OptionCall) and source.args:
            raw = list(source.args)
        else:
            hidden = _option_entry(morphism.options, "hidden_dim")
            if isinstance(hidden, OptionList):
                raw = list(hidden.items)
            elif hidden is not None:
                raw = [hidden]
        if not raw:
            return (64, 64)
        widths: list[int] = []
        for item in raw:
            if not isinstance(item, OptionNumber) or not float(item.value).is_integer():
                self._fail(
                    node,
                    f"morphism {morphism.names[0]!r} declares a hidden width that is "
                    "not an integer",
                    code="qiec-program",
                )
            widths.append(int(item.value))
        return tuple(widths)

    def _network_kernel(
        self: _Elaborator,
        step: SampleStep | ObserveStep | MarginalizeStep,
        kinds: Sequence[str],
    ) -> ResolvedDist | None:
        """Resolve a draw through a network-parameterized kernel morphism.

        The resolver refuses a morphism whose parameters come from a
        network or an embedding table, since no transpile target can
        read them; the elaboration reads them as typed inputs and maps
        them through the network's layers, so a multilayer perceptron
        with a bare family initializer resolves to that family, and an
        embedding to the Gaussian kernel at each element's centre.

        Parameters
        ----------
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        kinds : Sequence[str]
            The resolver's structured kinds.

        Returns
        -------
        ResolvedDist | None
            The family the morphism draws from, or ``None`` when the
            refusal is neither a perceptron nor an embedding at the
            site itself.
        """
        morphism = self._program_morphisms.get(step.morphism)
        if morphism is None:
            return None
        if f"embed:{step.morphism}" in kinds:
            return ResolvedDist("Normal", (), step.morphism)
        if "param-source:mlp" not in kinds and "param-source:linear" not in kinds:
            return None
        family = _bare_init_family(morphism)
        if family is None or family not in _CONDITIONAL_HEADS:
            return None
        return ResolvedDist(family, (), step.morphism)

    def _finite_domain(
        self: _Elaborator, morphism: MorphismDecl
    ) -> tuple[str, int] | None:
        """The one finite object a morphism's domain is, if it is one.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.

        Returns
        -------
        tuple[str, int] | None
            The object's name and extent when the domain is a single
            finite object, else ``None``.
        """
        factors = _factors(morphism.domain)
        if len(factors) != 1 or not isinstance(factors[0], TypeName):
            return None
        info = self._program_objects.get(factors[0].name)
        if info is None or not info.finite or info.extent is None:
            return None
        rows = _fixed_extent(info.extent)
        if rows is None:
            return None
        return factors[0].name, rows

    def _table_arguments(
        self: _Elaborator,
        morphism: MorphismDecl,
        step: SampleStep | ObserveStep | MarginalizeStep,
        heads: tuple[tuple[str, HeadTransform], ...],
        width: int,
        finite: tuple[str, int],
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
    ) -> list[tuple[str, Value]]:
        """The arguments of a draw through a kernel over a finite domain.

        The morphism's table is a program input with one row per
        element of the domain and one column per head coordinate. The
        step's one argument names the row: a bound ``Int``, a name of
        the program's domain, or, under a plate, a ``Tensor[Int]`` with
        one element per position, which reads a row per position.
        Without an argument the program's domain input names the row.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        heads : tuple[tuple[str, Literal["identity", "exp_floor"]], ...]
            The family's heads and their transforms.
        width : int
            The codomain's real width.
        finite : tuple[str, int]
            The domain object's name and extent.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        list[tuple[str, Value]]
            One table head per family parameter.

        Raises
        ------
        QiecDiagnosticError
            If the argument is not a binding, is unbound and not a
            domain name, or is not an index.
        """
        del morphism
        name, extent = finite
        rows_axis = _rows_axis(plate, step)
        row = Local(f"__{step.morphism}_row", INT)
        index_value: Value
        batched = False
        if step.args:
            argument = step.args[0]
            if not isinstance(argument, DrawArgName):
                self._fail(
                    step,
                    f"morphism {step.morphism!r} conditions on a value, not a "
                    f"{argument.kind} argument",
                    code="qiec-program",
                )
            local = scope.lookup(argument.text)
            if local is None:
                if argument.text in self._domain_names(state.declaration):
                    local = self._declare_parameter(
                        argument.text, INT, "domain", scope, state, step
                    )
                elif rows_axis is not None:
                    # An unbound index under a plate is a data input
                    # with one element of the domain per position.
                    local = self._declare_parameter(
                        argument.text,
                        tensor_type(INT, (rows_axis.size,)),
                        "data",
                        scope,
                        state,
                        step,
                    )
                else:
                    self._fail(
                        step,
                        f"conditioning binding {argument.text!r} is not bound",
                        code="qiec-program",
                    )
            shape = tensor_shape(local.type)
            if local.type == INT:
                index_value = Var(local)
            elif (
                shape is not None
                and shape[0] == INT
                and rows_axis is not None
                and len(shape[1]) == 1
                and shape[1][0] == rows_axis.size
            ):
                batched = True
                index_value = Gather(Var(local), Var(row), INT)
            else:
                self._fail(
                    step,
                    f"conditioning binding {argument.text!r} is not an index of {name}",
                    code="qiec-program",
                )
        else:
            index_value = Var(
                self._declare_parameter(name.lower(), INT, "domain", scope, state, step)
            )
        rows = width * len(heads)
        table = self._declare_parameter(
            f"{step.morphism}_param_table",
            tensor_type(REAL, (_extent(extent), _extent(rows))),
            "table",
            scope,
            state,
            step,
            axes=(name, f"{step.morphism}_param_row"),
        )
        arguments: list[tuple[str, Value]] = []
        for index, (head, transform) in enumerate(heads):
            result: TypeExpr = (
                REAL if width == 1 else tensor_type(REAL, (_extent(width),))
            )
            value: Value = TableMap(
                Var(table),
                index_value,
                index * width,
                width,
                _map_transform(transform),
                result,
            )
            value = self._transformed_head(value, transform, width, step, state, head)
            if batched:
                assert rows_axis is not None
                value = _per_row(value, width, row, rows_axis)
            arguments.append((head, value))
        return arguments

    def _domain_names(self: _Elaborator, declaration: ProgramDecl) -> tuple[str, ...]:
        """The names a program's domain factors are read through.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.

        Returns
        -------
        tuple[str, ...]
            The declared parameter names when the program names them,
            else each named factor's object name in lowercase.
        """
        if declaration.params is not None:
            return tuple(declaration.params)
        return tuple(
            factor.name.lower()
            for factor in _factors(declaration.domain)
            if isinstance(factor, TypeName)
        )

    def _structured_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        step: object,
    ) -> list[tuple[str, Value]]:
        """The arguments of a bare structured family morphism.

        A ``~ MultivariateNormal`` or ``~ MatrixNormal`` morphism reads
        its location and covariances from data inputs shaped by the
        plate's event axes; a ``~ GP`` morphism's mean is zero and its
        covariance a kernel matrix over the input locations.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl
            The morphism.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        step : object
            The step.

        Returns
        -------
        list[tuple[str, Value]]
            The named arguments.

        Raises
        ------
        QiecDiagnosticError
            If the plate names no event axes for the family's rank, or a
            kernel option is unreadable.
        """
        event = tuple(axis.size for axis in plate.event)
        if record.name == "GP":
            factors = [f for f in _factors(morphism.domain) if isinstance(f, TypeName)]
            grid = None
            for factor in factors:
                info = self._program_objects.get(factor.name)
                if info is not None and info.finite and info.extent is not None:
                    grid = PlateAxis(factor.name, _extent(info.extent))
                    break
            if grid is None:
                self._fail(
                    step,
                    f"GP morphism {morphism.names[0]!r} has no finite domain axis for its grid",
                    code="qiec-program",
                )
            kernel = _option_value(morphism.options, "kernel")
            length_scale = _option_value(morphism.options, "length_scale")
            if kernel is None:
                kernel = "rbf"
            if not isinstance(kernel, str) or kernel != "rbf":
                self._fail(
                    step,
                    f"GP kernel {kernel!r} has no elaboration; the kernels are rbf",
                    code="qiec-program",
                )
            if length_scale is None:
                length_scale = 1.0
            if not isinstance(length_scale, float | int):
                self._fail(
                    step, "GP length_scale must be a number", code="qiec-program"
                )
            inputs = self._declare_parameter(
                "x",
                tensor_type(REAL, (grid.size,)),
                "kernel-input",
                scope,
                state,
                step,
                axes=("grid",),
            )
            zeros = TensorValue(
                tuple(
                    LiteralValue(0.0, REAL)
                    for _ in range(int(cast(IndexLiteral, grid.size).value))
                ),
                tensor_type(REAL, (grid.size,)),
            )
            return [
                ("mean", zeros),
                (
                    "kernel",
                    KernelMatrix(
                        Var(inputs),
                        "rbf",
                        float(length_scale),
                        1e-8,
                        tensor_type(REAL, (grid.size, grid.size)),
                    ),
                ),
            ]
        if len(event) != record.event_rank:
            self._fail(
                step,
                f"morphism {morphism.names[0]!r} draws a rank-{record.event_rank} "
                f"{record.name} but its plate names {len(event)} event axes",
                code="qiec-program",
            )
        site = state.current_site
        if record.name == "MultivariateNormal":
            (n,) = event
            return [
                (
                    "loc",
                    Var(
                        self._declare_parameter(
                            f"{site}_loc",
                            tensor_type(REAL, (n,)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
                (
                    "covariance_matrix",
                    Var(
                        self._declare_parameter(
                            f"{site}_covariance_matrix",
                            tensor_type(REAL, (n, n)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
            ]
        if record.name == "MatrixNormal":
            rows, columns = event
            return [
                (
                    "loc",
                    Var(
                        self._declare_parameter(
                            f"{site}_loc",
                            tensor_type(REAL, (rows, columns)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
                (
                    "row_covariance",
                    Var(
                        self._declare_parameter(
                            f"{site}_row_covariance",
                            tensor_type(REAL, (rows, rows)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
                (
                    "col_covariance",
                    Var(
                        self._declare_parameter(
                            f"{site}_col_covariance",
                            tensor_type(REAL, (columns, columns)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
            ]
        self._fail(
            step,
            f"family {record.name!r} has no structured elaboration",
            code="qiec-program",
        )


def _semantic_family_parameter(
    record: DistributionFamily,
    surface_name: str,
    surface_family: str | None = None,
) -> FamilyParameter:
    """Resolve a QVR keyword to the semantic registry parameter it fills.

    Most names coincide. A few surface families name a parameter
    differently from the registry record (``LogitNormal(mu=..., sigma=...)``
    fills ``loc`` and ``scale``); `PARAMETER_ALIASES` states each such
    correspondence, so a keyword always fills the parameter meaning the
    same quantity, never one at the same position.

    Parameters
    ----------
    record : DistributionFamily
        The family's registry record.
    surface_name : str
        The keyword the source wrote.
    surface_family : str | None
        The family name the source wrote, when it is an alias of
        ``record``.

    Returns
    -------
    FamilyParameter
        The registry parameter the keyword fills.

    Raises
    ------
    KeyError
        If the keyword names no parameter of the family.
    """
    aliases = PARAMETER_ALIASES.get(surface_family or record.name, {})
    return record.parameter(aliases.get(surface_name, surface_name))


def _structured(
    record: DistributionFamily, morphism: MorphismDecl, step: object
) -> bool:
    """Whether a step draws a structured family through a bare morphism.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    morphism : MorphismDecl
        The morphism.
    step : object
        The step, whose own arguments override the structured reading.

    Returns
    -------
    bool
        ``True`` for ``GP`` always, and for ``MultivariateNormal`` and
        ``MatrixNormal`` when neither the step nor the initializer
        supplies arguments.
    """
    if record.name == "GP":
        return True
    if record.name not in ("MultivariateNormal", "MatrixNormal"):
        return False
    if getattr(step, "args", None):
        return False
    init = morphism.init_family
    return init is None or not init.args


class _ScalarParameter:
    """A stand-in family parameter for a list entry.

    Parameters
    ----------
    element : TypeApplication
        The entry's element type.
    """

    def __init__(self, element: TypeApplication) -> None:
        self.rank = 0
        self.constraint = (
            "boolean"
            if element == BOOL
            else "nonnegative_integer"
            if element == INT
            else "real"
        )


class _ParseBinding(dx.Model):
    """One ``("A", p)`` binding of a ``parse`` call.

    Parameters
    ----------
    label : str
        The left-hand side.
    value : LetExprNode
        The probability vector's expression.
    line : int
        The binding's source line.
    col : int
        The binding's source column.
    """

    label: str
    value: LetExprNode
    line: int
    col: int


class _ParseCall(dx.Model):
    """The parts of ``parse(D, x[, Plate][, ("A", p)...])``.

    Parameters
    ----------
    deduction : str
        The deduction.
    source : str
        The program input the deduction parses.
    plate : str | None
        The object whose elements index the corpus's sentences, when the
        input is a corpus.
    bindings : tuple[_ParseBinding, ...]
        The production probabilities, by left-hand side.
    """

    deduction: str
    source: str
    plate: str | None
    bindings: tuple[_ParseBinding, ...]


def _sampled_support(
    expr: LetExprNode, declaration: ProgramDecl
) -> tuple[str, str] | None:
    """The support of the family a variable is drawn from, if it is a draw.

    Parameters
    ----------
    expr : LetExprNode
        A binding's value.
    declaration : ProgramDecl
        The program.

    Returns
    -------
    tuple[str, str] | None
        The variable and its family's support when the value is a
        variable a ``sample`` step of a known family draws, else
        ``None``.
    """
    if not isinstance(expr, LetExprVar):
        return None
    for step in declaration.draws:
        if isinstance(step, SampleStep) and expr.name in step.vars:
            record = FAMILIES.get(step.morphism)
            return None if record is None else (expr.name, record.support)
    return None


@dataclass(slots=True)
class _ProgramState:
    """What accumulates while one program elaborates.

    Parameters
    ----------
    declaration
        The program.
    parameters
        The computation's parameters with their roles, in order.
    sites
        The probabilistic steps seen so far.
    helpers
        The marginalization helper computations declared so far.
    uses_random
        Whether any step performs on the canonical ``random`` instance.
    uses_score
        Whether any step performs on the canonical ``score`` instance.
    uses_params
        Whether any step calls a deduction, which reads its learned
        weights through the module's ``params`` instance.
    alphabets
        The class alphabet of each class-valued step, by step name.
    pending_alphabet
        The alphabet the step being elaborated draws from, if any.
    current_site
        The name of the step being elaborated.
    input_shapes
        The element type and batch axis inferred for each free name a
        let expression reads.
    tables
        The names let expressions gather from, with how many axes a
        gather indexes; a free one is a table of open extents.
    table_events
        The tables whose gathered rows fill a vector-valued family
        parameter, with the rank of one row.
    batch_shapes
        The plate extents each bound name ranges over, outermost first:
        a site's batch axes, a plated input's axes, and those a let
        inherits from the plated values it combines.
    parameter_axes
        The axis names of each parameter's dimensions.
    per_row
        The inputs that fill a family parameter of a plated draw
        directly, whose leading axis is that draw's plate.
    function_calls
        The number of learned-function calls bound so far, which numbers
        the binding of the next one.
    markov_row
        For the row of a marginalized chain's cell, the latent its block
        marginalizes, which the row binds to the block's weighted shots
        (one per state) rather than scoring their marginal; else
        ``None``.
    markov_entry
        For such a row, the parameter holding one position's entry,
        which an observation of that name scores in place.
    extents
        The index binders of the computation's telescope: the object
        parameters of a template, then the open extents of its inputs.
    open_extents
        The open extents a free name's typing introduced, which a later
        use may fix: a declared object, or another extent.
    extent_solution
        Each open extent a use has fixed, to the term it equals, with no
        solved extent on any right-hand side.
    template_objects
        The object parameters bound as objects of the body while the
        program elaborates.
    random
        The canonical ``Random`` instance.
    score
        The canonical ``Score`` instance.
    """

    declaration: ProgramDecl
    parameters: list[tuple[str, ParameterRole]] = field(default_factory=list)
    sites: list[ProgramSite] = field(default_factory=list)
    helpers: list[NamedComputation] = field(default_factory=list)
    uses_random: bool = False
    uses_score: bool = False
    uses_params: bool = False
    alphabets: dict[str, PlateAxis] = field(default_factory=dict)
    pending_alphabet: PlateAxis | None = None
    current_site: str = ""
    extents: dict[str, IndexBinder] = field(default_factory=dict)
    open_extents: set[str] = field(default_factory=set)
    extent_solution: dict[str, IndexTerm] = field(default_factory=dict)
    template_objects: list[str] = field(default_factory=list)
    input_shapes: dict[str, tuple[TypeApplication, PlateAxis | None]] = field(
        default_factory=dict
    )
    tables: dict[str, int] = field(default_factory=dict)
    table_events: dict[str, int] = field(default_factory=dict)
    batch_shapes: dict[str, tuple[IndexTerm, ...]] = field(default_factory=dict)
    parameter_axes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    per_row: set[str] = field(default_factory=set)
    function_calls: int = 0
    markov_row: str | None = None
    markov_entry: str | None = None
    random: NamedEffectInstance | None = None  # type: ignore[assignment]
    score: NamedEffectInstance | None = None  # type: ignore[assignment]


def _index_variables_in(terms: Sequence[StaticArgument]) -> frozenset[str]:
    """The names of every index variable a set of static terms mentions.

    Parameters
    ----------
    terms : Sequence[StaticArgument]
        The terms to walk.

    Returns
    -------
    frozenset[str]
        The variable names.
    """
    found: set[str] = set()
    pending: list[StaticArgument] = list(terms)
    while pending:
        term = pending.pop()
        if isinstance(term, TypeApplication):
            pending.extend(term.arguments)
        elif isinstance(term, ShapeIndex):
            pending.extend(term.dimensions)
        elif isinstance(term, IndexVariable):
            found.add(term.name)
    return frozenset(found)


def _bind_extents(
    parameter: TypeExpr, argument: TypeExpr, bindings: dict[str, IndexTerm]
) -> None:
    """Bind the index variables of a parameter type to an argument type's terms.

    Parameters
    ----------
    parameter : TypeExpr
        The declared parameter type, which may mention index variables.
    argument : TypeExpr
        The argument's type, whose corresponding terms are bound.
    bindings : dict[str, IndexTerm]
        The bindings found so far, extended in place; a variable already
        bound keeps its first binding.
    """
    if not isinstance(parameter, TypeApplication) or not isinstance(
        argument, TypeApplication
    ):
        return
    for expected, actual in zip(parameter.arguments, argument.arguments):
        if isinstance(expected, ShapeIndex) and isinstance(actual, ShapeIndex):
            for dimension, extent in zip(expected.dimensions, actual.dimensions):
                if isinstance(dimension, IndexVariable):
                    bindings.setdefault(dimension.name, extent)
        elif isinstance(expected, TypeApplication | TypeVariable) and isinstance(
            actual, TypeApplication | TypeVariable
        ):
            _bind_extents(expected, actual, bindings)


def _is_entry_point(declaration: ProgramDecl) -> bool:
    """Whether a program elaborates to a computation of its own.

    Parameters
    ----------
    declaration : ProgramDecl
        The program.

    Returns
    -------
    bool
        ``True`` unless the program is a template over morphisms, which
        only its call sites instantiate; scalar parameters make an
        ordinary computation with real or natural parameters, and object
        parameters bind the computation's telescope.
    """
    return declaration.type_params is None or not any(
        isinstance(parameter, MorphismParam) for parameter in declaration.type_params
    )


def _extent(size: int | IndexVariable) -> IndexTerm:
    """An axis extent as an index term.

    Parameters
    ----------
    size : int | IndexVariable
        The extent: a cardinality, or the index variable a program
        template binds for an object parameter.

    Returns
    -------
    IndexTerm
        The literal of the nat sort, or the variable itself.
    """
    if isinstance(size, IndexVariable):
        return size
    return IndexLiteral(size, NAT)


def _fixed_extent(size: int | IndexVariable) -> int | None:
    """An extent as a cardinality, when it has one.

    Parameters
    ----------
    size : int | IndexVariable
        The extent.

    Returns
    -------
    int | None
        The cardinality, or ``None`` for the index variable of a
        template's object parameter, which no call has fixed yet.
    """
    return None if isinstance(size, IndexVariable) else size


def _axis_name(expr: ObjectExpr) -> str:
    """A readable name for an object expression used as an axis.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.

    Returns
    -------
    str
        The object's name, the factor names joined by ``x``, or the
        constructor's spelling.
    """
    if isinstance(expr, TypeName):
        return expr.name
    if isinstance(expr, ObjectProduct):
        return "x".join(_axis_name(item) for item in expr.components)
    if isinstance(expr, DiscreteConstructor | ContinuousConstructor):
        return f"{expr.constructor}{''.join(expr.args)}"
    return expr.kind


def _index_map_names(step: ObserveStep) -> tuple[str, ...] | None:
    """The names of an observation's index map into its group.

    Parameters
    ----------
    step : ObserveStep
        The step.

    Returns
    -------
    tuple[str, ...] | None
        The single ``via`` name, the product's names, or ``None`` when
        the observation is not grouped.
    """
    if step.via_axes:
        return tuple(step.via_axes)
    if step.via is not None:
        return (step.via,)
    return None


def _product_axis(factors: tuple[PlateAxis, ...]) -> PlateAxis:
    """The flat axis a product grouping plate is indexed by.

    Parameters
    ----------
    factors : tuple[PlateAxis, ...]
        The product's axes, each of literal extent.

    Returns
    -------
    PlateAxis
        The axis named by the factors, of the product of their extents.
    """
    extent = 1
    for factor in factors:
        assert isinstance(factor.size, IndexLiteral)
        extent *= factor.size.value
    return PlateAxis("x".join(factor.name for factor in factors), _extent(extent))


def _draw_dependencies(step: SampleStep) -> set[str]:
    """The names a draw step's arguments read.

    Parameters
    ----------
    step : SampleStep
        The step.

    Returns
    -------
    set[str]
        Every bare name, indexed base, and index the arguments mention.
    """
    names: set[str] = set()

    def visit(argument: DrawArg | str | float) -> None:
        """Collect the names one argument reads.

        Parameters
        ----------
        argument : DrawArg | str | float
            The argument.
        """
        if isinstance(argument, DrawArgNamed):
            visit(argument.value)
        elif isinstance(argument, DrawArgName):
            names.add(argument.text)
        elif isinstance(argument, DrawArgIndex):
            names.add(argument.name)
            names.update(argument.indices)
        elif isinstance(argument, DrawArgDist | DrawArgList):
            for item in (
                argument.args if isinstance(argument, DrawArgDist) else argument.items
            ):
                visit(item)
        elif isinstance(argument, str) and not _is_number_text(argument):
            base, _, rest = argument.partition("[")
            names.add(base)
            if rest:
                names.update(item.rstrip("]") for item in rest.split("["))

    for argument in step.args or ():
        visit(argument)
    return names


def _step_binders(step: ProgramStep) -> tuple[str, ...]:
    """The names a step binds in its scope.

    Parameters
    ----------
    step : ProgramStep
        The step.

    Returns
    -------
    tuple[str, ...]
        The bound names; empty for a step binding nothing.
    """
    if isinstance(step, SampleStep | ObserveStep):
        return tuple(step.vars)
    if isinstance(step, LetStep | ScoreStep | CallStep):
        return (step.name,)
    if isinstance(step, MarginalizeStep):
        return (step.var,)
    return ()


def _group_projection(factors: tuple[PlateAxis, ...], outer: PlateAxis) -> Value | None:
    """The index projecting a product group's positions onto one factor.

    Parameters
    ----------
    factors : tuple[PlateAxis, ...]
        The product group's axes, row-major, each of literal extent.
    outer : PlateAxis
        The enclosing group's axis.

    Returns
    -------
    Value | None
        A literal ``Tensor[Int]`` over the product's positions naming
        each position's coordinate along the factor that is the outer
        axis; ``None`` when no factor is.
    """
    position = next(
        (index for index, factor in enumerate(factors) if factor.name == outer.name),
        None,
    )
    if position is None:
        return None
    sizes: list[int] = []
    for factor in factors:
        assert isinstance(factor.size, IndexLiteral)
        sizes.append(factor.size.value)
    stride = math.prod(sizes[position + 1 :])
    total = math.prod(sizes)
    entries = tuple(
        LiteralValue((flat // stride) % sizes[position], INT) for flat in range(total)
    )
    return TensorValue(entries, tensor_type(INT, (_extent(total),)))


def _enclosing_group(scope: _Scope) -> PlateAxis | None:
    """The grouping axis of the nearest grouped marginalization.

    Parameters
    ----------
    scope : _Scope
        The scope.

    Returns
    -------
    PlateAxis | None
        The axis, or ``None`` outside any grouped scope.
    """
    current: _Scope | None = scope
    while current is not None:
        if current.weight_instance is not None:
            return current.group
        current = current.outer
    return None


def _weight_scope(scope: _Scope) -> _Scope | None:
    """The nearest marginalization scope collecting weights.

    Parameters
    ----------
    scope : _Scope
        The scope.

    Returns
    -------
    _Scope | None
        The scope owning a ``Weight`` instance, or ``None`` in the program
        body.
    """
    current: _Scope | None = scope
    while current is not None:
        if current.weight_instance is not None:
            return current
        current = current.outer
    return None


def _binding_order(scope: _Scope, name: str) -> int:
    """Where a name was bound among a scope chain's bindings.

    Parameters
    ----------
    scope : _Scope
        The innermost scope.
    name : str
        The name.

    Returns
    -------
    int
        The binding's position, outermost scope first.
    """
    context = scope.context()
    for position, local in enumerate(context.locals):
        if local.name == name:
            return position
    return len(context.locals)


def _object_expr_info(
    expr: ObjectExpr, table: Mapping[str, ObjectInfo]
) -> ObjectInfo | None:
    """The shape an object expression denotes.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.
    table : Mapping[str, ObjectInfo]
        The objects declared so far.

    Returns
    -------
    ObjectInfo | None
        The shape, or ``None`` for an expression with no static one.
    """
    if isinstance(expr, TypeName):
        if expr.name.isdigit():
            return ObjectInfo(extent=int(expr.name), finite=True)
        return table.get(expr.name)
    if isinstance(expr, DiscreteConstructor):
        if not expr.args:
            return None
        size = _size_argument(expr.args[0], table)
        return None if size is None else ObjectInfo(extent=size, finite=True)
    if isinstance(expr, ContinuousConstructor):
        if not expr.args:
            return None
        sizes = [_size_argument(item, table) for item in expr.args]
        if any(size is None for size in sizes):
            return None
        extents = [cast(int, size) for size in sizes]
        if expr.constructor == "Real":
            width = 1
            for size in extents:
                width *= size
            return ObjectInfo(extent=width, real_width=width)
        return ObjectInfo(extent=extents[0])
    if isinstance(expr, ObjectProduct):
        factors = [_object_expr_info(item, table) for item in _factors(expr)]
        if any(
            item is None or item.extent is None or _fixed_extent(item.extent) is None
            for item in factors
        ):
            return None
        total = 1
        finite = True
        for item in factors:
            assert item is not None and item.extent is not None
            total *= cast(int, _fixed_extent(item.extent))
            finite = finite and item.finite
        return ObjectInfo(extent=total, finite=finite)
    return None


def _size_argument(argument: str, table: Mapping[str, ObjectInfo]) -> int | None:
    """Resolve a constructor size argument.

    Parameters
    ----------
    argument : str
        An integer literal or an object name.
    table : Mapping[str, ObjectInfo]
        The objects declared so far.

    Returns
    -------
    int | None
        The size, or ``None`` when the name has no extent.
    """
    if argument.isdigit():
        return int(argument)
    prior = table.get(argument)
    return (
        None if prior is None or prior.extent is None else _fixed_extent(prior.extent)
    )


def _object_table(statements: Sequence[object]) -> dict[str, ObjectInfo]:
    """The shapes of every object declaration in a module.

    Parameters
    ----------
    statements : Sequence[object]
        The module's statements.

    Returns
    -------
    dict[str, ObjectInfo]
        Name to shape, in declaration order, so a later declaration can
        size itself by an earlier one.
    """
    table: dict[str, ObjectInfo] = {}
    for statement in statements:
        if not isinstance(statement, ObjectDecl):
            continue
        info: ObjectInfo | None = None
        if isinstance(statement.init, TypeEnumSet):
            info = ObjectInfo(extent=len(statement.init.elements), finite=True)
        elif isinstance(statement.init, TypeFromExpr):
            info = _object_expr_info(statement.init.expr, table)
        if info is None:
            continue
        for name in statement.names:
            table[name] = info
    return table


def _origin_at(
    elaborator: _Elaborator,
    path: tuple[str | int, ...],
    role: str,
    node: object | None = None,
) -> SourceOrigin:
    """A source origin at a structural path.

    Parameters
    ----------
    elaborator : _Elaborator
        The elaborator, for the module name.
    path : tuple[str | int, ...]
        The structural path the identity derives from.
    role : str
        What the position is.
    node : object | None
        The source node whose line and column are recorded.

    Returns
    -------
    SourceOrigin
        The origin.
    """
    return elaborator._origin(
        node if node is not None else elaborator.source.syntax.statements[0], path, role
    )


__all__ = []
