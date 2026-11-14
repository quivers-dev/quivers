"""Canonical QIEC prelude effects and their small reference handlers.

The declarations in this module are stable structural data.  The factory
functions produce process-local :class:`~quivers.qiec.evaluator.RuntimeHandler`
attachments; samplers, distributions, semirings, and observation tables
consequently never become serializable QIEC fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
import operator
from typing import cast

from quivers.qiec.canonical import (
    ELEMENT,
    ELEMENT_BINDER,
    LOG_WEIGHT,
    SAMPLEABLE_CONSTRUCTOR,
    SITE_CONSTRUCTOR,
    builtin_constructor,
)
from quivers.qiec.effects import (
    ArgumentDef,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    RowEntry,
)
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec.families import family
from quivers.qiec.evaluator import (
    ClauseContext,
    Forward,
    TailResume,
    InvalidHandlerError,
    Resumption,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
    RuntimeTypeMismatch,
    RuntimeValidator,
)
from quivers.qiec.identifiers import (
    EffectId,
    EffectInstanceId,
    HandlerId,
    OperationId,
    SourceOrigin,
)
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.terms import Perform
from quivers.qiec.types import (
    STRING,
    UNIT,
    EffectRef,
    TypeApplication,
    TypeExpr,
    TypeVariable,
    render_static,
)


_constructor = builtin_constructor

_A_BINDER = ELEMENT_BINDER
_K_BINDER = TypeBinder(name="k")
_P_BINDER = TypeBinder(name="payload")
_S_BINDER = TypeBinder(name="state")
_ANSWER_BINDER = TypeBinder(name="answer")
A = ELEMENT
K = TypeVariable(name="k")
PAYLOAD = TypeVariable(name="payload")
STATE = TypeVariable(name="state")
ANSWER = TypeVariable(name="answer")

CHOICES_CONSTRUCTOR = _constructor("Choices", _A_BINDER)
CHOICE_RESULTS_CONSTRUCTOR = _constructor("ChoiceResults", _A_BINDER)
STATE_RESULT_CONSTRUCTOR = _constructor("StateResult", _A_BINDER, _S_BINDER)
WEIGHTED_RESULT_CONSTRUCTOR = _constructor("WeightedResult", _A_BINDER, _K_BINDER)

SITE_A = TypeApplication(constructor=SITE_CONSTRUCTOR, arguments=(A,))
SAMPLEABLE_A = TypeApplication(constructor=SAMPLEABLE_CONSTRUCTOR, arguments=(A,))
CHOICES_A = TypeApplication(constructor=CHOICES_CONSTRUCTOR, arguments=(A,))
CHOICE_RESULTS_A = TypeApplication(
    constructor=CHOICE_RESULTS_CONSTRUCTOR, arguments=(A,)
)
STATE_RESULT = TypeApplication(
    constructor=STATE_RESULT_CONSTRUCTOR, arguments=(ANSWER, STATE)
)
WEIGHTED_RESULT = TypeApplication(
    constructor=WEIGHTED_RESULT_CONSTRUCTOR, arguments=(ANSWER, K)
)


def _effect_ref(name: str) -> EffectRef:
    """An unsaturated reference to a prelude effect interface.

    Parameters
    ----------
    name : str
        The interface's name.

    Returns
    -------
    EffectRef
        The declaration reference, in the ``prelude`` namespace so it
        names the same interface in every module.
    """
    return EffectRef(id=EffectId.derive("prelude", name), name=name)


def _operation_id(effect: EffectRef, name: str) -> OperationId:
    """The identity of one operation on a prelude interface.

    Parameters
    ----------
    effect : EffectRef
        The interface that declares it.
    name : str
        The operation's name.

    Returns
    -------
    OperationId
        The identity, derived from the interface and the name, so two
        interfaces may each declare `get` without collision.
    """
    return OperationId.derive(str(effect.id), name)


RANDOM = _effect_ref("Random")
"""``Random``: the prelude interface for drawing a value at a sample site."""
RANDOM_SAMPLE = _operation_id(RANDOM, "sample")
"""``Random.sample``: draw a value of type ``A`` at a site from a ``Sampleable[A]``."""
RANDOM_EFFECT = EffectDef(
    ref=RANDOM,
    telescope=(),
    operations=(
        OperationDef(
            id=RANDOM_SAMPLE,
            name="sample",
            telescope=(_A_BINDER,),
            arguments=(
                ArgumentDef(name="site", type=SITE_A),
                ArgumentDef(name="sampleable", type=SAMPLEABLE_A),
            ),
            result_type=A,
        ),
    ),
)
"""The declaration of :data:`RANDOM`."""

SCORE = _effect_ref("Score")
"""``Score``: the prelude interface for adding a log-density contribution."""
SCORE_ADD = _operation_id(SCORE, "add")
"""``Score.add``: add one ``LogWeight`` to the running score."""
SCORE_EFFECT = EffectDef(
    ref=SCORE,
    telescope=(),
    operations=(
        OperationDef(
            id=SCORE_ADD,
            name="add",
            telescope=(),
            arguments=(ArgumentDef(name="weight", type=LOG_WEIGHT),),
            result_type=UNIT,
        ),
    ),
)
"""The declaration of :data:`SCORE`."""

STATE_EFFECT_REF = _effect_ref("State")
"""``State[S]``: the prelude interface for one mutable cell of type ``S``."""
STATE_GET = _operation_id(STATE_EFFECT_REF, "get")
"""``State.get``: read the cell."""
STATE_PUT = _operation_id(STATE_EFFECT_REF, "put")
"""``State.put``: overwrite the cell."""
STATE_EFFECT = EffectDef(
    ref=STATE_EFFECT_REF,
    telescope=(_S_BINDER,),
    operations=(
        OperationDef(
            id=STATE_GET, name="get", telescope=(), arguments=(), result_type=STATE
        ),
        OperationDef(
            id=STATE_PUT,
            name="put",
            telescope=(),
            arguments=(ArgumentDef(name="value", type=STATE),),
            result_type=UNIT,
        ),
    ),
)
"""The declaration of :data:`STATE_EFFECT_REF`; :func:`state_effect` applies it."""

ABORT = _effect_ref("Abort")
"""``Abort[P]``: the prelude interface for stopping with a payload of type ``P``."""
ABORT_ABORT = _operation_id(ABORT, "abort")
"""``Abort.abort``: stop the computation with an error payload."""
ABORT_EFFECT = EffectDef(
    ref=ABORT,
    telescope=(_P_BINDER,),
    operations=(
        OperationDef(
            id=ABORT_ABORT,
            name="abort",
            telescope=(_A_BINDER,),
            arguments=(ArgumentDef(name="error", type=PAYLOAD),),
            result_type=A,
        ),
    ),
)
"""The declaration of :data:`ABORT`; :func:`abort_effect` applies it."""

CHOOSE = _effect_ref("Choose")
"""``Choose``: the prelude interface for nondeterministic choice."""
CHOOSE_CHOOSE = _operation_id(CHOOSE, "choose")
"""``Choose.choose``: pick one of a collection of alternatives."""
CHOOSE_EFFECT = EffectDef(
    ref=CHOOSE,
    telescope=(),
    operations=(
        OperationDef(
            id=CHOOSE_CHOOSE,
            name="choose",
            telescope=(_A_BINDER,),
            arguments=(ArgumentDef(name="alternatives", type=CHOICES_A),),
            result_type=A,
        ),
    ),
)
"""The declaration of :data:`CHOOSE`."""

WEIGHT = _effect_ref("Weight")
"""``Weight[K]``: the prelude interface for accumulating a semiring weight of type ``K``."""
WEIGHT_ADD = _operation_id(WEIGHT, "add")
"""``Weight.add``: fold one weight into the running total in the handler's semiring."""
WEIGHT_EFFECT = EffectDef(
    ref=WEIGHT,
    telescope=(_K_BINDER,),
    operations=(
        OperationDef(
            id=WEIGHT_ADD,
            name="add",
            telescope=(),
            arguments=(ArgumentDef(name="weight", type=K),),
            result_type=UNIT,
        ),
    ),
)
"""The declaration of :data:`WEIGHT`; :func:`weight_effect` applies it."""

_X_BINDER = TypeBinder(name="input")
INPUT = TypeVariable(name="input")

COMPUTE = _effect_ref("Compute")
"""``Compute[X, A]``: the prelude interface for a host computation; see :data:`COMPUTE_EFFECT`."""
COMPUTE_APPLY = _operation_id(COMPUTE, "apply")
"""``Compute.apply``: run the host computation on an argument of type ``X``."""
COMPUTE_EFFECT = EffectDef(
    ref=COMPUTE,
    telescope=(_X_BINDER, _ANSWER_BINDER),
    operations=(
        OperationDef(
            id=COMPUTE_APPLY,
            name="apply",
            telescope=(),
            arguments=(ArgumentDef(name="argument", type=INPUT),),
            result_type=ANSWER,
        ),
    ),
)
"""``Compute[X, A]``: a host computation as an effect interface.

A morphism, a closure, or any other process-local function a program
depends on is not a kernel value; performing ``apply`` on an instance of
this interface asks the handler installed for it, which a runtime
provider supplies, so the dependence is explicit in the row.
"""

PARAM = _effect_ref("Param")
"""``Param``: the prelude interface for learned parameter lookup; see :data:`PARAM_EFFECT`."""
PARAM_GET = _operation_id(PARAM, "get")
"""``Param.get``: read a learned parameter by name."""
PARAM_EFFECT = EffectDef(
    ref=PARAM,
    telescope=(),
    operations=(
        OperationDef(
            id=PARAM_GET,
            name="get",
            telescope=(_A_BINDER,),
            arguments=(ArgumentDef(name="name", type=STRING),),
            result_type=A,
        ),
    ),
)
"""``Param``: learned parameter lookup as an effect.

A host computation reads each learned tensor it depends on by name
through ``get``, so a handler between it and the parameter store can
answer with a perturbed value, as a prior lift does, or record the read.
"""

BUILTIN_EFFECTS = (
    RANDOM_EFFECT,
    SCORE_EFFECT,
    STATE_EFFECT,
    ABORT_EFFECT,
    CHOOSE_EFFECT,
    WEIGHT_EFFECT,
    COMPUTE_EFFECT,
    PARAM_EFFECT,
)
"""Every prelude interface declaration, which every module may name without
declaring."""

_RUNTIME_ADD = cast(Callable[[object, object], object], operator.add)


def state_effect(state_type: TypeExpr) -> EffectRef:
    """Construct the concrete ``State[state_type]`` interface.

    Parameters
    ----------
    state_type : TypeExpr
        The type to instantiate the interface's binder at.

    Returns
    -------
    EffectRef
        The saturated application.

    Raises
    ------
    TypeError
        If the argument is not a type.
    ValueError
        If it is ill-kinded.
    """
    return STATE_EFFECT.apply((state_type,))


def abort_effect(error_type: TypeExpr) -> EffectRef:
    """Construct the concrete ``Abort[error_type]`` interface.

    Parameters
    ----------
    error_type : TypeExpr
        The type to instantiate the interface's binder at.

    Returns
    -------
    EffectRef
        The saturated application.

    Raises
    ------
    TypeError
        If the argument is not a type.
    ValueError
        If it is ill-kinded.
    """
    return ABORT_EFFECT.apply((error_type,))


def weight_effect(weight_type: TypeExpr) -> EffectRef:
    """Construct the concrete ``Weight[weight_type]`` interface.

    Parameters
    ----------
    weight_type : TypeExpr
        The type to instantiate the interface's binder at.

    Returns
    -------
    EffectRef
        The saturated application.

    Raises
    ------
    TypeError
        If the argument is not a type.
    ValueError
        If it is ill-kinded.
    """
    return WEIGHT_EFFECT.apply((weight_type,))


def _unit(value: object) -> bool:
    """Whether a host value inhabits `Unit`.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True only for None, which is `Unit`'s single inhabitant.
    """
    return value is None


def _accepts(validator: RuntimeValidator, value: object) -> bool:
    """Whether a validator admits a value.

    Parameters
    ----------
    validator : RuntimeValidator
        The predicate to apply.
    value : object
        The value to test.

    Returns
    -------
    bool
        True unless the validator returned False exactly. A validator
        returning None reads as acceptance, so one written for its side
        effect does not reject everything.
    """
    return validator(value) is not False


def _pair_of(
    first: RuntimeValidator,
    second: RuntimeValidator,
) -> RuntimeValidator:
    """A validator for a two-element tuple, component by component.

    Parameters
    ----------
    first : RuntimeValidator
        Validates the first component.
    second : RuntimeValidator
        Validates the second.

    Returns
    -------
    RuntimeValidator
        A validator accepting a pair whose components both pass.
    """

    def validate(value: object) -> bool:
        """Whether a value is a pair of accepted components.

        Parameters
        ----------
        value : object
            The value to test.

        Returns
        -------
        bool
            True for a two-element tuple whose components pass.
        """
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and _accepts(first, value[0])
            and _accepts(second, value[1])
        )

    return validate


def _tuple_of(item: RuntimeValidator) -> RuntimeValidator:
    """A validator for a tuple of uniformly typed elements.

    Parameters
    ----------
    item : RuntimeValidator
        Validates each element.

    Returns
    -------
    RuntimeValidator
        A validator accepting a tuple whose every element passes. An
        empty tuple passes, having no element to fail.
    """

    def validate(value: object) -> bool:
        """Whether a value is a tuple of accepted elements.

        Parameters
        ----------
        value : object
            The value to test.

        Returns
        -------
        bool
            True for a tuple whose elements all pass.
        """
        return isinstance(value, tuple) and all(_accepts(item, part) for part in value)

    return validate


def _handler_def(
    name: str,
    effect: EffectRef,
    clauses: tuple[tuple[OperationId, ResumptionGrade], ...],
    *,
    key: str,
    input_type: TypeExpr = ANSWER,
    output_type: TypeExpr = ANSWER,
    introduced: EffectRow = EffectRow(),
    total: bool = True,
    telescope: tuple[TypeBinder, ...] = (),
) -> HandlerDef:
    """Build one prelude handler's checked signature.

    Parameters
    ----------
    name : str
        The handler's name.
    effect : EffectRef
        The interface it handles.
    clauses : tuple[tuple[OperationId, ResumptionGrade], ...]
        The operations it covers, each with its resumption grade.
    key : str
        Distinguishes handlers sharing a name, entering the derived
        identity so two prelude handlers for one interface stay apart.
    input_type : TypeExpr
        What the handled computation returns.
    output_type : TypeExpr
        What the handler answers with.
    introduced : EffectRow
        Effects the handler itself performs.
    total : bool
        Whether it covers every declared operation.
    telescope : tuple[TypeBinder, ...]
        Static binders it takes.

    Returns
    -------
    HandlerDef
        The signature. Its clauses carry no bodies: a prelude handler's
        behavior is a runtime attachment, so the declaration is foreign.
    """
    return HandlerDef(
        id=HandlerId.derive("prelude", name, key),
        name=name,
        effect=effect,
        clauses=tuple(
            HandlerClauseDef(operation=operation, grade=grade)
            for operation, grade in clauses
        ),
        input_type=input_type,
        output_type=output_type,
        introduced=introduced,
        total=total,
        telescope=telescope,
    )


def _generic_binders(
    *,
    answer_type: TypeExpr | None = None,
    state_type: TypeExpr | None = None,
    error_type: TypeExpr | None = None,
    weight_type: TypeExpr | None = None,
) -> tuple[TypeBinder, ...]:
    """Declare exactly the free prelude parameters used by a handler.

    Parameters
    ----------
    answer_type : TypeExpr or None
        The handler's answer type, when it has one.
    state_type : TypeExpr or None
        Its state type, for a stateful handler.
    error_type : TypeExpr or None
        Its error payload type, for an aborting handler.
    weight_type : TypeExpr or None
        Its weight type, for a weighting handler.

    Returns
    -------
    tuple[TypeBinder, ...]
        Binders for exactly those parameters left generic. A handler
        instantiated at a concrete type binds nothing for it, so its
        telescope stays as small as its genericity requires.
    """
    result: list[TypeBinder] = []
    if answer_type == ANSWER:
        result.append(_ANSWER_BINDER)
    if state_type == STATE:
        result.append(_S_BINDER)
    if error_type == PAYLOAD:
        result.append(_P_BINDER)
    if weight_type == K:
        result.append(_K_BINDER)
    return tuple(result)


def _expect_arguments(
    request: RuntimeRequest, count: int, handler: str
) -> tuple[object, ...]:
    """Require a request to carry the argument count a clause expects.

    Parameters
    ----------
    request : RuntimeRequest
        The request being answered.
    count : int
        How many arguments the clause reads.
    handler : str
        The handler's name, for the diagnostic.

    Returns
    -------
    tuple[object, ...]
        The arguments.

    Raises
    ------
    InvalidHandlerError
        If the count differs. The static check settles this for an
        authored clause, so reaching here means a foreign handler was
        attached to an interface it does not match.
    """
    if len(request.arguments) != count:
        raise InvalidHandlerError(
            f"{handler} expected {count} operation arguments, got "
            f"{len(request.arguments)}"
        )
    return request.arguments


def _require(
    value: object,
    validator: RuntimeValidator,
    type: TypeExpr,
    subject: str,
) -> None:
    """Check a host value against the type a prelude clause expects.

    Parameters
    ----------
    value : object
        The host value.
    validator : RuntimeValidator
        The predicate to apply.
    type : TypeExpr
        The type it should inhabit, named in the diagnostic.
    subject : str
        What is being checked.

    Raises
    ------
    RuntimeTypeMismatch
        If the validator returns False, or raises. A raising validator is
        reported as a mismatch so a caller handles one failure class.
    """
    try:
        accepted = validator(value)
    except Exception as error:
        raise RuntimeTypeMismatch(f"validator raised for {subject}: {error}") from error
    if accepted is False:
        raise RuntimeTypeMismatch(f"{subject} does not inhabit {type!r}: {value!r}")


def _default_draw(sampleable: object) -> object:
    """Draw from a sampleable using whichever interface it offers.

    Parameters
    ----------
    sampleable
        The host distribution.

    Returns
    -------
    object
        The drawn value. A reparameterised draw is preferred where the
        object offers one, since it keeps a gradient path a plain sample
        would cut.

    Raises
    ------
    InvalidHandlerError
        If the object offers neither drawing interface.
    """
    rsample = getattr(sampleable, "rsample", None)
    if callable(rsample):
        return rsample()
    sample = getattr(sampleable, "sample", None)
    if callable(sample):
        return sample()
    if callable(sampleable):
        return sampleable()
    raise InvalidHandlerError(
        "Random.sample received a nonsampleable runtime attachment"
    )


def draw_handler(
    result_validator: RuntimeValidator,
    *,
    draw: Callable[[object, object, RuntimeRequest], object] | None = None,
    answer_type: TypeExpr = ANSWER,
    duplicable_context: bool = False,
    key: str = "draw",
) -> RuntimeHandler:
    """Interpret ``Random.sample`` by drawing exactly once.

    Parameters
    ----------
    result_validator
        Checks the value a resumption carries inhabits the
        operation's result type.
    draw
        Supplies the drawn value, given the site, the sampleable, and
        the request. None uses the sampleable's own reparameterised or
        plain sampling interface.
    answer_type
        What the handler answers with.
    duplicable_context
        Whether the handler's own state may be copied for a
        multi-shot resumption. Asserted rather than inferred: copying a
        handler owning mutable state would let two shots share it.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Random.draw",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def sample(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer a `Random.sample` request.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            What the resumed computation produced.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry a site and a sampleable, or the
            default draw finds no sampling interface on the sampleable.
        """
        site, sampleable = _expect_arguments(request, 2, definition.name)
        value = (
            draw(site, sampleable, request)
            if draw is not None
            else _default_draw(sampleable)
        )
        return TailResume(value)

    return RuntimeHandler(
        definition,
        {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
        duplicable_context=duplicable_context,
    )


def add_weights(left: object, right: object) -> object:
    """Add two log weights, entrywise over tensors of them.

    Parameters
    ----------
    left : object
        A float, or nested tuples of floats.
    right : object
        A float, or nested tuples of floats shaped like ``left`` when
        both are tensors.

    Returns
    -------
    object
        The entrywise sum; a scalar broadcasts over a tensor.

    Raises
    ------
    RuntimeTypeMismatch
        If two tensors differ in shape.
    """
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise RuntimeTypeMismatch(
                f"weights of lengths {len(left)} and {len(right)} cannot be added"
            )
        return tuple(add_weights(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, tuple):
        return tuple(add_weights(item, right) for item in left)
    if isinstance(right, tuple):
        return tuple(add_weights(left, item) for item in right)
    return cast(float, left) + cast(float, right)


def _weight_shape(value: object) -> tuple[int, ...]:
    """The shape of a host tensor of weights.

    Parameters
    ----------
    value : object
        A float or nested tuples of floats.

    Returns
    -------
    tuple[int, ...]
        The extents, outermost first; empty for a float.
    """
    if isinstance(value, tuple):
        return (len(value), *(_weight_shape(value[0]) if value else ()))
    return ()


def _entries(value: object) -> list[float]:
    """Every entry of a host tensor of weights in row-major order.

    Parameters
    ----------
    value : object
        A float or nested tuples of floats.

    Returns
    -------
    list[float]
        The entries.
    """
    if isinstance(value, tuple):
        return [entry for item in value for entry in _entries(item)]
    return [cast(float, value)]


def _reshape(entries: list[float], shape: tuple[int, ...]) -> object:
    """Arrange entries as nested tuples of a shape.

    Parameters
    ----------
    entries : list[float]
        The entries in row-major order.
    shape : tuple[int, ...]
        The shape.

    Returns
    -------
    object
        The single entry for the empty shape, else nested tuples.
    """
    if not shape:
        return entries[0]
    stride = len(entries) // shape[0] if shape[0] else 0
    return tuple(
        _reshape(entries[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    )


def _aggregate(values: list[float], reduction: str) -> float:
    """Aggregate the weighted shots of an enumeration at one position.

    Parameters
    ----------
    values : list[float]
        One weighted shot per support value.
    reduction : str
        ``"logsumexp"``, ``"sum"``, or ``"mean"``.

    Returns
    -------
    float
        The aggregate.
    """
    if reduction == "logsumexp":
        return _log_sum_exp(values)
    total = math.fsum(values)
    return total / len(values) if reduction == "mean" else total


def _log_sum_exp(values: list[float]) -> float:
    """The log of a sum of exponentials, stably.

    Parameters
    ----------
    values : list[float]
        The logarithms summed.

    Returns
    -------
    float
        ``log(sum(exp(v)))``; negative infinity for an empty list or
        one of only negative infinities.
    """
    finite = [value for value in values if value != float("-inf")]
    if not finite:
        return float("-inf")
    peak = max(finite)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in finite))


def _support(sampleable: object) -> tuple[object, ...]:
    """The finite support a distribution is enumerated over.

    Parameters
    ----------
    sampleable : object
        A runtime distribution.

    Returns
    -------
    tuple[object, ...]
        The values one draw can take: the family's declared finite
        support, or the class indices of a ``Categorical``, read off
        its probabilities or its logits.

    Raises
    ------
    InvalidHandlerError
        If the value is no runtime distribution or its family has no
        finite support.
    """
    if not isinstance(sampleable, RuntimeDistribution):
        raise InvalidHandlerError("enumeration needs a runtime distribution")
    record = family(sampleable.family)
    if record.finite_support is not None:
        return tuple(record.finite_support)
    if sampleable.family == "Categorical":
        weights = sampleable.arguments.get("probs", sampleable.arguments.get("logits"))
        classes = _weight_shape(weights)
        if not classes:
            raise InvalidHandlerError(
                "Categorical enumeration needs its probabilities or logits"
            )
        return tuple(range(classes[-1]))
    raise InvalidHandlerError(
        f"family {sampleable.family!r} has no finite support to enumerate"
    )


def enumerate_handler(
    result_validator: RuntimeValidator,
    *,
    answer_type: TypeExpr = ANSWER,
    weight_type: TypeExpr | None = None,
    reduction: str = "logsumexp",
    key: str = "enumerate",
) -> RuntimeHandler:
    """Interpret ``Random.sample`` by aggregating over a finite support.

    The handled computation must answer with its value paired with the
    log weight it accumulated, as the collecting ``Weight`` handler
    answers. The clause resumes once per support value, adds each shot's
    weight to the prior log probability of its value, and aggregates the
    shots by the reduction: ``logsumexp`` answers the log of the summed
    probabilities, the marginal; ``sum`` the sum of the weighted shots;
    ``mean`` their average. The aggregate is totalled over the plate's
    positions when the latent is plated, or kept one per position when
    a weight type is given, which is what a marginalization nested in a
    grouped one adds to its group's weights. ``table`` aggregates
    nothing: it answers the weighted shots themselves, one per support
    value in support order, which is the row of a marginalized chain's
    transition weights that the forward algorithm reduces.

    Parameters
    ----------
    result_validator
        Checks the value a resumption carries inhabits the operation's
        result type.
    answer_type
        What the handled computation answers with.
    weight_type
        The type of the per-position weights the clause answers with,
        the collected weights' own tensor type, whose shape the answer
        takes from the shots' weights; ``None`` totals the positions
        into one ``LogWeight``.
    reduction
        How the shots aggregate: ``"logsumexp"``, ``"sum"``, or
        ``"mean"``; or ``"table"``, which keeps them one per support
        value.
    key
        Distinguishes this handler from others of the same name.

    Returns
    -------
    RuntimeHandler
        The attachment.

    Raises
    ------
    ValueError
        If the reduction is not one of the four.
    """
    if reduction not in ("logsumexp", "sum", "mean", "table"):
        raise ValueError(
            "enumeration reduces by logsumexp, sum, or mean, or keeps a table, "
            f"not {reduction!r}"
        )
    definition = _handler_def(
        "Random.enumerate",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.UNRESTRICTED),),
        key=key,
        input_type=answer_type,
        output_type=LOG_WEIGHT if weight_type is None else weight_type,
        telescope=_generic_binders(answer_type=answer_type),
    )
    per_position = weight_type is not None

    def sample(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer a `Random.sample` request by enumeration.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, resumed once per support value.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            The log of the marginal likelihood the scope accumulates, or
            the per-position marginals shaped by the weight type.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry a site and a sampleable, the
            family has no finite support, a shot answers with something
            other than a value paired with its weight, or the shots'
            weights do not fill the weight type's positions.
        """
        _site, sampleable = _expect_arguments(request, 2, definition.name)
        support = _support(sampleable)
        distribution = cast(RuntimeDistribution, sampleable)
        shape = (*distribution.batch, *distribution.event)
        totals: list[list[float]] = []
        shapes: list[tuple[int, ...]] = []
        for choice in support:
            value = _reshape([choice] * math.prod(shape), shape) if shape else choice  # type: ignore[list-item]
            prior = distribution.log_prob(value, keep_batch=True)
            answer = resume(value)
            if not isinstance(answer, tuple) or len(answer) != 2:
                raise InvalidHandlerError(
                    "enumeration expects each shot to answer with a value and "
                    "its accumulated weight"
                )
            weight = add_weights(prior, answer[1])
            totals.append(_entries(weight))
            shapes.append(_weight_shape(weight))
        if reduction == "table":
            if any(len(entries) != 1 for entries in totals):
                raise InvalidHandlerError(
                    "enumeration keeps a table of one weight per support value "
                    "only for a single latent whose shots answer one weight"
                )
            return tuple(entries[0] for entries in totals)
        width = max(len(entries) for entries in totals)
        marginals = [
            _aggregate([entries[position] for entries in totals], reduction)
            for position in range(width)
        ]
        if not per_position:
            return math.fsum(marginals)
        widest = max(shapes, key=len)
        if any(shape not in ((), widest) for shape in shapes):
            raise InvalidHandlerError(
                "enumeration answers per-position weights only when every "
                "shot's weights share one shape"
            )
        return _reshape(marginals, widest)

    return RuntimeHandler(
        definition,
        {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
        duplicable_context=True,
    )


@dataclass(slots=True)
class ScoreAccumulator:
    """Mutable result owned by one installed Score handler.

    Parameters
    ----------
    total
        The running combination of every contribution so far, starting at
        the handler's identity.
    contributions
        Every contribution in the order it was added.
    """

    total: object
    contributions: list[object] = field(default_factory=list)


def score_handler(
    *,
    identity: object = 0.0,
    combine: Callable[[object, object], object] = _RUNTIME_ADD,
    weight_validator: RuntimeValidator = lambda _value: True,
    answer_validator: RuntimeValidator = lambda _value: True,
    total_validator: RuntimeValidator | None = None,
    expose_total: bool = False,
    answer_type: TypeExpr = ANSWER,
    key: str = "score",
    observer: Callable[[RuntimeRequest, object], None] | None = None,
) -> tuple[RuntimeHandler, ScoreAccumulator]:
    """Accumulate ``Score.add`` contributions in their evaluation order.

    Parameters
    ----------
    identity
        The accumulator's value before any contribution.
    combine
        Folds a contribution into the running total.
    weight_validator
        Checks an individual weight contribution.
    answer_validator
        Checks the handled computation's answer.
    total_validator
        Checks the accumulated total; ``weight_validator`` when omitted.
    expose_total
        Whether the accumulated total is returned alongside the
        answer, rather than kept internal.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.
    observer
        Called with each request and its contribution as it is
        accumulated, so a caller can attribute contributions to the
        sites their provenance names.

    Returns
    -------
    tuple[RuntimeHandler, ScoreAccumulator]
        The attachment, ready to bind into a runtime environment, and
        the accumulator each installation publishes into when its
        computation completes.
    """
    accumulator = ScoreAccumulator(identity)
    checked_total = total_validator or weight_validator
    output_type = (
        TypeApplication(
            constructor=WEIGHTED_RESULT_CONSTRUCTOR, arguments=(answer_type, LOG_WEIGHT)
        )
        if expose_total
        else answer_type
    )
    definition = _handler_def(
        "Score.accumulate",
        SCORE,
        ((SCORE_ADD, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=output_type,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_context() -> RuntimeHandler:
        """Build a fresh per-installation handler.

        Returns
        -------
        RuntimeHandler
            A handler with its own state, so two installations of this
            declaration do not share it.
        """
        local = ScoreAccumulator(identity)

        def add(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            """Answer a `Score.add` request by accumulating the weight.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                What the resumed computation produced.

            Raises
            ------
            InvalidHandlerError
                If the request does not carry exactly one argument.
            RuntimeTypeMismatch
                If the contribution does not inhabit the weight type.
            """
            (weight,) = _expect_arguments(request, 1, definition.name)
            _require(weight, weight_validator, LOG_WEIGHT, "score contribution")
            local.contributions.append(weight)
            local.total = combine(local.total, weight)
            if observer is not None:
                observer(request, weight)
            return TailResume(None)

        def finish(value: object, _context: ClauseContext) -> object:
            """Answer the handled computation's return.

            Parameters
            ----------
            value
                The value the handled computation returned.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                The value paired with the accumulated total when
                ``expose_total`` is set, else the value alone.
            """
            return (value, local.total) if expose_total else value

        def publish() -> None:
            """Copy this installation's total and contributions to the accumulator."""
            accumulator.total = local.total
            accumulator.contributions[:] = local.contributions

        return RuntimeHandler(
            definition,
            {SCORE_ADD: RuntimeClause(add, _unit)},
            return_clause=finish,
            input_validator=answer_validator,
            output_validator=(
                _pair_of(answer_validator, checked_total)
                if expose_total
                else answer_validator
            ),
            mutable_context=True,
            on_exit=publish,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype, accumulator


class MissingValuePolicy(str, Enum):
    """What a condition handler does with a sample site it has no observation for.

    ``FORWARD`` passes the request to an outer handler; ``ERROR`` rejects it.
    """

    FORWARD = "forward"
    ERROR = "error"


class ExtraValuePolicy(str, Enum):
    """What a handler does with a supplied value no sample site consumed.

    ``IGNORE`` discards it; ``ERROR`` rejects the run when the handled
    computation returns.
    """

    IGNORE = "ignore"
    ERROR = "error"


def _site_and_sampleable(
    request: RuntimeRequest, handler: str
) -> tuple[object, object]:
    """Read a sample request's site and sampleable arguments.

    Parameters
    ----------
    request
        The request being answered and its arguments.
    handler
        The handler's name, for any diagnostic.

    Returns
    -------
    tuple[object, object]
        The site and the sampleable.

    Raises
    ------
    InvalidHandlerError
        If the request does not carry exactly those two arguments, or
        if the site is not hashable and so cannot key an observation
        or replay table.
    """
    site, sampleable = _expect_arguments(request, 2, handler)
    try:
        hash(site)
    except TypeError as error:
        raise InvalidHandlerError("probabilistic site keys must be hashable") from error
    return site, sampleable


def _log_density(sampleable: object, value: object) -> object:
    """Score a value under a host distribution.

    Parameters
    ----------
    sampleable
        The host distribution.
    value
        The observed or replayed value to score.

    Returns
    -------
    object
        The log density.

    Raises
    ------
    InvalidHandlerError
        If the distribution offers no scoring interface.
    """
    log_prob = getattr(sampleable, "log_prob", None)
    if not callable(log_prob):
        raise InvalidHandlerError(
            "a scored observation/replay needs a runtime sampleable with log_prob"
        )
    return log_prob(value)


def _emit_score(
    context: ClauseContext,
    request: RuntimeRequest,
    score_instance: EffectInstanceId,
    weight: object,
    weight_validator: RuntimeValidator,
    *,
    role: str,
) -> None:
    """Send one log-density contribution to a `Score` instance.

    Parameters
    ----------
    context
        Runtime services available to the clause.
    request
        The request being answered and its arguments.
    score_instance
        The lexical `Score` instance to send the density to.
    weight
        The log density to contribute.
    weight_validator
        Checks the contribution.
    role
        What the contribution is for; it names the attachment and the
        generated request's site, which is recorded as derived from the
        original request's site.

    Raises
    ------
    RuntimeTypeMismatch
        If the contribution does not inhabit the weight type.
    """
    weight_ref = context.attach(
        weight,
        LOG_WEIGHT,
        role=role,
        validator=weight_validator,
    )
    source = request.core.origin.origin
    # A contribution derived from a contribution keeps the site the first
    # one was derived from, and the derived provenance carries the site's
    # full dynamic address, so a transformer between a site's handler and
    # the accumulator does not cut the attribution and the site's frames
    # survive the clause being evaluated outside the call that reached it.
    parents = request.core.origin.parents or (request.core.origin.static_site,)
    if request.core.origin.parents:
        dynamic_path = request.core.origin.dynamic_path
    else:
        dynamic_path = (*request.core.origin.dynamic_path, *request.dynamic_path)
    generated_origin = request.core.origin.with_(
        origin=SourceOrigin(
            module=source.module,
            structural_path=(*source.structural_path, "generated", role),
            role=role,
            source_protocol=source.source_protocol,
            file=source.file,
            line=source.line,
            column=source.column,
        ),
        dynamic_path=dynamic_path,
        relation="duplicate",
        parents=parents,
    )
    context.evaluate(
        Perform(
            request=EffectRequest(
                instance=score_instance,
                effect=SCORE,
                operation=SCORE_ADD,
                static_arguments=(),
                arguments=(weight_ref,),
                result_type=UNIT,
                origin=generated_origin,
            )
        )
    )


def condition_handler(
    observations: Mapping[object, object],
    *,
    score_instance: EffectInstanceId,
    result_validator: RuntimeValidator,
    weight_validator: RuntimeValidator = lambda _value: True,
    missing: MissingValuePolicy = MissingValuePolicy.FORWARD,
    extra: ExtraValuePolicy = ExtraValuePolicy.ERROR,
    answer_type: TypeExpr = ANSWER,
    key: str = "condition",
) -> RuntimeHandler:
    """Supply likelihood observations and emit their densities as Score.

    Unobserved sites forward to the next ``Random`` handler.  The explicit
    introduced Score row distinguishes likelihood observation from intervention
    and unscored replay.

    Parameters
    ----------
    observations
        The values to condition on, keyed by site.
    score_instance
        The lexical `Score` instance each observation's log density is
        sent to.
    result_validator
        Checks an observed value inhabits the request's result type.
    weight_validator
        Checks each log-density contribution.
    missing
        Whether a sample request at a site with no observation is
        forwarded outward or rejected.
    extra
        What to do, when the handled computation returns, with an
        observation at a site the computation never sampled.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Random.condition",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=EffectRow(
            entries=(RowEntry(instance=score_instance, effect=SCORE),)
        ),
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_context() -> RuntimeHandler:
        """Build a fresh per-installation handler.

        Returns
        -------
        RuntimeHandler
            A handler with its own state, so two installations of this
            declaration do not share it.
        """
        seen: set[object] = set()

        def sample(
            request: RuntimeRequest,
            resume: Resumption,
            context: ClauseContext,
        ) -> object:
            """Answer a `Random.sample` request.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            context
                Runtime services available to the clause.

            Returns
            -------
            object
                What the resumed computation produced, or a `Forward` when
                the site has no observation and ``missing`` forwards.

            Raises
            ------
            InvalidHandlerError
                If the request's site has no observation and ``missing``
                rejects, or the request does not carry a site and a
                sampleable.
            RuntimeTypeMismatch
                If the observed value does not inhabit the site's type.
            """
            site, sampleable = _site_and_sampleable(request, definition.name)
            if site not in observations:
                if missing is MissingValuePolicy.ERROR:
                    raise InvalidHandlerError(f"missing observation for site {site!r}")
                return Forward()
            value = observations[site]
            _require(
                value,
                result_validator,
                request.core.result_type,
                f"observation {site!r}",
            )
            seen.add(site)
            weight = _log_density(sampleable, value)
            _emit_score(
                context,
                request,
                score_instance,
                weight,
                weight_validator,
                role="condition-score",
            )
            return TailResume(value)

        def finish(value: object, _context: ClauseContext) -> object:
            """Answer the handled computation's return.

            Parameters
            ----------
            value
                The value the handled computation returned.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                The value, unchanged.

            Raises
            ------
            InvalidHandlerError
                If observations were supplied for sites the computation never
                sampled. Reported rather than ignored, since an unused
                observation usually means a site was renamed.
            """
            if extra is ExtraValuePolicy.ERROR:
                extras = set(observations) - seen
                if extras:
                    rendered = ", ".join(
                        repr(item) for item in sorted(extras, key=repr)
                    )
                    raise InvalidHandlerError(f"unused observations: {rendered}")
            return value

        return RuntimeHandler(
            definition,
            {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
            return_clause=finish,
            mutable_context=True,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype


class ReplayPolicy(str, Enum):
    """Whether a replayed value is scored as well as clamped.

    ``CLAMP_AND_SCORE`` sends the replayed value's log density to a `Score`
    instance; ``CLAMP_NO_SCORE`` and ``INTERVENE`` replay it without scoring.
    """

    CLAMP_AND_SCORE = "clamp-and-score"
    CLAMP_NO_SCORE = "clamp-no-score"
    INTERVENE = "intervene"


def replay_handler(
    values: Mapping[object, object],
    *,
    result_validator: RuntimeValidator,
    policy: ReplayPolicy = ReplayPolicy.CLAMP_NO_SCORE,
    score_instance: EffectInstanceId | None = None,
    weight_validator: RuntimeValidator = lambda _value: True,
    extra: ExtraValuePolicy = ExtraValuePolicy.ERROR,
    answer_type: TypeExpr = ANSWER,
    key: str = "replay",
    type_validator: Callable[[TypeExpr], RuntimeValidator | None] = lambda _type: None,
) -> RuntimeHandler:
    """Replay selected sites under an explicit scoring/intervention policy.

    A sample request at a site in ``values`` resumes with the recorded
    value; a request at any other site is forwarded to an outer handler.
    A site reached more than once, as a recurrence reaches the sites of
    its step at every position, replays the value keyed ``"<site>@<n>"``
    at its ``n``-th occurrence, counted from zero. A value keyed by the
    site alone replays a site reached once; given for a site reached
    more than once, it is refused rather than replayed at every
    occurrence.

    Parameters
    ----------
    values
        The recorded value for each replayed site, keyed by site, or by
        site and occurrence as ``"<site>@<n>"``.
    result_validator
        Checks a replayed value inhabits the request's result type.
    policy
        Whether a replayed value is also scored: ``CLAMP_AND_SCORE`` sends
        its log density to ``score_instance``, while ``CLAMP_NO_SCORE``
        and ``INTERVENE`` replay it without scoring.
    score_instance
        The lexical `Score` instance replayed densities are sent to;
        required under ``CLAMP_AND_SCORE``.
    weight_validator
        Checks each log-density contribution.
    extra
        What to do, when the handled computation returns, with a recorded
        site the computation never sampled.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.
    type_validator
        The validator for a closed type, or ``None`` where the runtime
        has none: each replayed value is checked against the type the
        request instantiates the site at, so a value of the wrong
        element type is refused where it enters, naming the site.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.

    Raises
    ------
    ValueError
        If ``policy`` is ``CLAMP_AND_SCORE`` and no ``score_instance`` is
        given.
    """
    if policy is ReplayPolicy.CLAMP_AND_SCORE and score_instance is None:
        raise ValueError("scored replay requires a Score effect instance")
    introduced = (
        EffectRow(entries=(RowEntry(instance=score_instance, effect=SCORE),))
        if score_instance is not None and policy is ReplayPolicy.CLAMP_AND_SCORE
        else EffectRow()
    )
    definition = _handler_def(
        f"Random.replay.{policy.value}",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=introduced,
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_context() -> RuntimeHandler:
        """Build a fresh per-installation handler.

        Returns
        -------
        RuntimeHandler
            A handler with its own state, so two installations of this
            declaration do not share it.
        """
        seen: set[object] = set()
        occurrences: dict[object, int] = {}

        def sample(
            request: RuntimeRequest,
            resume: Resumption,
            context: ClauseContext,
        ) -> object:
            """Answer a `Random.sample` request.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            context
                Runtime services available to the clause.

            Returns
            -------
            object
                What the resumed computation produced, or a `Forward` when
                the site has no recorded value.

            Raises
            ------
            InvalidHandlerError
                If the request does not carry a site and a sampleable, the
                value is scored and the sampleable has no ``log_prob``, or
                one value keyed by the site alone is given for a site
                reached more than once.
            RuntimeTypeMismatch
                If the recorded value does not inhabit the site's type.
            """
            site, sampleable = _site_and_sampleable(request, definition.name)
            occurrence = occurrences.get(site, 0)
            occurrences[site] = occurrence + 1
            key_at = f"{site}@{occurrence}"
            if key_at in values:
                site_key: object = key_at
            elif site in values:
                if occurrence > 0 or any(
                    isinstance(key, str) and key.startswith(f"{site}@")
                    for key in values
                ):
                    # One value for a site reached more than once would be
                    # replayed at every occurrence, a trajectory the caller
                    # never gave.
                    raise InvalidHandlerError(
                        f"replay {site!r}: the site is reached more than once, "
                        f"so each occurrence takes its own value under "
                        f"'{site}@<n>', not one value under {site!r}"
                    )
                site_key = site
            else:
                return Forward()
            value = values[site_key]
            _require(
                value,
                result_validator,
                request.core.result_type,
                f"replay {site!r}",
            )
            site_type = request.core.result_type
            site_validator = type_validator(site_type)
            if site_validator is not None and not _accepts(site_validator, value):
                raise RuntimeTypeMismatch(
                    f"the value given for site {site!r}, {value!r}, does not "
                    f"inhabit the site's checked type {render_static(site_type)}"
                )
            seen.add(site_key)
            if policy is ReplayPolicy.CLAMP_AND_SCORE:
                assert score_instance is not None
                _emit_score(
                    context,
                    request,
                    score_instance,
                    _log_density(sampleable, value),
                    weight_validator,
                    role="replay-score",
                )
            return TailResume(value)

        def finish(value: object, _context: ClauseContext) -> object:
            """Answer the handled computation's return.

            Parameters
            ----------
            value
                The value the handled computation returned.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                The value, unchanged.

            Raises
            ------
            InvalidHandlerError
                If recorded values remain unconsumed and the policy forbids it.
            """
            if extra is ExtraValuePolicy.ERROR:
                extras = set(values) - seen
                if extras:
                    rendered = ", ".join(
                        repr(item) for item in sorted(extras, key=repr)
                    )
                    raise InvalidHandlerError(f"unused replay values: {rendered}")
            return value

        return RuntimeHandler(
            definition,
            {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
            return_clause=finish,
            mutable_context=True,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One recorded request and the value produced for it.

    Parameters
    ----------
    address
        The request's dynamic key: its static site, address frames, and
        resumption path.
    instance
        The effect instance the request was performed on.
    operation
        The operation requested.
    arguments
        The evaluated value arguments.
    result
        The value produced for the request.
    mode
        How the value was produced, such as ``"forwarded"``.
    """

    address: tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
    instance: EffectInstanceId
    operation: OperationId
    arguments: tuple[object, ...]
    result: object
    mode: str


@dataclass(slots=True)
class TraceRecorder:
    """Ordered trace; repeated dynamic addresses remain observable as events.

    Parameters
    ----------
    events
        The recorded events, in order; empty to start.
    """

    events: list[TraceEvent] = field(default_factory=list)

    def record(self, request: RuntimeRequest, result: object, *, mode: str) -> object:
        """Record one request and the value produced for it.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        result
            The value produced for the request.
        mode
            How the value was produced.

        Returns
        -------
        object
            The value, unchanged, after recording. Returned so the caller can
            use this in place of the value it passed.
        """
        self.events.append(
            TraceEvent(
                request.address,
                request.instance,
                request.operation,
                request.arguments,
                result,
                mode,
            )
        )
        return result


def trace_handler(
    effect: EffectRef,
    operations: Iterable[OperationId],
    recorder: TraceRecorder,
    *,
    validators: Mapping[OperationId, RuntimeValidator],
    answer_type: TypeExpr = ANSWER,
    key: str = "trace",
) -> RuntimeHandler:
    """Observe forwarded operation replies without interpreting the effect.

    Parameters
    ----------
    effect
        The interface whose operations are recorded.
    operations
        Which operations to record.
    recorder
        Receives each recorded request.
    validators
        Per-operation result validators.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    operation_tuple = tuple(operations)
    definition = _handler_def(
        f"{effect.name}.trace",
        effect,
        tuple((operation, ResumptionGrade.AFFINE) for operation in operation_tuple),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_clause(operation: OperationId):
        """Build the clause implementing one operation.

        Parameters
        ----------
        operation
            The operation this clause is being built for.

        Returns
        -------
        RuntimeClause
            The clause and its result validator.

        Raises
        ------
        ValueError
            If the operation has no validator among those supplied.
        """

        def clause(
            request: RuntimeRequest,
            _resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            """Answer the request this clause covers.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            _resume
                The continuation, which this clause does not invoke.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                A `Forward` whose response hook records the outer
                handler's reply.
            """
            return Forward(
                lambda result: recorder.record(request, result, mode="forwarded")
            )

        try:
            validator = validators[operation]
        except KeyError as error:
            raise ValueError(
                f"trace handler needs a validator for {operation}"
            ) from error

        def checked_clause(
            request: RuntimeRequest,
            resume: Resumption,
            context: ClauseContext,
        ) -> object:
            """Answer the request, validating the resumed value.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            context
                Runtime services available to the clause.

            Returns
            -------
            object
                A `Forward` whose response hook validates the outer
                handler's reply before recording it.
            """
            answer = clause(request, resume, context)
            assert isinstance(answer, Forward)
            hook = answer.on_response

            def validate_and_record(result: object) -> object:
                """Check a produced value and record it.

                Parameters
                ----------
                result
                    The value produced for the request.

                Returns
                -------
                object
                    The value, unchanged, so the caller can use this in
                    place of what it passed.

                Raises
                ------
                RuntimeTypeMismatch
                    If the value does not inhabit the operation's result
                    type.
                """
                _require(
                    result,
                    validator,
                    request.core.result_type,
                    f"traced result of {request.operation}",
                )
                assert hook is not None
                return hook(result)

            return Forward(validate_and_record)

        return RuntimeClause(checked_clause, validator)

    return RuntimeHandler(
        definition,
        {operation: make_clause(operation) for operation in operation_tuple},
    )


@dataclass(slots=True)
class StateCell:
    """The mutable cell owned by one installed State handler.

    Parameters
    ----------
    value
        The current state.
    """

    value: object


def state_handler(
    initial: object,
    state_validator: RuntimeValidator,
    *,
    state_type: TypeExpr = STATE,
    expose_final: bool = False,
    answer_type: TypeExpr = ANSWER,
    answer_validator: RuntimeValidator = lambda _value: True,
    key: str = "state",
) -> tuple[RuntimeHandler, StateCell]:
    """Interpret lexical State.get/put with a nonduplicable mutable cell.

    Parameters
    ----------
    initial
        The state before any operation.
    state_validator
        Checks the initial state, each value put, and each value got.
    state_type
        The type of the handler's state.
    expose_final
        Whether the final state is returned alongside the answer.
    answer_type
        What the handler answers with.
    answer_validator
        Checks the handled computation's answer.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    tuple[RuntimeHandler, StateCell]
        The attachment, ready to bind into a runtime environment, and
        the cell each installation publishes its final state into when
        its computation completes.

    Raises
    ------
    RuntimeTypeMismatch
        If ``initial`` does not inhabit ``state_type``.
    """
    _require(initial, state_validator, state_type, "initial state")
    cell = StateCell(initial)
    effect = state_effect(state_type)
    definition = _handler_def(
        "State.run",
        effect,
        (
            (STATE_GET, ResumptionGrade.LINEAR),
            (STATE_PUT, ResumptionGrade.LINEAR),
        ),
        key=key,
        input_type=answer_type,
        output_type=(
            TypeApplication(
                constructor=STATE_RESULT_CONSTRUCTOR,
                arguments=(answer_type, state_type),
            )
            if expose_final
            else answer_type
        ),
        telescope=_generic_binders(
            answer_type=answer_type,
            state_type=state_type,
        ),
    )

    def make_context() -> RuntimeHandler:
        """Build a fresh per-installation handler.

        Returns
        -------
        RuntimeHandler
            A handler with its own state, so two installations of this
            declaration do not share it.
        """
        local = StateCell(initial)

        def get(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            """Answer a `State.get` request with the current state.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                What the resumed computation produced.

            Raises
            ------
            InvalidHandlerError
                If the request carries any argument.
            """
            _expect_arguments(request, 0, definition.name)
            return TailResume(local.value)

        def put(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            """Answer a `State.put` request by replacing the state.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                What the resumed computation produced.

            Raises
            ------
            InvalidHandlerError
                If the request does not carry exactly one argument.
            RuntimeTypeMismatch
                If the new state does not inhabit ``state_type``.
            """
            (new_value,) = _expect_arguments(request, 1, definition.name)
            _require(new_value, state_validator, state_type, "state update")
            local.value = new_value
            return TailResume(None)

        def finish(value: object, _context: ClauseContext) -> object:
            """Answer the handled computation's return.

            Parameters
            ----------
            value
                The value the handled computation returned.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                The value paired with the final state when
                ``expose_final`` is set, else the value alone.
            """
            return (value, local.value) if expose_final else value

        def publish() -> None:
            """Copy this installation's final state to the shared cell."""
            cell.value = local.value

        return RuntimeHandler(
            definition,
            {
                STATE_GET: RuntimeClause(get, state_validator),
                STATE_PUT: RuntimeClause(put, _unit),
            },
            return_clause=finish,
            input_validator=answer_validator,
            output_validator=(
                _pair_of(answer_validator, state_validator)
                if expose_final
                else answer_validator
            ),
            mutable_context=True,
            on_exit=publish,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype, cell


@dataclass(frozen=True, slots=True)
class Aborted:
    """The default answer of an abort handler, wrapping the abort payload.

    Parameters
    ----------
    error
        The payload the computation aborted with.
    """

    error: object


def abort_handler(
    *,
    on_abort: Callable[[object], object] = Aborted,
    result_validator: RuntimeValidator = lambda _value: True,
    error_type: TypeExpr = PAYLOAD,
    input_type: TypeExpr = ANSWER,
    output_type: TypeExpr = ANSWER,
    output_validator: RuntimeValidator = lambda _value: True,
    duplicable_context: bool = False,
    key: str = "abort",
) -> RuntimeHandler:
    """Handle Abort.abort without resuming the discarded continuation.

    Parameters
    ----------
    on_abort
        Receives the payload when the computation aborts and supplies
        the handler's answer; wraps it in `Aborted` by default.
    result_validator
        The abort operation's result validator; never exercised, since
        the clause does not resume.
    error_type
        The payload type an abort carries.
    input_type
        What the handled computation returns.
    output_type
        What the handler answers with.
    output_validator
        Checks the handler's answer.
    duplicable_context
        Whether the handler's own state may be copied for a
        multi-shot resumption. Asserted rather than inferred: copying a
        handler owning mutable state would let two shots share it.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    effect = abort_effect(error_type)
    definition = _handler_def(
        "Abort.run",
        effect,
        ((ABORT_ABORT, ResumptionGrade.ZERO),),
        key=key,
        input_type=input_type,
        output_type=output_type,
        telescope=tuple(
            dict.fromkeys(
                (
                    *_generic_binders(answer_type=input_type),
                    *_generic_binders(answer_type=output_type),
                    *_generic_binders(error_type=error_type),
                )
            )
        ),
    )

    def abort(
        request: RuntimeRequest,
        _resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer an abort request without resuming.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        _resume
            The continuation, which this clause does not invoke.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            What ``on_abort`` makes of the payload; the continuation is
            discarded.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry exactly one argument.
        """
        (error,) = _expect_arguments(request, 1, definition.name)
        return on_abort(error)

    return RuntimeHandler(
        definition,
        {ABORT_ABORT: RuntimeClause(abort, result_validator)},
        output_validator=output_validator,
        duplicable_context=duplicable_context,
    )


def choose_handler(
    choice_validator: RuntimeValidator,
    *,
    combine: Callable[[list[object]], object] | None = None,
    combine_validator: RuntimeValidator | None = None,
    answer_type: TypeExpr = ANSWER,
    answer_validator: RuntimeValidator = lambda _value: True,
    key: str = "choose",
) -> RuntimeHandler:
    """Enumerate finite choices using an unrestricted deep resumption.

    Parameters
    ----------
    choice_validator
        Checks each alternative resumed with.
    combine
        Folds the per-alternative results into the handler's answer;
        ``None`` concatenates them, the return clause having wrapped each
        answer in a one-element tuple.
    combine_validator
        Checks the combined answer; required with a custom ``combine``,
        and otherwise a tuple of ``answer_validator`` values.
    answer_type
        What the handled computation returns; the handler answers with
        ``ChoiceResults`` of it.
    answer_validator
        Checks the handled computation's answer.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.

    Raises
    ------
    ValueError
        If ``combine`` is given without ``combine_validator``.
    """
    if combine is not None and combine_validator is None:
        raise ValueError("a custom Choose combine needs an output validator")
    definition = _handler_def(
        "Choose.enumerate",
        CHOOSE,
        ((CHOOSE_CHOOSE, ResumptionGrade.UNRESTRICTED),),
        key=key,
        input_type=answer_type,
        output_type=TypeApplication(
            constructor=CHOICE_RESULTS_CONSTRUCTOR, arguments=(answer_type,)
        ),
        telescope=_generic_binders(answer_type=answer_type),
    )

    def choose(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer a choice request, resuming once per alternative.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            The combined result over every alternative.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry exactly one argument, the
            alternatives are a string, bytes, or not iterable, or, under
            the default combination, the return clause's results are not
            tuples.
        """
        (alternatives,) = _expect_arguments(request, 1, definition.name)
        if isinstance(alternatives, (str, bytes)):
            raise InvalidHandlerError("Choose alternatives must be a finite collection")
        try:
            options = tuple(alternatives)  # type: ignore[arg-type]
        except TypeError as error:
            raise InvalidHandlerError("Choose alternatives must be iterable") from error
        results = [resume(option) for option in options]
        if combine is not None:
            return combine(results)
        flattened: list[object] = []
        for result in results:
            if not isinstance(result, tuple):
                raise InvalidHandlerError(
                    "the default Choose handler expects its deep return clause "
                    "to produce tuples"
                )
            flattened.extend(result)
        return tuple(flattened)

    return RuntimeHandler(
        definition,
        {CHOOSE_CHOOSE: RuntimeClause(choose, choice_validator)},
        return_clause=lambda value, _context: (value,),
        input_validator=answer_validator,
        output_validator=(
            combine_validator
            if combine_validator is not None
            else _tuple_of(answer_validator)
        ),
        duplicable_context=True,
    )


SEARCH_REDUCTIONS: tuple[str, ...] = ("logsumexp", "max", "or", "sum")
"""The semiring additions :func:`search_handler` may combine its shots by."""


def _search_combine(reduction: str, weights: list[object]) -> object:
    """Combine the collected weights of a search's shots.

    Parameters
    ----------
    reduction : str
        One of ``SEARCH_REDUCTIONS``.
    weights : list[object]
        One weight per shot: floats for the log reductions, Booleans
        for ``or``, integers for ``sum``.

    Returns
    -------
    object
        The semiring sum; the semiring's zero for no shots.
    """
    if reduction == "logsumexp":
        return _log_sum_exp([cast(float, weight) for weight in weights])
    if reduction == "max":
        return max((cast(float, weight) for weight in weights), default=float("-inf"))
    if reduction == "or":
        return any(cast(bool, weight) for weight in weights)
    return sum(cast(int, weight) for weight in weights)


def search_handler(
    definition: HandlerDef,
    operation: OperationId,
    *,
    reduction: str,
    alternative_validator: RuntimeValidator,
) -> RuntimeHandler:
    """Enumerate a tensor of alternatives, summing the shots' weights.

    The handled computation answers with a unit paired with the weight
    a collecting handler inside it accumulated, which the return clause
    reduces to the weight. The clause resumes once per entry of the
    tensor it is asked to choose from and answers the semiring sum of
    what the resumptions answer, so an empty choice contributes the
    semiring's zero; a resumption's answer is itself a semiring sum when
    the continuation chooses again, and the sum is associative, so the
    nesting does not matter.

    Parameters
    ----------
    definition
        The module's declaration of the handler, whose one clause
        handles the choosing operation with an unrestricted resumption.
    operation
        The choosing operation.
    reduction
        The semiring addition, one of ``SEARCH_REDUCTIONS``.
    alternative_validator
        Checks the value each resumption carries.

    Returns
    -------
    RuntimeHandler
        The attachment.

    Raises
    ------
    ValueError
        If the reduction is unknown.
    """
    if reduction not in SEARCH_REDUCTIONS:
        raise ValueError(
            f"a search combines its shots by one of {', '.join(SEARCH_REDUCTIONS)}, "
            f"not {reduction!r}"
        )

    def choose(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer a choice by resuming once per alternative.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, resumed once per alternative.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            The semiring sum of the shots' weights.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry one tensor of alternatives.
        """
        (alternatives,) = _expect_arguments(request, 1, definition.name)
        if not isinstance(alternatives, tuple):
            raise InvalidHandlerError("a search chooses among the entries of a tensor")
        return _search_combine(
            reduction, [resume(alternative) for alternative in alternatives]
        )

    def finish(value: object, _context: ClauseContext) -> object:
        """Reduce the handled computation's answer to its weight.

        Parameters
        ----------
        value
            The value paired with its accumulated weight.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            The weight.

        Raises
        ------
        InvalidHandlerError
            If the answer is not a value paired with its weight.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise InvalidHandlerError(
                "a search expects the handled computation to answer with a value "
                "and its accumulated weight"
            )
        return value[1]

    return RuntimeHandler(
        definition,
        {operation: RuntimeClause(choose, alternative_validator)},
        return_clause=finish,
        duplicable_context=True,
    )


@dataclass(slots=True)
class WeightAccumulator:
    """Mutable result owned by one installed Weight handler.

    Parameters
    ----------
    total
        The running combination of every contribution so far, starting at
        the handler's identity.
    contributions
        Every contribution in the order it was added.
    """

    total: object
    contributions: list[object] = field(default_factory=list)


def weight_handler(
    identity: object,
    combine: Callable[[object, object], object],
    weight_validator: RuntimeValidator,
    *,
    weight_type: TypeExpr = K,
    expose_total: bool = True,
    answer_type: TypeExpr = ANSWER,
    answer_validator: RuntimeValidator = lambda _value: True,
    total_validator: RuntimeValidator | None = None,
    forkable: bool = False,
    key: str = "weight",
) -> tuple[RuntimeHandler, WeightAccumulator]:
    """Accumulate values in a caller-supplied semiring multiplication.

    Parameters
    ----------
    identity
        The accumulator's value before any contribution.
    combine
        Folds a contribution into the running total.
    weight_validator
        Checks an individual weight contribution.
    weight_type
        The type of an individual weight.
    expose_total
        Whether the accumulated total is returned alongside the
        answer, rather than kept internal.
    answer_type
        What the handler answers with.
    answer_validator
        Checks the handled computation's answer.
    total_validator
        Checks the accumulated total; ``weight_validator`` when omitted.
    forkable
        Whether an unrestricted resumption may capture an installation:
        each shot then continues from its own copy of the total so far,
        which is what a search inside the scope needs.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    tuple[RuntimeHandler, WeightAccumulator]
        The attachment, ready to bind into a runtime environment, and
        the accumulator each installation publishes into when its
        computation completes.
    """
    accumulator = WeightAccumulator(identity)
    checked_total = total_validator or weight_validator
    effect = weight_effect(weight_type)
    definition = _handler_def(
        "Weight.accumulate",
        effect,
        ((WEIGHT_ADD, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=(
            TypeApplication(
                constructor=WEIGHTED_RESULT_CONSTRUCTOR,
                arguments=(answer_type, weight_type),
            )
            if expose_total
            else answer_type
        ),
        telescope=_generic_binders(
            answer_type=answer_type,
            weight_type=weight_type,
        ),
    )

    def make_context(seed: WeightAccumulator | None = None) -> RuntimeHandler:
        """Build a fresh per-installation handler.

        Parameters
        ----------
        seed : WeightAccumulator | None
            The state a fork continues from; ``None`` starts at the
            identity.

        Returns
        -------
        RuntimeHandler
            A handler with its own state, so two installations of this
            declaration do not share it.
        """
        local = (
            WeightAccumulator(identity)
            if seed is None
            else WeightAccumulator(seed.total, list(seed.contributions))
        )

        def add(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            """Answer a `Score.add` request by accumulating the weight.

            Parameters
            ----------
            request
                The request being answered and its arguments.
            resume
                The continuation, invoked within the clause's declared grade.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                A tail resumption with the unit answer, so the scope
                continues on the machine's stack and a search inside it
                may resume any number of times.

            Raises
            ------
            InvalidHandlerError
                If the request does not carry exactly one argument.
            RuntimeTypeMismatch
                If the contribution does not inhabit the weight type.
            """
            del resume
            (weight,) = _expect_arguments(request, 1, definition.name)
            _require(weight, weight_validator, weight_type, "semiring weight")
            local.contributions.append(weight)
            local.total = combine(local.total, weight)
            return TailResume(None)

        def finish(value: object, _context: ClauseContext) -> object:
            """Answer the handled computation's return.

            Parameters
            ----------
            value
                The value the handled computation returned.
            _context
                Runtime services, unused by this clause.

            Returns
            -------
            object
                The value paired with the accumulated total when
                ``expose_total`` is set, else the value alone.
            """
            return (value, local.total) if expose_total else value

        def publish() -> None:
            """Copy this installation's total and contributions to the accumulator."""
            accumulator.total = local.total
            accumulator.contributions[:] = local.contributions

        def fork(_handler: RuntimeHandler) -> RuntimeHandler:
            """Copy this installation for another shot.

            Parameters
            ----------
            _handler : RuntimeHandler
                The installation forked, which is this one.

            Returns
            -------
            RuntimeHandler
                An installation continuing from the same total.
            """
            return make_context(local)

        return RuntimeHandler(
            definition,
            {WEIGHT_ADD: RuntimeClause(add, _unit)},
            return_clause=finish,
            input_validator=answer_validator,
            output_validator=(
                _pair_of(answer_validator, checked_total)
                if expose_total
                else answer_validator
            ),
            mutable_context=True,
            duplicable_context=forkable,
            fork_context=fork if forkable else None,
            on_exit=publish,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype, accumulator


def draw_scoring_handler(
    *,
    score_instance: EffectInstanceId,
    result_validator: RuntimeValidator,
    weight_validator: RuntimeValidator = lambda _value: True,
    draw: Callable[[object, object, RuntimeRequest], object] | None = None,
    answer_type: TypeExpr = ANSWER,
    key: str = "draw-scoring",
) -> RuntimeHandler:
    """Interpret ``Random.sample`` by drawing once and scoring the draw.

    The draw's log density under its own sampleable is sent to a `Score`
    instance, so a run whose latent sites are drawn accumulates the joint
    density of the draw, which is what a trace of the run reports.

    Parameters
    ----------
    score_instance
        The lexical `Score` instance each draw's log density is sent to.
    result_validator
        Checks the value a resumption carries inhabits the operation's
        result type.
    weight_validator
        Checks each log-density contribution.
    draw
        Supplies the drawn value, given the site, the sampleable, and
        the request. None uses the sampleable's own sampling interface.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Random.draw-scoring",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=EffectRow(
            entries=(RowEntry(instance=score_instance, effect=SCORE),)
        ),
        telescope=_generic_binders(answer_type=answer_type),
    )

    def sample(
        request: RuntimeRequest,
        resume: Resumption,
        context: ClauseContext,
    ) -> object:
        """Answer a `Random.sample` request by drawing and scoring.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        context
            Runtime services available to the clause.

        Returns
        -------
        object
            What the resumed computation produced.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry a site and a sampleable, or the
            sampleable offers no sampling or scoring interface.
        """
        site, sampleable = _site_and_sampleable(request, definition.name)
        value = (
            draw(site, sampleable, request)
            if draw is not None
            else _default_draw(sampleable)
        )
        _emit_score(
            context,
            request,
            score_instance,
            _log_density(sampleable, value),
            weight_validator,
            role="draw-score",
        )
        return TailResume(value)

    return RuntimeHandler(
        definition,
        {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
    )


def intervene_handler(
    values: Mapping[object, object],
    *,
    result_validator: RuntimeValidator,
    on_intervene: Callable[[RuntimeRequest, object], None] | None = None,
    answer_type: TypeExpr = ANSWER,
    key: str = "intervene",
) -> RuntimeHandler:
    """Fix selected sites to given values without scoring them.

    An intervened site's distribution is replaced by a point mass, so it
    contributes nothing to the joint; sites without a value forward.

    Parameters
    ----------
    values
        The value for each intervened site, keyed by site.
    result_validator
        Checks an intervened value inhabits the request's result type.
    on_intervene
        Called with the request and the value fixed for it, so a caller
        can mark the site as deterministic.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Random.intervene",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def sample(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer a `Random.sample` request with the intervened value.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            What the resumed computation produced, or a `Forward` when
            the site is not intervened.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry a site and a sampleable.
        RuntimeTypeMismatch
            If the value does not inhabit the site's type.
        """
        site, _ = _site_and_sampleable(request, definition.name)
        if site not in values:
            return Forward()
        value = values[site]
        _require(
            value, result_validator, request.core.result_type, f"intervention {site!r}"
        )
        if on_intervene is not None:
            on_intervene(request, value)
        return TailResume(value)

    return RuntimeHandler(
        definition, {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)}
    )


def reweight_handler(
    transform: Callable[[object], object],
    *,
    score_instance: EffectInstanceId,
    weight_validator: RuntimeValidator = lambda _value: True,
    answer_type: TypeExpr = ANSWER,
    key: str = "reweight",
) -> RuntimeHandler:
    """Transform every ``Score.add`` contribution before passing it outward.

    Scaling and masking are transformers: each contribution arriving at
    the handler is replaced by its image and sent to the same instance's
    outer handler, with the provenance of the site it came from kept.

    Parameters
    ----------
    transform
        The map applied to each contribution.
    score_instance
        The lexical `Score` instance handled, which the transformed
        contribution is sent on to.
    weight_validator
        Checks each transformed contribution.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Score.reweight",
        SCORE,
        ((SCORE_ADD, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=EffectRow(
            entries=(RowEntry(instance=score_instance, effect=SCORE),)
        ),
        telescope=_generic_binders(answer_type=answer_type),
    )

    def add(
        request: RuntimeRequest,
        resume: Resumption,
        context: ClauseContext,
    ) -> object:
        """Answer a `Score.add` request by re-emitting the transformed weight.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        context
            Runtime services available to the clause.

        Returns
        -------
        object
            What the resumed computation produced.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry exactly one argument.
        RuntimeTypeMismatch
            If the transformed contribution does not inhabit the weight
            type.
        """
        (weight,) = _expect_arguments(request, 1, definition.name)
        # The transformed contribution keeps the role the original was
        # emitted under, so a reader of the accumulator still knows
        # whether the site was drawn, conditioned, or replayed.
        _emit_score(
            context,
            request,
            score_instance,
            transform(weight),
            weight_validator,
            role=request.core.origin.origin.role,
        )
        return TailResume(None)

    return RuntimeHandler(definition, {SCORE_ADD: RuntimeClause(add, _unit)})


def block_handler(
    hidden: Callable[[object], bool],
    *,
    score_instance: EffectInstanceId,
    result_validator: RuntimeValidator,
    weight_validator: RuntimeValidator = lambda _value: True,
    draw: Callable[[object, object, RuntimeRequest], object] | None = None,
    answer_type: TypeExpr = ANSWER,
    key: str = "block",
) -> RuntimeHandler:
    """Answer hidden sites locally so no outer handler sees them.

    A hidden site is drawn and scored here, exactly as the default
    scoring draw would; every other site forwards.

    Parameters
    ----------
    hidden
        Whether a site is hidden from the handlers outside this one.
    score_instance
        The lexical `Score` instance a hidden draw's log density is sent
        to.
    result_validator
        Checks the value a resumption carries inhabits the operation's
        result type.
    weight_validator
        Checks each log-density contribution.
    draw
        Supplies the drawn value for a hidden site, given the site, the
        sampleable, and the request. None uses the sampleable's own
        sampling interface.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Random.block",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=EffectRow(
            entries=(RowEntry(instance=score_instance, effect=SCORE),)
        ),
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def sample(
        request: RuntimeRequest,
        resume: Resumption,
        context: ClauseContext,
    ) -> object:
        """Answer a hidden site's `Random.sample` request locally.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        context
            Runtime services available to the clause.

        Returns
        -------
        object
            What the resumed computation produced, or a `Forward` for a
            site that is not hidden.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry a site and a sampleable, or the
            sampleable offers no sampling or scoring interface.
        """
        site, sampleable = _site_and_sampleable(request, definition.name)
        if not hidden(site):
            return Forward()
        value = (
            draw(site, sampleable, request)
            if draw is not None
            else _default_draw(sampleable)
        )
        _emit_score(
            context,
            request,
            score_instance,
            _log_density(sampleable, value),
            weight_validator,
            role="block-score",
        )
        return TailResume(value)

    return RuntimeHandler(
        definition, {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)}
    )


def compute_handler(
    function: Callable[[object, ClauseContext], object],
    *,
    effect: EffectRef,
    result_validator: RuntimeValidator,
    introduced: EffectRow = EffectRow(),
    answer_type: TypeExpr = ANSWER,
    key: str = "compute",
) -> RuntimeHandler:
    """Serve a ``Compute`` instance with a host function.

    Parameters
    ----------
    function
        The host function, called with the request's argument and the
        clause context, through which it may perform effects the
        ``introduced`` row names.
    effect
        The applied ``Compute[X, A]`` interface the instance carries.
    result_validator
        Checks the function's result inhabits the operation's result
        type.
    introduced
        Effects the host function performs.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Compute.host",
        effect,
        ((COMPUTE_APPLY, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=introduced,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def apply(
        request: RuntimeRequest,
        resume: Resumption,
        context: ClauseContext,
    ) -> object:
        """Answer a `Compute.apply` request with the host function's result.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        context
            Runtime services available to the clause.

        Returns
        -------
        object
            What the resumed computation produced.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry exactly one argument.
        """
        (argument,) = _expect_arguments(request, 1, definition.name)
        return TailResume(function(argument, context))

    return RuntimeHandler(
        definition, {COMPUTE_APPLY: RuntimeClause(apply, result_validator)}
    )


def param_handler(
    lookup: Callable[[str, RuntimeRequest], object],
    *,
    result_validator: RuntimeValidator,
    total: bool = True,
    answer_type: TypeExpr = ANSWER,
    key: str = "param",
) -> RuntimeHandler:
    """Serve ``Param.get`` from a host parameter store.

    Parameters
    ----------
    lookup
        Returns the value of the named parameter, given the name and the
        request.
    result_validator
        Checks a value inhabits the request's result type.
    total
        Whether the handler discharges the instance; a handler layered
        over a store leaves the instance in the row so the store stays
        required outside it.
    answer_type
        What the handler answers with.
    key
        Distinguishes this handler from others of the same name,
        entering its derived identity.

    Returns
    -------
    RuntimeHandler
        The attachment, ready to bind into a runtime environment.
    """
    definition = _handler_def(
        "Param.store",
        PARAM,
        ((PARAM_GET, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        total=total,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def get(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        """Answer a `Param.get` request with the stored value.

        Parameters
        ----------
        request
            The request being answered and its arguments.
        resume
            The continuation, invoked within the clause's declared grade.
        _context
            Runtime services, unused by this clause.

        Returns
        -------
        object
            What the resumed computation produced.

        Raises
        ------
        InvalidHandlerError
            If the request does not carry exactly one string argument.
        RuntimeTypeMismatch
            If the value does not inhabit the request's result type.
        """
        (name,) = _expect_arguments(request, 1, definition.name)
        if not isinstance(name, str):
            raise InvalidHandlerError("Param.get takes a parameter name")
        value = lookup(name, request)
        _require(
            value, result_validator, request.core.result_type, f"parameter {name!r}"
        )
        return TailResume(value)

    return RuntimeHandler(definition, {PARAM_GET: RuntimeClause(get, result_validator)})


__all__ = [
    "ABORT",
    "ABORT_ABORT",
    "ABORT_EFFECT",
    "Aborted",
    "BUILTIN_EFFECTS",
    "CHOOSE",
    "CHOOSE_CHOOSE",
    "CHOOSE_EFFECT",
    "COMPUTE",
    "COMPUTE_APPLY",
    "COMPUTE_EFFECT",
    "ExtraValuePolicy",
    "MissingValuePolicy",
    "PARAM",
    "PARAM_EFFECT",
    "PARAM_GET",
    "RANDOM",
    "RANDOM_EFFECT",
    "RANDOM_SAMPLE",
    "ReplayPolicy",
    "SCORE",
    "SCORE_ADD",
    "SCORE_EFFECT",
    "SEARCH_REDUCTIONS",
    "STATE_EFFECT",
    "STATE_EFFECT_REF",
    "STATE_GET",
    "STATE_PUT",
    "ScoreAccumulator",
    "StateCell",
    "TraceEvent",
    "TraceRecorder",
    "WEIGHT",
    "WEIGHT_ADD",
    "WEIGHT_EFFECT",
    "WeightAccumulator",
    "abort_effect",
    "abort_handler",
    "add_weights",
    "block_handler",
    "choose_handler",
    "compute_handler",
    "condition_handler",
    "draw_handler",
    "draw_scoring_handler",
    "enumerate_handler",
    "intervene_handler",
    "param_handler",
    "replay_handler",
    "reweight_handler",
    "score_handler",
    "search_handler",
    "state_effect",
    "state_handler",
    "trace_handler",
    "weight_effect",
    "weight_handler",
]
