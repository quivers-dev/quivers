"""Run the checked full-span log mass of a finite lexical recurrence."""

from __future__ import annotations

from fractions import Fraction

import torch

from quivers._positions import demand
from quivers.core._closure_block import _make_block
from quivers.core.algebras import LOG_PROB, DivergentClosureError, UndecidedClosureError
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.checking import KernelError, KernelRegistry
from quivers.qiec.contracts import (
    SPECTRAL_RADIUS_BELOW_ONE,
    AbsoluteLogError,
    AccuracyContract,
    Control,
    DerivativeSemantics,
    Exact,
    InterpretationId,
    LinearCyclic,
    NoRetention,
    Numerics,
    OccurrenceId,
    ParameterId,
    QuerySpec,
    Rational,
    Target,
    Weight,
    check_admissible,
)
from quivers.qiec.effects import HandlerDef, ResumptionGrade
from quivers.qiec.evaluator import (
    ClauseContext,
    EvaluationError,
    InvalidHandlerError,
    Resumption,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
    TailResume,
)
from quivers.qiec.foreign import DeductionLinearGoal
from quivers.qiec.identifiers import EffectId, HandlerId, OperationId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.numbers import host_index
from quivers.qiec.rules import AxiomOccurrence, RulePresentation
from quivers.qiec.types import INT, REAL, STRING, IndexLiteral, IndexVariable
from quivers.stochastic.planning import SPAN_ALGORITHM, Plan, PlanRefused, plan

__all__: list[str] = []


def prepare_linear_goal(presentation: RulePresentation) -> Plan:
    """Check the positive-width lexical fragment and select exact closure.

    For width one the lexical seeds are the only non-unary contributions.
    Every binary premise lies in a strictly shorter width. Inducting on
    width, each cell is therefore a nonnegative linear system with the
    already-computed binary contributions plus the lexical seeds as its
    constant term. The span executor sums its unary paths by the star.

    Parameters
    ----------
    presentation
        The checked rule presentation for the lexical system.

    Returns
    -------
    Plan
        The exact span plan in the default number domain.

    Raises
    ------
    ValueError
        If the presentation does not admit this lexical span plan.
    """
    if (
        len(presentation.judgments) != 1
        or len(presentation.axioms) != 1
        or presentation.conditions
        or presentation.equations
        or any(rule.arity not in (1, 2) for rule in presentation.rules)
    ):
        raise ValueError(
            "linear deduction goals require a positive-width lexical span "
            "presentation with unconditional unary or binary rules"
        )
    (form,) = presentation.judgments
    (slot,) = presentation.axioms
    arguments = slot.judgment.arguments
    if (
        form.name != "span"
        or slot.judgment.family != form.id
        or slot.weight is not None
        or tuple(field.type for field in slot.values) != (STRING,)
        or len(arguments) != 3
        or tuple(
            argument.name
            for argument in arguments
            if isinstance(argument, IndexVariable)
        )
        != tuple(binder.name for binder in slot.telescope)
        or len({binder.name for binder in slot.telescope}) != 3
    ):
        raise ValueError(
            "linear deduction goals require one lexical axiom binding "
            "the ordered span and category"
        )
    result = plan(
        QuerySpec(
            presentation=presentation.id,
            interpretation=InterpretationId.derive(presentation.id, "weight"),
            observable=Weight(),
        ),
        presentation,
        LOG_PROB,
        LinearCyclic(),
        AccuracyContract(
            numerics=Numerics(dtype="float64", domain="log"),
            target=Target(
                quantity="least full-span log mass", metric=AbsoluteLogError()
            ),
            guarantee=Exact(),
        ),
        NoRetention(),
    )
    grammar = result.span
    if (
        result.record.algorithm != SPAN_ALGORITHM
        or grammar is None
        or (grammar.start, grammar.end, grammar.category_slots) != (0, 1, (2,))
    ):
        raise ValueError(
            "linear deduction goals need atomic categories and rules that "
            "preserve a span or concatenate two adjacent spans"
        )
    return result


def validate_linear_goal(
    definition: HandlerDef,
    semantics: DeductionLinearGoal,
    registry: KernelRegistry,
    module: str,
) -> None:
    """Bind the nominal lexical plan to the type of its one operation.

    Parameters
    ----------
    definition
        The declared handler.
    semantics
        The checked lexical recurrence and ordered input coefficients.
    registry
        The outer module containing the operation.
    module
        The lexical module in which the query was declared.

    Raises
    ------
    KernelError
        If the operation or handler disagrees with the descriptor.
    """
    interface = registry.effects[definition.effect.id]
    assert isinstance(semantics.presentation, RulePresentation)
    name = semantics.presentation.name
    effect_name, handler_name = f"{name}__LinearGoal", f"{name}__query_linear_goal"
    if (
        definition.name != handler_name
        or definition.id != HandlerId.derive(module, "handler", handler_name)
        or interface.ref.name != effect_name
        or interface.ref.id != EffectId.derive(module, "effect", effect_name)
    ):
        raise KernelError("linear deduction goal names a different nominal recurrence")
    if (
        definition.telescope
        or definition.introduced.entries
        or definition.introduced.tail is not None
        or not definition.total
        or definition.input_type != LOG_WEIGHT
        or definition.output_type != LOG_WEIGHT
        or len(definition.clauses) != 1
        or len(interface.operations) != 1
        or interface.telescope
    ):
        raise KernelError("linear deduction goal requires one closed LogWeight handler")
    (operation,) = interface.operations
    (clause,) = definition.clauses
    if (
        clause.operation != operation.id
        or operation.name != "query"
        or operation.id != OperationId.derive(interface.ref.id, "operation", "query")
        or clause.grade is not ResumptionGrade.LINEAR
        or len(operation.telescope) != 1
        or not isinstance(binder := operation.telescope[0], IndexBinder)
        or binder.sort != NAT
    ):
        raise KernelError(
            "linear deduction goal requires a linear query with a Nat extent"
        )
    extent = IndexVariable(name=binder.name, sort=NAT)
    count = IndexLiteral(
        value=len(semantics.lexicon) + len(semantics.parameters), sort=NAT
    )
    if (
        tuple(argument.type for argument in operation.arguments)
        != (tensor_type(STRING, (extent,)), tensor_type(REAL, (count,)), INT)
        or operation.result_type != LOG_WEIGHT
    ):
        raise KernelError("linear deduction goal has an invalid query signature")


def linear_goal_handler(
    definition: HandlerDef, operation: OperationId, semantics: DeductionLinearGoal
) -> RuntimeHandler:
    """Make the declared operation execute its certified lexical closure.

    Parameters
    ----------
    definition
        The checked handler declaration.
    operation
        Its one linear operation.
    semantics
        The nominal recurrence, selected goal, and typed coefficient slots.

    Returns
    -------
    RuntimeHandler
        The native handler, with request inputs left on their autograd graph.
    """
    selected = semantics.plan
    grammar = selected.span
    assert grammar is not None
    (slot,) = selected.presentation.axioms
    left, right, category = (binder.name for binder in slot.telescope)
    if grammar.unary_cyclic:
        block = _make_block(
            len(grammar.categories),
            tuple(
                (rule.left, rule.result, index)
                for index, rule in enumerate(grammar.unary)
            ),
        )
        closure_size = block.order
    else:
        closure_size = len(grammar.categories) if grammar.unary else 0
    lexicon: dict[str, list[tuple[int, str]]] = {}
    for index, (word, atom) in enumerate(semantics.lexicon):
        lexicon.setdefault(word, []).append((index, atom))

    def solve(
        request: RuntimeRequest, _resume: Resumption, context: ClauseContext
    ) -> object:
        """Return the raw full-span mass or preserve a closure refusal.

        Parameters
        ----------
        request
            The checked request with a token tuple and a log-weight vector.
        _resume
            The linear continuation, resumed via a tail directive.
        context
            The actual execution's numeric domain.

        Returns
        -------
        TailResume
            The full-span log mass.

        Raises
        ------
        InvalidHandlerError
            If the runtime request disagrees with its checked signature.
        EvaluationError
            If the cap or number domain disallows the chart, or the
            closure refuses the supplied coefficients.
        """
        tokens, weights, max_closure_size = request.arguments
        if not (
            isinstance(tokens, tuple)
            and all(isinstance(token, str) for token in tokens)
            and isinstance(weights, torch.Tensor)
            and weights.shape == (len(semantics.lexicon) + len(semantics.parameters),)
            and weights.is_floating_point()
        ):
            raise InvalidHandlerError("malformed linear deduction goal query")
        limit = host_index(max_closure_size, "linear deduction maximum closure size")
        if limit < 1:
            raise EvaluationError(
                "linear deduction maximum closure size must be positive"
            )
        if closure_size > limit:
            raise EvaluationError(
                f"linear deduction closure requires order {closure_size}, above the declared maximum {limit}"
            )
        numerics = context.attachments.numbers.real
        if numerics not in ("float64", "float32"):
            raise EvaluationError(
                "linear deduction goals require float64 or float32 weights"
            )
        # Each elimination adds at most two previous edge logs and a
        # nonnegative star log. Each lexical leaf needs one coefficient and
        # each binary node needs one, plus one unary path per cell. Keep
        # these finite sums away from overflow and structural negative infinity.
        divisor = 16 * max(1, len(tokens)) * (1 << (closure_size + 1))
        weight_limit = float(Fraction(torch.finfo(weights.dtype).max) / divisor)
        demand(
            ((weights.abs() <= weight_limit) | torch.isneginf(weights)).all(),
            lambda: EvaluationError(
                "linear deduction weights exceed the finite chart range"
            ),
        )
        axioms = []
        inputs: dict[ParameterId | OccurrenceId, torch.Tensor] = dict(
            zip(
                semantics.parameters,
                weights[len(semantics.lexicon) :].unbind(),
                strict=True,
            )
        )
        for position, token in enumerate(tokens):
            for index, atom in lexicon.get(token, ()):
                occurrence = OccurrenceId.derive(
                    selected.presentation.id, "lexicon", position, index
                )
                axioms.append(
                    AxiomOccurrence(
                        slot=slot.id,
                        occurrence=occurrence,
                        binding=(
                            (left, position),
                            (right, position + 1),
                            (category, atom),
                        ),
                        values=(token,),
                    )
                )
                inputs[occurrence] = weights[index]
        record = selected.record
        query = record.query
        accuracy = record.accuracy.with_(
            numerics=record.accuracy.numerics.with_(dtype=numerics),
            controls=(Control(name="max_closure_size", value=Rational.of(limit)),),
        )
        if weights.requires_grad and inputs:
            query = query.with_(differentiate_with_respect_to=tuple(inputs))
            accuracy = accuracy.with_(
                gradient=DerivativeSemantics.implicit(SPECTRAL_RADIUS_BELOW_ONE)
            )
            check_admissible(
                query,
                record.recursion,
                accuracy,
                algebra=record.algebra,
                claims=record.claims,
            )
        execution = selected.with_(record=record.with_(query=query, accuracy=accuracy))
        try:
            evaluated = execution.run(axioms, inputs, like=weights)
        except PlanRefused as error:
            cause = error.__cause__
            if isinstance(cause, UndecidedClosureError):
                reason = "could not establish a usable closure"
            elif isinstance(cause, DivergentClosureError):
                reason = "diverges"
            else:
                reason = "was refused"
            raise EvaluationError(f"linear deduction goal {reason}: {error}") from error
        full_span = [
            value
            for item, value in evaluated.values.items()
            if item.arguments[:2] == (0, len(tokens))
            and (semantics.start is None or item.arguments[2] == semantics.start)
        ]
        mass = LOG_PROB.join(torch.stack(full_span) if full_span else weights[:0], 0)
        demand(
            (torch.isfinite(mass) | torch.isneginf(mass)),
            lambda: EvaluationError(
                "linear deduction goal exceeds the finite chart range"
            ),
        )
        return TailResume(value=mass)

    return RuntimeHandler(
        definition=definition,
        clauses={
            operation: RuntimeClause(invoke=solve, result_validator=_scalar_log_weight)
        },
        duplicable_context=True,
    )


def _scalar_log_weight(value: object) -> bool:
    """Recognize a scalar machine value of the checked carrier.

    Parameters
    ----------
    value
        A candidate answer to the closed query.

    Returns
    -------
    bool
        Whether the value has the log carrier's machine representation.
    """
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and value.is_floating_point()
    )
