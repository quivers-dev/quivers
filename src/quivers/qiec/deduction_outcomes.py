"""Execute the checked finite lexical outcome query on the packed forest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from quivers._positions import demand
from quivers.core.algebras import LOG_PROB
from quivers.qiec.canonical import tensor_type
from quivers.qiec.checking import KernelError, KernelRegistry
from quivers.qiec.contracts import OccurrenceId, ParameterId
from quivers.qiec.effects import HandlerDef
from quivers.qiec.evaluator import (
    ClauseContext,
    EvaluationError,
    InvalidHandlerError,
    Resumption,
    RuntimeClause,
    RuntimeConstructor,
    RuntimeHandler,
    RuntimeRequest,
    TailResume,
)
from quivers.qiec.foreign import DEDUCTION_OUTCOME_KEYS, DeductionOutcomes
from quivers.qiec.identifiers import ConstructorId, OperationId
from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
from quivers.qiec.numbers import NumberDomain, decide
from quivers.qiec.rules import AxiomOccurrence, Ground, RulePresentation
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    TypeApplication,
)
from quivers.stochastic.forest import build_forest, edge_weight, forest_folds


def validate_outcome_adapter(
    definition: HandlerDef, semantics: DeductionOutcomes, registry: KernelRegistry
) -> None:
    """Check how a private span presentation exchanges typed values with QIEC.

    Parameters
    ----------
    definition
        The handler declaring the operation and its answer type.
    semantics
        The packed chart descriptor.
    registry
        The outer module's constructor and effect declarations.

    Raises
    ------
    KernelError
        If a type or constructor disagrees with the presented vocabulary
        or the operation cannot carry the query's inputs and answer.
    """
    presentation = semantics.presentation
    assert isinstance(presentation, RulePresentation)
    (form,) = presentation.judgments
    output = definition.output_type
    empty = registry.constructors.get(semantics.empty)
    entry = registry.constructors.get(semantics.entry)
    if (
        empty is None
        or entry is None
        or definition.telescope
        or output != definition.input_type
        or output
        != TypeApplication(
            constructor=registry.families[entry.family].type_constructor, arguments=()
        )
        or empty.family != entry.family
        or empty.fields
        or empty.telescope
        or entry.telescope
        or tuple(field.type for field in entry.fields)
        != (semantics.item_type, REAL, output)
    ):
        raise KernelError("deduction outcome adapter has an invalid result family")
    constructor_ids = dict(semantics.item_constructors)
    item_prefix = semantics.item_type.constructor.name.removesuffix("__Item")
    for name, identity in constructor_ids.items():
        constructor = registry.constructors.get(identity)
        if (
            constructor is None
            or constructor.name != f"{item_prefix}__{name}"
            or constructor.telescope
            or semantics.item_type
            != TypeApplication(
                constructor=registry.families[constructor.family].type_constructor,
                arguments=(),
            )
            or any(
                field.type not in (semantics.item_type, INT)
                for field in constructor.fields
            )
        ):
            raise KernelError(
                "deduction outcome adapter has an invalid item constructor"
            )
    span_fields = registry.constructors[constructor_ids[semantics.span]].fields
    if tuple(field.type for field in span_fields) != (
        INT,
        INT,
        *((semantics.item_type,) * (len(form.indices) - 2)),
    ):
        raise KernelError(
            "deduction outcome adapter does not preserve the span arguments"
        )
    terms = [
        term
        for rule in presentation.rules
        for judgment in (
            rule.conclusion,
            *(premise.judgment for premise in rule.premises),
        )
        for term in judgment.arguments[2:]
    ]
    while terms:
        term = terms.pop()
        if isinstance(term, IndexConstructor):
            label, arity = term.name, len(term.arguments)
            terms.extend(term.arguments)
        elif isinstance(term, IndexLiteral) and isinstance(term.value, str):
            label, arity = term.value, 0
        else:
            continue
        identity = constructor_ids.get(label)
        if identity is None or tuple(
            field.type for field in registry.constructors[identity].fields
        ) != ((semantics.item_type,) * arity):
            raise KernelError(
                f"deduction outcome adapter does not preserve {label!r} at arity {arity}"
            )

    interface = registry.effects[definition.effect.id]
    if len(interface.operations) != 1 or len(definition.clauses) != 1:
        raise KernelError("deduction outcome adapter must handle one query operation")
    (operation,) = interface.operations
    if len(operation.telescope) != 1:
        raise KernelError("deduction outcome adapter needs the sentence extent")
    (binder,) = operation.telescope
    if not isinstance(binder, IndexBinder) or binder.sort != NAT:
        raise KernelError("deduction outcome adapter needs a Nat sentence extent")
    extent = IndexVariable(name=binder.name, sort=binder.sort)
    term_count = IndexLiteral(
        value=len(semantics.words) * (len(form.indices) - 2), sort=NAT
    )
    weight_count = IndexLiteral(
        value=len(semantics.words) + len(semantics.parameters), sort=NAT
    )
    expected = (
        tensor_type(STRING, (extent,)),
        tensor_type(semantics.item_type, (term_count,)),
        tensor_type(REAL, (weight_count,)),
        STRING,
        BOOL,
    )
    if (
        tuple(argument.type for argument in operation.arguments) != expected
        or operation.result_type != output
    ):
        raise KernelError("deduction outcome adapter has an invalid query signature")


def _ground(
    value: object, names: Mapping[ConstructorId, str], sort: UserIndexSort
) -> Ground:
    """Read a checked lexical constructor as a ground rule argument.

    Parameters
    ----------
    value
        The runtime value of a lexical category or form.
    names
        The descriptor's source names keyed by outer constructor identity.
    sort
        The private index sort the resulting ground term must inhabit.

    Returns
    -------
    Ground
        The ground term in the rule presentation.

    Raises
    ------
    InvalidHandlerError
        If the lexical value does not inhabit the private index sort.
    """
    if isinstance(value, RuntimeConstructor) and value.constructor in names:
        label = names[value.constructor]
        if label in sort.constructors and sort.constructor_arity(label) == len(
            value.fields
        ):
            if not value.fields:
                return label
            return (label, *(_ground(field, names, sort) for field in value.fields))
    raise InvalidHandlerError(
        f"deduction outcome lexical data must inhabit {sort.name!r}"
    )


def _item(
    value: Ground,
    constructors: Mapping[str, ConstructorId],
    semantics: DeductionOutcomes,
    numbers: NumberDomain,
) -> RuntimeConstructor | torch.Tensor:
    """Rebuild a packed item's data in the checked item family.

    Parameters
    ----------
    value
        The selected structural key or one of its fields.
    constructors
        The outer constructors keyed by their names in the presentation.
    semantics
        The descriptor of the checked result family.
    numbers
        The execution's integer dtype and device.

    Returns
    -------
    RuntimeConstructor or torch.Tensor
        The checked term, or an integer tensor for a span endpoint.
    """
    if isinstance(value, int):
        return numbers.number("integer", value)
    name, *fields = (value,) if isinstance(value, str) else value
    assert isinstance(name, str)
    return RuntimeConstructor(
        constructor=constructors[name],
        static_arguments=(),
        fields=tuple(
            _item(field, constructors, semantics, numbers) for field in fields
        ),
        result_type=semantics.item_type,
    )


def _chart_outcomes(
    semantics: DeductionOutcomes,
    tokens: Sequence[object],
    terms: Sequence[object],
    weights: torch.Tensor,
    key: str,
    normalize: bool,
) -> dict[Ground, torch.Tensor]:
    """Push full-span goal weights forward to the selected structural keys.

    Parameters
    ----------
    semantics
        The descriptor of the lexical presentation and its admissible weights.
    tokens
        The input sentence, as a sequence of strings.
    terms
        The category and optional logical form of every lexical entry.
    weights
        The log weights of the lexical entries followed by the rule weights.
    key
        The projection requested for the keys of the result.
    normalize
        Whether to subtract the log total mass from each returned log weight.

    Returns
    -------
    dict of Ground to torch.Tensor
        The log mass of each positive-mass key, with gradients.

    Raises
    ------
    InvalidHandlerError
        If the request does not match the descriptor's admitted fragment.
    EvaluationError
        If a finite mass is out of range or normalization has no positive mass.
    """
    presentation = semantics.presentation
    assert isinstance(presentation, RulePresentation)
    (span,) = presentation.judgments
    (slot,) = presentation.axioms
    sorts = tuple(
        binder.sort
        for binder in span.indices[2:]
        if isinstance(binder.sort, UserIndexSort)
    )
    arity = len(sorts)
    if key not in DEDUCTION_OUTCOME_KEYS or (key == "form" and arity != 2):
        raise InvalidHandlerError(f"unavailable deduction outcome key {key!r}")
    if len(terms) != len(semantics.words) * arity:
        raise InvalidHandlerError("deduction outcome lexical terms have the wrong size")
    if weights.shape != (len(semantics.words) + len(semantics.parameters),):
        raise InvalidHandlerError("deduction outcome weights have the wrong size")
    # A derivation of n tokens has n lexical leaves and at most n - 1
    # internal nodes. Leave room for its log product, the joins at every
    # item, and the final subtraction, so finite mass cannot underflow
    # to the distinguished zero during the fold.
    limit = torch.finfo(weights.dtype).max / max(1, 4 * len(tokens))
    demand(
        ((weights.abs() <= limit) | torch.isneginf(weights)).all(),
        lambda: EvaluationError(
            "deduction outcome log weights exceed the finite chart range"
        ),
    )
    names = {identity: name for name, identity in semantics.item_constructors}
    lexicon: dict[str, list[tuple[int, tuple[Ground, ...]]]] = {}
    for index, word in enumerate(semantics.words):
        fields = terms[index * arity : (index + 1) * arity]
        args = tuple(
            _ground(field, names, sort)
            for field, sort in zip(fields, sorts, strict=True)
        )
        lexicon.setdefault(word, []).append((index, args))

    axioms = []
    inputs: dict[ParameterId | OccurrenceId, torch.Tensor] = dict(
        zip(semantics.parameters, weights[len(semantics.words) :].unbind(), strict=True)
    )
    bindings = tuple(binder.name for binder in slot.telescope)
    for position, token in enumerate(tokens):
        if not isinstance(token, str):
            raise InvalidHandlerError("a deduction outcome requires a string sentence")
        for index, args in lexicon.get(token, ()):
            occurrence = OccurrenceId.derive(
                presentation.id, "lexicon", position, index
            )
            axioms.append(
                AxiomOccurrence(
                    slot=slot.id,
                    occurrence=occurrence,
                    binding=tuple(
                        zip(bindings, (position, position + 1, *args), strict=True)
                    ),
                    values=(token,),
                )
            )
            inputs[occurrence] = weights[index]
    forest = build_forest(presentation, axioms)
    goals = tuple(
        item
        for item in forest.items
        if item.arguments[:2] == (0, len(tokens))
        and (semantics.start is None or item.arguments[2] == semantics.start)
    )
    values = forest_folds(
        forest,
        goals,
        LOG_PROB,
        lambda edge: edge_weight(forest, edge, LOG_PROB, inputs, like=weights),
        like=weights,
    )
    grouped: dict[Ground, list[torch.Tensor]] = {}
    for goal, weight in values.items():
        outcome = (
            (semantics.span, *goal.arguments)
            if key == "item"
            else goal.arguments[2 if key == "category" else 3]
        )
        grouped.setdefault(outcome, []).append(weight)
    rows = {}
    for outcome, weights_for_key in grouped.items():
        mass = LOG_PROB.join(torch.stack(weights_for_key), 0)
        if not decide(torch.isneginf(mass), "deduction outcome support"):
            demand(
                torch.isfinite(mass),
                lambda: EvaluationError("deduction outcome log mass is not finite"),
            )
            rows[outcome] = mass
    if not normalize:
        return rows
    if not rows:
        raise EvaluationError(
            "cannot normalize deduction outcomes without positive goal mass"
        )
    normalizer = LOG_PROB.join(torch.stack(tuple(rows.values())), 0)
    demand(
        torch.isfinite(normalizer),
        lambda: EvaluationError("deduction outcome normalizer is not finite"),
    )
    normalized = {outcome: weight - normalizer for outcome, weight in rows.items()}
    demand(
        torch.isfinite(torch.stack(tuple(normalized.values()))).all(),
        lambda: EvaluationError("normalized deduction outcome log mass is not finite"),
    )
    return normalized


def deduction_outcomes_handler(
    definition: HandlerDef,
    operation: OperationId,
    semantics: DeductionOutcomes,
) -> RuntimeHandler:
    """Provide one chart query with the semantics fixed by its descriptor.

    Parameters
    ----------
    definition
        The checked handler declaration.
    operation
        The identity of its single linear operation.
    semantics
        The data specifying the chart and its typed inputs and result.

    Returns
    -------
    RuntimeHandler
        The implementation of the declared operation.
    """

    def query(
        request: RuntimeRequest, _resume: Resumption, context: ClauseContext
    ) -> object:
        """Resume with the checked result family's list of weighted keys.

        Parameters
        ----------
        request
            The checked query operation and its arguments.
        _resume
            The continuation, resumed by the tail directive.
        context
            The runtime context supplying the run's number domain.

        Returns
        -------
        TailResume
            The continuation directive carrying the typed outcome list.

        Raises
        ------
        InvalidHandlerError
            If the actual request values are malformed.
        """
        tokens, terms, weights, key, normalize = request.arguments
        if not (
            isinstance(tokens, tuple)
            and isinstance(terms, tuple)
            and isinstance(weights, torch.Tensor)
            and isinstance(key, str)
        ):
            raise InvalidHandlerError("malformed deduction outcome query")
        rows = _chart_outcomes(
            semantics,
            tokens,
            terms,
            weights,
            key,
            decide(normalize, "outcome normalization"),
        )
        numbers = context.attachments.numbers
        constructors = dict(semantics.item_constructors)
        result = RuntimeConstructor(
            constructor=semantics.empty,
            static_arguments=(),
            fields=(),
            result_type=definition.output_type,
        )
        for outcome, weight in reversed(tuple(rows.items())):
            result = RuntimeConstructor(
                constructor=semantics.entry,
                static_arguments=(),
                fields=(
                    _item(outcome, constructors, semantics, numbers),
                    weight,
                    result,
                ),
                result_type=definition.output_type,
            )
        return TailResume(value=result)

    return RuntimeHandler(
        definition=definition,
        clauses={
            operation: RuntimeClause(
                invoke=query,
                result_validator=lambda result: (
                    isinstance(result, RuntimeConstructor)
                    and result.result_type == definition.output_type
                ),
            )
        },
        duplicable_context=True,
    )
