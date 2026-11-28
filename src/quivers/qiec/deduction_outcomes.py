"""Execute the checked finite lexical outcome query on the packed forest."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from fractions import Fraction

import torch

from quivers._positions import demand
from quivers._values import HeldValue
from quivers.core.algebras import LOG_PROB
from quivers.qiec._lexical_outcome_plan import OUTCOME_ALGORITHM, raw_item_plan_trace
from quivers.qiec.canonical import sampleable_type, tensor_type
from quivers.qiec.checking import KernelError, KernelRegistry
from quivers.qiec.contracts import OccurrenceId, ParameterId
from quivers.qiec.declarations import FamilyDecl
from quivers.qiec.effects import HandlerDef, ResumptionGrade
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
from quivers.qiec.foreign import (
    DEDUCTION_OUTCOME_KEYS,
    DeductionBooleanOutcomes,
    DeductionOutcomeGrouping,
    DeductionOutcomeLaw,
    DeductionOutcomes,
    DeductionProofs,
    _NodeBoundedDeductionProofs,
)
from quivers.qiec.identifiers import ConstructorId, OperationId
from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
from quivers.qiec.numbers import NumberDomain, decide, discrete_key, host_index
from quivers.qiec.rules import (
    AxiomOccurrence,
    Ground,
    Item,
    RulePresentation,
    closed_derivation_value,
)
from quivers.qiec.terms import Return
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    UNIT,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    TypeApplication,
    TypeExpr,
    render_static,
)
from quivers.stochastic.forest import (
    DerivationForest,
    ForestRefusal,
    ForestRefused,
    _weights_only,
    build_forest,
    edge_weight,
    fold_derivations_many,
    forest_folds,
)


def proof_entry_type(
    family: FamilyDecl,
) -> tuple[tuple[IndexBinder, ...], TypeApplication]:
    """Bind a proof's indices existentially in an unindexed result row.

    Parameters
    ----------
    family
        The exported lexical proof family.

    Returns
    -------
    tuple
        The row's constructor binders and its field type at those binders.
    """
    indices = tuple(
        binder.with_(refinable=False)
        for binder in family.indices
        if isinstance(binder, IndexBinder)
    )
    assert len(indices) == len(family.indices) and not family.parameters
    proof = TypeApplication(
        constructor=family.type_constructor,
        arguments=tuple(
            IndexVariable(name=binder.name, sort=binder.sort) for binder in indices
        ),
    )
    return indices, proof


def validate_outcome_adapter(
    definition: HandlerDef,
    semantics: DeductionOutcomes | DeductionProofs | DeductionBooleanOutcomes,
    registry: KernelRegistry,
) -> None:
    """Check how a lexical span presentation exchanges typed values with QIEC.

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
    lexical = (
        semantics.chart
        if isinstance(semantics, (DeductionProofs, DeductionBooleanOutcomes))
        else semantics
    )
    presentation = lexical.presentation
    assert isinstance(presentation, RulePresentation)
    (form,) = presentation.judgments
    result_binders: tuple[IndexBinder, ...] = ()
    key_type = lexical.item_type
    control = STRING
    if isinstance(semantics, DeductionProofs):
        (family,), constructors = semantics.proof_presentation.declarations()
        if registry.families.get(family.id) != family or any(
            registry.constructors.get(constructor.id) != constructor
            for constructor in constructors
        ):
            raise KernelError(
                "deduction proof adapter must expose its checked proof declarations"
            )
        result_binders, key_type = proof_entry_type(family)
        control = INT
    elif isinstance(semantics, DeductionBooleanOutcomes):
        key_type = BOOL
        control = tensor_type(
            BOOL, (IndexLiteral(value=semantics.table_size, sort=NAT),)
        )
    output = definition.output_type
    empty = registry.constructors.get(lexical.empty)
    entry = registry.constructors.get(lexical.entry)
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
        or entry.telescope != result_binders
        or tuple(field.type for field in entry.fields) != (key_type, REAL, output)
    ):
        raise KernelError("deduction outcome adapter has an invalid result family")
    constructor_ids = dict(lexical.item_constructors)
    item_prefix = lexical.item_type.constructor.name.removesuffix("__Item")
    for name, identity in constructor_ids.items():
        constructor = registry.constructors.get(identity)
        if (
            constructor is None
            or constructor.name != f"{item_prefix}__{name}"
            or constructor.telescope
            or lexical.item_type
            != TypeApplication(
                constructor=registry.families[constructor.family].type_constructor,
                arguments=(),
            )
            or any(
                field.type not in (lexical.item_type, INT)
                for field in constructor.fields
            )
        ):
            raise KernelError(
                "deduction outcome adapter has an invalid item constructor"
            )
    span_fields = registry.constructors[constructor_ids[lexical.span]].fields
    if tuple(field.type for field in span_fields) != (
        INT,
        INT,
        *((lexical.item_type,) * (len(form.indices) - 2)),
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
        ) != ((lexical.item_type,) * arity):
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
        value=len(lexical.words) * (len(form.indices) - 2), sort=NAT
    )
    weight_count = IndexLiteral(
        value=len(lexical.words) + len(lexical.parameters), sort=NAT
    )
    expected = (
        tensor_type(STRING, (extent,)),
        tensor_type(lexical.item_type, (term_count,)),
        tensor_type(REAL, (weight_count,)),
        control,
        BOOL,
    )
    if (
        tuple(argument.type for argument in operation.arguments) != expected
        or operation.result_type != output
    ):
        raise KernelError("deduction outcome adapter has an invalid query signature")


def validate_outcome_key(type_: TypeExpr, registry: KernelRegistry) -> None:
    """Require a discrete scalar or a closed regular algebraic data type.

    On this domain constructor values compare their nominal identities and
    their ordered fields recursively. Each admitted scalar representation
    has exact value equality, so the resulting equality is structural by
    induction on finite values. The visited set handles recursive families.

    Parameters
    ----------
    type_ : TypeExpr
        The checked result type of the key computation.
    registry : KernelRegistry
        The closed families and constructor declarations used by the type.

    Raises
    ------
    KernelError
        The type or one of its fields lacks the supported structural equality.
    """
    families = {
        family.type_constructor: family for family in registry.families.values()
    }
    pending = [type_]
    visited: set[TypeExpr] = set()
    while pending:
        current = pending.pop()
        if current in visited or current in (BOOL, INT, STRING, UNIT):
            continue
        visited.add(current)
        family = (
            families.get(current.constructor)
            if isinstance(current, TypeApplication)
            else None
        )
        if family is None or not family.closed or family.parameters or family.indices:
            raise KernelError(
                "deduction outcome keys require exact structural equality; "
                f"{render_static(current)} is outside the supported key types"
            )
        for identity in family.constructors:
            constructor = registry.constructors.get(identity)
            if constructor is None or constructor.telescope:
                raise KernelError(
                    "deduction outcome keys require exact structural equality; "
                    f"{family.name!r} must have ordinary closed constructors"
                )
            pending.extend(field.type for field in constructor.fields)


def validate_outcome_grouping(
    definition: HandlerDef, grouping: DeductionOutcomeGrouping, registry: KernelRegistry
) -> None:
    """Check a grouping handler against its typed data family and operation.

    Parameters
    ----------
    definition : HandlerDef
        The checked handler declaration to validate.
    grouping : DeductionOutcomeGrouping
        The descriptor naming the key type and row constructors.
    registry : KernelRegistry
        The declarations against which the descriptor is checked.

    Raises
    ------
    KernelError
        The key type, row family, or grouping operation has an invalid signature.
    """
    output = definition.output_type
    if definition.telescope or definition.input_type != output:
        raise KernelError("deduction outcome grouping has an invalid result family")
    _validate_outcome_row_family(output, grouping, registry)
    effect = registry.effects[definition.effect.id]
    if len(effect.operations) != 1 or len(definition.clauses) != 1:
        raise KernelError("deduction outcome grouping needs one operation")
    (operation,) = effect.operations
    if (
        effect.telescope
        or operation.telescope
        or tuple(argument.type for argument in operation.arguments) != (output, BOOL)
        or operation.result_type != output
    ):
        raise KernelError("deduction outcome grouping has an invalid query signature")


def _validate_outcome_row_family(
    output: TypeExpr, grouping: DeductionOutcomeGrouping, registry: KernelRegistry
) -> None:
    """Check the nominal closed row family before consuming its values.

    Parameters
    ----------
    output
        The nominal type of the weighted row list.
    grouping
        The expected key type and constructor identities.
    registry
        Constructor and family declarations for those identities.

    Raises
    ------
    KernelError
        The row family or the key type is invalid.
    """
    validate_outcome_key(grouping.key_type, registry)
    empty = registry.constructors.get(grouping.empty)
    entry = registry.constructors.get(grouping.entry)
    if (
        empty is None
        or entry is None
        or empty.family != entry.family
        or empty.fields
        or empty.telescope
        or entry.telescope
        or tuple(field.type for field in entry.fields)
        != (grouping.key_type, REAL, output)
    ):
        raise KernelError("deduction outcome grouping has an invalid result family")
    family = registry.families[entry.family]
    if (
        not family.closed
        or family.parameters
        or family.indices
        or set(family.constructors) != {grouping.empty, grouping.entry}
        or output != TypeApplication(constructor=family.type_constructor, arguments=())
    ):
        raise KernelError("deduction outcome grouping has an invalid result family")


def validate_outcome_law(
    definition: HandlerDef, law: DeductionOutcomeLaw, registry: KernelRegistry
) -> None:
    """Check the row and result types of a finite outcome law handler.

    Parameters
    ----------
    definition
        The handler declaration for the conversion operation.
    law
        The exact nominal layout of the weighted input rows.
    registry
        The checked effect and type declarations.

    Raises
    ------
    KernelError
        The handler does not implement the declared finite sampleable conversion.
    """
    effect = registry.effects[definition.effect.id]
    if len(effect.operations) != 1 or len(definition.clauses) != 1:
        raise KernelError("deduction outcome law needs one operation")
    (operation,) = effect.operations
    output = sampleable_type(law.layout.key_type)
    if (
        effect.telescope
        or operation.telescope
        or definition.telescope
        or len(operation.arguments) != 1
        or definition.clauses[0].grade is not ResumptionGrade.LINEAR
        or operation.result_type != output
        or definition.input_type != output
        or definition.output_type != output
    ):
        raise KernelError("deduction outcome law has an invalid query signature")
    _validate_outcome_row_family(operation.arguments[0].type, law.layout, registry)


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


def _lexical_forest(
    semantics: DeductionOutcomes,
    tokens: Sequence[object],
    terms: Sequence[object],
    weights: torch.Tensor,
) -> tuple[
    DerivationForest, dict[ParameterId | OccurrenceId, torch.Tensor], tuple[Item, ...]
]:
    """Ground shared lexical proof and outcome queries once.

    Parameters
    ----------
    semantics
        The finite span presentation and its ordered weight addresses.
    tokens, terms
        The input sentence and lexical entries' category or form values.
    weights
        Lexical log weights followed by rule log weights.

    Returns
    -------
    tuple
        The packed forest, differentiable weights, and accepted full spans.

    Raises
    ------
    InvalidHandlerError
        If the words or lexical values disagree with the typed presentation.
    EvaluationError
        If the supplied log values exceed the finite arithmetic range.
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
    if len(terms) != len(semantics.words) * arity:
        raise InvalidHandlerError("deduction outcome lexical terms have the wrong size")
    if weights.shape != (len(semantics.words) + len(semantics.parameters),):
        raise InvalidHandlerError("deduction outcome weights have the wrong size")
    # Nullable trees can have a large finite node bound. Dividing exact
    # integers keeps the bound from overflowing a floating-point divisor.
    divisor = semantics.span_order.weight_range_divisor(len(tokens))
    limit = float(Fraction(torch.finfo(weights.dtype).max) / divisor)
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
    return forest, inputs, goals


def _chart_outcomes(
    semantics: DeductionOutcomes,
    tokens: Sequence[object],
    terms: Sequence[object],
    weights: torch.Tensor,
    key: str,
    normalize: bool,
    *,
    context: ClauseContext,
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
    context
        The runtime trace hook, if the invocation has one.

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
    if key not in DEDUCTION_OUTCOME_KEYS or (key == "form" and len(span.indices) != 4):
        raise InvalidHandlerError(f"unavailable deduction outcome key {key!r}")
    forest, inputs, goals = _lexical_forest(semantics, tokens, terms, weights)
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
    rows = _outcome_masses(grouped, normalize)
    context.emit_trace(
        "deduction.outcomes",
        lambda: _outcome_trace(forest, key, normalize, len(goals), len(rows)),
    )
    return rows


def _outcome_trace(
    forest: DerivationForest,
    key: str,
    normalized: bool,
    goals: int,
    rows: int,
) -> Mapping[str, object]:
    """Report the observed schedule and counters of one completed outcome fold.

    Parameters
    ----------
    forest
        The actual indexed forest built for the request.
    key, normalized
        The field selected as the outcome key and whether the returned
        log masses subtract their common log total.
    goals, rows
        The number of full-span goal items and positive-mass result rows.

    Returns
    -------
    Mapping of str to object
        An execution record with construction measurements only.
    """
    return {
        **_packed_trace(forest, normalized, goals, rows),
        "observer": "weighted-outcome-rows",
        "key": key,
        "algorithm": OUTCOME_ALGORITHM,
        "unmeasured": (
            "fold_arithmetic",
            "backward_arithmetic",
            "elements_saved_for_backward",
            "peak_bytes",
        ),
    }


def _packed_trace(
    forest: DerivationForest, normalized: bool, goals: int, rows: int
) -> Mapping[str, object]:
    """Report the measurements common to completed lexical forest queries.

    Parameters
    ----------
    forest
        The indexed forest built by the invocation.
    normalized
        Whether the query subtracts the common normalizer.
    goals, rows
        The number of full-span goal items and positive-weight result rows.

    Returns
    -------
    Mapping of str to object
        The finite weight contract and measured construction work.
    """
    construction = forest.report()
    return {
        "presentation": str(forest.presentation.id),
        "normalized": normalized,
        "recursion": "finite-acyclic",
        "algebra": "LogProb",
        "derivative": "autograd-on-finite-log-weights-at-fixed-support",
        "retention": "packed-forest-during-query",
        **(
            {"checked_rule_guards": forest.work.conditions}
            if forest.presentation.conditions
            else {}
        ),
        "goal_items": goals,
        "positive_rows": rows,
        "construction_work": tuple(
            (measure.value, getattr(construction, measure.value))
            for measure in construction.measured
        ),
    }


def _outcome_masses[K](
    grouped: Mapping[K, list[torch.Tensor]], normalize: bool
) -> dict[K, torch.Tensor]:
    """Sum all occurrences per key, then normalize the positive support.

    Parameters
    ----------
    grouped : Mapping[K, list[torch.Tensor]]
        Each key's nonempty list of scalar log masses.
    normalize : bool
        Whether to subtract the total log mass from the supported rows.

    Returns
    -------
    dict[K, torch.Tensor]
        The grouped log masses with zero-mass keys omitted.

    Raises
    ------
    EvaluationError
        A supported mass or normalizer is nonfinite, or normalization was
        requested for empty support.
    """
    rows: dict[K, torch.Tensor] = {}
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


def _materialize_rows(
    output: TypeExpr,
    empty: ConstructorId,
    entry: ConstructorId,
    rows: Iterable[tuple[HeldValue, torch.Tensor]],
) -> RuntimeConstructor:
    """Produce the typed row list in first-occurrence order.

    Parameters
    ----------
    output : TypeExpr
        The checked row-list family type.
    empty : ConstructorId
        The nullary constructor for the end of the list.
    entry : ConstructorId
        The constructor for a key, log mass, and remaining rows.
    rows : Iterable[tuple[HeldValue, torch.Tensor]]
        The key and scalar log mass of each retained row, in order.

    Returns
    -------
    RuntimeConstructor
        The head of the typed row list, or its empty constructor.
    """
    result = RuntimeConstructor(
        constructor=empty, static_arguments=(), fields=(), result_type=output
    )
    for key, weight in reversed(tuple(rows)):
        result = RuntimeConstructor(
            constructor=entry,
            static_arguments=(),
            fields=(key, weight, result),
            result_type=output,
        )
    return result


def _typed_key(value: HeldValue, type_: TypeExpr) -> Hashable:
    """Use the machine's exact equality on the checked key domain.

    Parameters
    ----------
    value : HeldValue
        The runtime result of the key computation.
    type_ : TypeExpr
        The checked key result type.

    Returns
    -------
    Hashable
        A key using the runtime's value equality for this type.

    Raises
    ------
    InvalidHandlerError
        The runtime value does not have the expected representation.
    """
    if type_ == STRING and isinstance(value, str):
        return value
    if type_ == UNIT and value is None:
        return None
    if (
        type_ in (BOOL, INT)
        and isinstance(value, torch.Tensor)
        and value.ndim == 0
        and not value.is_floating_point()
        and not value.is_complex()
        and (value.dtype == torch.bool) == (type_ == BOOL)
    ):
        return discrete_key(value)
    if isinstance(value, RuntimeConstructor) and value.result_type == type_:
        return value
    raise InvalidHandlerError("a grouped deduction outcome has an invalid typed key")


def outcome_grouping_handler(
    definition: HandlerDef,
    operation: OperationId,
    grouping: DeductionOutcomeGrouping,
) -> RuntimeHandler:
    """Push forward checked weighted rows without traversing their proofs.

    Parameters
    ----------
    definition : HandlerDef
        The checked grouping handler declaration.
    operation : OperationId
        The operation that groups rows and optionally normalizes their weights.
    grouping : DeductionOutcomeGrouping
        The checked descriptor for the input and output row representation.

    Returns
    -------
    RuntimeHandler
        The handler that sums scalar log weights by their checked key.
    """

    def group(
        request: RuntimeRequest, _resume: Resumption, context: ClauseContext
    ) -> TailResume:
        """Fold the typed input rows and resume with their grouped masses.

        Parameters
        ----------
        request : RuntimeRequest
            The evaluated rows and the normalization flag.
        _resume : Resumption
            The resumption supplied by the machine.
        context : ClauseContext
            The optional execution trace supplied by the machine.

        Returns
        -------
        TailResume
            A tail resumption with the grouped rows.

        Raises
        ------
        InvalidHandlerError
            The input does not have the checked row representation.
        """
        row, normalize = request.arguments
        grouped: dict[Hashable, list[torch.Tensor]] = {}
        representatives: dict[Hashable, HeldValue] = {}
        while (
            isinstance(row, RuntimeConstructor)
            and row.result_type == definition.output_type
        ):
            if row.constructor == grouping.empty and not row.fields:
                normalized = decide(normalize, "outcome normalization")
                masses = _outcome_masses(grouped, normalized)
                context.emit_trace(
                    "deduction.grouped_outcomes",
                    lambda: {
                        "observer": "weighted-typed-key-rows",
                        "algorithm": "hash-grouped-structural-keys",
                        "key_type": render_static(grouping.key_type),
                        "normalized": normalized,
                        "derivative": "autograd-on-finite-log-weights-at-fixed-support",
                        "input_rows": sum(len(weights) for weights in grouped.values()),
                        "distinct_keys": len(grouped),
                        "positive_rows": len(masses),
                        "unmeasured": (
                            "key_computation",
                            "key_hash_and_equality",
                            "group_arithmetic",
                            "backward_arithmetic",
                            "elements_saved_for_backward",
                            "peak_bytes",
                        ),
                    },
                )
                return TailResume(
                    value=_materialize_rows(
                        definition.output_type,
                        grouping.empty,
                        grouping.entry,
                        ((representatives[key], mass) for key, mass in masses.items()),
                    )
                )
            if row.constructor != grouping.entry or len(row.fields) != 3:
                break
            value, weight, row = row.fields
            if (
                not isinstance(weight, torch.Tensor)
                or weight.ndim != 0
                or not weight.is_floating_point()
            ):
                break
            key = _typed_key(value, grouping.key_type)
            representatives.setdefault(key, value)
            grouped.setdefault(key, []).append(weight)
        raise InvalidHandlerError("malformed deduction outcome rows")

    return RuntimeHandler(
        definition=definition,
        clauses={
            operation: RuntimeClause(
                invoke=group,
                result_validator=lambda result: (
                    isinstance(result, RuntimeConstructor)
                    and result.result_type == definition.output_type
                ),
            )
        },
        duplicable_context=True,
    )


def outcome_law_handler(
    definition: HandlerDef,
    operation: OperationId,
    law: DeductionOutcomeLaw,
    registry: KernelRegistry,
) -> RuntimeHandler:
    """Bind a checked finite law to the live masses in a weighted outcome table.

    Parameters
    ----------
    definition
        The checked law-conversion handler.
    operation
        Its one linear conversion operation.
    law
        The declared input row layout and output carrier.
    registry
        The nominal constructor and effect declarations for this invocation.

    Returns
    -------
    RuntimeHandler
        The pure conversion implementation. Random consumes its result separately.
    """
    from didactic.api import ValidationError

    from quivers.qiec.distributions import DistributionError
    from quivers.qiec.structured_sampleables import FiniteSampleable

    layout = law.layout
    (conversion,) = registry.effects[definition.effect.id].operations
    row_type = conversion.arguments[0].type

    def convert(
        request: RuntimeRequest, _resume: Resumption, _context: ClauseContext
    ) -> TailResume:
        """Read the finite rows and resume with their normalized law.

        Parameters
        ----------
        request
            The single argument is the checked weighted row list.
        _resume
            The machine's continuation, used through the returned tail directive.
        _context
            The invocation context for this pure conversion.

        Returns
        -------
        TailResume
            A probabilistic value over the original outcome keys.

        Raises
        ------
        InvalidHandlerError
            If rows are malformed or there is no positive total mass.
        """
        (row,) = request.arguments
        values: list[HeldValue] = []
        weights: list[torch.Tensor] = []
        seen: set[int] = set()
        while (
            isinstance(row, RuntimeConstructor)
            and row.result_type == row_type
            and not row.static_arguments
            and id(row) not in seen
        ):
            seen.add(id(row))
            if row.constructor == layout.empty and not row.fields:
                if not weights:
                    raise InvalidHandlerError(
                        "deduction outcomes have no positive mass"
                    )
                try:
                    result = FiniteSampleable(
                        element=layout.key_type,
                        values=tuple(values),
                        log_weights=torch.stack(weights),
                        registry=registry,
                    )
                except (DistributionError, ValidationError) as error:
                    raise InvalidHandlerError(str(error)) from error
                return TailResume(value=result)
            if row.constructor != layout.entry or len(row.fields) != 3:
                break
            value, weight, row = row.fields
            if (
                not isinstance(weight, torch.Tensor)
                or weight.ndim != 0
                or not weight.is_floating_point()
                or (
                    weights
                    and (weight.device, weight.dtype)
                    != (weights[0].device, weights[0].dtype)
                )
            ):
                break
            values.append(value)
            weights.append(weight)
        raise InvalidHandlerError("malformed deduction outcome rows")

    return RuntimeHandler(
        definition=definition,
        clauses={
            operation: RuntimeClause(
                invoke=convert,
                result_validator=lambda value: (
                    isinstance(value, FiniteSampleable)
                    and value.element == layout.key_type
                ),
            )
        },
        duplicable_context=True,
    )


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
        normalized = decide(normalize, "outcome normalization")
        rows = _chart_outcomes(
            semantics,
            tokens,
            terms,
            weights,
            key,
            normalized,
            context=context,
        )
        numbers = context.attachments.numbers
        constructors = dict(semantics.item_constructors)
        result = _materialize_rows(
            definition.output_type,
            semantics.empty,
            semantics.entry,
            (
                (_item(outcome, constructors, semantics, numbers), weight)
                for outcome, weight in rows.items()
            ),
        )
        presentation = semantics.presentation
        assert isinstance(presentation, RulePresentation)
        if key == "item" and not normalized and not presentation.conditions:
            context.emit_trace(
                "deduction.outcomes.plan",
                lambda: raw_item_plan_trace(
                    semantics, definition, operation, len(tokens), len(terms), weights
                ),
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


def deduction_proofs_handler(
    definition: HandlerDef, operation: OperationId, semantics: DeductionProofs
) -> RuntimeHandler:
    """Provide a finite lexical proof list with existentially indexed rows.

    Parameters
    ----------
    definition, operation
        The checked handler and its single linear operation.
    semantics
        The checked span grammar and proof row constructors.

    Returns
    -------
    RuntimeHandler
        The handler running one packed proof enumeration for the query.
    """

    def query(
        request: RuntimeRequest, _resume: Resumption, context: ClauseContext
    ) -> TailResume:
        """Build typed values only after verifying the global proof limit.

        Parameters
        ----------
        request
            Sentence, lexical values, weights, proof limit, and normalization.
        _resume
            The continuation resumed by the returned directive.
        context
            The evaluator and number domain for reifying checked proof values.

        Returns
        -------
        TailResume
            The list of supported proofs and their differentiable weights.

        Raises
        ------
        InvalidHandlerError
            If the operation arguments have the wrong runtime representation.
        EvaluationError
            If the budget is nonpositive or exceeded, or normalization fails.
        """
        tokens, terms, weights, budget, normalize = request.arguments
        if not (
            isinstance(tokens, tuple)
            and isinstance(terms, tuple)
            and isinstance(weights, torch.Tensor)
        ):
            raise InvalidHandlerError("malformed deduction proof query")
        limit = host_index(budget, "deduction proof budget")
        if limit < 1:
            raise EvaluationError("deduction proof budgets must be positive")
        node_limit = (
            semantics.node_limit
            if isinstance(semantics, _NodeBoundedDeductionProofs)
            else None
        )
        forest, inputs, goals = _lexical_forest(semantics.chart, tokens, terms, weights)
        try:
            by_goal = fold_derivations_many(
                forest,
                goals,
                LOG_PROB,
                inputs,
                _weights_only(forest),
                limit=limit,
                like=weights,
                node_limit=node_limit,
            )
        except ForestRefused as error:
            reason = error.code.value
            context.emit_trace(
                "deduction.proofs.refused",
                lambda: {
                    "presentation": str(forest.presentation.id),
                    "observer": "weighted-proof-rows",
                    "proof_budget": limit,
                    "goal_items": len(goals),
                    "reason": reason,
                    **(
                        {"proof_node_budget": node_limit}
                        if node_limit is not None
                        else {}
                    ),
                    **(
                        {"minimum_structural_proof_nodes": node_limit + 1}
                        if node_limit is not None
                        and reason == ForestRefusal.TOO_MANY_NODES.value
                        else {}
                    ),
                },
            )
            raise EvaluationError(str(error)) from error
        listed = tuple(
            (proof, weight) for rows in by_goal.values() for proof, weight, _ in rows
        )
        normalized = decide(normalize, "proof normalization")
        masses = _outcome_masses(
            {index: [mass] for index, (_, mass) in enumerate(listed)},
            normalized,
        )
        result = RuntimeConstructor(
            constructor=semantics.chart.empty,
            static_arguments=(),
            fields=(),
            result_type=definition.output_type,
        )
        for index, weight in reversed(tuple(masses.items())):
            proof = closed_derivation_value(
                semantics.proof_presentation, listed[index][0]
            )
            value = context.evaluate(Return(value=proof))
            assert isinstance(value, RuntimeConstructor)
            assert isinstance(proof.result_type, TypeApplication)
            result = RuntimeConstructor(
                constructor=semantics.chart.entry,
                static_arguments=proof.result_type.arguments,
                fields=(value, weight, result),
                result_type=definition.output_type,
            )
        context.emit_trace(
            "deduction.proofs",
            lambda: {
                **_packed_trace(forest, normalized, len(goals), len(masses)),
                "observer": "weighted-proof-rows",
                "algorithm": "indexed-packed-forest-budgeted-enumeration",
                "proof_budget": limit,
                **({"proof_node_budget": node_limit} if node_limit is not None else {}),
                "structural_goal_proofs": len(listed),
                "unmeasured": (
                    "enumeration_arithmetic",
                    "reified_proof_nodes",
                    "backward_arithmetic",
                    "elements_saved_for_backward",
                    "peak_bytes",
                ),
            },
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


__all__: list[str] = []
