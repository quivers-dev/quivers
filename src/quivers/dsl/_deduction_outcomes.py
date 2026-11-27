"""Checked elaboration of typed finite lexical outcome queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import didactic.api as dx

from quivers.core.algebras import LOG_PROB
from quivers.qiec.canonical import tensor_type
from quivers.qiec.checking import ComputationSignature
from quivers.qiec.contracts import ParameterId
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.effects import (
    ArgumentDef,
    EffectDef,
    EffectRef,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    instantiate_effect,
)
from quivers.qiec.foreign import DeductionOutcomes
from quivers.qiec.identifiers import (
    ConstructorId,
    EffectId,
    FamilyId,
    HandlerId,
    OperationId,
)
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.terms import (
    Bind,
    Computation,
    Handle,
    Local,
    Return,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    REAL,
    STRING,
    IndexLiteral,
    IndexVariable,
    TypeApplication,
)

if TYPE_CHECKING:
    from quivers.dsl.deduction_elaboration import _Context, _System
    from quivers.dsl.qiec_lowering import _Elaborator


OUTCOME_CALLS = frozenset({"parse_outcomes", "parse_normalized_outcomes"})
OUTCOME_CODE = "qiec-deduction-outcomes"
_SPAN = "span"
_QUERY = "query"


@dataclass(frozen=True)
class OutcomeQuery:
    """The registered query, its ordered rule weights, and its result family."""

    instance: NamedEffectInstance
    handler: HandlerDef
    terms: TypeApplication
    weights: TypeApplication
    production_indices: tuple[int, ...]


def declare_outcomes(
    elaborator: _Elaborator, system: _System, node: object
) -> OutcomeQuery:
    """Declare the result family and one operation fixed to this presentation."""
    if system.semiring.algebra != LOG_PROB.subject or not system.entries:
        elaborator._fail(
            node,
            "outcome queries require a lexical LogProb deduction",
            code=OUTCOME_CODE,
        )
    if system.depth_cell[0] is not None:
        elaborator._fail(
            node,
            "outcome queries do not yet support a category depth bound",
            code=OUTCOME_CODE,
        )
    if not system.normalized_cell[0] and any(
        rule.presented.weighted for rule in system.rules
    ):
        elaborator._fail(
            node,
            "outcome queries require fixed or normalize=lhs rule weights. Free weights keyed by a rule's bindings are outside this fragment",
            code=OUTCOME_CODE,
        )
    presented = system.presentation_cell[0]
    assert presented is not None
    presentation = presented.presentation
    module = elaborator.source.module_name
    production_indices: tuple[int, ...] = ()
    parameters: dict[int, ParameterId] = {}
    if system.normalized_cell[0]:
        production_indices = tuple(
            dict.fromkeys(rule.index for rule in presented.rules)
        )
        parameters = {
            index: ParameterId.derive(presentation.id, "production", index)
            for index in production_indices
        }
        presentation = presentation.with_(
            rules=tuple(
                rule.rule.with_(weight=parameters[rule.index])
                for rule in presented.rules
            )
        )
    family_name = system.name("Outcomes")
    family_id = FamilyId.derive(module, "family", family_name)
    empty_id = ConstructorId.derive(family_id, "empty")
    entry_id = ConstructorId.derive(family_id, "entry")
    family = FamilyDecl(
        id=family_id,
        name=family_name,
        parameters=(),
        indices=(),
        constructors=(empty_id, entry_id),
    )
    result = TypeApplication(constructor=family.type_constructor, arguments=())
    try:
        semantics = DeductionOutcomes(
            presentation=presentation,
            words=tuple(entry.word for entry in system.entries),
            parameters=tuple(parameters.values()),
            span=_SPAN,
            start=system.start,
            item_type=system.item,
            item_constructors=tuple(
                (name, constructor.id)
                for name, constructor in system.constructors.items()
            ),
            empty=empty_id,
            entry=entry_id,
        )
    except (ValueError, dx.ValidationError) as error:
        elaborator._fail_kernel(node, error, fallback=OUTCOME_CODE)
    elaborator.families[family_name] = family
    elaborator.registry.register_family(family)
    empty = ConstructorDecl(
        id=empty_id,
        name=system.name("outcome_end"),
        family=family_id,
        telescope=(),
        fields=(),
        result_indices=(),
    )
    entry = ConstructorDecl(
        id=entry_id,
        name=system.name("outcome_entry"),
        family=family_id,
        telescope=(),
        fields=(
            FieldDef(name="key", type=system.item),
            FieldDef(name="weight", type=REAL),
            FieldDef(name="tail", type=result),
        ),
        result_indices=(),
    )
    for constructor in (empty, entry):
        elaborator.constructors[constructor.name] = constructor
        elaborator.registry.register_constructor(constructor)

    arity = 2 if presented.uses_forms else 1
    terms = tensor_type(
        system.item, (IndexLiteral(value=arity * len(system.entries), sort=NAT),)
    )
    weights = tensor_type(
        REAL, (IndexLiteral(value=len(system.entries) + len(parameters), sort=NAT),)
    )
    effect_name = system.name("OutcomeQuery")
    effect_id = EffectId.derive(module, "effect", effect_name)
    extent = IndexVariable(name="n", sort=NAT)
    operation = OperationDef(
        id=OperationId.derive(effect_id, "operation", _QUERY),
        name=_QUERY,
        telescope=(IndexBinder(name=extent.name, sort=NAT),),
        arguments=(
            ArgumentDef(name="tokens", type=tensor_type(STRING, (extent,))),
            ArgumentDef(name="terms", type=terms),
            ArgumentDef(name="weights", type=weights),
            ArgumentDef(name="key", type=STRING),
            ArgumentDef(name="normalize", type=BOOL),
        ),
        result_type=result,
    )
    effect = EffectDef(
        ref=EffectRef(id=effect_id, name=effect_name),
        telescope=(),
        operations=(operation,),
    )
    elaborator.effects[effect_name] = effect
    elaborator.registry.register_effect(effect)
    instance_name = system.name("outcome_query")
    row = instantiate_effect(
        effect.ref, module=module, lexical_path=("instances", instance_name)
    )
    instance = NamedEffectInstance(
        name=instance_name,
        entry=row,
        origin=elaborator._origin(
            system.decl, ("instances", instance_name), "effect-instance"
        ),
    )
    elaborator.instances[instance_name] = instance
    handler_name = system.name("query_outcomes")
    handler = HandlerDef(
        id=HandlerId.derive(module, "handler", handler_name),
        name=handler_name,
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=result,
        output_type=result,
        introduced=EffectRow(),
        foreign=semantics,
    )
    elaborator.handlers[handler_name] = handler
    return OutcomeQuery(instance, handler, terms, weights, production_indices)


def elaborate_outcomes(
    context: _Context, query: OutcomeQuery, signature: ComputationSignature
) -> NamedComputation:
    """Read weights through the grammar's checked readers and run the query."""
    system = context.system
    tokens, key, normalize = tuple(
        Local(name=name, type=type_)
        for name, type_ in zip(
            ("tokens", "key", "normalize"), context.inputs(signature), strict=True
        )
    )
    terms: list[Value] = []
    for entry in system.entries:
        terms.append(context.ground(entry.category))
        if entry.form is not None:
            terms.append(context.ground(entry.form))
    readings = [
        context.entry_weight(index, entry, lambda weight: Return(value=weight))
        for index, entry in enumerate(system.entries)
    ] + [
        context.production_weight(("rule", index), lambda weight: Return(value=weight))
        for index in query.production_indices
    ]
    values = tuple(system.local("weight", system.carrier) for _ in readings)
    body: Computation = Handle(
        instance=query.instance.entry.instance,
        handler=query.handler.id,
        computation=context.request(
            query.instance,
            _QUERY,
            (context.extent,),
            (
                Var(local=tokens),
                TensorValue(items=tuple(terms), result_type=query.terms),
                TensorValue(
                    items=tuple(context.real_of(Var(local=local)) for local in values),
                    result_type=query.weights,
                ),
                Var(local=key),
                Var(local=normalize),
            ),
            signature.result,
            "outcomes",
        ),
        static_arguments=(),
    )
    for local, reading in reversed(tuple(zip(values, readings, strict=True))):
        body = Bind(binder=local, first=reading, then=body)
    return context.computation(signature, (tokens, key, normalize), body)
