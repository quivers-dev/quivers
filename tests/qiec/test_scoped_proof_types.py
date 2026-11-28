"""Indexed proof constructors enforce ordered scope without a second checker."""

from __future__ import annotations

import pytest

from quivers.qiec._scoped_proof_interpretation import scoped_lookup_computations
from quivers.qiec._scoped_proof_types import scoped_proof_types
from quivers.qiec.checking import KernelError, infer_value
from quivers.qiec.contracts import OccurrenceId, PresentationId
from quivers.qiec.declarations import FieldDef
from quivers.qiec.effects import ComputationType, EffectRow
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import run_named
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    FamilyId,
    SourceOrigin,
)
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.rules import (
    AxiomSlot,
    CARTESIAN,
    ORDERED,
    Derivation,
    Hole,
    Item,
    Judgment,
    JudgmentDecl,
    OpenDerivation,
    Premise,
    Rule,
    RuleError,
    RulePresentation,
    rename_hypotheses,
)
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import Call, ConstructorValue, LiteralValue, Return, Var
from quivers.qiec.types import BOOL, REAL, TypeApplication


_IDENTITY = PresentationId.derive("scoped-proofs-example")
_A = JudgmentDecl(id=FamilyId.derive(_IDENTITY, "A"), name="A")
_B = JudgmentDecl(id=FamilyId.derive(_IDENTITY, "B"), name="B")
_RESULT = JudgmentDecl(id=FamilyId.derive(_IDENTITY, "Result"), name="Result")
A = Item(family=_A.id)
B = Item(family=_B.id)
RESULT = Item(family=_RESULT.id)
_JA = Judgment(family=_A.id)
_JB = Judgment(family=_B.id)
_JR = Judgment(family=_RESULT.id)
_PAIR = Rule(
    id=ConstructorId.derive(_IDENTITY, "pair"),
    name="pair",
    conclusion=_JR,
    premises=(Premise(judgment=_JA), Premise(judgment=_JB)),
)
_INTRODUCE_A = Rule(
    id=ConstructorId.derive(_IDENTITY, "introduce_a"),
    name="introduce_a",
    conclusion=_JR,
    premises=(Premise(judgment=_JR, after=(_JA,)),),
)
_INTRODUCE_B = Rule(
    id=ConstructorId.derive(_IDENTITY, "introduce_b"),
    name="introduce_b",
    conclusion=_JR,
    premises=(Premise(judgment=_JR, after=(_JB,)),),
)
_AXIOM = AxiomSlot(
    id=ConstructorId.derive(_IDENTITY, "given"),
    name="given",
    judgment=_JA,
    values=(FieldDef(name="value", type=REAL),),
)
PRESENTATION = RulePresentation(
    id=_IDENTITY,
    name="ExampleBinding",
    judgments=(_A, _B, _RESULT),
    rules=(_PAIR, _INTRODUCE_A, _INTRODUCE_B),
    axioms=(_AXIOM,),
    structure=CARTESIAN,
)


def _binding_example() -> OpenDerivation:
    pair = Derivation(
        constructor=_PAIR.id,
        item=RESULT,
        premises=(Hole(index=2, item=A), Hole(index=1, item=B)),
    )
    bind_a = Derivation(constructor=_INTRODUCE_A.id, item=RESULT, premises=(pair,))
    bind_b = Derivation(constructor=_INTRODUCE_B.id, item=RESULT, premises=(bind_a,))
    return OpenDerivation(context=(A,), body=bind_b)


def test_nested_scopes_keep_old_and_new_same_sort_positions() -> None:
    syntax = scoped_proof_types(PRESENTATION)
    proof = syntax.value(_binding_example())
    index = proof.static_arguments[-1]
    assert index.constructor == syntax.frames[_A.id].type_constructor
    (inside_b,) = proof.fields
    assert isinstance(inside_b, ConstructorValue)
    assert (
        inside_b.static_arguments[-1].constructor
        == syntax.frames[_B.id].type_constructor
    )
    (pair,) = inside_b.fields
    assert isinstance(pair, ConstructorValue)
    assert (
        pair.static_arguments[-1].constructor == syntax.frames[_A.id].type_constructor
    )
    first, second = pair.fields
    assert isinstance(first, ConstructorValue) and isinstance(second, ConstructorValue)
    (a,) = first.fields
    (b,) = second.fields
    assert isinstance(a, ConstructorValue) and isinstance(b, ConstructorValue)
    assert a.constructor == syntax.there[_A.id, _A.id]
    assert b.constructor == syntax.there[_B.id, _A.id]
    (a_under_b,) = a.fields
    (b_at_top,) = b.fields
    assert isinstance(a_under_b, ConstructorValue) and isinstance(
        b_at_top, ConstructorValue
    )
    assert a_under_b.constructor == syntax.there[_A.id, _B.id]
    assert b_at_top.constructor == syntax.here[_B.id]
    assert infer_value(proof, syntax.registry) == proof.result_type

    origin = SourceOrigin(
        module=syntax.module.module,
        structural_path=("proof",),
        role="computation",
        source_protocol=syntax.module.source_protocol,
    )
    quoted = NamedComputation(
        id=ComputationId.derive(syntax.module.module, "proof"),
        name="proof",
        telescope=(),
        parameters=(),
        type=ComputationType(effects=EffectRow(), result=proof.result_type),
        body=Return(value=proof),
        origin=origin,
    )
    restored = loads(dumps(syntax.module.with_(computations=(quoted,))))
    assert isinstance(restored, QiecModule)
    value = run_named(restored, "proof").value
    assert isinstance(value, RuntimeConstructor)
    assert value.result_type == proof.result_type


def test_renaming_same_sort_hypotheses_changes_membership_witness() -> None:
    syntax = scoped_proof_types(PRESENTATION)
    before = OpenDerivation(context=(A, A), body=Hole(index=0, item=A))
    after = rename_hypotheses(PRESENTATION, before, (1, 0), before.context)
    assert after.body.index == 1
    first = syntax.value(before)
    second = syntax.value(after)
    assert first.result_type == second.result_type
    assert first.fields[0].constructor == syntax.here[_A.id]
    assert second.fields[0].constructor == syntax.there[_A.id, _A.id]


def test_open_proof_formation_refuses_invalid_scope_and_inhabitant() -> None:
    syntax = scoped_proof_types(PRESENTATION)
    with pytest.raises(RuleError, match="does not match its context position"):
        syntax.value(OpenDerivation(context=(A, B), body=Hole(index=0, item=A)))
    with pytest.raises(RuleError, match="lies outside its context"):
        syntax.value(OpenDerivation(context=(A,), body=Hole(index=1, item=A)))
    with pytest.raises(RuleError, match="Cartesian structural permissions"):
        scoped_proof_types(PRESENTATION.with_(structure=ORDERED))
    with pytest.raises(RuleError, match="exactly"):
        syntax.value(
            OpenDerivation(context=(A,), body=Hole(index=0, item=A)), {_A.id: REAL}
        )
    first = syntax.value(OpenDerivation(context=(A,), body=Hole(index=0, item=A)))
    (reference,) = first.fields
    forged = first.with_(fields=(reference.with_(constructor=syntax.here[_B.id]),))
    with pytest.raises(KernelError, match="constructor"):
        infer_value(forged, syntax.registry)


def test_scoped_proof_values_keep_axiom_occurrence_identity() -> None:
    syntax = scoped_proof_types(PRESENTATION)
    conclusion = Derivation(
        constructor=_AXIOM.id,
        item=A,
        values=(3.0,),
        occurrence=OccurrenceId.derive(_IDENTITY, "first"),
    )
    first = syntax.value(OpenDerivation(context=(), body=conclusion))
    second_appearance = conclusion.with_(
        occurrence=OccurrenceId.derive(_IDENTITY, "second")
    )
    second = syntax.value(OpenDerivation(context=(), body=second_appearance))
    assert first.result_type == second.result_type
    assert first.fields[0].value == str(conclusion.occurrence)
    assert second.fields[0].value == str(second_appearance.occurrence)
    assert first != second
    bound = conclusion.with_(binding=(("unknown", 0),))
    with pytest.raises(RuleError, match="static bindings"):
        syntax.value(OpenDerivation(context=(), body=bound))
    assert validate_module(syntax.module).families


def test_serialized_scoped_lookups_follow_exact_positions() -> None:
    syntax = scoped_proof_types(PRESENTATION)
    carriers = {_A.id: REAL, _B.id: BOOL, _RESULT.id: REAL}
    lookups = scoped_lookup_computations(syntax, carriers)

    environment = ConstructorValue(
        constructor=syntax.empty.constructors[0],
        static_arguments=(),
        fields=(),
        result_type=TypeApplication(constructor=syntax.empty.type_constructor),
    )
    for family, value in ((_A.id, 2.5), (_B.id, True), (_A.id, 8.5)):
        frame = syntax.frames[family]
        arguments = (environment.result_type, carriers[family])
        environment = ConstructorValue(
            constructor=frame.constructors[0],
            static_arguments=arguments,
            fields=(environment, LiteralValue(value=value, type=carriers[family])),
            result_type=TypeApplication(
                constructor=frame.type_constructor, arguments=arguments
            ),
        )

    exports = []
    for name, item, index, result in (
        ("outer", A, 2, REAL),
        ("middle", B, 1, BOOL),
        ("inner", A, 0, REAL),
    ):
        open_proof = OpenDerivation(
            context=(A, B, A), body=Hole(index=index, item=item)
        )
        proof = syntax.value(open_proof, carriers)
        assert proof.constructor == syntax.value(open_proof).constructor
        assert proof.result_type != syntax.value(open_proof).result_type
        (reference,) = proof.fields
        lookup = lookups[item.family]
        origin = lookup.origin.with_(structural_path=(name,))
        exports.append(
            NamedComputation(
                id=ComputationId.derive(syntax.module.module, name),
                name=name,
                telescope=(),
                parameters=(),
                type=ComputationType(effects=EffectRow(), result=result),
                body=Call(
                    callee=lookup.id,
                    name=lookup.name,
                    static_arguments=(environment.result_type,),
                    arguments=(reference, environment),
                    result_type=result,
                    effects=EffectRow(),
                    origin=origin,
                ),
                origin=origin,
            )
        )

    module = syntax.module.with_(computations=(*lookups.values(), *exports))
    validate_module(module)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert run_named(restored, "outer").value == 2.5
    assert bool(run_named(restored, "middle").value) is True
    assert run_named(restored, "inner").value == 8.5

    outer = exports[0]
    _, correct_environment = outer.body.arguments
    previous, _ = correct_environment.fields
    forged = outer.with_(
        body=outer.body.with_(arguments=(outer.body.arguments[0], previous))
    )
    with pytest.raises(KernelError, match="argument"):
        validate_module(module.with_(computations=(*lookups.values(), forged)))

    # This mutation remains well typed because the ignored frame also carries
    # Real. It must produce the shadowing value, not the outer valuation.
    lookup = lookups[_A.id]
    selected = syntax.there[_A.id, _A.id]
    mutated = []
    for branch in lookup.body.branches:
        if branch.constructor == selected:
            (frame,) = branch.body.branches
            hypothesis = frame.fields[1]
            early = frame.with_(body=Return(value=Var(local=hypothesis)))
            branch = branch.with_(body=branch.body.with_(branches=(early,)))
        mutated.append(branch)
    wrong = lookup.with_(body=lookup.body.with_(branches=tuple(mutated)))
    wrong_module = module.with_(
        computations=(wrong, lookups[_B.id], lookups[_RESULT.id], *exports)
    )
    validate_module(wrong_module)
    assert run_named(wrong_module, "outer").value == 8.5
