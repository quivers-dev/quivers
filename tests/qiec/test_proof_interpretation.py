"""Closed proof folds execute checked rule operations in premise order."""

from __future__ import annotations

import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.dsl import loads as load_source
from quivers.qiec._proof_interpretation import lower_closed_interpretation
from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import InterpretationId, OccurrenceId, PresentationId
from quivers.qiec.declarations import FieldDef
from quivers.qiec.effects import ComputationType, EffectRow
from quivers.qiec.execution import run_named
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    FamilyId,
    SourceOrigin,
)
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    Derivation,
    HostInterpretation,
    InterpretationDecl,
    Judgment,
    JudgmentDecl,
    Item,
    Premise,
    Rule,
    RulePresentation,
)
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import Call, ConstructorValue, Handle, Local, Var
from quivers.qiec.types import BOOL, REAL, IndexLiteral, IndexVariable, TypeApplication
from quivers.stochastic.forest import (
    build_forest,
    edge_weight,
    fold_derivations,
    forest_folds,
)


OPERATIONS = """\
effect Inspection
    visit : Real -> Real

instance audit : Inspection

handler shift for Inspection : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    visit(value : Real) resumes 1 =>
        resume(value + 1.0)

define leaf(x : Real) : Real !{audit} =
    let shifted <- perform audit.visit(x)
    return shifted

define pair(left : Real, right : Real) : Real !{} =
    return 2.0 * left - right

define truth(x : Real) : Bool !{} =
    return x > 5.0

define conjunction(left : Bool, right : Bool) : Bool !{} =
    if left then
        return right
    else
        return false
"""


def _setup() -> tuple[QiecModule, RulePresentation, InterpretationDecl, FamilyId]:
    presentation_id = PresentationId.derive("proof-fold", "presentation")
    family_id = FamilyId.derive(presentation_id, "Proof")
    atom_id = FamilyId.derive(presentation_id, "Atom")
    i, j = (IndexVariable(name=name, sort=NAT) for name in ("i", "j"))
    left = Judgment(family=atom_id, arguments=(i,))
    right = Judgment(family=atom_id, arguments=(j,))
    leaf_id = ConstructorId.derive(presentation_id, "leaf")
    pair_id = ConstructorId.derive(presentation_id, "pair")
    presentation = RulePresentation(
        id=presentation_id,
        name="example_proof_fold",
        judgments=(
            JudgmentDecl(
                id=family_id,
                name="Proof",
                indices=(IndexBinder(name="position", sort=NAT, refinable=True),),
            ),
            JudgmentDecl(
                id=atom_id,
                name="Atom",
                indices=(IndexBinder(name="position", sort=NAT, refinable=True),),
            ),
        ),
        axioms=(
            AxiomSlot(
                id=leaf_id,
                name="assume",
                telescope=(IndexBinder(name=i.name, sort=NAT),),
                judgment=left,
                values=(FieldDef(name="value", type=REAL),),
            ),
        ),
        rules=(
            Rule(
                id=pair_id,
                name="combine",
                telescope=(
                    IndexBinder(name=i.name, sort=NAT),
                    IndexBinder(name=j.name, sort=NAT),
                ),
                premises=(Premise(judgment=left), Premise(judgment=right)),
                conclusion=Judgment(family=family_id, arguments=(i,)),
            ),
        ),
    )
    module = load_source(OPERATIONS).qiec
    assert isinstance(module, QiecModule)
    families, constructors = presentation.declarations()
    module = module.with_(
        families=(*module.families, *families),
        constructors=(*module.constructors, *constructors),
    )
    named = {computation.name: computation.id for computation in module.computations}
    interpretation = InterpretationDecl(
        id=InterpretationId.derive(presentation_id, "weighted_difference"),
        presentation=presentation_id,
        operations=((leaf_id, named["leaf"]), (pair_id, named["pair"])),
    )
    return module, presentation, interpretation, family_id


def _driver(
    module: QiecModule,
    presentation: RulePresentation,
    *,
    repeated: bool = False,
    name: str = "driver",
    fold_name: str = "interpret__Proof",
) -> NamedComputation:
    """Build a checked proof from two real inputs and handle its interpretation."""
    fold = next(
        computation
        for computation in module.computations
        if computation.name == fold_name
    )
    (family, atom), _ = presentation.declarations()
    (slot,) = presentation.axioms
    (rule,) = presentation.rules
    origin = SourceOrigin(
        module=module.module,
        source_protocol=module.source_protocol,
        structural_path=("proof_fold_test", name),
        role="computation",
    )
    left, right = (Local(name=name, type=REAL) for name in ("left", "right"))
    i = IndexLiteral(value=3, sort=NAT)
    j = i if repeated else IndexVariable(name="j", sort=NAT)
    left_type = TypeApplication(constructor=atom.type_constructor, arguments=(i,))
    left_proof = ConstructorValue(
        constructor=slot.id,
        static_arguments=(i,),
        fields=(Var(local=left),),
        result_type=left_type,
    )
    right_proof = (
        left_proof
        if repeated
        else ConstructorValue(
            constructor=slot.id,
            static_arguments=(j,),
            fields=(Var(local=right),),
            result_type=TypeApplication(
                constructor=atom.type_constructor, arguments=(j,)
            ),
        )
    )
    proof = ConstructorValue(
        constructor=rule.id,
        static_arguments=(i, j),
        fields=(left_proof, right_proof),
        result_type=TypeApplication(
            constructor=family.type_constructor, arguments=(i,)
        ),
    )
    instance = next(
        instance for instance in module.instances if instance.name == "audit"
    )
    handler = next(handler for handler in module.handlers if handler.name == "shift")
    call = Call(
        callee=fold.id,
        name=fold.name,
        static_arguments=(i,),
        arguments=(proof,),
        effects=fold.type.effects,
        result_type=fold.type.result,
        origin=origin.with_(
            structural_path=(*origin.structural_path, "fold"), role="call"
        ),
    )
    body = (
        Handle(
            instance=instance.entry.instance,
            handler=handler.id,
            static_arguments=(),
            computation=call,
        )
        if fold.type.effects.entries
        else call
    )
    return NamedComputation(
        id=ComputationId.derive(module.module, "computation", name),
        name=name,
        telescope=() if repeated else (IndexBinder(name="j", sort=NAT),),
        parameters=(left, right),
        type=ComputationType(effects=EffectRow(), result=fold.type.result),
        body=body,
        origin=origin,
    )


@pytest.mark.parametrize("repeated", [False, True])
def test_indexed_proof_fold_preserves_premise_occurrences_and_grads(
    repeated: bool,
) -> None:
    module, presentation, interpretation, _ = _setup()
    module = lower_closed_interpretation(
        module,
        presentation,
        interpretation,
        name="interpret",
        carriers={judgment.id: REAL for judgment in presentation.judgments},
    )
    driver = _driver(module, presentation, repeated=repeated)
    module = module.with_(computations=(*module.computations, driver))
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert restored == module

    arguments = torch.tensor([3.0, 11.0], dtype=torch.float64, requires_grad=True)
    found = run_named(
        restored,
        "driver",
        tuple(arguments.unbind()),
        static_arguments=() if repeated else (IndexLiteral(value=5, sort=NAT),),
    )
    assert isinstance(found.value, torch.Tensor)
    expected = 2 * (arguments[0] + 1) - (arguments[0 if repeated else 1] + 1)
    torch.testing.assert_close(found.value, expected)
    torch.testing.assert_close(
        torch.autograd.grad(found.value, arguments)[0],
        torch.autograd.grad(expected, arguments)[0],
    )
    order = [
        event.detail["computation"]
        for event in found.trace
        if event.event == "call.entered"
        and event.detail["computation"] in ("leaf", "pair")
    ]
    assert order == ["leaf", "leaf", "pair"]
    requests = [
        event.detail for event in found.trace if event.event == "operation.requested"
    ]
    assert len(requests) == 2
    assert requests[0]["instance"] == requests[1]["instance"]
    assert requests[0]["address"] != requests[1]["address"]

    assert isinstance(driver.body, Handle)
    unhandled = driver.with_(body=driver.body.computation)
    with pytest.raises(KernelError, match="not declared type"):
        validate_module(
            module.with_(computations=(*module.computations[:-1], unhandled))
        )


def test_rule_operations_must_match_their_constructor_arity() -> None:
    module, presentation, interpretation, _ = _setup()
    first, second = interpretation.operations
    wrong = interpretation.with_(operations=(first, (second[0], first[1])))
    with pytest.raises(
        KernelError, match="does not match the closed interpretation of 'combine'"
    ):
        lower_closed_interpretation(
            module,
            presentation,
            wrong,
            name="interpret",
            carriers={judgment.id: REAL for judgment in presentation.judgments},
        )


def test_proof_fold_refuses_a_premise_under_hypotheses() -> None:
    module, presentation, interpretation, _ = _setup()
    (rule,) = presentation.rules
    first, second = rule.premises
    scoped = rule.with_(premises=(first.with_(before=(first.judgment,)), second))
    presentation = presentation.with_(rules=(scoped,))
    with pytest.raises(KernelError, match="premise under hypotheses"):
        lower_closed_interpretation(
            module,
            presentation,
            interpretation,
            name="interpret",
            carriers={judgment.id: REAL for judgment in presentation.judgments},
        )


def test_one_presentation_drives_weights_meanings_and_proof_encodings() -> None:
    module, presentation, interpretation, goal_family = _setup()
    assert loads(dumps(interpretation)) == interpretation
    module = lower_closed_interpretation(
        module,
        presentation,
        interpretation,
        name="interpret",
        carriers={judgment.id: REAL for judgment in presentation.judgments},
    )
    truth = interpretation.with_(
        id=InterpretationId.derive(presentation.id, "truth"),
        operations=tuple(
            (rule, next(item.id for item in module.computations if item.name == name))
            for (rule, _), name in zip(
                interpretation.operations, ("truth", "conjunction"), strict=True
            )
        ),
    )
    module = lower_closed_interpretation(
        module,
        presentation,
        truth,
        name="truth",
        carriers={judgment.id: BOOL for judgment in presentation.judgments},
    )
    drivers = (
        _driver(module, presentation),
        _driver(module, presentation, name="truth_driver", fold_name="truth__Proof"),
    )
    module = module.with_(computations=(*module.computations, *drivers))

    (slot,) = presentation.axioms
    positions, values = (3, 3, 5), (3.0, 8.0, 11.0)
    occurrences = tuple(
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive(slot.id, index),
            binding=(("i", position),),
            values=(value,),
        )
        for index, (position, value) in enumerate(zip(positions, values, strict=True))
    )
    logs = torch.tensor([-0.4, -0.9, -0.3], dtype=torch.float64, requires_grad=True)
    weights = {
        row.occurrence: log for row, log in zip(occurrences, logs.unbind(), strict=True)
    }
    forest = build_forest(presentation, occurrences)
    goal = Item(family=goal_family, arguments=(3,))
    inside = forest_folds(
        forest,
        (goal,),
        LOG_PROB,
        lambda edge: edge_weight(forest, edge, LOG_PROB, weights, like=logs),
        like=logs,
    )[goal]
    trees = fold_derivations(
        forest,
        goal,
        LOG_PROB,
        weights,
        HostInterpretation(
            presentation=presentation,
            operations={
                identity: lambda node, children: None
                for identity in presentation.constructors_by_id
            },
        ),
        like=logs,
        limit=6,
    )
    assert len(trees) == 6
    masses = []
    encodings = []
    meanings = []
    for tree, weight, _ in trees:
        left, right = tree.premises
        assert isinstance(left, Derivation) and isinstance(right, Derivation)
        index = right.item.arguments[0]
        assert isinstance(index, int)
        static = (IndexLiteral(value=index, sort=NAT),)
        inputs = (*left.values, *right.values)
        encoded = run_named(module, "driver", inputs, static_arguments=static).value
        meaning = run_named(
            module, "truth_driver", inputs, static_arguments=static
        ).value
        assert isinstance(encoded, torch.Tensor) and isinstance(meaning, torch.Tensor)
        masses.append((weight - inside).exp())
        encodings.append(encoded)
        meanings.append(meaning.to(dtype=logs.dtype))
    posterior = torch.stack(masses)
    observed = torch.stack(
        (posterior @ torch.stack(encodings), posterior @ torch.stack(meanings))
    )

    # Six ordered pairs, including a repeated occurrence on the diagonal.
    left_rows, right_rows = [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]
    proof_logs = logs[left_rows] + logs[right_rows]
    data = torch.tensor(values, dtype=logs.dtype)
    wanted = torch.stack(
        (
            2 * data[left_rows] - data[right_rows] + 1,
            ((data[left_rows] > 5) & (data[right_rows] > 5)).to(dtype=logs.dtype),
        )
    ) @ proof_logs.softmax(0)
    torch.testing.assert_close(inside, proof_logs.logsumexp(0))
    torch.testing.assert_close(observed, wanted)
    actual_grad = torch.stack(
        [
            torch.autograd.grad(value, logs, retain_graph=True)[0]
            for value in observed.unbind()
        ]
    )
    wanted_grad = torch.stack(
        [
            torch.autograd.grad(value, logs, retain_graph=True)[0]
            for value in wanted.unbind()
        ]
    )
    torch.testing.assert_close(actual_grad, wanted_grad)
