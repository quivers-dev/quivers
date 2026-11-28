"""Dependent scope application is checked, serializable, and effectful."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import loads as loads_qvr
from quivers.qiec import ScopedProofFold, scoped_proof_fold, scoped_proof_types
from quivers.qiec.builtins import LOG_WEIGHT, SCORE_ADD
from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import OccurrenceId, PresentationId
from quivers.qiec.declarations import FieldDef
from quivers.qiec.effects import ComputationType, EffectRow
from quivers.qiec.execution import RuntimeConfiguration, RuntimeSelection, run_named
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    FamilyId,
    SourceOrigin,
)
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.primitives import primitive
from quivers.qiec.rules import (
    AxiomSlot,
    CARTESIAN,
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
    presentation_of,
    substitute_hypotheses,
    term_derivation,
)
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import (
    Bind,
    Call,
    ConstructorValue,
    Handle,
    LiteralValue,
    Local,
    PrimitiveApplication,
    Return,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    REAL,
    TypeApplication,
    TypeVariable,
    product_type,
)

from quivers.structural import Term


_ID = PresentationId.derive("scoped-fold-experiment")
_NUMBER = JudgmentDecl(id=FamilyId.derive(_ID, "Number"), name="Number")
_MEANING = JudgmentDecl(id=FamilyId.derive(_ID, "Meaning"), name="Meaning")
_NUMBER_ITEM = Item(family=_NUMBER.id)
_MEANING_ITEM = Item(family=_MEANING.id)
_A = Judgment(family=_NUMBER.id)
_B = Judgment(family=_MEANING.id)
_SUM = Rule(
    id=ConstructorId.derive(_ID, "sum"),
    name="sum",
    conclusion=_B,
    premises=(Premise(judgment=_A),) * 4,
)
_OUTER = Rule(
    id=ConstructorId.derive(_ID, "outer"),
    name="outer",
    conclusion=_B,
    premises=(Premise(judgment=_B, after=(_A,)),),
)
_INNER = Rule(
    id=ConstructorId.derive(_ID, "inner"),
    name="inner",
    conclusion=_B,
    premises=(Premise(judgment=_B, after=(_A,)),),
)
_BOTH = Rule(
    id=ConstructorId.derive(_ID, "both"),
    name="both",
    conclusion=_B,
    premises=(Premise(judgment=_B, after=(_A, _A)),),
)
_IGNORE = Rule(
    id=ConstructorId.derive(_ID, "ignore"),
    name="ignore",
    conclusion=_B,
    premises=(Premise(judgment=_B, after=(_A, _A)),),
)
_LEAF = AxiomSlot(
    id=ConstructorId.derive(_ID, "leaf"),
    name="leaf",
    judgment=_A,
    values=(FieldDef(name="value", type=REAL),),
)
_PRESENTATION = RulePresentation(
    id=_ID,
    name="ScopedFoldExperiment",
    judgments=(_NUMBER, _MEANING),
    rules=(_SUM, _OUTER, _INNER, _BOTH, _IGNORE),
    axioms=(_LEAF,),
    structure=CARTESIAN,
)

_LEAF_SOURCE = """\
instance score : Score

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define weighted_leaf(occurrence : String, value : Real) : Real !{score} =
    perform score.add(log_prob(Normal(0.0, 1.0), value))
    return value
"""
_RUNTIME = RuntimeConfiguration(
    selections=(
        RuntimeSelection(
            provider="core",
            options={"handlers": {"accumulate": {"kind": "score"}}},
        ),
    ),
)


def _origin(name: str) -> SourceOrigin:
    return SourceOrigin(
        module=_PRESENTATION.name,
        source_protocol="quivers.rules",
        structural_path=(name,),
        role="computation",
    )


def _literal(value: float) -> LiteralValue:
    return LiteralValue(value=value, type=REAL)


def _binary(symbol: str, left: Value, right: Value) -> PrimitiveApplication:
    operation = primitive({"+": "add_real", "*": "mul_real"}[symbol])
    return PrimitiveApplication(
        name=operation.name,
        primitive=operation.id,
        arguments=(left, right),
        result_type=REAL,
        origin=_origin(operation.name),
    )


def _operations(
    fold: ScopedProofFold,
    weighted_leaf: NamedComputation,
    score: EffectRow,
) -> dict[ConstructorId, NamedComputation]:
    ambient = TypeVariable(name="semantic_context")
    telescope = (TypeBinder(name=ambient.name),)
    applications = fold.applications()
    made = {_LEAF.id: weighted_leaf.with_(telescope=telescope)}
    for rule in _PRESENTATION.rules:
        domain = tuple(
            Local(name=f"argument{n}", type=type_)
            for n, type_ in enumerate(fold.operation_parameters(rule.id, ambient))
        )
        origin = _origin(rule.name)
        row = EffectRow()
        if rule == _SUM:
            old, outer, inner, payload = tuple(Var(local=local) for local in domain)
            result = _binary(
                "+",
                old,
                _binary(
                    "+",
                    _binary("*", _literal(10.0), outer),
                    _binary("*", _literal(100.0), inner),
                ),
            )
            body = Return(value=_binary("+", result, payload))
        elif rule == _IGNORE:
            body = Return(value=_literal(99.0))
        else:
            row = score
            target = applications[rule.id, 0]

            def apply(label: str, *values: float) -> Call:
                return Call(
                    callee=target.id,
                    name=target.name,
                    static_arguments=(ambient,),
                    arguments=(
                        Var(local=domain[0]),
                        *(_literal(value) for value in values),
                    ),
                    result_type=REAL,
                    effects=score,
                    origin=origin.with_(structural_path=(rule.name, label)),
                )

            if rule == _BOTH:
                body = apply("both", 2.0, 7.0)
            else:
                choices = (2.0, 5.0) if rule == _OUTER else (3.0, 7.0)
                left = Local(name="left", type=REAL)
                right = Local(name="right", type=REAL)
                body = Bind(
                    binder=left,
                    first=apply("first", choices[0]),
                    then=Bind(
                        binder=right,
                        first=apply("second", choices[1]),
                        then=Return(
                            value=_binary(
                                "+",
                                Var(local=left),
                                _binary("*", _literal(2.0), Var(local=right)),
                            )
                        ),
                    ),
                )
        made[rule.id] = NamedComputation(
            id=ComputationId.derive(_ID, rule.id),
            name=f"meaning_{rule.name}",
            telescope=telescope,
            parameters=domain,
            type=ComputationType(effects=row, result=REAL),
            body=body,
            origin=origin,
        )
    return made


def _proof() -> dict[str, OpenDerivation]:
    occurrence = Derivation(
        constructor=_LEAF.id,
        item=_NUMBER_ITEM,
        values=(1.25,),
        occurrence=OccurrenceId.derive(_ID, "first"),
    )
    arithmetic = Derivation(
        constructor=_SUM.id,
        item=_MEANING_ITEM,
        premises=(
            Hole(index=2, item=_NUMBER_ITEM),
            Hole(index=1, item=_NUMBER_ITEM),
            Hole(index=0, item=_NUMBER_ITEM),
            occurrence,
        ),
    )

    def apply(rule: Rule, child: Derivation) -> Derivation:
        return Derivation(constructor=rule.id, item=_MEANING_ITEM, premises=(child,))

    return {
        "nested": OpenDerivation(
            context=(_NUMBER_ITEM,), body=apply(_OUTER, apply(_INNER, arithmetic))
        ),
        "both": OpenDerivation(context=(_NUMBER_ITEM,), body=apply(_BOTH, arithmetic)),
        "ignore": OpenDerivation(
            context=(_NUMBER_ITEM,), body=apply(_IGNORE, arithmetic)
        ),
    }


def test_native_scoped_fold_keeps_lexical_values_multiplicity_and_effects() -> None:
    syntax = scoped_proof_types(_PRESENTATION)
    carriers = {_NUMBER.id: REAL, _MEANING.id: REAL}
    authored = loads_qvr(_LEAF_SOURCE).qiec
    assert isinstance(authored, QiecModule)
    (weighted_leaf,) = authored.computations
    leaf_body = weighted_leaf.body
    request = leaf_body.first.request
    site = request.origin.with_(origin=_origin("weighted_leaf_score"))
    rewritten_request = leaf_body.first.with_(request=request.with_(origin=site))
    weighted_leaf = weighted_leaf.with_(
        origin=_origin("weighted_leaf"), body=leaf_body.with_(first=rewritten_request)
    )
    (entry,) = authored.instances
    entry = entry.with_(origin=_origin("score"))
    (handler,) = authored.handlers
    score = weighted_leaf.type.effects
    rows = {family: score for family in carriers}
    plan = scoped_proof_fold(syntax, carriers, rows, name="evaluate")
    operations = _operations(plan, weighted_leaf, score)
    base = syntax.module.with_(
        effects=authored.effects,
        instances=(entry,),
        handlers=authored.handlers,
    )
    module, folds = plan.assemble(operations, module=base)
    fold = folds[_MEANING.id]
    environment = ConstructorValue(
        constructor=syntax.empty.constructors[0],
        static_arguments=(),
        fields=(),
        result_type=TypeApplication(constructor=syntax.empty.type_constructor),
    )
    root = Local(name="root", type=REAL)
    frame = syntax.frames[_NUMBER.id]
    static_arguments = (environment.result_type, REAL)
    environment = ConstructorValue(
        constructor=frame.constructors[0],
        static_arguments=static_arguments,
        fields=(environment, Var(local=root)),
        result_type=TypeApplication(
            constructor=frame.type_constructor, arguments=static_arguments
        ),
    )
    exports = []
    proof_examples = _proof()
    widened = (_NUMBER_ITEM, _NUMBER_ITEM)
    proof_examples["substituted"] = substitute_hypotheses(
        _PRESENTATION,
        proof_examples["nested"],
        (OpenDerivation(context=widened, body=Hole(index=1, item=_NUMBER_ITEM)),),
        widened,
    )
    for name, open_proof in proof_examples.items():
        valuation = environment
        if name == "substituted":
            extended = (environment.result_type, REAL)
            valuation = ConstructorValue(
                constructor=frame.constructors[0],
                static_arguments=extended,
                fields=(environment, _literal(-999.0)),
                result_type=TypeApplication(
                    constructor=frame.type_constructor, arguments=extended
                ),
            )
        proof = syntax.value(open_proof, carriers)
        origin = _origin(name)
        scoped_call = Call(
            callee=fold.id,
            name=fold.name,
            static_arguments=(valuation.result_type,),
            arguments=(proof, valuation),
            result_type=REAL,
            effects=score,
            origin=origin,
        )
        exports.append(
            NamedComputation(
                id=ComputationId.derive(_ID, name),
                name=name,
                telescope=(),
                parameters=(root,),
                type=ComputationType(
                    effects=EffectRow(), result=product_type(REAL, LOG_WEIGHT)
                ),
                origin=origin,
                body=Handle(
                    instance=entry.entry.instance,
                    handler=handler.id,
                    computation=scoped_call,
                ),
            )
        )
    closed = module.with_(computations=(*module.computations, *exports))
    validate_module(closed)
    restored = loads(dumps(closed))
    assert isinstance(restored, QiecModule)

    anchor = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    actual = run_named(restored, "nested", (anchor,), runtime=_RUNTIME)
    value, joint = actual.value
    expected = sum(
        outerm * innerm * (anchor.detach() + 10.0 * outer + 100.0 * inner + 1.25)
        for outer, outerm in ((2.0, 1.0), (5.0, 2.0))
        for inner, innerm in ((3.0, 1.0), (7.0, 2.0))
    )
    assert torch.equal(value, expected)
    assert torch.equal(
        torch.autograd.grad(value, anchor)[0], torch.tensor(9.0, dtype=anchor.dtype)
    )
    leaf_density = -0.5 * (1.25**2 + math.log(2 * math.pi))
    assert float(joint) == pytest.approx(4 * leaf_density)
    addresses = [
        operation.detail["address"]
        for operation in actual.trace
        if operation.event == "operation.requested"
        and operation.detail["operation"] == str(SCORE_ADD)
    ]
    assert len(addresses) == 4
    assert len(set(addresses)) == 4
    assert len({address[0] for address in addresses}) == 1
    rerun = run_named(restored, "nested", (anchor,), runtime=_RUNTIME)
    assert [
        operation.detail["address"]
        for operation in rerun.trace
        if operation.event == "operation.requested"
    ] == addresses

    substituted = run_named(restored, "substituted", (anchor,), runtime=_RUNTIME)
    substituted_value, substituted_score = substituted.value
    assert torch.equal(substituted_value, expected)
    assert float(substituted_score) == pytest.approx(4 * leaf_density)
    assert torch.equal(
        torch.autograd.grad(substituted_value, anchor)[0],
        torch.tensor(9.0, dtype=anchor.dtype),
    )

    both = run_named(restored, "both", (anchor,), runtime=_RUNTIME)
    both_value, both_score = both.value
    assert float(both_value.detach()) == pytest.approx(0.75 + 20.0 + 700.0 + 1.25)
    assert float(both_score) == pytest.approx(leaf_density)
    ignored = run_named(restored, "ignore", (anchor,), runtime=_RUNTIME)
    assert tuple(map(float, ignored.value)) == (99.0, 0.0)
    assert all(operation.event != "operation.requested" for operation in ignored.trace)

    bad_rows = scoped_proof_fold(
        syntax, carriers, {**rows, _NUMBER.id: EffectRow()}, name="evaluate"
    )
    with pytest.raises(KernelError, match="declared type"):
        bad_rows.assemble(operations, module=base)

    both_action = operations[_BOTH.id]
    argument, first, _ = both_action.body.arguments
    wrong_hypothesis = LiteralValue(value=False, type=BOOL)
    ill_typed = both_action.with_(
        body=both_action.body.with_(arguments=(argument, first, wrong_hypothesis))
    )
    with pytest.raises(KernelError, match="argument"):
        plan.assemble({**operations, _BOTH.id: ill_typed}, module=base)

    outer = operations[_OUTER.id]
    # Resetting the second call to the first bound value remains well typed
    # and preserves effect count. The valuation-sensitive oracle must detect it.
    second_call = outer.body.then.first
    mutated = second_call.with_(arguments=outer.body.first.arguments)
    wrong_outer = outer.with_(
        body=outer.body.with_(then=outer.body.then.with_(first=mutated))
    )
    wrong, _ = plan.assemble({**operations, _OUTER.id: wrong_outer}, module=base)
    wrong = wrong.with_(computations=(*wrong.computations, *exports))
    wrong_value, wrong_joint = run_named(
        wrong, "nested", (anchor,), runtime=_RUNTIME
    ).value
    assert not torch.equal(wrong_value, expected)
    assert torch.equal(wrong_joint, joint)


def test_authored_structural_signature_supplies_the_same_scoped_proof() -> None:
    authored = loads_qvr(
        """\
signature Binders
    sorts
        Expr : object
    constructors
        Pair : Expr, Expr -> Expr
    binders
        Outer : binds (outer : Expr) in (body : Expr) -> Expr
        Inner : binds (inner : Expr) in (body : Expr) -> Expr
"""
    )
    signature = authored.signatures["Binders"]
    ordered = presentation_of(signature)
    with pytest.raises(RuleError, match="Cartesian structural permissions"):
        scoped_proof_types(ordered)
    presentation = ordered.with_(structure=CARTESIAN)
    syntax = scoped_proof_types(presentation)
    (judgment,) = presentation.judgments
    carriers = {judgment.id: REAL}
    row = EffectRow()
    plan = scoped_proof_fold(syntax, carriers, {judgment.id: row}, name="interpret")
    applications = plan.applications()

    def variable(index: int) -> Term:
        return Term(op="BoundVar", args=(index,))

    core = Term(op="Pair", args=(variable(1), variable(0)))
    term = Term(op="Outer", args=(Term(op="Inner", args=(core,)),))
    conventional = term_derivation(signature, term)
    proof = syntax.value(conventional, carriers)

    ambient = TypeVariable(name="semantic_context")
    ops = {}
    for rule in presentation.rules:
        origin = SourceOrigin(
            module=syntax.module.module,
            source_protocol=syntax.module.source_protocol,
            structural_path=(rule.name,),
            role="computation",
        )
        params = tuple(
            Local(name=f"arg{i}", type=type_)
            for i, type_ in enumerate(plan.operation_parameters(rule.id, ambient))
        )
        arguments = tuple(Var(local=local) for local in params)
        if rule.name == "Pair":
            first, second = arguments
            body = Return(
                value=_binary("+", _binary("*", _literal(10.0), first), second)
            )
        else:
            target = applications[rule.id, 0]
            introduced = 2.0 if rule.name == "Outer" else 7.0
            body = Call(
                callee=target.id,
                name=target.name,
                static_arguments=(ambient,),
                arguments=(*arguments, _literal(introduced)),
                result_type=REAL,
                effects=row,
                origin=origin,
            )
        ops[rule.id] = NamedComputation(
            id=ComputationId.derive(
                syntax.module.module, rule.name, "semantic-operation"
            ),
            name=f"semantic_{rule.name}",
            telescope=(TypeBinder(name=ambient.name),),
            parameters=params,
            body=body,
            origin=origin,
            type=ComputationType(effects=row, result=REAL),
        )
    module, folds = plan.assemble(ops)
    (fold,) = folds.values()
    empty_type = TypeApplication(constructor=syntax.empty.type_constructor)
    environment = ConstructorValue(
        constructor=syntax.empty.constructors[0],
        static_arguments=(),
        fields=(),
        result_type=empty_type,
    )
    name = "meaning"
    origin = fold.origin.with_(structural_path=(name,))
    call = Call(
        callee=fold.id,
        name=fold.name,
        static_arguments=(empty_type,),
        arguments=(proof, environment),
        result_type=REAL,
        effects=row,
        origin=origin,
    )
    run = NamedComputation(
        id=ComputationId.derive(syntax.module.module, name),
        name=name,
        telescope=(),
        parameters=(),
        body=call,
        origin=origin,
        type=ComputationType(effects=row, result=REAL),
    )
    restored = loads(dumps(module.with_(computations=(*module.computations, run))))
    assert isinstance(restored, QiecModule)
    assert float(run_named(restored, name).value) == 27.0
    wrong = term.with_(
        args=(Term(op="Inner", args=(core.with_(args=(variable(2), variable(0))),)),)
    )
    with pytest.raises(RuleError, match="out of scope"):
        term_derivation(signature, wrong)
