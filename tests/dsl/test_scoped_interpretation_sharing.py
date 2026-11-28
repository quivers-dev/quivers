"""Scoped rule applications preserve ordered effects and explicit sharing."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.modeling import Model
from quivers.qiec import Bind, Case, QiecModule, RuleError, dumps, loads, run_named


_SOURCE = """\
effect Experiment
    mark : Real -> Real
    toss : Unit -> Bool

instance events : Experiment

signature Scope
    sorts
        Expr : object
    constructors
        Left : Expr -> Expr
        Right : -> Expr
        Fuse : Expr, Expr -> Expr
    binders
        Twice : binds (value : Expr) in (body : Expr) -> Expr

interpret meaning [structure=cartesian, effects=[events]] : Scope -> Real
    rule Left = left
    rule Right = right
    rule Fuse = fuse
    rule Twice = twice

define left[G : Type](value : Real) : Real !{events} =
    let mark <- perform events.mark(1.0)
    let heads <- perform events.toss()
    if heads then
        return value + mark
    else
        return mark

define right[G : Type]() : Real !{events} =
    perform events.mark(3.0)

define fuse[G : Type](first : Real, second : Real) : Real !{} =
    return first + 2.0 * second

define twice[G : Type](body : Scope__ScopedProof_Expr[Real, Scope__ScopedFrame_Expr[G, Real]] * G) : Real !{events} =
    let first <- meaning__apply_Twice_0[G](body, 2.0)
    let second <- meaning__apply_Twice_0[G](body, 2.0)
    if first == second then
        return 1.0
    else
        return 0.0

define evaluate() : Real !{events} =
    let here = construct Scope__ScopedHere_Expr[Real, Scope__ScopedEmpty]() as Scope__ScopedMember_Expr[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]]
    let x = construct Scope__ScopedHypothesis_Expr[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]](here) as Scope__ScopedProof_Expr[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]]
    let head = construct Scope__ScopedRule_Left[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]](x) as Scope__ScopedProof_Expr[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]]
    let tail = construct Scope__ScopedRule_Right[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]]() as Scope__ScopedProof_Expr[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]]
    let fused = construct Scope__ScopedRule_Fuse[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]](head, tail) as Scope__ScopedProof_Expr[Real, Scope__ScopedFrame_Expr[Scope__ScopedEmpty, Real]]
    let proof = construct Scope__ScopedRule_Twice[Real, Scope__ScopedEmpty](fused) as Scope__ScopedProof_Expr[Real, Scope__ScopedEmpty]
    meaning(proof)

handler marginalize for Experiment : Real -> Real with (p : Real) [coverage=total, forwards=none, implementation=authored]
    return result =>
        return result
    mark(tag : Real) resumes 1 =>
        resume(0.0)
    toss resumes omega =>
        let heads <- resume(true)
        let tails <- resume(false)
        return p * heads + (1.0 - p) * tails

handler audit for Experiment : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return result =>
        return result
    mark(tag : Real) resumes 1 =>
        let tail <- resume(0.0)
        return 10.0 * tail + tag
    toss resumes 1 =>
        resume(false)

define agreement(p : Real) : Real !{} =
    handle events with marginalize(p) in
        evaluate()

define log_order() : Real !{} =
    handle events with audit in
        evaluate()

object UnitInput : FinSet 1
object Output : Real 1

program scope_order : UnitInput -> Output
    let result <- log_order()
    return result
"""


def _checked(source: str) -> QiecModule:
    module = load_source(source).qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    return restored


def test_scoped_applications_share_a_suspension_or_explicitly_share_its_value() -> None:
    independent = _checked(_SOURCE)
    shared = _checked(
        _SOURCE.replace(
            "let second <- meaning__apply_Twice_0[G](body, 2.0)", "let second = first"
        )
    )
    for p in (0.5, 0.31):
        bias = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        actual = run_named(independent, "agreement", (bias,)).value
        reused = run_named(shared, "agreement", (bias,)).value
        expected = sum(
            mass
            for left, right, mass in (
                (True, True, bias**2),
                (True, False, bias * (1.0 - bias)),
                (False, True, (1.0 - bias) * bias),
                (False, False, (1.0 - bias) ** 2),
            )
            if left == right
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(reused, bias.new_tensor(1.0))
        (actual_gradient,) = torch.autograd.grad(actual, bias)
        (expected_gradient,) = torch.autograd.grad(expected, bias)
        (reused_gradient,) = torch.autograd.grad(reused, bias)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        torch.testing.assert_close(reused_gradient, bias.new_tensor(0.0))
    assert float(run_named(independent, "agreement", (0.5,)).value) == 0.5
    for module, repetitions in ((independent, 2), (shared, 1)):
        audit = run_named(module, "log_order")
        assert float(audit.value) == _order(*(1, 3) * repetitions)
        (effect,) = [
            effect for effect in module.effects if effect.ref.name == "Experiment"
        ]
        names = {str(operation.id): operation.name for operation in effect.operations}
        requests = [
            event.detail
            for event in audit.trace
            if event.event == "operation.requested"
        ]
        assert [names[request["operation"]] for request in requests] == [
            "mark",
            "toss",
            "mark",
        ] * repetitions
        addresses = [request["address"] for request in requests]
        assert len(set(addresses)) == 3 * repetitions
        assert len({request["instance"] for request in requests}) == 1


def test_nonapplication_rule_keeps_order_even_with_the_same_arguments() -> None:
    module = _checked(_SOURCE)
    (fold,) = [
        definition
        for definition in module.computations
        if definition.name == "meaning__fold_Expr"
    ]
    assert isinstance(fold.body, Case)
    (fuse,) = [
        constructor.id
        for constructor in module.constructors
        if constructor.name == "Scope__ScopedRule_Fuse"
    ]
    (branch,) = [branch for branch in fold.body.branches if branch.constructor == fuse]
    left = branch.body
    assert isinstance(left, Bind) and isinstance(left.then, Bind)
    right = left.then
    exchanged = branch.with_(body=right.with_(then=left.with_(then=right.then)))
    mutated_fold = fold.with_(
        body=fold.body.with_(
            branches=tuple(
                exchanged if candidate == branch else candidate
                for candidate in fold.body.branches
            )
        )
    )
    mutated = module.with_(
        computations=tuple(
            mutated_fold if other == fold else other for other in module.computations
        )
    )
    assert float(run_named(module, "log_order").value) == _order(1, 3, 1, 3)
    assert float(run_named(mutated, "log_order").value) == _order(3, 1, 3, 1)
    assert float(run_named(mutated, "agreement", (0.5,)).value) == 0.5

    model = Model(_SOURCE, entry="scope_order")
    (binding,) = model.bind({}).schema.scoped_interpretations
    binding.validate(module)
    assert binding.interpretation.preserves == ()
    (required,) = binding.instances
    assert required.name == "events"
    with pytest.raises(RuleError, match="assignments differ from checked fold bodies"):
        binding.validate(mutated)


def test_scope_handler_must_be_checked_for_both_resumptions() -> None:
    ungraded = _SOURCE.replace("toss resumes omega", "toss resumes 1")
    with pytest.raises(QiecDiagnosticError, match="grade"):
        load_source(ungraded)


def _order(*marks: int) -> float:
    # The audit handler interprets a mark m by postcomposing F_m(x)=10x+m.
    total = 1
    for mark in reversed(marks):
        total = 10 * total + mark
    return float(total)
