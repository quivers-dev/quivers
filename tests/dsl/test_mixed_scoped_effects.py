"""Effectful scoped folds can share one exact row over distinct judgments."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.modeling import Model
from quivers.qiec import (
    CARTESIAN,
    REAL,
    EffectRow,
    QiecModule,
    RuleError,
    ScopedInterpretationBinding,
    dumps,
    loads,
)
from quivers.qiec.checking import KernelError
from quivers.qiec.program_runtime import run_program


SOURCE = """\
object Token : FinSet 1
object Response : Real 1

effect Scale
    scale : Real -> Real

instance seed : Scale
instance view : Scale

handler scale_by for Scale : Real -> Real with (multiplier : Real) [coverage=total, forwards=none, implementation=authored]
    return result =>
        return result
    scale(value : Real) resumes 1 =>
        resume(multiplier * value)

signature MixedScope
    sorts
        Expr : object
        Atom : object
    constructors
        Merge : Atom, Expr -> Expr
        Read : Atom -> Expr
        AtomLeaf : -> Atom
        ExprLeaf : -> Expr
    binders
        Bind : binds (value : Atom) in (body : Expr) -> Expr

interpret meaning [structure=cartesian, effects=[seed, view]] : MixedScope -> Real
    rule Merge = merge
    rule Read = read_atom
    rule AtomLeaf = leaf
    rule ExprLeaf = spare
    rule Bind = introduce

define merge[G : Type](x : Real, y : Real) : Real !{} =
    return x + y

define read_atom[G : Type](value : Real) : Real !{view} =
    perform view.scale(value)

define leaf[G : Type]() : Real !{seed, view} =
    let baseline <- perform seed.scale(2.0)
    perform view.scale(baseline)

define spare[G : Type]() : Real !{seed, view} =
    let baseline <- perform seed.scale(1.0)
    perform view.scale(baseline)

define introduce[G : Type](body : MixedScope__ScopedProof_Expr[Real, Real, MixedScope__ScopedFrame_Atom[G, Real]] * G) : Real !{seed, view} =
    let value <- perform seed.scale(3.0)
    meaning__apply_Bind_0[G](body, value)

define evaluate() : Real !{seed, view} =
    let member = construct MixedScope__ScopedHere_Atom[Real, Real, MixedScope__ScopedEmpty]() as MixedScope__ScopedMember_Atom[Real, Real, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Real]]
    let variable = construct MixedScope__ScopedHypothesis_Atom[Real, Real, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Real]](member) as MixedScope__ScopedProof_Atom[Real, Real, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Real]]
    let read = construct MixedScope__ScopedRule_Read[Real, Real, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Real]](variable) as MixedScope__ScopedProof_Expr[Real, Real, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Real]]
    let bind = construct MixedScope__ScopedRule_Bind[Real, Real, MixedScope__ScopedEmpty](read) as MixedScope__ScopedProof_Expr[Real, Real, MixedScope__ScopedEmpty]
    let atom = construct MixedScope__ScopedRule_AtomLeaf[Real, Real, MixedScope__ScopedEmpty]() as MixedScope__ScopedProof_Atom[Real, Real, MixedScope__ScopedEmpty]
    let syntax = construct MixedScope__ScopedRule_Merge[Real, Real, MixedScope__ScopedEmpty](atom, bind) as MixedScope__ScopedProof_Expr[Real, Real, MixedScope__ScopedEmpty]
    meaning(syntax)

define execute(theta : Real, beta : Real) : Real !{} =
    handle seed with scale_by(theta) in
        handle view with scale_by(beta) in
            evaluate()

program interpreted(theta : Real, beta : Real) : Token -> Response
    let estimate <- execute(theta, beta)
    observe y <- Normal(loc=estimate, scale=1.0)
    return estimate
"""


def _module(source: str) -> QiecModule:
    compiled = load_source(source)
    bytes_ = dumps(compiled.qiec)
    restored = loads(bytes_)
    assert isinstance(restored, QiecModule) and restored == compiled.qiec
    assert dumps(restored) == bytes_
    return restored


def _normal_log_likelihood(value: torch.Tensor, observation: float) -> torch.Tensor:
    return -0.5 * ((observation - value) ** 2 + math.log(2 * math.pi))


@pytest.mark.parametrize("reassign_view", [False, True])
def test_mixed_sort_folds_preserve_bound_scope_and_nominal_requests(
    reassign_view,
) -> None:
    src = SOURCE
    if reassign_view:
        src = src.replace(
            "define read_atom[G : Type](value : Real) : Real !{view} =\n"
            "    perform view.scale(value)",
            "define read_atom[G : Type](value : Real) : Real !{seed} =\n"
            "    perform seed.scale(value)",
        )
    module = _module(src)
    theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    observation = 0.7
    run = run_program(
        module,
        "interpreted",
        data={"theta": theta, "beta": beta, "y": observation},
        sites={},
        distributions=TensorBackend(),
    )
    # The left premise contributes its calibrated literal 2. The second
    # premise introduces a fresh Atom carrying the distinct literal 3.
    expected = 2 * theta * beta + 3 * theta * (theta if reassign_view else beta)
    assert not torch.isclose(expected, 3 * theta * beta)
    torch.testing.assert_close(run.value, expected)
    likelihood = _normal_log_likelihood(expected, observation)
    torch.testing.assert_close(run.log_joint, likelihood)
    torch.testing.assert_close(
        torch.autograd.grad(run.log_joint, (theta, beta)),
        torch.autograd.grad(likelihood, (theta, beta)),
    )

    scale = next(effect for effect in module.effects if effect.ref.name == "Scale")
    (operation,) = scale.operations
    requests = [
        event.detail
        for event in run.result.trace
        if event.event == "operation.requested"
        and event.detail["operation"] == str(operation.id)
    ]
    instances = {
        str(instance.entry.instance): instance.name
        for instance in module.instances
        if instance.entry.effect.id == scale.ref.id
    }
    assert [instances[event["instance"]] for event in requests] == [
        "seed",
        "view",
        "seed",
        "seed" if reassign_view else "view",
    ]
    assert len({event["address"] for event in requests}) == 4
    # Lookup of an Atom in the retained right extension is pure. The
    # requests are the two rule actions and the scoped application's body.
    assert [
        next(
            frame for kind, frame in reversed(event["address"][1]) if kind == "call"
        ).split("#", 1)[0]
        for event in requests
    ] == ["leaf", "leaf", "introduce", "read_atom"]


def test_mixed_companion_rechecks_identical_type_but_different_rule_assignment() -> (
    None
):
    checked = Model(SOURCE, entry="interpreted")
    (binding,) = checked.schema.scoped_interpretations
    assert binding.presentation.structure == CARTESIAN
    identities = {item.name: item.id for item in binding.presentation.judgments}
    assert identities["Expr"] == binding.judgment != identities["Atom"]
    assert dict(binding.carriers) == dict.fromkeys(identities.values(), REAL)
    actual_row = EffectRow(entries=tuple(item.entry for item in binding.instances))
    assert {instance.name for instance in binding.instances} == {"seed", "view"}
    assert dict(binding.rows) == dict.fromkeys(identities.values(), actual_row)

    module = _module(SOURCE)
    restored = loads(dumps(binding))
    assert isinstance(restored, ScopedInterpretationBinding) and restored == binding
    restored.validate(module)
    assignments = dict(restored.interpretation.operations)
    local = {rule.name: rule.id for rule in restored.presentation.rules}
    by_computation = {
        computation.id: computation.name for computation in module.computations
    }
    assert by_computation[assignments[local["AtomLeaf"]]] == "leaf"
    assert by_computation[assignments[local["ExprLeaf"]]] == "spare"
    operations = {item.id: item for item in module.computations}
    leaf, spare = assignments[local["AtomLeaf"]], assignments[local["ExprLeaf"]]
    assert operations[leaf].type == operations[spare].type
    assignments[local["AtomLeaf"]] = spare
    forged = restored.with_(
        interpretation=restored.interpretation.with_(
            operations=tuple(assignments.items())
        )
    )
    with pytest.raises(RuleError, match="assignments differ from checked fold bodies"):
        forged.validate(module)


def test_common_row_does_not_erase_missing_instance_or_context_sort() -> None:
    with pytest.raises(
        QiecDiagnosticError, match="outside the interpretation's closed effects row"
    ):
        load_source(SOURCE.replace("effects=[seed, view]", "effects=[seed]"))
    with pytest.raises(QiecDiagnosticError, match="ScopedMember_Atom"):
        load_source(SOURCE.replace("ScopedHere_Atom", "ScopedHere_Expr"))


def test_shared_row_still_checks_each_generated_fold_for_exact_effect_type() -> None:
    source = SOURCE.replace(
        "define leaf[G : Type]() : Real !{seed, view} =\n"
        "    let baseline <- perform seed.scale(2.0)\n"
        "    perform view.scale(baseline)",
        "define leaf[G : Type]() : Real !{seed} =\n    perform seed.scale(2.0)",
    )
    with pytest.raises(KernelError, match=r"meaning__fold_Atom.*not declared type"):
        load_source(source)
