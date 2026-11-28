"""Discharged bodies retain their checked nominal calling types."""

from hashlib import sha256

import didactic.api as dx
import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import QVR_SOURCE_VERSION, lower_qvr_to_qiec
from quivers.qiec import (
    BOOL,
    RANDOM_SAMPLE,
    SCORE_ADD,
    EffectRow,
    KernelError,
    LiteralValue,
    NamedComputation,
    QiecModule,
    Return,
    SiteId,
    dumps,
    inferred_computation_type,
    loads,
    run_program,
    validate_module,
)
from quivers.transpile import transpile
from quivers.transpile.qiec_ir import lower_qiec_ir

_SUBJECT = """\
effect Stop
    stop : Real -> Real

instance stop : Stop
instance random : Random
instance other_random : Random
instance score : Score

handler abort for Stop : Real -> Real [coverage=total, forwards=none, implementation=authored]
    stop(p : Real) resumes 0 =>
        return p

define subject(p : Real) : Real !{random, score} =
    handle stop with abort in
        let probability <- perform stop.stop(p)
        let atom <- perform random.sample[Bool](site("dead"), Bernoulli(probability))
        perform score.add(weight(probability * probability))
        if atom then
            return probability + 1.0
        else
            return probability + 2.0
"""
_PROGRAM = (
    _SUBJECT
    + """
object UnitInput : FinSet 1
object Output : Real 1

define caller(p : Real) : Real !{random, score} =
    let value <- subject(p)
    perform score.add(weight(3.0 * p))
    return 2.0 * value

program measured(p : Real) : UnitInput -> Output
    let result <- caller(p)
    return result

export measured
"""
)


def _owner(module, name):
    return next(item for item in module.computations if item.name == name)


@pytest.fixture(scope="module")
def checked_route():
    return Compiler(parse(_PROGRAM)).effect_plan("measured")


def test_dropped_draw_and_score_keep_the_nominal_type_and_outer_gradient(checked_route):
    source, reduced = checked_route.bundle.module, checked_route.discharge.module
    owner = _owner(source, "subject")
    discharged = _owner(reduced, "subject")
    assert discharged.type == owner.type
    assert (discharged.id, discharged.origin) == (owner.id, owner.origin)
    assert _owner(reduced, "caller") is _owner(source, "caller")
    assert (
        inferred_computation_type(reduced, validate_module(reduced), "subject").effects
        == EffectRow()
    )
    restored = loads(dumps(reduced))
    assert isinstance(restored, QiecModule) and restored == reduced
    assert dumps(restored) == dumps(reduced)

    static = str(
        SiteId.derive(
            QVR_SOURCE_VERSION,
            "source",
            ("computations", "caller", "body", "then", "first"),
            "effect-request",
        )
    )
    expected = (
        (str(SCORE_ADD), (static, (("call", "measured#1"), ("call", "caller#2")), ())),
    )
    for graph in (source, reduced, restored):
        for probability in (0.25, 0.375):
            point = torch.tensor(probability, dtype=torch.float64, requires_grad=True)
            result = run_program(graph, "measured", data={"p": point}, sites={})
            actual = tuple(
                (event.detail["operation"], event.detail["address"])
                for event in result.result.trace
                if event.event == "operation.requested"
                and event.detail["operation"] in (str(RANDOM_SAMPLE), str(SCORE_ADD))
            )
            assert actual == expected
            assert result.value.item() == 2 * probability
            assert result.log_joint.item() == 3 * probability
            assert torch.autograd.grad(result.value, point) == (point.new_tensor(2.0),)
            assert torch.autograd.grad(result.log_joint, point) == (
                point.new_tensor(3.0),
            )

    resumed = _PROGRAM.replace(
        "resumes 0 =>\n        return p", "resumes 1 =>\n        resume(p)"
    )
    mutant = Compiler(parse(resumed)).effect_plan("measured").bundle.module
    witnessed = run_program(mutant, "measured", data={"p": 0.25}, sites={"dead": True})
    assert witnessed.value.item() == 2.5
    assert any(
        event.event == "operation.requested"
        and event.detail["operation"] == str(RANDOM_SAMPLE)
        for event in witnessed.result.trace
    )


def test_a_nonempty_body_row_keeps_the_original_reached_score():
    partial = _PROGRAM.replace(
        "    handle stop with abort in\n",
        "    handle stop with abort in\n        perform score.add(weight(2.0 * p))\n",
    )
    plan = Compiler(parse(partial)).effect_plan("measured")
    source, reduced = plan.bundle.module, plan.discharge.module
    score = next(item.entry for item in source.instances if item.name == "score")
    assert _owner(reduced, "subject").type == _owner(source, "subject").type
    assert inferred_computation_type(
        reduced, validate_module(reduced), "subject"
    ).effects == EffectRow(entries=(score,))
    effects = []
    for graph in (source, reduced):
        point = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        run = run_program(graph, "measured", data={"p": point}, sites={})
        effects.append(
            tuple(
                (e.detail["operation"], e.detail["address"])
                for e in run.result.trace
                if e.event == "operation.requested"
                and e.detail["operation"] in (str(RANDOM_SAMPLE), str(SCORE_ADD))
            )
        )
        assert run.value.item() == 0.5 and run.log_joint.item() == 1.25
        assert torch.autograd.grad(run.log_joint, point) == (point.new_tensor(5.0),)
    assert effects[0] == effects[1]
    assert tuple(op for op, _ in effects[0]) == (str(SCORE_ADD), str(SCORE_ADD))


def test_a_standalone_static_target_uses_the_checked_implementation_row():
    parsed = parse(_SUBJECT)
    plan = Compiler(parsed).effect_plan("subject")
    discharged = plan.discharge.module
    projection = lower_qiec_ir(discharged)
    (owner,) = projection.computations
    assert owner.name == "subject" and owner.type.effects.entries == ()
    artifact = transpile(parsed, target="stan")
    assert b"qiec_subject" in artifact
    assert b"bernoulli" not in artifact.lower()
    receipt = plan.target_evidence("stan")
    assert receipt.artifact_digest == sha256(artifact).hexdigest()
    assert receipt.projection == "discharged"


def test_a_claimed_reduced_row_cannot_hide_an_operation_or_wrong_result(checked_route):
    source, reduced = checked_route.bundle.module, checked_route.discharge.module
    owner = _owner(reduced, "subject")
    for body in (
        _owner(source, "subject").body,
        Return(value=LiteralValue(value=True, type=BOOL)),
    ):
        forged = owner.with_(body=body)
        module = reduced.with_(
            computations=tuple(
                forged if old is owner else old for old in reduced.computations
            )
        )
        restored = loads(dumps(module))
        assert isinstance(restored, QiecModule)
        for consumer in (validate_module, lower_qiec_ir):
            with pytest.raises(KernelError, match="not declared body type"):
                consumer(restored)


def test_weakening_rejects_another_instance_of_the_same_effect(checked_route):
    reduced = checked_route.discharge.module
    owner = _owner(reduced, "subject")
    extra = next(
        entry.entry for entry in reduced.instances if entry.name == "other_random"
    )
    assert extra.effect in tuple(entry.effect for entry in owner.type.effects.entries)
    with pytest.raises(dx.ValidationError, match="strict nominal closed subrow"):
        owner.with_(body_effects=EffectRow(entries=(extra,)))


def test_an_authored_closed_row_still_has_to_match_its_body():
    source = _SUBJECT.split("    handle stop with abort in", 1)[0] + "    return p\n"
    with pytest.raises(QiecDiagnosticError, match="declared"):
        lower_qvr_to_qiec(parse(source))
    assert "body_effects" not in NamedComputation.__field_specs__
