"""Reported effect plans preserve independent law evidence and ordinary dispatch."""

from dataclasses import replace
from fractions import Fraction

import pytest

from quivers.core.laws import Assumed
from quivers.core.laws import Tested as LawTest
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ComputationId,
    Condition,
    EffectEquation,
    EffectLawBundle,
    EffectLawEvidence,
    EffectLawObligation,
    ExactJvp,
    ExecutionFailure,
    FiniteEffectLawDecision,
    Handle,
    checked_effect_plan,
    discharge_module,
    dumps,
    loads,
)

_SOURCE = """
effect Reader
    read_value : Unit -> Bool

effect Gate
    pass_value : Bool -> Bool

instance reader : Reader
instance gate : Gate

handler stable for Reader : Bool -> Bool with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    read_value resumes 1 =>
        resume(current)

handler toggle for Reader : Bool -> Bool with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    read_value resumes 1 =>
        if current then
            resume(true) with (false)
        else
            resume(false) with (true)

handler stateless for Reader : Bool -> Bool [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    read_value resumes 1 =>
        resume(false)

define first_read() : Bool !{reader} =
    perform reader.read_value()

define second_read() : Bool !{reader} =
    let ignored <- perform reader.read_value()
    perform reader.read_value()

define at_reader(initial : Bool) : Bool !{} =
    handle reader with stable(initial) in
        first_read()

handler relay for Gate : Bool -> Bool [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    pass_value(v : Bool) resumes 1 =>
        let observed <- at_reader(v)
        resume(observed)

handler unattached for Gate : Bool -> Bool [coverage=total, implementation=foreign]
    pass_value(v : Bool) resumes 1

define gate_once(v : Bool) : Bool !{gate} =
    perform gate.pass_value(v)

define gate_twice(v : Bool) : Bool !{gate} =
    let first <- perform gate.pass_value(v)
    perform gate.pass_value(first)

define through_clause(v : Bool) : Bool !{} =
    handle gate with relay in
        perform gate.pass_value(v)

define retained(v : Bool) : Bool !{} =
    handle gate with unattached in
        let observed <- perform gate.pass_value(v)
        at_reader(observed)

define through_retained(v : Bool) : Bool !{} =
    retained(v)
"""


@pytest.fixture(scope="module")
def compiler() -> Compiler:
    return Compiler(parse(_SOURCE), module_name="effect-plan-test")


@pytest.fixture(scope="module")
def bundle(compiler):
    module = compiler.qiec_module
    assert module is not None
    owner = next(item.entry for item in module.instances if item.name == "reader")
    effect = next(item for item in module.effects if item.ref.id == owner.effect.id)
    computations = {item.name: item.id for item in module.computations}
    equation = EffectEquation(
        effect.ref.id,
        "repeated-read",
        owner.instance,
        tuple(item.id for item in effect.operations),
        computations["first_read"],
        computations["second_read"],
    )
    handlers = {item.name: item.id for item in module.handlers}
    gate = next(item.entry for item in module.instances if item.name == "gate")
    gate_effect = next(item for item in module.effects if item.ref.id == gate.effect.id)
    gate_equation = EffectEquation(
        gate_effect.ref.id,
        "repeated-gate",
        gate.instance,
        tuple(item.id for item in gate_effect.operations),
        computations["gate_once"],
        computations["gate_twice"],
    )
    obligations = (
        EffectLawObligation(
            equation=equation, handler=handlers["stable"], method="finite"
        ),
        EffectLawObligation(
            equation=equation, handler=handlers["toggle"], method="finite"
        ),
        EffectLawObligation(
            equation=equation, handler=handlers["stateless"], method="finite"
        ),
        EffectLawObligation(
            equation=equation, handler=handlers["stable"], method="polynomial"
        ),
        EffectLawObligation(
            equation=equation,
            handler=handlers["toggle"],
            method="rational",
            guards=(
                Condition(
                    name="proof-claimed", statement="The reads may be interchanged."
                ),
            ),
        ),
        EffectLawObligation(
            equation=gate_equation,
            handler=handlers["unattached"],
            method="finite",
            guards=(
                Condition(name="idempotence", statement="The gate is idempotent."),
            ),
        ),
    )
    return EffectLawBundle(module=module, obligations=obligations)


def _answers(plan):
    names = {item.id: item.name for item in plan.bundle.module.handlers}
    return {
        (names[item.obligation.handler], item.obligation.method): item
        for item in plan.laws
    }


def _root(bundle, name):
    return next(item for item in bundle.module.computations if item.name == name)


def test_compiler_report_uses_actual_graph_and_checks_every_obligation(
    compiler, bundle
):
    reported = compiler.effect_plan("through_clause", obligations=bundle.obligations)
    names = {item.id: item.name for item in bundle.module.handlers}
    assert {names[shape.handler]: shape.kind for shape in reported.carriers} == {
        "relay": "continuation",
        "stable": "tail-state",
    }
    assert reported.computation == _root(bundle, "through_clause").id

    answers = _answers(reported)
    assert {subject: item.status for subject, item in answers.items()} == {
        ("stable", "finite"): "checked",
        ("toggle", "finite"): "refuted",
        ("stateless", "finite"): "checked",
        ("stable", "polynomial"): "unresolved",
        ("toggle", "rational"): "unresolved",
        ("unattached", "finite"): "unresolved",
    }
    assert answers["stable", "polynomial"].reason
    reason = answers["toggle", "rational"].reason
    assert reason is not None and "proof-claimed" in reason
    checked = answers["stable", "finite"].decision
    assert isinstance(checked, FiniteEffectLawDecision)
    assert checked.result.checked == 2
    rejected = answers["toggle", "finite"].decision
    assert isinstance(rejected, FiniteEffectLawDecision)
    witness = rejected.result.counterexample
    assert witness is not None
    alternatives = [(current, not current) for current in (False, True)]
    assert (witness.left, witness.right) == alternatives[0]
    assert witness.arguments == (alternatives[0][0],)


def test_unresolved_guards_provide_no_rewrite_permission(compiler, bundle):
    before = dumps(bundle.module)
    direct = discharge_module(bundle.module)
    reported = compiler.effect_plan("retained", obligations=bundle.obligations)
    assert reported.discharge == direct
    assert direct.obstacle("retained") is not None
    original = _root(bundle, "retained")
    root = next(item for item in direct.module.computations if item.id == original.id)
    assert isinstance(root.body, Handle)
    assert root is original
    guard = _answers(reported)["unattached", "finite"]
    assert guard.status == "unresolved" and guard.decision is None
    assert guard.obligation.handler == root.body.handler
    assert guard.obligation.equation.instance == root.body.instance
    assert guard.reason is not None and "idempotence" in guard.reason
    assert dumps(bundle.module) == before


def test_stage_reports_a_retained_transitive_callee(compiler, bundle):
    report = compiler.effect_plan("through_retained")
    names = {source.id: source.name for source in bundle.module.computations}
    stages = {names[stage.computation]: stage for stage in report.stages}
    assert stages["through_retained"].stage == "direct"
    assert stages["through_retained"].rewrite is None
    assert stages["retained"].stage == "reference"
    assert stages["retained"].obstacle == report.discharge.obstacle("retained")
    assert stages["retained"].obstacle.kind == "foreign"
    assert stages["at_reader"].stage == "handler-discharge"
    assert stages["at_reader"].rewrite.computation == _root(bundle, "at_reader").id
    assert "through_clause" not in stages


def test_comparison_checks_the_derivative_even_when_primal_values_agree():
    source = """
effect Echo
    echo : Real -> Real

instance feedback : Echo

handler identity for Echo : Real -> Real [coverage=total, forwards=none, implementation=authored]
    echo(v : Real) resumes 1 =>
        resume(v)

define feedback_square(x : Real) : Real !{} =
    handle feedback with identity in
        let a <- perform feedback.echo(x)
        return a * a
"""
    name = "feedback-effect-plan"
    plan = Compiler(parse(source), module_name=name).effect_plan("feedback_square")
    point = (2,)
    direction = (Fraction(3, 2),)
    result = plan.check_jvp(point, direction)
    assert result.source == result.staged == ExactJvp(Fraction(4), Fraction(6))
    assert result.status == "checked" and result.computation == plan.computation
    assert result.arguments == point and result.direction == direction
    assert result.source_obstacle is None and result.staged_obstacle is None
    bounded = plan.check_jvp(point, direction, fuel=1)
    assert bounded.status == "unresolved" and bounded.source_obstacle

    changed = Compiler(
        parse(source.replace("return a * a", "return a + a")), module_name=name
    ).qiec_module
    assert changed is not None
    wrong_stage = replace(plan, discharge=discharge_module(changed))
    discrepancy = wrong_stage.check_jvp(point, direction)
    assert discrepancy.source.value == discrepancy.staged.value == 4
    assert discrepancy.source.tangent == 6 and discrepancy.staged.tangent == 3
    assert discrepancy.status == "refuted"


def test_comparison_reports_a_regular_point_refusal():
    source = """
define boundary(x : Real) : Real !{} =
    if x < 0.0 then
        return -x
    else
        return x
"""
    plan = Compiler(parse(source), module_name="comparison-boundary").effect_plan(
        "boundary"
    )
    result = plan.check_jvp((0,), (1,))
    assert result.status == "unresolved"
    assert result.source is None and result.staged is None
    assert result.source_obstacle and result.staged_obstacle
    assert "comparison" in result.source_obstacle


def test_assignment_exhaustion_does_not_hide_other_obligations(bundle):
    root = _root(bundle, "retained")
    plan = checked_effect_plan(bundle, root.id, max_assignments=1)
    reports = _answers(plan)
    assert reports["stateless", "finite"].status == "checked"
    for name in ("stable", "toggle"):
        item = reports[name, "finite"]
        assert item.status == "unresolved"
        assert item.reason is not None and "assignments" in item.reason
    assert len(reports) == 6


def test_report_rechecks_changed_source_even_with_the_same_ids(bundle):
    stable = next(item for item in bundle.module.handlers if item.name == "stable")
    toggle = next(item for item in bundle.module.handlers if item.name == "toggle")
    changed = bundle.module.with_(
        handlers=tuple(
            item.with_(clauses=toggle.clauses) if item.id == stable.id else item
            for item in bundle.module.handlers
        )
    )
    changed_bundle = bundle.with_(module=changed)
    root = _root(bundle, "through_clause")
    old = checked_effect_plan(bundle, root.id)
    new = checked_effect_plan(changed_bundle, root.id)
    assert _answers(old)["stable", "finite"].status == "checked"
    assert _answers(new)["stable", "finite"].status == "refuted"
    assert (
        next(item for item in old.carriers if item.handler == stable.id).kind
        == "tail-state"
    )
    assert (
        next(item for item in new.carriers if item.handler == stable.id).kind
        == "tail-state"
    )


def test_wire_roundtrip_reproduces_the_report(bundle):
    restored = loads(dumps(bundle))
    assert isinstance(restored, EffectLawBundle)
    root = _root(bundle, "through_clause")
    assert checked_effect_plan(bundle, root.id) == checked_effect_plan(
        restored, root.id
    )


def test_planner_errors_on_invalid_limits_or_unknown_named_entries(compiler, bundle):
    with pytest.raises(ValueError, match="fuel must be a positive integer"):
        checked_effect_plan(bundle, _root(bundle, "retained").id, fuel=True)
    with pytest.raises(ValueError, match="unknown computation"):
        checked_effect_plan(bundle, ComputationId.derive("absent"))
    with pytest.raises(ExecutionFailure, match="unknown entry point"):
        compiler.effect_plan("absent")


def test_compiler_reports_the_elaborated_program_identity():
    compiler = Compiler(
        parse("""
object Obs : FinSet 4

program sampled : Obs -> Obs
    sample x <- Normal(loc=0.0, scale=1.0)
    return x

export sampled
"""),
        module_name="effect-plan-program",
    )
    module = compiler.qiec_module
    assert module is not None
    (program,) = module.entries
    plan = compiler.effect_plan("sampled")
    assert plan.computation == program.computation
    assert plan.laws == ()


def test_reports_keep_test_and_assumption_separate_from_exact_decisions(
    compiler, bundle
):
    baseline = _answers(
        compiler.effect_plan("through_clause", obligations=bundle.obligations)
    )
    unsupported = baseline["stable", "polynomial"].obligation
    false = baseline["toggle", "finite"].obligation
    reported = (
        EffectLawEvidence.for_module(
            bundle.module,
            unsupported,
            LawTest(check="recorded-cases", cases=2, domain="Boolean initial state"),
        ),
        EffectLawEvidence.for_module(
            bundle.module,
            false,
            Assumed(declared_by="caller", reason="comparison"),
        ),
    )
    decoded = tuple(loads(dumps(record)) for record in reported)
    assert decoded == reported
    plan = compiler.effect_plan(
        "through_clause", obligations=bundle.obligations, law_evidence=decoded
    )
    answers = _answers(plan)
    assert answers["stable", "polynomial"].status == "unresolved"
    assert answers["stable", "polynomial"].declared_evidence == reported[0]
    assert answers["toggle", "finite"].status == "refuted"
    assert answers["toggle", "finite"].declared_evidence == reported[1]
    decision = answers["toggle", "finite"].decision
    assert isinstance(decision, FiniteEffectLawDecision)
    assert decision.result.counterexample is not None
    assert plan.discharge == discharge_module(bundle.module)

    with pytest.raises(ValueError, match="duplicate declared evidence"):
        compiler.effect_plan(
            "through_clause",
            obligations=bundle.obligations,
            law_evidence=(reported[0],) * 2,
        )
    with pytest.raises(ValueError, match="different module or obligation"):
        compiler.effect_plan(
            "through_clause", obligations=(false,), law_evidence=reported
        )

    replacement = _SOURCE.replace(
        "        return value\n    read_value",
        "        return false\n    read_value",
        1,
    )
    stale = Compiler(parse(replacement), module_name="effect-plan-test")
    with pytest.raises(ValueError, match="different module or obligation"):
        stale.effect_plan(
            "through_clause", obligations=bundle.obligations, law_evidence=reported
        )
