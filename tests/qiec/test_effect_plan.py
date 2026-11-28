"""Reported effect plans preserve independent law evidence and ordinary dispatch."""

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from typing import cast

import pytest
import pyro
import torch
from panproto import PanprotoError

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
    RationalDerivativeDeclaration,
    checked_effect_plan,
    discharge_module,
    dumps,
    loads,
)
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile import _entry as target_entry
from quivers.transpile.ir import IRSample
from quivers.transpile.qiec_ir import IRQiecHandle

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
    assert discrepancy.source is not None and discrepancy.staged is not None
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
    target = plan.check_target("pyro", arguments=(0,), direction=(1,))
    assert target.projection == "source" and target.native_jvp == result
    assert target.external_derivative.missing == ("external derivative unverified",)


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
    target = plan.check_target("pyro")
    assert target.projection == "program-plan"
    assert tuple(stage.computation for stage in target.stages) == (program.computation,)
    assert tuple(stage.stage for stage in target.stages) == ("program-plan",)
    assert target.refusals == ()

    wire = loads(dumps(plan.bundle))
    assert isinstance(wire, EffectLawBundle)
    wire_plan = checked_effect_plan(wire, plan.computation)
    assert wire_plan == plan
    assert wire_plan.check_target("pyro").refusals == (
        f"qiec:target-stage:program-plan-required:{program.name}",
    )


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


_TARGET_SOURCE = """
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

define entry(x : Real) : Real !{} =
    feedback_square(x)
"""


def test_target_uses_actual_emission_and_keeps_external_derivative_unverified():
    parsed = parse(_TARGET_SOURCE)
    plan = Compiler(parsed, module_name="target-effect-stages").effect_plan("entry")
    names = {owner.id: owner.name for owner in plan.bundle.module.computations}
    expected = ExactJvp(Fraction(4), Fraction(6))

    pyro = plan.check_target("pyro", arguments=(2,), direction=(Fraction(3, 2),))
    stan = plan.check_target("stan", arguments=(2,), direction=(Fraction(3, 2),))
    assert pyro.projection == "source"
    assert stan.projection == "discharged"
    assert {names[stage.computation]: stage.stage for stage in pyro.stages} == {
        "entry": "direct",
        "feedback_square": "source-handler",
    }
    assert {names[stage.computation]: stage.stage for stage in stan.stages} == {
        "entry": "direct",
        "feedback_square": "handler-discharge",
    }
    for report in (pyro, stan):
        assert report.refusals == ()
        assert report.laws == plan.laws
        assert report.computation == plan.computation
        assert report.native_jvp is not None
        assert report.native_jvp.source == report.native_jvp.staged == expected
        assert report.native_jvp.status == "checked"
        assert report.external_derivative.missing == ("external derivative unverified",)

    # Independent observations of public target artifacts at this point.
    namespace = {}
    exec(transpile(parsed, target="pyro"), namespace)
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    value = namespace["qiec_entry"](x)
    (gradient,) = torch.autograd.grad(value, x)
    assert value.item() == 4.0 and (gradient * 1.5).item() == 6.0
    assert b"real qiec_entry" in transpile(parsed, target="stan")

    with pytest.raises(UnsupportedConstruct) as refusal:
        transpile(parsed, target="jags")
    jags = plan.check_target("jags", arguments=(2,), direction=(1,))
    assert jags.refusals == tuple(refusal.value.kinds)
    assert "qiec:capability:arithmetic:feedback_square" in jags.refusals
    assert jags.detail is not None and "requires the `arithmetic`" in jags.detail
    assert jags.projection is None and jags.stages == () and jags.native_jvp is None
    assert jags.external_derivative.missing == ("target export refused",)


def test_only_the_discharge_target_compares_a_changed_discharged_graph():
    name = "target-effect-mutation"
    plan = Compiler(parse(_TARGET_SOURCE), module_name=name).effect_plan("entry")
    changed = Compiler(
        parse(_TARGET_SOURCE.replace("return a * a", "return a + a")),
        module_name=name,
    ).qiec_module
    assert changed is not None
    mutated = replace(plan, discharge=discharge_module(changed))
    pyro = mutated.check_target("pyro", arguments=(2,), direction=(3,))
    stan = mutated.check_target("stan", arguments=(2,), direction=(3,))
    assert pyro.native_jvp is not None and pyro.native_jvp.status == "checked"
    assert stan.native_jvp is not None and stan.native_jvp.status == "refuted"
    assert stan.native_jvp.source == ExactJvp(Fraction(4), Fraction(12))
    assert stan.native_jvp.staged == ExactJvp(Fraction(4), Fraction(6))
    assert stan.external_derivative.missing == ("external derivative unverified",)


def test_renderer_resource_refusal_agrees_with_public_emission():
    # Host bindings use nested continuations. Stan lowers the same checked
    # sequence to local statements without the host grammar nesting cost.
    lines = [
        "define relay(x : Real) : Real !{} =",
        "    return x",
        "",
        "define forward(x : Real) : Real !{} =",
    ]
    previous = "x"
    for i in range(48):
        name = f"edge_{i}"
        lines.append(f"    let {name} <- relay({previous})")
        previous = name
    lines.append(f"    return {previous}")
    source = parse("\n".join(lines) + "\n")
    plan = Compiler(source, module_name="target-depth").effect_plan("forward")
    with pytest.raises(PanprotoError, match="nesting exceeds") as emitted_error:
        transpile(source, target="pyro")
    pyro = plan.check_target("pyro")
    assert pyro.projection is None
    assert pyro.refusals == ("qiec:target-emission:PanprotoError",)
    assert pyro.detail == str(emitted_error.value)
    assert plan.check_target("stan").projection == "discharged"


def test_derivative_rule_evidence_is_checked_in_its_named_source_scope():
    compiler = Compiler(
        parse("""
define square(x : Real) : Real !{} =
    return x * x

define push(x : Real, direction : Real) : Real !{} =
    return 2.0 * x * direction

define undercounts(x : Real, direction : Real) : Real !{} =
    return x * direction
"""),
        module_name="effect-target-rules",
    )
    plan = compiler.effect_plan("square")
    ids = {owner.name: owner.id for owner in plan.bundle.module.computations}
    correct = RationalDerivativeDeclaration(primal=ids["square"], jvp=ids["push"])
    false = correct.with_(jvp=ids["undercounts"])
    tools = plan.check_target(
        "stan", derivative_rule=correct, arguments=(2,), direction=(3,)
    )
    assert tools.derivative_rule == correct and tools.rule_obstacle is None
    assert tools.rule_decision is not None and tools.rule_decision.certified
    assert tools.native_jvp is not None and tools.native_jvp.status == "checked"
    assert tools.external_derivative.missing == ("external derivative unverified",)

    wrong = plan.check_target("stan", derivative_rule=false)
    assert wrong.derivative_rule == false and wrong.rule_obstacle is None
    assert wrong.rule_decision is not None and not wrong.rule_decision.certified
    assert wrong.rule_decision.jvp is not None
    assert wrong.rule_decision.jvp.counterexample is not None
    assert wrong.external_derivative == tools.external_derivative

    unrelated = plan.check_target(
        "stan", derivative_rule=correct.with_(primal=ids["push"])
    )
    assert unrelated.rule_decision is None
    assert (
        unrelated.rule_obstacle
        == "the declared derivative belongs to another named computation"
    )
    limited = plan.check_target("stan", derivative_rule=correct, max_rule_work=1)
    assert limited.rule_decision is None and limited.rule_obstacle

    unknown = plan.check_target("unknown")
    assert unknown.refusals[0].startswith("target:unknown:unknown:")
    with pytest.raises(ValueError, match="both arguments and direction"):
        plan.check_target("stan", arguments=(2,))


_PROGRAM_EXPORT_SOURCE = (
    _TARGET_SOURCE
    + """
object Obs : FinSet 4

program primary : Obs -> Obs
    sample x <- Normal(loc=0.0, scale=1.0)
    return x

program alternative : Obs -> Obs
    sample y <- Normal(loc=5.0, scale=1.0)
    return y

export primary
"""
)


@pytest.mark.parametrize(
    ("target", "helpers"), [("pyro", "source"), ("stan", "discharged")]
)
def test_program_target_reports_the_actual_public_plan_and_output(
    target, helpers, monkeypatch
):
    parsed = parse(_PROGRAM_EXPORT_SOURCE)
    compiler = Compiler(parsed, module_name="target-program-source")
    plan = compiler.effect_plan("primary")
    emitted = []
    emit = target_entry._emit_target

    def record(plan, target):
        code = emit(plan, target)
        emitted.append((plan, code))
        return code

    monkeypatch.setattr(target_entry, "_emit_target", record)
    public = transpile(parsed, target=target)
    report = plan.check_target(target, arguments=(), direction=())
    assert report.refusals == () and report.projection == "program-plan"
    assert report.helper_projection == helpers
    assert report.native_jvp is None
    assert report.native_obstacle == "qiec:target-stage:program-jvp-unverified"
    assert report.external_derivative.missing == ("external derivative unverified",)
    assert tuple((entry.computation, entry.stage) for entry in report.stages) == (
        (plan.computation, "program-plan"),
    )
    assert len(emitted) == 2
    (public_plan, public_code), (reported_plan, reported_code) = emitted
    assert reported_plan == public_plan and reported_code == public_code == public
    assert reported_plan.name == "primary"
    assert len([node for node in reported_plan.body if isinstance(node, IRSample)]) == 1
    helper = next(
        item
        for item in reported_plan.module.computations
        if item.name == "feedback_square"
    )
    assert isinstance(helper.body, IRQiecHandle) == (helpers == "source")
    if target == "pyro":
        namespace = {"pyro": pyro}
        exec(reported_code, namespace)
        generated = cast(Callable[[], torch.Tensor], namespace["model"])
        model = pyro.poutine.condition(generated, data={"x": torch.tensor(2.0)})
        trace = pyro.poutine.trace(model).get_trace()
        draw = trace.nodes["x"].get("value")
        answer = trace.nodes["_RETURN"].get("value")
        assert isinstance(draw, torch.Tensor) and isinstance(answer, torch.Tensor)
        assert draw.item() == answer.item() == 2.0
        assert "y" not in trace.nodes
    else:
        assert b"normal_lpdf(x" in public and b"normal_lpdf(y" not in public

    nonexported = compiler.effect_plan("alternative").check_target(target)
    assert nonexported.projection is None
    assert nonexported.refusals == (
        "qiec:target-stage:program-not-exported:alternative:primary",
    )
    assert len(emitted) == 2


def test_program_target_rejects_a_changed_module_with_the_same_nominal_entry(
    monkeypatch,
):
    name = "program-provenance"
    original = Compiler(parse(_PROGRAM_EXPORT_SOURCE), module_name=name).effect_plan(
        "primary"
    )
    changed = Compiler(
        parse(_PROGRAM_EXPORT_SOURCE.replace("loc=0.0", "loc=3.0")),
        module_name=name,
    ).effect_plan("primary")
    assert (
        changed.computation == original.computation
        and changed.bundle != original.bundle
    )

    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("an unrelated program source reached public preparation")

    monkeypatch.setattr(target_entry, "_prepare_target", unexpected_preparation)
    for mismatched in (
        replace(original, bundle=changed.bundle),
        replace(original, _program_source=changed._program_source),
        replace(original, _program_source=parse(_PROGRAM_EXPORT_SOURCE)),
    ):
        report = mismatched.check_target("pyro")
        assert report.projection is None
        assert report.refusals == ("qiec:target-stage:program-source-mismatch:primary",)


def test_program_target_reports_actual_public_planner_refusal():
    parsed = parse("""
object Row : FinSet 3
object Out : Real 1

program weighted : Row -> Out
    observe y : Row <- Normal(loc=0.0, scale=1.0) [weights=w]
    return y

export weighted
""")
    compiler = Compiler(parsed, module_name="target-program-weighted")
    with pytest.raises(UnsupportedConstruct) as error:
        transpile(parsed, target="pyro")
    report = compiler.effect_plan("weighted").check_target("pyro")
    assert report.projection is None and report.stages == ()
    assert (
        report.refusals
        == tuple(error.value.kinds)
        == ("qiec:weighted-observe:weighted",)
    )
    assert report.detail == str(error.value)
    assert report.external_derivative.missing == ("target export refused",)
