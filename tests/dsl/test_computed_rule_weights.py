"""Telescope-dependent LogProb weights compose with checked deduction."""

from __future__ import annotations

from math import log

import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.dsl import Compiler, loads as load_source
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.parser import parse
from quivers.dsl.program_theory import extract_deduction_schema
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec import ComputedRuleWeight, QiecModule, dumps, loads, validate_module
from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import (
    AbsoluteLogError,
    AccuracyContract,
    Acyclic,
    Exact,
    InterpretationId,
    NoRetention,
    Numerics,
    OccurrenceId,
    QuerySpec,
    Target,
    Weight,
)
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.primitives import primitive
from quivers.qiec.program_runtime import deduction_item, run_deduction, run_program
from quivers.qiec.rules import AxiomOccurrence, Item, Rule
from quivers.qiec.terms import PrimitiveApplication, Return
from quivers.stochastic.forest import ForestRefused, build_forest, edge_weight
from quivers.stochastic.planning import PlanRefused, plan
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.qiec_ir import analyze_qiec_capabilities, lower_qiec_ir


SOURCE = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, D, S, same, lex
    rule combine : span(I, K, Left, lex), span(K, J, D, lex) |- span(I, J, S, same) #[weight=mass]
    rule repeated : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, S, same)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
        "y" : D = lex

define mass(i : Int, k : Int, left : Mix__Item, j : Int) : LogWeight !{} =
    let is_a <- Mix__eq(left, construct Mix__A() as Mix__Item)
    if k == 1 then
        if is_a then
            return weight(log(0.2))
        else
            return weight(log(0.6))
    else
        return weight(log(1.9))

program fit : Token -> Result
    let value = parse(Mix, sentence)
    score result = weight_value(value)
    return result
"""


CHART = """\
object Term : FinSet 2

deduction Agree : Term -> Term [semiring=LogProb, start=Goal]
    atoms P, Q, A, B, Goal
    rule combine : P(X), Q(X) |- Goal #[weight=mass]

define mass(x : Agree__Item) : LogWeight !{} =
    let is_a <- Agree__eq(x, construct Agree__A() as Agree__Item)
    if is_a then
        return weight(log(0.5))
    else
        return weight(log(0.25))
"""


@pytest.fixture(scope="module")
def compiled():
    return load_source(SOURCE + "\nexport fit\n")


def test_search_rule_weights_preserve_proof_multiplicity_and_gradients(
    compiled,
) -> None:
    data = {"sentence": ("x", "y")}
    assert compiled.run("fit", data=data).log_joint == pytest.approx(log(1.8))

    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule) and module == compiled.qiec
    logs = torch.tensor([-0.4, 0.25], dtype=torch.float64, requires_grad=True)
    a, b = logs.exp().unbind()
    # The sentence has one split and three distinct proofs: combine(A),
    # combine(B), and repeated(A). The last proof's rule weight is the unit.
    expected = (0.2 * a + 0.6 * b + a).log()
    wrong_position = (1.9 * a + 1.9 * b + a).log()
    assert abs(expected.item() - wrong_position.item()) > 0.2
    parameters = dict(zip(("Mix.lex.0", "Mix.lex.1"), logs.unbind(), strict=True))
    result = run_deduction(
        module, "Mix", tokens=data["sentence"], parameters=parameters, record_trace=True
    )
    program = run_program(module, "fit", data=data, sites={}, parameters=parameters)
    assert [
        event.detail["computation"]
        for event in result.result.trace
        if event.event == "call.entered" and event.detail["computation"] == "mass"
    ] == ["mass", "mass"]
    gradient = torch.autograd.grad(expected, logs)[0]
    for actual in (result.weight, program.log_joint):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, logs)[0], gradient)

    wrong = load_source(SOURCE.replace("if k == 1", "if i == 1") + "\nexport fit\n")
    mutated = run_program(wrong.qiec, "fit", data=data, sites={}, parameters=parameters)
    torch.testing.assert_close(mutated.log_joint, wrong_position)


def test_chart_rule_weight_is_constant_within_each_grounded_join() -> None:
    module = load_source(CHART).qiec
    weights = torch.tensor(
        [0.2, 0.3, 0.4, 0.7, 0.8], dtype=torch.float64, requires_grad=True
    )
    choices = (("P", "A"), ("P", "A"), ("Q", "A"), ("P", "B"), ("Q", "B"))
    axioms = [
        deduction_item(
            module, "Agree", relation, deduction_item(module, "Agree", value)
        )
        for relation, value in choices
    ]
    result = run_deduction(
        module,
        "Agree",
        axioms=axioms,
        axiom_weights=weights.log().unbind(),
        record_trace=True,
    )
    p1, p2, qa, pb, qb = weights.unbind()
    # Each P(A) occurrence can meet Q(A); P(B) can only meet Q(B).
    expected = (0.5 * p1 * qa + 0.5 * p2 * qa + 0.25 * pb * qb).log()
    assert any(
        event.event == "call.entered"
        and event.detail["computation"].startswith("Agree__chart_")
        for event in result.result.trace
    )
    torch.testing.assert_close(result.weight, expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.weight, weights)[0],
        torch.autograd.grad(expected, weights)[0],
    )


def test_zero_weight_preserves_exact_zero_and_finite_neighbor_gradient() -> None:
    source = SOURCE.replace("return weight(log(0.2))", "return weight(log(0.0))")
    source = "\n".join(
        line for line in source.split("\n") if "rule repeated" not in line
    )
    module = load_source(source).qiec
    weights = torch.tensor([-0.9, -0.1], dtype=torch.float64, requires_grad=True)
    result = run_deduction(
        module,
        "Mix",
        tokens=("x", "y"),
        parameters={"Mix.lex.0": weights[0], "Mix.lex.1": weights[1]},
    )
    torch.testing.assert_close(result.weight, log(0.6) + weights[1])
    torch.testing.assert_close(
        torch.autograd.grad(result.weight, weights)[0], weights.new_tensor([0.0, 1.0])
    )


@pytest.mark.parametrize("invalid", ["exp(1000.0) - exp(1000.0)", "exp(1000.0)"])
def test_invalid_log_domain_result_refuses_at_application(invalid) -> None:
    source = SOURCE.replace("return weight(log(0.2))", f"return weight({invalid})")
    module = load_source(source).qiec
    with pytest.raises(ExecutionFailure, match="must be finite or negative infinity"):
        run_deduction(module, "Mix", tokens=("x", "y"))


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("weight=mass", "weight=absent"),
        SOURCE.replace("j : Int) : LogWeight !{}", "j : Int) : Real !{}"),
        SOURCE.replace("define mass(", "define mass[n : Nat]("),
        SOURCE.replace("k : Int, left : Mix__Item", "left : Mix__Item, k : Int"),
    ],
)
def test_weight_operation_has_one_exact_closed_signature(source) -> None:
    with pytest.raises(
        QiecDiagnosticError, match="must name an authored computation"
    ) as error:
        load_source(source)
    assert error.value.code == "qiec-deduction-rule-weight"
    assert error.value.line == 6


def test_equal_shape_weight_fields_keep_their_deduction_identity() -> None:
    begin = SOURCE.index("deduction Mix ")
    end = SOURCE.index("define mass")
    other = (
        SOURCE[begin:end]
        .replace("deduction Mix", "deduction Other")
        .replace(" #[weight=mass]", "")
    )
    source = (
        SOURCE[:end]
        + other
        + SOURCE[end:].replace("left : Mix__Item", "left : Other__Item")
    )
    with pytest.raises(QiecDiagnosticError, match="Left: Mix__Item"):
        load_source(source)


def test_installed_handler_is_refused_through_a_pure_helper() -> None:
    helper = """\
effect Probe
    inspect : LogWeight -> LogWeight

instance probe : Probe

handler unchanged for Probe : LogWeight -> LogWeight [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    inspect(value : LogWeight) resumes 1 =>
        resume(value)

define intercepted(value : LogWeight) : LogWeight !{} =
    handle probe with unchanged in
        let weight <- perform probe.inspect(value)
        return weight

"""
    source = SOURCE.replace("define mass", helper + "define mass").replace(
        "return weight(log(0.2))", "intercepted(weight(log(0.2)))"
    )
    with pytest.raises(
        QiecDiagnosticError, match="'intercepted' installs an effect handler"
    ):
        load_source(source)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            SOURCE.replace("weight=mass", "weight=0.2"),
            "must name one checked computation",
        ),
        (SOURCE.replace("weight=mass", "weight=mass, learnable"), "cannot be combined"),
        (SOURCE.replace("start=S", "start=S, normalize=lhs"), "cannot be combined"),
        (
            SOURCE.replace("S, same)\n", "S, same) #[parent=combine]\n"),
            "cannot inherit",
        ),
        (SOURCE.replace("semiring=LogProb", "semiring=Viterbi"), "requires LogProb"),
    ],
)
def test_weight_policies_do_not_mix(source, reason) -> None:
    with pytest.raises((QiecDiagnosticError, CompileError), match=reason):
        load_source(source)


def test_presentation_keeps_weight_requirement_without_legacy_parameter(
    compiled,
) -> None:
    deduction = compiled.deductions["Mix"]
    assert deduction.presentation is not None
    (rule,) = deduction.presentation.computed_weight_rules
    assert rule.weight == ComputedRuleWeight(operation="mass")
    restored = loads(dumps(rule))
    assert isinstance(restored, Rule) and restored == rule
    assert "_rule_module" not in deduction.modules
    assert set(dict(deduction.modules["_axiom_module"].named_parameters())) == {
        "lex_weight_0",
        "lex_weight_1",
    }

    with pytest.raises(ValueError, match="require checked QVR execution"):
        deduction.run(("x", "y"))
    module = compiled.qiec
    missing = module.with_(
        computations=tuple(item for item in module.computations if item.name != "mass")
    )
    with pytest.raises(KernelError, match="computation"):
        validate_module(missing)


def test_deduction_schema_distinguishes_computation_from_parameter_weight() -> None:
    source = SOURCE.replace("S, same)\n", "S, same) #[learnable]\n")
    compiler = Compiler(parse(source))
    compiler.compile()
    schema = extract_deduction_schema(compiler)
    computed = {
        item.sort: item.value
        for item in schema.constraints_for("deduction:Mix/rule:combine")
    }
    parameter = {
        item.sort: item.value
        for item in schema.constraints_for("deduction:Mix/rule:repeated")
    }
    assert computed["weight-computation"] == "mass"
    assert "weight" not in computed
    assert parameter["weight"].startswith("qiec:parameter:")
    assert "weight-computation" not in parameter


def test_unbound_rule_weight_refuses_before_planning_and_forest_evaluation(
    compiled,
) -> None:
    presentation = compiled.deductions["Mix"].presentation
    assert presentation is not None
    weight_query = QuerySpec(
        presentation=presentation.id,
        interpretation=InterpretationId.derive(presentation.id, "weight"),
        observable=Weight(),
    )
    accuracy = AccuracyContract(
        numerics=Numerics(dtype="float64"),
        target=Target(quantity="inside mass", metric=AbsoluteLogError()),
        guarantee=Exact(),
    )
    for executor in ("auto", "eliminate"):
        with pytest.raises(PlanRefused, match="no executable attachment"):
            plan(
                weight_query,
                presentation,
                LOG_PROB,
                Acyclic(),
                accuracy,
                NoRetention(),
                executor=executor,
            )

    slot = next(item for item in presentation.axioms if item.name == "axiom_span")
    inputs = [(0, 1, "A", "lex"), (1, 2, "D", "lex")]
    names = [binder.name for binder in slot.telescope]
    axioms = [
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive(presentation.id, index),
            binding=tuple(zip(names, arguments, strict=True)),
            values=(token,),
        )
        for index, (arguments, token) in enumerate(zip(inputs, ("x", "y"), strict=True))
    ]
    forest = build_forest(presentation, axioms)
    goal = Item(family=slot.judgment.family, arguments=(0, 2, "S", "same"))
    (computed, unit) = forest.edges[goal]
    assert computed.constructor == presentation.computed_weight_rules[0].id
    with pytest.raises(ForestRefused, match="no executable attachment"):
        edge_weight(forest, computed, LOG_PROB, {}, like=torch.zeros(()))
    assert edge_weight(forest, unit, LOG_PROB, {}, like=torch.zeros(())) == 0.0


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        ("parse_outcomes(Mix, sentence)", "outcome and proof queries"),
        ("parse_proofs(Mix, sentence, 4)", "outcome and proof queries"),
        ("parse_linear(Mix, sentence, 32)", "parse_linear cannot execute"),
    ],
)
def test_optimized_source_queries_refuse_binding_dependent_weights(
    call, reason
) -> None:
    source = SOURCE.replace("parse(Mix, sentence)", call)
    with pytest.raises(QiecDiagnosticError, match=reason):
        load_source(source)


def test_external_target_refuses_only_reached_native_deduction() -> None:
    with pytest.raises(UnsupportedConstruct, match="Stan has no runtime") as error:
        transpile(parse(SOURCE + "\nexport fit\n"), target="stan")
    assert error.value.kinds == ["qiec:capability:search:Mix__run"]

    independent = parse(
        SOURCE
        + """\

program independent : Token -> Result
    sample number <- Normal(loc=0.0, scale=1.0)
    return number

export independent
"""
    )
    assert b"normal" in transpile(independent, target="stan")


def test_log_domain_guard_has_no_generic_host_primitive_lowering() -> None:
    source = """\
define guard(value : LogWeight) : LogWeight !{} =
    return value

export guard
"""
    module = lower_qvr_to_qiec(parse(source))
    (identity,) = module.computations
    assert isinstance(identity.body, Return)
    signature = primitive("require_rule_log_weight")
    checked = module.with_(
        computations=(
            identity.with_(
                body=Return(
                    value=PrimitiveApplication(
                        primitive=signature.id,
                        name=signature.name,
                        arguments=(identity.body.value,),
                        result_type=signature.result,
                        origin=identity.origin,
                    ),
                ),
            ),
        ),
    )
    validate_module(checked)
    ir = lower_qiec_ir(checked)
    (refusal,) = analyze_qiec_capabilities(ir, "pyro")
    assert refusal.kind == "qiec:capability:rule-log-weight:guard"
    assert refusal.origin is not None
    assert refusal.origin.line == 1
