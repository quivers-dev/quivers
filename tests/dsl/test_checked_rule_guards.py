"""Source rule guards preserve occurrence-sensitive checked deduction."""

from math import log

import didactic.api as dx
import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.dsl.program_theory import extract_deduction_schema
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.effects.checked_program import CheckedProgram
from quivers.qiec import (
    DeductionOutcomes,
    KernelError,
    LiteralValue,
    QiecModule,
    Return,
    deduction_item,
    dumps,
    loads,
    run_deduction,
    run_program,
    validate_module,
)
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.types import BOOL
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.qiec_ir import outcome_capability_diagnostics

SOURCE = """\
object Token : FinSet 8
object Result : Real 1

deduction Guarded : Token -> Token [semiring=LogProb, start=S, normalize=lhs]
    atoms span, A, S, word
    rule grow : span(I, K, A), span(K, J, A) |- span(I, J, A)
    rule first : span(I, K, A), span(K, J, A) |- span(I, J, S) #[guard=forward]
    rule second : span(I, K, A), span(K, J, A) |- span(I, J, S) #[guard=forward]
    rule reject : span(I, K, A), span(K, J, A) |- span(I, J, S) #[guard=never]
    lexicon
        "x" : A = word

define forward(i : Int, k : Int, j : Int) : Bool !{} =
    let left_width = k - i
    let right_width = j - k
    if left_width == 1 then
        return right_width == 2
    else
        return false

define never(i : Int, k : Int, j : Int) : Bool !{} =
    return i > j

define mass(rows : Guarded__Outcomes) : Real !{} =
    case rows motive => Real
        Guarded__outcome_end =>
            return 0.0
        Guarded__outcome_entry(key, weight, tail) =>
            return exp(weight)

program scalar : Token -> Result
    sample p_a <- Dirichlet(concentration=[1.0, 1.0])
    sample p_s <- Dirichlet(concentration=[1.0, 1.0, 1.0])
    let chart = parse(Guarded, sentence, ("A", p_a), ("S", p_s))
    score observed = chart.goal_weight()
    return observed

program packed : Token -> Result
    sample p_a <- Dirichlet(concentration=[1.0, 1.0])
    sample p_s <- Dirichlet(concentration=[1.0, 1.0, 1.0])
    let rows = parse_outcomes(Guarded, sentence, "item", ("A", p_a), ("S", p_s))
    let observed <- mass(rows)
    score response = log(observed)
    return response

"""

INDEPENDENT = """\
program independent : Token -> Result
    sample value <- Normal(loc=0.0, scale=1.0)
    return value
"""


def _descriptor(module):
    (handler,) = tuple(
        handler
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionOutcomes)
    )
    return handler, handler.foreign


@pytest.fixture(scope="module")
def compiled():
    return load_source(SOURCE + "\nexport scalar\n")


@pytest.mark.parametrize("chooses_second_split", (False, True))
def test_guards_filter_ground_matches_before_aggregation(
    compiled, chooses_second_split
):
    # There are two full-span splits. Each needs one A growth rule
    # and three lexical axioms. The first two S rules both accept just
    # the (1, 2) split, yielding distinct proofs of the same item.
    # Changing just the second guard's width admits both splits under
    # that rule. The third S rule's reverse endpoint condition never holds.
    if chooses_second_split:
        source = (
            SOURCE.replace(
                "#[guard=forward]\n    rule reject",
                "#[guard=both]\n    rule reject",
            )
            + """
define both(i : Int, k : Int, j : Int) : Bool !{} =
    return i < j
"""
        )
        compiled = load_source(source + "\nexport scalar\n")
    assert isinstance(compiled.morphism, CheckedProgram)
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule) and module == compiled.qiec
    _, descriptor = _descriptor(module)
    presentation = descriptor.presentation
    assert presentation.conditions_by_id
    conditions = {
        rule.name: (
            presentation.conditions_by_id[rule.side_condition.computation].name,
            rule.side_condition.arguments,
        )
        for rule in presentation.rules
        if rule.side_condition is not None
    }
    assert conditions == {
        "first": ("forward", ("I", "K", "J")),
        "second": ("both" if chooses_second_split else "forward", ("I", "K", "J")),
        "reject": ("never", ("I", "K", "J")),
    }

    logits_a = torch.tensor((-0.1, 0.4), dtype=torch.float64, requires_grad=True)
    logits_s = torch.tensor((-0.4, 0.2, 0.7), dtype=torch.float64, requires_grad=True)
    for _ in range(2):
        p_a, p_s = (values.softmax(0) for values in (logits_a, logits_s))
        data = {"sentence": ("x", "x", "x")}
        sites = {"p_a": p_a, "p_s": p_s}
        expected = (
            p_a[0] * p_a[1] ** 3 * (p_s[0] + (1 + chooses_second_split) * p_s[1])
        ).log()
        expected_gradient = torch.autograd.grad(
            expected, (logits_a, logits_s), retain_graph=True
        )
        expected_rule_gradient = torch.autograd.grad(expected, p_s, retain_graph=True)[
            0
        ]
        assert expected_rule_gradient[0] > 0 and expected_rule_gradient[1] > 0
        assert expected_rule_gradient[2] == 0
        for entry in ("scalar", "packed"):
            run = run_program(
                module, entry, data=data, sites=sites, distributions=TensorBackend()
            )
            torch.testing.assert_close(
                run.log_joint, expected + log(2), atol=1e-12, rtol=1e-12
            )
            gradient = torch.autograd.grad(
                run.log_joint, (logits_a, logits_s), retain_graph=True
            )
            raw_rule_score = run.log_joint + p_s.sum().log()
            direct = torch.autograd.grad(raw_rule_score, p_s, retain_graph=True)[0]
            torch.testing.assert_close(direct, expected_rule_gradient)
            if entry == "packed":
                (work,) = (
                    event.detail
                    for event in run.result.trace
                    if event.event == "deduction.outcomes"
                )
                assert work["checked_rule_guards"] == 12
                assert dict(work["construction_work"])["provenance_records"] == (
                    30 if chooses_second_split else 21
                )
            for actual, wanted in zip(gradient, expected_gradient, strict=True):
                torch.testing.assert_close(actual, wanted, atol=1e-12, rtol=1e-12)
        with torch.no_grad():
            logits_a.add_(0.2 * expected_gradient[0])
            logits_s.add_(0.2 * expected_gradient[1])


def test_guarded_presentation_survives_schema_and_refuses_legacy_runtime(compiled):
    compiler = Compiler(parse(SOURCE + "\nexport scalar\n"))
    compiler.compile()
    schema = extract_deduction_schema(compiler)
    constraints = schema.constraints_for("deduction:Guarded/rule:first")
    values = {
        item.sort: item.value for item in constraints if item.sort != "guard-argument"
    }
    assert values["guard-computation"].startswith("qiec:computation:")
    assert {item.value for item in constraints if item.sort == "guard-argument"} == {
        "0:I",
        "1:K",
        "2:J",
    }
    deduction = compiled.deductions["Guarded"]
    for sentence in (("x", "x", "x"), ("missing",)):
        with pytest.raises(
            ValueError, match="rule guards require checked QVR execution"
        ):
            deduction.run(sentence)


def test_guard_binding_refuses_a_different_body_with_the_same_identity(compiled):
    module = compiled.qiec
    original, descriptor = _descriptor(module)
    presentation = descriptor.presentation
    guard = next(guard for guard in presentation.conditions if guard.name == "forward")
    rewritten = guard.with_(body=Return(value=LiteralValue(value=False, type=BOOL)))
    assert rewritten.id == guard.id and rewritten.name == guard.name
    wrong = presentation.with_(
        conditions=tuple(
            rewritten if computation is guard else computation
            for computation in presentation.conditions
        )
    )
    modified = module.with_(
        handlers=tuple(
            handler.with_(foreign=descriptor.with_(presentation=wrong))
            if handler is original
            else handler
            for handler in module.handlers
        )
    )
    with pytest.raises(KernelError, match="outer module's checked computation"):
        validate_module(modified)
    with pytest.raises(
        (ValueError, KernelError), match="outer module's checked computation"
    ):
        reloaded = loads(dumps(modified))
        assert isinstance(reloaded, QiecModule)
        validate_module(reloaded)

    rule = next(rule for rule in presentation.rules if rule.name == "first")
    condition = rule.side_condition
    assert condition is not None
    wrong_order = presentation.with_(
        rules=tuple(
            rule.with_(side_condition=condition.with_(arguments=("K", "I", "J")))
            if actual is rule
            else actual
            for actual in presentation.rules
        )
    )
    with pytest.raises(dx.ValidationError, match="ordered matched Int fields"):
        descriptor.with_(presentation=wrong_order)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("guard=forward", "guard=absent", "names no checked computation"),
        (
            "i : Int, k : Int, j : Int",
            "i : Int, k : Int, j : Int, extra : Int",
            "matched Int coordinates in telescope order",
        ),
        (
            "i : Int, k : Int, j : Int",
            "i : Real, k : Int, j : Int",
            "Bool computation of Ints",
        ),
        ("return i > j", "forward(i, k, j)", "without named calls or handlers"),
        ("return i > j", "return log(-1.0) < 0.0", "only scalar Int and Bool"),
    ],
)
def test_guard_rejects_unavailable_names_or_carriers(old, new, message):
    with pytest.raises((QiecDiagnosticError, ValueError), match=message):
        load_source(SOURCE.replace(old, new, 1) + "\nexport scalar\n")


def test_native_target_refusal_remains_specific_to_reached_guard(compiled):
    (diagnostic,) = (
        item
        for item in outcome_capability_diagnostics(compiled.qiec, "stan")
        if item.computation == "scalar"
    )
    assert diagnostic.kind == "qiec:capability:deduction-rule-guard:scalar"
    assert diagnostic.origin is not None
    parse_line = next(
        lineno
        for lineno, line in enumerate(SOURCE.splitlines(), 1)
        if "let chart = parse(" in line
    )
    assert diagnostic.origin.line == parse_line
    with pytest.raises(UnsupportedConstruct) as error:
        transpile(parse(SOURCE + "\nexport scalar\n"), target="stan")
    assert diagnostic.kind in error.value.kinds
    assert b"normal" in transpile(
        parse(SOURCE + INDEPENDENT + "\nexport independent\n"), target="stan"
    )


def test_unsupported_guard_evaluation_fails_the_query():
    source = SOURCE.replace("return i > j", "return i > 9223372036854775808")
    module = load_source(source + "\nexport scalar\n").qiec
    for entry in ("scalar", "packed"):
        with pytest.raises(ExecutionFailure, match="Overflow"):
            run_program(
                module,
                entry,
                data={"sentence": ("x", "x", "x")},
                sites={"p_a": (0.35, 0.65), "p_s": (0.2, 0.3, 0.5)},
                distributions=TensorBackend(),
            )


CHART = """\
object Term : FinSet 2

deduction Chart : Term -> Term [semiring=LogProb, start=S]
    atoms A, B, S
    rule keep : A |- S #[guard=admit]
    rule skip : B |- S #[guard=deny,weight=unreachable]

define admit() : Bool !{} =
    return true

define deny() : Bool !{} =
    return false

define unreachable() : LogWeight !{} =
    return weight(log(-1.0))
"""


def test_guard_precedes_weight_evaluation_on_the_checked_fixed_point():
    module = load_source(CHART).qiec
    assert isinstance(module, QiecModule)
    first, second = (deduction_item(module, "Chart", kind) for kind in ("A", "B"))
    weights = torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64, requires_grad=True)
    actual = run_deduction(
        module,
        "Chart",
        axioms=(first, first, second),
        axiom_weights=weights.log().unbind(),
        record_trace=True,
    )
    assert any(
        event.event == "call.entered" and event.detail["computation"] == "admit"
        for event in actual.result.trace
    )
    assert all(
        event.event != "call.entered" or event.detail["computation"] != "unreachable"
        for event in actual.result.trace
    )
    expected = weights[:2].sum().log()
    torch.testing.assert_close(actual.weight, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.weight, weights)[0],
        torch.autograd.grad(expected, weights)[0],
    )
