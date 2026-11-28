"""Checked rule guards over the deduction's own matched constructor values."""

from math import log

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.dsl.program_theory import extract_deduction_schema
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    DeductionOutcomes,
    KernelError,
    LiteralValue,
    QiecModule,
    Return,
    dumps,
    loads,
    run_program,
    validate_module,
)
from quivers.qiec._deduction_guards import _TermGuard
from quivers.qiec.types import BOOL
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.qiec_ir import outcome_capability_diagnostics


SOURCE = """\
object Token : FinSet 2
object Result : Real 1

deduction Forms : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, Box, On, Off, plain
    rule first : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain) #[guard=select]
    rule second : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain) #[guard=select]
    rule shape : span(I, K, S, Box(On)), span(K, J, B, plain) |- span(I, J, S, plain)
    lexicon
        "x" : A = Box(On) #[learnable]
        "x" : A = Box(Off) #[learnable]
        "x" : A = On #[learnable]
        "y" : B = plain

define select(i : Int, k : Int, f : Forms__Item, j : Int) : Bool !{} =
    case f motive => Bool
        Forms__Box(content) =>
            case content motive => Bool
                Forms__On =>
                    return k == 1
                Forms__Off =>
                    return false
                Forms__Box(child) =>
                    return false
                Forms__A =>
                    return false
                Forms__B =>
                    return false
                Forms__S =>
                    return false
                Forms__span(left, right, category, form) =>
                    return false
                Forms__plain =>
                    return false
        Forms__On =>
            return false
        Forms__Off =>
            return false
        Forms__A =>
            return false
        Forms__B =>
            return false
        Forms__S =>
            return false
        Forms__span(left, right, category, form) =>
            return false
        Forms__plain =>
            return false

define read_mass(rows : Forms__Outcomes) : Real !{} =
    case rows motive => Real
        Forms__outcome_end =>
            return log(0.0)
        Forms__outcome_entry(key, weight, tail) =>
            return weight

program scalar : Token -> Result
    let chart = parse(Forms, sentence)
    score answer = weight_value(chart)
    return answer

program packed : Token -> Result
    let rows = parse_outcomes(Forms, sentence, "item")
    let weight <- read_mass(rows)
    score answer = weight
    return answer
"""


@pytest.fixture(scope="module")
def checked() -> QiecModule:
    module = load_source(SOURCE + "\nexport scalar\nexport packed\n").qiec
    assert isinstance(module, QiecModule)
    return module


def _term_guard(module):
    (handler,) = (
        item for item in module.handlers if isinstance(item.foreign, DeductionOutcomes)
    )
    chart = handler.foreign
    (guard,) = chart.presentation.conditions
    assert isinstance(guard, _TermGuard)
    return handler, chart, guard


@pytest.mark.parametrize("accepts_off", (False, True))
def test_same_checked_term_guard_controls_scalar_and_packed_mass(checked, accepts_off):
    if accepts_off:
        source = SOURCE.replace(
            "Forms__On =>\n                    return k == 1\n                Forms__Off =>",
            "Forms__Off =>\n                    return k == 1\n                Forms__On =>",
            1,
        )
        checked = load_source(source + "\nexport scalar\nexport packed\n").qiec
    wire = dumps(checked)
    module = loads(wire)
    assert isinstance(module, QiecModule) and dumps(module) == wire
    _, chart, guard = _term_guard(module)
    assert chart.item_type == guard.item_type
    assert all(
        rule.side_condition.arguments == ("I", "K", "F", "J")
        for rule in chart.presentation.rules
        if rule.side_condition is not None
    )
    # One split pairs each of the three lexical forms with "y". The
    # nested Case chooses exactly one form. Both rules prove the same
    # item. The shape rule has no S premise over a proper prefix.
    logs = torch.tensor([-0.4, -0.9, -0.2], dtype=torch.float64, requires_grad=True)
    expected_gradient = torch.zeros_like(logs)
    expected_gradient[int(accepts_off)] = 1
    for _ in range(2):
        for name in ("scalar", "packed"):
            result = run_program(
                module,
                name,
                data={"sentence": ("x", "y")},
                sites={},
                parameters={
                    f"Forms.lex.{i}": value for i, value in enumerate(logs.unbind())
                },
            )
            torch.testing.assert_close(
                result.log_joint, logs[int(accepts_off)] + log(2)
            )
            torch.testing.assert_close(
                torch.autograd.grad(result.log_joint, logs)[0], expected_gradient
            )
            if name == "packed":
                (trace,) = [
                    event.detail
                    for event in result.result.trace
                    if event.event == "deduction.outcomes"
                ]
                assert trace["checked_rule_guards"] == 6
                assert dict(trace["construction_work"])["provenance_records"] == 10
        with torch.no_grad():
            logs.add_(0.125 * expected_gradient)


def test_term_guard_requires_exact_body_and_constructor_capture_after_cache(checked):
    validate_module(checked)
    original, chart, guard = _term_guard(checked)
    box = next(item for item in guard.item_constructors if item.name == "Forms__Box")
    (field,) = box.fields
    different_capture = guard.with_(
        item_constructors=tuple(
            item.with_(fields=(field.with_(name="renamed"),)) if item == box else item
            for item in guard.item_constructors
        )
    )
    different_body = guard.with_(
        body=Return(value=LiteralValue(value=False, type=BOOL))
    )
    for bad, diagnostic in (
        (different_capture, "exact outer item declarations"),
        (different_body, "outer module's checked computation"),
    ):
        assert bad.id == guard.id and bad.item_type == guard.item_type
        presentation = chart.presentation.with_(conditions=(bad,))
        modified = checked.with_(
            handlers=tuple(
                handler.with_(foreign=chart.with_(presentation=presentation))
                if handler.id == original.id
                else handler
                for handler in checked.handlers
            )
        )
        with pytest.raises(KernelError, match=diagnostic):
            validate_module(modified)
        with pytest.raises((ValueError, KernelError), match=diagnostic):
            restored = loads(dumps(modified))
            assert isinstance(restored, QiecModule)
            validate_module(restored)


def test_same_shape_type_from_another_deduction_is_not_a_guard_argument():
    lo, hi = SOURCE.index("deduction Forms"), SOURCE.index("\ndefine select")
    same_shape = (
        SOURCE[lo:hi]
        .replace("deduction Forms", "deduction Other")
        .replace(" #[guard=select]", "")
    )
    source = (SOURCE[:hi] + same_shape + SOURCE[hi:]).replace(
        "f : Forms__Item", "f : Other__Item", 1
    )
    start, stop = source.index("define select"), source.index("define read_mass")
    source = (
        source[:start]
        + source[start:stop].replace("Forms__", "Other__")
        + source[stop:]
    )
    with pytest.raises(QiecDiagnosticError, match="own deduction's Item type"):
        load_source(source + "\nexport scalar\n")


def test_term_guard_schema_keeps_the_ordered_argument_and_body_requirements(checked):
    compiler = Compiler(parse(SOURCE + "\nexport scalar\n"))
    compiler.compile()
    schema = extract_deduction_schema(compiler)
    constraints = schema.constraints_for("deduction:Forms/rule:first")
    _, _, guard = _term_guard(checked)
    assert [item.value for item in constraints if item.sort == "guard-computation"] == [
        str(guard.id)
    ]
    assert {item.value for item in constraints if item.sort == "guard-argument"} == {
        "0:I",
        "1:K",
        "2:F",
        "3:J",
    }


def test_term_guard_may_not_recurse_through_a_named_call():
    source = SOURCE.replace("return k == 1", "select(i, k, f, j)", 1)
    with pytest.raises(QiecDiagnosticError, match="without named calls or handlers"):
        load_source(source + "\nexport scalar\n")


def test_term_guard_refusal_is_specific_to_reached_checked_entry(checked):
    (item,) = (
        diagnosis
        for diagnosis in outcome_capability_diagnostics(checked, "stan")
        if diagnosis.computation == "scalar"
    )
    assert item.kind == "qiec:capability:deduction-rule-guard:scalar"
    assert item.origin is not None
    assert "let chart = parse(" in SOURCE.splitlines()[item.origin.line - 1]
    with pytest.raises(UnsupportedConstruct, match="deduction-rule-guard"):
        transpile(parse(SOURCE + "\nexport scalar\n"), target="stan")
    source = (
        SOURCE
        + """
program independent : Token -> Result
    sample x <- Normal(loc=0.0, scale=1.0)
    return x
export independent
"""
    )
    assert b"normal" in transpile(parse(source), target="stan")
