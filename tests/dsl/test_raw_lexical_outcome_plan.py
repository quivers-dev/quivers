"""The typed plan describes the native raw item query that actually ran."""

import math

import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.dsl import loads as load_source
from quivers.qiec import QiecModule, dumps, loads, run_program
from quivers.qiec import deduction_outcomes
from quivers.qiec.canonical import tensor_type
from quivers.qiec.contracts import (
    Acyclic,
    DerivativeSemantics,
    Exact,
    NoRetention,
    Phase,
    ResourceBudget,
    Weight,
    check_admissible,
    check_complexity,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import RuntimeConfiguration, RuntimeSelection, run_named
from quivers.qiec.foreign import DeductionOutcomes
from quivers.qiec.kinds import NAT
from quivers.qiec.program_runtime import (
    PARAMS_HANDLER,
    foreign_handler_configuration,
    joint_module,
    program_entry,
    static_arguments_for,
)
from quivers.qiec.types import BOOL, REAL, STRING, IndexLiteral
from quivers.stochastic.planning import PlanRecord
from tests.dsl.test_deduction_keys import GRAMMAR as TYPED_KEY_SOURCE
from tests.dsl.test_deduction_outcomes import GRAMMAR
from tests.dsl.test_term_rule_guards import SOURCE as GUARDED_SOURCE
from tests.evidence.counters import CallCounter

SOURCE = (
    GRAMMAR.replace("[semiring=LogProb]", "[semiring=LogProb, start=S]")
    .replace("span(I, J, T, same)", "span(I, J, S, same)")
    .replace(
        "    lexicon",
        "    rule ignored : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, T, inverse)\n"
        "    lexicon",
    )
    + """
program listed : Token -> Result
    let rows = parse_outcomes(Mix, sentence, "item")
    return rows

program normalized : Token -> Result
    let rows = parse_normalized_outcomes(Mix, sentence, "item")
    return rows

program forms : Token -> Result
    let rows = parse_outcomes(Mix, sentence, "form")
    return rows

export listed
export normalized
export forms
"""
)
DATA = {"sentence": ("x", "y")}


@pytest.fixture(scope="module")
def checked() -> QiecModule:
    original = load_source(SOURCE).qiec
    module = loads(dumps(original))
    assert isinstance(module, QiecModule) and module == original
    return module


def test_raw_item_plan_keeps_the_checked_identity_and_matches_live_mass(checked):
    (handler,) = (
        value
        for value in checked.handlers
        if isinstance(value.foreign, DeductionOutcomes)
    )
    descriptor = handler.foreign
    interface = next(
        effect for effect in checked.effects if effect.ref == handler.effect
    )
    (operation,) = interface.operations
    weights = torch.tensor([-0.1, -1.2, -0.7], dtype=torch.float64, requires_grad=True)
    parameters = dict(
        zip(("Mix.lex.0", "Mix.lex.1", "Mix.lex.2"), weights.unbind(), strict=True)
    )
    with CallCounter(
        {
            "packed_chart": deduction_outcomes.build_forest,
            "packed_fold": deduction_outcomes.forest_folds,
            "unfold": deduction_outcomes.fold_derivations_many,
        }
    ) as work:
        result = run_program(
            checked, "listed", data=DATA, sites={}, parameters=parameters
        )
    assert work.counts == {"packed_chart": 1, "packed_fold": 1, "unfold": 0}
    names = dict(descriptor.item_constructors)
    row = result.value
    masses = {}
    for _ in range(2):
        assert (
            isinstance(row, RuntimeConstructor) and row.constructor == descriptor.entry
        )
        item, weight, row = row.fields
        assert (
            isinstance(item, RuntimeConstructor) and item.constructor == names["span"]
        )
        left, right, category, form = item.fields
        assert (int(left), int(right)) == (0, 2)
        assert (
            isinstance(category, RuntimeConstructor)
            and category.constructor == names["S"]
        )
        assert isinstance(form, RuntimeConstructor)
        assert isinstance(weight, torch.Tensor)
        masses[form.constructor] = weight
    assert isinstance(row, RuntimeConstructor) and row.constructor == descriptor.empty
    assert len(masses) == 2
    wanted = (weights[:2].logsumexp(0), weights[2])
    torch.testing.assert_close(masses[names["same"]], wanted[0])
    torch.testing.assert_close(masses[names["inverse"]], wanted[1])
    actual = (
        masses[names["same"]].exp() * 0.7 + masses[names["inverse"]].exp() * 0.2
    ).log()
    reference = (weights + weights.new_tensor([0.7, 0.7, 0.2]).log()).logsumexp(0)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual, weights)[0],
        torch.autograd.grad(reference, weights)[0],
    )

    (original,) = [
        event for event in result.result.trace if event.event == "deduction.outcomes"
    ]
    assert set(original.detail) == {
        "presentation",
        "normalized",
        "recursion",
        "algebra",
        "derivative",
        "retention",
        "goal_items",
        "positive_rows",
        "construction_work",
        "observer",
        "key",
        "algorithm",
        "unmeasured",
    }
    assert original.detail["goal_items"] == original.detail["positive_rows"] == 2
    assert dict(original.detail["construction_work"])["provenance_records"] == 16

    (event,) = [
        entry
        for entry in result.result.trace
        if entry.event == "deduction.outcomes.plan"
    ]
    assert original.sequence < event.sequence
    detail = event.detail
    record = loads(detail["plan"])
    assert isinstance(record, PlanRecord) and dumps(record) == detail["plan"]
    assert detail["handler"] == str(handler.id) and detail["operation"] == str(
        operation.id
    )
    assert detail["presentation"] == str(descriptor.presentation.id)
    assert loads(detail["output_type"]) == handler.output_type
    assert detail["selector"] == "item" and detail["normalized"] is False
    assert detail["sentence_length"] == 2 and detail["start"] == "S"
    assert dict(detail["span_order"]) == {
        "equal_width_height": 0,
        "maximum_arity": 2,
        "nullable": False,
    }
    assert detail["temporary_provenance"] == "packed-forest"
    assert loads(detail["derivative"]) == DerivativeSemantics()
    assert detail["derivative_inputs"] == "ordered finite log weights at fixed support"
    assert record.query.presentation == descriptor.presentation.id
    assert record.structure == descriptor.presentation.analysis
    assert record.query.observable == Weight()
    assert record.query.input_types == (
        tensor_type(STRING, (IndexLiteral(value=2, sort=NAT),)),
        tensor_type(descriptor.item_type, (IndexLiteral(value=8, sort=NAT),)),
        tensor_type(REAL, (IndexLiteral(value=4, sort=NAT),)),
        STRING,
        BOOL,
    )
    assert record.query.retention == record.retention == NoRetention()
    assert record.recursion == Acyclic() and record.backward == "autograd"
    assert record.algebra == LOG_PROB.subject and record.claims == LOG_PROB.claims()
    assert (
        record.accuracy.numerics.dtype == "float64"
        and record.accuracy.numerics.domain == "log"
    )
    assert record.accuracy.numerics.rounding is None
    assert (
        record.accuracy.guarantee == Exact()
        and record.accuracy.budget == ResourceBudget()
    )
    assert record.algorithm == original.detail["algorithm"]
    assert tuple(join.rule for join in record.joins) == tuple(
        rule.id for rule in descriptor.presentation.rules
    )
    assert all(
        join.algorithm == "generic join" and join.rho.fraction == 2
        for join in record.joins
    )
    check_admissible(
        record.query,
        record.recursion,
        record.accuracy,
        algebra=record.algebra,
        claims=record.claims,
    )
    check_complexity(record.complexity)
    assert {
        (bound.phase, bound.resource): bound.expression
        for bound in record.complexity.bounds
    }[Phase.DIFFERENTIATION, "time"] == "(A + 1) (W + 1) + V"
    assert event.to_data()["detail"]["plan"] == detail["plan"]


def test_plan_record_is_scoped_and_skipped_when_not_traced(checked, monkeypatch):
    for name in ("forms", "normalized"):
        result = run_program(checked, name, data=DATA, sites={})
        events = tuple(event.event for event in result.result.trace)
        assert "deduction.outcomes" in events
        assert "deduction.outcomes.plan" not in events

    guarded = load_source(GUARDED_SOURCE + "\nexport packed\n").qiec
    result = run_program(guarded, "packed", data=DATA, sites={})
    (ordinary,) = [
        event for event in result.result.trace if event.event == "deduction.outcomes"
    ]
    assert ordinary.detail["key"] == "item"
    assert ordinary.detail["normalized"] is False
    assert ordinary.detail["checked_rule_guards"] == 6
    assert all(
        event.event != "deduction.outcomes.plan" for event in result.result.trace
    )

    def forbidden(*args, **kwargs):
        pytest.fail("an untraced result must not construct plan metadata")

    monkeypatch.setattr(deduction_outcomes, "raw_item_plan_trace", forbidden)
    extended, entry_name = joint_module(checked, "listed")
    entry = program_entry(checked, "listed")
    handlers = foreign_handler_configuration(checked)
    handlers[PARAMS_HANDLER] = {"kind": "param", "values": {}}
    runtime = RuntimeConfiguration(
        selections=(RuntimeSelection(provider="core", options={"handlers": handlers}),)
    )
    plain = run_named(
        extended,
        entry_name,
        tuple(DATA[parameter.name] for parameter in entry.parameters),
        static_arguments=static_arguments_for(entry, DATA, ()),
        runtime=runtime,
        record_trace=False,
    )
    assert plain.trace == ()
    returned, *_ = plain.value
    assert isinstance(returned, RuntimeConstructor)
    _, weight, _ = returned.fields
    assert float(weight) in (0, math.log(2))


def test_raw_plan_covers_only_the_item_stage_of_a_typed_key_query():
    checked = load_source(TYPED_KEY_SOURCE + "\nexport normalized\n").qiec
    result = run_program(
        checked,
        "normalized",
        data={**DATA, "yes": 0.8, "no": 0.3},
        sites={},
    )
    (item_plan,) = [
        event
        for event in result.result.trace
        if event.event == "deduction.outcomes.plan"
    ]
    (grouping,) = [
        event
        for event in result.result.trace
        if event.event == "deduction.grouped_outcomes"
    ]
    assert item_plan.sequence < grouping.sequence
    assert item_plan.detail["selector"] == "item"
    assert item_plan.detail["normalized"] is False
    assert grouping.detail["normalized"] is True
    assert grouping.detail["input_rows"] == 4
    assert grouping.detail["distinct_keys"] == 3
