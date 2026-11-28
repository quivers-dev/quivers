"""The normalized native plans report the checked operation and its finite Jacobian."""

import math

import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.laws import DIVISION
from quivers.dsl import loads as load_source
from quivers.qiec import QiecModule, dumps, loads, run_program
from quivers.qiec.contracts import (
    Exact,
    InadmissibleContract,
    NormalizedWeightRows,
    Outcome,
    Phase,
    Weight,
    check_admissible,
    check_complexity,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.foreign import DeductionOutcomes
from quivers.stochastic.planning import PlanRecord, PlanRefusal, PlanRefused, plan
from tests.dsl.test_deduction_keys import GRAMMAR as NAMED_GRAMMAR
from tests.dsl.test_raw_lexical_outcome_plan import DATA, SOURCE as ITEM_SOURCE


KEY_SOURCE = (
    NAMED_GRAMMAR
    + """
program normalized_rows : Token -> Result
    let rows = parse_normalized_outcomes(Mix, sentence, classify)
    return rows

export normalized_rows
"""
)


@pytest.fixture(scope="module", params=("item", "key"))
def checked(request):
    source = ITEM_SOURCE if request.param == "item" else KEY_SOURCE
    original = load_source(source).qiec
    copied = loads(dumps(original))
    assert isinstance(copied, QiecModule) and copied == original
    return copied, request.param


def _weights_by_name(module, bare_rows, mode):
    row = bare_rows
    masses = {}
    constructor_names = {decl.id: decl.name for decl in module.constructors}
    for _ in range(2):
        assert isinstance(row, RuntimeConstructor)
        key, weight, row = row.fields
        assert isinstance(key, RuntimeConstructor)
        assert isinstance(weight, torch.Tensor)
        if mode == "item":
            category, form = key.fields[2:]
            assert constructor_names[category.constructor] == "Mix__S"
            name = constructor_names[form.constructor]
        else:
            assert constructor_names[key.constructor] == "Envelope"
            (leaf,) = key.fields
            name = constructor_names[leaf.constructor]
            if name == "Number":
                (number,) = leaf.fields
                assert int(number) == 1
        assert name not in masses
        masses[name] = weight
    assert isinstance(row, RuntimeConstructor) and row.fields == ()
    names = ("Mix__same", "Mix__inverse") if mode == "item" else ("Truth", "Number")
    return torch.stack(tuple(masses[name] for name in names))


def _normalized_entry(mode):
    return "normalized" if mode == "item" else "normalized_rows"


def _normalized_event(mode):
    return "deduction.outcomes" if mode == "item" else "deduction.grouped_outcomes"


def test_normalized_plan_reports_one_actual_divisor_and_exact_jacobian(checked):
    module, mode = checked
    (chart,) = (
        entry.foreign
        for entry in module.handlers
        if isinstance(entry.foreign, DeductionOutcomes)
    )
    if mode == "item":
        # A and B give the same S item. C gives the other S item. The
        # extra T item uses A again but must contribute no denominator mass.
        logs = tuple(math.log(count / 8) for count in (1, 2, 4))
        probabilities = (3 / 7, 4 / 7)
        jacobian = ((4 / 21, 8 / 21, -4 / 7), (-1 / 7, -2 / 7, 3 / 7))
        observer = Weight()
    else:
        # A and B share one item. C is a different item with the same
        # key. D gives another key, and the structural-zero row is absent.
        logs = (*tuple(math.log(count / 16) for count in (1, 2, 4, 8)), -math.inf)
        probabilities = (7 / 15, 8 / 15)
        jacobian = (
            (8 / 105, 16 / 105, 32 / 105, -8 / 15, 0),
            (-1 / 15, -2 / 15, -4 / 15, 7 / 15, 0),
        )
        (key,) = (comp for comp in module.computations if comp.name == "classify")
        observer = Outcome(key=key.id)

    for shift in (0.0, math.log(2)):
        weights = (torch.tensor(logs, dtype=torch.float64) + shift).requires_grad_()
        run = run_program(
            module,
            _normalized_entry(mode),
            data=DATA,
            sites={},
            parameters={
                f"Mix.lex.{index}": weight
                for index, weight in enumerate(weights.unbind())
            },
        )
        output = _weights_by_name(module, run.value, mode)
        torch.testing.assert_close(output.exp(), weights.new_tensor(probabilities))
        torch.testing.assert_close(output.exp().sum(), weights.new_tensor(1.0))
        measured = torch.stack(
            tuple(
                torch.autograd.grad(value, weights, retain_graph=True)[0]
                for value in output
            )
        )
        torch.testing.assert_close(measured, weights.new_tensor(jacobian))
        torch.testing.assert_close(measured.sum(dim=1), torch.zeros_like(output))

        event_name = _normalized_event(mode)
        (ordinary,) = [event for event in run.result.trace if event.event == event_name]
        (record_event,) = [
            event for event in run.result.trace if event.event == f"{event_name}.plan"
        ]
        assert ordinary.sequence < record_event.sequence
        assert (
            ordinary.detail["normalized"] is record_event.detail["normalized"] is True
        )
        assert ordinary.detail["positive_rows"] == 2
        assert record_event.detail["positive_total_mass"] is True
        if mode == "key":
            (item_plan,) = [
                event
                for event in run.result.trace
                if event.event == "deduction.outcomes.plan"
            ]
            assert item_plan.sequence < ordinary.sequence
            assert item_plan.detail["normalized"] is False
            assert ordinary.detail["input_rows"] == 3
        record = loads(record_event.detail["plan"])
        assert (
            isinstance(record, PlanRecord)
            and dumps(record) == record_event.detail["plan"]
        )
        assert record.query.observable == NormalizedWeightRows(source=observer)
        assert record.accuracy.guarantee == Exact()
        check_admissible(
            record.query,
            record.recursion,
            record.accuracy,
            algebra=record.algebra,
            claims=record.claims,
        )
        without_division = tuple(
            claim for claim in record.claims if claim.law != DIVISION
        )
        with pytest.raises(InadmissibleContract, match="division"):
            check_admissible(
                record.query,
                record.recursion,
                record.accuracy,
                algebra=record.algebra,
                claims=without_division,
            )
        check_admissible(
            record.query.with_(observable=observer),
            record.recursion,
            record.accuracy,
            algebra=record.algebra,
            claims=without_division,
        )
        with pytest.raises(PlanRefused) as refused:
            plan(
                record.query,
                chart.presentation,
                LOG_PROB,
                record.recursion,
                record.accuracy,
                record.retention,
            )
        assert refused.value.code == PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR
        check_complexity(record.complexity)
        assert all(
            bound.expression.endswith(" + H") == (bound.phase != Phase.CONSTRUCTION)
            for bound in record.complexity.bounds
        )


def test_zero_mass_refusal_precedes_any_normalized_plan(checked):
    module, mode = checked
    seen = []
    parameters = {f"Mix.lex.{i}": -math.inf for i in range(3 if mode == "item" else 5)}
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        run_program(
            module,
            _normalized_entry(mode),
            data=DATA,
            sites={},
            parameters=parameters,
            observer=seen.append,
        )
    event_name = _normalized_event(mode)
    assert all(event.event not in (event_name, f"{event_name}.plan") for event in seen)


@pytest.mark.parametrize("selector", ("category", "form"))
def test_normalized_builtin_projections_keep_their_ordinary_trace(selector):
    source = ITEM_SOURCE.replace(
        'parse_normalized_outcomes(Mix, sentence, "item")',
        f'parse_normalized_outcomes(Mix, sentence, "{selector}")',
    )
    module = load_source(source).qiec
    assert isinstance(module, QiecModule)
    result = run_program(module, "normalized", data=DATA, sites={})
    (ordinary,) = [
        event for event in result.result.trace if event.event == "deduction.outcomes"
    ]
    assert ordinary.detail["key"] == selector and ordinary.detail["normalized"] is True
    assert all(
        event.event != "deduction.outcomes.plan" for event in result.result.trace
    )
