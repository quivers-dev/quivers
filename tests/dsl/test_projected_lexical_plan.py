"""Built-in outcome projections report the judgment coordinate actually reduced."""

import math

import pytest
import torch

from quivers.core.laws import DIVISION
from quivers.dsl import loads as load_source
from quivers.qiec import QiecModule, dumps, loads, run_program
from quivers.qiec.contracts import (
    InadmissibleContract,
    NormalizedWeightRows,
    Phase,
    ProjectedWeightRows,
    check_admissible,
    check_complexity,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.foreign import DeductionOutcomes
from quivers.stochastic.planning import PlanRecord
from tests.dsl.test_deduction_outcomes import GRAMMAR

PROGRAM = """
program selected : Token -> Result
    let rows = {query}(Mix, sentence, "{selector}")
    return rows

export selected
"""


@pytest.mark.parametrize("normalized", (False, True))
@pytest.mark.parametrize("selector", ("category", "form"))
def test_distinct_checked_indices_give_distinct_partitions_and_jacobians(
    selector, normalized
):
    query = "parse_normalized_outcomes" if normalized else "parse_outcomes"
    module = loads(
        dumps(
            load_source(GRAMMAR + PROGRAM.format(query=query, selector=selector)).qiec
        )
    )
    assert isinstance(module, QiecModule)
    (chart,) = (
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionOutcomes)
    )
    (span,) = chart.presentation.judgments
    logs = torch.tensor(
        [math.log(1 / 8), math.log(2 / 8), math.log(4 / 8)],
        dtype=torch.float64,
        requires_grad=True,
    )
    run = run_program(
        module,
        "selected",
        data={"sentence": ("x", "y")},
        sites={},
        parameters={f"Mix.lex.{i}": weight for i, weight in enumerate(logs.unbind())},
    )
    if selector == "category":
        # A and C prove category S; B proves T, at the same endpoints.
        position, names = 2, ("S", "T")
        total = (5, 2)
        raw_jacobian = ((1 / 5, 0, 4 / 5), (0, 1, 0))
    else:
        # A and B prove form same; C proves inverse, at the same endpoints.
        position, names = 3, ("same", "inverse")
        total = (3, 4)
        raw_jacobian = ((1 / 3, 2 / 3, 0), (0, 0, 1))
    constructors = dict(chart.item_constructors)
    row = run.value
    masses = {}
    for _ in range(2):
        assert isinstance(row, RuntimeConstructor)
        key, mass, row = row.fields
        assert isinstance(key, RuntimeConstructor) and isinstance(mass, torch.Tensor)
        assert key.constructor not in masses
        masses[key.constructor] = mass
    assert isinstance(row, RuntimeConstructor) and row.constructor == chart.empty
    actual = torch.stack(tuple(masses[constructors[name]] for name in names))
    torch.testing.assert_close(
        actual.exp(), logs.new_tensor(total) / (7 if normalized else 8)
    )
    gradients = torch.stack(
        tuple(
            torch.autograd.grad(value, logs, retain_graph=True)[0] for value in actual
        )
    )
    expected = logs.new_tensor(raw_jacobian)
    if normalized:
        expected = expected - logs.new_tensor((1 / 7, 2 / 7, 4 / 7))
        torch.testing.assert_close(actual.exp().sum(), logs.new_tensor(1.0))
    torch.testing.assert_close(gradients, expected)

    (original,) = [
        event for event in run.result.trace if event.event == "deduction.outcomes"
    ]
    (event,) = [
        event for event in run.result.trace if event.event == "deduction.outcomes.plan"
    ]
    assert original.sequence < event.sequence
    assert original.detail["key"] == event.detail["selector"] == selector
    assert original.detail["goal_items"] == 3 and original.detail["positive_rows"] == 2
    assert original.detail["normalized"] is event.detail["normalized"] is normalized
    predicate = ProjectedWeightRows(
        judgment=span.id, position=position, field=span.indices[position]
    )
    if normalized:
        assert event.detail["positive_total_mass"] is True
    record = loads(event.detail["plan"])
    assert isinstance(record, PlanRecord) and dumps(record) == event.detail["plan"]
    assert record.query.observable == (
        NormalizedWeightRows(source=predicate) if normalized else predicate
    )
    missing = {
        bound.phase
        for bound in record.complexity.bounds
        if " + G" not in bound.expression
    }
    assert missing == {Phase.CONSTRUCTION}
    assert all(
        bound.expression.endswith(" + H")
        == (normalized and bound.phase != Phase.CONSTRUCTION)
        for bound in record.complexity.bounds
    )
    check_complexity(record.complexity)
    check_admissible(
        record.query,
        record.recursion,
        record.accuracy,
        algebra=record.algebra,
        claims=record.claims,
    )

    without_division = tuple(claim for claim in record.claims if claim.law != DIVISION)
    if normalized:
        with pytest.raises(InadmissibleContract, match="division"):
            check_admissible(
                record.query,
                record.recursion,
                record.accuracy,
                algebra=record.algebra,
                claims=without_division,
            )
    check_admissible(
        record.query.with_(observable=predicate),
        record.recursion,
        record.accuracy,
        algebra=record.algebra,
        claims=without_division,
    )


@pytest.mark.parametrize("selector", ("category", "form"))
def test_zero_total_mass_refuses_before_emitting_the_projected_plan(selector):
    module = load_source(
        GRAMMAR + PROGRAM.format(query="parse_normalized_outcomes", selector=selector)
    ).qiec
    assert isinstance(module, QiecModule)
    seen = []
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        run_program(
            module,
            "selected",
            data={"sentence": ("x", "y")},
            sites={},
            parameters={f"Mix.lex.{i}": -math.inf for i in range(3)},
            observer=seen.append,
        )
    assert all(
        event.event not in ("deduction.outcomes", "deduction.outcomes.plan")
        for event in seen
    )
