"""A packed Boolean plan describes the actual ordered truth table and fold."""

import math

import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.laws import DIVISION
from quivers.dsl import loads as load_source
from quivers.qiec import DeductionBooleanOutcomes, QiecModule, dumps, loads, run_program
from quivers.qiec.canonical import tensor_type
from quivers.qiec.contracts import (
    BooleanProofWeightRows,
    InadmissibleContract,
    InterpretationId,
    NoRetention,
    Phase,
    check_admissible,
    check_complexity,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.kinds import NAT
from quivers.qiec.types import BOOL, IndexLiteral
from quivers.stochastic.planning import PlanRecord, PlanRefusal, PlanRefused, plan
from tests.dsl.test_packed_boolean_outcomes import SOURCE


PROGRAM = """
define projection(left : Bool, right : Bool) : Bool !{} =
    return {argument}

program raw_rows : Token -> Result
    let rows = parse_packed_boolean_outcomes(Parity, sentence, parity)
    return rows

program normalized_rows : Token -> Result
    let rows = parse_normalized_packed_boolean_outcomes(Parity, sentence, parity)
    return rows

export raw_rows
export normalized_rows
"""


def _source(argument):
    return SOURCE.replace("rule join = xor", "rule join = projection").replace(
        "\ninterpret parity : Parity -> Bool",
        '\n        "y" : A = word #[learnable]\n        "y" : B = word #[learnable]\n'
        "\ninterpret parity : Parity -> Bool",
    ) + PROGRAM.replace("{argument}", argument)


@pytest.mark.parametrize(("argument", "position"), (("left", 0), ("right", 1)))
def test_live_asymmetric_table_has_the_recorded_meaning_lanes(argument, position):
    source = load_source(_source(argument)).qiec
    wire = dumps(source)
    module = loads(wire)
    assert isinstance(module, QiecModule) and dumps(module) == wire
    (handler,) = tuple(
        handler
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionBooleanOutcomes)
    )
    chart = handler.foreign.chart
    presentation = chart.presentation
    effect = next(effect for effect in module.effects if effect.ref == handler.effect)
    (operation,) = effect.operations
    weights = torch.tensor(
        tuple(map(math.log, (0.1, 0.4, 0.7, 0.2))),
        dtype=torch.float64,
        requires_grad=True,
    )
    word = weights.reshape(2, 2)
    for normalized in (False, True):
        run = run_program(
            module,
            "normalized_rows" if normalized else "raw_rows",
            data={"sentence": ("x", "y")},
            sites={},
            parameters={
                f"Parity.lex.{i}": value for i, value in enumerate(weights.unbind())
            },
        )
        rows = []
        row = run.value
        for bit in (False, True):
            assert (
                isinstance(row, RuntimeConstructor) and row.constructor == chart.entry
            )
            key, mass, row = row.fields
            assert bool(key) is bit
            rows.append(mass)
        assert isinstance(row, RuntimeConstructor) and row.constructor == chart.empty
        actual = torch.stack(rows)

        # Each position independently picks its A or B lexical occurrence.
        # The table keeps just one position's reading. Summing over the
        # other position contributes that word's entire mass.
        reference = (
            word[position].log_softmax(0)
            if normalized
            else word[position] + word[1 - position].logsumexp(0)
        )
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        jacobian = torch.stack(
            tuple(
                torch.autograd.grad(value, weights, retain_graph=True)[0]
                for value in actual
            )
        )
        expected = weights.new_zeros((2, 2, 2))
        expected[:, position, :] = torch.eye(2, dtype=weights.dtype)
        if normalized:
            expected[:, position, :] -= word[position].detach().softmax(0)
        else:
            expected[:, 1 - position, :] = word[1 - position].detach().softmax(0)
        torch.testing.assert_close(
            jacobian, expected.reshape(2, 4), atol=1e-12, rtol=1e-12
        )

        ordinary, planned = [
            event
            for event in run.result.trace
            if event.event
            in {"deduction.boolean_outcomes", "deduction.boolean_outcomes.plan"}
        ]
        assert (
            ordinary.event == "deduction.boolean_outcomes"
            and planned.event == "deduction.boolean_outcomes.plan"
        )
        report = planned.detail
        assert report["handler"] == str(handler.id) and report["operation"] == str(
            operation.id
        )
        assert report["output_type"] == dumps(handler.output_type)
        assert (
            report["normalized"] is normalized
            and ("positive_total_mass" in report) is normalized
        )
        # Four lexical edges, five unary edges and one binary edge remain
        # after selecting the goal's ancestors. The grammar has two lexical
        # table entries, six entries for unary rules and four for its join.
        for key, count in {
            "evaluated_items": 8,
            "truth_table_entries": 12,
            "edge_meaning_lanes": 18,
            "weight_products": 18,
            "reduced_entries": 18,
        }.items():
            assert report[key] == ordinary.detail[key] == count
        lexical, rules = loads(report["truth_table"])
        assert lexical == (("x", False), ("y", False))
        expected_tables = {
            "no": (False, False),
            "yes": (True, True),
            "base": (False, True),
            "join": (False, False, True, True)
            if position == 0
            else (False, True, False, True),
        }
        assert rules == tuple(
            (rule.id, expected_tables[rule.name]) for rule in presentation.rules
        )
        assert dumps((lexical, rules)) == report["truth_table"]

        record = loads(report["plan"])
        assert isinstance(record, PlanRecord) and dumps(record) == report["plan"]
        assert record.query.presentation == presentation.id
        assert record.query.interpretation == InterpretationId.derive(
            presentation.id, "boolean-proof-table", operation.id
        )
        assert record.query.observable == BooleanProofWeightRows(normalized=normalized)
        assert record.query.input_types[-2:] == (
            tensor_type(BOOL, (IndexLiteral(value=12, sort=NAT),)),
            BOOL,
        )
        assert record.retention == NoRetention()
        assert (
            record.algorithm == ordinary.detail["algorithm"]
            and record.backward == "autograd"
        )
        check_admissible(
            record.query,
            record.recursion,
            record.accuracy,
            algebra=record.algebra,
            claims=record.claims,
        )
        claims = tuple(claim for claim in record.claims if claim.law != DIVISION)
        if normalized:
            with pytest.raises(InadmissibleContract, match="division"):
                check_admissible(
                    record.query,
                    record.recursion,
                    record.accuracy,
                    algebra=record.algebra,
                    claims=claims,
                )
        else:
            check_admissible(
                record.query,
                record.recursion,
                record.accuracy,
                algebra=record.algebra,
                claims=claims,
            )
        check_complexity(record.complexity)
        time = {
            bound.phase: bound.expression
            for bound in record.complexity.bounds
            if bound.resource == "time"
        }
        assert time[Phase.CONSTRUCTION].endswith(" + F")
        assert time[Phase.EXECUTION] == "V + A + K + L + M" + (
            " + H" if normalized else ""
        )
        assert "W + 1" in time[Phase.DIFFERENTIATION]
        with pytest.raises(PlanRefused) as refused:
            plan(
                record.query,
                presentation,
                LOG_PROB,
                record.recursion,
                record.accuracy,
                record.retention,
            )
        assert refused.value.code == PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR


def test_no_completed_plan_when_boolean_normalization_refuses():
    module = load_source(_source("left")).qiec
    observed = []
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        run_program(
            module,
            "normalized_rows",
            data={"sentence": ("missing", "y")},
            sites={},
            observer=observed.append,
        )
    assert all(
        event.event
        not in {"deduction.boolean_outcomes", "deduction.boolean_outcomes.plan"}
        for event in observed
    )
