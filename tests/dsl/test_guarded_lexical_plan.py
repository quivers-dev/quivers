"""The packed query reports its exact guards and weights the admitted matches."""

import math

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.qiec import DeductionOutcomes, QiecModule, dumps, loads, run_program
from quivers.qiec._deduction_guards import _TermGuard
from quivers.qiec.contracts import NormalizedWeightRows, ProjectedWeightRows, Weight
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.kinds import NAT, UserIndexSort
from quivers.qiec.types import INT
from quivers.stochastic.planning import PlanRecord
from tests.dsl.test_term_rule_guards import SOURCE


PROGRAM = """
program raw : Token -> Result
    let rows = parse_outcomes(Forms, sentence, "{selector}")
    return rows

program normalized : Token -> Result
    let rows = parse_normalized_outcomes(Forms, sentence, "{selector}")
    return rows

export raw
export normalized
"""
DATA = {"sentence": ("x", "y")}


def _source(selector, accepts_off):
    text = SOURCE.replace(", start=S]", "]").replace(
        "    lexicon",
        "    rule alternate : span(I, K, A, On), span(K, J, B, plain) "
        "|- span(I, J, B, On)\n    lexicon",
    )
    if accepts_off:
        text = text.replace(
            "Forms__On =>\n                    return k == 1\n                Forms__Off =>",
            "Forms__Off =>\n                    return k == 1\n                Forms__On =>",
            1,
        )
    return text + PROGRAM.format(selector=selector)


@pytest.mark.parametrize(
    ("selector", "position"), (("item", None), ("category", 2), ("form", 3))
)
def test_guard_plan_attaches_actual_body_and_counts_to_raw_and_normalized_rows(
    selector, position
):
    reports = []
    for accepts_off in (False, True):
        wire = dumps(load_source(_source(selector, accepts_off)).qiec)
        module = loads(wire)
        assert isinstance(module, QiecModule) and dumps(module) == wire
        (chart,) = (
            handler.foreign
            for handler in module.handlers
            if isinstance(handler.foreign, DeductionOutcomes)
        )
        presentation = chart.presentation
        (span,) = presentation.judgments
        (condition,) = presentation.conditions
        assert (
            isinstance(condition, _TermGuard) and condition.item_type == chart.item_type
        )
        source = (
            Weight()
            if position is None
            else ProjectedWeightRows(
                judgment=span.id, position=position, field=span.indices[position]
            )
        )
        constructors = dict(chart.item_constructors)
        names = ("S", "B") if selector == "category" else ("plain", "On")
        scores = torch.tensor(
            (-0.4, -0.9, -0.2), dtype=torch.float64, requires_grad=True
        )
        parameters = {
            f"Forms.lex.{i}": value for i, value in enumerate(scores.unbind())
        }
        for normalized in (False, True):
            run = run_program(
                module,
                "normalized" if normalized else "raw",
                data=DATA,
                sites={},
                parameters=parameters,
            )
            row = run.value
            weights = {}
            for _ in range(2):
                assert (
                    isinstance(row, RuntimeConstructor)
                    and row.constructor == chart.entry
                )
                key, mass, row = row.fields
                if position is None:
                    assert key.constructor == constructors["span"]
                    key = key.fields[3]
                assert key.constructor not in weights
                weights[key.constructor] = mass
            assert (
                isinstance(row, RuntimeConstructor) and row.constructor == chart.empty
            )
            actual = torch.stack(tuple(weights[constructors[name]] for name in names))

            # The matching Box input has two rule applications. The atom On
            # only goes through alternate, which produces the other key.
            chosen = int(accepts_off)
            raw = torch.stack((scores[chosen] + math.log(2), scores[2]))
            jacobian = scores.new_zeros((2, 3))
            jacobian[0, chosen] = jacobian[1, 2] = 1
            if normalized:
                jacobian = jacobian - raw.detach().softmax(0) @ jacobian
                raw = raw - raw.logsumexp(0)
            torch.testing.assert_close(actual, raw)
            torch.testing.assert_close(
                torch.stack(
                    tuple(
                        torch.autograd.grad(weight, scores, retain_graph=True)[0]
                        for weight in actual
                    )
                ),
                jacobian,
            )

            ordinary, report = (
                event
                for event in run.result.trace
                if event.event in {"deduction.outcomes", "deduction.outcomes.plan"}
            )
            assert (
                ordinary.event == "deduction.outcomes"
                and report.event == "deduction.outcomes.plan"
            )
            assert (
                ordinary.detail["checked_rule_guards"]
                == report.detail["checked_rule_guards"]
                == 6
            )
            assert ordinary.detail["positive_rows"] == 2
            assert (
                dict(ordinary.detail["construction_work"])["provenance_records"] == 13
            )
            bindings, bodies = loads(report.detail["guard_bindings"])
            assert dumps((bindings, bodies)) == report.detail["guard_bindings"]
            assert bodies == (condition,)
            guarded = tuple(
                rule for rule in presentation.rules if rule.side_condition is not None
            )
            assert bindings == tuple(
                (rule.id, rule.side_condition, rule.telescope) for rule in guarded
            )
            assert (
                len(bindings) == 2
                and len({guard.computation for _, guard, _ in bindings}) == 1
            )
            for _, _, (i, k, f, j) in bindings:
                assert (i.sort, k.sort, j.sort) == (NAT, NAT, NAT)
                assert isinstance(f.sort, UserIndexSort)
            assert tuple(parameter.type for parameter in condition.parameters) == (
                INT,
                INT,
                chart.item_type,
                INT,
            )
            plan = loads(report.detail["plan"])
            assert isinstance(plan, PlanRecord) and dumps(plan) == report.detail["plan"]
            assert plan.query.presentation == presentation.id
            assert plan.query.observable == (
                NormalizedWeightRows(source=source) if normalized else source
            )
            assert ("positive_total_mass" in report.detail) is normalized
        reports.append((condition, report.detail["guard_bindings"]))

    (first, first_wire), (second, second_wire) = reports
    assert first.id == second.id and first.item_type == second.item_type
    assert first.body != second.body and first_wire != second_wire


def test_zero_mass_guarded_normalization_refuses_before_emitting_success():
    module = load_source(_source("form", False)).qiec
    seen = []
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        run_program(
            module,
            "normalized",
            data=DATA,
            sites={},
            parameters={f"Forms.lex.{i}": -math.inf for i in range(3)},
            observer=seen.append,
        )
    assert all(
        event.event not in {"deduction.outcomes", "deduction.outcomes.plan"}
        for event in seen
    )
