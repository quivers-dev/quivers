"""Count matched-term conversion separately from lexical decoding and refits."""

from math import log

import pytest
import torch

from quivers.dsl import loads
from quivers.qiec import QiecModule, loads as load_checked, run_program
from quivers.qiec._deduction_guards import _TermGuard
from quivers.qiec.contracts import Phase, check_complexity
from quivers.qiec.deduction_outcomes import _ground
from quivers.stochastic.planning import PlanRecord
from tests.evidence.counters import CallCounter


def _source(entries: int, depth: int) -> str:
    """Give two structurally different forms the same lexical multiplicity."""
    child = "Leaf"
    for _ in range(depth - 1):
        child = f"Keep({child})"
    forms = (f"Keep({child})", f"Drop({child})")
    lexical = "\n".join(
        f'        "x" : A = {forms[index % 2]} #[learnable]' for index in range(entries)
    )
    return f"""\
object Token : FinSet 2
object Result : Real 1

deduction Cost : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, Keep, Drop, Leaf, plain
    rule first : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain) #[guard=admit]
    rule second : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain) #[guard=admit]
    lexicon
{lexical}
        "y" : B = plain

define admit(i : Int, k : Int, form : Cost__Item, j : Int) : Bool !{{}} =
    case form motive => Bool
        Cost__Keep(child) =>
            return k == 1
        Cost__Drop(child) =>
            return false
        Cost__Leaf =>
            return false
        Cost__A =>
            return false
        Cost__B =>
            return false
        Cost__S =>
            return false
        Cost__span(left, right, category, meaning) =>
            return false
        Cost__plain =>
            return false

define mass(rows : Cost__Outcomes) : Real !{{}} =
    case rows motive => Real
        Cost__outcome_end =>
            return log(0.0)
        Cost__outcome_entry(key, weight, tail) =>
            return weight

program fit : Token -> Result
    let rows = parse_outcomes(Cost, sentence, "item")
    let value <- mass(rows)
    score response = value
    return response

export fit
"""


@pytest.mark.parametrize(("entries", "depth"), [(4, 1), (4, 8), (16, 1), (16, 8)])
def test_guard_term_conversion_depends_on_distinct_matches_and_term_size(
    entries: int, depth: int
) -> None:
    module = loads(_source(entries, depth)).qiec
    assert isinstance(module, QiecModule)
    logits = (-torch.arange(1, entries + 1, dtype=torch.float64) / 16).requires_grad_()
    for _ in range(2):
        # All occurrences remain probabilistic alternatives. The helper
        # selects the even-numbered rows by the outer constructor alone.
        expected = logits.detach()[::2].logsumexp(0) + log(2.0)
        expected_gradient = torch.zeros_like(logits)
        expected_gradient[::2] = logits.detach()[::2].softmax(0)
        parameters = {
            f"Cost.lex.{i}": weight for i, weight in enumerate(logits.unbind())
        }
        with CallCounter(
            {"lexical_decode": _ground, "guard_reification": _TermGuard.ground_item}
        ) as counter:
            result = run_program(
                module,
                "fit",
                data={"sentence": ("x", "y")},
                sites={},
                parameters=parameters,
            )

        # Count one decode per term node and category per lexical row,
        # including two nullary nodes for "y". At each of the two rules
        # there are just two distinct complete matches. Each reconstructs
        # one d+1-node form. A shallow Case still receives the whole form.
        assert counter.counts == {
            "lexical_decode": entries * (depth + 2) + 2,
            "guard_reification": 4 * (depth + 1),
        }
        (trace,) = [
            event.detail
            for event in result.result.trace
            if event.event == "deduction.outcomes"
        ]
        assert trace["checked_rule_guards"] == 4
        (report,) = [
            event.detail
            for event in result.result.trace
            if event.event == "deduction.outcomes.plan"
        ]
        assert report["checked_rule_guards"] == 4
        record = load_checked(report["plan"])
        assert isinstance(record, PlanRecord)
        check_complexity(record.complexity)
        assert {
            bound.phase
            for bound in record.complexity.bounds
            if "P_g X_g + S_g U_g" in bound.expression
        } == {Phase.CONSTRUCTION, Phase.REFIT}
        # Each lexical occurrence keeps its edge. The two accepted binary
        # applications keep an edge and two ordered premise references each.
        assert dict(trace["construction_work"])["provenance_records"] == entries + 7
        torch.testing.assert_close(result.log_joint, expected, rtol=1e-12, atol=1e-12)
        (gradient,) = torch.autograd.grad(result.log_joint, logits)
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
        with torch.no_grad():
            logits.add_(gradient, alpha=0.1)
