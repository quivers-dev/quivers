"""All three executors compute the same literal acyclic proof sum."""

from __future__ import annotations

import pytest

from quivers.dsl import loads
from quivers.qiec.contracts import WorkField
from tests.evidence.checks.partition import SENTENCE
from tests.evidence.conformance import parameters, run
from tests.evidence.sources import WEIGHTED_LEXICAL, deduction


@pytest.mark.parametrize("name", parameters("partition"))
def test_partition(name: str) -> None:
    run(name)


def test_retained_ground_instances_are_distinct_from_proof_occurrences() -> None:
    """The finite agenda fixture records ground rule instances and axiom positions."""
    system = deduction(loads(WEIGHTED_LEXICAL.replace("SEMIRING", "LogProb")), "G")
    result = system.run(SENTENCE)
    assert result.axiom_occurrences == [
        ("span", 0, 1, ("atom", "S")),
        ("span", 0, 1, ("atom", "V")),
        ("span", 1, 2, ("atom", "V")),
        ("span", 2, 3, ("atom", "S")),
        ("span", 2, 3, ("atom", "V")),
    ]
    assert sum(rule == "lift" for _, rule, _, _ in result.applications) == 3
    assert sum(rule == "join" for _, rule, _, _ in result.applications) == 4
    assert WorkField.PROVENANCE_RECORDS in result.work.measured
    # One record per axiom occurrence, and one per rule instance with
    # one more for each ordered premise occurrence within that instance.
    assert result.work.provenance_records == 5 + 3 * 2 + 4 * 3
