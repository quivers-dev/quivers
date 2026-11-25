"""Every deliberately wrong executor is rejected by the checks it names."""

from __future__ import annotations

import pytest

from tests.evidence.checks.all import CHECKS
from tests.evidence.compare import Mismatch
from tests.evidence.mutations import MUTATIONS, MovedAnchor, Mutation, Rewrite, installed
from tests.evidence.mutations import rewrite

CASES = [
    pytest.param(mutation, check, id=f"{mutation.name}-{check}")
    for mutation in MUTATIONS
    for check in mutation.rejected_by
]


@pytest.mark.parametrize(("mutation", "check"), CASES)
def test_mutation_is_rejected(mutation: Mutation, check: str) -> None:
    """Require a check to pass, and to reject the mutated library.

    Parameters
    ----------
    mutation : Mutation
        The wrong executor.
    check : str
        The check that has to reject it.
    """
    CHECKS[check].run()
    with installed(mutation):
        with pytest.raises(Mismatch) as caught:
            CHECKS[check].run()
    assert caught.value.deviation >= mutation.min_effect, (
        f"{mutation.name} moved {check} by {caught.value.deviation}, recorded "
        f"at least {mutation.min_effect}"
    )


def test_every_mutation_names_registered_checks() -> None:
    """Require every mutation to be rejected by at least one registered check."""
    for mutation in MUTATIONS:
        assert mutation.rejected_by, mutation.name
        assert set(mutation.rejected_by) <= set(CHECKS), mutation.name
        assert mutation.min_effect > 0, mutation.name


def test_the_classes_named_by_the_harness_are_covered() -> None:
    """Require a mutation of every defect class the harness has to reject."""
    classes = {mutation.defect_class for mutation in MUTATIONS}
    assert classes >= {
        "repeated premise",
        "normalizer",
        "aggregation",
        "gradient path",
        "effect sharing",
        "stopping rule",
        "budget",
        "omitted alternative",
        "support",
        "multiplicity",
        "graph work",
        "provenance",
        "observer",
        "empty reduction",
        "scaling",
    }


def test_a_moved_anchor_fails() -> None:
    """Require a replacement whose text occurs another number of times to fail."""
    with pytest.raises(MovedAnchor):
        rewrite("a + a", (Rewrite(old="a", new="b"),), "text")
    with pytest.raises(MovedAnchor):
        rewrite("a + a", (Rewrite(old="c", new="b"),), "text")
    assert rewrite("a + a", (Rewrite(old="a", new="b", occurrences=2),), "t") == "b + b"
