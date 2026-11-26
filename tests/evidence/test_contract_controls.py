"""Every wrong version of the contract records is rejected by its checks."""

from __future__ import annotations

import pytest

from quivers.qiec.contracts import RefusalCode
from tests.evidence.compare import Mismatch
from tests.evidence.contract_controls import (
    CHECKS,
    CLAIM_CASES,
    CONTRACT_CASES,
    MUTATIONS,
    REPORT_CASES,
)
from tests.evidence.mutations import Mutation, installed

CASES = [
    pytest.param(mutation, check, id=f"{mutation.name}-{check}")
    for mutation in MUTATIONS
    for check in mutation.rejected_by
]


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_check_passes(name: str) -> None:
    """Require every check of the contract records to pass.

    Parameters
    ----------
    name : str
        The check.
    """
    CHECKS[name].run()


@pytest.mark.parametrize(("mutation", "check"), CASES)
def test_mutation_is_rejected(mutation: Mutation, check: str) -> None:
    """Require a check to pass, and to reject the mutated library.

    Parameters
    ----------
    mutation : Mutation
        The wrong version.
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


def test_every_refusal_has_a_case_and_a_mutation() -> None:
    """Require a check and a removing mutation for every kind of refusal."""
    cases = {*CONTRACT_CASES, *REPORT_CASES, *CLAIM_CASES}
    assert cases == set(RefusalCode)
    removed = {
        mutation.name.removeprefix("refusal-").removesuffix("-removed")
        for mutation in MUTATIONS
        if mutation.defect_class == "refusal"
    }
    assert removed == {code.value for code in RefusalCode}


def test_every_mutation_names_registered_checks() -> None:
    """Require every mutation to be rejected by at least one registered check."""
    for mutation in MUTATIONS:
        assert mutation.rejected_by, mutation.name
        assert set(mutation.rejected_by) <= set(CHECKS), mutation.name
        assert mutation.min_effect > 0, mutation.name


def test_the_status_and_serialization_classes_are_covered() -> None:
    """Require mutations of a cap or residual reported as a bound and of lost identities."""
    classes = {mutation.defect_class for mutation in MUTATIONS}
    assert classes >= {"refusal", "status", "serialization"}
