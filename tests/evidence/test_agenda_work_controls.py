"""Every wrong version of agenda work reports is rejected by its checks."""

from __future__ import annotations

import pytest

from tests.evidence.compare import Mismatch
from tests.evidence.mutations import Mutation, installed
from tests.evidence.agenda_work_controls import CHECKS, MUTATIONS

CASES = [
    pytest.param(mutation, check, id=f"{mutation.name}-{check}")
    for mutation in MUTATIONS
    for check in mutation.rejected_by
]


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_check_passes(name: str) -> None:
    """Require every check of agenda work reports to pass.

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
