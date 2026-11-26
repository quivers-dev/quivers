"""Every law a member of the algebra hierarchy claims holds, with its status."""

from __future__ import annotations

import inspect

import pytest
import torch

from quivers.core import algebras
from quivers.core.algebras import LAWS, Lawful
from quivers.stochastic import semiring as chart
from tests.evidence.compare import Mismatch
from tests.evidence.laws import (
    ASSUMED,
    CELLS,
    DTYPES,
    LAW_CHECKS,
    LAW_MUTATIONS,
    LAWLESS,
    MEMBERS,
    ROWS,
    cell,
    claimed,
    run,
)
from tests.evidence.mutations import Mutation, installed


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", CELLS)
def test_cell(name: str, dtype: torch.dtype) -> None:
    """Require one claimed law, or the declared operations, of one row.

    Parameters
    ----------
    name : str
        ``"<row>/<law>"``.
    dtype : torch.dtype
        The precision.
    """
    member, law = cell(name)
    run(member, law, dtype)


@pytest.mark.parametrize("row", ROWS)
def test_claims_have_statuses(row: str) -> None:
    """Require every claim of a shipped row to be proved or checked.

    The custom rows are the exception, and their laws about the carrier
    have to be reported as assumed.

    Parameters
    ----------
    row : str
        The row.
    """
    structure = MEMBERS[row].structure
    for claim in structure.claims():
        assert claim.member == structure.name
        assert claim.evidence
        if row in ASSUMED and claim.law.subject == "carrier":
            assert claim.status == "assumed", claim
        else:
            assert claim.status in ("proved", "checked"), claim


def test_every_law_has_a_check() -> None:
    """Require a check for every law of the catalogue."""
    assert set(LAW_CHECKS) == set(LAWS)


def _concrete_classes() -> set[type]:
    """The concrete members of the hierarchy that the library defines.

    Returns
    -------
    set of type
        Every class of the algebra and chart modules that claims laws and
        can be built.
    """
    found: set[type] = set()
    for module in (algebras, chart):
        for _, value in inspect.getmembers(module, inspect.isclass):
            if (
                value.__module__ == module.__name__
                and issubclass(value, Lawful)
                and not inspect.isabstract(value)
                and value.laws()
            ):
                found.add(value)
    return found


def test_every_concrete_class_has_a_row() -> None:
    """Require a row for every class of the hierarchy that claims a law."""
    rows = {type(member.structure) for member in MEMBERS.values()}
    assert _concrete_classes() <= rows, sorted(
        cls.__name__ for cls in _concrete_classes() - rows
    )


@pytest.mark.parametrize("law", sorted(LAWS))
def test_a_lawless_member_is_rejected(law: str) -> None:
    """Require the check of each law to reject a member that breaks it.

    Parameters
    ----------
    law : str
        The law.
    """
    member = LAWLESS[law]
    assert law in claimed(member), member.name
    with pytest.raises(Mismatch):
        run(member, law)


def test_every_law_has_a_lawless_member() -> None:
    """Require a deliberately lawless member for every law."""
    assert set(LAWLESS) == set(LAWS)


MUTATED = [
    pytest.param(mutation, name, id=f"{mutation.name}-{name}")
    for mutation in LAW_MUTATIONS
    for name in mutation.rejected_by
]


@pytest.mark.parametrize(("mutation", "name"), MUTATED)
def test_law_mutation_is_rejected(mutation: Mutation, name: str) -> None:
    """Require a cell to pass, and to reject wrong algebra code.

    Parameters
    ----------
    mutation : Mutation
        The wrong code.
    name : str
        The cell that has to reject it.
    """
    assert name in CELLS, name
    member, law = cell(name)
    run(member, law)
    with installed(mutation):
        with pytest.raises(Mismatch) as caught:
            run(member, law)
    assert caught.value.deviation >= mutation.min_effect, (
        f"{mutation.name} moved {name} by {caught.value.deviation}, recorded at "
        f"least {mutation.min_effect}"
    )
