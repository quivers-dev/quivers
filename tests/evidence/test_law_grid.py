"""Every law a member of the algebra hierarchy claims holds, with its status."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import torch

from quivers.core import algebras
from quivers.core.laws import LAWS, Lawful, LawStatus, Proved, Tested
from quivers.stochastic import semiring as chart
from tests.evidence.compare import Mismatch
from tests.evidence.laws import (
    ACTION_LAWLESS,
    ASSUMED,
    CELLS,
    DTYPES,
    LAW_CHECKS,
    LAW_MUTATIONS,
    LAWLESS,
    MEMBERS,
    Member,
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


DOCS = Path(__file__).resolve().parents[2] / "docs"
"""The documentation, where the arguments of the shipped classes are written."""

DTYPE_FREE = frozenset({"zero-form", "unit-form"})
"""Laws whose check asks for every dtype itself."""


def _anchors(page: Path) -> set[str]:
    """The explicit anchors of a documentation page.

    Parameters
    ----------
    page : Path
        The page.

    Returns
    -------
    set of str
        Every ``{#anchor}`` of a heading.
    """
    return set(re.findall(r"\{#([A-Za-z0-9_-]+)\}", page.read_text()))


def _cases(row: str, law: str) -> int:
    """The number of cases the law grid evaluates for one cell.

    Parameters
    ----------
    row : str
        The row.
    law : str
        A law about the implementation.

    Returns
    -------
    int
        The cases over every precision the cell runs in.
    """
    member = MEMBERS[row]
    if law in DTYPE_FREE:
        found = LAW_CHECKS[law](member, DTYPES[0])
        assert found is not None
        return found
    total = 0
    for dtype in DTYPES:
        found = LAW_CHECKS[law](member, dtype)
        assert found is not None
        total += found
    return total


@pytest.mark.parametrize("row", ROWS)
def test_claims_carry_their_evidence(row: str) -> None:
    """Require the evidence of every claim of a row to be what it states.

    A shipped row proves its laws about the carrier by an argument written
    at an existing anchor of the documentation, and tests its laws about
    the implementation on exactly as many cases as the grid evaluates. The
    custom rows assume their laws about the carrier.

    Parameters
    ----------
    row : str
        The row.
    """
    structure = MEMBERS[row].structure
    for claim in structure.claims():
        assert claim.subject == structure.subject
        if claim.law.scope == "carrier":
            if row in ASSUMED:
                assert claim.status is LawStatus.ASSUMED, claim
                continue
            assert isinstance(claim.evidence, Proved), claim
            assert claim.established
            page, anchor = claim.evidence.argument.split("#")
            assert anchor in _anchors(DOCS / page), claim.evidence.argument
            continue
        assert isinstance(claim.evidence, Tested), claim
        assert not claim.established
        assert claim.evidence.cases == _cases(row, claim.law.name), claim


def test_every_claimed_law_has_a_check() -> None:
    """Require a check for every law some row claims."""
    assert {law for row in ROWS for law in claimed(MEMBERS[row])} <= set(LAW_CHECKS)
    assert set(LAW_CHECKS) <= set(LAWS)


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


@pytest.mark.parametrize("law", sorted(LAW_CHECKS))
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


def test_every_checked_law_has_a_lawless_member() -> None:
    """Require a deliberately lawless member for every law the grid checks."""
    assert set(LAWLESS) == set(LAW_CHECKS)


@pytest.mark.parametrize("member", ACTION_LAWLESS, ids=lambda m: m.name)
def test_each_part_of_the_action_is_checked(member: Member) -> None:
    """Require the action check to reject a module that breaks any one part.

    Parameters
    ----------
    member : Member
        A lawless module.
    """
    with pytest.raises(Mismatch):
        run(member, "semimodule-action")


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
