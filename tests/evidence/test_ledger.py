"""Known defects hold exactly, in both directions."""

from __future__ import annotations

import pytest

from tests.evidence.grid import LEDGERED
from tests.evidence.ledger import ENTRIES, Entry, verify
from tests.evidence.compare import Mismatch


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_entry(name: str) -> None:
    """Require one entry to show exactly its recorded wrong facts.

    Parameters
    ----------
    name : str
        The entry.
    """
    verify(ENTRIES[name])


def test_every_entry_is_named_by_a_cell() -> None:
    """Require every entry to be a cell of the grid and every cell's entry to exist."""
    named = {
        entry for row in LEDGERED.values() for cell in row.values() for entry in cell
    }
    assert named == set(ENTRIES)


def _with(entry: Entry, **changes: dict[str, str]) -> Entry:
    """An entry with other recorded facts.

    Parameters
    ----------
    entry : Entry
        The entry.
    **changes : dict of str to str
        The replaced fields.

    Returns
    -------
    Entry
        The changed entry.
    """
    return entry.with_(**changes)


def test_a_repaired_defect_fails_its_entry() -> None:
    """Require an entry whose library now gives the required facts to fail."""
    entry = ENTRIES["dense-join"]
    repaired = entry.with_(observe=entry.require)
    with pytest.raises(Mismatch, match="repaired"):
        verify(repaired)


def test_a_changed_defect_fails_its_entry() -> None:
    """Require an entry whose wrong facts moved to fail."""
    entry = ENTRIES["residual-is-the-value"]
    moved = dict(entry.wrong) | {"agenda at 9/10": "2.29"}
    with pytest.raises(Mismatch, match="changed"):
        verify(_with(entry, wrong=moved))


def test_a_fact_that_is_right_cannot_be_recorded_as_wrong() -> None:
    """Require every recorded fact to differ from its requirement."""
    entry = ENTRIES["dense-join"]
    with pytest.raises(Mismatch, match="not wrong"):
        verify(_with(entry, wrong=dict(entry.right)))


def test_a_wrong_requirement_fails() -> None:
    """Require the recorded requirement to be the computed one."""
    entry = ENTRIES["residual-is-the-value"]
    with pytest.raises(Mismatch, match="required"):
        verify(_with(entry, right=dict.fromkeys(entry.right, "-1")))
