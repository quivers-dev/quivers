"""The grid of executors by problem classes is complete and true."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tests.evidence.catalogue import PINS
from tests.evidence.checks.all import CHECKS
from tests.evidence.grid import (
    COLUMNS,
    CONSUMER_COLUMNS,
    LEDGERED,
    ROWS,
    cells,
    recorded_refusals,
    refusal,
)
from tests.evidence.ledger import ENTRIES
from tests.evidence.runs import CONSUMERS

PASSING = Counter((check.executor, check.question) for check in CHECKS.values())
GRID = cells(PASSING)
REFUSED = sorted(
    f"{row}/{column}"
    for (row, column), cell in GRID.items()
    if cell.status == "refuses"
)


def test_every_check_is_a_cell() -> None:
    """Require every check to name a row and a column of the grid."""
    assert set(PASSING) <= set(GRID)


def test_every_cell_has_a_value() -> None:
    """Require a value in every cell of both blocks."""
    rows = {row for row, _ in GRID}
    assert rows == set(ROWS) | set(CONSUMERS)
    for row in ROWS:
        assert all(GRID[(row, column)].status for column in COLUMNS)
    for row in CONSUMERS:
        assert all(GRID[(row, column)].status for column in CONSUMER_COLUMNS)


def test_every_ledger_cell_names_entries() -> None:
    """Require ledger cells to name entries that exist."""
    for row, columns in LEDGERED.items():
        assert row in set(ROWS) | set(CONSUMERS), row
        for column, entries in columns.items():
            assert column in COLUMNS or column in CONSUMER_COLUMNS, column
            assert entries and set(entries) <= set(ENTRIES), (row, column)


def test_the_recorded_refusals_are_the_refusing_cells() -> None:
    """Require one recorded diagnostic per refusing cell and no other."""
    assert sorted(recorded_refusals()) == REFUSED


@pytest.mark.parametrize("cell", REFUSED)
def test_a_refusing_cell_refuses_exactly(cell: str) -> None:
    """Require a refused question to raise the recorded diagnostic.

    Parameters
    ----------
    cell : str
        ``"row/column"``.
    """
    row, column = cell.rsplit("/", 1)
    assert refusal(row, column) == recorded_refusals()[cell]


def test_cells_move_to_pass_only() -> None:
    """Require every cell that passed when pinned to pass now."""
    pinned = set(json.loads(PINS.read_text())["passing cells"])
    now = {
        f"{row}/{column}"
        for (row, column), cell in GRID.items()
        if cell.status == "pass"
    }
    assert pinned <= now, sorted(pinned - now)
