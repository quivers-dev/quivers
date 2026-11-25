"""Write the pins of the harness after a reviewed change.

The pins hold the digest of every harness file and the cells of the grid
that pass. Writing refuses to drop a cell that passed before, and
`test_grid` fails when a pinned passing cell no longer passes, so a cell
moves to ``pass`` only.
"""

from __future__ import annotations

import json
from collections import Counter

from tests.evidence.catalogue import PINS, digests
from tests.evidence.checks.all import CHECKS
from tests.evidence.grid import cells


def passing_cells() -> list[str]:
    """The cells of the grid that pass.

    Returns
    -------
    list of str
        ``"row/column"`` of each passing cell, sorted.
    """
    grid = cells(Counter((check.executor, check.question) for check in CHECKS.values()))
    return sorted(
        f"{row}/{column}"
        for (row, column), cell in grid.items()
        if cell.status == "pass"
    )


def write() -> None:
    """Write the pins, keeping every cell that passed before.

    Raises
    ------
    SystemExit
        If a cell that passed no longer passes.
    """
    before: list[str] = []
    if PINS.exists():
        before = json.loads(PINS.read_text()).get("passing cells", [])
    now = passing_cells()
    lost = sorted(set(before) - set(now))
    if lost:
        raise SystemExit(f"cells that passed no longer pass: {lost}")
    PINS.write_text(
        json.dumps({"files": digests(), "passing cells": now}, indent=1, sort_keys=True)
        + "\n"
    )


if __name__ == "__main__":
    write()
