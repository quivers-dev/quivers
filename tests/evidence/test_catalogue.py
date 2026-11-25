"""The pinned files of the harness are unchanged."""

from __future__ import annotations

import ast
import json

from tests.evidence.catalogue import OPERATIONS, PINS, ROOT, digests
from tests.evidence.problems import DEDUCTIONS, operation_bound


def test_operation_counts_are_derived_again() -> None:
    """Require every pinned operation count to be the derived one."""
    assert OPERATIONS == {name: operation_bound(name) for name in DEDUCTIONS}


def test_pinned_files_are_unchanged() -> None:
    """Require the digests of the harness files to be the pinned ones.

    A change of a tolerance, a point set, a minimum effect, a bound, or a
    defect changes a digest; the change and its new digest are reviewed
    together.
    """
    pinned = json.loads(PINS.read_text())["files"]
    assert digests() == pinned


def test_oracles_import_nothing_from_the_library() -> None:
    """Require the oracles to be independent of the code they check."""
    for path in sorted((ROOT / "oracles").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not name.startswith(("quivers", "tests")), (path.name, name)
