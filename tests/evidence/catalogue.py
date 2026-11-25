"""Pinned numbers of the evidence harness, and the digest of its files.

The tolerances of the deduction checks are computed from the operation
counts below, which `tests.evidence.problems.operation_bound` derives
again from each problem. Every file of the harness that holds a
tolerance, a point set, a minimum effect, a claimed bound, or a known
defect is listed with the digest of its bytes in `pins.json`, so a
change of any of them is a visible change of the pins.

`python -m tests.evidence.pin` writes the pins after a reviewed change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

OPERATIONS: dict[str, int] = {
    "repeated_premise": 66,
    "three_equal_premises": 132,
    "diamond": 336,
    "skip": 404,
    "reach": 426,
    "ambiguous": 870,
    "nested": 128,
    "high_arity": 280,
    "side_condition": 198,
    "equal_proofs": 128,
    "exact_zero": 128,
    "tiny": 128,
    "impossible": 30,
    "edit_lattice": 812,
}
"""Roundings on the longest path of each deduction problem."""

ROOT = Path(__file__).parent
"""The harness."""

PINS = ROOT / "pins.json"
"""The recorded digests."""


def pinned_files() -> list[Path]:
    """The files whose bytes are pinned.

    Returns
    -------
    list of Path
        Every Python file and recorded table of the harness except the
        pins themselves.
    """
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.suffix in (".py", ".json")
        and path.name != PINS.name
        and "__pycache__" not in path.parts
    )


def digests() -> dict[str, str]:
    """The digest of every pinned file.

    Returns
    -------
    dict of str to str
        The SHA-256 of each file's bytes, by its path below the harness.
    """
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in pinned_files()
    }

