"""Register every check, and the repeated-run check of every executor."""

from __future__ import annotations

from functools import partial

from tests.evidence import runs
from tests.evidence.checks._registry import CHECKS, add


def _register() -> None:
    """Register the repeated runs."""
    for name in runs.RUNS:
        add(f"repeat/{name}", name, "Q24", partial(runs.check_repeat, name))


_register()

__all__ = ["CHECKS"]
