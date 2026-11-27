"""Register every check, and the repeated-run check of every executor."""

from __future__ import annotations

from functools import partial

from tests.evidence import runs
from tests.evidence.checks import (
    closure,
    consumers,
    contraction,
    deduction,
    machine,
    mixture,
    presented,
    span,
    terms,
    work,
)
from tests.evidence.checks._registry import CHECKS, add
from tests.evidence import newton_controls, span_controls

MODULES = (
    closure,
    consumers,
    contraction,
    deduction,
    machine,
    mixture,
    presented,
    span,
    terms,
    work,
)
"""The modules whose checks are registered."""


def _register() -> None:
    """Register the repeated runs and the checks of the plan executors."""
    span_controls.register()
    newton_controls.register()
    for name in runs.RUNS:
        add(f"repeat/{name}", name, "Q24", partial(runs.check_repeat, name))


_register()

__all__ = ["CHECKS"]
