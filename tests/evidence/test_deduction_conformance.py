"""Conformance of the agenda engine against enumerated derivations."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize(
    "name",
    parameters(
        "deduction",
        "repeat/agenda-fifo",
        "repeat/agenda-lifo",
        "repeat/agenda-priority",
    ),
)
def test_check(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    run(name)
