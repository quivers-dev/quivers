"""Conformance of closures and traces against exact inverses and enumerated paths."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize("name", parameters("closure", "repeat/kleene-star", "repeat/kleene-star-trace"))
def test_check(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    run(name)
