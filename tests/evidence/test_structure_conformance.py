"""Conformance of structural folds and term distributions against enumerated terms."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize("name", parameters("terms", "repeat/encoder", "repeat/decoder"))
def test_check(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    run(name)
