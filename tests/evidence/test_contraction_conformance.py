"""Conformance of wiring contractions against sums over index assignments."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize("name", parameters("contraction", "repeat/einsum-wiring"))
def test_check(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    run(name)
