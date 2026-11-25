"""Conformance of deductions on the reference machine against enumerated derivations."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize("name", parameters("machine", "repeat/machine-lexical", "repeat/machine-nonlexical", "repeat/machine-chained"))
def test_check(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    run(name)
