"""Conformance of summed-out latents, scans, and collapsed pairs against enumerated draws."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize(
    "name",
    parameters(
        "mixture",
        "repeat/marginalize-grouped",
        "repeat/marginalize-ungrouped",
        "repeat/marginalize-chain",
        "repeat/scan",
        "repeat/collapse",
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
