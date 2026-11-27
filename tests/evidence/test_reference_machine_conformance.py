"""Compare the reference machine with independent oracles and the compiled runtime."""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize(
    "name",
    parameters(
        "machine",
        "checked",
        "equivalence",
        "repeat/machine-lexical",
        "repeat/machine-nonlexical",
        "repeat/machine-chained",
        "repeat/machine-program",
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
