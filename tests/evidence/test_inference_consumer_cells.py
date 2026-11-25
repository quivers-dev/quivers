"""Every inference consumer against exact log joints.

A consumer is checked on the problems it accepts, and its refusals of
the others are cells of the grid (`test_grid`). The long runs that check
draws against a posterior are slow.
"""

from __future__ import annotations

import pytest

from tests.evidence.conformance import parameters, run


@pytest.mark.parametrize("name", parameters("consumer"))
def test_check(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    run(name)
