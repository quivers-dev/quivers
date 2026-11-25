"""Shared helpers of the conformance tests."""

from __future__ import annotations

import pytest
from _pytest.mark import ParameterSet

from tests.evidence.checks.all import CHECKS


def parameters(*prefixes: str) -> list[ParameterSet]:
    """The registered checks below some prefixes, the slow ones marked.

    Parameters
    ----------
    *prefixes : str
        The first parts of the check names.

    Returns
    -------
    list of ParameterSet
        One parameter per check.
    """
    return [
        pytest.param(name, id=name, marks=(pytest.mark.slow,) if check.slow else ())
        for name, check in CHECKS.items()
        if any(name.startswith(prefix + "/") for prefix in prefixes)
    ]


def run(name: str) -> None:
    """Run one registered check.

    Parameters
    ----------
    name : str
        The check.
    """
    CHECKS[name].run()
