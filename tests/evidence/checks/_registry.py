"""The registry of named checks.

A check runs one executor on one problem and raises
`tests.evidence.compare.Mismatch` when the result is wrong. The
conformance tests run every check once. The mutation controls run the
checks a defect has to be rejected by, under the defect, so a defect is
tied to the name of the check that rejects it. The grid names the checks
that back each of its cells.
"""

from __future__ import annotations

from collections.abc import Callable

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel

type Run = Callable[[], None]
"""A check's body."""


class Check(ValueModel):
    """One executor on one problem.

    Parameters
    ----------
    name : str
        The check's name.
    executor : str
        The executor, a row of the grid.
    question : str
        The problem class, a column of the grid.
    run : Callable[[], None]
        The body, which raises on a wrong result.
    slow : bool
        Whether the check runs only with the slow tests.
    """

    name: str
    executor: str
    question: str
    run: Run = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    slow: bool = False


CHECKS: dict[str, Check] = {}
"""Every registered check by name."""


def add(
    name: str, executor: str, question: str, run: Run, *, slow: bool = False
) -> None:
    """Register a check.

    Parameters
    ----------
    name : str
        The check's name.
    executor : str
        The executor.
    question : str
        The problem class.
    run : Callable[[], None]
        The body.
    slow : bool
        Whether the check runs only with the slow tests.

    Raises
    ------
    ValueError
        If the name is taken.
    """
    if name in CHECKS:
        raise ValueError(f"the check {name!r} is registered twice")
    CHECKS[name] = Check(
        name=name, executor=executor, question=question, run=run, slow=slow
    )


def under(prefix: str, *, slow: bool = False) -> list[str]:
    """The names of the checks below a prefix.

    Parameters
    ----------
    prefix : str
        The first part of the name.
    slow : bool
        Which speed to list.

    Returns
    -------
    list of str
        The names in registration order.
    """
    return [
        name
        for name, check in CHECKS.items()
        if name.startswith(prefix + "/") and check.slow == slow
    ]
