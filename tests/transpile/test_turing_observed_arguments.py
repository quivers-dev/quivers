"""A Turing model never rebinds the data it is given.

The observed sites of a Turing `@model` are its arguments, and each
`~` on an argument scores the argument's value. A statement that
assigns the argument a new array, such as an uninitialized
`Array{Float64, N}(undef, ...)`, makes Turing score that array instead
of the data: whatever the memory holds, which on one machine is a value
near zero and on another a NaN. The check reads the rendered source of
every fixture the Turing target renders and requires that no statement
assigns to a model argument.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from quivers.dsl import parse
from quivers.transpile import UnsupportedConstruct, transpile

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_SIGNATURE = re.compile(r"@model\s+function\s+\w+\s*\(([^)]*)\)")


def _arguments(source: str) -> tuple[str, ...]:
    """The argument names of the rendered `@model` function.

    Parameters
    ----------
    source : str
        The rendered Turing source.

    Returns
    -------
    tuple[str, ...]
        The names, without defaults or types.
    """
    match = _SIGNATURE.search(source)
    assert match is not None, "the rendered source declares no @model function"
    names = []
    for part in match.group(1).split(","):
        name = part.split("=")[0].split("::")[0].strip()
        if name:
            names.append(name)
    return tuple(names)


def _rendered(fixture: pathlib.Path) -> str | None:
    """A fixture's Turing source, or ``None`` when the target refuses it.

    Parameters
    ----------
    fixture : pathlib.Path
        The QVR fixture.

    Returns
    -------
    str | None
        The source.
    """
    try:
        source = transpile(parse(fixture.read_text()), target="turing")
    except UnsupportedConstruct:
        return None
    return source if isinstance(source, str) else source.decode()


_CASES = sorted(_FIXTURES.rglob("*.qvr"))


@pytest.mark.parametrize(
    "fixture", _CASES, ids=[str(p.relative_to(_FIXTURES)) for p in _CASES]
)
def test_no_statement_rebinds_a_model_argument(fixture: pathlib.Path) -> None:
    source = _rendered(fixture)
    if source is None:
        return
    for name in _arguments(source):
        rebinding = re.compile(rf"^\s*{re.escape(name)}\s*=(?!=)", re.MULTILINE)
        found = rebinding.search(source)
        assert found is None, (
            f"the model rebinds its argument {name!r}: {found.group(0).strip()!r}"
        )


def test_the_matrix_normal_observation_is_scored_as_given() -> None:
    """The observed matrix site that a row loop scores stays the argument."""
    for axis in ("row", "col"):
        source = _rendered(
            _FIXTURES / "steps" / f"observe_matrix_normal_{axis}_factor.qvr"
        )
        assert source is not None
        assert _arguments(source) == ("v",)
        assert "v[m_Row,:,:] ~ MatrixNormal(" in source
        assert "undef" not in source
