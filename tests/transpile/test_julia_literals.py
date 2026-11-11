"""Julia spells each let literal at the type the source gave it.

A real literal keeps its decimal point, so real arithmetic on whole
numbers stays in `Float64` instead of wrapping around past
`typemax(Int)`; an integral literal, such as a factor binder's value or a
subscript, stays an `Int`, which Julia's indexing requires.
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile

SOURCE = """
object Step : FinSet 3
object Val : Real 1

program p : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    let ys = [1.0, 2.0, 3.0]
    let wide = 3000000000.0 * 4000000000.0 * 1.0e-18
    let lag = factor t : Step in ys[max(t - 1, 0)]
    score s = mu * wide + sum(lag)
    return mu

export p
"""


def _line(target: str, name: str) -> str:
    rendered = transpile(parse(SOURCE), target=target).decode()
    return next(
        line.strip()
        for line in rendered.splitlines()
        if line.strip().startswith(f"{name} =")
    )


@pytest.mark.parametrize("target", ["turing", "gen"])
def test_a_whole_number_real_keeps_its_decimal_point(target: str) -> None:
    assert _line(target, "wide") == "wide = (3000000000.0 * 4000000000.0) * 1e-18"


@pytest.mark.parametrize("target", ["turing", "gen"])
def test_an_integral_literal_stays_an_integer(target: str) -> None:
    line = _line(target, "lag")
    assert "ys[max.(0 - 1, 0) + 1]" in line
    assert "1.0" in _line(target, "ys")
