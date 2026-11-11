"""A let that calls a name nothing defines is refused when the program loads.

The compiled runtime resolves a call against its builtins, its own
combinators, the compiled deductions, and the declared constructors; a
call naming none of them would fail at the first trace, after ``loads``
had returned a program. ``loads`` refuses it at the call instead.
"""

from __future__ import annotations

import pytest

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError

_TEMPLATE = """
object Row : FinSet 4
program p : Row -> Row
    sample v : Row <- Normal(loc=0.0, scale=1.0)
    STEP
    observe y : Row <- Normal(loc=v, scale=1.0)
    return y
export p
"""


@pytest.mark.parametrize(
    ("step", "name"),
    [
        ("let z = matmul(v, v)", "matmul"),
        ("let z = exp(where(v, v, v))", "where"),
        ("score z = sum(frobnicate(v))", "frobnicate"),
    ],
)
def test_loads_refuses_a_call_to_an_unknown_name_at_the_call(
    step: str, name: str
) -> None:
    source = _TEMPLATE.replace("STEP", step)
    column = source.splitlines()[4].index(f"{name}(")
    with pytest.raises(CompileError, match=f"unknown function {name!r}") as raised:
        loads(source)
    assert raised.value.line == 5
    assert raised.value.col == column


@pytest.mark.parametrize(
    "step",
    ["let z = exp(v)", "let z = softmax(v)", "score z = sum(log1p(exp(v)))"],
)
def test_loads_accepts_a_call_to_a_builtin(step: str) -> None:
    loads(_TEMPLATE.replace("STEP", step))
