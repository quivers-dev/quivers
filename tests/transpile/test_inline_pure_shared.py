"""A called pure computation's bindings are stated once each in the plan.

`step` binds `depth` locals in sequence, each reading the previous one
three times. Inlined by substitution, the rendered expression would
triple with every binding; the plan instead states each binding as a
statement of its own, so the rendered program grows linearly in the
depth and every statement stays the same size.
"""

from __future__ import annotations

import re
import time

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile


def _source(depth: int) -> str:
    lines = ["    let v0 = x * 0.5"]
    for k in range(1, depth + 1):
        lines.append(f"    let v{k} = v{k - 1} * v{k - 1} * 0.1 + v{k - 1}")
    body = "\n".join(lines)
    return f"""
object Val : Real 1

define step(x : Real) : Real !{{}} =
{body}
    return v{depth}

program p : Val -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    let out <- step(mu)
    score s = out
    return mu

export p
"""


def _bindings(depth: int, target: str) -> list[str]:
    """The rendered statements that bind the computation's locals."""
    rendered = transpile(parse(_source(depth)), target=target).decode()
    statements = rendered.replace(";", ";\n").splitlines()
    return [
        line.strip()
        for line in statements
        if re.search(r"\bv\d+\s*=", line) and "lambda" not in line
    ]


@pytest.mark.parametrize("target", ["numpyro", "pyro", "pymc", "turing", "webppl"])
def test_each_binding_is_one_statement_of_bounded_size(target: str) -> None:
    shallow, deep = _bindings(4, target), _bindings(16, target)
    assert len(shallow) == 5
    assert len(deep) == 17
    assert max(len(line) for line in deep) <= max(len(line) for line in shallow) + 4


def test_a_deep_computation_transpiles_quickly() -> None:
    start = time.perf_counter()
    transpile(parse(_source(32)), target="numpyro")
    # Generous: linear rendering takes seconds; substitution took hours.
    assert time.perf_counter() - start < 300.0
