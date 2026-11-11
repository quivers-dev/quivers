"""Reductions nested in one let collapse one axis each.

``sum(sum(t))`` means the two lets ``u = sum(t)`` and ``s = sum(u)``: each
reduction collapses the last axis its argument has. Nested reductions over
factor tensors of rank two, three, and four agree with the split form on
the compiled runtime, with the reference machine, and with the rendered
NumPyro and Pyro bodies evaluated on the same arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quivers.dsl import loads
from tests.dsl.test_reduction_agreement import (
    _NUMPYRO_NAMESPACE,
    EXPECTED,
    _assert_values,
    _rendered_lets,
)


def _nested_source(rank: int, depth: int, builtin: str, *, split: bool) -> str:
    """A program reducing a factor tensor several times.

    Parameters
    ----------
    rank : int
        The number of axes the factor builds.
    depth : int
        How many reductions apply to it.
    builtin : str
        The reduction.
    split : bool
        Whether each reduction is its own let, rather than all of them
        nested in one.

    Returns
    -------
    str
        QVR source returning the reduced value as ``r``.
    """
    binders = ", ".join(f"i{axis} : A" for axis in range(rank))
    body = " * ".join(f"w{axis}[i{axis}]" for axis in range(rank))
    sites = "".join(
        f"    sample w{axis} : A <- Normal(loc=0.0, scale=1.0)\n"
        for axis in range(rank)
    )
    if split:
        names = [f"u{level}" for level in range(depth - 1)] + ["r"]
        sources = ["t", *names[:-1]]
        reductions = "".join(
            f"    let {name} = {builtin}({source})\n"
            for name, source in zip(names, sources, strict=True)
        )
    else:
        nested = "t"
        for _ in range(depth):
            nested = f"{builtin}({nested})"
        reductions = f"    let r = {nested}\n"
    return (
        "\nobject A : FinSet 2\n\nprogram total : A -> A\n"
        f"{sites}    let t = factor {binders} in {body}\n"
        f"{reductions}    return r\n\nexport total\n"
    )


@pytest.mark.parametrize("builtin", ("sum", "max"))
@pytest.mark.parametrize(("rank", "depth"), ((2, 2), (3, 2), (3, 3), (4, 2), (4, 3)))
def test_nested_reductions_in_one_let_collapse_one_axis_each(
    rank: int, depth: int, builtin: str
) -> None:
    source = _nested_source(rank, depth, builtin, split=False)
    values = {f"w{axis}": [0.5 + axis, -1.25 + 0.5 * axis] for axis in range(rank)}
    tensor = torch.tensor(values["w0"])
    for axis in range(1, rank):
        tensor = tensor.unsqueeze(-1) * torch.tensor(values[f"w{axis}"])
    expected = tensor
    for _ in range(depth):
        expected = EXPECTED[builtin](expected)
    program = loads(source)
    assert not program.qiec.gap, program.qiec.gap
    split = loads(_nested_source(rank, depth, builtin, split=True))
    for compiled in (program, split):
        value = compiled.morphism.rsample(
            torch.zeros(1, 1),
            observations={name: torch.tensor(item) for name, item in values.items()},
        )
        _assert_values(value, expected, "compiled runtime")
    reference = program.run(
        "total", sites={name: tuple(item) for name, item in values.items()}
    )
    _assert_values(reference.value, expected, "reference machine")
    for target in ("numpyro", "pyro"):
        lets = _rendered_lets(source, target, frozenset({"t", "r"}))
        namespace: dict[str, object] = (
            {**_NUMPYRO_NAMESPACE, **{n: np.asarray(v) for n, v in values.items()}}
            if target == "numpyro"
            else {"torch": torch, **{n: torch.tensor(v) for n, v in values.items()}}
        )
        for statement in lets:
            exec(statement, namespace)
        _assert_values(namespace["r"], expected, target)
