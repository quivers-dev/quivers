"""`map` in a program let on the compiled runtime.

`map(xs, x -> body)` builds a tensor with the leading extent of `xs`, each
entry the lambda's value at one element. The compiled runtime returns the
tensor the checked type names, for a lambda returning numbers and for one
returning vectors, and agrees with the reference machine, also when the
mapped tensor only feeds a later reduction.
"""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads

W = [1.0, 2.0]

#: Each case: the program body after its draw of `w`, the checked result
#: type, and the value the body returns at `w`.
CASES: dict[str, tuple[str, str, list[float] | list[list[float]] | float]] = {
    "map returning numbers": (
        "    let d = map(w, x -> x * 2.0)\n    return d",
        "Tensor[Real]([2])",
        [2.0, 4.0],
    ),
    "map returning vectors": (
        "    let t = factor a : A, b : A in w[a] * w[b]\n"
        "    let d = map(t, row -> row * 2.0)\n"
        "    return d",
        "Tensor[Real]([2, 2])",
        [[2.0, 4.0], [4.0, 8.0]],
    ),
    "map of a map": (
        "    let t = factor a : A, b : A in w[a] * w[b]\n"
        "    let d = map(t, row -> map(row, x -> x + 1.0))\n"
        "    return d",
        "Tensor[Real]([2, 2])",
        [[2.0, 3.0], [3.0, 5.0]],
    ),
    "map feeding a reduction": (
        "    let d = map(w, x -> x * 2.0)\n    let s = sum(d)\n    return s",
        "Real",
        6.0,
    ),
    "map of reductions": (
        "    let t = factor a : A, b : A in w[a] * w[b]\n"
        "    let d = map(t, row -> sum(row))\n"
        "    return d",
        "Tensor[Real]([2])",
        [3.0, 6.0],
    ),
}


def _nested(value: list[float] | list[list[float]] | float) -> object:
    """A value as the reference machine's nested tuples.

    Parameters
    ----------
    value : list[float] | list[list[float]] | float
        The value.

    Returns
    -------
    object
        A number, or nested tuples of numbers.
    """
    if isinstance(value, list):
        return tuple(_nested(item) for item in value)
    return value


@pytest.mark.parametrize("case", sorted(CASES))
def test_map_returns_the_checked_tensor_on_both_paths(case: str) -> None:
    body, result, expected = CASES[case]
    program = loads(
        "object A : FinSet 2\n\n"
        "program total : A -> A\n"
        "    sample w : A <- Normal(loc=0.0, scale=1.0)\n"
        f"{body}\n\nexport total\n"
    )
    (entry,) = program.entry_points()
    assert entry.result == result
    reference = program.run("total", sites={"w": tuple(W)}).value
    assert reference == _nested(expected)
    compiled = program.morphism.rsample(
        torch.zeros(1, 1), observations={"w": torch.tensor(W)}
    )
    assert isinstance(compiled, torch.Tensor)
    torch.testing.assert_close(
        compiled.reshape(torch.tensor(expected).shape), torch.tensor(expected)
    )
