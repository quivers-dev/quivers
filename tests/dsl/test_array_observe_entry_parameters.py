"""An observe over `over` axes reads a parameter that varies along them per entry.

`observe y : Row <- Normal(loc=slice, scale=s) [over=[Part, Class]]` with
`slice` one `(Part, Class)` slice per row scores each entry of a row's
array at its own entry of the slice. The compiled runtime and the
reference machine both score that joint, pinned to an independent
`torch.distributions` computation, with the scale shared by a row's
entries and with a scale that varies per entry too; repeating the whole
slice for every entry, which scores each entry against every location,
is a joint the oracle rejects.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads

_SOURCE = """
object Group : FinSet 2
object Part : FinSet 2
object Class : FinSet 3
object Row : FinSet 4
object Val : Real 1

program slices : Row -> Val
    sample a : Group <- Normal(loc=0.0, scale=1.0)
    let table = factor g : Group, p : Part, k : Class in a[g] * real(p + 1) + real(k)
    let spread = factor g : Group, p : Part, k : Class in exp(0.1 * real(k) - 0.2 * real(p))
    let row_slice = table[group_idx]
    let row_spread = spread[group_idx]
    observe y : Row <- Normal(loc=row_slice, scale=SCALE) [over=[Part, Class]]
    return a

export slices
"""

_A = torch.tensor([0.4, -1.3], dtype=torch.float64)
_IDX = torch.tensor([0, 1, 1, 0])
_Y = torch.tensor(
    [
        [[0.3, 1.2, 2.5], [0.1, 0.9, 1.7]],
        [[-0.4, 0.8, 2.2], [-1.0, 0.2, 1.1]],
        [[0.6, 1.4, 2.9], [0.9, 2.1, 2.8]],
        [[0.0, 1.0, 2.0], [0.2, 1.3, 2.4]],
    ],
    dtype=torch.float64,
)


def _tables() -> tuple[torch.Tensor, torch.Tensor]:
    """The location and scale tables of the program at `_A`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The ``(Group, Part, Class)`` location and scale tables.
    """
    part = torch.arange(2, dtype=torch.float64)[None, :, None]
    cls = torch.arange(3, dtype=torch.float64)[None, None, :]
    table = _A[:, None, None] * (part + 1.0) + cls
    spread = torch.exp(0.1 * cls - 0.2 * part).expand(2, 2, 3)
    return table, spread


def _nested(value: torch.Tensor) -> float | tuple:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    float | tuple
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(item) for item in value)


@pytest.mark.parametrize("scale", ["0.5", "row_spread"])
def test_each_entry_is_scored_at_its_own_parameters(scale: str) -> None:
    program = loads(_SOURCE.replace("SCALE", scale))
    table, spread = _tables()
    sigma = torch.tensor(0.5, dtype=torch.float64) if scale == "0.5" else spread[_IDX]
    expected = float(
        D.Normal(0.0, 1.0).log_prob(_A).sum()
        + D.Normal(table[_IDX], sigma).log_prob(_Y).sum()
    )
    values = {"a": _A.float(), "y": _Y.float(), "group_idx": _IDX}
    joint = program.morphism.log_joint(torch.zeros(1, dtype=torch.long), values)
    assert float(joint.sum()) == pytest.approx(expected, rel=1e-5)
    run = program.run(
        "slices",
        data={"y": _nested(_Y), "group_idx": _nested(_IDX)},
        sites={"a": _nested(_A)},
    )
    assert run.log_joint == pytest.approx(expected, rel=1e-9)


def test_the_oracle_rejects_the_whole_slice_at_every_entry() -> None:
    table, _ = _tables()
    rows = table[_IDX].reshape(4, 1, 6)
    whole = (
        D.Normal(0.0, 1.0).log_prob(_A).sum()
        + D.Normal(rows, 0.5).log_prob(_Y.reshape(4, 6, 1)).sum()
    )
    right = (
        D.Normal(0.0, 1.0).log_prob(_A).sum()
        + D.Normal(table[_IDX], 0.5).log_prob(_Y).sum()
    )
    assert abs(float(right) - float(whole)) > 1.0
