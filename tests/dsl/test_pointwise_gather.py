"""A subscript with several index vectors selects one entry per position.

`M[r, c]` with `r` and `c` index vectors over the same rows reads
`M[r[n], c[n]]` at each row `n`, as advanced indexing does: the result
has the index vectors' shape followed by the axes of `M` the subscript
leaves, and an `Int` index stands for the same value at every row. Each
program is scored on the compiled runtime and on the reference machine,
both pinned to an independent `torch.distributions` joint whose mean is
computed with explicit loops; a row-gather reading of the same subscript
is shown to miss that joint, and index vectors of two different shapes
are refused at the subscript.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.entries import HostValue
from quivers.transpile import available_targets, transpile

_HEAD = (
    "object A : FinSet 3\nobject B : FinSet 2\nobject Row : FinSet 6\n"
    "program p : Row -> Row\n"
    "    sample a : A <- Normal(loc=0.0, scale=1.0)\n"
    "    sample b : B <- Normal(loc=0.0, scale=1.0)\n"
)
_TAIL = "    observe y : Row <- Normal(loc=m, scale=1.0)\n    return y\nexport p\n"

_M = torch.tensor([[1.0, -2.0], [0.5, 4.0], [-3.0, 6.0]], dtype=torch.float64)
_R = torch.tensor([0, 1, 2, 0, 1, 2])
_C = torch.tensor([0, 0, 0, 1, 1, 1])
_A = torch.tensor([0.3, -1.1, 0.8], dtype=torch.float64)
_B = torch.tensor([-0.4, 1.5], dtype=torch.float64)
_Y = torch.tensor([0.2, -0.7, 1.1, 0.4, 2.0, -1.3], dtype=torch.float64)


def _host_mean(m: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> list[float]:
    del a, b
    return [float(m[int(_R[n]), int(_C[n])]) for n in range(6)]


def _table(m: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    del m
    return torch.stack([torch.stack([a[i] + b[j] for j in range(2)]) for i in range(3)])


#: Each case: its let lines, whether it reads `M`, and its mean by loops.
_CASES: dict[
    str,
    tuple[str, bool, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], list[float]]],
] = {
    "host-matrix": ("    let m = M[r_idx, c_idx]\n", True, _host_mean),
    "host-matrix-times-gather": (
        "    let m = M[r_idx, c_idx] * a[r_idx]\n",
        True,
        lambda m, a, b: [
            float(m[int(_R[n]), int(_C[n])] * a[int(_R[n])]) for n in range(6)
        ],
    ),
    "factor-built-matrix": (
        "    let mt = factor i : A, j : B in a[i] + b[j]\n"
        "    let m = mt[r_idx, c_idx]\n",
        False,
        lambda m, a, b: [
            float(_table(m, a, b)[int(_R[n]), int(_C[n])]) for n in range(6)
        ],
    ),
    "integer-column": (
        "    let m = M[r_idx, 1]\n",
        True,
        lambda m, a, b: [float(m[int(_R[n]), 1]) for n in range(6)],
    ),
}


def _oracle(mean: list[float]) -> float:
    """The joint at the fixed sites, computed without the DSL.

    Parameters
    ----------
    mean : list[float]
        The observation's mean at each row.

    Returns
    -------
    float
        The log joint.
    """
    standard = D.Normal(0.0, 1.0)
    return float(
        standard.log_prob(_A).sum()
        + standard.log_prob(_B).sum()
        + D.Normal(torch.tensor(mean, dtype=torch.float64), 1.0).log_prob(_Y).sum()
    )


def _nested(value: torch.Tensor) -> HostValue:
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(entry) for entry in value)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_compiled_runtime_selects_one_entry_per_row(case: str) -> None:
    lets, reads_m, mean = _CASES[case]
    model = loads(_HEAD + lets + _TAIL).morphism
    sites = {
        "a": _A.float(),
        "b": _B.float(),
        "y": _Y.float(),
        "r_idx": _R,
        "c_idx": _C,
    }
    if reads_m:
        sites["M"] = _M.float()
    joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
    assert joint == pytest.approx(_oracle(mean(_M, _A, _B)), rel=1e-5)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_reference_machine_selects_one_entry_per_row(case: str) -> None:
    lets, reads_m, mean = _CASES[case]
    program = loads(_HEAD + lets + _TAIL)
    data: dict[str, HostValue] = {"y": _nested(_Y), "r_idx": _nested(_R)}
    if "c_idx" in lets:
        data["c_idx"] = _nested(_C)
    if reads_m:
        data["M"] = _nested(_M)
    run = program.run(
        "p", data=data, sites={"a": _nested(_A), "b": _nested(_B)}, seed=0
    )
    assert run.log_joint == pytest.approx(_oracle(mean(_M, _A, _B)), rel=1e-9)


@pytest.mark.parametrize(
    "wrong",
    [
        # The column index ignored: the first entry of each gathered row.
        lambda n: float(_M[int(_R[n]), 0]),
        # The two index vectors swapped onto each other's axes.
        lambda n: float(_M[int(_C[n]), int(_R[n]) % 2]),
        # The other column.
        lambda n: float(_M[int(_R[n]), 1 - int(_C[n])]),
    ],
)
def test_the_pinned_joint_rejects_a_misread_subscript(
    wrong: Callable[[int], float],
) -> None:
    program = loads(_HEAD + _CASES["host-matrix"][0] + _TAIL)
    run = program.run(
        "p",
        data={
            "y": _nested(_Y),
            "r_idx": _nested(_R),
            "c_idx": _nested(_C),
            "M": _nested(_M),
        },
        sites={"a": _nested(_A), "b": _nested(_B)},
        seed=0,
    )
    assert run.log_joint != pytest.approx(
        _oracle([wrong(n) for n in range(6)]), rel=1e-3
    )


def test_index_vectors_of_different_shapes_are_refused_at_the_subscript() -> None:
    source = (
        _HEAD + "    let mt = factor i : A, j : B in a[i] + b[j]\n"
        "    let rows = factor n : Row in n % 3\n"
        "    let cols = factor n : B in n\n"
        "    let m = mt[rows, cols]\n" + _TAIL
    )
    with pytest.raises(QiecDiagnosticError, match="share one shape") as raised:
        loads(source)
    assert ":10:" in str(raised.value)


@pytest.mark.parametrize("target", sorted(available_targets()))
def test_every_target_renders_the_pairwise_selection(target: str) -> None:
    rendered = transpile(
        parse(_HEAD + _CASES["host-matrix"][0] + _TAIL), target=target
    ).decode()
    assert "c_idx" in rendered and "r_idx" in rendered
