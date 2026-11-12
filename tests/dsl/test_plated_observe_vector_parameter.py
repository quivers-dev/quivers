"""A plated observe reads a vector over its rows one entry per row.

``observe y : Row <- Normal(loc=z, scale=0.5)`` with ``z`` one vector over
``Row`` scores row ``i`` at ``z[i]``. The compiled runtime and the
reference machine both score that joint, pinned to an independent
`torch.distributions` computation, for one run and for several runs
whose vectors differ; reading the whole of ``z`` at every row, which
scores each row against every entry, is a joint the oracle rejects.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads

_SOURCE = """
object Row : FinSet 4
program p : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0) [over=Row]
    let z = tanh(a)
    let r = exp(a)
    observe y : Row <- Normal(loc=z, scale=0.5)
    observe c : Row <- Poisson(rate=r)
    return z
export p
"""

_A = torch.tensor([[0.4, -0.8, 1.5, -0.2], [-1.1, 0.2, 0.6, 1.4]], dtype=torch.float64)
_Y = torch.tensor([[0.9, -1.2, 0.3, 0.0], [0.1, 0.4, -0.6, 1.1]], dtype=torch.float64)
_C = torch.tensor([[1, 0, 4, 2], [0, 3, 1, 5]])


def _oracle(a: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The joint of one run, row ``i`` scored at entry ``i``.

    Parameters
    ----------
    a : torch.Tensor
        The drawn vector.
    y : torch.Tensor
        The Normal observations.
    c : torch.Tensor
        The Poisson counts.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    return (
        D.Normal(0.0, 1.0).log_prob(a).sum()
        + D.Normal(torch.tanh(a), 0.5).log_prob(y).sum()
        + D.Poisson(torch.exp(a)).log_prob(c.double()).sum()
    )


def test_one_run_scores_each_row_at_its_entry() -> None:
    program = loads(_SOURCE)
    expected = float(_oracle(_A[0], _Y[0], _C[0]))
    values = {"a": _A[0].float(), "y": _Y[0].float(), "c": _C[0]}
    joint = program.morphism.log_joint(torch.zeros(1, dtype=torch.long), values)
    assert float(joint.sum()) == pytest.approx(expected, rel=1e-5)
    run = program.run(
        "p",
        data={"y": tuple(_Y[0].tolist()), "c": tuple(_C[0].tolist())},
        sites={"a": tuple(_A[0].tolist())},
    )
    assert run.log_joint == pytest.approx(expected, rel=1e-9)


def test_each_run_scores_its_own_rows() -> None:
    program = loads(_SOURCE)
    values = {"a": _A.float(), "y": _Y.float(), "c": _C}
    joint = program.morphism.log_joint(torch.zeros(2, dtype=torch.long), values)
    expected = [float(_oracle(_A[r], _Y[r], _C[r])) for r in range(2)]
    assert joint.shape == (2,)
    assert [float(v) for v in joint] == pytest.approx(expected, rel=1e-5)


def test_the_oracle_rejects_reading_the_whole_vector_at_every_row() -> None:
    a, y, c = _A[0], _Y[0], _C[0]
    whole = (
        D.Normal(0.0, 1.0).log_prob(a).sum()
        + D.Normal(torch.tanh(a)[None, :], 0.5).log_prob(y[:, None]).sum()
        + D.Poisson(torch.exp(a)).log_prob(c.double()).sum()
    )
    assert abs(float(_oracle(a, y, c)) - float(whole)) > 1.0
