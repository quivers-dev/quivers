"""A let lays out a number from a continuous site as the checked number it is.

Within one run the compiled runtime holds a scalar draw as a vector of one
entry. A let that lays numbers out along a new axis (a list literal, a
``factor``) or reduces them reads each such operand as a number, so
``[h, 1.0]`` is one vector of two entries per run, as the checked type and
the reference machine have it. Each program is scored on both paths
against an independent ``torch.distributions`` joint, with one and with
two input rows.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.inference import trace

_PACK = """
object Row : FinSet 1
object K : FinSet 3
object V2 : Real 2
program p : Row -> V2
    sample h <- Normal(loc=0.0, scale=1.0)
    let packed = [h, 1.0]
    let f = factor k : K in h * real(k)
    let total = sum(packed) + sum(f)
    observe y <- Normal(loc=total, scale=1.0)
    return packed
export p
"""

_H = [0.7, -1.3]
_Y = [0.4, 2.2]


def _oracle(h: float, y: float) -> float:
    """One run's joint, computed without the DSL.

    Parameters
    ----------
    h : float
        The scalar site.
    y : float
        The observation.

    Returns
    -------
    float
        The log joint: ``total = (h + 1) + (0 + h + 2 h)``.
    """
    total = (h + 1.0) + 3.0 * h
    return float(
        D.Normal(0.0, 1.0).log_prob(torch.tensor(h, dtype=torch.float64))
        + D.Normal(total, 1.0).log_prob(torch.tensor(y, dtype=torch.float64))
    )


@pytest.mark.parametrize("runs", [1, 2])
def test_compiled_runtime_lays_out_the_list_one_vector_per_run(runs: int) -> None:
    model = loads(_PACK).morphism
    sites = {
        "h": torch.tensor(_H[:runs]).reshape(runs, 1),
        "y": torch.tensor(_Y[:runs]),
    }
    recorded = trace(model, torch.zeros(runs, 1), observations=sites)
    assert tuple(recorded.sites["packed"].value.shape) == (runs, 2)
    assert tuple(recorded.sites["f"].value.shape) == (runs, 3)
    expected = torch.tensor(
        [[_H[run], 1.0] for run in range(runs)], dtype=torch.float32
    )
    torch.testing.assert_close(recorded.sites["packed"].value, expected)
    joints = model.log_joint(torch.zeros(runs, 1), sites)
    assert joints.shape == (runs,)
    for run in range(runs):
        assert float(joints[run]) == pytest.approx(_oracle(_H[run], _Y[run]), rel=1e-5)
    # A total that read the stray one-entry axis as data would sum the
    # factor's three entries per row of a (3, 1) table: the same here,
    # but `packed` would then be (2, 1) and its sum (1,), which the
    # shape assertions above reject.


def test_reference_machine_agrees() -> None:
    run = loads(_PACK).run("p", data={"y": _Y[0]}, sites={"h": _H[0]}, seed=0)
    assert run.value == pytest.approx((_H[0], 1.0))
    assert run.log_joint == pytest.approx(_oracle(_H[0], _Y[0]), rel=1e-6)


_SCAN = """
object Token : FinSet 5
object Embedded : Real 1
object Flagged : Real 2
morphism tok_embed : Token -> Embedded [role=embed]

program cell(x_t, state) : Embedded * Flagged -> Flagged
    let level = sum(x_t) + state[0]
    let packed = [level, 1.0]
    return packed

define run = tok_embed >> scan(cell)

program lm : Token -> Flagged
    sample s <- run
    return s

export lm
"""


def test_a_scan_step_may_return_a_list_of_numbers_as_its_state() -> None:
    program = loads(_SCAN)
    tokens = torch.tensor([[1, 4, 2], [0, 3, 3]])
    recorded = trace(program.morphism, tokens, observations={})
    level = torch.zeros(2)
    for position in range(3):
        embedded = recorded.sites[f"s$x_in@{position}"].value
        state = recorded.sites[f"packed@{position}"].value
        assert tuple(state.shape) == (2, 2)
        level = embedded[:, 0] + level
        torch.testing.assert_close(state, torch.stack([level, torch.ones(2)], dim=1))
