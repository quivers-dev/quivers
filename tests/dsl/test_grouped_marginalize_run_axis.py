"""Grouped marginals score one run at a time under the tensor executor.

Checked observations identify a leading run axis. The executor separates
independent runs before evaluating a grouped block; each result agrees with
the grouped mixture computed by hand.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads

_SOURCE = """\
object Component : FinSet 2
object Group : FinSet 2
object Item : FinSet 4
object Weights : Real 2
program grouped_mixture : Item -> Weights [effects=[Sample, Score, Marginal]]
    sample probs : Group <- Dirichlet(concentration=1.0) [over=Component, iid_over=Group]
    sample mu_k : Component <- Normal(loc=0.0, scale=5.0)
    sample sd_k : Component <- HalfNormal(scale=1.0)
    marginalize z : Component <- Categorical(probs=probs) [over=Group]
        observe y : Item <- Normal(loc=mu_k[z], scale=sd_k[z]) [via=group_idx]
    return probs
export grouped_mixture
"""

_PROBS = [[0.25, 0.75], [0.5, 0.5]]
_MU = [0.0, 2.0]
_Y = [0.125, 0.25, 1.5, 2.0]
_IDX = [0, 0, 1, 1]


def _expected(
    observations: Sequence[float] = _Y, means: Sequence[float] = _MU
) -> float:
    """The joint in double precision: priors plus the grouped mixture.

    Parameters
    ----------
    observations : Sequence[float]
        Responses for one run.
    means : Sequence[float]
        Component locations for that run.

    Returns
    -------
    float
        The log joint at the fixed latents.
    """
    f64 = torch.float64
    prior = sum(
        float(
            D.Dirichlet(torch.ones(2, dtype=f64)).log_prob(torch.tensor(row, dtype=f64))
        )
        for row in _PROBS
    )
    prior += float(
        D.Normal(torch.tensor(0.0, dtype=f64), 5.0)
        .log_prob(torch.tensor(means, dtype=f64))
        .sum()
    )
    prior += 2 * float(
        D.HalfNormal(torch.tensor(1.0, dtype=f64)).log_prob(
            torch.tensor(1.0, dtype=f64)
        )
    )
    total = prior
    for group in range(2):
        terms = []
        for k in range(2):
            rows = [
                float(
                    D.Normal(torch.tensor(means[k], dtype=f64), 1.0).log_prob(
                        torch.tensor(observations[n], dtype=f64)
                    )
                )
                for n in range(4)
                if _IDX[n] == group
            ]
            terms.append(math.log(_PROBS[group][k]) + sum(rows))
        top = max(terms)
        total += top + math.log(sum(math.exp(t - top) for t in terms))
    return total


def _values(y: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "probs": torch.tensor([_PROBS]),
        "mu_k": torch.tensor([_MU]),
        "sd_k": torch.tensor([[1.0, 1.0]]),
        "group_idx": torch.tensor(_IDX),
        "y": y,
    }


@pytest.mark.parametrize("leading", [False, True])
def test_one_run_scores_the_grouped_mixture(leading: bool) -> None:
    program = loads(_SOURCE).morphism
    y = torch.tensor([_Y]) if leading else torch.tensor(_Y)
    joint = float(program.log_joint(torch.zeros(1, 1), _values(y))[0])
    assert joint == pytest.approx(_expected(), rel=2.0**-18)


def test_several_runs_score_independent_grouped_mixtures() -> None:
    program = loads(_SOURCE).morphism
    second_y = [0.0, 0.4, 1.7, 2.2]
    second_mu = [-0.5, 1.5]
    values = {
        "probs": torch.tensor([_PROBS, _PROBS]),
        "mu_k": torch.tensor([_MU, second_mu]),
        "sd_k": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "group_idx": torch.tensor(_IDX),
        "y": torch.tensor([_Y, second_y]),
    }
    actual = program.log_joint(torch.zeros(2, 1), values)
    expected = torch.tensor([_expected(), _expected(second_y, second_mu)])
    torch.testing.assert_close(actual, expected, rtol=2.0**-18, atol=2.0**-18)
