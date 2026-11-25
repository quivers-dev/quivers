"""Counting work changes no bit of any result."""

from __future__ import annotations

import pytest
import torch

from tests.evidence.compare import Mismatch
from tests.evidence.counters import CallCounter, TensorCounter, forwarding, saved_elements
from tests.evidence.mutations import Mutation, Rewrite, installed
from tests.evidence.runs import RUNS, Result, same_bits
from tests.evidence.checks import deduction
from tests.evidence.problems import DEDUCTIONS
from quivers.stochastic.agenda import ChartView, match
from quivers.stochastic.semiring import LOG_PROB, ChartSemiring


def counted(name: str) -> Result:
    """Run an executor under every counter at once.

    Parameters
    ----------
    name : str
        The executor.

    Returns
    -------
    Result
        The result.
    """
    with CallCounter({"match": match}), saved_elements(), TensorCounter():
        return RUNS[name]()


@pytest.mark.parametrize("name", sorted(RUNS))
def test_counted_run_is_bitwise_equal(name: str) -> None:
    """Require a counted run to equal a plain run bit by bit.

    Parameters
    ----------
    name : str
        The executor.
    """
    assert same_bits(RUNS[name](), counted(name))


def forwarded(semiring: ChartSemiring) -> Result:
    """Run the agenda on a problem in a given semiring.

    Parameters
    ----------
    semiring : ChartSemiring
        The semiring.

    Returns
    -------
    Result
        The weight and its gradients.
    """
    problem = DEDUCTIONS["ambiguous"]
    leaves, parameters = deduction.leaves_of(problem, "logprob", requires_grad=True)
    value = ChartView(deduction.run(problem, semiring, "fifo", leaves)).goal_weight()
    found = torch.autograd.grad(value, list(parameters.values()))
    return [value.detach(), *found]


def check_forwarding() -> None:
    """Require the forwarding semiring to equal its base bit by bit.

    Raises
    ------
    Mismatch
        If a bit differs.
    """
    counting, counts = forwarding(LOG_PROB)
    if not same_bits(forwarded(LOG_PROB), forwarded(counting)):
        raise Mismatch("value", float("inf"), "a counted run differs")
    assert counts["sums"] and counts["products"]


def test_forwarding_semiring_is_bitwise_equal() -> None:
    """Require counted semiring operations to change no bit."""
    check_forwarding()


REASSOCIATED = Mutation(
    name="counter-reassociates-a-sum",
    defect_class="instrumentation",
    provenance="a counter that recomputes a sum in its own order",
    target="tests.evidence.counters:forwarding",
    rewrites=(
        Rewrite(
            old="result = parent.plus(self, scores, dim)",
            new="peak = scores.amax(dim=dim, keepdim=True)\n"
            "        result = (scores - peak).exp().sum(dim=dim).log() + peak.squeeze(dim)",
        ),
    ),
    rejected_by=("forwarding",),
    min_effect=1.0,
)
"""An instrument that changes the value it measures."""


def test_instrument_that_changes_a_value_is_rejected() -> None:
    """Require the bitwise check to reject a counter that reassociates a sum."""
    with installed(REASSOCIATED):
        with pytest.raises(Mismatch):
            check_forwarding()
