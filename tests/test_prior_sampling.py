"""Prior workflows omit likelihoods while retaining genuine prior potentials."""

import math

import pytest
import torch

from quivers.dsl import loads
from quivers.effects import observation_scoring
from quivers.inference import trace
from quivers.continuous.program_steps import Score


SOURCE = """object Row : FinSet 3
object Out : Real 1
program p : Row -> Out
    sample b <- Normal(loc=0.0,scale=1.0)
    observe y:Row <- Normal(loc=b,scale=1.0) [weights=w]
    score potential = -0.5 * b * b
    return b
export p
"""


def test_prior_only_omits_observations_and_retains_prior_and_explicit_potential_gradients():
    program = loads(SOURCE).morphism
    b = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    data = {
        "b": b,
        "y": torch.tensor([0.2, 0.5, -0.1], dtype=torch.float64),
        "w": torch.tensor([0.1, 0.0, 2.0], dtype=torch.float64),
    }
    with observation_scoring(False):
        recorded = trace(program, torch.zeros(1, 1), data)
    expected = -0.5 * (b.square() + math.log(2 * math.pi)) - 0.5 * b.square()
    torch.testing.assert_close(recorded.log_joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(recorded.log_joint.sum(), b)[0], -2 * b
    )
    assert bool((recorded.sites["y"].log_prob == 0).all())
    raw = recorded.sites["y"].sampleable.raw_log_prob_pointwise(data["y"])
    torch.testing.assert_close(
        raw, torch.distributions.Normal(b, 1.0).log_prob(data["y"])
    )
    ordinary = trace(program, torch.zeros(1, 1), data)
    torch.testing.assert_close(ordinary.log_joint, expected + (data["w"] * raw).sum())


def test_prior_only_does_not_evaluate_an_inactive_invalid_likelihood():
    source = (
        SOURCE.replace("    observe y:Row", "    let sigma = exp(b)\n    observe y:Row")
        .replace("Normal(loc=b,scale=1.0) [weights=w]", "Normal(loc=b,scale=sigma)")
        .replace("    score potential = -0.5 * b * b\n", "")
    )
    program = loads(source).morphism
    data = {"b": torch.tensor([-1000.0]), "y": torch.ones(3)}
    with observation_scoring(False):
        recorded = trace(program, torch.zeros(1, 1), data)
    assert bool(torch.isfinite(recorded.log_joint).all())
    with pytest.raises(ValueError):
        trace(program, torch.zeros(1, 1), data)


def test_innermost_observation_policy_and_exception_restore_the_scope():
    program = loads(SOURCE).morphism
    data = {"b": torch.tensor([0.3]), "y": torch.zeros(3), "w": torch.ones(3)}
    ordinary = trace(program, torch.zeros(1, 1), data).log_joint
    with pytest.raises(RuntimeError):
        with observation_scoring(False):
            with observation_scoring(True):
                torch.testing.assert_close(
                    trace(program, torch.zeros(1, 1), data).log_joint, ordinary
                )
            raise RuntimeError("exit")
    torch.testing.assert_close(
        trace(program, torch.zeros(1, 1), data).log_joint, ordinary
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_compiler_classifies_marginalized_likelihoods_and_prior_policy_omits_them(
    grouped,
):
    from tests.dsl.test_marginal_observation_events import SOURCE as marginal

    source = (
        marginal
        if grouped
        else marginal.replace(
            "sample pi : Group <- Dirichlet(concentration=1.0) [over=Class, iid_over=Group]",
            "sample pi <- Dirichlet(concentration=1.0) [over=Class]",
        )
        .replace(" [over=Group]\n", "\n")
        .replace("via=group, ", "")
    )
    program = loads(source).morphism
    scores = [step for step in program.steps if isinstance(step, Score)]
    assert scores and all(step.role == "likelihood" for step in scores)
    mu = torch.tensor([-0.2, 0.7], dtype=torch.float64, requires_grad=True)
    pi = torch.tensor(
        [[0.3, 0.7], [0.6, 0.4]] if grouped else [0.3, 0.7],
        dtype=torch.float64,
        requires_grad=True,
    )
    data = {
        "mu": mu,
        "pi": pi,
        "y": torch.tensor([0.1, 0.4, -0.3, 0.2], dtype=torch.float64),
        "status": torch.tensor([0, 1, -1, 2]),
        "upper": torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float64),
        "weight": torch.ones(4),
        "group": torch.tensor([0, 0, 1, 1]),
    }
    with observation_scoring(False):
        recorded = trace(program, torch.zeros(1, 1), data)
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(mu).sum()
        + torch.distributions.Dirichlet(torch.ones(2, dtype=torch.float64))
        .log_prob(pi)
        .sum()
    )
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    torch.testing.assert_close(
        torch.autograd.grad(recorded.log_joint.sum(), mu)[0], -mu
    )
