"""Replay independent observations of superposition and product-space completion."""

from itertools import product
from math import log, pi
from pathlib import Path

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.qiec.builtins import RANDOM_SAMPLE
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads


_DOC = Path(__file__).resolve().parents[2] / "docs/api/qiec/effects.md"


def _source() -> str:
    section = _DOC.read_text().split(
        "## Superposition, repeated draws, and completion", 1
    )[1]
    return section.split("\n```qvr\n", 1)[1].split("\n```", 1)[0] + "\n"


@pytest.fixture(scope="module")
def module() -> QiecModule:
    result = loads(dumps(load_source(_source()).qiec))
    assert isinstance(result, QiecModule)
    return result


def _addresses(run):
    return tuple(
        event.detail["address"]
        for event in run.result.trace
        if event.event == "operation.requested"
        and event.detail["operation"] == str(RANDOM_SAMPLE)
    )


def test_mixture_replay_has_one_density_and_resumption_has_two_draws(module):
    theta = torch.tensor([0.4, -0.4, 0.8], dtype=torch.float64, requires_grad=True)
    p, left, right = theta.unbind()
    inputs = dict(
        data={"p": p, "mu_a": left, "mu_b": right}, distributions=TensorBackend()
    )
    actual = run_program(module, "superposed", sites={"x": 0.31}, **inputs)
    branches = torch.stack((p.log(), (-p).log1p())) - (theta[1:] - 0.31).square() / 2
    expected = branches.logsumexp(0) - 0.5 * log(2 * pi)
    torch.testing.assert_close(actual.log_joint, expected)
    assert len(_addresses(actual)) == 1
    (actual_gradient,) = torch.autograd.grad(actual.log_joint, theta, retain_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected, theta)
    torch.testing.assert_close(actual_gradient, expected_gradient)

    separate = run_program(
        module, "two_shots", sites={"x@0": 0.31, "x@1": -0.2}, **inputs
    )
    answer = p * 0.31 + (1 - p) * -0.2
    density = -0.5 * ((left - 0.31).square() + (right + 0.2).square()) - log(2 * pi)
    torch.testing.assert_close(separate.value, answer)
    torch.testing.assert_close(separate.log_joint, density)
    first, second = _addresses(separate)
    assert first[:2] == second[:2] and (first[2], second[2]) == ((0,), (1,))
    (actual_gradient,) = torch.autograd.grad(
        separate.log_joint, theta, retain_graph=True
    )
    (expected_gradient,) = torch.autograd.grad(density, theta)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("first,active", tuple(product((False, True), repeat=2)))
def test_summing_inactive_completion_sites_preserves_mass_and_its_derivative(
    module, first, active
):
    theta = torch.tensor([0.4, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    q, left, right = theta.unbind()
    data = {"p": q, "p_left": left, "p_right": right}
    active_name, inactive_name = ("left", "right") if first else ("right", "left")
    selected = {"first": first, active_name: active}
    args = dict(data=data, distributions=TensorBackend())
    disjoint = run_program(module, "disjoint", sites=selected, **args)
    extensions = tuple(
        run_program(
            module, "completed", sites={**selected, inactive_name: extra}, **args
        )
        for extra in (False, True)
    )
    expected_q = q if first else 1 - q
    probability = left if first else right
    expected = expected_q.log() + (probability if active else 1 - probability).log()
    marginal = torch.stack(tuple(run.log_joint for run in extensions)).logsumexp(0)
    torch.testing.assert_close(disjoint.log_joint, expected)
    torch.testing.assert_close(marginal, expected)
    assert disjoint.value == float(active) and len(_addresses(disjoint)) == 2
    for run in extensions:
        assert run.value == float(active) and len(set(_addresses(run))) == 3
    (reference,) = torch.autograd.grad(expected, theta, retain_graph=True)
    (current,) = torch.autograd.grad(disjoint.log_joint, theta, retain_graph=True)
    (summed,) = torch.autograd.grad(marginal, theta)
    torch.testing.assert_close(current, reference)
    torch.testing.assert_close(summed, reference)


def test_a_nonunit_auxiliary_score_changes_the_marginal(module):
    source = _source().replace(
        "    sample right <- Bernoulli(probs=p_right)\n",
        "    sample right <- Bernoulli(probs=p_right)\n    score extra = log(2.0)\n",
    )
    changed = load_source(source).qiec
    assert isinstance(changed, QiecModule)
    data = {"p": 0.4, "p_left": 0.2, "p_right": 0.8}
    event = {"first": True, "left": True}
    disjoint = run_program(module, "disjoint", data=data, sites=event)
    altered = torch.stack(
        tuple(
            run_program(
                changed, "completed", data=data, sites={**event, "right": extra}
            ).log_joint
            for extra in (False, True)
        )
    ).logsumexp(0)
    torch.testing.assert_close(altered, disjoint.log_joint + log(2.0))
