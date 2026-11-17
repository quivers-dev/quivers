"""Lexical observations score computed data without adding guide coordinates."""

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace


def _program(weighted=False):
    weight = " [weights=weights]" if weighted else ""
    return loads(
        "object Row : FinSet 3\n"
        "program model(raw: Tensor[Real]([Row]), weights: Tensor[Real]([Row])) : Row -> Row\n"
        "    sample theta <- Normal(loc=0.0, scale=1.0)\n"
        "    let y = 2.0 * raw + theta\n"
        f"    observe y : Row <- Normal(loc=theta, scale=1.0){weight}\n"
        "    return y\n"
        "export model\n"
    ).morphism.double()


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_computed_plate_density_gradient_and_static_dynamic_registry(
    weighted, override
):
    program = _program(weighted)
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    raw = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    weights = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    observations = dict(raw=raw, weights=weights, theta=theta)
    if override:
        observations["y"] = torch.tensor([-0.5, 0.2, 0.7], dtype=torch.float64)
    recorded = trace(program, torch.zeros(1, 1), observations)
    value = observations["y"] if override else 2 * raw + theta
    row_weights = weights if weighted else torch.ones_like(weights)
    expected = torch.distributions.Normal(theta, 1.0).log_prob(value) * row_weights
    response = recorded.sites["y"]
    assert response.is_observed and not response.is_deterministic
    torch.testing.assert_close(response.value, value)
    torch.testing.assert_close(response.log_prob.sum(), expected.sum())
    actual_gradient = torch.autograd.grad(
        response.log_prob.sum(), theta, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected.sum(), theta)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    static = LatentRegistry.from_model(program, {"raw", "weights"})
    dynamic = LatentRegistry.from_trace(
        program, trace(program, torch.zeros(1, 1), dict(raw=raw, weights=weights))
    )
    assert static.names == dynamic.names == ("theta",)
    assert static.observed_names == {"y", "raw", "weights"}
    assert dynamic.observed_names == {"y"}


@pytest.mark.parametrize("weighted", [False, True])
def test_computed_response_has_no_variational_coordinates(weighted):
    program = _program(weighted)
    guide = AutoNormalGuide(program, observed_names={"raw", "weights"})
    assert guide.registry.names == ("theta",)
    observations = {
        "raw": torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64),
        "weights": torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64),
    }
    runner = SVI(program, guide, torch.optim.Adam(guide.parameters()), ELBO())
    losses = runner.run(torch.zeros(1, 1), observations, num_steps=3)
    assert len(losses) == 3 and all(
        torch.isfinite(torch.tensor(loss)) for loss in losses
    )


def test_scanned_observation_recomputes_each_position_and_each_run():
    program = loads(
        "object Val : Real 1\n"
        "program cell(u, previous) : Val * Val -> Val\n"
        "    let y = 2.0 * u\n"
        "    observe y <- Normal(loc=previous, scale=1.0)\n"
        "    return y\n"
        "define run = scan(cell, init=initial)\n"
        "program model : Val -> Val\n"
        "    let initial = 0.0\n"
        "    sample last <- run\n"
        "    return last\n"
        "export model\n"
    ).morphism.double()
    inputs = torch.tensor(
        [[[0.1], [0.4], [-0.2]], [[0.8], [-0.1], [0.3]]], dtype=torch.float64
    )
    recorded = trace(program, inputs)
    expected_values = inputs * 2
    total = torch.zeros(2, dtype=torch.float64)
    previous = torch.zeros(2, 1, dtype=torch.float64)
    for position in range(3):
        response = recorded.sites[f"y@{position}"]
        assert response.is_observed
        value = expected_values[:, position]
        torch.testing.assert_close(response.value, value)
        total += torch.distributions.Normal(previous, 1.0).log_prob(value).sum(-1)
        previous = value
    torch.testing.assert_close(recorded.log_joint.reshape(-1), total)
    assert not LatentRegistry.from_trace(program, recorded).names
