"""Immutable kernel-state updates and unchanged checkpoint values."""

import didactic.api as dx
import pytest
import torch

from quivers.inference.mcmc import KernelState, PotentialFn
from quivers.inference.mcmc.kernel import MCMCKernel
from quivers.inference.registry import LatentRegistry
from tests.test_inference_mcmc import _standard_normal_program


def test_first_evaluation_returns_a_state_without_mutating_input() -> None:
    """Initial density and gradient agree with the closed-form Gaussian."""
    model = _standard_normal_program(2)
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    position = torch.tensor([0.5, -1.25], dtype=torch.float64)
    state = KernelState(
        position=position,
        log_density=torch.tensor(0.0, dtype=torch.float64),
        grad_log_density=torch.zeros_like(position),
    )
    evaluated = MCMCKernel._evaluate_first(state, potential)
    expected = (
        torch.distributions.Normal(
            torch.zeros_like(position), torch.ones_like(position)
        )
        .log_prob(position)
        .sum()
    )
    torch.testing.assert_close(evaluated.log_density, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(evaluated.grad_log_density, -position, rtol=0, atol=0)
    assert evaluated.position is position
    assert evaluated is not state
    assert state.log_density == 0
    torch.testing.assert_close(state.grad_log_density, torch.zeros_like(position))
    advanced = evaluated.with_(step_count=1)
    assert MCMCKernel._evaluate_first(advanced, potential) is advanced
    assert evaluated.step_count == 0
    restored = KernelState.from_state_dict(advanced.state_dict())
    torch.testing.assert_close(restored.position, advanced.position, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.log_density, advanced.log_density, rtol=0, atol=0
    )
    torch.testing.assert_close(restored.grad_log_density, -position, rtol=0, atol=0)
    assert restored.step_count == 1
    with pytest.raises(dx.ValidationError, match="torch.Tensor"):
        state.with_(position=[0.5, -1.25])
    payload = advanced.state_dict()
    payload["extras"] = {1: 3}
    with pytest.raises(ValueError, match="string-keyed dictionary"):
        KernelState.from_state_dict(payload)
