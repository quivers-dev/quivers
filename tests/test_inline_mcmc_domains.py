"""Invalid inline parameters reject proposals without hiding program errors."""

import math

import pytest
import torch

from quivers.continuous.param_transforms import positive_parameter
from quivers.inference.mcmc import PotentialFn
from quivers.inference.mcmc.kernel import _rejects_position
from quivers.inference.registry import LatentRegistry
from tests.benchmarks.datasets import neal_funnel


def _potential() -> PotentialFn:
    data = neal_funnel()
    registry = LatentRegistry.from_model(data.model, {"x"})
    return PotentialFn(data.model, registry, torch.zeros(1, 1), data.observations)


def test_inline_scale_underflow_rejects_proposal() -> None:
    potential = _potential()
    assert torch.isfinite(potential.check_initial(torch.zeros(1)))
    value, grad = potential.value_and_grad(torch.tensor([-1000.0]))
    assert torch.isneginf(value)
    torch.testing.assert_close(grad, torch.zeros(1), rtol=0, atol=0)


def test_invalid_initial_scale_still_raises() -> None:
    with pytest.raises(ValueError, match="positive parameter"):
        _potential().check_initial(torch.tensor([-1000.0]))


def test_tiny_valid_scale_keeps_its_potential_gradient() -> None:
    value, grad = _potential().value_and_grad(torch.tensor([-40.0]))
    expected = -0.5 * (-40 / 3) ** 2 - math.log(3) - 5 * math.log(2 * math.pi) + 180
    torch.testing.assert_close(value, torch.tensor(expected), rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(grad, torch.tensor([40 / 9 - 4.5]), rtol=1e-4, atol=1e-6)


def test_parameter_validator_is_identified_by_its_raising_frame() -> None:
    with pytest.raises(ValueError) as genuine:
        positive_parameter(torch.tensor(0.0))
    assert _rejects_position(genuine.value)
    with pytest.raises(ValueError) as unrelated:
        raise ValueError(str(genuine.value))
    assert not _rejects_position(unrelated.value)
