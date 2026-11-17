"""Checked weight primitives retain each Python host's arrays and gradients."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import transpile
from tests.transpile.test_observation_weight_host_bridges import _bridge

SOURCE = """define checked(w : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return observation_weight(w)
define scaled(w : Tensor[Real]([3]), ell : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return weight_value(scale_weight(observation_weight(w), weight(ell)))
"""


def test_generated_pyro_helper_preserves_host_tensor_values_and_gradients() -> None:
    namespace: dict[str, object] = {}
    exec(transpile(parse(SOURCE), target="pyro"), namespace)
    checked = cast(Callable, namespace["qiec_checked"])
    scaled = cast(Callable, namespace["qiec_scaled"])
    weights = torch.tensor([0.0, 2.0, 0.5], dtype=torch.float64, requires_grad=True)
    density = torch.tensor(
        [-torch.inf, -3.0, -4.0], dtype=torch.float64, requires_grad=True
    )
    assert checked(weights) is weights
    result = scaled(weights, density)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([0.0, -6.0, -2.0], dtype=torch.float64))
    left, right = torch.autograd.grad(result.sum(), (weights, density))
    assert torch.equal(left, torch.tensor([0.0, -3.0, -4.0], dtype=torch.float64))
    assert torch.equal(right, torch.tensor([0.0, 2.0, 0.5], dtype=torch.float64))
    for value in (
        torch.tensor([0.0, -1.0, 1.0]),
        torch.tensor([0.0, torch.inf, 1.0]),
        torch.tensor([0.0, torch.nan, 1.0]),
        torch.tensor([True, False, True]),
        torch.tensor([0, 1, 2]),
    ):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            checked(value)


def test_python_dispatch_selects_host_implementations_inside_mixed_tuples() -> None:
    primitive = _bridge("pyro", torch=torch)
    first = torch.tensor(0.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)
    checked = primitive("observation_weight", ((first, second),))
    result = primitive("scale_weight", (checked, (-torch.inf, -3.0)))
    assert all(isinstance(value, torch.Tensor) for value in result)
    total = torch.stack(result).sum()
    assert total.item() == -6.0
    gradients = torch.autograd.grad(total, (first, second))
    assert [value.item() for value in gradients] == [0.0, -3.0]


@pytest.mark.parametrize("host_coefficient", [True, False])
def test_python_host_vector_and_canonical_tensor_share_elementwise_positions(
    host_coefficient: bool,
) -> None:
    primitive = _bridge("pyro", torch=torch)
    coefficient = (0.0, 2.0, 0.5)
    density = (-float("inf"), -3.0, -4.0)
    host_value = torch.tensor(
        coefficient if host_coefficient else density, requires_grad=True
    )
    result = primitive(
        "scale_weight",
        (host_value, density) if host_coefficient else (coefficient, host_value),
    )
    value = torch.stack(result) if isinstance(result, tuple) else result
    assert tuple(value.shape) == (3,)
    assert torch.equal(value, torch.tensor([0.0, -6.0, -2.0]))
    (gradient,) = torch.autograd.grad(value.sum(), (host_value,))
    expected = [0.0, -3.0, -4.0] if host_coefficient else [0.0, 2.0, 0.5]
    assert torch.equal(gradient, torch.tensor(expected))


def test_tuple_of_live_host_scalars_and_host_vector_retain_every_gradient() -> None:
    primitive = _bridge("pyro", torch=torch)
    weights = tuple(
        torch.tensor(value, requires_grad=True) for value in (0.0, 2.0, 0.5)
    )
    density = torch.tensor([-torch.inf, -3.0, -4.0], requires_grad=True)
    result = primitive("scale_weight", (weights, density))
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([0.0, -6.0, -2.0]))
    gradients = torch.autograd.grad(result.sum(), (*weights, density))
    assert [gradient.item() for gradient in gradients[:3]] == [0.0, -3.0, -4.0]
    assert torch.equal(gradients[3], torch.tensor([0.0, 2.0, 0.5]))
