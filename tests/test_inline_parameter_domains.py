"""Inline family parameters preserve valid values near their boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
import torch.distributions as distributions

from quivers.continuous.family_spec import ParamSpec
from quivers.dsl import loads


@pytest.fixture(autouse=True)
def double_precision_parameters() -> Iterator[None]:
    """Use representable interior parameters in both construction paths."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("transform", ["softplus", "softplus_shifted", "exp"])
def test_positive_inline_parameters_are_preserved(transform: str) -> None:
    """Valid positive values retain their value and derivative."""
    parameter = ParamSpec(name="scale", transform=transform)
    values = torch.tensor(
        [1e-14, 1e-10, 1e-7, 0.5], dtype=torch.float64, requires_grad=True
    )
    actual = parameter.inline_clamp(values)
    torch.testing.assert_close(actual, values, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), values)[0], torch.ones_like(values)
    )


@pytest.mark.parametrize(
    "family,parameters,value",
    [
        ("Normal", {"loc": 0.0, "scale": 1e-10}, 0.0),
        ("LogNormal", {"loc": 0.0, "scale": 1e-10}, 1.0),
        ("HalfNormal", {"scale": 1e-10}, 5e-11),
        ("HalfCauchy", {"scale": 1e-10}, 2e-10),
        ("Exponential", {"rate": 1e-10}, 3.0),
        ("Gamma", {"concentration": 1e-10, "rate": 0.7}, 0.8),
        ("Gamma", {"concentration": 2.0, "rate": 1e-10}, 1.0),
        ("Beta", {"concentration1": 1e-10, "concentration0": 0.7}, 0.3),
        ("Laplace", {"loc": 0.0, "scale": 1e-10}, 2e-10),
        ("StudentT", {"df": 1e-10, "loc": 0.0, "scale": 1.0}, 0.5),
        ("Weibull", {"scale": 1e-10, "concentration": 2.0}, 5e-11),
        ("Geometric", {"probs": 1e-10}, 0.0),
        ("NegativeBinomial", {"total_count": 2.0, "probs": 1.0 - 1e-10}, 0.0),
    ],
)
@pytest.mark.parametrize("literal", [False, True])
def test_inline_density_preserves_valid_parameters(
    family: str, parameters: dict[str, float], value: float, literal: bool
) -> None:
    """Literal and runtime arguments agree with independent distributions."""
    arguments = ", ".join(
        f"{name}={number!r}" if literal else f"{name}=p_{name}"
        for name, number in parameters.items()
    )
    source = f"""
object Resp : FinSet 2
object Out : Real 1
program p : Resp -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
    observe y : Resp <- {family}({arguments})
    return a
export p
"""
    dtype = torch.float64
    tensors = {
        name: torch.tensor(number, dtype=dtype) for name, number in parameters.items()
    }
    values = torch.full((2,), value, dtype=dtype)
    observations = {"a": torch.tensor(0.0), "y": values}
    observations.update({f"p_{name}": tensor for name, tensor in tensors.items()})
    distribution = getattr(distributions, family)
    expected = distribution(**tensors).log_prob(values).sum() - 0.5 * math.log(
        2 * math.pi
    )
    actual = loads(source).morphism.log_joint(torch.zeros(1, 1), observations)[0]
    torch.testing.assert_close(
        actual.to(expected.dtype), expected, rtol=1e-5, atol=1e-5
    )
    clipped = {
        name: tensor.clamp(1e-7, 1.0 - 1e-7)
        if name == "probs"
        else tensor.clamp_min(1e-7)
        if name != "loc"
        else tensor
        for name, tensor in tensors.items()
    }
    wrong = distribution(**clipped).log_prob(values).sum() - 0.5 * math.log(2 * math.pi)
    assert not torch.isclose(expected, wrong, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("value", [-20.0, -32.0, -40.0, -60.0])
def test_conditioned_funnel_joint_and_gradient(value: float) -> None:
    """The conditioned joint has the Gaussian conditional's exact slope."""
    source = """
object Latent : FinSet 9
object Out : Real 1
program funnel : Latent -> Out
    sample v <- Normal(loc=0.0, scale=3.0)
    let sigma = exp(0.5 * v)
    observe x : Latent <- Normal(loc=0.0, scale=sigma)
    return v
export funnel
"""
    v = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    actual = loads(source).morphism.log_joint(
        torch.zeros(1, 1), {"v": v, "x": torch.zeros(9, dtype=torch.float64)}
    )[0]
    expected = -0.5 * (v / 3) ** 2 - math.log(3) - 5 * math.log(2 * math.pi) - 4.5 * v
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    gradient = torch.autograd.grad(actual, v)[0]
    torch.testing.assert_close(gradient, -v / 9 - 4.5, rtol=0, atol=1e-6)
