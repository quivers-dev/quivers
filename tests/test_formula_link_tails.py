"""Released brms links preserve values and derivatives in both tails."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import FormulaToQVRModule, families, formula_from_data
from quivers.formulas._posterior import response_distribution
from quivers.inference import trace


@pytest.mark.parametrize("link", ["squareplus", "softit", "softplus"])
@pytest.mark.parametrize("value", [-80.0, -2.0, 0.0, 2.0, 80.0, 1000.0])
def test_links_match_independent_first_and_second_derivatives(link, value):
    parsed = formula_from_data("y ~ 1", pd.DataFrame({"y": [0.0, 1.0]}))
    lens = FormulaToQVRModule(
        families["gaussian"].with_links(mu=link), fixed_parameters={"sigma": 1.0}
    )
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    loads(module_to_source(module))
    eta = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    recorded = trace(
        program, torch.zeros(1, 1), {"intercept": eta, "y": torch.zeros(2)}
    )
    actual = response_distribution(recorded, "y").loc.reshape(-1)[0]
    first = torch.autograd.grad(actual, eta, create_graph=True)[0]
    second = torch.autograd.grad(first, eta)[0]
    if link == "squareplus":
        radius = math.hypot(value, 2)
        expected = (value + radius) / 2 if value >= 0 else 2 / (radius - value)
        expected_first = (1 + value / radius) / 2
        expected_second = 2 / radius**3
    else:
        positive = max(value, 0) + math.log1p(math.exp(-abs(value)))
        sigmoid = 1 / (1 + math.exp(-value))
        curvature = sigmoid * (1 - sigmoid)
        if link == "softplus":
            expected, expected_first, expected_second = positive, sigmoid, curvature
        else:
            expected = positive / (1 + positive)
            expected_first = sigmoid / (1 + positive) ** 2
            expected_second = (
                curvature / (1 + positive) ** 2 - 2 * sigmoid**2 / (1 + positive) ** 3
            )
    assert actual.item() == pytest.approx(expected, rel=2e-13, abs=1e-30)
    assert first.item() == pytest.approx(expected_first, rel=2e-10, abs=1e-16)
    assert second.item() == pytest.approx(expected_second, rel=2e-10, abs=1e-16)


@pytest.mark.parametrize(
    "dtype,magnitude", [(torch.float32, 1e30), (torch.float64, 1e300)]
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_squareplus_avoids_overflow_cancellation_and_lost_positive_tail_gradient(
    dtype, magnitude, sign
):
    parsed = formula_from_data("y ~ 1", pd.DataFrame({"y": [0.0]}))
    lens = FormulaToQVRModule(
        families["gaussian"].with_links(mu="squareplus"),
        fixed_parameters={"sigma": 1.0},
    )
    module, _ = lens.forward(parsed)
    eta = torch.tensor(sign * magnitude, dtype=dtype, requires_grad=True)
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1),
        {"intercept": eta, "y": torch.zeros(1)},
    )
    value = response_distribution(recorded, "y").loc.reshape(-1)[0]
    assert torch.isfinite(value) and value > 0
    expected = magnitude if sign > 0 else 1 / magnitude
    assert value.item() == pytest.approx(expected, rel=3e-6, abs=0)
    gradient = torch.autograd.grad(value, eta)[0]
    assert torch.isfinite(gradient)
    if sign > 0:
        assert gradient.item() == pytest.approx(1.0, rel=3e-6)
