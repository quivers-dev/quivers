"""A host datum combined with a matrix product takes the product's shape."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.inference.trace import trace

DESIGN = torch.tensor(
    [[1.0, 0.5], [0.2, -1.0], [-0.3, 0.7], [2.0, 0.1]], dtype=torch.float64
)
OFFSET = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float64)
RESPONSE = torch.tensor([0.5, 0.2, -0.1, 1.0], dtype=torch.float64)
COEFFICIENTS = torch.tensor([0.6, -0.4], dtype=torch.float64)
MIXING = torch.tensor([[1.0, 0.3], [-0.2, 0.8]], dtype=torch.float64)

HEAD = """
object Row : FinSet 4
object Basis : FinSet 2
program model({inputs}) : Row -> Row
    sample b : Basis <- Normal(loc=0.0, scale=1.0)
{body}
    observe y : Row <- Normal(loc=eta, scale=1.0)
    return y
export model
"""

CASES = {
    "vector": (
        "d : Tensor[Real]([Row, Basis])",
        "    let eta = o + matmul(d, b)",
        lambda: OFFSET + DESIGN @ COEFFICIENTS,
    ),
    "bound": (
        "d : Tensor[Real]([Row, Basis])",
        "    let r = matmul(d, b)\n    let eta = o + r",
        lambda: OFFSET + DESIGN @ COEFFICIENTS,
    ),
    "chained": (
        "d : Tensor[Real]([Row, Basis]), m : Tensor[Real]([Basis, Basis])",
        "    let eta = o * matmul(matmul(d, m), b)",
        lambda: OFFSET * (DESIGN @ MIXING @ COEFFICIENTS),
    ),
    "reduced": (
        "d : Tensor[Real]([Row, Basis])",
        "    let eta = o + sum(matmul(d, b))",
        lambda: OFFSET + (DESIGN @ COEFFICIENTS).sum(),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_a_datum_meets_a_matrix_product(name):
    """The joint density is the one of the written linear predictor."""
    inputs, body, mean = CASES[name]
    program = (
        Compiler(parse(HEAD.format(inputs=inputs, body=body))).compile().morphism
    )
    data = {"d": DESIGN, "o": OFFSET, "y": RESPONSE, "m": MIXING}
    values = {key: value for key, value in data.items() if key != "m" or "m :" in inputs}
    value = trace(program, torch.zeros(1, 1), values | {"b": COEFFICIENTS}).log_joint
    assert value is not None
    normal = torch.distributions.Normal
    expected = (
        normal(0.0, 1.0).log_prob(COEFFICIENTS).sum()
        + normal(mean(), 1.0).log_prob(RESPONSE).sum()
    )
    torch.testing.assert_close(value.sum().double(), expected)
