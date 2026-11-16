"""Measure normalization must retain specific external capability diagnoses."""

import math

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec.program_runtime import run_program
from quivers.qiec.terms import Bind, Comprehension, DistributionValue, LogDensity
from quivers.qiec.terms import Perform, WeightSum
from quivers.transpile import UnsupportedConstruct, available_targets, transpile


def _source(family: str, plated: bool) -> str:
    """Construct the same measure with scalar or row-indexed observations."""
    expression = (
        "Mixture(weights=[3.0, 7.0], components=[Normal(loc=-1.0, scale=0.5), "
        "Normal(loc=1.0, scale=0.5)])"
        if family == "Mixture"
        else "Pushforward(base=Normal(loc=0.0, scale=1.0), bijector=Exp)"
    )
    axis = " : Row" if plated else ""
    return f"""object Row : FinSet 3
program model : Row -> Row
    observe y{axis} <- {expression}
    return y
export model
"""


@pytest.mark.parametrize("target", sorted(available_targets()))
@pytest.mark.parametrize("family", ["Mixture", "Transformed"])
@pytest.mark.parametrize("plated", [False, True])
def test_nested_measure_refusal_names_the_construction_through_its_plate(
    target: str, family: str, plated: bool
) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(family, plated)), target=target)
    assert len(caught.value.kinds) == 1
    assert caught.value.kinds[0].startswith(f"nested-distribution-arg:{family}:")


@pytest.mark.parametrize("family", ["Mixture", "Transformed"])
def test_normalized_product_density_is_preserved_before_external_refusal(
    family: str,
) -> None:
    source = _source(family, True)
    module = lower_qvr_to_qiec(parse(source))
    body = module.computations[-1].body
    assert isinstance(body, Bind) and isinstance(body.first, Perform)
    weight = body.first.request.arguments[0]
    assert isinstance(weight, WeightSum) and isinstance(weight.value, LogDensity)
    normalized = weight.value.sampleable
    assert isinstance(normalized, DistributionValue)
    assert normalized.name == "Normalize"
    product = dict(normalized.arguments)["base"]
    assert isinstance(product, Comprehension)
    assert isinstance(product.body, DistributionValue)
    assert product.body.name == family

    values = torch.tensor([0.5, 1.0, 2.0])
    if family == "Mixture":
        # Original weights sum to ten: dropping Normalize changes the score
        # by three log(10), so the oracle also checks normalization per row.
        components = torch.distributions.Normal(
            torch.tensor([-1.0, 1.0]), torch.tensor([0.5, 0.5])
        )
        expected = torch.logsumexp(
            components.log_prob(values[:, None]) + torch.tensor([0.3, 0.7]).log(),
            dim=-1,
        ).sum()
        raw = expected + len(values) * math.log(10)
        assert not torch.isclose(raw, expected)
    else:
        expected = torch.distributions.LogNormal(0.0, 1.0).log_prob(values).sum()
    reference = run_program(
        module, "model", data={"y": tuple(values.tolist())}, sites={}
    )
    assert reference.log_joint == pytest.approx(float(expected), abs=1e-6)
    native = trace(
        loads(source).morphism,
        torch.zeros(1, 1, dtype=torch.long),
        {"y": values},
    )
    assert native.log_joint is not None
    torch.testing.assert_close(native.log_joint.squeeze(), expected)
