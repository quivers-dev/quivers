"""Literal constructors use the same registered kernels and row packing as values."""

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import trace


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("rows", [2, 5])
def test_static_nested_count_family_matches_dynamic_and_independent_probability(rows):
    def compile(probability):
        header = "(p:Real)" if probability == "p" else ""
        return loads(f"""object Row:FinSet {rows}
program model{header}:Row->Row
    observe y:Row <- Normalize(base=Restrict(base=ZeroInflatedBinomial(zero_prob=0.2,total_count=4,probs={probability}),low=0,high=2))
    return y
export model
""")

    y = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)[:rows]
    p = torch.tensor(0.3, requires_grad=True)
    probabilities = (
        torch.distributions.Binomial(total_count=4, probs=p)
        .log_prob(torch.arange(5))
        .exp()
        * 0.8
    )
    probabilities = probabilities + torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0])
    expected = (probabilities[y].log() - probabilities[:3].sum().log()).sum()
    fixed, dynamic = compile("0.3"), compile("p")
    actual_fixed = trace(fixed.morphism, torch.zeros(1, 1), {"y": y}).log_joint.sum()
    actual_dynamic = trace(
        dynamic.morphism, torch.zeros(1, 1), {"y": y, "p": p}
    ).log_joint.sum()
    torch.testing.assert_close(actual_fixed, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(actual_dynamic, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual_dynamic, p)[0], torch.autograd.grad(expected, p)[0]
    )
    reference = fixed.run("model", data={"y": tuple(y.tolist())}, sites={}).log_joint
    assert reference == pytest.approx(expected.item(), abs=2e-12)


@pytest.mark.parametrize("rows", [2, 5])
def test_static_nested_response_time_family_retains_each_row_density(rows):
    from quivers.continuous._response_time import ExGaussian

    source = loads(f"""object Row:FinSet {rows}
program model:Row->Row
    observe y:Row <- Restrict(base=ExGaussian(loc=0.1,scale=0.7,rate=1.2),low=-1.0,high=2.0)
    return y
export model
""")
    y = torch.tensor([-0.2, 0.3, 0.7, 1.0, 1.4])[:rows]
    base = ExGaussian(torch.tensor(0.1), torch.tensor(0.7), torch.tensor(1.2))
    expected = (
        base.log_prob(y)
        - (base.cdf(torch.tensor(2.0)) - base.cdf(torch.tensor(-1.0))).log()
    ).sum()
    actual = trace(source.morphism, torch.zeros(1, 1), {"y": y}).log_joint.sum()
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    reference = source.run("model", data={"y": tuple(y.tolist())}, sites={}).log_joint
    assert reference == pytest.approx(expected.item(), abs=2e-12)
