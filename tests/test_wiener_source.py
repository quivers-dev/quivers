"""Wiener joint-law source contracts remain consistent across native and QIEC."""

import math

import pytest
import torch

from quivers.continuous import Wiener
from quivers.continuous.families import ConditionalWiener
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads, module_to_source, parse
from quivers.inference.trace import trace
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from tests.continuous.test_wiener import oracle
from tests.transpile.fixtures._generate_families import fixture_sources


@pytest.mark.parametrize("syntax", ["named", "typed"])
def test_joint_signed_source_likelihood_and_all_runtime_parameter_gradients(syntax):
    args = (
        "a, delay, bias, drift"
        if syntax == "positional"
        else "boundary=a, ndt=delay, bias=bias, drift=drift"
    )
    header = (
        "(a:Real, delay:Real, bias:Real, drift:Real, y:Tensor[Real]([Row]))"
        if syntax == "typed"
        else ""
    )
    source = f"""object Row : FinSet 4
program model{header} : Row -> Row
    observe y : Row <- Wiener({args})
    return y
export model
"""
    restored = module_to_source(parse(source))
    assert module_to_source(parse(restored)) == restored
    compiled = loads(restored)
    values = [1.2, 0.1, 0.4, 0.7]
    names = ["a", "delay", "bias", "drift"]
    params = {
        name: torch.tensor(value, dtype=torch.double, requires_grad=True)
        for name, value in zip(names, values, strict=True)
    }
    y = torch.tensor([-0.25, -0.8, 0.4, 1.1], dtype=torch.double)
    recorded = trace(
        compiled.morphism, torch.zeros(1, 1, dtype=torch.long), params | {"y": y}
    )
    assert recorded.log_joint is not None
    expected = math.fsum(float(oracle(*values, p)) for p in y.tolist())
    assert recorded.log_joint.sum().item() == pytest.approx(
        expected, rel=2e-6, abs=1e-6
    )
    actual = torch.autograd.grad(recorded.log_joint.sum(), tuple(params.values()))
    independent = Wiener(*params.values(), validate_args=False).log_prob(y).sum()
    expected_grad = torch.autograd.grad(independent, tuple(params.values()))
    for observed, target in zip(actual, expected_grad, strict=True):
        torch.testing.assert_close(observed, target, atol=2e-5, rtol=2e-6)
    reference = compiled.run(
        "model", data=dict(zip(names, values, strict=True)) | {"y": tuple(y.tolist())}
    )
    assert reference.log_joint == pytest.approx(expected, abs=2e-11)


def test_generator_fixture_scores_both_choices_and_is_registered():
    source = fixture_sources()["Wiener"]
    y = torch.tensor([-0.2, -0.35, 0.25, 0.6, -0.7, 1.2, -0.9, 0.4], dtype=torch.double)
    compiled = loads(source)
    recorded = trace(compiled.morphism, torch.zeros(1, 1, dtype=torch.long), {"y": y})
    expected = math.fsum(float(oracle(1.2, 0.1, 0.4, 0.7, p)) for p in y.tolist())
    assert recorded.log_joint is not None
    assert recorded.log_joint.sum().item() == pytest.approx(expected, rel=2e-6)
    assert compiled.run(
        "wiener_fixture", data={"y": tuple(y.tolist())}
    ).log_joint == pytest.approx(expected, abs=2e-11)


@pytest.mark.parametrize("target", available_targets())
def test_every_external_target_explicitly_refuses_wiener(target):
    with pytest.raises(UnsupportedConstruct) as error:
        transpile(parse(fixture_sources()["Wiener"]), target=target)
    assert any(
        "Wiener" in kind and kind.startswith("family:") for kind in error.value.kinds
    )


def test_conditional_native_family_scores_its_actual_joint_distribution():
    morphism = ConditionalWiener(
        Euclidean(name="Input", dim=2), Euclidean(name="Response", dim=1)
    )
    x = torch.tensor([[0.2, -0.1], [0.5, 0.4]])
    law = morphism._get_dist(x)
    assert isinstance(law, Wiener)
    y = (law.ndt + 0.3) * torch.tensor([[-1.0], [1.0]], dtype=torch.double)
    expected = law.log_prob(y).sum(-1)
    torch.testing.assert_close(morphism.log_prob(x, y), expected)
    torch.manual_seed(1)
    sampled = morphism.rsample(x)
    assert sampled.shape == (2, 1)
    assert not sampled.requires_grad


def test_positional_wiener_source_arguments_follow_keyword_only_family_convention():
    from quivers.dsl.compiler import CompileError

    with pytest.raises(CompileError, match="keyword-only"):
        loads("""object X: Real 1
program model: X -> X
    observe y <- Wiener(1.2,0.1,0.4,0.7)
    return y
export model
""")
