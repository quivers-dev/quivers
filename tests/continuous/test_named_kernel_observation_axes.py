"""Declared kernel coordinates are consumed once by indexed observations."""

from pathlib import Path

import pytest
import torch

from quivers.dsl import loads
from tests.transpile._gallery_data import _DATA_BLOCK_RE


@pytest.mark.parametrize("width", [1, 3, 5])
def test_named_normal_kernel_owns_its_declared_response_coordinates(width):
    source = f"""object Row : FinSet 4
object Input : Real 2
object Output : Real {width}
morphism decoder : Input -> Output [param_source=mlp(3)] ~ Normal
program p : Row -> Output
    sample z : Row <- Normal(loc=0.0,scale=1.0) [over=Input]
    observe y : Row <- decoder(z)
    return y
export p
export decoder
"""
    compiled = loads(source)
    observation = compiled.morphism.steps[1].morphism
    kernel = compiled.decoder
    assert observation.array_shape == ()
    assert observation.response.shape == ((4,) if width == 1 else (4, width))
    inputs = torch.arange(8, dtype=torch.float64).reshape(4, 2).div(5).requires_grad_()
    response = torch.arange(4 * width, dtype=torch.float64).reshape(4, width) / 7 - 0.3
    kernel.double()
    first, second = kernel.param_source.net[0], kernel.param_source.net[2]
    raw = (
        torch.tanh(inputs @ first.weight.T + first.bias) @ second.weight.T + second.bias
    )
    expected = (
        torch.distributions.Normal(raw[:, :width], raw[:, width:].exp())
        .log_prob(response)
        .sum(-1)
    )
    actual = observation.log_prob_pointwise(inputs, response)
    torch.testing.assert_close(actual, expected)
    parameters = (inputs, *kernel.parameters())
    actual_grad = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), parameters)
    for got, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, wanted)
    if width > 1:
        with pytest.raises(ValueError, match="expected.*for a"):
            observation.log_prob_pointwise(inputs, response.flatten())


def test_inline_scalar_law_retains_its_explicit_independent_array_axes():
    source = """object Row : FinSet 3
object Col : FinSet 2
object One : FinSet 1
program p(mu:Real) : Row -> Row
    observe y : Row <- Normal(loc=mu,scale=1.0) [over=[Col,One]]
    return y
export p
"""
    observation = loads(source).morphism.steps[0].morphism
    assert observation.array_shape == (2, 1)
    values = torch.arange(6, dtype=torch.float64).reshape(3, 2, 1) / 4
    means = torch.full((3, 1), 0.7, dtype=torch.float64)
    actual = observation.log_prob_pointwise(means, values)
    expected = torch.distributions.Normal(0.7, 1.0).log_prob(values)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_actual_vae_gallery_loader_and_data_snippet_match_the_handwritten_joint():
    path = Path(__file__).resolve().parents[2] / "docs/examples/vae.md"
    matched = _DATA_BLOCK_RE.search(path.read_text())
    assert matched is not None
    with torch.random.fork_rng(devices=[]):
        namespace = {"__name__": "__gallery_data__"}
        exec(compile(matched.group(1), str(path), "exec"), namespace)
    assert torch.allclose(
        namespace["log_joint_true"], namespace["hand"].reshape(1), rtol=1e-5
    )
