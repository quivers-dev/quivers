"""Correlated ALR densities, Gaussian ratio sampling, and affine event typing."""

import random

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import torch

from quivers.continuous import (
    AdditiveLogRatioNormal,
    ConditionalAdditiveLogRatioNormal,
    Euclidean,
    Simplex,
)
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.modeling import Model
from quivers.qiec.program_runtime import run_program
from quivers.qiec.reference_families import DENSITIES, SAMPLERS


@pytest.mark.parametrize("rows,width", [(2, 2), (3, 2), (2, 3)])
def test_density_jacobian_and_all_head_gradients(rows, width):
    loc = (
        torch.linspace(-0.3, 0.4, rows * width, dtype=torch.double)
        .reshape(rows, width)
        .requires_grad_()
    )
    raw = torch.eye(width, dtype=torch.double).expand(rows, width, width).clone()
    raw[..., 1, 0] = 0.3
    raw.requires_grad_()
    y = torch.arange(1, rows * (width + 1) + 1, dtype=torch.double).reshape(
        rows, width + 1
    )
    y = y / y.sum(-1, keepdim=True)
    dist = AdditiveLogRatioNormal(loc, raw)
    actual = dist.log_prob(y)
    expected = torch.tensor(
        [
            multivariate_normal.logpdf(np.log(v[:-1] / v[-1]), mean=m, cov=l @ l.T)
            - np.log(v).sum()
            for v, m, l in zip(
                y.numpy(), loc.detach().numpy(), raw.detach().numpy(), strict=True
            )
        ]
    )
    torch.testing.assert_close(actual, expected)

    def score(m, l):
        return AdditiveLogRatioNormal(m, l, validate_args=False).log_prob(y)

    assert torch.autograd.gradcheck(score, (loc, raw))
    assert torch.autograd.gradgradcheck(score, (loc, raw))
    gradient = torch.func.grad(lambda m: score(m, raw).sum())(loc)
    torch.testing.assert_close(
        gradient, torch.autograd.grad(actual.sum(), loc, retain_graph=True)[0]
    )
    _, tangent = torch.func.jvp(
        lambda m: score(m, raw), (loc,), (torch.ones_like(loc),)
    )
    torch.testing.assert_close(tangent, gradient.sum(-1))


def test_sample_ratios_recover_full_gaussian_and_expand():
    loc = torch.tensor([0.2, -0.4], dtype=torch.double)
    factor = torch.tensor([[0.8, 0.0], [0.35, 0.6]], dtype=torch.double)
    dist = AdditiveLogRatioNormal(loc, factor)
    with torch.random.fork_rng():
        torch.manual_seed(401)
        y = dist.rsample((60000,))
    ratios = (y[:, :-1] / y[:, -1:]).log()
    torch.testing.assert_close(ratios.mean(0), loc, atol=0.008, rtol=0)
    torch.testing.assert_close(
        torch.cov(ratios.T), factor @ factor.T, atol=0.013, rtol=0
    )
    assert dist.expand((4,)).sample((3,)).shape == (3, 4, 3)
    with pytest.raises(NotImplementedError):
        _ = dist.mean
    with pytest.raises(FloatingPointError, match="underflow"):
        AdditiveLogRatioNormal([1e5], [[1.0]]).rsample()


@pytest.mark.parametrize(
    "point",
    [[0.0, 0.4, 0.6], [-0.1, 0.5, 0.6], [0.2, 0.3, 0.6], [float("nan"), 0.3, 0.7]],
)
def test_outside_support_is_negative_infinity_with_zero_parameter_gradients(point):
    loc = torch.tensor([0.1, 0.3], dtype=torch.double, requires_grad=True)
    actual = AdditiveLogRatioNormal(loc, torch.eye(2, dtype=torch.double)).log_prob(
        torch.tensor(point, dtype=torch.double)
    )
    assert actual == -torch.inf
    torch.testing.assert_close(
        torch.autograd.grad(actual, loc)[0], torch.zeros_like(loc)
    )


def _source(rows, coordinates):
    return f"""object Row : FinSet {rows}
object Coordinate : FinSet {coordinates}
object Component : FinSet {coordinates + 1}
program model(loc:Tensor[Real]([Row,Coordinate]), factor:Tensor[Real]([Row,Coordinate,Coordinate]), y:Tensor[Real]([Row,Component])):Row->Row
    observe y : Row <- AdditiveLogRatioNormal(loc=loc,scale_tril=factor) [over=Component]
    return y
export model
"""


def _host(x):
    return tuple(_host(v) for v in x) if isinstance(x, list) else x


@pytest.mark.parametrize("rows,width", [(2, 2), (3, 2), (2, 3)])
def test_checked_source_native_reference_and_independent_density(rows, width):
    loc = (
        torch.linspace(-0.3, 0.4, rows * width, dtype=torch.double)
        .reshape(rows, width)
        .requires_grad_()
    )
    factor = (
        torch.eye(width, dtype=torch.double)
        .expand(rows, width, width)
        .clone()
        .requires_grad_()
    )
    y = torch.full((rows, width + 1), 1 / (width + 1), dtype=torch.double)
    source = _source(rows, width)
    checked = lower_qvr_to_qiec(parse(source))
    expected = AdditiveLogRatioNormal(loc, factor).log_prob(y).sum()
    actual = (
        Model(source)
        .bind({"loc": loc, "factor": factor, "y": y})
        .trace({})
        .log_joint.sum()
    )
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual, (loc, factor)),
        torch.autograd.grad(expected, (loc, factor)),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    reference = run_program(
        checked,
        "model",
        data={
            k: _host(v.detach().tolist())
            for k, v in {"loc": loc, "factor": factor, "y": y}.items()
        },
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-10)


def test_reference_sampling_ratios_and_parameter_validation():
    params = {"loc": (0.2, -0.4), "scale_tril": ((0.8, 0.0), (0.35, 0.6))}
    rng = random.Random(24)
    values = np.array(
        [SAMPLERS["AdditiveLogRatioNormal"](params, rng) for _ in range(20000)]
    )
    ratios = np.log(values[:, :-1] / values[:, -1:])
    np.testing.assert_allclose(ratios.mean(0), [0.2, -0.4], atol=0.015)
    np.testing.assert_allclose(
        np.cov(ratios.T),
        np.array(params["scale_tril"]) @ np.array(params["scale_tril"]).T,
        atol=0.016,
    )
    assert DENSITIES["AdditiveLogRatioNormal"](params, (0.0, 0.4, 0.6)) == -float("inf")
    with pytest.raises(ValueError, match="Cholesky"):
        DENSITIES["AdditiveLogRatioNormal"](
            {"loc": (0.2, -0.4), "scale_tril": ((1.0, 0.1), (0.0, 1.0))},
            (0.2, 0.3, 0.5),
        )


def test_conditional_uses_shared_gaussian_heads_with_one_fewer_coordinate():
    morphism = ConditionalAdditiveLogRatioNormal(
        Euclidean(name="x", dim=2), Simplex(name="y", dim=3)
    )
    x = torch.randn(4, 2)
    y = morphism.rsample(x)
    assert y.shape == (4, 3)
    assert morphism.log_prob(x, y).shape == (4,)
    morphism.log_prob(x, y.detach()).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in morphism.parameters()
    )


def test_float32_simplex_roundoff_has_same_native_and_reference_support():
    y = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32)
    native = AdditiveLogRatioNormal(torch.zeros(2), torch.eye(2)).log_prob(y)
    reference = DENSITIES["AdditiveLogRatioNormal"](
        {"loc": (0.0, 0.0), "scale_tril": ((1.0, 0.0), (0.0, 1.0))}, tuple(y.tolist())
    )
    assert reference == pytest.approx(native.item(), abs=1e-6)
