"""Asymmetric Laplace laws agree with independent analytic and SciPy oracles."""

import math

import numpy as np
import pytest
import scipy.stats
import torch

from quivers.continuous import AsymmetricLaplace
from quivers.continuous.families import ConditionalAsymmetricLaplace
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference.trace import trace


@pytest.mark.parametrize("quantile", [0.05, 0.25, 0.5, 0.8, 0.99])
def test_density_cdf_and_quantiles_match_scipy(quantile):
    loc, scale = 0.4, 0.7
    q = quantile
    # SciPy uses kappa and scale b: kappa=sqrt(q/(1-q)),
    # b=sigma/sqrt(q*(1-q)), giving both identical exponential rates.
    reference = scipy.stats.laplace_asymmetric(
        math.sqrt(q / (1 - q)), loc=loc, scale=scale / math.sqrt(q * (1 - q))
    )
    law = AsymmetricLaplace(torch.tensor(loc, dtype=torch.float64), scale, q)
    points = torch.tensor([-500.0, -3.0, 0.4, 1.2, 12.0, 500.0], dtype=torch.float64)
    np.testing.assert_allclose(
        law.log_prob(points).numpy(), reference.logpdf(points), atol=1e-12
    )
    np.testing.assert_allclose(
        law.cdf(points).numpy(), reference.cdf(points), atol=1e-13
    )
    probabilities = torch.tensor([1e-8, 0.02, q, 0.8, 1 - 1e-8], dtype=torch.float64)
    np.testing.assert_allclose(
        law.icdf(probabilities).numpy(), reference.ppf(probabilities), atol=1e-11
    )
    np.testing.assert_allclose(law.entropy().numpy(), reference.entropy(), atol=1e-12)
    np.testing.assert_allclose(law.mean.numpy(), reference.mean(), atol=1e-10)
    np.testing.assert_allclose(law.variance.numpy(), reference.var(), atol=1e-9)
    assert law.cdf(torch.tensor(loc, dtype=torch.float64)).item() == pytest.approx(q)


def test_density_derivatives_and_far_tail_cdf_gradients_are_finite():
    loc = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    points = torch.tensor([-3.0, -0.5, 1.2], dtype=torch.float64)
    law = AsymmetricLaplace(loc, scale, q)
    residual = (points - loc) / scale
    oracle = (
        q.log()
        + (1 - q).log()
        - scale.log()
        - residual * (q - (points < loc).to(points))
    )
    for actual, expected in zip(
        torch.autograd.grad(
            law.log_prob(points).sum(), (loc, scale, q), retain_graph=True
        ),
        torch.autograd.grad(oracle.sum(), (loc, scale, q)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    extremes = torch.tensor([-1e6, 1e6], dtype=torch.float64)
    assert all(
        torch.isfinite(part)
        for part in torch.autograd.grad(law.cdf(extremes).sum(), (loc, scale, q))
    )


def test_rsample_moments_quantile_and_pathwise_derivatives():
    loc = torch.tensor([-0.2, 0.8], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([0.5, 0.7], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    law = AsymmetricLaplace(loc, scale, q)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        draws = law.rsample(torch.Size((80000,)))
    torch.testing.assert_close(draws.mean(0), law.mean, atol=0.04, rtol=0.0)
    # The fourth cumulant of E1*a-E2*b is 6*(a**4+b**4).
    a, b = scale / q, scale / (1 - q)
    variance_error = ((6 * (a**4 + b**4) + 2 * law.variance.square()) / 80000).sqrt()
    assert ((draws.var(0) - law.variance).abs() < 4 * variance_error).all()
    torch.testing.assert_close((draws < loc).double().mean(0), q, atol=0.005, rtol=0.0)
    empirical = torch.autograd.grad(
        draws.mean(0).sum(), (loc, scale, q), retain_graph=True
    )
    expected = torch.autograd.grad(law.mean.sum(), (loc, scale, q))
    for actual, oracle in zip(empirical, expected, strict=True):
        torch.testing.assert_close(actual, oracle, atol=0.15, rtol=0.015)
    expanded = law.expand((3, 2))
    assert expanded.rsample().shape == (3, 2)
    torch.testing.assert_close(
        expanded.log_prob(torch.zeros(3, 2)), law.log_prob(torch.zeros(3, 2))
    )


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_quantile_parameter_refuses_endpoints_and_invalid_values(quantile):
    with pytest.raises(ValueError):
        AsymmetricLaplace(0.0, 1.0, quantile, validate_args=True)


@pytest.mark.parametrize("runtime", [False, True])
def test_dsl_fixed_and_mixed_density_match_native_distribution(runtime):
    args = (
        "loc=mu, scale=sigma, quantile=q"
        if runtime
        else "loc=0.3, scale=0.7, quantile=0.2"
    )
    source = f"""object X : Real 1
program model : X -> X
    observe y <- AsymmetricLaplace({args})
    return y
export model
"""
    program = loads(source).morphism
    point = torch.tensor([[-0.4], [0.8]])
    values = {"y": point}
    if runtime:
        values |= {
            "mu": torch.tensor(0.3),
            "sigma": torch.tensor(0.7),
            "q": torch.tensor(0.2),
        }
    recorded = trace(program, torch.zeros(2, 1), values)
    torch.testing.assert_close(
        recorded.log_joint, AsymmetricLaplace(0.3, 0.7, 0.2).log_prob(point).squeeze(-1)
    )


def test_conditional_registered_family_has_differentiable_parameters():
    family = ConditionalAsymmetricLaplace(
        Euclidean(name="X", dim=1), Euclidean(name="Y", dim=2)
    )
    inputs = torch.tensor([[-0.5], [0.5]])
    draws = family.rsample(inputs)
    assert draws.shape == (2, 2)
    density = family.log_prob(inputs, draws.detach())
    assert torch.isfinite(density).all()
    density.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in family.parameters()
    )


def test_reference_backend_matches_density_and_quantile_probability():
    import random
    from quivers.qiec.distributions import ReferenceBackend, DistributionError

    backend = ReferenceBackend()
    arguments = {"loc": 0.4, "scale": 0.7, "quantile": 0.8}
    oracle = scipy.stats.laplace_asymmetric(2.0, loc=0.4, scale=1.75)
    for point in [-3.0, 0.2, 0.4, 1.5]:
        assert backend.family_log_prob("AsymmetricLaplace", arguments, point) == pytest.approx(
            oracle.logpdf(point), abs=1e-12
        )
    rng = random.Random(31)
    draws = np.asarray(
        [backend.family_sample("AsymmetricLaplace", arguments, rng) for _ in range(20000)]
    )
    assert np.mean(draws < 0.4) == pytest.approx(0.8, abs=0.012)
    assert np.mean(draws) == pytest.approx(oracle.mean(), abs=0.06)
    with pytest.raises(DistributionError, match="0 < quantile < 1"):
        backend.family_log_prob("AsymmetricLaplace", arguments | {"quantile": 1.0}, 0.0)


def test_cdf_inverse_endpoints_and_symmetric_laplace_scale():
    law = AsymmetricLaplace(0.3, 0.7, 0.5)
    points = torch.linspace(-20.0, 20.0, 101)
    torch.testing.assert_close(
        law.log_prob(points), torch.distributions.Laplace(0.3, 1.4).log_prob(points)
    )
    endpoints = law.icdf(torch.tensor([0.0, 1.0]))
    assert torch.isneginf(endpoints[0]) and torch.isposinf(endpoints[1])
    torch.testing.assert_close(law.cdf(endpoints), torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="probabilities in"):
        law.icdf(torch.tensor([-0.01, 1.01]))
