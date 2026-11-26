"""Response-time laws match SciPy and independently differentiated CDFs."""

import numpy as np
import pytest
import scipy.stats
import torch

from quivers.continuous import ExGaussian, InverseGaussian
from quivers.continuous.families import (
    ConditionalExGaussian,
    ConditionalInverseGaussian,
)
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference.trace import trace


@pytest.mark.parametrize(
    "loc, scale, rate", [(0.3, 0.7, 1.4), (-1.0, 2.0, 0.1), (2.0, 0.1, 30.0)]
)
def test_exgaussian_density_cdf_moments_match_scipy(loc, scale, rate):
    law = ExGaussian(torch.tensor(loc, dtype=torch.float64), scale, rate)
    reference = scipy.stats.exponnorm(1 / (rate * scale), loc=loc, scale=scale)
    points = torch.tensor([-20.0, -3.0, 0.0, 0.5, 5.0, 100.0], dtype=torch.float64)
    np.testing.assert_allclose(
        law.log_prob(points).numpy(), reference.logpdf(points), atol=1e-9
    )
    np.testing.assert_allclose(
        law.cdf(points).numpy(), reference.cdf(points), atol=1e-12
    )
    assert law.mean.item() == pytest.approx(reference.mean())
    assert law.variance.item() == pytest.approx(reference.var())


@pytest.mark.parametrize("mu, shape", [(0.5, 1.3), (2.0, 3.0), (5.0, 0.2)])
def test_inverse_gaussian_density_cdf_moments_match_scipy(mu, shape):
    law = InverseGaussian(torch.tensor(mu, dtype=torch.float64), shape)
    reference = scipy.stats.invgauss(mu / shape, scale=shape)
    points = torch.tensor([0.001, 0.1, 0.5, 2.0, 10.0, 1000.0], dtype=torch.float64)
    np.testing.assert_allclose(
        law.log_prob(points).numpy(), reference.logpdf(points), atol=1e-10
    )
    np.testing.assert_allclose(
        law.cdf(points).numpy(), reference.cdf(points), atol=1e-13
    )
    assert law.mean.item() == pytest.approx(reference.mean())
    assert law.variance.item() == pytest.approx(reference.var())


@pytest.mark.parametrize(
    "distribution, parameters, point",
    [
        (ExGaussian, (0.3, 0.7, 1.4), 0.5),
        (InverseGaussian, (1.3, 2.7), 0.5),
    ],
)
def test_density_and_cdf_autograd_match_numerical_derivatives(
    distribution, parameters, point
):
    tensors = tuple(
        torch.tensor(item, dtype=torch.float64, requires_grad=True)
        for item in parameters
    )
    value = torch.tensor(point, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda *args: distribution(*args).log_prob(value), tensors
    )
    assert torch.autograd.gradcheck(
        lambda *args: distribution(*args).cdf(value), tensors
    )


def test_inverse_gaussian_sample_derivative_matches_independent_scipy_cdf_difference():
    mu = torch.tensor([0.7, 1.8], dtype=torch.float64, requires_grad=True)
    shape = torch.tensor([1.1, 2.4], dtype=torch.float64, requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        draws = InverseGaussian(mu, shape).rsample(torch.Size((3,)))
    actual_mu, actual_shape = torch.autograd.grad(draws.sum(), (mu, shape))
    expected_mu, expected_shape = [], []
    step = 1e-5
    for index in range(2):
        m, s = mu[index].item(), shape[index].item()
        y = draws[:, index].detach().numpy()
        density = scipy.stats.invgauss(m / s, scale=s).pdf(y)
        derivative_mu = (
            scipy.stats.invgauss((m + step) / s, scale=s).cdf(y)
            - scipy.stats.invgauss((m - step) / s, scale=s).cdf(y)
        ) / (2 * step)
        derivative_shape = (
            scipy.stats.invgauss(m / (s + step), scale=s + step).cdf(y)
            - scipy.stats.invgauss(m / (s - step), scale=s - step).cdf(y)
        ) / (2 * step)
        expected_mu.append(np.sum(-derivative_mu / density))
        expected_shape.append(np.sum(-derivative_shape / density))
    np.testing.assert_allclose(actual_mu.numpy(), expected_mu, atol=1e-8)
    np.testing.assert_allclose(actual_shape.numpy(), expected_shape, atol=1e-8)


@pytest.mark.parametrize("distribution", [ExGaussian, InverseGaussian])
def test_reparameterized_samples_recover_moments_and_moment_gradients(distribution):
    parameters = (0.3, 0.7, 1.4) if distribution is ExGaussian else (1.3, 2.4)
    tensors = tuple(
        torch.tensor(item, dtype=torch.float64, requires_grad=True)
        for item in parameters
    )
    law = distribution(*tensors)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        draws = law.rsample(torch.Size((100000,)))
    assert draws.mean().item() == pytest.approx(law.mean.item(), abs=0.025)
    assert draws.var().item() == pytest.approx(law.variance.item(), abs=0.06)
    for degree in (1, 2):
        expected_moment = law.mean if degree == 1 else law.mean.square() + law.variance
        actual = torch.autograd.grad(
            draws.pow(degree).mean(), tensors, retain_graph=True
        )
        expected = torch.autograd.grad(
            expected_moment, tensors, allow_unused=True, retain_graph=True
        )
        for derivative, oracle in zip(actual, expected, strict=True):
            oracle = torch.zeros_like(derivative) if oracle is None else oracle
            torch.testing.assert_close(derivative, oracle, atol=0.04, rtol=0.025)
    expanded = law.expand((2, 3))
    assert expanded.rsample().shape == (2, 3)


@pytest.mark.parametrize("distribution", [ExGaussian, InverseGaussian])
def test_parameter_validation_rejects_nonpositive_scale_or_shape(distribution):
    with pytest.raises(ValueError):
        distribution(
            *((0.0, -1.0, 1.0) if distribution is ExGaussian else (1.0, 0.0)),
            validate_args=True,
        )


@pytest.mark.parametrize(
    "family, args, values",
    [
        (
            "ExGaussian",
            "loc=mu, scale=sigma, rate=rate",
            {"mu": 0.3, "sigma": 0.7, "rate": 1.4},
        ),
        ("InverseGaussian", "mu=mu, shape=shape", {"mu": 1.3, "shape": 2.4}),
    ],
)
def test_mixed_dsl_observations_match_native_density(family, args, values):
    source = f"""object X : Real 1
program model : X -> X
    observe y <- {family}({args})
    return y
export model
"""
    point = torch.tensor([[0.2], [0.8]])
    program = loads(source).morphism
    law = (
        ExGaussian(0.3, 0.7, 1.4)
        if family == "ExGaussian"
        else InverseGaussian(1.3, 2.4)
    )
    recorded = trace(
        program,
        torch.zeros(2, 1),
        {name: torch.tensor(value) for name, value in values.items()} | {"y": point},
    )
    torch.testing.assert_close(recorded.log_joint, law.log_prob(point).squeeze(-1))


@pytest.mark.parametrize("family", [ConditionalExGaussian, ConditionalInverseGaussian])
def test_conditional_family_has_differentiable_sample_and_density(family):
    model = family(Euclidean(name="X", dim=1), Euclidean(name="Y", dim=2))
    inputs = torch.tensor([[-0.5], [0.5]])
    draws = model.rsample(inputs)
    density = model.log_prob(inputs, draws)
    assert draws.shape == (2, 2)
    assert torch.isfinite(density).all()
    density.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "family, parameters, reference",
    [
        (
            "ExGaussian",
            {"loc": 0.3, "scale": 0.7, "rate": 1.4},
            scipy.stats.exponnorm(1 / (1.4 * 0.7), loc=0.3, scale=0.7),
        ),
        (
            "InverseGaussian",
            {"mu": 1.3, "shape": 2.4},
            scipy.stats.invgauss(1.3 / 2.4, scale=2.4),
        ),
    ],
)
def test_reference_backend_scores_and_samples_official_parameterization(
    family, parameters, reference
):
    import random
    from quivers.qiec.distributions import ReferenceBackend

    backend = ReferenceBackend()
    for point in [0.05, 0.4, 2.0, 20.0]:
        assert backend.family_log_prob(family, parameters, point) == pytest.approx(
            reference.logpdf(point), abs=1e-10
        )
    rng = random.Random(18)
    draws = np.array([backend.family_sample(family, parameters, rng) for _ in range(30000)])
    assert draws.mean() == pytest.approx(reference.mean(), abs=0.04)
    assert draws.var() == pytest.approx(reference.var(), abs=0.06)


@pytest.mark.parametrize(
    "distribution, parameters",
    [(ExGaussian, (0.3, 0.7, 1.4)), (InverseGaussian, (1.3, 2.4))],
)
def test_cdf_limits_and_broadcast_gradients(distribution, parameters):
    law = distribution(*parameters)
    torch.testing.assert_close(
        law.cdf(torch.tensor([-torch.inf, torch.inf])), torch.tensor([0.0, 1.0])
    )
    if distribution is InverseGaussian:
        torch.testing.assert_close(law.cdf(torch.tensor([-2.0, 0.0])), torch.zeros(2))
    tensor_parameters = tuple(
        torch.tensor(item, dtype=torch.float64, requires_grad=True)
        for item in parameters
    )
    base = distribution(*tensor_parameters)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        draws = base.expand((2, 3)).rsample(torch.Size((4,)))
    gradient = torch.autograd.grad(draws.sum(), tensor_parameters)
    assert draws.shape == (4, 2, 3)
    assert all(torch.isfinite(item).all() for item in gradient)


def test_exgaussian_left_tail_remains_finite_at_large_rate_scale():
    rate = torch.tensor(1e8, dtype=torch.float64, requires_grad=True)
    law = ExGaussian(torch.tensor(0.0, dtype=torch.float64), 1.0, rate)
    point = torch.tensor(-3.0, dtype=torch.float64)
    density = law.log_prob(point)
    reference = torch.distributions.Normal(0.0, 1.0).log_prob(point)
    torch.testing.assert_close(density, reference, atol=1e-7, rtol=0.0)
    assert torch.isfinite(torch.autograd.grad(density, rate)[0])
