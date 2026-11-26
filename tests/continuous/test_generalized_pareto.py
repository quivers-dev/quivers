"""Generalized Pareto native and reference tails agree with independent SciPy laws."""

import random

import numpy as np
import pytest
from scipy.stats import genpareto
import torch

from quivers.continuous._generalized_pareto import GeneralizedPareto
from quivers.qiec.distributions import ReferenceBackend


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("concentration", [-2.0, -1.0, -0.2, 0.0, 1e-10, 0.2, 2.0])
def test_density_cdf_and_quantile_match_scipy(dtype, concentration):
    parameters = [torch.tensor(v, dtype=dtype) for v in (-0.5, 1.3, concentration)]
    law = GeneralizedPareto(*parameters)
    probabilities = torch.tensor([0.01, 0.25, 0.5, 0.9, 0.99], dtype=dtype)
    values = law.icdf(probabilities)
    oracle = genpareto(concentration, loc=-0.5, scale=1.3)
    np.testing.assert_allclose(
        values.numpy(),
        oracle.ppf(probabilities.numpy()),
        rtol=2e-5 if dtype == torch.float32 else 2e-13,
        atol=2e-6 if dtype == torch.float32 else 2e-13,
    )
    np.testing.assert_allclose(
        law.log_prob(values).numpy(),
        oracle.logpdf(values.numpy()),
        rtol=2e-4 if dtype == torch.float32 else 2e-11,
        atol=2e-4 if dtype == torch.float32 else 2e-11,
    )
    torch.testing.assert_close(
        law.cdf(values),
        probabilities,
        atol=2e-5 if dtype == torch.float32 else 2e-13,
        rtol=2e-5 if dtype == torch.float32 else 2e-13,
    )


@pytest.mark.parametrize("concentration", [-1.0, -0.2, 0.0, 1e-10, 0.2])
def test_density_and_quantile_first_second_derivatives(concentration):
    parameters = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (-0.5, 1.3, concentration)
    )

    def score(*args):
        law = GeneralizedPareto(*args)
        return torch.cat(
            (
                law.log_prob(torch.tensor([0.0, 0.2], dtype=torch.float64)),
                law.icdf(torch.tensor([0.25, 0.7], dtype=torch.float64)),
            )
        )

    assert torch.autograd.gradcheck(score, parameters)
    assert torch.autograd.gradgradcheck(score, parameters)


@pytest.mark.parametrize("concentration", [-2.0, -1.0, -0.2, 0.0, 0.2])
def test_reference_density_sampling_and_support(concentration):
    backend = ReferenceBackend()
    arguments = dict(loc=-0.5, scale=1.3, concentration=concentration)
    oracle = genpareto(concentration, loc=-0.5, scale=1.3)
    for probability in (0.01, 0.5, 0.99):
        point = oracle.ppf(probability)
        assert backend.family_log_prob(
            "GeneralizedPareto", arguments, float(point)
        ) == pytest.approx(oracle.logpdf(point), abs=3e-12)
    rng = random.Random(327)
    draws = np.array(
        [backend.family_sample("GeneralizedPareto", arguments, rng) for _ in range(5000)]
    )
    assert np.mean(draws <= oracle.ppf(0.4)) == pytest.approx(0.4, abs=0.025)
    assert backend.family_log_prob("GeneralizedPareto", arguments, -1.0) == -np.inf


def test_reparameterized_samples_preserve_device_shape_and_gradients():
    parameters = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (-0.5, 1.3, 0.0)
    )
    law = GeneralizedPareto(*parameters).expand((3,))
    samples = law.rsample((5000,))
    assert samples.shape == (5000, 3)
    assert samples.dtype == torch.float64
    assert samples.device == parameters[0].device
    for gradient in torch.autograd.grad(samples.mean(), parameters):
        assert bool(torch.isfinite(gradient))
    assert samples.mean().item() == pytest.approx(0.8, abs=0.04)


@pytest.mark.parametrize(
    "concentration, expected", [(-2.0, np.inf), (-1.0, -np.log(1.3)), (-0.5, -np.inf)]
)
def test_finite_upper_endpoint_density_limit(concentration, expected):
    law = GeneralizedPareto(
        torch.tensor(-0.5, dtype=torch.float64),
        torch.tensor(1.3, dtype=torch.float64),
        torch.tensor(concentration, dtype=torch.float64),
        validate_args=False,
    )
    upper = -0.5 - 1.3 / concentration
    assert law.log_prob(
        torch.tensor(upper, dtype=torch.float64)
    ).item() == pytest.approx(expected)
    assert (
        law.log_prob(torch.tensor(upper + 1.0, dtype=torch.float64)).item() == -np.inf
    )


@pytest.mark.parametrize("concentration", [-2.0, -0.2, 0.0, 0.2])
def test_quantile_endpoints(concentration):
    law = GeneralizedPareto(
        torch.tensor(-0.5, dtype=torch.float64),
        torch.tensor(1.3, dtype=torch.float64),
        torch.tensor(concentration, dtype=torch.float64),
    )
    endpoints = law.icdf(torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert endpoints[0].item() == -0.5
    assert endpoints[1].item() == pytest.approx(
        -0.5 - 1.3 / concentration if concentration < 0 else np.inf
    )
