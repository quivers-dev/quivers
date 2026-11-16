"""Discrete reference densities reject fractional observations without truncation."""

import math

import pytest
import torch
from scipy import stats

from quivers.continuous._zip_hurdle import HurdlePoisson, ZeroInflatedPoisson
from quivers.qiec.distributions import ReferenceBackend
from quivers.qiec.reference_support import _integer_observation


PARAMETERS = {
    "Bernoulli": {"probs": 0.3},
    "Categorical": {"probs": (0.2, 0.3, 0.5)},
    "Binomial": {"total_count": 5, "probs": 0.3},
    "Poisson": {"rate": 2.0},
    "Geometric": {"probs": 0.3},
    "NegativeBinomial": {"total_count": 2.0, "probs": 0.3},
    "BetaBinomial": {"total_count": 5, "concentration1": 0.7, "concentration0": 1.3},
    "ZeroInflatedPoisson": {"zero_prob": 0.2, "rate": 2.0},
    "HurdlePoisson": {"zero_prob": 0.2, "rate": 2.0},
    "DiscreteWeibull": {"probs": 0.4, "concentration": 1.3},
}


@pytest.mark.parametrize("family", PARAMETERS)
@pytest.mark.parametrize(
    "value", [-0.5, 0.5, 1.5, float("nan"), float("inf"), -float("inf")]
)
def test_nonintegral_observations_have_no_counting_measure_mass(family, value):
    assert ReferenceBackend().log_prob(family, PARAMETERS[family], value) == -math.inf


@pytest.mark.parametrize("value", [-1.0, 2.0, 3.0])
def test_bernoulli_rejects_numeric_values_outside_zero_and_one(value):
    assert ReferenceBackend().log_prob("Bernoulli", {"probs": 0.3}, value) == -math.inf


def test_integer_reader_preserves_exact_host_integers_and_boolean_carrier():
    assert _integer_observation(2**53 + 1, "Poisson") == 2**53 + 1
    assert _integer_observation(True, "Bernoulli") == 1
    assert _integer_observation(2.0, "Poisson") == 2
    assert _integer_observation(0.5, "Poisson") is None


@pytest.mark.parametrize("count", [1000001, 33554433, 2147483647])
def test_scalar_and_inflated_poisson_share_stable_large_count_scores(count):
    backend = ReferenceBackend()
    oracle = stats.poisson.logpmf(count, count)
    assert backend.log_prob("Poisson", {"rate": count}, count) == pytest.approx(
        oracle, abs=5e-6
    )
    for family in ("ZeroInflatedPoisson", "HurdlePoisson"):
        actual = backend.log_prob(family, {"zero_prob": 0.2, "rate": count}, count)
        assert actual == pytest.approx(math.log(0.8) + oracle, abs=5e-6)
    rate = torch.tensor(float(count), dtype=torch.float64, requires_grad=True)
    zero = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    actual = HurdlePoisson(zero, rate).log_prob(torch.tensor(count))
    assert actual.item() == pytest.approx(math.log(0.8) + oracle, abs=5e-6)
    gradients = torch.autograd.grad(actual, (rate, zero))
    assert gradients[0].item() == pytest.approx(0.0, abs=2e-14)
    assert gradients[1].item() == pytest.approx(-1 / 0.8)


def test_zero_inflated_poisson_zero_atom_keeps_log_domain_extreme_tail():
    assert (
        ReferenceBackend().log_prob(
            "ZeroInflatedPoisson", {"zero_prob": 0.0, "rate": 1000.0}, 0
        )
        == -1000.0
    )


def test_hurdle_poisson_zero_rate_retains_first_and_second_derivatives():
    rate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    score = HurdlePoisson(torch.tensor(0.2), rate).log_prob(torch.tensor(1.0))
    derivative = torch.autograd.grad(score, rate, create_graph=True)[0]
    second = torch.autograd.grad(derivative, rate)[0]
    assert score.item() == pytest.approx(math.log(0.8), abs=1e-7)
    assert derivative.item() == pytest.approx(-0.5)
    assert second.item() == pytest.approx(-1 / 12)


@pytest.mark.parametrize("zero", [0.0, 0.3, 1.0])
def test_poisson_zero_atom_retains_closed_endpoint_derivatives(zero):
    probability = torch.tensor(zero, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    score = ZeroInflatedPoisson(probability, rate).log_prob(torch.tensor(0.0))
    mass = zero + (1 - zero) * math.exp(-2)
    assert score.item() == pytest.approx(math.log(mass))
    first = torch.autograd.grad(score, (probability, rate), create_graph=True)
    assert first[0].item() == pytest.approx((1 - math.exp(-2)) / mass)
    assert first[1].item() == pytest.approx(-(1 - zero) * math.exp(-2) / mass)
    second = torch.autograd.grad(first[0], probability)[0]
    assert second.item() == pytest.approx(-(((1 - math.exp(-2)) / mass) ** 2))


def test_shared_poisson_count_sampling_preserves_shape_atoms_and_moments():
    law = ZeroInflatedPoisson(
        torch.tensor([1.0, 0.0, 0.2]), torch.tensor([1e20, 0.0, 2.0])
    )
    with torch.random.fork_rng():
        torch.manual_seed(62)
        sample = law.expand((2, 3)).sample((20000,))
    assert sample.dtype == torch.int64
    assert sample.shape == (20000, 2, 3)
    assert sample[..., :2].eq(0).all()
    assert sample[..., 2].double().mean().item() == pytest.approx(1.6, abs=0.03)
    assert sample[..., 2].double().var().item() == pytest.approx(2.24, abs=0.06)
