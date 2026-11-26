"""Common-reference mixture scores and selective draws have independent oracles."""

import math

import pytest
import torch
from torch.distributions import Beta, Binomial, Exponential, Normal

from quivers.continuous._closed_probability import ClosedPoisson
from quivers.continuous.bijectors import AffineBijector
from quivers.continuous.measure import (
    Mixture,
    Normalize,
    PointMass,
    Pushforward,
    Restrict,
)
from quivers.qiec import HostLaw


def _ref(name, **arguments):
    return HostLaw(family=name, arguments=arguments)


@pytest.mark.parametrize("weight", [0.0, 0.3, 1.0])
def test_declared_atom_carrier_survives_zero_weight(weight):
    mixture = Mixture(
        torch.tensor([weight, 1 - weight]), [PointMass(0.0), Normal(0.0, 1.0)]
    )
    expected = math.log(weight) if weight else -math.inf
    assert mixture.log_prob(torch.tensor(0.0)).item() == pytest.approx(expected)
    reference = _ref(
        "Mixture",
        weights=(weight, 1 - weight),
        component=(_ref("PointMass", value=0.0), _ref("Normal", loc=0.0, scale=1.0)),
    )
    assert reference.log_prob(0.0) == pytest.approx(expected)


def test_integer_reference_does_not_change_with_binomial_trial_count():
    mixture = Mixture(
        torch.tensor([0.5, 0.5]), [Binomial(1, probs=0.3), Normal(0.0, 1.0)]
    )
    assert mixture.log_prob(torch.tensor(2.0)).item() == -math.inf
    expected = math.log(0.5) + Normal(0.0, 1.0).log_prob(torch.tensor(2.1)).item()
    assert mixture.log_prob(torch.tensor(2.1)).item() == pytest.approx(expected)


def test_component_support_validation_does_not_reject_other_components():
    mixture = Mixture(torch.tensor([0.4, 0.6]), [Exponential(1.0), Normal(0.0, 1.0)])
    expected = math.log(0.6) + Normal(0.0, 1.0).log_prob(torch.tensor(-1.0)).item()
    assert mixture.log_prob(torch.tensor(-1.0)).item() == pytest.approx(expected)


@pytest.mark.parametrize("point", [0.0, 0.4])
def test_inactive_atomic_and_continuous_scores_keep_finite_weight_hessians(point):
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    mean = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    mixture = Mixture(weights, [PointMass(0.0), Normal(mean, 1.0)])
    actual = mixture.log_prob(torch.tensor(point, dtype=torch.float64))
    expected = (
        weights[0].log()
        if point == 0
        else weights[1].log() + Normal(mean, 1.0).log_prob(torch.tensor(point))
    )
    actual_gradient = torch.autograd.grad(actual, weights, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, weights, create_graph=True)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(
        torch.autograd.grad(actual_gradient.sum(), weights)[0],
        torch.autograd.grad(expected_gradient.sum(), weights)[0],
    )


def test_zero_weight_density_keeps_true_one_sided_derivative():
    weights = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    components = [Normal(0.0, 1.0), Normal(1.0, 1.0)]
    actual = Mixture(weights, components).log_prob(torch.tensor(0.2))
    scores = torch.stack([c.log_prob(torch.tensor(0.2)) for c in components]).double()
    expected = (weights * scores.exp()).sum().log()
    torch.testing.assert_close(
        torch.autograd.grad(actual, weights, create_graph=True)[0],
        torch.autograd.grad(expected, weights, create_graph=True)[0],
    )


def test_infinite_beta_endpoint_density_is_inactive_at_the_dirac_atom():
    shape = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    mixture = Mixture(torch.tensor([0.2, 0.8]), [PointMass(0.0), Beta(shape, 0.4)])
    score = mixture.log_prob(torch.tensor(0.0))
    assert score.item() == pytest.approx(math.log(0.2))
    assert torch.autograd.grad(score, shape)[0].item() == 0


def test_bijective_transport_applies_no_jacobian_to_singleton_mass():
    base = Mixture(torch.tensor([0.3, 0.7]), [PointMass(0.0), Normal(0.0, 1.0)])
    transported = Pushforward(base, AffineBijector(shift=1.0, scale=3.0))
    assert transported.log_prob(torch.tensor(1.0)).item() == pytest.approx(
        math.log(0.3)
    )
    continuous = transported.log_prob(torch.tensor(2.0))
    expected = (
        math.log(0.7)
        + Normal(0.0, 1.0).log_prob(torch.tensor(1 / 3)).item()
        - math.log(3)
    )
    assert continuous.item() == pytest.approx(expected)


def test_restriction_and_normalization_retain_atom_reference():
    base = Mixture(torch.tensor([0.3, 0.7]), [PointMass(0.0), Normal(0.0, 1.0)])
    restricted = Normalize(Restrict(base, low=0.0))
    assert restricted.log_prob(torch.tensor(0.0)).item() == pytest.approx(
        math.log(0.3 / 0.65)
    )


def test_mixture_never_draws_an_unselected_count_batch_with_unrepresentable_rate():
    law = Mixture(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        [
            ClosedPoisson(torch.tensor([1.0, 1e20], dtype=torch.float64)),
            Normal(torch.tensor([0.0, 2.0]), 1.0),
        ],
    )
    samples = law.sample(torch.Size((16,)))
    assert samples.shape == (16, 2)
    assert bool(torch.isfinite(samples).all())
    assert bool((samples[:, 0] == samples[:, 0].floor()).all())


def test_selective_point_mass_sampling_needs_no_native_constructor_metadata():
    law = Mixture(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        [PointMass(torch.zeros(2)), Normal(torch.ones(2), 1.0)],
    )
    samples = law.sample(torch.Size((8,)))
    assert samples.shape == (8, 2)
    assert bool((samples[:, 0] == 0).all())
