"""Interval normalization is deterministic, normalized, and differentiable."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import integrate, stats
from torch.distributions import Gamma, LogNormal, MultivariateNormal, Normal, Poisson

from quivers.continuous._interval import log_interval_mass
from quivers.continuous.families import ConditionalNormal, Truncated
from quivers.continuous.measure import Restrict
from quivers.continuous.param_source import IdentitySource
from quivers.continuous.spaces import Euclidean


@pytest.mark.parametrize(
    "bounds", [(None, -9.0), (8.0, None), (8.0, 8.1), (-0.001, 0.001), (-2.0, 3.0)]
)
def test_normal_interval_matches_independent_integral(
    bounds: tuple[float | None, float | None],
) -> None:
    low, high = bounds
    got = log_interval_mass(
        Normal(torch.tensor(0.0, dtype=torch.float64), 1.0), low, high
    )
    mass = integrate.quad(
        stats.norm.pdf,
        -np.inf if low is None else low,
        np.inf if high is None else high,
        epsabs=1e-30,
    )[0]
    assert float(got) == pytest.approx(math.log(mass), abs=2e-12)


def _conditional(dimensions: int = 1) -> ConditionalNormal:
    return ConditionalNormal(
        Euclidean(name="parameters", dim=2 * dimensions),
        Euclidean(name="response", dim=dimensions),
        param_source=IdentitySource(2 * dimensions),
    )


def test_conditional_normalization_preserves_rng_and_matches_density_integral() -> None:
    model = Truncated(_conditional(), lower=0.2, upper=1.4)
    x = torch.tensor([[0.3, math.log(1.2)]], dtype=torch.float64)
    y = torch.tensor([[0.8]], dtype=torch.float64)
    state = torch.random.get_rng_state().clone()
    first = model.log_prob(x, y)
    assert torch.equal(first, model.log_prob(x, y))
    assert torch.equal(state, torch.random.get_rng_state())
    mass = integrate.quad(
        lambda value: stats.norm.pdf(value, loc=0.3, scale=1.2), 0.2, 1.4
    )[0]
    assert float(first) == pytest.approx(
        stats.norm.logpdf(0.8, loc=0.3, scale=1.2) - math.log(mass), abs=1e-13
    )


def test_conditional_normalization_has_first_and_second_parameter_derivatives() -> None:
    model = Truncated(_conditional(), lower=0.2, upper=1.4)
    x = torch.tensor([[0.3, math.log(1.2)]], dtype=torch.float64, requires_grad=True)
    y = x.new_tensor([[0.8]])
    assert torch.autograd.gradcheck(
        lambda parameters: model.log_prob(parameters, y), (x,)
    )
    assert torch.autograd.gradgradcheck(
        lambda parameters: model.log_prob(parameters, y), (x,)
    )
    epsilon = 1e-5

    def oracle(location: float) -> float:
        mass = integrate.quad(
            lambda value: stats.norm.pdf(value, loc=location, scale=1.2), 0.2, 1.4
        )[0]
        return stats.norm.logpdf(0.8, loc=location, scale=1.2) - math.log(mass)

    derivative = torch.autograd.grad(model.log_prob(x, y).sum(), x)[0][0, 0]
    assert float(derivative) == pytest.approx(
        (oracle(0.3 + epsilon) - oracle(0.3 - epsilon)) / (2 * epsilon), abs=1e-9
    )


def test_independent_box_normalizer_sums_each_coordinate() -> None:
    model = Truncated(_conditional(2), lower=0.0, upper=2.0)
    x = torch.tensor([[0.2, 1.3, 0.0, math.log(0.7)]], dtype=torch.float64)
    y = x.new_tensor([[0.8, 1.2]])
    expected = sum(
        stats.truncnorm.logpdf(
            value, (0 - loc) / scale, (2 - loc) / scale, loc=loc, scale=scale
        )
        for value, loc, scale in [(0.8, 0.2, 1.0), (1.2, 1.3, 0.7)]
    )
    assert float(model.log_prob(x, y)) == pytest.approx(expected, abs=1e-12)


def test_far_tail_normal_sampling_retains_leading_shape_and_distribution() -> None:
    torch.manual_seed(62)
    model = Truncated(_conditional(), lower=8.0, upper=9.0)
    x = torch.zeros(2, 2, dtype=torch.float64)
    draws = model.rsample(x, torch.Size((10_000,)))
    assert draws.shape == (10_000, 2, 1)
    assert bool(((draws >= 8) & (draws <= 9)).all())
    assert float(draws.mean()) == pytest.approx(stats.truncnorm.mean(8, 9), abs=0.004)


@pytest.mark.parametrize(
    "bounds", [(0.0, 5.0), (3.2, 12.7), (None, 3.0), (2.0, None), (-4.0, 2.0)]
)
def test_poisson_intervals_include_exact_integer_endpoints(
    bounds: tuple[float | None, float | None],
) -> None:
    low, high = bounds
    rate = torch.tensor(4.2, dtype=torch.float64, requires_grad=True)
    result = log_interval_mass(Poisson(rate), low, high)
    lower = 0 if low is None else max(0, math.ceil(low))
    upper = 200 if high is None else math.floor(high)
    expected = stats.poisson.pmf(np.arange(lower, upper + 1), 4.2).sum()
    assert float(result.detach()) == pytest.approx(math.log(expected), abs=2e-13)
    assert torch.autograd.gradcheck(
        lambda value: log_interval_mass(Poisson(value), low, high), (rate,)
    )


def test_poisson_zero_rate_interval_has_finite_score_gradient() -> None:
    rate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    mass = log_interval_mass(Poisson(rate), 0.0, 4.0)
    assert mass.item() == 0.0
    assert torch.autograd.grad(mass, rate)[0].item() == 0.0


def test_gamma_one_sided_prior_is_normalized_with_rate_gradient() -> None:
    rate = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    prior = Restrict(Gamma(2.0, rate), low=1.0)
    expected = stats.gamma.logsf(1.0, a=2.0, scale=10.0)
    assert float(prior.log_normalizer().detach()) == pytest.approx(expected, abs=1e-13)
    assert torch.autograd.gradcheck(
        lambda value: Restrict(Gamma(2.0, value), low=1.0).log_normalizer(), (rate,)
    )


def test_zero_probability_is_not_floored() -> None:
    result = log_interval_mass(Poisson(torch.tensor(1.0)), -3.0, -1.0)
    assert result.item() == -torch.inf


def test_coupled_events_require_rectangle_probability() -> None:
    base = MultivariateNormal(torch.zeros(2), torch.eye(2))
    with pytest.raises(NotImplementedError, match="joint rectangle probability"):
        log_interval_mass(base, 0.0, 1.0)


def test_conditional_truncation_reports_its_actual_support() -> None:
    model = Truncated(_conditional(), lower=0.0, upper=2.0)
    assert model.support.check(torch.tensor([-0.1, 0.1, 2.1])).tolist() == [
        False,
        True,
        False,
    ]
    bijector = torch.distributions.biject_to(model.support)
    assert bijector(torch.tensor(0.0)).item() == 1.0


@pytest.mark.parametrize("iterations", [True, 0, 1.5])
def test_rejection_limit_requires_positive_integer(iterations: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Truncated(_conditional(), lower=0.0, max_rejection_iterations=iterations)


def test_transformed_positive_prior_preserves_normalization_and_derivatives() -> None:
    location = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    positive = Restrict(LogNormal(location, scale), low=0.0)
    assert positive.log_normalizer().item() == 0.0
    mass = log_interval_mass(LogNormal(location, scale), 0.4, 2.3)
    oracle = stats.lognorm.cdf(2.3, s=0.7, scale=math.exp(0.3)) - stats.lognorm.cdf(
        0.4, s=0.7, scale=math.exp(0.3)
    )
    assert mass.item() == pytest.approx(math.log(oracle), abs=1e-13)
    assert torch.autograd.gradcheck(
        lambda loc, std: log_interval_mass(LogNormal(loc, std), 0.4, 2.3),
        (location, scale),
    )
    assert torch.autograd.gradgradcheck(
        lambda loc, std: log_interval_mass(LogNormal(loc, std), 0.4, 2.3),
        (location, scale),
    )
