"""Censoring retains boundary probability atoms and normalized interiors."""

import math

import pytest
import torch
from scipy.integrate import quad
from scipy.stats import beta as scipy_beta
from scipy.stats import norm, poisson
from torch.distributions import (
    AffineTransform,
    Beta,
    Exponential,
    Normal,
    Poisson,
    TransformedDistribution,
)

from quivers.continuous._censored import Censored
from quivers.continuous._interval import log_interval_mass


@pytest.mark.parametrize(
    "loc,scale,low,high",
    [(0.3, 1.2, -0.5, 1.7), (2.0, 0.5, 0.0, 1.0), (0.0, 1.0, 9.0, 10.0)],
)
def test_normal_censoring_atoms_interior_and_normalization(loc, scale, low, high):
    law = Censored(
        Normal(
            torch.tensor(loc, dtype=torch.double),
            torch.tensor(scale, dtype=torch.double),
        ),
        low,
        high,
    )
    oracle = norm(loc, scale)
    assert law.log_prob(low).item() == pytest.approx(oracle.logcdf(low), abs=2e-13)
    assert law.log_prob(high).item() == pytest.approx(oracle.logsf(high), abs=2e-13)
    middle = (low + high) / 2
    assert law.log_prob(middle).item() == pytest.approx(
        oracle.logpdf(middle), abs=2e-13
    )
    interior = quad(lambda x: math.exp(law.log_prob(x).item()), low, high)[0]
    total = interior + law.log_prob(low).exp().item() + law.log_prob(high).exp().item()
    assert total == pytest.approx(1.0, abs=2e-13)
    assert law.log_prob(low - 1).item() == -math.inf
    assert law.log_prob(high + 1).item() == -math.inf


@pytest.mark.parametrize(
    "low,high", [(None, None), (None, 0.4), (-0.5, None), (-0.5, 0.4)]
)
def test_censoring_cdf_support_includes_lower_atom_and_excludes_upper_atom(low, high):
    law = Censored(
        Normal(
            torch.tensor(0.2, dtype=torch.double), torch.tensor(0.8, dtype=torch.double)
        ),
        low,
        high,
    )
    points = [-math.inf, -1.0, -0.5, 0.0, 0.4, 1.0, math.inf]
    for point in points:
        expected = (
            0.0
            if point == -math.inf or low is not None and point < low
            else 1.0
            if point == math.inf or high is not None and point >= high
            else norm.cdf(point, 0.2, 0.8)
        )
        assert law.cdf(point).item() == pytest.approx(expected, abs=3e-16)
        cdf, sf = law.log_cdf(point), law.log_survival_function(point)
        assert torch.logaddexp(cdf, sf).item() == pytest.approx(0.0, abs=3e-16)


@pytest.mark.parametrize("low,high", [(1.0, 4.0), (1.5, 4.5)])
def test_discrete_censoring_uses_inclusive_upper_atom(low, high):
    law = Censored(Poisson(torch.tensor(2.3, dtype=torch.double)), low, high)
    expected_low = poisson.cdf(math.floor(low), 2.3)
    expected_high = poisson.sf(math.ceil(high) - 1, 2.3)
    assert law.log_prob(low).exp().item() == pytest.approx(expected_low)
    assert law.log_prob(high).exp().item() == pytest.approx(expected_high)
    middle = sum(
        law.log_prob(float(x)).exp().item()
        for x in range(math.floor(low) + 1, math.ceil(high))
    )
    assert expected_low + middle + expected_high == pytest.approx(1.0)


def test_beta_tails_are_shared_by_affine_censoring_and_interval_normalization():
    a, b = torch.tensor(0.2, dtype=torch.double), torch.tensor(3.0, dtype=torch.double)
    base = Beta(a, b)
    transformed = TransformedDistribution(base, [AffineTransform(-0.4, 1.8)])
    law = Censored(transformed, 0.0, 1.0)
    lower, upper = 0.4 / 1.8, 1.4 / 1.8
    assert law.log_prob(0.0).item() == pytest.approx(
        scipy_beta.logcdf(lower, 0.2, 3.0), abs=3e-14
    )
    assert law.log_prob(1.0).item() == pytest.approx(
        scipy_beta.logsf(upper, 0.2, 3.0), abs=3e-14
    )
    expected = scipy_beta.cdf(upper, 0.2, 3.0) - scipy_beta.cdf(lower, 0.2, 3.0)
    assert log_interval_mass(base, lower, upper).exp().item() == pytest.approx(expected)
    expected_density = scipy_beta.logpdf((0.3 + 0.4) / 1.8, 0.2, 3.0) - math.log(1.8)
    assert law.log_prob(0.3).item() == pytest.approx(expected_density, abs=2e-14)


def test_censored_beta_endpoint_gradients_do_not_score_unsafe_inactive_density():
    a = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    b = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    score = Censored(Beta(a, b), 0.0, 1.0).log_prob(
        torch.tensor([0.0, 0.3, 1.0], dtype=torch.double)
    )
    assert torch.isneginf(score[[0, 2]]).all()
    gradients = torch.autograd.grad(score.sum(), (a, b), create_graph=True)
    assert all(torch.isfinite(value) for value in gradients)
    assert all(
        torch.isfinite(value) for value in torch.autograd.grad(sum(gradients), (a, b))
    )


def test_censoring_first_second_gradients_and_reparameterized_sample():
    values = tuple(
        torch.tensor(v, dtype=torch.double, requires_grad=True)
        for v in (0.2, 0.7, -0.4, 1.1)
    )

    def score(loc, scale, low, high):
        law = Censored(Normal(loc, scale), low, high)
        return torch.stack(
            (
                law.log_prob(low),
                law.log_prob(torch.tensor(0.5, dtype=torch.double)),
                law.log_prob(high),
            )
        )

    assert torch.autograd.gradcheck(score, values, atol=1e-7, rtol=1e-6)
    assert torch.autograd.gradgradcheck(score, values, atol=1e-6, rtol=1e-5)
    law = Censored(Normal(values[0], values[1]), values[2], values[3])
    draws = law.rsample((2000,))
    assert ((draws >= values[2]) & (draws <= values[3])).all()
    assert all(
        torch.isfinite(value) for value in torch.autograd.grad(draws.mean(), values)
    )


def test_censored_sampling_is_seeded_batched_and_matches_boundary_frequencies():
    base = Exponential(torch.tensor([0.8, 2.1], dtype=torch.double))
    law = Censored(base, 0.2, torch.tensor([1.1, 0.9], dtype=torch.double))
    torch.manual_seed(372)
    samples = law.sample((20000,))
    torch.manual_seed(372)
    assert torch.equal(samples, law.sample((20000,)))
    assert samples.shape == (20000, 2)
    for axis in range(2):
        assert (
            samples[:, axis] == law.low[axis]
        ).double().mean().item() == pytest.approx(
            law.log_prob(law.low).exp()[axis].item(), abs=0.012
        )
        assert (
            samples[:, axis] == law.high[axis]
        ).double().mean().item() == pytest.approx(
            law.log_prob(law.high).exp()[axis].item(), abs=0.012
        )
    assert law.expand((3, 2)).sample((4,)).shape == (4, 3, 2)


@pytest.mark.parametrize("low,high", [(1, 0), (0, 0), (math.nan, 1), (0, math.nan)])
def test_censoring_rejects_invalid_bounds(low, high):
    with pytest.raises(ValueError, match="ordered"):
        Censored(Normal(0.0, 1.0), low, high)


def test_censoring_rejects_nan_observations_and_coupled_events():
    with pytest.raises(ValueError, match="scalar events"):
        Censored(
            torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)), 0, 1
        )
    law = Censored(Normal(0.0, 1.0), 0, 1)
    for method in (law.log_prob, law.cdf):
        with pytest.raises(ValueError, match="NaN"):
            method(math.nan)
