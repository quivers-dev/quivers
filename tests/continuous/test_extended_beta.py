"""Censored Beta moments and atoms agree with independent positive integrals."""

import math

import numpy as np
import mpmath as mp
import pytest
import torch
from scipy import integrate, stats

from quivers.continuous._extended_beta import ExtendedBeta
from quivers.continuous._simplex_sampling import (
    _dirichlet_log_weights,
    _dirichlet_weights,
)


def _oracle(mu, phi, extension):
    if extension < 1e-10 or phi >= 1e3 or extension > 1e4:
        with mp.workdps(80):
            m, p, k = map(mp.mpf, (mu, phi, extension))
            a, b, scale = m * p, (1 - m) * p, 1 + 2 * k
            lower = k / scale
            p0 = mp.betainc(a, b, 0, lower, regularized=True)
            p1 = mp.betainc(b, a, 0, lower, regularized=True)
            pieces = [
                1
                - mp.betainc(a + j, b, 0, lower, regularized=True)
                - mp.betainc(b, a + j, 0, lower, regularized=True)
                for j in range(3)
            ]
            first = p1 + scale * m * pieces[1] - k * pieces[0]
            second = (
                p1
                + scale**2 * a * (a + 1) / (p * (p + 1)) * pieces[2]
                - 2 * scale * k * m * pieces[1]
                + k * k * pieces[0]
            )
            return tuple(map(float, (p0, p1, first, second - first * first)))
    a, b, scale = mu * phi, (1 - mu) * phi, 1 + 2 * extension
    lower, upper = extension / scale, (1 + extension) / scale
    p0, p1 = stats.beta.cdf(lower, a, b), stats.beta.sf(upper, a, b)
    if extension == 0:
        return p0, p1, mu, mu * (1 - mu) / (phi + 1)

    def density(y):
        return stats.beta.pdf((y + extension) / scale, a, b) / scale

    # These integrals remain positive when an extensive partial-moment
    # subtraction would destroy the precision of the requested mean.
    mean = (
        p1
        + integrate.quad(lambda y: y * density(y), 0, 1, epsabs=2e-13, epsrel=2e-12)[0]
    )
    variance = (
        p0 * mean**2
        + p1 * (1 - mean) ** 2
        + integrate.quad(
            lambda y: (y - mean) ** 2 * density(y), 0, 1, epsabs=2e-13, epsrel=2e-12
        )[0]
    )
    return p0, p1, mean, variance


@pytest.mark.parametrize(
    "mu,phi,kappa",
    [
        (0.3, 4.0, 0.2),
        (0.8, 0.3, 0.0001),
        (0.2, 20.0, 1.0),
        (0.7, 0.5, 10.0),
        (0.4, 1e4, 0.1),
        (0.3, 8.0, 1e8),
        (0.7, 20.0, 1e12),
        (0.3, 0.4, 1e-16),
    ],
)
def test_density_atoms_normalization_and_true_moments(mu, phi, kappa):
    law = ExtendedBeta(mu, phi, kappa)
    point = torch.tensor([0.0, 0.1, 0.4, 0.9, 1.0], dtype=torch.float64)
    actual = law.log_prob(point).detach().numpy()
    a, b, scale = mu * phi, (1 - mu) * phi, 1 + 2 * kappa
    oracle = stats.beta.logpdf((point.numpy() + kappa) / scale, a, b) - math.log(scale)
    with mp.workdps(80):
        m, p, k = map(mp.mpf, (mu, phi, kappa))
        cutoff = k / (1 + 2 * k)
        oracle[0] = float(
            mp.log(mp.betainc(m * p, (1 - m) * p, 0, cutoff, regularized=True))
        )
        oracle[-1] = float(
            mp.log(mp.betainc((1 - m) * p, m * p, 0, cutoff, regularized=True))
        )
    np.testing.assert_allclose(actual, oracle, atol=2e-9, rtol=1e-10)
    p0, p1, mean, variance = _oracle(mu, phi, kappa)
    assert law.mean.item() == pytest.approx(mean, abs=2e-10, rel=2e-9)
    assert law.variance.item() == pytest.approx(variance, abs=2e-10, rel=2e-9)
    assert 0 <= law.mean.item() <= 1
    assert law.variance.item() >= 0
    if kappa > 1e-10 and phi < 1e3:
        assert p0 + p1 + integrate.quad(
            lambda y: stats.beta.pdf((y + kappa) / scale, a, b) / scale,
            0,
            1,
            epsabs=1e-12,
            epsrel=1e-10,
        )[0] == pytest.approx(1.0, abs=2e-9)
    else:
        # A response-scale double quadrature cannot resolve endpoint layers
        # narrower than its panels. Independent high precision incomplete
        # Beta mass identities cover those cases without dropping the tails.
        with mp.workdps(80):
            a, b, k = (
                mp.mpf(mu) * mp.mpf(phi),
                (1 - mp.mpf(mu)) * mp.mpf(phi),
                mp.mpf(kappa),
            )
            cutoff = k / (1 + 2 * k)
            interior = (
                1
                - mp.betainc(a, b, 0, cutoff, regularized=True)
                - mp.betainc(b, a, 0, cutoff, regularized=True)
            )
            assert float(interior) + p0 + p1 == pytest.approx(1.0, abs=2e-14)


@pytest.mark.parametrize("kappa", [0.0, 1e-16, 0.1, 1.0, 1e8, 1e12])
def test_uniform_beta_has_closed_mean_and_variance_for_every_extension(kappa):
    law = ExtendedBeta(0.5, 2.0, kappa)
    assert law.mean.item() == pytest.approx(0.5, abs=3e-14)
    assert law.variance.item() == pytest.approx(
        (kappa / 2 + 1 / 12) / (1 + 2 * kappa), abs=3e-14
    )


@pytest.mark.parametrize("mu,phi", [(0.3, 0.4), (0.2, 6.0), (0.5, 1e3)])
def test_zero_extension_keeps_beta_moments_and_one_sided_extension_gradients(mu, phi):
    parameters = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (mu, phi, 0.0)
    )
    law = ExtendedBeta(*parameters)
    mean, variance = law.mean, law.variance
    first = torch.autograd.grad(mean, parameters, retain_graph=True)
    second = torch.autograd.grad(variance, parameters)
    assert mean.item() == pytest.approx(mu)
    assert variance.item() == pytest.approx(mu * (1 - mu) / (phi + 1))
    assert first[0].item() == pytest.approx(1.0)
    assert first[1].item() == pytest.approx(0.0)
    assert first[2].item() == pytest.approx(2 * mu - 1)
    assert second[2].item() == pytest.approx(4 * mu * (1 - mu) / (phi + 1))


def test_ordinary_density_and_moment_autograd_matches_independent_finite_differences():
    parameters = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (0.3, 4.0, 0.2)
    )
    law = ExtendedBeta(*parameters)
    score = law.log_prob(torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)).sum()
    assert torch.autograd.gradcheck(
        lambda *v: ExtendedBeta(*v).log_prob(torch.tensor(0.3)), parameters
    )
    assert torch.autograd.gradgradcheck(
        lambda *v: ExtendedBeta(*v).log_prob(torch.tensor(0.3)), parameters
    )
    for target, oracle_index in [(law.mean, 2), (law.variance, 3)]:
        actual = torch.autograd.grad(target, parameters, retain_graph=True)
        for index, gradient in enumerate(actual):
            values = [0.3, 4.0, 0.2]
            step = 1e-5
            values[index] += step
            upper = _oracle(*values)[oracle_index]
            values[index] -= 2 * step
            lower = _oracle(*values)[oracle_index]
            assert gradient.item() == pytest.approx(
                (upper - lower) / (2 * step), abs=2e-8, rel=2e-6
            )
    assert all(torch.isfinite(x) for x in torch.autograd.grad(score, parameters))


def test_moment_batches_have_no_shape_collisions_and_keep_learned_head_gradients():
    mu = torch.tensor(
        [[0.2, 0.5, 0.8], [0.3, 0.4, 0.6]], dtype=torch.float64, requires_grad=True
    )
    phi = torch.tensor([0.4, 4.0, 30.0], dtype=torch.float64, requires_grad=True)
    kappa = torch.tensor([[0.0], [0.2]], dtype=torch.float64, requires_grad=True)
    law = ExtendedBeta(mu, phi, kappa)
    mean, variance = law.mean, law.variance
    assert mean.shape == variance.shape == (2, 3)
    for row in range(2):
        for col in range(3):
            oracle = _oracle(mu[row, col].item(), phi[col].item(), kappa[row, 0].item())
            assert mean[row, col].item() == pytest.approx(oracle[2], abs=2e-10)
            assert variance[row, col].item() == pytest.approx(oracle[3], abs=2e-10)
    assert all(
        torch.isfinite(x).all()
        for x in torch.autograd.grad(mean.sum() + variance.sum(), (mu, phi, kappa))
    )
    assert law.expand((5, 2, 3)).sample((4,)).shape == (4, 5, 2, 3)


def test_sampling_preserves_atoms_and_response_moments_with_pathwise_gradients():
    law = ExtendedBeta(0.3, 4.0, 0.2)
    p0, p1, mean, variance = _oracle(0.3, 4.0, 0.2)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        sample = law.sample((60000,))
    assert sample.eq(0).double().mean().item() == pytest.approx(p0, abs=0.006)
    assert sample.eq(1).double().mean().item() == pytest.approx(p1, abs=0.001)
    assert sample.mean().item() == pytest.approx(mean, abs=0.004)
    assert sample.var().item() == pytest.approx(variance, abs=0.003)
    params = tuple(
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in (0.3, 4.0, 0.2)
    )
    draw = ExtendedBeta(*params).rsample((100,)).mean()
    assert all(torch.isfinite(x) for x in torch.autograd.grad(draw, params))


def test_cdf_support_boundaries_and_unrepresentable_extensions_are_explicit():
    law = ExtendedBeta(0.3, 4.0, 0.2)
    assert law.cdf(torch.tensor(-1.0)).item() == 0
    assert law.cdf(torch.tensor(1.0)).item() == 1
    assert law.log_survival_function(torch.tensor(1.0)).item() == -math.inf
    assert law.log_prob(torch.tensor(-0.1)).item() == -math.inf
    assert law.log_prob(torch.tensor(1.1)).item() == -math.inf
    with pytest.raises(RuntimeError, match="resolution"):
        ExtendedBeta(0.5, 2.0, 2**53)
    with pytest.raises(RuntimeError, match="budget"):
        _ = ExtendedBeta(0.8, 0.3, 0.0001, moment_max_panels=1).mean


def test_log_gamma_helper_preserves_existing_rng_order_and_extreme_weights():
    concentration = torch.tensor(
        [[0.1, 0.4, 2.0], [1e-310, 2e-310, 1e-310]], dtype=torch.float64
    )
    with torch.random.fork_rng():
        torch.manual_seed(82)
        logs = _dirichlet_log_weights(concentration)
        state = torch.random.get_rng_state()
        torch.manual_seed(82)
        values = _dirichlet_weights(concentration)
        assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(logs.exp(), values)
    torch.testing.assert_close(values.sum(-1), torch.ones(2, dtype=torch.float64))
    assert not logs.isnan().any()


def test_tiny_extension_classifies_atoms_before_upper_coordinate_rounding(monkeypatch):
    import quivers.continuous._affine_beta as module

    kappa = 1e-16
    q = torch.tensor([0.5 * kappa, 2 * kappa], dtype=torch.float64)
    logs = torch.stack((torch.log1p(-q), q.log()), -1)
    monkeypatch.setattr(module, "_dirichlet_log_weights", lambda *a, **kw: logs)
    sample = ExtendedBeta(0.3, 0.4, kappa).sample((2,))
    assert sample[0].item() == 1.0
    assert 0 < sample[1].item() < 1.0
    q = torch.tensor([1.01 * kappa], dtype=torch.float64)
    logs = torch.stack((torch.log1p(-q), q.log()), -1)
    with pytest.raises(RuntimeError, match="endpoint resolution"):
        ExtendedBeta(0.3, 0.4, kappa).sample((1,))


def test_left_cdf_preserves_the_two_censoring_atoms():
    law = ExtendedBeta(0.3, 4.0, 0.2)
    assert law.log_cdf_left(0.0).item() == -math.inf
    assert law.log_survival_function_left(0.0).item() == 0.0
    assert law.log_cdf(1.0).item() == 0.0
    assert law.log_survival_function(1.0).item() == -math.inf
    torch.testing.assert_close(law.log_survival_function_left(1.0), law.log_prob(1.0))
    torch.testing.assert_close(law.log_cdf(0.0), law.log_prob(0.0))


def test_tiny_positive_extension_moment_gradient_retains_endpoint_tail_contributions():
    kappa = torch.tensor(1e-16, dtype=torch.float64, requires_grad=True)
    law = ExtendedBeta(0.3, 0.04, kappa)
    actual = torch.autograd.grad(law.mean, kappa)[0].item()
    variance_gradient = torch.autograd.grad(law.variance, kappa)[0].item()
    with mp.workdps(80):
        mu, phi, k = map(mp.mpf, (0.3, 0.04, 1e-16))
        a, b, cutoff = mu * phi, (1 - mu) * phi, k / (1 + 2 * k)
        interior0 = (
            1
            - mp.betainc(a, b, 0, cutoff, regularized=True)
            - mp.betainc(b, a, 0, cutoff, regularized=True)
        )
        interior1 = mu * (
            1
            - mp.betainc(a + 1, b, 0, cutoff, regularized=True)
            - mp.betainc(b, a + 1, 0, cutoff, regularized=True)
        )
        expected = float(2 * interior1 - interior0)
        interior2 = (
            mu
            * (a + 1)
            / (phi + 1)
            * (
                1
                - mp.betainc(a + 2, b, 0, cutoff, regularized=True)
                - mp.betainc(b, a + 2, 0, cutoff, regularized=True)
            )
        )
        expected_variance = float(
            2
            * (
                (1 + 2 * k) * (2 * interior2 - interior1)
                - (k + mu) * (2 * interior1 - interior0)
            )
        )
    assert actual == pytest.approx(expected, abs=2e-12)
    assert variance_gradient == pytest.approx(expected_variance, abs=2e-12)
    assert abs(actual - (2 * 0.3 - 1)) > 0.01


def test_pathwise_sampling_refuses_unrepresentable_shape_gradients_without_restricting_samples():
    phi = torch.tensor(1e-300, dtype=torch.float64, requires_grad=True)
    law = ExtendedBeta(0.3, phi, 0.2)
    assert torch.isfinite(law.sample((20,))).all()
    with pytest.raises(RuntimeError, match="shape-gradient capacity"):
        law.rsample((20,))
