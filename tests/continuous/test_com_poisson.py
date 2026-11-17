"""Independent count-series, automatic differentiation, and sampling oracles."""

import math
import random

import pytest
import torch
from scipy.special import gammaln, logsumexp
import numpy as np

from quivers.continuous._com_poisson import COMPoisson
from quivers.qiec._com_poisson import density, sample


def oracle(lr, nu):
    k = np.arange(20000, dtype=float)
    logw = k * lr - nu * gammaln(k + 1)
    logz = logsumexp(logw)
    p = np.exp(logw - logz)
    mean = np.sum(k * p)
    return logz, mean, np.sum((k - mean) ** 2 * p), np.sum(gammaln(k + 1) * p)


@pytest.mark.parametrize(
    "lr,nu",
    [(0.0, 0.2), (2.0, 0.5), (2.0, 1.0), (4.0, 2.0), (-15.0, 3.0), (100.0, 100.0)],
)
def test_density_moments_and_reference(lr, nu):
    logz, mean, var, _ = oracle(lr, nu)
    d = COMPoisson(log_rate=lr, dispersion=nu)
    points = torch.arange(12, dtype=torch.double)
    expected = points * lr - nu * torch.lgamma(points + 1) - logz
    torch.testing.assert_close(d.log_prob(points), expected, atol=3e-11, rtol=3e-11)
    assert d.mean.item() == pytest.approx(mean, rel=3e-11, abs=3e-11)
    assert d.variance.item() == pytest.approx(var, rel=3e-10, abs=3e-11)
    for k in [0, 1, 5]:
        assert density({"log_rate": lr, "dispersion": nu}, k) == pytest.approx(
            expected[k].item(), abs=3e-11
        )


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
def test_natural_score_and_second_forward_reverse_derivatives(nu):
    x = torch.tensor([1.0, nu], dtype=torch.double, requires_grad=True)

    def score(z):
        return COMPoisson(log_rate=z[0], dispersion=z[1]).log_prob(torch.tensor(3.0))

    _, mean, _, factorial = oracle(1.0, nu)
    expected = torch.tensor([3 - mean, factorial - math.lgamma(4)], dtype=torch.double)
    torch.testing.assert_close(
        torch.func.jacrev(score)(x), expected, atol=2e-10, rtol=2e-10
    )
    torch.testing.assert_close(
        torch.func.jacfwd(score)(x), expected, atol=2e-10, rtol=2e-10
    )
    assert torch.autograd.gradcheck(score, (x,))
    assert torch.autograd.gradgradcheck(score, (x,))
    torch.testing.assert_close(
        torch.func.jacfwd(torch.func.jacrev(score))(x),
        torch.func.jacrev(torch.func.jacrev(score))(x),
    )


def test_moment_derivatives_at_poisson_limit():
    x = torch.tensor([1.0, 1.0], dtype=torch.double, requires_grad=True)

    def moments(z):
        d = COMPoisson(log_rate=z[0], dispersion=z[1])
        return torch.stack((d.mean, d.variance))

    assert torch.autograd.gradcheck(moments, (x,))
    assert torch.autograd.gradgradcheck(moments, (x,))
    jac = torch.func.jacfwd(moments)(x)
    assert jac[:, 1].abs().min() > 1


def test_log_rate_parameterization_exceeds_rate_capacity():
    d = COMPoisson(log_rate=1000 * math.log(100), dispersion=1000.0)
    assert torch.isinf(d.rate)
    assert 99 < d.mean < 100
    assert 0.24 < d.variance < 0.26
    assert torch.isfinite(d.log_prob(torch.tensor(100.0)))
    tiny = COMPoisson(log_rate=-1000.0, dispersion=2.0)
    assert tiny.log_prob(torch.tensor(1.0)) == -1000.0


def test_zero_rate_and_invalid_support():
    d = COMPoisson(rate=0.0, dispersion=0.1)
    torch.testing.assert_close(
        d.log_prob(torch.tensor([0.0, 1.0, -1.0, 0.5])),
        torch.tensor([0.0, -torch.inf, -torch.inf, -torch.inf], dtype=torch.double),
    )
    assert d.mean == 0 and d.variance == 0
    assert not d.sample((20,)).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"rate": 1.0, "log_rate": 0.0},
        {"rate": -1.0},
        {"rate": float("inf")},
        {"rate": 1.0, "dispersion": 0.0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        COMPoisson(**kwargs)


def test_explicit_sum_capacity_refusal():
    with pytest.raises(NotImplementedError, match="certify"):
        COMPoisson(rate=10.0, dispersion=0.5, max_terms=1).log_prob(torch.tensor(0.0))
    with pytest.raises(NotImplementedError, match="capacity"):
        COMPoisson(log_rate=100.0, dispersion=1.0).mean


@pytest.mark.parametrize(
    "lr,nu",
    [(0.0, 0.2), (2.0, 0.5), (2.0, 1.0), (4.0, 2.0), (-4.0, 3.0), (100.0, 100.0)],
)
def test_sampler_moments_and_shape(lr, nu):
    torch.manual_seed(17)
    d = COMPoisson(log_rate=lr, dispersion=nu, max_terms=1)
    values = d.sample((40000,))
    _, mean, var, _ = oracle(lr, nu)
    assert values.dtype == torch.int64
    assert values.double().mean().item() == pytest.approx(
        mean, abs=6 * math.sqrt(var / len(values)) + 1e-3
    )
    assert values.double().var().item() == pytest.approx(var, rel=0.09, abs=0.015)
    assert d.expand((2, 3)).sample((4,)).shape == (4, 2, 3)


def test_sampler_reference_and_closed_probability_atom():
    rng = random.Random(7)
    draws = [sample({"rate": 2.0, "dispersion": 1.3}, rng) for _ in range(5000)]
    _, mean, var, _ = oracle(math.log(2), 1.3)
    assert np.mean(draws) == pytest.approx(mean, abs=6 * math.sqrt(var / len(draws)))
    d = COMPoisson(rate=torch.tensor([0.0, 2.0]), dispersion=0.7)
    assert not d.sample((100,))[:, 0].any()
    with pytest.raises(RuntimeError, match="capacity"):
        COMPoisson(log_rate=100.0, dispersion=1.0).sample()


def test_poisson_exact_law_without_dropping_dispersion_score():
    rate = torch.tensor([0.01, 1.0, 25.0, 1000.0], dtype=torch.double)
    point = torch.tensor([0.0, 2.0, 30.0, 999.0], dtype=torch.double)
    torch.testing.assert_close(
        COMPoisson(rate=rate, dispersion=1.0).log_prob(point),
        torch.distributions.Poisson(rate).log_prob(point),
        atol=3e-11,
        rtol=3e-11,
    )


def test_closed_rate_first_derivatives_and_expanded_views():
    rate = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    d = COMPoisson(rate=rate, dispersion=2.0)
    assert (
        torch.autograd.grad(d.log_prob(torch.tensor(0.0)), rate, retain_graph=True)[0]
        == -1
    )
    assert torch.autograd.grad(d.mean, rate, retain_graph=True)[0] == 1
    assert torch.autograd.grad(d.variance, rate, retain_graph=True)[0] == 1
    assert torch.autograd.grad(d.expand((3,)).mean.sum(), rate)[0] == 3


def test_compact_batch_preserves_float_head_gradients():
    lr = torch.tensor([-0.4, 1.0, 4.0], dtype=torch.float32, requires_grad=True)
    nu = torch.tensor([0.7, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    d = COMPoisson(log_rate=lr, dispersion=nu)
    result = d.log_prob(torch.tensor([0, 2, 8]))
    assert result.dtype == torch.double
    gradients = torch.autograd.grad(result.sum(), (lr, nu))
    assert all(
        value.dtype == torch.float32 and torch.isfinite(value).all()
        for value in gradients
    )


def test_rejection_budget_is_explicit_and_does_not_normalize(monkeypatch):
    # At lambda1,nu100, the uniform center proposes2, which is virtually impossible.
    def high_uniform(value):
        return torch.full_like(value, 0.999)

    monkeypatch.setattr(torch, "rand_like", high_uniform)
    with pytest.raises(RuntimeError, match="max_proposals"):
        COMPoisson(rate=1.0, dispersion=100.0, max_terms=1, max_proposals=1).sample()


def test_checked_source_rate_and_log_rate_schemas():
    from quivers.modeling import Model

    for parameter in ["rate=2.0", "log_rate=0.6931471805599453"]:
        source = f"""object Out : Real 1
program count[n: Nat](y: Tensor[Int]([n])): n -> Out
    sample theta <- Normal(loc=0.0, scale=1.0)
    observe y: n <- COMPoisson({parameter}, dispersion=1.3)
    return theta
export count
"""
        point = torch.tensor([0, 1, 4])
        model = Model(source).bind({"y": point})
        expected = COMPoisson(rate=2.0, dispersion=1.3).log_prob(
            point
        ).sum() - 0.5 * math.log(2 * math.pi)
        actual = model.trace({"theta": torch.tensor([0.0])}).log_joint.sum()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("lr,nu", [(0.0, 0.2), (2.0, 0.5), (2.0, 1.0), (4.0, 2.0)])
def test_sampler_cell_probabilities_against_independent_series(lr, nu):
    torch.manual_seed(2026)
    draws = COMPoisson(log_rate=lr, dispersion=nu).sample((50000,))
    logz, *_ = oracle(lr, nu)
    for count in range(7):
        probability = math.exp(count * lr - nu * math.lgamma(count + 1) - logz)
        observed = (draws == count).double().mean().item()
        assert observed == pytest.approx(
            probability,
            abs=6 * math.sqrt(probability * (1 - probability) / len(draws))
            + 1 / len(draws),
        )


def test_natural_hessian_is_negative_sufficient_statistic_covariance():
    lr, nu = 1.0, 1.0
    counts = np.arange(1000, dtype=float)
    stats = np.stack((counts, -gammaln(counts + 1)), axis=-1)
    logweights = counts * lr - nu * gammaln(counts + 1)
    probability = np.exp(logweights - logsumexp(logweights))
    centered = stats - (probability[:, None] * stats).sum(0)
    expected = -np.einsum("n,ni,nj->ij", probability, centered, centered)

    def score(z):
        return COMPoisson(log_rate=z[0], dispersion=z[1]).log_prob(torch.tensor(3.0))

    actual = torch.func.hessian(score)(torch.tensor([lr, nu], dtype=torch.double))
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=1e-10, rtol=1e-10
    )
