"""Private count-tail kernels agree with independent cumulative probabilities."""

import math

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.special import logsumexp

from quivers.continuous._count_tails import (
    binomial_log_tails,
    negative_binomial_log_tails,
)
from quivers.continuous._count_tails import inflated_count_tails
from quivers.continuous._count_tails import beta_binomial_log_tails


@pytest.mark.parametrize(
    "n,p",
    [(0, 0.3), (1, 0.3), (10, 0.2), (1000, 0.4), (1000001, 0.01), (33554433, 0.5)],
)
def test_binomial_probability_tails_match_scipy(n, p):
    queries = torch.tensor(
        [-math.inf, -1.0, 0.0, 1.0, 2.0, n * p, n * p + 1, n - 1, n, math.inf],
        dtype=torch.float64,
    )
    actual = binomial_log_tails(
        torch.tensor(n), queries, torch.tensor(p, dtype=torch.float64)
    )
    lower, upper = (
        stats.binom.logcdf(queries.numpy(), n, p),
        stats.binom.logsf(queries.numpy(), n, p),
    )
    for index, value in enumerate(queries.tolist()):
        if 0 <= value <= 2 and n > value:

            def logmass(k):
                combination = math.fsum(
                    math.log(n - j) - math.log(j + 1) for j in range(k)
                )
                return combination + k * math.log(p) + (n - k) * math.log1p(-p)

            lower[index] = logsumexp([logmass(k) for k in range(math.floor(value) + 1)])
        if value == n - 1 and n > 0:
            upper[index] = n * math.log(p)
    for got, want in zip(actual, (lower, upper), strict=True):
        np.testing.assert_allclose(got.detach().numpy(), want, atol=2e-8, rtol=2e-9)


@pytest.mark.parametrize(
    "r,p", [(0.01, 0.2), (0.1, 0.7), (0.2, 0.7), (3.0, 0.2), (10000.0, 0.01)]
)
def test_negative_binomial_tails_match_scipy(r, p):
    queries = torch.tensor(
        [-math.inf, -1.0, 0.0, 1.0, 2.0, 4.0, 10.0, r * p / (1 - p), math.inf],
        dtype=torch.float64,
    )
    actual = negative_binomial_log_tails(
        torch.tensor(r, dtype=torch.float64),
        queries,
        torch.tensor(p, dtype=torch.float64),
    )
    for got, want in zip(
        actual,
        (
            stats.nbinom.logcdf(queries.numpy(), r, 1 - p),
            stats.nbinom.logsf(queries.numpy(), r, 1 - p),
        ),
        strict=True,
    ):
        np.testing.assert_allclose(got.detach().numpy(), want, atol=3e-9, rtol=3e-9)


@pytest.mark.parametrize(
    "kernel,total,query",
    [(binomial_log_tails, 10.0, 2.0), (negative_binomial_log_tails, 3.0, 2.0)],
)
def test_interior_first_second_gradients(kernel, total, query):
    p = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    n = torch.tensor(total, dtype=torch.float64, requires_grad=True)
    point = torch.tensor(query, dtype=torch.float64)
    # Only the NB shape is a continuous parameter.
    if kernel is negative_binomial_log_tails:
        assert torch.autograd.gradcheck(lambda r, p: kernel(r, point, p)[0], (n, p))
        assert torch.autograd.gradgradcheck(lambda r, p: kernel(r, point, p)[0], (n, p))
    else:
        assert torch.autograd.gradcheck(lambda p: kernel(n.detach(), point, p)[0], (p,))
        assert torch.autograd.gradgradcheck(
            lambda p: kernel(n.detach(), point, p)[0], (p,)
        )


def test_negative_binomial_zero_shape_has_real_one_sided_derivative():
    p = 0.4
    for k in [0, 1, 2, 10]:
        r = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        cdf = negative_binomial_log_tails(
            r, torch.tensor(float(k)), torch.tensor(p, dtype=torch.float64)
        )[0]
        derivative = torch.autograd.grad(cdf, r)[0]
        expected = math.log1p(-p) + sum(p**j / j for j in range(1, k + 1))
        assert cdf.item() == 0
        assert derivative.item() == pytest.approx(expected, abs=2e-13)


def test_extreme_finite_logits_keep_finite_short_log_tails():
    for logits in [-1000.0, 1000.0]:
        parameter = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
        cdf, sf = binomial_log_tails(
            torch.tensor(10.0), torch.tensor(2.0), parameter, logits=True
        )
        assert torch.isfinite(cdf) & torch.isfinite(sf)
        assert all(torch.isfinite(v) for v in torch.autograd.grad(cdf + sf, parameter))


@pytest.mark.parametrize(
    "kernel,total", [(binomial_log_tails, 10.0), (negative_binomial_log_tails, 3.0)]
)
@pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
def test_closed_zero_probability_retains_finite_first_and_second_cdf_derivatives(
    kernel, total, k
):
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    cdf = kernel(torch.tensor(total), torch.tensor(k), p)[0]
    first = torch.autograd.grad(cdf, p, create_graph=True)[0]
    second = torch.autograd.grad(first, p)[0]
    expected_first = -total if k == 0 else 0.0
    sign = -1 if kernel is binomial_log_tails else 1
    expected_second = -total if k == 0 else -total * (total + sign) if k == 1 else 0.0
    assert cdf.item() == 0
    assert first.item() == pytest.approx(expected_first, abs=1e-13)
    assert second.item() == pytest.approx(expected_second, abs=1e-13)


@pytest.mark.parametrize("distance", [0.0, 1.0, 2.0])
def test_closed_one_probability_retains_binomial_survival_derivatives(distance):
    p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    sf = binomial_log_tails(torch.tensor(10.0), torch.tensor(9.0 - distance), p)[1]
    first = torch.autograd.grad(sf, p, create_graph=True)[0]
    second = torch.autograd.grad(first, p)[0]
    assert sf.item() == 0
    assert first.item() == pytest.approx(10.0 if distance == 0 else 0.0)
    assert second.item() == pytest.approx(
        -10.0 if distance == 0 else -90.0 if distance == 1 else 0.0
    )


def test_small_positive_nb_shape_keeps_both_order_gradients_and_forward_ad():
    parameters = torch.tensor([0.01, 0.3], dtype=torch.float64, requires_grad=True)

    def score(v):
        return negative_binomial_log_tails(v[0], torch.tensor(2.0), v[1])[0]

    assert torch.autograd.gradcheck(score, (parameters,))
    assert torch.autograd.gradgradcheck(score, (parameters,))
    actual = torch.func.jacfwd(score)(parameters)
    torch.testing.assert_close(
        actual, torch.autograd.grad(score(parameters), parameters)[0]
    )


@pytest.mark.parametrize("zi", [0.0, 0.3, 1.0])
def test_inflated_probability_tails_keep_atom_derivatives_under_component_underflow(zi):
    probability = torch.tensor(zi, dtype=torch.float64, requires_grad=True)
    logcdf, logsf, cdf, sf = inflated_count_tails(
        probability,
        torch.tensor(2.0),
        torch.tensor(-1000.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert cdf.item() == pytest.approx(zi)
    assert sf.item() == pytest.approx(1 - zi)
    assert torch.autograd.grad(cdf, probability, retain_graph=True)[0].item() == 1
    assert torch.autograd.grad(sf, probability)[0].item() == -1
    if zi:
        assert logcdf.item() == pytest.approx(math.log(zi))
    else:
        assert logcdf.item() == -1000.0


def test_inflated_count_logtails_agree_with_scipy_component_mixture():
    query = torch.tensor([-1.0, 0.0, 2.0, 10.0, math.inf], dtype=torch.float64)
    cdf, sf = binomial_log_tails(
        torch.tensor(10.0), query, torch.tensor(0.3, dtype=torch.float64)
    )
    actual = inflated_count_tails(
        torch.tensor(0.2, dtype=torch.float64), query, cdf, sf
    )
    base = stats.binom.cdf(query.numpy(), 10, 0.3)
    expected = np.where(query.numpy() >= 0, 0.2 + 0.8 * base, 0.0)
    np.testing.assert_allclose(actual[2].detach().numpy(), expected, atol=2e-14)
    np.testing.assert_allclose(actual[3].detach().numpy(), 1 - expected, atol=2e-14)


@pytest.mark.parametrize(
    "n,a,b",
    [
        (0, 0.2, 0.3),
        (1, 1.0, 1.0),
        (10, 0.2, 0.3),
        (10, 0.2, 7.0),
        (50, 20.0, 0.1),
        (3000, 2.0, 3.0),
    ],
)
def test_beta_binomial_complete_finite_tails_match_scipy(n, a, b):
    query = torch.tensor(
        [-math.inf, -1.0, 0.0, 1.0, n // 3, n - 1.0, n, math.inf], dtype=torch.float64
    )
    actual = beta_binomial_log_tails(
        torch.tensor(n),
        query,
        torch.tensor(a, dtype=torch.float64),
        torch.tensor(b, dtype=torch.float64),
    )
    lower, upper = (
        stats.betabinom.logcdf(query.numpy(), n, a, b),
        stats.betabinom.logsf(query.numpy(), n, a, b),
    )
    if n > 0:
        # SciPy's beta-binomial SF subtracts its CDF and loses precision
        # when the last atom is tiny. That atom has a direct log-PMF oracle.
        upper[query.numpy() == n - 1] = stats.betabinom.logpmf(n, n, a, b)
    for got, want in zip(actual, (lower, upper), strict=True):
        np.testing.assert_allclose(got.detach().numpy(), want, atol=3e-9, rtol=3e-9)


def test_beta_binomial_ragged_batches_and_all_shape_gradients():
    n = torch.tensor([1.0, 4.0, 130.0, 3.0, 1000.0], dtype=torch.float64)
    query = torch.tensor([0.0, 2.0, 20.0, 1.0, 300.0], dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: beta_binomial_log_tails(n, query, a, b)[0], (a, b)
    )
    assert torch.autograd.gradgradcheck(
        lambda a, b: beta_binomial_log_tails(n, query, a, b)[0], (a, b)
    )


def test_beta_binomial_rejects_incomplete_finite_tail_budget():
    with pytest.raises(RuntimeError, match="max_terms"):
        beta_binomial_log_tails(
            torch.tensor(1000.0),
            torch.tensor(400.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
            max_terms=10,
        )
