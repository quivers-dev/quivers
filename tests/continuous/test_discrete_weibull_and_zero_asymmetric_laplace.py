"""Count survival and atomic quantile laws match independent numerical oracles."""

import math
import random

import numpy as np
import pytest
import torch

from quivers.continuous._asymmetric_laplace import AsymmetricLaplace
from quivers.continuous._discrete_weibull import DiscreteWeibull
from quivers.continuous._zero_inflated_asymmetric_laplace import (
    ZeroInflatedAsymmetricLaplace,
)
from quivers.qiec.distributions import ReferenceBackend


def _double(value, *, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def _dw_oracle(q, power, values):
    values = np.asarray(values, dtype=np.longdouble)
    q, power = np.longdouble(q), np.longdouble(power)
    return q**values**power - q ** (values + 1) ** power


def _al_log_density(loc, scale, quantile, value):
    residual = (np.asarray(value) - loc) / scale
    loss = residual * (quantile - (residual < 0))
    return math.log(quantile) + math.log1p(-quantile) - math.log(scale) - loss


def _al_cdf(loc, scale, quantile, value):
    residual = (np.asarray(value) - loc) / scale
    left = quantile * np.exp((1 - quantile) * np.minimum(residual, 0))
    right = 1 - (1 - quantile) * np.exp(-quantile * np.maximum(residual, 0))
    return np.where(residual < 0, left, right)


@pytest.mark.parametrize(
    "q,power", [(0.0, 0.7), (0.2, 0.4), (0.4, 1.3), (0.8, 2.5), (0.9, 4.0)]
)
def test_discrete_weibull_density_and_cdf_match_direct_survival_powers(q, power):
    law = DiscreteWeibull(_double(q), _double(power))
    values = torch.arange(12, dtype=torch.float64)
    probabilities = _dw_oracle(q, power, values.numpy()).astype(float)
    np.testing.assert_allclose(
        law.log_prob(values).exp().numpy(), probabilities, rtol=2e-12, atol=1e-16
    )
    expected_cdf = 1 - np.longdouble(q) ** (
        values.numpy().astype(np.longdouble) + 1
    ) ** np.longdouble(power)
    np.testing.assert_allclose(
        law.cdf(values).numpy(), expected_cdf.astype(float), rtol=2e-12, atol=1e-16
    )
    torch.testing.assert_close(
        torch.logaddexp(law.log_cdf(values), law.log_survival_function(values)),
        torch.zeros_like(values),
    )
    assert (law.log_prob(_double([-1.0, 0.5])) == -torch.inf).all()


@pytest.mark.parametrize("power", [0.4, 0.9, 1.0, 2.0, 20.0])
def test_discrete_weibull_density_tails_and_first_second_parameter_derivatives(power):
    q, concentration = _double(0.6, grad=True), _double(power, grad=True)
    values = _double([2.0] if power > 10 else [0.0, 1.0, 4.0, 15.0])

    def function(a, b):
        return DiscreteWeibull(a, b).log_prob(values)

    assert torch.autograd.gradcheck(function, (q, concentration))
    assert torch.autograd.gradgradcheck(function, (q, concentration))


@pytest.mark.parametrize(
    "q,power", [(0.2, 0.4), (0.4, 1.3), (0.6, 0.7), (0.8, 2.5), (0.9, 4.0)]
)
def test_discrete_weibull_certified_moments_match_independent_long_double_sum(q, power):
    law = DiscreteWeibull(q, power)
    values = np.arange(100000, dtype=np.longdouble)
    probabilities = _dw_oracle(q, power, values)
    expected_mean = np.sum(values * probabilities)
    expected_variance = np.sum((values - expected_mean) ** 2 * probabilities)
    assert law.mean.item() == pytest.approx(float(expected_mean), rel=2e-10, abs=1e-14)
    assert law.variance.item() == pytest.approx(
        float(expected_variance), rel=2e-10, abs=1e-14
    )


def test_discrete_weibull_geometric_limit_is_exact_for_large_mean():
    q = _double(1 - 1e-10)
    law = DiscreteWeibull(q, _double(1))
    torch.testing.assert_close(law.mean, q / (1 - q), rtol=0, atol=0)
    torch.testing.assert_close(law.variance, q / (1 - q) ** 2, rtol=0, atol=0)


def test_discrete_weibull_moment_gradients_include_shape_at_the_geometric_limit():
    q, power = _double(0.4, grad=True), _double(1, grad=True)
    for method in (lambda law: law.mean, lambda law: law.variance):

        def function(a, b):
            return method(DiscreteWeibull(a, b))

        assert torch.autograd.gradcheck(function, (q, power), eps=1e-5, atol=2e-7)
        assert torch.autograd.gradgradcheck(function, (q, power), eps=1e-5, atol=3e-6)


def test_discrete_weibull_zero_endpoint_retains_density_and_moment_derivatives():
    q, power = _double(0, grad=True), _double(0.7, grad=True)
    law = DiscreteWeibull(q, power)
    for value, derivative in [
        (law.log_prob(_double(0)), -1.0),
        (law.log_cdf(_double(0)), -1.0),
        (law.mean, 1.0),
        (law.variance, 1.0),
    ]:
        actual = torch.autograd.grad(
            value, (q, power), allow_unused=True, retain_graph=True
        )
        assert actual[0].item() == derivative
        assert actual[1] is None or actual[1].item() == 0
    assert torch.equal(law.sample((100,)), torch.zeros(100, dtype=torch.long))


def test_discrete_weibull_moment_budget_refuses_uncertified_estimates():
    law = DiscreteWeibull(0.8, 0.3, moment_max_terms=1)
    with pytest.raises(NotImplementedError, match="did not certify"):
        _ = law.mean
    with pytest.raises(NotImplementedError, match="did not certify"):
        _ = law.variance


@pytest.mark.parametrize("q,power", [(0.2, 0.5), (0.5, 1.0), (0.7, 2.0)])
def test_discrete_weibull_quantiles_and_vectorized_samples_match_count_cdf(q, power):
    law = DiscreteWeibull(q, power)
    probabilities = _double([0.0, 1e-8, 0.1, 0.5, 0.9, 1 - 1e-8, 1.0])
    quantiles = law.icdf(probabilities)
    finite = torch.isfinite(quantiles)
    assert (law.cdf(quantiles[finite]) >= probabilities[finite]).all()
    previous = quantiles[finite] - 1
    assert (
        (law.cdf(previous) < probabilities[finite]) | (probabilities[finite] == 0)
    ).all()
    assert torch.isinf(quantiles[-1])
    torch.manual_seed(19)
    sample = law.expand((2,)).sample((30000,))
    assert sample.shape == (30000, 2) and sample.dtype == torch.long
    for point in range(4):
        expected = law.cdf(_double(point)).item()
        actual = (sample <= point).double().mean().item()
        standard_error = math.sqrt(
            max(expected * (1 - expected), 1e-12) / sample.numel()
        )
        assert abs(actual - expected) <= 6 * standard_error + 2 / sample.numel()


def test_discrete_weibull_sampling_refuses_unrepresentable_counts():
    with pytest.raises(RuntimeError, match="2\\*\\*53"):
        DiscreteWeibull(0.9, 0.001).sample((20,))


@pytest.mark.parametrize("zero_prob", [0.0, 0.2, 0.8, 1.0])
@pytest.mark.parametrize("loc,quantile", [(-0.4, 0.2), (0.7, 0.7)])
def test_zero_inflated_asymmetric_laplace_density_cdf_and_moments(
    zero_prob, loc, quantile
):
    law = ZeroInflatedAsymmetricLaplace(*map(_double, (zero_prob, loc, 0.7, quantile)))
    values = _double([-3.0, -0.1, 0.0, 0.1, 3.0])
    expected = (
        np.log1p(-zero_prob) + _al_log_density(loc, 0.7, quantile, values.numpy())
        if zero_prob < 1
        else np.full(5, -np.inf)
    )
    expected[2] = math.log(zero_prob) if zero_prob > 0 else -math.inf
    np.testing.assert_allclose(
        law.log_prob(values).numpy(), expected, rtol=1e-13, atol=1e-13
    )
    expected_cdf = (1 - zero_prob) * _al_cdf(
        loc, 0.7, quantile, values.numpy()
    ) + zero_prob * (values.numpy() >= 0)
    np.testing.assert_allclose(
        law.cdf(values).numpy(), expected_cdf, rtol=1e-13, atol=1e-13
    )
    torch.testing.assert_close(
        torch.logaddexp(law.log_cdf(values), law.log_survival_function(values)),
        torch.zeros_like(values),
    )
    component_mean = loc + 0.7 * (1 - 2 * quantile) / (quantile * (1 - quantile))
    component_variance = 0.7**2 * (1 / quantile**2 + 1 / (1 - quantile) ** 2)
    assert law.mean.item() == pytest.approx((1 - zero_prob) * component_mean)
    assert law.variance.item() == pytest.approx(
        (1 - zero_prob) * (component_variance + zero_prob * component_mean**2)
    )


def test_zero_inflated_asymmetric_laplace_quantiles_cover_jump_and_closed_endpoints():
    law = ZeroInflatedAsymmetricLaplace(*map(_double, (0.3, 0.1, 0.7, 0.4)))
    left = 0.7 * law._component().cdf(_double(0)).item()
    probabilities = _double([0.0, 0.05, left + 0.01, left + 0.29, 0.9, 1.0])
    values = law.icdf(probabilities)
    assert values[2:4].tolist() == [0.0, 0.0]
    assert values[0] == -torch.inf and values[-1] == torch.inf
    torch.testing.assert_close(law.cdf(values[[1, 4]]), probabilities[[1, 4]])
    assert torch.equal(
        ZeroInflatedAsymmetricLaplace(1.0, 0.1, 0.7, 0.4).icdf(probabilities),
        torch.zeros_like(probabilities),
    )


def test_zero_inflated_asymmetric_laplace_first_second_density_and_tail_derivatives():
    arguments = tuple(_double(x, grad=True) for x in (0.3, 0.1, 0.7, 0.4))
    points = _double([-2.0, 0.0, 2.0])
    for method in ("log_prob", "log_cdf", "log_survival_function"):

        def function(*args):
            return getattr(ZeroInflatedAsymmetricLaplace(*args), method)(points)

        assert torch.autograd.gradcheck(function, arguments)
        assert torch.autograd.gradgradcheck(function, arguments)


@pytest.mark.parametrize("zero_prob", [0.0, 1.0])
def test_zero_inflated_asymmetric_laplace_finite_endpoint_score_derivatives(zero_prob):
    zi, loc, scale, quantile = tuple(
        _double(x, grad=True) for x in (zero_prob, 0.1, 0.7, 0.4)
    )
    law = ZeroInflatedAsymmetricLaplace(zi, loc, scale, quantile)
    point = _double(0.0 if zero_prob == 1 else 0.5)
    score = law.log_prob(point)
    derivative = torch.autograd.grad(score, (zi, loc, scale, quantile))
    assert derivative[0].item() == (1.0 if zero_prob == 1 else -1.0)
    assert all(torch.isfinite(x) for x in derivative)
    if zero_prob == 1:
        assert all(x.item() == 0 for x in derivative[1:])


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (AsymmetricLaplace, (0.1, 0.7, 0.4)),
        (ZeroInflatedAsymmetricLaplace, (0.3, 0.1, 0.7, 0.4)),
        (DiscreteWeibull, (0.4, 1.3)),
    ],
)
@pytest.mark.parametrize("bound", [-math.inf, math.inf])
def test_infinite_tail_bounds_have_zero_parameter_gradients(cls, parameters, bound):
    args = tuple(_double(x, grad=True) for x in parameters)
    law = cls(*args)
    for method in (law.log_cdf, law.log_survival_function):
        value = method(_double(bound))
        assert value.item() in (0.0, -math.inf)
        assert all(x.item() == 0 for x in torch.autograd.grad(value, args))


def test_asymmetric_laplace_log_tails_remain_finite_after_probability_underflow():
    law = AsymmetricLaplace(*map(_double, (0.0, 0.1, 0.2)))
    assert law.log_cdf(_double(-1000)).item() == pytest.approx(math.log(0.2) - 8000)
    assert law.log_survival_function(_double(1000)).item() == pytest.approx(
        math.log(0.8) - 2000
    )


def test_zero_atom_sampler_does_not_evaluate_inactive_huge_component(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inactive component was sampled")

    monkeypatch.setattr(AsymmetricLaplace, "sample", forbidden)
    law = ZeroInflatedAsymmetricLaplace(1.0, 1e30, 1e30, 0.1)
    assert torch.equal(law.sample((20,)), torch.zeros(20))


def test_zero_inflated_asymmetric_laplace_sampling_matches_atom_frequency_and_component_moments():
    law = ZeroInflatedAsymmetricLaplace(*map(_double, (0.3, 0.1, 0.7, 0.4)))
    torch.manual_seed(17)
    draws = law.expand((2,)).sample((50000,))
    assert draws.shape == (50000, 2) and draws.dtype == torch.float64
    assert (draws == 0).double().mean().item() == pytest.approx(0.3, abs=0.008)
    standard_error = law.variance.sqrt().item() / math.sqrt(draws.numel())
    assert abs(draws.mean().item() - law.mean.item()) < 6 * standard_error
    assert draws.var().item() == pytest.approx(law.variance.item(), rel=0.04)


@pytest.mark.parametrize(
    "name,parameters,points",
    [
        ("DiscreteWeibull", {"probs": 0.4, "concentration": 1.3}, [0.0, 1.0, 4.0]),
        (
            "ZeroInflatedAsymmetricLaplace",
            {"zero_prob": 0.3, "loc": 0.1, "scale": 0.7, "quantile": 0.4},
            [-2.0, 0.0, 2.0],
        ),
    ],
)
def test_checked_reference_density_and_sampling_follow_independent_law(
    name, parameters, points
):
    backend = ReferenceBackend()
    cls = (
        DiscreteWeibull if name == "DiscreteWeibull" else ZeroInflatedAsymmetricLaplace
    )
    law = cls(**{k: _double(v) for k, v in parameters.items()})
    for point in points:
        assert backend.log_prob(name, parameters, point) == pytest.approx(
            law.log_prob(_double(point)).item(), abs=2e-13
        )
    rng = random.Random(7)
    draws = np.asarray([backend.sample(name, parameters, rng) for _ in range(5000)])
    if name == "DiscreteWeibull":
        assert np.mean(draws <= 1) == pytest.approx(
            law.cdf(_double(1)).item(), abs=0.025
        )
    else:
        assert np.mean(draws == 0) == pytest.approx(0.3, abs=0.025)


def test_float32_heads_keep_gradient_dtype_and_count_law_uses_float64():
    q = torch.tensor(0.4, requires_grad=True)
    power = torch.tensor(1.3, requires_grad=True)
    score = DiscreteWeibull(q, power).log_prob(_double(2))
    assert score.dtype == torch.float64
    assert all(g.dtype == torch.float32 for g in torch.autograd.grad(score, (q, power)))


def test_discrete_weibull_moment_forward_and_reverse_derivatives_agree_at_shape_one():
    parameters = _double([0.4, 1.0])

    def function(values):
        law = DiscreteWeibull(values[0], values[1], validate_args=False)
        return torch.stack((law.mean, law.variance))

    torch.testing.assert_close(
        torch.func.jacfwd(function)(parameters), torch.func.jacrev(function)(parameters)
    )


def test_discrete_weibull_moment_batch_keeps_independent_row_parameters():
    q = _double([[0.0, 0.4], [0.7, 0.2]], grad=True)
    power = _double([[0.7, 1.0], [2.0, 0.5]], grad=True)
    law = DiscreteWeibull(q, power)
    for parameter in ("mean", "variance"):
        actual = getattr(law, parameter)
        expected = torch.stack(
            tuple(
                getattr(DiscreteWeibull(a, b), parameter)
                for a, b in zip(q.flatten(), power.flatten(), strict=True)
            )
        ).reshape(2, 2)
        torch.testing.assert_close(actual, expected)
        assert all(
            torch.isfinite(x).all()
            for x in torch.autograd.grad(actual.sum(), (q, power), retain_graph=True)
        )


@pytest.mark.parametrize("zero_prob", [0.0, 1.0])
def test_zero_inflated_asymmetric_laplace_cdf_endpoint_derivatives_match_linear_mixture(
    zero_prob,
):
    zi = _double(zero_prob, grad=True)
    law = ZeroInflatedAsymmetricLaplace(zi, _double(0.1), _double(0.7), _double(0.4))
    for point in [-1.0, 0.0, 1.0]:
        value = _double(point)
        actual = law.cdf(value)
        component = law._component().cdf(value)
        expected_derivative = float(point >= 0) - component.item()
        assert torch.autograd.grad(actual, zi)[0].item() == pytest.approx(
            expected_derivative, abs=1e-13
        )


@pytest.mark.parametrize("probability", [0.05, 0.9])
def test_zero_inflated_asymmetric_laplace_quantile_first_second_derivatives(
    probability,
):
    arguments = tuple(_double(x, grad=True) for x in (probability, 0.3, 0.1, 0.7, 0.4))

    def function(p, zi, loc, scale, quantile):
        return ZeroInflatedAsymmetricLaplace(zi, loc, scale, quantile).icdf(p)

    assert torch.autograd.gradcheck(function, arguments)
    assert torch.autograd.gradgradcheck(function, arguments)
