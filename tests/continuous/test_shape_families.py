"""Skew, shifted, and extreme-value laws match independent numerical oracles."""

import math
import random

import numpy as np
import pytest
from scipy import stats
import torch

from quivers.continuous import (
    SkewNormal,
    ShiftedLogNormal,
    Frechet,
    GeneralizedExtremeValue,
)
from quivers.qiec.distributions import ReferenceBackend


CASES = [
    *[
        (SkewNormal, (0.2, 1.3, a), stats.skewnorm(a, loc=0.2, scale=1.3))
        for a in (-4.0, 0.0, 2.0)
    ],
    (
        ShiftedLogNormal,
        (0.2, 0.5, 0.3),
        stats.lognorm(0.5, loc=0.3, scale=math.exp(0.2)),
    ),
    *[
        (Frechet, (1.3, a), stats.invweibull(a, scale=1.3))
        for a in (0.7, 1.2, 2.1, 3.0)
    ],
    *[
        (
            GeneralizedExtremeValue,
            (0.2, 1.3, c),
            stats.genextreme(-c, loc=0.2, scale=1.3),
        )
        for c in (-2.0, -1.0, -0.3, 0.0, 1e-10, 0.3, 0.8, 1.2)
    ],
]


def _law(cls, parameters, dtype=torch.float64):
    return cls(*(torch.tensor(v, dtype=dtype) for v in parameters))


@pytest.mark.parametrize("cls,parameters,oracle", CASES)
def test_density_cdf_survival_quantiles_and_moments_match_scipy(
    cls, parameters, oracle
):
    law = _law(cls, parameters)
    p = torch.tensor([0.001, 0.02, 0.2, 0.5, 0.8, 0.98, 0.999], dtype=torch.float64)
    value = torch.tensor(oracle.ppf(p.numpy()), dtype=torch.float64)
    np.testing.assert_allclose(
        law.log_prob(value), oracle.logpdf(value), rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        law.log_cdf(value), oracle.logcdf(value), rtol=3e-10, atol=3e-11
    )
    np.testing.assert_allclose(
        law.log_survival_function(value), oracle.logsf(value), rtol=3e-10, atol=3e-11
    )
    torch.testing.assert_close(law.cdf(value), p, rtol=3e-10, atol=3e-11)
    torch.testing.assert_close(law.icdf(p), value, rtol=4e-9, atol=3e-10)
    mean_exists = not (
        (cls is Frechet and parameters[1] <= 1)
        or (cls is GeneralizedExtremeValue and parameters[2] >= 1)
    )
    variance_exists = not (
        (cls is Frechet and parameters[1] <= 2)
        or (cls is GeneralizedExtremeValue and parameters[2] >= 0.5)
    )
    stable_oracle_moments = (
        cls is not GeneralizedExtremeValue
        or parameters[2] == 0
        or abs(parameters[2]) > 1e-6
    )
    if mean_exists and stable_oracle_moments and np.isfinite(oracle.mean()):
        assert law.mean.item() == pytest.approx(oracle.mean(), rel=2e-10, abs=2e-11)
    if variance_exists and stable_oracle_moments and np.isfinite(oracle.var()):
        assert law.variance.item() == pytest.approx(oracle.var(), rel=2e-10, abs=2e-11)


@pytest.mark.parametrize("cls,parameters,oracle", CASES)
def test_first_second_derivatives_density_and_log_tails(cls, parameters, oracle):
    params = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in parameters
    )
    value = torch.tensor(oracle.ppf([0.2, 0.6]), dtype=torch.float64)

    def function(*args):
        law = cls(*args)
        return torch.cat(
            (law.log_prob(value), law.log_cdf(value), law.log_survival_function(value))
        )

    assert torch.autograd.gradcheck(function, params)
    assert torch.autograd.gradgradcheck(function, params)


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (SkewNormal, (0.2, 1.3, 0.0)),
        (SkewNormal, (0.2, 1.3, 2.0)),
        (ShiftedLogNormal, (0.2, 0.5, 0.3)),
        (Frechet, (1.3, 3.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.2)),
    ],
)
def test_quantile_first_second_derivatives(cls, parameters):
    params = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in parameters
    )
    p = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)

    def function(probability, *args):
        return cls(*args).icdf(probability)

    assert torch.autograd.gradcheck(function, (p, *params))
    assert torch.autograd.gradgradcheck(function, (p, *params))


@pytest.mark.parametrize("cls,parameters,oracle", CASES)
def test_scalar_reference_density_and_support_sampling(cls, parameters, oracle):
    arguments = dict(zip(cls.arg_constraints, parameters, strict=True))
    backend = ReferenceBackend()
    for point in oracle.ppf([0.1, 0.4, 0.8]):
        assert backend.family_log_prob(
            cls.__name__, arguments, float(point)
        ) == pytest.approx(oracle.logpdf(point), abs=2e-11)
    rng = random.Random(71)
    draws = np.array(
        [backend.family_sample(cls.__name__, arguments, rng) for _ in range(4000)]
    )
    assert np.mean(draws <= oracle.ppf(0.3)) == pytest.approx(0.3, abs=0.025)


@pytest.mark.parametrize("cls,parameters,oracle", CASES)
def test_native_sampling_shape_dtype_device_and_quantiles(cls, parameters, oracle):
    torch.manual_seed(17)
    law = _law(cls, parameters).expand((2,))
    draws = law.rsample((10000,))
    assert draws.shape == (10000, 2)
    assert draws.dtype == torch.float64
    assert draws.device == law.mean.device
    assert bool(torch.isfinite(draws).all())
    assert ((draws <= oracle.ppf(0.3)).double().mean()).item() == pytest.approx(
        0.3, abs=0.018
    )


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (SkewNormal, (0.2, 1.3, 0.0)),
        (ShiftedLogNormal, (0.2, 0.5, 0.3)),
        (Frechet, (1.3, 3.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.0)),
    ],
)
def test_pathwise_sample_mean_derivatives_agree_with_analytic_mean(cls, parameters):
    torch.manual_seed(40)
    params = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in parameters
    )
    law = cls(*params)
    actual = torch.autograd.grad(law.rsample((100000,)).mean(), params)
    expected = torch.autograd.grad(law.mean, params)
    for a, e in zip(actual, expected, strict=True):
        assert a.item() == pytest.approx(e.item(), abs=0.025, rel=0.04)


@pytest.mark.parametrize(
    "cls,parameters,value",
    [
        (ShiftedLogNormal, (0.2, 0.5, 0.3), 0.2),
        (Frechet, (1.3, 3.0), -0.1),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.3), -5.0),
        (GeneralizedExtremeValue, (0.2, 1.3, -0.3), 6.0),
    ],
)
def test_parameter_dependent_outside_support_scores_negative_infinity_with_zero_gradients(
    cls, parameters, value
):
    params = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in parameters
    )
    law = cls(*params)
    score = law.log_prob(torch.tensor(value, dtype=torch.float64))
    assert score.item() == -math.inf
    gradients = torch.autograd.grad(score, params)
    assert all(g.item() == 0 for g in gradients)
    cdf, survival = (
        law.log_cdf(torch.tensor(value)),
        law.log_survival_function(torch.tensor(value)),
    )
    assert not torch.isnan(cdf) and not torch.isnan(survival)


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (Frechet, (1.3, 0.7)),
        (Frechet, (1.3, 1.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 1.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 2.0)),
    ],
)
def test_nonexistent_first_moments_are_explicit(cls, parameters):
    law = _law(cls, parameters)
    assert law.mean.item() == math.inf
    assert law.variance.item() == math.inf


@pytest.mark.parametrize(
    "cls,parameters",
    [(Frechet, (1.3, 2.0)), (GeneralizedExtremeValue, (0.2, 1.3, 0.5))],
)
def test_finite_mean_with_infinite_variance(cls, parameters):
    law = _law(cls, parameters)
    assert torch.isfinite(law.mean)
    assert law.variance.item() == math.inf


def test_frechet_concentrated_limit_keeps_small_variance():
    law = _law(Frechet, (1.3, 1e10))
    assert law.variance.item() == pytest.approx(
        1.3**2 * (math.pi**2 / 6) / 1e20, rel=2e-9
    )


def test_zero_shape_extreme_value_moment_derivatives_are_smooth():
    c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    law = GeneralizedExtremeValue(torch.tensor(0.2), torch.tensor(1.3), c)
    for method in (
        lambda x: GeneralizedExtremeValue(0.0, 1.0, x).mean,
        lambda x: GeneralizedExtremeValue(0.0, 1.0, x).variance,
    ):
        assert torch.autograd.gradcheck(method, (c,))
        assert torch.autograd.gradgradcheck(method, (c,))
    assert law.mean.item() == pytest.approx(0.2 + 1.3 * np.euler_gamma, rel=1e-7)


def test_skew_zero_stationary_cdf_gradient_matches_density():
    point = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    shape = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    law = SkewNormal(0.0, 1.0, shape)
    cdf = law.cdf(point)
    dz, da = torch.autograd.grad(cdf, (point, shape), create_graph=True)
    assert cdf.item() == 0.5
    assert dz.item() == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-14)
    assert da.item() == pytest.approx(-1 / math.pi, abs=1e-14)
    assert torch.isfinite(torch.autograd.grad(dz, shape)[0])


def test_skew_cdf_reverse_forward_and_batched_transforms_agree():
    parameters = torch.tensor([-0.4, 0.7], dtype=torch.float64)

    def function(values):
        return SkewNormal(0.0, 1.0, values[1]).log_cdf(values[0])

    reverse = torch.func.jacrev(function)(parameters)
    forward = torch.func.jacfwd(function)(parameters)
    torch.testing.assert_close(reverse, forward)
    batched = torch.func.vmap(function)(
        torch.stack((parameters, parameters + torch.tensor([0.1, 0.2])))
    )
    assert batched.shape == (2,)


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (SkewNormal, (0.2, 1.3, 0.7)),
        (ShiftedLogNormal, (0.2, 0.5, 0.3)),
        (Frechet, (1.3, 3.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.2)),
    ],
)
@pytest.mark.parametrize("point", [-math.inf, math.inf])
def test_infinite_censoring_bounds_have_exact_probabilities_and_zero_parameter_gradients(
    cls, parameters, point
):
    params = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in parameters
    )
    law = cls(*params)
    value = torch.tensor(point, dtype=torch.float64)
    for method in (law.log_cdf, law.log_survival_function):
        result = method(value)
        assert result.item() in (0.0, -math.inf)
        assert all(g.item() == 0 for g in torch.autograd.grad(result, params))


@pytest.mark.parametrize(
    "point,skewness,expected",
    [
        (-0.001, 500.01, -8.0609538538608393358278568468037664),
        (-0.1, 5.0001, -3.4808662468604856679648517223662061),
        (-1.0, 0.50001, -2.6238307236724066979718453852661981),
        (
            -0.0013961353908567902,
            6480.286440412508,
            -55.288729430597879507405307653937283,
        ),
        (-40.0, 20.0, -320817.51218717461926614408221559799),
        (-1e-8, 1e6, -14.972802154561342039143973275923934),
    ],
)
def test_skew_extreme_log_tails_match_80_digit_integrated_density(
    point, skewness, expected
):
    # Constants integrate 2*phi(t)*Phi(alpha*t) with 80 decimal digits and a
    # boundary-scaled adaptive integral, independently of the native quadrature.
    alpha = torch.tensor(skewness, dtype=torch.float64, requires_grad=True)
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    law = SkewNormal(0.0, 1.0, alpha)
    actual = law.log_cdf(value)
    assert actual.item() == pytest.approx(expected, abs=8e-11)
    reflected = SkewNormal(0.0, 1.0, -alpha).log_survival_function(-value)
    torch.testing.assert_close(actual, reflected)
    derivatives = torch.autograd.grad(actual, (value, alpha))
    assert all(torch.isfinite(g) for g in derivatives)


@pytest.mark.parametrize("skewness", [-100.0, 0.0, 100.0])
def test_skew_quantiles_preserve_both_extreme_log_tails(skewness):
    law = _law(SkewNormal, (0.0, 1.0, skewness))
    p = torch.tensor(
        [1e-100, 1e-20, 1e-12, 0.2, 0.8, 1 - 1e-12, 1 - 1e-15], dtype=torch.float64
    )
    values = law.icdf(p)
    torch.testing.assert_close(
        law.log_cdf(values[:4]), p[:4].log(), atol=3e-10, rtol=3e-10
    )
    torch.testing.assert_close(
        law.log_survival_function(values[4:]),
        torch.log1p(-p[4:]),
        atol=3e-10,
        rtol=3e-10,
    )


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (SkewNormal, (0.2, 1.3, 0.7)),
        (ShiftedLogNormal, (0.2, 0.5, 0.3)),
        (Frechet, (1.3, 3.0)),
        (GeneralizedExtremeValue, (0.2, 1.3, 0.0)),
    ],
)
def test_float32_parameters_keep_output_and_gradient_dtypes(cls, parameters):
    args = tuple(
        torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in parameters
    )
    law = cls(*args)
    value = law.icdf(torch.tensor([0.2, 0.7]))
    score = (
        law.log_prob(value.detach()).sum()
        + law.log_cdf(value.detach()).sum()
        + law.log_survival_function(value.detach()).sum()
    )
    assert score.dtype == torch.float32
    for gradient in torch.autograd.grad(score, args):
        assert gradient.dtype == torch.float32
        assert torch.isfinite(gradient)


def test_skew_quantile_reverse_forward_and_batched_transforms_agree():
    parameters = torch.tensor([0.8, -1.3], dtype=torch.float64)

    def function(values):
        return SkewNormal(0.0, 1.0, values[1], validate_args=False).icdf(values[0])

    reverse = torch.func.jacrev(function)(parameters)
    forward = torch.func.jacfwd(function)(parameters)
    torch.testing.assert_close(reverse, forward)
    cases = torch.stack((parameters, torch.tensor([0.2, 2.0], dtype=torch.float64)))
    batched = torch.func.vmap(function)(cases)
    torch.testing.assert_close(
        batched, torch.stack(tuple(function(row) for row in cases))
    )
