"""Log tails behind censored and truncated responses, against independent oracles.

Oracles: SciPy ``stats`` distribution functions, mpmath at 50 digits where
SciPy underflows or has no CDF (incomplete gamma and beta, von Mises
integrals, Conway-Maxwell-Poisson sums), and central finite differences of
those oracles for first and second derivatives. Tolerances are fixed from
float64 rounding before any comparison: 1e-12 relative for values, 1e-7 for
first and 1e-5 for second derivatives taken by finite differences.
"""

from __future__ import annotations

import math
from typing import cast

import mpmath as mp
import numpy as np
import pytest
import torch
from scipy import stats
from torch.distributions import Beta, Exponential, Gamma, StudentT, VonMises

from quivers.continuous._com_poisson import COMPoisson
from quivers.continuous._count_tails import negative_binomial_log_tails
from quivers.continuous._gamma_tails import log_gamma_tails
from quivers.continuous._interval import _log_tails, log_interval_mass
from quivers.continuous._observation_events import event_log_prob
from quivers.continuous._positive_hurdle import HurdleGamma, HurdleLogNormal
from quivers.continuous._recorded_event_tails import (
    VON_MISES_SERIES_FLOOR,
    invert_log_cdf,
    log_com_poisson_tails,
    log_exponential_tails,
    log_student_t_tails,
    log_von_mises_tails,
)
from quivers.continuous._zip_hurdle import MixtureNormal, ZeroOneInflatedBeta
from quivers.continuous.measure import Normalize, Restrict
from quivers.inference.mcmc.kernel import _rejects_position

mp.mp.dps = 50
VALUE_RTOL = 1e-12
FIRST_RTOL = 1e-7
SECOND_RTOL = 1e-5


@pytest.fixture(autouse=True)
def _float64():
    """Build every literal tensor in float64, matching the oracles' inputs."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _close(actual, expected, rtol):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    same = actual == expected
    error = np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))
    assert bool(np.all(same | (error <= rtol))), (actual, expected)


def _derivatives(function, point, step):
    """Central first and second differences of a scalar mpmath function."""
    point = [mp.mpf(value) for value in point]
    size = len(point)
    first, second = [], [[None] * size for _ in range(size)]
    for i in range(size):
        up = point.copy()
        down = point.copy()
        up[i] += step
        down[i] -= step
        first.append((function(*up) - function(*down)) / (2 * step))
        for j in range(size):
            corners = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                shifted = point.copy()
                shifted[i] += si * step
                shifted[j] += sj * step
                corners.append(si * sj * function(*shifted))
            second[i][j] = sum(corners) / (4 * step * step)
    return np.array(first, float), np.array(second, float)


def _autograd(function, point):
    parameters = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    value = function(parameters)
    (gradient,) = torch.autograd.grad(value, parameters, create_graph=True)
    hessian = torch.stack(
        [
            torch.autograd.grad(gradient[i], parameters, retain_graph=True)[0]
            for i in range(parameters.numel())
        ]
    )
    return value.item(), gradient.detach().numpy(), hessian.detach().numpy()


# Student t -----------------------------------------------------------------


@pytest.mark.parametrize("df", [0.6, 3.0, 30.0, 4000.0])
def test_student_t_tails_match_scipy_and_mpmath_including_far_tails(df):
    values = torch.tensor([-1e5, -40.0, -3.0, -0.2, 0.5, 0.5 + 1e-12, 0.9, 2.0, 60.0])
    log_cdf, log_sf = log_student_t_tails(
        torch.tensor(df), torch.tensor(0.5), torch.tensor(1.7), values
    )

    def exact(value, upper):
        nu, t = mp.mpf(df), (mp.mpf(value) - mp.mpf("0.5")) / mp.mpf("1.7")
        far = (
            mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + t * t), regularized=True)
            / 2
        )
        on_far_side = (t < 0) != upper
        return float(mp.log(far if on_far_side else 1 - far))

    _close(log_cdf.numpy(), [exact(v, False) for v in values.tolist()], VALUE_RTOL)
    _close(log_sf.numpy(), [exact(v, True) for v in values.tolist()], VALUE_RTOL)
    finite = values.abs() < 100
    _close(
        log_cdf[finite].numpy(),
        stats.t.logcdf(values[finite].numpy(), df, 0.5, 1.7),
        1e-11,
    )


def test_student_t_tails_are_exact_at_infinity_and_reject_nan():
    log_cdf, log_sf = log_student_t_tails(
        torch.tensor(3.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor([-math.inf, math.inf]),
    )
    assert log_cdf.tolist() == [-math.inf, 0.0]
    assert log_sf.tolist() == [0.0, -math.inf]
    with pytest.raises(ValueError, match="Student-t tails"):
        log_student_t_tails(
            torch.tensor(3.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
            torch.tensor(math.nan),
        )


@pytest.mark.parametrize("value", [0.5, 0.62, -2.0, 25.0])
@pytest.mark.parametrize("side", [0, 1])
def test_student_t_tail_derivatives_in_df_location_and_scale(value, side):
    point = (3.3, 0.5, 1.7)

    def exact(nu, mu, sigma):
        t = (mp.mpf(value) - mu) / sigma
        far = (
            mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + t * t), regularized=True)
            / 2
        )
        on_far_side = (t < 0) != bool(side)
        return mp.log(far if on_far_side else 1 - far)

    first, second = _derivatives(exact, point, mp.mpf("1e-12"))
    _, gradient, hessian = _autograd(
        lambda p: log_student_t_tails(p[0], p[1], p[2], torch.tensor(value))[side],
        point,
    )
    _close(gradient, first, FIRST_RTOL)
    _close(hessian, second, SECOND_RTOL)


# von Mises -----------------------------------------------------------------


def _von_mises_mass(mu, kappa, low, high):
    """mpmath integral of the density over ``[low, high]``.

    The interval is split at the mode and antimode, and every piece is graded
    geometrically toward both of its ends so that a peak at an endpoint is
    resolved at any concentration.
    """
    with mp.workdps(30):
        mu, kappa = mp.mpf(mu), mp.mpf(kappa)
        low, high = mp.mpf(low), mp.mpf(high)
        breaks = sorted(
            {low, high}
            | {point for point in (mu, mu - mp.pi, mu + mp.pi) if low < point < high}
        )
        points = set(breaks)
        for left, right in zip(breaks[:-1], breaks[1:], strict=True):
            for j in range(1, 40):
                points.add(left + (right - left) * mp.mpf(2) ** -j)
                points.add(right - (right - left) * mp.mpf(2) ** -j)
        density = lambda t: mp.e ** (kappa * (mp.cos(t - mu) - 1))  # noqa: E731
        normalizer = 2 * mp.pi * mp.besseli(0, kappa) * mp.e ** (-kappa)
        return mp.quad(density, sorted(points)) / normalizer


def _von_mises_cdf(mu, kappa, value):
    return _von_mises_mass(mu, kappa, -mp.pi, value)


def _von_mises_sf(mu, kappa, value):
    return _von_mises_mass(mu, kappa, value, mp.pi)


@pytest.mark.parametrize("kappa", [0.01, 1.0, 20.0, 400.0])
def test_von_mises_tails_match_the_integrated_density(kappa):
    values = [-3.14, -2.5, -1.0, 0.0, 0.4, 2.9, 3.14]
    log_cdf, log_sf = log_von_mises_tails(
        torch.tensor(0.3), torch.tensor(kappa), torch.tensor(values)
    )
    cdf = [_von_mises_cdf(0.3, kappa, v) for v in values]
    sf = [_von_mises_sf(0.3, kappa, v) for v in values]
    _close(log_cdf.numpy(), [float(mp.log(p)) for p in cdf], 1e-10)
    _close(log_sf.numpy(), [float(mp.log(p)) for p in sf], 1e-10)


def test_von_mises_tails_follow_the_minus_pi_convention_outside_the_circle():
    log_cdf, log_sf = log_von_mises_tails(
        torch.tensor(0.0),
        torch.tensor(2.0),
        torch.tensor([-4.0, -math.pi, math.pi, 5.0]),
    )
    assert log_cdf.tolist() == [-math.inf, -math.inf, 0.0, 0.0]
    assert log_sf.tolist() == [0.0, 0.0, -math.inf, -math.inf]


@pytest.mark.parametrize(
    ("value", "kappa"), [(0.9, 2.0), (-2.8, 3.0), (-2.0, 60.0), (2.5, 0.5)]
)
@pytest.mark.parametrize("side", [0, 1])
def test_von_mises_tail_derivatives_in_location_and_concentration(value, kappa, side):
    point = (0.3, kappa)

    def exact(mu, concentration):
        tail = _von_mises_sf if side else _von_mises_cdf
        return mp.log(tail(mu, concentration, value))

    first, second = _derivatives(exact, point, mp.mpf("1e-10"))
    log_value, gradient, hessian = _autograd(
        lambda p: log_von_mises_tails(p[0], p[1], torch.tensor(value))[side], point
    )
    assert math.isfinite(log_value)
    _close(gradient, first, FIRST_RTOL)
    _close(hessian, second, SECOND_RTOL)


def test_von_mises_series_and_direct_tail_agree_at_the_switch():
    # Locate a point just above and just below the series floor and check the
    # two evaluations against the same oracle.
    kappa, mu = 4.0, 0.0
    values = np.linspace(-3.1, -0.5, 27)
    log_cdf, _ = log_von_mises_tails(
        torch.tensor(mu), torch.tensor(kappa), torch.tensor(values)
    )
    exact = [float(mp.log(_von_mises_cdf(mu, kappa, v))) for v in values]
    assert min(exact) < math.log(VON_MISES_SERIES_FLOOR) < max(exact)
    _close(log_cdf.numpy(), exact, 1e-10)


# Conway-Maxwell-Poisson ----------------------------------------------------


@pytest.mark.parametrize(
    ("rate", "dispersion"), [(3.0, 1.0), (10.0, 0.6), (2.0, 2.5), (200.0, 1.3)]
)
def test_com_poisson_tails_match_the_normalizing_series(rate, dispersion):
    law = COMPoisson(torch.tensor(rate), cast(float, torch.tensor(dispersion)))
    values = [-1.0, -0.5, 0.0, 1.5, 3.0, 10.0, 40.0, 200.0, math.inf]
    log_cdf, log_sf = log_com_poisson_tails(law, torch.tensor(values))
    weights = [mp.mpf(rate) ** j / mp.factorial(j) ** dispersion for j in range(3000)]
    total = mp.fsum(weights)
    expected_cdf, expected_sf = [], []
    for value in values:
        if value < 0:
            expected_cdf.append(-math.inf)
            expected_sf.append(0.0)
        elif math.isinf(value):
            expected_cdf.append(0.0)
            expected_sf.append(-math.inf)
        else:
            k = math.floor(value)
            expected_cdf.append(float(mp.log(mp.fsum(weights[: k + 1]) / total)))
            expected_sf.append(float(mp.log(mp.fsum(weights[k + 1 :]) / total)))
    # The law's own normalizer is certified to 1e-12 relative.
    _close(log_cdf.numpy(), expected_cdf, 2e-12)
    _close(log_sf.numpy(), expected_sf, 2e-12)


@pytest.mark.parametrize("value", [1.0, 6.0])
def test_com_poisson_tail_derivatives_in_rate_and_dispersion(value):
    point = (4.0, 1.4)

    def exact(rate, dispersion):
        weights = [rate**j / mp.factorial(j) ** dispersion for j in range(400)]
        k = int(value)
        return mp.log(mp.fsum(weights[k + 1 :]) / mp.fsum(weights))

    first, second = _derivatives(exact, point, mp.mpf("1e-12"))
    _, gradient, hessian = _autograd(
        lambda p: log_com_poisson_tails(
            COMPoisson(p[0], cast(float, p[1])), torch.tensor(value)
        )[1],
        point,
    )
    # The normalizer certificate bounds omitted mass at 1e-12 relative,
    # which bounds the derivative error at the same order.
    _close(gradient, first, FIRST_RTOL)
    _close(hessian, second, SECOND_RTOL)


# Gamma and exponential -----------------------------------------------------


@pytest.mark.parametrize("shape", [0.05, 0.7, 1.0, 2.0, 7.3, 60.0])
@pytest.mark.parametrize("point", [1e-3, 0.5, 1.0, 3.0, 20.0, 200.0])
def test_gamma_tail_shape_derivatives_match_mpmath(shape, point):
    def exact(a, upper):
        if upper:
            return mp.log(mp.gammainc(a, point, mp.inf, regularized=True))
        return mp.log(mp.gammainc(a, 0, point, regularized=True))

    for upper in (False, True):
        value = float(exact(mp.mpf(shape), upper))
        if not math.isfinite(value):
            continue
        first = float(mp.diff(lambda a: exact(a, upper), mp.mpf(shape)))
        second = float(mp.diff(lambda a: exact(a, upper), mp.mpf(shape), 2))
        actual, gradient, hessian = _autograd(
            lambda p: log_gamma_tails(p[0], torch.tensor(point))[int(upper)], (shape,)
        )
        _close([actual], [value], VALUE_RTOL * 10)
        _close(gradient, [first], 1e-9)
        _close(hessian, [[second]], 1e-8)


def test_gamma_and_exponential_far_tails_are_finite_with_finite_gradients():
    rate = torch.tensor([0.7, 0.7, 0.7, 0.7], dtype=torch.float64, requires_grad=True)
    shape = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    points = torch.tensor([1e-300, 1e-8, 900.0, 1500.0], dtype=torch.float64)
    gamma_cdf, gamma_sf = _log_tails(Gamma(shape, rate), points)
    exp_cdf, exp_sf = log_exponential_tails(rate, points)
    for a, x, cdf, sf in zip(
        [2.5] * 4,
        (0.7 * points).tolist(),
        gamma_cdf.tolist(),
        gamma_sf.tolist(),
        strict=True,
    ):
        _close([cdf], [float(mp.log(mp.gammainc(a, 0, x, regularized=True)))], 1e-12)
        _close(
            [sf], [float(mp.log(mp.gammainc(a, x, mp.inf, regularized=True)))], 1e-12
        )
    _close(exp_sf.detach().numpy(), (-0.7 * points).numpy(), 1e-15)
    _close(
        exp_cdf.detach().numpy(),
        [float(mp.log(-mp.expm1(-mp.mpf(0.7) * mp.mpf(x)))) for x in points.tolist()],
        1e-12,
    )
    total = gamma_cdf.sum() + gamma_sf.sum() + exp_cdf.sum() + exp_sf.sum()
    gradients = torch.autograd.grad(total, (rate, shape))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)


@pytest.mark.parametrize("family", ["gamma", "exponential"])
def test_infinite_and_zero_interval_endpoints_leave_gradients_finite(family):
    rate = torch.full((4,), 0.7, dtype=torch.float64, requires_grad=True)
    law = (
        Gamma(torch.full((4,), 2.0, dtype=torch.float64), rate)
        if family == "gamma"
        else Exponential(rate)
    )
    value = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64)
    score = event_log_prob(
        law,
        value,
        status=torch.tensor([0, 1, -1, 2]),
        interval_upper=value + 1,
        trunc_upper=torch.full((4,), 50.0, dtype=torch.float64),
    )
    (gradient,) = torch.autograd.grad(score.sum(), rate)
    assert bool(torch.isfinite(score).all()) and bool(torch.isfinite(gradient).all())
    if family == "gamma":
        log_pdf = lambda x: stats.gamma.logpdf(x, 2.0, scale=1 / 0.7)  # noqa: E731
        cdf = lambda x: stats.gamma.cdf(x, 2.0, scale=1 / 0.7)  # noqa: E731
    else:
        log_pdf = lambda x: stats.expon.logpdf(x, scale=1 / 0.7)  # noqa: E731
        cdf = lambda x: stats.expon.cdf(x, scale=1 / 0.7)  # noqa: E731
    v = value.numpy()
    expected = np.array(
        [
            log_pdf(v[0]),
            np.log(cdf(50.0) - cdf(v[1])),
            np.log(cdf(v[2])),
            np.log(cdf(v[3] + 1) - cdf(v[3])),
        ]
    ) - np.log(cdf(50.0))
    _close(score.detach().numpy(), expected, 1e-12)


# Zero atoms and mixtures ---------------------------------------------------


@pytest.mark.parametrize("component", ["gamma", "lognormal"])
def test_hurdle_tails_compose_the_atom_and_differentiate_the_gamma_shape(component):
    hu = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    first = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    second = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    law = (
        HurdleGamma(hu, first, second)
        if component == "gamma"
        else HurdleLogNormal(hu, first, second)
    )
    points = torch.tensor([-1.0, 0.0, 0.4, 2.0, 30.0], dtype=torch.float64)
    log_cdf, log_sf = law._log_tails(points)
    if component == "gamma":
        g = stats.gamma(0.8, scale=1 / 1.3).cdf(points.numpy())
    else:
        g = stats.lognorm(1.3, scale=math.exp(0.8)).cdf(points.numpy())
    g = np.where(points.numpy() > 0, g, 0.0)
    cdf = np.where(points.numpy() >= 0, 0.3 + 0.7 * g, 0.0)
    _close(np.exp(log_cdf.detach().numpy()), cdf, 1e-13)
    _close(np.exp(log_sf.detach().numpy()), 1 - cdf, 1e-12)
    left_cdf = law.log_cdf_left(torch.tensor([0.0], dtype=torch.float64))
    left_sf = law.log_survival_function_left(torch.tensor([0.0], dtype=torch.float64))
    assert left_cdf.item() == -math.inf and left_sf.item() == 0.0
    gradients = torch.autograd.grad(log_sf[2:].sum(), (hu, first, second))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    with torch.no_grad():
        assert torch.isfinite(law.cdf(points)).all()


def test_zero_inflated_beta_tails_match_the_composed_scipy_law():
    law = ZeroOneInflatedBeta(
        torch.tensor(0.4, dtype=torch.float64),
        torch.tensor(5.0, dtype=torch.float64),
        torch.tensor(0.2, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    points = torch.tensor([-0.1, 0.0, 0.3, 0.9, 1.0, 1.2], dtype=torch.float64)
    log_cdf, log_sf = law._log_tails(points)
    beta = stats.beta(2.0, 3.0).cdf(np.clip(points.numpy(), 0, 1))
    cdf = np.where(points.numpy() >= 0, 0.2 + 0.8 * beta, 0.0)
    _close(np.exp(log_cdf.numpy()), cdf, 1e-13)
    _close(np.exp(log_sf.numpy()), 1 - cdf, 1e-12)
    zero = torch.tensor([0.0], dtype=torch.float64)
    assert law.log_cdf_left(zero).item() == -math.inf
    assert law.log_survival_function_left(zero).item() == 0.0
    mass = log_interval_mass(law, 0.0, 0.5)
    _close([math.exp(mass.item())], [0.2 + 0.8 * stats.beta(2.0, 3.0).cdf(0.5)], 1e-13)


def test_mixture_truncation_normalizes_each_component_as_brms_does():
    weights = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    loc = torch.tensor([[-1.0, 2.0]], dtype=torch.float64)
    scale = torch.tensor([[0.3, 1.5]], dtype=torch.float64)
    law = MixtureNormal(weights, loc, scale)
    y = torch.tensor([0.5], dtype=torch.float64)
    score = event_log_prob(law, y, trunc_lower=-0.5, trunc_upper=1.0)
    components = [(-1.0, 0.3), (2.0, 1.5)]

    def pdf(component, x):
        return stats.norm.pdf(x, *component)

    def cdf(component, x):
        return stats.norm.cdf(x, *component)

    def sf(component, x):
        return stats.norm.sf(x, *component)

    brms = math.log(
        sum(
            w * pdf(c, 0.5) / (cdf(c, 1.0) - cdf(c, -0.5))
            for w, c in zip((0.3, 0.7), components, strict=True)
        )
    )
    whole = math.log(
        sum(w * pdf(c, 0.5) for w, c in zip((0.3, 0.7), components, strict=True))
    ) - math.log(
        sum(
            w * (cdf(c, 1.0) - cdf(c, -0.5))
            for w, c in zip((0.3, 0.7), components, strict=True)
        )
    )
    _close([score.item()], [brms], 1e-13)
    assert abs(score.item() - whole) > 0.1
    censored = event_log_prob(law, y, status=1)
    mixture = sum(w * sf(c, 0.5) for w, c in zip((0.3, 0.7), components, strict=True))
    _close([censored.item()], [math.log(mixture)], 1e-13)
    torch.manual_seed(4)
    draws = law.componentwise_truncation(-0.5, 1.0).sample(torch.Size([2000]))
    assert bool(((draws >= -0.5) & (draws <= 1.0)).all())


@pytest.mark.parametrize(
    "law",
    [
        StudentT(torch.tensor(3.0), torch.tensor(0.0), torch.tensor(1.0)),
        VonMises(torch.tensor(0.0), torch.tensor(1.0)),
    ],
)
def test_torch_laws_without_cdfs_gain_finite_truncated_scores(law):
    score = event_log_prob(law, torch.tensor(0.4), trunc_lower=-1.0, trunc_upper=2.0)
    assert math.isfinite(score.item())


# Truncated sampling and sampler rejection ----------------------------------


def test_bisection_quantiles_match_scipy_for_laws_without_icdf():
    targets = torch.tensor([1e-12, 0.01, 0.4, 0.9, 1 - 1e-9])
    quantiles = invert_log_cdf(
        lambda point: log_student_t_tails(
            torch.tensor(2.5), torch.tensor(0.3), torch.tensor(1.4), point
        )[0],
        targets.log(),
        torch.full_like(targets, -math.inf),
        torch.full_like(targets, math.inf),
    )
    _close(quantiles.numpy(), stats.t.ppf(targets.numpy(), 2.5, 0.3, 1.4), 1e-9)


def test_truncated_laws_without_icdf_sample_inside_low_mass_intervals():
    # The interval holds about 1e-6 of the mass, beyond rejection's budget.
    torch.manual_seed(3)
    law = Normalize(Restrict(Beta(torch.tensor(1e-4), torch.tensor(1e-4)), 0.3, 0.7))
    draws = law.sample(torch.Size([4000]))
    assert bool(((draws >= 0.3) & (draws <= 0.7)).all())
    # A Kolmogorov-Smirnov check against the SciPy conditional CDF.
    beta = stats.beta(1e-4, 1e-4)
    lower, upper = beta.cdf(0.3), beta.cdf(0.7)
    conditional = lambda x: (beta.cdf(x) - lower) / (upper - lower)  # noqa: E731
    assert stats.kstest(draws.numpy(), conditional).pvalue > 1e-3


def test_zero_truncation_mass_is_a_rejected_sampler_position():
    with pytest.raises(ValueError, match="positive finite mass") as caught:
        event_log_prob(
            Exponential(torch.tensor([2.0])),
            torch.tensor([0.5]),
            trunc_lower=-3.0,
            trunc_upper=-1.0,
        )
    assert _rejects_position(caught.value)


@pytest.mark.parametrize("kappa", [2e4, 1e8])
def test_von_mises_tails_at_large_concentration_use_the_direct_integral(kappa):
    scale = 1 / math.sqrt(kappa)
    values = [-3.0, 0.3 - scale, 0.3, 0.3 + scale, 2.0]
    kappa_tensor = torch.tensor(kappa, requires_grad=True)
    log_cdf, log_sf = log_von_mises_tails(
        torch.tensor(0.3), kappa_tensor, torch.tensor(values)
    )
    # Within one standard deviation of the mode the normal limit is exact to
    # O(1/kappa) relative; the far
    # points have masses far below float64 range in the normal limit and are
    # checked against the integrated density.
    near = [1, 2, 3]
    normal = stats.norm.logcdf((np.array(values)[near] - 0.3) / scale)
    _close(log_cdf.detach().numpy()[near], normal, 10 / kappa)
    for index in (0, 4):
        exact_cdf = float(mp.log(_von_mises_cdf(0.3, kappa, values[index])))
        exact_sf = float(mp.log(_von_mises_sf(0.3, kappa, values[index])))
        _close([log_cdf[index].item()], [exact_cdf], 1e-10)
        _close([log_sf[index].item()], [exact_sf], 1e-10)
    (gradient,) = torch.autograd.grad((log_cdf + log_sf).sum(), kappa_tensor)
    assert math.isfinite(gradient.item())


@pytest.mark.parametrize("success", [0.999, 1 - 2.0**-20, 1 - 2.0**-40])
@pytest.mark.parametrize("shape", [7e-5, 0.05])
def test_small_shape_negative_binomial_tails_near_certain_success(success, shape):
    # These coordinates exhausted the direct continued fraction; the tails
    # follow I_(1-p)(r, k + 1) as in Torch's convention for successes.
    log_cdf, log_sf = negative_binomial_log_tails(
        torch.tensor(shape), torch.tensor([2.0, 3.0, 12.0]), torch.tensor(success)
    )
    p = mp.mpf(success)
    for index, k in enumerate((2, 3, 12)):
        cdf = mp.betainc(shape, k + 1, 0, 1 - p, regularized=True)
        sf = mp.betainc(k + 1, shape, 0, p, regularized=True)
        _close([log_cdf[index].item()], [float(mp.log(cdf))], 1e-11)
        _close([log_sf[index].item()], [float(mp.log(sf))], 1e-11)


def test_divergent_parameters_are_rejected_sampler_positions():
    with pytest.raises(ValueError, match="NaN or positive infinity") as caught:
        event_log_prob(
            Exponential(torch.tensor([math.inf])), torch.tensor([0.5]), status=1
        )
    assert _rejects_position(caught.value)
    with pytest.raises(NotImplementedError, match="capacity") as capacity:
        log_com_poisson_tails(
            COMPoisson(torch.tensor(5.0), cast(float, torch.tensor(1e-14))),
            torch.tensor(1.0),
        )
    assert _rejects_position(capacity.value)


def test_truncated_count_laws_sample_the_conditional_mass_by_inversion():
    torch.manual_seed(5)
    law = Normalize(Restrict(torch.distributions.Poisson(torch.tensor(3.0)), 2.0, 6.0))
    draws = law.sample(torch.Size([20000])).long().numpy()
    assert draws.min() >= 2 and draws.max() <= 6
    expected = stats.poisson.pmf(np.arange(2, 7), 3.0)
    observed = np.bincount(draws, minlength=7)[2:7]
    result = stats.chisquare(observed, expected / expected.sum() * draws.size)
    assert result.pvalue > 1e-3
    # The interval holds about exp(-1e17) of the base mass, which underflows
    # in probability coordinates; the draw is made in log CDF coordinates.
    huge = Normalize(
        Restrict(torch.distributions.Poisson(torch.tensor(1e17)), 1.0, 12.0)
    )
    draws = huge.sample(torch.Size([50]))
    assert bool(((draws >= 1) & (draws <= 12)).all())


@pytest.mark.parametrize("shape", [2e3, 4e5, 1e9])
@pytest.mark.parametrize("success", [1e-6, 1e-4, 3e-3])
def test_large_shape_negative_binomial_tails_sum_the_count_masses(shape, success):
    # Large shapes previously exhausted the continued-fraction budget.
    log_cdf, log_sf = negative_binomial_log_tails(
        torch.tensor(shape), torch.tensor([2.0, 12.0, 60.0]), torch.tensor(success)
    )
    r, p = mp.mpf(shape), mp.mpf(success)

    def mass(j):
        return mp.binomial(r + j - 1, j) * (1 - p) ** r * p**j

    for index, k in enumerate((2, 12, 60)):
        cdf = mp.fsum(mass(j) for j in range(k + 1))
        _close([log_cdf[index].item()], [float(mp.log(cdf))], 1e-10)
        if cdf < mp.mpf("0.999999"):
            _close([log_sf[index].item()], [float(mp.log(1 - cdf))], 1e-8)
