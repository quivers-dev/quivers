"""Deterministic means of truncated scalar laws against independent references.

Three references are used. brms's own ``posterior_epred_trunc_<family>``
functions, run by ``tests/formulas/oracles/brms_truncated_epred.R`` and
pinned here, cover the families brms supports. SciPy adaptive quadrature of
``y f(y)`` and ``f(y)`` over the interval covers every continuous law, and a
NumPy sum of ``y p(y)`` covers every count law. Far-tail cases use 50-digit
mpmath closed forms.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest
import torch
from scipy import integrate, special, stats
from torch import distributions as D

from quivers.continuous import _truncated_mean as module
from quivers.continuous._closed_probability import (
    ClosedBinomial,
    ClosedGeometric,
    ClosedNegativeBinomial,
    ClosedPoisson,
)
from quivers.continuous._asymmetric_laplace import AsymmetricLaplace
from quivers.continuous._positive_hurdle import HurdleGamma, HurdleLogNormal
from quivers.continuous._response_time import ExGaussian, InverseGaussian
from quivers.continuous._shape_families import (
    Frechet,
    GeneralizedExtremeValue,
    ShiftedLogNormal,
)
from quivers.continuous._skew_normal import SkewNormal
from quivers.continuous._zero_inflated_count import BetaBinomial
from quivers.continuous._zip_hurdle import (
    HurdlePoisson,
    ZeroInflatedPoisson,
    ZeroOneInflatedBeta,
)
from quivers.continuous.measure import Mixture, Normalize, Restrict


def _t(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64)


def _mean(law, low, high, **flags) -> float:
    return float(Normalize(Restrict(law, low, high, **flags)).mean)


WEIBULL_SCALE = 1.4 / math.gamma(1 + 1 / 1.7)
BRMS_LAWS = {
    "gaussian": lambda: D.Normal(_t(0.3), _t(1.2)),
    "student": lambda: D.StudentT(_t(3.0), _t(0.3), _t(1.2)),
    "lognormal": lambda: D.LogNormal(_t(0.2), _t(0.7)),
    # brms predicts the mean: rate = shape / mu, scale = mu / Gamma(1 + 1/shape).
    "gamma": lambda: D.Gamma(_t(2.5), _t(2.5 / 1.4)),
    "exponential": lambda: D.Exponential(_t(1 / 1.4)),
    "weibull": lambda: D.Weibull(_t(WEIBULL_SCALE), _t(1.7)),
}
BRMS_CONTINUOUS = {
    (0.5, 2.0): {
        "gaussian": 1.1343213839560995,
        "student": 1.1269765775785738,
        "lognormal": 1.1321563074966021,
        "gamma": 1.1670220537036649,
        "exponential": 1.1185657541369076,
        "weibull": 1.1964129669634773,
    },
    (0.5, None): {
        "gaussian": 1.3883127343174126,
        "student": 1.79282791853893,
        "lognormal": 1.6939887840306773,
        "gamma": 1.5480077164995565,
        "exponential": 1.8999999999999999,
        "weibull": 1.5677459465664527,
    },
    (None, 0.8): {
        "gaussian": -0.36349363981773614,
        "student": -0.66582747823949728,
        "lognormal": 0.54941891876200299,
        "gamma": 0.51419243187556474,
        "exponential": 0.36211048415330416,
        "weibull": 0.48541101273256076,
    },
    (3.0, None): {
        "gaussian": 3.4156830464243599,
        "student": 4.7784250532929491,
        "lognormal": 4.3814383524939107,
        "gamma": 3.7003020736989938,
        "exponential": 4.400000000000003,
        "weibull": 3.528540664305055,
    },
}
"""Output of ``brms:::posterior_epred_trunc_<family>`` (brms 2.20.4)."""

# brms forms each mean from a difference of two incomplete gamma or normal
# distribution values, which keeps about twelve digits on these intervals.
BRMS_RTOL = 1e-11


@pytest.mark.parametrize("bounds", sorted(BRMS_CONTINUOUS, key=str))
@pytest.mark.parametrize("family", sorted(BRMS_LAWS))
def test_continuous_truncated_means_match_brms(family: str, bounds) -> None:
    low, high = bounds
    observed = _mean(BRMS_LAWS[family](), low, high)
    assert observed == pytest.approx(BRMS_CONTINUOUS[bounds][family], rel=BRMS_RTOL)


BRMS_DISCRETE = {
    "poisson": (lambda: ClosedPoisson(_t(3.2)), 3.4842207859799204),
    "negbinomial": (
        lambda: ClosedNegativeBinomial(_t(2.5), logits=_t(math.log(3.2 / 2.5))),
        3.4842143018816163,
    ),
    "geometric": (lambda: ClosedGeometric(probs=_t(1 / 4.2)), 3.4728259526770495),
    "binomial": (lambda: ClosedBinomial(_t(9.0), probs=_t(0.35)), 3.4225705338268693),
}
"""``brms:::posterior_epred_trunc_discrete`` over ``1 < y <= 6``."""


@pytest.mark.parametrize("family", sorted(BRMS_DISCRETE))
def test_count_truncated_means_match_brms(family: str) -> None:
    build, expected = BRMS_DISCRETE[family]
    # brms's bounds are open below and closed above.
    assert _mean(build(), 1.0, 6.0, lower_closed=False) == pytest.approx(
        expected, rel=BRMS_RTOL
    )
    assert _mean(build(), 2.0, 6.0) == pytest.approx(expected, rel=BRMS_RTOL)


CONTINUOUS = {
    "normal": (lambda: D.Normal(_t(0.3), _t(1.2)), stats.norm(0.3, 1.2)),
    "student": (lambda: D.StudentT(_t(3.0), _t(0.3), _t(1.2)), stats.t(3.0, 0.3, 1.2)),
    "lognormal": (
        lambda: D.LogNormal(_t(0.2), _t(0.7)),
        stats.lognorm(0.7, scale=math.exp(0.2)),
    ),
    "shifted_lognormal": (
        lambda: ShiftedLogNormal(_t(0.2), _t(0.7), _t(0.25)),
        stats.lognorm(0.7, loc=0.25, scale=math.exp(0.2)),
    ),
    "gamma": (lambda: D.Gamma(_t(2.5), _t(1.5)), stats.gamma(2.5, scale=1 / 1.5)),
    "exponential": (lambda: D.Exponential(_t(1.5)), stats.expon(scale=1 / 1.5)),
    "weibull": (lambda: D.Weibull(_t(1.3), _t(1.7)), stats.weibull_min(1.7, scale=1.3)),
    "frechet": (lambda: Frechet(_t(1.3), _t(2.7)), stats.invweibull(2.7, scale=1.3)),
    "inverse_gaussian": (
        lambda: InverseGaussian(_t(1.4), _t(2.0)),
        stats.invgauss(1.4 / 2.0, scale=2.0),
    ),
    "beta": (lambda: D.Beta(_t(2.0), _t(3.0)), stats.beta(2.0, 3.0)),
    "skew_normal": (
        lambda: SkewNormal(_t(0.3), _t(1.2), _t(2.0)),
        stats.skewnorm(2.0, 0.3, 1.2),
    ),
    "exgaussian": (
        lambda: ExGaussian(_t(0.3), _t(1.2), _t(1.5)),
        stats.exponnorm(1 / (1.2 * 1.5), 0.3, 1.2),
    ),
    "gen_extreme_value": (
        lambda: GeneralizedExtremeValue(_t(0.3), _t(1.2), _t(0.2)),
        stats.genextreme(-0.2, 0.3, 1.2),
    ),
    "gen_extreme_value_bounded_above": (
        lambda: GeneralizedExtremeValue(_t(0.3), _t(1.2), _t(-0.2)),
        stats.genextreme(0.2, 0.3, 1.2),
    ),
}
INTERVALS = [(0.5, 2.0), (0.5, None), (None, 0.8), (0.1, 0.9), (3.0, None)]
# The quadrature rule is certified to 1e-8 of the interval mass and SciPy's
# adaptive rule is run to 1e-12; 1e-8 relative covers both.
QUAD_RTOL = 1e-8


def _scipy_mean(reference, low, high, kink: float | None = None) -> float:
    lower, upper = reference.support()
    a = max(-math.inf if low is None else low, lower)
    b = min(math.inf if high is None else high, upper)
    points = [kink] if kink is not None and a < kink < b else None
    options = {"epsabs": 0.0, "epsrel": 1e-12, "limit": 400}
    if points is not None and math.isfinite(a) and math.isfinite(b):
        options["points"] = points
    moment = integrate.quad(lambda y: y * reference.pdf(y), a, b, **options)[0]
    mass = integrate.quad(reference.pdf, a, b, **options)[0]
    return moment / mass


@pytest.mark.parametrize("interval", INTERVALS, ids=str)
@pytest.mark.parametrize("family", sorted(CONTINUOUS))
def test_continuous_truncated_means_match_scipy_quadrature(family, interval) -> None:
    low, high = interval
    build, reference = CONTINUOUS[family]
    if family == "beta" and (low is None or low >= 1 or high is None):
        pytest.skip("the interval does not restrict the unit interval")
    expected = _scipy_mean(reference, low, high)
    assert _mean(build(), low, high) == pytest.approx(expected, rel=QUAD_RTOL)


@pytest.mark.parametrize("interval", [(-1.0, 2.0), (0.5, None), (None, 0.1)], ids=str)
@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_asymmetric_laplace_truncated_mean_matches_its_piecewise_form(
    quantile: float, interval
) -> None:
    low, high = interval
    loc, scale = 0.3, 0.9
    law = AsymmetricLaplace(_t(loc), _t(scale), _t(quantile))

    def density(y: float) -> float:
        u = (y - loc) / scale
        check = u * (quantile - (u < 0))
        return quantile * (1 - quantile) / scale * math.exp(-check)

    a = -math.inf if low is None else low
    b = math.inf if high is None else high
    pieces = [a, loc, b] if a < loc < b else [a, b]
    moment = mass = 0.0
    for left, right in zip(pieces[:-1], pieces[1:], strict=True):
        options = {"epsabs": 0.0, "epsrel": 1e-12, "limit": 400}
        moment += integrate.quad(lambda y: y * density(y), left, right, **options)[0]
        mass += integrate.quad(density, left, right, **options)[0]
    assert _mean(law, low, high) == pytest.approx(moment / mass, rel=QUAD_RTOL)


def test_far_tail_means_keep_their_digits() -> None:
    mpmath.mp.dps = 50
    # Normal on [30, inf): phi(30) / Phi(-30).
    expected = float(mpmath.npdf(30) / (mpmath.erfc(30 / mpmath.sqrt(2)) / 2))
    assert _mean(D.Normal(_t(0.0), _t(1.0)), 30.0, None) == pytest.approx(
        expected, rel=1e-13
    )
    # Gamma(2.5, rate 1.5) on [60, inf): Gamma(3.5, 90) / (1.5 Gamma(2.5, 90)).
    expected = float(
        mpmath.gammainc(3.5, 90) / (mpmath.mpf("1.5") * mpmath.gammainc(2.5, 90))
    )
    assert _mean(D.Gamma(_t(2.5), _t(1.5)), 60.0, None) == pytest.approx(
        expected, rel=1e-13
    )
    # Lognormal(0, 1) on (0, 1e-6]: exp(1/2) Phi(log b - 1) / Phi(log b).
    log_b = mpmath.log(mpmath.mpf("1e-6"))
    expected = float(mpmath.exp(0.5) * mpmath.ncdf(log_b - 1) / mpmath.ncdf(log_b))
    assert _mean(D.LogNormal(_t(0.0), _t(1.0)), None, 1e-6) == pytest.approx(
        expected, rel=1e-13
    )
    # Poisson(3) on [40, inf): 3 P(Y >= 39) / P(Y >= 40).
    upper_39 = mpmath.gammainc(39, 0, 3, regularized=True)
    upper_40 = mpmath.gammainc(40, 0, 3, regularized=True)
    expected = float(3 * upper_39 / upper_40)
    assert _mean(ClosedPoisson(_t(3.0)), 40.0, None) == pytest.approx(
        expected, rel=1e-12
    )


COUNTS = {
    "poisson": lambda: ClosedPoisson(_t(3.2)),
    "negative_binomial": lambda: ClosedNegativeBinomial(_t(2.5), logits=_t(0.3)),
    "geometric": lambda: ClosedGeometric(probs=_t(0.3)),
    "binomial": lambda: ClosedBinomial(_t(9.0), probs=_t(0.35)),
    "beta_binomial": lambda: BetaBinomial(_t(9.0), _t(2.0), _t(3.0)),
    "zero_inflated_poisson": lambda: ZeroInflatedPoisson(_t(0.2), _t(3.2)),
    "hurdle_poisson": lambda: HurdlePoisson(_t(0.2), _t(3.2)),
}
COUNT_INTERVALS = [(1.0, 5.0), (2.0, None), (None, 4.0), (0.0, 3.0), (2.5, 6.5)]


@pytest.mark.parametrize("interval", COUNT_INTERVALS, ids=str)
@pytest.mark.parametrize("family", sorted(COUNTS))
def test_count_truncated_means_match_a_direct_sum(family: str, interval) -> None:
    low, high = interval
    law = COUNTS[family]()
    cap = 9 if family in {"binomial", "beta_binomial"} else 400
    first = 0 if low is None else math.ceil(low)
    last = cap if high is None else min(math.floor(high), cap)
    grid = np.arange(first, last + 1, dtype=float)
    probability = np.exp(law.log_prob(torch.tensor(grid)).numpy())
    expected = (grid * probability).sum() / probability.sum()
    assert _mean(law, low, high) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize("interval", [(0.0, 2.0), (0.5, None), (None, 1.5)], ids=str)
def test_hurdle_truncated_means_weigh_the_atom_at_zero(family: str, interval) -> None:
    low, high = interval
    zero = 0.3
    if family == "gamma":
        law = HurdleGamma(_t(zero), _t(2.5), _t(1.5))
        component = stats.gamma(2.5, scale=1 / 1.5)
    else:
        law = HurdleLogNormal(_t(zero), _t(0.2), _t(0.7))
        component = stats.lognorm(0.7, scale=math.exp(0.2))
    a = max(0.0 if low is None else low, 0.0)
    b = math.inf if high is None else high
    options = {"epsabs": 0.0, "epsrel": 1e-12, "limit": 400}
    moment = integrate.quad(lambda y: y * component.pdf(y), a, b, **options)[0]
    mass = integrate.quad(component.pdf, a, b, **options)[0]
    holds_zero = low is None or low <= 0
    expected = (1 - zero) * moment / (zero * holds_zero + (1 - zero) * mass)
    assert _mean(law, low, high) == pytest.approx(expected, rel=QUAD_RTOL)


@pytest.mark.parametrize(
    ("low", "high", "upper_closed"),
    [(0.0, 0.9, True), (0.2, 1.0, True), (0.2, 1.0, False), (0.0, 1.0, True)],
)
def test_zero_one_inflated_beta_truncated_mean_counts_both_atoms(
    low: float, high: float, upper_closed: bool
) -> None:
    mu, phi, zoi, coi = 0.4, 5.0, 0.3, 0.6
    law = ZeroOneInflatedBeta(_t(mu), _t(phi), _t(zoi), _t(coi))
    beta = stats.beta(mu * phi, (1 - mu) * phi)
    options = {"epsabs": 0.0, "epsrel": 1e-12, "limit": 400}
    moment = integrate.quad(lambda y: y * beta.pdf(y), low, high, **options)[0]
    mass = integrate.quad(beta.pdf, low, high, **options)[0]
    at_zero = zoi * (1 - coi) * (low <= 0)
    at_one = zoi * coi * (high >= 1 and upper_closed)
    expected = ((1 - zoi) * moment + at_one) / ((1 - zoi) * mass + at_zero + at_one)
    observed = _mean(law, low, high, upper_closed=upper_closed)
    assert observed == pytest.approx(expected, rel=QUAD_RTOL)


def test_a_mixture_of_truncated_normals_averages_the_component_means() -> None:
    components = [
        Normalize(Restrict(D.Normal(_t(-1.0), _t(0.8)), -2.0, 1.0)),
        Normalize(Restrict(D.Normal(_t(1.5), _t(1.1)), -2.0, 1.0)),
    ]
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
    mixture = Mixture(weights, components)
    means = [
        _scipy_mean(stats.norm(-1.0, 0.8), -2.0, 1.0),
        _scipy_mean(stats.norm(1.5, 1.1), -2.0, 1.0),
    ]
    expected = 0.3 * means[0] + 0.7 * means[1]
    assert float(mixture.mean) == pytest.approx(expected, rel=QUAD_RTOL)


def test_heavy_tails_are_infinite_or_refused_and_bounded_ranges_integrate() -> None:
    heavy = D.StudentT(_t(0.8), _t(0.0), _t(1.0))
    assert _mean(heavy, 1.0, None) == math.inf
    assert _mean(heavy, None, -1.0) == -math.inf
    expected = _scipy_mean(stats.t(0.8), 0.5, 4.0)
    assert _mean(heavy, 0.5, 4.0) == pytest.approx(expected, rel=QUAD_RTOL)
    frechet = Frechet(_t(1.0), _t(0.9))
    with pytest.raises(NotImplementedError, match="no finite mean"):
        _mean(frechet, 1.0, None)
    expected = _scipy_mean(stats.invweibull(0.9), 0.5, 4.0)
    assert _mean(frechet, 0.5, 4.0) == pytest.approx(expected, rel=QUAD_RTOL)


def test_quadrature_refuses_a_law_with_an_atom_away_from_zero() -> None:
    law = ZeroOneInflatedBeta(_t(0.4), _t(5.0), _t(0.3), _t(0.6))
    low, high = _t(0.2), _t(1.0)
    log_mass = module._log_mass(law, low, high)
    with pytest.raises(NotImplementedError, match="missed the interval mass by 0.2"):
        module._quadrature_mean(law, low, high, True, True, log_mass)


def test_count_ranges_beyond_the_deterministic_rules_are_refused(monkeypatch) -> None:
    law = ZeroInflatedPoisson(_t(0.2), _t(3.2))
    with pytest.raises(NotImplementedError, match="holds less than 1e-06"):
        _mean(law, 40.0, None)
    monkeypatch.setattr(module, "MAX_SUMMED_COUNTS", 8)
    with pytest.raises(NotImplementedError, match="would sum 11 integers"):
        _mean(law, 0.0, 10.0)
    assert math.isfinite(_mean(law, 0.0, 7.0))


def test_only_a_scalar_law_restricted_to_an_interval_has_this_mean() -> None:
    joint = D.Independent(D.Normal(torch.zeros(2), torch.ones(2)), 1)
    with pytest.raises(NotImplementedError, match="scalar law"):
        module.truncated_mean(joint, _t(0.0), None)
    with pytest.raises(NotImplementedError, match="restricted to an interval"):
        _ = Normalize(D.Normal(_t(0.0), _t(1.0))).mean


def test_batched_bounds_and_parameters_broadcast() -> None:
    loc = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    low = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    law = D.Normal(loc, torch.ones_like(loc))
    observed = Normalize(Restrict(law, low, None)).mean
    assert observed.shape == (2, 3)
    for i, mean in enumerate((0.0, 1.0)):
        for j, bound in enumerate((-1.0, 0.5, 2.0)):
            z = bound - mean
            expected = mean + stats.norm.pdf(z) / stats.norm.sf(z)
            assert float(observed[i, j]) == pytest.approx(expected, rel=1e-13)
    assert special.ndtr(0.0) == 0.5
