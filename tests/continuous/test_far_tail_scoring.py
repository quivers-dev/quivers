"""Log tails far from the bulk of a law, against independent references.

Transformed laws are checked against SciPy's direct ``logcdf`` and ``logsf``
(closed forms for the Weibull and Frechet laws, the normal log tails of the
log response for the lognormal law). The skew-normal upper tail is checked
against 50-digit mpmath quadrature of its density. Hurdle interval masses are
checked against the atom plus the SciPy component mass.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest
import torch
from scipy import stats
from torch.distributions import (
    AffineTransform,
    Exponential,
    LogNormal,
    Normal,
    PowerTransform,
    SigmoidTransform,
    TransformedDistribution,
    Weibull,
)

from quivers.continuous._interval import (
    _log_tails,
    _monotone_pullback,
    log_interval_mass,
)
from quivers.continuous._positive_hurdle import HurdleGamma, HurdleLogNormal
from quivers.continuous._skew_normal import SkewNormal

# The pulled-back point costs a division, a logarithm and a power in float64,
# and the references are accurate to a few ulp; 1e-12 relative leaves three
# orders of margin.
RTOL = 1e-12
POSITIVE = np.array([1e-300, 1e-30, 1e-12, 1e-4, 0.5, 5.0, 20.0, 200.0, 1e4, 1e8])


def _t(value: float | np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _assert_tails(law, reference, points: np.ndarray) -> None:
    log_cdf, log_sf = _log_tails(law, _t(points))
    expected_cdf, expected_sf = reference.logcdf(points), reference.logsf(points)
    # Every reference value is finite or an exact zero mass on this grid.
    np.testing.assert_allclose(log_cdf.numpy(), expected_cdf, rtol=RTOL, atol=1e-300)
    np.testing.assert_allclose(log_sf.numpy(), expected_sf, rtol=RTOL, atol=1e-300)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [0.5, 1.5, 4.0])
@pytest.mark.parametrize("scale", [0.25, 1.0, 64.0])
def test_weibull_log_tails_match_scipy(dtype, shape: float, scale: float) -> None:
    law = Weibull(torch.tensor(scale, dtype=dtype), torch.tensor(shape, dtype=dtype))
    points = POSITIVE[
        np.isfinite((POSITIVE / scale) ** shape) & ((POSITIVE / scale) ** shape < 1e300)
    ]
    reference = stats.weibull_min(shape, scale=scale)
    with np.errstate(divide="ignore"):
        assert np.isneginf(np.log1p(-reference.cdf(points))).any()
    _assert_tails(law, reference, points)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sigma", [0.25, 1.0, 2.0])
def test_lognormal_log_tails_match_the_normal_tails_of_the_log(
    dtype, sigma: float
) -> None:
    loc = 0.5
    law = LogNormal(torch.tensor(loc, dtype=dtype), torch.tensor(sigma, dtype=dtype))
    standardized = (np.log(POSITIVE) - loc) / sigma
    log_cdf, log_sf = _log_tails(law, _t(POSITIVE))
    np.testing.assert_allclose(
        log_cdf.numpy(), stats.norm.logcdf(standardized), rtol=RTOL, atol=1e-300
    )
    np.testing.assert_allclose(
        log_sf.numpy(), stats.norm.logsf(standardized), rtol=RTOL, atol=1e-300
    )
    assert bool(torch.isfinite(log_cdf).all()) and bool(torch.isfinite(log_sf).all())


def test_a_decreasing_power_exchanges_the_tails() -> None:
    # X ~ Exponential(1) and Y = scale * X**(-1 / shape) is Frechet.
    shape, scale = 2.0, 3.0
    law = TransformedDistribution(
        Exponential(_t(1.0)),
        [PowerTransform(_t(-1.0 / shape)), AffineTransform(_t(0.0), _t(scale))],
    )
    points = np.array([1e-3, 0.5, 5.0, 200.0, 1e8])
    log_cdf, log_sf = _log_tails(law, _t(points))
    expected_cdf = -((points / scale) ** -shape)
    np.testing.assert_allclose(log_cdf.numpy(), expected_cdf, rtol=RTOL)
    expected_sf = np.where(
        expected_cdf > -math.log(2.0),
        np.log(-np.expm1(expected_cdf)),
        np.log1p(-np.exp(expected_cdf)),
    )
    np.testing.assert_allclose(log_sf.numpy(), expected_sf, rtol=RTOL)
    # SciPy agrees wherever its own Frechet tails are finite.
    reference = stats.invweibull(shape, scale=scale)
    finite = np.isfinite(reference.logcdf(points))
    assert finite.sum() == 4
    np.testing.assert_allclose(
        log_cdf.numpy()[finite], reference.logcdf(points)[finite], rtol=RTOL
    )


def test_a_negative_affine_scale_exchanges_the_tails() -> None:
    # Y = 1 - 2 X for a lognormal X: P(Y <= y) = P(X >= (1 - y) / 2).
    law = TransformedDistribution(
        LogNormal(_t(0.0), _t(1.0)), [AffineTransform(_t(1.0), _t(-2.0))]
    )
    points = np.array([-1e8, -400.0, -3.0, 0.5, 1.0 - 2.0**-40])
    pulled = np.log((1.0 - points) / 2.0)
    assert np.isfinite(pulled).all()
    log_cdf, log_sf = _log_tails(law, _t(points))
    np.testing.assert_allclose(log_cdf.numpy(), stats.norm.logsf(pulled), rtol=RTOL)
    np.testing.assert_allclose(log_sf.numpy(), stats.norm.logcdf(pulled), rtol=RTOL)
    # Points at or above the image's supremum hold the whole mass below them.
    edge_cdf, edge_sf = _log_tails(law, _t([1.0, 2.0, math.inf, -math.inf]))
    assert edge_cdf.tolist() == [0.0, 0.0, 0.0, -math.inf]
    assert edge_sf.tolist() == [-math.inf, -math.inf, -math.inf, 0.0]


def test_points_outside_the_image_carry_exact_masses_and_finite_gradients() -> None:
    loc = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    points = _t([-math.inf, -1.0, 0.0, 2.0, math.inf])
    log_cdf, log_sf = _log_tails(LogNormal(loc, sigma), points)
    assert log_cdf[[0, 1, 2, 4]].tolist() == [-math.inf, -math.inf, -math.inf, 0.0]
    assert log_sf[[0, 1, 2, 4]].tolist() == [0.0, 0.0, 0.0, -math.inf]
    gradients = torch.autograd.grad(
        log_cdf[3] + log_sf[3], (loc, sigma), retain_graph=True
    )
    z = (math.log(2.0) - 0.3) / 0.7
    hazard_low = math.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
    hazard_high = math.exp(stats.norm.logpdf(z) - stats.norm.logsf(z))
    expected_loc = (-hazard_low + hazard_high) / 0.7
    assert float(gradients[0]) == pytest.approx(expected_loc, rel=1e-10)
    assert float(gradients[1]) == pytest.approx(expected_loc * z, rel=1e-10)
    masked = (
        torch.where(torch.isfinite(log_cdf), log_cdf, 0.0).sum()
        + torch.where(torch.isfinite(log_sf), log_sf, 0.0).sum()
    )
    for gradient in torch.autograd.grad(masked, (loc, sigma)):
        assert bool(torch.isfinite(gradient))


def test_a_map_outside_the_monotone_family_is_left_to_the_caller() -> None:
    assert _monotone_pullback([SigmoidTransform()], _t([0.5])) is None
    mixed = AffineTransform(_t([0.0, 0.0]), _t([1.0, -1.0]))
    assert _monotone_pullback([mixed], _t([0.5, 0.5])) is None
    mixed_power = PowerTransform(_t([1.0, -1.0]))
    assert _monotone_pullback([mixed_power], _t([0.5, 0.5])) is None
    # The generic CDF path still scores such a law.
    law = TransformedDistribution(Normal(_t(0.0), _t(1.0)), [SigmoidTransform()])
    log_cdf, log_sf = _log_tails(law, _t([0.5]))
    assert float(log_cdf) == pytest.approx(math.log(0.5))
    assert float(log_sf) == pytest.approx(math.log(0.5))


def _skew_normal_log_tail(z: float, alpha: float) -> float:
    """Reference ``log P(Z > z)`` for the density ``2 phi(t) Phi(alpha t)``.

    With ``t = z + u`` the tail is ``2 phi(z) Phi(alpha z)`` times the
    integral over ``u > 0`` of ``exp(-z u - u**2 / 2) R(u)``, where
    ``R(u) = erfc(-alpha (z + u) / sqrt 2) / erfc(-alpha z / sqrt 2)``. The
    integrand is of order one, so the quadrature's absolute tolerance is a
    relative one, and every factor is positive.
    """
    mpmath.mp.dps = 50
    root = mpmath.sqrt(2)
    front = mpmath.erfc(-alpha * z / root)

    def integrand(u):
        ratio = mpmath.erfc(-alpha * (z + u) / root) / front
        return mpmath.exp(-z * u - u * u / 2) * ratio

    integral, error = mpmath.quad(integrand, [0, 1, mpmath.inf], error=True)
    assert error < mpmath.mpf(10) ** -30 * integral
    return float(
        mpmath.log(2 * mpmath.npdf(z)) + mpmath.log(front / 2) + mpmath.log(integral)
    )


@pytest.mark.parametrize("alpha", [-3.0, -0.5, 0.5, 2.0])
@pytest.mark.parametrize("z", [3.0, 12.0, 30.0])
def test_skew_normal_upper_tail_matches_mpmath_quadrature(
    alpha: float, z: float
) -> None:
    law = SkewNormal(_t(0.0), _t(1.0), _t(alpha))
    expected = _skew_normal_log_tail(z, alpha)
    # The series and quadrature rules of the law are accurate to about 1e-12.
    assert float(law.log_survival_function(_t(z))) == pytest.approx(expected, rel=1e-10)
    # The lower tail of the mirrored law is the same number.
    mirrored = SkewNormal(_t(0.0), _t(1.0), _t(-alpha))
    assert float(mirrored.log_cdf(_t(-z))) == pytest.approx(expected, rel=1e-10)


def test_skew_normal_upper_tail_stays_finite_beyond_float64_underflow() -> None:
    law = SkewNormal(_t(0.0), _t(1.0), _t(2.0))
    points = _t([40.0, 200.0, 1e4])
    observed = law.log_survival_function(points)
    # For positive skewness the upper tail is 2 Phi(-z) (1 + o(1)).
    expected = math.log(2.0) + stats.norm.logsf(points.numpy())
    with np.errstate(divide="ignore"):
        naive = np.log1p(-stats.skewnorm(2.0).cdf(points.numpy()))
    assert np.isneginf(naive).all()
    np.testing.assert_allclose(observed.numpy(), expected, rtol=1e-12)


INTERVALS = [
    (None, None, True, True),
    (-1.0, 2.0, True, True),
    (0.0, 2.0, True, True),
    (0.0, 2.0, False, True),
    (0.0, None, False, True),
    (1.0, None, True, True),
    (None, 0.0, True, True),
    (None, 0.0, True, False),
    (3.0, 2.0, True, True),
    (1e-4, 1.6e-4, False, True),
    (2e-9, 3e-9, True, True),
]


@pytest.mark.parametrize("family", ["lognormal", "gamma"])
@pytest.mark.parametrize("interval", INTERVALS)
def test_hurdle_interval_mass_adds_the_atom_and_the_component(
    family: str, interval: tuple[float | None, float | None, bool, bool]
) -> None:
    low, high, lower_closed, upper_closed = interval
    zero = 0.25
    if family == "lognormal":
        law = HurdleLogNormal(_t(zero), _t(0.0), _t(1.0))
        component = stats.lognorm(1.0)
    else:
        law = HurdleGamma(_t(zero), _t(2.0), _t(1.0))
        component = stats.gamma(2.0)
    start = max(-math.inf if low is None else low, 0.0)
    stop = math.inf if high is None else high
    holds_atom = (low is None or low < 0 or (low == 0 and lower_closed)) and (
        high is None or high > 0 or (high == 0 and upper_closed)
    )
    # The component mass is taken from the smaller pair of tails, so the
    # reference itself does not cancel.
    if stop <= start:
        part = 0.0
    elif component.cdf(stop) < 0.5:
        part = component.cdf(stop) - component.cdf(start)
    else:
        part = component.sf(start) - component.sf(stop)
    expected = zero * holds_atom + (1 - zero) * part
    observed = log_interval_mass(
        law,
        None if low is None else _t(low),
        None if high is None else _t(high),
        lower_closed=lower_closed,
        upper_closed=upper_closed,
    )
    if expected == 0.0:
        assert float(observed) == -math.inf
    else:
        assert float(observed) == pytest.approx(math.log(expected), rel=1e-9)
        assert math.isfinite(float(observed))
