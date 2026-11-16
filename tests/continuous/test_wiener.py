"""Independent first-passage series, ODE, derivative, and sampling oracles."""

import math
import random

import mpmath as mp
import pytest
import torch
from scipy.integrate import quad

from quivers.continuous._wiener import Wiener
from quivers.qiec._wiener import _accept, _pi_interval, _ratio_interval, density, sample
from quivers.qiec.reference_support import DistributionError


def oracle(boundary, ndt, bias, drift, point):
    edge = min(float(bias), 1 - float(bias))
    digits = int(-math.log10(edge)) + 80 if edge > 0 else 100
    with mp.workdps(max(100, mp.mp.dps, digits)):
        a, delay, beta, mu, y = map(mp.mpf, (boundary, ndt, bias, drift, point))
        t = abs(y) - delay
        w = 1 - beta if y > 0 else beta
        v = mu if y > 0 else -mu
        s = t / a**2
        if s < mp.mpf(".15"):
            f = mp.fsum(
                (w + 2 * k) * mp.exp(-((w + 2 * k) ** 2) / (2 * s))
                for k in range(-25, 26)
            ) / mp.sqrt(2 * mp.pi * s**3)
        else:
            f = mp.pi * mp.fsum(
                k * mp.sin(k * mp.pi * w) * mp.exp(-k * k * mp.pi**2 * s / 2)
                for k in range(1, 80)
            )
        return mp.log(f) - 2 * mp.log(a) + v * a * w - v * v * t / 2


def laplace_moments(a, beta, mu):
    # Solve (1/2)F''+mu F'=sF with absorbing endpoint values, independently
    # of the density series, then differentiate the joint Laplace transforms.
    with mp.workdps(70):
        a, beta, mu = map(mp.mpf, (a, beta, mu))
        x = a * beta

        def sinhc(z):
            return mp.sinh(z) / z if z else mp.mpf(1)

        def joint(s, upper):
            q = mp.sqrt(mu * mu + 2 * s)
            r = x if upper else a - x
            tilt = mu * (a - x) if upper else -mu * x
            return mp.exp(tilt) * r / a * sinhc(r * q) / sinhc(a * q)

        p = [mp.re(joint(0, u)) for u in (False, True)]
        m = [-mp.re(mp.diff(lambda s: joint(s, u), 0)) for u in (False, True)]
        q = [mp.re(mp.diff(lambda s: joint(s, u), 0, 2)) for u in (False, True)]
        return tuple(float(v) for v in p + m + q)


@pytest.mark.parametrize("bias", [1e-12, 0.03, 0.4, 0.5, 0.91, 1 - 1e-12])
@pytest.mark.parametrize(
    "unit_time", [0.0003, 0.04, 0.124999, 0.125001, 0.4, 3.0, 40.0]
)
def test_independent_density_series_and_reference(bias, unit_time):
    a, delay, mu = 1.3, 0.2, -0.7
    d = Wiener(a, delay, bias, mu, validate_args=False)
    for sign in (-1, 1):
        y = sign * (delay + a * a * unit_time)
        expected = float(oracle(a, delay, bias, mu, y))
        assert d.log_prob(y).item() == pytest.approx(expected, rel=2e-12, abs=2e-10)
        assert density(
            dict(boundary=a, ndt=delay, bias=bias, drift=mu), y
        ) == pytest.approx(expected, rel=2e-12, abs=2e-10)


@pytest.mark.parametrize(
    "sign,time,bias,drift",
    [
        (1, 0.04, 0.2, 0.0),
        (-1, 0.04, 0.8, 0.0),
        (1, 0.125, 0.4, 0.7),
        (-1, 0.4, 0.7, -2.0),
        (1, 2.0, 0.05, 5.0),
        (1, 0.4, 0.5, 0.0),
        (-1, 0.04, 0.5, -0.7),
    ],
)
def test_every_parameter_first_second_and_forward_derivatives(sign, time, bias, drift):
    p = torch.tensor(
        [1.0, 0.1, bias, drift, sign * (time + 0.1)],
        dtype=torch.double,
        requires_grad=True,
    )

    def score(z):
        return Wiener(*z[:4], validate_args=False).log_prob(z[4])

    first = []
    second = []
    with mp.workdps(70):
        z = tuple(mp.mpf(float(v)) for v in p.detach())
        for i in range(5):
            first.append(
                float(mp.diff(oracle, z, tuple(int(k == i) for k in range(5))))
            )
            second.append(
                [
                    float(
                        mp.diff(
                            oracle,
                            z,
                            tuple(int(k == i) + int(k == j) for k in range(5)),
                        )
                    )
                    for j in range(5)
                ]
            )
    first = torch.tensor(first, dtype=torch.double)
    second = torch.tensor(second, dtype=torch.double)
    torch.testing.assert_close(torch.func.jacrev(score)(p), first, atol=2e-8, rtol=2e-9)
    torch.testing.assert_close(torch.func.jacfwd(score)(p), first, atol=2e-8, rtol=2e-9)
    torch.testing.assert_close(
        torch.func.jacfwd(torch.func.jacrev(score))(p), second, atol=2e-7, rtol=2e-8
    )
    assert torch.autograd.gradcheck(score, (p,))
    assert torch.autograd.gradgradcheck(score, (p,))


def test_regime_boundary_scores_and_derivatives_match():
    p = torch.tensor([1.0, 0.0, 0.31, -0.4, 0.125], dtype=torch.double)

    def score(z):
        return Wiener(*z[:4], validate_args=False).log_prob(z[4])

    left, right = p.clone(), p.clone()
    left[-1] -= 1e-12
    right[-1] += 1e-12
    torch.testing.assert_close(score(left), score(right), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        torch.func.jacfwd(score)(left),
        torch.func.jacfwd(score)(right),
        atol=1e-8,
        rtol=1e-8,
    )
    torch.testing.assert_close(
        torch.func.hessian(score)(left),
        torch.func.hessian(score)(right),
        atol=2e-7,
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    "a,bias,drift", [(1.0, 0.3, 0.0), (1.7, 0.65, 1.2), (0.7, 0.15, -3.0)]
)
def test_joint_normalization_choice_probabilities_and_moments(a, bias, drift):
    d = Wiener(a, 0.0, bias, drift, validate_args=False)
    plo, phi, ml, mh, ql, qh = laplace_moments(a, bias, drift)
    for sign, p, m, q in [(-1, plo, ml, ql), (1, phi, mh, qh)]:
        for power, expected in [(0, p), (1, m), (2, q)]:
            actual = quad(
                lambda t: t**power * math.exp(d.log_prob(sign * t).item()),
                0,
                math.inf,
                epsabs=2e-10,
                epsrel=2e-10,
            )[0]
            assert actual == pytest.approx(expected, abs=2e-9, rel=2e-9)
    assert plo + phi == pytest.approx(1.0, abs=1e-14)
    if drift == 0:
        assert phi == pytest.approx(bias)
        assert ml + mh == pytest.approx(a * a * bias * (1 - bias))
    else:
        assert phi == pytest.approx(
            math.expm1(-2 * drift * a * bias) / math.expm1(-2 * drift * a)
        )
        assert ml + mh == pytest.approx(a * (phi - bias) / drift)


@pytest.mark.parametrize("drift", [0.0, -2.0, 3.0])
def test_symmetry_and_brownian_scaling(drift):
    y = torch.tensor([-0.3, -0.9, 0.4, 1.7], dtype=torch.double)
    d = Wiener(1.2, 0.15, 0.3, drift, validate_args=False)
    reflected = Wiener(1.2, 0.15, 0.7, -drift, validate_args=False)
    torch.testing.assert_close(
        d.log_prob(y), reflected.log_prob(-y), atol=1e-12, rtol=1e-12
    )
    c = 3.7
    scaled = Wiener(1.2 * c, 0.15 * c * c, 0.3, drift / c, validate_args=False)
    torch.testing.assert_close(
        scaled.log_prob(y * c * c),
        d.log_prob(y) - 2 * math.log(c),
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize("bias", [1e-15, 0.05, 0.5, 0.9, 1 - 1e-15])
@pytest.mark.parametrize("time", [0.0001, 0.02, 0.125, 0.126, 1.0, 20.0])
@pytest.mark.parametrize("terms", [1, 2, 4, 12])
def test_certified_ratio_bounds_enclose_independent_series(bias, time, terms):
    with mp.workdps(150):
        for upper in (False, True):
            w = 1 - mp.mpf(bias) if upper else mp.mpf(bias)
            t = mp.mpf(time)
            if time <= 0.125:
                expected = mp.fsum(
                    (w + 2 * k) / w * mp.exp(-((w + 2 * k) ** 2 - w * w) / (2 * t))
                    for k in range(-25, 26)
                )
            else:
                f = mp.pi * mp.fsum(
                    k * mp.sin(k * mp.pi * w) * mp.exp(-k * k * mp.pi**2 * t / 2)
                    for k in range(1, 80)
                )
                expected = f / (
                    w / mp.sqrt(2 * mp.pi * t**3) * mp.exp(-w * w / (2 * t))
                )
            low, high = _ratio_interval(time, bias, upper, terms, 48)
            # The oracle's tiny allowance covers cancellation rounding 1+1e-150.
            assert (
                mp.mpf(str(low)) - mp.mpf("1e-140")
                <= expected
                <= mp.mpf(str(high)) + mp.mpf("1e-140")
            )
            assert 0 <= low <= high <= 1
            assert _accept(time, bias, upper, 0.1, max_terms=64, max_precision=192) == (
                expected > mp.mpf(".1")
            )


def test_certified_pi_and_comparison_refinement():
    with mp.workdps(120):
        low, high = _pi_interval(80)
        assert mp.mpf(str(low)) < mp.pi < mp.mpf(str(high))
    with pytest.raises(DistributionError, match="budgets"):
        _accept(0.13, 0.5, True, 0.99, max_terms=1, max_precision=1)
    assert isinstance(
        _accept(0.13, 0.5, True, 0.99, max_terms=64, max_precision=192), bool
    )


@pytest.mark.parametrize("bias,drift", [(0.3, 0.0), (0.7, 2.0), (0.2, -3.0)])
def test_certified_sampler_choice_and_time_moments(bias, drift):
    n = 4000
    a, delay = 1.1, 0.15
    torch.manual_seed(491)
    draws = Wiener(a, delay, bias, drift).sample((n,))
    assert bool((draws.abs() > delay).all())
    plo, phi, ml, mh, ql, qh = laplace_moments(a, bias, drift)
    empirical = (draws > 0).double().mean().item()
    assert abs(empirical - phi) < 6 * math.sqrt(phi * plo / n) + 1 / n
    mean = ml + mh
    variance = ql + qh - mean * mean
    duration = draws.abs() - delay
    assert abs(duration.mean().item() - mean) < 6 * math.sqrt(variance / n)
    for upper, m, q in [(False, ml, ql), (True, mh, qh)]:
        observed = torch.where((draws > 0) == upper, duration, 0.0).mean().item()
        assert abs(observed - m) < 6 * math.sqrt((q - m * m) / n)
    with mp.workdps(60):
        x = mp.mpf(a) * bias
        q = mp.sqrt(mp.mpf(drift) ** 2 + 2)
        target = float(
            (
                mp.exp(drift * (a - x)) * mp.sinh(x * q)
                + mp.exp(-drift * x) * mp.sinh((a - x) * q)
            )
            / mp.sinh(a * q)
        )
    assert abs((-duration).exp().mean().item() - target) < 0.025
    assert duration.square().mean().item() == pytest.approx(
        ql + qh, rel=0.16, abs=0.002
    )


def test_seed_ownership_broadcast_empty_expand_and_reference_sampler():
    d = Wiener(torch.tensor([1.0, 1.4]), 0.2, torch.tensor([0.3, 0.7]), 0.0)
    torch.manual_seed(81)
    first = d.sample((9,))
    random.seed(99)
    torch.manual_seed(81)
    second = d.sample((9,))
    assert torch.equal(first, second) and first.shape == (9, 2)
    assert d.sample((0,)).shape == (0, 2)
    expanded = d.expand((3, 2))
    assert expanded.boundary.data_ptr() == d.boundary.data_ptr()
    assert expanded.sample((2,)).shape == (2, 3, 2)
    a = dict(boundary=1.1, ndt=0.1, bias=0.4, drift=0.2)
    r1, r2 = random.Random(20), random.Random(20)
    assert [sample(a, r1) for _ in range(6)] == [sample(a, r2) for _ in range(6)]


@pytest.mark.parametrize(
    "slot,value",
    [
        (0, 0.0),
        (0, -1.0),
        (1, -0.1),
        (2, 0.0),
        (2, 1.0),
        (3, math.inf),
        (0, True),
        (2, torch.tensor(True)),
        (3, float("nan")),
    ],
)
def test_parameter_domains(slot, value):
    p = [1.0, 0.1, 0.5, 0.0]
    p[slot] = value
    with pytest.raises(ValueError, match="Wiener"):
        Wiener(*p, validate_args=False)
    if not isinstance(value, torch.Tensor):
        with pytest.raises(DistributionError, match="Wiener"):
            density(
                dict(zip(("boundary", "ndt", "bias", "drift"), p, strict=True)), 0.5
            )


@pytest.mark.parametrize("point", [0.0, 0.2, -0.2, 0.1, -0.1, math.inf, float("nan")])
def test_invalid_reaction_times_are_masked_with_finite_parameter_gradients(point):
    a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
    y = Wiener(a, 0.2, 0.4, 0.7, validate_args=False).log_prob(point)
    assert y.item() == -math.inf
    assert torch.autograd.grad(y, a)[0].item() == 0.0
    assert density(dict(boundary=1.0, ndt=0.2, bias=0.4, drift=0.7), point) == -math.inf


def test_boolean_observation_invalid_budgets_and_proposal_exhaustion():
    with pytest.raises(ValueError, match="nonboolean"):
        Wiener(1.0, 0.0, 0.5, 0.0).log_prob(True)
    for kwargs in [dict(max_terms=0), dict(max_proposals=True), dict(max_precision=0)]:
        with pytest.raises(ValueError, match="budgets"):
            Wiener(1.0, 0.0, 0.5, 0.0, **kwargs)
        with pytest.raises(DistributionError, match="budgets"):
            sample(
                dict(boundary=1.0, ndt=0.0, bias=0.5, drift=0.0),
                random.Random(0),
                **kwargs,
            )

    class RejectingRandom:
        def random(self):
            return 0.99

        def gauss(self, mu, sigma):
            return 0.1

    with pytest.raises(DistributionError, match="proposal budget"):
        sample(
            dict(boundary=1.0, ndt=0.0, bias=0.5, drift=0.0),
            RejectingRandom(),
            max_proposals=1,
        )


def test_large_drift_completed_square_and_float32_gradient_ownership():
    mu = 1e10
    y = 0.5 / mu
    expected = math.log(0.5) - 0.5 * math.log(2 * math.pi) - 1.5 * math.log(y)
    assert Wiener(1.0, 0.0, 0.5, mu).log_prob(y).item() == pytest.approx(
        expected, abs=1e-12
    )
    assert density(dict(boundary=1.0, ndt=0.0, bias=0.5, drift=mu), y) == pytest.approx(
        expected, abs=1e-12
    )
    p = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)
    result = Wiener(1.0, 0.1, p, 0.7).log_prob(0.4)
    assert result.dtype == torch.float64
    gradient = torch.autograd.grad(result, p)[0]
    assert gradient.dtype == torch.float32 and torch.isfinite(gradient)


def test_batched_scores_and_all_parameter_gradients_equal_scalar_oracles():
    p = torch.tensor(
        [[0.8, 1.3, 2.0], [0.1, 0.2, 0.05], [0.2, 0.8, 0.4], [-1.0, 0.0, 2.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    y = torch.tensor([-0.3, 0.55, 1.6], dtype=torch.double, requires_grad=True)
    actual = Wiener(*p).log_prob(y)
    expected = torch.tensor(
        [float(oracle(*p[:, i].detach().tolist(), y[i].item())) for i in range(3)],
        dtype=torch.double,
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gp, gy = torch.autograd.grad(actual.sum(), (p, y))
    for i in range(3):
        a = p[:, i].detach().clone().requires_grad_()
        b = y[i].detach().clone().requires_grad_()
        sp, sy = torch.autograd.grad(Wiener(*a).log_prob(b), (a, b))
        torch.testing.assert_close(gp[:, i], sp, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(gy[i], sy, atol=1e-11, rtol=1e-11)
    assert torch.isfinite(gp).all() and torch.isfinite(gy).all()


@pytest.mark.parametrize(
    "bias,drift", [(0.5, 0.0), (0.2, 3.0), (0.8, -3.0), (0.03, 10.0)]
)
def test_one_boundary_domination_acceptance_mass(bias, drift):
    class CountedRandom(random.Random):
        proposals = 0

        def gauss(self, mu, sigma):
            self.proposals += 1
            return super().gauss(mu, sigma)

    rng = CountedRandom(710)
    n = 1000
    values = [
        sample(dict(boundary=1.0, ndt=0.0, bias=bias, drift=drift), rng)
        for _ in range(n)
    ]
    mass = math.exp(min(0.0, 2 * drift * (1 - bias))) + math.exp(
        min(0.0, -2 * drift * bias)
    )
    expected = 1 / mass
    assert expected >= 0.5
    assert (
        abs(n / rng.proposals - expected)
        < 6 * math.sqrt(expected * (1 - expected) / rng.proposals) + 0.002
    )
    assert all(math.isfinite(v) and v != 0 for v in values)


@pytest.mark.parametrize("bias", [5e-324, 1e-100, 1e-10])
def test_subnormal_bias_density_without_sine_or_distance_underflow(bias):
    for y in [-0.7, 0.7]:
        expected = float(oracle(0.1, 0.0, bias, 0.0, y))
        assert Wiener(0.1, 0.0, bias, 0.0).log_prob(y).item() == pytest.approx(
            expected, abs=1e-10
        )
        assert density(
            dict(boundary=0.1, ndt=0.0, bias=bias, drift=0.0), y
        ) == pytest.approx(expected, abs=1e-10)


def test_sampling_arithmetic_capacity_failure_is_explicit():
    class ZeroNormal:
        def random(self):
            return 0.3

        def gauss(self, mu, sigma):
            return 0.0

    with pytest.raises(DistributionError, match="capacity"):
        sample(dict(boundary=1.0, ndt=0.0, bias=0.5, drift=0.0), ZeroNormal())
    with pytest.raises(ValueError, match="capacity"):
        Wiener(1e-200, 0.0, 0.5, 0.0).log_prob(0.5)


@pytest.mark.parametrize(
    "a,bias,drift",
    [
        (1.0, 0.3, 0.0),
        (1.0, 0.5, 0.0),
        (1.7, 0.65, 1.2),
        (0.7, 0.15, -3.0),
        (1.0, 0.3, 0.5),
        (1.0, 0.3, 0.50001),
        (1.0, 0.3, -0.50001),
        (1.0, 1e-12, 1.0),
        (1.0, 1 - 1e-12, -1.0),
        (1.0, 0.3, 100.0),
    ],
)
def test_closed_moments_against_independent_boundary_value_transform(a, bias, drift):
    delay = 0.2
    d = Wiener(a, delay, bias, drift)
    pl, pu, ml, mu, ql, qu = laplace_moments(a, bias, drift)
    signed = delay * (pu - pl) + mu - ml
    unsigned = delay + mu + ml
    second = delay * delay + 2 * delay * (mu + ml) + ql + qu
    torch.testing.assert_close(
        d.choice_probs,
        torch.tensor([pl, pu], dtype=torch.double),
        atol=2e-15,
        rtol=2e-13,
    )
    assert d.mean.item() == pytest.approx(signed, rel=2e-11, abs=2e-14)
    assert d.mean_response_time.item() == pytest.approx(unsigned, rel=2e-12, abs=2e-14)
    assert d.variance.item() == pytest.approx(
        second - signed * signed, rel=2e-10, abs=2e-14
    )
    assert d.variance_response_time.item() == pytest.approx(
        second - unsigned * unsigned, rel=2e-10, abs=2e-14
    )


@pytest.mark.parametrize("drift", [0.0, 0.01, 0.5, 0.5001, -0.7, 3.0])
def test_every_moment_first_second_and_forward_derivatives(drift):
    p = torch.tensor([1.0, 0.1, 0.3, drift], dtype=torch.double, requires_grad=True)

    def moments(z):
        d = Wiener(*z)
        return torch.cat(
            (
                d.choice_probs,
                torch.stack(
                    (d.mean, d.variance, d.mean_response_time, d.variance_response_time)
                ),
            )
        )

    assert torch.autograd.gradcheck(moments, (p,), atol=2e-7, rtol=1e-6)
    assert torch.autograd.gradgradcheck(moments, (p,), atol=3e-7, rtol=1e-6)
    torch.testing.assert_close(
        torch.func.jacfwd(moments)(p),
        torch.func.jacrev(moments)(p),
        atol=1e-11,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        torch.func.jacfwd(torch.func.jacrev(moments))(p),
        torch.func.jacrev(torch.func.jacrev(moments))(p),
        atol=1e-10,
        rtol=1e-9,
    )


def test_zero_drift_choice_and_moment_derivatives_have_the_analytic_limits():
    p = torch.tensor([1.3, 0.2, 0.3, 0.0], dtype=torch.double, requires_grad=True)
    jac = torch.func.jacrev(lambda z: Wiener(*z).choice_probs)(p)
    a, _, b, _ = p.detach()
    expected = a * b * (1 - b)
    assert jac[1, 3].item() == pytest.approx(expected.item(), abs=1e-14)
    assert jac[0, 3].item() == pytest.approx(-expected.item(), abs=1e-14)
    d = Wiener(*p)
    assert d.mean_response_time.item() == pytest.approx(
        (0.2 + a * a * b * (1 - b)).item(), abs=1e-14
    )
    assert d.variance_response_time.item() == pytest.approx(
        (a**4 * b * (1 - b) * (1 - 2 * b + 2 * b * b) / 3).item(), abs=1e-14
    )


def test_default_zero_density_and_explicit_strict_observation_validation():
    assert Wiener(1.0, 0.3, 0.5, 0.0).log_prob(0.2).item() == -math.inf
    with pytest.raises(ValueError, match="support"):
        Wiener(1.0, 0.3, 0.5, 0.0, validate_args=True).log_prob(0.2)
