"""Spline hazards and normalized event times agree with independent oracles."""

import dataclasses
import math
import os
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest
from scipy import integrate, interpolate, stats
import torch

from quivers._splines import MSplineBasis
from quivers.continuous._cox import CoxPH, spline_values
from quivers.continuous._interval import log_interval_mass
from quivers.qiec._cox import CoxPH as ReferenceCoxPH


def _oracle(basis):
    lower, upper = basis.boundary_knots
    p = basis.degree
    knots = [lower] * (p + 1) + list(basis.interior_knots) + [upper] * (p + 1)
    coefficients = np.eye(basis.size)
    splines = []
    for i in range(basis.size):
        b = interpolate.BSpline(knots, coefficients[i], p)
        factor = (p + 1) / (knots[i + p + 1] - knots[i])
        splines.append((b, b.antiderivative(), factor))

    def evaluate(value):
        if value < lower:
            return np.zeros(basis.size), np.zeros(basis.size)
        x = min(value, upper)
        m = np.array([factor * b(x) for b, _, factor in splines])
        integral = np.array([factor * (a(x) - a(lower)) for _, a, factor in splines])
        return m, integral + m * max(0, value - upper)

    return evaluate


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("knots", [(), (0.3, 1.2, 1.8)])
def test_basis_matches_scipy_splines_integrals_and_retains_value_structure(
    degree, knots
):
    basis = MSplineBasis((0.0, 2.0), knots, degree)
    oracle = _oracle(basis)
    values = torch.tensor([-1.0, 0.0, 0.17, *knots, 1.9, 2.0, 4.0], dtype=torch.float64)
    m, integral = spline_values(basis, values)
    for i, value in enumerate(values.tolist()):
        expected_m, expected_i = oracle(value)
        np.testing.assert_allclose(m[i], expected_m, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(integral[i], expected_i, atol=3e-14, rtol=3e-14)
        actual_m, actual_i = basis.evaluate(value)
        np.testing.assert_allclose(actual_m, expected_m, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(actual_i, expected_i, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(basis.evaluate(2)[1], 1, atol=3e-14)
    assert pickle.loads(pickle.dumps(basis)) == basis
    assert dataclasses.replace(basis, degree=degree + 1) != basis
    with pytest.raises(dataclasses.FrozenInstanceError):
        basis.degree = 7


@pytest.mark.parametrize("rate", [0.03, 0.7, 4.0])
def test_normalization_density_cdf_and_reference_match_independent_spline_oracle(rate):
    basis = MSplineBasis((0, 3), (0.25, 0.8, 2.1), 3)
    weights = torch.softmax(torch.linspace(-1, 1, basis.size, dtype=torch.float64), -1)
    oracle_weights = weights.numpy()
    law = CoxPH(basis, weights, rate=rate)
    reference = ReferenceCoxPH(basis, weights.tolist(), rate=rate)
    oracle = _oracle(basis)

    def density(t):
        m, cumulative = oracle(t)
        return (
            rate
            * np.dot(oracle_weights, m)
            * math.exp(-rate * np.dot(oracle_weights, cumulative))
        )

    total = sum(
        integrate.quad(density, a, b, epsabs=1e-12)[0]
        for a, b in zip(basis.breaks, basis.breaks[1:])
    )
    total += integrate.quad(density, 3, np.inf, epsabs=1e-12)[0]
    assert total == pytest.approx(1.0, abs=2e-12)
    for t in (0, 0.1, 0.25, 0.5, 0.8, 2.1, 3, 4, 100):
        x = torch.tensor(t, dtype=torch.float64)
        m, cumulative = oracle(t)
        expected = (
            math.log(rate)
            + math.log(np.dot(oracle_weights, m))
            - rate * np.dot(oracle_weights, cumulative)
        )
        assert law.log_prob(x).item() == pytest.approx(expected, rel=2e-13, abs=2e-13)
        assert reference.log_prob(t) == pytest.approx(expected, rel=2e-13, abs=2e-13)
        expected_sf = -rate * np.dot(oracle_weights, cumulative)
        assert law.log_survival_function(x).item() == pytest.approx(
            expected_sf, rel=2e-13, abs=2e-13
        )
        assert reference.log_survival_function(t) == pytest.approx(
            expected_sf, rel=2e-13, abs=2e-13
        )
        assert law.cdf(x).item() == pytest.approx(reference.cdf(t), abs=2e-15)
        if t:
            assert law.log_cdf(x).item() == pytest.approx(
                reference.log_cdf(t), abs=2e-14
            )


def test_constant_basis_is_exponential_and_broadcasts_sample_batch_dimensions():
    basis = MSplineBasis((0, 2), (), 0)
    rate = torch.tensor([0.2, 3.0], dtype=torch.float64)
    law = CoxPH(basis, torch.ones(2, 1, dtype=torch.float64), rate=rate)
    exponential = torch.distributions.Exponential(rate / 2)
    x = torch.tensor([[0.0], [0.5], [2.0], [8.0]], dtype=torch.float64)
    torch.testing.assert_close(law.log_prob(x), exponential.log_prob(x))
    p = torch.tensor([[1e-100], [0.1], [0.6], [1 - 1e-15]], dtype=torch.float64)
    torch.testing.assert_close(law.icdf(p), exponential.icdf(p), rtol=3e-14, atol=0)
    assert law.rsample((4, 3)).shape == (4, 3, 2)
    expanded = law.expand((5, 2))
    assert expanded.basis is basis
    assert expanded.rsample((3,)).shape == (3, 5, 2)


def test_density_and_cdf_first_second_and_forward_derivatives_include_knots_and_tail():
    basis = MSplineBasis((0, 3), (0.5, 1.5), 3)
    raw = torch.linspace(-0.5, 0.5, basis.size, dtype=torch.float64, requires_grad=True)
    log_rate = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    times = torch.tensor(
        [0.2, 0.5, 1.5, 2.7, 4.0], dtype=torch.float64, requires_grad=True
    )

    def function(raw, log_rate, times):
        law = CoxPH(basis, raw.softmax(-1), log_rate=log_rate)
        return torch.cat(
            (law.log_prob(times), law.log_cdf(times), law.log_survival_function(times))
        )

    assert torch.autograd.gradcheck(
        function, (raw, log_rate, times), atol=2e-5, rtol=2e-4
    )
    assert torch.autograd.gradgradcheck(
        function, (raw, log_rate, times), atol=3e-5, rtol=3e-4
    )
    tangent = tuple(torch.ones_like(x) for x in (raw, log_rate, times))
    actual = torch.func.jvp(function, (raw, log_rate, times), tangent)[1]
    _, expected = torch.autograd.functional.jvp(
        function, (raw, log_rate, times), tangent
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    # dF/dt must be the full density on both sides of the retained boundary.
    law = CoxPH(basis, raw.softmax(-1), log_rate=log_rate)
    derivative = torch.autograd.grad(law.cdf(times).sum(), times)[0]
    torch.testing.assert_close(
        derivative, law.log_prob(times).exp(), rtol=2e-12, atol=2e-12
    )

    # The hazard is continuous at the constant-tail join, so CDF has a time
    # derivative there. The time derivative of log density can have a kink.
    boundary = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(law.cdf(boundary), boundary)[0]
    torch.testing.assert_close(
        derivative, law.log_prob(boundary).exp(), rtol=2e-12, atol=2e-12
    )
    assert torch.autograd.gradcheck(
        lambda raw, lr: function(raw, lr, boundary[None]), (raw, log_rate)
    )
    assert torch.autograd.gradgradcheck(
        lambda raw, lr: function(raw, lr, boundary[None]), (raw, log_rate)
    )


def test_quantile_reference_roundtrip_and_implicit_first_second_derivatives():
    basis = MSplineBasis((0, 2), (0.4, 1.3), 3)
    raw = torch.linspace(-0.5, 0.5, basis.size, dtype=torch.float64, requires_grad=True)
    log_rate = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    probabilities = torch.tensor(
        [1e-100, 1e-5, 0.1, 0.6, 0.99, 1 - 1e-15], dtype=torch.float64
    )
    law = CoxPH(basis, raw.softmax(-1), log_rate=log_rate)
    quantiles = law.icdf(probabilities)
    torch.testing.assert_close(law.cdf(quantiles), probabilities, rtol=3e-13, atol=0)
    reference = ReferenceCoxPH(basis, raw.detach().softmax(-1).tolist(), log_rate=0.3)
    np.testing.assert_allclose(
        quantiles.detach(),
        [reference.icdf(p) for p in probabilities.tolist()],
        rtol=3e-13,
        atol=0,
    )

    def function(raw, log_rate):
        return CoxPH(basis, raw.softmax(-1), log_rate=log_rate).icdf(probabilities[2:5])

    assert torch.autograd.gradcheck(function, (raw, log_rate), atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(function, (raw, log_rate), atol=3e-5, rtol=3e-4)


def test_join_quantile_has_correct_first_implicit_derivative():
    basis = MSplineBasis((0, 2), (0.4,), 3)
    raw = torch.linspace(-0.5, 0.5, basis.size, dtype=torch.float64, requires_grad=True)
    log_rate = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    probability = -torch.expm1(-log_rate.detach().exp())

    def function(raw, lr):
        return CoxPH(basis, raw.softmax(-1), log_rate=lr).icdf(probability)

    assert function(raw, log_rate).item() == pytest.approx(2, abs=1e-13)
    assert torch.autograd.gradcheck(function, (raw, log_rate), atol=3e-5, rtol=3e-4)


def test_log_domain_tail_rescues_overflowing_unscaled_integral_and_quantile():
    basis = MSplineBasis((0, 1e-100), (), 0)
    law = CoxPH(basis, [1.0], log_rate=-800)
    reference = ReferenceCoxPH(basis, [1.0], log_rate=-800)
    value = 1e245
    log_effective_rate = -800 - math.log(1e-100)
    expected_cumulative = math.exp(log_effective_rate + math.log(value))
    assert law.log_survival_function(value).item() == pytest.approx(
        -expected_cumulative, rel=2e-13
    )
    assert reference.log_survival_function(value) == pytest.approx(
        -expected_cumulative, rel=2e-13
    )
    assert law.log_prob(value).item() == pytest.approx(
        log_effective_rate - expected_cumulative, abs=2e-12
    )
    quantile = math.exp(math.log(math.log(2)) - log_effective_rate)
    assert law.icdf(torch.tensor(0.5)).item() == pytest.approx(quantile, rel=2e-13)
    assert reference.icdf(0.5) == pytest.approx(quantile, rel=2e-13)


@pytest.mark.parametrize("log_rate", [-1000.0, 1000.0])
def test_nonrepresentable_positive_quantiles_refuse_without_clipping(log_rate):
    basis = MSplineBasis(degree=0)
    for constructor in (CoxPH, ReferenceCoxPH):
        law = constructor(basis, [1.0], log_rate=log_rate)
        with pytest.raises((FloatingPointError, OverflowError), match="capacity"):
            law.icdf(0.5)
        assert law.icdf(0.0) == 0
        assert law.icdf(1.0) == math.inf


@pytest.mark.parametrize("log_rate", [math.nan, math.inf, -math.inf])
def test_log_rate_must_be_finite(log_rate):
    for constructor in (CoxPH, ReferenceCoxPH):
        with pytest.raises(ValueError, match="finite"):
            constructor(MSplineBasis(degree=0), [1.0], log_rate=log_rate)


def test_log_cdf_retains_representable_small_intensity_log_probability_and_gradients():
    basis = MSplineBasis(degree=0)
    log_rate = torch.tensor(-1000.0, dtype=torch.float64, requires_grad=True)
    law = CoxPH(basis, [1.0], log_rate=log_rate)
    reference = ReferenceCoxPH(basis, [1.0], log_rate=-1000)
    assert law.log_cdf(1.0).item() == -1000
    assert reference.log_cdf(1.0) == -1000
    assert torch.autograd.grad(law.log_cdf(1.0), log_rate)[0].item() == 1


def test_large_basis_request_has_explicit_preallocation_resource_refusal():
    with pytest.raises(NotImplementedError, match="coefficients"):
        MSplineBasis(degree=1_000_000)


def test_nan_observations_are_refused_by_density_and_both_tail_interfaces():
    for constructor in (CoxPH, ReferenceCoxPH):
        law = constructor(MSplineBasis(degree=0), [1.0], rate=1)
        for method in (law.log_prob, law.log_cdf, law.log_survival_function):
            with pytest.raises(ValueError, match="NaN"):
                method(math.nan)


def test_inactive_tiny_tail_never_contaminates_interior_quantile_gradients():
    basis = MSplineBasis((0, 2), (0.4,), 3)
    weights = torch.tensor(
        [0.25, 0.25, 0.25, 0.25, 1e-320], dtype=torch.float64, requires_grad=True
    )
    log_rate = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    law = CoxPH(basis, weights, log_rate=log_rate)
    quantile = law.icdf(0.1)
    assert torch.isfinite(quantile)
    gradients = torch.autograd.grad(quantile, (weights, log_rate))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_quantile_forward_ad_broadcasting_and_original_float32_head_gradients():
    basis = MSplineBasis((0, 2), (0.4,), 3)
    raw = torch.linspace(-1, 1, basis.size, requires_grad=True)
    log_rate = torch.tensor(0.2, requires_grad=True)
    probabilities = torch.tensor([0.02, 0.3, 0.9], dtype=torch.float64)

    def function(raw, lr):
        return CoxPH(basis, raw.softmax(-1), log_rate=lr).icdf(probabilities)

    tangent = (torch.ones_like(raw), torch.ones_like(log_rate))
    actual = torch.func.jvp(function, (raw, log_rate), tangent)[1]
    _, expected = torch.autograd.functional.jvp(function, (raw, log_rate), tangent)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    result = function(raw, log_rate)
    assert result.dtype == torch.float64
    result.sum().backward()
    assert raw.grad.dtype == torch.float32 and torch.isfinite(raw.grad).all()
    assert log_rate.grad.dtype == torch.float32 and torch.isfinite(log_rate.grad)
    law = CoxPH(
        basis, torch.ones(3, basis.size) / basis.size, rate=torch.tensor([[0.2], [3.0]])
    )
    p = probabilities[:, None, None]
    quantiles = law.icdf(p)
    assert quantiles.shape == (3, 2, 3)
    torch.testing.assert_close(
        law.cdf(quantiles), p.expand(3, 2, 3), rtol=3e-13, atol=0
    )


def test_censoring_interval_protocol_retains_full_tail_and_parameter_gradients():
    basis = MSplineBasis((0, 2), (0.4,), 3)
    raw = torch.linspace(-1, 1, basis.size, dtype=torch.float64, requires_grad=True)
    log_rate = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    law = CoxPH(basis, raw.softmax(-1), log_rate=log_rate)
    lower = torch.tensor([-torch.inf, 0.2, 4.0, 1000.0], dtype=torch.float64)
    upper = torch.tensor([0.5, 1.2, torch.inf, 1000.001], dtype=torch.float64)
    masses = log_interval_mass(law, lower, upper)
    reference = ReferenceCoxPH(basis, raw.detach().softmax(-1).tolist(), log_rate=0.2)
    expected = []
    for lo, hi in zip(lower.tolist(), upper.tolist()):
        if lo == -math.inf:
            expected.append(reference.log_cdf(hi))
        elif hi == math.inf:
            expected.append(reference.log_survival_function(lo))
        else:
            a, b = (
                reference.log_survival_function(lo),
                reference.log_survival_function(hi),
            )
            expected.append(a + math.log(-math.expm1(b - a)))
    torch.testing.assert_close(
        masses, torch.tensor(expected, dtype=torch.float64), rtol=3e-13, atol=2e-10
    )
    assert torch.autograd.gradcheck(
        lambda raw, lr: log_interval_mass(
            CoxPH(basis, raw.softmax(-1), log_rate=lr), lower, upper
        ),
        (raw, log_rate),
        rtol=2e-4,
        atol=3e-5,
    )


def test_native_and_reference_need_no_optional_scientific_libraries():
    source = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'pandas', 'xarray', 'arviz'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
from quivers._splines import MSplineBasis
from quivers.continuous._cox import CoxPH
from quivers.qiec._cox import CoxPH as Reference
basis = MSplineBasis((0, 2), (), 0)
assert CoxPH(basis, [1.], rate=2).log_prob(1).item() == -1
assert Reference(basis, [1.], rate=2).log_prob(1) == -1
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_both_samplers_reproduce_uniform_probability_integral_transform_and_tail_mass():
    basis = MSplineBasis((0, 2), (0.4, 1.3), 3)
    weights = torch.full((basis.size,), 1 / basis.size, dtype=torch.float64)
    law = CoxPH(basis, weights, rate=0.7)
    reference = ReferenceCoxPH(basis, weights.tolist(), rate=0.7)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        native = law.rsample((16000,))
    rng = random.Random(41)
    samples = [reference.sample(rng) for _ in range(5000)]
    assert stats.kstest(law.cdf(native).numpy(), "uniform").pvalue > 0.001
    assert stats.kstest([reference.cdf(t) for t in samples], "uniform").pvalue > 0.001
    assert (native > 2).double().mean().item() == pytest.approx(
        math.exp(-0.7), abs=0.015
    )
    # An omitted tail would leave almost half of this law's mass unrepresented.
    assert max(samples) > 10


def test_support_endpoints_infinite_bounds_and_zero_inactive_gradients():
    basis = MSplineBasis((0, 2), (0.4,), 3)
    raw = torch.zeros(basis.size, dtype=torch.float64, requires_grad=True)
    log_rate = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    law = CoxPH(basis, raw.softmax(-1), log_rate=log_rate)
    points = torch.tensor([-torch.inf, -1.0, 0.0, torch.inf], dtype=torch.float64)
    assert torch.isneginf(law.log_prob(points)[[0, 1, 3]]).all()
    torch.testing.assert_close(
        law.cdf(points), torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    )
    assert law.icdf(torch.tensor(0.0)) == 0
    assert law.icdf(torch.tensor(1.0)) == torch.inf
    loss = (
        law.log_prob(points)[[0, 1, 3]].sum()
        + law.log_cdf(points).sum()
        + law.log_survival_function(points).sum()
    )
    gradients = torch.autograd.grad(loss, (raw, log_rate))
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": -1},
        {"degree": True},
        {"interior_knots": (0.5, 0.5)},
        {"interior_knots": (2.0,)},
        {"boundary_knots": (1, 0)},
        {"boundary_knots": (0, math.inf)},
        {"tail": "none"},
    ],
)
def test_invalid_basis_is_refused(kwargs):
    with pytest.raises(ValueError):
        MSplineBasis(**kwargs)


@pytest.mark.parametrize(
    "weights,rate",
    [
        ([0.0, 1.0], 1),
        ([0.2, 0.2], 1),
        ([math.nan, 1], 1),
        ([0.5, 0.5], 0),
        ([0.5, 0.5], -1),
        ([0.5, 0.5], math.inf),
    ],
)
def test_invalid_law_parameters_refused_by_both_implementations(weights, rate):
    basis = MSplineBasis(degree=1)
    for constructor in (CoxPH, ReferenceCoxPH):
        with pytest.raises(ValueError):
            constructor(basis, weights, rate=rate)
