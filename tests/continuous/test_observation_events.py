"""Recorded censoring events and inclusive sampling bounds share one measure."""

import math

import pytest
import torch
from scipy.special import ndtr

from quivers.continuous._closed_probability import ClosedPoisson
from quivers.continuous._observation_events import (
    event_log_prob,
)
from quivers.continuous.measure import Mixture, PointMass


def test_continuous_censoring_and_truncation_matches_independent_cdf_ratios():
    mu = torch.tensor([0.1, 0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.5, 0.8, -0.2, 0.1], dtype=torch.float64)
    lower = torch.tensor([-1.0, -0.5, -1.0, -0.2], dtype=torch.float64)
    upper = torch.tensor([1.5, 1.0, 1.5, 0.9], dtype=torch.float64)
    interval = torch.tensor([math.nan, math.nan, math.nan, 0.7], dtype=torch.float64)
    actual = event_log_prob(
        torch.distributions.Normal(mu, 1.1),
        y,
        status=torch.tensor([0, 1, -1, 2]),
        interval_upper=interval,
        trunc_lower=lower,
        trunc_upper=upper,
    )
    m = mu.detach().numpy()
    normalization = ndtr((upper.numpy() - m) / 1.1) - ndtr((lower.numpy() - m) / 1.1)
    numerator = [
        math.exp(-0.5 * ((float(y[0]) - m[0]) / 1.1) ** 2)
        / (1.1 * math.sqrt(2 * math.pi)),
        ndtr((float(upper[1]) - m[1]) / 1.1) - ndtr((float(y[1]) - m[1]) / 1.1),
        ndtr((float(y[2]) - m[2]) / 1.1) - ndtr((float(lower[2]) - m[2]) / 1.1),
        ndtr((float(interval[3]) - m[3]) / 1.1) - ndtr((float(y[3]) - m[3]) / 1.1),
    ]
    expected = torch.tensor(
        [math.log(float(n) / z) for n, z in zip(numerator, normalization, strict=True)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert bool(
        torch.isfinite(
            torch.autograd.grad(actual.sum(), mu, create_graph=True)[0]
        ).all()
    )


def test_continuous_event_score_first_and_second_derivatives_match_direct_math():
    mu = torch.tensor([0.2, -0.3, 0.5, 0.1], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.7, 0.2, 0.9, -0.4], dtype=torch.float64)
    status = torch.tensor([0, 1, -1, 2])
    actual = event_log_prob(
        torch.distributions.Normal(mu, scale),
        y,
        status=status,
        interval_upper=torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float64),
    )
    z = (y - mu) / scale
    expected = torch.stack(
        [
            -0.5 * z[0] ** 2 - scale.log() - 0.5 * math.log(2 * math.pi),
            torch.special.log_ndtr(-z[1]),
            torch.special.log_ndtr(z[2]),
            torch.log(
                torch.special.ndtr((0.8 - mu[3]) / scale) - torch.special.ndtr(z[3])
            ),
        ]
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    observed = torch.autograd.grad(actual.sum(), (mu, scale), create_graph=True)
    oracle = torch.autograd.grad(expected.sum(), (mu, scale), create_graph=True)
    for a, b in zip(observed, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    for a, b in zip(
        torch.autograd.grad(sum(v.sum() for v in observed), (mu, scale)),
        torch.autograd.grad(sum(v.sum() for v in oracle), (mu, scale)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


def test_discrete_event_endpoints_and_truncation_are_not_clamp_probabilities():
    rate = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    actual = event_log_prob(
        ClosedPoisson(rate),
        torch.tensor([1, 1, 1, 1]),
        status=torch.tensor([0, 1, -1, 2]),
        interval_upper=torch.tensor([99.0, 99.0, 99.0, 2.0]),
        trunc_lower=torch.tensor(1.0),
        trunc_upper=torch.tensor(3.0),
    )
    mass = [math.exp(-1.4) * 1.4**k / math.factorial(k) for k in range(4)]
    denominator = sum(mass[1:])
    expected = torch.tensor(
        [math.log(p / denominator) for p in [mass[1], sum(mass[2:]), mass[1], mass[2]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert bool(torch.isfinite(torch.autograd.grad(actual.sum(), rate)[0]))


def test_mixed_atom_events_include_only_the_correct_boundaries():
    base = Mixture(
        torch.tensor([0.3, 0.7], dtype=torch.float64),
        [
            PointMass(torch.tensor(0.0, dtype=torch.float64)),
            torch.distributions.Uniform(
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            ),
        ],
    )
    actual = event_log_prob(
        base,
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        status=torch.tensor([-1, 1, 2, 0]),
        interval_upper=torch.tensor(
            [math.nan, math.nan, 0.5, math.nan], dtype=torch.float64
        ),
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(
            [math.log(0.3), math.log(0.7), math.log(0.35), math.log(0.3)],
            dtype=torch.float64,
        ),
        atol=2e-12,
        rtol=2e-12,
    )


def test_inactive_log_density_does_not_receive_invalid_positive_support_points():
    shape = torch.tensor([0.4, 1.2, 2.1], dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Beta(shape, torch.tensor(0.7, dtype=torch.float64))
    actual = event_log_prob(
        base,
        torch.tensor([0.2, -1.0, 2.0], dtype=torch.float64),
        status=torch.tensor([0, 1, -1]),
    )
    assert bool(torch.isfinite(actual).all())
    first = torch.autograd.grad(actual.sum(), shape, create_graph=True)[0]
    assert bool(torch.isfinite(first).all())
    assert bool(torch.isfinite(torch.autograd.grad(first.sum(), shape)[0]).all())


def test_integer_measurement_endpoints_keep_inactive_continuous_points_interior():
    shape = torch.tensor([0.4, 1.2], dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Gamma(shape, torch.tensor(1.0, dtype=torch.float64))
    actual = event_log_prob(base, torch.tensor([1, -1]), status=torch.tensor([0, 1]))
    expected = torch.stack((-shape[0].lgamma() - 1, shape[1] * 0))
    torch.testing.assert_close(actual, expected)
    derivative = torch.autograd.grad(actual.sum(), shape, create_graph=True)[0]
    assert bool(torch.isfinite(derivative).all())
    assert bool(torch.isfinite(torch.autograd.grad(derivative.sum(), shape)[0]).all())


@pytest.mark.parametrize(
    "status",
    [torch.tensor([3]), torch.tensor([-2]), torch.tensor([0.5]), torch.tensor([True])],
)
def test_invalid_censoring_codes_fail_before_density(status):
    with pytest.raises(ValueError, match="integer codes"):
        event_log_prob(
            torch.distributions.Normal(0.0, 1.0), torch.tensor([0.0]), status=status
        )


def test_zero_mass_truncation_and_missing_interval_endpoints_fail():
    base = torch.distributions.Uniform(0.0, 1.0)
    with pytest.raises(ValueError, match="positive finite mass"):
        event_log_prob(
            base,
            torch.tensor(0.5),
            trunc_lower=torch.tensor(2.0),
            trunc_upper=torch.tensor(3.0),
        )
    with pytest.raises(ValueError, match="upper endpoint"):
        event_log_prob(base, torch.tensor(0.5), status=torch.tensor(2))
    with pytest.raises(ValueError, match="greater than"):
        event_log_prob(
            base,
            torch.tensor(0.5),
            status=torch.tensor(2),
            interval_upper=torch.tensor(0.4),
        )


@pytest.mark.parametrize("upper", [math.inf, math.nan, -math.inf])
def test_active_interval_requires_finite_upper_endpoint(upper):
    with pytest.raises(ValueError, match="upper endpoint"):
        event_log_prob(
            torch.distributions.Normal(0.0, 1.0),
            torch.tensor(0.0),
            status=torch.tensor(2),
            interval_upper=torch.tensor(upper),
        )


@pytest.mark.parametrize("tensor_endpoint", [False, True])
def test_literal_endpoint_precision_and_existing_tensor_carrier_are_preserved(
    tensor_endpoint,
):
    location = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Normal(location, 1.0)
    upper = torch.tensor(1.3, dtype=torch.float32) if tensor_endpoint else 1.3
    lower, ceiling = -0.7, 1.7
    y = torch.tensor(0.3, dtype=torch.float64)
    score = event_log_prob(
        base, y, status=2, interval_upper=upper, trunc_lower=lower, trunc_upper=ceiling
    )
    endpoint = (
        upper.double() if tensor_endpoint else torch.tensor(upper, dtype=torch.float64)
    )
    expected = (base.cdf(endpoint) - base.cdf(y)).log() - (
        base.cdf(torch.tensor(ceiling, dtype=torch.float64))
        - base.cdf(torch.tensor(lower, dtype=torch.float64))
    ).log()
    torch.testing.assert_close(score, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(score, location)[0],
        torch.autograd.grad(expected, location)[0],
        rtol=2e-12,
        atol=2e-12,
    )
