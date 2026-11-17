"""A shared positive baseline kernel has an independent normalized law."""

import math
import random

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from quivers.continuous._bernstein_hazard import BernsteinHazard
from quivers.qiec._bernstein_hazard import BernsteinHazard as ReferenceHazard


def _cumulative(value):
    return 0.5 * value + 0.5 * value**2 if value <= 1 else 1 + 1.5 * (value - 1)


@pytest.mark.parametrize("rate", [0.01, 1.0, 7.0])
def test_linear_baseline_density_tails_and_inverse_match_independent_formula(rate):
    law = BernsteinHazard([0.0, 1.0], [[0.5, 1.5]], rate=rate)
    host = ReferenceHazard([0.0, 1.0], [[0.5, 1.5]], rate=rate)
    for time in [0.0, 0.1, 0.8, 1.0, 1.01, 4.0]:
        h = 0.5 + time if time <= 1 else 1.5
        cumulative = rate * _cumulative(time)
        expected = math.log(rate * h) - cumulative
        assert law.log_prob(time).item() == pytest.approx(expected, abs=2e-13)
        assert host.log_prob(time) == pytest.approx(expected, abs=2e-13)
        assert law.log_survival_function(time).item() == pytest.approx(-cumulative)
        assert host.log_survival_function(time) == pytest.approx(-cumulative)
    probabilities = torch.tensor([1e-12, 0.1, 0.5, 0.9, 1 - 1e-12], dtype=torch.float64)
    targets = -np.log1p(-probabilities.numpy()) / rate
    oracle = np.where(
        targets <= 1, np.sqrt(0.25 + 2 * targets) - 0.5, 1 + (targets - 1) / 1.5
    )
    torch.testing.assert_close(
        law.icdf(probabilities), torch.tensor(oracle), rtol=2e-11, atol=1e-15
    )
    for p, y in zip(probabilities, oracle):
        assert host.icdf(p.item()) == pytest.approx(y, rel=2e-11, abs=1e-15)
    mass = (
        quad(lambda t: math.exp(host.log_prob(t)), 0, 1, epsabs=1e-12)[0]
        + quad(lambda t: math.exp(host.log_prob(t)), 1, math.inf, epsabs=1e-12)[0]
    )
    assert mass == pytest.approx(1, abs=2e-12)


def test_batched_boundaries_and_tables_are_structural_not_response_axes():
    law = BernsteinHazard(
        [[0.0, 1.0], [0.0, 2.0]],
        [[[1.0]], [[0.5]]],
        rate=torch.tensor([2.0, 3.0], dtype=torch.float64),
    )
    times = torch.tensor([[0.1, 0.3], [1.0, 2.0], [4.0, 9.0]], dtype=torch.float64)
    expected = torch.distributions.Exponential(
        torch.tensor([2.0, 1.5], dtype=torch.float64)
    )
    torch.testing.assert_close(law.log_prob(times), expected.log_prob(times))
    torch.testing.assert_close(law.cdf(times), expected.cdf(times))
    assert law.rsample((5,)).shape == (5, 2)
    expanded = law.expand((3, 2))
    assert expanded.event_shape == torch.Size()
    torch.testing.assert_close(
        expanded.log_prob(torch.ones(3, 2)), expected.log_prob(torch.ones(3, 2))
    )


def test_shape_controls_breaks_and_rate_have_first_second_and_forward_derivatives():
    raw = torch.tensor(
        [[0.2, -0.3, 0.1], [-0.1, 0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    intervals = torch.tensor([0.6, 1.4], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    time = torch.tensor([0.2, 0.9, 2.5], dtype=torch.float64)
    p = torch.tensor([0.1, 0.5, 0.95], dtype=torch.float64)

    def evaluate(raw, intervals, eta):
        breaks = torch.cat((intervals.new_zeros(1), intervals.cumsum(0)))
        controls = raw.exp()
        controls = controls / (intervals * (controls / 3).sum(-1)).sum()
        law = BernsteinHazard(breaks, controls, log_rate=eta)
        return torch.cat((law.log_prob(time), law.log_cdf(time), law.icdf(p)))

    assert torch.autograd.gradcheck(
        evaluate, (raw, intervals, eta), atol=2e-5, rtol=2e-4
    )
    assert torch.autograd.gradgradcheck(
        evaluate, (raw, intervals, eta), atol=3e-5, rtol=3e-4
    )
    tangent = tuple(torch.ones_like(v) for v in (raw, intervals, eta))
    actual = torch.func.jvp(evaluate, (raw, intervals, eta), tangent)[1]
    expected = torch.autograd.functional.jvp(evaluate, (raw, intervals, eta), tangent)[
        1
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_native_and_reference_samples_follow_the_full_tail_law():
    law = BernsteinHazard([0.0, 1.0], [[0.5, 1.5]], rate=0.7)
    host = ReferenceHazard([0.0, 1.0], [[0.5, 1.5]], rate=0.7)
    with torch.random.fork_rng():
        torch.manual_seed(26)
        samples = law.sample((16000,))
    assert (samples > 1).double().mean().item() == pytest.approx(
        math.exp(-0.7), abs=0.015
    )
    uniform = law.cdf(samples)
    assert uniform.mean().item() == pytest.approx(0.5, abs=0.01)
    rng = random.Random(33)
    host_samples = [host.sample(rng) for _ in range(2000)]
    assert sum(x > 1 for x in host_samples) / len(host_samples) == pytest.approx(
        math.exp(-0.7), abs=0.04
    )


@pytest.mark.parametrize(
    "breaks,controls",
    [
        ([0.0, 1.0, 2.0], [[1.0]]),
        ([0.0, 1.0], [[0.5], [0.5]]),
        ([0.0, 0.0], [[1.0]]),
        ([1.0, 2.0], [[1.0]]),
        ([0.0, math.inf], [[1.0]]),
        ([0.0, 1.0], [[0.0, 2.0]]),
        ([0.0, 1.0], [[1.0, math.nan]]),
        ([0.0, 1.0], [[0.5, 0.5]]),
    ],
)
def test_malformed_structural_parameters_fail_in_both_constructors(breaks, controls):
    for constructor in (BernsteinHazard, ReferenceHazard):
        with pytest.raises((ValueError, RuntimeError)):
            constructor(breaks, controls, rate=1.0)
