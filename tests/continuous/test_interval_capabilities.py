"""Interval extensions share paired tails and the guarded local integrator."""

import math

import torch
from torch.distributions import Distribution, constraints
from torch.nn import functional as F

from quivers.continuous._interval import log_interval_mass


class _LogisticCapabilities(Distribution):
    arg_constraints = {}
    support = constraints.real

    def __init__(self, loc):
        self.loc = loc
        self.tail_calls = 0
        self.variation_calls = 0
        super().__init__(loc.shape, validate_args=False)

    def log_prob(self, value):
        point = value - self.loc
        return -F.softplus(point) - F.softplus(-point)

    def _log_tails(self, value):
        self.tail_calls += 1
        point = value - self.loc
        return F.logsigmoid(point), F.logsigmoid(-point)

    def log_density_variation(self, lower, upper):
        self.variation_calls += 1
        # The absolute logistic score is bounded by one everywhere.
        return upper - lower

    def cdf(self, value):
        raise AssertionError("paired tails must supply interval probabilities")


def test_paired_extension_runs_once_per_endpoint_and_reuses_local_rule():
    loc = torch.tensor([0.2, 0.2], dtype=torch.double, requires_grad=True)
    law = _LogisticCapabilities(loc)
    lower = torch.tensor([0.1, 0.1], dtype=torch.double)
    upper = torch.tensor([1.0, math.nextafter(0.1, math.inf)], dtype=torch.double)
    actual = log_interval_mass(law, lower, upper)
    a, b = lower - loc, upper - loc
    # Exact logistic interval identity; unlike CDF subtraction it retains
    # adjacent floating-point endpoints and provides independent derivatives.
    expected = (
        b + torch.log(-torch.expm1(lower - upper)) - F.softplus(a) - F.softplus(b)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
    actual_gradient = torch.autograd.grad(actual.sum(), loc, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), loc, create_graph=True)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual_gradient.sum(), loc)[0],
        torch.autograd.grad(expected_gradient.sum(), loc)[0],
        rtol=0,
        atol=2e-14,
    )
    assert law.tail_calls == 2
    assert law.variation_calls == 1
