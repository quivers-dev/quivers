"""Gamma interval scores retain tails and shape/rate derivatives."""

import math

import pytest
import torch
from scipy import special

from quivers.continuous._gamma_tails import log_gamma_tails
from quivers.continuous._interval import log_interval_mass


@pytest.mark.parametrize("shape", [1e-20, 1e-6, 0.1, 0.7, 1.0, 2.0, 3.4, 10.0, 80.5])
@pytest.mark.parametrize("point", [1e-40, 1e-4, 0.1, 1.0, 5.0, 50.0, 200.0])
def test_log_tails_match_independent_incomplete_gamma(shape, point):
    actual = log_gamma_tails(
        torch.tensor(shape, dtype=torch.float64),
        torch.tensor(point, dtype=torch.float64),
    )
    for got, probability in zip(
        actual,
        (special.gammainc(shape, point), special.gammaincc(shape, point)),
        strict=True,
    ):
        if probability > 0:
            assert got.item() == pytest.approx(math.log(probability), abs=3e-12)


@pytest.mark.parametrize(
    "shape,point",
    [
        (1.0, 3.0),
        (2.0, 4.0),
        (3.0, 6.0),
        (0.4, 0.2),
        (3.4, 2.0),
        (3.4, 6.0),
        (10.0, 11.0),
    ],
)
def test_shape_and_point_first_and_second_derivatives(shape, point):
    a = torch.tensor(shape, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        log_gamma_tails, (a, x), eps=1e-5, atol=2e-6, rtol=2e-5
    )
    assert torch.autograd.gradgradcheck(
        log_gamma_tails, (a, x), eps=1e-5, atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize("point", [100.0, 1000.0, 10_000.0])
def test_extreme_erlang_survival_stays_in_log_space(point):
    got = log_gamma_tails(torch.tensor(2.0), torch.tensor(point))[1]
    assert got.item() == pytest.approx(-point + math.log1p(point), abs=2e-12)


def test_mixed_tail_batches_preserve_gradients_and_exact_zero_endpoint():
    shape = torch.tensor([0.5, 2.0, 3.4, 10.0], dtype=torch.float64, requires_grad=True)
    point = torch.tensor(
        [0.0, 0.1, 1000.0, 5.0], dtype=torch.float64, requires_grad=True
    )
    log_p, log_q = log_gamma_tails(shape, point)
    assert log_p[0] == -torch.inf and log_q[0] == 0
    gradient = torch.autograd.grad(log_q.sum(), (shape, point))
    assert all(torch.isfinite(part).all() for part in gradient)
    assert gradient[0][0] == gradient[1][0] == 0


def test_far_gamma_interval_retains_rate_and_shape_derivatives():
    shape = torch.tensor(2.3, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)

    def mass(a, b):
        return log_interval_mass(torch.distributions.Gamma(a, b), 1.0, 1.1)

    assert torch.autograd.gradcheck(mass, (shape, rate), eps=1e-4, atol=2e-6, rtol=2e-5)
