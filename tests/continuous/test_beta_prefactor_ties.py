"""Tied Beta shapes retain density curvature in independent directions."""

import math

import mpmath as mp
import numpy as np
import pytest
import torch
from scipy.special import digamma, polygamma

from quivers.continuous._beta_tails import _log_front
from quivers.continuous._extended_beta import ExtendedBeta


@pytest.mark.parametrize("shape", [0.2, 3.0, 7.9, 8.0, 20.0])
def test_tied_beta_prefactor_first_and_second_derivatives_match_independent_oracles(
    shape,
):
    parameters = torch.tensor([shape, shape], dtype=torch.float64, requires_grad=True)
    point = torch.tensor(0.3, dtype=torch.float64)

    def function(values):
        return _log_front(values[0], values[1], point)

    actual_first = torch.autograd.grad(function(parameters), parameters)[0]
    expected_first = [
        math.log(0.3) - digamma(shape) + digamma(2 * shape),
        math.log(0.7) - digamma(shape) + digamma(2 * shape),
    ]
    np.testing.assert_allclose(
        actual_first.detach().numpy(), expected_first, atol=2e-12
    )
    actual = torch.autograd.functional.hessian(function, parameters)
    cross = polygamma(1, 2 * shape)
    expected = np.array(
        [[cross - polygamma(1, shape), cross], [cross, cross - polygamma(1, shape)]]
    )
    # Torch's trigamma approximation itself differs from SciPy/mpmath by
    # about 1e-9 at small shapes. The former tie projection loses O(1)
    # curvature, well above this independent numerical tolerance.
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-10, rtol=1e-9)
    # Independent 80-digit finite differences do not use Torch's gamma AD.
    with mp.workdps(80):
        a = mp.mpf(shape)
        h = mp.mpf("1e-5") * min(1, a)

        def oracle(left, right):
            return (
                left * mp.log(mp.mpf(".3"))
                + right * mp.log(mp.mpf(".7"))
                - mp.loggamma(left)
                - mp.loggamma(right)
                + mp.loggamma(left + right)
            )

        diagonal = (oracle(a + h, a) - 2 * oracle(a, a) + oracle(a - h, a)) / h**2
        mixed = (
            oracle(a + h, a + h)
            - oracle(a + h, a - h)
            - oracle(a - h, a + h)
            + oracle(a - h, a - h)
        ) / (4 * h * h)
    assert actual[0, 0].item() == pytest.approx(float(diagonal), rel=3e-9, abs=3e-11)
    assert actual[0, 1].item() == pytest.approx(float(mixed), rel=3e-9, abs=3e-11)
    forward = torch.func.jacfwd(torch.func.grad(function))(parameters)
    torch.testing.assert_close(forward, actual)
    if shape == 8.0:
        # Central perturbations cross the direct/Stirling dispatch boundary.
        assert torch.autograd.gradcheck(function, (parameters,))
        assert torch.autograd.gradgradcheck(function, (parameters,))


def test_tied_beta_density_retains_antisymmetric_mean_curvature_after_censoring():
    mu = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda m: ExtendedBeta(m, 6.0, 0.2).log_prob(
            torch.tensor(0.3, dtype=torch.float64)
        ),
        (mu,),
    )
