"""Beta tails agree with independent probability and derivative oracles."""

import math

import mpmath
import pytest
import torch
from scipy.special import betainc, betaincc

from quivers.continuous._beta_tails import log_beta_tails
from quivers.qiec._beta_tails import beta_fraction, log_beta_tails as reference_tails
from quivers.qiec._beta_tails import log_beta_interval_mass
from quivers.continuous._interval import log_interval_mass
from quivers.qiec.distributions import HostLaw


@pytest.mark.parametrize(
    "a,b", [(0.01, 0.1), (0.1, 8), (1, 1), (2, 3), (8, 8), (10, 20), (1000, 2000)]
)
@pytest.mark.parametrize("point", [1e-12, 1e-4, 0.1, 0.3, 0.5, 0.9, 1 - 1e-12])
def test_beta_tails_match_scipy_and_reference(a, b, point):
    native = log_beta_tails(
        *(torch.tensor(v, dtype=torch.double) for v in (a, b, point))
    )
    reference = reference_tails(a, b, point)
    expected = (betainc(a, b, point), betaincc(a, b, point))
    for tensor, scalar, probability in zip(native, reference, expected, strict=True):
        if probability > 1e-280:
            assert tensor.item() == pytest.approx(
                math.log(probability), abs=2e-10, rel=3e-12
            )
        assert tensor.item() == pytest.approx(scalar, abs=2e-10, rel=3e-12)
    assert torch.logaddexp(*native).item() == pytest.approx(0.0, abs=3e-16)


@pytest.mark.parametrize(
    "a,b,point", [(1e-10, 5, 0.2), (5, 1e-10, 0.8), (1e6, 1e6, 0.5), (1e4, 2e4, 1 / 3)]
)
def test_beta_tails_keep_small_complements_and_concentrated_centers(a, b, point):
    with mpmath.workdps(70):
        if a == b and point == 0.5:
            first = mpmath.mpf("0.5")
        elif a >= 1e4:
            aa, bb, xx = map(mpmath.mpf, (a, b, point))
            center = aa / (aa + bb)
            scale = mpmath.sqrt(aa * bb / ((aa + bb) ** 2 * (aa + bb + 1)))
            normalizer = (
                mpmath.loggamma(aa) + mpmath.loggamma(bb) - mpmath.loggamma(aa + bb)
            )

            def density(z):
                t = center + scale * z
                if t <= 0 or t >= 1:
                    return mpmath.mpf(0)
                return scale * mpmath.exp(
                    (aa - 1) * mpmath.log(t) + (bb - 1) * mpmath.log1p(-t) - normalizer
                )

            first = mpmath.quad(
                density, [-center / scale, -10, -1, (xx - center) / scale]
            )
        else:
            first = mpmath.betainc(a, b, 0, point, regularized=True)
        expected = float(mpmath.log(first)), float(mpmath.log1p(-first))
    native = log_beta_tails(
        *(torch.tensor(v, dtype=torch.double) for v in (a, b, point))
    )
    reference = reference_tails(a, b, point)
    for actual, plain, oracle in zip(native, reference, expected, strict=True):
        assert actual.item() == pytest.approx(oracle, abs=3e-11, rel=2e-11)
        assert plain == pytest.approx(oracle, abs=3e-11, rel=2e-11)


@pytest.mark.parametrize(
    "parameters",
    [(0.2, 0.7, 0.1), (2.0, 3.0, 0.3), (12.0, 20.0, 0.375), (8.0, 8.0, 0.5)],
)
def test_beta_tails_have_first_second_and_forward_derivatives(parameters):
    inputs = tuple(
        torch.tensor(v, dtype=torch.double, requires_grad=True) for v in parameters
    )
    assert torch.autograd.gradcheck(
        log_beta_tails, inputs, eps=2e-5, atol=1e-6, rtol=2e-5
    )
    assert torch.autograd.gradgradcheck(
        log_beta_tails, inputs, eps=2e-5, atol=2e-6, rtol=5e-5
    )
    with mpmath.workdps(60):
        for tail in (0, 1):

            def independent(a, b, x):
                probability = mpmath.betainc(a, b, 0, x, regularized=True)
                return mpmath.log(probability if tail == 0 else 1 - probability)

            actual = torch.autograd.grad(
                log_beta_tails(*inputs)[tail], inputs, retain_graph=True
            )
            for axis in range(3):
                order = tuple(1 if index == axis else 0 for index in range(3))
                oracle = float(mpmath.diff(independent, parameters, order))
                assert actual[axis].item() == pytest.approx(oracle, rel=2e-8, abs=1e-9)
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(value.detach(), torch.ones_like(value))
            for value in inputs
        )
        forward = log_beta_tails(*duals)
        for tail in (0, 1):
            tangent = torch.autograd.forward_ad.unpack_dual(forward[tail]).tangent
            expected = sum(torch.autograd.grad(log_beta_tails(*inputs)[tail], inputs))
            torch.testing.assert_close(tangent, expected, atol=1e-10, rtol=1e-10)


def test_beta_tails_broadcast_and_mask_support_without_inactive_nan_gradients():
    a = torch.tensor([0.2, 3.0, 40.0], dtype=torch.double, requires_grad=True)
    b = torch.tensor(1.4, dtype=torch.double, requires_grad=True)
    points = torch.tensor(
        [[-math.inf], [0.0], [0.2], [1.0], [math.inf]], dtype=torch.double
    )
    cdf, sf = log_beta_tails(a, b, points)
    assert cdf.shape == sf.shape == (5, 3)
    assert torch.isneginf(cdf[:2]).all() and (sf[:2] == 0).all()
    assert torch.isneginf(sf[3:]).all() and (cdf[3:] == 0).all()
    for point in (-math.inf, 0.0, 1.0, math.inf):
        scalar = log_beta_tails(a, b, torch.tensor(point, dtype=torch.double))
        gradients = torch.autograd.grad(
            sum(value.sum() for value in scalar), (a, b), retain_graph=True
        )
        assert all(
            torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients
        )


@pytest.mark.parametrize(
    "a,b,point", [(0, 1, 0.5), (1, -1, 0.5), (math.inf, 1, 0.5), (1, 1, math.nan)]
)
def test_beta_tails_reject_invalid_parameters(a, b, point):
    with pytest.raises(ValueError, match="finite positive"):
        reference_tails(a, b, point)
    with pytest.raises(ValueError, match="finite positive"):
        log_beta_tails(*(torch.tensor(v, dtype=torch.double) for v in (a, b, point)))


def test_beta_fraction_has_an_explicit_iteration_budget():
    with pytest.raises(RuntimeError, match="within 1 iterations"):
        beta_fraction(
            2.0,
            3.0,
            0.2,
            nonzero=lambda value: value,
            converged=lambda value: True,
            max_iterations=1,
        )


@pytest.mark.parametrize(
    "a,b,low,high",
    [
        (2, 3, 0.1, 0.8),
        (2, 3, 1e-100, 2e-100),
        (3, 2, 1 - 2e-10, 1 - 1e-10),
        (1e6, 1e6, 0.5, 0.500000000001),
        (1e-6, 20, 0.2, 0.20000000001),
        (2, 3, 0.2, math.nextafter(0.2, 1.0)),
    ],
)
def test_beta_interval_native_reference_and_restriction_use_independent_integral(
    a, b, low, high
):
    with mpmath.workdps(90):
        aa, bb, ll, hh = map(mpmath.mpf, (a, b, low, high))
        integral = mpmath.quad(
            lambda point: point ** (aa - 1) * (1 - point) ** (bb - 1), [ll, hh]
        ) / mpmath.beta(aa, bb)
        expected = float(mpmath.log(integral))
    law = torch.distributions.Beta(
        torch.tensor(a, dtype=torch.double), torch.tensor(b, dtype=torch.double)
    )
    native = log_interval_mass(law, low, high).item()
    scalar = log_beta_interval_mass(a, b, low, high)
    base = HostLaw(
        family="Beta", arguments=dict(concentration1=a, concentration0=b)
    )
    restricted = HostLaw(
        family="Restrict", arguments=dict(base=base, low=low, high=high)
    )
    for actual in (native, scalar, restricted.log_mass()):
        assert actual == pytest.approx(expected, abs=3e-9)


def test_beta_interval_retains_first_and_second_parameter_endpoint_derivatives():
    def mass(a, b, low, high):
        return log_interval_mass(torch.distributions.Beta(a, b), low, high)

    inputs = tuple(
        torch.tensor(value, dtype=torch.double, requires_grad=True)
        for value in (2.3, 3.7, 0.2, 0.8)
    )
    assert torch.autograd.gradcheck(mass, inputs, atol=2e-6, rtol=2e-5)
    assert torch.autograd.gradgradcheck(mass, inputs, atol=3e-6, rtol=3e-5)
