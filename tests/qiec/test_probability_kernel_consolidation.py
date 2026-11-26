"""Legacy ordinal and interval adapters retain the shared stable kernel laws."""

import math
import random

import numpy as np
import pytest
import torch
from scipy import special

from quivers.continuous._discrete_weibull import DiscreteWeibull
from quivers.continuous._interval import log_interval_mass
from quivers.continuous._ordered import OrderedLogistic, OrderedProbit
from quivers.qiec._count_tails import log_count_tails
from quivers.qiec._interval import log_interval_mass as reference_interval
from quivers.qiec.distributions import ReferenceBackend, HostLaw


@pytest.mark.parametrize("q,power", [(0.4, 1e-20), (0.7, 0.1), (0.9, 2.0), (0.0, 1.3)])
def test_discrete_weibull_full_mass_does_not_use_negligible_individual_term_heuristic(
    q, power
):
    base = HostLaw(
        family="DiscreteWeibull", arguments={"probs": q, "concentration": power}
    )
    assert reference_interval(base, None, None) == 0.0
    # Conditioning away zero retains q even when every representable positive
    # point has negligible individual mass and most draws exceed capacity.
    actual = reference_interval(base, 1, None)
    expected = math.log(q) if q else -math.inf
    assert actual == pytest.approx(expected, abs=2e-15)
    assert log_interval_mass(
        DiscreteWeibull(
            torch.tensor(q, dtype=torch.double), torch.tensor(power, dtype=torch.double)
        ),
        1,
        None,
    ).item() == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("q,power", [(0.4, 1.3), (0.9, 0.1), (1e-300, 0.5), (0.0, 2.0)])
@pytest.mark.parametrize("point", [-math.inf, -0.5, 0.0, 0.9, 3.1, 1000, math.inf])
def test_discrete_weibull_reference_tail_contract_matches_closed_survival(
    q, power, point
):
    args = {"probs": q, "concentration": power}
    actual = log_count_tails("DiscreteWeibull", args, point)
    if point < 0:
        expected = (-math.inf, 0.0)
    elif point == math.inf or q == 0:
        expected = (0.0, -math.inf)
    else:
        log_sf = math.log(q) * ((math.floor(point) + 1) ** power)
        log_cdf = math.log(-math.expm1(log_sf)) if log_sf else -math.inf
        expected = (log_cdf, log_sf)
    assert actual == pytest.approx(expected, rel=3e-13, abs=2e-15)


@pytest.mark.parametrize(
    "family,cls,eta",
    [
        ("OrderedLogistic", OrderedLogistic, -1000.0),
        ("OrderedLogistic", OrderedLogistic, 1000.0),
        ("OrderedProbit", OrderedProbit, -40.0),
        ("OrderedProbit", OrderedProbit, 40.0),
    ],
)
def test_ordinal_host_log_scores_keep_extreme_probabilities_and_all_head_gradients(
    family, cls, eta
):
    backend = ReferenceBackend()
    parameters = torch.tensor([eta, -1.0, 1.0], dtype=torch.double, requires_grad=True)
    law = cls(parameters[0], parameters[1:])
    args = {"eta": eta, "cutpoints": [-1.0, 1.0]}
    scores = [backend.family_log_prob(family, args, point) for point in range(3)]
    torch.testing.assert_close(
        torch.tensor(scores, dtype=torch.double),
        law.log_prob(torch.arange(3)).detach(),
        atol=2e-11,
        rtol=2e-13,
    )
    # Independent special-function oracle, reflecting both probit tails.
    a, b = -1 - eta, 1 - eta
    if family == "OrderedLogistic":
        oracle = (
            -np.logaddexp(0, -b) - np.logaddexp(0, a) + math.log(-math.expm1(a - b))
        )
    else:
        larger, smaller = (
            (special.log_ndtr(-a), special.log_ndtr(-b))
            if a > 0
            else (special.log_ndtr(b), special.log_ndtr(a))
        )
        oracle = larger + math.log(-math.expm1(smaller - larger))
    assert scores[1] == pytest.approx(oracle, abs=2e-11)
    (gradient,) = torch.autograd.grad(law.log_prob(torch.tensor(1)), parameters)
    step = 1e-4
    differences = []
    for index in range(3):
        positive = parameters.detach().numpy().copy()
        negative = positive.copy()
        positive[index] += step
        negative[index] -= step
        plus = backend.family_log_prob(
            family, {"eta": float(positive[0]), "cutpoints": positive[1:].tolist()}, 1
        )
        minus = backend.family_log_prob(
            family, {"eta": float(negative[0]), "cutpoints": negative[1:].tolist()}, 1
        )
        differences.append((plus - minus) / (2 * step))
    np.testing.assert_allclose(
        differences, gradient.detach().numpy(), atol=4e-7, rtol=4e-7
    )


@pytest.mark.parametrize("family", ["OrderedLogistic", "OrderedProbit"])
@pytest.mark.parametrize(
    "eta,cuts",
    [
        (0.0, []),
        (0.0, [1.0, -1.0]),
        (0.0, [1.0, 1.0]),
        (math.nan, [-1.0, 1.0]),
        (0.0, [math.nan]),
        (math.inf, [-1.0, 1.0]),
        (0.0, [math.inf]),
    ],
)
def test_ordinal_density_and_sampling_share_native_parameter_validation(
    family, eta, cuts
):
    args = {"eta": eta, "cutpoints": cuts}
    backend = ReferenceBackend()
    with pytest.raises(ValueError):
        backend.family_log_prob(family, args, 0)
    with pytest.raises(ValueError):
        backend.family_sample(family, args, random.Random(2))


@pytest.mark.parametrize(
    "family,cls",
    [("OrderedLogistic", OrderedLogistic), ("OrderedProbit", OrderedProbit)],
)
def test_ordinal_host_sampling_uses_the_same_log_probability_vector(family, cls):
    backend = ReferenceBackend()
    args = {"eta": 0.3, "cutpoints": [-1.0, 0.5, 2.0]}
    rng = random.Random(101)
    draws = [backend.family_sample(family, args, rng) for _ in range(20000)]
    expected = (
        cls(
            torch.tensor(0.3, dtype=torch.double),
            torch.tensor(args["cutpoints"], dtype=torch.double),
        )
        .log_prob(torch.arange(4))
        .exp()
        .numpy()
    )
    actual = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0)


def _decimal_interval(q, power, first, last):
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 400
        probability = Decimal(str(q))
        concentration = Decimal(str(power))
        left = probability ** (Decimal(first) ** concentration) if first else Decimal(1)
        right = probability ** (Decimal(last + 1) ** concentration)
        return float((left - right).ln())


@pytest.mark.parametrize(
    "q,power,first,last",
    [
        (0.4, 1e-20, 1, 50),
        (0.4, 1e-20, 1, 1),
        (0.4, 1e-20, 2**53, 2**53),
        (0.4, 1e-100, 1, 50),
        (0.7, 0.1, 1, 100),
        (1e-300, 1e-20, 1, 2),
        (0.4, 1.3, 3, 9),
        (0.8, 20.0, 1, 2),
        (0.4, 1e-20, 33_554_433, 33_554_434),
    ],
)
def test_factored_weibull_intervals_retain_mass_between_rounded_equal_tails(
    q, power, first, last
):
    args = {"probs": q, "concentration": power}
    native = DiscreteWeibull(
        torch.tensor(q, dtype=torch.double), torch.tensor(power, dtype=torch.double)
    )
    reference = HostLaw(family="DiscreteWeibull", arguments=args)
    expected = _decimal_interval(q, power, first, last)
    assert log_interval_mass(native, first, last).item() == pytest.approx(
        expected, abs=3e-11, rel=5e-13
    )
    assert reference_interval(reference, first, last) == pytest.approx(
        expected, abs=3e-11, rel=5e-13
    )
    if first == last:
        assert native.log_prob(torch.tensor(first)).item() == pytest.approx(expected)


@pytest.mark.parametrize("first,last", [(0, 0), (1, 1), (1, 50), (2, 4)])
@pytest.mark.parametrize(
    "lower_closed,upper_closed",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_weibull_endpoint_membership_matches_explicit_finite_count_sum(
    first, last, lower_closed, upper_closed
):
    q, power = 0.4, 0.7
    args = {"probs": q, "concentration": power}
    native = DiscreteWeibull(
        torch.tensor(q, dtype=torch.double), torch.tensor(power, dtype=torch.double)
    )
    reference = HostLaw(family="DiscreteWeibull", arguments=args)
    start = first + (not lower_closed)
    stop = last - (not upper_closed)
    masses = [
        math.exp(ReferenceBackend().family_log_prob("DiscreteWeibull", args, k))
        for k in range(start, stop + 1)
    ]
    expected = math.log(math.fsum(masses)) if masses else -math.inf
    assert log_interval_mass(
        native, first, last, lower_closed=lower_closed, upper_closed=upper_closed
    ).item() == pytest.approx(expected, abs=2e-13)
    assert reference_interval(
        reference, first, last, lower_closed=lower_closed, upper_closed=upper_closed
    ) == pytest.approx(expected, abs=2e-13)


def test_tiny_weibull_interval_native_head_gradients_and_hessian_match_decimal_oracle():
    from decimal import Decimal, localcontext

    parameters = torch.tensor(
        [0.3, math.log(1e-20)], dtype=torch.double, requires_grad=True
    )

    def function(heads):
        return log_interval_mass(
            DiscreteWeibull(heads[0].sigmoid(), heads[1].exp()), 1, 50
        )

    assert torch.autograd.gradcheck(function, (parameters,), atol=2e-7)
    assert torch.autograd.gradgradcheck(function, (parameters,), atol=2e-7)
    (actual_gradient,) = torch.autograd.grad(function(parameters), parameters)
    actual_hessian = torch.func.hessian(function)(parameters)

    def oracle(values):
        with localcontext() as ctx:
            ctx.prec = 150
            q = 1 / (1 + (-Decimal(str(values[0]))).exp())
            power = Decimal(str(values[1])).exp()
            return _decimal_interval(q, power, 1, 50)

    point = parameters.detach().numpy()
    step = 1e-4
    gradient = []
    hessian = np.empty((2, 2))
    for i in range(2):
        ei = np.eye(2)[i] * step
        gradient.append((oracle(point + ei) - oracle(point - ei)) / (2 * step))
        for j in range(2):
            ej = np.eye(2)[j] * step
            hessian[i, j] = (
                oracle(point + ei + ej)
                - oracle(point + ei - ej)
                - oracle(point - ei + ej)
                + oracle(point - ei - ej)
            ) / (4 * step * step)
    np.testing.assert_allclose(
        actual_gradient.detach().numpy(), gradient, atol=2e-7, rtol=2e-7
    )
    np.testing.assert_allclose(
        actual_hessian.detach().numpy(), hessian, atol=5e-7, rtol=2e-6
    )


def test_weibull_interval_batch_flags_and_zero_endpoint_derivatives():
    q = torch.tensor([0.0, 0.4, 0.7], dtype=torch.double, requires_grad=True)
    power = torch.tensor([0.7, 1e-20, 1.3], dtype=torch.double, requires_grad=True)
    native = DiscreteWeibull(q, power)
    low = torch.tensor([0.0, 1.0, 2.0])
    high = torch.tensor([0.0, 50.0, 4.0])
    closed = torch.tensor([True, True, False])
    scores = log_interval_mass(native, low, high, lower_closed=closed)
    expected = [
        0.0,
        _decimal_interval(0.4, 1e-20, 1, 50),
        _decimal_interval(0.7, 1.3, 3, 4),
    ]
    torch.testing.assert_close(
        scores, torch.tensor(expected, dtype=torch.double), atol=2e-13, rtol=2e-13
    )
    dq, db = torch.autograd.grad(scores.sum(), (q, power))
    assert dq[0].item() == -1.0
    assert db[0].item() == 0.0
    assert torch.isfinite(dq).all() and torch.isfinite(db).all()


def test_probit_unrepresentable_log_tail_remains_impossible_without_nan_difference():
    args = {"eta": 1e200, "cutpoints": [-1.0, 1.0]}
    assert ReferenceBackend().family_log_prob("OrderedProbit", args, 1) == -math.inf
    assert torch.isneginf(
        OrderedProbit(
            torch.tensor(1e200, dtype=torch.double),
            torch.tensor([-1.0, 1.0], dtype=torch.double),
        ).log_prob(torch.tensor(1))
    )


def test_weibull_log_cdf_preserves_tiny_finite_complement_after_probability_rounds_one():
    q, power = 0.4, 6.0
    args = {"probs": q, "concentration": power}
    actual = log_count_tails("DiscreteWeibull", args, 1)
    expected_sf = math.log(q) * (2**power)
    expected_cdf = math.log1p(-math.exp(expected_sf))
    assert expected_cdf < 0
    assert actual[0] == pytest.approx(expected_cdf, rel=2e-13, abs=0)
    assert actual[1] == pytest.approx(expected_sf, rel=2e-15)
