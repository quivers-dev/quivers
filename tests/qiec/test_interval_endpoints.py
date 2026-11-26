"""Reference endpoint policies agree with atomic and continuous event masses."""

import math
import random

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec, QiecDiagnosticError
from quivers.qiec import run_named, HostLaw
from quivers.qiec._interval import log_interval_mass
from quivers.qiec.reference_support import DistributionError


def _distribution(family, **arguments):
    return HostLaw(family=family, arguments=arguments)


@pytest.mark.parametrize("lower_closed", [False, True])
@pytest.mark.parametrize("upper_closed", [False, True])
def test_atom_membership_has_no_quadrature_or_normalization(lower_closed, upper_closed):
    atom = _distribution("PointMass", value=2.0)
    mass = log_interval_mass(
        atom, 2, 2, lower_closed=lower_closed, upper_closed=upper_closed
    )
    assert mass == (0 if lower_closed and upper_closed else -math.inf)


def test_nested_raw_mass_is_not_multiplied_twice():
    uniform = _distribution("Uniform", low=0.0, high=1.0)
    half = _distribution("Restrict", base=uniform, low=0.0, high=0.5)
    outer = _distribution("Restrict", base=half, low=0.25, high=0.75)
    assert outer.log_mass() == pytest.approx(math.log(0.25), abs=1e-11)
    normalized = _distribution("Normalize", base=outer)
    assert log_interval_mass(normalized, 0.3, 0.4) == pytest.approx(
        math.log(0.4), abs=1e-11
    )


def test_mixture_event_includes_only_selected_atoms_and_raw_component_mass():
    half = _distribution(
        "Restrict", base=_distribution("Uniform", low=0.0, high=1.0), low=0.0, high=0.5
    )
    atom = _distribution("PointMass", value=0.0)
    mixture = _distribution("Mixture", weights=(2.0, 3.0), component=(half, atom))
    assert mixture.log_mass() == pytest.approx(math.log(4.0), abs=1e-11)
    assert log_interval_mass(mixture, 0, 0.25) == pytest.approx(
        math.log(3.5), abs=1e-11
    )
    assert log_interval_mass(mixture, 0, 0.25, lower_closed=False) == pytest.approx(
        math.log(0.5), abs=1e-11
    )


def test_source_can_express_half_open_event_mass():
    source = """define mass() : LogWeight !{} =
    return log_mass(Restrict(Poisson(2.3), 1.0, 3.0, false, true))
"""
    program = lower_qvr_to_qiec(parse(source), file_path="endpoints.qvr")
    base = _distribution("Poisson", rate=2.3)
    expected = math.log(sum(math.exp(base.log_prob(x)) for x in (2, 3)))
    assert run_named(program, "mass").value == pytest.approx(expected)


def test_source_checks_boolean_endpoint_flags():
    source = """define mass() : LogWeight !{} =
    return log_mass(Restrict(Poisson(2.3), 1.0, 3.0, 0, true))
"""
    with pytest.raises(QiecDiagnosticError, match="Bool"):
        lower_qvr_to_qiec(parse(source), file_path="endpoints.qvr")


def test_extended_beta_interval_keeps_boundary_atoms():
    from quivers.qiec._extended_beta import distribution

    arguments = {"mu": 0.3, "phi": 4.0, "kappa": 0.2}
    base = _distribution("ExtendedBeta", **arguments)
    kernel = distribution(arguments)
    assert log_interval_mass(base, 0, 0) == pytest.approx(kernel.log_prob(0))
    assert log_interval_mass(base, 1, 1) == pytest.approx(kernel.log_prob(1))
    assert log_interval_mass(base, 0, 1) == pytest.approx(0.0)
    assert log_interval_mass(base, 0, 1, lower_closed=False, upper_closed=False) < 0


def test_empty_reference_restriction_refuses_without_rejection_loop():
    base = _distribution(
        "Restrict", base=_distribution("PointMass", value=0.0), low=1.0
    )
    with pytest.raises(DistributionError, match="finite positive interval mass"):
        base.sample(random.Random(1))


def test_reference_normal_narrow_interval_preserves_small_positive_mass():
    base = _distribution("Normal", loc=0.3, scale=1.2)
    low, high = 0.4, 0.400000000000001
    midpoint = (low + high) / 2
    expected = base.log_prob(midpoint) + math.log(high - low)
    assert log_interval_mass(base, low, high) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "rate,low,high",
    [(1e6, 999999, 1000001), (20.0, 100.0, 110.0), (2.0, 0.0, math.inf)],
)
def test_reference_poisson_tail_mass_avoids_long_count_sums(rate, low, high):
    from scipy import stats

    base = _distribution("Poisson", rate=rate)
    expected = (
        stats.poisson.cdf(high, rate) - stats.poisson.cdf(math.ceil(low) - 1, rate)
        if high < rate
        else stats.poisson.sf(math.ceil(low) - 1, rate) - stats.poisson.sf(high, rate)
    )
    assert log_interval_mass(base, low, high) == pytest.approx(
        math.log(expected), abs=2e-9
    )


@pytest.mark.parametrize(
    "family,arguments,point",
    [
        ("Binomial", {"total_count": 5, "probs": 0.4}, 1),
        (
            "BetaBinomial",
            {"total_count": 5, "concentration1": 2.0, "concentration0": 3.0},
            1,
        ),
        ("Geometric", {"probs": 0.4}, 1),
    ],
)
def test_reference_discrete_closed_singletons_keep_their_mass(family, arguments, point):
    base = _distribution(family, **arguments)
    assert log_interval_mass(base, point, point) == pytest.approx(base.log_prob(point))
    assert log_interval_mass(base, point, point, lower_closed=False) == -math.inf
