"""Reference rectangle restrictions use declared product axes and original points."""

import math
import random

import pytest
import torch

from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec._interval import log_interval_mass


@pytest.mark.parametrize("batch,event", [((), (2, 3)), ((2,), (3,))])
def test_vector_restriction_density_mass_and_sampling_match_independent_product(
    batch, event
):
    loc = torch.tensor([[-0.4, 0.1, 0.7], [0.2, -0.3, 0.8]], dtype=torch.double)
    base = RuntimeDistribution(
        family="Normal",
        arguments={"loc": tuple(tuple(row) for row in loc.tolist()), "scale": 0.7},
        batch=batch,
        event=event,
        ranks={"loc": 0, "scale": 0},
    )
    restricted = RuntimeDistribution(
        family="Restrict", arguments={"base": base, "low": -1.0, "high": 1.0}
    )
    normalized = RuntimeDistribution(family="Normalize", arguments={"base": restricted})
    law = torch.distributions.Normal(loc, 0.7)
    logmass = (
        (law.cdf(loc.new_tensor(1.0)) - law.cdf(loc.new_tensor(-1.0)))
        .log()
        .sum()
        .item()
    )
    assert log_interval_mass(base, -1.0, 1.0) == pytest.approx(logmass, abs=1e-12)
    assert restricted.log_mass() == pytest.approx(logmass, abs=1e-12)
    point = ((0.1, -0.2, 0.3), (0.2, -0.1, 0.4))
    expected = law.log_prob(loc.new_tensor(point)).sum().item()
    assert restricted.log_prob(point) == pytest.approx(expected, abs=1e-12)
    assert normalized.log_prob(point) == pytest.approx(expected - logmass, abs=1e-12)
    assert normalized.log_prob(((0.1, -0.2, 2.0), (0.2, -0.1, 0.4))) == -math.inf
    draw = normalized.sample(random.Random(12))
    assert len(draw) == 2 and all(len(row) == 3 for row in draw)
    assert all(-1 <= value <= 1 for row in draw for value in row)


def test_plated_discrete_singleton_mass_is_product_of_original_count_atoms():
    base = RuntimeDistribution(
        family="Binomial",
        arguments={"total_count": (0, 1, 2, 3), "probs": 0.5},
        event=(4,),
        ranks={"total_count": 0, "probs": 0},
    )
    assert log_interval_mass(base, 0, 0) == pytest.approx(6 * math.log(0.5), abs=1e-12)
    assert log_interval_mass(base, 0, 0, lower_closed=False) == -math.inf


@pytest.mark.parametrize("value", [(0.0, 0.5), ((0.0, 0.5), (0.25, 0.75)), True])
def test_point_mass_rectangle_uses_the_same_coordinate_membership(value):
    point = RuntimeDistribution(family="PointMass", arguments={"value": value})
    restricted = RuntimeDistribution(
        family="Restrict", arguments={"base": point, "low": 0.0, "high": 1.0}
    )
    assert restricted.log_mass() == 0.0
    assert restricted.log_prob(value) == 0.0
    assert log_interval_mass(point, 0.0, 1.0, upper_closed=False) == (
        -math.inf if value is True else 0.0
    )


def test_point_mass_reference_uses_exact_singletons_without_float_rounding():
    atom = RuntimeDistribution(family="PointMass", arguments={"value": 0.0})
    assert atom.log_prob(1e-13) == -math.inf
    large = RuntimeDistribution(family="PointMass", arguments={"value": 2**53 + 1})
    assert large.sample(random.Random(1)) == 2**53 + 1
    assert large.log_prob(2**53) == -math.inf
