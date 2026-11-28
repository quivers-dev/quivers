"""Check finite whole-unit inclusion and its exact directional action."""

from fractions import Fraction
from itertools import combinations, product

import pytest
import torch

from quivers.continuous.program_steps import UnitFactorization
from quivers.continuous.unit_subsets import unit_coefficients
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExactJvp,
    JvpLimit,
    JvpUnsupported,
    UnitSubset,
    exact_jvp,
    plan_unit_subsampling,
)

_SOURCE = """
define unit(value : Real, rate : Real, theta : Real) : Real !{} =
    return value + rate * theta + rate * theta * theta

define logged(value : Real) : Real !{} =
    return log(value)
"""
_PROBABILITIES = (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))
_DESIGN = tuple(
    UnitSubset(positions, probability)
    for positions, probability in zip(
        ((0, 1), (1, 2), (2,)), _PROBABILITIES, strict=True
    )
)


def _jet(value, direction):
    return ExactJvp(Fraction(value), Fraction(direction))


@pytest.fixture(scope="module")
def module():
    module = Compiler(parse(_SOURCE), module_name="finite-unit-subsampling").qiec_module
    assert module is not None
    return module


@pytest.fixture(scope="module")
def units(module):
    return tuple(
        exact_jvp(module, "unit", (value, rate, 0), (0, 0, 1))
        for value, rate in ((1, 5), (4, -1), (-2, 3))
    )


def test_nonuniform_whole_unit_design_matches_native_population_total(units):
    plan = plan_unit_subsampling(units, _DESIGN)
    assert plan.work == 8
    probabilities = Fraction(1, 2), Fraction(4, 5), Fraction(1, 2)
    assert plan.inclusion_probabilities == probabilities
    assert plan.exact == _jet(3, 7)
    assert [item.estimate for item in plan.outcomes] == [
        _jet(7, Fraction(35, 4)),
        _jet(1, Fraction(19, 4)),
        _jet(-4, 6),
    ]
    for name in ("value", "tangent"):
        expected = getattr(plan.exact, name)
        sampled = sum(
            item.subset.probability * getattr(item.estimate, name)
            for item in plan.outcomes
        )
        omitted = sum(
            subset.probability * sum(getattr(units[i], name) for i in subset.positions)
            for subset in _DESIGN
        )
        ratio = sum(
            item.subset.probability
            * 3
            * getattr(item.estimate, name)
            / sum(1 / probabilities[i] for i in item.subset.positions)
            for item in plan.outcomes
        )
        assert sampled == expected
        assert omitted != expected
        assert ratio != expected


def test_uniform_design_agrees_with_existing_whole_plate_unit_coefficients(units):
    def weight(_):
        return torch.zeros(3)

    plate = UnitFactorization(
        latent="unit",
        axes=("Units",),
        sizes=(3,),
        weights=weight,
        raw_weights=weight,
    )
    rows = tuple(torch.tensor(positions) for positions in combinations(range(3), 2))
    design = tuple(
        UnitSubset.from_plate_rows("Units", index, Fraction(1, 3)) for index in rows
    )
    plan = plan_unit_subsampling(units, design)
    assert plan.inclusion_probabilities == (Fraction(2, 3),) * 3
    for index, action in zip(rows, plan.outcomes, strict=True):
        native = unit_coefficients(plate, {}, {"Units": index})
        coefficients = tuple(Fraction(x) for x in native.tolist())
        assert coefficients == tuple(
            Fraction(3, 2) if i in action.subset.positions else 0 for i in range(3)
        )
        assert action.estimate == _jet(
            sum(c * unit.value for c, unit in zip(coefficients, units, strict=True)),
            sum(c * unit.tangent for c, unit in zip(coefficients, units, strict=True)),
        )
    with pytest.raises(ValueError, match="distinct rows"):
        UnitSubset.from_plate_rows("Units", torch.tensor([0, 0]), Fraction(1, 3))
    with pytest.raises(ValueError, match="must hold integer rows"):
        UnitSubset.from_plate_rows("Units", torch.tensor([0.0]), Fraction(1, 3))


def test_exact_subset_distribution_uses_joint_design_probabilities(units, monkeypatch):
    plan = plan_unit_subsampling(units, _DESIGN)
    seen = [0] * 3
    for first, second in product(range(2), range(5)):
        picks = iter(((2, first), (5, second)))

        class Script:
            def randrange(self, stop):
                expected, pick = next(picks)
                assert stop == expected
                return pick

        monkeypatch.setattr(
            "quivers.qiec.finite_subsampling.Random", lambda seed: Script()
        )
        result = plan.sample(seed=42)
        index = 0 if first == 0 else 1 if second < 3 else 2
        assert result == plan.outcomes[index]
        seen[index] += 1
    assert [Fraction(x, 10) for x in seen] == list(_PROBABILITIES)


def test_noncovered_units_and_invalid_design_do_not_rebase_or_renormalize(units):
    with pytest.raises(JvpUnsupported, match=r"unit position 2 has zero inclusion"):
        plan_unit_subsampling(units, (UnitSubset((0, 1), Fraction(1)),))
    for subset in (
        UnitSubset((0, 1, 2), Fraction(1, 2)),
        UnitSubset((0, 1, 2), Fraction(0)),
        UnitSubset((0, 1, 2), Fraction(-1)),
        UnitSubset((0, 1, 2), Fraction(2)),
    ):
        with pytest.raises(ValueError, match="unit subset probabilities"):
            plan_unit_subsampling(units, (subset,))
    for positions in ((0, 0), (1, 0), (0, 4), (), (True,)):
        with pytest.raises(ValueError, match="distinct positions"):
            plan_unit_subsampling(units, (UnitSubset(positions, Fraction(1)),))
    q = UnitSubset((0, 1, 2), Fraction(1, 2))
    with pytest.raises(ValueError, match="same subset twice"):
        plan_unit_subsampling(units, (q, q))
    with pytest.raises(ValueError, match="nonempty"):
        plan_unit_subsampling(units, ())


def test_exact_outcome_and_membership_budgets_are_separate(units):
    exact = plan_unit_subsampling(units, _DESIGN, max_work=8, max_outcomes=3)
    assert exact.work == 8 and len(exact.outcomes) == 3
    with pytest.raises(JvpLimit, match="exceeds 7 unit positions and memberships"):
        plan_unit_subsampling(units, _DESIGN, max_work=7)
    with pytest.raises(JvpLimit, match="exceeds 2 distinct subsets"):
        plan_unit_subsampling(units, _DESIGN, max_outcomes=2)


def test_raw_total_subsampling_and_log_objectives_have_separate_expectations(module):
    units = (_jet(1, 2), _jet(3, 1))
    design = (UnitSubset((0,), Fraction(1, 2)), UnitSubset((1,), Fraction(1, 2)))
    plan = plan_unit_subsampling(units, design)
    assert plan.exact == _jet(4, 3)
    assert [item.estimate for item in plan.outcomes] == [_jet(2, 4), _jet(6, 2)]
    left, right = [item.estimate.value for item in plan.outcomes]
    assert left * right < plan.exact.value**2
    direct_score = sum(unit.tangent / unit.value for unit in units)
    assert direct_score == Fraction(7, 3)
    assert direct_score != plan.exact.tangent / plan.exact.value
    with pytest.raises(JvpUnsupported, match="primitive"):
        exact_jvp(module, "logged", (plan.exact.value,), (plan.exact.tangent,))


def test_tiny_first_order_coverage_does_not_clip_a_signed_estimate():
    tiny = Fraction(1, 2**180)
    units = (_jet(-2, 3), _jet(1, 2))
    design = (UnitSubset((0,), tiny), UnitSubset((1,), 1 - tiny))
    plan = plan_unit_subsampling(units, design)
    assert plan.exact == _jet(-1, 5)
    assert plan.inclusion_probabilities == (tiny, 1 - tiny)
    assert plan.outcomes[0].estimate == _jet(-(2**181), 3 * 2**180)
