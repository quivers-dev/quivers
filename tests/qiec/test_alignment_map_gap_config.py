"""Check exact alignment-MAP gap targets independently of path agreement."""

from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from itertools import product

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import dual_alignment_map
from quivers.core.relations.alignment_dual import _DualAlignmentRun
from quivers.qiec import (
    JointAlignmentMapGapConfig,
    SerializationError,
    StringFactorGraphSpec,
    dumps,
    from_data,
    loads,
    to_data,
)
from tests.qiec.test_string_solver_configuration import (
    _SCORES,
    _TAPE,
    _UNIT,
    _map_config,
    _relation,
)


def _solver():
    return _map_config().with_(max_iterations=4, max_features=0)


def _fixture(left, right):
    relation_left = _relation(_TAPE, ((0, _SCORES[0]), (1, _SCORES[1])))
    relation_right = _relation(_TAPE, ((1, _SCORES[3]), (0, _SCORES[2])))
    model = StringFactorGraphSpec(
        variables=(("x", "Letter", 2, None),),
        relations=(("left", relation_left), ("right", relation_right)),
        factors=(("left", ("x",)), ("right", ("x",))),
    )
    values = {_UNIT: torch.tensor(0.0, dtype=torch.float64)}
    values.update(
        {
            owner: torch.tensor(
                float("-inf") if weight is None else weight, dtype=torch.float64
            )
            for owner, weight in zip(_SCORES, (*left, *right), strict=True)
        }
    )
    return model, values


def _independent_bounds(left, right):
    complete = [
        Fraction(a + b)
        for (x, a), (y, b) in product(enumerate(left), enumerate(right))
        if x == y and a is not None and b is not None
    ]
    separated = [
        Fraction(a + b)
        for a, b in product(left, right)
        if a is not None and b is not None
    ]
    return max(complete, default=None), max(separated, default=None)


@pytest.mark.parametrize(
    "offset,relative",
    [(0, Fraction(3, 2)), (2, None), (6, Fraction(3, 10))],
)
def test_exact_absolute_gap_reaches_a_bound_without_agreement(offset, relative):
    left = (-1 + offset, -4 + offset)
    right = (-4 + offset, -1 + offset)
    model, parameters = _fixture(left, right)
    independent_optimum, local_bound = _independent_bounds(left, right)
    config = JointAlignmentMapGapConfig(solver=_solver(), absolute_gap_ratio=(3, 1))
    original = dumps(config)
    assert isinstance(config := loads(original), JointAlignmentMapGapConfig)
    assert dumps(config) == original
    actual = config.solve(model, parameters)
    assert actual.status == "gap" and actual.gap_target_met
    assert actual.certificate is None and actual.iterations == 1
    assert actual.assignment == (("x", (0,)),)
    assert actual.score == independent_optimum and actual.upper_bound == local_bound
    assert actual.absolute_gap == Fraction(3)
    assert actual.relative_gap == relative
    assert actual.score <= independent_optimum <= actual.upper_bound

    too_small = config.with_(absolute_gap_ratio=(299, 100)).solve(model, parameters)
    assert too_small.status == "limit" and not too_small.gap_target_met
    assert too_small.iterations == 4 and too_small.certificate is None


@pytest.mark.parametrize("offset", [0, 2, 6])
def test_relative_target_uses_an_exact_bound_and_refuses_a_zero_crossing(offset):
    left, right = (-1 + offset, -4 + offset), (-4 + offset, -1 + offset)
    model, parameters = _fixture(left, right)
    optimum, upper = _independent_bounds(left, right)
    assert optimum is not None and upper is not None
    tolerance = (
        (100, 1)
        if optimum < 0 < upper
        else ((upper - optimum) / abs(upper)).as_integer_ratio()
    )
    config = JointAlignmentMapGapConfig(solver=_solver(), relative_gap_ratio=tolerance)
    result = config.solve(model, parameters)
    assert result.certificate is None
    if optimum < 0 < upper:
        assert result.status == "limit" and not result.gap_target_met
        assert result.relative_gap is None and result.absolute_gap == 3
        return
    assert result.status == "gap" and result.gap_target_met
    bound = result.relative_gap
    assert bound is not None
    for n in range(11):
        possible_optimum = optimum + Fraction(n, 10) * (upper - optimum)
        assert (possible_optimum - optimum) / abs(possible_optimum) <= bound
    stricter = (bound - Fraction(1, 100)).as_integer_ratio()
    assert (
        config.with_(relative_gap_ratio=stricter).solve(model, parameters).status
        == "limit"
    )
    both = config.with_(absolute_gap_ratio=(2, 1)).solve(model, parameters)
    assert not both.gap_target_met and both.status == "limit"


def test_zero_gap_with_tied_disagreement_does_not_create_an_agreement_certificate():
    identical = (0, 0)
    model, parameters = _fixture(identical, identical)
    config = JointAlignmentMapGapConfig(solver=_solver(), relative_gap_ratio=(0, 1))
    ordinary = config.solver.solve(model, parameters)
    assert ordinary.status == "limit" and ordinary.certificate is None
    result = config.solve(model, parameters)
    assert result.status == "gap" and result.gap_target_met
    assert result.absolute_gap == result.relative_gap == Fraction(0)
    assert result.score == result.upper_bound == Fraction(0)
    assert result.certificate is None

    agreed_model, agreed_parameters = _fixture((-1, -4), (-1, -4))
    agreed = config.solve(agreed_model, agreed_parameters)
    assert agreed.status == "optimal" and agreed.gap_target_met
    assert agreed.certificate is not None

    disjoint, disjoint_parameters = _fixture((-1, None), (None, -1))
    impossible = config.with_(relative_gap_ratio=(100, 1)).solve(
        disjoint, disjoint_parameters
    )
    assert impossible.status == "limit" and not impossible.gap_target_met
    assert impossible.score is None and impossible.upper_bound == -2
    assert impossible.absolute_gap is None and impossible.relative_gap is None


@pytest.mark.parametrize("shift,relative", [(0, None), (1, Fraction(1))])
def test_relative_zero_endpoints_have_distinct_guarantees(shift, relative):
    model, parameters = _fixture((shift, shift - 2), (shift - 2, shift))
    config = JointAlignmentMapGapConfig(solver=_solver(), relative_gap_ratio=(1, 1))
    actual = config.solve(model, parameters)
    assert actual.score == 2 * shift - 2 and actual.upper_bound == 2 * shift
    assert actual.absolute_gap == 2 and actual.relative_gap == relative
    assert actual.gap_target_met is (relative is not None)
    assert actual.certificate is None


def test_inconsistent_local_bound_refuses_even_under_a_generous_gap(monkeypatch):
    model, parameters = _fixture((-1, -4), (-4, -1))
    original = _DualAlignmentRun.decode

    def underestimate(self, words, adjustments):
        exact = original(self, words, adjustments)
        if exact is not None and words["x"] is None:
            return tuple(
                replace(path, adjusted_score=path.adjusted_score - 8) for path in exact
            )
        return exact

    monkeypatch.setattr(_DualAlignmentRun, "decode", underestimate)
    config = JointAlignmentMapGapConfig(solver=_solver(), absolute_gap_ratio=(100, 1))
    with pytest.raises(ArithmeticError, match="score exceeds the dual upper bound"):
        config.solve(model, parameters)


def test_suboptimal_agreement_refuses_a_self_consistent_false_certificate(monkeypatch):
    model, parameters = _fixture((-1, -4), (-1, -4))
    correct = _solver().solve(model, parameters)
    optimum, _ = _independent_bounds((-1, -4), (-1, -4))
    assert correct.score == optimum == -2 and correct.certificate is not None
    original = _DualAlignmentRun.decode

    def agree_on_wrong_word(self, words, shifts):
        forced = {"x": (1,)} if words["x"] is None else words
        return original(self, forced, shifts)

    monkeypatch.setattr(_DualAlignmentRun, "decode", agree_on_wrong_word)
    # Both paths really score -4, so the proposed primal and dual agree
    # at -8. The scalar subproblem verifier must refuse independently.
    with pytest.raises(
        ArithmeticError, match="does not attain the independent local optimum"
    ):
        _solver().solve(model, parameters)


def test_wrong_dual_witness_cannot_claim_a_gap_with_a_valid_lower_path(monkeypatch):
    model, parameters = _fixture((-4, -1), (-4, -1))
    original = _DualAlignmentRun.decode

    def return_one_suboptimal_path(self, words, shifts):
        if words["x"] is not None:
            return original(self, words, shifts)
        zero = original(self, {"x": (0,)}, shifts)
        one = original(self, {"x": (1,)}, shifts)
        assert zero is not None and one is not None
        return zero[0], one[1]

    monkeypatch.setattr(_DualAlignmentRun, "decode", return_one_suboptimal_path)
    policy = JointAlignmentMapGapConfig(solver=_solver(), absolute_gap_ratio=(3, 1))
    # The feasible score -8 and claimed upper -5 give a nonnegative gap
    # of three, though the actual optimum is -2.
    with pytest.raises(
        ArithmeticError, match="does not attain the independent local optimum"
    ):
        policy.solve(model, parameters)


def test_gap_settings_cannot_claim_an_achieved_bound_or_change_the_old_wire():
    solver = _solver()
    old_digest = sha256(dumps(solver).encode()).hexdigest()
    assert (
        old_digest == "329fb5a3e486e0188eed86886680d334dfc11cb96027a5a9e5930d00a08c4d2b"
    )
    with pytest.raises(dx.ValidationError, match="at least one target"):
        JointAlignmentMapGapConfig(solver=solver)
    with pytest.raises(dx.ValidationError, match="nonnegative rational ratios"):
        JointAlignmentMapGapConfig(solver=solver, relative_gap_ratio=(1, 0))
    model, parameters = _fixture((-1, -4), (-4, -1))
    with pytest.raises(ValueError, match="nonnegative rational"):
        dual_alignment_map(
            model.bind(parameters),
            max_iterations=1,
            max_ngram_order=1,
            max_features=0,
            max_states=8,
            max_work=16,
            absolute_gap=-1,
        )
    payload = to_data(
        JointAlignmentMapGapConfig(solver=solver, absolute_gap_ratio=(3, 1))
    )
    payload["root"]["fields"].append(
        {"name": "gap_target_met", "value": to_data(True)["root"]}
    )
    with pytest.raises(SerializationError, match="field"):
        from_data(payload)
