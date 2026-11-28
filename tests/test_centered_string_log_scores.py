"""Independent dyadic path oracles for opt-in binary64 normalization."""

from fractions import Fraction
from math import exp, inf, log
from sys import float_info

import pytest
import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
)
from tests.test_relation_composition import _INPUT, _OUTPUT

_LIMITS = {"max_states": 50, "max_work": 100}
_CENTER = {"max_center_occurrences": 100, "max_dyadic_work": 700}
_PATHS = (
    ((0,), (0, 1, 0)),
    ((0,), (0, 1, 0)),
    ((1,), (0, 1, 1, 2, 3, 0)),
    ((1,), (0, 4, 0)),
)


def _fixture():
    topology = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        5,
        (
            Arc(0, 4, (0, 0), 1),
            Arc(0, 4, (0, 0), 1),
            Arc(0, 1, (0, None), 1),
            Arc(1, 2, (None, None), 1),
            Arc(2, 3, (None, None), 2),
            Arc(3, 4, (None, 1), 3),
            Arc(0, 4, (0, 1), 4),
        ),
        (Boundary(0, 0),),
        (Boundary(4, 0),),
        ("unit", "shared", "opposite", "bias", "optional"),
    )
    weights = torch.nn.Parameter(
        torch.tensor([0, -1e20, 1e20, 0, -inf], dtype=torch.float64)
    )
    return RationalRelation(topology, weights, algebra=LOG_PROB), weights


def _oracle(weights, observed, *, shift_per_arc=False):
    live = weights.detach().tolist()
    usable = [
        (word, slots)
        for word, slots in _PATHS
        if all(live[slot] != -inf for slot in slots)
    ]
    logs = [
        sum(Fraction(live[slot]) for slot in slots)
        - (len(slots) - 2) * Fraction(live[1]) * shift_per_arc
        for _, slots in usable
    ]
    maximum = max(logs)
    masses = [exp(float(value - maximum)) for value in logs]
    expected = []
    selected = [
        mass if observed == word else 0.0
        for mass, (word, _) in zip(masses, usable, strict=True)
    ]
    numerator, denominator = sum(selected), sum(masses)
    for slot in range(len(live)):
        counts = [path.count(slot) for _, path in usable]
        restricted = sum(m * c for m, c in zip(selected, counts, strict=True))
        total = sum(m * c for m, c in zip(masses, counts, strict=True))
        expected.append(restricted / numerator - total / denominator)
    return log(numerator / denominator), expected


def _tree(relation):
    graph = StringFactorGraph(
        (StringVariable("input", _INPUT, (0,)), StringVariable("output", _OUTPUT)),
        (StringFactor(("input", "output"), relation),),
    )
    return graph.tree_marginal("output", **_LIMITS)


@pytest.mark.parametrize("mode", ("conditional", "tree"))
def test_exact_gauge_handles_unequal_lengths_and_parallel_live_occurrences(mode):
    relation, weights = _fixture()
    conditional = relation.conditional(((0,),), **_LIMITS)
    if mode == "conditional":
        slots = 5
        first = conditional.centered_query(((0,),), **_CENTER)
        second = conditional.centered_query(((1,),), **_CENTER)
        impossible = conditional.centered_query(((),), **_CENTER)
    else:
        slots = 7
        marginal = _tree(relation)
        first = marginal.centered_query((0,), **_CENTER)
        second = marginal.centered_query((1,), **_CENTER)
        impossible = marginal.centered_query((), **_CENTER)
    legacy = conditional.log_prob(((0,),))
    assert legacy.item() == 0.0
    assert _oracle(weights, (1,), shift_per_arc=True)[0] == 0.0

    for bias, optional in ((0.0, -inf), (log(2), -1e20)):
        with torch.no_grad():
            weights[3] = bias
            weights[4] = optional
        outcomes = []
        for word, score in zip(((0,), (1,)), (first, second), strict=True):
            expected_value, expected_grad = _oracle(weights, word)
            result = score()
            assert result.log_prob.item() == pytest.approx(expected_value, abs=1e-15)
            (gradient,) = torch.autograd.grad(result.log_prob, weights)
            assert gradient.tolist() == pytest.approx(expected_grad, abs=1e-15)
            assert result.weight_slots == slots
            assert result.dyadic_work <= _CENTER["max_dyadic_work"]
            outcomes.append(result.log_prob.exp().item())
        assert sum(outcomes) == pytest.approx(1.0, abs=1e-15)
        zero = impossible().log_prob
        assert zero.item() == -inf and not zero.requires_grad

    with torch.no_grad():
        weights[1] = weights[4] = -inf
    with pytest.raises(ValueError, match="positive finite normalizer"):
        first()


def test_centered_work_limits_include_both_observations_before_evaluation():
    relation, _ = _fixture()
    conditional = relation.conditional(((0,),), **_LIMITS)
    modes = {"max_center_occurrences": 13, "max_dyadic_work": 84}
    report = conditional.centered_query(((0,),), **modes)()
    assert (report.states, report.occurrences, report.weight_slots) == (7, 13, 5)
    assert report.dyadic_work == 80
    for field in modes:
        limits = modes | {field: modes[field] - 1}
        with pytest.raises(RelationLimitError, match=field):
            conditional.centered_query(((0,),), **limits)


def _columns(depth, rare, common=0.0, *, dtype=torch.float64, algebra=LOG_PROB):
    topology = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        depth + 1,
        tuple(
            Arc(state, state + 1, (None, symbol), symbol)
            for state in range(depth)
            for symbol in range(2)
        ),
        (Boundary(0, 2),),
        (Boundary(depth, 2),),
        ("rare", "common", "unit"),
    )
    values = torch.tensor([rare, common, 0.0], dtype=dtype, requires_grad=True)
    return RationalRelation(topology, values, algebra=algebra), values


def test_range_checks_distinguish_overflowed_reduction_and_final_difference():
    relation, weights = _columns(2, -8e307)
    conditional = relation.conditional(((),), **_LIMITS)
    report = conditional.centered_query(((0, 0),), **_CENTER)()
    assert report.log_prob.item() == float(Fraction(weights[0].item()) * 2)
    (gradient,) = torch.autograd.grad(report.log_prob, weights)
    assert gradient.tolist() == [2.0, -2.0, 0.0]

    relation, weights = _columns(3, -9e307)
    conditional = relation.conditional(((),), **_LIMITS)
    exact = Fraction(weights[0].item()) * 3
    assert exact < -Fraction(float_info.max)
    with pytest.raises(OverflowError, match="centered log ratio.*outside binary64"):
        conditional.centered_query(((0, 0, 0),), **_CENTER)()

    relation, _ = _columns(1, -float_info.max, float_info.max)
    conditional = relation.conditional(((),), **_LIMITS)
    with pytest.raises(OverflowError, match="reduced path factor.*outside binary64"):
        conditional.centered_query(((0,),), **_CENTER)()


def test_centered_ratio_does_not_narrow_an_absolute_path_log_mass():
    relation, weights = _columns(2, float_info.max, float_info.max)
    conditional = relation.conditional(((),), **_LIMITS)
    report = conditional.centered_query(((0, 0),), **_CENTER)()
    assert 2 * Fraction(weights[0].item()) > Fraction(float_info.max)
    assert report.log_prob.item() == pytest.approx(-log(4), abs=1e-15)
    (gradient,) = torch.autograd.grad(report.log_prob, weights)
    assert gradient.tolist() == [1.0, -1.0, 0.0]


def test_zero_prefix_prunes_unrepresentable_reductions_until_it_becomes_live():
    topology = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        4,
        (
            Arc(0, 3, (0, 0), 0),
            Arc(0, 1, (0, None), 1),
            Arc(1, 3, (None, 1), 2),
            Arc(1, 3, (None, 1), 3),
        ),
        (Boundary(0, 0),),
        (Boundary(3, 0),),
        ("unit", "prefix", "rare", "common"),
    )
    weights = torch.tensor(
        [0, -inf, -float_info.max, float_info.max], dtype=torch.float64
    )
    relation = RationalRelation(topology, weights, algebra=LOG_PROB)
    conditional = relation.conditional(((0,),), **_LIMITS)
    reachable = conditional.centered_query(((0,),), **_CENTER)
    inaccessible = conditional.centered_query(((1,),), **_CENTER)
    assert reachable().log_prob.item() == 0.0
    assert inaccessible().log_prob.item() == -inf
    weights[1] = 0
    with pytest.raises(OverflowError, match="reduced path factor"):
        inaccessible()


def test_centered_scope_checks_dtype_algebra_and_structural_cycles():
    relation, _ = _columns(1, -8, dtype=torch.float32)
    conditional = relation.conditional(((),), **_LIMITS)
    with pytest.raises(TypeError, match="binary64"):
        conditional.centered_query(((0,),), **_CENTER)()
    markov, _ = _columns(1, 0.5, 0.25, algebra=MARKOV)
    with pytest.raises(TypeError, match="LogProb"):
        markov.conditional(((),), **_LIMITS).centered_query(((0,),), **_CENTER)

    relation, weights = _columns(1, -8)
    cyclic = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        2,
        (*relation.topology.arcs, Arc(0, 0, (None, None), 0)),
        relation.topology.initial,
        relation.topology.final,
        relation.topology.weight_names,
    )
    cyclic_relation = RationalRelation(cyclic, weights, algebra=LOG_PROB)
    conditional = cyclic_relation.conditional(((),), max_closure_size=2, **_LIMITS)
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic productive"):
        conditional.centered_query(((0,),), **_CENTER)
