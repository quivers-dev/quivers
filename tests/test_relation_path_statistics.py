"""Entropy and additive feature oracles over explicitly enumerated paths."""

from math import log, prod

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationTopology,
)
from tests.evidence.oracles import paths
from tests.test_relation_composition import _INPUT, _topology
from tests.test_relation_path_inference import _fixture


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_entropy_cost_and_derivatives_match_complete_paths(algebra):
    machine, values = _fixture()
    topology = _topology(machine, (_INPUT,))
    raw = torch.tensor(
        [float(values[name]) for name in topology.weight_names], dtype=torch.float64
    )
    slots = nn.Parameter(raw if algebra is MARKOV else raw.log())
    relation = RationalRelation(topology, slots, algebra=algebra)
    starts = torch.arange(4, dtype=slots.dtype).reshape(2, 2)
    arc_features = nn.Parameter(
        torch.arange(2 * len(machine.arcs), dtype=slots.dtype).reshape(-1, 2) / 3
    )
    stops = torch.arange(4, dtype=slots.dtype).reshape(2, 2) / 5
    query = relation.query(((0,),))
    stats = query.path_statistics(
        initial_costs=starts, arc_costs=arc_features, final_costs=stops
    )

    # The oracle multiplies each full path before normalizing any choice.
    # It includes both boundary marks and uses the original tied slots.
    lookup = dict(zip(topology.weight_names, slots.unbind(), strict=True))
    start_ids = {mark.state: index for index, mark in enumerate(topology.initial)}
    stop_ids = {mark.state: index for index, mark in enumerate(topology.final)}
    masses = []
    costs = []
    for path in paths.accepting_paths(machine):
        if path.strings() != (("a",),) or not prod(
            values[name] for name in path.factors()
        ):
            continue
        weights = torch.stack([lookup[name] for name in path.factors()])
        masses.append(weights.prod() if algebra is MARKOV else weights.sum().exp())
        path_cost = (
            starts[start_ids[path.start.state]] + stops[stop_ids[path.end.state]]
        )
        for arc in path.arcs:
            path_cost = path_cost + arc_features[int(arc.name)]
        costs.append(path_cost)
    normalized = torch.stack(masses) / torch.stack(masses).sum()
    oracle_entropy = -(normalized * normalized.log()).sum()
    oracle_cost = normalized @ torch.stack(costs)
    torch.testing.assert_close(stats.log_normalizer, torch.stack(masses).sum().log())
    torch.testing.assert_close(stats.entropy, oracle_entropy)
    assert stats.expected_cost is not None
    torch.testing.assert_close(stats.expected_cost, oracle_cost)
    actual = torch.autograd.grad(
        stats.entropy + stats.expected_cost.sum(), (slots, arc_features)
    )
    oracle = torch.autograd.grad(
        oracle_entropy + oracle_cost.sum(), (slots, arc_features)
    )
    for computed, expected in zip(actual, oracle, strict=True):
        torch.testing.assert_close(computed, expected, rtol=2e-14, atol=2e-14)


def test_log_probabilities_near_minus_one_thousand_keep_entropy_over_duplicate_paths():
    topology = RelationTopology(
        (_INPUT,),
        0,
        2,
        (Arc(0, 1, (0,), 0), Arc(0, 1, (0,), 0), Arc(0, 1, (0,), 2)),
        (Boundary(0, 1),),
        (Boundary(1, 1),),
        ("shared", "unit", "zero"),
    )
    weights = nn.Parameter(torch.tensor([-1000, 0, -torch.inf], dtype=torch.float64))
    relation = RationalRelation(topology, weights, algebra=LOG_PROB)
    stats = relation.query(((0,),)).path_statistics()
    assert stats.log_normalizer.item() == pytest.approx(-1000 + log(2))
    assert stats.entropy.item() == pytest.approx(log(2))
    assert stats.expected_cost is None
    (gradient,) = torch.autograd.grad(stats.entropy, weights)
    assert torch.isfinite(gradient).all()
    assert gradient.tolist() == pytest.approx([0, 0, 0], abs=5e-13)


def test_empty_path_counts_explicit_boundary_costs_and_has_zero_entropy():
    topology = RelationTopology(
        (_INPUT,),
        1,
        1,
        (),
        (Boundary(0, 0),),
        (Boundary(0, 0),),
        ("tiny",),
    )
    weights = nn.Parameter(torch.tensor([1e-200], dtype=torch.float64))
    relation = RationalRelation(topology, weights, algebra=MARKOV)
    stats = relation.query(((),)).path_statistics(
        initial_costs=weights.new_tensor([2.0]),
        final_costs=weights.new_tensor([3.0]),
    )
    assert stats.log_normalizer.item() == pytest.approx(2 * log(1e-200))
    assert stats.entropy.item() == 0
    assert stats.expected_cost is not None
    assert stats.expected_cost.item() == pytest.approx(5.0)


def test_zero_support_and_invalid_costs_are_refused():
    machine, values = _fixture()
    topology = _topology(machine, (_INPUT,))
    weights = torch.tensor([float(values[name]) for name in topology.weight_names])
    query = RationalRelation(topology, weights, algebra=MARKOV).query(((0,),))
    with pytest.raises(ValueError, match="one finite real row"):
        query.path_statistics(arc_costs=torch.zeros(1))
    with pytest.raises(ValueError, match="one finite real row"):
        query.path_statistics(
            initial_costs=torch.full((len(topology.initial),), float("nan"))
        )
    with pytest.raises(ValueError, match="trailing shape"):
        query.path_statistics(
            initial_costs=torch.zeros(len(topology.initial)),
            final_costs=torch.zeros(len(topology.final), 2),
        )
    impossible = RationalRelation(topology, weights, algebra=MARKOV).query(((0, 0),))
    with pytest.raises(ValueError, match="positive finite normalizer"):
        impossible.path_statistics()


def test_repeated_original_arc_counts_once_at_each_observed_offset():
    topology = RelationTopology(
        (_INPUT,),
        0,
        1,
        (Arc(0, 0, (0,), 0),),
        (Boundary(0, 1),),
        (Boundary(0, 1),),
        ("repeat", "unit"),
    )
    weights = torch.tensor([0.5, 1.0], dtype=torch.float64)
    source = RationalRelation(topology, weights, algebra=MARKOV)
    stats = source.query(((0, 0, 0),)).path_statistics(
        arc_costs=weights.new_tensor([[2.0, 3.0]])
    )
    assert stats.entropy.item() == pytest.approx(0.0)
    assert stats.expected_cost is not None
    torch.testing.assert_close(stats.expected_cost, weights.new_tensor([6, 9]))


def test_productive_epsilon_cycle_has_no_acyclic_statistics_contract():
    topology = RelationTopology(
        (_INPUT,),
        0,
        1,
        (Arc(0, 0, (None,), 0),),
        (Boundary(0, 1),),
        (Boundary(0, 1),),
        ("repeat", "unit"),
    )
    weights = torch.tensor([0.5, 1.0], dtype=torch.float64)
    source = RationalRelation(topology, weights, algebra=MARKOV)
    query = source.query(((),), max_closure_size=1)
    assert query.weight().item() == 2.0
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic productive"):
        query.path_statistics()
