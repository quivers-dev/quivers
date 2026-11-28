"""Exact rational and geometric checks for cyclic fixed-word posteriors."""

from collections import Counter
from fractions import Fraction
from math import log

import pytest
import torch

from quivers.core.algebras import LOG_PROB, MARKOV, DivergentClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
)
from tests.test_relation_composition import _INPUT


def _loops(first=0.25, second=0.25):
    topology = RelationTopology(
        (_INPUT,),
        1,
        1,
        (Arc(0, 0, (None,), 0), Arc(0, 0, (None,), 1)),
        (Boundary(0, 2),),
        (Boundary(0, 2),),
        ("loop1", "loop2", "unit"),
    )
    values = torch.nn.Parameter(torch.tensor([first, second, 1], dtype=torch.float64))
    return RationalRelation(topology, values, algebra=MARKOV), values


def test_parallel_cycles_produce_exact_counts_entropy_and_cost_derivatives():
    source, weights = _loops()
    query = source.query(((),), max_closure_size=2)
    count = query.expected_arc_counts()
    assert count.arc_ids == (0, 1)
    assert count.normalizer.item() == pytest.approx(2.0)
    assert count.counts.tolist() == pytest.approx([0.5, 0.5])
    stats = query.path_statistics(arc_costs=weights.new_tensor([2, 3]))
    assert stats.log_normalizer.item() == pytest.approx(log(2))
    assert stats.entropy.item() == pytest.approx(log(2) + log(4))
    assert stats.expected_cost is not None
    assert stats.expected_cost.item() == pytest.approx(2.5)
    (actual,) = torch.autograd.grad(stats.expected_cost + stats.entropy, weights)
    continuation = weights[0] + weights[1]
    mean = weights[:2] / (1 - continuation)
    expected = (
        mean @ weights.new_tensor([2.0, 3.0])
        - (1 - continuation).log()
        - mean @ weights[:2].log()
    )
    (reference,) = torch.autograd.grad(expected, weights)
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_two_node_asymmetric_scc_uses_closed_exit_prefix_and_suffix(algebra):
    topology = RelationTopology(
        (_INPUT,),
        0,
        2,
        (Arc(0, 1, (None,), 0), Arc(1, 0, (None,), 1)),
        (Boundary(0, 2),),
        (Boundary(1, 2),),
        ("go", "back", "unit"),
    )
    weights = torch.tensor([0.25, 0.5, 1], dtype=torch.float64)
    if algebra is LOG_PROB:
        weights = weights.log()
    relation = RationalRelation(topology, weights, algebra=algebra)
    query = relation.query(((),), max_closure_size=2)
    counts = query.expected_arc_counts()
    assert dict(
        zip(counts.arc_ids, counts.counts.tolist(), strict=True)
    ) == pytest.approx({0: float(Fraction(8, 7)), 1: float(Fraction(1, 7))})
    stats = query.path_statistics()
    expected_z = float(Fraction(2, 7))
    assert stats.log_normalizer.item() == pytest.approx(log(expected_z))
    assert stats.entropy.item() == pytest.approx(-log(7 / 8) - 1 / 7 * log(1 / 8))


def test_budgeted_cyclic_sampler_has_no_truncation_or_silent_fallback():
    relation, _ = _loops()
    query = relation.query(((),), max_closure_size=2)
    with pytest.raises(ProductiveEpsilonCycleError, match="explicit max_steps"):
        query.sample_path()
    generator = torch.Generator().manual_seed(331)
    counts = Counter()
    failures = 0
    for _ in range(300):
        try:
            chosen = query.sample_path(generator=generator, max_steps=2)
        except RelationLimitError:
            failures += 1
            continue
        counts[chosen.arc_ids] += 1
    assert failures / 300 == pytest.approx(1 / 8, abs=0.05)
    assert counts[()] / 300 == pytest.approx(0.5, abs=0.08)
    assert sum(
        count for ids, count in counts.items() if len(ids) == 1
    ) / 300 == pytest.approx(0.25, abs=0.07)
    assert counts[(0,)] and counts[(1,)]
    assert all(len(ids) <= 2 for ids in counts)


def test_divergent_posterior_and_structural_zero_weights():
    relation, weights = _loops(first=0, second=0.25)
    query = relation.query(((),), max_closure_size=2)
    assert query.expected_arc_counts().counts.tolist() == pytest.approx([0, 1 / 3])
    assert torch.isfinite(query.path_statistics().entropy)
    with torch.no_grad():
        weights[0] = 0.75
    for call in (
        query.expected_arc_counts,
        query.path_statistics,
        lambda: query.sample_path(max_steps=10),
    ):
        with pytest.raises(DivergentClosureError):
            call()


def test_tied_loop_weights_accumulate_original_occurrence_gradients():
    topology = RelationTopology(
        (_INPUT,),
        0,
        1,
        (Arc(0, 0, (None,), 0), Arc(0, 0, (None,), 0)),
        (Boundary(0, 1),),
        (Boundary(0, 1),),
        ("shared", "unit"),
    )
    weights = torch.nn.Parameter(torch.tensor([0.125, 1.0], dtype=torch.float64))
    relation = RationalRelation(topology, weights, algebra=MARKOV)
    counts = relation.query(((),), max_closure_size=2).expected_arc_counts()
    assert counts.arc_ids == (0, 1)
    assert counts.counts.tolist() == pytest.approx([1 / 6, 1 / 6])
    (first,) = torch.autograd.grad(counts.counts.sum(), weights, create_graph=True)
    assert first[0].item() == pytest.approx(32 / 9)
    (second,) = torch.autograd.grad(first[0], weights)
    assert second[0].item() == pytest.approx(512 / 27)
