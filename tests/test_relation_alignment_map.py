"""Exhaustive alignment and dual certificate checks for sparse string factors."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from itertools import product

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
    Words,
    dual_alignment_map,
    intersect,
)
from quivers.core.relations.alignment_dual import _NgramScorer
from tests.benchmarks import string_alignment_map as scaling

_TAPE = StringSet(alphabet=FinSet(name="Two", cardinality=2))
_SETTINGS = dict(
    max_iterations=8, max_ngram_order=3, max_features=8, max_states=400, max_work=1000
)


def _unary(words_and_scores):
    arcs = []
    ends = []
    scores = [0]
    state = 1
    for word, score in words_and_scores:
        prior = 0
        for symbol in word:
            arcs.append(Arc(prior, state, (symbol,), 0))
            prior = state
            state += 1
        ends.append(Boundary(prior, len(scores)))
        scores.append(score)
    topology = RelationTopology(
        (_TAPE,),
        0,
        state,
        tuple(arcs),
        (Boundary(0, 0),),
        tuple(ends),
        tuple(f"score{i}" for i in range(len(scores))),
    )
    return RationalRelation(
        topology, torch.tensor(scores, dtype=torch.float64), algebra=LOG_PROB
    )


def _all_paths(
    relation: RationalRelation,
) -> list[tuple[Words, int, tuple[int, ...], int, Fraction]]:
    topology = relation.topology
    weights = [
        None if not torch.isfinite(value) else Fraction(value.item())
        for value in relation.module()()
    ]
    final = {mark.state: weights[mark.weight] for mark in topology.final}
    results = []
    pending: list[tuple[int, int, tuple[int, ...], Words, Fraction]] = [
        (mark.state, mark.state, (), ((),) * len(topology.tapes), weight)
        for mark in topology.initial
        if (weight := weights[mark.weight]) is not None
    ]
    while pending:
        state, start, ids, words, score = pending.pop()
        if (stop := final.get(state)) is not None:
            results.append((words, start, ids, state, score + stop))
        for i, arc in enumerate(topology.arcs):
            if arc.source == state and (weight := weights[arc.weight]) is not None:
                emitted = tuple(
                    word if label is None else (*word, label)
                    for word, label in zip(words, arc.labels, strict=True)
                )
                pending.append((arc.target, start, (*ids, i), emitted, score + weight))
    return results


def _oracle_certificate(result, factors):
    """Check every exact local upper bound and the globally canceled shifts."""
    certificate = result.certificate
    assert certificate is not None
    total_multipliers = defaultdict(Fraction)
    brute = []
    base_total = Fraction(0)
    adjusted_total = Fraction(0)
    for factor, selected, shifts in zip(
        factors, certificate.alignments, certificate.multipliers, strict=True
    ):
        paths = _all_paths(factor.relation)
        brute.append(paths)
        identity = (
            selected.words,
            selected.start,
            selected.arc_ids,
            selected.stop,
            selected.base_score,
        )
        assert identity in paths
        for var, shift in zip(factor.variables, shifts, strict=True):
            for gram, value in shift:
                total_multipliers[var, gram] += value
        candidate_values = []
        for words, _, _, _, score in paths:
            adjustment = Fraction(0)
            for word, shift in zip(words, shifts, strict=True):
                padded = (-1, *word, -2)
                for gram, value in shift:
                    positions = (padded[i : i + len(gram)] for i in range(len(padded)))
                    adjustment += value * sum(item == gram for item in positions)
            candidate_values.append(score + adjustment)
        assert selected.adjusted_score == max(candidate_values)
        base_total += selected.base_score
        adjusted_total += selected.adjusted_score
    assert not any(total_multipliers.values())
    feasible = []
    for local in product(*brute):
        assigned = defaultdict(set)
        for factor, (words, _, _, _, _) in zip(factors, local, strict=True):
            for name, word in zip(factor.variables, words, strict=True):
                assigned[name].add(word)
        if all(len(ws) == 1 for ws in assigned.values()):
            feasible.append(sum(item[-1] for item in local))
    assert (
        base_total
        == adjusted_total
        == result.score
        == result.upper_bound
        == max(feasible)
    )


def test_adaptive_exact_ngram_alignment_certificate_and_unigram_disagreement():
    ab = (0, 1)
    ba = (1, 0)
    factors = (
        StringFactor(("x",), _unary(((ab, 1), (ba, 0)))),
        StringFactor(("x",), _unary(((ab, 0), (ba, 0.5)))),
    )
    graph = StringFactorGraph((StringVariable("x", _TAPE),), factors)
    result = dual_alignment_map(graph, step_size=Fraction(3, 2), **_SETTINGS)
    assert result.status == "optimal"
    assert result.assignment == (("x", ab),)
    assert result.score == 1
    assert result.iterations == 2
    assert result.active_features == (("x", ((-1, 0),)),)
    _oracle_certificate(result, factors)

    limited = dual_alignment_map(
        graph,
        step_size=Fraction(3, 2),
        **(_SETTINGS | dict(max_ngram_order=1, max_iterations=2)),
    )
    assert limited.status == "limit"
    assert limited.certificate is None
    assert limited.active_features == (("x", ()),)
    assert limited.score == 1
    assert limited.upper_bound == Fraction(3, 2)


def test_joint_alignment_objective_differs_from_output_marginalization():
    source = _unary((((0,), 0), ((1,), -0.125), ((1,), -0.125)))
    factors = (StringFactor(("x",), source),)
    graph = StringFactorGraph((StringVariable("x", _TAPE),), factors)
    choice = dual_alignment_map(graph, **_SETTINGS)
    assert choice.assignment == (("x", (0,)),)
    assert source.weight(((1,),)).item() > source.weight(((0,),)).item()
    _oracle_certificate(choice, factors)


def test_separate_local_decodes_avoid_materializing_full_implicit_intersection():
    left = _unary((((0, 1), float(i) / 16) for i in range(12)))
    right = _unary((((0, 1), -float(i) / 16) for i in range(12)))
    factors = (StringFactor(("x",), left), StringFactor(("x",), right))
    graph = StringFactorGraph((StringVariable("x", _TAPE),), factors)
    result = dual_alignment_map(graph, **_SETTINGS)
    explicit = intersect(left, right, max_states=1000, max_work=1000)
    assert result.max_local_states * 10 < explicit.topology.n_states
    assert result.local_arc_examinations == 48
    assert result.verification_arc_examinations == 100
    assert result.max_verification_states == result.max_local_states
    assert result.score == max(path[-1] for path in _all_paths(explicit))
    _oracle_certificate(result, factors)


class _LiveScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.tensor([0.0, 0.125, -0.125], dtype=torch.float64)
        )
        self.calls = 0

    def forward(self):
        self.calls += 1
        return self.weights


def test_observed_tapes_constrain_each_local_trace_and_shared_source_reads_once():
    scores = _LiveScores()
    topology = RelationTopology(
        (_TAPE, _TAPE),
        1,
        2,
        (Arc(0, 1, (0, 0), 2), Arc(0, 1, (0, 1), 1), Arc(0, 1, (1, 0), 2)),
        (Boundary(0, 0),),
        (Boundary(1, 0),),
        ("unit", "better", "worse"),
    )
    relation = RationalRelation(topology, scores, algebra=LOG_PROB)
    factors = (StringFactor(("x", "y"), relation), StringFactor(("x", "z"), relation))
    graph = StringFactorGraph(
        (
            StringVariable("x", _TAPE),
            StringVariable("y", _TAPE, (0,)),
            StringVariable("z", _TAPE, (1,)),
        ),
        factors,
    )
    result = dual_alignment_map(graph, **_SETTINGS)
    assert result.assignment == (("x", (0,)), ("y", (0,)), ("z", (1,)))
    assert result.score == 0
    assert scores.calls == 1


def test_productive_cycles_and_unsupported_algebra_are_refused():
    unary = _unary((((0,), 0),))
    factor = StringFactor(("x",), unary)
    graph = StringFactorGraph((StringVariable("x", _TAPE),), (factor,))
    with pytest.raises(RelationLimitError, match="max_work"):
        dual_alignment_map(graph, **(_SETTINGS | dict(max_work=0)))
    old = unary.topology
    cyclic_topology = replace(old, arcs=old.arcs + (Arc(1, 1, (None,), 0),))
    cyclic = RationalRelation(cyclic_topology, unary.module(), algebra=LOG_PROB)
    graph = StringFactorGraph(
        (StringVariable("x", _TAPE),), (StringFactor(("x",), cyclic),)
    )
    with pytest.raises(ValueError, match="acyclic positive"):
        dual_alignment_map(graph, **_SETTINGS)
    markov = RationalRelation(old, torch.ones(2), algebra=MARKOV)
    graph = StringFactorGraph(
        (StringVariable("x", _TAPE),), (StringFactor(("x",), markov),)
    )
    with pytest.raises(TypeError, match="requires LogProb"):
        dual_alignment_map(graph, **_SETTINGS)


def test_overlapping_suffix_features_score_signed_counts_and_boundary_markers():
    coefficients = {
        (0,): Fraction(1, 2),
        (1, 0): Fraction(-3, 4),
        (0, 1, 0): Fraction(7, 8),
        (-1, 0): Fraction(2, 3),
        (0, -2): Fraction(-3, 5),
        (-1, -2): Fraction(4, 5),
    }
    scorer = _NgramScorer(coefficients)
    for width in range(6):
        for word in product(range(2), repeat=width):
            padded = (-1, *word, -2)
            expected = sum(
                (
                    coefficient
                    * sum(padded[i : i + len(gram)] == gram for i in range(len(padded)))
                    for gram, coefficient in coefficients.items()
                ),
                Fraction(0),
            )
            state = 0
            observed = Fraction(0)
            for symbol in padded:
                state, emission = scorer.step(state, symbol)
                observed += emission
            assert observed == expected


def test_scaling_protocol_compares_both_alignment_families_and_reports_work():
    for mode, score, calls, features in (
        ("agree", "3/2", 2, 0),
        ("adaptive", "1", 6, 1),
    ):
        for width in (2, 5):
            row = scaling.compare(scaling.fixture(mode, width))
            assert row["status"] == "optimal" and row["certificate"] is True
            assert row["primal"] == row["upper_bound"] == score
            assert row["enumerated_compatible_pairs"] == 2 * width**2
            assert row["implicit_largest_decode_configurations"] == 6 * width + 1
            assert row["implicit_largest_verification_configurations"] == 6 * width + 1
            assert row["implicit_verification_arc_examinations"] == 24 * width + 6
            assert row["implicit_local_decode_calls"] == calls
            assert row["implicit_peak_active_features"] == features
            assert row["explicit_peak_product_states"] == 10 * width**2 + 2 * width + 1


def test_scaling_protocol_checks_live_updates_and_uncertified_controls():
    checks = scaling.controls()
    assert checks["after_update"]["status"] == "optimal"
    assert checks["after_update"]["primal"] == "2"
    assert checks["unigram_only"]["certificate"] is False
    assert checks["zero_weight_right_support"]["status"] == "infeasible"
