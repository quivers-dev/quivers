"""Explicit lexicographic ranking and original-occurrence weight oracles."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from itertools import product
from random import Random

import pytest
import torch

from quivers import FinSet, StringSet
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    ranked_output_candidates,
)


def _brute_force(
    relation: RationalRelation,
    word: tuple[int, ...],
    violations: tuple[tuple[int, ...], ...],
    *,
    caps: tuple[int, ...] | None = None,
) -> dict[tuple[int, ...], Fraction]:
    """Enumerate complete source paths before comparing any ranking."""
    topology = relation.topology
    weights = list(map(Fraction, relation.module()().tolist()))
    final = {mark.state: weights[mark.weight] for mark in topology.final}
    candidates = []
    zeros = (0,) * len(violations[0]) if violations else ()
    pending: list[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], Fraction]
    ] = [(mark.state, (), (), zeros, weights[mark.weight]) for mark in topology.initial]
    while pending:
        state, read, emitted, rank, weight = pending.pop()
        if state in final and read == word:
            profile = (
                rank
                if caps is None
                else tuple(min(cap, n) for cap, n in zip(caps, rank, strict=True))
            )
            candidates.append((profile, emitted, weight * final[state]))
        for index, arc in enumerate(topology.arcs):
            if arc.source == state:
                incoming, outgoing = arc.labels
                pending.append(
                    (
                        arc.target,
                        read + (() if incoming is None else (incoming,)),
                        emitted + (() if outgoing is None else (outgoing,)),
                        tuple(
                            a + b for a, b in zip(rank, violations[index], strict=True)
                        ),
                        weight * weights[arc.weight],
                    )
                )
    optimum = min((rank for rank, _, _ in candidates), default=None)
    expected: defaultdict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    for rank, emitted, weight in candidates:
        if rank == optimum:
            expected[emitted] += weight
    return expected


def test_random_lattices_match_an_independent_complete_path_oracle():
    rng = Random(83301)
    tape = StringSet(alphabet=FinSet(name="ab", cardinality=2))
    weights = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    for _ in range(32):
        arcs = tuple(
            Arc(
                source,
                rng.randrange(source + 1, 5),
                (rng.choice((0, 1, None)), rng.choice((0, 1, None))),
                rng.randrange(4),
            )
            for source in range(4)
            for _ in range(3)
        )
        topology = RelationTopology(
            (tape, tape),
            1,
            5,
            arcs,
            (Boundary(0, rng.randrange(4)), Boundary(1, 3)),
            tuple(Boundary(state, rng.randrange(4)) for state in range(5)),
            ("zero", "quarter", "half", "one"),
        )
        relation = RationalRelation(topology, weights, algebra=MARKOV)
        violations = tuple((rng.randrange(2), rng.randrange(5)) for _ in arcs)
        for word in [(), (0,), (1,), (0, 1)]:
            expected = _brute_force(relation, word, violations)
            restricted = ranked_output_candidates(
                relation,
                word,
                violations,
                max_states=32,
                max_work=100,
            )
            for length in range(5):
                for output in product(range(2), repeat=length):
                    assert restricted.weight((output,)).item() == expected[output]


def _cyclic_generator():
    tape = StringSet(alphabet=FinSet(name="ab", cardinality=2))
    topology = RelationTopology(
        (tape, tape),
        1,
        4,
        (
            Arc(0, 1, (0, 0), 0),
            Arc(0, 1, (0, 0), 0),
            Arc(0, 2, (0, 0), 0),
            Arc(0, 3, (0, 1), 1),
            Arc(1, 1, (None, None), 2),
            Arc(2, 2, (None, None), 2),
        ),
        (Boundary(0, 3),),
        (Boundary(1, 3), Boundary(2, 3), Boundary(3, 3)),
        ("ties", "loses", "loop", "one"),
    )
    weights = torch.nn.Parameter(
        torch.tensor([0.25, 0.75, 0.25, 1.0], dtype=torch.float64)
    )
    return RationalRelation(topology, weights, algebra=MARKOV), weights


def test_only_zero_rank_cycles_remain_and_tie_weights_stay_live():
    relation, weights = _cyclic_generator()
    violations = ((0, 5**20), (0, 5**20 + 1), (0, 5**20), (1, 0), (0, 0), (0, 1))
    ranked = ranked_output_candidates(
        relation,
        (0,),
        violations,
        max_states=8,
        max_work=8,
    )
    assert ranked.module() is relation.module()
    assert len(ranked.topology.arcs) == 3
    good = ranked.query(((0,),), max_closure_size=1)
    assert good.weight().item() == pytest.approx(float(Fraction(7, 12)))
    assert ranked.weight(((1,),)).item() == 0
    gradient = torch.autograd.grad(good.weight(), weights)[0]
    assert gradient.tolist()[:3] == pytest.approx([7 / 3, 0, 4 / 9])
    with torch.no_grad():
        weights[0] = 0
    assert good.weight().item() == 0
    assert ranked.weight(((1,),)).item() == 0


@pytest.mark.parametrize(
    "bad", [(), ((True,),) * 6, ((-1,),) * 6, ((), (0,), (), (), (), ())]
)
def test_bad_vectors_are_refused(bad):
    relation, _ = _cyclic_generator()
    with pytest.raises(ValueError, match="vector per arc"):
        ranked_output_candidates(relation, (0,), bad, max_states=8, max_work=8)


def test_construction_budgets_and_tape_orientation_are_checked():
    relation, _ = _cyclic_generator()
    violations = ((0,),) * 6
    with pytest.raises(RelationLimitError, match="max_work"):
        ranked_output_candidates(relation, (0,), violations, max_states=8, max_work=0)
    with pytest.raises(RelationLimitError, match="max_states"):
        ranked_output_candidates(relation, (0,), violations, max_states=1, max_work=8)
    same_side = RationalRelation(
        replace(relation.topology, input_tapes=2), relation.module(), algebra=MARKOV
    )
    with pytest.raises(TypeError, match="one input tape"):
        ranked_output_candidates(same_side, (0,), violations, max_states=8, max_work=8)
