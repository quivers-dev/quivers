"""Exact all-input ranked OT oracles for finite profiles and epsilon paths."""

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
    compile_ranked_ot,
)
from tests.test_relation_ranked import _brute_force, _cyclic_generator


def test_uniform_compile_matches_independent_full_path_ranks_for_every_small_input():
    rng = Random(4953)
    tape = StringSet(alphabet=FinSet(name="ab", cardinality=2))
    values = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    words = [word for n in range(4) for word in product(range(2), repeat=n)]
    for _ in range(16):
        arcs = tuple(
            Arc(
                q,
                rng.randrange(q + 1, 4),
                (rng.choice((0, 1, None)), rng.choice((0, 1, None))),
                rng.randrange(4),
            )
            for q in range(3)
            for _ in range(4)
        )
        topology = RelationTopology(
            (tape, tape),
            1,
            4,
            arcs,
            (Boundary(0, rng.randrange(4)), Boundary(1, 3)),
            tuple(Boundary(q, rng.randrange(4)) for q in range(4)),
            ("zero", "quarter", "half", "one"),
        )
        relation = RationalRelation(topology, values, algebra=MARKOV)
        violations = tuple((rng.randrange(3), rng.randrange(4)) for _ in arcs)
        caps = (1, rng.randrange(3))
        compiled = compile_ranked_ot(
            relation, violations, caps, max_states=2000, max_work=15000
        )
        for given in words:
            oracle = _brute_force(relation, given, violations, caps=caps)
            for predicted in words:
                assert compiled.weight((given, predicted)).item() == oracle[predicted]


def test_saturation_can_reverse_prefix_order_so_all_competitor_ranks_matter():
    tape = StringSet(alphabet=FinSet(name="huge_ab", cardinality=1 << 25))
    # Prefix (0,) favors the first path. The last arc raises the first
    # count to its cap, so the second coordinate decides the longer input.
    arcs = (
        Arc(0, 1, (0, 0), 0),
        Arc(0, 1, (0, 1), 0),
        Arc(1, 2, (1, None), 0),
    )
    topology = RelationTopology(
        (tape, tape),
        1,
        1 << 29,
        arcs,
        (Boundary(0, 0),),
        (Boundary(1, 0), Boundary(2, 0)),
        ("one",),
    )
    relation = RationalRelation(topology, torch.ones(1), algebra=MARKOV)
    result = compile_ranked_ot(
        relation,
        ((0, 1), (1, 0), (1, 0)),
        (1, 1),
        max_states=16,
        max_work=200,
    )
    assert result.weight(((0,), (0,))).item() == 1
    assert result.weight(((0,), (1,))).item() == 0
    assert result.weight(((0, 1), (0,))).item() == 0
    assert result.weight(((0, 1), (1,))).item() == 1
    assert result.weight(((1 << 24,), (1,))).item() == 0


def test_capped_epsilon_cycles_keep_exact_path_once_and_original_derivatives():
    source, weights = _cyclic_generator()
    # The two parallel arcs to state 1 have different profiles. The
    # winner can repeat its epsilon loop freely. The other winning
    # branch at state 2 cannot take any of its positive-rank loops.
    profiles = ((0, 1), (0, 3), (0, 1), (1, 0), (0, 0), (0, 2))
    result = compile_ranked_ot(source, profiles, (1, 3), max_states=40, max_work=300)
    assert result.module() is source.module()
    logit = result.query(((0,), (0,)), max_closure_size=1).weight()
    assert logit.item() == pytest.approx(float(Fraction(7, 12)))
    assert result.weight(((0,), (1,))).item() == 0
    (grad,) = torch.autograd.grad(logit, weights)
    assert grad.tolist()[:3] == pytest.approx([7 / 3, 0, 4 / 9])


def test_limits_caps_and_empty_candidate_paths_are_explicit():
    relation, _ = _cyclic_generator()
    profiles = ((0,),) * 6
    with pytest.raises(RelationLimitError, match="max_work"):
        compile_ranked_ot(relation, profiles, (2,), max_states=8, max_work=0)
    with pytest.raises(RelationLimitError, match="max_states"):
        compile_ranked_ot(relation, profiles, (2,), max_states=1, max_work=100)
    with pytest.raises(ValueError, match="integer cap"):
        compile_ranked_ot(relation, profiles, (False,), max_states=8, max_work=100)
    tape = relation.topology.tapes[0]
    empty_words = RelationTopology(
        (tape, tape), 1, 1, (), (Boundary(0, 0),), (Boundary(0, 0),), ("start_stop",)
    )
    unit = RationalRelation(empty_words, torch.tensor([0.5]), algebra=MARKOV)
    got = compile_ranked_ot(unit, (), (1, 2), max_states=2, max_work=4)
    assert got.weight(((), ())).item() == 0.25
