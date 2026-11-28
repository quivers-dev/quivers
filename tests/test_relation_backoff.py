"""Independent fully expanded oracle for sparse fixed-word backoff."""

from dataclasses import replace
from fractions import Fraction
from itertools import product

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    FailureArc,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    backoff_query,
)


def _fixture():
    tape = StringSet(alphabet=FinSet(name="LargeAlphabet", cardinality=1 << 30))
    names = ("unit", "a", "b", "c", "fallback0", "fallback1", "zero")
    topology = RelationTopology(
        (tape,),
        0,
        3,
        (
            Arc(0, 1, (0,), 1),
            Arc(1, 1, (0,), 6),
            Arc(1, 0, (1,), 2),
            Arc(2, 1, (2,), 3),
            Arc(2, 0, (0,), 0),
        ),
        (Boundary(0, 0),),
        (Boundary(0, 0), Boundary(1, 0)),
        names,
    )
    weights = nn.Parameter(
        torch.tensor(
            [1, 3 / 4, 5 / 8, 1 / 8, 1 / 2, 1 / 4, 0],
            dtype=torch.float64,
        )
    )
    relation = RationalRelation(topology, weights, algebra=MARKOV)
    return relation, weights, (FailureArc(0, 1, 4), FailureArc(1, 2, 5))


def test_sparse_backoff_matches_an_independent_expanded_table():
    relation, weights, failures = _fixture()
    expanded = {
        (0, 0): (1, Fraction(3, 4)),
        (0, 1): (0, Fraction(5, 16)),
        (0, 2): (1, Fraction(1, 64)),
        (1, 0): (1, Fraction(0)),
        (1, 1): (0, Fraction(5, 8)),
        (1, 2): (1, Fraction(1, 32)),
        (2, 0): (0, Fraction(1)),
        (2, 2): (1, Fraction(1, 8)),
    }

    def oracle(word):
        state, score = 0, Fraction(1)
        for symbol in word:
            if (state, symbol) not in expanded:
                return 0
            state, weight = expanded[state, symbol]
            score *= weight
        return score if state in (0, 1) else 0

    for length in range(4):
        for word in product((0, 1, 2, (1 << 30) - 1), repeat=length):
            actual = backoff_query(relation, failures, word, max_states=16, max_work=16)
            assert actual.weight().item() == oracle(word)
    plan = backoff_query(relation, failures, (2,), max_states=4, max_work=3)
    (gradient,) = torch.autograd.grad(plan.weight(), weights)
    assert gradient.tolist() == [1 / 32, 0, 0, 1 / 8, 1 / 32, 1 / 16, 0]
    with torch.no_grad():
        weights[5] = 0.5
    assert plan.weight().item() == Fraction(1, 32)


def test_backoff_requires_deterministic_labels_acyclic_fallback_and_actual_work():
    relation, _, failures = _fixture()
    for bound, value in (("max_states", 3), ("max_work", 2)):
        limits = {"max_states": 4, "max_work": 3}
        limits[bound] = value
        with pytest.raises(RelationLimitError, match=bound):
            backoff_query(relation, failures, (2,), **limits)
    with pytest.raises(ValueError, match="acyclic"):
        backoff_query(
            relation,
            failures + (FailureArc(2, 0, 0),),
            (2,),
            max_states=16,
            max_work=16,
        )
    with pytest.raises(ValueError, match="at most one failure arc"):
        backoff_query(
            relation,
            failures + (FailureArc(0, 2, 0),),
            (2,),
            max_states=16,
            max_work=16,
        )
    for added in (Arc(0, 1, (0,), 0), Arc(0, 1, (None,), 0)):
        topology = replace(relation.topology, arcs=relation.topology.arcs + (added,))
        nondeterministic = RationalRelation(topology, relation.module(), algebra=MARKOV)
        with pytest.raises(ValueError, match="epsilon-free deterministic"):
            backoff_query(nondeterministic, failures, (2,), max_states=16, max_work=16)
