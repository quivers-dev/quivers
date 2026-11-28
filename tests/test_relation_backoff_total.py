"""Full-word and work-budget oracles for ring-free backoff aggregation."""

from fractions import Fraction
from itertools import product
from random import Random

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import COUNTING, LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    FailureArc,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    backoff_total_weight,
)


def _explicit_word_sum(
    ordinary: dict[tuple[int, int], tuple[int, int]],
    fallback: dict[int, tuple[int, int]],
    initial: tuple[int, int],
    final: dict[int, int],
    values: tuple[Fraction, ...],
    alphabet: int,
    depth: int,
) -> Fraction:
    total = Fraction(0)
    for length in range(depth + 1):
        for word in product(range(alphabet), repeat=length):
            state, slot = initial
            mass = values[slot]
            for symbol in word:
                while (state, symbol) not in ordinary and state in fallback:
                    state, slot = fallback[state]
                    mass *= values[slot]
                target = ordinary.get((state, symbol))
                if target is None:
                    break
                state, slot = target
                mass *= values[slot]
            else:
                if state in final:
                    total += mass * values[final[state]]
    return total


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_persistent_defaults_match_explicit_word_scores_and_original_gradients(algebra):
    tape = StringSet(alphabet=FinSet(name="Three", cardinality=3))
    explicit = {(0, 0): (2, 0), (1, 0): (2, 2), (1, 1): (2, 2)}
    failures = (FailureArc(0, 1, 1),)
    initial = (0, 3)
    final = {0: 3, 2: 3}
    topology = RelationTopology(
        (tape,),
        0,
        3,
        tuple(Arc(q, r, (a,), slot) for (q, a), (r, slot) in explicit.items()),
        (Boundary(*initial),),
        tuple(Boundary(q, slot) for q, slot in final.items()),
        ("zero", "fallback", "shared", "unit"),
    )
    values = tuple(map(Fraction, (0, 0.5, 0.25, 1)))
    raw = torch.tensor([float(value) for value in values], dtype=torch.float64)
    weights = nn.Parameter(raw if algebra is MARKOV else raw.log())
    relation = RationalRelation(topology, weights, algebra=algebra)
    exact = _explicit_word_sum(explicit, {0: (1, 1)}, initial, final, values, 3, 2)
    total = backoff_total_weight(relation, failures, max_states=12, max_work=80)
    assert (total if algebra is MARKOV else total.exp()).item() == pytest.approx(
        float(exact)
    )
    (gradient,) = torch.autograd.grad(total, weights)
    expected = (1, 0.25, 0.5) if algebra is MARKOV else (0, 1 / 9, 1 / 9)
    assert gradient.tolist()[:3] == pytest.approx(expected)
    with torch.no_grad():
        weights[2] = 0.5 if algebra is MARKOV else torch.log(weights.new_tensor(0.5))
    updated = backoff_total_weight(relation, failures, max_states=12, max_work=80)
    assert (updated if algebra is MARKOV else updated.exp()).item() == pytest.approx(
        1.25
    )


def test_small_dags_match_exhaustive_rational_word_sums():
    rng = Random(891)
    n, alphabet = 5, 3
    tape = StringSet(alphabet=FinSet(name="Three", cardinality=alphabet))
    values = (Fraction(0), Fraction(1, 2), Fraction(1, 4), Fraction(1))
    for _ in range(120):
        ordinary = {
            (q, a): (rng.randrange(q + 1, n), rng.randrange(len(values)))
            for q in range(n - 1)
            for a in range(alphabet)
            if rng.random() < 0.55
        }
        fallback = {
            q: (rng.randrange(q + 1, n), rng.randrange(len(values)))
            for q in range(n - 1)
            if rng.random() < 0.6
        }
        initial = (0, rng.randrange(len(values)))
        final = {q: rng.randrange(len(values)) for q in range(n) if rng.random() < 0.55}
        topology = RelationTopology(
            (tape,),
            0,
            n,
            tuple(Arc(q, r, (a,), slot) for (q, a), (r, slot) in ordinary.items()),
            (Boundary(*initial),),
            tuple(Boundary(q, slot) for q, slot in final.items()),
            ("zero", "half", "quarter", "one"),
        )
        relation = RationalRelation(
            topology,
            torch.tensor([float(value) for value in values], dtype=torch.float64),
            algebra=MARKOV,
        )
        total = backoff_total_weight(
            relation,
            tuple(FailureArc(q, r, slot) for q, (r, slot) in fallback.items()),
            max_states=n,
            max_work=150,
        )
        expected = _explicit_word_sum(
            ordinary, fallback, initial, final, values, alphabet, n - 1
        )
        assert Fraction(total.item()) == expected


def test_aggregation_does_not_expand_all_effective_transitions():
    branches, symbols = 64, 512
    tape = StringSet(alphabet=FinSet(name="Sparse", cardinality=1 << 30))
    terminal = branches + 1
    topology = RelationTopology(
        (tape,),
        0,
        branches + 2,
        (
            *(Arc(0, state, (state - 1,), 0) for state in range(1, branches + 1)),
            *(
                Arc(branches, terminal, (branches + symbol,), 0)
                for symbol in range(symbols)
            ),
            *(
                Arc(state, terminal, (branches + state,), 0)
                for state in range(1, branches)
            ),
        ),
        (Boundary(0, 0),),
        (Boundary(terminal, 0),),
        ("one",),
    )
    relation = RationalRelation(
        topology, torch.ones(1, dtype=torch.int64), algebra=COUNTING
    )
    failures = tuple(FailureArc(q, q + 1, 0) for q in range(1, branches))
    # The expanded effective graph has 32768 second-symbol arcs alone.
    value = backoff_total_weight(relation, failures, max_states=80, max_work=10000)
    assert value.item() == branches * symbols
    with pytest.raises(RelationLimitError, match="max_work"):
        backoff_total_weight(relation, failures, max_states=80, max_work=100)


def test_cycles_in_the_combined_backoff_graph_are_refused():
    tape = StringSet(alphabet=FinSet(name="Two", cardinality=2))
    topology = RelationTopology(
        (tape,),
        0,
        2,
        (Arc(1, 0, (0,), 0),),
        (Boundary(0, 0),),
        (Boundary(1, 0),),
        ("unit",),
    )
    model = RationalRelation(topology, torch.ones(1), algebra=MARKOV)
    with pytest.raises(ValueError, match="acyclic ordinary and failure"):
        backoff_total_weight(model, (FailureArc(0, 1, 0),), max_states=8, max_work=20)
