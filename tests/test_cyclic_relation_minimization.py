"""Check cyclic minimization by exact reachable linear spans."""

from collections import deque
from dataclasses import replace
from fractions import Fraction
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
    minimize_static_acceptor_with_work,
)


def _machine(size, arcs, initial, final, values, *, alphabet_size=4):
    tape = StringSet(alphabet=FinSet(name="Alphabet", cardinality=alphabet_size))
    topology = RelationTopology(
        (tape,),
        0,
        size,
        tuple(
            Arc(source, target, (symbol,), slot)
            for source, target, symbol, slot in arcs
        ),
        tuple(Boundary(state, slot) for state, slot in initial),
        tuple(Boundary(state, slot) for state, slot in final),
        tuple(f"w{i}" for i in range(len(values))),
    )
    return RationalRelation(
        topology, torch.tensor(values, dtype=torch.float64), algebra=MARKOV
    )


def _linear_form(relation):
    topology = relation.topology
    weights = list(map(Fraction, relation.module()().tolist()))
    initial = [Fraction(0)] * topology.n_states
    final = [Fraction(0)] * topology.n_states
    transitions = {}
    for mark in topology.initial:
        initial[mark.state] = weights[mark.weight]
    for mark in topology.final:
        final[mark.state] = weights[mark.weight]
    for arc in topology.arcs:
        transitions[arc.source, arc.labels[0]] = arc.target, weights[arc.weight]
    return initial, final, transitions


def _span_agrees(initial, final, transitions):
    """Test the whole weighted series by closure of an exact row basis."""
    symbols = {symbol for _, symbol in transitions}
    basis = {}
    pending = deque([initial])
    while pending:
        row = pending.popleft()
        for pivot, generator in sorted(basis.items()):
            multiplier = row[pivot]
            row = [x - multiplier * y for x, y in zip(row, generator, strict=True)]
        pivot = next((i for i, x in enumerate(row) if x), None)
        if pivot is None:
            continue
        if sum(x * y for x, y in zip(row, final, strict=True)):
            return False
        row = [x / row[pivot] for x in row]
        basis[pivot] = row
        for symbol in symbols:
            successor = [Fraction(0)] * len(row)
            for state, coefficient in enumerate(row):
                if (state, symbol) in transitions:
                    target, weight = transitions[state, symbol]
                    successor[target] += coefficient * weight
            pending.append(successor)
    return True


def _same_series(left, right):
    i_left, f_left, t_left = _linear_form(left)
    i_right, f_right, t_right = _linear_form(right)
    offset = len(i_left)
    return _span_agrees(
        i_left + [-weight for weight in i_right],
        f_left + f_right,
        t_left
        | {
            (state + offset, label): (target + offset, weight)
            for (state, label), (target, weight) in t_right.items()
        },
    )


def _accepted_witness(state, final, transitions):
    """Find any positive accepting suffix by graph search in the original."""
    pending = [(state, ())]
    seen = {state}
    while pending:
        state, word = pending.pop()
        if final[state]:
            return word
        for (source, symbol), (target, weight) in transitions.items():
            if source == state and weight and target not in seen:
                seen.add(target)
                pending.append((target, (*word, symbol)))
    return None


def _suffix_weight(state, word, final, transitions):
    product = Fraction(1)
    for symbol in word:
        if (state, symbol) not in transitions:
            return Fraction(0)
        state, weight = transitions[state, symbol]
        product *= weight
    return product * final[state]


def _minimum_width(relation):
    """Compare original reachable states up to a separately witnessed ratio."""
    initial, final, transitions = _linear_form(relation)
    reachable = {i for i, coefficient in enumerate(initial) if coefficient}
    pending = list(reachable)
    while pending:
        state = pending.pop()
        for (source, _), (target, weight) in transitions.items():
            if source == state and weight and target not in reachable:
                reachable.add(target)
                pending.append(target)
    representatives = []
    for state in sorted(reachable):
        witness = _accepted_witness(state, final, transitions)
        if witness is None:
            continue
        own_weight = _suffix_weight(state, witness, final, transitions)
        for other in representatives:
            other_weight = _suffix_weight(other, witness, final, transitions)
            if other_weight:
                start = [Fraction(0)] * len(initial)
                start[state] = other_weight
                start[other] = -own_weight
                if _span_agrees(start, final, transitions):
                    break
        else:
            representatives.append(state)
    return len(representatives)


def test_proportional_cyclic_residuals_merge_and_preserve_all_finite_words():
    source = _machine(
        5,
        [
            (0, 1, 0, 0),
            (0, 2, 1, 1),
            (1, 2, 2, 3),
            (2, 1, 2, 4),
            (0, 3, 2, 4),
            (1, 4, 3, 5),
            (4, 4, 2, 4),
        ],
        [(0, 4)],
        [(1, 2), (2, 1), (4, 2)],
        [0.125, 0.25, 0.5, 4.0, 1.0, 0.0],
    )
    reduced, work = minimize_static_acceptor_with_work(source)
    assert work.active_states == 3
    assert work.active_arcs == 4
    assert reduced.topology.n_states == _minimum_width(source) == 2
    assert _same_series(source, reduced)
    # The merged loop has weight two, so the length-unbounded sum diverges.
    assert reduced.weight(((0,),)).item() == Fraction(1, 16)
    assert reduced.weight(((1,) + (2,) * 37,)).item() == 2**33
    weights = reduced.module()().clone()
    loop = next(arc for arc in reduced.topology.arcs if arc.source == arc.target)
    weights[loop.weight] *= 0.5
    wrong = RationalRelation(reduced.topology, weights, algebra=MARKOV)
    assert not _same_series(source, wrong)
    source.module()()[3] = 1.0
    assert not _same_series(source, reduced)


def test_sparse_cyclic_families_match_exact_span_and_minimal_width():
    rng = Random(7031)
    mask = [0.0, 0.125, 0.5, 1.0, 2.0]
    for size in range(1, 8):
        for _ in range(24):
            source = _machine(
                size,
                [
                    (state, rng.randrange(size), symbol, rng.randrange(len(mask)))
                    for state in range(size)
                    for symbol in range(3)
                    if rng.random() < 0.6
                ],
                [(0, rng.randrange(len(mask)))],
                [(state, rng.randrange(len(mask))) for state in range(size)],
                mask,
            )
            reduced, work = minimize_static_acceptor_with_work(source)
            assert _same_series(source, reduced)
            assert reduced.topology.n_states == _minimum_width(source)
            charges = work.active_states.bit_length()
            assert work.splitter_states <= work.active_states * charges
            assert work.predecessor_examinations <= work.active_arcs * charges


def test_shortest_suffix_ties_use_symbol_order_rather_than_arc_order():
    source = _machine(
        2,
        [(0, 1, 3, 0), (0, 1, 0, 1), (1, 1, 1, 1)],
        [(0, 2)],
        [(1, 2)],
        [0.75, 0.5, 1.0],
    )
    reduced, _ = minimize_static_acceptor_with_work(source)
    assert _same_series(source, reduced)
    topology = reduced.topology
    weights = reduced.module()()
    initial = topology.initial[0]
    assert weights[initial.weight].item() == 0.5
    outgoing = {
        arc.labels[0]: weights[arc.weight].item()
        for arc in topology.arcs
        if arc.source == initial.state
    }
    assert outgoing == {0: 1.0, 3: 1.5}


def test_smaller_half_counts_and_limits_on_a_partial_sparse_ladder():
    size = 1024
    source = _machine(
        size,
        [(state, state - 1, 0, 0) for state in range(1, size)]
        + [(state, state, 1, 0) for state in range(size)],
        [(size - 1, 0)],
        [(0, 0)],
        [1.0],
        alphabet_size=1 << 30,
    )
    reduced, work = minimize_static_acceptor_with_work(source, max_states=size)
    assert reduced.topology.n_states == size
    assert work.active_arcs == len(source.topology.arcs)
    assert work.splitter_states <= 3 * size
    assert work.predecessor_examinations <= 5 * size
    charge = work.input_records + work.splitter_states + work.predecessor_examinations
    assert minimize_static_acceptor_with_work(source, max_work=charge)[1] == work
    with pytest.raises(RelationLimitError, match="max_work"):
        minimize_static_acceptor_with_work(source, max_work=charge - 1)
    with pytest.raises(RelationLimitError, match="max_work"):
        minimize_static_acceptor_with_work(source, max_work=work.input_records - 1)
    with pytest.raises(RelationLimitError, match="max_states"):
        minimize_static_acceptor_with_work(source, max_states=size - 1)


def test_cyclic_minimizer_refuses_unrepresentable_gauge_and_unsupported_inputs():
    source = _machine(
        2,
        [(0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 0), (1, 1, 1, 0)],
        [(0, 0)],
        [(0, 1), (1, 0)],
        [1.0, 0.75],
    )
    with pytest.raises(ValueError, match="exactly representable"):
        minimize_static_acceptor_with_work(source)

    for extra in (Arc(0, 1, (0,), 0), Arc(0, 1, (None,), 0)):
        topology = replace(source.topology, arcs=(*source.topology.arcs, extra))
        with pytest.raises(ValueError, match="epsilon-free deterministic"):
            minimize_static_acceptor_with_work(
                RationalRelation(topology, source.module(), algebra=MARKOV)
            )
    with pytest.raises(TypeError, match="nontrainable"):
        minimize_static_acceptor_with_work(
            RationalRelation(
                source.topology, torch.nn.Parameter(source.module()()), algebra=MARKOV
            )
        )
