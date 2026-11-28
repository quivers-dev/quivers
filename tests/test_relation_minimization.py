"""Independent finite-language and residual-series oracles for minimization."""

from dataclasses import replace
from fractions import Fraction
from itertools import product
from random import Random
from struct import pack

import pytest
import torch

from quivers import FinSet, StringSet
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationTopology,
    minimize_static_acceptor,
)


def _machine(first=0.25, second=0.5, *, trainable=False):
    alphabet = StringSet(alphabet=FinSet(name="ABC", cardinality=3))
    topology = RelationTopology(
        (alphabet,),
        0,
        1 << 30,
        (
            Arc(0, 1, (0,), 0),
            Arc(0, 2, (1,), 1),
            Arc(1, 3, (2,), 2),
            Arc(2, 3, (2,), 3),
        ),
        (Boundary(0, 4),),
        (Boundary(1, 2), Boundary(2, 3), Boundary(3, 4)),
        ("a", "b", "left", "right", "unit"),
    )
    values = torch.tensor([0.25, 0.5, first, second, 1.0], dtype=torch.float64)
    weights = torch.nn.Parameter(values) if trainable else values
    return RationalRelation(topology, weights, algebra=MARKOV), weights


def _language(relation):
    """Enumerate every nonzero accepted word in the acyclic test topology."""
    topology = relation.topology
    values = list(map(Fraction, relation.module()().tolist()))
    accepted = {}
    pending = [
        (mark.state, (), values[mark.weight])
        for mark in topology.initial
        if values[mark.weight]
    ]
    while pending:
        state, word, weight = pending.pop()
        for mark in topology.final:
            if mark.state == state and values[mark.weight]:
                accepted[word] = weight * values[mark.weight]
        for arc in topology.arcs:
            if arc.source == state and values[arc.weight]:
                pending.append(
                    (arc.target, word + arc.labels, weight * values[arc.weight])
                )
    return accepted


def _minimum_state_count(language):
    # A single deterministic state can serve two positive prefixes exactly
    # when their restrictions to suffixes differ by a multiplicative scale.
    # Normalize independently by total mass and compare explicit word maps.
    prefixes = {word[:i] for word in language for i in range(len(word) + 1)}
    classes = set()
    for prefix in prefixes:
        residual = {
            word[len(prefix) :]: value
            for word, value in language.items()
            if word[: len(prefix)] == prefix
        }
        total = sum(residual.values())
        classes.add(
            frozenset((word, weight / total) for word, weight in residual.items())
        )
    return len(classes)


def test_proportional_residuals_reduce_even_with_nondyadic_ratios():
    source, weights = _machine(first=0.375, second=0.625)
    reduced = minimize_static_acceptor(source)
    assert reduced.topology.n_states == 3
    assert reduced.topology.initial == (Boundary(2, 0),)
    assert reduced.topology.final == (Boundary(0, 1), Boundary(1, 1))
    assert reduced.topology.arcs == (
        Arc(1, 0, (2,), 1),
        Arc(2, 1, (0,), 2),
        Arc(2, 1, (1,), 3),
    )
    assert bytes(reduced.module()().view(torch.uint8).tolist()) == pack(
        "=4d", 1 / 32, 1, 3, 10
    )
    expected = {
        (0,): Fraction(3, 32),
        (1,): Fraction(5, 16),
        (0, 2): Fraction(3, 32),
        (1, 2): Fraction(5, 16),
    }
    for n in range(3):
        for word in product(range(3), repeat=n):
            assert reduced.weight((word,)).item() == expected.get(word, 0)
    weights[2] = 0.5
    assert source.weight(((0,),)).item() == Fraction(1, 8)
    assert reduced.weight(((0,),)).item() == Fraction(3, 32)


def test_random_dags_preserve_explicit_rational_series_and_have_minimum_width():
    rng = Random(31509)
    alphabet = StringSet(alphabet=FinSet(name="ab", cardinality=2))
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    for size in range(1, 8):
        for _ in range(16):
            topology = RelationTopology(
                (alphabet,),
                0,
                size,
                tuple(
                    Arc(
                        state,
                        rng.randrange(state + 1, size),
                        (label,),
                        rng.randrange(5),
                    )
                    for state in range(size - 1)
                    for label in range(2)
                    if rng.random() < 0.8
                ),
                (Boundary(0, rng.randrange(5)),),
                tuple(Boundary(state, rng.randrange(5)) for state in range(size)),
                ("zero", "quarter", "half", "three-quarters", "one"),
            )
            source = RationalRelation(topology, values, algebra=MARKOV)
            oracle = _language(source)
            reduced = minimize_static_acceptor(source)
            assert _language(reduced) == oracle
            assert reduced.topology.n_states == _minimum_state_count(oracle)


def test_static_quotient_refuses_unrepresentable_coefficients_and_trainable_weights():
    base, _ = _machine()
    topology = replace(
        base.topology,
        arcs=(Arc(0, 1, (0,), 0),),
        initial=(Boundary(0, 0),),
        final=(Boundary(1, 1),),
        weight_names=("one-neighbor", "one"),
    )
    values = torch.tensor([1.0 + 2.0**-52, 1.0], dtype=torch.float64)
    rounded = RationalRelation(topology, values, algebra=MARKOV)
    with pytest.raises(ValueError, match="exactly representable"):
        minimize_static_acceptor(rounded)
    trainable, _ = _machine(trainable=True)
    with pytest.raises(TypeError, match="nontrainable"):
        minimize_static_acceptor(trainable)


def test_only_productive_positive_cycles_are_refused():
    base, weights = _machine()
    cyclic = replace(base.topology, arcs=base.topology.arcs + (Arc(3, 3, (0,), 0),))
    with pytest.raises(ValueError, match="acyclic productive support"):
        minimize_static_acceptor(RationalRelation(cyclic, weights, algebra=MARKOV))
    new_weights = torch.cat((weights, weights.new_zeros(1)))
    nonproductive = replace(
        base.topology,
        arcs=base.topology.arcs + (Arc(3, 3, (0,), 5), Arc(4, 4, (0,), 0)),
        weight_names=(*base.topology.weight_names, "zero"),
    )
    source = RationalRelation(nonproductive, new_weights, algebra=MARKOV)
    assert _language(minimize_static_acceptor(source)) == _language(source)


@pytest.mark.parametrize("extra", [Arc(0, 2, (0,), 0), Arc(0, 2, (None,), 0)])
def test_ambiguous_symbols_and_epsilons_are_refused(extra):
    base, _ = _machine()
    illegal = replace(base.topology, arcs=base.topology.arcs + (extra,))
    with pytest.raises(ValueError, match="epsilon-free deterministic"):
        minimize_static_acceptor(
            RationalRelation(illegal, base.module(), algebra=MARKOV)
        )
