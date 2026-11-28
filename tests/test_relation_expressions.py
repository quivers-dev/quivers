"""Exact path oracles for weighted union, concatenation and repetition."""

from fractions import Fraction
from itertools import product

import pytest
import torch
from torch import nn

from quivers.core.algebras import COUNTING, MARKOV, DivergentClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationTopology,
    closure,
    concatenate,
    invert,
    union,
)
from tests.evidence.oracles import paths
from tests.test_relation_composition import _INPUT, _topology, _words


def test_union_and_concatenation_preserve_the_exact_sum_of_path_multiplicities():
    def machine(names):
        return paths.Machine(
            arcs=tuple(
                paths.Arc(
                    name=str(i), source=0, target=1, labels=(label,), weight=weight
                )
                for i, (label, weight) in enumerate(names)
            ),
            initial=(paths.Mark(state=0, weight="unit"),),
            final=(paths.Mark(state=1, weight="unit"),),
        )

    def relation(obj):
        topology = _topology(obj, (_INPUT,))
        data = torch.tensor([int(weights[name]) for name in topology.weight_names])
        return RationalRelation(topology, data, algebra=COUNTING)

    left = machine([(None, "two"), ("a", "one"), ("a", "two")])
    right = machine([(None, "one"), ("a", "one"), ("b", "three")])
    weights = dict(
        unit=Fraction(1), one=Fraction(1), two=Fraction(2), three=Fraction(3)
    )
    a, b = (
        paths.relation(left, weights, tapes=1),
        paths.relation(right, weights, tapes=1),
    )
    a_relation, b_relation = relation(left), relation(right)
    added = union(a_relation, b_relation)
    serial = concatenate(a_relation, b_relation)
    for size in range(4):
        for word in product("ab", repeat=size):
            observed = _words((word,))
            assert added.weight(observed).item() == a.get((word,), 0) + b.get(
                (word,), 0
            )
            assert serial.weight(observed).item() == sum(
                a.get((word[:split],), 0) * b.get((word[split:],), 0)
                for split in range(size + 1)
            )

    with pytest.raises(TypeError, match="identical ordered tape types"):
        concatenate(a_relation, invert(b_relation))
    with pytest.raises(TypeError, match="common scalar semiring"):
        union(a_relation, RationalRelation(b_relation.topology, b_relation.module()))


def test_binary_expressions_read_an_aliased_source_once_and_share_derivatives():
    class Source(nn.Module):
        def __init__(self):
            super().__init__()
            self.slots = nn.Parameter(torch.tensor([0.5, 1.0], dtype=torch.float64))
            self.calls = 0

        def forward(self):
            self.calls += 1
            return self.slots

    topology = RelationTopology(
        (_INPUT,),
        1,
        2,
        (Arc(0, 1, (0,), 0),),
        (Boundary(0, 1),),
        (Boundary(1, 1),),
        ("arc", "unit"),
    )
    source = Source()
    language = RationalRelation(topology, source, algebra=MARKOV)
    for operation, word, mass, expected in (
        (union, (0,), 1.0, [2.0, 2.0]),
        (concatenate, (0, 0), 0.25, [1.0, 1.0]),
    ):
        result = operation(language, language)
        assert list(result.module().parameters())[0] is source.slots
        assert len(list(result.module().parameters())) == 1
        before = source.calls
        weight = result.weight((word,))
        assert source.calls == before + 1
        (gradient,) = torch.autograd.grad(weight, source.slots)
        assert weight.item() == mass
        assert gradient.tolist() == expected


def test_closure_matches_the_geometric_series_and_its_derivatives():
    topology = RelationTopology(
        (_INPUT,),
        1,
        2,
        (Arc(0, 1, (0,), 1),),
        (Boundary(0, 0),),
        (Boundary(0, 2), Boundary(1, 2)),
        ("p", "q", "unit"),
    )
    p, q = Fraction(1, 4), Fraction(1, 2)
    slots = nn.Parameter(torch.tensor([float(p), float(q), 1.0], dtype=torch.float64))
    repeated = closure(RationalRelation(topology, slots, algebra=MARKOV))
    assert list(repeated.module().parameters())[0] is slots
    with pytest.raises(ProductiveEpsilonCycleError):
        repeated.weight(((),))
    for count in range(5):
        observed = ((0,) * count,)
        # Every 'a' comes from one operand path. The count + 1 gaps
        # independently admit any number of empty operand paths.
        expected = (p * q) ** count / (1 - p) ** (count + 1)
        score = repeated.query(observed, max_closure_size=2).weight()
        (gradient,) = torch.autograd.grad(score, slots)
        assert score.item() == pytest.approx(float(expected))
        assert gradient[0].item() == pytest.approx(
            float(expected * (count / p + (count + 1) / (1 - p)))
        )
        assert gradient[1].item() == pytest.approx(float(expected * count / q))
    with torch.no_grad():
        slots[0] = 1.0
    with pytest.raises(DivergentClosureError):
        repeated.query(((),), max_closure_size=2).weight()
