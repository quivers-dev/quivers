"""Exact table oracles for one synchronized tape in a multi-tape product."""

from dataclasses import replace
from fractions import Fraction
from typing import cast

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    join,
    project,
)
from tests.evidence.oracles import paths
from tests.test_relation_composition import _INPUT, _OUTPUT, _SHARED, _topology, _words


def test_single_tape_join_preserves_nonmatching_tapes_and_counts_path_pairs():
    def machine(records):
        arcs = tuple(
            paths.Arc(name=str(i), source=src, target=dst, labels=labels, weight=slot)
            for i, (src, dst, labels, slot) in enumerate(records)
        )
        return paths.Machine(
            arcs=arcs,
            initial=(paths.Mark(state=0, weight="unit"),),
            final=(paths.Mark(state=2, weight="unit"),),
        )

    another = StringSet(alphabet=FinSet(name="other", cardinality=2))
    left = machine(
        (
            (0, 1, ("x", None, None), "tied"),
            (0, 1, ("x", None, None), "tied"),
            (1, 2, (None, "a", "z"), "unit"),
            (0, 2, ("q", "b", "r"), "zero"),
        )
    )
    right = machine(
        (
            (0, 1, (None, None, "r"), "unit"),
            (1, 2, ("q", "a", None), "tied"),
            (0, 2, ("x", "b", "z"), "tied"),
        )
    )
    left_topology, right_topology = (
        replace(_topology(obj, tapes), input_tapes=2)
        for obj, tapes in (
            (left, (_INPUT, _SHARED, another)),
            (right, (another, _SHARED, _OUTPUT)),
        )
    )
    tied = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    class Source(nn.Module):
        def __init__(self, names):
            super().__init__()
            self.weight = tied
            self.names = names

        def forward(self):
            return torch.stack(
                [
                    self.weight
                    if name == "tied"
                    else self.weight.new_tensor(name == "unit")
                    for name in self.names
                ]
            )

    a, b = (
        RationalRelation(topology, Source(topology.weight_names), algebra=MARKOV)
        for topology in (left_topology, right_topology)
    )

    def product(*, keep_shared: bool = False, max_work: int = 256, left_tape: int = 1):
        return join(
            a,
            b,
            left_tape=left_tape,
            right_tape=1,
            max_states=128,
            max_work=max_work,
            keep_shared=keep_shared,
        )

    eliminated = product()
    visible = product(keep_shared=True)
    assert eliminated.topology.tapes == (_INPUT, another, another, _OUTPUT)
    assert eliminated.topology.input_tapes == 2
    assert visible.topology.tapes == (_INPUT, _SHARED, another, another, _OUTPUT)
    assert visible.topology.input_tapes == 3

    weights = dict(tied=Fraction(1, 4), zero=Fraction(0), unit=Fraction(1))
    l_table, r_table = (paths.relation(obj, weights, tapes=3) for obj in (left, right))
    projected_table = {}
    visible_table = {}
    for (x, middle, y), l_weight in l_table.items():
        for (z, check, w), r_weight in r_table.items():
            if check == middle:
                outer = (x, z, y, w)
                projected_table[outer] = (
                    projected_table.get(outer, 0) + l_weight * r_weight
                )
                full = (x, middle, z, y, w)
                visible_table[full] = visible_table.get(full, 0) + l_weight * r_weight
    for word_set, result, expected in (
        (set(projected_table) | {((),) * 4}, eliminated, projected_table),
        (set(visible_table) | {((),) * 5}, visible, visible_table),
    ):
        assert len(list(result.module().parameters())) == 1
        for word in word_set:
            assert result.weight(_words(word)).item() == expected.get(word, 0)
    score = eliminated.weight(_words((("x",), ("q",), ("z",), ("r",))))
    (gradient,) = torch.autograd.grad(score, tied)
    assert score.item() == Fraction(1, 8)
    assert gradient.item() == 1
    with pytest.raises(RelationLimitError, match="max_work"):
        product(max_work=1)
    with pytest.raises(TypeError, match="same shared tape type"):
        product(left_tape=0)
    with pytest.raises(ValueError, match="one valid tape position"):
        product(left_tape=cast(int, (0, 1)))
    with pytest.raises(ValueError, match="at least one tape"):
        join(
            project(a, (1,)),
            project(b, (1,)),
            left_tape=0,
            right_tape=0,
            max_states=128,
            max_work=256,
        )


def test_total_weight_sums_all_words_with_live_parallel_slot_derivatives():
    tape = (_INPUT,)
    topology = RelationTopology(
        tape,
        1,
        1,
        (Arc(0, 0, (0,), 0), Arc(0, 0, (1,), 0)),
        (Boundary(0, 1),),
        (Boundary(0, 2),),
        ("step", "unit", "stop"),
    )
    slots = nn.Parameter(torch.tensor([0.25, 1.0, 0.5], dtype=torch.float64))
    language = RationalRelation(topology, slots, algebra=MARKOV)
    assert language.weight(((),)).item() == Fraction(1, 2)
    with pytest.raises(ProductiveEpsilonCycleError):
        language.total_weight()
    total = language.total_weight(max_states=1, max_work=2, max_closure_size=2)
    (gradient,) = torch.autograd.grad(total, slots)
    assert total.item() == pytest.approx(1.0)
    assert gradient.tolist() == pytest.approx([4.0, 1.0, 2.0])
