"""Exact weighted path tests for certified same-relation tape equality."""

from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    auto_intersect,
)
from tests.evidence.oracles import paths
from tests.test_relation_composition import _OUTPUT, _SHARED, _topology, _words


def test_bounded_delay_equality_matches_independent_path_table():
    machine = paths.Machine(
        arcs=tuple(
            paths.Arc(name=str(i), source=a, target=b, labels=labels, weight=weight)
            for i, (a, b, labels, weight) in enumerate(
                (
                    (0, 1, ("a", None, "r"), "tied"),
                    (0, 1, ("a", None, "r"), "tied"),
                    (1, 3, (None, "a", None), "unit"),
                    (1, 3, (None, "b", None), "tail"),
                    (0, 2, (None, "a", "z"), "zero"),
                    (2, 3, ("a", None, None), "unit"),
                    (0, 3, (None, None, "z"), "tail"),
                )
            )
        ),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=(paths.Mark(state=3, weight="unit"),),
    )
    values = dict(
        tied=Fraction(1, 4), unit=Fraction(1), tail=Fraction(1, 2), zero=Fraction(0)
    )
    topology = _topology(machine, (_SHARED, _SHARED, _OUTPUT))
    slots = nn.Parameter(
        torch.tensor(
            [float(values[name]) for name in topology.weight_names], dtype=torch.float64
        )
    )
    relation = RationalRelation(topology, slots, algebra=MARKOV)
    filtered = auto_intersect(relation, 0, 1, max_states=32, max_work=64)
    assert filtered.topology.tapes == topology.tapes
    assert filtered.module() is relation.module()
    scores = paths.relation(machine, values, tapes=3)
    for observation, weight in scores.items():
        expected = weight if observation[0] == observation[1] else 0
        assert filtered.weight(_words(observation)).item() == expected
    observed = filtered.weight(_words((("a",), ("a",), ("r",))))
    (gradient,) = torch.autograd.grad(observed, slots)
    assert gradient[topology.weight_names.index("tied")].item() == 2
    reversed_tapes = auto_intersect(relation, 1, 0, max_states=32, max_work=64)
    assert (
        reversed_tapes.weight(_words((("a",), ("a",), ("r",)))).item()
        == observed.item()
    )
    with pytest.raises(RelationLimitError, match="max_states"):
        auto_intersect(relation, 0, 1, max_states=1, max_work=64)
    with pytest.raises(RelationLimitError, match="max_work"):
        auto_intersect(relation, 0, 1, max_states=32, max_work=1)
    with pytest.raises(TypeError, match="same type"):
        auto_intersect(relation, 0, 2, max_states=32, max_work=64)
    with pytest.raises(ValueError, match="two distinct"):
        auto_intersect(relation, 0, 0, max_states=32, max_work=64)


def test_zero_net_productive_cycle_needs_no_approximate_lag_cutoff():
    active = 19
    transient = 27
    marks = (Boundary(active, 2),)
    topology = RelationTopology(
        (_SHARED, _SHARED),
        1,
        1 << 30,
        (
            Arc(active, transient, (0, None), 0),
            Arc(transient, active, (None, 0), 0),
            Arc(transient, active, (None, 1), 1),
            Arc(1, 1, (0, None), 0),  # unreachable drift is not productive
        ),
        marks,
        marks,
        ("half", "incompatible", "unit"),
    )
    slots = nn.Parameter(torch.tensor([0.5, 0.25, 1.0], dtype=torch.float64))
    filtered = auto_intersect(
        RationalRelation(topology, slots, algebra=MARKOV),
        0,
        1,
        max_states=2,
        max_work=3,
    )
    assert filtered.topology.n_states == 2
    assert len(filtered.topology.arcs) == 2
    for count in range(6):
        word = (0,) * count
        assert filtered.weight((word, word)).item() == Fraction(1, 4) ** count
    total = filtered.total_weight(max_states=2, max_work=2, max_closure_size=2)
    (gradient,) = torch.autograd.grad(total, slots)
    assert total.item() == pytest.approx(4 / 3)
    assert gradient.tolist() == pytest.approx([16 / 9, 0.0, 8 / 3])


def test_drift_refuses_even_if_the_current_drifting_arc_has_zero_weight():
    mark = (Boundary(0, 1),)
    topology = RelationTopology(
        (_SHARED, _SHARED),
        1,
        1,
        (Arc(0, 0, (0, None), 0),),
        mark,
        mark,
        ("arc", "unit"),
    )
    source = RationalRelation(topology, torch.tensor([0.0, 1.0]), algebra=MARKOV)
    with pytest.raises(ValueError, match="zero tape delay on every productive cycle"):
        auto_intersect(source, 0, 1, max_states=64, max_work=64)
