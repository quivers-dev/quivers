"""Exact relation oracles for unobserved product topology and tape operations."""

from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.core.algebras import MARKOV, DivergentClosureError, UndecidedClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    ComposedRelation,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    invert,
    project,
    trim,
)
from tests.evidence.compare import Accuracy, compare_gradients, compare_number
from tests.evidence.oracles import paths
from tests.evidence.oracles.closed_forms import geometric
from tests.test_relation_composition import (
    _INPUT,
    _OUTPUT,
    _SHARED,
    _topology,
    _weighted_machines,
    _words,
)


def _product(left, right):
    return ComposedRelation(left, right).materialize(max_states=256, max_work=512)


def test_sparse_topology_cascades_and_projection_match_independent_tables():
    left, right = _weighted_machines()
    third = paths.Machine(
        arcs=(
            paths.Arc(
                name="last1", source=0, target=1, labels=("z", "x"), weight="unit"
            ),
            paths.Arc(
                name="last2", source=1, target=2, labels=("r", "q"), weight="tail"
            ),
        ),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=tuple(paths.Mark(state=i, weight="unit") for i in range(3)),
    )
    values = {
        "tied": Fraction(1, 4),
        "tail": Fraction(1, 2),
        "zero": Fraction(0),
        "unit": Fraction(1),
    }
    a, b, c = (
        RationalRelation(
            topology,
            torch.tensor(
                [float(values[slot]) for slot in topology.weight_names],
                dtype=torch.float64,
            ),
        )
        for topology in (
            _topology(left, (_INPUT, _SHARED)),
            _topology(right, (_SHARED, _OUTPUT)),
            _topology(third, (_OUTPUT, _INPUT)),
        )
    )
    ab = _product(a, b)
    lhs = _product(ab, c)
    rhs = _product(a, _product(b, c))
    reference_ab = paths.compose_relations(
        paths.relation(left, values, tapes=2),
        paths.relation(right, values, tapes=2),
    )
    expected = paths.compose_relations(
        reference_ab, paths.relation(third, values, tapes=2)
    )
    for words in (*expected, (("q",), ("x",))):
        mass = expected.get(words, Fraction(0))
        assert lhs.weight(_words(words)).item() == mass
        assert rhs.weight(_words(words)).item() == mass
    assert lhs.domain == _INPUT and lhs.codomain == _INPUT

    marginal = project(ab, (1,))
    assert marginal.topology.tapes == (_OUTPUT,)
    assert marginal.topology.input_tapes == 0
    assert marginal.module() is ab.module()
    exchanged = invert(ab)
    assert exchanged.domain == _OUTPUT and exchanged.codomain == _INPUT
    assert exchanged.module() is ab.module()
    for x, z in reference_ab:
        assert exchanged.weight(_words((z, x))).item() == reference_ab[x, z]
    for word in {z for _, z in reference_ab}:
        mass = sum(value for (_, z), value in reference_ab.items() if z == word)
        assert marginal.weight(_words((word,))).item() == mass
    for positions in ((), (1, 0), (0, 0), (-1,), (2,), (True,)):
        with pytest.raises(ValueError, match="projection requires"):
            project(ab, positions)


def test_trim_keeps_zero_weight_parallel_occurrences_and_original_source():
    topology = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        1 << 32,
        (
            Arc(0, 1, (0, 0), 0),
            Arc(0, 1, (0, 0), 0),
            Arc(0, 2, (None, None), 1),
            Arc(2, 2, (None, None), 1),
            Arc(3, 1, (None, None), 1),
        ),
        (Boundary(0, 1),),
        (Boundary(1, 1),),
        ("edge", "unit"),
    )
    slots = nn.Parameter(torch.tensor([0.0, 1.0], dtype=torch.float64))
    relation = RationalRelation(topology, slots)
    trimmed = trim(relation)
    assert trimmed.topology == replace(topology, arcs=topology.arcs[:2])
    assert trimmed.module() is relation.module()
    score = trimmed.weight(((0,), (0,)))
    (gradient,) = torch.autograd.grad(score, slots)
    assert score.item() == 0
    assert gradient.tolist() == [2.0, 0.0]


def _hidden_repeat():
    marks = (Boundary(0, 1),)
    left = RelationTopology(
        (_INPUT, _SHARED),
        1,
        1,
        (Arc(0, 0, (None, 0), 0),),
        marks,
        marks,
        ("loop", "unit"),
    )
    right = RelationTopology(
        (_SHARED, _OUTPUT),
        1,
        1,
        (Arc(0, 0, (0, None), 0),),
        marks,
        marks,
        ("loop", "unit"),
    )
    return left, right


def test_materialized_cyclic_product_preserves_each_primitive_weight_and_gradient():
    left, right = _hidden_repeat()
    slots = nn.Parameter(torch.tensor([0.5, 1.0], dtype=torch.float64))
    composed = ComposedRelation(
        RationalRelation(left, slots, algebra=MARKOV),
        RationalRelation(right, slots, algebra=MARKOV),
    )
    materialized = composed.materialize(max_states=2, max_work=3)
    assert [
        (arc.source, arc.target, arc.weight) for arc in materialized.topology.arcs
    ] == [(0, 1, 0), (1, 0, 2)]
    assert list(materialized.module().parameters())[0] is slots
    with pytest.raises(RelationLimitError, match="max_closure_size"):
        materialized.query(((), ()), max_closure_size=1)
    query = materialized.query(((), ()), max_closure_size=2)
    value = query.weight()
    (gradient,) = torch.autograd.grad(value, slots)
    mass = geometric(Fraction(1, 4))
    accuracy = Accuracy(domain="linear", operations=160)
    compare_number(value.item(), mass, accuracy, "cyclic composed mass")
    compare_gradients(
        {"loop": gradient[0], "boundary": gradient[1]},
        {"loop": mass**2, "boundary": 4 * mass},
        accuracy,
        "cyclic composed gradients",
    )
    with torch.no_grad():
        slots[0] = 1
    with pytest.raises(DivergentClosureError):
        query.weight()


def test_cyclic_product_does_not_certify_a_rounded_primitive_product_as_divergent():
    left, right = _hidden_repeat()
    below, above = 1 - 2**-52, 1 + 2**-52
    ratio = Fraction(below) * Fraction(above)
    assert ratio < 1 and below * above == 1
    materialized = _product(
        RationalRelation(
            left, torch.tensor([below, 1.0], dtype=torch.float64), algebra=MARKOV
        ),
        RationalRelation(
            right, torch.tensor([above, 1.0], dtype=torch.float64), algebra=MARKOV
        ),
    )
    query = materialized.query(((), ()), max_closure_size=2)
    try:
        value = query.weight().item()
    except UndecidedClosureError:
        pass
    else:
        compare_number(
            value,
            geometric(ratio),
            Accuracy(domain="linear", operations=160),
            "near-boundary cyclic product",
        )
