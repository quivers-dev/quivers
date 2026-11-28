"""Evidence for fixed-tape slices and weighted-language intersection."""

from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.core.objects import Unit
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    condition,
    intersect,
    invert,
    project,
)
from tests.evidence.oracles import paths
from tests.test_relation_composition import (
    _INPUT,
    _OUTPUT,
    _SHARED,
    _topology,
    _weighted_machines,
    _words,
)


def test_conditioning_a_tape_keeps_the_original_weight_and_gradient():
    marks = (Boundary(0, 1),)
    topology = RelationTopology(
        (_INPUT, _OUTPUT), 1, 1, (Arc(0, 0, (0, 1), 0),), marks, marks, ("arc", "unit")
    )
    slots = nn.Parameter(torch.tensor([0.5, 1.0], dtype=torch.float64))
    relation = RationalRelation(topology, slots)
    remaining = condition(relation, 0, (0, 0), max_states=3, max_work=2)
    assert remaining.module() is relation.module()
    assert remaining.domain == Unit and remaining.codomain == _OUTPUT
    score = remaining.weight(((1, 1),))
    (gradient,) = torch.autograd.grad(score, slots)
    assert score.item() == Fraction(1, 4)
    assert gradient.tolist() == [1.0, 0.5]
    assert remaining.weight(((1,),)).item() == 0
    conditioned_output = condition(relation, 1, (1, 1), max_states=3, max_work=2)
    assert conditioned_output.domain == _INPUT and conditioned_output.codomain == Unit
    assert conditioned_output.weight(((0, 0),)).item() == Fraction(1, 4)
    with pytest.raises(RelationLimitError, match="max_states"):
        condition(relation, 0, (0, 0), max_states=2, max_work=2)
    with pytest.raises(RelationLimitError, match="max_work"):
        condition(relation, 0, (0, 0), max_states=3, max_work=1)
    with pytest.raises(ValueError, match="invalid symbol"):
        condition(relation, 0, (2,), max_states=3, max_work=2)
    with pytest.raises(ValueError, match="one remaining tape"):
        condition(remaining, 0, (1, 1), max_states=3, max_work=2)


def test_weighted_language_intersection_matches_path_table_product():
    left, right = _weighted_machines()
    weights = {
        "tied": Fraction(1, 8),
        "tail": Fraction(1, 2),
        "zero": Fraction(0),
        "unit": Fraction(1),
    }
    a, b = (
        RationalRelation(
            obj,
            torch.tensor(
                [float(weights[slot]) for slot in obj.weight_names], dtype=torch.float64
            ),
        )
        for obj in (
            _topology(left, (_INPUT, _SHARED)),
            _topology(right, (_SHARED, _OUTPUT)),
        )
    )
    l_lang = project(a, (1,))
    r_lang = invert(project(b, (0,)))
    expected_left = paths.relation(left, weights, tapes=2)
    expected_right = paths.relation(right, weights, tapes=2)
    joint = intersect(l_lang, r_lang, max_states=128, max_work=256)
    joint_input = intersect(
        invert(l_lang), invert(r_lang), max_states=128, max_work=256
    )
    assert joint.domain == Unit and joint.codomain == _SHARED
    assert joint_input.domain == _SHARED and joint_input.codomain == Unit
    words = (
        {(), ("b",)} | {y for _, y in expected_left} | {y for y, _ in expected_right}
    )
    for word in words:
        l_mass = sum(weight for (_, y), weight in expected_left.items() if y == word)
        r_mass = sum(weight for (y, _), weight in expected_right.items() if y == word)
        expected = l_mass * r_mass
        assert joint.weight(_words((word,))).item() == expected
        assert joint_input.weight(_words((word,))).item() == expected
    with pytest.raises(TypeError, match="one tape per operand"):
        intersect(a, b, max_states=128, max_work=256)
    with pytest.raises(TypeError, match="same tape orientation"):
        intersect(l_lang, invert(r_lang), max_states=128, max_work=256)
