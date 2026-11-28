"""Exact support and differentiable message-scoring oracles."""

from itertools import product
from math import log

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV, DivergentClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    NgramMessagePlan,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
)


def _ambiguous():
    tape = StringSet(alphabet=FinSet(name="AB and unused symbols", cardinality=1 << 30))
    topology = RelationTopology(
        (tape,),
        0,
        1 << 26,
        (
            Arc(0, 1, (0,), 0),
            Arc(0, 2, (None,), 1),
            Arc(2, 1, (0,), 2),
            Arc(1, 3, (1,), 3),
            Arc(3, 4, (0,), 3),
        ),
        (Boundary(0, 4),),
        (Boundary(1, 4), Boundary(4, 4)),
        ("direct", "enter", "emit", "repeat", "zero"),
    )
    weights = nn.Parameter(
        torch.tensor([-0.25, -0.5, -0.125, -0.375, 0], dtype=torch.float64)
    )
    return RationalRelation(topology, weights, algebra=LOG_PROB), weights


def test_matching_feature_occurrences_weight_each_ambiguous_word_once():
    base, weights = _ambiguous()
    features = ((0,), (0, 1, 0), (1, 0))
    plan = NgramMessagePlan(base, features, max_states=20, max_work=100)
    coefficients = nn.Parameter(weights.new_tensor([0.375, -0.125, 0.625]))
    relation = plan.bind(coefficients)
    assert {id(parameter) for parameter in relation.module().parameters()} == {
        id(weights),
        id(coefficients),
    }
    assert plan.n_states <= 8
    for length in range(4):
        for word in product(range(2), repeat=length):
            counts = [
                sum(
                    word[index : index + len(feature)] == feature
                    for index in range(len(word))
                )
                for feature in features
            ]
            expected = base.weight((word,)) + sum(
                k * t for k, t in zip(counts, coefficients, strict=True)
            )
            torch.testing.assert_close(relation.weight((word,)), expected)
    result = relation.weight(((0, 1, 0),))
    p_grad, c_grad = torch.autograd.grad(result, (weights, coefficients))
    (direct_grad,) = torch.autograd.grad(base.weight(((0, 1, 0),)), weights)
    torch.testing.assert_close(p_grad, direct_grad)
    assert c_grad.tolist() == pytest.approx([2.0, 1.0, 1.0])
    with torch.no_grad():
        coefficients[0] += 0.25
    assert relation.weight(((0, 1, 0),)).item() == pytest.approx(result.item() + 0.5)


def test_convergent_geometric_message_checked_by_existing_star():
    tape = StringSet(alphabet=FinSet(name="Single", cardinality=1))
    topology = RelationTopology(
        (tape,),
        0,
        1,
        (Arc(0, 0, (0,), 0),),
        (Boundary(0, 1),),
        (Boundary(0, 2),),
        ("continue", "unit", "stop"),
    )
    base = RationalRelation(
        topology,
        torch.tensor([log(0.25), 0, log(0.75)], dtype=torch.float64),
        algebra=LOG_PROB,
    )
    plan = NgramMessagePlan(base, ((0,),), max_states=8, max_work=20)
    theta = nn.Parameter(torch.tensor([log(2)], dtype=torch.float64))
    model = plan.bind(theta)
    value = model.total_weight(max_states=8, max_work=20, max_closure_size=1)
    assert value.item() == pytest.approx(log(1.5))
    (derivative,) = torch.autograd.grad(value, theta)
    assert derivative.tolist() == pytest.approx([1.0])
    with torch.no_grad():
        theta[0] = log(5)
    with pytest.raises(DivergentClosureError):
        model.total_weight(max_states=8, max_work=20, max_closure_size=1)


def test_limits_and_unsupported_coefficients():
    relation, weights = _ambiguous()
    with pytest.raises(RelationLimitError, match="max_states"):
        NgramMessagePlan(relation, ((0,),), max_states=1, max_work=100)
    with pytest.raises(RelationLimitError, match="max_work"):
        NgramMessagePlan(relation, ((0,) * 100,), max_states=10, max_work=10)
    with pytest.raises(ValueError, match="distinct nonempty"):
        NgramMessagePlan(relation, ((0,), (0,)), max_states=10, max_work=100)
    markov = RationalRelation(relation.topology, weights.exp().detach(), algebra=MARKOV)
    with pytest.raises(TypeError, match="one-tape LogProb"):
        NgramMessagePlan(markov, ((0,),), max_states=10, max_work=100)
    plan = NgramMessagePlan(relation, ((0,),), max_states=10, max_work=100)
    empty = plan.bind(weights.new_zeros(0))
    with pytest.raises(ValueError, match="feature count"):
        empty.weight(((0,),))
