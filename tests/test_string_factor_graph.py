"""Exact enumeration oracles for the string-factor tree schedule."""

from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
)
from tests.test_relation_composition import _INPUT, _OUTPUT, _SHARED


class _UnarySource(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.slots = nn.Parameter(torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64))
        self.calls = 0

    def forward(self):
        self.calls += 1
        return self.slots.log() if self.algebra is LOG_PROB else self.slots


def _two_words(tape):
    return RelationTopology(
        (tape,),
        0,
        2,
        (Arc(0, 1, (0,), 0), Arc(0, 1, (1,), 1)),
        (Boundary(0, 2),),
        (Boundary(1, 2),),
        ("a", "b", "unit"),
    )


def _graph(algebra, observed=None):
    source = _UnarySource(algebra)
    data = torch.tensor([0.25, 0.5, 0.125, 0.75, 1.0], dtype=torch.float64)
    channel = RationalRelation(
        RelationTopology(
            (_INPUT, _OUTPUT),
            1,
            3,
            (
                Arc(0, 1, (0, 0), 0),
                Arc(0, 2, (0, None), 1),
                Arc(0, 2, (0, None), 1),
                Arc(2, 1, (None, 1), 2),
                Arc(0, 1, (1, 1), 3),
            ),
            (Boundary(0, 4),),
            (Boundary(1, 4),),
            ("same_a", "lead", "tail", "same_b", "unit"),
        ),
        data.log() if algebra is LOG_PROB else data,
        algebra=algebra,
    )
    variables = (
        StringVariable("x", _INPUT),
        StringVariable("y", _OUTPUT, observed),
        StringVariable("z", _SHARED),
    )
    other = RationalRelation(
        _two_words(_SHARED),
        torch.ones(3, dtype=torch.float64) * (0.0 if algebra is LOG_PROB else 1.0),
        algebra=algebra,
    )
    factors = (
        StringFactor(
            ("x",), RationalRelation(_two_words(_INPUT), source, algebra=algebra)
        ),
        StringFactor(("x", "y"), channel),
        StringFactor(
            ("y",), RationalRelation(_two_words(_OUTPUT), source, algebra=algebra)
        ),
        StringFactor(("z",), other),
    )
    return variables, factors, source


@pytest.mark.parametrize("algebra", (MARKOV, LOG_PROB))
@pytest.mark.parametrize("observed", (None, (1,)))
def test_tree_messages_and_gradients_match_exact_full_assignment_enumeration(
    algebra, observed
):
    variables, factors, source = _graph(algebra, observed)
    graph = StringFactorGraph(variables, factors)
    marginal = graph.tree_marginal("x", max_states=512, max_work=1024)
    assert source.calls == 0
    assert marginal.relation.topology.tapes == (_INPUT,)
    exact = [Fraction(1, 4), Fraction(1, 2)]
    rows = (
        (Fraction(1, 4), Fraction(1, 8)),
        (Fraction(0), Fraction(3, 4)),
    )
    support = range(2) if observed is None else observed
    masses = [sum(exact[x] * rows[x][y] * exact[y] for y in support) for x in range(2)]
    partition = marginal.partition()
    assert source.calls == 1
    assert algebra.to_reals(partition).item() == pytest.approx(float(2 * sum(masses)))
    for x in range(2):
        actual = marginal.log_prob((x,))
        assert actual.exp().item() == pytest.approx(float(masses[x] / sum(masses)))
    assert source.calls == 3
    assert marginal.log_prob((0, 0)).item() == -torch.inf

    values = source.slots
    independent = torch.stack(
        [
            sum(
                (
                    values[x] * float(rows[x][y]) * values[y] * values[2] ** 4
                    for y in support
                ),
                values.new_zeros(()),
            )
            for x in range(2)
        ]
    )
    (expected_partition_gradient,) = torch.autograd.grad(
        2 * independent.sum(), values, retain_graph=True
    )
    (expected_prob_gradient,) = torch.autograd.grad(
        (independent[0] / independent.sum()).log(), values
    )
    (partition_gradient,) = torch.autograd.grad(algebra.to_reals(partition), values)
    (prob_gradient,) = torch.autograd.grad(marginal.log_prob((0,)), values)
    assert torch.allclose(partition_gradient, expected_partition_gradient)
    assert torch.allclose(prob_gradient, expected_prob_gradient)


def test_tree_messages_refuse_cycles_bad_types_limits_and_impossible_evidence():
    variables, factors, _ = _graph(MARKOV, ())
    graph = StringFactorGraph(variables, factors)
    marginal = graph.tree_marginal("x", max_states=512, max_work=1024)
    assert marginal.partition().item() == 0
    with pytest.raises(ValueError, match="positive finite normalizer"):
        marginal.log_prob((0,))
    with pytest.raises(RelationLimitError, match="max_states"):
        graph.tree_marginal("x", max_states=1, max_work=1024)
    with pytest.raises(ValueError, match="unknown string variable"):
        graph.tree_marginal("unbound", max_states=512, max_work=1024)
    cycle = StringFactorGraph(variables, factors + (factors[1],))
    with pytest.raises(ValueError, match="acyclic factor graph"):
        cycle.tree_marginal("x", max_states=512, max_work=1024)
    with pytest.raises(ValueError, match="cannot repeat"):
        StringFactor(("x", "x"), factors[1].relation)
    with pytest.raises(TypeError, match="mistyped variable"):
        StringFactorGraph(
            variables, factors + (StringFactor(("y",), factors[0].relation),)
        )


def test_a_free_unary_leaf_projects_every_word_and_the_tree_partition_is_invariant():
    variables, factors, source = _graph(MARKOV)
    graph = StringFactorGraph(variables[:2], factors[:2])
    a = graph.tree_marginal("x", max_states=512, max_work=1024)
    b = graph.tree_marginal("y", max_states=512, max_work=1024)
    value = Fraction(1, 4) * (Fraction(1, 4) + Fraction(1, 8))
    value += Fraction(1, 2) * Fraction(3, 4)
    assert a.partition().item() == value
    assert b.partition().item() == value
    assert source.calls == 2


def test_tree_mass_accepts_a_productive_relation_cycle_under_the_closure_contract():
    topology = RelationTopology(
        (_INPUT,),
        0,
        1,
        (Arc(0, 0, (0,), 0),),
        (Boundary(0, 1),),
        (Boundary(0, 2),),
        ("step", "unit", "stop"),
    )
    weights = nn.Parameter(torch.tensor([0.25, 1.0, 0.75], dtype=torch.float64))
    factor = StringFactor(("x",), RationalRelation(topology, weights, algebra=MARKOV))
    graph = StringFactorGraph((StringVariable("x", _INPUT),), (factor,))
    marginal = graph.tree_marginal("x", max_states=3, max_work=3, max_closure_size=1)
    assert marginal.partition().item() == 1.0
    word_probability = marginal.log_prob((0, 0)).exp()
    assert word_probability.item() == pytest.approx(Fraction(3, 64))
    (gradient,) = torch.autograd.grad(word_probability, weights)
    assert gradient.tolist() == pytest.approx([0.3125, 0.0, 0.0])
