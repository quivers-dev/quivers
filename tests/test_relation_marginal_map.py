"""Exact bounded output ranking and alignment-sum differentiation."""

from dataclasses import replace
from fractions import Fraction
from math import log

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    best_marginal_output,
)
from tests.test_relation_composition import _INPUT


class _CountingWeights(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float64))
        self.calls = 0

    def forward(self):
        self.calls += 1
        return self.values


def _relation(solo=0.5, twice=0.375):
    topology = RelationTopology(
        (_INPUT, _INPUT),
        1,
        3,
        (
            Arc(0, 1, (0, 0), 0),
            Arc(0, 2, (0, 1), 1),
            Arc(0, 2, (0, 1), 1),
        ),
        (Boundary(0, 2),),
        (Boundary(1, 2), Boundary(2, 2)),
        ("solo", "shared", "unit"),
    )
    source = _CountingWeights([solo, twice, 1])
    return RationalRelation(topology, source, algebra=MARKOV), source


def test_marginal_output_can_differ_from_the_best_alignment_and_retains_gradients():
    relation, source = _relation()
    answer = best_marginal_output(
        relation, ((0,),), (((0,),), ((1,),)), max_states=12, max_work=20
    )
    assert answer is not None
    assert answer.output == ((1,),)
    assert answer.exact_weight == Fraction(3, 4)
    assert answer.log_pair_weight.item() == pytest.approx(log(0.75))
    assert source.calls == 1
    (gradient,) = torch.autograd.grad(answer.log_pair_weight, source.values)
    assert gradient.tolist() == pytest.approx([0, 8 / 3, 2])
    best0 = relation.query(((0,), (0,))).best_path()
    best1 = relation.query(((0,), (1,))).best_path()
    assert best0 is not None and best1 is not None
    assert best0.weight > best1.weight


def test_exact_rational_order_can_break_a_native_float_tie():
    relation, source = _relation(solo=1.0, twice=2.0**-60)
    topology = replace(
        relation.topology,
        arcs=(*relation.topology.arcs, Arc(0, 2, (0, 1), 0)),
    )
    source_machine = RationalRelation(topology, source, algebra=MARKOV)
    assert (
        source_machine.weight(((0,), (0,))).item()
        == source_machine.weight(((0,), (1,))).item()
    )
    answer = best_marginal_output(
        source_machine, ((0,),), (((0,),), ((1,),)), max_states=12, max_work=20
    )
    assert answer is not None
    assert answer.output == ((1,),)
    assert answer.exact_weight == 1 + 2 * Fraction(1, 2**60)


def test_budgets_and_unsupported_observations_are_explicit():
    relation, _ = _relation(solo=0.75)
    best = best_marginal_output(
        relation, ((0,),), (((1,),), ((0,),)), max_states=12, max_work=20
    )
    assert best is not None and best.output == ((1,),)
    assert (
        best_marginal_output(
            relation,
            ((1,),),
            (((0,),), ((1,),)),
            max_states=12,
            max_work=20,
        )
        is None
    )
    with pytest.raises(RelationLimitError, match="max_states"):
        best_marginal_output(
            relation,
            ((0,),),
            (((1,),), ((0,),)),
            max_states=3,
            max_work=20,
        )
    with pytest.raises(RelationLimitError, match="max_work"):
        best_marginal_output(
            relation,
            ((0,),),
            (((1,),), ((0,),)),
            max_states=12,
            max_work=4,
        )
    cyclic = replace(
        relation.topology, arcs=(*relation.topology.arcs, Arc(1, 1, (None, None), 0))
    )
    source = RationalRelation(cyclic, relation.module(), algebra=MARKOV)
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic productive"):
        best_marginal_output(source, ((0,),), (((0,),),), max_states=12, max_work=20)
    log_relation = RationalRelation(relation.topology, torch.zeros(3), algebra=LOG_PROB)
    with pytest.raises(TypeError, match="requires Markov"):
        best_marginal_output(
            log_relation, ((0,),), (((0,),),), max_states=12, max_work=20
        )
