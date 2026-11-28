"""Check candidate selection, complementary mass, and live derivatives."""

from math import log

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringCandidatePlan,
)


_TAPE = StringSet(alphabet=FinSet(name="CandidateAlphabet", cardinality=2))
_CANDIDATES = ((), (1,), (0,))


class _GeometricWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.calls = 0

    def forward(self):
        self.calls += 1
        return torch.stack(
            (self.p.log() - log(2), (-self.p).log1p(), self.p.new_zeros(()))
        )


def _geometric():
    source = _GeometricWeights()
    topology = RelationTopology(
        (_TAPE,),
        0,
        1,
        (Arc(0, 0, (0,), 0), Arc(0, 0, (1,), 0)),
        (Boundary(0, 2),),
        (Boundary(0, 1),),
        ("repeat", "stop", "unit"),
    )
    return RationalRelation(topology, source, algebra=LOG_PROB), source


def test_beam_keeps_scale_and_sums_complement_with_live_gradients():
    relation, source = _geometric()
    plan = StringCandidatePlan(
        relation, _CANDIDATES, max_states=64, max_work=192, max_closure_size=8
    )
    assert source.calls == 0
    for iteration, probability in enumerate((0.2, 0.6), 1):
        with torch.no_grad():
            source.p.fill_(probability)
        message = plan.evaluate(beam=2)
        assert source.calls == iteration
        assert message.tape == _TAPE
        assert message.words == ((), (0,))
        assert message.status == "truncated" and message.boundary_tie

        p = source.p
        expected = torch.stack((1 - p, (1 - p) * p / 2))
        omitted = p.square() + expected[1]
        torch.testing.assert_close(message.log_total_mass.exp(), p.new_ones(()))
        torch.testing.assert_close(message.log_weights.exp(), expected)
        torch.testing.assert_close(
            message.normalized_log_weights().exp(), expected / expected.sum()
        )
        torch.testing.assert_close(message.log_omitted_mass.exp(), omitted)
        (actual_gradient,) = torch.autograd.grad(message.log_omitted_mass, p)
        (expected_gradient,) = torch.autograd.grad(omitted.log(), p)
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_direct_complement_retains_tiny_positive_mass():
    relation, source = _geometric()
    with torch.no_grad():
        source.p.fill_(1e-120)
    plan = StringCandidatePlan(
        relation, _CANDIDATES, max_states=64, max_work=192, max_closure_size=8
    )
    message = plan.evaluate()
    assert message.status == "truncated" and not message.boundary_tie
    torch.testing.assert_close(message.log_omitted_mass, 2 * source.p.log())
    (gradient,) = torch.autograd.grad(message.log_omitted_mass, source.p)
    torch.testing.assert_close(gradient, 2 / source.p)


def test_full_support_zero_beam_and_resource_refusal_are_distinct():
    topology = RelationTopology(
        (_TAPE,),
        0,
        2,
        (Arc(0, 1, (1,), 0),),
        (Boundary(0, 0),),
        (Boundary(1, 0),),
        ("unit",),
    )
    relation = RationalRelation(topology, torch.zeros(1), algebra=LOG_PROB)
    plan = StringCandidatePlan(relation, ((0,), (1,)), max_states=64, max_work=192)
    complete = plan.evaluate(beam=1)
    assert complete.words == ((1,),)
    assert complete.status == "exact"
    assert torch.isneginf(complete.log_omitted_mass)
    dropped = plan.evaluate(beam=0)
    assert dropped.words == () and dropped.status == "truncated"
    assert dropped.log_omitted_mass.item() == 0
    with pytest.raises(ValueError, match="normalizer"):
        dropped.normalized_log_weights()
    with pytest.raises(RelationLimitError, match="max_work"):
        StringCandidatePlan(relation, ((0,),), max_states=64, max_work=0)
    with pytest.raises(ValueError, match="must be distinct"):
        StringCandidatePlan(relation, ((0,), (0,)), max_states=64, max_work=192)
    cyclic, _ = _geometric()
    with pytest.raises(ProductiveEpsilonCycleError):
        StringCandidatePlan(cyclic, ((0,),), max_states=64, max_work=192)
