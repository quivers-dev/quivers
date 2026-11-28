"""Independent geometric and tied-parameter checks for string conditionals."""

from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV, DivergentClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationTopology,
)
from tests.test_relation_composition import _INPUT, _OUTPUT, _SHARED


@pytest.mark.parametrize("algebra", (MARKOV, LOG_PROB))
def test_conditional_output_mass_uses_exact_insertion_closure_and_live_weights(algebra):
    class Source(nn.Module):
        def __init__(self):
            super().__init__()
            self.slots = nn.Parameter(
                torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64)
            )
            self.calls = 0

        def forward(self):
            self.calls += 1
            return self.slots.log() if algebra is LOG_PROB else self.slots

    topology = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        2,
        (Arc(0, 0, (None, 1), 0), Arc(0, 1, (0, 0), 1)),
        (Boundary(0, 2),),
        (Boundary(1, 2),),
        ("insert", "consume", "unit"),
    )
    source = Source()
    relation = RationalRelation(topology, source, algebra=algebra)
    with pytest.raises(ProductiveEpsilonCycleError):
        relation.conditional(((0,),), max_states=16, max_work=32)
    query = relation.conditional(
        ((0,),), max_states=16, max_work=32, max_closure_size=1
    )
    insert = Fraction(1, 4)
    assert algebra.to_reals(query.normalizer()).item() == pytest.approx(
        float(Fraction(2, 3))
    )
    for n in range(4):
        before = source.calls
        log_probability = query.log_prob(((1,) * n + (0,),))
        assert source.calls == before + 1
        assert log_probability.exp().item() == pytest.approx(
            float(insert**n * (1 - insert))
        )
        (gradient,) = torch.autograd.grad(log_probability, source.slots)
        assert gradient.tolist() == pytest.approx(
            [float(n / insert - 1 / (1 - insert)), 0.0, 0.0], abs=1e-13
        )
    assert query.log_prob(((1, 1),)).item() == -torch.inf
    absent = relation.conditional(
        ((1,),), max_states=16, max_work=32, max_closure_size=1
    )
    with pytest.raises(ValueError, match="positive finite normalizer"):
        absent.log_prob(((0,),))
    with torch.no_grad():
        source.slots[0] = 1.0
    with pytest.raises(DivergentClosureError):
        query.log_prob(((0,),))


def test_all_input_tapes_are_conditioned_in_their_original_order():
    topology = RelationTopology(
        (_INPUT, _SHARED, _OUTPUT),
        2,
        3,
        (
            Arc(0, 1, (0, None, 1), 0),
            Arc(1, 2, (None, 1, None), 1),
            Arc(0, 2, (0, 1, 0), 1),
        ),
        (Boundary(0, 2),),
        (Boundary(2, 2),),
        ("step", "finish", "unit"),
    )
    relation = RationalRelation(
        topology, torch.tensor([0.5, 0.25, 1.0], dtype=torch.float64), algebra=MARKOV
    )
    query = relation.conditional(((0,), (1,)), max_states=16, max_work=32)
    assert query.normalizer().item() == Fraction(3, 8)
    assert query.log_prob(((0,),)).exp().item() == pytest.approx(2 / 3)
    assert query.log_prob(((1,),)).exp().item() == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="each input tape"):
        relation.conditional(((0,),), max_states=16, max_work=32)
