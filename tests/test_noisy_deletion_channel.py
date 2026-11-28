"""Exact negative-binomial oracles for a composed noisy string channel."""

from fractions import Fraction
from math import comb

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationTopology,
    condition,
    join,
)


def _model():
    class Weights(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            self.calls = 0

        def forward(self):
            self.calls += 1
            return torch.stack((self.p, 1 - self.p, self.p.new_ones(())))

    tape = StringSet(alphabet=FinSet(name="UnaryAlphabet", cardinality=1))
    weights = Weights()
    names = ("repeat_or_delete", "stop_or_keep", "unit")
    unit = (Boundary(0, 2),)
    source = RationalRelation(
        RelationTopology(
            (tape,), 0, 1, (Arc(0, 0, (0,), 0),), unit, (Boundary(0, 1),), names
        ),
        weights,
        algebra=MARKOV,
    )
    channel = RationalRelation(
        RelationTopology(
            (tape, tape),
            1,
            1,
            (Arc(0, 0, (0, None), 0), Arc(0, 0, (0, 0), 1)),
            unit,
            unit,
            names,
        ),
        weights,
        algebra=MARKOV,
    )
    joint = join(
        source,
        channel,
        left_tape=0,
        right_tape=0,
        keep_shared=True,
        max_states=64,
        max_work=128,
    )
    return tape, weights, joint


@pytest.mark.parametrize("observed_length", (0, 1, 2, 3))
def test_unbounded_deletion_channel_evidence_posterior_and_tied_gradient(
    observed_length,
):
    tape, source, joint = _model()
    assert joint.topology.tapes == (tape, tape)
    assert joint.topology.n_states == 3
    assert list(joint.module().parameters())[0] is source.p
    assert len(list(joint.module().parameters())) == 1
    m = observed_length
    observed = condition(joint, 1, (0,) * m, max_states=64, max_work=128)
    before = source.calls
    mass = observed.total_weight(max_states=64, max_work=128, max_closure_size=2)
    assert source.calls == before + 1
    p = Fraction(1, 4)
    expected = (1 - p) * (p * (1 - p)) ** m / (1 - p**2) ** (m + 1)
    expected_derivative = expected * (
        m / p - (m + 1) / (1 - p) + (m + 1) * 2 * p / (1 - p**2)
    )
    assert mass.item() == pytest.approx(float(expected), abs=1e-14)
    (gradient,) = torch.autograd.grad(mass, source.p, retain_graph=True)
    assert gradient.item() == pytest.approx(float(expected_derivative), abs=1e-13)
    for n in range(7):
        probability = observed.weight(((0,) * n,)) / mass
        if n < m:
            assert probability.item() == 0
            continue
        exact = comb(n, m) * (p**2) ** (n - m) * (1 - p**2) ** (m + 1)
        exact_derivative = exact * (2 * (n - m) / p - 2 * p * (m + 1) / (1 - p**2))
        assert probability.item() == pytest.approx(float(exact), abs=1e-14)
        (derivative,) = torch.autograd.grad(probability, source.p, retain_graph=True)
        assert derivative.item() == pytest.approx(float(exact_derivative), abs=1e-13)
