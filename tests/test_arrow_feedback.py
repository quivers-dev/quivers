"""Feedback capabilities respect the carriers of finite arrows."""

import pytest
import torch

from quivers.arrows.instances import Function, Stochastic, VRel
from quivers.arrows.typeclasses import ArrowChoice, ArrowLoop
from quivers.core.algebras import MARKOV
from quivers.core.morphisms import identity, observed
from quivers.core.objects import FinSet, ProductSet, Unit


@pytest.mark.parametrize("arrow", [Function(), Stochastic()])
def test_normalized_arrows_do_not_offer_general_feedback(arrow) -> None:
    """Two feedback fixed points have mass two, outside both carriers."""
    feedback = FinSet(name="Feedback", cardinality=2)
    product = ProductSet(components=(Unit, feedback))
    morphism = identity(product, algebra=MARKOV)
    contraction = VRel().loop_arr(morphism, feedback)
    assert contraction.tensor.item() == 2.0
    assert isinstance(arrow, ArrowChoice)
    assert not isinstance(arrow, ArrowLoop)
    assert not hasattr(arrow, "loop_arr")


@pytest.mark.parametrize("arrow", [Function(), Stochastic()])
def test_composition_retains_normalized_rows(arrow) -> None:
    """The supported operation stays within each declared carrier."""
    obj = FinSet(name="State", cardinality=2)
    data = (
        torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        if isinstance(arrow, Function)
        else torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    )
    morphism = observed(obj, obj, data, algebra=MARKOV)
    result = arrow.compose(morphism, morphism).tensor
    torch.testing.assert_close(result.sum(-1), torch.ones(2))
    if isinstance(arrow, Function):
        assert ((result == 0) | (result == 1)).all()


def test_weighted_relation_feedback_satisfies_yanking() -> None:
    """Tracing the swap leaves the identity on the untraced factor."""
    obj = FinSet(name="State", cardinality=2)
    product = ProductSet(components=(obj, obj))
    swap = torch.zeros(2, 2, 2, 2)
    for left in range(2):
        for right in range(2):
            swap[left, right, right, left] = 1
    morphism = observed(product, product, swap, algebra=MARKOV)
    result = VRel().loop_arr(morphism, obj)
    torch.testing.assert_close(result.tensor, torch.eye(2))


def test_weighted_relation_feedback_over_unit_is_vanishing() -> None:
    """Tracing a singleton feedback factor retains every weight."""
    obj = FinSet(name="State", cardinality=2)
    product = ProductSet(components=(obj, Unit))
    data = torch.tensor([[0.2, 0.7], [1.3, 0.4]])
    morphism = observed(product, product, data.reshape(2, 1, 2, 1), algebra=MARKOV)
    torch.testing.assert_close(VRel().loop_arr(morphism, Unit).tensor, data)
