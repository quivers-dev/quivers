"""Runtime domain failures identify the checked binding that evaluated them."""

import pytest
import torch

from quivers.dsl import loads


@pytest.mark.parametrize("kind", ["let", "score"])
def test_checked_probability_failure_names_its_binding_and_preserves_value_error(kind):
    source = f"""
object Unit : FinSet 1
object Out : Real 1
program model(p : Real) : Unit -> Out
    {kind} checked_probability = log_probability(p)
    return p
export model
"""
    compiled = loads(source)
    with pytest.raises(
        ValueError, match=rf"{kind} 'checked_probability'.*log_probability"
    ) as caught:
        compiled.morphism.log_joint(torch.zeros(1, 1), {"p": torch.tensor(1.5)})
    assert isinstance(caught.value.__cause__, ValueError)
