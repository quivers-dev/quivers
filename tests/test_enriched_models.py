"""Runtime ownership and validation of enriched records."""

import copy

import didactic.api as dx
import pytest
import torch

from quivers.core.algebras import GODEL, PRODUCT_FUZZY
from quivers.core.objects import FinSet
from quivers.enriched.weighted_limits import Weight
from quivers.enriched.yoneda import Presheaf


def test_weight_updates_preserve_tensor_identity_and_gradients() -> None:
    """A weight holds its live tensor and validates immutable updates."""
    values = torch.tensor([0.2, 0.7], requires_grad=True)
    weight = Weight(values=values, algebra=GODEL)
    updated = weight.with_(algebra=GODEL)
    assert updated.values is values
    assert updated == weight
    assert hash(updated) == hash(weight)
    assert weight.with_(values=values.detach().clone()) != weight
    updated.values.square().sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.4, 1.4]))
    with pytest.raises(dx.ValidationError, match="1D"):
        weight.with_(values=values.reshape(1, 2))
    with pytest.raises(dx.ValidationError, match="IdempotentSemiring"):
        weight.with_(algebra=PRODUCT_FUZZY)


def test_presheaf_retains_mutable_mapping_and_live_tensors() -> None:
    """Mapping growth preserves record identity and differentiable values."""
    objects = [FinSet(name="A", cardinality=2)]
    tensor = torch.tensor([0.2, 0.7], requires_grad=True)
    values = {0: tensor}
    presheaf = Presheaf(objects=objects, values=values, algebra=GODEL)
    copied = copy.copy(presheaf)
    initial_hash = hash(presheaf)
    assert copied.objects is objects
    assert copied.values is values
    objects.append(FinSet(name="B", cardinality=2))
    values[1] = tensor * 3
    assert copied.size == 2
    assert copied.evaluate(1) is values[1]
    assert hash(presheaf) == initial_hash
    assert copied == presheaf
    assert presheaf.with_(values=dict(values)) != presheaf
    copied.evaluate(1).sum().backward()
    torch.testing.assert_close(tensor.grad, torch.tensor([3.0, 3.0]))
    with pytest.raises(dx.ValidationError, match="integer indices to tensors"):
        presheaf.with_(values={0: "bad"})
    with pytest.raises(dx.ValidationError, match="IdempotentSemiring"):
        presheaf.with_(algebra=PRODUCT_FUZZY)
