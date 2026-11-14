"""Runtime identity and autograd contracts for structural value records."""

import copy

import didactic.api as dx
import pytest
import torch

from quivers.structural.encoder import PerOpFn
from quivers.structural.losses import LossEntry, LossRegistry
from quivers.structural.signature import Context, ContextEntry, Term


def test_context_preserves_nested_terms_and_embedding_gradients() -> None:
    """Scope extension and immutable updates preserve live tensor references."""
    embedding = torch.tensor([2.0, 3.0], requires_grad=True)
    annotation = Term(op="Arrow", args=(Term(op="A"), Term(op="B")))
    outer = Context().push("Term", embedding, annotation)
    inner = outer.push("Term", embedding * 2)
    assert outer.depth() == 1
    assert inner.depth() == 2
    assert inner.var(1) is embedding
    assert inner.type_of(1) == annotation
    assert inner.type_of(1).args == (Term(op="A"), Term(op="B"))
    updated = inner.with_(entries=inner.entries[1:])
    copied = copy.copy(updated)
    assert copied == outer
    assert hash(copied) == hash(outer)
    assert copied.var(0) is embedding
    inner.var(0).sum().backward()
    torch.testing.assert_close(embedding.grad, torch.tensor([2.0, 2.0]))
    different_tensor = outer.entries[0].with_(embedding=embedding.detach().clone())
    assert different_tensor != outer.entries[0]


def test_operation_record_retains_callable_and_parameter_identity() -> None:
    """A copied operation keeps its callable and its parameter gradients."""
    operation = torch.nn.Linear(2, 1, bias=False)
    record = PerOpFn(op="Node", mode="plain", args=("child",), fn=operation)
    updated = record.with_(op="Leaf")
    assert updated.fn is operation
    assert copy.copy(record) == record
    assert hash(copy.copy(record)) == hash(record)
    assert record.with_(fn=torch.nn.Linear(2, 1, bias=False)) != record
    updated.fn(torch.tensor([2.0, 3.0])).sum().backward()
    torch.testing.assert_close(operation.weight.grad, torch.tensor([[2.0, 3.0]]))
    with pytest.raises(dx.ValidationError, match="callable"):
        record.with_(fn=3)


def test_context_rejects_invalid_runtime_fields() -> None:
    """Model updates enforce opaque field types rather than storing bad state."""
    entry = ContextEntry(var_sort="Term", embedding=torch.zeros(2))
    with pytest.raises(dx.ValidationError, match="torch.Tensor"):
        entry.with_(embedding=(0.0, 0.0))
    with pytest.raises(dx.ValidationError, match="Term or None"):
        entry.with_(type_term="A")
    with pytest.raises(dx.ValidationError, match="tuple of ContextEntry"):
        Context(entries=("invalid",))


def test_loss_records_preserve_callable_gradients_and_registry_mutation() -> None:
    """Weighted evaluation agrees with its scalar formula and exact gradient."""
    parameter = torch.tensor(3.0, requires_grad=True)
    entry = LossEntry(name="square", body=lambda env: parameter.square())
    weighted = entry.with_(weight=lambda env: parameter.new_tensor(2.0))
    registry = LossRegistry()
    other = LossRegistry()
    registry.add(weighted)
    copied = copy.copy(registry)
    assert copied.entries is registry.entries
    assert copied == registry
    assert other.entries == []
    assert weighted.body is entry.body
    assert copy.copy(weighted) == weighted
    total = registry.evaluate()
    torch.testing.assert_close(total, torch.tensor(18.0), rtol=0, atol=0)
    total.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(12.0), rtol=0, atol=0)
    assert total != entry.body({})
    with pytest.raises(dx.ValidationError, match="callable"):
        entry.with_(body=3)
    with pytest.raises(dx.ValidationError, match="LossEntry"):
        registry.with_(entries=["invalid"])
