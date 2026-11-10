"""Tests for `quivers._values.ValueModel`: equality, hashing, freezing,
copying, and pickling over every declared field, opaque fields
included, under the per-field value and identity policies.
"""

from __future__ import annotations

import copy
import pickle
from typing import Literal

import didactic.api as dx
import numpy as np
import pytest
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel


class _Column(ValueModel):
    name: str
    data: np.ndarray = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class _Table(ValueModel):
    title: str = ""
    columns: tuple[_Column, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    arrays: dict[str, np.ndarray] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    levels: dict[str, tuple[str, ...]] = {}
    handle: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class _Node(ValueModel):
    label: str
    children: tuple = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)


class _Step(dx.TaggedUnion, ValueModel, discriminator="kind"):
    pass


class _Bind(_Step):
    kind: Literal["bind"] = "bind"
    name: str
    fn: torch.nn.Module = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def _table(**overrides: np.ndarray | str | tuple[_Column, ...]) -> _Table:
    fields: dict[str, np.ndarray | str | tuple[_Column, ...]] = {
        "title": "t",
        "columns": (_Column(name="x", data=np.arange(4.0)),),
    }
    fields.update(overrides)
    return _Table(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# arrays
# ---------------------------------------------------------------------------


def test_equal_arrays_built_separately_are_equal_and_hash_alike():
    left = _Column(name="x", data=np.arange(4.0))
    right = _Column(name="x", data=np.arange(4.0))
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


@pytest.mark.parametrize(
    "other",
    [
        np.arange(4.0) + 1.0,
        np.arange(4, dtype=np.int64),
        np.arange(4.0, dtype=np.float32),
        np.arange(4.0).reshape(2, 2),
        np.arange(5.0),
        np.array([-0.0, 1.0, 2.0, 3.0]),
    ],
    ids=["values", "int-dtype", "float32", "shape", "length", "negative-zero"],
)
def test_arrays_compare_exactly_by_dtype_shape_and_bytes(other: np.ndarray):
    base = _Column(name="x", data=np.array([0.0, 1.0, 2.0, 3.0]))
    changed = _Column(name="x", data=other)
    assert base != changed
    assert hash(base) != hash(changed)


def test_nan_arrays_with_equal_bits_are_equal():
    left = _Column(name="x", data=np.array([np.nan, 1.0]))
    right = _Column(name="x", data=np.array([np.nan, 1.0]))
    assert left == right
    assert hash(left) == hash(right)


def test_object_arrays_compare_by_element():
    left = _Column(name="x", data=np.array(["a", 1], dtype=object))
    right = _Column(name="x", data=np.array(["a", 1], dtype=object))
    other = _Column(name="x", data=np.array(["a", 2], dtype=object))
    assert left == right
    assert hash(left) == hash(right)
    assert left != other


def test_arrays_are_stored_as_read_only_copies():
    source = np.arange(4.0)
    column = _Column(name="x", data=source)
    before = hash(column)
    source[0] = 99.0
    assert column.data[0] == 0.0
    assert not column.data.flags.writeable
    with pytest.raises(ValueError):
        column.data[0] = 1.0
    assert hash(column) == before
    assert column == _Column(name="x", data=np.arange(4.0))


def test_non_contiguous_arrays_are_compared_by_content():
    strided = np.arange(8.0)[::2]
    assert _Column(name="x", data=strided) == _Column(
        name="x", data=np.array([0.0, 2.0, 4.0, 6.0])
    )


# ---------------------------------------------------------------------------
# nested models, mappings, typed fields
# ---------------------------------------------------------------------------


def test_nested_value_models_compare_elementwise():
    assert _table() == _table()
    other = _table(columns=(_Column(name="x", data=np.arange(4.0) * 2),))
    assert _table() != other
    assert hash(_table()) != hash(other)


def test_value_mappings_ignore_insertion_order_and_compare_values():
    left = _Table(arrays={"a": np.zeros(2), "b": np.ones(2)})
    right = _Table(arrays={"b": np.ones(2), "a": np.zeros(2)})
    changed = _Table(arrays={"a": np.zeros(2), "b": np.zeros(2)})
    assert left == right
    assert hash(left) == hash(right)
    assert left != changed


def test_value_mappings_are_read_only():
    source = {"a": np.zeros(2)}
    table = _Table(arrays=source)
    source["b"] = np.ones(2)
    assert set(table.arrays) == {"a"}
    with pytest.raises(TypeError):
        table.arrays["c"] = np.ones(2)  # type: ignore[index]


def test_typed_mappings_ignore_insertion_order():
    left = _Table(levels={"g": ("a", "b"), "h": ("c",)})
    right = _Table(levels={"h": ("c",), "g": ("a", "b")})
    assert left == right
    assert hash(left) == hash(right)
    assert left != _Table(levels={"g": ("b", "a"), "h": ("c",)})


def test_typed_fields_take_part_in_equality():
    assert _table(title="a") != _table(title="b")


# ---------------------------------------------------------------------------
# identity policy
# ---------------------------------------------------------------------------


def test_identity_fields_compare_by_object_identity():
    handle = torch.ones(2)
    assert _Table(handle=handle) == _Table(handle=handle)
    assert hash(_Table(handle=handle)) == hash(_Table(handle=handle))
    assert _Table(handle=handle) != _Table(handle=torch.ones(2))


def test_tagged_union_variants_use_value_semantics():
    module = torch.nn.Linear(1, 1)
    assert _Bind(name="a", fn=module) == _Bind(name="a", fn=module)
    assert _Bind(name="a", fn=module) != _Bind(name="a", fn=torch.nn.Linear(1, 1))
    assert _Bind(name="a", fn=module) != _Bind(name="b", fn=module)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_opaque_field_without_a_policy_is_rejected():
    class _Undeclared(ValueModel):
        payload: int = dx.field(default=0, opaque=True)

    with pytest.raises(dx.ValidationError, match="declares no equality policy"):
        _Undeclared()


@pytest.mark.parametrize(
    "arrays",
    [{"a": [1, 2]}, {"a": {1, 2}}, {"a": torch.ones(2)}],
    ids=["list", "set", "tensor"],
)
def test_value_fields_reject_values_without_a_content_key(arrays):
    with pytest.raises(dx.ValidationError, match="_Table.arrays"):
        _Table(arrays=arrays)


def test_with_refreezes_and_rehashes():
    table = _table()
    hash(table)
    source = np.arange(3.0)
    changed = table.with_(columns=(_Column(name="x", data=source),))
    source[0] = 5.0
    assert changed.columns[0].data[0] == 0.0
    assert changed != table
    assert hash(changed) == hash(
        _table(columns=(_Column(name="x", data=np.arange(3.0)),))
    )


# ---------------------------------------------------------------------------
# deep values, copy, pickle
# ---------------------------------------------------------------------------


def test_deep_nesting_compares_and_hashes_without_recursion():
    def chain(depth: int, leaf: str) -> _Node:
        node = _Node(label=leaf)
        for i in range(depth):
            node = _Node(label=str(i), children=(node,))
        return node

    assert chain(20_000, "a") == chain(20_000, "a")
    assert hash(chain(20_000, "a")) == hash(chain(20_000, "a"))
    assert chain(20_000, "a") != chain(20_000, "b")


def test_copy_deepcopy_and_pickle_keep_value_fields():
    table = _Table(
        title="t",
        columns=(_Column(name="x", data=np.arange(3.0)),),
        arrays={"a": np.ones(2)},
        levels={"g": ("a",)},
    )
    for clone in (
        copy.copy(table),
        copy.deepcopy(table),
        pickle.loads(pickle.dumps(table)),
    ):
        assert clone == table
        assert hash(clone) == hash(table)
        np.testing.assert_array_equal(clone.columns[0].data, np.arange(3.0))
        assert not clone.columns[0].data.flags.writeable


def test_copy_keeps_identity_fields_and_deepcopy_replaces_them():
    handle = torch.ones(2)
    table = _Table(handle=handle)
    assert copy.copy(table).handle is handle
    assert copy.copy(table) == table
    clone = copy.deepcopy(table)
    assert clone.handle is not handle
    torch.testing.assert_close(clone.handle, handle)
    assert clone != table
    restored = pickle.loads(pickle.dumps(table))
    torch.testing.assert_close(restored.handle, handle)


def test_tagged_union_variants_copy_and_pickle():
    step = _Bind(name="a", fn=torch.nn.Linear(1, 1))
    assert copy.copy(step) == step
    assert copy.copy(step).fn is step.fn
    restored = pickle.loads(pickle.dumps(step))
    assert isinstance(restored, _Bind)
    assert restored.name == "a"
