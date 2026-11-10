"""Value semantics for didactic models that hold opaque fields.

didactic's `Model.__eq__` and `Model.__hash__` read only a model's
encoded storage. A field declared ``dx.field(opaque=True)`` is kept
out of that storage (its value lives by reference in a side table and
the storage carries a constant placeholder), so didactic equality and
hashing ignore it, and didactic's pickle hooks drop it. didactic also
encodes a ``dict`` field in insertion order, so two equal mappings
built in different orders encode differently.

`ValueModel` replaces those operations with ones derived from the
class's declared fields:

- A typed field enters the comparison key through its decoded value,
  with mappings keyed independently of insertion order.
- An opaque field enters through the policy it declares in its field
  extras, which every opaque field of a `ValueModel` must do:

  - `OPAQUE_BY_VALUE` (``"value"``): the value is compared by content.
    Scalars compare by type and value (``1``, ``1.0``, and ``True``
    differ); tuples elementwise; mappings by their key set and values;
    numpy arrays exactly, by dtype, shape, and bytes; nested models by
    their own equality. Arrays are stored as C-contiguous read-only
    copies and mappings as read-only proxies, made at construction, so
    neither the comparison nor the cached hash can go stale.
  - `OPAQUE_BY_IDENTITY` (``"identity"``): the value is compared by
    object identity. This is the policy for live runtime objects
    (tensors, modules, callables, AST nodes, dataframes) whose content
    comparison is undefined, ambiguous, or too costly.

Nested `ValueModel` instances found in a field are compared and hashed
one node at a time with an explicit stack, so deeply nested values
(long structural term lists, for instance) do not exhaust the
interpreter's recursion limit. A model's hash is computed once and
cached. Copying and pickling rebuild the model from its declared field
values, so opaque values survive both.

This module imports nothing from quivers, so every package can use it
without an import cycle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol, Self, cast

import didactic.api as dx
import numpy as np

__all__: list[str] = []


type OpaquePolicy = Literal["value", "identity"]
"""How an opaque field enters `ValueModel` equality and hashing."""

POLICY_KEY = "quivers.opaque_eq"
"""The field-extras key under which an opaque field declares its policy."""

OPAQUE_BY_VALUE: Mapping[str, OpaquePolicy] = MappingProxyType({POLICY_KEY: "value"})
"""Field extras declaring that an opaque field compares by content."""

OPAQUE_BY_IDENTITY: Mapping[str, OpaquePolicy] = MappingProxyType(
    {POLICY_KEY: "identity"}
)
"""Field extras declaring that an opaque field compares by identity."""

_HASH_ATTR = "_quivers_value_hash"


class HeldValue(Protocol):
    """Any Python value held in a model field.

    Every Python value satisfies this protocol; it names the values a
    field can hold without committing to their type.
    """


class _Marker(Enum):
    """The placeholder for a nested `ValueModel` inside a comparison key."""

    CHILD = "child"


_SCALAR_TYPES: tuple[type, ...] = (str, bytes, bool, int, float, complex, type(None))


class _ArrayKey:
    """The comparison key of a numpy array.

    Two keys are equal exactly when the arrays share dtype and shape
    and their bytes agree (for object arrays, when their elements'
    keys agree); the hash is a digest of the same data.
    """

    __slots__ = ("_array", "_elements")

    def __init__(self, array: np.ndarray) -> None:
        """Wrap a C-contiguous array.

        Parameters
        ----------
        array : np.ndarray
            The array to key.

        Raises
        ------
        TypeError
            If an object array holds an element without a value key.
        """
        self._array = np.ascontiguousarray(array)
        self._elements: tuple[Hashable, ...] | None = None
        if array.dtype.hasobject:
            children: list[ValueModel] = []
            self._elements = tuple(
                _value_key(cast("HeldValue", v), children)
                for v in self._array.ravel().tolist()
            )
            if children:
                raise TypeError(
                    "an object array compared by value cannot hold a model "
                    "with value semantics"
                )

    def __eq__(self, other: _ArrayKey | HeldValue) -> bool:  # type: ignore[override]
        """Compare two array keys exactly.

        Parameters
        ----------
        other : _ArrayKey or HeldValue
            The value to compare against.

        Returns
        -------
        bool
            ``True`` when dtype, shape, and data agree.
        """
        if not isinstance(other, _ArrayKey):
            return NotImplemented
        left, right = self._array, other._array
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        if self._elements is not None:
            return self._elements == other._elements
        return memoryview(left).cast("B") == memoryview(right).cast("B")

    def __hash__(self) -> int:
        """Hash the array's dtype, shape, and a digest of its data.

        Returns
        -------
        int
            The hash.
        """
        if self._elements is not None:
            return hash((self._array.dtype.str, self._array.shape, self._elements))
        digest = hashlib.blake2b(memoryview(self._array).cast("B"), digest_size=16)
        return hash((self._array.dtype.str, self._array.shape, digest.digest()))


def _value_key(value: HeldValue, children: list[ValueModel]) -> Hashable:
    """Build the content key of one value.

    Parameters
    ----------
    value : HeldValue
        The value to key.
    children : list[ValueModel]
        Accumulator for nested `ValueModel` instances, which are keyed
        by a placeholder and compared separately in positional order.

    Returns
    -------
    Hashable
        The key.

    Raises
    ------
    TypeError
        If the value (or a value nested in it) has no content key: a
        mutable list or set, or an object of a type outside the
        supported scalars, containers, arrays, classes, and models.
    """
    if isinstance(value, ValueModel):
        children.append(value)
        return _Marker.CHILD
    if isinstance(value, dx.Model):
        return value
    if isinstance(value, _SCALAR_TYPES):
        return (type(value), value)
    if isinstance(value, type):
        return value
    if isinstance(value, np.ndarray):
        return _ArrayKey(value)
    if isinstance(value, np.generic):
        return (type(value), value.item())
    if isinstance(value, tuple):
        items = cast("tuple[HeldValue, ...]", value)
        return (tuple, tuple(_value_key(v, children) for v in items))
    if isinstance(value, frozenset):
        members = cast("frozenset[HeldValue]", value)
        inner: list[ValueModel] = []
        keyed = frozenset(_value_key(v, inner) for v in members)
        if inner:
            raise TypeError("a frozenset compared by value cannot hold a model")
        return (frozenset, keyed)
    if isinstance(value, Mapping):
        mapping = cast("Mapping[str, HeldValue]", value)
        return (
            Mapping,
            tuple(
                (k, _value_key(mapping[k], children))
                for k in sorted(mapping, key=lambda k: (type(k).__name__, k))
            ),
        )
    raise TypeError(
        f"a value of type {type(value).__name__} has no content key; declare "
        "the field with identity semantics or hold an immutable value"
    )


def _freeze(value: HeldValue) -> HeldValue:
    """Return an immutable equivalent of a value compared by content.

    Arrays become C-contiguous read-only copies, mappings become
    read-only proxies over a copy, and tuples are frozen elementwise.

    Parameters
    ----------
    value : HeldValue
        The value to freeze.

    Returns
    -------
    HeldValue
        The frozen value.

    Raises
    ------
    TypeError
        If the value is a mutable list or set, or has no content key.
    """
    if isinstance(value, np.ndarray):
        if (
            value.flags.owndata
            and not value.flags.writeable
            and value.flags.c_contiguous
        ):
            return value
        frozen = np.array(value, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, tuple):
        items = cast("tuple[HeldValue, ...]", value)
        return tuple(_freeze(v) for v in items)
    if isinstance(value, Mapping) and not isinstance(value, dx.Model):
        mapping = cast("Mapping[str, HeldValue]", value)
        return MappingProxyType({k: _freeze(v) for k, v in mapping.items()})
    _value_key(value, [])
    return value


def _thaw(value: HeldValue) -> HeldValue:
    """Undo `_freeze`'s proxies so a value can be pickled and rebuilt.

    Parameters
    ----------
    value : HeldValue
        A frozen value.

    Returns
    -------
    HeldValue
        The value with every read-only mapping proxy replaced by a
        ``dict``.
    """
    if isinstance(value, tuple):
        items = cast("tuple[HeldValue, ...]", value)
        return tuple(_thaw(v) for v in items)
    if isinstance(value, MappingProxyType):
        mapping = cast("Mapping[str, HeldValue]", value)
        return {k: _thaw(v) for k, v in mapping.items()}
    return value


def opaque_policy(model_cls: type[dx.Model], field_name: str) -> OpaquePolicy | None:
    """Return the policy an opaque field declares, if any.

    Parameters
    ----------
    model_cls : type[dx.Model]
        The model class.
    field_name : str
        The field's name.

    Returns
    -------
    OpaquePolicy or None
        ``"value"`` or ``"identity"`` when the field declares one;
        ``None`` otherwise.
    """
    spec = model_cls.__field_specs__[field_name]
    policy = spec.extras.get(POLICY_KEY)
    if policy == "value":
        return "value"
    if policy == "identity":
        return "identity"
    return None


def _rebuild(model_cls: type[ValueModel], fields: dict[str, HeldValue]) -> ValueModel:
    """Construct a model from its declared field values.

    Parameters
    ----------
    model_cls : type[ValueModel]
        The class to construct.
    fields : dict[str, HeldValue]
        One value per declared field.

    Returns
    -------
    ValueModel
        The rebuilt model.
    """
    return model_cls(**fields)


class ValueModel(dx.Model):
    """A didactic model whose equality, hashing, copying, and pickling
    cover every declared field, opaque fields included.

    Subclasses declare each opaque field's policy through its extras,
    as ``dx.field(opaque=True, extras=OPAQUE_BY_VALUE)`` or
    ``dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)``; see the module
    docstring for what each policy compares. Construction rejects an
    opaque field without a declared policy, and freezes every field
    compared by value.
    """

    @dx.model_validator()
    def _freeze_value_fields(self) -> None:
        """Check opaque policies and freeze the fields compared by value.

        Raises
        ------
        TypeError
            If an opaque field declares no policy, or a field compared
            by value holds a value with no content key.
        """
        cls = type(self)
        side_table = cast(
            "dict[str, HeldValue]", object.__getattribute__(self, "_opaque_storage")
        )
        for name, spec in cls.__field_specs__.items():
            if not spec.is_opaque:
                continue
            policy = opaque_policy(cls, name)
            if policy is None:
                raise TypeError(
                    f"{cls.__name__}.{name} is opaque but declares no equality "
                    "policy; pass extras=OPAQUE_BY_VALUE or extras=OPAQUE_BY_IDENTITY"
                )
            if policy == "value":
                try:
                    side_table[name] = _freeze(side_table[name])
                except TypeError as exc:
                    raise TypeError(f"{cls.__name__}.{name}: {exc}") from exc

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Split this model into its local key and its nested value models.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            One key entry per declared field, in declaration order, and
            the nested `ValueModel` instances in the order the key
            mentions them.
        """
        cls = type(self)
        children: list[ValueModel] = []
        key: list[Hashable] = []
        for name, spec in cls.__field_specs__.items():
            value = cast("HeldValue", getattr(self, name))
            if spec.is_opaque and opaque_policy(cls, name) == "identity":
                key.append(("identity", id(value)))
            else:
                key.append(_value_key(value, children))
        return tuple(key), tuple(children)

    def __hash__(self) -> int:
        """Hash the model over every declared field, consistently with `__eq__`.

        Returns
        -------
        int
            The hash of the model's class, local key, and nested models'
            hashes, computed bottom-up and cached on each model.
        """
        cached = self.__dict__.get(_HASH_ATTR)
        if cached is not None:
            return cached
        stack: list[tuple[ValueModel, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if _HASH_ATTR in node.__dict__:
                continue
            key, children = node._local_key()
            if not expanded:
                stack.append((node, True))
                stack.extend(
                    (c, False) for c in children if _HASH_ATTR not in c.__dict__
                )
                continue
            child_hashes = tuple(c.__dict__[_HASH_ATTR] for c in children)
            object.__setattr__(node, _HASH_ATTR, hash((type(node), key, child_hashes)))
        return self.__dict__[_HASH_ATTR]

    def __eq__(self, other: Self | HeldValue) -> bool:  # type: ignore[override]
        """Compare two models over every declared field.

        Parameters
        ----------
        other : ValueModel or HeldValue
            The value to compare against.

        Returns
        -------
        bool
            ``True`` when ``other`` is of the same class and every field
            agrees under its policy, recursively through nested value
            models; ``NotImplemented`` when ``other`` is of another class.
        """
        if not isinstance(other, ValueModel) or type(other) is not type(self):
            return NotImplemented
        stack: list[tuple[ValueModel, ValueModel]] = [(self, other)]
        while stack:
            left, right = stack.pop()
            if left is right:
                continue
            if type(left) is not type(right) or hash(left) != hash(right):
                return False
            left_key, left_children = left._local_key()
            right_key, right_children = right._local_key()
            if left_key != right_key:
                return False
            stack.extend(zip(left_children, right_children, strict=True))
        return True

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[type[ValueModel], dict[str, HeldValue]], ValueModel],
        tuple[type[ValueModel], dict[str, HeldValue]],
    ]:
        """Rebuild the model from its declared field values.

        Returns
        -------
        tuple
            A callable and its arguments: the model class and one value
            per declared field, with read-only mapping proxies turned
            back into dicts.
        """
        cls = type(self)
        fields = {
            name: _thaw(cast("HeldValue", getattr(self, name)))
            for name in cls.__field_specs__
        }
        return (_rebuild, (cls, fields))
