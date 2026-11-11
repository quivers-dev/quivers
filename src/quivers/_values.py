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
    an object exporting a read-only buffer (a numpy array, say) exactly,
    by its element format, shape, and bytes in logical order; nested
    models by their own equality. A buffer must be read-only when the
    model is built, and mappings are stored as read-only proxies, so
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
import struct
import sys
from collections.abc import Callable, Hashable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol, Self, cast

import didactic.api as dx

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

FREEZE_KEY = "quivers.opaque_freeze"
"""The field-extras key under which a value field names its freezer."""

type Freezer = Callable[[HeldValue], HeldValue]
"""A function returning an immutable equivalent of a field's value."""


def opaque_by_value[V: HeldValue](
    freeze: Callable[[V], V],
) -> Mapping[str, OpaquePolicy | Callable[[V], V]]:
    """Field extras for a value field whose owner freezes its values.

    A buffer compared by value must be read-only, and only the library
    that made it can copy it read-only (a numpy array, by a copy whose
    writeable flag is cleared). A field holding such buffers names the
    function that does so; the model applies it to the field's value
    when it is built, before checking the value is immutable.

    Parameters
    ----------
    freeze : Callable[[V], V]
        The function returning an immutable equivalent of a value.

    Returns
    -------
    Mapping[str, OpaquePolicy | Callable[[V], V]]
        The extras: the value policy and the freezer.
    """
    return MappingProxyType({POLICY_KEY: "value", FREEZE_KEY: freeze})


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


#: The struct codes of integer elements, signed and unsigned. Codes of
#: one signedness and size denote the same element type (an exporter
#: may write a 64-bit integer as ``l`` or ``q``), so a buffer's integer
#: format is keyed by signedness and size.
_SIGNED_CODES = frozenset("bhilqn")
_UNSIGNED_CODES = frozenset("BHILQN")

#: The byte-order prefixes that mean the platform's own order.
_NATIVE_ORDER = frozenset(
    {"", "@", "="} | ({"<"} if sys.byteorder == "little" else {">", "!"})
)


def _element_format(view: memoryview) -> Hashable:
    """The element type a buffer's format string names.

    Parameters
    ----------
    view : memoryview
        The buffer.

    Returns
    -------
    Hashable
        The format, with a native byte-order prefix dropped and an
        integer code read as its signedness and size.
    """
    fmt = view.format
    order = fmt[0] if fmt and fmt[0] in "@=<>!" else ""
    code = fmt[len(order) :]
    native = order in _NATIVE_ORDER
    if code in _SIGNED_CODES or code in _UNSIGNED_CODES:
        signed = code in _SIGNED_CODES
        return ("integer", native or order, signed, struct.calcsize(order + code))
    return ("format", native or order, code)


class _BufferKey:
    """The comparison key of a read-only buffer, a numpy array say.

    Two keys are equal exactly when the buffers share element type and
    shape and their bytes in logical (C) order agree; the hash is a
    digest of the same data. A buffer of Python object references has
    no key: its bytes are addresses, not values.
    """

    __slots__ = ("_bytes", "_format", "_shape")

    def __init__(self, view: memoryview) -> None:
        """Key a buffer by its element type, shape, and bytes.

        Parameters
        ----------
        view : memoryview
            The buffer, read in logical order whatever its strides.

        Raises
        ------
        TypeError
            If the buffer holds Python object references.
        """
        if "O" in view.format:
            raise TypeError(
                "an array of Python objects compared by value has no content "
                "key; hold numbers, or declare the field with identity semantics"
            )
        self._format = _element_format(view)
        self._shape = tuple(view.shape or ())
        self._bytes = view.tobytes()

    def __eq__(self, other: _BufferKey | HeldValue) -> bool:  # type: ignore[override]
        """Compare two buffer keys exactly.

        Parameters
        ----------
        other : _BufferKey or HeldValue
            The value to compare against.

        Returns
        -------
        bool
            ``True`` when element type, shape, and bytes agree.
        """
        if not isinstance(other, _BufferKey):
            return NotImplemented
        return (
            self._format == other._format
            and self._shape == other._shape
            and self._bytes == other._bytes
        )

    def __hash__(self) -> int:
        """Hash the element type, shape, and a digest of the bytes.

        Returns
        -------
        int
            The hash.
        """
        digest = hashlib.blake2b(self._bytes, digest_size=16).digest()
        return hash((self._format, self._shape, digest))


def _buffer(value: HeldValue) -> memoryview | None:
    """The buffer a value exports, if it exports one.

    Parameters
    ----------
    value : HeldValue
        The value.

    Returns
    -------
    memoryview | None
        A view of the value's buffer, or ``None`` for a value that
        exports none.
    """
    try:
        return memoryview(cast("bytes", value))
    except TypeError:
        return None


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
        mutable list or set, a buffer of Python objects, or an object of
        a type outside the supported scalars, containers, buffers,
        classes, and models.
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
    view = _buffer(value)
    if view is not None:
        return _BufferKey(view)
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

    A buffer (an array, say) must already be read-only, since only its
    owner can make a read-only copy of it; mappings become read-only
    proxies over a copy, and tuples are frozen elementwise.

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
        If the value is a mutable list or set, a writable buffer, or has
        no content key.
    """
    view = _buffer(value)
    if view is not None:
        if not view.readonly:
            raise TypeError(
                "a buffer compared by value must be read-only when the model "
                "is built; pass a read-only copy (for a numpy array, one whose "
                "writeable flag is cleared)"
            )
        _BufferKey(view)
        return value
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
    ``dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)``, or, for a
    field holding buffers its owner freezes,
    ``dx.field(opaque=True, extras=opaque_by_value(freeze))``; see the
    module docstring for what each policy compares. Construction rejects
    an opaque field without a declared policy, applies each field's
    freezer, and freezes every field compared by value.
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
                freezer = spec.extras.get(FREEZE_KEY)
                try:
                    value = side_table[name]
                    if callable(freezer):
                        value = cast("Freezer", freezer)(value)
                    side_table[name] = _freeze(value)
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
