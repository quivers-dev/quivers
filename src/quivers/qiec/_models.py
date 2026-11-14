"""Validated immutable records for recursive kernel graphs."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from functools import cache
from inspect import get_annotations
import sys
from typing import Literal, TypeAliasType, Union, cast, get_args, get_origin

import didactic.api as dx

from quivers._values import HeldValue, ValueModel, _value_key


@cache
def field_types(class_: type[dx.Model]) -> dict[str, HeldValue]:
    """Resolve the declared types of one recursive kernel record.

    Parameters
    ----------
    class_ : type[didactic.api.Model]
        The record whose annotations are resolved in its defining module.

    Returns
    -------
    dict[str, HeldValue]
        Resolved field annotations, including recursive type aliases.
    """
    namespace = vars(sys.modules[class_.__module__])
    annotations = get_annotations(
        class_, globals=namespace, locals=namespace, eval_str=True
    )
    return {name: annotations[name] for name in class_.__field_specs__}


def validate_shape(value: HeldValue, annotation: HeldValue, *, path: str) -> None:
    """Validate a field against the closed kernel schema.

    Nested records have already validated their own fields, so this check
    inspects their classes without traversing their graphs again.

    Parameters
    ----------
    value : HeldValue
        The field value.
    annotation : HeldValue
        Its declared Python type or recursive alias.
    path : str
        The field path used in diagnostics.

    Raises
    ------
    TypeError
        If the value does not inhabit its annotation, or the annotation
        is outside the closed scalar, sequence, mapping, union, and model schema.
    """
    if annotation is HeldValue:
        return
    if isinstance(annotation, TypeAliasType):
        validate_shape(value, annotation.__value__, path=path)
        return
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if any(type(value) is type(item) and value == item for item in arguments):
            return
    elif origin is Union:
        for candidate in arguments:
            try:
                validate_shape(value, candidate, path=path)
            except TypeError:
                continue
            return
    elif origin is tuple:
        if isinstance(value, tuple):
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                for index, item in enumerate(value):
                    validate_shape(item, arguments[0], path=f"{path}[{index}]")
                return
            if len(value) == len(arguments):
                for index, (item, item_type) in enumerate(
                    zip(value, arguments, strict=True)
                ):
                    validate_shape(item, item_type, path=f"{path}[{index}]")
                return
    elif origin in {list, frozenset}:
        if isinstance(value, origin):
            for item in value:
                validate_shape(item, arguments[0], path=f"{path}.member")
            return
    elif origin in {dict, Mapping}:
        if isinstance(value, origin):
            key_type, value_type = arguments
            for key, item in value.items():
                validate_shape(key, key_type, path=f"{path}.key")
                validate_shape(item, value_type, path=f"{path}[{key!r}]")
            return
    elif isinstance(annotation, type):
        primitive = annotation in {bool, bytes, float, int, str, type(None)}
        if type(value) is annotation if primitive else isinstance(value, annotation):
            return
    else:
        raise TypeError(f"unsupported kernel annotation {annotation!s} at {path}")
    raise TypeError(
        f"invalid field {path}: expected {annotation!s}, got {type(value).__name__}"
    )


class QiecModel(ValueModel):
    """A kernel record with typed recursive fields and structural equality.

    Recursive fields retain their child models by value rather than
    flattening their runtime storage into a parent's encoded fields.
    Construction and immutable updates validate those field types.
    """

    @dx.model_validator()
    def _validate_recursive_fields(self) -> None:
        """Validate recursive fields against their resolved annotations.

        Raises
        ------
        TypeError
            If an opaque field does not inhabit its declared kernel type.
        """
        cls = type(self)
        annotations = field_types(cls)
        for name, spec in cls.__field_specs__.items():
            if spec.is_opaque:
                validate_shape(
                    getattr(self, name),
                    annotations[name],
                    path=f"{cls.__name__}.{name}",
                )

    def _comparison_key(
        self, *values: HeldValue
    ) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Build a structural key for the fields defining nominal identity.

        Parameters
        ----------
        values : HeldValue
            The fields that equality compares, in order.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Local comparison keys and recursively compared child models.
        """
        children: list[ValueModel] = []
        return tuple(_value_key(value, children) for value in values), tuple(children)

    def _normalize_field(self, name: str, value: HeldValue) -> None:
        """Store a canonical recursive field during model validation.

        Parameters
        ----------
        name : str
            The declared opaque field to normalize.
        value : HeldValue
            Its validated canonical value.

        Raises
        ------
        TypeError
            If the field is encoded or the normalized value has the wrong type.
        """
        cls = type(self)
        if not cls.__field_specs__[name].is_opaque:
            raise TypeError(f"{name} is not a recursive kernel field")
        validate_shape(value, field_types(cls)[name], path=f"{cls.__name__}.{name}")
        storage = cast(
            "dict[str, HeldValue]", object.__getattribute__(self, "_opaque_storage")
        )
        storage[name] = value
