"""Validated immutable records for recursive kernel graphs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from functools import cache
from inspect import get_annotations
import sys
from typing import (
    Literal,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
    get_protocol_members,
    is_protocol,
)

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


@cache
def _model_classes(annotation: HeldValue) -> tuple[type[dx.Model], ...] | None:
    """Resolve a closed union of model types without inspecting model values.

    Parameters
    ----------
    annotation : HeldValue
        A field annotation or recursive alias.

    Returns
    -------
    tuple[type[didactic.api.Model], ...] or None
        The model classes in the union, or ``None`` when other forms need
        the general shape validator.
    """
    if isinstance(annotation, TypeAliasType):
        return _model_classes(annotation.__value__)
    if get_origin(annotation) is Union:
        groups = tuple(_model_classes(item) for item in get_args(annotation))
        if all(group is not None for group in groups):
            return tuple(cls for group in groups if group is not None for cls in group)
        return None
    if isinstance(annotation, type) and issubclass(annotation, dx.Model):
        return (annotation,)
    return None


@cache
def _shape_predicate(annotation: HeldValue) -> Callable[[HeldValue], bool]:
    """Compile a field's type check independently of its values.

    Parameters
    ----------
    annotation : HeldValue
        The declared field type.

    Returns
    -------
    Callable[[HeldValue], bool]
        A predicate that checks the complete field shape. Recursive aliases
        resolve lazily, and no model or runtime value is retained in this cache.
    """
    if annotation is HeldValue:
        return lambda value: True
    classes = _model_classes(annotation)
    if classes is not None:
        return lambda value: isinstance(value, classes)
    if isinstance(annotation, TypeAliasType):
        return lambda value: _shape_predicate(annotation.__value__)(value)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return lambda value: any(
            type(value) is type(item) and value == item for item in arguments
        )
    if origin is Union:
        alternatives = tuple(_shape_predicate(item) for item in arguments)
        return lambda value: any(check(value) for check in alternatives)
    if origin is tuple:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            check = _shape_predicate(arguments[0])
            return lambda value: (
                isinstance(value, tuple) and all(check(item) for item in value)
            )
        checks = tuple(_shape_predicate(item) for item in arguments)
        return lambda value: (
            isinstance(value, tuple)
            and len(value) == len(checks)
            and all(check(item) for check, item in zip(checks, value, strict=True))
        )
    if origin in {list, set, frozenset}:
        container = cast(type, origin)
        check = _shape_predicate(arguments[0])
        return lambda value: (
            isinstance(value, container)
            and all(check(item) for item in cast(list[HeldValue], value))
        )
    if origin in {dict, Mapping}:
        key_check, value_check = (_shape_predicate(item) for item in arguments)
        return lambda value: (
            isinstance(value, Mapping)
            and (origin is Mapping or isinstance(value, dict))
            and all(key_check(key) and value_check(item) for key, item in value.items())
        )
    if origin is Callable or annotation is Callable:
        return callable
    if isinstance(annotation, type) and is_protocol(annotation):
        annotations = get_annotations(annotation, eval_str=True)
        members = tuple(
            (
                name,
                _shape_predicate(annotations[name])
                if name in annotations
                else callable
                if callable(getattr(annotation, name, None))
                else None,
            )
            for name in get_protocol_members(annotation)
        )

        def matches_protocol(value: HeldValue) -> bool:
            """Check the protocol's declared attributes and methods.

            Parameters
            ----------
            value : HeldValue
                The candidate runtime value.

            Returns
            -------
            bool
                Whether all declared members have the required shapes.
            """
            for name, check in members:
                try:
                    member = getattr(value, name)
                except AttributeError:
                    return False
                if check is not None and not check(member):
                    return False
            return True

        return matches_protocol
    if isinstance(annotation, type):
        if annotation in {bool, bytes, float, int, str, type(None)}:
            return lambda value: type(value) is annotation
        return lambda value: isinstance(value, annotation)
    return lambda value: False


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
        is outside the scalar, sequence, mapping, union, model, callable,
        and protocol schema.
    """
    if _shape_predicate(annotation)(value):
        return
    model_classes = _model_classes(annotation)
    if model_classes is not None:
        if isinstance(value, model_classes):
            return
        raise TypeError(
            f"invalid field {path}: expected {annotation!s}, got {type(value).__name__}"
        )
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
    elif origin in {list, set, frozenset}:
        if isinstance(value, (list, set, frozenset)) and isinstance(
            value, cast(type, origin)
        ):
            for item in value:
                validate_shape(item, arguments[0], path=f"{path}.member")
            return
    elif origin in {dict, Mapping}:
        if isinstance(value, Mapping) and (
            origin is Mapping or isinstance(value, dict)
        ):
            key_type, value_type = arguments
            for key, item in value.items():
                validate_shape(key, key_type, path=f"{path}.key")
                validate_shape(item, value_type, path=f"{path}[{key!r}]")
            return
    elif origin is Callable or annotation is Callable:
        if callable(value):
            return
    elif isinstance(annotation, type) and is_protocol(annotation):
        annotations = get_annotations(annotation, eval_str=True)
        for name in get_protocol_members(annotation):
            try:
                member = getattr(value, name)
            except AttributeError:
                break
            if name in annotations:
                validate_shape(member, annotations[name], path=f"{path}.{name}")
            elif callable(getattr(annotation, name, None)) and not callable(member):
                break
        else:
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
