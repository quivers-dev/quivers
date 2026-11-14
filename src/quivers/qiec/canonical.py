"""Canonical type constructors shared by the kernel, prelude, and compiler.

These constructors live in the ``builtin`` namespace, so ``Site``,
``Sampleable``, ``Tensor``, and ``LogWeight`` mean the same type in every
module without any module declaring them. The prelude's handlers are typed
against them, the checker types distribution construction and log-density
evaluation with them, and the compiler elaborates a program's steps into
them.
"""

from __future__ import annotations

from quivers.qiec.identifiers import TypeId
from quivers.qiec.kinds import IndexBinder, ShapeSort, TypeBinder
from quivers.qiec.types import (
    IndexTerm,
    ShapeIndex,
    TypeApplication,
    TypeConstructorRef,
    TypeExpr,
    TypeVariable,
)


def builtin_constructor(
    name: str, *binders: TypeBinder | IndexBinder
) -> TypeConstructorRef:
    """A type constructor in the ``builtin`` namespace.

    Parameters
    ----------
    name : str
        The constructor's name.
    *binders : TypeBinder | IndexBinder
        Its kinding telescope.

    Returns
    -------
    TypeConstructorRef
        The constructor. Its identity derives from the namespace and the
        name alone, so it is the same constructor in every module.
    """
    return TypeConstructorRef(
        id=TypeId.derive("builtin", name),
        name=name,
        telescope=tuple(binders),
    )


ELEMENT_BINDER = TypeBinder(name="a")
"""The element-type binder ``a`` of the canonical constructors' telescopes."""

SHAPE_BINDER = IndexBinder(name="shape", sort=ShapeSort())
"""The shape-index binder of :data:`TENSOR_CONSTRUCTOR`'s telescope."""

ELEMENT = TypeVariable(name="a")
"""The type variable :data:`ELEMENT_BINDER` binds."""

SITE_CONSTRUCTOR = builtin_constructor("Site", ELEMENT_BINDER)
"""``Site[A]``: a named sample site whose value has type ``A``."""

SAMPLEABLE_CONSTRUCTOR = builtin_constructor("Sampleable", ELEMENT_BINDER)
"""``Sampleable[A]``: a distribution over values of type ``A``."""

TENSOR_CONSTRUCTOR = builtin_constructor("Tensor", ELEMENT_BINDER, SHAPE_BINDER)
"""``Tensor[A](shape)``: an array of ``A`` with a static shape index."""

LOG_WEIGHT = TypeApplication(constructor=builtin_constructor("LogWeight"))
"""``LogWeight``: a log-density contribution."""

BUILTIN_TYPE_CONSTRUCTORS: dict[str, TypeConstructorRef] = {
    "Site": SITE_CONSTRUCTOR,
    "Sampleable": SAMPLEABLE_CONSTRUCTOR,
    "Tensor": TENSOR_CONSTRUCTOR,
    "LogWeight": LOG_WEIGHT.constructor,
}
"""The canonical names a source type may refer to without declaring them."""


def site_type(element: TypeExpr) -> TypeApplication:
    """The type of a sample site producing ``element``.

    Parameters
    ----------
    element : TypeExpr
        The sampled value's type.

    Returns
    -------
    TypeApplication
        ``Site[element]``.
    """
    return TypeApplication(constructor=SITE_CONSTRUCTOR, arguments=(element,))


def sampleable_type(element: TypeExpr) -> TypeApplication:
    """The type of a distribution over ``element``.

    Parameters
    ----------
    element : TypeExpr
        The sampled value's type.

    Returns
    -------
    TypeApplication
        ``Sampleable[element]``.
    """
    return TypeApplication(constructor=SAMPLEABLE_CONSTRUCTOR, arguments=(element,))


def tensor_type(
    element: TypeExpr, dimensions: tuple[IndexTerm, ...]
) -> TypeApplication:
    """The type of an array of ``element`` with the given static shape.

    Parameters
    ----------
    element : TypeExpr
        The element type.
    dimensions : tuple[IndexTerm, ...]
        One index term per dimension, outermost first.

    Returns
    -------
    TypeApplication
        ``Tensor[element](shape)``.
    """
    return TypeApplication(
        constructor=TENSOR_CONSTRUCTOR,
        arguments=(element, ShapeIndex(dimensions=dimensions)),
    )


def sampled_element(type_: TypeExpr) -> TypeExpr | None:
    """The value type a ``Sampleable`` produces.

    Parameters
    ----------
    type_ : TypeExpr
        A type that may be a sampleable application.

    Returns
    -------
    TypeExpr | None
        The element type, or ``None`` when ``type_`` is not
        ``Sampleable[A]``.
    """
    if (
        isinstance(type_, TypeApplication)
        and type_.constructor == SAMPLEABLE_CONSTRUCTOR
        and len(type_.arguments) == 1
    ):
        argument = type_.arguments[0]
        if isinstance(argument, TypeApplication | TypeVariable):
            return argument
    return None


def tensor_shape(type_: TypeExpr) -> tuple[TypeExpr, tuple[IndexTerm, ...]] | None:
    """Split a tensor type into its element type and dimensions.

    Parameters
    ----------
    type_ : TypeExpr
        A type that may be a tensor application.

    Returns
    -------
    tuple[TypeExpr, tuple[IndexTerm, ...]] | None
        The element type and dimensions, or ``None`` when ``type_`` is
        not ``Tensor[A](shape)`` with a literal shape.
    """
    if (
        isinstance(type_, TypeApplication)
        and type_.constructor == TENSOR_CONSTRUCTOR
        and len(type_.arguments) == 2
    ):
        element, shape = type_.arguments
        if isinstance(element, TypeApplication | TypeVariable) and isinstance(
            shape, ShapeIndex
        ):
            return element, shape.dimensions
    return None


__all__ = [
    "BUILTIN_TYPE_CONSTRUCTORS",
    "ELEMENT",
    "ELEMENT_BINDER",
    "LOG_WEIGHT",
    "SAMPLEABLE_CONSTRUCTOR",
    "SHAPE_BINDER",
    "SITE_CONSTRUCTOR",
    "TENSOR_CONSTRUCTOR",
    "builtin_constructor",
    "sampleable_type",
    "sampled_element",
    "site_type",
    "tensor_shape",
    "tensor_type",
]
