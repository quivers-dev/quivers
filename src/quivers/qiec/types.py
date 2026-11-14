"""Static terms for the Quivers Indexed Effect Core (QIEC).

The module intentionally models only the kernel's small static language.
Didactic and Panproto check a first-order projection of its indexed-family
declarations, while these frozen records retain the QIEC-specific distinction
between uniform parameters and refinable indices.
"""

from __future__ import annotations

from collections.abc import Hashable
from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import QiecModel
import didactic.api as dx

from typing import Literal

from quivers.qiec.identifiers import EffectId, StaticVariableId, TypeId
from quivers.qiec.kinds import (
    EFFECT,
    NAT,
    TYPE,
    ContextSort,
    EffectBinder,
    IndexBinder,
    IndexSort,
    Kind,
    NatSort,
    ShapeSort,
    Telescope,
    TypeBinder,
    UserIndexSort,
)


class IndexVariable(QiecModel):
    """A variable standing for an index term.

    Parameters
    ----------
    name
        The variable's display name.
    sort
        The closed index sort of the terms it stands for.
    identity
        The rigid identity of a case branch's skolem, or ``None`` for a
        declaration binder matched by name.
    tag
        The serialization discriminator; always ``"index_variable"``.
    """

    name: str
    sort: IndexSort = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    identity: StaticVariableId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["index_variable"] = "index_variable"

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Key the fields that define this kernel object's identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Its nominal fields and recursively compared children.
        """
        return self._comparison_key(
            self.sort, self.identity if self.identity is not None else self.name
        )


class IndexLiteral(QiecModel):
    """A closed index literal.

    Parameters
    ----------
    value
        A natural number for the nat sort, or a nullary constructor name for
        a user-defined index sort.
    sort
        The sort the literal inhabits.
    tag
        The serialization discriminator; always ``"index_literal"``.
    """

    value: int | str
    sort: IndexSort = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["index_literal"] = "index_literal"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Check the literal inhabits the sort it claims.

        Raises
        ------
        ValueError
            If a `Nat` literal is not a nonnegative integer, if a user
            sort's literal names a constructor that sort does not
            declare, or if it names one that takes arguments. The last is
            the subtle case: a nullary constructor is a literal, while
            one with arguments has to be applied.
        """

        if isinstance(self.sort, NatSort):
            if not isinstance(self.value, int) or self.value < 0:
                raise ValueError("Nat literals must be nonnegative integers")
        if (
            isinstance(self.sort, UserIndexSort)
            and self.value not in self.sort.constructors
        ):
            raise ValueError(
                f"{self.value!r} is not a constructor of index sort {self.sort.name!r}"
            )
        if (
            isinstance(self.sort, UserIndexSort)
            and isinstance(self.value, str)
            and self.sort.constructor_arity(self.value) != 0
        ):
            raise ValueError(f"index constructor {self.value!r} is not nullary")


class IndexConstructor(QiecModel):
    """A fully applied constructor of a user-defined index sort.

    Parameters
    ----------
    name
        The constructor's name, which must belong to ``sort``.
    arguments
        The constructor's index arguments, in order.
    sort
        The sort the constructor builds.
    tag
        The serialization discriminator; always ``"index_constructor"``.
    """

    name: str
    arguments: tuple[IndexTerm, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sort: IndexSort = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["index_constructor"] = "index_constructor"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Check the constructor belongs to its sort and is fully applied.

        Raises
        ------
        ValueError
            If the sort is not a user-defined one, if it declares no such
            constructor, or if the argument count differs from the
            declared arity. A partially applied index constructor is
            rejected rather than curried, since an index is a value of
            its sort and a partial application is not one.
        """

        if not isinstance(self.sort, UserIndexSort):
            raise ValueError("named index constructors require a user index sort")
        expected = self.sort.constructor_arity(self.name)
        if len(self.arguments) != expected:
            raise ValueError(
                f"index constructor {self.name!r} expects {expected} arguments, "
                f"got {len(self.arguments)}"
            )
        for argument in self.arguments:
            if index_sort(argument) != self.sort:
                raise ValueError(
                    f"argument of index constructor {self.name!r} has the wrong sort"
                )


class ShapeIndex(QiecModel):
    """A shape given by its dimensions.

    Parameters
    ----------
    dimensions
        One index term per dimension, outermost first.
    tag
        The serialization discriminator; always ``"shape_index"``.
    """

    dimensions: tuple[IndexTerm, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["shape_index"] = "shape_index"


type IndexTerm = IndexVariable | IndexLiteral | IndexConstructor | ShapeIndex
"""A static index: a variable, a literal, a constructor application, or a shape."""


class TypeVariable(QiecModel):
    """A variable standing for a type or effect-kinded static argument.

    Parameters
    ----------
    name
        The variable's display name.
    kind
        The kind of the arguments it stands for.
    identity
        The rigid identity of a case branch's skolem, or ``None`` for a
        declaration binder matched by name.
    tag
        The serialization discriminator; always ``"type_variable"``.
    """

    name: str
    kind: Kind = dx.field(default=TYPE, opaque=True, extras=OPAQUE_BY_VALUE)
    identity: StaticVariableId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["type_variable"] = "type_variable"

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Key the fields that define this kernel object's identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Its nominal fields and recursively compared children.
        """
        return self._comparison_key(
            self.kind, self.identity if self.identity is not None else self.name
        )


class TypeConstructorRef(QiecModel):
    """A fully qualified type constructor and its kinding telescope.

    Parameters
    ----------
    id
        The constructor's stable identity, which alone determines equality.
    name
        The constructor's display name.
    telescope
        The binders the constructor's arguments instantiate, in order.
    """

    id: TypeId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)

    @classmethod
    def builtin(cls, name: str) -> TypeConstructorRef:
        """A constructor for one of the language's own types.

        Parameters
        ----------
        name : str
            The builtin's name.

        Returns
        -------
        TypeConstructorRef
            A nullary constructor whose identity derives from the
            ``builtin`` namespace, so `Int` means the same type in every
            module without being declared in any of them.
        """
        return cls(id=TypeId.derive("builtin", name), name=name)

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Key the fields that define this kernel object's identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Its nominal fields and recursively compared children.
        """
        return self._comparison_key(self.id)


class TypeApplication(QiecModel):
    """A type constructor applied to static arguments.

    Parameters
    ----------
    constructor
        The constructor applied.
    arguments
        The arguments instantiating the constructor's telescope, in order;
        empty for a nullary constructor.
    tag
        The serialization discriminator; always ``"type_application"``.
    """

    constructor: TypeConstructorRef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arguments: tuple[StaticArgument, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["type_application"] = "type_application"


class FunctionType(QiecModel):
    """A pure function type.

    Effectful codomains are represented explicitly by
    :class:`quivers.qiec.effects.ComputationType`, keeping the value and
    computation strata separate.

    Parameters
    ----------
    parameter
        The argument type.
    result
        The result type.
    tag
        The serialization discriminator; always ``"function_type"``.
    """

    parameter: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["function_type"] = "function_type"


class EqualityType(QiecModel):
    """The proposition that two static arguments are equal.

    Parameters
    ----------
    kind
        The kind or index sort both sides inhabit.
    left
        The left side of the equation.
    right
        The right side of the equation.
    tag
        The serialization discriminator; always ``"equality_type"``.
    """

    kind: Kind | IndexSort = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    left: StaticArgument = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    right: StaticArgument = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["equality_type"] = "equality_type"


type TypeExpr = TypeVariable | TypeApplication | FunctionType | EqualityType
"""A type: a variable, a constructor application, a function type, or an equality."""


class EffectVariable(QiecModel):
    """A variable standing for an effect interface application.

    Parameters
    ----------
    name
        The variable's display name.
    identity
        The rigid identity of a case branch's skolem, or ``None`` for a
        declaration binder matched by name.
    tag
        The serialization discriminator; always ``"effect_variable"``.
    """

    name: str
    identity: StaticVariableId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["effect_variable"] = "effect_variable"

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Key the fields that define this kernel object's identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Its nominal fields and recursively compared children.
        """
        return self._comparison_key(
            self.identity if self.identity is not None else self.name
        )


class EffectRef(QiecModel):
    """One closed effect interface application.

    Parameters
    ----------
    id
        The stable identity of the interface declaration.
    name
        The interface's display name.
    arguments
        The arguments instantiating the interface's telescope, in order.
    tag
        The serialization discriminator; always ``"effect_ref"``.
    """

    id: EffectId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    arguments: tuple[StaticArgument, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["effect_ref"] = "effect_ref"

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Key the fields that define this kernel object's identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Its nominal fields and recursively compared children.
        """
        return self._comparison_key(self.id, self.arguments)


type StaticArgument = TypeExpr | IndexTerm | EffectRef | EffectVariable
"""Anything a telescope binder may be instantiated at: a type, an index, or an effect."""


UNIT = TypeApplication(constructor=TypeConstructorRef.builtin("Unit"))
"""The unit type, inhabited by ``None`` alone."""
BOOL = TypeApplication(constructor=TypeConstructorRef.builtin("Bool"))
"""The Boolean type."""
INT = TypeApplication(constructor=TypeConstructorRef.builtin("Int"))
"""The integer type."""
REAL = TypeApplication(constructor=TypeConstructorRef.builtin("Real"))
"""The real-number type."""
STRING = TypeApplication(constructor=TypeConstructorRef.builtin("String"))
"""The string type."""


def product_type(*components: TypeExpr) -> TypeApplication:
    """Construct the canonical finite-product type of the given arity.

    Parameters
    ----------
    components : TypeExpr
        The component types, in order.

    Returns
    -------
    TypeApplication
        The product. Its constructor identity derives from the arity, so
        every two-component product in every module is the same type
        constructor and two products of different arity are different
        ones.
    """

    arity = len(components)
    constructor = TypeConstructorRef(
        id=TypeId.derive("builtin", "Product", arity),
        name=f"Product{arity}",
        telescope=tuple(
            TypeBinder(name=f"item{position}") for position in range(arity)
        ),
    )
    return TypeApplication(constructor=constructor, arguments=components)


def static_kind(term: StaticArgument) -> Kind:
    """Return the outer kind of a static term.

    Index sorts are deliberately not collapsed into ``Type``: callers that
    validate an index binder compare the term's concrete ``sort`` separately.

    Parameters
    ----------
    term : StaticArgument
        The static term to classify.

    Returns
    -------
    Kind
        The term's kind.

    Raises
    ------
    TypeError
        If the term is an index, which carries a sort rather than a kind.
        Raising rather than returning a placeholder is what keeps a
        caller from comparing an index against a kind and finding them
        equal.
    """

    if isinstance(term, (EffectRef, EffectVariable)):
        return EFFECT
    if isinstance(term, TypeVariable):
        return term.kind
    if isinstance(term, (TypeApplication, FunctionType, EqualityType)):
        return TYPE
    raise TypeError("index terms have a sort, not a QIEC kind")


def index_sort(term: IndexTerm) -> IndexSort:
    """Return an index term's sort, checking shape dimensions.

    Parameters
    ----------
    term : IndexTerm
        The index whose sort is wanted.

    Returns
    -------
    IndexSort
        The term's sort. A shape reports its concrete rank, which is what
        lets a rank-polymorphic binder accept it while a binder of a
        fixed rank does not.

    Raises
    ------
    TypeError
        If a shape dimension is not of `Nat` sort. A dimension is a
        count, so admitting anything else would let a shape be indexed by
        a value that cannot be one.
    """

    if isinstance(term, ShapeIndex):
        for dimension in term.dimensions:
            if index_sort(dimension) != NAT:
                raise TypeError("shape dimensions must have Nat sort")
        return ShapeSort(rank=len(term.dimensions))
    return term.sort


def _sort_matches(expected: object, actual: object) -> bool:
    """Whether an index of one sort satisfies a binder of another.

    Parameters
    ----------
    expected : object
        The sort the binder declares.
    actual : object
        The sort the supplied index carries.

    Returns
    -------
    bool
        True when the index is acceptable. Sorts match exactly, except
        that a shape sort of unspecified rank accepts any rank, which is
        what makes a rank-polymorphic binder usable.
    """
    if isinstance(expected, ShapeSort) and isinstance(actual, ShapeSort):
        return expected.rank is None or expected.rank == actual.rank
    return expected == actual


def validate_static_argument(term: StaticArgument) -> None:
    """Recursively validate one intrinsically kinded static argument.

    Parameters
    ----------
    term : StaticArgument
        The static term to validate, together with everything nested in
        it.

    Raises
    ------
    TypeError
        If a term is of a class that cannot appear in a static position.
    ValueError
        If an application is ill-kinded, or an index constructor is
        applied at the wrong arity.
    """
    if isinstance(term, TypeVariable | EffectVariable | IndexVariable | IndexLiteral):
        return
    if isinstance(term, TypeApplication):
        check_static_arguments(term.constructor.telescope, term.arguments)
        return
    if isinstance(term, FunctionType):
        validate_static_argument(term.parameter)
        validate_static_argument(term.result)
        if static_kind(term.parameter) != TYPE:
            raise TypeError("function parameter must have Type kind")
        if static_kind(term.result) != TYPE:
            raise TypeError("function result must have Type kind")
        return
    if isinstance(term, EqualityType):
        validate_static_argument(term.left)
        validate_static_argument(term.right)
        index_kinds = (NatSort, ShapeSort, ContextSort, UserIndexSort)
        index_nodes = (IndexVariable, IndexLiteral, IndexConstructor, ShapeIndex)
        if isinstance(term.kind, index_kinds):
            if not isinstance(term.left, index_nodes) or not isinstance(
                term.right, index_nodes
            ):
                raise TypeError("index equality endpoints must be index terms")
            if not _sort_matches(term.kind, index_sort(term.left)) or not _sort_matches(
                term.kind, index_sort(term.right)
            ):
                raise TypeError("index equality endpoint has the wrong sort")
        elif (
            static_kind(term.left) != term.kind or static_kind(term.right) != term.kind
        ):
            raise TypeError("equality endpoint has the wrong kind")
        return
    if isinstance(term, EffectRef):
        for argument in term.arguments:
            validate_static_argument(argument)
        return
    if isinstance(term, IndexConstructor):
        for argument in term.arguments:
            validate_static_argument(argument)
        index_sort(term)
        return
    if isinstance(term, ShapeIndex):
        index_sort(term)
        for dimension in term.dimensions:
            validate_static_argument(dimension)
        return
    raise TypeError(f"unknown static argument {term!r}")


def check_static_arguments(
    telescope: Telescope,
    arguments: tuple[StaticArgument, ...],
) -> tuple[
    tuple[tuple[str, TypeExpr], ...],
    tuple[tuple[str, IndexTerm], ...],
    tuple[tuple[str, EffectRef | EffectVariable], ...],
]:
    """Kind-check static arguments against a telescope and classify them.

    Parameters
    ----------
    telescope : Telescope
        The binders the arguments instantiate, in order.
    arguments : tuple[StaticArgument, ...]
        One argument per binder.

    Returns
    -------
    tuple[tuple[tuple[str, TypeExpr], ...], tuple[tuple[str, IndexTerm], ...], tuple[tuple[str, EffectRef | EffectVariable], ...]]
        The type, index, and effect bindings the arguments make, each
        paired with its binder's name, in telescope order.

    Raises
    ------
    TypeError
        If the counts differ, an argument is of the wrong static class
        for its binder, or an argument is of the wrong kind or index
        sort.
    ValueError
        If a nested application is ill-kinded, or an index constructor is
        applied at the wrong arity.
    """
    if len(telescope) != len(arguments):
        raise TypeError(
            f"expected {len(telescope)} static arguments, got {len(arguments)}"
        )

    types: list[tuple[str, TypeExpr]] = []
    indices: list[tuple[str, IndexTerm]] = []
    effects: list[tuple[str, EffectRef | EffectVariable]] = []
    for binder, argument in zip(telescope, arguments, strict=True):
        validate_static_argument(argument)
        if isinstance(binder, TypeBinder):
            if not isinstance(
                argument,
                (TypeVariable, TypeApplication, FunctionType, EqualityType),
            ):
                raise TypeError(f"{binder.name!r} expects a type argument")
            if static_kind(argument) != binder.kind:
                raise TypeError(f"type argument for {binder.name!r} has wrong kind")
            types.append((binder.name, argument))
        elif isinstance(binder, IndexBinder):
            if not isinstance(
                argument,
                (IndexVariable, IndexConstructor, ShapeIndex),
            ) and not hasattr(argument, "sort"):
                raise TypeError(f"{binder.name!r} expects an index argument")
            actual_sort = index_sort(argument)  # type: ignore[arg-type]
            if not _sort_matches(binder.sort, actual_sort):
                raise TypeError(
                    f"index argument for {binder.name!r} has sort {actual_sort!r}, "
                    f"expected {binder.sort!r}"
                )
            indices.append((binder.name, argument))  # type: ignore[arg-type]
        elif isinstance(binder, EffectBinder):
            if not isinstance(argument, (EffectRef, EffectVariable)):
                raise TypeError(f"{binder.name!r} expects an effect argument")
            effects.append((binder.name, argument))
        else:  # pragma: no cover - closed binder union
            raise TypeError(f"unknown telescope binder {binder!r}")
    return tuple(types), tuple(indices), tuple(effects)


def render_static(term: StaticArgument) -> str:
    """Render a static term in the surface spelling.

    Parameters
    ----------
    term : StaticArgument
        The type, index, or effect term to render.

    Returns
    -------
    str
        A readable rendering: a constructor applied to its arguments in
        brackets, with shape arguments in parentheses after them; an
        arrow for a function type; an equality with ``~``; a shape as a
        bracketed dimension list; an index constructor as a call.

    Raises
    ------
    TypeError
        If the term is not a static term.
    """
    if isinstance(term, TypeVariable | EffectVariable | IndexVariable):
        return term.name
    if isinstance(term, IndexLiteral):
        return str(term.value)
    if isinstance(term, IndexConstructor):
        if not term.arguments:
            return term.name
        inner = ", ".join(render_static(argument) for argument in term.arguments)
        return f"{term.name}({inner})"
    if isinstance(term, ShapeIndex):
        inner = ", ".join(render_static(dimension) for dimension in term.dimensions)
        return f"[{inner}]"
    if isinstance(term, TypeApplication | EffectRef):
        name = term.constructor.name if isinstance(term, TypeApplication) else term.name
        statics = [
            render_static(argument)
            for argument in term.arguments
            if not isinstance(argument, ShapeIndex)
        ]
        shapes = [
            render_static(argument)
            for argument in term.arguments
            if isinstance(argument, ShapeIndex)
        ]
        rendered = name
        if statics:
            rendered += f"[{', '.join(statics)}]"
        if shapes:
            rendered += f"({', '.join(shapes)})"
        return rendered
    if isinstance(term, FunctionType):
        return f"{render_static(term.parameter)} -> {render_static(term.result)}"
    if isinstance(term, EqualityType):
        return f"{render_static(term.left)} ~ {render_static(term.right)}"
    raise TypeError(f"not a static term: {term!r}")


__all__ = [
    "BOOL",
    "EffectRef",
    "EffectVariable",
    "EqualityType",
    "FunctionType",
    "INT",
    "IndexConstructor",
    "IndexLiteral",
    "IndexTerm",
    "IndexVariable",
    "REAL",
    "STRING",
    "ShapeIndex",
    "StaticArgument",
    "TypeApplication",
    "TypeConstructorRef",
    "TypeExpr",
    "TypeVariable",
    "UNIT",
    "check_static_arguments",
    "index_sort",
    "product_type",
    "render_static",
    "static_kind",
    "validate_static_argument",
]
