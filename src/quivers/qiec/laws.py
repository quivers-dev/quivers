"""Exhaustive laws of finite checked QIEC computations.

A finite continuation is a product with one result cell for each argument
tuple. The same immutable table is passed to both sides of a law, so repeated
continuation applications can read the same cell. Enumeration covers all total
tables, including those that distinguish two values a handler might return.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod
from typing import cast

from quivers.qiec.effects import EMPTY_ROW, EffectRequest, HandlerDef
from quivers.qiec.execution import ExecutionFailure, run_named
from quivers.qiec.identifiers import ComputationId, HandlerId
from quivers.qiec.module import (
    NamedComputation,
    QiecModule,
    _children,
    called_computations,
    installed_handlers,
    validate_module,
)
from quivers.qiec.numbers import host_value
from quivers.qiec.terms import (
    Bind,
    Call,
    CallAddress,
    Case,
    Computation,
    ConditionalValue,
    HandlerReturnAddress,
    InstanceAddress,
    LetValue,
    LiteralValue,
    Local,
    Projection,
    TupleValue,
    Value,
    Var,
    WithTraceAddresses,
)
from quivers.qiec.types import (
    BOOL,
    UNIT,
    EqualityType,
    FunctionType,
    TypeApplication,
    TypeExpr,
    TypeVariable,
    product_type,
    render_static,
)

type FiniteValue = bool | None | tuple[FiniteValue, ...]
"""The host encoding of a Boolean, unit, or recursively finite product."""


class _Atom(Enum):
    """The atomic carriers available to the finite checker."""

    BOOL = "bool"
    UNIT = "unit"


type _Shape = _Atom | tuple[_Shape, ...]

_DEFAULT_LIMIT = 65_536
_VALUE_TERMS = tuple(
    item for item in Value.__value__.__args__ if isinstance(item, type)
)
_FINITE_TERMS = (Var, LiteralValue, TupleValue, Projection, ConditionalValue, LetValue)
_TYPE_TERMS = (TypeVariable, TypeApplication, FunctionType, EqualityType)


class FiniteLawUnsupported(ValueError):
    """The law falls outside the deterministic finite fragment."""


class FiniteLawLimit(ValueError):
    """The law checker exhausted a limit before deciding the law."""


@dataclass(frozen=True, slots=True)
class FiniteLawCounterexample:
    """A complete assignment on which both sides return different values.

    Parameters
    ----------
    arguments : tuple[FiniteValue, ...]
        Inputs to both sides, including the full continuation tables.
    left : FiniteValue
        The left result.
    right : FiniteValue
        The right result.
    """

    arguments: tuple[FiniteValue, ...]
    left: FiniteValue
    right: FiniteValue


@dataclass(frozen=True, slots=True)
class FiniteLawDecision:
    """The result of checking a typed finite law on the reference machine.

    Parameters
    ----------
    checked : int
        The number of assignments checked.
    counterexample : FiniteLawCounterexample or None
        The first failing assignment, if any. A positive decision has
        checked every assignment.
    """

    checked: int
    counterexample: FiniteLawCounterexample | None = None


def _shape(type_: TypeExpr) -> _Shape:
    """Read the finite carrier of a checked type.

    Parameters
    ----------
    type_ : TypeExpr
        The checked kernel type.

    Returns
    -------
    _Shape
        The shape of its host values.

    Raises
    ------
    FiniteLawUnsupported
        If the type has no carrier in this fragment.
    """
    if type_ == BOOL:
        return _Atom.BOOL
    if type_ == UNIT:
        return _Atom.UNIT
    if isinstance(type_, TypeApplication) and all(
        isinstance(item, _TYPE_TERMS) for item in type_.arguments
    ):
        components = cast(tuple[TypeExpr, ...], type_.arguments)
        if type_ == product_type(*components):
            return tuple(_shape(item) for item in components)
    raise FiniteLawUnsupported(f"unsupported finite law type {render_static(type_)}")


def _size(shape: _Shape) -> int:
    """Count a carrier's values before materializing them.

    Parameters
    ----------
    shape : _Shape
        The finite carrier.

    Returns
    -------
    int
        Its cardinality.
    """
    if isinstance(shape, tuple):
        return prod(_size(item) for item in shape)
    return 2 if shape is _Atom.BOOL else 1


def _values(shape: _Shape) -> tuple[FiniteValue, ...]:
    """Enumerate a finite carrier in table index order.

    Parameters
    ----------
    shape : _Shape
        The finite carrier.

    Returns
    -------
    tuple[FiniteValue, ...]
        Each inhabitant once, in lexicographic order.
    """
    if isinstance(shape, tuple):
        return tuple(product(*(_values(item) for item in shape)))
    return (False, True) if shape is _Atom.BOOL else (None,)


def _validate_limit(name: str, value: int) -> None:
    """Require a positive integer bound.

    Parameters
    ----------
    name : str
        The argument named in the error.
    value : int
        The requested bound.

    Raises
    ------
    ValueError
        If the bound is not positive or is not an integer.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def finite_continuation_type(
    arguments: tuple[TypeExpr, ...],
    result: TypeExpr,
    *,
    max_cells: int = _DEFAULT_LIMIT,
) -> TypeApplication:
    """Construct the QIEC product type of a total finite continuation.

    Parameters
    ----------
    arguments : tuple[TypeExpr, ...]
        The continuation's argument types, including any handler state.
    result : TypeExpr
        The answer type.
    max_cells : int
        The largest product the checker will construct.

    Returns
    -------
    TypeApplication
        A product of result cells in lexicographic argument order. Booleans
        order false before true, unit has one value, and products order
        lexicographically over their component carriers. An argument list
        of length zero has one cell.

    Raises
    ------
    FiniteLawUnsupported
        If an argument or answer is outside the supported finite types.
    FiniteLawLimit
        If the continuation has more than ``max_cells`` cells.
    ValueError
        If ``max_cells`` is not a positive integer.
    """
    _validate_limit("max_cells", max_cells)
    _shape(result)
    size = prod(_size(_shape(argument)) for argument in arguments)
    if size > max_cells:
        raise FiniteLawLimit(f"continuation has {size} cells, exceeding {max_cells}")
    return product_type(*((result,) * size))


def _check_body(body: Computation) -> None:
    """Check that a body uses finite values and exact primitives.

    Parameters
    ----------
    body : Computation
        One named computation or handler clause.

    Raises
    ------
    FiniteLawUnsupported
        If a term or type leaves the finite fragment.
    """
    pending: list[object] = [body]
    seen: set[int] = set()
    while pending:
        term = pending.pop()
        if id(term) in seen:
            continue
        seen.add(id(term))
        if isinstance(term, Bind) and isinstance(
            term.first, CallAddress | InstanceAddress | HandlerReturnAddress
        ):
            pending.append(term.then)
            continue
        if isinstance(term, WithTraceAddresses):
            pending.append(term.body)
            continue
        if isinstance(term, Case) or (
            isinstance(term, _VALUE_TERMS) and not isinstance(term, _FINITE_TERMS)
        ):
            raise FiniteLawUnsupported(
                f"{type(term).__name__} is outside the exact finite law fragment"
            )
        if isinstance(term, (LiteralValue, Local)):
            _shape(term.type)
        elif isinstance(term, (Call, EffectRequest)):
            _shape(term.result_type)
        pending.extend(_children(term))


def _check_reachable(
    module: QiecModule, left: NamedComputation, right: NamedComputation
) -> None:
    """Check called bodies and installed clauses before running either side.

    Parameters
    ----------
    module : QiecModule
        The module declaring the bodies and handlers.
    left : NamedComputation
        The left root.
    right : NamedComputation
        The right root.

    Raises
    ------
    FiniteLawUnsupported
        If a reachable definition is foreign, unspecialized, or uses a
        term outside the finite fragment.
    """
    computations = {item.id: item for item in module.computations}
    handlers = {item.id: item for item in module.handlers}
    pending: list[NamedComputation | HandlerDef] = [left, right]
    seen: set[ComputationId | HandlerId] = set()
    while pending:
        item = pending.pop()
        if item.id in seen:
            continue
        seen.add(item.id)
        if item.telescope:
            raise FiniteLawUnsupported(f"`{item.name}` has static parameters")
        for parameter in item.parameters:
            _shape(parameter.type)
        if isinstance(item, HandlerDef):
            if item.implementation != "authored":
                raise FiniteLawUnsupported(f"`{item.name}` is a foreign handler")
            _shape(item.input_type)
            _shape(item.output_type)
            bodies = [clause.body for clause in item.clauses if clause.body is not None]
            if item.return_clause is not None:
                bodies.append(item.return_clause.body)
        else:
            _shape(item.type.result)
            bodies = [item.body]
        for body in bodies:
            _check_body(body)
            pending.extend(computations[callee] for callee in called_computations(body))
            pending.extend(handlers[handler] for handler in installed_handlers(body))


def _observed(shape: _Shape, value: object) -> FiniteValue:
    """Read a reference result in the host representation of its type.

    Parameters
    ----------
    shape : _Shape
        The result carrier.
    value : object
        The result converted to host data.

    Returns
    -------
    FiniteValue
        The result with the type's exact shape.

    Raises
    ------
    FiniteLawUnsupported
        If the result is outside that carrier.
    """
    if shape is _Atom.BOOL and type(value) is bool:
        return value
    if shape is _Atom.UNIT and value is None:
        return None
    if (
        isinstance(shape, tuple)
        and isinstance(value, tuple)
        and len(shape) == len(value)
    ):
        return tuple(
            _observed(component, item)
            for component, item in zip(shape, value, strict=True)
        )
    raise FiniteLawUnsupported("a computation returned a value outside its finite type")


def check_finite_law(
    module: QiecModule,
    left: str,
    right: str,
    *,
    max_assignments: int = _DEFAULT_LIMIT,
    fuel: int = _DEFAULT_LIMIT,
) -> FiniteLawDecision:
    """Decide equality on every assignment to two named finite computations.

    Parameters
    ----------
    module : QiecModule
        The checked kernel module.
    left : str
        The left computation's name.
    right : str
        The right computation's name. The computations must have the same
        positional finite argument types, the same finite result type,
        and closed empty residual rows. Continuations can be supplied as
        products from :func:`finite_continuation_type`.
    max_assignments : int
        The largest number of assignments the checker will enumerate.
    fuel : int
        The reference evaluator's step limit for each side of each assignment.

    Returns
    -------
    FiniteLawDecision
        A decision after all assignments agree or a concrete assignment
        whose results differ.

    Raises
    ------
    FiniteLawUnsupported
        If the signatures differ or reachable code leaves the exact finite
        fragment of unit, Booleans, finite products, and authored handlers.
    FiniteLawLimit
        If the assignment or evaluator step limit prevents a decision.
    ExecutionFailure
        If a checked assignment cannot be evaluated.
    ValueError
        If an assignment or evaluator limit is not a positive integer.
    """
    _validate_limit("max_assignments", max_assignments)
    _validate_limit("fuel", fuel)
    validate_module(module)
    named = {item.name: item for item in module.computations}
    if left not in named or right not in named:
        missing = left if left not in named else right
        raise FiniteLawUnsupported(f"unknown finite law computation `{missing}`")
    lhs, rhs = named[left], named[right]
    parameters = tuple(parameter.type for parameter in lhs.parameters)
    if (
        parameters != tuple(parameter.type for parameter in rhs.parameters)
        or lhs.type.result != rhs.type.result
    ):
        raise FiniteLawUnsupported(
            "a finite law requires matching positional signatures"
        )
    if lhs.type.effects != EMPTY_ROW or rhs.type.effects != EMPTY_ROW:
        raise FiniteLawUnsupported("a finite law requires an empty residual effect row")
    shapes = tuple(_shape(parameter) for parameter in parameters)
    answer = _shape(lhs.type.result)
    count = prod(_size(shape) for shape in shapes)
    if count > max_assignments:
        raise FiniteLawLimit(
            f"law has {count} assignments, exceeding {max_assignments}"
        )
    _check_reachable(module, lhs, rhs)
    for index, arguments in enumerate(
        product(*(_values(shape) for shape in shapes)), 1
    ):
        try:
            first = _observed(
                answer,
                host_value(
                    run_named(
                        module, left, arguments, fuel=fuel, record_trace=False
                    ).value
                ),
            )
            second = _observed(
                answer,
                host_value(
                    run_named(
                        module, right, arguments, fuel=fuel, record_trace=False
                    ).value
                ),
            )
        except ExecutionFailure as error:
            if error.diagnostic.code == "qiec-run-fuel":
                raise FiniteLawLimit(
                    f"fuel exhausted at assignment {arguments!r}"
                ) from error
            raise
        if first != second:
            return FiniteLawDecision(
                index, FiniteLawCounterexample(arguments, first, second)
            )
    return FiniteLawDecision(count)


__all__ = [
    "FiniteLawCounterexample",
    "FiniteLawDecision",
    "FiniteLawLimit",
    "FiniteLawUnsupported",
    "FiniteValue",
    "check_finite_law",
    "finite_continuation_type",
]
