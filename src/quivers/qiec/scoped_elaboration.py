"""Check static computation arguments before elaborating them to QIEC handlers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from quivers.qiec.checking import (
    CheckContext,
    KernelError,
    infer_computation,
    infer_value,
)
from quivers.qiec.effects import EMPTY_ROW, ComputationType, HandlerDef, ResumptionGrade
from quivers.qiec.identifiers import EffectInstanceId, HandlerId
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.primitives import PRIMITIVES
from quivers.qiec.terms import (
    Bind,
    Computation,
    ConditionalValue,
    Handle,
    If,
    LetValue,
    LiteralValue,
    Local,
    Perform,
    PrimitiveApplication,
    Resume,
    Return,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, INT, REAL, UNIT, TypeExpr

_SCALAR_TYPES = (BOOL, INT, REAL, UNIT)
_PURE = frozenset(
    PRIMITIVES[name].id
    for name in (
        "add_real",
        "sub_real",
        "mul_real",
        "neg_real",
        "eq_real",
        "lt_real",
        "le_real",
        "gt_real",
        "ge_real",
        "and",
        "or",
        "not",
        "bool_to_real",
    )
)


@dataclass(frozen=True, slots=True)
class ScopedOperation:
    """The compile-time interface of a scope with one static computation.

    Parameters
    ----------
    name : str
        The source name of the scoped operation.
    body_type : ComputationType
        The exact authored type of the computation argument, with one
        named effect instance.
    result_type : ComputationType
        The type of the elaborated operation, with an empty effect row.
    arguments : tuple[TypeExpr, ...]
        The scalar value arguments available to an implementation.
    """

    name: str
    body_type: ComputationType
    result_type: ComputationType
    arguments: tuple[TypeExpr, ...] = ()


@dataclass(frozen=True, slots=True)
class ScopedHandlerElaboration:
    """Implement one static scope with an ordinary authored handler.

    Parameters
    ----------
    instance : EffectInstanceId
        The exact named instance to intercept in the static computation.
    handler : HandlerId
        The checked authored handler, receiving the scope's value
        arguments in order.
    """

    instance: EffectInstanceId
    handler: HandlerId


def _check_term(
    source: Value | Computation, *, instances: Collection[EffectInstanceId] = ()
) -> set[str]:
    """Check the first-order terms that can run inside a single-entry scope.

    Parameters
    ----------
    source : Value or Computation
        The term whose structural control and value constructors are inspected.
    instances : Collection[EffectInstanceId]
        The nominal instances allowed in the source. Handler argument
        and response values carry no requests.

    Returns
    -------
    set[str]
        The names bound in the source.

    Raises
    ------
    KernelError
        If a term leaves the single-entry portable fragment.
    """
    pending = [source]
    names: set[str] = set()
    while pending:
        term = pending.pop()
        if isinstance(term, Bind):
            names.add(term.binder.name)
            pending.extend((term.first, term.then))
        elif isinstance(term, Return):
            pending.append(term.value)
        elif isinstance(term, If | ConditionalValue):
            pending.extend((term.condition, term.then, term.otherwise))
        elif isinstance(term, Perform):
            request = term.request
            if request.instance not in instances or request.static_arguments:
                raise KernelError(
                    "scope body requests must use the declared named instance"
                )
            if request.result_type not in _SCALAR_TYPES:
                raise KernelError("scope body requests must return a scalar")
            pending.extend(request.arguments)
        elif isinstance(term, LetValue):
            names.update(item.binder.name for item in term.bindings)
            pending.extend(item.value for item in term.bindings)
            pending.append(term.body)
        elif isinstance(term, PrimitiveApplication) and term.primitive in _PURE:
            pending.extend(term.arguments)
        elif not (
            isinstance(term, Var)
            and term.local.type in _SCALAR_TYPES
            or isinstance(term, LiteralValue)
            and term.type in _SCALAR_TYPES
        ):
            raise KernelError(
                f"{type(term).__name__} is outside the portable static scope"
            )
    return names


def _check_handler(handler: HandlerDef) -> None:
    """Require an authored handler with one direct tail resume per request.

    Parameters
    ----------
    handler : HandlerDef
        The validated declaration the portable elaboration would install.

    Raises
    ------
    KernelError
        If a clause may skip, copy, or otherwise re-enter the static body,
        performs an extra operation, or changes the return boundary.
    """
    if (
        handler.implementation != "authored"
        or not handler.total
        or handler.telescope
        or handler.return_clause is not None
        or handler.introduced != EMPTY_ROW
    ):
        raise KernelError(
            "static scope elaboration requires a total authored handler with an identity return"
        )
    for clause in handler.clauses:
        resume = clause.body
        if not isinstance(resume, Resume) or clause.grade is not ResumptionGrade.LINEAR:
            raise KernelError(
                "static scope elaboration requires one direct tail resume per clause"
            )
        for value in (resume.value, *resume.arguments):
            _check_term(value)


def _capture_context(
    captures: tuple[Local, ...], ambient: CheckContext
) -> CheckContext:
    """Restrict the computation argument to its declared lexical captures.

    Parameters
    ----------
    captures : tuple[Local, ...]
        The immutable scalar locals named by the scope's static body.
    ambient : CheckContext
        The actual enclosing lexical context at the source operation.

    Returns
    -------
    CheckContext
        A context with only the declared captures, without a resumption or
        an enclosing branch's static evidence.

    Raises
    ------
    KernelError
        If a capture is unavailable, duplicated, or requires a resource
        whose immutable scalar use this fragment does not establish.
    """
    restricted = CheckContext()
    for local in captures:
        if local not in ambient.locals:
            raise KernelError(f"scope capture {local.name!r} is unavailable")
        if local.type not in _SCALAR_TYPES:
            raise KernelError(
                f"scope capture {local.name!r} must be an immutable scalar"
            )
        restricted = restricted.extend(local)
    return restricted


def elaborate_static_scope(
    module: QiecModule,
    operation: ScopedOperation,
    elaboration: ScopedHandlerElaboration,
    body: Computation,
    arguments: tuple[Value, ...] = (),
    *,
    captures: tuple[Local, ...] = (),
    context: CheckContext = CheckContext(),
) -> Handle:
    """Use a static body once in an ordinary, checked QIEC handler scope.

    The authored region remains the exact child of the handler. Its
    computation and original source provenance are available to the
    ordinary evaluator, discharge and serialization. The compile-time
    operation and elaborator disappear.

    Parameters
    ----------
    module : QiecModule
        The module supplying the actual effect instances and handler.
    operation : ScopedOperation
        The source operation's static body, result and value-argument types.
    elaboration : ScopedHandlerElaboration
        The selected implementation on the original nominal instance.
    body : Computation
        The exact computation argument written at the operation site.
    arguments : tuple[Value, ...]
        The scalar values supplied to the elaboration at installation.
    captures : tuple[Local, ...]
        The lexical bindings the body can read.
    context : CheckContext
        The actual enclosing source context.

    Returns
    -------
    Handle
        One ordinary handler installation whose only computation child
        is the supplied body.

    Raises
    ------
    KernelError
        If the types, nominal effect, lexical scope or once-use rule fail.
    """
    registry = validate_module(module)
    body_row = operation.body_type.effects
    if (
        body_row.tail is not None
        or len(body_row.entries) != 1
        or body_row.entries[0].instance != elaboration.instance
    ):
        raise KernelError(
            f"scoped operation {operation.name!r} needs its declared named instance"
        )
    if (
        operation.body_type.result not in _SCALAR_TYPES
        or operation.result_type.result != operation.body_type.result
        or operation.result_type.effects != EMPTY_ROW
        or any(item not in _SCALAR_TYPES for item in operation.arguments)
    ):
        raise KernelError(
            f"scoped operation {operation.name!r} needs scalar types and a closed result row"
        )
    capture_context = _capture_context(captures, context)
    _check_term(body, instances=(elaboration.instance,))
    if infer_computation(body, registry, capture_context) != operation.body_type:
        raise KernelError("the static scope body does not match its declared type")
    for argument in arguments:
        _check_term(argument)
    if tuple(infer_value(argument, registry, context) for argument in arguments) != (
        operation.arguments
    ):
        raise KernelError("scope value arguments do not match their declared types")

    handler = registry.handler(elaboration.handler)
    _check_handler(handler)
    lowered = Handle(
        instance=elaboration.instance,
        handler=elaboration.handler,
        arguments=arguments,
        computation=body,
    )
    if infer_computation(lowered, registry, context) != operation.result_type:
        raise KernelError("static scope elaboration does not match its result type")
    return lowered


__all__ = [
    "ScopedHandlerElaboration",
    "ScopedOperation",
    "elaborate_static_scope",
]
