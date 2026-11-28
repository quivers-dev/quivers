"""Elaborate a static catch to a QIEC handler and a first-order case."""

from __future__ import annotations

from quivers.qiec.checking import CheckContext, KernelError, infer_computation
from quivers.qiec.effects import EMPTY_ROW, ComputationType, HandlerDef, ResumptionGrade
from quivers.qiec.identifiers import ConstructorId, SourceOrigin, StaticScopeId
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.scoped_elaboration import (
    ScopedHandlerElaboration,
    _SCALAR_TYPES,
    _capture_context,
    _check_term,
)
from quivers.qiec.terms import (
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    ConstructorValue,
    Handle,
    Local,
    Return,
    Var,
)
from quivers.qiec.types import TypeApplication, TypeExpr


def _return_tag(
    body: Computation | None, binder: Local, result: TypeExpr
) -> ConstructorId:
    """Recognize an authored return of a tag containing one unchanged value.

    Parameters
    ----------
    body : Computation or None
        The actual return or aborting handler clause body.
    binder : Local
        The original result or thrown value supplied to the clause.
    result : TypeExpr
        The common ordinary data type of the two handler answers.

    Returns
    -------
    ConstructorId
        The nominal data constructor used by the clause.

    Raises
    ------
    KernelError
        If the clause does more than tag its own input value.
    """
    if isinstance(body, Return) and isinstance(value := body.value, ConstructorValue):
        if (
            value.result_type == result
            and not value.static_arguments
            and value.fields == (Var(local=binder),)
        ):
            return value.constructor
    raise KernelError(
        "a static catch handler must return each unchanged payload in a tag"
    )


def _catch_tags(
    handler: HandlerDef, error: Local
) -> tuple[ConstructorId, ConstructorId]:
    """Check the actual handler's success and grade-zero failure actions.

    Parameters
    ----------
    handler : HandlerDef
        The handler resolved from the checked declarations.
    error : Local
        The recovery binder with the thrown value's type.

    Returns
    -------
    tuple[ConstructorId, ConstructorId]
        The success and failure constructors in that order.

    Raises
    ------
    KernelError
        If the selected handler lacks the first-order catch contract.
    """
    if (
        handler.implementation != "authored"
        or not handler.total
        or handler.telescope
        or handler.parameters
        or handler.introduced != EMPTY_ROW
        or len(handler.clauses) != 1
        or handler.return_clause is None
        or not isinstance(handler.output_type, TypeApplication)
        or handler.output_type.arguments
    ):
        raise KernelError(
            "static catch requires a total authored single-operation handler returning two data tags"
        )
    clause = handler.clauses[0]
    if (
        clause.grade is not ResumptionGrade.ZERO
        or len(clause.parameters) != 1
        or clause.parameters[0].type != error.type
    ):
        raise KernelError(
            "static catch requires one grade-zero clause with the thrown type"
        )
    success = _return_tag(
        handler.return_clause.body, handler.return_clause.binder, handler.output_type
    )
    failure = _return_tag(clause.body, clause.parameters[0], handler.output_type)
    if success == failure:
        raise KernelError(
            "static catch needs different success and failure constructors"
        )
    return success, failure


def _fresh_local(prefix: str, type_: TypeExpr, used: set[str]) -> Local:
    """Allocate an internal local without shadowing an authored binding.

    Parameters
    ----------
    prefix : str
        The deterministic source-derived name to try first.
    type_ : TypeExpr
        The binding's checked QIEC type.
    used : set[str]
        The names already in scope or bound within the recovery body.

    Returns
    -------
    Local
        A fresh local with its name added to the occupied set.
    """
    name = prefix
    while name in used:
        name += "_"
    used.add(name)
    return Local(name=name, type=type_)


def elaborate_static_catch(
    module: QiecModule,
    elaboration: ScopedHandlerElaboration,
    protected: Computation,
    recovery: Computation,
    *,
    error: Local,
    origin: SourceOrigin,
    captures: tuple[Local, ...] = (),
    context: CheckContext = CheckContext(),
) -> Bind:
    """Catch one named throw inside a static body and recover outside its handler.

    The ordinary authored handler returns a success or failure tag with
    its original payload. A case outside the handler returns the success
    value or executes the recovery with the thrown value bound. Residual
    requests go through the surrounding handlers as usual. The catch
    adds no rollback and no surrounding continuation is copied by it.

    Parameters
    ----------
    module : QiecModule
        The source declarations supplying the named instance and handler.
    elaboration : ScopedHandlerElaboration
        The handler installation for the exact nominal exception instance.
    protected, recovery : Computation
        Authored first-order bodies returning the same immutable scalar
        type. Recovery may perform the original exception again.
    error : Local
        The fresh scalar binder available only in the recovery.
    origin : SourceOrigin
        The lexical catch occurrence, used for fresh static case scopes.
    captures : tuple[Local, ...]
        The existing scalar locals either computation is allowed to read.
    context : CheckContext
        The enclosing lexical context of the catch.

    Returns
    -------
    Bind
        The ordinary handler followed by a case with one recovery body.

    Raises
    ------
    KernelError
        If a term, capture, nominal instance, payload tag or result fails
        its structural or ordinary QIEC type check.
    """
    if error.type not in _SCALAR_TYPES or any(
        item.name == error.name for item in context.locals
    ):
        raise KernelError("a static catch needs a fresh scalar recovery binder")
    registry = validate_module(module)
    captured = _capture_context(captures, context)
    checked = infer_computation(protected, registry, captured)
    recovered = infer_computation(recovery, registry, captured.extend(error))
    if (
        checked.result not in _SCALAR_TYPES
        or recovered.result != checked.result
        or checked.effects.tail is not None
        or recovered.effects.tail is not None
    ):
        raise KernelError(
            "static catch needs closed effect rows and one scalar result type"
        )
    handler = registry.handler(elaboration.handler)
    if (
        checked.effects.lookup(elaboration.instance) != handler.effect
        or checked.result != handler.input_type
    ):
        raise KernelError(
            "static catch needs its declared named exception instance and input"
        )
    success, failure = _catch_tags(handler, error)
    _check_term(
        protected, instances=tuple(item.instance for item in checked.effects.entries)
    )
    body_names = _check_term(
        recovery, instances=tuple(item.instance for item in recovered.effects.entries)
    )
    used = body_names | {item.name for item in context.locals} | {error.name}
    site = str(origin.site_id())
    prefix = f"_qvr_caught_{site[-24:]}"
    result = _fresh_local(prefix, handler.output_type, used)
    value = _fresh_local(f"{prefix}_value", checked.result, used)
    lowered = Bind(
        first=Handle(
            instance=elaboration.instance,
            handler=elaboration.handler,
            computation=protected,
        ),
        binder=result,
        then=Case(
            scrutinee=Var(local=result),
            motive=CaseMotive(indices=(), result_type=checked.result),
            branches=(
                CaseBranch(
                    constructor=success,
                    static_arguments=(),
                    fields=(value,),
                    body=Return(value=Var(local=value)),
                    scope=StaticScopeId.derive(site, "catch-success"),
                ),
                CaseBranch(
                    constructor=failure,
                    static_arguments=(),
                    fields=(error,),
                    body=recovery,
                    scope=StaticScopeId.derive(site, "catch-failure"),
                ),
            ),
        ),
    )
    expected = ComputationType(
        effects=checked.effects.without(elaboration.instance).union(recovered.effects),
        result=checked.result,
    )
    if infer_computation(lowered, registry, context) != expected:
        raise KernelError("static catch produced a different ordinary QIEC result type")
    return lowered


__all__ = ["elaborate_static_catch"]
