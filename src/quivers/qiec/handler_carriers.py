"""Classify checked handlers by the position of their resumptions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from quivers.qiec.effects import (
    EMPTY_ROW,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
)
from quivers.qiec.identifiers import ComputationId, HandlerId, OperationId
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.serialization import dumps
from quivers.qiec.terms import (
    Bind,
    Call,
    Computation,
    If,
    LiteralValue,
    Resume,
    Return,
    Var,
)
from quivers.qiec.types import BOOL

type _CarrierKind = Literal["tail-state", "tail-or-abort", "continuation"]
type _Completion = Literal["abort", "resume"]
type _ReturnAction = Literal["identity", "authored", "opaque"]


@dataclass(frozen=True, slots=True)
class ClauseCarrierShape:
    """The possible terminal actions of one checked operation clause.

    Parameters
    ----------
    operation : OperationId
        The nominal operation this clause answers.
    grade : ResumptionGrade
        The declared grade, retained even when the body uses fewer shots.
    completions : tuple of {"abort", "resume"}
        Terminal actions on the recognized paths through the clause.
    obstacle : str or None
        The first reason the clause needs a general continuation.
    """

    operation: OperationId
    grade: ResumptionGrade
    completions: tuple[_Completion, ...]
    obstacle: str | None

    @property
    def kind(self) -> _CarrierKind:
        """The carrier shape of the clause's recognized terminal actions.

        Returns
        -------
        _CarrierKind
            Tail state, tail with an abort arm, or a general continuation.
        """
        if self.obstacle is not None:
            return "continuation"
        return "tail-or-abort" if "abort" in self.completions else "tail-state"


@dataclass(frozen=True, slots=True)
class HandlerCarrierShape:
    """The carrier and return-action shapes of an actual checked handler.

    Parameters
    ----------
    handler : HandlerId
        The nominal handler whose clauses were examined.
    fingerprint : str
        A binding to the checked module, handler and classification version.
    kind : {"tail-state", "tail-or-abort", "continuation"}
        The shape required by the whole handler.
    return_action : {"identity", "authored", "opaque"}
        Whether completing the handled computation uses the identity,
        an authored return clause, or provider behavior.
    clauses : tuple[ClauseCarrierShape, ...]
        The checked clause shapes in declaration order.
    obstacle : str or None
        The first source construct requiring a general continuation.
    """

    handler: HandlerId
    fingerprint: str
    kind: _CarrierKind
    return_action: _ReturnAction
    clauses: tuple[ClauseCarrierShape, ...]
    obstacle: str | None


def _branches(term: If) -> tuple[Computation, ...]:
    """Select a literal conditional arm or retain both possible arms.

    Parameters
    ----------
    term : If
        The checked conditional.

    Returns
    -------
    tuple[Computation, ...]
        The branches to inspect.
    """
    condition = term.condition
    if (
        isinstance(condition, LiteralValue)
        and condition.type == BOOL
        and type(condition.value) is bool
    ):
        return (term.then if condition.value else term.otherwise,)
    return term.then, term.otherwise


class _CarrierAnalysis:
    """Recognize ordinary prefixes and tail actions in one checked module.

    Parameters
    ----------
    module : QiecModule
        The validated module supplying the actual callees of each prefix.
    """

    def __init__(self, module: QiecModule) -> None:
        self.computations = {item.id: item for item in module.computations}
        self.calls: dict[ComputationId, str | None] = {}
        self.active: set[ComputationId] = set()

    def ordinary(self, term: Computation) -> str | None:
        """Check a computation which must return without invoking a resumption.

        Parameters
        ----------
        term : Computation
            A checked prefix, call body or return action.

        Returns
        -------
        str or None
            A reason to require general control, if found.
        """
        if isinstance(term, Return):
            return None
        if isinstance(term, Bind):
            return self.ordinary(term.first) or self.ordinary(term.then)
        if isinstance(term, If):
            for branch in _branches(term):
                if (obstacle := self.ordinary(branch)) is not None:
                    return obstacle
            return None
        if isinstance(term, Call):
            owner = self.computations.get(term.callee)
            if owner is None or owner.telescope or owner.type.effects != EMPTY_ROW:
                return f"call `{term.name}` is outside ordinary closed prefixes"
            return self.call(owner)
        if isinstance(term, Resume):
            return "a resumption is used before its clause returns"
        return f"{type(term).__name__} requires general control"

    def call(self, declaration: NamedComputation) -> str | None:
        """Inspect the complete ordinary call graph of a prefix.

        Parameters
        ----------
        declaration : NamedComputation
            The closed callee whose body must return ordinarily.

        Returns
        -------
        str or None
            A conservative refusal for a cycle or general control.
        """
        identity = declaration.id
        if identity in self.calls:
            return self.calls[identity]
        if identity in self.active:
            return f"recursive call `{declaration.name}` needs a separate carrier proof"
        self.active.add(identity)
        try:
            result = self.ordinary(declaration.body)
        finally:
            self.active.remove(identity)
        self.calls[identity] = result
        return result

    def completions(self, term: Computation) -> frozenset[_Completion] | str:
        """Collect the recognized terminal actions of a whole clause.

        Parameters
        ----------
        term : Computation
            A checked computation in tail position.

        Returns
        -------
        frozenset[_Completion] or str
            Its possible terminal actions, or a reason to keep the continuation.
        """
        if isinstance(term, Resume):
            return frozenset(("resume",))
        if isinstance(term, Bind):
            obstacle = self.ordinary(term.first)
            return obstacle if obstacle is not None else self.completions(term.then)
        if isinstance(term, If):
            result: set[_Completion] = set()
            for branch in _branches(term):
                actions = self.completions(branch)
                if isinstance(actions, str):
                    return actions
                result.update(actions)
            return frozenset(result)
        obstacle = self.ordinary(term)
        return obstacle if obstacle is not None else frozenset(("abort",))

    def clause(
        self, definition: HandlerClauseDef, obstacle: str | None
    ) -> ClauseCarrierShape:
        """Summarize one clause while retaining its nominal identity and grade.

        Parameters
        ----------
        definition : HandlerClauseDef
            The checked operation clause.
        obstacle : str or None
            An obstacle applying to the entire handler, if present.

        Returns
        -------
        ClauseCarrierShape
            The recognized actions and any reason for a continuation.
        """
        if obstacle is not None or definition.body is None:
            return ClauseCarrierShape(
                definition.operation,
                definition.grade,
                (),
                obstacle or "opaque operation clause",
            )
        result = self.completions(definition.body)
        if isinstance(result, str):
            return ClauseCarrierShape(
                definition.operation, definition.grade, (), result
            )
        return ClauseCarrierShape(
            definition.operation, definition.grade, tuple(sorted(result)), None
        )


def _handler_obstacle(handler: HandlerDef) -> str | None:
    """Check the shared hypotheses of the first-order carrier analysis.

    Parameters
    ----------
    handler : HandlerDef
        The validated handler declaration.

    Returns
    -------
    str or None
        The first unproved handler hypothesis, if any.
    """
    if handler.implementation != "authored":
        return "opaque foreign implementation"
    if handler.telescope:
        return "handler needs static specialization"
    if not handler.total:
        return "handler has incomplete operation coverage"
    if handler.introduced != EMPTY_ROW:
        return "handler introduces residual effects"
    return None


def _return_action(handler: HandlerDef) -> _ReturnAction:
    """Identify a syntactic identity return and record other return actions.

    Parameters
    ----------
    handler : HandlerDef
        The declaration whose return action is examined.

    Returns
    -------
    _ReturnAction
        Identity, other authored code, or provider behavior.
    """
    if handler.implementation != "authored":
        return "opaque"
    clause = handler.return_clause
    if clause is None or (
        isinstance(clause.body, Return)
        and isinstance(clause.body.value, Var)
        and clause.body.value.local == clause.binder
    ):
        return "identity"
    return "authored"


def classify_handler_carrier(
    module: QiecModule, handler: HandlerId
) -> HandlerCarrierShape:
    """Recognize tail-state and tail-or-abort shapes of a checked handler.

    A recognized operation clause returns an answer directly or resumes
    in tail position after ordinary prefixes. Every reachable named
    prefix call must have an ordinary acyclic body. Handler metadata and
    the exact return action stay visible in the result. Transformations
    that use a carrier shape also need their own semantic hypotheses.

    Parameters
    ----------
    module : QiecModule
        The checked module containing the actual handler and its callees.
    handler : HandlerId
        The nominal handler to classify.

    Returns
    -------
    HandlerCarrierShape
        A source-bound classification with clause shapes and obstacles.

    Raises
    ------
    ValueError
        If the nominal handler is absent from the module.
    """
    registry = validate_module(module)
    target = registry.handlers.get(handler)
    if target is None:
        raise ValueError(f"unknown handler for carrier classification `{handler}`")
    fingerprint = sha256(
        dumps(("handler-carrier-v1", module, handler)).encode("utf-8")
    ).hexdigest()
    analysis = _CarrierAnalysis(module)
    obstacle = _handler_obstacle(target)
    try:
        clauses = tuple(analysis.clause(clause, obstacle) for clause in target.clauses)
        if obstacle is None and target.return_clause is not None:
            why = analysis.ordinary(target.return_clause.body)
            if why is not None:
                obstacle = f"return action: {why}"
    except RecursionError:
        obstacle = "carrier analysis nesting limit reached"
        clauses = tuple(analysis.clause(clause, obstacle) for clause in target.clauses)
    if obstacle is None:
        obstacle = next(
            (clause.obstacle for clause in clauses if clause.obstacle is not None), None
        )
    kind: _CarrierKind = "tail-state"
    if obstacle is not None:
        kind = "continuation"
    elif any(clause.kind == "tail-or-abort" for clause in clauses):
        kind = "tail-or-abort"
    return HandlerCarrierShape(
        handler, fingerprint, kind, _return_action(target), clauses, obstacle
    )


__all__ = ["ClauseCarrierShape", "HandlerCarrierShape", "classify_handler_carrier"]
