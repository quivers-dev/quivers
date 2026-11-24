"""Static discharge of authored handlers.

Discharge rewrites a checked computation to an equal one that installs no
handler. It evaluates the term with the reference machine's own rules and
symbolic values: a ``handle`` is a frame of an explicit continuation, a
request finds the innermost frame of its instance, the frame's clause body
runs with the continuation between the request and the frame captured, and
``resume`` puts the captured continuation back with the parameters it names.
A value that a step computes is bound under a fresh name in the rewritten
term, a request no frame answers stays a request, and an ``if`` whose
condition is not a literal stays an ``if`` over the rewritten branches, which
meet again where both reach one continuation.

The four resumption grades are one rule. A clause that resumes twice puts its
captured continuation back twice, and one that never resumes drops it.

The result depends on no target and no runtime. The compiled runtime, the
transpile planner, the static targets' projection, and the capability analysis
read it, so a handler is discharged for all of them or for none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, HeldValue
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import tensor_shape, tensor_type
from quivers.qiec.declarations import ConstructorDecl
from quivers.qiec.effects import (
    EffectDef,
    EffectRequest,
    HandlerDef,
    OperationDef,
)
from quivers.qiec.evidence import BranchGiven, Reflexivity
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    EffectInstanceId,
    HandlerId,
    OperationId,
    SourceOrigin,
    StaticScopeId,
    StaticVariableId,
)
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_telescope,
    substitute_computation,
    substitute_type,
)
from quivers.qiec.terms import (
    AttachmentRef,
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Comprehension,
    Computation,
    ConstructorValue,
    EvidenceValue,
    Gather,
    Handle,
    If,
    LetBinding,
    LetValue,
    LiteralValue,
    Local,
    NewInstance,
    Perform,
    Projection,
    Resume,
    Return,
    SiteValue,
    TensorValue,
    TupleValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    UNIT,
    EffectVariable,
    IndexVariable,
    StaticArgument,
    TypeExpr,
    TypeVariable,
    product_type,
)

DISCHARGE_STATEMENTS = 2048
"""The most statements a discharged computation may have.

A clause that resumes ``m`` times copies the rest of the computation ``m``
times, so ``n`` such requests in sequence yield ``m ** n`` copies. A
computation past this bound is left as it is.
"""

DISCHARGE_STEPS = 65_536
"""The most reduction steps one computation's discharge may take.

Discharge reduces a term without entering a recursive call, so it terminates.
The bound stops a reduction whose copies produce no statement of their own.
"""

type ObstacleKind = Literal["foreign", "forwarding", "recursion", "case", "growth"]
"""What stops a handler from being discharged."""


class DischargeObstacle(QiecModel):
    """Why a computation's handlers are not discharged.

    Parameters
    ----------
    computation
        The computation left as it is.
    kind
        ``"foreign"`` for a handler a runtime provider implements,
        ``"forwarding"`` for a request whose handler has no clause for it
        and which no outer handler of the computation answers,
        ``"recursion"`` for a recursive call that makes handled requests,
        ``"case"`` for a case analysis on a value known only at run time
        whose branches do not meet in one continuation, and ``"growth"``
        for a rewriting past the size bounds.
    reason
        The obstacle as a clause naming the handler and the clause, to
        follow "the computation".
    origin
        Where the obstacle is written, when the term records it.
    """

    computation: str
    kind: ObstacleKind = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    reason: str
    origin: SourceOrigin | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class DischargedModule(QiecModel):
    """A module whose computations install no handler that can be discharged.

    Parameters
    ----------
    module
        The module with every dischargeable computation's body rewritten.
        A computation keeps its declared type, whose row may name an
        instance the rewritten body no longer requests.
    obstacles
        One obstacle per computation whose handlers remain.
    """

    module: QiecModule = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    obstacles: tuple[DischargeObstacle, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def obstacle(self, computation: str) -> DischargeObstacle | None:
        """Why a computation's handlers remain.

        Parameters
        ----------
        computation : str
            The computation's name.

        Returns
        -------
        DischargeObstacle or None
            The obstacle, or ``None`` when the computation installs no
            handler after discharge.
        """
        return next(
            (item for item in self.obstacles if item.computation == computation), None
        )

    def requester(self, request: EffectRequest) -> str | None:
        """The handler clause a request is written in.

        Parameters
        ----------
        request : EffectRequest
            A request of a discharged body.

        Returns
        -------
        str or None
            ``clause `op` of handler `h```, or ``None`` when the request is
            not written in a clause.
        """
        names = {
            operation.id: operation.name
            for effect in self.module.effects
            for operation in effect.operations
        }
        for handler in self.module.handlers:
            for clause, made in clause_requests(handler, names):
                if made.origin == request.origin:
                    return clause
        return None


class _Undischarged(Exception):
    """An obstacle met while discharging one computation.

    Parameters
    ----------
    kind : ObstacleKind
        The obstacle's kind.
    reason : str
        The obstacle's clause.
    origin : SourceOrigin or None
        Where it is written.
    """

    def __init__(
        self, kind: ObstacleKind, reason: str, origin: SourceOrigin | None
    ) -> None:
        super().__init__(reason)
        self.kind: ObstacleKind = kind
        self.reason = reason
        self.origin = origin


type _Environment = Mapping[str, Value]
"""The value each local of a body stands for, by name."""

type _Statements = tuple[tuple[Local, Computation], ...]
"""Statements of a rewritten term: each binder and what it binds."""


class _Frame(QiecModel):
    """One frame of the symbolic continuation.

    Frames compare by identity, so two continuations are one exactly when
    they hold the same frames.
    """

    __slots__ = ()


class _BindFrame(_Frame):
    """A continuation awaiting the first half of a bind.

    Parameters
    ----------
    binder
        The local the awaited value is bound to.
    then
        The computation run once the value arrives.
    environment
        The values ``then`` is read under.
    """

    binder: Local = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    then: Computation = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    environment: _Environment = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


class _HandlerFrame(_Frame):
    """An installed handler.

    Parameters
    ----------
    serial
        The installation's ordinal, shared by the frames that carry it at
        other parameter values.
    instance
        The effect instance handled.
    handler
        The handler's declaration.
    statics
        The declaration's telescope at this installation.
    arguments
        The current values of the handler's parameters.
    """

    serial: int
    instance: EffectInstanceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    handler: HandlerDef = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    statics: StaticSubstitution = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    arguments: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def carrying(self, arguments: tuple[Value, ...]) -> _HandlerFrame:
        """This installation with its parameters at other values.

        Parameters
        ----------
        arguments : tuple[Value, ...]
            One value per parameter.

        Returns
        -------
        _HandlerFrame
            The frame.
        """
        return _HandlerFrame(
            serial=self.serial,
            instance=self.instance,
            handler=self.handler,
            statics=self.statics,
            arguments=arguments,
        )


class _ClauseFrame(_Frame):
    """A running clause body, standing where its handler stood.

    Parameters
    ----------
    captured
        The continuation from the request to the handler, the handler's
        frame last.
    answer
        The type the clause answers.
    """

    captured: tuple[_Frame, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    answer: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class _CallFrame(_Frame):
    """An entered computation.

    Parameters
    ----------
    callee
        The computation's identity.
    result
        The call's result type.
    """

    callee: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class _JoinFrame(_Frame):
    """The point at which the branches of one conditional meet.

    Parameters
    ----------
    serial
        The conditional's ordinal.
    """

    serial: int


class _Arrival(QiecModel):
    """A path that ended in a value.

    Parameters
    ----------
    join
        The join the value arrived at, or ``None`` at the computation's end.
    statements
        The statements the path made.
    value
        The value.
    stack
        The continuation beyond the join.
    resumed
        Whether the path left the conditional by a ``resume`` in tail
        position of its clause, so that the value returns to the resumed
        continuation and the rest of the clause has nothing to do.
    """

    join: int | None
    statements: _Statements = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    stack: tuple[_Frame, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    resumed: bool = False


class _Fork(QiecModel):
    """A path that ended in a conditional whose branches do not meet.

    Parameters
    ----------
    statements
        The statements made before the conditional.
    rebuild
        The conditional over its rewritten branches and the type they have.
    paths
        How each branch ended.
    """

    statements: _Statements = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    rebuild: Callable[[tuple[Computation, ...], TypeExpr], Computation] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    paths: tuple[_Arrival | _Fork, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class _Meeting(QiecModel):
    """The point at which a conditional's branches meet.

    Parameters
    ----------
    statements
        The statements up to the meeting, the conditional's included.
    value
        The value after it.
    stack
        The continuation after it.
    """

    statements: _Statements = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    value: Value = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    stack: tuple[_Frame, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


type _Path = _Arrival | _Fork
"""How the reduction of one term ended."""

type _State = Computation | Value
"""What a reduction continues from: a term, or a value returned to a frame."""

type _Term = (
    Value
    | LetBinding
    | Local
    | str
    | int
    | float
    | bool
    | bytes
    | None
    | tuple[_Term, ...]
)
"""A value term or one of the parts a substitution descends into."""

#: The value terms that are read in place rather than bound to a name.
_ATOMS = (Var, LiteralValue, SiteValue, AttachmentRef, EvidenceValue)

#: The value variants, as one runtime tuple.
_VALUES: tuple[type, ...] = tuple(
    item for item in Value.__value__.__args__ if isinstance(item, type)
)


def _operations(
    module: QiecModule,
) -> dict[OperationId, tuple[EffectDef, OperationDef]]:
    """The operations of a module's interfaces, with their interfaces.

    Parameters
    ----------
    module : QiecModule
        The module.

    Returns
    -------
    dict[OperationId, tuple[EffectDef, OperationDef]]
        Each operation by identity.
    """
    return {
        operation.id: (effect, operation)
        for effect in module.effects
        for operation in effect.operations
    }


def _subterms(term: Computation) -> tuple[Computation, ...]:
    """A computation and every computation nested in it.

    Parameters
    ----------
    term : Computation
        The computation.

    Returns
    -------
    tuple[Computation, ...]
        The terms, outermost first.
    """
    found: list[Computation] = []
    pending: list[Computation] = [term]
    while pending:
        item = pending.pop()
        found.append(item)
        if isinstance(item, Bind):
            pending.extend((item.then, item.first))
        elif isinstance(item, Handle):
            pending.append(item.computation)
        elif isinstance(item, If):
            pending.extend((item.otherwise, item.then))
        elif isinstance(item, Case):
            pending.extend(branch.body for branch in item.branches)
        elif isinstance(item, NewInstance):
            pending.append(item.body)
    return tuple(found)


def clause_requests(
    handler: HandlerDef, names: Mapping[OperationId, str]
) -> tuple[tuple[str, EffectRequest], ...]:
    """The requests a handler's clause bodies make.

    Parameters
    ----------
    handler : HandlerDef
        The handler.
    names : Mapping[OperationId, str]
        The source name of each operation.

    Returns
    -------
    tuple[tuple[str, EffectRequest], ...]
        Each request with the clause making it, as ``clause `op` of handler
        `h``` or ``the return clause of handler `h```.
    """
    bodies: list[tuple[str, Computation | None]] = [
        (f"clause `{names[clause.operation]}`", clause.body)
        for clause in handler.clauses
    ]
    if handler.return_clause is not None:
        bodies.append(("the return clause", handler.return_clause.body))
    return tuple(
        (f"{subject} of handler `{handler.name}`", term.request)
        for subject, body in bodies
        if body is not None
        for term in _subterms(body)
        if isinstance(term, Perform)
    )


def _read(value: Value) -> frozenset[str]:
    """The names a value reads.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    frozenset[str]
        The name of every variable in it.
    """
    found: set[str] = set()
    pending: list[HeldValue] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Var):
            found.add(item.local.name)
        elif isinstance(item, tuple):
            pending.extend(item)
        elif isinstance(item, QiecModel):
            pending.extend(getattr(item, name) for name in type(item).__field_specs__)
    return frozenset(found)


def _computation_reads(term: Computation) -> frozenset[str]:
    """The names a computation's values read, its own binders included.

    Parameters
    ----------
    term : Computation
        The computation.

    Returns
    -------
    frozenset[str]
        The name of every variable in it.
    """
    found: set[str] = set()
    for item in _subterms(term):
        if isinstance(item, Return):
            found |= _read(item.value)
        elif isinstance(item, If):
            found |= _read(item.condition)
        elif isinstance(item, Call):
            for argument in item.arguments:
                found |= _read(argument)
        elif isinstance(item, Perform):
            for argument in item.request.arguments:
                found |= _read(argument)
        elif isinstance(item, Case):
            found |= _read(item.scrutinee)
    return frozenset(found)


def _pure(term: Computation) -> bool:
    """Whether a rewritten computation makes no request.

    Parameters
    ----------
    term : Computation
        The computation.

    Returns
    -------
    bool
        True when every call in it has the empty closed row and nothing in
        it requests.
    """
    return all(
        not isinstance(item, Perform)
        and not (
            isinstance(item, Call)
            and (item.effects.entries or item.effects.tail is not None)
        )
        for item in _subterms(term)
    )


def _live(statements: _Statements, reads: frozenset[str]) -> _Statements:
    """The statements a rewritten term keeps.

    A value nothing reads is not computed, while a call or a request is
    made whether or not its result is read.

    Parameters
    ----------
    statements : tuple[tuple[Local, Computation], ...]
        The statements, in order.
    reads : frozenset[str]
        The names read after them.

    Returns
    -------
    tuple[tuple[Local, Computation], ...]
        The statements kept.
    """
    needed = set(reads)
    kept: list[tuple[Local, Computation]] = []
    for binder, first in reversed(statements):
        if isinstance(first, Return) and binder.name not in needed:
            continue
        kept.append((binder, first))
        needed |= _computation_reads(first)
    return tuple(reversed(kept))


def _sequence(statements: _Statements, tail: Computation) -> Computation:
    """A chain of statements ending in a computation.

    Parameters
    ----------
    statements : tuple[tuple[Local, Computation], ...]
        The statements, in order.
    tail : Computation
        What the chain ends in.

    Returns
    -------
    Computation
        The chain, without the values nothing reads. A chain that returns
        what its last statement binds ends in that statement.
    """
    kept = _live(statements, _computation_reads(tail))
    result = tail
    if (
        kept
        and isinstance(tail, Return)
        and isinstance(tail.value, Var)
        and tail.value.local == kept[-1][0]
    ):
        result = kept[-1][1]
        kept = kept[:-1]
    for binder, first in reversed(kept):
        result = Bind(binder=binder, first=first, then=result)
    return result


def _combined(
    outer: StaticSubstitution, inner: StaticSubstitution
) -> StaticSubstitution:
    """Two substitutions as one simultaneous substitution.

    Parameters
    ----------
    outer : StaticSubstitution
        The bindings of the enclosing telescope.
    inner : StaticSubstitution
        The bindings of the nested telescope, which shadow the enclosing
        ones of the same name.

    Returns
    -------
    StaticSubstitution
        The combined bindings.
    """
    return StaticSubstitution(
        types=tuple({**dict(outer.types), **dict(inner.types)}.items()),
        indices=tuple({**dict(outer.indices), **dict(inner.indices)}.items()),
        effects=tuple({**dict(outer.effects), **dict(inner.effects)}.items()),
    )


def _refined[NodeT: QiecModel](
    node: NodeT, rigid: Mapping[StaticVariableId, StaticArgument]
) -> NodeT:
    """A term with a case branch's rigid variables at their arguments.

    The evidence a branch grants becomes reflexivity, since both sides of
    its equality are one term once the variables are replaced.

    Parameters
    ----------
    node : NodeT
        The branch's term, or a part of it.
    rigid : Mapping[StaticVariableId, StaticArgument]
        The argument of each rigid variable.

    Returns
    -------
    NodeT
        The refined term.
    """
    return cast(NodeT, _refine(node, rigid))


def _refine(
    node: HeldValue, rigid: Mapping[StaticVariableId, StaticArgument]
) -> HeldValue:
    """Refine one node, sharing what does not change.

    Parameters
    ----------
    node : HeldValue
        A kernel record, a tuple of records, or a leaf.
    rigid : Mapping[StaticVariableId, StaticArgument]
        The argument of each rigid variable.

    Returns
    -------
    HeldValue
        The refined node.
    """
    if isinstance(node, TypeVariable | IndexVariable | EffectVariable):
        return node if node.identity is None else rigid.get(node.identity, node)
    if isinstance(node, tuple):
        return tuple(_refine(item, rigid) for item in node)
    if isinstance(node, QiecModel):
        changes: dict[str, HeldValue] = {}
        for name in type(node).__field_specs__:
            current = getattr(node, name)
            if isinstance(current, tuple | QiecModel):
                refined = _refine(current, rigid)
                if refined != current:
                    changes[name] = refined
        rewritten = node.with_(**changes) if changes else node
        if isinstance(rewritten, BranchGiven) and (
            rewritten.equality.left == rewritten.equality.right
        ):
            return Reflexivity(equality=rewritten.equality)
        return rewritten
    return node


class _Discharger:
    """The discharge of one computation.

    Parameters
    ----------
    owner : NamedComputation
        The computation.
    tables : _Tables
        The module's declarations, by identity.
    """

    def __init__(self, owner: NamedComputation, tables: _Tables) -> None:
        self.owner = owner
        self.tables = tables
        self.occupied = {parameter.name for parameter in owner.parameters}
        self.scopes: set[StaticScopeId] = set()
        self.tensors: dict[str, tuple[Value, ...]] = {}
        self.awaited: set[int] = set()
        self.serial = 0
        self.steps = 0
        self.statements = 0

    def body(self) -> Computation:
        """The computation's body with its handlers discharged.

        Returns
        -------
        Computation
            The rewritten body.

        Raises
        ------
        _Undischarged
            If a handler cannot be discharged.
        """
        environment: _Environment = {
            parameter.name: Var(local=parameter) for parameter in self.owner.parameters
        }
        path = self.reduce(self.owner.body, environment, (), None)
        return self.close(path, self.owner.type.result)

    # ------------------------------------------------------------------
    # names and bounds

    def fresh(self, name: str) -> str:
        """A name the rewritten body does not bind yet.

        Parameters
        ----------
        name : str
            The preferred name.

        Returns
        -------
        str
            The name, suffixed until it is unused.
        """
        candidate = name
        suffix = 0
        while candidate in self.occupied:
            suffix += 1
            candidate = f"{name}_{suffix}"
        self.occupied.add(candidate)
        return candidate

    def ordinal(self) -> int:
        """The next ordinal of an installation or a conditional.

        Returns
        -------
        int
            The ordinal.
        """
        self.serial += 1
        return self.serial

    def count(self, statements: int) -> None:
        """Charge statements against the size bound.

        Parameters
        ----------
        statements : int
            The statements made.

        Raises
        ------
        _Undischarged
            If the rewritten body passes the bound.
        """
        self.statements += statements
        if self.statements > DISCHARGE_STATEMENTS:
            raise _Undischarged(
                "growth",
                f"has more than {DISCHARGE_STATEMENTS} statements once its "
                f"handlers are discharged, since each resumption after the first "
                f"of a clause copies the rest of the handled computation",
                self.owner.origin,
            )

    # ------------------------------------------------------------------
    # values

    def value(self, term: Value, environment: _Environment) -> Value:
        """A value of a body read under an environment.

        Parameters
        ----------
        term : Value
            The value.
        environment : Mapping[str, Value]
            The value each local stands for.

        Returns
        -------
        Value
            The value over the rewritten body's names. A projection of a
            tuple is its component, and a selection from a tensor of known
            entries at a literal index is the entry.
        """
        return cast(Value, self.rewritten(term, environment))

    def rewritten(self, term: _Term, environment: _Environment) -> _Term:
        """One part of a value read under an environment.

        Parameters
        ----------
        term : _Term
            The part.
        environment : Mapping[str, Value]
            The value each local stands for.

        Returns
        -------
        _Term
            The rewritten part, the input itself where nothing changed.
        """
        if isinstance(term, Var):
            return environment[term.local.name]
        if isinstance(term, tuple):
            return tuple(self.rewritten(item, environment) for item in term)
        if isinstance(term, Comprehension):
            binder = Local(name=self.fresh(term.binder.name), type=term.binder.type)
            inner = {**environment, term.binder.name: Var(local=binder)}
            return term.with_(binder=binder, body=self.value(term.body, inner))
        if isinstance(term, LetValue):
            scope = dict(environment)
            bindings: list[LetBinding] = []
            for binding in term.bindings:
                bound = self.value(binding.value, scope)
                binder = Local(
                    name=self.fresh(binding.binder.name), type=binding.binder.type
                )
                bindings.append(LetBinding(binder=binder, value=bound))
                scope[binding.binder.name] = Var(local=binder)
            return term.with_(
                bindings=tuple(bindings), body=self.value(term.body, scope)
            )
        if isinstance(term, Projection):
            source = self.value(term.value, environment)
            if isinstance(source, TupleValue):
                return source.items[term.position]
            return term.with_(value=source)
        if isinstance(term, Gather):
            source = self.value(term.value, environment)
            index = self.value(term.index, environment)
            entry = self.entry(source, index)
            return term.with_(value=source, index=index) if entry is None else entry
        if isinstance(term, _VALUES):
            record = cast(Value, term)
            changes: dict[str, _Term] = {}
            for name in type(record).__field_specs__:
                current = getattr(record, name)
                if isinstance(current, (tuple, *_VALUES)):
                    changes[name] = self.rewritten(current, environment)
            return record.with_(**changes) if changes else record
        return term

    def entry(self, source: Value, index: Value) -> Value | None:
        """The entry of a tensor of known entries at a literal index.

        Parameters
        ----------
        source : Value
            The tensor.
        index : Value
            The index.

        Returns
        -------
        Value or None
            The entry, or ``None`` when the tensor's entries or the index
            are known only at run time.
        """
        if not isinstance(source, Var) or not isinstance(index, LiteralValue):
            return None
        entries = self.tensors.get(source.local.name)
        position = index.value
        if (
            entries is None
            or not isinstance(position, int)
            or not 0 <= position < len(entries)
        ):
            return None
        return entries[position]

    def atom(
        self, value: Value, hint: str, type_: TypeExpr
    ) -> tuple[_Statements, Value]:
        """A value as one that can be read any number of times.

        A variable, a literal, a tuple, and a constructor application are
        read in place, their components bound first where they are neither.
        Any other value is bound to a fresh name. A tensor's entries are
        bound before it, so a selection at a literal index reads the entry.

        Parameters
        ----------
        value : Value
            The value, over the rewritten body's names.
        hint : str
            The preferred name of the binding.
        type_ : TypeExpr
            The value's type.

        Returns
        -------
        tuple[tuple[tuple[Local, Computation], ...], Value]
            The bindings made, and the value reading them.
        """
        if isinstance(value, _ATOMS):
            return (), value
        if isinstance(value, TupleValue):
            statements: list[tuple[Local, Computation]] = []
            items: list[Value] = []
            for item, component in zip(
                value.items,
                cast(tuple[TypeExpr, ...], _arguments(value.result_type)),
                strict=True,
            ):
                made, read = self.atom(item, hint, component)
                statements.extend(made)
                items.append(read)
            return tuple(statements), value.with_(items=tuple(items))
        if isinstance(value, ConstructorValue):
            fields = self.tables.field_types(value)
            statements = []
            items = []
            for item, component in zip(value.fields, fields, strict=True):
                made, read = self.atom(item, hint, component)
                statements.extend(made)
                items.append(read)
            return tuple(statements), value.with_(fields=tuple(items))
        statements = []
        if isinstance(value, TensorValue):
            entries: list[Value] = []
            for item in value.items:
                made, read = self.atom(item, hint, _entry_type(type_))
                statements.extend(made)
                entries.append(read)
            value = value.with_(items=tuple(entries))
        binder = Local(name=self.fresh(hint), type=type_)
        self.count(1)
        if isinstance(value, TensorValue):
            self.tensors[binder.name] = value.items
        return (*statements, (binder, Return(value=value))), Var(local=binder)

    # ------------------------------------------------------------------
    # reduction

    def reduce(  # noqa: C901, PLR0912, PLR0915
        self,
        state: _State,
        environment: _Environment,
        stack: tuple[_Frame, ...],
        join: int | None,
    ) -> _Path:
        """Reduce a term under a continuation.

        Parameters
        ----------
        state : Computation | Value
            The term, or a value over the rewritten body's names returned to
            the continuation.
        environment : Mapping[str, Value]
            The value each local of the term stands for.
        stack : tuple[_Frame, ...]
            The continuation, innermost frame first.
        join : int or None
            The join the reduction stops at, or ``None`` for the
            computation's end.

        Returns
        -------
        _Path
            How the reduction ended. A value that arrives at a join an
            enclosing reduction awaits ends this one there.

        Raises
        ------
        _Undischarged
            If a handler cannot be discharged.
        """
        made: list[tuple[Local, Computation]] = []
        while True:
            self.steps += 1
            if self.steps > DISCHARGE_STEPS:
                raise _Undischarged(
                    "growth",
                    f"takes more than {DISCHARGE_STEPS} steps to discharge",
                    self.owner.origin,
                )
            if isinstance(state, _VALUES):
                value = cast(Value, state)
                if not stack:
                    return _Arrival(
                        join=None, statements=tuple(made), value=value, stack=()
                    )
                frame, stack = stack[0], stack[1:]
                if isinstance(frame, _BindFrame):
                    bound, read = self.atom(value, frame.binder.name, frame.binder.type)
                    made.extend(bound)
                    environment = {**frame.environment, frame.binder.name: read}
                    state = frame.then
                elif isinstance(frame, _HandlerFrame):
                    answer = frame.handler.return_clause
                    if answer is not None:
                        bound, read = self.atom(
                            value,
                            answer.binder.name,
                            substitute_type(answer.binder.type, frame.statics),
                        )
                        made.extend(bound)
                        environment = {
                            **self.parameters(frame),
                            answer.binder.name: read,
                        }
                        state = substitute_computation(answer.body, frame.statics)
                elif isinstance(frame, _JoinFrame) and frame.serial in self.awaited:
                    return _Arrival(
                        join=frame.serial,
                        statements=tuple(made),
                        value=value,
                        stack=stack,
                    )
                continue
            term = cast(Computation, state)
            if isinstance(term, Return):
                state = self.value(term.value, environment)
            elif isinstance(term, Bind):
                stack = (
                    _BindFrame(
                        binder=term.binder, then=term.then, environment=environment
                    ),
                    *stack,
                )
                state = term.first
            elif isinstance(term, NewInstance):
                state = term.body
            elif isinstance(term, Handle):
                frame, bound = self.install(term, environment)
                made.extend(bound)
                stack = (frame, *stack)
                state = term.computation
            elif isinstance(term, Call):
                entered = self.enter(term, stack)
                if entered is None:
                    binder = Local(
                        name=self.fresh(_awaiting(stack, "called")),
                        type=term.result_type,
                    )
                    call = term.with_(
                        arguments=tuple(
                            self.value(item, environment) for item in term.arguments
                        )
                    )
                    made.append((binder, call))
                    self.count(1)
                    state = Var(local=binder)
                    continue
                parameters, body = entered
                scope: dict[str, Value] = {}
                for parameter, argument in zip(parameters, term.arguments, strict=True):
                    bound, read = self.atom(
                        self.value(argument, environment),
                        parameter.name,
                        parameter.type,
                    )
                    made.extend(bound)
                    scope[parameter.name] = read
                stack = (
                    _CallFrame(callee=term.callee, result=term.result_type),
                    *stack,
                )
                environment = scope
                state = body
            elif isinstance(term, Perform):
                request = term.request.with_(
                    arguments=tuple(
                        self.value(item, environment) for item in term.request.arguments
                    )
                )
                position = self.answering(request, stack)
                if position is None:
                    binder = Local(
                        name=self.fresh(_awaiting(stack, "answer")),
                        type=request.result_type,
                    )
                    made.append((binder, Perform(request=request)))
                    self.count(1)
                    state = Var(local=binder)
                    continue
                bound, environment, state = self.clause(
                    request, cast(_HandlerFrame, stack[position])
                )
                made.extend(bound)
                answering = cast(_HandlerFrame, stack[position])
                stack = (
                    _ClauseFrame(
                        captured=stack[: position + 1],
                        answer=substitute_type(
                            answering.handler.output_type, answering.statics
                        ),
                    ),
                    *stack[position + 1 :],
                )
            elif isinstance(term, Resume):
                bound, value, captured = self.resume(term, environment, stack)
                made.extend(bound)
                joins, beyond = self.clause_end(stack)
                if joins:
                    return _Arrival(
                        join=joins[0],
                        statements=tuple(made),
                        value=value,
                        stack=(*captured, *beyond),
                        resumed=True,
                    )
                state = value
                stack = (*captured, *beyond)
            elif isinstance(term, If):
                condition = self.value(term.condition, environment)
                if isinstance(condition, LiteralValue):
                    state = term.then if condition.value else term.otherwise
                    continue
                path = self.branch(
                    tuple(made),
                    _if(condition),
                    ((term.then, environment), (term.otherwise, environment)),
                    stack,
                    join,
                    duplicable=True,
                )
                if not isinstance(path, _Meeting):
                    return path
                made = list(path.statements)
                stack = path.stack
                state = path.value
            elif isinstance(term, Case):
                scrutinee = self.value(term.scrutinee, environment)
                if isinstance(scrutinee, ConstructorValue):
                    environment, state = self.matched(term, scrutinee, environment)
                    continue
                path = self.case(tuple(made), term, scrutinee, environment, stack, join)
                if not isinstance(path, _Meeting):
                    return path
                made = list(path.statements)
                stack = path.stack
                state = path.value
            else:  # pragma: no cover - closed union
                raise TypeError(f"unknown computation term {term!r}")

    def parameters(self, frame: _HandlerFrame) -> dict[str, Value]:
        """The environment a clause body of an installed handler starts from.

        Parameters
        ----------
        frame : _HandlerFrame
            The installation.

        Returns
        -------
        dict[str, Value]
            The handler's parameters at their current values.
        """
        return {
            parameter.name: argument
            for parameter, argument in zip(
                frame.handler.parameters, frame.arguments, strict=True
            )
        }

    def install(
        self, term: Handle, environment: _Environment
    ) -> tuple[_HandlerFrame, _Statements]:
        """The frame a ``handle`` puts on the continuation.

        Parameters
        ----------
        term : Handle
            The term.
        environment : Mapping[str, Value]
            The value each local stands for.

        Returns
        -------
        tuple[_HandlerFrame, tuple[tuple[Local, Computation], ...]]
            The frame, and the statements binding its parameters' values.

        Raises
        ------
        _Undischarged
            If the handler is foreign.
        """
        handler = self.tables.handlers[term.handler]
        if handler.implementation != "authored":
            clauses = " and ".join(
                f"clause `{self.tables.operations[clause.operation][1].name}`"
                for clause in handler.clauses
            )
            raise _Undischarged(
                "foreign",
                f"installs the foreign handler `{handler.name}`, whose {clauses} a "
                f"runtime provider implements",
                self.owner.origin,
            )
        statics = instantiate_telescope(handler.telescope, term.static_arguments)
        made: list[tuple[Local, Computation]] = []
        arguments: list[Value] = []
        for parameter, argument in zip(handler.parameters, term.arguments, strict=True):
            bound, read = self.atom(
                self.value(argument, environment),
                parameter.name,
                substitute_type(parameter.type, statics),
            )
            made.extend(bound)
            arguments.append(read)
        frame = _HandlerFrame(
            serial=self.ordinal(),
            instance=term.instance,
            handler=handler,
            statics=statics,
            arguments=tuple(arguments),
        )
        return frame, tuple(made)

    def answering(
        self, request: EffectRequest, stack: tuple[_Frame, ...]
    ) -> int | None:
        """The frame of the continuation that answers a request.

        Parameters
        ----------
        request : EffectRequest
            The request.
        stack : tuple[_Frame, ...]
            The continuation, innermost frame first.

        Returns
        -------
        int or None
            The position of the innermost handler of the request's instance
            with a clause for its operation, or ``None`` when the
            computation handles the instance nowhere around the request.

        Raises
        ------
        _Undischarged
            If every handler of the instance forwards the request.
        """
        forwarding: HandlerDef | None = None
        for position, frame in enumerate(stack):
            if (
                not isinstance(frame, _HandlerFrame)
                or frame.instance != request.instance
            ):
                continue
            if frame.handler.clause(request.operation) is not None:
                return position
            forwarding = frame.handler
        if forwarding is None:
            return None
        operation = self.tables.operations[request.operation][1].name
        raise _Undischarged(
            "forwarding",
            f"requests `{operation}` under the handler `{forwarding.name}`, which "
            f"has no clause `{operation}` and forwards it, and no outer handler of "
            f"the computation answers it",
            request.origin.origin,
        )

    def clause(
        self, request: EffectRequest, frame: _HandlerFrame
    ) -> tuple[_Statements, dict[str, Value], Computation]:
        """The clause body that answers a request, and what it is read under.

        Parameters
        ----------
        request : EffectRequest
            The request, over the rewritten body's names.
        frame : _HandlerFrame
            The handler that answers it.

        Returns
        -------
        tuple[tuple[tuple[Local, Computation], ...], dict[str, Value], Computation]
            The statements binding the request's arguments, the clause's
            environment, and its body at the request's static arguments.
        """
        clause = frame.handler.clause(request.operation)
        assert clause is not None and clause.body is not None
        operation = self.tables.operations[request.operation][1]
        statics = _combined(
            frame.statics,
            instantiate_telescope(operation.telescope, request.static_arguments),
        )
        made: list[tuple[Local, Computation]] = []
        environment = self.parameters(frame)
        for parameter, argument in zip(
            clause.parameters, request.arguments, strict=True
        ):
            bound, read = self.atom(
                argument, parameter.name, substitute_type(parameter.type, statics)
            )
            made.extend(bound)
            environment[parameter.name] = read
        return tuple(made), environment, substitute_computation(clause.body, statics)

    def resume(
        self, term: Resume, environment: _Environment, stack: tuple[_Frame, ...]
    ) -> tuple[_Statements, Value, tuple[_Frame, ...]]:
        """The captured continuation a ``resume`` puts back.

        Parameters
        ----------
        term : Resume
            The resumption.
        environment : Mapping[str, Value]
            The value each local of the clause stands for.
        stack : tuple[_Frame, ...]
            The continuation of the resumption.

        Returns
        -------
        tuple[tuple[tuple[Local, Computation], ...], Value, tuple[_Frame, ...]]
            The statements binding the parameters' values, the resumed
            value, and the captured continuation with the handler at those
            parameters.
        """
        frame = next(item for item in stack if isinstance(item, _ClauseFrame))
        handler = cast(_HandlerFrame, frame.captured[-1])
        made: list[tuple[Local, Computation]] = []
        arguments: list[Value] = []
        for parameter, argument in zip(
            handler.handler.parameters, term.arguments, strict=True
        ):
            bound, read = self.atom(
                self.value(argument, environment),
                parameter.name,
                substitute_type(parameter.type, handler.statics),
            )
            made.extend(bound)
            arguments.append(read)
        return (
            tuple(made),
            self.value(term.value, environment),
            (*frame.captured[:-1], handler.carrying(tuple(arguments))),
        )

    def clause_end(
        self, stack: tuple[_Frame, ...]
    ) -> tuple[tuple[int, ...], tuple[_Frame, ...]]:
        """What follows a resumption once its clause has nothing left to do.

        Parameters
        ----------
        stack : tuple[_Frame, ...]
            The continuation of the resumption.

        Returns
        -------
        tuple[tuple[int, ...], tuple[_Frame, ...]]
            For a resumption in tail position, at the end of its clause or of
            conditionals that end it, the joins of those conditionals still
            awaited, innermost first, and the continuation beyond the clause.
            For any other resumption, no join and its own continuation.
        """
        joins: list[int] = []
        for position, frame in enumerate(stack):
            if isinstance(frame, _ClauseFrame):
                return tuple(joins), stack[position + 1 :]
            if not isinstance(frame, _JoinFrame):
                break
            if frame.serial in self.awaited:
                joins.append(frame.serial)
        return (), stack

    def enter(
        self, call: Call, stack: tuple[_Frame, ...]
    ) -> tuple[tuple[Local, ...], Computation] | None:
        """The body of a callee that requests from a handler around the call.

        Parameters
        ----------
        call : Call
            The call.
        stack : tuple[_Frame, ...]
            The continuation of the call.

        Returns
        -------
        tuple[tuple[Local, ...], Computation] or None
            The callee's parameters and body at the call's static arguments,
            or ``None`` for a callee whose row names no handled instance,
            which stays a call.

        Raises
        ------
        _Undischarged
            If the callee is being entered already, since the requests it
            makes then depend on the values it is given.
        """
        handled = {
            frame.instance: frame.handler
            for frame in reversed(stack)
            if isinstance(frame, _HandlerFrame)
        }
        requested = [
            handled[entry.instance]
            for entry in call.effects.entries
            if entry.instance in handled
        ]
        if not requested and (call.effects.tail is None or not handled):
            return None
        callee = self.tables.computations[call.callee]
        if callee.id == self.owner.id or call.callee in _entered(stack):
            handler = requested[0] if requested else next(iter(handled.values()))
            raise _Undischarged(
                "recursion",
                f"reaches a recursive call of `{call.name}` that requests from the "
                f"handler `{handler.name}`, so how often its clauses resume depends "
                f"on the values the call is given",
                call.origin,
            )
        statics = instantiate_telescope(callee.telescope, call.static_arguments)
        return (
            tuple(
                Local(name=local.name, type=substitute_type(local.type, statics))
                for local in callee.parameters
            ),
            substitute_computation(callee.body, statics),
        )

    # ------------------------------------------------------------------
    # conditionals

    def branch(
        self,
        made: _Statements,
        rebuild: Callable[[tuple[Computation, ...], TypeExpr], Computation],
        branches: tuple[tuple[Computation, _Environment], ...],
        stack: tuple[_Frame, ...],
        join: int | None,
        *,
        duplicable: bool,
    ) -> _Path | _Meeting:
        """Reduce the branches of a conditional, and bring them together.

        Branches that arrive at one join with one continuation, up to the
        values of handler parameters, meet: the conditional is bound to what
        the branches' values and parameters differ in, and the continuation
        is reduced once. Any other branches are followed separately.

        Parameters
        ----------
        made : tuple[tuple[Local, Computation], ...]
            The statements before the conditional.
        rebuild : Callable[[tuple[Computation, ...], TypeExpr], Computation]
            The conditional over its rewritten branches and their type.
        branches : tuple[tuple[Computation, Mapping[str, Value]], ...]
            Each branch, and the environment it is read under.
        stack : tuple[_Frame, ...]
            The continuation of the conditional.
        join : int or None
            The join the enclosing reduction stops at.
        duplicable : bool
            Whether the conditional may be stated more than once.

        Returns
        -------
        _Path | _Meeting
            Where the branches meet, when the reduction continues there, or
            how the conditional ended.

        Raises
        ------
        _Undischarged
            If a handler cannot be discharged.
        """
        if not stack and join is None:
            # Nothing follows, so the conditional is the result.
            paths = tuple(
                self.reduce(body, environment, (), None)
                for body, environment in branches
            )
            return _Fork(statements=made, rebuild=rebuild, paths=paths)
        serial = self.ordinal()
        self.awaited.add(serial)
        paths = tuple(
            self.reduce(body, environment, (_JoinFrame(serial=serial), *stack), serial)
            for body, environment in branches
        )
        self.awaited.discard(serial)
        meeting = self.meet(made, rebuild, paths, duplicable=duplicable)
        if meeting is not None:
            if meeting.join != serial:
                return meeting
            outer = stack[0] if stack else None
            if (
                meeting.resumed
                and isinstance(outer, _JoinFrame)
                and outer.serial in self.awaited
            ):
                # The resumption that ends this conditional ends the one
                # around it.
                return meeting.with_(join=outer.serial)
            return _Meeting(
                statements=meeting.statements,
                value=meeting.value,
                stack=meeting.stack,
            )
        return _Fork(
            statements=made,
            rebuild=rebuild,
            paths=tuple(self.follow(path, serial, join) for path in paths),
        )

    def follow(self, path: _Path, serial: int, join: int | None) -> _Path:
        """Continue the paths that arrived at a join past it.

        Parameters
        ----------
        path : _Path
            How a branch ended.
        serial : int
            The join.
        join : int or None
            The join the enclosing reduction stops at.

        Returns
        -------
        _Path
            How the branch ends past the join.
        """
        if isinstance(path, _Fork):
            return path.with_(
                paths=tuple(self.follow(item, serial, join) for item in path.paths)
            )
        if path.join != serial:
            return path
        further = self.reduce(path.value, {}, path.stack, join)
        return further.with_(statements=(*path.statements, *further.statements))

    def meet(
        self,
        made: _Statements,
        rebuild: Callable[[tuple[Computation, ...], TypeExpr], Computation],
        paths: tuple[_Path, ...],
        *,
        duplicable: bool,
    ) -> _Arrival | None:
        """Bring paths together that arrived at one join.

        Parameters
        ----------
        made : tuple[tuple[Local, Computation], ...]
            The statements before the conditional.
        rebuild : Callable[[tuple[Computation, ...], TypeExpr], Computation]
            The conditional over its rewritten branches and their type.
        paths : tuple[_Path, ...]
            How each branch ended.
        duplicable : bool
            Whether the conditional may be stated more than once.

        Returns
        -------
        _Arrival or None
            The arrival of the whole conditional, or ``None`` when the paths
            do not arrive at one join with one continuation.
        """
        arrivals = [path for path in paths if isinstance(path, _Arrival)]
        first = arrivals[0] if arrivals else None
        if (
            first is None
            or len(arrivals) != len(paths)
            or first.join is None
            or any(
                item.join != first.join
                or item.resumed != first.resumed
                or not _same(item.stack, first.stack)
                for item in arrivals
            )
        ):
            return None
        differing: list[tuple[TypeExpr, str, tuple[Value, ...], Local]] = []
        value = self.reconciled(
            tuple(item.value for item in arrivals),
            self.accepts(first.stack),
            "chosen",
            differing,
        )
        frames: list[_Frame] = []
        for position, frame in enumerate(first.stack):
            if not isinstance(frame, _HandlerFrame):
                frames.append(frame)
                continue
            rows = [
                cast(_HandlerFrame, item.stack[position]).arguments for item in arrivals
            ]
            carried = tuple(
                self.reconciled(
                    tuple(row[slot] for row in rows),
                    substitute_type(parameter.type, frame.statics),
                    parameter.name,
                    differing,
                )
                for slot, parameter in enumerate(frame.handler.parameters)
            )
            frames.append(
                frame if carried == frame.arguments else frame.carrying(carried)
            )
        stated = self.stated(rebuild, arrivals, differing, duplicable=duplicable)
        self.count(len(stated))
        return _Arrival(
            join=first.join,
            resumed=first.resumed,
            statements=(*made, *stated),
            value=_read_chosen(value, differing, stated),
            stack=tuple(
                frame.carrying(
                    tuple(
                        _read_chosen(item, differing, stated)
                        for item in frame.arguments
                    )
                )
                if isinstance(frame, _HandlerFrame) and differing
                else frame
                for frame in frames
            ),
        )

    def reconciled(
        self,
        values: tuple[Value, ...],
        type_: TypeExpr,
        hint: str,
        differing: list[tuple[TypeExpr, str, tuple[Value, ...], Local]],
    ) -> Value:
        """The value a conditional gives from the value of each branch.

        Parameters
        ----------
        values : tuple[Value, ...]
            The value of each branch.
        type_ : TypeExpr
            Their type.
        hint : str
            The preferred name of the conditional's result.
        differing : list[tuple[TypeExpr, str, tuple[Value, ...], Local]]
            What the branches differ in, extended with the parts of the
            values that do.

        Returns
        -------
        Value
            The value: the branches' own where they agree, and otherwise a
            local standing for the conditional, component by component of
            tuples and of applications of one constructor.
        """
        first = values[0]
        if all(item == first for item in values):
            return first
        if all(isinstance(item, TupleValue) for item in values):
            rows = [cast(TupleValue, item).items for item in values]
            return cast(TupleValue, first).with_(
                items=tuple(
                    self.reconciled(
                        tuple(row[slot] for row in rows),
                        cast(TypeExpr, component),
                        hint,
                        differing,
                    )
                    for slot, component in enumerate(_arguments(type_))
                )
            )
        if all(
            isinstance(item, ConstructorValue)
            and item.constructor == cast(ConstructorValue, first).constructor
            and item.static_arguments == cast(ConstructorValue, first).static_arguments
            for item in values
        ):
            head = cast(ConstructorValue, first)
            rows = [cast(ConstructorValue, item).fields for item in values]
            return head.with_(
                fields=tuple(
                    self.reconciled(
                        tuple(row[slot] for row in rows), component, hint, differing
                    )
                    for slot, component in enumerate(self.tables.field_types(head))
                )
            )
        local = Local(name=self.fresh(hint), type=type_)
        differing.append((type_, hint, values, local))
        return Var(local=local)

    def stated(
        self,
        rebuild: Callable[[tuple[Computation, ...], TypeExpr], Computation],
        arrivals: list[_Arrival],
        differing: list[tuple[TypeExpr, str, tuple[Value, ...], Local]],
        *,
        duplicable: bool,
    ) -> _Statements:
        """The statements binding what a conditional's branches differ in.

        Branches that make no request are stated once per part they differ
        in, each with the values the part reads. Branches that make requests
        are stated once, their parts as one tuple.

        Parameters
        ----------
        rebuild : Callable[[tuple[Computation, ...], TypeExpr], Computation]
            The conditional over its rewritten branches and their type.
        arrivals : list[_Arrival]
            The branches.
        differing : list[tuple[TypeExpr, str, tuple[Value, ...], Local]]
            The parts: their type, hint, value in each branch, and local.
        duplicable : bool
            Whether the conditional may be stated more than once.

        Returns
        -------
        tuple[tuple[Local, Computation], ...]
            The statements.
        """
        pure = all(
            _pure(first) for item in arrivals for _binder, first in item.statements
        )
        if not differing:
            if not any(_live(item.statements, frozenset()) for item in arrivals):
                return ()
            unit = LiteralValue(value=None, type=UNIT)
            binder = Local(name=self.fresh("done"), type=UNIT)
            branches = tuple(
                _sequence(item.statements, Return(value=unit)) for item in arrivals
            )
            return ((binder, rebuild(branches, UNIT)),)
        if len(differing) == 1 or (pure and duplicable):
            return tuple(
                (
                    local,
                    rebuild(
                        tuple(
                            _sequence(item.statements, Return(value=part))
                            for item, part in zip(arrivals, values, strict=True)
                        ),
                        type_,
                    ),
                )
                for type_, _hint, values, local in differing
            )
        joint = product_type(*(type_ for type_, _hint, _values, _local in differing))
        binder = Local(name=self.fresh("chosen"), type=joint)
        branches = tuple(
            _sequence(
                item.statements,
                Return(
                    value=TupleValue(
                        items=tuple(part[2][row] for part in differing),
                        result_type=joint,
                    )
                ),
            )
            for row, item in enumerate(arrivals)
        )
        return ((binder, rebuild(branches, joint)),)

    def accepts(self, stack: tuple[_Frame, ...]) -> TypeExpr:
        """The type of the value a continuation awaits.

        Parameters
        ----------
        stack : tuple[_Frame, ...]
            The continuation, innermost frame first.

        Returns
        -------
        TypeExpr
            The type.
        """
        for frame in stack:
            if isinstance(frame, _BindFrame):
                return frame.binder.type
            if isinstance(frame, _HandlerFrame):
                return substitute_type(frame.handler.input_type, frame.statics)
            if isinstance(frame, _ClauseFrame):
                return frame.answer
            if isinstance(frame, _CallFrame):
                return frame.result
        return self.owner.type.result

    def matched(
        self, term: Case, scrutinee: ConstructorValue, environment: _Environment
    ) -> tuple[_Environment, Computation]:
        """The branch of a case analysis a constructor application selects.

        Parameters
        ----------
        term : Case
            The case analysis.
        scrutinee : ConstructorValue
            The application.
        environment : Mapping[str, Value]
            The value each local stands for.

        Returns
        -------
        tuple[Mapping[str, Value], Computation]
            The environment extended by the branch's fields, and the
            branch's body at the application's static arguments.
        """
        branch = next(
            item for item in term.branches if item.constructor == scrutinee.constructor
        )
        local = scrutinee.static_arguments[
            len(scrutinee.static_arguments) - len(branch.static_arguments) :
        ]
        rigid = {
            variable.identity: argument
            for variable, argument in zip(branch.static_arguments, local, strict=True)
            if isinstance(variable, TypeVariable | IndexVariable | EffectVariable)
            and variable.identity is not None
        }
        body = cast(Computation, _refined(branch.body, rigid)) if rigid else branch.body
        fields = {
            field.name: value
            for field, value in zip(branch.fields, scrutinee.fields, strict=True)
        }
        return {**environment, **fields}, body

    def case(
        self,
        made: _Statements,
        term: Case,
        scrutinee: Value,
        environment: _Environment,
        stack: tuple[_Frame, ...],
        join: int | None,
    ) -> _Path | _Meeting:
        """Reduce a case analysis on a value known only at run time.

        Parameters
        ----------
        made : tuple[tuple[Local, Computation], ...]
            The statements before it.
        term : Case
            The case analysis.
        scrutinee : Value
            The value analyzed, over the rewritten body's names.
        environment : Mapping[str, Value]
            The value each local stands for.
        stack : tuple[_Frame, ...]
            The continuation.
        join : int or None
            The join the enclosing reduction stops at.

        Returns
        -------
        _Path | _Meeting
            Where the branches meet, or how the case analysis ended.

        Raises
        ------
        _Undischarged
            If the branches do not meet, or the case analysis is reached
            more than once, so its static scopes would be entered twice.
        """
        branches: list[tuple[Computation, _Environment]] = []
        patterns: list[CaseBranch] = []
        for branch in term.branches:
            fields = tuple(
                Local(name=self.fresh(field.name), type=field.type)
                for field in branch.fields
            )
            branches.append(
                (
                    branch.body,
                    {
                        **environment,
                        **{
                            field.name: Var(local=local)
                            for field, local in zip(branch.fields, fields, strict=True)
                        },
                    },
                )
            )
            patterns.append(branch.with_(fields=fields))
        repeated = any(branch.scope in self.scopes for branch in term.branches)
        self.scopes.update(branch.scope for branch in term.branches)

        def rebuild(bodies: tuple[Computation, ...], type_: TypeExpr) -> Computation:
            """The case analysis over its rewritten branches.

            Parameters
            ----------
            bodies : tuple[Computation, ...]
                The rewritten branches.
            type_ : TypeExpr
                The type they have.

            Returns
            -------
            Computation
                The case analysis.
            """
            return Case(
                scrutinee=scrutinee,
                motive=CaseMotive(indices=term.motive.indices, result_type=type_),
                branches=tuple(
                    pattern.with_(body=body)
                    for pattern, body in zip(patterns, bodies, strict=True)
                ),
            )

        path = self.branch(
            made, rebuild, tuple(branches), stack, join, duplicable=False
        )
        if repeated or (isinstance(path, _Fork) and stack):
            raise _Undischarged(
                "case",
                "branches with `case` on a value known only at run time, and the "
                "handled requests of its branches, or a clause that resumes more "
                "than once, would need the rest of the computation in every branch",
                self.owner.origin,
            )
        return path

    # ------------------------------------------------------------------
    # assembly

    def close(self, path: _Path, result: TypeExpr) -> Computation:
        """A finished path as a computation.

        Parameters
        ----------
        path : _Path
            The path.
        result : TypeExpr
            The type of its value.

        Returns
        -------
        Computation
            The computation.
        """
        if isinstance(path, _Arrival):
            return _sequence(path.statements, Return(value=path.value))
        return _sequence(
            path.statements,
            path.rebuild(
                tuple(self.close(item, result) for item in path.paths), result
            ),
        )


class _Tables:
    """A module's declarations, by identity.

    Parameters
    ----------
    module : QiecModule
        The module.
    """

    def __init__(self, module: QiecModule) -> None:
        self.handlers: dict[HandlerId, HandlerDef] = {
            item.id: item for item in module.handlers
        }
        self.computations: dict[ComputationId, NamedComputation] = {
            item.id: item for item in module.computations
        }
        self.operations = _operations(module)
        self.constructors: dict[ConstructorId, ConstructorDecl] = {
            item.id: item for item in module.constructors
        }
        self.families = {item.id: item for item in module.families}

    def field_types(self, value: ConstructorValue) -> tuple[TypeExpr, ...]:
        """The types of a constructor application's fields.

        Parameters
        ----------
        value : ConstructorValue
            The application.

        Returns
        -------
        tuple[TypeExpr, ...]
            One type per field, at the application's static arguments.
        """
        constructor = self.constructors[value.constructor]
        family = self.families[constructor.family]
        count = len(family.parameters)
        substitution = _combined(
            instantiate_telescope(family.parameters, value.static_arguments[:count]),
            instantiate_telescope(
                constructor.telescope, value.static_arguments[count:]
            ),
        )
        return tuple(
            substitute_type(field.type, substitution) for field in constructor.fields
        )


def _entry_type(type_: TypeExpr) -> TypeExpr:
    """The type of a tensor's entries along its outermost axis.

    Parameters
    ----------
    type_ : TypeExpr
        The tensor type.

    Returns
    -------
    TypeExpr
        The element type, or the tensor type one rank lower.
    """
    split = tensor_shape(type_)
    assert split is not None
    element, dimensions = split
    return tensor_type(element, dimensions[1:]) if dimensions[1:] else element


def _arguments(type_: TypeExpr) -> tuple[StaticArgument, ...]:
    """The components of a product type.

    Parameters
    ----------
    type_ : TypeExpr
        The product type.

    Returns
    -------
    tuple[StaticArgument, ...]
        Its components.
    """
    return cast(tuple[StaticArgument, ...], getattr(type_, "arguments"))


def _if(
    condition: Value,
) -> Callable[[tuple[Computation, ...], TypeExpr], Computation]:
    """The conditional on a condition, over its rewritten branches.

    Parameters
    ----------
    condition : Value
        The condition.

    Returns
    -------
    Callable[[tuple[Computation, ...], TypeExpr], Computation]
        The conditional.
    """

    def rebuild(branches: tuple[Computation, ...], _type: TypeExpr) -> Computation:
        """The conditional over its rewritten branches.

        Parameters
        ----------
        branches : tuple[Computation, ...]
            The branch where the condition holds, and the other.
        _type : TypeExpr
            The type they have, which a conditional does not record.

        Returns
        -------
        Computation
            The conditional.
        """
        then, otherwise = branches
        return If(condition=condition, then=then, otherwise=otherwise)

    return rebuild


def _awaiting(stack: tuple[_Frame, ...], default: str) -> str:
    """The name a continuation binds its value to.

    Parameters
    ----------
    stack : tuple[_Frame, ...]
        The continuation, innermost frame first.
    default : str
        The name when the continuation binds nothing.

    Returns
    -------
    str
        The binder's name, which a call's sites are named under.
    """
    frame = stack[0] if stack else None
    return frame.binder.name if isinstance(frame, _BindFrame) else default


def _entered(stack: tuple[_Frame, ...]) -> frozenset[ComputationId]:
    """The computations a continuation is inside of.

    Parameters
    ----------
    stack : tuple[_Frame, ...]
        The continuation.

    Returns
    -------
    frozenset[ComputationId]
        Every computation entered and not returned from. A computation a
        running clause captured is left out, since the clause runs outside
        the handler that captured it.
    """
    return frozenset(frame.callee for frame in stack if isinstance(frame, _CallFrame))


def _same(left: tuple[_Frame, ...], right: tuple[_Frame, ...]) -> bool:
    """Whether two continuations are one up to handler parameter values.

    Parameters
    ----------
    left : tuple[_Frame, ...]
        One continuation.
    right : tuple[_Frame, ...]
        The other.

    Returns
    -------
    bool
        True when they hold the same frames in order, a handler's frame
        being the same as another of its installation.
    """
    return len(left) == len(right) and all(
        a is b
        or (
            isinstance(a, _HandlerFrame)
            and isinstance(b, _HandlerFrame)
            and a.serial == b.serial
        )
        for a, b in zip(left, right, strict=True)
    )


def _read_chosen(
    value: Value,
    differing: list[tuple[TypeExpr, str, tuple[Value, ...], Local]],
    stated: _Statements,
) -> Value:
    """A reconciled value reading the statements that bind its parts.

    Parameters
    ----------
    value : Value
        The value, whose differing parts are locals.
    differing : list[tuple[TypeExpr, str, tuple[Value, ...], Local]]
        The parts and their locals.
    stated : tuple[tuple[Local, Computation], ...]
        The statements made for them: one per part, or one tuple of them.

    Returns
    -------
    Value
        The value, its parts read by name or as components of the tuple.
    """
    if len(stated) == len(differing):
        return value
    joint = Var(local=stated[0][0])
    reads = {
        local.name: Projection(value=joint, position=position, result_type=type_)
        for position, (type_, _hint, _values, local) in enumerate(differing)
    }

    def rewrite(term: Value) -> Value:
        """Read each part as a component of the tuple.

        Parameters
        ----------
        term : Value
            A part of the reconciled value.

        Returns
        -------
        Value
            The rewritten part.
        """
        if isinstance(term, Var):
            return reads.get(term.local.name, term)
        if isinstance(term, TupleValue):
            return term.with_(items=tuple(rewrite(item) for item in term.items))
        if isinstance(term, ConstructorValue):
            return term.with_(fields=tuple(rewrite(item) for item in term.fields))
        return term

    return rewrite(value)


def installs_handler(computation: NamedComputation) -> bool:
    """Whether a computation's body installs a handler or makes an instance.

    Parameters
    ----------
    computation : NamedComputation
        The computation.

    Returns
    -------
    bool
        True when a ``handle`` or a local instance occurs in the body.
    """
    return any(
        isinstance(term, Handle | NewInstance) for term in _subterms(computation.body)
    )


def discharge_computation(
    computation: NamedComputation, module: QiecModule
) -> NamedComputation | DischargeObstacle:
    """Discharge one computation's handlers.

    Parameters
    ----------
    computation : NamedComputation
        A checked computation of the module.
    module : QiecModule
        The module, whose handlers and callees are read.

    Returns
    -------
    NamedComputation | DischargeObstacle
        The computation with a body that installs no handler, or why its
        handlers remain.
    """
    discharger = _Discharger(computation, _Tables(module))
    try:
        body = discharger.body()
    except _Undischarged as error:
        return DischargeObstacle(
            computation=computation.name,
            kind=error.kind,
            reason=error.reason,
            origin=error.origin,
        )
    return computation.with_(body=body)


def discharge_module(module: QiecModule) -> DischargedModule:
    """Discharge the handlers of every computation of a module.

    Parameters
    ----------
    module : QiecModule
        The checked module.

    Returns
    -------
    DischargedModule
        The module itself when no computation installs a handler, else the
        module with every dischargeable computation rewritten and one
        obstacle per computation left as it is.
    """
    computations: list[NamedComputation] = []
    obstacles: list[DischargeObstacle] = []
    for computation in module.computations:
        if not installs_handler(computation):
            computations.append(computation)
            continue
        result = discharge_computation(computation, module)
        if isinstance(result, DischargeObstacle):
            obstacles.append(result)
            computations.append(computation)
        else:
            computations.append(result)
    if all(
        new is old for new, old in zip(computations, module.computations, strict=True)
    ):
        return DischargedModule(module=module, obstacles=tuple(obstacles))
    return DischargedModule(
        module=module.with_(computations=tuple(computations)),
        obstacles=tuple(obstacles),
    )


__all__ = [
    "DISCHARGE_STATEMENTS",
    "DISCHARGE_STEPS",
    "DischargeObstacle",
    "DischargedModule",
    "clause_requests",
    "discharge_computation",
    "discharge_module",
    "installs_handler",
]
