"""Thread the state of one closed linear handler through recursive calls."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

from quivers.qiec._models import QiecModel
from quivers.qiec.effects import (
    EMPTY_ROW,
    ComputationType,
    EffectDef,
    EffectRow,
    HandlerDef,
    ResumptionGrade,
)
from quivers.qiec.handler_carriers import _branches, _CarrierAnalysis, _handler_obstacle
from quivers.qiec.identifiers import ComputationId, OperationId, SourceOrigin
from quivers.qiec.module import (
    NamedComputation,
    QiecModule,
    _children,
    _return_observes_address,
    validate_module,
)
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    Computation,
    Handle,
    HandlerReturnAddress,
    If,
    Local,
    Perform,
    Projection,
    Resume,
    Return,
    TupleValue,
    Value,
    Var,
    _RETURN_ADDRESS,
)
from quivers.qiec.types import TypeExpr, product_type


class StateSpecializationUnsupported(ValueError):
    """The selected installation is outside the proved state fragment."""


class StateSpecializationLimit(StateSpecializationUnsupported):
    """State specialization exceeds the requested construction bound."""


class _LocalNames:
    """Allocate locals fresh even under source binders inside value terms.

    Parameters
    ----------
    owner : NamedComputation
        The declaration whose lexical names to reserve.
    """

    def __init__(self, owner: NamedComputation) -> None:
        self.occupied: set[str] = set()
        pending: list[object] = [owner.parameters, owner.body]
        visited: set[int] = set()
        while pending:
            item = pending.pop()
            if isinstance(item, Local):
                self.occupied.add(item.name)
            elif id(item) not in visited:
                visited.add(id(item))
                pending.extend(_children(item))
        self.ordinal = 0

    def fresh(self, type_: TypeExpr) -> Local:
        """Make a binder disjoint from all authored locals.

        Parameters
        ----------
        type_ : TypeExpr
            The new local's checked type.

        Returns
        -------
        Local
            A fresh local.
        """
        while True:
            self.ordinal += 1
            name = f"__state_{self.ordinal}"
            if name not in self.occupied:
                return Local(name=name, type=type_)


def _relocate(node: object, names: _LocalNames, renamed: dict[str, str]) -> object:
    """Rename locally bound values in code transplanted from a closed clause.

    Parameters
    ----------
    node : object
        The code or parameter record being copied.
    names : _LocalNames
        Fresh names disjoint from the destination's existing bindings.
    renamed : dict[str, str]
        The shared correspondence between old and new lexical names.

    Returns
    -------
    object
        The input with its local names replaced.
    """
    if isinstance(node, Local):
        name = renamed.get(node.name)
        if name is None:
            name = names.fresh(node.type).name
            renamed[node.name] = name
        return node.with_(name=name)
    if isinstance(node, tuple):
        items = tuple(_relocate(item, names, renamed) for item in node)
        return node if all(a is b for a, b in zip(node, items, strict=True)) else items
    if isinstance(node, QiecModel):
        changes: dict[str, object] = {}
        for field in type(node).__field_specs__:
            current = getattr(node, field)
            moved = _relocate(current, names, renamed)
            if moved is not current:
                changes[field] = moved
        return node.with_(**changes) if changes else node
    return node


@dataclass(frozen=True, slots=True)
class _Transition:
    """The checked source of one state transition.

    Parameters
    ----------
    parameters : tuple[Local, ...]
        Request binders followed by the old state binders, in order.
    body : Computation
        The operation clause with one tail resumption on each path.
    result : TypeExpr
        The product of its reply and new state.
    """

    parameters: tuple[Local, ...]
    body: Computation
    result: TypeExpr


@dataclass(frozen=True, slots=True)
class _Companion:
    """The signature shared by calls and the generated definition.

    Parameters
    ----------
    id : ComputationId
        The companion's nominal identity.
    name : str
        Its unique display name.
    result : TypeExpr
        The result of its pure body.
    """

    id: ComputationId
    name: str
    result: TypeExpr

    def call(self, arguments: tuple[Value, ...], origin: SourceOrigin) -> Call:
        """Apply a pure companion at a known source origin.

        Parameters
        ----------
        arguments : tuple[Value, ...]
            The arguments in evaluation order.
        origin : SourceOrigin
            The original or generated call site.

        Returns
        -------
        Call
            The checked-signature call.
        """
        return Call(
            callee=self.id,
            name=self.name,
            static_arguments=(),
            arguments=arguments,
            result_type=self.result,
            effects=EMPTY_ROW,
            origin=origin,
        )

    def declaration(
        self, parameters: tuple[Local, ...], body: Computation, origin: SourceOrigin
    ) -> NamedComputation:
        """Declare the companion with the previously allocated signature.

        Parameters
        ----------
        parameters : tuple[Local, ...]
            The companion's ordered binders.
        body : Computation
            Its pure body.
        origin : SourceOrigin
            The source location of its declaration.

        Returns
        -------
        NamedComputation
            The definition referred to by companion calls.
        """
        return NamedComputation(
            id=self.id,
            name=self.name,
            telescope=(),
            parameters=parameters,
            body=body,
            type=ComputationType(effects=EMPTY_ROW, result=self.result),
            origin=origin,
        )


class _Specialization:
    """Build a finite companion graph for one authored installation.

    Parameters
    ----------
    module : QiecModule
        The checked source and its call graph.
    owner : NamedComputation
        The declaration to specialize.
    installation : Handle
        The selected installation.
    handler : HandlerDef
        The actual linked handler declaration.
    effect : EffectDef
        The interface of its handled instance.
    row : EffectRow
        The singleton row of that exact instance.
    max_terms : int
        The bound on constructed computation terms.
    """

    def __init__(
        self,
        module: QiecModule,
        owner: NamedComputation,
        installation: Handle,
        handler: HandlerDef,
        effect: EffectDef,
        row: EffectRow,
        max_terms: int,
    ) -> None:
        self.module = module
        self.owner = owner
        self.installation = installation
        self.handler = handler
        self.effect = effect
        self.row = row
        self.max_terms = max_terms
        self.terms = 0
        self.ordinary = _CarrierAnalysis(module)
        self.state_types = tuple(parameter.type for parameter in handler.parameters)
        self.case_constructors = frozenset(
            constructor
            for family in module.families
            if family.closed and not family.parameters and not family.indices
            for constructor in family.constructors
        )
        self.computations = self.ordinary.computations
        self.names = {item.name for item in module.computations}
        self.identities = set(self.computations)
        self.operations: dict[OperationId, _Transition] = {}
        self.companions: dict[ComputationId, _Companion] = {}
        self.pending: deque[NamedComputation] = deque()
        self.generated: list[NamedComputation] = []

    def spend(self, terms: int) -> None:
        """Bound the number of constructed computation terms.

        Parameters
        ----------
        terms : int
            The number of new nodes to charge.

        Raises
        ------
        StateSpecializationLimit
            If the resulting tree would exceed the bound.
        """
        self.terms += terms
        if self.terms > self.max_terms:
            raise StateSpecializationLimit(
                f"state specialization exceeds its bound of {self.max_terms} computation terms"
            )

    def pair(self, result: TypeExpr) -> TypeExpr:
        """The result paired with each coordinate of the new state.

        Parameters
        ----------
        result : TypeExpr
            The original result type.

        Returns
        -------
        TypeExpr
            The product with the result in position zero.
        """
        return product_type(result, *self.state_types)

    def split(self, local: Local, result: TypeExpr) -> tuple[Value, tuple[Value, ...]]:
        """Project the result and new state from one evaluated computation.

        Parameters
        ----------
        local : Local
            The binder for the whole evaluated product.
        result : TypeExpr
            The type of its ordinary result.

        Returns
        -------
        tuple[Value, tuple[Value, ...]]
            Projections for its result and state.
        """
        value = Var(local=local)
        return (
            Projection(value=value, position=0, result_type=result),
            tuple(
                Projection(value=value, position=position, result_type=type_)
                for position, type_ in enumerate(self.state_types, start=1)
            ),
        )

    def allocate(
        self, role: str, source: ComputationId | OperationId, result: TypeExpr
    ) -> _Companion:
        """Allocate a deterministic computation disjoint from existing declarations.

        Parameters
        ----------
        role : str
            Whether this companion is for an operation, call or return.
        source : ComputationId or OperationId
            The originating nominal declaration.
        result : TypeExpr
            The result type of its companion.

        Returns
        -------
        _Companion
            A reserved identity and signature.
        """
        ordinal = 0
        while True:
            identity = ComputationId.derive(
                "linear-state-v1", self.owner.id, self.handler.id, role, source, ordinal
            )
            name = f"__state_{identity.digest}"
            if identity not in self.identities and name not in self.names:
                self.identities.add(identity)
                self.names.add(name)
                return _Companion(identity, name, result)
            ordinal += 1

    def origin(self, role: str, key: str) -> SourceOrigin:
        """Attribute a generated definition to its original installation.

        Parameters
        ----------
        role : str
            The role of the generated source position.
        key : str
            The structural discriminator within this installation.

        Returns
        -------
        SourceOrigin
            A stable generated source location.
        """
        origin = self.owner.origin
        return origin.with_(
            structural_path=(
                *origin.structural_path,
                "state-specialization",
                role,
                key,
            ),
            role=role,
        )

    def pure(self, term: Computation) -> Computation:
        """Copy an ordinary prefix while dropping statically dead arms.

        Parameters
        ----------
        term : Computation
            The already recognized ordinary computation.

        Returns
        -------
        Computation
            The remaining ordinary code.
        """
        if isinstance(term, If):
            branches = _branches(term)
            if len(branches) == 1:
                return self.pure(branches[0])
            self.spend(1)
            return term.with_(
                then=self.pure(term.then), otherwise=self.pure(term.otherwise)
            )
        self.spend(1)
        if isinstance(term, Bind):
            return term.with_(first=self.pure(term.first), then=self.pure(term.then))
        return term

    def transition(self, term: Computation, result: TypeExpr) -> Computation:
        """Replace a clause's terminal resume by its reply and simultaneous update.

        Parameters
        ----------
        term : Computation
            A recognized tail-position clause term.
        result : TypeExpr
            The product of the reply and new state.

        Returns
        -------
        Computation
            The ordinary operation transition.

        Raises
        ------
        StateSpecializationUnsupported
            If a terminal action is not a resumption.
        """
        if isinstance(term, If):
            branches = _branches(term)
            if len(branches) == 1:
                return self.transition(branches[0], result)
            self.spend(1)
            return term.with_(
                then=self.transition(term.then, result),
                otherwise=self.transition(term.otherwise, result),
            )
        self.spend(1)
        if isinstance(term, Bind):
            return term.with_(
                first=self.pure(term.first), then=self.transition(term.then, result)
            )
        if isinstance(term, Resume):
            return Return(
                value=TupleValue(
                    items=(term.value, *term.arguments), result_type=result
                )
            )
        raise StateSpecializationUnsupported(
            "an operation clause does not resume in tail position"
        )

    def declare_transitions(self) -> None:
        """Check and prepare one ordinary transition per covered operation.

        Raises
        ------
        StateSpecializationUnsupported
            If a return or operation clause leaves the linear state fragment.
        """
        if self.handler.return_clause is not None:
            obstacle = self.ordinary.ordinary(self.handler.return_clause.body)
            if obstacle is not None:
                raise StateSpecializationUnsupported(f"return action: {obstacle}")
        for clause in self.handler.clauses:
            operation = self.effect.operation(clause.operation)
            if operation.telescope or clause.body is None:
                raise StateSpecializationUnsupported(
                    f"operation {operation.name!r} requires an authored monomorphic clause"
                )
            if clause.grade is not ResumptionGrade.LINEAR:
                raise StateSpecializationUnsupported(
                    f"operation {operation.name!r} must have linear grade"
                )
            actions = self.ordinary.completions(clause.body)
            if actions != frozenset(("resume",)):
                raise StateSpecializationUnsupported(
                    f"operation {operation.name!r} needs a tail resumption on every path"
                )
            self.operations[operation.id] = _Transition(
                (*clause.parameters, *self.handler.parameters),
                clause.body,
                self.pair(operation.result_type),
            )

    def bind_arguments(
        self,
        parameters: tuple[Local, ...],
        arguments: tuple[Value, ...],
        body: Computation,
    ) -> Computation:
        """Evaluate all arguments once in order, then execute the body.

        Parameters
        ----------
        parameters : tuple[Local, ...]
            Fresh binders already used to name the body's arguments.
        arguments : tuple[Value, ...]
            The caller's value expressions.
        body : Computation
            The computation in scope of every parameter.

        Returns
        -------
        Computation
            Strict argument bindings enclosing the body.
        """
        self.spend(2 * len(parameters))
        for parameter, argument in reversed(
            tuple(zip(parameters, arguments, strict=True))
        ):
            body = Bind(binder=parameter, first=Return(value=argument), then=body)
        return body

    def inline(
        self,
        body: Computation,
        parameters: tuple[Local, ...],
        arguments: tuple[Value, ...],
        names: _LocalNames,
    ) -> Computation:
        """Move a closed clause into scope without allocating a runtime call.

        Parameters
        ----------
        body : Computation
            Code whose free locals are among its parameters.
        parameters : tuple[Local, ...]
            The source lexical binders for inputs and handler state.
        arguments : tuple[Value, ...]
            The actual arguments in source evaluation order.
        names : _LocalNames
            The destination's fresh-name supply.

        Returns
        -------
        Computation
            Code with local names transported to the destination scope.
        """
        renamed: dict[str, str] = {}
        binders = cast(tuple[Local, ...], _relocate(parameters, names, renamed))
        copied = cast(Computation, _relocate(body, names, renamed))
        return self.bind_arguments(binders, arguments, copied)

    def thread(
        self,
        term: Computation,
        state: tuple[Value, ...],
        result: TypeExpr,
        names: _LocalNames,
    ) -> Computation:
        """Evaluate the source once, carrying every new state into its continuation.

        Parameters
        ----------
        term : Computation
            The source computation to transform.
        state : tuple[Value, ...]
            Pure reads of the current state coordinates.
        result : TypeExpr
            The result type of the source computation.
        names : _LocalNames
            The fresh-name supply for its lexical declaration.

        Returns
        -------
        Computation
            Code that returns the result and the outgoing state.

        Raises
        ------
        StateSpecializationUnsupported
            If the term uses unsupported control or a different instance.
        """
        if isinstance(term, Return):
            self.spend(1)
            return Return(
                value=TupleValue(
                    items=(term.value, *state), result_type=self.pair(result)
                )
            )
        if isinstance(term, Bind):
            self.spend(3)
            pair = names.fresh(self.pair(term.binder.type))
            value, updated = self.split(pair, term.binder.type)
            return Bind(
                binder=pair,
                first=self.thread(term.first, state, term.binder.type, names),
                then=Bind(
                    binder=term.binder,
                    first=Return(value=value),
                    then=self.thread(term.then, updated, result, names),
                ),
            )
        if isinstance(term, If):
            self.spend(1)
            return term.with_(
                then=self.thread(term.then, state, result, names),
                otherwise=self.thread(term.otherwise, state, result, names),
            )
        if isinstance(term, Case):
            if term.motive.indices or any(
                branch.static_arguments
                or branch.constructor not in self.case_constructors
                for branch in term.branches
            ):
                raise StateSpecializationUnsupported(
                    "a state case needs a monomorphic family without indices "
                    "or static pattern arguments"
                )
            self.spend(1)
            return term.with_(
                motive=term.motive.with_(result_type=self.pair(result)),
                branches=tuple(
                    branch.with_(body=self.thread(branch.body, state, result, names))
                    for branch in term.branches
                ),
            )
        if isinstance(term, Perform):
            request = term.request
            if (
                request.instance != self.installation.instance
                or request.static_arguments
            ):
                raise StateSpecializationUnsupported(
                    "a request reaches a different instance or requires static specialization"
                )
            clause = self.operations[request.operation]
            return self.inline(
                self.transition(clause.body, clause.result),
                clause.parameters,
                (*request.arguments, *state),
                names,
            )
        if isinstance(term, Call):
            return self.thread_call(term, state, names)
        raise StateSpecializationUnsupported(
            f"{type(term).__name__} is outside the closed state computation"
        )

    def thread_call(
        self, term: Call, state: tuple[Value, ...], names: _LocalNames
    ) -> Computation:
        """Retain an ordinary call or use the unique companion for a stateful call.

        Parameters
        ----------
        term : Call
            The source call to transform.
        state : tuple[Value, ...]
            The current state coordinates.
        names : _LocalNames
            The lexical name supply for its result binder.

        Returns
        -------
        Computation
            Code that returns the call's result and outgoing state.

        Raises
        ------
        StateSpecializationUnsupported
            If the target needs static specialization, residual effects or general control.
        """
        target = self.computations[term.callee]
        if target.telescope:
            raise StateSpecializationUnsupported(
                f"call {target.name!r} needs static specialization"
            )
        if term.effects == EMPTY_ROW:
            obstacle = self.ordinary.call(target)
            if obstacle is not None:
                raise StateSpecializationUnsupported(obstacle)
            self.spend(3)
            value = names.fresh(term.result_type)
            return Bind(
                binder=value,
                first=term,
                then=Return(
                    value=TupleValue(
                        items=(Var(local=value), *state),
                        result_type=self.pair(term.result_type),
                    )
                ),
            )
        if term.effects != self.row:
            raise StateSpecializationUnsupported(
                f"call {target.name!r} has a residual effect row"
            )
        self.spend(1)
        companion = self.companions.get(target.id)
        if companion is None:
            companion = self.allocate(
                "computation", target.id, self.pair(target.type.result)
            )
            self.companions[target.id] = companion
            self.pending.append(target)
        return companion.call((*term.arguments, *state), term.origin)

    def body(self) -> Computation:
        """Initialize state once, run the installation and apply its return once.

        Returns
        -------
        Computation
            The replacement body of the named installation.
        """
        names = _LocalNames(self.owner)
        parameters = tuple(names.fresh(type_) for type_ in self.state_types)
        result = self.handler.input_type
        root = self.thread(
            self.installation.computation,
            tuple(Var(local=parameter) for parameter in parameters),
            result,
            names,
        )
        self.spend(1)
        pair = names.fresh(self.pair(result))
        value, state = self.split(pair, result)
        final = self.handler.return_clause
        if final is None:
            self.spend(1)
            finish: Computation = Return(value=value)
        else:
            finish = self.inline(
                self.pure(final.body),
                (final.binder, *self.handler.parameters),
                (value, *state),
                names,
            )
        root = Bind(binder=pair, first=root, then=finish)
        if _return_observes_address(self.handler):
            self.spend(2)
            root = Bind(
                binder=names.fresh(_RETURN_ADDRESS),
                first=HandlerReturnAddress(handler=self.handler.id),
                then=root,
            )
        return self.bind_arguments(parameters, self.installation.arguments, root)

    def build(self) -> QiecModule:
        """Finish each reachable recursive companion once and recheck the result.

        Returns
        -------
        QiecModule
            The original declarations, with the selected body transformed, followed by the companions.
        """
        self.declare_transitions()
        rewritten = self.owner.with_(body=self.body())
        while self.pending:
            source = self.pending.popleft()
            names = _LocalNames(source)
            parameters = tuple(names.fresh(type_) for type_ in self.state_types)
            body = self.thread(
                source.body,
                tuple(Var(local=parameter) for parameter in parameters),
                source.type.result,
                names,
            )
            self.generated.append(
                self.companions[source.id].declaration(
                    (*source.parameters, *parameters),
                    body,
                    self.origin("computation", str(source.id)),
                )
            )
        computations = tuple(
            rewritten if item.id == self.owner.id else item
            for item in self.module.computations
        )
        output = self.module.with_(computations=(*computations, *self.generated))
        validate_module(output)
        return output


def specialize_recursive_state(
    module: QiecModule, computation: ComputationId, *, max_terms: int = 2048
) -> QiecModule:
    """Eliminate one closed tail-linear state installation from a named body.

    The selected computation must install a total authored handler on one
    module-scoped nominal instance. Its clauses must be ordinary prefixes
    followed by one tail resumption, with no introduced effects. Reached
    stateful recursive calls use pure companions that return their result
    and updated state. The selected boundary alone applies the authored
    return action. A case in a stateful body can select an arm of a closed
    monomorphic family without indices or static pattern arguments.
    Inlining the pure clauses preserves the number and order
    of calls that allocate runtime addresses. Evaluation agrees on returned
    values under the checked contracts. Handler and request diagnostic
    events and evaluator fuel counts can differ.

    Parameters
    ----------
    module : QiecModule
        The checked source module.
    computation : ComputationId
        The named installation to replace. Its signature and identity stay
        the same, and other declarations are available for their original callers.
    max_terms : int
        Positive bound on the number of constructed computation terms,
        counting sequencings, branches, calls and returns. Source validation
        and runtime recursion have their own costs.

    Returns
    -------
    QiecModule
        The checked module with this installation replaced and its companions added.

    Raises
    ------
    StateSpecializationUnsupported
        If the installation is outside the closed state fragment.
    StateSpecializationLimit
        If construction exceeds the bound or the source is too deeply nested.
    ValueError
        If the computation is absent or the requested bound is invalid.
    """
    if type(max_terms) is not int or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    registry = validate_module(module)
    owner = next((item for item in module.computations if item.id == computation), None)
    if owner is None:
        raise ValueError(
            f"unknown computation for state specialization `{computation}`"
        )
    term = owner.body
    if (
        not isinstance(term, Handle)
        or owner.telescope
        or owner.type.effects != EMPTY_ROW
    ):
        raise StateSpecializationUnsupported(
            "the named body must be a closed monomorphic handler installation"
        )
    handler = registry.handlers[term.handler]
    if (obstacle := _handler_obstacle(handler)) is not None:
        raise StateSpecializationUnsupported(obstacle)
    effect = registry.effects[handler.effect.id]
    if effect.telescope or term.static_arguments or not handler.parameters:
        raise StateSpecializationUnsupported(
            "the installation needs nonempty monomorphic state"
        )
    instance = next(
        (item for item in module.instances if item.entry.instance == term.instance),
        None,
    )
    if instance is None:
        raise StateSpecializationUnsupported(
            "the handled instance must be module-scoped"
        )
    row = EffectRow(entries=(instance.entry,))
    try:
        return _Specialization(
            module, owner, term, handler, effect, row, max_terms
        ).build()
    except RecursionError as error:
        raise StateSpecializationLimit(
            "state specialization exceeds its structural nesting limit"
        ) from error


__all__ = [
    "StateSpecializationLimit",
    "StateSpecializationUnsupported",
    "specialize_recursive_state",
]
