"""Elaborate checked lexical forward differentiation to authored QIEC handlers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from quivers.qiec.checking import KernelError, KernelRegistry
from quivers.qiec.effects import (
    EMPTY_ROW,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    RowEntry,
)
from quivers.qiec.identifiers import (
    ConstructorId,
    EffectInstanceId,
    HandlerId,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.primitives import PRIMITIVES
from quivers.qiec.terms import (
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    ConstructorValue,
    Handle,
    If,
    LiteralValue,
    Local,
    Perform,
    PrimitiveApplication,
    Resume,
    Return,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, REAL, TypeApplication, TypeExpr

_MAX_NESTING = 2
_MAX_SOURCE_NODES = 16


@dataclass(frozen=True, slots=True)
class ForwardDiffInstance:
    """A named perturbation with a distinct two-field number type.

    Parameters
    ----------
    instance : EffectInstanceId
        The existing lexical instance implementing the forward operations.
    handler : HandlerId
        Its total linear handler signature, to elaborate to authored clauses.
    constructor : ConstructorId
        The one constructor of this instance's nominal number type. Its
        coefficient type is Real or the enclosing instance's number type.
    """

    instance: EffectInstanceId
    handler: HandlerId
    constructor: ConstructorId


@dataclass(frozen=True, slots=True)
class _Scope:
    """The declarations of a checked forward instance.

    Parameters
    ----------
    entry : RowEntry
        The actual nominal effect row entry.
    handler : HandlerDef
        Its ordinary handler signature.
    constructor : ConstructorId
        The number carrier's constructor.
    number : TypeExpr
        The nominal type of numbers at this instance.
    coefficient : TypeExpr
        The ground or parent coefficient type.
    operations : dict[str, OperationDef]
        The resolved seed, arithmetic and stopping operations.
    """

    entry: RowEntry
    handler: HandlerDef
    constructor: ConstructorId
    number: TypeExpr
    coefficient: TypeExpr
    operations: dict[str, OperationDef]


def _scope(
    module: QiecModule, registry: KernelRegistry, declaration: ForwardDiffInstance
) -> _Scope:
    """Resolve the actual carrier, effect signatures and handler.

    Parameters
    ----------
    module : QiecModule
        The source module with the ordinary named instance.
    registry : KernelRegistry
        Its validated declarations.
    declaration : ForwardDiffInstance
        The requested lexical perturbation.

    Returns
    -------
    _Scope
        Its checked nominal declarations and field types.

    Raises
    ------
    KernelError
        If the selected carrier, effect or handler fails its contract.
    """
    entry = next(
        (
            item.entry
            for item in module.instances
            if item.entry.instance == declaration.instance
        ),
        None,
    )
    if entry is None:
        raise KernelError("forward differentiation needs a declared named instance")
    constructor = registry.constructor(declaration.constructor)
    family = registry.families[constructor.family]
    if (
        not family.closed
        or family.parameters
        or family.indices
        or family.constructors != (constructor.id,)
        or constructor.telescope
        or len(constructor.fields) != 2
        or constructor.fields[0].type != constructor.fields[1].type
    ):
        raise KernelError("a forward number needs a nominal two-coefficient data type")
    number = TypeApplication(constructor=family.type_constructor, arguments=())
    coefficient = constructor.fields[0].type
    interface = registry.effects[entry.effect.id]
    operations = {op.name: op for op in interface.operations}
    expected = {
        "seed": ((coefficient, coefficient), number),
        "add": ((number, number), number),
        "mul": ((number, number), number),
        "stop_primal": ((number,), coefficient),
        "stop_tangent": ((number,), coefficient),
    }
    if "less" in operations:
        expected["less"] = ((number, number), BOOL)
    actual = {
        op.name: (tuple(arg.type for arg in op.arguments), op.result_type)
        for op in interface.operations
    }
    if (
        interface.telescope
        or any(op.telescope for op in operations.values())
        or actual != expected
    ):
        raise KernelError(
            "forward operations do not match their instance's coefficient types"
        )
    handler = registry.handler(declaration.handler)
    if (
        handler.effect != entry.effect
        or handler.implementation != "foreign"
        or handler.foreign is not None
        or not handler.total
        or handler.telescope
        or handler.parameters
        or handler.return_clause is not None
        or handler.input_type != handler.output_type
        or any(clause.grade is not ResumptionGrade.LINEAR for clause in handler.clauses)
    ):
        raise KernelError(
            "forward differentiation needs its total linear handler signature"
        )
    return _Scope(entry, handler, constructor.id, number, coefficient, operations)


def _check_source(
    entry: NamedComputation, scopes: dict[EffectInstanceId, _Scope]
) -> None:
    """Check scope order and the only ways an authored term can use a carrier.

    Parameters
    ----------
    entry : NamedComputation
        The single closed Real entry.
    scopes : dict[EffectInstanceId, _Scope]
        The actual declared, distinct perturbations.

    Raises
    ------
    KernelError
        If a carrier is constructed, inspected or exported directly, or
        a named request or handler lies outside its checked lexical scope.
    """
    if (
        entry.telescope
        or entry.type.effects != EMPTY_ROW
        or entry.type.result != REAL
        or any(parameter.type != REAL for parameter in entry.parameters)
    ):
        raise KernelError("forward differentiation needs a closed scalar Real entry")
    installed: set[EffectInstanceId] = set()
    pending: list[tuple[Computation, tuple[_Scope, ...], bool]] = [
        (entry.body, (), False)
    ]
    nodes = 0
    while pending:
        term, active, comparison_guard = pending.pop()
        if isinstance(term, Bind | Handle):
            nodes += 1
            if isinstance(term, Bind) and isinstance(term.then, If):
                nodes += 1
            if nodes > _MAX_SOURCE_NODES:
                raise KernelError(
                    f"forward source exceeds {_MAX_SOURCE_NODES} binds and handlers or branches",
                    "qiec-forward-limit",
                )
        if isinstance(term, Bind):
            if isinstance(term.then, If):
                branch = term.then
                if (
                    not isinstance(term.first, Perform)
                    or term.binder.type != BOOL
                    or branch.condition != Var(local=term.binder)
                ):
                    raise KernelError(
                        "a forward branch must immediately use its named comparison",
                        "qiec-forward-comparison-use",
                    )
                pending.extend(
                    (
                        (branch.otherwise, active, False),
                        (branch.then, active, False),
                        (term.first, active, True),
                    )
                )
            else:
                pending.extend(
                    ((term.then, active, False), (term.first, active, False))
                )
            continue
        if isinstance(term, Handle):
            if len(active) == _MAX_NESTING:
                raise KernelError(
                    f"forward source exceeds {_MAX_NESTING} nested instances",
                    "qiec-forward-limit",
                )
            scope = scopes.get(term.instance)
            if (
                scope is None
                or term.instance in installed
                or term.handler != scope.handler.id
                or term.arguments
                or term.static_arguments
            ):
                raise KernelError(
                    "a forward scope needs one matching lexical installation"
                )
            parent = active[-1] if active else None
            coefficient = parent.number if parent is not None else REAL
            introduced = (
                EffectRow(entries=(parent.entry,)) if parent is not None else EMPTY_ROW
            )
            if scope.coefficient != coefficient or scope.handler.introduced not in (
                EMPTY_ROW,
                introduced,
            ):
                raise KernelError(
                    "a nested forward scope needs the named parent coefficient algebra"
                )
            if scope.handler.output_type not in (
                REAL,
                *(item.number for item in active),
            ):
                raise KernelError(
                    "a forward number must stop before leaving its instance"
                )
            installed.add(term.instance)
            pending.append((term.computation, (*active, scope), False))
            continue
        if isinstance(term, Perform):
            request = term.request
            scope = next(
                (
                    candidate
                    for candidate in active
                    if candidate.entry.instance == request.instance
                ),
                None,
            )
            if scope is None or request.operation not in {
                op.id for op in scope.operations.values()
            }:
                raise KernelError(
                    "a forward request must use its active named instance"
                )
            less = scope.operations.get("less")
            is_comparison = less is not None and request.operation == less.id
            if comparison_guard != is_comparison:
                raise KernelError(
                    "a forward comparison must immediately guard a branch",
                    "qiec-forward-comparison-use",
                )
            values = request.arguments
        elif isinstance(term, Return):
            values = (term.value,)
        elif isinstance(term, If):
            raise KernelError(
                "a forward branch must immediately use its named comparison",
                "qiec-forward-comparison-use",
            )
        else:
            raise KernelError(
                f"{type(term).__name__} is outside the finite forward scope"
            )
        for value in values:
            if isinstance(value, Var):
                continue
            if (
                isinstance(value, LiteralValue)
                and value.type == REAL
                and isinstance(value.value, (float, int))
                and not isinstance(value.value, bool)
                and (isinstance(value.value, int) or isfinite(value.value))
            ):
                continue
            raise KernelError(
                "forward numbers must be constructed and read by named operations"
            )
    if installed != scopes.keys():
        raise KernelError(
            "each declared forward instance must be installed once in the source"
        )


class _ForwardOperation:
    """Generate an authored clause from the checked coefficient types.

    Parameters
    ----------
    module : QiecModule
        The module owning the generated clause origins.
    scope : _Scope
        The operation's actual nominal carrier.
    carriers : dict[TypeExpr, _Scope]
        The checked representations of the possible parent carriers.
    operation : OperationDef
        The actual nominal operation the clause answers.
    """

    def __init__(
        self,
        module: QiecModule,
        scope: _Scope,
        carriers: dict[TypeExpr, _Scope],
        operation: OperationDef,
    ) -> None:
        self.module = module
        self.scope = scope
        self.carriers = carriers
        self.operation = operation
        self.matches: list[tuple[Value, _Scope, tuple[Local, Local]]] = []

    def origin(self, role: str, *suffix: str | int) -> SourceOrigin:
        """Place generated arithmetic under the actual handler and operation.

        Parameters
        ----------
        role : str
            The generated term's source role.
        suffix : str or int
            The remaining structural path.

        Returns
        -------
        SourceOrigin
            The generated occurrence.
        """
        return SourceOrigin(
            module=self.module.module,
            source_protocol=self.module.source_protocol,
            structural_path=(
                "forward",
                str(self.scope.handler.id),
                self.operation.name,
                *suffix,
            ),
            role=role,
        )

    def open_pair(self, value: Value, scope: _Scope) -> tuple[Var, Var]:
        """Read the two coefficients of one trusted nominal number.

        Parameters
        ----------
        value : Value
            The number to open.
        scope : _Scope
            Its exact nominal carrier.

        Returns
        -------
        tuple[Var, Var]
            The primal and tangent bound in a pending ordinary case.
        """
        position = len(self.matches)
        x = Local(name=f"primal_{position}", type=scope.coefficient)
        dx = Local(name=f"tangent_{position}", type=scope.coefficient)
        self.matches.append((value, scope, (x, dx)))
        return Var(local=x), Var(local=dx)

    def coefficients(self, value: Value, scope: _Scope | None) -> tuple[Value, ...]:
        """Open a checked carrier down to its ordinary Real coefficients.

        Parameters
        ----------
        value : Value
            The number to open, or an ordinary Real at the root algebra.
        scope : _Scope or None
            The precise carrier of that number.

        Returns
        -------
        tuple[Value, ...]
            The flattened primal coefficients followed by the tangents.
        """
        if scope is None:
            return (value,)
        x, dx = self.open_pair(value, scope)
        parent = self.carriers.get(scope.coefficient)
        return (*self.coefficients(x, parent), *self.coefficients(dx, parent))

    def pair(self, scope: _Scope, x: Value, dx: Value) -> ConstructorValue:
        """Construct one pair of the given nominal carrier.

        Parameters
        ----------
        scope : _Scope
            The nominal carrier served by the constructor.
        x, dx : Value
            The two values in its coefficient type.

        Returns
        -------
        ConstructorValue
            The ordinary data value.
        """
        return ConstructorValue(
            constructor=scope.constructor,
            static_arguments=(),
            fields=(x, dx),
            result_type=scope.number,
        )

    def pack(self, values: tuple[Value, ...], scope: _Scope | None) -> Value:
        """Build the original nested carrier from its Real coefficients.

        Parameters
        ----------
        values : tuple[Value, ...]
            The flat coefficient vector in primal and tangent order.
        scope : _Scope or None
            The nominal result carrier, or the ground Real level.

        Returns
        -------
        Value
            The nested ordinary constructor value.
        """
        if scope is None:
            (value,) = values
            return value
        half = len(values) // 2
        parent = self.carriers.get(scope.coefficient)
        return self.pair(
            scope, self.pack(values[:half], parent), self.pack(values[half:], parent)
        )

    def primitive(self, name: str, x: Value, y: Value) -> PrimitiveApplication:
        """Use the ordinary ground Real operation.

        Parameters
        ----------
        name : str
            The checked scalar arithmetic or comparison operation.
        x, y : Value
            The underlying Real coefficients.

        Returns
        -------
        PrimitiveApplication
            The pure ground operation with its checked result type.
        """
        primitive = PRIMITIVES[f"{name}_real"]
        return PrimitiveApplication(
            primitive=primitive.id,
            name=primitive.name,
            arguments=(x, y),
            origin=self.origin(name, "coefficient"),
            result_type=primitive.result,
        )

    def ground_primal(self, value: Value) -> Value:
        """Read the zeroth coefficient through all checked nominal pairs.

        Parameters
        ----------
        value : Value
            A value belonging to this operation's nominal carrier.

        Returns
        -------
        Value
            Its Real primal, independent of every perturbation coefficient.
        """
        scope: _Scope | None = self.scope
        while scope is not None:
            value, _ = self.open_pair(value, scope)
            scope = self.carriers.get(scope.coefficient)
        return value

    def add(self, x: tuple[Value, ...], y: tuple[Value, ...]) -> tuple[Value, ...]:
        """Add two vectors of coefficients.

        Parameters
        ----------
        x, y : tuple[Value, ...]
            The coefficients in the same carrier order.

        Returns
        -------
        tuple[Value, ...]
            The coefficientwise sums.
        """
        return tuple(self.primitive("add", a, b) for a, b in zip(x, y, strict=True))

    def multiply(self, x: tuple[Value, ...], y: tuple[Value, ...]) -> tuple[Value, ...]:
        """Specialize the multiplication equations down to ordinary Real.

        Parameters
        ----------
        x, y : tuple[Value, ...]
            The flattened coefficients of the same nested carrier.

        Returns
        -------
        tuple[Value, ...]
            The product coefficients, in their original nesting order.
        """
        if len(x) == 1:
            return (self.primitive("mul", x[0], y[0]),)
        half = len(x) // 2
        primal = self.multiply(x[:half], y[:half])
        left = self.multiply(x[half:], y[:half])
        right = self.multiply(x[:half], y[half:])
        return (*primal, *self.add(left, right))

    def clause(self) -> HandlerClauseDef:
        """Generate the pure prefix and single tail resumption of a clause.

        Returns
        -------
        HandlerClauseDef
            The ordinary authored operation implementation.
        """
        name = self.operation.name
        parameters = tuple(
            Local(name=f"argument_{i}", type=arg.type)
            for i, arg in enumerate(self.operation.arguments)
        )
        operands = tuple(Var(local=parameter) for parameter in parameters)
        if name == "seed":
            x, dx = operands
            result = self.pair(self.scope, x, dx)
        elif name in ("stop_primal", "stop_tangent"):
            x, dx = self.open_pair(operands[0], self.scope)
            result = x if name == "stop_primal" else dx
        elif name == "less":
            result = self.primitive(
                "strict_lt",
                self.ground_primal(operands[0]),
                self.ground_primal(operands[1]),
            )
        else:
            x_values = self.coefficients(operands[0], self.scope)
            y_values = self.coefficients(operands[1], self.scope)
            values = (
                self.add(x_values, y_values)
                if name == "add"
                else self.multiply(x_values, y_values)
            )
            result = self.pack(values, self.scope)
        body: Computation = Resume(value=result, origin=self.origin("resume"))
        for index, (source, scope, fields) in reversed(tuple(enumerate(self.matches))):
            body = Case(
                scrutinee=source,
                motive=CaseMotive(
                    indices=(), result_type=self.scope.handler.output_type
                ),
                branches=(
                    CaseBranch(
                        constructor=scope.constructor,
                        static_arguments=(),
                        fields=fields,
                        body=body,
                        scope=StaticScopeId.derive(
                            str(self.origin("carrier", index).site_id())
                        ),
                    ),
                ),
            )
        return HandlerClauseDef(
            operation=self.operation.id,
            grade=ResumptionGrade.LINEAR,
            parameters=parameters,
            body=body,
        )


def elaborate_forward_diff(
    module: QiecModule, instances: tuple[ForwardDiffInstance, ...]
) -> QiecModule:
    """Generate ordinary authored forward handlers for checked nominal scopes.

    Each instance's seed embeds a coefficient and a tangent in its own
    number type. Addition and multiplication use the dual-number rules.
    Coefficient arithmetic specializes to pure Real operations in each
    generated clause. The portable fragment admits two simultaneously
    active instances and sixteen source binds, handlers and branches in
    total. An optional named `less` compares only the two ground primals
    and immediately guards a lazy branch. Equality and nonfinite primals
    refuse the comparison at runtime.
    The two explicit stop operations project the primal or tangent into
    the coefficient algebra, retaining the parent perturbation when
    nested. Those original requests remain visible in ordinary traces.

    Parameters
    ----------
    module : QiecModule
        The single closed Real computation, its ordinary data and effect
        declarations, and total linear forward handler signatures.
    instances : tuple[ForwardDiffInstance, ...]
        The lexical perturbations, with distinct carrier and handler identities.

    Returns
    -------
    QiecModule
        The rechecked ordinary module with the selected authored handlers.

    Raises
    ------
    KernelError
        If the module, named operations, carrier abstraction or lexical
        nesting leaves the checked finite forward fragment. The code
        `qiec-forward-limit` identifies an exceeded size or nesting bound.
    """
    if len(module.computations) != 1 or not instances:
        raise KernelError(
            "forward differentiation needs one entry and at least one instance"
        )
    registry = validate_module(module)
    resolved = tuple(_scope(module, registry, item) for item in instances)
    by_instance = {item.entry.instance: item for item in resolved}
    by_number = {item.number: item for item in resolved}
    by_handler = {item.handler.id: item for item in resolved}
    if any(
        len(items) != len(resolved) for items in (by_instance, by_number, by_handler)
    ):
        raise KernelError(
            "forward instances need distinct nominal carriers and handlers"
        )
    _check_source(module.computations[0], by_instance)
    generated = {
        item.handler.id: item.handler.with_(
            implementation="authored",
            clauses=tuple(
                _ForwardOperation(module, item, by_number, operation).clause()
                for operation in item.operations.values()
            ),
        )
        for item in resolved
    }
    lowered = module.with_(
        handlers=tuple(
            generated.get(handler.id, handler) for handler in module.handlers
        )
    )
    validate_module(lowered)
    return lowered


__all__ = ["ForwardDiffInstance", "elaborate_forward_diff"]
