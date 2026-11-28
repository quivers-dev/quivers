"""Elaborate or compile a nominal reverse graph with immutable cotangents."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from math import isfinite
from typing import Literal

from quivers.qiec.checking import KernelError
from quivers.qiec.effects import (
    EMPTY_ROW,
    HandlerClauseDef,
    HandlerDef,
    HandlerReturnClauseDef,
    OperationDef,
    ResumptionGrade,
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
    ConditionalValue,
    ConstructorValue,
    Handle,
    If,
    LetBinding,
    LetValue,
    LiteralValue,
    Local,
    Perform,
    PrimitiveApplication,
    Projection,
    Resume,
    Return,
    TupleValue,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, INT, REAL, TypeApplication, TypeExpr, product_type

_MAX_VERTICES = 6
_MAX_COMPILED_VERTICES = 256
_MAX_SINK_DEPTH = 15
_MAX_SINK_COMPARISONS = 256
_ZERO = LiteralValue(value=0.0, type=REAL)
_ONE = LiteralValue(value=1.0, type=REAL)
_SINK_CHOICE = product_type(INT, REAL)


@dataclass(frozen=True)
class _SeedVertex:
    """One checked input or literal and the source request that binds it.

    Parameters
    ----------
    value : Value
        The checked parameter reference or Real literal.
    origin : SourceOrigin
        The source seed request's origin.
    """

    value: Value
    origin: SourceOrigin


@dataclass(frozen=True)
class _BinaryVertex:
    """The ordered operand occurrences of an arithmetic vertex.

    Parameters
    ----------
    operation : {"add", "mul"}
        The arithmetic used at this vertex.
    left, right : int
        The two operand slots in source order, possibly equal.
    origin : SourceOrigin
        The source arithmetic request's origin.
    """

    operation: Literal["add", "mul"]
    left: int
    right: int
    origin: SourceOrigin


type _ReverseVertex = _SeedVertex | _BinaryVertex


@dataclass(frozen=True)
class _SinkGuard:
    """A strict comparison and its two lazy choices of output slot."""

    left: int
    right: int
    then: _ReverseSink
    otherwise: _ReverseSink
    origin: SourceOrigin


type _ReverseSink = int | _SinkGuard


def _source_sink(
    term: Computation,
    instance: EffectInstanceId,
    operations: dict[str, OperationDef],
    positions: dict[Local, int],
) -> _ReverseSink:
    """Check a lazy decision that only compares or selects existing vertices."""
    less = operations.get("less")
    comparisons = 0

    def visit(current: Computation, depth: int) -> _ReverseSink:
        nonlocal comparisons
        if isinstance(current, Perform):
            request = current.request
            if (
                request.instance == instance
                and request.operation == operations["stop"].id
                and isinstance(chosen := request.arguments[0], Var)
                and chosen.local in positions
            ):
                return positions[chosen.local]
            raise KernelError(
                "reverse source must stop on a vertex of its own instance"
            )
        if (
            less is None
            or not isinstance(current, Bind)
            or not isinstance(current.first, Perform)
            or current.first.request.instance != instance
            or current.first.request.operation != less.id
            or not isinstance(branch := current.then, If)
            or branch.condition != Var(local=current.binder)
        ):
            raise KernelError(
                "reverse sink selection must branch immediately on a comparison "
                "of its own instance or stop on one of its vertices"
            )
        if depth == _MAX_SINK_DEPTH or comparisons == _MAX_SINK_COMPARISONS:
            raise KernelError(
                f"reverse sink selection exceeds {_MAX_SINK_DEPTH} nested or "
                f"{_MAX_SINK_COMPARISONS} total comparisons",
                "qiec-reverse-sink-limit",
            )
        left, right = current.first.request.arguments
        if (
            not isinstance(left, Var)
            or left.local not in positions
            or not isinstance(right, Var)
            or right.local not in positions
        ):
            raise KernelError("reverse comparison requires vertices of its instance")
        comparisons += 1
        return _SinkGuard(
            positions[left.local],
            positions[right.local],
            visit(branch.then, depth + 1),
            visit(branch.otherwise, depth + 1),
            current.first.request.origin.origin,
        )

    return visit(term, 0)


def _source_vertices(
    entry: NamedComputation,
    instance: EffectInstanceId,
    handler: HandlerDef,
    operations: dict[str, OperationDef],
    *,
    maximum: int,
    limit_code: str,
) -> tuple[tuple[_ReverseVertex, ...], _ReverseSink]:
    """Check sequential allocation and collect the ordered operand slots.

    Parameters
    ----------
    entry : NamedComputation
        The source entry with its original requests.
    instance : EffectInstanceId
        The one lexical reverse instance.
    handler : HandlerDef
        Its linear handler signature.
    operations : dict[str, OperationDef]
        The checked seed, arithmetic and stop operations.
    maximum : int
        The largest graph supported by the selected elaboration.
    limit_code : str
        The diagnostic code for exceeding that maximum.

    Returns
    -------
    tuple
        The vertices in allocation order and the lazy output-slot selection.

    Raises
    ------
    KernelError
        If the source control, scope, inputs, size, allocation or stop fails.
    """
    term = entry.body
    if (
        entry.telescope
        or entry.type.effects != EMPTY_ROW
        or any(parameter.type != REAL for parameter in entry.parameters)
        or not isinstance(term, Handle)
        or term.instance != instance
        or term.handler != handler.id
        or term.arguments
        or term.static_arguments
    ):
        raise KernelError("reverse differentiation needs one closed named handler")
    term = term.computation
    positions: dict[Local, int] = {}
    vertices: list[_ReverseVertex] = []
    while isinstance(term, Bind):
        node = term.first
        if not isinstance(node, Perform) or node.request.instance != instance:
            raise KernelError("a reverse vertex must use the active named instance")
        request = node.request
        name = request.operation
        if (less := operations.get("less")) is not None and name == less.id:
            break
        if len(vertices) == maximum:
            raise KernelError(f"reverse source exceeds {maximum} vertices", limit_code)
        if name not in (
            operations["seed"].id,
            operations["add"].id,
            operations["mul"].id,
        ):
            raise KernelError("a reverse vertex must be seeded, added or multiplied")
        slot, *operands = request.arguments
        position = len(vertices)
        if (
            not isinstance(slot, LiteralValue)
            or slot.type != INT
            or type(slot.value) is not int
            or slot.value != position
        ):
            raise KernelError("reverse vertex slots must be fresh and consecutive")
        origin = request.origin.origin
        if position < len(entry.parameters):
            if name != operations["seed"].id or operands != [
                Var(local=entry.parameters[position])
            ]:
                raise KernelError(
                    "reverse source must seed each input once in parameter order"
                )
            vertices.append(_SeedVertex(operands[0], origin))
        elif name == operations["seed"].id:
            (constant,) = operands
            if (
                not isinstance(constant, LiteralValue)
                or constant.type != REAL
                or isinstance(constant.value, bool)
                or not isinstance(constant.value, (int, float))
                or (isinstance(constant.value, float) and not isfinite(constant.value))
            ):
                raise KernelError(
                    "a reverse seed after the inputs must be a Real constant"
                )
            vertices.append(_SeedVertex(constant, origin))
        else:
            left, right = operands
            if (
                not isinstance(left, Var)
                or left.local not in positions
                or not isinstance(right, Var)
                or right.local not in positions
            ):
                raise KernelError(
                    "reverse operands must be earlier vertices of the same instance"
                )
            vertices.append(
                _BinaryVertex(
                    "add" if name == operations["add"].id else "mul",
                    positions[left.local],
                    positions[right.local],
                    origin,
                )
            )
        positions[term.binder] = position
        term = term.then
    if len(vertices) < max(1, len(entry.parameters)):
        raise KernelError("reverse source must stop on a vertex of its own instance")
    return tuple(vertices), _source_sink(term, instance, operations, positions)


class _ReverseHandler:
    """Generate the authored handler for the checked, numbered source DAG.

    Parameters
    ----------
    module : QiecModule
        The module owning generated terms.
    handler : HandlerDef
        The checked nominal linear signature.
    constructor : ConstructorId
        The private carrier's primal and vertex slot constructor.
    vertices : int
        The exact finite size of the cotangent vector.
    """

    def __init__(
        self,
        module: QiecModule,
        handler: HandlerDef,
        constructor: ConstructorId,
        vertices: int,
    ) -> None:
        self.module = module
        self.handler = handler
        self.constructor = constructor
        self.vertices = vertices

    def origin(self, name: str, role: str) -> SourceOrigin:
        """Derive a generated term's stable identity.

        Parameters
        ----------
        name, role : str
            Its clause name and function in the generated body.

        Returns
        -------
        SourceOrigin
            The normal origin in the input module's protocol.
        """
        return SourceOrigin(
            module=self.module.module,
            source_protocol=self.module.source_protocol,
            structural_path=("reverse", str(self.handler.id), name, role),
            role=role,
        )

    def primitive(self, name: str, *arguments: Value) -> PrimitiveApplication:
        """Use one ordinary pure QIEC primitive.

        Parameters
        ----------
        name : str
            The nominal primitive's registry name.
        arguments : Value
            The primitive's actual input values.

        Returns
        -------
        PrimitiveApplication
            The checked pure term.
        """
        return _scalar_operation(name, self.origin(name, "coefficient"), *arguments)

    def at(self, slot: Value, position: int, coefficient: Value) -> Value:
        """Select a coefficient at exactly one statically admitted slot.

        Parameters
        ----------
        slot : Value
            The source vertex identity.
        position : int
            The candidate index in the finite vector.
        coefficient : Value
            The Real coefficient to keep if it matches.

        Returns
        -------
        Value
            The coefficient at that index, or zero.
        """
        index = LiteralValue(value=position, type=INT)
        test = self.primitive("eq_int", slot, index)
        return self.primitive("select_real", test, coefficient, _ZERO)

    def number(self, primal: Value, slot: Value) -> ConstructorValue:
        """Construct one carrier value with its retained primal and slot.

        Parameters
        ----------
        primal, slot : Value
            The scalar result and allocated vertex identity.

        Returns
        -------
        ConstructorValue
            The nominal number of the active reverse scope.
        """
        return ConstructorValue(
            constructor=self.constructor,
            static_arguments=(),
            fields=(primal, slot),
            result_type=self.handler.input_type,
        )

    def split(
        self, name: str, number: Local, fields: tuple[Local, Local], then: Computation
    ) -> Case:
        """Read the primal and slot inside the generated operation.

        Parameters
        ----------
        name : str
            The known operation or return clause.
        number : Local
            The nominal carrier being read.
        fields : tuple[Local, Local]
            The Real primal and Int slot binders.
        then : Computation
            The ordinary handler body in their scope.

        Returns
        -------
        Case
            The exhaustive match on the carrier's one constructor.
        """
        return Case(
            scrutinee=Var(local=number),
            motive=CaseMotive(indices=(), result_type=self.handler.output_type),
            branches=(
                CaseBranch(
                    constructor=self.constructor,
                    static_arguments=(),
                    fields=fields,
                    body=then,
                    scope=StaticScopeId.derive(
                        str(self.origin(name, number.name).site_id())
                    ),
                ),
            ),
        )

    def result(self, primal: Value, adjoints: tuple[Value, ...]) -> TupleValue:
        """Return a primal and the complete immutable cotangent vector.

        Parameters
        ----------
        primal : Value
            The selected output value.
        adjoints : tuple[Value, ...]
            The cotangents in source vertex order.

        Returns
        -------
        TupleValue
            The ordinary product of Real values.
        """
        return TupleValue(
            items=(primal, *adjoints), result_type=self.handler.output_type
        )

    def return_clause(self) -> HandlerReturnClauseDef:
        """Seed unit cotangent at the selected output when the body returns.

        Returns
        -------
        HandlerReturnClauseDef
            The authored initialization of the reverse sweep.
        """
        chosen = Local(name="chosen", type=self.handler.input_type)
        primal, slot = Local(name="primal", type=REAL), Local(name="slot", type=INT)
        adjoints = tuple(
            self.at(Var(local=slot), position, _ONE)
            for position in range(self.vertices)
        )
        terminal = Return(value=self.result(Var(local=primal), adjoints))
        return HandlerReturnClauseDef(
            binder=chosen, body=self.split("return", chosen, (primal, slot), terminal)
        )

    def update(
        self,
        name: str,
        previous: Local,
        output_slot: Var,
        left: tuple[Var, Var],
        right: tuple[Var, Var],
    ) -> Return:
        """Accumulate both edges in the answer of the already-resumed continuation.

        Parameters
        ----------
        name : {"add", "mul"}
            The vertex's own polynomial primitive.
        previous : Local
            The immutable answer returned by the continuation.
        output_slot : Var
            The index of the current vertex.
        left, right : tuple[Var, Var]
            The primal and slot of each incoming operand edge.

        Returns
        -------
        Return
            A fresh answer carrying the two post-resumption contributions.
        """
        prior = Var(local=previous)
        adjoints = tuple(
            Projection(value=prior, position=i + 1, result_type=REAL)
            for i in range(self.vertices)
        )
        output_cotangent: Value = _ZERO
        for i, cotangent in enumerate(adjoints):
            output_cotangent = self.primitive(
                "add_real", output_cotangent, self.at(output_slot, i, cotangent)
            )
        weight = Local(name="weight", type=REAL)
        left_weight = Local(name="left_weight", type=REAL)
        right_weight = Local(name="right_weight", type=REAL)
        weight_value = Var(local=weight)
        x, x_slot = left
        y, y_slot = right
        dx = (
            weight_value
            if name == "add"
            else self.primitive("mul_real", weight_value, y)
        )
        dy = (
            weight_value
            if name == "add"
            else self.primitive("mul_real", weight_value, x)
        )
        additions = tuple(
            self.primitive(
                "add_real",
                cotangent,
                self.primitive(
                    "add_real",
                    self.at(x_slot, i, Var(local=left_weight)),
                    self.at(y_slot, i, Var(local=right_weight)),
                ),
            )
            for i, cotangent in enumerate(adjoints)
        )
        primal = Projection(value=prior, position=0, result_type=REAL)
        return Return(
            value=LetValue(
                bindings=(
                    LetBinding(binder=weight, value=output_cotangent),
                    LetBinding(binder=left_weight, value=dx),
                    LetBinding(binder=right_weight, value=dy),
                ),
                body=self.result(primal, additions),
                result_type=self.handler.output_type,
            )
        )

    def clause(self, operation: OperationDef) -> HandlerClauseDef:
        """Generate a linear clause for a vertex, comparison or stop.

        Parameters
        ----------
        operation : OperationDef
            The actual registered nominal operation.

        Returns
        -------
        HandlerClauseDef
            An authored clause, with the reverse update after resumption.
        """
        name = operation.name
        parameters = tuple(
            Local(name=f"argument_{i}", type=argument.type)
            for i, argument in enumerate(operation.arguments)
        )
        resume = self.origin(name, "resume")
        if name == "seed":
            slot, primal = (Var(local=parameter) for parameter in parameters)
            body: Computation = Resume(value=self.number(primal, slot), origin=resume)
        elif name == "stop":
            body = Resume(value=Var(local=parameters[0]), origin=resume)
        else:
            left = (Local(name="x", type=REAL), Local(name="x_slot", type=INT))
            right = (Local(name="y", type=REAL), Local(name="y_slot", type=INT))
            x, x_slot = (Var(local=local) for local in left)
            y, y_slot = (Var(local=local) for local in right)
            if name == "less":
                continuation: Computation = Resume(
                    value=self.primitive("strict_lt_real", x, y), origin=resume
                )
            else:
                slot = Var(local=parameters[0])
                primal = self.primitive(f"{name}_real", x, y)
                previous = Local(name="answer", type=self.handler.output_type)
                continuation = Bind(
                    first=Resume(value=self.number(primal, slot), origin=resume),
                    binder=previous,
                    then=self.update(name, previous, slot, (x, x_slot), (y, y_slot)),
                )
            body = self.split(
                name,
                parameters[-2],
                left,
                self.split(name, parameters[-1], right, continuation),
            )
        return HandlerClauseDef(
            operation=operation.id,
            grade=ResumptionGrade.LINEAR,
            parameters=parameters,
            body=body,
        )


def _checked_reverse_graph(
    module: QiecModule,
    instance: EffectInstanceId,
    handler: HandlerId,
    constructor: ConstructorId,
    *,
    maximum: int,
    limit_code: str,
) -> tuple[
    HandlerDef, tuple[OperationDef, ...], tuple[_ReverseVertex, ...], _ReverseSink
]:
    """Check the source and its nominal linear handler signature together.

    Parameters
    ----------
    module : QiecModule
        The module containing the selected graph and its declarations.
    instance : EffectInstanceId
        The actual declared lexical reverse instance.
    handler : HandlerId
        Its total grade-one handler.
    constructor : ConstructorId
        Its nominal constructor with Real and Int fields.
    maximum : int
        The largest graph supported by the selected elaboration.
    limit_code : str
        The diagnostic code for exceeding that maximum.

    Returns
    -------
    tuple
        The handler, its operations, the vertices and the slot selection.

    Raises
    ------
    KernelError
        If module validation or the nominal graph contract fails.
    """
    if len(module.computations) != 1:
        raise KernelError("reverse differentiation needs one selected source entry")
    registry = validate_module(module)
    declaration = registry.handler(handler)
    entry = next(
        (item.entry for item in module.instances if item.entry.instance == instance),
        None,
    )
    if entry is None or declaration.effect != entry.effect:
        raise KernelError("reverse differentiation needs its declared nominal instance")
    vertex = registry.constructor(constructor)
    family = registry.families[vertex.family]
    if (
        not family.closed
        or family.parameters
        or family.indices
        or family.constructors != (vertex.id,)
        or vertex.telescope
        or tuple(field.type for field in vertex.fields) != (REAL, INT)
    ):
        raise KernelError("a reverse carrier needs one nominal Real and Int pair")
    number = TypeApplication(constructor=family.type_constructor, arguments=())
    interface = registry.effects[entry.effect.id]
    operations = {op.name: op for op in interface.operations}
    expected = {
        "seed": ((INT, REAL), number),
        "add": ((INT, number, number), number),
        "mul": ((INT, number, number), number),
        "stop": ((number,), number),
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
        raise KernelError("reverse operations do not match their nominal carrier")
    if (
        declaration.input_type != number
        or declaration.implementation != "foreign"
        or declaration.foreign is not None
        or not declaration.total
        or declaration.telescope
        or declaration.parameters
        or declaration.return_clause is not None
        or declaration.introduced != EMPTY_ROW
        or any(
            clause.grade is not ResumptionGrade.LINEAR for clause in declaration.clauses
        )
    ):
        raise KernelError(
            "reverse differentiation needs its total linear handler signature"
        )
    vertices, chosen = _source_vertices(
        module.computations[0],
        instance,
        declaration,
        operations,
        maximum=maximum,
        limit_code=limit_code,
    )
    if declaration.output_type != product_type(*((REAL,) * (len(vertices) + 1))):
        raise KernelError(
            "reverse answer needs the primal and one Real cotangent per vertex"
        )
    return declaration, tuple(operations.values()), vertices, chosen


def elaborate_reverse_diff(
    module: QiecModule,
    instance: EffectInstanceId,
    handler: HandlerId,
    constructor: ConstructorId,
) -> QiecModule:
    """Fill a named reverse handler with a bounded immutable cotangent sweep.

    The source seeds its Real parameters in order, can seed Real constants
    thereafter, and adds or multiplies earlier vertices. A terminal stop
    chooses the output vertex. Optional named comparisons may choose
    between stops on that same graph. Each reached comparison needs finite
    distinct primals and resumes once with its Boolean result. Arithmetic
    clauses first resume once and update both operand slots in the returned
    answer. The answer consists of the primal followed by one cotangent
    per vertex in source order. Parameter cotangents come first.

    Parameters
    ----------
    module : QiecModule
        One checked named reverse graph with an optional lazy sink selection.
    instance : EffectInstanceId
        The actual declared lexical reverse instance.
    handler : HandlerId
        Its total grade-one handler signature.
    constructor : ConstructorId
        Its one nominal constructor with Real and Int fields.

    Returns
    -------
    QiecModule
        The ordinary revalidated module with an authored reverse handler.

    Raises
    ------
    KernelError
        If the numeric carrier, signature or source violates the exact
        graph shape. The code `qiec-reverse-limit` identifies a graph
        with more than six vertices. The code `qiec-reverse-sink-limit`
        identifies more than 15 nested or 256 total comparisons.
    """
    declaration, operations, vertices, _ = _checked_reverse_graph(
        module,
        instance,
        handler,
        constructor,
        maximum=_MAX_VERTICES,
        limit_code="qiec-reverse-limit",
    )
    builder = _ReverseHandler(module, declaration, constructor, len(vertices))
    authored = declaration.with_(
        implementation="authored",
        return_clause=builder.return_clause(),
        clauses=tuple(builder.clause(operation) for operation in operations),
    )
    lowered = module.with_(
        handlers=tuple(
            authored if item.id == handler else item for item in module.handlers
        )
    )
    validate_module(lowered)
    return lowered


def _scalar_operation(
    name: str, origin: SourceOrigin, *arguments: Value
) -> PrimitiveApplication:
    """Create a scalar primitive at its associated source request.

    Parameters
    ----------
    name : str
        The registered scalar operation.
    origin : SourceOrigin
        The source request associated with the primitive.
    arguments : Value
        The checked operands in order.

    Returns
    -------
    PrimitiveApplication
        The pure scalar term.
    """
    primitive = PRIMITIVES[name]
    return PrimitiveApplication(
        primitive=primitive.id,
        name=primitive.name,
        arguments=arguments,
        result_type=primitive.result,
        origin=origin,
    )


def _compiled_sink(sink: _ReverseSink, primals: list[Var]) -> Value:
    """Select the output slot and primal using only reached comparisons."""
    if isinstance(sink, int):
        return TupleValue(
            items=(LiteralValue(value=sink, type=INT), primals[sink]),
            result_type=_SINK_CHOICE,
        )
    return ConditionalValue(
        condition=_scalar_operation(
            "strict_lt_real", sink.origin, primals[sink.left], primals[sink.right]
        ),
        then=_compiled_sink(sink.then, primals),
        otherwise=_compiled_sink(sink.otherwise, primals),
        result_type=_SINK_CHOICE,
    )


def _compiled_body(
    entry: NamedComputation, vertices: tuple[_ReverseVertex, ...], chosen: _ReverseSink
) -> Return:
    """Bind every primal and adjoint in dependency order.

    Parameters
    ----------
    entry : NamedComputation
        The source entry providing parameters and the result type.
    vertices : tuple[_ReverseVertex, ...]
        The checked vertices in allocation order.
    chosen : int or _SinkGuard
        The terminal output slot or the finite lazy decision selecting it.

    Returns
    -------
    Return
        One flat value binding sequence producing the primal and adjoints.
    """
    used = {parameter.name for parameter in entry.parameters}
    bindings: list[LetBinding] = []
    supply = count()

    def bind(value: Value, type_: TypeExpr = REAL) -> Var:
        """Append one binding and return its reference.

        Parameters
        ----------
        value : Value
            The Real value to bind in the generated sequence.
        type_ : TypeExpr
            The result type of the bound value.

        Returns
        -------
        Var
            The fresh reference.
        """
        # Target name sanitization cannot turn a different source name into
        # one of these ASCII alphanumeric names.
        name = next(
            f"reverseValue{i}" for i in supply if f"reverseValue{i}" not in used
        )
        binder = Local(name=name, type=type_)
        bindings.append(LetBinding(binder=binder, value=value))
        return Var(local=binder)

    primals: list[Var] = []
    for vertex in vertices:
        value = (
            vertex.value
            if isinstance(vertex, _SeedVertex)
            else _scalar_operation(
                f"{vertex.operation}_real",
                vertex.origin,
                primals[vertex.left],
                primals[vertex.right],
            )
        )
        primals.append(bind(value))

    pending: list[list[Value]] = [[] for _ in vertices]
    primal: Value
    if isinstance(chosen, int):
        primal = primals[chosen]
        pending[chosen].append(_ONE)
    else:
        selection = bind(_compiled_sink(chosen, primals), _SINK_CHOICE)
        selected = Projection(value=selection, position=0, result_type=INT)
        primal = Projection(value=selection, position=1, result_type=REAL)
        for slot, vertex in enumerate(vertices):
            at_slot = _scalar_operation(
                "eq_int", vertex.origin, selected, LiteralValue(value=slot, type=INT)
            )
            pending[slot].append(
                _scalar_operation("select_real", vertex.origin, at_slot, _ONE, _ZERO)
            )
    adjoints: list[Var] = []
    for slot in reversed(range(len(vertices))):
        vertex = vertices[slot]
        terms = iter(pending[slot])
        total = next(terms, _ZERO)
        for term in terms:
            total = bind(_scalar_operation("add_real", vertex.origin, total, term))
        cotangent = bind(total)
        adjoints.append(cotangent)
        if isinstance(vertex, _BinaryVertex):
            # The two positions are separate edges even when their slots agree.
            for predecessor, other in (
                (vertex.left, vertex.right),
                (vertex.right, vertex.left),
            ):
                contribution = (
                    cotangent
                    if vertex.operation == "add"
                    else _scalar_operation(
                        "mul_real", vertex.origin, cotangent, primals[other]
                    )
                )
                pending[predecessor].append(contribution)

    result_type = entry.type.result
    result = TupleValue(items=(primal, *reversed(adjoints)), result_type=result_type)
    return Return(
        value=LetValue(bindings=tuple(bindings), body=result, result_type=result_type)
    )


def compile_reverse_diff(
    module: QiecModule,
    instance: EffectInstanceId,
    handler: HandlerId,
    constructor: ConstructorId,
) -> QiecModule:
    """Compile a checked nominal reverse graph to an immutable pure computation.

    The compiled module contains the specialized computation and its entry
    points. Its arithmetic retains source vertex origins and emits no
    original operation events. Use `elaborate_reverse_diff` to run the
    named handler and observe those events.

    Each binary vertex adds one primal operation and sends a contribution
    along each ordered operand edge, including two uses of the same
    operand. All contributions to a vertex are added before its adjoint is
    passed down. With B binary vertices the computation executes at most
    5B scalar arithmetic operations. Lazy sink selection adds one
    comparison per reached test and two scalar operations per vertex to
    seed the one shared backward sweep. The bound excludes parsing and
    validation. The recurrence gives the exact selected polynomial
    derivative where the reached comparisons are strict. Runtime Real
    values can round or overflow.

    Parameters
    ----------
    module : QiecModule
        One checked named reverse graph with an optional lazy sink selection.
    instance : EffectInstanceId
        The actual declared lexical reverse instance.
    handler : HandlerId
        Its total grade-one handler signature.
    constructor : ConstructorId
        Its one nominal constructor with Real and Int fields.

    Returns
    -------
    QiecModule
        The revalidated pure computation and its entry points. The result
        is the selected primal followed by all vertex cotangents in the
        original allocation order.

    Raises
    ------
    KernelError
        If the carrier, handler or source violates the nominal graph
        contract. The code `qiec-reverse-compile-limit` identifies a graph
        with more than 256 vertices. The code `qiec-reverse-sink-limit`
        identifies more than 15 nested or 256 total comparisons.
    """
    _, _, vertices, chosen = _checked_reverse_graph(
        module,
        instance,
        handler,
        constructor,
        maximum=_MAX_COMPILED_VERTICES,
        limit_code="qiec-reverse-compile-limit",
    )
    (entry,) = module.computations
    compiled = QiecModule(
        module=module.module,
        source_protocol=module.source_protocol,
        computations=(entry.with_(body=_compiled_body(entry, vertices, chosen)),),
        entries=module.entries,
        gap=module.gap,
    )
    validate_module(compiled)
    return compiled


__all__ = ["compile_reverse_diff", "elaborate_reverse_diff"]
