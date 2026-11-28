"""Compile a bounded nominal reverse handler with immutable cotangents."""

from __future__ import annotations

from math import isfinite

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
    ConstructorValue,
    Handle,
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
from quivers.qiec.types import INT, REAL, TypeApplication, product_type

_MAX_VERTICES = 6
_ZERO = LiteralValue(value=0.0, type=REAL)
_ONE = LiteralValue(value=1.0, type=REAL)


def _source_vertices(
    entry: NamedComputation,
    instance: EffectInstanceId,
    handler: HandlerDef,
    operations: dict[str, OperationDef],
) -> int:
    """Validate sequential allocation and the terminal stop of the actual source.

    Parameters
    ----------
    entry : NamedComputation
        The single source entry with the original nominal requests.
    instance : EffectInstanceId
        The sole lexical reverse scope.
    handler : HandlerDef
        Its linear handler signature.
    operations : dict[str, OperationDef]
        The checked seed, binary operators and stopping operation.

    Returns
    -------
    int
        The vertex count, in allocation order.

    Raises
    ------
    KernelError
        If source control, inputs, aliases, allocation or stopping fails.
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
    seen: set[Local] = set()
    while isinstance(term, Bind):
        if len(seen) == _MAX_VERTICES:
            raise KernelError(
                f"reverse source exceeds {_MAX_VERTICES} vertices", "qiec-reverse-limit"
            )
        node = term.first
        if not isinstance(node, Perform) or node.request.instance != instance:
            raise KernelError("a reverse vertex must use the active named instance")
        request = node.request
        name = request.operation
        if name not in (
            operations["seed"].id,
            operations["add"].id,
            operations["mul"].id,
        ):
            raise KernelError("a reverse vertex must be seeded, added or multiplied")
        slot, *operands = request.arguments
        position = len(seen)
        if (
            not isinstance(slot, LiteralValue)
            or slot.type != INT
            or type(slot.value) is not int
            or slot.value != position
        ):
            raise KernelError("reverse vertex slots must be fresh and consecutive")
        if position < len(entry.parameters):
            if name != operations["seed"].id or operands != [
                Var(local=entry.parameters[position])
            ]:
                raise KernelError(
                    "reverse source must seed each input once in parameter order"
                )
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
        elif not all(
            isinstance(operand, Var) and operand.local in seen for operand in operands
        ):
            raise KernelError(
                "reverse operands must be earlier vertices of the same instance"
            )
        seen.add(term.binder)
        term = term.then
    if (
        len(seen) < max(1, len(entry.parameters))
        or not isinstance(term, Perform)
        or term.request.instance != instance
        or term.request.operation != operations["stop"].id
        or not isinstance(chosen := term.request.arguments[0], Var)
        or chosen.local not in seen
    ):
        raise KernelError("reverse source must stop on a vertex of its own instance")
    return len(seen)


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
        primitive = PRIMITIVES[name]
        return PrimitiveApplication(
            primitive=primitive.id,
            name=primitive.name,
            arguments=arguments,
            result_type=primitive.result,
            origin=self.origin(name, "coefficient"),
        )

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
        """Generate a linear clause for a seed, binary vertex or stop.

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
            slot = Var(local=parameters[0])
            left = (Local(name="x", type=REAL), Local(name="x_slot", type=INT))
            right = (Local(name="y", type=REAL), Local(name="y_slot", type=INT))
            x, x_slot = (Var(local=local) for local in left)
            y, y_slot = (Var(local=local) for local in right)
            primal = self.primitive(f"{name}_real", x, y)
            previous = Local(name="answer", type=self.handler.output_type)
            continuation = Bind(
                first=Resume(value=self.number(primal, slot), origin=resume),
                binder=previous,
                then=self.update(name, previous, slot, (x, x_slot), (y, y_slot)),
            )
            body = self.split(
                name,
                parameters[1],
                left,
                self.split(name, parameters[2], right, continuation),
            )
        return HandlerClauseDef(
            operation=operation.id,
            grade=ResumptionGrade.LINEAR,
            parameters=parameters,
            body=body,
        )


def elaborate_reverse_diff(
    module: QiecModule,
    instance: EffectInstanceId,
    handler: HandlerId,
    constructor: ConstructorId,
) -> QiecModule:
    """Fill a named reverse handler with a bounded immutable cotangent sweep.

    The source seeds its Real parameters in order, can seed Real constants
    thereafter, and adds or multiplies earlier vertices. The explicit
    final stop chooses the output vertex. Arithmetic clauses first resume
    once and update both operand slots in the returned answer. The answer
    consists of the primal followed by one cotangent per vertex in source
    order. The parameter cotangents form the first part of that vector.

    Parameters
    ----------
    module : QiecModule
        One checked source entry with a straight-line named reverse graph.
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
        forward graph shape. The code `qiec-reverse-limit` identifies a
        graph with more than six vertices.
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
    vertices = _source_vertices(
        module.computations[0], instance, declaration, operations
    )
    if declaration.output_type != product_type(*((REAL,) * (vertices + 1))):
        raise KernelError(
            "reverse answer needs the primal and one Real cotangent per vertex"
        )
    builder = _ReverseHandler(module, declaration, constructor, vertices)
    authored = declaration.with_(
        implementation="authored",
        return_clause=builder.return_clause(),
        clauses=tuple(builder.clause(operation) for operation in operations.values()),
    )
    lowered = module.with_(
        handlers=tuple(
            authored if item.id == handler else item for item in module.handlers
        )
    )
    validate_module(lowered)
    return lowered


__all__ = ["elaborate_reverse_diff"]
