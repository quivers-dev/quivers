"""Check nominal effect equations against authored QIEC handlers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from quivers.qiec.effects import (
    EMPTY_ROW,
    ComputationType,
    HandlerDef,
    RowEntry,
)
from quivers.qiec.identifiers import (
    ComputationId,
    EffectId,
    EffectInstanceId,
    HandlerId,
    OperationId,
)
from quivers.qiec.laws import FiniteLawDecision, FiniteLawUnsupported, check_finite_law
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.serialization import dumps
from quivers.qiec.terms import (
    Bind,
    Call,
    Handle,
    If,
    LiteralValue,
    Local,
    Perform,
    Return,
    Var,
)
from quivers.qiec.types import BOOL, TypeExpr


@dataclass(frozen=True, slots=True)
class EffectEquation:
    """Declare a pair of checked computations over one nominal effect.

    The law's name is scoped to its effect. Its statement is the two
    computation bodies, checked under the listed nominal operation set.

    Parameters
    ----------
    effect : EffectId
        The declared interface owning the law.
    name : str
        The law's name within that interface.
    instance : EffectInstanceId
        The concrete checked instance whose requests form the two sides.
    operations : tuple[OperationId, ...]
        The operations occurring across both bodies.
    left : ComputationId
        The left computation.
    right : ComputationId
        The right computation, with the same positional domain and result.
    """

    effect: EffectId
    name: str
    instance: EffectInstanceId
    operations: tuple[OperationId, ...]
    left: ComputationId
    right: ComputationId

    def __post_init__(self) -> None:
        """Require a law name and a nonempty, distinct operation set.

        Raises
        ------
        ValueError
            If the name is empty or the operation set is empty or repeated.
        """
        if not self.name.strip():
            raise ValueError("an effect equation needs a name")
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("an effect equation needs distinct performed operations")
        object.__setattr__(self, "operations", tuple(sorted(self.operations, key=str)))


@dataclass(frozen=True, slots=True)
class FiniteEffectLawDecision:
    """An exhaustive handler-law decision on a finite carrier.

    Parameters
    ----------
    equation : EffectEquation
        The checked equation.
    handler : HandlerId
        The exact handler installed around each side.
    fingerprint : str
        The canonical checked module and equation binding, including the
        handler identity and the checking fragment.
    parameter_types : tuple[TypeExpr, ...]
        The common domain in order, followed by the handler's initial
        parameter types. The checker exhausts every assignment.
    answer_type : TypeExpr
        The type of the observable handled answers.
    result : FiniteLawDecision
        The exhaustive result or a separating assignment and both answers.
    """

    equation: EffectEquation
    handler: HandlerId
    fingerprint: str
    parameter_types: tuple[TypeExpr, ...]
    answer_type: TypeExpr
    result: FiniteLawDecision

    @property
    def certified(self) -> bool:
        """Whether the installed handler satisfies the finite equation.

        Returns
        -------
        bool
            Whether every enumerated assignment agrees.
        """
        return self.result.counterexample is None


def _operations(side: NamedComputation, instance: RowEntry) -> set[OperationId]:
    """Inspect every operation in an open equation side.

    Parameters
    ----------
    side : NamedComputation
        The checked left or right computation of the equation.
    instance : RowEntry
        The one instance the equation can use.

    Returns
    -------
    set[OperationId]
        The nominal operations performed by its body.

    Raises
    ------
    FiniteLawUnsupported
        If the side can use another instance or hide a computation.
    """
    row = side.type.effects
    if (
        side.telescope
        or row.tail is not None
        or any(entry != instance for entry in row.entries)
    ):
        raise FiniteLawUnsupported(
            f"equation side `{side.name}` uses an incompatible row"
        )
    result: set[OperationId] = set()
    pending = [side.body]
    while pending:
        body = pending.pop()
        if isinstance(body, Bind):
            pending.extend((body.first, body.then))
        elif isinstance(body, If):
            pending.extend((body.then, body.otherwise))
        elif isinstance(body, Perform):
            request = body.request
            if (
                request.instance != instance.instance
                or request.effect != instance.effect
            ):
                raise FiniteLawUnsupported(
                    "equation request uses another nominal instance"
                )
            result.add(request.operation)
        elif not isinstance(body, Return):
            raise FiniteLawUnsupported(
                f"equation side `{side.name}` contains unsupported {type(body).__name__}"
            )
    return result


def _probe(
    equation: EffectEquation,
    side: NamedComputation,
    other: NamedComputation,
    handler: HandlerDef,
    label: str,
) -> NamedComputation:
    """Install the selected handler around one checked equation side.

    Parameters
    ----------
    equation : EffectEquation
        The law whose instance is handled.
    side : NamedComputation
        The checked equation side to run.
    other : NamedComputation
        The other side, which can supply a row for a pure side.
    handler : HandlerDef
        The actual authored handler under test.
    label : str
        The side's positional tag in the fresh probe.

    Returns
    -------
    NamedComputation
        The closed probe parameterized by all equation inputs and handler state.
    """
    identity = ComputationId.derive(
        "finite-effect-equation-v1",
        str(equation.effect),
        equation.name,
        str(equation.instance),
        str(side.id),
        str(handler.id),
        label,
    )
    parameters = tuple(
        Local(name=f"law_arg_{index}", type=argument.type)
        for index, argument in enumerate(side.parameters)
    )
    states = tuple(
        Local(name=f"law_state_{index}", type=argument.type)
        for index, argument in enumerate(handler.parameters)
    )
    call = Call(
        callee=side.id,
        name=side.name,
        static_arguments=(),
        arguments=tuple(Var(local=parameter) for parameter in parameters),
        result_type=side.type.result,
        effects=side.type.effects,
        origin=side.origin,
    )
    computation: Call | If = call
    if side.type.effects == EMPTY_ROW:
        # The inactive branch supplies the row required by the delimiter
        # so even a pure side runs the actual handler's return clause.
        computation = If(
            condition=LiteralValue(value=False, type=BOOL),
            then=call.with_(
                callee=other.id,
                name=other.name,
                effects=other.type.effects,
                origin=other.origin,
            ),
            otherwise=call,
        )
    return NamedComputation(
        id=identity,
        name=f"effect_law_{identity.digest}",
        telescope=(),
        parameters=(*parameters, *states),
        body=Handle(
            instance=equation.instance,
            handler=handler.id,
            computation=computation,
            arguments=tuple(Var(local=state) for state in states),
        ),
        type=ComputationType(result=handler.output_type, effects=EMPTY_ROW),
        origin=side.origin,
    )


def _fingerprint(
    module: QiecModule,
    equation: EffectEquation,
    handler: HandlerId,
) -> str:
    """Bind a finite comparison to its actual checked source and hypothesis.

    Parameters
    ----------
    module : QiecModule
        The canonical module, including the actual bodies and parameter types.
    equation : EffectEquation
        The exact nominal equation under verification.
    handler : HandlerId
        The selected actual handler.

    Returns
    -------
    str
        The SHA-256 binding fingerprint.
    """
    return sha256(
        dumps(
            (
                "finite-effect-equation-v1",
                module,
                equation.effect,
                equation.name,
                equation.instance,
                equation.operations,
                equation.left,
                equation.right,
                handler,
            )
        ).encode("utf-8")
    ).hexdigest()


def check_finite_effect_law(
    module: QiecModule,
    equation: EffectEquation,
    handler: HandlerId,
    *,
    prior: FiniteEffectLawDecision | None = None,
    max_assignments: int = 65_536,
    fuel: int = 65_536,
) -> FiniteEffectLawDecision:
    """Check the actual handler on both sides of a declared finite equation.

    Each side is open over the equation's concrete instance and built from
    returns, sequences, branches and direct requests. The checker installs
    the same authored handler around each side and checks every assignment
    to their common parameters and initial handler state. Continuations can
    be represented by finite products in the equation's parameter types.

    An earlier result can supply the expected subject and fingerprint.
    The current implementation is then checked again on that subject.

    Parameters
    ----------
    module : QiecModule
        The checked source containing the interface, instance, equation
        sides and handler.
    equation : EffectEquation
        The law's exact nominal scope and two named open bodies.
    handler : HandlerId
        The authored handler to install on that instance for both probes.
    prior : FiniteEffectLawDecision or None
        An earlier result whose equation, handler and binding must agree.
    max_assignments : int
        The maximum number of finite input assignments.
    fuel : int
        The reference step limit per side and assignment.

    Returns
    -------
    FiniteEffectLawDecision
        The current checker decision and its exact subject binding.

    Raises
    ------
    FiniteLawUnsupported
        If the subject has a mismatched identity, scope, checked carrier,
        unsupported term, or implementation.
    FiniteLawLimit
        If an assignment or evaluator limit prevents an exhaustive decision.
    """
    registry = validate_module(module)
    interface = registry.effects.get(equation.effect)
    instance = next(
        (
            item.entry
            for item in module.instances
            if item.entry.instance == equation.instance
        ),
        None,
    )
    if interface is None or instance is None or instance.effect.id != equation.effect:
        raise FiniteLawUnsupported(
            "effect equation names an unknown or mismatched nominal instance"
        )
    target = registry.handlers.get(handler)
    if (
        target is None
        or target.effect != instance.effect
        or target.telescope
        or target.implementation != "authored"
        or not target.total
        or target.introduced != EMPTY_ROW
    ):
        raise FiniteLawUnsupported(
            "effect equation needs a matching closed authored total handler"
        )
    roots = {item.id: item for item in module.computations}
    lhs, rhs = roots.get(equation.left), roots.get(equation.right)
    if lhs is None or rhs is None:
        raise FiniteLawUnsupported("effect equation names an unknown computation")
    if tuple(local.type for local in lhs.parameters) != tuple(
        local.type for local in rhs.parameters
    ):
        raise FiniteLawUnsupported(
            "effect equation needs matching positional parameter types"
        )
    if lhs.type.result != target.input_type or rhs.type.result != target.input_type:
        raise FiniteLawUnsupported(
            "equation sides disagree with the handler's input carrier"
        )
    uses = _operations(lhs, instance) | _operations(rhs, instance)
    allowed = {operation.id for operation in interface.operations}
    if uses != set(equation.operations) or not uses <= allowed:
        raise FiniteLawUnsupported(
            "effect equation's performed operations do not match its nominal declaration"
        )

    binding = _fingerprint(module, equation, handler)
    left = _probe(equation, lhs, rhs, target, "left")
    right = _probe(equation, rhs, lhs, target, "right")
    parameters = tuple(parameter.type for parameter in left.parameters)
    if prior is not None and (
        prior.equation != equation
        or prior.handler != handler
        or prior.fingerprint != binding
        or prior.parameter_types != parameters
        or prior.answer_type != target.output_type
    ):
        raise FiniteLawUnsupported(
            "effect law evidence does not match the checked subject"
        )
    checked = module.with_(computations=(*module.computations, left, right))
    result = check_finite_law(
        checked, left.name, right.name, max_assignments=max_assignments, fuel=fuel
    )
    return FiniteEffectLawDecision(
        equation, handler, binding, parameters, target.output_type, result
    )


__all__ = ["EffectEquation", "FiniteEffectLawDecision", "check_finite_effect_law"]
