"""Exact derivatives on regular scalar executions of checked QIEC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from operator import eq, ge, gt, le, lt, ne
from typing import TYPE_CHECKING, cast

from quivers.qiec.checking import KernelRegistry
from quivers.qiec.contracts import Condition, DerivativeSemantics
from quivers.qiec.derivative_rule_schema import RationalDerivativeDeclaration
from quivers.qiec.evaluator import EvaluationError, Evaluator, FuelExhaustedError
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.primitives import PRIMITIVES
from quivers.qiec.terms import (
    ConditionalValue,
    LetValue,
    LiteralValue,
    Local,
    PrimitiveApplication,
    Projection,
    TupleValue,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, REAL, UNIT

if TYPE_CHECKING:
    from quivers.qiec.derivative_rules import RationalDerivativeDecision

_ARITHMETIC = {
    PRIMITIVES[name].id: name
    for name in ("add_real", "sub_real", "mul_real", "div_real", "neg_real", "pow_real")
}
_COMPARISONS = {
    PRIMITIVES[name].id: operation
    for name, operation in (
        ("eq_real", eq),
        ("ne_real", ne),
        ("lt_real", lt),
        ("le_real", le),
        ("gt_real", gt),
        ("ge_real", ge),
    )
}
_BOOLEANS = {PRIMITIVES[name].id for name in ("and", "or", "not", "eq_bool", "ne_bool")}
_MAX_POWER = 1_024
_MAX_COEFFICIENT_BITS = 16_384
_IMPLICIT_ROOT = DerivativeSemantics.implicit(
    Condition(
        name="regular-residual",
        statement="The residual is continuously differentiable near the supplied root.",
    ),
    Condition(
        name="zero-residual",
        statement="The residual at the supplied root is exactly zero.",
    ),
    Condition(
        name="nonzero-state-derivative",
        statement="The residual derivative with respect to its first parameter is nonzero.",
    ),
)


class JvpUnsupported(ValueError):
    """A checked computation lies outside the exact scalar derivative fragment."""


class JvpBoundaryError(JvpUnsupported):
    """Comparison stability could not be certified at an equality boundary."""


class JvpLimit(ValueError):
    """Exact derivative evaluation exhausted a resource bound."""


class ImplicitRootError(ValueError):
    """The supplied point cannot establish a regular implicit root."""


@dataclass(frozen=True, slots=True)
class ExactGradient:
    """A scalar and the exact gradient returned by its reverse pass.

    Parameters
    ----------
    value : Fraction
        The result at the supplied point.
    gradient : tuple[Fraction, ...]
        Its partial derivatives, in parameter order.
    """

    value: Fraction
    gradient: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class ExactImplicitGradient:
    """The exact gradient of a local scalar solution branch.

    Parameters
    ----------
    value : Fraction
        The supplied root verified by the residual computation.
    gradient : tuple[Fraction, ...]
        The local solution's partial derivatives in parameter order.
    state_derivative : Fraction
        The residual derivative with respect to the root coordinate.
    """

    value: Fraction
    gradient: tuple[Fraction, ...]
    state_derivative: Fraction

    @property
    def semantics(self) -> DerivativeSemantics:
        """The derivative report and its checked implicit hypotheses.

        Returns
        -------
        DerivativeSemantics
            An implicit derivative over the regular zero of the residual.
        """
        return _IMPLICIT_ROOT


@dataclass(frozen=True, slots=True)
class ExactJvp:
    """A computed value and its exact directional derivative.

    Parameters
    ----------
    value : Fraction
        The result of the admitted scalar computation.
    tangent : Fraction
        Its Jacobian applied to the supplied input direction.
    """

    value: Fraction
    tangent: Fraction


@dataclass(frozen=True, slots=True)
class ExactMixedDerivative:
    """A scalar and its derivatives along two independent input directions.

    Parameters
    ----------
    value : Fraction
        The result at the supplied point.
    first : Fraction
        The derivative along the first direction.
    second : Fraction
        The derivative along the second direction.
    mixed : Fraction
        The Hessian evaluated on the two directions.
    """

    value: Fraction
    first: Fraction
    second: Fraction
    mixed: Fraction


@dataclass(frozen=True, slots=True)
class _Dual:
    """Coefficients of a scalar in two square-zero perturbation variables.

    Parameters
    ----------
    value : Fraction
        The scalar primal.
    tangent : Fraction
        Its propagated directional derivative.
    dependent : bool
        Whether the expression depends syntactically on any input. A
        zero tangent in one direction does not establish independence.
    second_tangent : Fraction
        Its derivative along the second perturbation.
    mixed : Fraction
        The coefficient of the product of the two perturbations.
    """

    value: Fraction
    tangent: Fraction
    dependent: bool
    second_tangent: Fraction
    mixed: Fraction


@dataclass(frozen=True, slots=True)
class _ReverseNumber:
    """An exact scalar paired with its occurrence in the reverse graph.

    Parameters
    ----------
    value : Fraction
        The exact primal at this occurrence.
    index : int or None
        Its graph node, or no node for an input-independent expression.
    """

    value: Fraction
    index: int | None

    @property
    def dependent(self) -> bool:
        """Whether the value may depend on the input parameters.

        Returns
        -------
        bool
            Whether the graph carries a node for this occurrence.
        """
        return self.index is not None


type _Scalar = _Dual | _ReverseNumber
type _ReverseEdge = tuple[int, Fraction]


def _bound(*coefficients: Fraction) -> None:
    """Check the size of exact arithmetic coefficients.

    Parameters
    ----------
    coefficients : Fraction
        The rational values whose numerator and denominator are checked.

    Raises
    ------
    JvpLimit
        If a numerator or denominator exceeds the coefficient bound.
    """
    if any(
        max(coefficient.numerator.bit_length(), coefficient.denominator.bit_length())
        > _MAX_COEFFICIENT_BITS
        for coefficient in coefficients
    ):
        raise JvpLimit("exact derivative coefficient limit exceeded")


def _dual(
    value: Fraction,
    tangent: Fraction,
    dependent: bool,
    *,
    second_tangent: Fraction = Fraction(),
    mixed: Fraction = Fraction(),
) -> _Dual:
    """Construct a scalar in the perturbation algebra under the coefficient bound.

    Parameters
    ----------
    value : Fraction
        The primal.
    tangent : Fraction
        The directional derivative.
    dependent : bool
        Whether either operand can depend on the inputs.
    second_tangent : Fraction
        The derivative along the second perturbation.
    mixed : Fraction
        The coefficient of the product of both perturbations.

    Returns
    -------
    _Dual
        The exact immutable coefficients and their dependence flag.

    Raises
    ------
    JvpLimit
        If any coefficient exceeds the size limit.
    """
    _bound(value, tangent, second_tangent, mixed)
    return _Dual(value, tangent, dependent, second_tangent, mixed)


class _ScalarDerivativeEvaluator(Evaluator):
    """Use exact rational control and arithmetic in the checked reference machine."""

    def _constant(self, value: Fraction) -> _Scalar:
        """Embed a literal in the current scalar arithmetic.

        Parameters
        ----------
        value : Fraction
            The input-independent primal.

        Returns
        -------
        _Scalar
            The scalar with no derivative contribution.
        """
        return _dual(value, Fraction(), False)

    def _duplicable_value(self, value: object) -> bool:
        """Allow immutable tangent values to be shared by separate shots.

        Parameters
        ----------
        value : object
            A value in a captured continuation.

        Returns
        -------
        bool
            Whether it can be carried into another shot.
        """
        return isinstance(value, _Dual | _ReverseNumber) or super()._duplicable_value(
            value
        )

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Read a supported value with its scalar derivative.

        Parameters
        ----------
        value : Value
            The checked source expression.
        environment : Mapping[Local, object]
            The values in its lexical scope.

        Returns
        -------
        object
            The exact value and tangent, or the reference discrete value.

        Raises
        ------
        JvpUnsupported
            If the value does not have a rule in the scalar fragment.
        """
        if isinstance(value, PrimitiveApplication):
            return self.primitive(value, environment)
        if isinstance(value, LiteralValue):
            data = value.value
            if (
                value.type == REAL
                and isinstance(data, (int, float))
                and not isinstance(data, bool)
                and (isinstance(data, int) or isfinite(data))
            ):
                return self._constant(Fraction(data))
            if value.type in (BOOL, UNIT):
                return super()._value(value, environment)
        if isinstance(
            value, Var | LetValue | TupleValue | Projection | ConditionalValue
        ):
            return super()._value(value, environment)
        raise JvpUnsupported(
            f"{type(value).__name__} has no exact scalar derivative rule"
        )

    def primitive(
        self, term: PrimitiveApplication, environment: Mapping[Local, object]
    ) -> object:
        """Apply the exact derivative rule for a checked nominal primitive.

        Parameters
        ----------
        term : PrimitiveApplication
            The source primitive application.
        environment : Mapping[Local, object]
            The exact values bound in that application.

        Returns
        -------
        object
            A dual scalar or a locally constant Boolean value.

        Raises
        ------
        JvpUnsupported
            If the primitive is unsupported or undefined at the given input.
        JvpBoundaryError
            If an input-dependent comparison is at equality.
        """
        if term.primitive in _BOOLEANS and term.result_type == BOOL:
            return super()._value(term, environment)
        name = _ARITHMETIC.get(term.primitive)
        compare = _COMPARISONS.get(term.primitive)
        if not (
            name is not None
            and term.result_type == REAL
            or compare is not None
            and term.result_type == BOOL
        ):
            raise JvpUnsupported(
                f"primitive `{term.name}` has no exact derivative rule"
            )
        arguments = tuple(
            self._value(argument, environment) for argument in term.arguments
        )
        assert all(isinstance(item, _Dual | _ReverseNumber) for item in arguments)
        scalars = cast(tuple[_Scalar, ...], arguments)
        if compare is not None:
            first, second = scalars
            dependent = first.dependent or second.dependent
            if dependent and first.value == second.value:
                raise JvpBoundaryError(
                    f"comparison `{term.name}` is at an input-dependent equality boundary"
                )
            return self.numbers.number("truth", compare(first.value, second.value))
        assert name is not None
        return self._calculate(name, scalars)

    def _calculate(self, name: str, operands: tuple[_Scalar, ...]) -> _Scalar:
        """Evaluate arithmetic with the active derivative representation.

        Parameters
        ----------
        name : str
            The resolved arithmetic primitive.
        operands : tuple[_Scalar, ...]
            Its checked operands.

        Returns
        -------
        _Scalar
            The exact scalar result and its derivative data.
        """
        return self._arithmetic(name, cast(tuple[_Dual, ...], operands))

    def _arithmetic(self, name: str, operands: tuple[_Dual, ...]) -> _Dual:
        """Compute the Taylor coefficients of one arithmetic operation.

        Parameters
        ----------
        name : str
            The resolved arithmetic primitive.
        operands : tuple[_Dual, ...]
            The primal and perturbation coefficients of the operands.

        Returns
        -------
        _Dual
            The Taylor coefficients of the result.

        Raises
        ------
        JvpUnsupported
            If the primitive is undefined on the given operands.
        """
        first = operands[0]
        if name == "neg_real":
            return _dual(
                -first.value,
                -first.tangent,
                first.dependent,
                second_tangent=-first.second_tangent,
                mixed=-first.mixed,
            )
        _, second = operands
        dependent = first.dependent or second.dependent
        if name == "add_real":
            return _dual(
                first.value + second.value,
                first.tangent + second.tangent,
                dependent,
                second_tangent=first.second_tangent + second.second_tangent,
                mixed=first.mixed + second.mixed,
            )
        if name == "sub_real":
            return _dual(
                first.value - second.value,
                first.tangent - second.tangent,
                dependent,
                second_tangent=first.second_tangent - second.second_tangent,
                mixed=first.mixed - second.mixed,
            )
        if name == "mul_real":
            return _dual(
                first.value * second.value,
                first.tangent * second.value + first.value * second.tangent,
                dependent,
                second_tangent=first.second_tangent * second.value
                + first.value * second.second_tangent,
                mixed=first.mixed * second.value
                + first.tangent * second.second_tangent
                + first.second_tangent * second.tangent
                + first.value * second.mixed,
            )
        if name == "div_real":
            if not second.value:
                raise JvpUnsupported(
                    "division by zero is outside the derivative domain"
                )
            quotient = first.value / second.value
            tangent = (first.tangent - quotient * second.tangent) / second.value
            other = (
                first.second_tangent - quotient * second.second_tangent
            ) / second.value
            return _dual(
                quotient,
                tangent,
                dependent,
                second_tangent=other,
                mixed=(
                    first.mixed
                    - quotient * second.mixed
                    - tangent * second.second_tangent
                    - other * second.tangent
                )
                / second.value,
            )
        return self.power(first, second)

    @staticmethod
    def power(base: _Dual, exponent: _Dual) -> _Dual:
        """Differentiate a power with a locally fixed integral exponent.

        Parameters
        ----------
        base : _Dual
            The base and its derivative.
        exponent : _Dual
            The exponent, which must be independent of the inputs.

        Returns
        -------
        _Dual
        The primal and derivative coefficients.

        Raises
        ------
        JvpUnsupported
            If the power is outside the admitted integral domain.
        JvpLimit
            If the exponent exceeds the computation bound.
        """
        if exponent.dependent or exponent.value.denominator != 1:
            raise JvpUnsupported("an exact JVP requires a fixed integral exponent")
        power = exponent.value.numerator
        if abs(power) > _MAX_POWER:
            raise JvpLimit("exact JVP exponent limit exceeded")
        if power < 0 and not base.value:
            raise JvpUnsupported("a negative power of zero has no derivative")
        if not power:
            return _dual(Fraction(1), Fraction(), False)
        if power == 1:
            return base
        slope = power * base.value ** (power - 1)
        curvature = power * (power - 1) * base.value ** (power - 2)
        return _dual(
            base.value**power,
            slope * base.tangent,
            base.dependent,
            second_tangent=slope * base.second_tangent,
            mixed=slope * base.mixed + curvature * base.tangent * base.second_tangent,
        )


class _ReverseEvaluator(_ScalarDerivativeEvaluator):
    """Record immutable scalar occurrences during one reference execution.

    Parameters
    ----------
    module : QiecModule
        The checked module of this run.
    fuel : int
        The maximum reference machine step count.
    """

    def __init__(self, module: QiecModule, fuel: int) -> None:
        super().__init__(module=module, fuel=fuel)
        self.parents: list[tuple[_ReverseEdge, ...]] = []

    def _constant(self, value: Fraction) -> _ReverseNumber:
        """Embed an input-independent literal without a graph node.

        Parameters
        ----------
        value : Fraction
            The exact primal value.

        Returns
        -------
        _ReverseNumber
            The primal and an absent node index.
        """
        _bound(value)
        return _ReverseNumber(value, None)

    def input(self, value: Fraction) -> _ReverseNumber:
        """Allocate a distinct graph leaf for one input parameter.

        Parameters
        ----------
        value : Fraction
            The parameter's value at the evaluation point.

        Returns
        -------
        _ReverseNumber
            The primal at the new leaf.
        """
        _bound(value)
        index = len(self.parents)
        self.parents.append(())
        return _ReverseNumber(value, index)

    def run(
        self,
        declaration: NamedComputation,
        registry: KernelRegistry,
        arguments: tuple[int | Fraction, ...],
        *,
        record: bool,
    ) -> _ReverseNumber:
        """Evaluate a validated scalar with or without its reverse graph.

        Parameters
        ----------
        declaration : NamedComputation
            The checked scalar computation.
        registry : KernelRegistry
            Its validated module registry.
        arguments : tuple[int or Fraction, ...]
            The complete argument vector in parameter order.
        record : bool
            Whether to create a reverse graph during this run.

        Returns
        -------
        _ReverseNumber
            The exact scalar value and its graph node when recorded.
        """
        embed = self.input if record else self._constant
        environment = {
            parameter: embed(_fraction(value))
            for parameter, value in zip(declaration.parameters, arguments, strict=True)
        }
        result = _run_derivative(self, declaration, registry, environment)
        assert isinstance(result, _ReverseNumber)
        return result

    def _local(
        self, name: str, operands: tuple[_ReverseNumber, ...], position: int | None
    ) -> _Dual:
        """Use the forward rule on one operand basis vector of a primitive.

        Parameters
        ----------
        name : str
            The resolved arithmetic primitive.
        operands : tuple[_ReverseNumber, ...]
            The operand primals and their input dependence.
        position : int or None
            The operand to seed, or no seed for a constant result.

        Returns
        -------
        _Dual
            The primal and its derivative with respect to that operand.
        """
        return self._arithmetic(
            name,
            tuple(
                _dual(
                    operand.value, Fraction(int(index == position)), operand.dependent
                )
                for index, operand in enumerate(operands)
            ),
        )

    def _calculate(self, name: str, operands: tuple[_Scalar, ...]) -> _ReverseNumber:
        """Record one edge per dependent operand occurrence.

        Parameters
        ----------
        name : str
            The resolved arithmetic primitive.
        operands : tuple[_Scalar, ...]
            The scalars from this execution.

        Returns
        -------
        _ReverseNumber
            The computed primal and its node if it depends on the inputs.
        """
        scalars = cast(tuple[_ReverseNumber, ...], operands)
        active = tuple(
            (position, operand.index)
            for position, operand in enumerate(scalars)
            if operand.index is not None
        )
        local = self._local(name, scalars, active[0][0] if active else None)
        if not local.dependent:
            return _ReverseNumber(local.value, None)
        edges = [(active[0][1], local.tangent)]
        edges.extend(
            (parent, self._local(name, scalars, position).tangent)
            for position, parent in active[1:]
        )
        index = len(self.parents)
        self.parents.append(tuple(edges))
        return _ReverseNumber(local.value, index)

    def gradient(self, output: _ReverseNumber, arity: int) -> tuple[Fraction, ...]:
        """Accumulate every parent edge in reverse topological order.

        Parameters
        ----------
        output : _ReverseNumber
            The returned scalar of the evaluation.
        arity : int
            The number of input leaves, allocated in parameter order.

        Returns
        -------
        tuple[Fraction, ...]
            The exact output cotangent at every input leaf.
        """
        if output.index is None:
            return (Fraction(),) * arity
        cotangents = [Fraction()] * len(self.parents)
        cotangents[output.index] = Fraction(1)
        for index in range(len(self.parents) - 1, -1, -1):
            for parent, slope in self.parents[index]:
                cotangents[parent] += slope * cotangents[index]
                _bound(cotangents[parent])
        return tuple(cotangents[:arity])


def _fraction(value: int | Fraction) -> Fraction:
    """Read an explicitly exact input scalar.

    Parameters
    ----------
    value : int or Fraction
        A primal or input direction supplied by the caller.

    Returns
    -------
    Fraction
        The exact rational input.

    Raises
    ------
    ValueError
        If the supplied object is outside the exact input carrier.
    """
    if type(value) is not int and not isinstance(value, Fraction):
        raise ValueError("exact derivative inputs must be integers or Fractions")
    return Fraction(value)


def _check_action(
    module: QiecModule,
    owner: NamedComputation,
    rule: RationalDerivativeDeclaration,
    max_work: int,
    fuel: int,
) -> RationalDerivativeDecision:
    """Certify an action against the source owner of the requested evaluation.

    Parameters
    ----------
    module : QiecModule
        The checked module that supplies all the computations.
    owner : NamedComputation
        The actual requested primal computation.
    rule : RationalDerivativeDeclaration
        The nominal declaration to verify before use.
    max_work : int
        The symbolic certification work limit.
    fuel : int
        The reference machine step limit per computation.

    Returns
    -------
    RationalDerivativeDecision
        A fresh certificate over the actual checked source.

    Raises
    ------
    JvpUnsupported
        If the declaration names another primal or cannot be certified.
    JvpLimit
        If the certification exhausts a work or step limit.
    """
    from quivers.qiec.rational_laws import RationalLawLimit, RationalLawUnsupported

    if rule.primal != owner.id:
        raise JvpUnsupported("the declared derivative belongs to another computation")
    try:
        decision = rule.check(module, max_work=max_work, fuel=fuel)
    except RationalLawUnsupported as error:
        raise JvpUnsupported(str(error)) from error
    except RationalLawLimit as error:
        raise JvpLimit(str(error)) from error
    if not decision.certified:
        raise JvpUnsupported("the declared action differs from the exact derivative")
    return decision


def _scalar_value(
    module: QiecModule,
    name: str,
    arguments: tuple[int | Fraction, ...],
    fuel: int,
) -> Fraction:
    """Execute a certified scalar expression at its primal coordinates.

    Parameters
    ----------
    module : QiecModule
        The module that defines the named expression.
    name : str
        The primal or explicitly authored derivative computation.
    arguments : tuple[int or Fraction, ...]
        The exact inputs to that computation.
    fuel : int
        The reference step limit.

    Returns
    -------
    Fraction
        The exact returned real value with all perturbation seeds zero.
    """
    return _evaluate_dual(
        module, name, arguments, (0,) * len(arguments), None, fuel
    ).value


def exact_gradient(
    module: QiecModule,
    computation: str,
    arguments: tuple[int | Fraction, ...],
    *,
    fuel: int = 65_536,
    rule: RationalDerivativeDeclaration | None = None,
    max_rule_work: int = 65_536,
) -> ExactGradient:
    """Evaluate a checked scalar computation and its exact reverse gradient.

    The reference execution records scalar primitive occurrences. The
    reverse pass sums every use of a recorded value, including uses in
    several resumptions of one captured continuation. The admitted
    operations and comparisons are those of :func:`exact_jvp`, subject
    to the bounds on local slopes and accumulated cotangents.

    Parameters
    ----------
    module : QiecModule
        The checked QIEC module.
    computation : str
        A named, closed scalar real computation.
    arguments : tuple[int or Fraction, ...]
        The evaluation point in parameter order.
    fuel : int
        The maximum number of reference machine steps.
    rule : RationalDerivativeDeclaration or None
        If given, certify and execute the corresponding authored action.
    max_rule_work : int
        The symbolic work limit for certifying a supplied rule.

    Returns
    -------
    ExactGradient
        The scalar and its partial derivatives in parameter order.

    Raises
    ------
    JvpUnsupported
        If the computation or a reached operation has no rule.
    JvpBoundaryError
        If an input-dependent comparison is at equality.
    JvpLimit
        If the evaluation exhausts a bound.
    ValueError
        If the point or limit is malformed.
    """
    registry, declaration = _prepare_scalar(module, computation, fuel)
    parameters = declaration.parameters
    if len(arguments) != len(parameters):
        raise ValueError("arguments must match the computation's arity")
    if rule is not None:
        if rule.vjp is None:
            raise JvpUnsupported("the declaration has no reverse action")
        decision = _check_action(module, declaration, rule, max_rule_work, fuel)
        assert decision.vjp is not None
        value = _scalar_value(module, computation, arguments, fuel)
        gradient = tuple(
            _scalar_value(module, action.computation, (*arguments, 1), fuel)
            for action in decision.vjp
        )
        return ExactGradient(value, gradient)
    evaluator = _ReverseEvaluator(module, fuel)
    result = evaluator.run(declaration, registry, arguments, record=True)
    return ExactGradient(result.value, evaluator.gradient(result, len(parameters)))


def exact_implicit_gradient(
    module: QiecModule,
    residual: str,
    root: int | Fraction,
    arguments: tuple[int | Fraction, ...],
    *,
    fuel: int = 65_536,
) -> ExactImplicitGradient:
    """Differentiate the local solution of a scalar residual equation.

    The residual runs in the regular scalar fragment of
    :func:`exact_gradient`. Its first parameter is the unknown. At the
    supplied point, the residual must vanish exactly and its derivative
    in that first coordinate must be nonzero. The result differentiates
    the local branch through this root with respect to the remaining
    parameters.

    Parameters
    ----------
    module : QiecModule
        The checked module declaring the residual.
    residual : str
        The named real computation defining the scalar equation.
    root : int or Fraction
        The exact value to verify in its first parameter.
    arguments : tuple[int or Fraction, ...]
        Values of its other parameters in order.
    fuel : int
        The maximum number of reference machine steps.

    Returns
    -------
    ExactImplicitGradient
        The verified root, local derivatives and residual state derivative.

    Raises
    ------
    ImplicitRootError
        If the residual is nonzero or its first partial derivative vanishes.
    JvpUnsupported
        If the residual or a reached operation is unsupported.
    JvpBoundaryError
        If an input-dependent comparison is at equality.
    JvpLimit
        If the evaluation or exact gradient exhausts a bound.
    ValueError
        If the point or limit is malformed.
    """
    value = _fraction(root)
    evaluated = exact_gradient(module, residual, (value, *arguments), fuel=fuel)
    if evaluated.value != 0:
        raise ImplicitRootError("the supplied root does not make the residual zero")
    state_derivative, *other_partials = evaluated.gradient
    if state_derivative == 0:
        raise ImplicitRootError("the residual's state derivative is zero at the root")
    gradient = tuple(-partial / state_derivative for partial in other_partials)
    _bound(*gradient)
    return ExactImplicitGradient(value, gradient, state_derivative)


def exact_jvp(
    module: QiecModule,
    computation: str,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    *,
    fuel: int = 65_536,
    rule: RationalDerivativeDeclaration | None = None,
    max_rule_work: int = 65_536,
) -> ExactJvp:
    """Evaluate a named real computation and its exact forward derivative.

    The run follows the checked reference machine, including authored
    clauses and resumptions. The result uses rational arithmetic for the
    admitted primitives. An input-dependent comparison is admitted only
    when its operands are unequal at the supplied point.

    Parameters
    ----------
    module : QiecModule
        The module declaring the checked computation.
    computation : str
        The computation's name. Its arguments and result are scalar
        reals and its residual effect row is empty.
    arguments : tuple[int or Fraction, ...]
        The point at which to evaluate.
    direction : tuple[int or Fraction, ...]
        The input tangent, in the same parameter order.
    fuel : int
        The maximum number of reference machine steps.
    rule : RationalDerivativeDeclaration or None
        If given, certify and execute the corresponding authored action.
    max_rule_work : int
        The symbolic work limit for certifying a supplied rule.

    Returns
    -------
    ExactJvp
        The exact value and Jacobian-vector product.

    Raises
    ------
    JvpUnsupported
        If the checked computation or a reached operation has no rule.
    JvpBoundaryError
        If an input-dependent comparison is at equality.
    JvpLimit
        If the evaluation exhausts a bound.
    ValueError
        If the supplied arguments, direction, or limit is malformed.
    """
    if rule is not None:
        _, declaration = _prepare_scalar(module, computation, fuel)
        if len(arguments) != len(declaration.parameters) or len(direction) != len(
            arguments
        ):
            raise ValueError(
                "arguments and direction must each match the computation's arity"
            )
        if rule.jvp is None:
            raise JvpUnsupported("the declaration has no forward action")
        decision = _check_action(module, declaration, rule, max_rule_work, fuel)
        assert decision.jvp is not None
        value = _scalar_value(module, computation, arguments, fuel)
        tangent = _scalar_value(
            module, decision.jvp.computation, (*arguments, *direction), fuel
        )
        return ExactJvp(value, tangent)
    result = _evaluate_dual(module, computation, arguments, direction, None, fuel)
    return ExactJvp(result.value, result.tangent)


def exact_mixed_derivative(
    module: QiecModule,
    computation: str,
    arguments: tuple[int | Fraction, ...],
    first: tuple[int | Fraction, ...],
    second: tuple[int | Fraction, ...],
    *,
    fuel: int = 65_536,
) -> ExactMixedDerivative:
    """Evaluate a scalar's first derivatives and mixed second derivative.

    The first and second input directions seed separate perturbation
    variables. Each variable squares to zero and their product gives
    the mixed derivative. The run requires the same regularity and
    exact arithmetic as :func:`exact_jvp`.

    Parameters
    ----------
    module : QiecModule
        The checked QIEC module.
    computation : str
        A named, closed scalar real computation.
    arguments : tuple[int or Fraction, ...]
        The point where both derivatives are evaluated.
    first : tuple[int or Fraction, ...]
        The first input direction.
    second : tuple[int or Fraction, ...]
        The second input direction.
    fuel : int
        The maximum number of reference machine steps.

    Returns
    -------
    ExactMixedDerivative
        The scalar value, both first directional derivatives, and the
        Hessian's bilinear action on the given directions.

    Raises
    ------
    JvpUnsupported
        If the checked computation or a reached operation has no rule.
    JvpBoundaryError
        If an input-dependent comparison is at equality.
    JvpLimit
        If the evaluation exhausts a bound.
    ValueError
        If a point, direction, or limit is malformed.
    """
    result = _evaluate_dual(module, computation, arguments, first, second, fuel)
    return ExactMixedDerivative(
        result.value, result.tangent, result.second_tangent, result.mixed
    )


def _evaluate_dual(
    module: QiecModule,
    computation: str,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    second_direction: tuple[int | Fraction, ...] | None,
    fuel: int,
) -> _Dual:
    """Run the checked computation with seeded perturbation coefficients.

    Parameters
    ----------
    module : QiecModule
        The module declaring the computation.
    computation : str
        The computation's name.
    arguments : tuple[int or Fraction, ...]
        The primal values of its parameters.
    direction : tuple[int or Fraction, ...]
        The first perturbation coefficients.
    second_direction : tuple[int or Fraction, ...] or None
        The second coefficients, or no second seed for a JVP.
    fuel : int
        The maximum reference step count.

    Returns
    -------
    _Dual
        The scalar coefficients computed by the reference machine.

    Raises
    ------
    JvpUnsupported
        If the computation or a reached operation has no rule.
    JvpLimit
        If the evaluation exhausts a bound.
    ValueError
        If an input or limit is malformed.
    """
    registry, declaration = _prepare_scalar(module, computation, fuel)
    parameters = declaration.parameters
    second = (0,) * len(parameters) if second_direction is None else second_direction
    if any(len(vector) != len(parameters) for vector in (arguments, direction, second)):
        raise ValueError(
            "arguments and direction must each match the computation's arity"
        )
    environment = {
        parameter: _dual(
            _fraction(value), _fraction(tangent), True, second_tangent=_fraction(other)
        )
        for parameter, value, tangent, other in zip(
            parameters, arguments, direction, second, strict=True
        )
    }
    evaluator = _ScalarDerivativeEvaluator(module=module, fuel=fuel)
    result = _run_derivative(evaluator, declaration, registry, environment)
    assert isinstance(result, _Dual)
    return result


def _prepare_scalar(
    module: QiecModule, computation: str, fuel: int
) -> tuple[KernelRegistry, NamedComputation]:
    """Resolve a checked declaration in the shared derivative fragment.

    Parameters
    ----------
    module : QiecModule
        The module declaring the computation.
    computation : str
        The computation's name.
    fuel : int
        The reference machine step bound.

    Returns
    -------
    tuple[KernelRegistry, NamedComputation]
        The checked constraints and resolved declaration.

    Raises
    ------
    JvpUnsupported
        If the declaration is absent or outside the scalar fragment.
    ValueError
        If the step bound is malformed.
    """
    if type(fuel) is not int or fuel < 1:
        raise ValueError("fuel must be a positive integer")
    registry = validate_module(module)
    declaration = next(
        (item for item in module.computations if item.name == computation), None
    )
    if declaration is None:
        raise JvpUnsupported(f"unknown derivative computation `{computation}`")
    row = declaration.type.effects
    if (
        declaration.telescope
        or row.entries
        or row.tail is not None
        or declaration.type.result != REAL
        or any(parameter.type != REAL for parameter in declaration.parameters)
    ):
        raise JvpUnsupported(
            "exact derivatives require a closed scalar real computation"
        )
    return registry, declaration


def _run_derivative(
    evaluator: _ScalarDerivativeEvaluator,
    declaration: NamedComputation,
    registry: KernelRegistry,
    environment: Mapping[Local, object],
) -> object:
    """Run the selected scalar interpretation of a checked computation.

    Parameters
    ----------
    evaluator : _ScalarDerivativeEvaluator
        The active scalar algebra.
    declaration : NamedComputation
        The selected computation.
    registry : KernelRegistry
        Its validated module registry.
    environment : Mapping[Local, object]
        The seeded parameters of this run.

    Returns
    -------
    object
        The returned scalar represented in the active algebra.

    Raises
    ------
    JvpUnsupported
        If evaluation encounters an unsupported operation.
    JvpLimit
        If the reference machine exhausts a bound.
    """
    try:
        return evaluator.evaluate_validated(declaration.body, registry, environment)
    except FuelExhaustedError as error:
        raise JvpLimit(str(error)) from error
    except RecursionError as error:
        raise JvpLimit("exact derivative expression nesting limit exhausted") from error
    except EvaluationError as error:
        raise JvpUnsupported(str(error)) from error


__all__ = [
    "ExactGradient",
    "ExactImplicitGradient",
    "ExactJvp",
    "ExactMixedDerivative",
    "ImplicitRootError",
    "JvpBoundaryError",
    "JvpLimit",
    "JvpUnsupported",
    "exact_gradient",
    "exact_implicit_gradient",
    "exact_jvp",
    "exact_mixed_derivative",
]
