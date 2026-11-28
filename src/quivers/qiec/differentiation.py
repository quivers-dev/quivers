"""Exact forward derivatives on regular scalar executions of checked QIEC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from operator import eq, ge, gt, le, lt, ne

from quivers.qiec.evaluator import EvaluationError, Evaluator, FuelExhaustedError
from quivers.qiec.module import QiecModule, validate_module
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


class JvpUnsupported(ValueError):
    """A checked computation lies outside the exact scalar derivative fragment."""


class JvpBoundaryError(JvpUnsupported):
    """Comparison stability could not be certified at an equality boundary."""


class JvpLimit(ValueError):
    """Exact forward evaluation exhausted a resource bound."""


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
class _Dual:
    """An immutable primal and tangent in the rational arithmetic fragment.

    Parameters
    ----------
    value : Fraction
        The scalar primal.
    tangent : Fraction
        Its propagated directional derivative.
    dependent : bool
        Whether the expression depends syntactically on any input. A
        zero tangent in one direction does not establish independence.
    """

    value: Fraction
    tangent: Fraction
    dependent: bool


def _dual(value: Fraction, tangent: Fraction, dependent: bool) -> _Dual:
    """Construct a dual number under the coefficient bound.

    Parameters
    ----------
    value : Fraction
        The primal.
    tangent : Fraction
        The directional derivative.
    dependent : bool
        Whether either operand can depend on the inputs.

    Returns
    -------
    _Dual
        The exact immutable pair and its dependence flag.

    Raises
    ------
    JvpLimit
        If the primal or tangent exceeds the coefficient limit.
    """
    if (
        max(
            value.numerator.bit_length(),
            value.denominator.bit_length(),
            tangent.numerator.bit_length(),
            tangent.denominator.bit_length(),
        )
        > _MAX_COEFFICIENT_BITS
    ):
        raise JvpLimit("exact JVP coefficient limit exceeded")
    return _Dual(value, tangent, dependent)


class _ForwardEvaluator(Evaluator):
    """Interpret scalar values in the checked reference machine as exact duals."""

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
        return isinstance(value, _Dual) or super()._duplicable_value(value)

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
                return _dual(Fraction(data), Fraction(), False)
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
        first = arguments[0]
        assert isinstance(first, _Dual)
        if name == "neg_real":
            return _dual(-first.value, -first.tangent, first.dependent)
        _, second = arguments
        assert isinstance(second, _Dual)
        dependent = first.dependent or second.dependent
        if compare is not None:
            if dependent and first.value == second.value:
                raise JvpBoundaryError(
                    f"comparison `{term.name}` is at an input-dependent equality boundary"
                )
            return self.numbers.number("truth", compare(first.value, second.value))
        if name == "add_real":
            return _dual(
                first.value + second.value, first.tangent + second.tangent, dependent
            )
        if name == "sub_real":
            return _dual(
                first.value - second.value, first.tangent - second.tangent, dependent
            )
        if name == "mul_real":
            return _dual(
                first.value * second.value,
                first.tangent * second.value + first.value * second.tangent,
                dependent,
            )
        if name == "div_real":
            if not second.value:
                raise JvpUnsupported(
                    "division by zero is outside the derivative domain"
                )
            quotient = first.value / second.value
            return _dual(
                quotient,
                (first.tangent - quotient * second.tangent) / second.value,
                dependent,
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
            The primal and tangent.

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
        return _dual(
            base.value**power,
            power * base.value ** (power - 1) * base.tangent,
            base.dependent,
        )


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
        raise ValueError("exact JVP inputs must be integers or Fractions")
    return Fraction(value)


def exact_jvp(
    module: QiecModule,
    computation: str,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    *,
    fuel: int = 65_536,
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
    if type(fuel) is not int or fuel < 1:
        raise ValueError("fuel must be a positive integer")
    registry = validate_module(module)
    declaration = next(
        (item for item in module.computations if item.name == computation), None
    )
    if declaration is None:
        raise JvpUnsupported(f"unknown derivative computation `{computation}`")
    row = declaration.type.effects
    parameters = declaration.parameters
    if (
        declaration.telescope
        or row.entries
        or row.tail is not None
        or declaration.type.result != REAL
        or any(parameter.type != REAL for parameter in parameters)
    ):
        raise JvpUnsupported("an exact JVP needs a closed scalar real computation")
    if len(arguments) != len(parameters) or len(direction) != len(parameters):
        raise ValueError(
            "arguments and direction must each match the computation's arity"
        )
    environment = {
        parameter: _dual(_fraction(value), _fraction(tangent), True)
        for parameter, value, tangent in zip(
            parameters, arguments, direction, strict=True
        )
    }
    evaluator = _ForwardEvaluator(module=module, fuel=fuel)
    try:
        result = evaluator.evaluate_validated(declaration.body, registry, environment)
    except FuelExhaustedError as error:
        raise JvpLimit(str(error)) from error
    except RecursionError as error:
        raise JvpLimit("exact JVP expression nesting limit exhausted") from error
    except EvaluationError as error:
        raise JvpUnsupported(str(error)) from error
    assert isinstance(result, _Dual)
    return ExactJvp(result.value, result.tangent)


__all__ = ["ExactJvp", "JvpBoundaryError", "JvpLimit", "JvpUnsupported", "exact_jvp"]
