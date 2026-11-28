"""Exact identities of partial scalar rational functions computed by QIEC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from operator import eq, ge, gt, le, lt, ne

from quivers.qiec.evaluator import EvaluationError, Evaluator, FuelExhaustedError
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.polynomial_laws import (
    _ARITHMETIC,
    _COMPARISONS,
    PolynomialLawCounterexample,
    PolynomialLawLimit,
    PolynomialLawUnsupported,
    _Polynomial,
    _PolynomialRing,
    _Terms,
    _rational,
    _scalar,
)
from quivers.qiec.terms import ConditionalValue, LetValue, LiteralValue, Local
from quivers.qiec.terms import PrimitiveApplication, Value, Var
from quivers.qiec.types import BOOL, REAL, UNIT

_COMPARE = {
    "eq_real": eq,
    "ge_real": ge,
    "gt_real": gt,
    "le_real": le,
    "lt_real": lt,
    "ne_real": ne,
}


class RationalLawUnsupported(ValueError):
    """The checker cannot decide a law in the supported rational fragment."""


class RationalLawLimit(ValueError):
    """Exact rational evaluation exhausted a resource limit."""


@dataclass(frozen=True, slots=True)
class RationalLawDecision:
    """A comparison on an exactly identified common rational domain.

    Parameters
    ----------
    parameters : tuple[str, ...]
        The left computation's inputs, in position order.
    domain_polynomial : tuple[tuple[tuple[int, ...], Fraction], ...]
        The domain consists of inputs at which this polynomial is nonzero.
    difference_numerator : tuple[tuple[tuple[int, ...], Fraction], ...]
        The numerator after writing the difference over a common
        denominator. The empty tuple is the zero polynomial.
    counterexample : PolynomialLawCounterexample or None
        A separating integer input in the common domain with exact values,
        or no witness if both sides agree throughout the domain.
    """

    parameters: tuple[str, ...]
    domain_polynomial: _Terms
    difference_numerator: _Terms
    counterexample: PolynomialLawCounterexample | None


@dataclass(frozen=True, slots=True)
class _Fraction:
    """A pair of immutable exact polynomials.

    Parameters
    ----------
    numerator : _Terms
        The numerator's nonzero monomials.
    denominator : _Terms
        The denominator's nonzero monomials.
    """

    numerator: _Terms
    denominator: _Terms


class _RationalEvaluator(Evaluator):
    """Run checked control and handlers with symbolic scalar arithmetic.

    Parameters
    ----------
    module : QiecModule
        The checked module whose calls and handlers are available.
    ring : _PolynomialRing
        Shared arithmetic and work budget for both law roots.
    fuel : int
        The reference step budget per side.
    """

    def __init__(self, module: QiecModule, ring: _PolynomialRing, fuel: int) -> None:
        super().__init__(module=module, fuel=fuel)
        self.ring = ring
        self.one: _Polynomial = {ring.zero: Fraction(1)}
        self.guards: set[_Terms] = set()

    def fraction(self, numerator: _Polynomial, denominator: _Polynomial) -> _Fraction:
        """Freeze a quotient so that a continuation may share it.

        Parameters
        ----------
        numerator : _Polynomial
            The numerator, which can be zero.
        denominator : _Polynomial
            The nonzero denominator.

        Returns
        -------
        _Fraction
            The immutable quotient.
        """
        self.ring.spend(len(numerator) + len(denominator))
        return _Fraction(
            tuple(sorted(numerator.items())),
            tuple(sorted((denominator if numerator else self.one).items())),
        )

    def _duplicable_value(self, value: object) -> bool:
        """Allow an unrestricted resumption to share immutable quotients.

        Parameters
        ----------
        value : object
            The captured reference value.

        Returns
        -------
        bool
            Whether the capture can copy the value.
        """
        return isinstance(value, _Fraction) or super()._duplicable_value(value)

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Read a scalar term without approximating its arithmetic.

        Parameters
        ----------
        value : Value
            The checked source value.
        environment : Mapping[Local, object]
            The surrounding exact scalar values.

        Returns
        -------
        object
            The exact quotient or the reference Boolean or unit value.

        Raises
        ------
        RationalLawUnsupported
            If the evaluated term is outside the scalar rational fragment.
        """
        self.ring.spend()
        if isinstance(value, PrimitiveApplication):
            return self.primitive(value, environment)
        if isinstance(value, LiteralValue):
            if (number := _rational(value)) is not None:
                constant: _Polynomial = {}
                self.ring.accumulate(constant, self.ring.zero, number)
                return self.fraction(constant, self.one)
            if value.type in (BOOL, UNIT):
                return super()._value(value, environment)
        if isinstance(value, (Var, LetValue, ConditionalValue)):
            return super()._value(value, environment)
        raise RationalLawUnsupported(
            f"{type(value).__name__} is outside exact scalar rational laws"
        )

    def primitive(
        self, term: PrimitiveApplication, environment: Mapping[Local, object]
    ) -> object:
        """Interpret a checked nominal arithmetic primitive or comparison.

        Parameters
        ----------
        term : PrimitiveApplication
            The primitive application.
        environment : Mapping[Local, object]
            The symbolic values its arguments may read.

        Returns
        -------
        object
            The exact quotient or a Boolean constant.

        Raises
        ------
        RationalLawUnsupported
            If the primitive has no admitted exact rule, uses a zero
            divisor, or compares input-dependent results.
        """
        name = _ARITHMETIC.get(term.primitive)
        comparison = term.result_type == BOOL and term.primitive in _COMPARISONS
        if not comparison and (name is None or term.result_type != REAL):
            raise RationalLawUnsupported(
                f"primitive `{term.name}` is outside exact scalar rational laws"
            )
        arguments = tuple(self._value(item, environment) for item in term.arguments)
        left = arguments[0]
        assert isinstance(left, _Fraction)
        num, den = dict(left.numerator), dict(left.denominator)
        if name == "neg_real":
            return self.fraction(self.ring.scale(num, Fraction(-1)), den)
        _, right = arguments
        assert isinstance(right, _Fraction)
        if comparison:
            first, second = self.constant(left), self.constant(right)
            if first is None or second is None:
                raise RationalLawUnsupported(
                    "rational law branch comparisons must be constant"
                )
            return self.numbers.number("truth", _COMPARE[term.name](first, second))
        other_num, other_den = dict(right.numerator), dict(right.denominator)
        if name == "pow_real":
            exponent = self.constant(right)
            if exponent is None or exponent.denominator != 1:
                raise RationalLawUnsupported(
                    "rational law powers require an integral constant exponent"
                )
            if exponent < 0:
                self.guard(num)
                num, den = den, num
            return self.fraction(
                self.ring.power(num, abs(exponent.numerator)),
                self.ring.power(den, abs(exponent.numerator)),
            )
        if name == "div_real":
            self.guard(other_num)
            other_num, other_den = other_den, other_num
        if name in ("div_real", "mul_real"):
            return self.fraction(
                self.ring.multiply(num, other_num), self.ring.multiply(den, other_den)
            )
        sign = -1 if name == "sub_real" else 1
        if den == other_den:
            return self.fraction(self.ring.add(num, other_num, sign), den)
        return self.fraction(
            self.ring.add(
                self.ring.multiply(num, other_den),
                self.ring.multiply(other_num, den),
                sign,
            ),
            self.ring.multiply(den, other_den),
        )

    def constant(self, value: _Fraction) -> Fraction | None:
        """Read a quotient constant throughout its admitted domain.

        Parameters
        ----------
        value : _Fraction
            The exact quotient.

        Returns
        -------
        Fraction or None
            The constant if the monomial maps are proportional.
        """
        if not value.numerator:
            return Fraction()
        num, den = dict(value.numerator), dict(value.denominator)
        index = value.denominator[-1][0]
        scalar = num.get(index, Fraction()) / den[index]
        return scalar if self.ring.scale(den, scalar) == num else None

    def guard(self, polynomial: _Polynomial) -> None:
        """Record that an evaluated divisor must be nonzero.

        Parameters
        ----------
        polynomial : _Polynomial
            A numerator used to check a rational divisor.

        Raises
        ------
        RationalLawUnsupported
            If the domain is empty because the divisor is identically zero.
        """
        if not polynomial:
            raise RationalLawUnsupported("rational law divisor is identically zero")
        if set(polynomial) != {self.ring.zero}:
            leading = polynomial[max(polynomial)]
            monic = self.ring.scale(polynomial, 1 / leading)
            self.guards.add(tuple(sorted(monic.items())))

    def domain(self) -> _Polynomial:
        """Gather and clear the current run's domain guards.

        Returns
        -------
        _Polynomial
            A product whose nonzero set is the run's domain.
        """
        result = self.one
        for guard in sorted(self.guards):
            result = self.ring.multiply(result, dict(guard))
        self.guards.clear()
        return result


def _value_at(
    ring: _PolynomialRing, value: _Fraction, arguments: tuple[int, ...]
) -> Fraction:
    """Evaluate a rational form at a checked point in its domain.

    Parameters
    ----------
    ring : _PolynomialRing
        The arithmetic interpreter.
    value : _Fraction
        The exact quotient.
    arguments : tuple[int, ...]
        An assignment known to satisfy the domain guards.

    Returns
    -------
    Fraction
        The exact rational value.
    """
    numerator, denominator = dict(value.numerator), dict(value.denominator)
    for position, coordinate in enumerate(arguments):
        numerator = ring.substitute(numerator, position, coordinate)
        denominator = ring.substitute(denominator, position, coordinate)
    return numerator.get(ring.zero, Fraction()) / denominator[ring.zero]


def check_rational_law(
    module: QiecModule,
    left: str,
    right: str,
    *,
    max_work: int = 65_536,
    fuel: int = 65_536,
) -> RationalLawDecision:
    """Compare two checked scalar computations on a common rational domain.

    The checked reference evaluator preserves the order and strictness of
    calls, authored clauses and resumptions. Each side accumulates exact
    divisor guards even for unused intermediate results. A decision
    requires equal normalized products of monic nonzero guards.

    Parameters
    ----------
    module : QiecModule
        The module declaring both computations.
    left : str
        The first computation's name.
    right : str
        The second computation's name. The computations have matching
        scalar real signatures and empty residual rows.
    max_work : int
        The budget for exact arithmetic and finding a witness.
    fuel : int
        The reference step budget per side.

    Returns
    -------
    RationalLawDecision
        The shared domain and the difference numerator, with an exact
        witness in the domain if the values can differ.

    Raises
    ------
    RationalLawUnsupported
        If a computation or domain is outside the admitted exact fragment.
    RationalLawLimit
        If arithmetic or the reference evaluator exhausts a limit.
    ValueError
        If a limit is not a positive integer.
    """
    _validate_limit("max_work", max_work)
    _validate_limit("fuel", fuel)
    registry = validate_module(module)
    named = {item.name: item for item in module.computations}
    if left not in named or right not in named:
        unknown = left if left not in named else right
        raise RationalLawUnsupported(f"unknown rational law computation `{unknown}`")
    lhs, rhs = named[left], named[right]
    width = len(lhs.parameters)
    ring = _PolynomialRing(width, max_work)
    evaluator = _RationalEvaluator(module, ring, fuel)
    try:
        _scalar(lhs)
        _scalar(rhs)
        if width != len(rhs.parameters):
            raise RationalLawUnsupported(
                "a rational law requires matching positional signatures"
            )
        ring.spend(width * width)
        symbolic = tuple(
            evaluator.fraction(
                {tuple(int(i == j) for i in range(width)): Fraction(1)}, evaluator.one
            )
            for j in range(width)
        )
        first = evaluator.evaluate_validated(
            lhs.body, registry, dict(zip(lhs.parameters, symbolic, strict=True))
        )
        domain = evaluator.domain()
        second = evaluator.evaluate_validated(
            rhs.body, registry, dict(zip(rhs.parameters, symbolic, strict=True))
        )
        if domain != evaluator.domain():
            raise RationalLawUnsupported(
                "could not establish equal rational law domains"
            )
        assert isinstance(first, _Fraction) and isinstance(second, _Fraction)
        difference = ring.add(
            ring.multiply(dict(first.numerator), dict(second.denominator)),
            ring.multiply(dict(second.numerator), dict(first.denominator)),
            -1,
        )
        witness = ring.counterexample(ring.multiply(difference, domain), {})
        if witness is not None:
            arguments = witness.arguments
            witness = PolynomialLawCounterexample(
                arguments,
                _value_at(ring, first, arguments),
                _value_at(ring, second, arguments),
            )
    except (FuelExhaustedError, PolynomialLawLimit) as error:
        raise RationalLawLimit(str(error)) from error
    except RecursionError as error:
        raise RationalLawLimit("rational law nesting limit exhausted") from error
    except (EvaluationError, PolynomialLawUnsupported) as error:
        raise RationalLawUnsupported(str(error)) from error
    return RationalLawDecision(
        tuple(parameter.name for parameter in lhs.parameters),
        tuple(sorted(domain.items())),
        tuple(sorted(difference.items())),
        witness,
    )


__all__ = [
    "RationalLawDecision",
    "RationalLawLimit",
    "RationalLawUnsupported",
    "check_rational_law",
]
