"""Exact polynomial laws for the scalar real fragment of QIEC.

The checker interprets the supported scalar operations in the polynomial
ring over the rationals. A QIEC real literal contributes its represented
rational value. For instance, ``1.0 / 3.0`` gives the exact coefficient
one third. Named continuations can expose their answers as shared scalar
parameters after their handlers have discharged.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from quivers.qiec.discharge import DischargedModule, discharge_module
from quivers.qiec.effects import HandlerDef
from quivers.qiec.identifiers import ComputationId, PrimitiveId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import NamedComputation, QiecModule, _children, validate_module
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    Computation,
    ConditionalValue,
    Handle,
    If,
    LetValue,
    LiteralValue,
    Local,
    PrimitiveApplication,
    Return,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, REAL, UNIT

type _Exponents = tuple[int, ...]
type _Polynomial = dict[_Exponents, Fraction]
type _Environment = dict[Local, _Polynomial]
type _Terms = tuple[tuple[_Exponents, Fraction], ...]

_ARITHMETIC = {
    PrimitiveId.derive("builtin", name): name
    for name in ("add_real", "sub_real", "mul_real", "div_real", "neg_real", "pow_real")
}
_COMPARISONS = {
    PrimitiveId.derive("builtin", f"{name}_real")
    for name in ("eq", "ne", "lt", "le", "gt", "ge")
}
_VALUES = tuple(item for item in Value.__value__.__args__ if isinstance(item, type))
_SCALARS = (Var, LiteralValue, PrimitiveApplication, ConditionalValue, LetValue)
_MAX_DEGREE = 1024
_MAX_COEFFICIENT_BITS = 16_384


class PolynomialLawUnsupported(ValueError):
    """The law is outside the exact polynomial fragment."""


class PolynomialLawLimit(ValueError):
    """Normalization or witness construction exceeded a resource limit."""


@dataclass(frozen=True, slots=True)
class PolynomialLawCounterexample:
    """An exact input at which the two polynomials differ.

    Parameters
    ----------
    arguments : tuple[int, ...]
        The integer arguments, in parameter order.
    left : Fraction
        The exact rational value on the left.
    right : Fraction
        The exact rational value on the right.
    """

    arguments: tuple[int, ...]
    left: Fraction
    right: Fraction


@dataclass(frozen=True, slots=True)
class PolynomialLawDecision:
    """Normalized polynomial terms and a witness if they differ.

    Parameters
    ----------
    parameters : tuple[str, ...]
        The left computation's parameter names, in position order.
    left : tuple[tuple[tuple[int, ...], Fraction], ...]
        The left monomials with exact coefficients, sorted by exponents.
    right : tuple[tuple[tuple[int, ...], Fraction], ...]
        The corresponding terms on the right.
    counterexample : PolynomialLawCounterexample or None
        A separating input, or no witness when the normal forms agree.
    """

    parameters: tuple[str, ...]
    left: _Terms
    right: _Terms
    counterexample: PolynomialLawCounterexample | None = None


def _scalar(declaration: NamedComputation) -> None:
    """Restrict a declaration to the scalar real, empty-row interface.

    Parameters
    ----------
    declaration : NamedComputation
        A law root or a callee.

    Raises
    ------
    PolynomialLawUnsupported
        If the computation requires static arguments, has residual
        effects, or takes or returns a nonscalar value.
    """
    row = declaration.type.effects
    if (
        declaration.telescope
        or row.entries
        or row.tail is not None
        or declaration.type.result != REAL
        or any(parameter.type != REAL for parameter in declaration.parameters)
    ):
        raise PolynomialLawUnsupported(
            f"`{declaration.name}` is outside the closed scalar real fragment"
        )


def _condition(value: Value) -> bool:
    """Read a literal branch condition.

    Parameters
    ----------
    value : Value
        The condition of a conditional term.

    Returns
    -------
    bool
        The literal Boolean.

    Raises
    ------
    PolynomialLawUnsupported
        If the branch selection depends on runtime input.
    """
    if (
        isinstance(value, LiteralValue)
        and value.type == BOOL
        and type(value.value) is bool
    ):
        return value.value
    raise PolynomialLawUnsupported(
        "a polynomial law requires literal branch conditions"
    )


def _rational(value: LiteralValue) -> Fraction | None:
    """Interpret the represented value of a finite real literal.

    Parameters
    ----------
    value : LiteralValue
        A checked literal of any type.

    Returns
    -------
    Fraction or None
        The rational real value if the literal represents one.
    """
    data = value.value
    if (
        value.type == REAL
        and isinstance(data, (int, float))
        and not isinstance(data, bool)
        and (isinstance(data, int) or isfinite(data))
    ):
        return Fraction(data)
    return None


class _PolynomialRing:
    """Sparse exact arithmetic for a fixed list of polynomial variables.

    Parameters
    ----------
    width : int
        The number of scalar variables.
    max_work : int
        The normalization and witness work budget.
    """

    def __init__(self, width: int, max_work: int) -> None:
        self.zero = (0,) * width
        self.remaining = max_work

    def spend(self, amount: int = 1) -> None:
        """Charge a step or generated terms against the work budget.

        Parameters
        ----------
        amount : int
            How many units of work this step takes.

        Raises
        ------
        PolynomialLawLimit
            If the remaining budget cannot cover the work.
        """
        self.remaining -= amount
        if self.remaining < 0:
            raise PolynomialLawLimit("polynomial law work limit exhausted")

    def accumulate(
        self, result: _Polynomial, powers: _Exponents, coefficient: Fraction
    ) -> None:
        """Add one term and delete the coefficient if it vanishes.

        Parameters
        ----------
        result : _Polynomial
            The polynomial under construction.
        powers : _Exponents
            The monomial's exponents in parameter order.
        coefficient : Fraction
            Its signed contribution.

        Raises
        ------
        PolynomialLawLimit
            If the total degree or rational coefficient exceeds the limit.
        """
        value = result.get(powers, Fraction()) + coefficient
        if value:
            if sum(powers) > _MAX_DEGREE:
                raise PolynomialLawLimit("polynomial law degree limit exceeded")
            if (
                max(value.numerator.bit_length(), value.denominator.bit_length())
                > _MAX_COEFFICIENT_BITS
            ):
                raise PolynomialLawLimit("polynomial law coefficient limit exceeded")
            result[powers] = value
        else:
            result.pop(powers, None)

    def add(self, left: _Polynomial, right: _Polynomial, sign: int = 1) -> _Polynomial:
        """Add or subtract two normal forms.

        Parameters
        ----------
        left : _Polynomial
            The first summand.
        right : _Polynomial
            The second summand.
        sign : int
            One to add, minus one to subtract.

        Returns
        -------
        _Polynomial
            Their normal form after signed addition.
        """
        self.spend(len(left) + len(right))
        result = dict(left)
        for powers, coefficient in right.items():
            self.accumulate(result, powers, sign * coefficient)
        return result

    def scale(self, polynomial: _Polynomial, coefficient: Fraction) -> _Polynomial:
        """Multiply a normal form by a rational constant.

        Parameters
        ----------
        polynomial : _Polynomial
            The polynomial to scale.
        coefficient : Fraction
            The constant multiplier.

        Returns
        -------
        _Polynomial
            The scaled normal form.
        """
        self.spend(len(polynomial))
        result: _Polynomial = {}
        for powers, value in polynomial.items():
            self.accumulate(result, powers, value * coefficient)
        return result

    def multiply(self, left: _Polynomial, right: _Polynomial) -> _Polynomial:
        """Multiply monomials and combine coefficients of equal products.

        Parameters
        ----------
        left : _Polynomial
            The left factor.
        right : _Polynomial
            The right factor.

        Returns
        -------
        _Polynomial
            Their product in normal form.
        """
        self.spend(len(left) * len(right))
        result: _Polynomial = {}
        for lp, lc in left.items():
            for rp, rc in right.items():
                powers = tuple(a + b for a, b in zip(lp, rp, strict=True))
                self.accumulate(result, powers, lc * rc)
        return result

    def power(self, base: _Polynomial, exponent: int) -> _Polynomial:
        """Raise a polynomial to a nonnegative integral exponent.

        Parameters
        ----------
        base : _Polynomial
            The base.
        exponent : int
            The power.

        Returns
        -------
        _Polynomial
            The normalized power, including one at exponent zero.

        Raises
        ------
        PolynomialLawLimit
            If the exponent itself exceeds the degree limit.
        """
        if exponent > _MAX_DEGREE:
            raise PolynomialLawLimit("polynomial law exponent limit exceeded")
        result: _Polynomial = {self.zero: Fraction(1)}
        while exponent:
            if exponent % 2:
                result = self.multiply(result, base)
            exponent //= 2
            if exponent:
                base = self.multiply(base, base)
        return result

    def substitute(
        self, polynomial: _Polynomial, coordinate: int, value: int
    ) -> _Polynomial:
        """Substitute one integer at a chosen coordinate.

        Parameters
        ----------
        polynomial : _Polynomial
            The normal form.
        coordinate : int
            The variable's position.
        value : int
            Its assigned value.

        Returns
        -------
        _Polynomial
            A normal form with zero exponents at the chosen coordinate.
        """
        self.spend(len(polynomial))
        result: _Polynomial = {}
        for powers, coefficient in polynomial.items():
            exponent = powers[coordinate]
            reduced = (*powers[:coordinate], 0, *powers[coordinate + 1 :])
            self.accumulate(result, reduced, coefficient * value**exponent)
        return result

    def counterexample(
        self, left: _Polynomial, right: _Polynomial
    ) -> PolynomialLawCounterexample | None:
        """Separate unequal normal forms with an exact integer assignment.

        Parameters
        ----------
        left : _Polynomial
            The left normal form.
        right : _Polynomial
            The right normal form.

        Returns
        -------
        PolynomialLawCounterexample or None
            A separating assignment if their difference is nonzero.

        Raises
        ------
        AssertionError
            If a nonzero polynomial violates the root bound.
        """
        if left == right:
            return None
        difference = self.add(left, right, -1)
        arguments: list[int] = []
        for coordinate in range(len(self.zero)):
            degree = max(powers[coordinate] for powers in difference)
            for value in range(degree + 1):
                reduced = self.substitute(difference, coordinate, value)
                if reduced:
                    arguments.append(value)
                    difference = reduced
                    break
            else:  # pragma: no cover - root bound over the fraction field
                raise AssertionError(
                    "nonzero polynomial vanished on its separating grid"
                )
        for coordinate, value in enumerate(arguments):
            left = self.substitute(left, coordinate, value)
            right = self.substitute(right, coordinate, value)
        return PolynomialLawCounterexample(
            tuple(arguments),
            left.get(self.zero, Fraction()),
            right.get(self.zero, Fraction()),
        )


class _Normalizer(_PolynomialRing):
    """Normalize two computations in one polynomial ring.

    Parameters
    ----------
    discharged : DischargedModule
        The module after supported authored handlers are discharged.
    width : int
        The number of scalar variables.
    max_work : int
        The normalization and witness work budget.
    """

    def __init__(self, discharged: DischargedModule, width: int, max_work: int) -> None:
        super().__init__(width, max_work)
        self.discharged = discharged
        self.computations = {item.id: item for item in discharged.module.computations}

    def check_source(
        self,
        module: QiecModule,
        left: str,
        right: str,
    ) -> None:
        """Check source values before relying on discharge to erase unused ones.

        Parameters
        ----------
        module : QiecModule
            The checked module before discharge.
        left : str
            The first law root's name.
        right : str
            The second law root's name.

        Raises
        ------
        PolynomialLawUnsupported
            If either root can reach source outside the exact scalar fragment.
        """
        computations = {item.id: item for item in module.computations}
        handlers = {item.id: item for item in module.handlers}
        pending: list[object] = [
            item for item in module.computations if item.name in (left, right)
        ]
        seen: set[int] = set()
        while pending:
            term = pending.pop()
            if id(term) in seen:
                continue
            seen.add(id(term))
            self.spend()
            if isinstance(term, NamedComputation):
                if term.telescope:
                    raise PolynomialLawUnsupported(
                        f"`{term.name}` has static parameters"
                    )
                pending.append(term.body)
            elif isinstance(term, HandlerDef):
                if term.telescope or term.implementation != "authored":
                    raise PolynomialLawUnsupported(
                        f"`{term.name}` is outside the authored polynomial fragment"
                    )
                pending.extend(
                    clause.body for clause in term.clauses if clause.body is not None
                )
                if term.return_clause is not None:
                    pending.append(term.return_clause.body)
            else:
                self.check_source_term(term)
                pending.extend(_children(term))
                if isinstance(term, Call):
                    pending.append(computations[term.callee])
                elif isinstance(term, Handle):
                    pending.append(handlers[term.handler])

    def check_source_term(self, term: object) -> None:
        """Verify that discharge can erase a value without changing its domain.

        Parameters
        ----------
        term : object
            The kernel node or value being visited.

        Raises
        ------
        PolynomialLawUnsupported
            If a term may be undefined or lies outside exact scalar arithmetic.
        """
        if isinstance(term, Case) or (
            isinstance(term, _VALUES) and not isinstance(term, _SCALARS)
        ):
            raise PolynomialLawUnsupported(
                f"{type(term).__name__} is outside polynomial laws"
            )
        if isinstance(term, Local) and term.type not in (BOOL, UNIT, REAL):
            raise PolynomialLawUnsupported(
                f"`{term.name}` is outside the scalar polynomial fragment"
            )
        if isinstance(term, LiteralValue) and not (
            term.type in (BOOL, UNIT) or _rational(term) is not None
        ):
            raise PolynomialLawUnsupported("a polynomial literal must be finite")
        if not isinstance(term, PrimitiveApplication):
            return
        if term.result_type == BOOL and term.primitive in _COMPARISONS:
            return
        name = _ARITHMETIC.get(term.primitive)
        if term.result_type != REAL or name is None:
            raise PolynomialLawUnsupported(
                f"primitive `{term.name}` is outside polynomial laws"
            )
        if name not in ("div_real", "pow_real"):
            return
        # Discharge may remove an unread operand. Its domain must be checked
        # with the original binding structure still present.
        try:
            constant = self.value(term.arguments[1], {}).get(self.zero, Fraction())
        except KeyError:
            constant = None
        if name == "div_real":
            if not constant:
                raise PolynomialLawUnsupported(
                    "division requires a nonzero constant divisor"
                )
        elif constant is None or constant < 0 or constant.denominator != 1:
            raise PolynomialLawUnsupported(
                "power requires a nonnegative integral constant exponent"
            )

    def value(self, value: Value, environment: _Environment) -> _Polynomial:
        """Interpret a scalar value in the rational polynomial ring.

        Parameters
        ----------
        value : Value
            The checked value term.
        environment : _Environment
            The surrounding scalar bindings.

        Returns
        -------
        _Polynomial
            The value's normal form.

        Raises
        ------
        PolynomialLawUnsupported
            If the term or primitive is outside the polynomial fragment.
        """
        self.spend()
        if isinstance(value, Var):
            return environment[value.local]
        if isinstance(value, LiteralValue) and (number := _rational(value)) is not None:
            result: _Polynomial = {}
            self.accumulate(result, self.zero, number)
            return result
        if isinstance(value, LetValue):
            scope = dict(environment)
            for binding in value.bindings:
                scope[binding.binder] = self.value(binding.value, scope)
            return self.value(value.body, scope)
        if isinstance(value, ConditionalValue):
            source = value.then if _condition(value.condition) else value.otherwise
            return self.value(source, environment)
        if isinstance(value, PrimitiveApplication):
            return self.primitive(value, environment)
        raise PolynomialLawUnsupported(
            f"{type(value).__name__} is outside polynomial laws"
        )

    def primitive(
        self, term: PrimitiveApplication, environment: _Environment
    ) -> _Polynomial:
        """Interpret a registered polynomial operation.

        Parameters
        ----------
        term : PrimitiveApplication
            The checked nominal primitive.
        environment : _Environment
            The scalar bindings of its arguments.

        Returns
        -------
        _Polynomial
            The normalized result.

        Raises
        ------
        PolynomialLawUnsupported
            If the primitive is not polynomial on its checked domain.
        """
        name = _ARITHMETIC.get(term.primitive)
        if name is None or term.result_type != REAL:
            raise PolynomialLawUnsupported(
                f"primitive `{term.name}` is outside polynomial laws"
            )
        arguments = tuple(self.value(item, environment) for item in term.arguments)
        if name == "neg_real":
            return self.scale(arguments[0], Fraction(-1))
        left, right = arguments
        if name == "add_real":
            return self.add(left, right)
        if name == "sub_real":
            return self.add(left, right, -1)
        if name == "mul_real":
            return self.multiply(left, right)
        constant = right.get(self.zero, Fraction())
        nonconstant = any(powers != self.zero for powers in right)
        if name == "div_real":
            if nonconstant or not constant:
                raise PolynomialLawUnsupported(
                    "division requires a nonzero constant divisor"
                )
            return self.scale(left, 1 / constant)
        if nonconstant or constant < 0 or constant.denominator != 1:
            raise PolynomialLawUnsupported(
                "power requires a nonnegative integral constant exponent"
            )
        return self.power(left, int(constant))

    def body(
        self,
        term: Computation,
        environment: _Environment,
        active: frozenset[ComputationId],
    ) -> _Polynomial:
        """Interpret the pure computation that remains after discharge.

        Parameters
        ----------
        term : Computation
            The current computation.
        environment : _Environment
            Its scalar bindings.
        active : frozenset[ComputationId]
            The calls that have yet to return.

        Returns
        -------
        _Polynomial
            The normalized scalar result.

        Raises
        ------
        PolynomialLawUnsupported
            If a residual computation still needs an effect or a
            nonpolynomial operation.
        """
        scope = environment
        while True:
            self.spend()
            if isinstance(term, Bind):
                result = self.body(term.first, scope, active)
                scope = {**scope, term.binder: result}
                term = term.then
            elif isinstance(term, If):
                term = term.then if _condition(term.condition) else term.otherwise
            elif isinstance(term, Return):
                return self.value(term.value, scope)
            elif isinstance(term, Call):
                arguments = tuple(self.value(item, scope) for item in term.arguments)
                return self.call(self.computations[term.callee], arguments, active)
            else:
                raise PolynomialLawUnsupported(
                    f"{type(term).__name__} remains after polynomial law discharge"
                )

    def call(
        self,
        declaration: NamedComputation,
        arguments: tuple[_Polynomial, ...],
        active: frozenset[ComputationId],
    ) -> _Polynomial:
        """Instantiate a scalar callee with symbolic arguments.

        Parameters
        ----------
        declaration : NamedComputation
            The callee after discharge.
        arguments : tuple[_Polynomial, ...]
            The call's normalized arguments.
        active : frozenset[ComputationId]
            The computations active in the caller.

        Returns
        -------
        _Polynomial
            The callee's normalized result.

        Raises
        ------
        PolynomialLawUnsupported
            If a handler remains, the callee is outside the scalar
            fragment, or the call would re-enter it recursively.
        """
        _scalar(declaration)
        obstacle = self.discharged.obstacle(declaration.name)
        if obstacle is not None:
            raise PolynomialLawUnsupported(f"`{declaration.name}` {obstacle.reason}")
        if declaration.id in active:
            raise PolynomialLawUnsupported(
                f"recursive polynomial call to `{declaration.name}`"
            )
        environment = dict(zip(declaration.parameters, arguments, strict=True))
        return self.body(declaration.body, environment, active | {declaration.id})


def check_polynomial_law(
    module: QiecModule,
    left: str,
    right: str,
    *,
    max_work: int = 65_536,
) -> PolynomialLawDecision:
    """Check a scalar polynomial identity between two QIEC computations.

    Parameters
    ----------
    module : QiecModule
        The module with the checked computations.
    left : str
        The left computation's name.
    right : str
        The right computation's name. Both sides must have equally many
        scalar real arguments, a scalar real result, and empty residual
        effect rows.
    max_work : int
        The budget for visiting terms, expanding monomials, and finding
        a separating assignment.

    Returns
    -------
    PolynomialLawDecision
        Exact rational normal forms and a witness if they differ.

    Raises
    ------
    PolynomialLawUnsupported
        If either side leaves the scalar polynomial fragment.
    PolynomialLawLimit
        If normalization or witness construction exhausts a limit.
    ValueError
        If the work limit is not a positive integer.
    """
    _validate_limit("max_work", max_work)
    validate_module(module)
    discharged = discharge_module(module)
    named = {item.name: item for item in discharged.module.computations}
    if left not in named or right not in named:
        missing = left if left not in named else right
        raise PolynomialLawUnsupported(
            f"unknown polynomial law computation `{missing}`"
        )
    lhs, rhs = named[left], named[right]
    _scalar(lhs)
    _scalar(rhs)
    width = len(lhs.parameters)
    if width != len(rhs.parameters):
        raise PolynomialLawUnsupported(
            "a polynomial law requires matching positional signatures"
        )
    interpreter = _Normalizer(discharged, width, max_work)
    interpreter.spend(width * width)
    arguments = tuple(
        {tuple(int(i == j) for i in range(width)): Fraction(1)} for j in range(width)
    )
    try:
        interpreter.check_source(module, left, right)
        first = interpreter.call(lhs, arguments, frozenset())
        second = interpreter.call(rhs, arguments, frozenset())
        witness = interpreter.counterexample(first, second)
    except RecursionError as error:
        raise PolynomialLawLimit("polynomial law nesting limit exhausted") from error
    return PolynomialLawDecision(
        tuple(parameter.name for parameter in lhs.parameters),
        tuple(sorted(first.items())),
        tuple(sorted(second.items())),
        witness,
    )


__all__ = [
    "PolynomialLawCounterexample",
    "PolynomialLawDecision",
    "PolynomialLawLimit",
    "PolynomialLawUnsupported",
    "check_polynomial_law",
]
