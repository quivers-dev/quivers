"""Check authored scalar derivative rules by exact rational identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from quivers.qiec.checking import KernelRegistry
from quivers.qiec.derivative_rule_schema import RationalDerivativeDeclaration
from quivers.qiec.evaluator import EvaluationError, FuelExhaustedError
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.polynomial_laws import (
    PolynomialLawCounterexample,
    PolynomialLawLimit,
    PolynomialLawUnsupported,
    _Exponents,
    _Polynomial,
    _PolynomialRing,
    _Terms,
    _scalar,
)
from quivers.qiec.rational_laws import (
    RationalLawLimit,
    RationalLawUnsupported,
    _Fraction,
    _RationalEvaluator,
    _value_at,
)


@dataclass(frozen=True, slots=True)
class RationalDerivativeComponent:
    """A checked equality for an authored derivative computation.

    Parameters
    ----------
    computation : str
        The authored computation's name.
    difference_numerator : tuple[tuple[tuple[int, ...], Fraction], ...]
        The numerator of the authored result minus the formal derivative
        action. Monomial exponents are in the authored parameter order.
    counterexample : PolynomialLawCounterexample or None
        An exact argument vector in the primal domain at which the
        authored value (left) and formal derivative action (right) differ.
    """

    computation: str
    difference_numerator: _Terms
    counterexample: PolynomialLawCounterexample | None


@dataclass(frozen=True, slots=True)
class RationalDerivativeDecision:
    """Derivative rule checks on one strictly evaluated primal domain.

    Parameters
    ----------
    primal : str
        The primal computation's name.
    parameters : tuple[str, ...]
        Its input parameters in positional order.
    domain_polynomial : tuple[tuple[tuple[int, ...], Fraction], ...]
        The primal inputs where this polynomial is nonzero are the
        domain of the symbolic reference execution.
    jvp : RationalDerivativeComponent or None
        The result for the supplied forward rule, if present.
    vjp : tuple[RationalDerivativeComponent, ...] or None
        The supplied reverse component results in primal parameter order.
    """

    primal: str
    parameters: tuple[str, ...]
    domain_polynomial: _Terms
    jvp: RationalDerivativeComponent | None
    vjp: tuple[RationalDerivativeComponent, ...] | None

    @property
    def certified(self) -> bool:
        """Whether all supplied components implement the exact derivatives.

        Returns
        -------
        bool
            Whether every checked difference is identically zero.
        """
        forward = self.jvp is None or not self.jvp.difference_numerator
        return forward and all(not item.difference_numerator for item in self.vjp or ())


def _partial(
    ring: _PolynomialRing, polynomial: _Polynomial, coordinate: int
) -> _Polynomial:
    """Apply a formal partial derivative to a sparse polynomial.

    Parameters
    ----------
    ring : _PolynomialRing
        The common exact ring and work budget.
    polynomial : _Polynomial
        The polynomial to differentiate.
    coordinate : int
        The differentiated variable's position.

    Returns
    -------
    _Polynomial
        Its formal derivative in the same ring.
    """
    ring.spend(len(polynomial))
    result: _Polynomial = {}
    for powers, coefficient in polynomial.items():
        power = powers[coordinate]
        if power:
            reduced = (*powers[:coordinate], power - 1, *powers[coordinate + 1 :])
            ring.accumulate(result, reduced, coefficient * power)
    return result


def _monomial_order(powers: _Exponents) -> tuple[int, _Exponents]:
    """Order monomials by total degree and then lexicographically.

    Parameters
    ----------
    powers : _Exponents
        The monomial exponents.

    Returns
    -------
    tuple[int, _Exponents]
        The monomial's position in the chosen order.
    """
    return sum(powers), powers


def _remainder(
    ring: _PolynomialRing, polynomial: _Polynomial, divisor: _Polynomial
) -> _Polynomial:
    """Reduce a polynomial by one nonzero divisor in graded lexicographic order.

    Parameters
    ----------
    ring : _PolynomialRing
        The common exact ring and work budget.
    polynomial : _Polynomial
        The expression to reduce.
    divisor : _Polynomial
        The nonzero reduction divisor.

    Returns
    -------
    _Polynomial
        The reduced remainder, zero exactly when the divisor divides it.
    """
    ring.spend(len(divisor) + len(polynomial))
    working = dict(polynomial)
    result: _Polynomial = {}
    leading = max(divisor, key=_monomial_order)
    while working:
        ring.spend(len(working))
        powers = max(working, key=_monomial_order)
        quotient = tuple(a - b for a, b in zip(powers, leading, strict=True))
        if min(quotient, default=0) < 0:
            result[powers] = working.pop(powers)
            continue
        coefficient = working[powers] / divisor[leading]
        ring.spend(len(divisor))
        for term, factor in divisor.items():
            target = tuple(a + b for a, b in zip(term, quotient, strict=True))
            ring.accumulate(working, target, -coefficient * factor)
    return result


def _covers(ring: _PolynomialRing, source: _Polynomial, rule: _Polynomial) -> bool:
    """Prove the source's nonzero guard implies the rule's nonzero guard.

    Parameters
    ----------
    ring : _PolynomialRing
        The common exact ring and work budget.
    source : _Polynomial
        The product of the primal's strict nonzero guards.
    rule : _Polynomial
        The product of the rule's strict nonzero guards.

    Returns
    -------
    bool
        Whether the rule guard divides a power of the source guard.
    """
    exponent = max(map(sum, rule))
    if exponent == 0 or source == rule:
        return True
    # Work modulo the rule guard so high powers do not have to be expanded.
    power = _remainder(ring, source, rule)
    remainder: _Polynomial = {ring.zero: Fraction(1)}
    while exponent:
        if exponent % 2:
            remainder = _remainder(ring, ring.multiply(remainder, power), rule)
            if not remainder:
                return True
        exponent //= 2
        if exponent:
            power = _remainder(ring, ring.multiply(power, power), rule)
    return False


class _RuleChecker:
    """Compare authored derivative actions in a common guarded rational ring.

    Parameters
    ----------
    module : QiecModule
        The checked module containing the primal and authored rules.
    arity : int
        The number of primal input coordinates.
    max_work : int
        The total symbolic arithmetic and witness budget.
    fuel : int
        The reference machine step budget for each computation.
    """

    def __init__(
        self, module: QiecModule, arity: int, max_work: int, fuel: int
    ) -> None:
        self.registry: KernelRegistry = validate_module(module)
        width = arity + max(arity, 1)
        self.ring = _PolynomialRing(width, max_work)
        self.evaluator = _RationalEvaluator(module, self.ring, fuel)
        self.ring.spend(width * width)
        self.symbols: tuple[_Polynomial, ...] = tuple(
            {tuple(int(i == j) for i in range(width)): Fraction(1)}
            for j in range(width)
        )
        self.values = tuple(
            self.evaluator.fraction(symbol, self.evaluator.one)
            for symbol in self.symbols
        )

    def evaluate(self, declaration: NamedComputation) -> tuple[_Fraction, _Polynomial]:
        """Run one checked scalar computation and collect strict domain guards.

        Parameters
        ----------
        declaration : NamedComputation
            The computation in the common positional coordinate system.

        Returns
        -------
        tuple[_Fraction, _Polynomial]
            Its guarded quotient and domain polynomial.
        """
        environment = dict(
            zip(
                declaration.parameters,
                self.values[: len(declaration.parameters)],
                strict=True,
            )
        )
        result = self.evaluator.evaluate_validated(
            declaration.body, self.registry, environment
        )
        assert isinstance(result, _Fraction)
        return result, self.evaluator.domain()

    def compare(
        self,
        declaration: NamedComputation,
        formal: _Fraction,
        source_domain: _Polynomial,
    ) -> RationalDerivativeComponent:
        """Check definedness and equality against one formal action.

        Parameters
        ----------
        declaration : NamedComputation
            The authored scalar derivative rule.
        formal : _Fraction
            The formal result derived from the primal.
        source_domain : _Polynomial
            The strict guard of the primal expression.

        Returns
        -------
        RationalDerivativeComponent
            The exact equality decision on the primal domain.

        Raises
        ------
        RationalLawUnsupported
            If the rule cannot be shown to be defined throughout that domain.
        """
        authored, rule_domain = self.evaluate(declaration)
        if not _covers(self.ring, source_domain, rule_domain):
            raise RationalLawUnsupported(
                f"could not establish derivative rule `{declaration.name}` on the primal domain"
            )
        difference = self.ring.add(
            self.ring.multiply(dict(authored.numerator), dict(formal.denominator)),
            self.ring.multiply(dict(formal.numerator), dict(authored.denominator)),
            -1,
        )
        width = len(declaration.parameters)
        separating = self.ring.counterexample(
            self.ring.multiply(difference, source_domain), {}
        )
        witness = None
        if separating is not None:
            point = separating.arguments
            witness = PolynomialLawCounterexample(
                point[:width],
                _value_at(self.ring, authored, point),
                _value_at(self.ring, formal, point),
            )
        terms = tuple(
            sorted((powers[:width], value) for powers, value in difference.items())
        )
        return RationalDerivativeComponent(declaration.name, terms, witness)

    def expected(
        self,
        primal: _Fraction,
        arity: int,
        forward: NamedComputation | None,
        reverse: tuple[NamedComputation, ...] | None,
        domain: _Polynomial,
    ) -> tuple[
        RationalDerivativeComponent | None,
        tuple[RationalDerivativeComponent, ...] | None,
    ]:
        """Derive an exact JVP and each output of its transposed linear map.

        Parameters
        ----------
        primal : _Fraction
            The strictly evaluated primal quotient.
        arity : int
            The number of primal input coordinates.
        forward : NamedComputation or None
            The supplied JVP, if present.
        reverse : tuple[NamedComputation, ...] or None
            The supplied VJP components, if present.
        domain : _Polynomial
            The shared primal domain guard.

        Returns
        -------
        tuple
            The verification outcomes for the two supplied rules.
        """
        p, q = dict(primal.numerator), dict(primal.denominator)
        ring = self.ring
        denominator = ring.multiply(q, q)
        partials = tuple(
            ring.add(
                ring.multiply(_partial(ring, p, coordinate), q),
                ring.multiply(p, _partial(ring, q, coordinate)),
                -1,
            )
            for coordinate in range(arity)
        )
        jvp = None
        if forward is not None:
            numerator: _Polynomial = {}
            for partial, direction in zip(
                partials, self.symbols[arity : 2 * arity], strict=True
            ):
                numerator = ring.add(numerator, ring.multiply(partial, direction))
            jvp = self.compare(
                forward, self.evaluator.fraction(numerator, denominator), domain
            )
        vjp = None
        if reverse is not None:
            seed = self.symbols[arity]
            vjp = tuple(
                self.compare(
                    declaration,
                    self.evaluator.fraction(ring.multiply(partial, seed), denominator),
                    domain,
                )
                for declaration, partial in zip(reverse, partials, strict=True)
            )
        return jvp, vjp


def _rule(
    declarations: Mapping[str, NamedComputation], name: str, arity: int
) -> NamedComputation:
    """Require a closed scalar rule of the declared positional arity.

    Parameters
    ----------
    declarations : Mapping[str, NamedComputation]
        The module's named checked computations.
    name : str
        The supplied derivative rule name.
    arity : int
        The expected number of scalar real parameters.

    Returns
    -------
    NamedComputation
        The selected derivative computation.

    Raises
    ------
    RationalLawUnsupported
        If the name or the rule's arity is wrong.
    """
    result = declarations.get(name)
    if result is None:
        raise RationalLawUnsupported(f"unknown derivative rule computation `{name}`")
    _scalar(result)
    if len(result.parameters) != arity:
        raise RationalLawUnsupported(
            f"derivative rule `{name}` requires {arity} positional parameters"
        )
    return result


def check_rational_derivative_rule(
    module: QiecModule,
    primal: str | RationalDerivativeDeclaration,
    *,
    jvp: str | None = None,
    vjp: tuple[str, ...] | None = None,
    max_work: int = 65_536,
    fuel: int = 65_536,
) -> RationalDerivativeDecision:
    """Certify authored derivative actions of a checked scalar computation.

    A JVP takes the n primal coordinates followed by their n tangents.
    A VJP has one scalar computation per primal coordinate. Each takes
    the primal coordinates followed by an output cotangent. A certificate
    establishes agreement with the formal derivative for every such seed
    on the primal's domain. Thus each action is linear in its seed and
    both actions, if supplied, are transposes at each primal point.

    Parameters
    ----------
    module : QiecModule
        The module declaring the primal and the supplied rules.
    primal : str or RationalDerivativeDeclaration
        A named computation with scalar real inputs and result and an
        empty residual effect row. A declaration supplies the primal
        and actions by their nominal IDs in this module.
    jvp : str or None
        A named scalar computation implementing the forward rule.
    vjp : tuple[str, ...] or None
        Named scalar computations implementing the reverse action in
        primal parameter order.
    max_work : int
        The exact arithmetic and counterexample-search budget.
    fuel : int
        The reference machine step budget per computation.

    Returns
    -------
    RationalDerivativeDecision
        The verified primal domain and component checks. A false identity
        carries an exact separating input in the domain.

    Raises
    ------
    RationalLawUnsupported
        If a computation is outside the rational fragment, or a rule's
        domain cannot be shown to cover the primal domain for every seed.
    RationalLawLimit
        If arithmetic or reference evaluation exhausts a limit.
    ValueError
        If no rule was specified, a declaration was combined with
        separately named actions, or a limit is invalid.
    """
    _validate_limit("max_work", max_work)
    _validate_limit("fuel", fuel)
    if isinstance(primal, RationalDerivativeDeclaration):
        if jvp is not None or vjp is not None:
            raise ValueError("a derivative declaration already specifies its actions")
        names = {item.id: item.name for item in module.computations}
        try:
            jvp = None if primal.jvp is None else names[primal.jvp]
            vjp = (
                None
                if primal.vjp is None
                else tuple(names[rule] for rule in primal.vjp)
            )
            primal = names[primal.primal]
        except KeyError as error:
            raise RationalLawUnsupported(
                "a rational derivative declaration names an unknown computation"
            ) from error
    if jvp is None and vjp is None:
        raise ValueError("at least one derivative rule is required")
    declarations = {item.name: item for item in module.computations}
    owner = declarations.get(primal)
    if owner is None:
        raise RationalLawUnsupported(f"unknown primal computation `{primal}`")
    arity = len(owner.parameters)
    try:
        _scalar(owner)
        forward = None if jvp is None else _rule(declarations, jvp, 2 * arity)
        reverse = None
        if vjp is not None:
            if len(vjp) != arity:
                raise RationalLawUnsupported(
                    "a VJP requires one rule per primal parameter"
                )
            reverse = tuple(_rule(declarations, name, arity + 1) for name in vjp)
        checker = _RuleChecker(module, arity, max_work, fuel)
        value, domain = checker.evaluate(owner)
        checked_jvp, checked_vjp = checker.expected(
            value, arity, forward, reverse, domain
        )
    except (FuelExhaustedError, PolynomialLawLimit) as error:
        raise RationalLawLimit(str(error)) from error
    except RecursionError as error:
        raise RationalLawLimit(
            "rational derivative rule nesting limit exhausted"
        ) from error
    except (EvaluationError, PolynomialLawUnsupported) as error:
        raise RationalLawUnsupported(str(error)) from error
    return RationalDerivativeDecision(
        primal,
        tuple(parameter.name for parameter in owner.parameters),
        tuple(sorted((powers[:arity], value) for powers, value in domain.items())),
        checked_jvp,
        checked_vjp,
    )


__all__ = [
    "RationalDerivativeComponent",
    "RationalDerivativeDecision",
    "check_rational_derivative_rule",
]
