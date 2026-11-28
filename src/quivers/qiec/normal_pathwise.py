"""Locally dominated scalar Normal pathwise derivatives and exact moments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import prod

from quivers._values import HeldValue
from quivers.qiec._random_site_graph import (
    _Address,
    _check_first_order_graph,
    _random_computation,
)
from quivers.qiec.builtins import RANDOM, RANDOM_SAMPLE
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import (
    ExactJvp,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    _ARITHMETIC,
    _BOOLEANS,
    _COMPARISONS,
    _MAX_POWER,
    _ScalarDerivativeEvaluator,
    _bound,
    _Dual,
    _fraction,
)
from quivers.qiec.effects import EffectRequest
from quivers.qiec.evaluator import (
    Evaluator,
    FuelExhaustedError,
    RuntimeRequest,
    _Continue,
    _Frame,
    _Returned,
)
from quivers.qiec.families import FAMILIES
from quivers.qiec.identifiers import ComputationId, EffectInstanceId, SiteId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import QiecModule
from quivers.qiec.polynomial_laws import (
    PolynomialLawLimit,
    _Polynomial,
    _PolynomialRing,
)
from quivers.qiec.scope import Scope
from quivers.qiec.terms import (
    DistributionValue,
    Local,
    PrimitiveApplication,
    SiteValue,
    Value,
)
from quivers.qiec.types import BOOL, REAL

type _Terms = tuple[tuple[tuple[int, ...], Fraction], ...]


@dataclass(frozen=True, slots=True)
class NormalVisit:
    """The checked occurrence of one scalar location-scale draw.

    Parameters
    ----------
    site : SiteId
        The static site carrying the explicit reparameterization policy.
    address : tuple
        Its full request address in the reference machine.
    scale : Fraction
        The strictly positive, noise-independent scale.
    scale_tangent : Fraction
        The directional derivative of that scale.
    """

    site: SiteId
    address: _Address
    scale: Fraction
    scale_tangent: Fraction


@dataclass(frozen=True, slots=True)
class NormalPathwisePlan:
    """A checked scalar Normal derivative represented by noise polynomials.

    Parameters
    ----------
    computation : ComputationId
        The named source entry.
    exact : ExactJvp
        The exact expected return and its directional derivative.
    visits : tuple[NormalVisit, ...]
        The addressed draws, in order of their independent noise inputs.
    primal, tangent : _Terms
        Sparse polynomials in those standard-normal inputs, with one
        exponent per draw in every term.
    source_steps : int
        The measured reference steps used to build the plan.
    polynomial_work : int
        The sparse arithmetic and moment work charged during construction.
    """

    computation: ComputationId
    exact: ExactJvp
    visits: tuple[NormalVisit, ...]
    primal: _Terms
    tangent: _Terms
    source_steps: int
    polynomial_work: int

    def at_noise(self, noise: tuple[int | Fraction, ...]) -> ExactJvp:
        """Evaluate the pathwise action at given base-noise coordinates.

        Parameters
        ----------
        noise : tuple[int or Fraction, ...]
            One exact noise coordinate per visit, in source order.

        Returns
        -------
        ExactJvp
            The return and local directional derivative for those fixed
            coordinates. The plan's `exact` field integrates over standard
            normals instead.

        Raises
        ------
        ValueError
            If the noise vector has the wrong size or invalid coordinates.
        JvpLimit
            If exact pointwise output exceeds the coefficient bound.
        """
        if len(noise) != len(self.visits):
            raise ValueError("noise coordinates must match the checked Normal visits")
        coordinates = tuple(map(_fraction, noise))
        _bound(*coordinates)
        return ExactJvp(
            _evaluate_terms(self.primal, coordinates),
            _evaluate_terms(self.tangent, coordinates),
        )


def _evaluate_terms(terms: _Terms, coordinates: tuple[Fraction, ...]) -> Fraction:
    """Read a polynomial at fixed exact noise coordinates.

    Parameters
    ----------
    terms : _Terms
        The sorted sparse monomials and coefficients.
    coordinates : tuple[Fraction, ...]
        The base-noise point.

    Returns
    -------
    Fraction
        The exact value at that point.
    """
    result = Fraction()
    for powers, coefficient in terms:
        result += coefficient * prod(
            point**power for point, power in zip(coordinates, powers, strict=True)
        )
        _bound(result)
    return result


@dataclass(frozen=True, slots=True)
class _NoiseDual:
    """One scalar in the exact noise polynomial interpretation.

    Parameters
    ----------
    primal, tangent : _Polynomial
        Its polynomial value and directional derivative.
    noisy : bool
        Whether it syntactically depends on a sampled noise. Evaluating
        its coefficients at one parameter point can erase that dependence.
    parameter : bool
        Whether it syntactically depends on a source parameter.
    """

    primal: _Polynomial
    tangent: _Polynomial
    noisy: bool
    parameter: bool


@dataclass(frozen=True, slots=True)
class _NormalOperands:
    """The exact Normal parameters in their written scope.

    Parameters
    ----------
    location, scale : _NoiseDual
        The local location and scale before the operation is answered.
    """

    location: _NoiseDual
    scale: _NoiseDual


def _normal_arguments(value: DistributionValue) -> tuple[Value, Value]:
    """Require the nominal unplated location-scale constructor.

    Parameters
    ----------
    value : DistributionValue
        The checked distribution term.

    Returns
    -------
    tuple[Value, Value]
        The location and scale operands in parameter order.

    Raises
    ------
    JvpUnsupported
        If the distribution constructor leaves the fragment.
    """
    if (
        value.family != FAMILIES["Normal"].id
        or value.name != "Normal"
        or not value.plate.empty
        or tuple(name for name, _ in value.arguments) != ("loc", "scale")
    ):
        raise JvpUnsupported("pathwise estimation requires unplated Normal(loc, scale)")
    return value.arguments[0][1], value.arguments[1][1]


def _check_request(request: EffectRequest, instance: EffectInstanceId) -> None:
    """Check a direct real Normal request in the selected row.

    Parameters
    ----------
    request : EffectRequest
        The checked source request.
    instance : EffectInstanceId
        The only admitted nominal Random instance.

    Raises
    ------
    JvpUnsupported
        If the operation uses different control, type or constructor.
    """
    if (
        request.instance != instance
        or request.effect != RANDOM
        or request.operation != RANDOM_SAMPLE
        or request.result_type != REAL
        or request.static_arguments != (REAL,)
    ):
        raise JvpUnsupported(
            "pathwise estimation requires Random.sample[Real] on one instance"
        )
    site, sampleable = request.arguments
    if not isinstance(site, SiteValue) or not isinstance(sampleable, DistributionValue):
        raise JvpUnsupported(
            "pathwise estimation requires a direct site and Normal constructor"
        )
    _normal_arguments(sampleable)


class _NormalEvaluator(_ScalarDerivativeEvaluator):
    """Execute the source once with exact polynomial-valued real scalars.

    Parameters
    ----------
    module : QiecModule
        The checked module of the reference evaluator.
    fuel : int
        Maximum reference steps in the source run.
    ring : _PolynomialRing
        The sparse ring of standard-normal noise coordinates.
    methods : frozenset[SiteId]
        The static sites explicitly licensed for reparameterization.
    """

    def __init__(
        self,
        module: QiecModule,
        fuel: int,
        ring: _PolynomialRing,
        methods: frozenset[SiteId],
    ) -> None:
        super().__init__(module=module, fuel=fuel)
        self.ring = ring
        self.methods = methods
        self.visits: list[NormalVisit] = []
        self.seen: set[_Address] = set()

    def input(
        self, value: Fraction, tangent: Fraction, *, parameter: bool
    ) -> _NoiseDual:
        """Insert a noise-independent scalar and tangent.

        Parameters
        ----------
        value, tangent : Fraction
            The exact current value and directional derivative.
        parameter : bool
            Whether this input carries source parameter dependence.

        Returns
        -------
        _NoiseDual
            The corresponding two constant polynomials.
        """
        _bound(value, tangent)
        zero = self.ring.zero
        return _NoiseDual(
            {zero: value} if value else {},
            {zero: tangent} if tangent else {},
            False,
            parameter,
        )

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Interpret the checked expressions over the noise ring.

        Parameters
        ----------
        value : Value
            The source expression.
        environment : Mapping[Local, object]
            Its exact lexical runtime scope.

        Returns
        -------
        object
            A noise scalar or one of the reference machine's other values.
        """
        if isinstance(value, PrimitiveApplication):
            if value.primitive in _BOOLEANS and value.result_type == BOOL:
                return Evaluator._value(self, value, environment)
            return self.polynomial_primitive(value, environment)
        if isinstance(value, SiteValue):
            return value.label
        if isinstance(value, DistributionValue):
            loc, scale = map(
                lambda operand: self._value(operand, environment),
                _normal_arguments(value),
            )
            assert isinstance(loc, _NoiseDual) and isinstance(scale, _NoiseDual)
            return _NormalOperands(loc, scale)
        actual = super()._value(value, environment)
        if isinstance(actual, _Dual):
            return self.input(actual.value, actual.tangent, parameter=actual.dependent)
        return actual

    def polynomial_primitive(
        self, term: PrimitiveApplication, environment: Mapping[Local, object]
    ) -> object:
        """Apply a nominal arithmetic rule or check a stable comparison.

        Parameters
        ----------
        term : PrimitiveApplication
            The checked nominal primitive.
        environment : Mapping[Local, object]
            The term's exact runtime bindings.

        Returns
        -------
        object
            A polynomial coefficient pair or a stable Boolean value.

        Raises
        ------
        JvpUnsupported
            If a primitive or comparison does not admit this interpretation.
        JvpBoundaryError
            If parameter-dependent operands meet on a decision boundary.
        """
        name = _ARITHMETIC.get(term.primitive)
        compare = _COMPARISONS.get(term.primitive)
        if not (
            name is not None
            and term.result_type == REAL
            or compare is not None
            and term.result_type == BOOL
        ):
            raise JvpUnsupported(
                f"primitive `{term.name}` leaves the Normal pathwise fragment"
            )
        operands = tuple(
            self._value(operand, environment) for operand in term.arguments
        )
        left = operands[0]
        assert isinstance(left, _NoiseDual)
        if name == "neg_real":
            return _NoiseDual(
                self.ring.scale(left.primal, Fraction(-1)),
                self.ring.scale(left.tangent, Fraction(-1)),
                left.noisy,
                left.parameter,
            )
        _, right = operands
        assert isinstance(right, _NoiseDual)
        if compare is not None:
            if left.noisy or right.noisy:
                raise JvpUnsupported(
                    "a noise-dependent comparison cannot license a pathwise derivative"
                )
            a, b = self.constant(left.primal), self.constant(right.primal)
            if a == b and (left.parameter or right.parameter):
                raise JvpBoundaryError(
                    f"comparison `{term.name}` is at an input-dependent equality boundary"
                )
            return self.numbers.number("truth", compare(a, b))
        assert name is not None
        return self.arithmetic(name, left, right)

    def constant(self, polynomial: _Polynomial) -> Fraction:
        """Read the constant coefficient from a noise-independent expression.

        Parameters
        ----------
        polynomial : _Polynomial
            A polynomial already known to be constant in the noise.

        Returns
        -------
        Fraction
            The exact scalar coefficient, or zero when it is absent.
        """
        return polynomial.get(self.ring.zero, Fraction())

    def arithmetic(self, name: str, left: _NoiseDual, right: _NoiseDual) -> _NoiseDual:
        """Evaluate one binary arithmetic operation in the dual noise ring.

        Parameters
        ----------
        name : str
            The resolved arithmetic primitive.
        left, right : _NoiseDual
            The exact polynomial operands.

        Returns
        -------
        _NoiseDual
            Its primal and first directional polynomial.

        Raises
        ------
        JvpUnsupported
            If division uses a noise-dependent or zero divisor.
        """
        ring = self.ring
        a, da, b, db = left.primal, left.tangent, right.primal, right.tangent
        if name in ("add_real", "sub_real"):
            sign = 1 if name == "add_real" else -1
            primal = ring.add(a, b, sign)
            tangent = ring.add(da, db, sign)
        elif name == "mul_real":
            primal = ring.multiply(a, b)
            tangent = ring.add(ring.multiply(da, b), ring.multiply(a, db))
        elif name == "div_real":
            if right.noisy:
                raise JvpUnsupported(
                    "pathwise polynomial division requires a noise-independent divisor"
                )
            divisor = self.constant(b)
            if divisor == 0:
                raise JvpUnsupported(
                    "division by zero is outside the Normal pathwise domain"
                )
            primal = ring.scale(a, 1 / divisor)
            correction = ring.scale(primal, self.constant(db))
            tangent = ring.scale(ring.add(da, correction, -1), 1 / divisor)
        else:
            assert name == "pow_real"
            return self.polynomial_power(left, right)
        return _NoiseDual(
            primal,
            tangent,
            left.noisy or right.noisy,
            left.parameter or right.parameter,
        )

    def polynomial_power(self, base: _NoiseDual, exponent: _NoiseDual) -> _NoiseDual:
        """Raise a polynomial to a fixed integral exponent.

        Parameters
        ----------
        base : _NoiseDual
            The base and its directional derivative.
        exponent : _NoiseDual
            The exponent, which must be independent of noise and parameters.

        Returns
        -------
        _NoiseDual
            Its value and first directional derivative.

        Raises
        ------
        JvpUnsupported
            If the power is not fixed integral or has a noise-dependent or
            zero base at a negative exponent.
        JvpLimit
            If the integral power exceeds the resource bound.
        """
        power = self.constant(exponent.primal)
        if exponent.noisy or exponent.parameter or power.denominator != 1:
            raise JvpUnsupported(
                "a Normal pathwise power needs a fixed integral exponent"
            )
        n = power.numerator
        if abs(n) > _MAX_POWER:
            raise JvpLimit("Normal pathwise power exceeds its exponent bound")
        if n == 0:
            # Preserve syntactic dependence; it cannot certify a later decision.
            return _NoiseDual(
                {self.ring.zero: Fraction(1)}, {}, base.noisy, base.parameter
            )
        primal: _Polynomial
        if n < 0:
            if base.noisy:
                raise JvpUnsupported("a negative power needs a noise-independent base")
            constant = self.constant(base.primal)
            if constant == 0:
                raise JvpUnsupported("a negative power needs a nonzero base")
            value = constant**n
            _bound(value)
            primal = {self.ring.zero: value}
            tangent = self.ring.scale(base.tangent, n * constant ** (n - 1))
        else:
            primal = self.ring.power(base.primal, n)
            slope = self.ring.scale(self.ring.power(base.primal, n - 1), Fraction(n))
            tangent = self.ring.multiply(slope, base.tangent)
        return _NoiseDual(primal, tangent, base.noisy, base.parameter)

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: Scope[Local, HeldValue],
        stack: list[_Frame],
        search_index: int,
    ) -> _Continue:
        """Answer a checked site with a fresh independent noise variable.

        Parameters
        ----------
        request : RuntimeRequest
            The canonical addressed Normal request.
        environment : Scope[Local, HeldValue]
            The current runtime bindings kept by the continuation.
        stack : list[_Frame]
            The unmodified reference continuation.
        search_index : int
            The search origin for a handler in the ordinary machine.

        Returns
        -------
        _Continue
            The same continuation supplied with the reparameterized draw.

        Raises
        ------
        JvpUnsupported
            If a site or scale cannot be licensed.
        JvpLimit
            If a new independent noise coordinate exceeds the bound.
        """
        site, address = request.core.origin.static_site, request.address
        if site not in self.methods:
            raise JvpUnsupported(f"missing pathwise estimator for reached site {site}")
        if address in self.seen:
            raise JvpUnsupported(f"repeated Normal address along a trace {address!r}")
        _, operands = request.arguments
        assert isinstance(operands, _NormalOperands)
        scale = self.constant(operands.scale.primal)
        if operands.scale.noisy or scale <= 0:
            raise JvpUnsupported(
                "a Normal scale must be strictly positive and noise-independent"
            )
        scale_tangent = self.constant(operands.scale.tangent)
        position = len(self.visits)
        width = len(self.ring.zero)
        if position == width:
            raise JvpLimit(
                f"Normal pathwise evaluation exceeds {width} noise coordinates"
            )
        powers = tuple(int(index == position) for index in range(width))
        self.ring.spend()
        noise = {powers: Fraction(1)}
        loc = operands.location
        value = _NoiseDual(
            self.ring.add(loc.primal, self.ring.scale(noise, scale)),
            self.ring.add(loc.tangent, self.ring.scale(noise, scale_tangent)),
            True,
            loc.parameter or operands.scale.parameter,
        )
        self.seen.add(address)
        self.visits.append(NormalVisit(site, address, scale, scale_tangent))
        return _Continue(
            current=_Returned(value=value),
            environment=environment,
            resumption_path=request.resumption_path,
        )


def _gaussian_expectation(polynomial: _Polynomial, ring: _PolynomialRing) -> Fraction:
    """Integrate a noise polynomial against independent standard normals.

    Parameters
    ----------
    polynomial : _Polynomial
        The sparse noise polynomial.
    ring : _PolynomialRing
        The same resource meter used during compilation.

    Returns
    -------
    Fraction
        The exact integral from even one-dimensional Gaussian moments.
    """
    moments = {0: 1, 1: 0}
    exact = Fraction()
    for powers, coefficient in polynomial.items():
        ring.spend(len(powers))
        if any(power % 2 for power in powers):
            continue
        for power in sorted(set(powers) - moments.keys()):
            ring.spend(power // 2)
            moments[power] = prod(range(1, power, 2))
        exact += coefficient * prod(moments[power] for power in powers)
        _bound(exact)
    return exact


def _normal_path(
    computation: ComputationId,
    evaluator: _NormalEvaluator,
    result: _NoiseDual,
    work: int,
) -> NormalPathwisePlan:
    """Integrate one completed path and retain its pointwise action.

    Parameters
    ----------
    computation : ComputationId
        The original source entry.
    evaluator : _NormalEvaluator
        The reference machine with its final step and visit counts.
    result : _NoiseDual
        The actual scalar return of that path.
    work : int
        The polynomial budget available before the replay began.

    Returns
    -------
    NormalPathwisePlan
        The visited request order, exact polynomials and their integrals.
    """
    exact = ExactJvp(
        _gaussian_expectation(result.primal, evaluator.ring),
        _gaussian_expectation(result.tangent, evaluator.ring),
    )
    width = len(evaluator.visits)
    return NormalPathwisePlan(
        computation,
        exact,
        tuple(evaluator.visits),
        tuple(
            (powers[:width], value) for powers, value in sorted(result.primal.items())
        ),
        tuple(
            (powers[:width], value) for powers, value in sorted(result.tangent.items())
        ),
        evaluator._steps,
        work - evaluator.ring.remaining,
    )


def plan_normal_pathwise(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    max_sites: int = 16,
    max_work: int = 65_536,
    fuel: int = 65_536,
) -> NormalPathwisePlan:
    """License a locally dominated Normal pathwise directional derivative.

    Draws use independent standard-normal coordinates in source order.
    Ordinary first-order code runs over noise polynomials, with locally
    regular rational coefficients. Source decisions must be independent of
    noise and the Normal scales must be noise-independent and positive.
    Locations may depend on earlier draws through admitted expressions.

    The exact result integrates the return and its directional derivative.
    The plan's pointwise action evaluates the same polynomials at a fixed
    supplied vector of noise coordinates.

    Parameters
    ----------
    module : QiecModule
        The checked source supplying the actual first-order graph.
    computation : ComputationId
        The scalar-returning entry with one nominal Random instance.
    arguments : tuple[int or Fraction, ...]
        The exact parameter point.
    direction : tuple[int or Fraction, ...]
        The exact directional tangent in the same parameter order.
    policies : tuple[SiteEstimator, ...]
        One explicit `reparam` policy for each reached static site.
    max_sites : int
        Maximum number of distinct noise coordinates in the checked path.
    max_work : int
        Maximum sparse coefficient and moment work.
    fuel : int
        Maximum reference steps to assemble the plan.

    Returns
    -------
    NormalPathwisePlan
        The addressed noise polynomials and their exact expected JVP.

    Raises
    ------
    ValueError
        If the entry or input vectors are absent or malformed.
    JvpUnsupported
        If the actual source leaves the licensed fragment.
    JvpBoundaryError
        If a source decision hits an input-dependent equality boundary.
    JvpLimit
        If a step, noise coordinate, coefficient or work bound is exceeded.
    """
    for name, limit in (
        ("max_sites", max_sites),
        ("max_work", max_work),
        ("fuel", fuel),
    ):
        _validate_limit(name, limit)
    registry, owner = _random_computation(module, computation, "Normal pathwise")
    if not len(arguments) == len(direction) == len(owner.parameters):
        raise ValueError("arguments and direction must match the computation's arity")
    methods = frozenset(policy.site for policy in policies)
    if len(methods) != len(policies):
        raise ValueError("duplicate pathwise estimator for one site")
    if any(policy.estimator != "reparam" for policy in policies):
        raise JvpUnsupported(
            "Normal pathwise estimation needs an explicit reparam policy"
        )
    _check_first_order_graph(module, owner, "Normal pathwise", _check_request)
    ring = _PolynomialRing(min(max_sites, max_work, fuel), max_work)
    evaluator = _NormalEvaluator(module, fuel, ring, methods)
    environment = {
        parameter: evaluator.input(_fraction(value), _fraction(tangent), parameter=True)
        for parameter, value, tangent in zip(
            owner.parameters, arguments, direction, strict=True
        )
    }
    try:
        result = evaluator.evaluate_validated(owner.body, registry, environment)
        assert isinstance(result, _NoiseDual)
        path = _normal_path(computation, evaluator, result, max_work)
    except FuelExhaustedError as error:
        raise JvpLimit(
            f"Normal pathwise run exceeded its step budget of {fuel}"
        ) from error
    except PolynomialLawLimit as error:
        raise JvpLimit(f"Normal pathwise expansion: {error}") from error
    unused = methods - {visit.site for visit in evaluator.visits}
    if unused:
        raise JvpUnsupported(
            f"estimator policies name unreached sites: {', '.join(sorted(map(str, unused)))}"
        )
    return path


__all__ = ["NormalPathwisePlan", "NormalVisit", "plan_normal_pathwise"]
