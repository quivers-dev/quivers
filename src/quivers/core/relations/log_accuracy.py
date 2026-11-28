"""Directed forward-error certificates for acyclic LogProb path sums."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Clamped,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    Subnormal,
    Underflow,
)
from fractions import Fraction
from math import isfinite

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.relations import RationalRelation
from quivers.core.relations.accuracy import (
    RelationAccuracyError,
    _RationalBudget,
    _UnsupportedAccuracyPrecision,
)
from quivers.core.relations.queries import (
    RelationLimitError,
    _ActiveObservation,
    _compile_paths,
)
from quivers.core.relations.topology import Words

type _Bounds = tuple[Decimal, Decimal]
_ZERO = Decimal(0)
_ONE = Decimal(1)
_ZERO_BOUNDS = (_ZERO, _ZERO)
_LOG_ZERO = -float("inf")
_RANGE_SIGNALS = (Clamped, Overflow, Subnormal, Underflow)
_TRAPS = [*_RANGE_SIGNALS, DivisionByZero, FloatOperation, InvalidOperation]


@dataclass(frozen=True, slots=True)
class LogPathWeightCertificate:
    """A packed LogProb score and exact forward-error bounds in log units.

    A verified zero mass has two absent log endpoints, a score of negative
    infinity, and zero error by convention. For positive mass, the two
    rational endpoints enclose its mathematical log weight. The bounds
    concern the evaluated binary64 slots and carry no derivative assurance.
    """

    weight: torch.Tensor
    lower: Fraction | None
    upper: Fraction | None
    absolute_log_error: Fraction
    decimal_work: int
    exact_work: int
    exact_bits: int


class _DirectedDecimal:
    def __init__(self, precision: int, max_exponent: int, max_work: int) -> None:
        options = dict(prec=precision, Emin=-max_exponent, Emax=max_exponent, clamp=0)
        self.nearest = Context(
            **options, rounding=ROUND_HALF_EVEN, flags=[], traps=_TRAPS
        )
        self.down = Context(**options, rounding=ROUND_FLOOR, flags=[], traps=_TRAPS)
        self.up = Context(**options, rounding=ROUND_CEILING, flags=[], traps=_TRAPS)
        self.max_exponent = max_exponent
        self.max_work = max_work
        self.work = 0

    def _charge(self) -> None:
        if self.work == self.max_work:
            raise RelationLimitError(
                "LogProb path certificate exceeds max_decimal_work"
            )
        self.work += 1

    def _run(self, name: str, operation: Callable[[], Decimal]) -> Decimal:
        self._charge()
        try:
            result = operation()
        except _RANGE_SIGNALS as error:
            raise RelationLimitError(
                f"Decimal {name} exceeds max_decimal_exponent ({type(error).__name__})"
            ) from error
        except DecimalException as error:
            raise RelationAccuracyError(
                f"Decimal {name} cannot establish a finite enclosure"
            ) from error
        if not result.is_zero() and not result.is_normal(self.nearest):
            raise RelationLimitError(f"Decimal {name} exceeds max_decimal_exponent")
        return result

    def _neighbor(self, center: Decimal, *, lower: bool, name: str) -> Decimal:
        if not center.is_normal(self.nearest):
            raise RelationLimitError(f"Decimal {name} exceeds max_decimal_exponent")
        adjacent = self.nearest.next_minus if lower else self.nearest.next_plus
        bound = self._run(f"{name} neighbor", lambda: adjacent(center))
        if not bound.is_normal(self.nearest):
            raise RelationLimitError(
                f"Decimal {name} neighbor exceeds max_decimal_exponent"
            )
        return bound

    def exp(self, log_weight: float) -> _Bounds:
        if log_weight == 0.0:
            return _ONE, _ONE
        # e**2 < 10, so the guarded input alone cannot put its
        # exponential beyond the configured decimal exponent range.
        if abs(log_weight) > 2 * self.max_exponent:
            raise RelationLimitError("finite log weight exceeds max_decimal_exponent")
        self._charge()
        value = Decimal.from_float(log_weight)
        center = self._run("exponential", lambda: value.exp(context=self.nearest))
        return (
            self._neighbor(center, lower=True, name="exponential"),
            self._neighbor(center, lower=False, name="exponential"),
        )

    def accumulate(self, total: _Bounds, weight: _Bounds, suffix: _Bounds) -> _Bounds:
        return (
            self._run(
                "lower accumulation",
                lambda: self.down.fma(weight[0], suffix[0], total[0]),
            ),
            self._run(
                "upper accumulation",
                lambda: self.up.fma(weight[1], suffix[1], total[1]),
            ),
        )

    def from_rational(self, value: Fraction, *, lower: bool) -> Decimal:
        if not value:
            return _ZERO
        try:
            self._charge()
            numerator = Decimal(value.numerator, context=self.nearest)
            self._charge()
            denominator = Decimal(value.denominator, context=self.nearest)
        except DecimalException as error:
            raise RelationLimitError(
                "exact integer conversion exceeds the Decimal implementation range"
            ) from error
        context = self.down if lower else self.up
        bound = self._run(
            "rational endpoint", lambda: context.divide(numerator, denominator)
        )
        if bound.is_zero():
            raise RelationLimitError(
                "Decimal rational endpoint exceeds max_decimal_exponent"
            )
        return bound

    def ln(self, mass: Decimal, *, lower: bool) -> Decimal:
        if mass == _ONE:
            return _ZERO
        center = self._run("logarithm", lambda: mass.ln(context=self.nearest))
        return self._neighbor(center, lower=lower, name="logarithm")


def _mass_slots(
    weights: list[float], decimal: _DirectedDecimal
) -> Callable[[int], _Bounds]:
    slots: dict[int, _Bounds] = {}

    def mass(slot: int) -> _Bounds:
        if slot not in slots:
            log_weight = weights[slot]
            slots[slot] = (
                _ZERO_BOUNDS if log_weight == _LOG_ZERO else decimal.exp(log_weight)
            )
        return slots[slot]

    return mass


def _mass_bounds(
    graph: _ActiveObservation, weights: list[float], decimal: _DirectedDecimal
) -> _Bounds:
    # Discard paths that already contain a log-zero boundary or edge.
    # A finite weight on such a path needs no exponential enclosure.
    reachable = {state for state, slot in graph.initial if weights[slot] != _LOG_ZERO}
    for (state,) in graph.components:
        if state in reachable:
            reachable.update(
                target
                for target, slot in graph.arcs[state]
                if weights[slot] != _LOG_ZERO
            )
    final = {state: slot for state, slot in graph.final if weights[slot] != _LOG_ZERO}
    mass = _mass_slots(weights, decimal)
    suffixes: dict[int, _Bounds] = {}
    for (state,) in reversed(graph.components):
        if state not in reachable:
            continue
        total = mass(final[state]) if state in final else _ZERO_BOUNDS
        for target, slot in graph.arcs[state]:
            if (
                weights[slot] != _LOG_ZERO
                and (suffix := suffixes[target]) != _ZERO_BOUNDS
            ):
                total = decimal.accumulate(total, mass(slot), suffix)
        suffixes[state] = total
    total = _ZERO_BOUNDS
    for state, slot in graph.initial:
        if weights[slot] != _LOG_ZERO and (suffix := suffixes[state]) != _ZERO_BOUNDS:
            total = decimal.accumulate(total, mass(slot), suffix)
    return total


def _rational(decimal: Decimal, budget: _RationalBudget) -> Fraction:
    if decimal.is_zero():
        return budget.value(0)
    _, digits, exponent = decimal.as_tuple()
    assert isinstance(exponent, int)  # Finite endpoints already checked.
    # Bound the unreduced coefficient and power of ten before converting.
    # An integer with d decimal digits needs at most 4*d binary bits.
    budget._charge(4 * (len(digits) + max(0, exponent)), 1 + 4 * max(0, -exponent))
    return Fraction(*decimal.as_integer_ratio())


def _eliminate_walks(
    matrix: list[list[Fraction]],
    suffix: list[Fraction],
    budget: _RationalBudget,
) -> list[Fraction]:
    size = len(matrix)
    for state, row in enumerate(matrix):
        # The diagonal sums the nonempty return walks through previously
        # eliminated states. Its star exists exactly below one.
        pivot = budget.add(Fraction(1), row[state], subtract=True)
        if pivot <= 0:
            raise RelationAccuracyError(
                "LogProb path certificate could not establish a finite upper star"
            )
        suffix[state] = budget.divide(suffix[state], pivot)
        for target in range(state + 1, size):
            if row[target]:
                row[target] = budget.divide(row[target], pivot)
        for source in range(state + 1, size):
            incoming = matrix[source][state]
            if not incoming:
                continue
            suffix[source] = budget.add(
                suffix[source], budget.multiply(incoming, suffix[state])
            )
            for target in range(state + 1, size):
                if row[target]:
                    matrix[source][target] = budget.add(
                        matrix[source][target], budget.multiply(incoming, row[target])
                    )
    for state in reversed(range(size)):
        for target in range(state + 1, size):
            weight = matrix[state][target]
            if weight and suffix[target]:
                suffix[state] = budget.add(
                    suffix[state], budget.multiply(weight, suffix[target])
                )
    return suffix


def _component_endpoint(
    edges: list[tuple[int, int, int]],
    rhs: list[_Bounds],
    mass: Callable[[int], _Bounds],
    decimal: _DirectedDecimal,
    budget: _RationalBudget,
    *,
    lower: bool,
) -> tuple[Decimal, ...]:
    column = 0 if lower else 1
    order = len(rhs)
    zero = Fraction(0)
    matrix = [[zero] * order for _ in rhs]
    used: dict[int, Fraction] = {}
    for source, target, slot in edges:
        if slot not in used:
            used[slot] = _rational(mass(slot)[column], budget)
        if weight := used[slot]:
            matrix[source][target] = budget.add(matrix[source][target], weight)
    suffix = [_rational(bound[column], budget) for bound in rhs]
    result = _eliminate_walks(matrix, suffix, budget)
    return tuple(decimal.from_rational(value, lower=lower) for value in result)


def _cyclic_mass_bounds(
    graph: _ActiveObservation,
    weights: list[float],
    decimal: _DirectedDecimal,
    budget: _RationalBudget,
) -> _Bounds:
    final = dict(graph.final)
    suffixes: dict[int, _Bounds] = {}
    mass = _mass_slots(weights, decimal)
    for members in reversed(graph.components):
        local = {state: index for index, state in enumerate(members)}
        rhs = []
        edges = []
        for source, state in enumerate(members):
            total = mass(final[state]) if state in final else _ZERO_BOUNDS
            for target, slot in graph.arcs[state]:
                if (index := local.get(target)) is not None:
                    edges.append((source, index, slot))
                elif (suffix := suffixes[target]) != _ZERO_BOUNDS:
                    weight = mass(slot)
                    if weight != _ZERO_BOUNDS:
                        total = decimal.accumulate(total, weight, suffix)
            rhs.append(total)
        if edges:
            # Certify finiteness before computing the lower endpoint.
            high = _component_endpoint(edges, rhs, mass, decimal, budget, lower=False)
            low = _component_endpoint(edges, rhs, mass, decimal, budget, lower=True)
            rhs = list(zip(low, high, strict=True))
        suffixes.update(zip(members, rhs, strict=True))
    result = _ZERO_BOUNDS
    for state, slot in graph.initial:
        if (suffix := suffixes[state]) != _ZERO_BOUNDS:
            weight = mass(slot)
            if weight != _ZERO_BOUNDS:
                result = decimal.accumulate(result, weight, suffix)
    return result


def certify_logprob_path_weight(
    relation: RationalRelation,
    words: Words,
    *,
    absolute_log_tolerance: Fraction | int,
    max_states: int,
    max_work: int,
    decimal_precision: int,
    max_decimal_exponent: int,
    max_decimal_work: int,
    max_exact_bits: int,
    max_exact_work: int,
) -> LogPathWeightCertificate:
    """Certify absolute log error for a float64 fixed-word LogProb DAG score.

    The ordinary packed score and the directed verifier use the same
    evaluated binary64 slot snapshot. The returned tensor keeps its
    ordinary derivative. The certificate covers its value.

    Parameters
    ----------
    relation : RationalRelation
        A LogProb relation with float64 weights.
    words : Words
        One observed word per original tape.
    absolute_log_tolerance : Fraction or int
        The required nonnegative absolute error in log units.
    max_states, max_work : int
        Observation state and candidate arc budgets.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Decimal significant digits, symmetric adjusted exponent bound,
        and maximum primitive calls and exact binary64 conversions.
    max_exact_bits, max_exact_work : int
        Bit and operation bounds for the final rational error test.

    Returns
    -------
    LogPathWeightCertificate
        The usual score and exact log endpoints or verified zero mass.

    Raises
    ------
    TypeError
        The algebra or evaluated dtype lacks the supported contract.
    ValueError
        The words, tolerance, or resource declaration is invalid.
    ProductiveEpsilonCycleError
        The structurally productive observation has a cycle.
    RelationLimitError
        A state, work, range, or bit limit is exceeded.
    RelationAccuracyError
        The packed value's error could not be certified at this tolerance.
    """
    return _logprob_accuracy_plan(
        relation,
        words,
        absolute_log_tolerance=absolute_log_tolerance,
        max_states=max_states,
        max_work=max_work,
        decimal_precision=decimal_precision,
        max_decimal_exponent=max_decimal_exponent,
        max_decimal_work=max_decimal_work,
        max_exact_bits=max_exact_bits,
        max_exact_work=max_exact_work,
    )()


def certify_cyclic_logprob_path_weight(
    relation: RationalRelation,
    words: Words,
    *,
    absolute_log_tolerance: Fraction | int,
    max_states: int,
    max_work: int,
    max_closure_size: int,
    max_closure_cells: int,
    decimal_precision: int,
    max_decimal_exponent: int,
    max_decimal_work: int,
    max_exact_bits: int,
    max_exact_work: int,
) -> LogPathWeightCertificate:
    """Certify absolute log error with explicitly bounded productive epsilon stars.

    Each structurally productive cyclic component must admit a finite
    upper-enclosure star under exact rational elimination. The resulting
    enclosure concerns the evaluated binary64 slots. The returned tensor
    has the ordinary LogProb derivative. The certificate covers its value.

    Parameters
    ----------
    relation : RationalRelation
        A LogProb relation with float64 weights.
    words : Words
        One observed word per original tape.
    absolute_log_tolerance : Fraction or int
        The required nonnegative absolute error in log units.
    max_states, max_work : int
        Observation state and candidate arc limits.
    max_closure_size, max_closure_cells : int
        Local star order and aggregate cell limits, including auxiliaries.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Decimal significant digits, symmetric adjusted exponent bound,
        and maximum primitive calls and exact input conversions.
    max_exact_bits, max_exact_work : int
        Bit and operation limits for the exact cyclic solves and error test.

    Returns
    -------
    LogPathWeightCertificate
        The usual score and exact log endpoints or verified zero mass.

    Raises
    ------
    TypeError
        The algebra or evaluated dtype is outside the supported domain.
    ValueError
        The words, tolerance, or resource declaration is invalid.
    RelationLimitError
        A structural, range, work, or exact bit limit is exhausted.
    RelationAccuracyError
        The directed upper star cannot be proved finite or the score's
        error cannot be certified at the requested tolerance.
    UndecidedClosureError
        The ordinary numerical LogProb evaluator cannot establish closure.
    """
    for name, limit in (
        ("max_closure_size", max_closure_size),
        ("max_closure_cells", max_closure_cells),
    ):
        if type(limit) is not int or limit < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return _logprob_accuracy_plan(
        relation,
        words,
        absolute_log_tolerance=absolute_log_tolerance,
        max_states=max_states,
        max_work=max_work,
        decimal_precision=decimal_precision,
        max_decimal_exponent=max_decimal_exponent,
        max_decimal_work=max_decimal_work,
        max_exact_bits=max_exact_bits,
        max_exact_work=max_exact_work,
        max_closure_size=max_closure_size,
        max_closure_cells=max_closure_cells,
    )()


def _validate_decimal_settings(precision: int, max_exponent: int) -> None:
    if type(precision) is not int or not 1 <= precision <= MAX_PREC:
        raise ValueError("decimal_precision is outside the Decimal context range")
    if type(max_exponent) is not int or not 1 <= max_exponent <= min(
        MAX_EMAX, -MIN_EMIN
    ):
        raise ValueError("max_decimal_exponent is outside the Decimal context range")


def _logprob_accuracy_plan(
    relation: RationalRelation,
    words: Words,
    *,
    absolute_log_tolerance: Fraction | int,
    max_states: int,
    max_work: int,
    decimal_precision: int,
    max_decimal_exponent: int,
    max_decimal_work: int,
    max_exact_bits: int,
    max_exact_work: int,
    max_closure_size: int | None = None,
    max_closure_cells: int | None = None,
) -> Callable[[], LogPathWeightCertificate]:
    if relation.algebra is not LOG_PROB:
        raise TypeError("directed path accuracy requires LogProb weights")
    if (
        type(absolute_log_tolerance) not in (Fraction, int)
        or absolute_log_tolerance < 0
    ):
        raise ValueError("absolute_log_tolerance must be a nonnegative rational")
    for name, limit in (
        ("max_states", max_states),
        ("max_work", max_work),
        ("max_decimal_work", max_decimal_work),
        ("max_exact_bits", max_exact_bits),
        ("max_exact_work", max_exact_work),
    ):
        if type(limit) is not int or limit < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    _validate_decimal_settings(decimal_precision, max_decimal_exponent)
    topology = relation.topology
    topology._validate_words(words)
    plan = _compile_paths(
        topology,
        words,
        max_states=max_states,
        max_work=max_work,
        max_closure_size=max_closure_size,
        max_closure_cells=max_closure_cells,
        retain_observation=True,
    )
    graph = plan.observation
    assert graph is not None

    def certify() -> LogPathWeightCertificate:
        budget = _RationalBudget(max_exact_bits, max_exact_work)
        tolerance = budget.value(absolute_log_tolerance)
        weights = relation._weights().clone()
        relation._check_weights(weights)
        if weights.dtype != torch.float64:
            raise _UnsupportedAccuracyPrecision(
                "directed LogProb path accuracy requires float64 weights"
            )
        decimal = _DirectedDecimal(
            decimal_precision, max_decimal_exponent, max_decimal_work
        )
        slots = weights.detach().tolist()
        low, high = (
            _cyclic_mass_bounds(graph, slots, decimal, budget)
            if plan.closures
            else _mass_bounds(graph, slots, decimal)
        )
        score = plan.evaluate(weights, LOG_PROB)
        observed = score.item()
        if high.is_zero() and low.is_zero():
            if observed != _LOG_ZERO:
                raise RelationAccuracyError(
                    "packed LogProb score disagrees with exact zero mass"
                )
            return LogPathWeightCertificate(
                score, None, None, Fraction(0), decimal.work, budget.work, budget.bits
            )
        if low.is_zero() or not isfinite(observed):
            raise RelationAccuracyError(
                "LogProb path certificate needs a positive mass bound and finite score"
            )
        lower = _rational(decimal.ln(low, lower=True), budget)
        upper = _rational(decimal.ln(high, lower=False), budget)
        estimate = budget.value(observed)
        error = budget.maximum(
            abs(budget.add(estimate, lower, subtract=True)),
            abs(budget.add(estimate, upper, subtract=True)),
        )
        if budget.less(tolerance, error):
            raise RelationAccuracyError(
                "could not establish the requested absolute log path error"
            )
        return LogPathWeightCertificate(
            score, lower, upper, error, decimal.work, budget.work, budget.bits
        )

    return certify


__all__ = [
    "LogPathWeightCertificate",
    "certify_cyclic_logprob_path_weight",
    "certify_logprob_path_weight",
]
