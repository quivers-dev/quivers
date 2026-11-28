"""Directed centered ratio certificates over the evaluated binary64 model."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import isfinite

import torch

from quivers.core.relations._centered import (
    _CenteredBinding,
    _CenteredPaths,
    _CenteredRounding,
    _RoundedFactor,
)
from quivers.core.relations.accuracy import (
    RelationAccuracyError,
    _RationalBudget,
    _UnsupportedAccuracyPrecision,
)
from quivers.core.relations.log_accuracy import (
    LogPathWeightCertificate,
    _DirectedDecimal,
    _mass_bounds,
    _rational,
    _validate_decimal_settings,
)
from quivers.core.relations.queries import _ActiveObservation

type _Interval = tuple[Fraction, Fraction]
_LOG_ZERO = -float("inf")


def _observation(chart: _CenteredPaths) -> _ActiveObservation:
    return _ActiveObservation(
        chart.arcs,
        chart.initial,
        tuple(chart.final.items()),
        tuple((state,) for state in chart.order),
    )


def _merge_shifted(
    current: _Interval | None,
    suffix: _Interval,
    error: Fraction,
    budget: _RationalBudget,
) -> _Interval:
    lower, upper = suffix
    if error:
        lower, upper = budget.add(lower, error), budget.add(upper, error)
    if current is None:
        return lower, upper
    return (
        lower if budget.less(lower, current[0]) else current[0],
        budget.maximum(upper, current[1]),
    )


def _path_error_extrema(
    graph: _ActiveObservation, errors: dict[int, Fraction], budget: _RationalBudget
) -> _Interval | None:
    finals: dict[int, _Interval] = {}
    for state, mark in graph.final:
        if mark in errors:
            error = errors[mark]
            finals[state] = _merge_shifted(
                finals.get(state), (error, error), Fraction(0), budget
            )
    suffixes: dict[int, _Interval] = {}
    for (state,) in reversed(graph.components):
        interval = finals.get(state)
        for target, slot in graph.arcs[state]:
            if slot in errors and target in suffixes:
                interval = _merge_shifted(
                    interval, suffixes[target], errors[slot], budget
                )
        if interval is not None:
            suffixes[state] = interval
    starts = None
    for state, slot in graph.initial:
        if slot in errors and state in suffixes:
            starts = _merge_shifted(starts, suffixes[state], errors[slot], budget)
    return starts


def _rounded_inputs(
    occurrence_count: int, factors: list[_RoundedFactor], budget: _RationalBudget
) -> tuple[list[float], dict[int, Fraction]]:
    rounded = [_LOG_ZERO] * occurrence_count
    errors: dict[int, Fraction] = {}
    for slot, original, candidate in factors:
        rounded[slot] = candidate
        errors[slot] = budget.add(
            budget.value(original), budget.value(candidate), subtract=True
        )
    return rounded, errors


def _residual_bounds(
    graph: _ActiveObservation,
    occurrence_count: int,
    factors: list[_RoundedFactor],
    decimal: _DirectedDecimal,
    budget: _RationalBudget,
) -> _Interval | None:
    rounded, errors = _rounded_inputs(occurrence_count, factors, budget)
    low, high = _mass_bounds(graph, rounded, decimal)
    extrema = _path_error_extrema(graph, errors, budget)
    if low.is_zero() and high.is_zero() and extrema is None:
        return None
    if low.is_zero() or extrema is None:
        raise RelationAccuracyError(
            "centered ratio requires a positive residual mass bound"
        )

    # A path's exact reduced log score differs from its rounded path
    # score by the sum of its occurrence errors. The extreme path
    # errors therefore bound the shift of the entire log sum.
    lower = _rational(decimal.ln(low, lower=True), budget)
    upper = _rational(decimal.ln(high, lower=False), budget)
    return budget.add(lower, extrema[0]), budget.add(upper, extrema[1])


def _ratio_certificate(
    score: torch.Tensor,
    part: _Interval | None,
    whole: _Interval | None,
    difference: Fraction | None,
    tolerance: Fraction,
    decimal: _DirectedDecimal,
    budget: _RationalBudget,
) -> LogPathWeightCertificate:
    observed = score.item()
    if whole is None:
        raise RelationAccuracyError(
            "centered ratio requires a positive denominator enclosure"
        )
    if part is None and difference is None and observed == _LOG_ZERO:
        return LogPathWeightCertificate(
            score, None, None, Fraction(0), decimal.work, budget.work, budget.bits
        )
    if part is None or difference is None or not isfinite(observed):
        raise RelationAccuracyError(
            "centered ratio candidate disagrees with its positive numerator support"
        )

    shift = budget.value(difference)
    lower = budget.add(shift, budget.add(part[0], whole[1], subtract=True))
    upper = budget.add(shift, budget.add(part[1], whole[0], subtract=True))
    estimate = budget.value(observed)
    error = budget.maximum(
        abs(budget.add(estimate, lower, subtract=True)),
        abs(budget.add(estimate, upper, subtract=True)),
    )
    if budget.less(tolerance, error):
        raise RelationAccuracyError(
            "could not establish the requested absolute centered log ratio error"
        )
    return LogPathWeightCertificate(
        score, lower, upper, error, decimal.work, budget.work, budget.bits
    )


def _centered_accuracy_plan(
    binding: _CenteredBinding,
    *,
    absolute_log_tolerance: Fraction,
    decimal_precision: int,
    max_decimal_exponent: int,
    max_decimal_work: int,
    max_exact_bits: int,
    max_exact_work: int,
) -> Callable[[], LogPathWeightCertificate]:
    if type(absolute_log_tolerance) is not Fraction or absolute_log_tolerance < 0:
        raise ValueError("absolute_log_tolerance must be a nonnegative rational")
    for name, limit in (
        ("max_decimal_work", max_decimal_work),
        ("max_exact_bits", max_exact_bits),
        ("max_exact_work", max_exact_work),
    ):
        if type(limit) is not int or limit < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    _validate_decimal_settings(decimal_precision, max_decimal_exponent)
    plan, read_weights = binding
    numerator = _observation(plan.numerator)
    denominator = _observation(plan.denominator)

    def certify() -> LogPathWeightCertificate:
        budget = _RationalBudget(max_exact_bits, max_exact_work)
        tolerance = budget.value(absolute_log_tolerance)
        rounding = _CenteredRounding()
        with torch.no_grad():
            weights = read_weights().detach().clone()
            if weights.dtype != torch.float64:
                raise _UnsupportedAccuracyPrecision(
                    "directed centered ratio accuracy requires float64 weights"
                )
            score = plan.evaluate(weights, rounding=rounding).log_prob
        decimal = _DirectedDecimal(
            decimal_precision, max_decimal_exponent, max_decimal_work
        )
        whole = _residual_bounds(
            denominator,
            len(plan.denominator.source_slots),
            rounding.denominator,
            decimal,
            budget,
        )
        part = _residual_bounds(
            numerator,
            len(plan.numerator.source_slots),
            rounding.numerator,
            decimal,
            budget,
        )
        return _ratio_certificate(
            score, part, whole, rounding.maximum_difference, tolerance, decimal, budget
        )

    return certify
