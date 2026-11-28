"""Forward-error certificates for bounded Markov path sums."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

import torch

from quivers.core.algebras import MARKOV
from quivers.core.relations import RationalRelation
from quivers.core.relations.queries import (
    RelationLimitError,
    _ActiveObservation,
    _compile_paths,
    _validate_limits,
)
from quivers.core.relations.topology import Words

type _Interval = tuple[Fraction, Fraction]
type _Row = tuple[tuple[int, Fraction], ...]
_ZERO = Fraction(0)
_TWO = Fraction(2)


class RelationAccuracyError(ArithmeticError):
    """The available evaluation does not establish the requested accuracy."""


@dataclass(frozen=True, slots=True)
class PathWeightCertificate:
    """A scalar Markov weight and forward-error evidence for its input snapshot.

    The weight is the normal differentiable packed result. The exact
    interval concerns the mathematical path sum of its evaluated binary
    weight vector. The error bounds apply to the returned scalar value.
    The relative bound is reported when requested.
    They give no assurance about its derivative or about computations
    that produced the input vector.
    """

    weight: torch.Tensor
    lower: Fraction
    upper: Fraction
    absolute_error: Fraction
    relative_error: Fraction | None
    exact_work: int
    exact_bits: int
    closure_cells: int


class _RationalBudget:
    def __init__(self, max_bits: int, max_work: int) -> None:
        self.max_bits = max_bits
        self.max_work = max_work
        self.bits = 0
        self.work = 0

    def _charge(self, *bit_bounds: int) -> None:
        if self.work == self.max_work:
            raise RelationLimitError("path certificate exceeds max_exact_work")
        bits = max(bit_bounds)
        if bits > self.max_bits:
            raise RelationLimitError("path certificate exceeds max_exact_bits")
        self.work += 1
        self.bits = max(self.bits, bits)

    def value(self, number: Fraction | float | int) -> Fraction:
        if isinstance(number, float) and not isfinite(number):
            raise RelationAccuracyError(
                "path certificate needs finite candidate values"
            )
        # Conversion from float64 itself has at most 1,075 bits. Check
        # both resulting integers before any subsequent exact arithmetic.
        value = Fraction(number)
        self._charge(value.numerator.bit_length(), value.denominator.bit_length())
        return value

    def add(
        self, left: Fraction, right: Fraction, *, subtract: bool = False
    ) -> Fraction:
        a, b, c, d = (
            left.numerator,
            left.denominator,
            right.numerator,
            right.denominator,
        )
        self._charge(
            max(a.bit_length() + d.bit_length(), c.bit_length() + b.bit_length()) + 1,
            b.bit_length() + d.bit_length(),
        )
        return left - right if subtract else left + right

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        self._charge(
            left.numerator.bit_length() + right.numerator.bit_length(),
            left.denominator.bit_length() + right.denominator.bit_length(),
        )
        return left * right

    def divide(self, left: Fraction, right: Fraction) -> Fraction:
        self._charge(
            left.numerator.bit_length() + right.denominator.bit_length(),
            left.denominator.bit_length() + right.numerator.bit_length(),
        )
        return left / right

    def less(self, left: Fraction, right: Fraction) -> bool:
        self._charge(
            left.numerator.bit_length() + right.denominator.bit_length(),
            right.numerator.bit_length() + left.denominator.bit_length(),
        )
        return left < right

    def maximum(self, left: Fraction, right: Fraction) -> Fraction:
        return right if self.less(left, right) else left

    def dot(self, row: _Row, values: tuple[Fraction, ...]) -> Fraction:
        total = _ZERO
        for index, weight in row:
            total = self.add(total, self.multiply(weight, values[index]))
        return total


def _propose_suffixes(
    closed: torch.Tensor, rhs: tuple[float, ...]
) -> tuple[list[float], list[float]]:
    size = len(rhs)
    with torch.no_grad():
        matrix = closed.detach().reshape(size, size)
        return (matrix @ matrix.new_tensor(rhs)).tolist(), matrix.sum(dim=1).tolist()


def _cyclic_intervals(
    closed: torch.Tensor,
    rows: tuple[_Row, ...],
    rhs: tuple[_Interval, ...],
    budget: _RationalBudget,
) -> tuple[_Interval, ...]:
    try:
        midpoint = tuple(
            float(budget.divide(budget.add(low, high), _TWO)) for low, high in rhs
        )
    except OverflowError as error:
        raise RelationAccuracyError(
            "path certificate candidate exceeds float64"
        ) from error
    centers, witnesses = _propose_suffixes(closed, midpoint)
    y = tuple(budget.value(item) for item in centers)
    v = tuple(budget.value(item) for item in witnesses)
    multiplier = _ZERO
    for row, (lower, upper), center, direction in zip(rows, rhs, y, v, strict=True):
        margin = budget.add(direction, budget.dot(row, v), subtract=True)
        if direction.numerator <= 0 or margin.numerator <= 0:
            raise RelationAccuracyError("path certificate could not verify contraction")
        linear = budget.add(center, budget.dot(row, y), subtract=True)
        residual = budget.maximum(
            abs(budget.add(lower, linear, subtract=True)),
            abs(budget.add(upper, linear, subtract=True)),
        )
        multiplier = budget.maximum(multiplier, budget.divide(residual, margin))
    intervals = []
    for center, direction in zip(y, v, strict=True):
        radius = budget.multiply(multiplier, direction)
        low = budget.add(center, radius, subtract=True)
        high = budget.add(center, radius)
        intervals.append((low if low.numerator > 0 else _ZERO, high))
    return tuple(intervals)


def _suffix_intervals(
    graph: _ActiveObservation,
    closed: tuple[torch.Tensor, ...],
    weights: tuple[Fraction, ...],
    budget: _RationalBudget,
) -> dict[int, _Interval]:
    final = dict(graph.final)
    intervals: dict[int, _Interval] = {}
    matrices: Iterator[torch.Tensor] = reversed(closed)
    for members in reversed(graph.components):
        local = {state: index for index, state in enumerate(members)}
        rows = []
        rhs = []
        for state in members:
            lower = upper = weights[final[state]] if state in final else _ZERO
            row = []
            for target, slot in graph.arcs[state]:
                weight = weights[slot]
                if (index := local.get(target)) is not None:
                    row.append((index, weight))
                    continue
                low, high = intervals[target]
                lower = budget.add(lower, budget.multiply(weight, low))
                upper = budget.add(upper, budget.multiply(weight, high))
            rhs.append((lower, upper))
            rows.append(tuple(row))
        local_bounds = (
            _cyclic_intervals(next(matrices), tuple(rows), tuple(rhs), budget)
            if any(rows)
            else tuple(rhs)
        )
        intervals.update(zip(members, local_bounds, strict=True))
    return intervals


def _tolerance(value: Fraction | int | None) -> Fraction | None:
    if value is not None and (type(value) not in (Fraction, int) or value < 0):
        raise ValueError("path accuracy tolerances must be nonnegative rationals")
    return None if value is None else Fraction(value)


def certify_markov_path_weight(
    relation: RationalRelation,
    words: Words,
    *,
    max_states: int,
    max_work: int,
    max_exact_bits: int,
    max_exact_work: int,
    max_closure_cells: int,
    max_closure_size: int | None = None,
    absolute_tolerance: Fraction | int | None = None,
    relative_tolerance: Fraction | int | None = None,
) -> PathWeightCertificate:
    """Certify an absolute or relative error bound for a Markov pair weight.

    The certificate describes the exact path sum at the evaluated
    binary64 weight vector. It uses exact sparse reverse recurrences
    on acyclic states. Cyclic components reuse the packed evaluator's
    local star matrices as proposals and check their error against
    original sparse rows in rational arithmetic.

    Parameters
    ----------
    relation : RationalRelation
        A Markov relation with binary64 scalar weights.
    words : Words
        One observed word per original tape.
    max_states, max_work : int
        Fixed-word observation state and arc budgets.
    max_exact_bits, max_exact_work : int
        Maximum conservative integer bit bound per exact operation and
        number of conversions, arithmetic operations, or comparisons.
    max_closure_cells : int
        Aggregate cell bound before constructing closure expansion or matrices.
    max_closure_size : int or None
        Maximum local star order. None requires an acyclic observation.
    absolute_tolerance, relative_tolerance : Fraction or int or None
        Requested error bounds, each required if supplied. Relative
        error needs a positive lower bound or certified exact zero.

    Returns
    -------
    PathWeightCertificate
        The usual packed weight and exact bounds at the evaluated input.

    Raises
    ------
    TypeError
        The algebra or scalar dtype has no supported accuracy contract.
    ValueError
        A limit, observation, or tolerance is invalid.
    RelationAccuracyError
        A finite witness or requested forward-error bound was not established.
    RelationLimitError
        A resource bound is exceeded.
    """
    if relation.algebra is not MARKOV:
        raise TypeError("certified fixed-word accuracy requires Markov weights")
    _validate_limits(max_states, max_work, max_closure_size)
    for name, value in (
        ("max_states", max_states),
        ("max_work", max_work),
        ("max_exact_bits", max_exact_bits),
        ("max_exact_work", max_exact_work),
        ("max_closure_cells", max_closure_cells),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    absolute = _tolerance(absolute_tolerance)
    relative = _tolerance(relative_tolerance)
    if absolute is None and relative is None:
        raise ValueError(
            "path certification requires an absolute or relative tolerance"
        )
    budget = _RationalBudget(max_exact_bits, max_exact_work)
    absolute = budget.value(absolute) if absolute is not None else None
    relative = budget.value(relative) if relative is not None else None
    topology = relation.topology
    topology._validate_words(words)
    if len(topology.weight_names) > max_exact_work - budget.work:
        raise RelationLimitError(
            "path certificate exceeds max_exact_work reading weight slots"
        )
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
    values = relation._weights().clone()
    relation._check_weights(values)
    if values.dtype != torch.float64:
        raise TypeError("certified fixed-word accuracy requires float64 weights")
    weights = tuple(budget.value(value) for value in values.detach().tolist())
    closed = plan._closed_weights(values, MARKOV)
    suffixes = _suffix_intervals(graph, closed, weights, budget)
    lower = upper = _ZERO
    for state, slot in graph.initial:
        low, high = suffixes[state]
        lower = budget.add(lower, budget.multiply(weights[slot], low))
        upper = budget.add(upper, budget.multiply(weights[slot], high))
    score_inputs = torch.cat((values, *closed)) if closed else values
    score = plan.packed.evaluate(score_inputs, MARKOV)
    estimate = budget.value(score.item())
    lower_error = abs(budget.add(estimate, lower, subtract=True))
    upper_error = abs(budget.add(estimate, upper, subtract=True))
    absolute_error = budget.maximum(lower_error, upper_error)
    relative_error = None
    if relative is not None and lower.numerator > 0:
        relative_error = budget.maximum(
            budget.divide(lower_error, lower), budget.divide(upper_error, upper)
        )
    elif relative is not None and upper.numerator == estimate.numerator == 0:
        relative_error = _ZERO
    if absolute is not None and budget.less(absolute, absolute_error):
        raise RelationAccuracyError(
            "could not establish the requested absolute path error"
        )
    if relative is not None and (
        relative_error is None or budget.less(relative, relative_error)
    ):
        raise RelationAccuracyError(
            "could not establish the requested relative path error"
        )
    return PathWeightCertificate(
        score,
        lower,
        upper,
        absolute_error,
        relative_error,
        budget.work,
        budget.bits,
        sum(size**2 for size in plan.work.closure_sizes),
    )


__all__ = [
    "PathWeightCertificate",
    "RelationAccuracyError",
    "certify_markov_path_weight",
]
