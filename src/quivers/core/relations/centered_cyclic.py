"""Bounded cyclic string likelihoods with component-constant log gauges."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from math import isfinite

import torch

from quivers.core.algebras import LOG_PROB, DivergentClosureError, UndecidedClosureError
from quivers.core.relations import RationalRelation
from quivers.core.relations._centered import (
    _as_float,
    _centered_score,
    _RoundedFactor,
    _UndefinedCenteredNormalizer,
)
from quivers.core.relations._centered_accuracy import (
    _Interval,
    _path_error_extrema,
    _ratio_certificate,
    _rounded_inputs,
)
from quivers.core.relations._packed import _take
from quivers.core.relations.accuracy import (
    RelationAccuracyError,
    _RationalBudget,
    _UnsupportedAccuracyPrecision,
)
from quivers.core.relations.log_accuracy import (
    LogPathWeightCertificate,
    _cyclic_mass_bounds,
    _DirectedDecimal,
    _rational,
    _validate_decimal_settings,
)
from quivers.core.relations.queries import (
    RelationLimitError,
    _ActiveObservation,
    _Observation,
    _observe,
    _PathPlan,
    _trim_paths,
)
from quivers.core.relations.topology import Words
from quivers.core.relations.transforms import condition

_LOG_ZERO = -float("inf")


@dataclass(frozen=True, slots=True)
class _CyclicCenteredChart:
    plan: _PathPlan
    graph: _ActiveObservation
    dag: _ActiveObservation
    source_slots: tuple[int, ...]
    internal: tuple[int, ...]

    @classmethod
    def compile(
        cls, observed: _Observation, *, max_closure_size: int, max_closure_cells: int
    ) -> _CyclicCenteredChart:
        source_slots: list[int] = []

        def lift(slot: int) -> int:
            occurrence = len(source_slots)
            source_slots.append(slot)
            return occurrence

        arcs = [[(target, lift(slot)) for target, slot in row] for row in observed.arcs]
        initial = [(state, lift(slot)) for state, slot in observed.initial]
        final = [(state, lift(slot)) for state, slot in observed.final]
        plan = _trim_paths(
            arcs,
            initial,
            final,
            observed.examined,
            max_closure_size=max_closure_size,
            max_closure_cells=max_closure_cells,
            retain_observation=True,
            weight_slots=len(source_slots),
        )
        graph = plan.observation
        assert graph is not None
        component = {
            state: i for i, members in enumerate(graph.components) for state in members
        }
        internal: list[int] = []
        rows: list[tuple[tuple[int, int], ...]] = []
        for i, members in enumerate(graph.components):
            outward = []
            for state in members:
                for target, slot in graph.arcs[state]:
                    if (other := component[target]) == i:
                        internal.append(slot)
                    else:
                        outward.append((other, slot))
            rows.append(tuple(outward))
        dag = _ActiveObservation(
            tuple(rows),
            tuple((component[state], slot) for state, slot in graph.initial),
            tuple((component[state], slot) for state, slot in graph.final),
            tuple((i,) for i in range(len(rows))),
        )
        return cls(plan, graph, dag, tuple(source_slots), tuple(internal))

    @property
    def closure_cells(self) -> int:
        return sum(order * order for order in self.plan.work.closure_sizes)

    def _gauge(
        self, original: tuple[Fraction | None, ...], budget: _RationalBudget
    ) -> tuple[Fraction, list[_RoundedFactor]]:
        values = [original[slot] for slot in self.source_slots]
        count = len(self.dag.components)
        candidates: list[list[tuple[int, Fraction]]] = [[] for _ in range(count)]
        for state, slot in self.dag.final:
            if (weight := values[slot]) is not None:
                candidates[state].append((slot, weight))
        factors: list[_RoundedFactor] = [
            (slot, weight, float(weight))
            for slot in self.internal
            if (weight := values[slot]) is not None
        ]
        potentials = [Fraction(0)] * count
        for state in reversed(range(count)):
            outgoing = candidates[state]
            for target, slot in self.dag.arcs[state]:
                if (weight := values[slot]) is not None:
                    outgoing.append((slot, budget.add(weight, potentials[target])))
            potentials[state] = _balance(outgoing, factors, budget)
        starts = [
            (slot, budget.add(weight, potentials[state]))
            for state, slot in self.dag.initial
            if (weight := values[slot]) is not None
        ]
        return _balance(starts, factors, budget), factors

    def evaluate(
        self,
        snapshot: torch.Tensor,
        original: tuple[Fraction | None, ...],
        decimal: _DirectedDecimal,
        budget: _RationalBudget,
    ) -> tuple[Fraction, torch.Tensor, _Interval | None]:
        offset, factors = self._gauge(original, budget)
        rounded, errors = _rounded_inputs(len(self.source_slots), factors, budget)
        for slot in self.internal:
            expected = original[self.source_slots[slot]]
            if expected is None:
                preserved = rounded[slot] == _LOG_ZERO and slot not in errors
            else:
                preserved = rounded[slot] == float(expected) and errors.get(slot) == 0
            if not preserved:
                raise RelationAccuracyError(
                    "a centered cyclic gauge changed an original internal occurrence"
                )

        low, high = _cyclic_mass_bounds(self.graph, rounded, decimal, budget)
        offsets = snapshot.new_tensor(rounded)
        sources = _take(snapshot, self.source_slots)
        live = torch.where(torch.isfinite(offsets), sources, 0.0)
        try:
            mass = self.plan.evaluate(offsets + (live - live.detach()), LOG_PROB)
        except DivergentClosureError as error:
            raise UndecidedClosureError(
                "the numerical star disagrees with the verified finite upper star"
            ) from error
        observed = mass.item()
        if low.is_zero() and high.is_zero() and observed == _LOG_ZERO:
            return offset, mass, None
        if low.is_zero() or not isfinite(observed):
            raise RelationAccuracyError(
                "centered cyclic evaluation requires a positive residual bound and finite score"
            )

        # This condensation permits extra routes if a zero internal
        # weight disconnects a structural SCC. Its extrema still
        # enclose every supported original path perturbation.
        extrema = _path_error_extrema(self.dag, errors, budget)
        if extrema is None:
            raise RelationAccuracyError(
                "a positive centered mass has no condensation path"
            )
        lower = _rational(decimal.ln(low, lower=True), budget)
        upper = _rational(decimal.ln(high, lower=False), budget)
        return (
            offset,
            mass,
            (budget.add(lower, extrema[0]), budget.add(upper, extrema[1])),
        )


def _balance(
    candidates: list[tuple[int, Fraction]],
    factors: list[_RoundedFactor],
    budget: _RationalBudget,
) -> Fraction:
    if not candidates:
        return Fraction(0)
    maximum = candidates[0][1]
    for _, value in candidates[1:]:
        maximum = budget.maximum(maximum, value)
    for slot, value in candidates:
        exact = budget.add(value, maximum, subtract=True)
        factors.append((slot, exact, _as_float(exact, "a reduced cyclic path factor")))
    return maximum


def _centered_cyclic_accuracy_plan(
    relation: RationalRelation,
    input_words: Words,
    output_words: Words,
    *,
    absolute_log_tolerance: Fraction | int,
    max_states: int,
    max_work: int,
    max_center_occurrences: int,
    max_dyadic_work: int,
    max_closure_size: int,
    max_closure_cells: int,
    decimal_precision: int,
    max_decimal_exponent: int,
    max_decimal_work: int,
    max_exact_bits: int,
    max_exact_work: int,
) -> Callable[[], LogPathWeightCertificate]:
    if relation.algebra is not LOG_PROB:
        raise TypeError("centered cyclic log ratios require LogProb weights")
    if (
        type(absolute_log_tolerance) not in (Fraction, int)
        or absolute_log_tolerance < 0
    ):
        raise ValueError("absolute_log_tolerance must be a nonnegative rational")
    for name, value in (
        ("max_states", max_states),
        ("max_work", max_work),
        ("max_center_occurrences", max_center_occurrences),
        ("max_dyadic_work", max_dyadic_work),
        ("max_closure_size", max_closure_size),
        ("max_closure_cells", max_closure_cells),
        ("max_decimal_work", max_decimal_work),
        ("max_exact_bits", max_exact_bits),
        ("max_exact_work", max_exact_work),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    _validate_decimal_settings(decimal_precision, max_decimal_exponent)
    original = relation.topology
    if not 0 < original.input_tapes < len(original.tapes):
        raise ValueError("centered cyclic ratios require input and output tapes")
    if len(input_words) != original.input_tapes:
        raise ValueError("a centered cyclic ratio requires one word per input tape")
    original._validate_words(input_words + output_words)
    for tape, word in reversed(tuple(enumerate(input_words))):
        relation = condition(
            relation, tape, word, max_states=max_states, max_work=max_work
        )
    topology = relation.topology
    erased = replace(
        topology,
        arcs=tuple(
            replace(arc, labels=(None,) * len(topology.tapes)) for arc in topology.arcs
        ),
    )
    observations = (
        _observe(
            erased,
            ((),) * len(topology.tapes),
            max_states=max_states,
            max_work=max_work,
        ),
        _observe(topology, output_words, max_states=max_states, max_work=max_work),
    )
    visits = sum(len(item.states) for item in observations)
    occurrences = sum(
        len(item.initial) + len(item.final) + sum(map(len, item.arcs))
        for item in observations
    )
    if occurrences > max_center_occurrences:
        raise RelationLimitError("centered cyclic ratio exceeds max_center_occurrences")
    if len(topology.weight_names) + 2 * visits + 5 * occurrences > max_dyadic_work:
        raise RelationLimitError("centered cyclic ratio exceeds max_dyadic_work")
    whole = _CyclicCenteredChart.compile(
        observations[0],
        max_closure_size=max_closure_size,
        max_closure_cells=max_closure_cells,
    )
    part = _CyclicCenteredChart.compile(
        observations[1],
        max_closure_size=max_closure_size,
        max_closure_cells=max_closure_cells - whole.closure_cells,
    )

    def certify() -> LogPathWeightCertificate:
        budget = _RationalBudget(max_exact_bits, max_exact_work)
        tolerance = budget.value(absolute_log_tolerance)
        decimal = _DirectedDecimal(
            decimal_precision, max_decimal_exponent, max_decimal_work
        )
        snapshot = relation._weights().clone()
        if snapshot.dtype != torch.float64:
            raise _UnsupportedAccuracyPrecision(
                "directed centered cyclic ratio accuracy requires float64 weights"
            )
        exact = tuple(
            budget.value(value) if isfinite(value) else None
            for value in snapshot.detach().cpu().tolist()
        )
        whole_offset, whole_score, denominator = whole.evaluate(
            snapshot, exact, decimal, budget
        )
        if denominator is None:
            raise _UndefinedCenteredNormalizer(
                "path operation requires a positive finite normalizer"
            )
        part_offset, part_score, numerator = part.evaluate(
            snapshot, exact, decimal, budget
        )
        if numerator is None:
            difference = None
            score = part_score
        else:
            difference = budget.add(part_offset, whole_offset, subtract=True)
            score = _centered_score(snapshot, difference, part_score, whole_score)
        return _ratio_certificate(
            score, numerator, denominator, difference, tolerance, decimal, budget
        )

    return certify


def certify_centered_cyclic_log_ratio(
    relation: RationalRelation,
    input_words: Words,
    output_words: Words,
    *,
    absolute_log_tolerance: Fraction | int,
    max_states: int,
    max_work: int,
    max_center_occurrences: int,
    max_dyadic_work: int,
    max_closure_size: int,
    max_closure_cells: int,
    decimal_precision: int,
    max_decimal_exponent: int,
    max_decimal_work: int,
    max_exact_bits: int,
    max_exact_work: int,
) -> LogPathWeightCertificate:
    """Certify a centered fixed-input cyclic LogProb likelihood.

    Each productive component keeps its original internal log factors.
    The value certificate uses finite directed upper stars and compares
    the delivered score to bounds for the evaluated binary64 model.
    The returned score retains ordinary autograd through bounded
    numerical closure. Only the value error is certified.

    Parameters
    ----------
    relation : RationalRelation
        The typed relation with original live LogProb weights.
    input_words, output_words : Words
        Fixed words in the relation's input and output tape order.
    absolute_log_tolerance : Fraction or int
        Nonnegative absolute error target in log units.
    max_states, max_work : int
        Per-observation state and candidate arc caps.
    max_center_occurrences, max_dyadic_work : int
        Aggregate lifted occurrence and graph visit preflights.
    max_closure_size, max_closure_cells : int
        Local numeric order and combined cell caps, including auxiliaries.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Directed precision, symmetric exponent bound and shared work cap.
    max_exact_bits, max_exact_work : int
        Bit and work limits on the gauges, finite stars and value check.

    Returns
    -------
    LogPathWeightCertificate
        The live numerical score and exact endpoint error witness.

    Raises
    ------
    RelationLimitError
        A graph, cell or arithmetic resource limit is exhausted.
    RelationAccuracyError
        The upper star or the requested accuracy cannot be established.
    DivergentClosureError
        A directed lower endpoint proves divergent productive closure.
    UndecidedClosureError
        The bounded numerical star is unavailable.
    OverflowError
        A required reduced factor or reported score exceeds binary64 range.
    ValueError
        The model, bounds or normalization domain is invalid.
    TypeError
        The evaluated algebra or dtype is outside this certificate's scope.
    """
    return _centered_cyclic_accuracy_plan(
        relation,
        input_words,
        output_words,
        absolute_log_tolerance=absolute_log_tolerance,
        max_states=max_states,
        max_work=max_work,
        max_center_occurrences=max_center_occurrences,
        max_dyadic_work=max_dyadic_work,
        max_closure_size=max_closure_size,
        max_closure_cells=max_closure_cells,
        decimal_precision=decimal_precision,
        max_decimal_exponent=max_decimal_exponent,
        max_decimal_work=max_decimal_work,
        max_exact_bits=max_exact_bits,
        max_exact_work=max_exact_work,
    )()


__all__ = ["certify_centered_cyclic_log_ratio"]
