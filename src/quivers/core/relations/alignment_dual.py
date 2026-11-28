"""Adaptive joint alignment MAP with exact sparse local decoders."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from typing import Literal

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.relations import _weight_snapshot
from quivers.core.relations._ngram import _NgramAutomaton
from quivers.core.relations.graph import StringFactor, StringFactorGraph
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Words
from quivers.core.relations.transforms import _reached

type _Word = tuple[int, ...]
type _Ngram = tuple[int, ...]
type _Adjustments = tuple[Mapping[_Ngram, Fraction], ...]
type _Configuration = tuple[tuple[int, ...], tuple[int, ...]]

_START = -1
_STOP = -2


class _NgramScorer:
    """Count active contiguous features with a deterministic suffix automaton."""

    def __init__(self, coefficients: Mapping[_Ngram, Fraction]) -> None:
        active = {gram: weight for gram, weight in coefficients.items() if weight}
        self._machine = _NgramAutomaton(tuple(active))
        weights = tuple(active.values())
        self._scores = tuple(
            sum((weights[index] for index in matched), Fraction(0))
            for matched in self._machine.emissions
        )

    def step(self, state: int, symbol: int) -> tuple[int, Fraction]:
        target = self._machine.step(state, symbol)
        return target, self._scores[target]


@dataclass(frozen=True, slots=True)
class StringAlignment:
    """One local maximizing path with original relation occurrence identities."""

    words: Words
    start: int
    arc_ids: tuple[int, ...]
    stop: int
    base_score: Fraction
    adjusted_score: Fraction


@dataclass(frozen=True, slots=True)
class _Trace:
    base: Fraction
    adjusted: Fraction
    start: int
    predecessor: _Trace | None = None
    arc: int | None = None


class _LocalAlignments:
    def __init__(self, factor: StringFactor, weights: torch.Tensor) -> None:
        self._topology = topology = factor.relation.topology
        self._weights = values = tuple(
            None if value == -torch.inf else Fraction(value)
            for value in weights.detach().tolist()
        )
        forward: dict[int, list[int]] = defaultdict(list)
        backward: dict[int, list[int]] = defaultdict(list)
        by_state: dict[int, list[int]] = defaultdict(list)
        for i, arc in enumerate(topology.arcs):
            if values[arc.weight] is not None:
                forward[arc.source].append(arc.target)
                backward[arc.target].append(arc.source)
                by_state[arc.source].append(i)
        initial = {
            mark.state: mark.weight
            for mark in topology.initial
            if values[mark.weight] is not None
        }
        final = {
            mark.state: mark.weight
            for mark in topology.final
            if values[mark.weight] is not None
        }
        live = _reached(initial, forward) & _reached(final, backward)
        dependencies = {
            state: [target for target in forward[state] if target in live]
            for state in sorted(live)
        }
        try:
            self._order = tuple(
                reversed(tuple(TopologicalSorter(dependencies).static_order()))
            )
        except CycleError as error:
            raise ValueError(
                "string alignment MAP requires acyclic positive local factor support"
            ) from error
        self._outgoing = {
            state: tuple(
                index
                for index in by_state[state]
                if topology.arcs[index].target in live
            )
            for state in live
        }
        self._initial = {
            state: weight for state, weight in initial.items() if state in live
        }
        self._final = {
            state: weight for state, weight in final.items() if state in live
        }

    def _result(
        self, trace: _Trace, stop: int, base: Fraction, adjusted: Fraction
    ) -> StringAlignment:
        arcs: list[int] = []
        start = trace.start
        while trace.arc is not None and trace.predecessor is not None:
            arcs.append(trace.arc)
            trace = trace.predecessor
        ordered = tuple(reversed(arcs))
        words = tuple(
            tuple(
                symbol
                for index in ordered
                if (symbol := self._topology.arcs[index].labels[tape]) is not None
            )
            for tape in range(len(self._topology.tapes))
        )
        return StringAlignment(words, start, ordered, stop, base, adjusted)

    def best(
        self,
        observed: tuple[_Word | None, ...],
        adjustments: _Adjustments,
        max_states: int,
        max_work: int,
    ) -> tuple[StringAlignment | None, int, int]:
        scorers = tuple(map(_NgramScorer, adjustments))
        starts = tuple(scorer.step(0, _START) for scorer in scorers)
        offset_zero = (0,) * len(observed)
        initial_config = offset_zero, tuple(state for state, _ in starts)
        bonus = sum((score for _, score in starts), Fraction(0))
        chart: dict[int, dict[_Configuration, _Trace]] = defaultdict(dict)
        visits, work = 0, 0
        if len(self._initial) > max_states:
            raise RelationLimitError("string alignment decoder exceeds max_states")
        for state, slot in self._initial.items():
            weight = self._weights[slot]
            assert weight is not None
            visits += 1
            chart[state][initial_config] = _Trace(weight, weight + bonus, state)

        best: tuple[_Trace, int, Fraction, Fraction] | None = None
        for source in self._order:
            for (offsets, suffixes), trace in chart.pop(source, {}).items():
                if (slot := self._final.get(source)) is not None and all(
                    word is None or offset == len(word)
                    for word, offset in zip(observed, offsets, strict=True)
                ):
                    weight = self._weights[slot]
                    assert weight is not None
                    terminal_bonus = sum(
                        (
                            scorer.step(state, _STOP)[1]
                            for scorer, state in zip(scorers, suffixes, strict=True)
                        ),
                        Fraction(0),
                    )
                    adjusted = trace.adjusted + weight + terminal_bonus
                    if best is None or best[3] < adjusted:
                        best = trace, source, trace.base + weight, adjusted
                for index in self._outgoing[source]:
                    if work == max_work:
                        raise RelationLimitError(
                            "string alignment decoder exceeds max_work"
                        )
                    work += 1
                    arc = self._topology.arcs[index]
                    next_offsets = list(offsets)
                    next_suffixes = list(suffixes)
                    weight = self._weights[arc.weight]
                    assert weight is not None
                    adjusted = trace.adjusted + weight
                    for tape, label in enumerate(arc.labels):
                        if label is None:
                            continue
                        word, offset = observed[tape], offsets[tape]
                        if word is not None:
                            if offset == len(word) or word[offset] != label:
                                break
                            next_offsets[tape] += 1
                        next_suffixes[tape], score = scorers[tape].step(
                            suffixes[tape], label
                        )
                        adjusted += score
                    else:
                        config = tuple(next_offsets), tuple(next_suffixes)
                        target = chart[arc.target]
                        old = target.get(config)
                        if old is None:
                            if visits == max_states:
                                raise RelationLimitError(
                                    "string alignment decoder exceeds max_states"
                                )
                            visits += 1
                        if old is None or old.adjusted < adjusted:
                            target[config] = _Trace(
                                trace.base + weight, adjusted, trace.start, trace, index
                            )
        return (self._result(*best) if best is not None else None), visits, work


type _Assignment = tuple[tuple[str, _Word], ...]
type _Multiplier = dict[_Ngram, Fraction]
type _Multipliers = tuple[tuple[_Multiplier, ...], ...]
type _FrozenMultipliers = tuple[tuple[tuple[tuple[_Ngram, Fraction], ...], ...], ...]


@dataclass(frozen=True, slots=True)
class StringAlignmentCertificate:
    """Agreeing local maximizing paths and their exact n-gram potentials.

    Alignments and multiplier groups are listed in factor order. One
    multiplier table per tape maps n-grams to exact rational weights.
    Symbols -1 and -2 mark the beginning and end of a word.
    """

    alignments: tuple[StringAlignment, ...]
    multipliers: _FrozenMultipliers


@dataclass(frozen=True, slots=True)
class StringAlignmentMapResult:
    """A joint alignment optimum, infeasibility, or a bounded search.

    The score is from a feasible complete word assignment. The upper
    bound is the smallest adjusted score sum observed so far. Feature
    sets apply to the named variables. Local work includes both
    adjusted maximization and exact primal feasibility checks.
    Reaching a requested objective gap has status `gap` when the
    local paths have not agreed. Only `optimal` carries the agreement
    certificate.
    """

    status: Literal["optimal", "infeasible", "gap", "limit"]
    iterations: int
    assignment: _Assignment | None
    score: Fraction | None
    upper_bound: Fraction | None
    active_features: tuple[tuple[str, tuple[_Ngram, ...]], ...]
    max_local_states: int
    local_arc_examinations: int
    certificate: StringAlignmentCertificate | None = None
    gap_target_met: bool = False

    @property
    def absolute_gap(self) -> Fraction | None:
        """The exact upper bound on the feasible score's suboptimality."""
        if self.score is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.score

    @property
    def relative_gap(self) -> Fraction | None:
        """The relative suboptimality bound when the objective scale permits it."""
        if self.score is None or self.upper_bound is None:
            return None
        return _relative_gap(self.score, self.upper_bound)


def _relative_gap(lower: Fraction, upper: Fraction) -> Fraction | None:
    if lower == upper:
        return Fraction(0)
    if lower < 0 <= upper:
        return None
    return (upper - lower) / abs(upper)


def _grams(word: _Word, size: int) -> Counter[_Ngram]:
    extended = (_START, *word, _STOP)
    return Counter(extended[i : i + size] for i in range(len(extended) - size + 1))


def _expand(
    selected: tuple[_Word, ...],
    active: set[_Ngram],
    max_order: int,
) -> _Ngram | None:
    for size in range(1, min(max_order, max(map(len, selected)) + 2) + 1):
        counts = tuple(_grams(word, size) for word in selected)
        for gram in sorted(set().union(*(count.keys() for count in counts)) - active):
            if any(count[gram] != counts[0][gram] for count in counts[1:]):
                return gram
    return None


def _feature_counts(
    words: tuple[_Word, ...], features: set[_Ngram]
) -> tuple[dict[_Ngram, int], ...]:
    orders = {len(gram) for gram in features}
    result = []
    for word in words:
        counts = {size: _grams(word, size) for size in orders}
        result.append({gram: counts[len(gram)][gram] for gram in features})
    return tuple(result)


@dataclass(slots=True)
class _DualAlignmentRun:
    decoders: tuple[_LocalAlignments, ...]
    scopes: tuple[tuple[str, ...], ...]
    max_states: int
    max_work: int
    max_local_states: int = 0
    work: int = 0

    def decode(
        self,
        words: Mapping[str, _Word | None],
        adjustments: _Multipliers,
    ) -> tuple[StringAlignment, ...] | None:
        paths = []
        for decoder, names, shifts in zip(
            self.decoders, self.scopes, adjustments, strict=True
        ):
            path, states, arcs = decoder.best(
                tuple(words[name] for name in names),
                shifts,
                self.max_states,
                self.max_work,
            )
            self.max_local_states = max(self.max_local_states, states)
            self.work += arcs
            if path is None:
                return None
            paths.append(path)
        return tuple(paths)


def dual_alignment_map(
    graph: StringFactorGraph,
    *,
    max_iterations: int,
    max_ngram_order: int,
    max_features: int,
    max_states: int,
    max_work: int,
    step_size: Fraction | int = Fraction(1),
    absolute_gap: Fraction | int | None = None,
    relative_gap: Fraction | int | None = None,
) -> StringAlignmentMapResult:
    """Maximize joint string and local alignment scores by dual decomposition.

    The graph must use LogProb and each factor must have acyclic
    productive positive support. A factor's score adds the original
    log weights on its path, including both boundary marks. Original
    weights are read once per shared source and interpreted as exact
    binary rationals. Decoding has no gradient contract.

    Each factor receives an independent copy of its incident string
    variables. Updates keep each shared n-gram's multipliers centered
    across those copies. In case of disagreement, a new distinguishing
    feature can be added within the configured order and count limits.

    Parameters
    ----------
    graph : StringFactorGraph
        The typed variables, observations and weighted relation factors.
    max_iterations : int
        Positive limit on the number of dual iterations.
    max_ngram_order : int
        Positive maximum order of a newly selected distinguishing feature.
    max_features : int
        Maximum number of selected variable and feature pairs.
    max_states : int
        Maximum reached configurations per local decoding.
    max_work : int
        Maximum examined candidate transitions per local decoding.
    step_size : Fraction or int
        Positive multiplier step. The step is divided by the iteration.
    absolute_gap : Fraction or int or None
        A nonnegative target for dual upper bound minus feasible score.
    relative_gap : Fraction or int or None
        A nonnegative target for suboptimality divided by the magnitude
        of the unknown optimum. A finite bound is unavailable when the
        feasible score is negative and the upper bound is nonnegative.
        If both targets are supplied, both must be established.

    Returns
    -------
    StringAlignmentMapResult
        The best feasible assignment found, exact upper bound, and an
        optimum certificate only if all incident copies agree on their
        full words. A proved gap target permits early stopping without
        that certificate. Budget-limited runs can remain uncertified.

    Raises
    ------
    TypeError
        If the graph does not use LogProb weights.
    ValueError
        If variables are isolated, the solver settings are invalid, or
        a factor has a productive positive cycle.
    RelationLimitError
        If an exact local decoder exceeds a state or work budget.
    """
    if any(
        type(value) is not int or value < 1
        for value in (max_iterations, max_ngram_order)
    ):
        raise ValueError("max_iterations and max_ngram_order must be positive integers")
    if type(max_features) is not int or max_features < 0:
        raise ValueError("max_features must be a nonnegative integer")
    if type(step_size) not in (Fraction, int) or step_size <= 0:
        raise ValueError("step_size must be a positive rational")
    for name, target in (
        ("absolute_gap", absolute_gap),
        ("relative_gap", relative_gap),
    ):
        if target is not None and (type(target) not in (Fraction, int) or target < 0):
            raise ValueError(f"{name} must be a nonnegative rational or None")
    if max_states is None or max_work is None:
        raise ValueError("string alignment MAP requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    factors = graph._factors
    scopes = tuple(factor.variables for factor in factors)
    if any(factor.relation.algebra is not LOG_PROB for factor in factors):
        raise TypeError("joint string alignment MAP requires LogProb factors")
    observations = {
        name: variable.observed for name, variable in graph._variables.items()
    }
    if set(observations) != {name for scope in scopes for name in scope}:
        raise ValueError("joint string alignment MAP does not allow isolated variables")
    with _weight_snapshot():
        decoders = tuple(
            _LocalAlignments(factor, factor.relation._weights()) for factor in factors
        )
    run = _DualAlignmentRun(decoders, scopes, max_states, max_work)
    multipliers: _Multipliers = tuple(tuple({} for _ in scope) for scope in scopes)
    zero: _Multipliers = tuple(tuple({} for _ in scope) for scope in scopes)
    features: dict[str, set[_Ngram]] = {name: set() for name in observations}
    n_features = 0
    upper = None
    primal = None
    assignment = None
    certificate = None
    status: Literal["optimal", "infeasible", "gap", "limit"] = "limit"
    gap_target_met = False
    gap_requested = absolute_gap is not None or relative_gap is not None
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        paths = run.decode(observations, multipliers)
        if paths is None:
            status = "infeasible"
            break
        bound = sum((path.adjusted_score for path in paths), Fraction(0))
        upper = bound if upper is None else min(upper, bound)
        copies: dict[str, list[tuple[_Word, _Multiplier]]] = defaultdict(list)
        for scope, path, shifts in zip(scopes, paths, multipliers, strict=True):
            for name, word, shift in zip(scope, path.words, shifts, strict=True):
                copies[name].append((word, shift))
        proposed = {name: copies[name][0][0] for name in observations}
        agreement = all(
            word == proposed[name]
            for name, local in copies.items()
            for word, _ in local
        )
        feasible = paths if agreement else run.decode(proposed, zero)
        if feasible is not None:
            score = sum((path.base_score for path in feasible), Fraction(0))
            if primal is None or primal < score:
                assignment = tuple(proposed.items())
                primal = score
        if primal is not None:
            if primal > upper:
                raise ArithmeticError(
                    "feasible alignment score exceeds the dual upper bound"
                )
            if gap_requested:
                relative_bound = (
                    None if relative_gap is None else _relative_gap(primal, upper)
                )
                gap_target_met = (
                    absolute_gap is None or upper - primal <= absolute_gap
                ) and (
                    relative_gap is None
                    or (relative_bound is not None and relative_bound <= relative_gap)
                )
        if agreement:
            if primal != bound:
                raise ArithmeticError(
                    "agreeing alignment maximizers must attain the dual bound"
                )
            assignment = tuple(proposed.items())
            frozen = tuple(
                tuple(tuple(sorted(shift.items())) for shift in shifts)
                for shifts in multipliers
            )
            certificate = StringAlignmentCertificate(paths, frozen)
            status = "optimal"
            break
        if gap_target_met:
            status = "gap"
            break
        if iteration == max_iterations:
            break
        step = Fraction(step_size) / iteration
        for name, local in copies.items():
            selected = tuple(word for word, _ in local)
            if len(set(selected)) == 1:
                continue
            active = features[name]
            if n_features < max_features:
                gram = _expand(selected, active, max_ngram_order)
                if gram is not None:
                    active.add(gram)
                    n_features += 1
            counts = _feature_counts(selected, active)
            for gram in active:
                average = Fraction(sum(row[gram] for row in counts), len(local))
                for row, (_, shift) in zip(counts, local, strict=True):
                    shift[gram] = shift.get(gram, Fraction(0)) + step * (
                        average - row[gram]
                    )
    return StringAlignmentMapResult(
        status,
        iteration,
        assignment,
        primal,
        upper,
        tuple((name, tuple(sorted(active))) for name, active in features.items()),
        run.max_local_states,
        run.work,
        certificate,
        gap_target_met,
    )


__all__ = [
    "StringAlignment",
    "StringAlignmentCertificate",
    "StringAlignmentMapResult",
    "dual_alignment_map",
]
