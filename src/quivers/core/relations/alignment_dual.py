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
from quivers.core.relations.graph import StringFactorGraph
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import RelationTopology, Words
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
    def __init__(
        self, topology: RelationTopology, values: tuple[Fraction | None, ...]
    ) -> None:
        self._topology = topology
        self._weights = values
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

    def _start(
        self, observed: tuple[_Word | None, ...], adjustments: _Adjustments
    ) -> tuple[tuple[_NgramScorer, ...], _Configuration, Fraction]:
        scorers = tuple(map(_NgramScorer, adjustments))
        starts = tuple(scorer.step(0, _START) for scorer in scorers)
        initial = (0,) * len(observed), tuple(state for state, _ in starts)
        bonus = sum((score for _, score in starts), Fraction(0))
        return scorers, initial, bonus

    def _transition(
        self,
        index: int,
        config: _Configuration,
        observed: tuple[_Word | None, ...],
        scorers: tuple[_NgramScorer, ...],
    ) -> tuple[_Configuration, Fraction] | None:
        offsets, suffixes = config
        next_offsets, next_suffixes = list(offsets), list(suffixes)
        arc = self._topology.arcs[index]
        adjusted = self._weights[arc.weight]
        assert adjusted is not None
        for tape, label in enumerate(arc.labels):
            if label is None:
                continue
            word, offset = observed[tape], offsets[tape]
            if word is not None:
                if offset == len(word) or word[offset] != label:
                    return None
                next_offsets[tape] += 1
            next_suffixes[tape], score = scorers[tape].step(suffixes[tape], label)
            adjusted += score
        return (tuple(next_offsets), tuple(next_suffixes)), adjusted

    def _terminal(
        self,
        source: int,
        config: _Configuration,
        observed: tuple[_Word | None, ...],
        scorers: tuple[_NgramScorer, ...],
    ) -> tuple[Fraction, Fraction] | None:
        offsets, suffixes = config
        if (slot := self._final.get(source)) is None or any(
            word is not None and offset != len(word)
            for word, offset in zip(observed, offsets, strict=True)
        ):
            return None
        weight = self._weights[slot]
        assert weight is not None
        bonus = sum(
            (
                scorer.step(state, _STOP)[1]
                for scorer, state in zip(scorers, suffixes, strict=True)
            ),
            Fraction(0),
        )
        return weight, weight + bonus

    def best(
        self,
        observed: tuple[_Word | None, ...],
        adjustments: _Adjustments,
        max_states: int,
        max_work: int,
    ) -> tuple[StringAlignment | None, int, int]:
        scorers, initial_config, bonus = self._start(observed, adjustments)
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
            for config, trace in chart.pop(source, {}).items():
                if (
                    terminal := self._terminal(source, config, observed, scorers)
                ) is not None:
                    weight, final_score = terminal
                    adjusted = trace.adjusted + final_score
                    if best is None or best[3] < adjusted:
                        best = trace, source, trace.base + weight, adjusted
                for index in self._outgoing[source]:
                    if work == max_work:
                        raise RelationLimitError(
                            "string alignment decoder exceeds max_work"
                        )
                    work += 1
                    transition = self._transition(index, config, observed, scorers)
                    if transition is None:
                        continue
                    next_config, adjustment = transition
                    arc = self._topology.arcs[index]
                    weight = self._weights[arc.weight]
                    assert weight is not None
                    adjusted = trace.adjusted + adjustment
                    target = chart[arc.target]
                    old = target.get(next_config)
                    if old is None:
                        if visits == max_states:
                            raise RelationLimitError(
                                "string alignment decoder exceeds max_states"
                            )
                        visits += 1
                    if old is None or old.adjusted < adjusted:
                        target[next_config] = _Trace(
                            trace.base + weight, adjusted, trace.start, trace, index
                        )
        return (self._result(*best) if best is not None else None), visits, work

    def suffix_optimum(
        self,
        observed: tuple[_Word | None, ...],
        adjustments: _Adjustments,
        max_states: int,
        max_work: int,
    ) -> tuple[Fraction | None, int, int]:
        """Derive an optimum without the decoder's prefix scores or traceback."""
        scorers, initial, bonus = self._start(observed, adjustments)
        reached: dict[int, set[_Configuration]] = defaultdict(set)
        states = len(self._initial)
        if states > max_states:
            raise RelationLimitError("alignment verification exceeds max_states")
        for state in self._initial:
            reached[state].add(initial)
        work = 0
        for state in self._order:
            for config in reached.get(state, ()):
                for index in self._outgoing[state]:
                    if work == max_work:
                        raise RelationLimitError(
                            "alignment verification exceeds max_work"
                        )
                    work += 1
                    transition = self._transition(index, config, observed, scorers)
                    if transition is None:
                        continue
                    target = reached[self._topology.arcs[index].target]
                    next_config, _ = transition
                    if next_config not in target:
                        if states == max_states:
                            raise RelationLimitError(
                                "alignment verification exceeds max_states"
                            )
                        states += 1
                        target.add(next_config)

        suffix: dict[int, dict[_Configuration, Fraction]] = defaultdict(dict)
        for state in reversed(self._order):
            for config in reached.get(state, ()):
                terminal = self._terminal(state, config, observed, scorers)
                best = None if terminal is None else terminal[1]
                for index in self._outgoing[state]:
                    transition = self._transition(index, config, observed, scorers)
                    if transition is None:
                        continue
                    next_config, adjustment = transition
                    target = self._topology.arcs[index].target
                    tail = suffix[target].get(next_config)
                    if tail is not None:
                        candidate = adjustment + tail
                        if best is None or candidate > best:
                            best = candidate
                if best is not None:
                    suffix[state][config] = best
        optimum = None
        for state, slot in self._initial.items():
            if (tail := suffix[state].get(initial)) is not None:
                weight = self._weights[slot]
                assert weight is not None
                candidate = weight + bonus + tail
                if optimum is None or candidate > optimum:
                    optimum = candidate
        return optimum, states, 2 * work

    def check_trace(
        self,
        path: StringAlignment,
        observed: tuple[_Word | None, ...],
        adjustments: _Adjustments,
    ) -> None:
        """Check a claimed path score from original occurrences and word counts."""
        invalid = "alignment witness is not a valid original trace"
        start = self._initial.get(path.start)
        final = self._final.get(path.stop)
        if (
            type(path.start) is not int
            or type(path.stop) is not int
            or start is None
            or final is None
            or len(path.arc_ids) >= len(self._order)
        ):
            raise ArithmeticError(invalid)
        start_weight, stop_weight = self._weights[start], self._weights[final]
        assert start_weight is not None and stop_weight is not None
        base = start_weight + stop_weight
        state = path.start
        labels: list[list[int]] = [[] for _ in observed]
        for index in path.arc_ids:
            if type(index) is not int or not 0 <= index < len(self._topology.arcs):
                raise ArithmeticError(invalid)
            arc = self._topology.arcs[index]
            weight = self._weights[arc.weight]
            if arc.source != state or weight is None:
                raise ArithmeticError(invalid)
            base += weight
            state = arc.target
            for tape, label in enumerate(arc.labels):
                if label is not None:
                    labels[tape].append(label)
        words = tuple(tuple(row) for row in labels)
        if (
            state != path.stop
            or words != path.words
            or any(
                fixed is not None and word != fixed
                for word, fixed in zip(words, observed, strict=True)
            )
        ):
            raise ArithmeticError(invalid)
        adjusted = base
        for word, shifts in zip(words, adjustments, strict=True):
            counts = {
                length: _grams(word, length)
                for length in {len(gram) for gram in shifts}
            }
            adjusted += sum(
                (shift * counts[len(gram)][gram] for gram, shift in shifts.items()),
                Fraction(0),
            )
        if path.base_score != base or path.adjusted_score != adjusted:
            raise ArithmeticError(
                "alignment witness score differs from its original path"
            )


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
    certificate. Verification counts cover the independent bound checks
    and original trace examinations, separately from the search.
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
    max_verification_states: int = 0
    verification_arc_examinations: int = 0

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
    max_verification_states: int = 0
    verification_work: int = 0

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

    def _independent_optimum(
        self,
        decoder: _LocalAlignments,
        words: tuple[_Word | None, ...],
        shifts: _Adjustments,
    ) -> Fraction | None:
        optimum, states, work = decoder.suffix_optimum(
            words, shifts, self.max_states, self.max_work
        )
        self.max_verification_states = max(self.max_verification_states, states)
        self.verification_work += work
        return optimum

    def verify_paths(
        self,
        words: Mapping[str, _Word | None],
        adjustments: _Multipliers,
        paths: tuple[StringAlignment, ...],
        *,
        maximizing: bool,
    ) -> None:
        if len(paths) != len(self.decoders):
            raise ArithmeticError("alignment witnesses must cover every factor")
        centered: dict[tuple[str, _Ngram], Fraction] = defaultdict(Fraction)
        for decoder, scope, shifts, path in zip(
            self.decoders, self.scopes, adjustments, paths, strict=True
        ):
            local = tuple(words[name] for name in scope)
            decoder.check_trace(path, local, shifts)
            self.verification_work += len(path.arc_ids)
            if maximizing:
                if (
                    self._independent_optimum(decoder, local, shifts)
                    != path.adjusted_score
                ):
                    raise ArithmeticError(
                        "alignment witness does not attain the independent local optimum"
                    )
                for name, shift in zip(scope, shifts, strict=True):
                    for gram, value in shift.items():
                        centered[name, gram] += value
        if any(centered.values()):
            raise ArithmeticError("alignment upper-bound multipliers are not centered")

    def verify_infeasible(self, words: Mapping[str, _Word | None]) -> None:
        for decoder, scope in zip(self.decoders, self.scopes, strict=True):
            local = tuple(words[name] for name in scope)
            no_adjustments: _Adjustments = tuple({} for _ in scope)
            if self._independent_optimum(decoder, local, no_adjustments) is None:
                return
        raise ArithmeticError(
            "alignment infeasibility has no independent local witness"
        )


@dataclass(frozen=True, slots=True)
class _AlignmentSettings:
    max_iterations: int
    max_ngram_order: int
    max_features: int
    max_states: int
    max_work: int
    step_size: Fraction | int
    absolute_gap: Fraction | int | None
    relative_gap: Fraction | int | None

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.max_iterations, self.max_ngram_order)
        ):
            raise ValueError(
                "max_iterations and max_ngram_order must be positive integers"
            )
        if type(self.max_features) is not int or self.max_features < 0:
            raise ValueError("max_features must be a nonnegative integer")
        if type(self.step_size) not in (Fraction, int) or self.step_size <= 0:
            raise ValueError("step_size must be a positive rational")
        for name, target in (
            ("absolute_gap", self.absolute_gap),
            ("relative_gap", self.relative_gap),
        ):
            if target is not None and (
                type(target) not in (Fraction, int) or target < 0
            ):
                raise ValueError(f"{name} must be a nonnegative rational or None")
        if self.max_states is None or self.max_work is None:
            raise ValueError(
                "string alignment MAP requires explicit state and work limits"
            )
        _validate_limits(self.max_states, self.max_work)


def _alignment_inputs(
    graph: StringFactorGraph,
) -> tuple[
    tuple[_LocalAlignments, ...], tuple[tuple[str, ...], ...], dict[str, _Word | None]
]:
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
            _LocalAlignments(
                factor.relation.topology,
                tuple(
                    None if value == -torch.inf else Fraction(value)
                    for value in factor.relation._weights().detach().tolist()
                ),
            )
            for factor in factors
        )
    return decoders, scopes, observations


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
        Maximum examined candidate transitions per local decoding or
        individual verification scan. Each independently verified local
        witness has two scans and an additional original trace check.
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
    settings = _AlignmentSettings(
        max_iterations,
        max_ngram_order,
        max_features,
        max_states,
        max_work,
        step_size,
        absolute_gap,
        relative_gap,
    )
    decoders, scopes, observations = _alignment_inputs(graph)
    run = _DualAlignmentRun(decoders, scopes, max_states, max_work)
    result, _ = _run_alignment_map(run, observations, settings)
    return result


def _run_alignment_map(
    run: _DualAlignmentRun,
    observations: Mapping[str, _Word | None],
    settings: _AlignmentSettings,
) -> tuple[StringAlignmentMapResult, tuple[StringAlignment, ...] | None]:
    scopes = run.scopes
    absolute_gap, relative_gap = settings.absolute_gap, settings.relative_gap
    multipliers: _Multipliers = tuple(tuple({} for _ in scope) for scope in scopes)
    zero: _Multipliers = tuple(tuple({} for _ in scope) for scope in scopes)
    features: dict[str, set[_Ngram]] = {name: set() for name in observations}
    n_features = 0
    upper = None
    upper_witness: tuple[tuple[StringAlignment, ...], _Multipliers] | None = None
    primal = None
    primal_paths = None
    assignment = None
    certificate = None
    status: Literal["optimal", "infeasible", "gap", "limit"] = "limit"
    gap_target_met = False
    gap_requested = absolute_gap is not None or relative_gap is not None
    iteration = 0
    for iteration in range(1, settings.max_iterations + 1):
        paths = run.decode(observations, multipliers)
        if paths is None:
            run.verify_infeasible(observations)
            status = "infeasible"
            break
        bound = sum((path.adjusted_score for path in paths), Fraction(0))
        if upper is None or bound < upper:
            upper = bound
            stored = tuple(
                tuple(dict(shift) for shift in group) for group in multipliers
            )
            upper_witness = paths, stored
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
                primal_paths = feasible
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
            run.verify_paths(observations, multipliers, paths, maximizing=True)
            assignment = tuple(proposed.items())
            primal_paths = paths
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
        if iteration == settings.max_iterations:
            break
        step = Fraction(settings.step_size) / iteration
        for name, local in copies.items():
            selected = tuple(word for word, _ in local)
            if len(set(selected)) == 1:
                continue
            active = features[name]
            if n_features < settings.max_features:
                gram = _expand(selected, active, settings.max_ngram_order)
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
    if certificate is None:
        if upper_witness is not None:
            paths, stored = upper_witness
            run.verify_paths(observations, stored, paths, maximizing=True)
        if primal_paths is not None:
            assert assignment is not None
            run.verify_paths(dict(assignment), zero, primal_paths, maximizing=False)
    result = StringAlignmentMapResult(
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
        run.max_verification_states,
        run.verification_work,
    )
    return result, primal_paths


__all__ = [
    "StringAlignment",
    "StringAlignmentCertificate",
    "StringAlignmentMapResult",
    "dual_alignment_map",
]
