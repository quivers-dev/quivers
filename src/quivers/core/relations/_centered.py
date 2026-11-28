"""Binary64 log ratios from exact dyadic gauges and packed acyclic paths."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.relations._packed import _PackedPaths, _pack_paths, _take
from quivers.core.relations.queries import (
    ProductiveEpsilonCycleError,
    RelationLimitError,
    _observe,
    _productive_graph,
    _ProductiveGraph,
)
from quivers.core.relations.topology import RelationTopology, Words


@dataclass(frozen=True, slots=True)
class CenteredStringLogScore:
    """A normalized log probability and work on its two acyclic charts.

    The state and occurrence counts include both structurally productive
    observations. Occurrences include boundary marks. The dyadic work
    counts slot reads and state and occurrence visits in the exact pass.
    """

    log_prob: torch.Tensor
    states: int
    occurrences: int
    weight_slots: int
    dyadic_work: int


def _as_float(value: Fraction, operation: str) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        rounded = -torch.inf
    if not isfinite(rounded):
        raise OverflowError(f"{operation} is outside binary64 precision")
    return rounded


def _active_dag(
    topology: RelationTopology,
    words: Words,
    max_states: int,
    max_work: int,
) -> tuple[_ProductiveGraph, list[tuple[int, int]]]:
    topology._validate_words(words)
    observed = _observe(topology, words, max_states=max_states, max_work=max_work)
    graph = _productive_graph(observed.arcs, observed.initial, observed.final)
    if len(graph.order) != len(graph.states):
        raise ProductiveEpsilonCycleError(
            "centered log ratios require acyclic productive observations"
        )
    return graph, observed.final


@dataclass(frozen=True, slots=True)
class _CenteredPaths:
    order: tuple[int, ...]
    arcs: tuple[tuple[tuple[int, int], ...], ...]
    initial: tuple[tuple[int, int], ...]
    final: dict[int, int]
    source_slots: tuple[int, ...]
    reverse: _PackedPaths

    @classmethod
    def pack(
        cls, graph: _ProductiveGraph, final: list[tuple[int, int]]
    ) -> _CenteredPaths:
        slots: list[int] = []

        def lift(slot: int) -> int:
            occurrence = len(slots)
            slots.append(slot)
            return occurrence

        # The observation already separates repeated traversals of the
        # same slot at different positions. Preserve each occurrence.
        arcs = tuple(
            tuple((target, lift(slot)) for target, slot in row) for row in graph.arcs
        )
        starts = tuple((state, lift(slot)) for state, slot in graph.initial)
        stops = tuple((state, lift(slot)) for state, slot in final)
        transposed: list[list[tuple[int, int]]] = [[] for _ in arcs]
        for state in graph.order:
            for target, occurrence in arcs[state]:
                transposed[target].append((state, occurrence))
        reverse = _pack_paths(
            list(reversed(graph.order)),
            [tuple(row) for row in transposed],
            stops,
            list(starts),
        )
        return cls(tuple(graph.order), arcs, starts, dict(stops), tuple(slots), reverse)

    def evaluate(
        self, weights: torch.Tensor, exact: list[Fraction | None]
    ) -> tuple[tuple[Fraction, torch.Tensor] | None, int]:
        values = [exact[slot] for slot in self.source_slots]
        work = len(values) + len(self.initial)
        reachable = {state for state, slot in self.initial if values[slot] is not None}
        for state in self.order:
            work += 1
            if state in reachable:
                work += len(self.arcs[state])
                reachable.update(
                    target
                    for target, slot in self.arcs[state]
                    if values[slot] is not None
                )

        reduced = [-torch.inf] * len(values)
        potentials: dict[int, Fraction] = {}
        for state in reversed(self.order):
            work += 1
            if state not in reachable:
                continue
            candidates: list[tuple[int, Fraction]] = []
            if (stop := self.final.get(state)) is not None:
                work += 1
                if (value := values[stop]) is not None:
                    candidates.append((stop, value))
            work += len(self.arcs[state])
            for target, slot in self.arcs[state]:
                if (value := values[slot]) is not None and target in potentials:
                    candidates.append((slot, value + potentials[target]))
            if candidates:
                work += 2 * len(candidates)
                potential = max(value for _, value in candidates)
                potentials[state] = potential
                # The maximum is one of the candidates, so every
                # positive suffix chart row has a zero reduced choice.
                for slot, value in candidates:
                    reduced[slot] = _as_float(
                        value - potential, "a reduced path factor"
                    )

        work += len(self.initial)
        starts = [
            (slot, value + potentials[state])
            for state, slot in self.initial
            if (value := values[slot]) is not None and state in potentials
        ]
        if not starts:
            return None, work
        work += 2 * len(starts)
        maximum = max(value for _, value in starts)
        for slot, value in starts:
            reduced[slot] = _as_float(value - maximum, "a reduced path factor")
        offsets = weights.new_tensor(reduced)
        sources = _take(weights, self.source_slots)
        live = torch.where(torch.isfinite(offsets), sources, 0.0)
        # The detached values encode exact gauges rounded once. Each
        # included occurrence has its own unit derivative to its source.
        balanced = offsets + (live - live.detach())
        residual = self.reverse.evaluate(balanced, LOG_PROB)
        if not bool(torch.isfinite(residual)):
            raise OverflowError("centered log mass is outside binary64 precision")
        return (maximum, residual), work


@dataclass(frozen=True, slots=True)
class _CenteredLogRatio:
    numerator: _CenteredPaths
    denominator: _CenteredPaths
    states: int
    occurrences: int
    weight_slots: int

    @classmethod
    def compile(
        cls,
        topology: RelationTopology,
        total_topology: RelationTopology,
        words: Words,
        *,
        max_states: int,
        max_work: int,
        max_center_occurrences: int,
        max_dyadic_work: int,
    ) -> _CenteredLogRatio:
        for name, limit in (
            ("max_center_occurrences", max_center_occurrences),
            ("max_dyadic_work", max_dyadic_work),
        ):
            if type(limit) is not int or limit < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        part = _active_dag(topology, words, max_states, max_work)
        whole = _active_dag(
            total_topology, ((),) * len(total_topology.tapes), max_states, max_work
        )
        states = sum(len(graph.order) for graph, _ in (part, whole))
        occurrences = sum(
            graph.edge_count + len(graph.initial) + len(final)
            for graph, final in (part, whole)
        )
        weight_slots = len(topology.weight_names)
        if occurrences > max_center_occurrences:
            raise RelationLimitError(
                "centered log ratio exceeds max_center_occurrences"
            )
        # Slot mapping, positive reachability, candidate collection,
        # maximization and reduction each visit an occurrence at most once.
        if weight_slots + 2 * states + 5 * occurrences > max_dyadic_work:
            raise RelationLimitError("centered log ratio exceeds max_dyadic_work")
        return cls(
            _CenteredPaths.pack(*part),
            _CenteredPaths.pack(*whole),
            states,
            occurrences,
            weight_slots,
        )

    def evaluate(self, weights: torch.Tensor) -> CenteredStringLogScore:
        if weights.dtype != torch.float64:
            raise TypeError("centered log ratios require binary64 LogProb weights")
        exact = [
            Fraction(value) if isfinite(value) else None
            for value in weights.detach().cpu().tolist()
        ]
        whole, work = self.denominator.evaluate(weights, exact)
        if whole is None:
            raise ValueError("path operation requires a positive finite normalizer")
        part, part_work = self.numerator.evaluate(weights, exact)
        if part is None:
            score = weights.new_full((), -torch.inf)
        else:
            part_max, part_residual = part
            whole_max, whole_residual = whole
            difference = part_max - whole_max
            difference += Fraction(part_residual.detach().item())
            difference -= Fraction(whole_residual.detach().item())
            nominal = weights.new_tensor(
                _as_float(difference, "the centered log ratio")
            )
            correction = part_residual - whole_residual
            score = nominal + (correction - correction.detach())
        return CenteredStringLogScore(
            score,
            self.states,
            self.occurrences,
            self.weight_slots,
            self.weight_slots + work + part_work,
        )
