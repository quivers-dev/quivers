"""Decoding, sampling and inside–outside counts for acyclic observations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from heapq import nlargest
from operator import add, mul

import torch

from quivers.core.algebras import (
    LOG_PROB,
    MARKOV,
    LogProbAlgebra,
    MarkovAlgebra,
    Semiring,
)
from quivers.core.relations._packed import _PackedPaths, _pack_paths, _take
from quivers.core.relations.queries import (
    ProductiveEpsilonCycleError,
    _observe,
    _productive_graph,
)
from quivers.core.relations.topology import RelationTopology, Words


@dataclass(frozen=True, slots=True)
class RelationPath:
    """An accepting path, including its original arc occurrence IDs.

    The initial and final states distinguish paths with no arcs. The
    weight is in the relation's algebra, with floating-point rounding
    at the input dtype, and retains its derivatives.
    """

    initial_state: int
    arc_ids: tuple[int, ...]
    final_state: int
    weight: torch.Tensor


@dataclass(frozen=True, slots=True)
class ArcCounts:
    """Sparse expected traversal counts conditional on the observed words.

    The one-dimensional counts tensor aligns with the original arc IDs
    and uses ordinary nonnegative reals. The normalizer is the observed
    path sum in the relation's algebra.
    """

    arc_ids: tuple[int, ...]
    counts: torch.Tensor
    normalizer: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Chart:
    packed: _PackedPaths
    positions: dict[int, int]

    @classmethod
    def build(
        cls,
        order: list[int],
        arcs: list[tuple[tuple[int, int], ...]],
        initial: tuple[tuple[int, int], ...],
        final: list[tuple[int, int]],
    ) -> _Chart:
        states: list[int] = []
        packed = _pack_paths(order, arcs, initial, final, visit_order=states)
        return cls(packed, {state: position for position, state in enumerate(states)})

    def evaluate(
        self, weights: torch.Tensor, algebra: Semiring
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visits: list[torch.Tensor] = []
        total = self.packed.evaluate(weights, algebra, visits=visits)
        return total, torch.cat(visits) if visits else weights[:0]


@dataclass(frozen=True, slots=True)
class _Suffix:
    score: Fraction
    final: int
    arc_id: int | None = None
    tail: _Suffix | None = None

    def arc_ids(self) -> tuple[int, ...]:
        arcs: list[int] = []
        suffix = self
        while suffix.tail is not None and suffix.arc_id is not None:
            arcs.append(suffix.arc_id)
            suffix = suffix.tail
        return tuple(arcs)


def _probability_algebra(weights: torch.Tensor, algebra: Semiring) -> Semiring:
    if type(algebra) not in (MarkovAlgebra, LogProbAlgebra):
        raise TypeError("path operations require Markov or LogProb weights")
    if not weights.is_floating_point():
        raise TypeError("path operations require floating-point probability weights")
    return LOG_PROB if type(algebra) is LogProbAlgebra else MARKOV


def _require_normalizer(total: torch.Tensor, algebra: Semiring) -> None:
    if not torch.isfinite(total).item() or (algebra is MARKOV and total.item() <= 0):
        raise ValueError("path operation requires a positive finite normalizer")


class _AcyclicWalks:
    def __init__(
        self,
        topology: RelationTopology,
        words: Words,
        max_states: int | None,
        max_work: int | None,
    ) -> None:
        observation = _observe(
            topology, words, max_states=max_states, max_work=max_work, arc_ids=True
        )
        graph = _productive_graph(
            observation.arcs, observation.initial, observation.final
        )
        if len(graph.order) != len(graph.states):
            raise ProductiveEpsilonCycleError(
                "path decoding, sampling and counts require an acyclic productive graph"
            )
        self._topology = topology
        self._original = tuple(state for state, _ in observation.states)
        self._order = graph.order
        self._arcs = graph.arcs
        self._initial = graph.initial
        self._final = dict(observation.final)

    @cached_property
    def _forward(self) -> _Chart:
        arcs = [
            tuple(
                (target, self._topology.arcs[arc_id].weight) for target, arc_id in row
            )
            for row in self._arcs
        ]
        return _Chart.build(self._order, arcs, self._initial, list(self._final.items()))

    @cached_property
    def _backward(self) -> _Chart:
        reverse: list[list[tuple[int, int]]] = [[] for _ in self._arcs]
        for source in self._order:
            for target, arc_id in self._arcs[source]:
                reverse[target].append((source, self._topology.arcs[arc_id].weight))
        return _Chart.build(
            self._order[::-1],
            list(map(tuple, reverse)),
            tuple(self._final.items()),
            list(self._initial),
        )

    @cached_property
    def _occurrences(
        self,
    ) -> tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        # The same original arc can occur at more than one observed offset.
        # Keep its appearances in each chart separate up to normalization.
        arc_index: dict[int, int] = {}
        rows: list[tuple[int, int, int, int]] = []
        for source in self._order:
            for target, arc_id in self._arcs[source]:
                rows.append(
                    (
                        self._forward.positions[source],
                        self._backward.positions[target],
                        self._topology.arcs[arc_id].weight,
                        arc_index.setdefault(arc_id, len(arc_index)),
                    )
                )
        if not rows:
            return (), (), (), (), ()
        starts, ends, slots, destinations = map(tuple, zip(*rows, strict=True))
        return tuple(arc_index), starts, ends, slots, destinations

    def _path(
        self,
        start: int,
        initial_slot: int,
        arcs: tuple[int, ...],
        end: int,
        weights: torch.Tensor,
        algebra: Semiring,
    ) -> RelationPath:
        slots = (
            initial_slot,
            *(self._topology.arcs[arc_id].weight for arc_id in arcs),
            self._final[end],
        )
        values = _take(weights, slots)
        score = values.sum() if algebra is LOG_PROB else values.prod()
        if not torch.isfinite(score).item() or (
            algebra is MARKOV and score.item() == 0
        ):
            raise OverflowError("positive path weight is outside the input precision")
        return RelationPath(self._original[start], arcs, self._original[end], score)

    @staticmethod
    def _log_weights(weights: torch.Tensor, algebra: Semiring) -> torch.Tensor:
        values = weights.detach().to(device="cpu", dtype=torch.float64)
        return values if algebra is LOG_PROB else values.log()

    def _options(
        self,
        source: int,
        suffixes: dict[int, list[_Suffix]],
        exact_weights: list[Fraction | None],
        scale: Callable[[Fraction, Fraction], Fraction],
    ) -> Iterator[_Suffix]:
        if (slot := self._final.get(source)) is not None:
            weight = exact_weights[slot]
            if weight is not None:
                yield _Suffix(weight, source)
        for target, arc_id in self._arcs[source]:
            weight = exact_weights[self._topology.arcs[arc_id].weight]
            if weight is not None:
                for tail in suffixes[target]:
                    yield _Suffix(scale(weight, tail.score), tail.final, arc_id, tail)

    def k_best(
        self, k: int, weights: torch.Tensor, algebra: Semiring
    ) -> tuple[RelationPath, ...]:
        algebra = _probability_algebra(weights, algebra)
        if k == 0:
            return ()
        # Binary floating inputs are exact rationals. Compare their products
        # (sums in log space) before any rounding of a selected path score.
        zero, scale = (-torch.inf, add) if algebra is LOG_PROB else (0.0, mul)
        values = weights.detach().cpu().tolist()
        exact = [None if value == zero else Fraction(value) for value in values]
        suffixes: dict[int, list[_Suffix]] = {}
        for state in reversed(self._order):
            suffixes[state] = nlargest(
                k,
                self._options(state, suffixes, exact, scale),
                key=lambda tail: tail.score,
            )
        starts = (
            (scale(weight, tail.score), state, slot, tail)
            for state, slot in self._initial
            if (weight := exact[slot]) is not None
            for tail in suffixes[state]
        )
        return tuple(
            self._path(state, slot, tail.arc_ids(), tail.final, weights, algebra)
            for _, state, slot, tail in nlargest(
                k, starts, key=lambda choice: choice[0]
            )
        )

    def sample(
        self,
        weights: torch.Tensor,
        algebra: Semiring,
        generator: torch.Generator | None,
    ) -> RelationPath:
        algebra = _probability_algebra(weights, algebra)
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("path sampling requires a CPU generator")
        log_weights = self._log_weights(weights, algebra)
        normalizer, suffix = self._backward.evaluate(log_weights, LOG_PROB)
        _require_normalizer(normalizer, LOG_PROB)
        start_logits = torch.stack(
            [
                log_weights[slot] + suffix[self._backward.positions[state]]
                for state, slot in self._initial
            ]
        )
        initial = int(
            torch.multinomial(start_logits.softmax(0), 1, generator=generator).item()
        )
        start, initial_slot = self._initial[initial]
        state = start
        arcs: list[int] = []
        while True:
            edges = self._arcs[state]
            stop_slot = self._final.get(state)
            logits = [
                log_weights[self._topology.arcs[arc_id].weight]
                + suffix[self._backward.positions[target]]
                for target, arc_id in edges
            ]
            if stop_slot is not None:
                logits.append(log_weights[stop_slot])
            choice = int(
                torch.multinomial(
                    torch.stack(logits).softmax(0), 1, generator=generator
                ).item()
            )
            if choice == len(edges):
                return self._path(
                    start, initial_slot, tuple(arcs), state, weights, algebra
                )
            state, arc_id = edges[choice]
            arcs.append(arc_id)

    def counts(self, weights: torch.Tensor, algebra: Semiring) -> ArcCounts:
        algebra = _probability_algebra(weights, algebra)
        normalizer, prefix = self._forward.evaluate(weights, algebra)
        _require_normalizer(normalizer, algebra)
        _, suffix = self._backward.evaluate(weights, algebra)
        ids, starts, ends, slots, destinations = self._occurrences
        terms = algebra.tensor_op(
            algebra.tensor_op(_take(prefix, starts), _take(weights, slots)),
            _take(suffix, ends),
        )
        normalized = (
            (terms - normalizer).exp() if algebra is LOG_PROB else terms / normalizer
        )
        if not torch.isfinite(normalized).all().item():
            raise OverflowError(
                "path expectations are outside the evaluation precision"
            )
        index = torch.tensor(destinations, dtype=torch.long, device=weights.device)
        counts = normalized.new_zeros((len(ids),)).index_add(0, index, normalized)
        return ArcCounts(ids, counts, normalizer)


__all__ = ["ArcCounts", "RelationPath"]
