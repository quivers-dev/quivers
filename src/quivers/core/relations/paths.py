"""Decoding, sampling, and inside–outside statistics for string paths."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from heapq import merge
from operator import add, mul

import torch

from quivers.core.algebras import (
    LOG_PROB,
    MARKOV,
    LogProbAlgebra,
    MarkovAlgebra,
    Semiring,
)
from quivers.core.relations._packed import _take
from quivers.core.relations.queries import (
    ProductiveEpsilonCycleError,
    RelationLimitError,
    _PathPlan,
    _observe,
    _productive_graph,
    _trim_paths,
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
class PathStatistics:
    """Statistics of the conditional distribution over accepting paths.

    The entropy is in nats. The normalizer is represented in log
    space for both input algebras. The expected cost sums the supplied
    feature or loss tensors along each path, including boundary marks.
    """

    log_normalizer: torch.Tensor
    entropy: torch.Tensor
    expected_cost: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _PathEvents:
    original_ids: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]
    denominator: tuple[int, ...]
    slots: tuple[int, ...]

    def costs(
        self,
        probabilities: torch.Tensor,
        feature_groups: tuple[torch.Tensor | None, ...],
        size_groups: tuple[int, ...],
    ) -> torch.Tensor | None:
        expected = None
        for masses, indices, features, size in zip(
            probabilities.split(tuple(map(len, self.original_ids))),
            self.original_ids,
            feature_groups,
            size_groups,
            strict=True,
        ):
            if features is None:
                continue
            if (
                not isinstance(features, torch.Tensor)
                or features.ndim == 0
                or features.shape[0] != size
                or features.device != probabilities.device
                or features.is_complex()
                or not bool(torch.isfinite(features).all())
            ):
                raise ValueError(
                    "each path cost tensor must have one finite real row per occurrence "
                    "on the weight device"
                )
            if expected is not None and features.shape[1:] != expected.shape:
                raise ValueError("path cost tensors must have the same trailing shape")
            scale = masses.reshape((len(indices),) + (1,) * (features.ndim - 1))
            term = (scale * _take(features, indices)).sum(0)
            expected = term if expected is None else expected + term
        if expected is not None and not bool(torch.isfinite(expected).all()):
            raise OverflowError(
                "the expected path cost is outside the evaluation precision"
            )
        return expected


@dataclass(frozen=True, slots=True)
class _Chart:
    plan: _PathPlan
    positions: dict[int, int]
    visit_count: int

    @classmethod
    def build(
        cls,
        arcs: list[list[tuple[int, int]]],
        initial: list[tuple[int, int]],
        final: list[tuple[int, int]],
        max_closure_size: int | None,
        weight_slots: int,
    ) -> _Chart:
        states: list[int] = []
        exits: dict[int, int] = {}
        plan = _trim_paths(
            arcs,
            initial,
            final,
            0,
            max_closure_size=max_closure_size,
            weight_slots=weight_slots,
            visit_order=states,
            closed_exits=exits,
        )
        # Within a closed component, the exit prefix includes every
        # possible internal segment from an entry to that original state.
        visits = {state: position for position, state in enumerate(states)}
        positions = {
            state: visits[exits.get(state, state)]
            for state in states
            if state < len(arcs)
        }
        return cls(plan, positions, len(states))

    def evaluate(
        self, weights: torch.Tensor, algebra: Semiring
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visits: list[torch.Tensor] = []
        total = self.plan.evaluate(weights, algebra, visits=visits)
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


def _markov_log_weights(weights: torch.Tensor) -> torch.Tensor:
    # Preserve zero support without differentiating a logarithm at zero.
    positive = weights > 0
    return torch.where(positive, weights, 1).log().masked_fill(~positive, -torch.inf)


def _require_normalizer(total: torch.Tensor, algebra: Semiring) -> None:
    if not torch.isfinite(total).item() or (algebra is MARKOV and total.item() <= 0):
        raise ValueError("path operation requires a positive finite normalizer")


def _conditional_log_ratio(
    score: torch.Tensor, mass: torch.Tensor, algebra: Semiring
) -> torch.Tensor:
    _require_normalizer(mass, algebra)
    if not algebra.contains(score).item():
        raise OverflowError("a conditional weight is outside the evaluation precision")
    if algebra is LOG_PROB:
        return score - mass
    positive = score > 0
    safe_score = torch.where(positive, score, torch.ones_like(score))
    return torch.where(positive, safe_score.log() - mass.log(), -torch.inf)


class _PathWalks:
    def __init__(
        self,
        topology: RelationTopology,
        words: Words,
        max_states: int | None,
        max_work: int | None,
        max_closure_size: int | None,
    ) -> None:
        observation = _observe(
            topology, words, max_states=max_states, max_work=max_work, arc_ids=True
        )
        graph = _productive_graph(
            observation.arcs, observation.initial, observation.final
        )
        self._cyclic = len(graph.order) != len(graph.states)
        self._max_closure_size = max_closure_size
        self._topology = topology
        self._original = tuple(state for state, _ in observation.states)
        self._order = sorted(graph.states) if self._cyclic else graph.order
        self._arcs = graph.arcs
        self._initial = graph.initial
        self._final = dict(observation.final)

    @cached_property
    def _forward(self) -> _Chart:
        arcs = [
            [(target, self._topology.arcs[arc_id].weight) for target, arc_id in row]
            for row in self._arcs
        ]
        return _Chart.build(
            arcs,
            list(self._initial),
            list(self._final.items()),
            self._max_closure_size,
            len(self._topology.weight_names),
        )

    @cached_property
    def _backward(self) -> _Chart:
        reverse: list[list[tuple[int, int]]] = [[] for _ in self._arcs]
        for source in self._order:
            for target, arc_id in self._arcs[source]:
                reverse[target].append((source, self._topology.arcs[arc_id].weight))
        return _Chart.build(
            reverse,
            list(self._final.items()),
            list(self._initial),
            self._max_closure_size,
            len(self._topology.weight_names),
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

    @staticmethod
    def _prepend(
        weight: Fraction,
        arc_id: int,
        tails: list[_Suffix],
        scale: Callable[[Fraction, Fraction], Fraction],
    ) -> Iterator[_Suffix]:
        for tail in tails:
            yield _Suffix(scale(weight, tail.score), tail.final, arc_id, tail)

    def _suffix_streams(
        self,
        source: int,
        suffixes: dict[int, list[_Suffix]],
        exact_weights: list[Fraction | None],
        scale: Callable[[Fraction, Fraction], Fraction],
    ) -> Iterator[Iterator[_Suffix]]:
        if (slot := self._final.get(source)) is not None:
            weight = exact_weights[slot]
            if weight is not None:
                yield iter((_Suffix(weight, source),))
        for target, arc_id in self._arcs[source]:
            weight = exact_weights[self._topology.arcs[arc_id].weight]
            if weight is not None and suffixes[target]:
                yield self._prepend(weight, arc_id, suffixes[target], scale)

    def k_best(
        self, k: int, weights: torch.Tensor, algebra: Semiring
    ) -> tuple[RelationPath, ...]:
        if self._cyclic:
            raise ProductiveEpsilonCycleError(
                "path ranking requires an acyclic productive graph"
            )
        algebra = _probability_algebra(weights, algebra)
        if k == 0:
            return ()
        # A positive arc scales every suffix score by the same
        # order-preserving operation. Merge only the required heads of
        # these ranked streams, in original stream order on ties.
        zero, scale = (-torch.inf, add) if algebra is LOG_PROB else (0.0, mul)
        values = weights.detach().cpu().tolist()
        exact = [None if value == zero else Fraction(value) for value in values]
        suffixes: dict[int, list[_Suffix]] = {}
        for state in reversed(self._order):
            tails = merge(
                *self._suffix_streams(state, suffixes, exact, scale),
                key=lambda tail: tail.score,
                reverse=True,
            )
            suffixes[state] = [tail for _, tail in zip(range(k), tails, strict=False)]

        def initial_stream(
            state: int, slot: int, weight: Fraction
        ) -> Iterator[tuple[Fraction, int, int, _Suffix]]:
            for tail in suffixes[state]:
                yield scale(weight, tail.score), state, slot, tail

        starts = merge(
            *(
                initial_stream(state, slot, weight)
                for state, slot in self._initial
                if (weight := exact[slot]) is not None
            ),
            key=lambda choice: choice[0],
            reverse=True,
        )
        return tuple(
            self._path(state, slot, tail.arc_ids(), tail.final, weights, algebra)
            for _, (_, state, slot, tail) in zip(range(k), starts, strict=False)
        )

    def sample(
        self,
        weights: torch.Tensor,
        algebra: Semiring,
        generator: torch.Generator | None,
        max_steps: int | None,
    ) -> RelationPath:
        algebra = _probability_algebra(weights, algebra)
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError("max_steps must be a nonnegative integer")
        if self._cyclic and max_steps is None:
            raise ProductiveEpsilonCycleError(
                "cyclic path sampling requires explicit max_steps"
            )
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
            if max_steps is not None and len(arcs) == max_steps:
                raise RelationLimitError("sampled path exceeds max_steps")
            state, arc_id = edges[choice]
            arcs.append(arc_id)

    @cached_property
    def _events(self) -> _PathEvents:
        prefix_count = self._forward.visit_count
        suffix_count = self._backward.visit_count
        prefix = self._forward.positions
        suffix = self._backward.positions
        initial_ids = {
            mark.state: index for index, mark in enumerate(self._topology.initial)
        }
        final_ids = {
            mark.state: index for index, mark in enumerate(self._topology.final)
        }
        original_starts, original_arcs, original_stops = [], [], []
        events: list[tuple[int, int, int, int]] = []
        for state, slot in self._initial:
            original_starts.append(initial_ids[self._original[state]])
            events.append((prefix_count, suffix[state], suffix_count + 1, slot))
        for state in self._order:
            for target, arc_id in self._arcs[state]:
                original_arcs.append(arc_id)
                events.append(
                    (
                        prefix[state],
                        suffix[target],
                        suffix[state],
                        self._topology.arcs[arc_id].weight,
                    )
                )
        for state in self._order:
            if (slot := self._final.get(state)) is not None:
                original_stops.append(final_ids[self._original[state]])
                events.append((prefix[state], suffix_count, suffix[state], slot))
        prefixes, suffixes, denominators, slots = map(tuple, zip(*events, strict=True))
        return _PathEvents(
            (tuple(original_starts), tuple(original_arcs), tuple(original_stops)),
            prefixes,
            suffixes,
            denominators,
            slots,
        )

    def statistics(
        self,
        weights: torch.Tensor,
        algebra: Semiring,
        initial_costs: torch.Tensor | None,
        arc_costs: torch.Tensor | None,
        final_costs: torch.Tensor | None,
    ) -> PathStatistics:
        algebra = _probability_algebra(weights, algebra)
        if algebra is MARKOV:
            weights = _markov_log_weights(weights)
        normalizer, prefix = self._forward.evaluate(weights, LOG_PROB)
        _require_normalizer(normalizer, LOG_PROB)
        _, suffix = self._backward.evaluate(weights, LOG_PROB)
        events = self._events
        unit = weights.new_zeros(1)
        prefix = torch.cat((prefix, unit))
        suffix = torch.cat((suffix, unit, normalizer.reshape(1)))
        local = _take(weights, events.slots) + _take(suffix, events.suffix)
        log_mass = _take(prefix, events.prefix) + local - normalizer
        if not bool(LOG_PROB.contains(log_mass).all()):
            raise OverflowError("posterior paths are outside the evaluation precision")
        supported = torch.isfinite(log_mass)
        choices = local.masked_fill(~supported, 0) - _take(
            suffix, events.denominator
        ).masked_fill(~supported, 0)
        masses = log_mass.exp()
        entropy = -(masses * choices).sum()
        if not bool(torch.isfinite(entropy)):
            raise OverflowError("path entropy is outside the evaluation precision")
        costs = events.costs(
            masses,
            (initial_costs, arc_costs, final_costs),
            (
                len(self._topology.initial),
                len(self._topology.arcs),
                len(self._topology.final),
            ),
        )
        return PathStatistics(normalizer, entropy, costs)

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


__all__ = ["ArcCounts", "PathStatistics", "RelationPath"]
