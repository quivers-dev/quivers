"""Exact fixed-input and capped all-input constraints on weighted relations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from heapq import heapify, heappop, heappush
from operator import add

from quivers.core.relations import RationalRelation
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import (
    EPSILON,
    Arc,
    Boundary,
    Label,
    RelationTopology,
)
from quivers.core.relations.transforms import _condition_topology, trim

type _Violations = tuple[int, ...]


def _validate_ranks(
    topology: RelationTopology,
    violations: Sequence[_Violations],
    width: int,
) -> None:
    if len(topology.tapes) != 2 or topology.input_tapes != 1:
        raise TypeError("ranked output requires one input tape and one output tape")
    if len(violations) != len(topology.arcs) or any(
        len(vector) != width or any(type(n) is not int or n < 0 for n in vector)
        for vector in violations
    ):
        raise ValueError("supply one fixed-width nonnegative integer vector per arc")


def _suffix_ranks(
    topology: RelationTopology,
    ranks: list[_Violations],
    zeros: _Violations,
) -> dict[int, _Violations]:
    incoming: dict[int, list[int]] = defaultdict(list)
    for index, arc in enumerate(topology.arcs):
        incoming[arc.target].append(index)
    best = {mark.state: zeros for mark in topology.final}
    pending = [(zeros, state) for state in best]
    heapify(pending)
    while pending:
        distance, state = heappop(pending)
        if best[state] != distance:
            continue
        for index in incoming[state]:
            source = topology.arcs[index].source
            trial = tuple(map(add, ranks[index], distance))
            previous = best.get(source)
            if previous is None or trial < previous:
                best[source] = trial
                heappush(pending, (trial, source))
    return best


def ranked_output_candidates(
    relation: RationalRelation,
    input_word: tuple[int, ...],
    violations: Sequence[tuple[int, ...]],
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Restrict a fixed input to its lexicographically optimal candidate paths.

    A path sums the violation vectors of its arcs. Initial and final
    marks contribute zero. Earlier coordinates take priority over all
    later coordinates. Ranking is independent of numerical path weights.

    Parameters
    ----------
    relation : RationalRelation
        The candidate generator, with one input tape and one output tape.
    input_word : tuple[int, ...]
        The input whose candidates compete.
    violations : Sequence[tuple[int, ...]]
        One vector of nonnegative integers per original arc occurrence.
        All vectors must have the same length.
    max_states : int
        Maximum reachable state and input position configurations.
    max_work : int
        Maximum arcs compatible with the fixed input that the planner
        examines.

    Returns
    -------
    RationalRelation
        An acceptor on the output tape. All and only optimal original
        paths contribute with their original live weights. Zero-vector
        cycles can remain, so scoring may require an algebraic closure.

    Raises
    ------
    TypeError
        If the generator does not have one input and one output tape.
    ValueError
        If the input word, violation table, or construction limits
        are invalid.
    RelationLimitError
        If the fixed-input graph exceeds the state or arc budget.
    """
    topology = relation.topology
    width = len(violations[0]) if violations else 0
    _validate_ranks(topology, violations, width)

    origins: list[int] = []
    fixed = _condition_topology(
        topology,
        0,
        input_word,
        max_states=max_states,
        max_work=max_work,
        origins=origins,
    )
    ranks = [violations[index] for index in origins]
    best = _suffix_ranks(fixed, ranks, (0,) * width)
    optimum = min(
        (best[mark.state] for mark in fixed.initial if mark.state in best),
        default=None,
    )

    # Each retained arc attains equality in the suffix bound. A path
    # starting at an optimal initial and ending at any final telescopes
    # to the optimum. Any optimum must have equality at every step.
    winning = replace(
        fixed,
        initial=tuple(
            mark
            for mark in fixed.initial
            if mark.state in best and best[mark.state] == optimum
        ),
        arcs=tuple(
            arc
            for arc, rank in zip(fixed.arcs, ranks, strict=True)
            if arc.target in best
            and best[arc.source] == tuple(map(add, rank, best[arc.target]))
        ),
    )
    return trim(RationalRelation(winning, relation.module(), algebra=relation.algebra))


type _RankedState = tuple[int, _Violations]
type _Competitors = frozenset[_RankedState]


class _RankedCompiler:
    def __init__(
        self,
        topology: RelationTopology,
        violations: Sequence[_Violations],
        caps: _Violations,
        max_work: int,
    ) -> None:
        self.topology = topology
        self.violations = violations
        self.caps = caps
        self.remaining = max_work
        self.zero = (0,) * len(caps)
        self.outgoing: dict[int, dict[Label, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, arc in enumerate(topology.arcs):
            self.outgoing[arc.source][arc.labels[0]].append(index)
        self.subsets: list[_Competitors] = []
        self.subset_ids: dict[_Competitors, int] = {}
        self.best: list[_Violations | None] = []
        self.transitions: dict[tuple[int, int], int] = {}

    def _spend(self) -> None:
        if self.remaining == 0:
            raise RelationLimitError("ranked compilation exceeds max_work")
        self.remaining -= 1

    def _take(self, state: _RankedState, index: int) -> _RankedState:
        _, profile = state
        return self.topology.arcs[index].target, tuple(
            min(cap, used + count)
            for cap, used, count in zip(
                self.caps, profile, self.violations[index], strict=True
            )
        )

    def _consume(
        self, states: Iterable[_RankedState], label: Label
    ) -> Iterator[_RankedState]:
        for state in states:
            self._spend()
            for index in self.outgoing.get(state[0], {}).get(label, ()):
                self._spend()
                yield self._take(state, index)

    def _close(self, seeds: Iterable[_RankedState]) -> int:
        seen = set(seeds)
        # Process each discovered ranked configuration exactly once.
        pending = list(seen)
        for successor in self._consume(pending, EPSILON):
            if successor not in seen:
                seen.add(successor)
                pending.append(successor)
        subset = frozenset(seen)
        index = self.subset_ids.get(subset)
        if index is None:
            index = len(self.subsets)
            self.subset_ids[subset] = index
            self.subsets.append(subset)
            self.best.append(
                min(
                    (
                        rank
                        for state, rank in subset
                        if state in self.topology._final_slots
                    ),
                    default=None,
                )
            )
        return index

    def _next(self, subset: int, label: int) -> int:
        key = subset, label
        result = self.transitions.get(key)
        if result is None:
            result = self._close(self._consume(self.subsets[subset], label))
            self.transitions[key] = result
        return result

    def weighted_product(self, max_states: int) -> RelationTopology:
        seeds = [(mark.state, self.zero) for mark in self.topology.initial]
        if len(seeds) > max_states:
            raise RelationLimitError("ranked compilation exceeds max_states")
        starting_subset = self._close(seeds)
        states: list[tuple[_RankedState, int]] = [
            (seed, starting_subset) for seed in seeds
        ]
        identifiers = {state: index for index, state in enumerate(states)}
        initial = tuple(
            Boundary(index, mark.weight)
            for index, mark in enumerate(self.topology.initial)
        )
        final: list[Boundary] = []
        arcs: list[Arc] = []
        for source, (candidate, subset) in enumerate(states):
            self._spend()
            state, rank = candidate
            if (
                rank == self.best[subset]
                and (weight := self.topology._final_slots.get(state)) is not None
            ):
                final.append(Boundary(source, weight))
            for label, indices in self.outgoing.get(state, {}).items():
                target_subset = subset if label is None else self._next(subset, label)
                for index in indices:
                    self._spend()
                    config = self._take(candidate, index), target_subset
                    target = identifiers.get(config)
                    if target is None:
                        target = len(states)
                        if target == max_states:
                            raise RelationLimitError(
                                "ranked compilation exceeds max_states"
                            )
                        identifiers[config] = target
                        states.append(config)
                    arc = self.topology.arcs[index]
                    arcs.append(Arc(source, target, arc.labels, arc.weight))
        return replace(
            self.topology,
            n_states=len(states),
            arcs=tuple(arcs),
            initial=initial,
            final=tuple(final),
        )


def compile_ranked_ot(
    relation: RationalRelation,
    violations: Sequence[tuple[int, ...]],
    caps: tuple[int, ...],
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Compile all-input optimum candidates for capped ranked constraints.

    A candidate path sums its violations and caps each coordinate
    separately. Among paths for the same input, all lexicographically
    minimum profiles survive. The finite caps are part of the model.

    Parameters
    ----------
    relation : RationalRelation
        A candidate generator with one input tape and one output tape.
    violations : Sequence[tuple[int, ...]]
        A vector of nonnegative integers per original arc occurrence.
    caps : tuple[int, ...]
        A finite nonnegative cap for each rank. A cap of one makes a
        constraint binary, and a cap of zero ignores that coordinate.
    max_states : int
        Maximum reached states in the weighted product.
    max_work : int
        Maximum product states, original arc examinations and
        competitor configurations examined by the compilation.

    Returns
    -------
    RationalRelation
        A transducer with the original tape types and live weight
        source. Every winning original path occurs exactly once.
        Ranking is independent of numerical path weights.

    Raises
    ------
    TypeError
        If there is no one-input, one-output tape split.
    ValueError
        If the caps, violation vectors or limits are invalid.
    RelationLimitError
        If the weighted product or competitor tracker exceeds a limit.
    """
    topology = relation.topology
    _validate_ranks(topology, violations, len(caps))
    if any(type(cap) is not int or cap < 0 for cap in caps):
        raise ValueError("supply one nonnegative integer cap per rank")
    if max_states is None or max_work is None:
        raise ValueError("ranked compilation requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    compiler = _RankedCompiler(topology, violations, caps, max_work)
    winning = compiler.weighted_product(max_states)
    return trim(RationalRelation(winning, relation.module(), algebra=relation.algebra))


__all__ = ["compile_ranked_ot", "ranked_output_candidates"]
