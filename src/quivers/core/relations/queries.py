"""Finite observation plans for weighted rational relations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from quivers.core.algebras import Semiring
from quivers.core.relations._packed import _PackedPaths, _pack_paths
from quivers.core.relations.topology import EPSILON, RelationTopology, Words


class ProductiveEpsilonCycleError(ValueError):
    """The observation leaves a cycle on accepting walks."""


class RelationLimitError(ValueError):
    """The available construction budget does not suffice."""


@dataclass(frozen=True, slots=True)
class QueryWork:
    """Counts for construction of a fixed-word path plan."""

    visited_states: int
    candidate_arcs: int
    active_states: int
    active_arcs: int


@dataclass(frozen=True, slots=True)
class _PathPlan:
    packed: _PackedPaths
    work: QueryWork

    def evaluate(self, weights: torch.Tensor, algebra: Semiring) -> torch.Tensor:
        return self.packed.evaluate(weights, algebra)


def _advance(
    words: Words, offsets: tuple[int, ...], topology: RelationTopology, arc_id: int
) -> tuple[int, ...] | None:
    labels = topology.arcs[arc_id].labels
    result: list[int] = []
    for word, offset, label in zip(words, offsets, labels):
        if label is EPSILON:
            result.append(offset)
        elif offset == len(word) or word[offset] != label:
            return None
        else:
            result.append(offset + 1)
    return tuple(result)


def _validate_limits(max_states: int | None, max_work: int | None) -> None:
    for name, limit in (("max_states", max_states), ("max_work", max_work)):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")


def _compile_paths(
    topology: RelationTopology,
    words: Words,
    *,
    max_states: int | None,
    max_work: int | None,
) -> _PathPlan:
    initial_offsets: tuple[int, ...] = (0,) * len(words)
    final_offsets = tuple(map(len, words))
    states: list[tuple[int, tuple[int, ...]]] = [
        (mark.state, initial_offsets) for mark in topology.initial
    ]
    if max_states is not None and len(states) > max_states:
        raise RelationLimitError("fixed-word path plan exceeds max_states")
    state_ids = {state: index for index, state in enumerate(states)}
    arcs: list[list[tuple[int, int]]] = [[] for _ in states]
    reverse: list[list[int]] = [[] for _ in states]
    examined = 0
    final: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(states):
        state, offsets = states[cursor]
        if (
            offsets == final_offsets
            and (slot := topology._final_slots.get(state)) is not None
        ):
            final.append((cursor, slot))
        for arc_id in topology._candidate_arcs(state, words, offsets):
            if max_work is not None and examined == max_work:
                raise RelationLimitError("fixed-word path plan exceeds max_work")
            examined += 1
            next_offsets = _advance(words, offsets, topology, arc_id)
            if next_offsets is None:
                continue
            destination = topology.arcs[arc_id].target, next_offsets
            destination_id = state_ids.get(destination)
            if destination_id is None:
                destination_id = len(states)
                if max_states is not None and destination_id == max_states:
                    raise RelationLimitError("fixed-word path plan exceeds max_states")
                state_ids[destination] = destination_id
                states.append(destination)
                arcs.append([])
                reverse.append([])
            arcs[cursor].append((destination_id, arc_id))
            reverse[destination_id].append(cursor)
        cursor += 1

    productive = {state for state, _ in final}
    agenda = list(productive)
    while agenda:
        state = agenda.pop()
        for source in reverse[state]:
            if source not in productive:
                productive.add(source)
                agenda.append(source)

    active_arcs: list[tuple[tuple[int, int], ...]] = []
    incoming_count = dict.fromkeys(productive, 0)
    for state, candidates in enumerate(arcs):
        active = (
            tuple((target, arc) for target, arc in candidates if target in productive)
            if state in productive
            else ()
        )
        active_arcs.append(active)
        for target, _ in active:
            incoming_count[target] += 1
    ready = deque(
        state
        for state in range(len(states))
        if state in productive and incoming_count[state] == 0
    )
    order: list[int] = []
    while ready:
        state = ready.popleft()
        order.append(state)
        for target, _ in active_arcs[state]:
            incoming_count[target] -= 1
            if incoming_count[target] == 0:
                ready.append(target)
    if len(order) != len(productive):
        # Every emitting arc strictly increases a tape offset, so any
        # cycle here consists entirely of all-epsilon arcs.
        raise ProductiveEpsilonCycleError(
            "fixed-word path sum requires closure of a productive epsilon cycle"
        )
    initial = tuple(
        (index, mark.weight)
        for index, mark in enumerate(topology.initial)
        if index in productive
    )
    return _PathPlan(
        _pack_paths(topology, order, active_arcs, initial, final),
        QueryWork(len(states), examined, len(order), sum(map(len, active_arcs))),
    )


__all__ = [
    "ProductiveEpsilonCycleError",
    "QueryWork",
    "RelationLimitError",
]
