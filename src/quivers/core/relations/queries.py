"""Finite observation plans for weighted rational relations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from quivers.core.algebras import Semiring, Star, UndecidedClosureError
from quivers.core.laws import SEMIRING_LAWS, STAR_LEAST_SOLUTION
from quivers.core.relations._packed import _PackedPaths, _pack_paths, _take
from quivers.core.relations.topology import EPSILON, RelationTopology, Words


class ProductiveEpsilonCycleError(ValueError):
    """The observation has a productive cycle outside the supported closure."""


class RelationLimitError(ValueError):
    """The available construction budget does not suffice."""


@dataclass(frozen=True, slots=True)
class QueryWork:
    """Graph counts and planned star matrix orders for a fixed-word query.

    State and arc counts refer to the observation graph. Each entry of
    `closure_sizes` is the order of a local matrix, including any
    auxiliary vertices.
    """

    visited_states: int
    candidate_arcs: int
    active_states: int
    active_arcs: int
    closure_sizes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _PathPlan:
    packed: _PackedPaths
    work: QueryWork
    loops: tuple[tuple[int, ...], ...] = ()

    def evaluate(self, weights: torch.Tensor, algebra: Semiring) -> torch.Tensor:
        if self.loops:
            if not isinstance(algebra, Star) or not algebra.establishes(
                *SEMIRING_LAWS, STAR_LEAST_SOLUTION
            ):
                raise ProductiveEpsilonCycleError(
                    "a productive epsilon cycle requires an established semiring star"
                )
            if not weights.is_floating_point():
                raise TypeError(
                    "productive epsilon closure requires floating-point weights"
                )
            closed = torch.stack(
                [_close_loops(weights, slots, algebra) for slots in self.loops]
            )
            weights = torch.cat((weights, closed))
        return self.packed.evaluate(weights, algebra)


def _close_loops(
    weights: torch.Tensor, slots: tuple[int, ...], algebra: Star
) -> torch.Tensor:
    # Keep one weight per entry in the solver's initial matrix. Additional
    # parallel loops visit a distinct auxiliary node and then return with
    # unit weight. Pre-summing the entries could round across the solver's
    # convergence boundary before it had a chance to enclose the sum.
    auxiliary = len(slots) - 1
    top = torch.cat(
        (
            algebra.zero_like((auxiliary, auxiliary), weights),
            algebra.unit_like((auxiliary, 1), weights),
        ),
        dim=1,
    )
    matrix = torch.cat((top, _take(weights, slots).unsqueeze(0)), dim=0)
    repeat = algebra.star(matrix)[-1, -1]
    if not bool(torch.isfinite(repeat)):
        raise UndecidedClosureError(
            "productive epsilon closure did not yield a finite weight"
        )
    return repeat


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


def _validate_limits(
    max_states: int | None,
    max_work: int | None,
    max_closure_size: int | None = None,
) -> None:
    for name, limit in (
        ("max_states", max_states),
        ("max_work", max_work),
        ("max_closure_size", max_closure_size),
    ):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")


def _compile_paths(
    topology: RelationTopology,
    words: Words,
    *,
    max_states: int | None,
    max_work: int | None,
    max_closure_size: int | None = None,
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
            arcs[cursor].append((destination_id, topology.arcs[arc_id].weight))
        cursor += 1

    initial = [(index, mark.weight) for index, mark in enumerate(topology.initial)]
    return _trim_paths(
        arcs,
        initial,
        final,
        examined,
        max_closure_size=max_closure_size,
        weight_slots=len(topology.weight_names),
    )


def _trim_paths(
    arcs: list[list[tuple[int, int]]],
    initial: list[tuple[int, int]],
    final: list[tuple[int, int]],
    examined: int,
    *,
    max_closure_size: int | None = None,
    weight_slots: int = 0,
) -> _PathPlan:
    # An edge is an occurrence of a (target, weight slot) record. Several
    # occurrences can have the same target and slot.
    reverse: list[list[int]] = [[] for _ in arcs]
    for source, edges in enumerate(arcs):
        for target, _ in edges:
            reverse[target].append(source)
    productive = {state for state, _ in final}
    agenda = list(productive)
    while agenda:
        state = agenda.pop()
        for source in reverse[state]:
            if source not in productive:
                productive.add(source)
                agenda.append(source)

    active_arcs: list[tuple[tuple[int, int], ...]] = []
    self_loops: dict[int, tuple[int, ...]] = {}
    incoming_count = dict.fromkeys(productive, 0)
    active_count = 0
    for state, candidates in enumerate(arcs):
        active = (
            tuple((target, arc) for target, arc in candidates if target in productive)
            if state in productive
            else ()
        )
        active_count += len(active)
        if max_closure_size is not None:
            loops = tuple(slot for target, slot in active if target == state)
            if loops:
                self_loops[state] = loops
                active = tuple(
                    (target, slot) for target, slot in active if target != state
                )
        active_arcs.append(active)
        for target, _ in active:
            incoming_count[target] += 1
    ready = deque(
        state
        for state in range(len(arcs))
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
        message = (
            "fixed-word path sum requires closure of a productive epsilon cycle"
            if max_closure_size is None
            else "fixed-word closure supports only singleton productive epsilon components"
        )
        raise ProductiveEpsilonCycleError(message)
    if max_closure_size is not None and any(
        len(slots) > max_closure_size for slots in self_loops.values()
    ):
        raise RelationLimitError("fixed-word closure exceeds max_closure_size")

    active_initial = tuple(mark for mark in initial if mark[0] in productive)
    closure_slots: list[tuple[int, ...]] = []
    if self_loops:
        # Split a state into entry and exit with one closure edge. A path
        # takes that edge before emitting another symbol or accepting.
        expanded_order: list[int] = []
        exits: dict[int, int] = {}
        for state in order:
            expanded_order.append(state)
            if (slots := self_loops.get(state)) is not None:
                exit_ = len(active_arcs)
                exits[state] = exit_
                expanded_order.append(exit_)
                active_arcs.append(active_arcs[state])
                active_arcs[state] = ((exit_, weight_slots + len(closure_slots)),)
                closure_slots.append(slots)
        final = [(exits.get(state, state), slot) for state, slot in final]
        order = expanded_order

    return _PathPlan(
        _pack_paths(order, active_arcs, active_initial, final),
        QueryWork(
            len(arcs),
            examined,
            len(productive),
            active_count,
            tuple(map(len, closure_slots)),
        ),
        tuple(closure_slots),
    )


__all__ = [
    "ProductiveEpsilonCycleError",
    "QueryWork",
    "RelationLimitError",
]
