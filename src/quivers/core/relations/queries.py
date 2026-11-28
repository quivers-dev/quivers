"""Finite observation plans for weighted rational relations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from quivers.core.algebras import Semiring, Star
from quivers.core._closure_block import _ClosureBlock, _make_block
from quivers.core.laws import SEMIRING_LAWS, STAR_LEAST_SOLUTION
from quivers.core.relations._cyclic import _close_block, _components
from quivers.core.relations._packed import _PackedPaths, _pack_paths
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
class _ActiveObservation:
    arcs: tuple[tuple[tuple[int, int], ...], ...]
    initial: tuple[tuple[int, int], ...]
    final: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _PathPlan:
    packed: _PackedPaths
    work: QueryWork
    closures: tuple[_ClosureBlock, ...] = ()
    observation: _ActiveObservation | None = None

    def _closed_weights(
        self, weights: torch.Tensor, algebra: Semiring
    ) -> tuple[torch.Tensor, ...]:
        if not self.closures:
            return ()
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
        return tuple(_close_block(block, weights, algebra) for block in self.closures)

    def evaluate(
        self,
        weights: torch.Tensor,
        algebra: Semiring,
        *,
        visits: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if closed := self._closed_weights(weights, algebra):
            weights = torch.cat((weights, *closed))
        return self.packed.evaluate(weights, algebra, visits=visits)


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


@dataclass(slots=True)
class _Observation:
    states: list[tuple[int, tuple[int, ...]]]
    arcs: list[list[tuple[int, int]]]
    initial: list[tuple[int, int]]
    final: list[tuple[int, int]]
    examined: int


def _observe(
    topology: RelationTopology,
    words: Words,
    *,
    max_states: int | None,
    max_work: int | None,
    arc_ids: bool = False,
) -> _Observation:
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
            payload = arc_id if arc_ids else topology.arcs[arc_id].weight
            arcs[cursor].append((destination_id, payload))
        cursor += 1

    initial = [(index, mark.weight) for index, mark in enumerate(topology.initial)]
    return _Observation(states, arcs, initial, final, examined)


def _compile_paths(
    topology: RelationTopology,
    words: Words,
    *,
    max_states: int | None,
    max_work: int | None,
    max_closure_size: int | None = None,
    max_closure_cells: int | None = None,
    retain_observation: bool = False,
) -> _PathPlan:
    observed = _observe(topology, words, max_states=max_states, max_work=max_work)
    return _trim_paths(
        observed.arcs,
        observed.initial,
        observed.final,
        observed.examined,
        max_closure_size=max_closure_size,
        max_closure_cells=max_closure_cells,
        retain_observation=retain_observation,
        weight_slots=len(topology.weight_names),
    )


@dataclass(slots=True)
class _ProductiveGraph:
    arcs: list[tuple[tuple[int, int], ...]]
    initial: tuple[tuple[int, int], ...]
    order: list[int]
    states: set[int]
    reverse: list[list[int]]
    edge_count: int


def _productive_graph(
    arcs: list[list[tuple[int, int]]],
    initial: list[tuple[int, int]],
    final: list[tuple[int, int]],
) -> _ProductiveGraph:
    # Edge payloads preserve occurrence order and may share a weight slot.
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
    incoming_count = dict.fromkeys(productive, 0)
    active_count = 0
    for state, candidates in enumerate(arcs):
        active = (
            tuple((target, arc) for target, arc in candidates if target in productive)
            if state in productive
            else ()
        )
        active_count += len(active)
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
    active_initial = tuple(mark for mark in initial if mark[0] in productive)
    return _ProductiveGraph(
        active_arcs, active_initial, order, productive, reverse, active_count
    )


def _trim_paths(
    arcs: list[list[tuple[int, int]]],
    initial: list[tuple[int, int]],
    final: list[tuple[int, int]],
    examined: int,
    *,
    max_closure_size: int | None = None,
    max_closure_cells: int | None = None,
    retain_observation: bool = False,
    weight_slots: int = 0,
    visit_order: list[int] | None = None,
    closed_exits: dict[int, int] | None = None,
) -> _PathPlan:
    graph = _productive_graph(arcs, initial, final)
    order = graph.order
    active_arcs = graph.arcs
    productive = graph.states
    closures: tuple[_ClosureBlock, ...] = ()
    original = (
        (tuple(active_arcs), graph.initial, tuple(final))
        if retain_observation
        else None
    )
    components: tuple[tuple[int, ...], ...] = ()
    if len(order) != len(productive):
        # Every emitting arc strictly increases a tape offset, so all
        # arcs inside a cyclic component have epsilon on every tape.
        if max_closure_size is None:
            raise ProductiveEpsilonCycleError(
                "fixed-word path sum requires closure of a productive epsilon cycle"
            )
        order, final, closures, components = _expand_components(
            active_arcs,
            graph.reverse,
            productive,
            final,
            max_closure_size,
            max_closure_cells,
            weight_slots,
            closed_exits if closed_exits is not None else {},
        )

    elif retain_observation:
        components = tuple((state,) for state in order)

    return _PathPlan(
        _pack_paths(order, active_arcs, graph.initial, final, visit_order=visit_order),
        QueryWork(
            len(arcs),
            examined,
            len(productive),
            graph.edge_count,
            tuple(block.order for block in closures),
        ),
        closures,
        _ActiveObservation(*original, components) if original is not None else None,
    )


def _expand_components(
    arcs: list[tuple[tuple[int, int], ...]],
    reverse: list[list[int]],
    productive: set[int],
    final: list[tuple[int, int]],
    max_closure_size: int,
    max_closure_cells: int | None,
    weight_slots: int,
    exits: dict[int, int],
) -> tuple[
    list[int],
    list[tuple[int, int]],
    tuple[_ClosureBlock, ...],
    tuple[tuple[int, ...], ...],
]:
    # Within a component an accepting walk has one entry, then one
    # internal segment, then an exit. Replace all internal segments with
    # the corresponding entries of the local star.
    order: list[int] = []
    closures: list[_ClosureBlock] = []
    cells = 0
    components = tuple(_components(arcs, reverse, productive))
    for members in components:
        local = {state: index for index, state in enumerate(members)}
        edges = [
            (local[source], local[target], slot)
            for source in members
            for target, slot in arcs[source]
            if target in local
        ]
        order.extend(members)
        if not edges:
            continue
        block = _make_block(len(members), edges)
        if block.order > max_closure_size:
            raise RelationLimitError("fixed-word closure exceeds max_closure_size")
        cells += block.order**2
        if max_closure_cells is not None and cells > max_closure_cells:
            raise RelationLimitError("fixed-word closure exceeds max_closure_cells")
        closures.append(block)
        first_exit = len(arcs)
        for source in members:
            exit_ = len(arcs)
            exits[source] = exit_
            order.append(exit_)
            arcs.append(
                tuple(
                    (target, slot)
                    for target, slot in arcs[source]
                    if target not in local
                )
            )
            arcs[source] = tuple(
                (first_exit + column, weight_slots + column)
                for column in range(block.states)
            )
            weight_slots += block.states
    return (
        order,
        [(exits.get(state, state), slot) for state, slot in final],
        tuple(closures),
        components,
    )


__all__ = [
    "ProductiveEpsilonCycleError",
    "QueryWork",
    "RelationLimitError",
]
