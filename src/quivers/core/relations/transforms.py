"""Topology transformations of a weighted rational relation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace

from quivers.core.relations import RationalRelation


def _select_tapes(
    relation: RationalRelation, positions: tuple[int, ...], input_tapes: int
) -> RationalRelation:
    topology = relation.topology
    transformed = replace(
        topology,
        tapes=tuple(topology.tapes[index] for index in positions),
        input_tapes=input_tapes,
        arcs=tuple(
            replace(arc, labels=tuple(arc.labels[index] for index in positions))
            for arc in topology.arcs
        ),
    )
    return RationalRelation(transformed, relation.module(), algebra=relation.algebra)


def invert(relation: RationalRelation) -> RationalRelation:
    """Exchange the input and output tapes of a relation.

    The order within each side and all arc and boundary weights are
    preserved. Each path realizes the exchanged word tuple.
    """
    topology = relation.topology
    split = topology.input_tapes
    count = len(topology.tapes)
    positions = (*range(split, count), *range(split))
    return _select_tapes(relation, positions, count - split)


def project(relation: RationalRelation, positions: tuple[int, ...]) -> RationalRelation:
    """Retain the supplied original tape positions in increasing order.

    Each arc retains its occurrence and weight. Erasing a tape sums over
    the words realized on it, so projection can expose a productive
    epsilon cycle on the retained tapes. At least one tape must remain.
    """
    topology = relation.topology
    if not positions or any(
        type(index) is not int or not 0 <= index < len(topology.tapes)
        for index in positions
    ):
        raise ValueError("projection requires at least one valid retained tape")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("projection requires strictly increasing tape positions")
    input_tapes = sum(index < topology.input_tapes for index in positions)
    return _select_tapes(relation, positions, input_tapes)


def _reached(roots: Iterable[int], successors: Mapping[int, list[int]]) -> set[int]:
    reached = set(roots)
    pending = list(reached)
    while pending:
        state = pending.pop()
        for target in successors.get(state, ()):
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return reached


def trim(relation: RationalRelation) -> RationalRelation:
    """Retain the arcs and boundary marks on structurally accepting walks.

    The retained arcs keep their original relative order and distinct
    occurrences. State IDs and weight slots are retained. Structural
    reachability is independent of the current weights.
    """
    topology = relation.topology
    forward: dict[int, list[int]] = defaultdict(list)
    backward: dict[int, list[int]] = defaultdict(list)
    for arc in topology.arcs:
        forward[arc.source].append(arc.target)
        backward[arc.target].append(arc.source)
    productive = _reached((mark.state for mark in topology.initial), forward)
    productive &= _reached((mark.state for mark in topology.final), backward)
    trimmed = replace(
        topology,
        arcs=tuple(
            arc
            for arc in topology.arcs
            if arc.source in productive and arc.target in productive
        ),
        initial=tuple(mark for mark in topology.initial if mark.state in productive),
        final=tuple(mark for mark in topology.final if mark.state in productive),
    )
    return RationalRelation(trimmed, relation.module(), algebra=relation.algebra)


__all__ = ["invert", "project", "trim"]
