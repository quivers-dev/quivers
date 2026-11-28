"""Topology transformations of a weighted rational relation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace

from quivers.core.relations import RationalRelation
from quivers.core.relations.joins import join
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import (
    EPSILON,
    Arc,
    Boundary,
    Label,
    RelationTopology,
)


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


def condition(
    relation: RationalRelation,
    tape: int,
    word: tuple[int, ...],
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Fix the word on one tape and return the relation on the remaining tapes.

    A path contributes its original weight when it realizes the given
    word on the chosen tape. The result preserves the mass of that
    slice. At least one other tape must remain. The limits bound visits
    to state and position configurations and examinations of matching arcs.
    """
    conditioned = _condition_topology(
        relation.topology, tape, word, max_states=max_states, max_work=max_work
    )
    return RationalRelation(conditioned, relation.module(), algebra=relation.algebra)


def _condition_topology(
    topology: RelationTopology,
    tape: int,
    word: tuple[int, ...],
    *,
    max_states: int,
    max_work: int,
    origins: list[int] | None = None,
) -> RelationTopology:
    if type(tape) is not int or not 0 <= tape < len(topology.tapes):
        raise ValueError("conditioning requires a valid tape position")
    if len(topology.tapes) == 1:
        raise ValueError("conditioning requires at least one remaining tape")
    if max_states is None or max_work is None:
        raise ValueError("conditioning requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    topology._validate_words(
        tuple(word if index == tape else () for index in range(len(topology.tapes)))
    )

    if len(topology.initial) > max_states:
        raise RelationLimitError("conditioned relation exceeds max_states")
    groups: dict[tuple[int, Label], list[int]] = defaultdict(list)
    for index, arc in enumerate(topology.arcs):
        groups[arc.source, arc.labels[tape]].append(index)
    configurations = [(mark.state, 0) for mark in topology.initial]
    state_ids = {config: index for index, config in enumerate(configurations)}
    initial = tuple(
        Boundary(index, mark.weight) for index, mark in enumerate(topology.initial)
    )
    final: list[Boundary] = []
    arcs: list[Arc] = []
    for source, (state, position) in enumerate(configurations):
        if (
            position == len(word)
            and (slot := topology._final_slots.get(state)) is not None
        ):
            final.append(Boundary(source, slot))
        labels = (EPSILON, word[position]) if position < len(word) else (EPSILON,)
        for label in labels:
            next_position = position + (label is not EPSILON)
            for index in groups.get((state, label), ()):
                arc = topology.arcs[index]
                if len(arcs) == max_work:
                    raise RelationLimitError("conditioned relation exceeds max_work")
                config = arc.target, next_position
                target = state_ids.get(config)
                if target is None:
                    target = len(configurations)
                    if target == max_states:
                        raise RelationLimitError(
                            "conditioned relation exceeds max_states"
                        )
                    state_ids[config] = target
                    configurations.append(config)
                arcs.append(
                    Arc(
                        source,
                        target,
                        arc.labels[:tape] + arc.labels[tape + 1 :],
                        arc.weight,
                    )
                )
                if origins is not None:
                    origins.append(index)
    return RelationTopology(
        topology.tapes[:tape] + topology.tapes[tape + 1 :],
        topology.input_tapes - (tape < topology.input_tapes),
        len(configurations),
        tuple(arcs),
        initial,
        tuple(final),
        topology.weight_names,
    )


def intersect(
    left: RationalRelation,
    right: RationalRelation,
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Multiply the weights of two weighted languages on their common tape.

    Each operand has one tape, of the same type and on the same side of
    the relation. At each word, the result sums the products of all
    pairs of accepting paths that realize that word. The bounds apply
    to the sparse product construction.
    """
    if any(len(operand.topology.tapes) != 1 for operand in (left, right)):
        raise TypeError("weighted-language intersection requires one tape per operand")
    if left.topology.input_tapes != right.topology.input_tapes:
        raise TypeError(
            "weighted-language intersection requires the same tape orientation"
        )
    return join(
        left,
        right,
        left_tape=0,
        right_tape=0,
        keep_shared=True,
        max_states=max_states,
        max_work=max_work,
    )


__all__ = ["condition", "intersect", "invert", "project", "trim"]
