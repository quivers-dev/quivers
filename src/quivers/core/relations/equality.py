"""Bounded tape equality when productive paths do not mix cycle signs."""

from __future__ import annotations

from collections import defaultdict

from quivers.core.relations import RationalRelation
from quivers.core.relations._cyclic import _components
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Arc, Boundary, RelationTopology
from quivers.core.relations.transforms import trim

type _Buffer = tuple[int, ...]
type _State = tuple[int, _Buffer, _Buffer]


def _balanced(rows: list[list[tuple[int, int]]], members: set[int], start: int) -> bool:
    potential = {start: 0}
    agenda = [start]
    for source in agenda:
        for target, shift in rows[source]:
            if target not in members:
                continue
            expected = potential[source] + shift
            if target not in potential:
                potential[target] = expected
                agenda.append(target)
            elif potential[target] != expected:
                return False
    return True


def _negative_cycle(
    component: tuple[int, ...],
    edges: list[tuple[int, int, int]],
    direction: int,
    remaining: int,
) -> tuple[bool, int]:
    # A zero-distance virtual source reaches every vertex. An update
    # during the nth pass witnesses a negative cycle of this sign.
    distance = dict.fromkeys(component, 0)
    for _ in component:
        remaining -= len(edges)
        if remaining < 0:
            raise RelationLimitError(
                "auto-intersection cycle analysis exceeds max_work"
            )
        changed = False
        for source, target, shift in edges:
            candidate = distance[source] + direction * shift
            if candidate < distance[target]:
                distance[target] = candidate
                changed = True
        if not changed:
            return False, remaining
    return True, remaining


def _certify_delay(
    topology: RelationTopology, first: int, second: int, remaining: int
) -> tuple[int, int]:
    states = dict.fromkeys(
        (
            *(mark.state for mark in topology.initial),
            *(mark.state for mark in topology.final),
            *(end for arc in topology.arcs for end in (arc.source, arc.target)),
        )
    )
    indexes = {state: i for i, state in enumerate(states)}
    rows: list[list[tuple[int, int]]] = [[] for _ in states]
    reverse: list[list[int]] = [[] for _ in states]
    for arc in topology.arcs:
        source, target = indexes[arc.source], indexes[arc.target]
        shift = (arc.labels[first] is not None) - (arc.labels[second] is not None)
        rows[source].append((target, shift))
        reverse[target].append(source)

    components = tuple(
        _components(list(map(tuple, rows)), reverse, set(range(len(states))))
    )
    component_of = {
        state: index
        for index, component in enumerate(components)
        for state in component
    }
    previous = [0] * len(components)
    for index, component in enumerate(components):
        members = set(component)
        local = 0
        if not _balanced(rows, members, component[0]):
            edges = [
                (state, target, shift)
                for state in component
                for target, shift in rows[state]
                if target in members
            ]
            negative, remaining = _negative_cycle(component, edges, 1, remaining)
            positive, remaining = _negative_cycle(component, edges, -1, remaining)
            local = int(positive) | (int(negative) << 1)
        # Opposite signs in a neutral component's ancestors can be on
        # exclusive branches. Only a signed current component can
        # combine a preceding sign with its opposite on the same path.
        seen = local | previous[index]
        if local and seen == 3:
            raise ValueError(
                "auto-intersection refuses productive paths with both positive and negative cycle delays"
            )
        for state in component:
            for target, _ in rows[state]:
                if (successor := component_of[target]) != index:
                    previous[successor] |= seen

    # Remove cycles independently from any prefix and its suffix.
    # Both simple remainders have length at most Q-1. All removed
    # nonzero cycles along one accepting path have the same sign.
    # A zero total then bounds the original prefix on both sides.
    return max(0, len(states) - 1), remaining


def auto_intersect(
    relation: RationalRelation,
    first: int,
    second: int,
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Keep only paths realizing the same word on two same-type tapes.

    The tapes and primitive weight sources stay in place. Productive
    paths may pass through cycles with nonzero signed length difference
    of a common sign. Opposite signs on the same path are refused.
    The resulting finite graph tracks the suffix of whichever tape is
    ahead. Every created arc consumes exactly one original occurrence.

    After structural trimming, the limits bound the reached residual
    configurations, signed relaxations, and examined arc occurrences.
    """
    topology = relation.topology
    if (
        type(first) is not int
        or type(second) is not int
        or not 0 <= first < len(topology.tapes)
        or not 0 <= second < len(topology.tapes)
        or first == second
    ):
        raise ValueError("auto-intersection requires two distinct valid tape positions")
    if topology.tapes[first] != topology.tapes[second]:
        raise TypeError("auto-intersection requires the same type on both tapes")
    if max_states is None or max_work is None:
        raise ValueError("auto-intersection requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    topology = trim(relation).topology
    if len(topology.initial) > max_states:
        raise RelationLimitError("auto-intersection exceeds max_states")
    delay, remaining = _certify_delay(topology, first, second, max_work)
    native: dict[int, list[Arc]] = defaultdict(list)
    for arc in topology.arcs:
        native[arc.source].append(arc)

    states: list[_State] = [(mark.state, (), ()) for mark in topology.initial]
    indexes = {state: i for i, state in enumerate(states)}
    initial = tuple(Boundary(i, mark.weight) for i, mark in enumerate(topology.initial))
    final: list[Boundary] = []
    arcs: list[Arc] = []
    for source, (state, a, b) in enumerate(states):
        if not a and not b and (slot := topology._final_slots.get(state)) is not None:
            final.append(Boundary(source, slot))
        for arc in native.get(state, ()):
            if remaining == 0:
                raise RelationLimitError("auto-intersection exceeds max_work")
            remaining -= 1
            left, right = arc.labels[first], arc.labels[second]
            a_suffix = a if left is None else (*a, left)
            b_suffix = b if right is None else (*b, right)
            if abs(len(a_suffix) - len(b_suffix)) > delay:
                continue
            common = min(len(a_suffix), len(b_suffix))
            if a_suffix[:common] != b_suffix[:common]:
                continue
            config = arc.target, a_suffix[common:], b_suffix[common:]
            target = indexes.get(config)
            if target is None:
                target = len(states)
                if target == max_states:
                    raise RelationLimitError("auto-intersection exceeds max_states")
                indexes[config] = target
                states.append(config)
            arcs.append(Arc(source, target, arc.labels, arc.weight))
    checked = RelationTopology(
        topology.tapes,
        topology.input_tapes,
        len(states),
        tuple(arcs),
        initial,
        tuple(final),
        topology.weight_names,
    )
    return RationalRelation(checked, relation.module(), algebra=relation.algebra)


__all__ = ["auto_intersect"]
