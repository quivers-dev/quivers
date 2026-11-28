"""Auto-intersection when productive cycles have zero signed tape delay."""

from __future__ import annotations

from collections import defaultdict

from quivers.core.relations import RationalRelation
from quivers.core.relations._cyclic import _components
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Arc, Boundary, RelationTopology
from quivers.core.relations.transforms import trim

type _Buffer = tuple[int, ...]
type _State = tuple[int, _Buffer, _Buffer]


def _require_balanced_cycles(
    topology: RelationTopology, first: int, second: int
) -> None:
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
    for arc_id, arc in enumerate(topology.arcs):
        source, target = indexes[arc.source], indexes[arc.target]
        rows[source].append((target, arc_id))
        reverse[target].append(source)

    for component in _components(
        list(map(tuple, rows)), reverse, set(range(len(states)))
    ):
        members = set(component)
        potential = {component[0]: 0}
        agenda = [component[0]]
        for source in agenda:
            for target, arc_id in rows[source]:
                if target not in members:
                    continue
                arc = topology.arcs[arc_id]
                shift = (arc.labels[first] is not None) - (
                    arc.labels[second] is not None
                )
                expected = potential[source] + shift
                if target not in potential:
                    potential[target] = expected
                    agenda.append(target)
                elif potential[target] != expected:
                    raise ValueError(
                        "auto-intersection requires zero tape delay on every productive cycle"
                    )


def auto_intersect(
    relation: RationalRelation,
    first: int,
    second: int,
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Keep only paths realizing the same word on two same-type tapes.

    The tapes and primitive weight sources stay in place. Each productive
    cycle must have zero signed difference in its emitted tape lengths.
    The resulting finite graph tracks the suffix of whichever tape is
    ahead. Every created arc consumes exactly one original occurrence.

    After structural trimming and the cycle check, the limits bound the
    reached residual configurations and examined arc occurrences.
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
    _require_balanced_cycles(topology, first, second)
    if len(topology.initial) > max_states:
        raise RelationLimitError("auto-intersection exceeds max_states")
    native: dict[int, list[Arc]] = defaultdict(list)
    for arc in topology.arcs:
        native[arc.source].append(arc)

    states: list[_State] = [(mark.state, (), ()) for mark in topology.initial]
    indexes = {state: i for i, state in enumerate(states)}
    initial = tuple(Boundary(i, mark.weight) for i, mark in enumerate(topology.initial))
    final: list[Boundary] = []
    arcs: list[Arc] = []
    work = 0
    for source, (state, a, b) in enumerate(states):
        if not a and not b and (slot := topology._final_slots.get(state)) is not None:
            final.append(Boundary(source, slot))
        for arc in native.get(state, ()):
            if work == max_work:
                raise RelationLimitError("auto-intersection exceeds max_work")
            work += 1
            left, right = arc.labels[first], arc.labels[second]
            a_suffix = a if left is None else (*a, left)
            b_suffix = b if right is None else (*b, right)
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
