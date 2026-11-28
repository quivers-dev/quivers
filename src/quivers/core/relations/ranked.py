"""Exact cast of fixed-input ranked constraints as sparse relation filtering."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from heapq import heapify, heappop, heappush
from operator import add

from quivers.core.relations import RationalRelation
from quivers.core.relations.topology import RelationTopology
from quivers.core.relations.transforms import _condition_topology, trim

type _Violations = tuple[int, ...]


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
    if len(topology.tapes) != 2 or topology.input_tapes != 1:
        raise TypeError("ranked output requires one input tape and one output tape")
    width = len(violations[0]) if violations else 0
    if len(violations) != len(topology.arcs) or any(
        len(vector) != width or any(type(n) is not int or n < 0 for n in vector)
        for vector in violations
    ):
        raise ValueError("supply one fixed-width nonnegative integer vector per arc")

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


__all__ = ["ranked_output_candidates"]
