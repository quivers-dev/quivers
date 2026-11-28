"""Deterministic fixed-word queries with declared failure arcs."""

from __future__ import annotations

from dataclasses import dataclass

from quivers.core.relations import FixedStringQuery, RationalRelation
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import (
    EPSILON,
    Arc,
    Boundary,
    Label,
    RelationTopology,
)


@dataclass(frozen=True, slots=True)
class FailureArc:
    """A weighted transition taken when a state lacks the requested symbol."""

    source: int
    target: int
    weight: int


def _failure_index(
    topology: RelationTopology, failures: tuple[FailureArc, ...]
) -> dict[int, FailureArc]:
    index: dict[int, FailureArc] = {}
    for failure in failures:
        topology._check_state(failure.source)
        topology._check_state(failure.target)
        topology._check_weight(failure.weight)
        if failure.source in index:
            raise ValueError("backoff permits at most one failure arc at a state")
        index[failure.source] = failure
    acyclic: set[int] = set()
    for root in index:
        state = root
        chain: set[int] = set()
        while state in index and state not in acyclic:
            if state in chain:
                raise ValueError("backoff failure arcs must be acyclic")
            chain.add(state)
            state = index[state].target
        acyclic.update(chain)
    return index


def backoff_query(
    relation: RationalRelation,
    failures: tuple[FailureArc, ...],
    word: tuple[int, ...],
    *,
    max_states: int,
    max_work: int,
) -> FixedStringQuery:
    """Compile one word against an epsilon-free deterministic acceptor.

    An explicit transition takes precedence even if its current weight
    is zero. For a missing symbol, the query multiplies the weights of
    successive failure arcs and then the explicit matching weight, in
    that order. A failure arc never consumes a symbol or acts as a
    final transition.

    The source stays live. Only the selected path is built. Once the
    deterministic topology and acyclic failure links are checked, the
    work limit counts symbol probes and the state limit counts visited
    path configurations.
    """
    topology = relation.topology
    if len(topology.tapes) != 1 or len(topology.initial) > 1:
        raise TypeError(
            "backoff requires a one-tape acceptor with at most one initial state"
        )
    topology._validate_words((word,))
    if max_states is None or max_work is None:
        raise ValueError("backoff requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    by_symbol: dict[tuple[int, int], Arc] = {}
    for arc in topology.arcs:
        label = arc.labels[0]
        if label is None or (arc.source, label) in by_symbol:
            raise ValueError(
                "backoff requires epsilon-free deterministic symbol transitions"
            )
        by_symbol[arc.source, label] = arc
    failure_index = _failure_index(topology, failures)

    n_states = 0
    marks: tuple[Boundary, ...] = ()
    finals: tuple[Boundary, ...] = ()
    arcs: list[Arc] = []
    if topology.initial:
        initial = topology.initial[0]
        state = initial.state
        marks = (Boundary(0, initial.weight),)
        n_states = 1
        if max_states == 0:
            raise RelationLimitError("backoff exceeds max_states")
        offset = 0
        work = 0
        while offset < len(word):
            if work == max_work:
                raise RelationLimitError("backoff exceeds max_work")
            work += 1
            label: Label
            move: Arc | FailureArc | None = by_symbol.get((state, word[offset]))
            if move is not None:
                label = word[offset]
                offset += 1
            else:
                move = failure_index.get(state)
                label = EPSILON
            if move is None:
                break
            if n_states == max_states:
                raise RelationLimitError("backoff exceeds max_states")
            arcs.append(Arc(n_states - 1, n_states, (label,), move.weight))
            n_states += 1
            state = move.target
        if (
            offset == len(word)
            and (weight := topology._final_slots.get(state)) is not None
        ):
            finals = (Boundary(n_states - 1, weight),)

    observed = RelationTopology(
        topology.tapes,
        topology.input_tapes,
        n_states,
        tuple(arcs),
        marks,
        finals,
        topology.weight_names,
    )
    return RationalRelation(
        observed, relation.module(), algebra=relation.algebra
    ).query((word,))


__all__ = ["FailureArc", "backoff_query"]
