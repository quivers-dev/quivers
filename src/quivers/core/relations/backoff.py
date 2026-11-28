"""Deterministic acceptor queries with declared failure arcs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

import torch

from quivers.core.algebras import Semiring
from quivers.core.laws import SEMIRING_LAWS
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


def _ordinary_index(topology: RelationTopology) -> dict[tuple[int, int], Arc]:
    by_symbol: dict[tuple[int, int], Arc] = {}
    for arc in topology.arcs:
        label = arc.labels[0]
        if label is None or (arc.source, label) in by_symbol:
            raise ValueError(
                "backoff requires epsilon-free deterministic symbol transitions"
            )
        by_symbol[arc.source, label] = arc
    return by_symbol


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
    by_symbol = _ordinary_index(topology)
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


@dataclass(frozen=True, slots=True)
class _SymbolSums:
    total: torch.Tensor
    left: _SymbolSums | None = None
    right: _SymbolSums | None = None
    pending: torch.Tensor | None = None


class _SymbolTree:
    def __init__(self, algebra: Semiring, charge: Callable[[int], None]) -> None:
        self._algebra = algebra
        self._charge = charge

    def add(
        self, first: torch.Tensor | None, second: torch.Tensor | None
    ) -> torch.Tensor | None:
        if first is None:
            return second
        if second is None:
            return first
        return self._algebra.join(torch.stack((first, second)), dim=0)

    def scale(
        self, factor: torch.Tensor, node: _SymbolSums | None
    ) -> _SymbolSums | None:
        if node is None:
            return None
        self._charge(1)
        product = self._algebra.tensor_op
        pending = None
        if node.left is not None or node.right is not None:
            pending = factor if node.pending is None else product(factor, node.pending)
        return _SymbolSums(product(factor, node.total), node.left, node.right, pending)

    def override(
        self,
        node: _SymbolSums | None,
        position: int,
        value: torch.Tensor,
        start: int,
        stop: int,
    ) -> _SymbolSums:
        self._charge(1)
        if stop - start == 1:
            return _SymbolSums(value)
        left, right = (None, None) if node is None else (node.left, node.right)
        if node is not None and node.pending is not None:
            left, right = (
                self.scale(node.pending, left),
                self.scale(node.pending, right),
            )
        middle = (start + stop) // 2
        if position < middle:
            left = self.override(left, position, value, start, middle)
        else:
            right = self.override(right, position, value, middle, stop)
        total = self.add(
            None if left is None else left.total,
            None if right is None else right.total,
        )
        assert total is not None
        return _SymbolSums(total, left, right)


def backoff_total_weight(
    relation: RationalRelation,
    failures: tuple[FailureArc, ...],
    *,
    max_states: int,
    max_work: int,
) -> torch.Tensor:
    """Sum all accepted word weights of an acyclic deterministic backoff acceptor.

    An explicit symbol transition takes precedence over the fallback,
    including at zero weight. A state can finish only by its own final
    mark. The union of reachable ordinary and failure edges must be
    acyclic. The transition and boundary values come from the original
    weight source.

    Parameters
    ----------
    relation : RationalRelation
        An epsilon-free deterministic acceptor with at most one
        initial state and an established scalar semiring.
    failures : tuple[FailureArc, ...]
        The weighted fallback at each state, with at most one per state.
    max_states : int
        Maximum states reached in the combined transition graph.
    max_work : int
        Maximum examined reachable base edges and persistent sum-tree
        node updates.

    Returns
    -------
    torch.Tensor
        The total weight in the original semiring, with live derivatives
        where the algebra supports autograd.

    Raises
    ------
    TypeError
        If the tapes, starting boundary, or scalar algebra are unsupported.
    ValueError
        If the graph is cyclic or its transition records are invalid.
    RelationLimitError
        If an explicit construction limit is exceeded.
    """
    topology = relation.topology
    algebra = relation.algebra
    if len(topology.tapes) != 1 or len(topology.initial) > 1:
        raise TypeError(
            "backoff requires a one-tape acceptor with at most one initial state"
        )
    if not algebra.establishes(*SEMIRING_LAWS):
        raise TypeError("backoff total weight requires an established scalar semiring")
    if max_states is None or max_work is None:
        raise ValueError("backoff requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    ordinary = _ordinary_index(topology)
    fallback = _failure_index(topology, failures)
    outgoing: dict[int, list[tuple[int, Arc]]] = defaultdict(list)
    for (state, symbol), arc in ordinary.items():
        outgoing[state].append((symbol, arc))
    weights = relation._weights()
    zero = algebra.zero_like((), weights)
    if not topology.initial:
        return zero

    initial = topology.initial[0]
    remaining = max_work

    def charge(work: int) -> None:
        nonlocal remaining
        remaining -= work
        if remaining < 0:
            raise RelationLimitError("backoff total weight exceeds max_work")

    states = [initial.state]
    reached = set(states)
    dependencies: dict[int, list[int]] = {}
    if max_states == 0:
        raise RelationLimitError("backoff total weight exceeds max_states")
    for state in states:
        children = [arc.target for _, arc in outgoing[state]]
        if (failure := fallback.get(state)) is not None:
            children.append(failure.target)
        charge(len(children))
        dependencies[state] = children
        for target in children:
            if target not in reached:
                if len(states) == max_states:
                    raise RelationLimitError("backoff total weight exceeds max_states")
                reached.add(target)
                states.append(target)
    try:
        order = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError(
            "backoff total weight requires acyclic ordinary and failure edges"
        ) from error

    positions = dict.fromkeys(symbol for q in states for symbol, _ in outgoing[q])
    symbol_indices = {symbol: index for index, symbol in enumerate(positions)}
    tree = _SymbolTree(algebra, charge)
    roots: dict[int, _SymbolSums | None] = {}
    totals: dict[int, torch.Tensor] = {}
    for state in order:
        failure = fallback.get(state)
        root = (
            None
            if failure is None
            else tree.scale(weights[failure.weight], roots[failure.target])
        )
        for symbol, arc in outgoing[state]:
            weight = algebra.tensor_op(weights[arc.weight], totals[arc.target])
            root = tree.override(
                root, symbol_indices[symbol], weight, 0, len(symbol_indices)
            )
        roots[state] = root
        slot = topology._final_slots.get(state)
        total = tree.add(
            None if root is None else root.total,
            None if slot is None else weights[slot],
        )
        totals[state] = zero if total is None else total
    return algebra.tensor_op(weights[initial.weight], totals[initial.state])


__all__ = ["FailureArc", "backoff_query", "backoff_total_weight"]
