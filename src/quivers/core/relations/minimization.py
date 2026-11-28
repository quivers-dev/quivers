"""Exact residual-series minimization for static deterministic acceptors."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from math import gcd, lcm
from typing import cast

import torch

from quivers.core.algebras import MARKOV
from quivers.core.relations import RationalRelation
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Arc, Boundary, RelationTopology
from quivers.core.relations.transforms import trim

type _Signature = tuple[Fraction, tuple[tuple[int, Fraction, int], ...]]


@dataclass(frozen=True, slots=True)
class StaticMinimizationWork:
    """Input, positive-support, and refinement counts for static minimization.

    ``input_records`` counts original arc, boundary and weight-slot
    entries. ``active_states`` and ``active_arcs`` count the trimmed
    positive support. ``splitter_states`` and ``predecessor_examinations``
    count state visits and incoming arc visits during refinement, including
    revisits. Their sum with ``input_records`` is the work charge.
    """

    input_records: int
    active_states: int
    active_arcs: int
    splitter_states: int
    predecessor_examinations: int


def _residual_classes(
    topology: RelationTopology,
    values: list[Fraction],
) -> tuple[tuple[_Signature, ...], dict[int, int], dict[int, Fraction]]:
    arcs: dict[int, list[Arc]] = defaultdict(list)
    for arc in topology.arcs:
        arcs[arc.source].append(arc)
    for mark in topology.final:
        arcs.setdefault(mark.state, [])
    dependencies = {
        state: [arc.target for arc in edges] for state, edges in arcs.items()
    }
    try:
        order = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError(
            "static minimization requires acyclic productive support"
        ) from error

    class_of: dict[int, int] = {}
    divisors: dict[int, Fraction] = {}
    classes: dict[_Signature, int] = {}
    for state in order:
        marks = topology._final_slots
        accept = values[marks[state]] if state in marks else Fraction(0)
        entries: list[tuple[int, Fraction, int]] = []
        divisor = accept
        for arc in arcs[state]:
            label = arc.labels[0]
            assert label is not None
            contribution = values[arc.weight] * divisors[arc.target]
            # Distinct labels give disjoint suffix supports. The divisor
            # of their union is the gcd of their separate divisors.
            divisor = Fraction(
                gcd(divisor.numerator, contribution.numerator),
                lcm(divisor.denominator, contribution.denominator),
            )
            entries.append((label, contribution, class_of[arc.target]))
        divisors[state] = divisor
        signature = (
            accept / divisor,
            tuple(
                (label, value / divisor, target)
                for label, value, target in sorted(entries)
            ),
        )
        class_of[state] = classes.setdefault(signature, len(classes))
    return tuple(classes), class_of, divisors


def _render_exact(values: list[Fraction], like: torch.Tensor) -> torch.Tensor:
    try:
        result = like.new_tensor([float(value) for value in values])
    except (OverflowError, RuntimeError) as error:
        raise ValueError(
            "minimized weights are not exactly representable in the input dtype"
        ) from error
    if not bool(torch.isfinite(result).all()) or any(
        Fraction(value) != original
        for value, original in zip(result.tolist(), values, strict=True)
    ):
        raise ValueError(
            "minimized weights are not exactly representable in the input dtype"
        )
    return result


def _static_support(
    relation: RationalRelation,
) -> tuple[RelationTopology, list[Fraction], torch.Tensor]:
    topology = relation.topology
    if relation.algebra is not MARKOV:
        raise TypeError("static acceptor minimization requires the Markov algebra")
    if len(topology.tapes) != 1 or len(topology.initial) > 1:
        raise TypeError(
            "static minimization requires a one-tape acceptor with at most one initial state"
        )
    transitions = {(arc.source, arc.labels[0]) for arc in topology.arcs}
    if any(label is None for _, label in transitions) or len(transitions) != len(
        topology.arcs
    ):
        raise ValueError(
            "static minimization requires epsilon-free deterministic symbol transitions"
        )
    weights = relation._weights()
    if (
        not weights.is_floating_point()
        or weights.requires_grad
        or any(parameter.requires_grad for parameter in relation.module().parameters())
    ):
        raise TypeError(
            "static minimization requires nontrainable floating-point weights"
        )
    values = list(map(Fraction, weights.tolist()))
    nonzero = replace(
        topology,
        arcs=tuple(arc for arc in topology.arcs if values[arc.weight]),
        initial=tuple(mark for mark in topology.initial if values[mark.weight]),
        final=tuple(mark for mark in topology.final if values[mark.weight]),
    )
    positive = trim(
        RationalRelation(nonzero, relation.module(), algebra=MARKOV)
    ).topology
    return positive, values, weights


def _quotient(
    topology: RelationTopology,
    classes: tuple[_Signature, ...],
    class_of: dict[int, int],
    divisors: dict[int, Fraction],
    values: list[Fraction],
    weights: torch.Tensor,
) -> RationalRelation:
    slots: dict[Fraction, int] = {}

    def slot(value: Fraction) -> int:
        return slots.setdefault(value, len(slots))

    initial = tuple(
        Boundary(class_of[mark.state], slot(values[mark.weight] * divisors[mark.state]))
        for mark in topology.initial
    )
    final: list[Boundary] = []
    edges: list[Arc] = []
    for source, (accept, outgoing) in enumerate(classes):
        if accept:
            final.append(Boundary(source, slot(accept)))
        edges.extend(
            Arc(source, target, (label,), slot(weight))
            for label, weight, target in outgoing
        )
    minimized = RelationTopology(
        topology.tapes,
        topology.input_tapes,
        len(classes),
        tuple(edges),
        initial,
        tuple(final),
        tuple(f"coefficient:{i}" for i in range(len(slots))),
    )
    return RationalRelation(
        minimized, _render_exact(list(slots), weights), algebra=MARKOV
    )


def minimize_static_acceptor(relation: RationalRelation) -> RationalRelation:
    """Minimize an epsilon-free deterministic acceptor on its acyclic positive support.

    Each reachable nonzero residual series is factored by the rational
    gcd of its coefficients. Two states share a class exactly when the
    resulting primitive series coincide. The resulting class count
    is minimal among epsilon-free deterministic acceptors with one
    initial state.

    Parameters
    ----------
    relation : RationalRelation
        A one-tape Markov acceptor with at most one initial state and
        nontrainable floating-point weights.

    Returns
    -------
    RationalRelation
        A static snapshot in the input dtype and on the same device.
        Original floating-point coefficients are interpreted as their
        exact binary rationals.

    Raises
    ------
    TypeError
        If the relation has an incompatible algebra, tape or initial
        boundary count, or unsupported weight type.
    ValueError
        If the relation has epsilon arcs, duplicate symbol
        transitions, or a cycle on productive positive support, or
        if a canonical quotient coefficient is unrepresentable.
    """
    positive, values, weights = _static_support(relation)
    classes, class_of, divisors = _residual_classes(positive, values)
    return _quotient(positive, classes, class_of, divisors, values, weights)


def _shortlex_scales(
    outgoing: dict[int, list[Arc]],
    incoming: dict[int, list[Arc]],
    final: dict[int, Fraction],
    values: list[Fraction],
) -> dict[int, Fraction]:
    distance = {state: 0 for state in final}
    pending = deque(final)
    while pending:
        target = pending.popleft()
        for arc in incoming[target]:
            if arc.source not in distance:
                distance[arc.source] = distance[target] + 1
                pending.append(arc.source)

    scales = dict(final)
    # Breadth-first order ensures that every shortest suffix's tail has a scale.
    for state, length in distance.items():
        if length:
            first = min(
                (arc for arc in outgoing[state] if distance[arc.target] == length - 1),
                key=lambda arc: cast(int, arc.labels[0]),
            )
            scales[state] = values[first.weight] * scales[first.target]
    return scales


def _refine(
    final: dict[int, Fraction],
    incoming: dict[int, list[tuple[int, int]]],
    remaining_work: int | None,
) -> tuple[list[set[int]], dict[int, int], int, int]:
    groups: dict[Fraction, set[int]] = defaultdict(set)
    for state, accept in final.items():
        groups[accept].add(state)
    blocks = list(groups.values())
    class_of = {state: i for i, block in enumerate(blocks) for state in block}
    queue = deque(range(len(blocks)))
    state_visits = arc_visits = 0
    while queue:
        splitter = queue.popleft()
        predecessors: dict[int, list[int]] = defaultdict(list)
        # Gather against the entire popped block before any of its members move.
        for target in blocks[splitter]:
            entries = incoming[target]
            state_visits += 1
            arc_visits += len(entries)
            if (
                remaining_work is not None
                and state_visits + arc_visits > remaining_work
            ):
                raise RelationLimitError(
                    "static acceptor minimization exceeds max_work"
                )
            for source, label in entries:
                predecessors[label].append(source)

        for sources in predecessors.values():
            touched: dict[int, set[int]] = defaultdict(set)
            for source in sources:
                touched[class_of[source]].add(source)
            for index, marked in touched.items():
                block = blocks[index]
                if len(marked) == len(block):
                    continue
                # The old index remains on the worklist if it was already queued.
                # Otherwise only the new, smaller half needs to be queued.
                small = marked if 2 * len(marked) <= len(block) else block - marked
                block.difference_update(small)
                created = len(blocks)
                blocks.append(small)
                for state in small:
                    class_of[state] = created
                queue.append(created)
    return blocks, class_of, state_visits, arc_visits


def minimize_static_acceptor_with_work(
    relation: RationalRelation,
    *,
    max_states: int | None = None,
    max_work: int | None = None,
) -> tuple[RationalRelation, StaticMinimizationWork]:
    """Find the smallest static deterministic Markov acceptor, with work counts.

    Every productive state is scaled by the exact weight of its shortest
    accepted suffix, breaking ties by symbol order. A sparse smaller-half
    refinement identifies states whose scaled series coincide. Each
    finite word has the same weight in the quotient, even when the sum
    over all accepted lengths is infinite.

    Parameters
    ----------
    relation : RationalRelation
        A static one-tape Markov acceptor with at most one initial mark,
        epsilon-free deterministic symbols, and floating-point weights.
    max_states : int or None
        Limit on the number of states in the positive productive support.
    max_work : int or None
        Limit on the input records plus state and incoming arc visits
        during refinement. The counters measure structural work, with
        exact rational arithmetic costs dependent on coefficient lengths.

    Returns
    -------
    tuple[RationalRelation, StaticMinimizationWork]
        The detached minimal quotient and the structural work counts.
        Coefficients use the original dtype and device. Comparisons and
        scaling treat source floats as exact binary rationals.

    Raises
    ------
    TypeError
        If the algebra, tape or initial boundary count, or weight type
        is unsupported, or parameters are trainable.
    ValueError
        If limits are invalid, symbol transitions are ambiguous or use
        epsilon, or a shortest-suffix quotient coefficient cannot be
        represented exactly in the original dtype.
    RelationLimitError
        If a state or work limit is exhausted.
    """
    _validate_limits(max_states, max_work)
    source = relation.topology
    records = (
        len(source.arcs)
        + len(source.initial)
        + len(source.final)
        + len(source.weight_names)
    )
    if max_work is not None and records > max_work:
        raise RelationLimitError("static acceptor minimization exceeds max_work")
    topology, values, weights = _static_support(relation)
    final = {mark.state: values[mark.weight] for mark in topology.final}
    states = dict.fromkeys(final)
    for arc in topology.arcs:
        states[arc.source] = states[arc.target] = None
    if max_states is not None and len(states) > max_states:
        raise RelationLimitError("static acceptor minimization exceeds max_states")

    outgoing: dict[int, list[Arc]] = {state: [] for state in states}
    incoming: dict[int, list[Arc]] = {state: [] for state in states}
    for arc in topology.arcs:
        outgoing[arc.source].append(arc)
        incoming[arc.target].append(arc)
    scales = _shortlex_scales(outgoing, incoming, final, values)
    accept = {state: final.get(state, Fraction(0)) / scales[state] for state in states}
    pushed: dict[int, list[tuple[int, Fraction, int]]] = {state: [] for state in states}
    predecessors: dict[int, list[tuple[int, int]]] = {state: [] for state in states}
    labels: dict[tuple[int, Fraction], int] = {}
    for arc in topology.arcs:
        symbol = arc.labels[0]
        assert symbol is not None
        weight = values[arc.weight] * scales[arc.target] / scales[arc.source]
        label = labels.setdefault((symbol, weight), len(labels))
        pushed[arc.source].append((symbol, weight, arc.target))
        predecessors[arc.target].append((arc.source, label))

    remaining = None if max_work is None else max_work - records
    blocks, class_of, splitter_states, predecessor_examinations = _refine(
        accept, predecessors, remaining
    )
    representatives = tuple(map(min, blocks))
    classes = tuple(
        (
            accept[state],
            tuple(
                (symbol, value, class_of[target])
                for symbol, value, target in pushed[state]
            ),
        )
        for state in representatives
    )
    work = StaticMinimizationWork(
        records,
        len(states),
        len(topology.arcs),
        splitter_states,
        predecessor_examinations,
    )
    return _quotient(topology, classes, class_of, scales, values, weights), work


__all__ = [
    "StaticMinimizationWork",
    "minimize_static_acceptor",
    "minimize_static_acceptor_with_work",
]
