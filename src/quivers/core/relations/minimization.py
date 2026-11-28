"""Exact residual-series minimization for static acyclic acceptors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from math import gcd, lcm

import torch

from quivers.core.algebras import MARKOV
from quivers.core.relations import RationalRelation
from quivers.core.relations.topology import Arc, Boundary, RelationTopology
from quivers.core.relations.transforms import trim

type _Signature = tuple[Fraction, tuple[tuple[int, Fraction, int], ...]]


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
    classes, class_of, divisors = _residual_classes(positive, values)

    slots: dict[Fraction, int] = {}

    def slot(value: Fraction) -> int:
        return slots.setdefault(value, len(slots))

    initial = tuple(
        Boundary(class_of[mark.state], slot(values[mark.weight] * divisors[mark.state]))
        for mark in positive.initial
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


__all__ = ["minimize_static_acceptor"]
