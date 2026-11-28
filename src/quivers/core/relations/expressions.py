"""Union, concatenation and repetition of weighted rational relations."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from quivers.core.relations import RationalRelation
from quivers.core.relations.calculus import _read_weight_pair
from quivers.core.relations.topology import EPSILON, Arc, Boundary, RelationTopology


class _ExpressionWeights(nn.Module):
    def __init__(
        self, left: RationalRelation, right: RationalRelation | None = None
    ) -> None:
        super().__init__()
        self._left = left
        self._right = right
        sources = (left.module(),)
        if right is not None and right.module() is not sources[0]:
            sources = (*sources, right.module())
        self.sources = nn.ModuleList(sources)

    def forward(self) -> torch.Tensor:
        if self._right is None:
            left = self._left._weights()
            return torch.cat((left, self._left.algebra.unit_like((1,), left)))
        left, right = _read_weight_pair(self._left, self._right)
        if left.dtype != right.dtype or left.device != right.device:
            raise ValueError(
                "relation expressions require a common weight dtype and device"
            )
        return torch.cat((left, right))


def _offset_boundary(
    marks: tuple[Boundary, ...], states: int, slots: int
) -> tuple[Boundary, ...]:
    return tuple(Boundary(mark.state + states, mark.weight + slots) for mark in marks)


def _binary_expression(
    left: RationalRelation, right: RationalRelation, *, serial: bool
) -> RationalRelation:
    a, b = left.topology, right.topology
    if a.tapes != b.tapes or a.input_tapes != b.input_tapes:
        raise TypeError("relation expressions require identical ordered tape types")
    if left.algebra is not right.algebra:
        raise TypeError("relation expressions require a common scalar semiring")
    states, slots = a.n_states, len(a.weight_names)
    arcs = a.arcs + tuple(
        Arc(arc.source + states, arc.target + states, arc.labels, arc.weight + slots)
        for arc in b.arcs
    )
    right_initial = _offset_boundary(b.initial, states, slots)
    right_final = _offset_boundary(b.final, states, slots)
    n_states = states + b.n_states
    if serial:
        labels = (EPSILON,) * len(a.tapes)
        arcs += tuple(
            Arc(mark.state, n_states, labels, mark.weight) for mark in a.final
        )
        arcs += tuple(
            Arc(n_states, mark.state, labels, mark.weight) for mark in right_initial
        )
        n_states += 1
        initial, final = a.initial, right_final
    else:
        initial, final = a.initial + right_initial, a.final + right_final
    topology = RelationTopology(
        a.tapes,
        a.input_tapes,
        n_states,
        arcs,
        initial,
        final,
        tuple(f"left:{name}" for name in a.weight_names)
        + tuple(f"right:{name}" for name in b.weight_names),
    )
    return RationalRelation(
        topology, _ExpressionWeights(left, right), algebra=left.algebra
    )


def union(left: RationalRelation, right: RationalRelation) -> RationalRelation:
    """Add the weight of the operands at each tuple of words.

    The operands have the same ordered tape types and scalar semiring.
    Their disjoint state sets preserve each original path as a separate
    occurrence. Construction is linear in the topology records, apart
    from compiling the ordinary label indexes.
    """
    return _binary_expression(left, right, serial=False)


def concatenate(left: RationalRelation, right: RationalRelation) -> RationalRelation:
    """Sum over path pairs whose tape words concatenate to the result.

    Each tape reads its left word followed by its right word. A common
    connecting state takes one left final weight and one right initial
    weight, preserving them as separate factors. Distinct decompositions
    of the same word tuple contribute separately. The operands have
    identical tape types and one shared scalar semiring.
    """
    return _binary_expression(left, right, serial=True)


def closure(relation: RationalRelation) -> RationalRelation:
    """Build the relation of all finite repetitions, including zero.

    Every return to the new boundary state completes one accepting
    operand path. The topology leaves all repetition counts available.
    A fixed-word query must request a closure bound when a productive
    cycle exists, and its algebra must license the resulting sum.
    """
    topology = relation.topology
    bridge = topology.n_states
    unit = len(topology.weight_names)
    # The fresh slot name must stay distinct even if the input uses
    # "unit" or any names introduced by earlier transforms.
    marks = (Boundary(bridge, unit),)
    labels = (EPSILON,) * len(topology.tapes)
    entries = tuple(
        Arc(bridge, mark.state, labels, mark.weight) for mark in topology.initial
    )
    exits = tuple(
        Arc(mark.state, bridge, labels, mark.weight) for mark in topology.final
    )
    repeated = replace(
        topology,
        n_states=bridge + 1,
        arcs=topology.arcs + entries + exits,
        initial=marks,
        final=marks,
        weight_names=tuple(f"operand:{name}" for name in topology.weight_names)
        + ("unit",),
    )
    return RationalRelation(
        repeated, _ExpressionWeights(relation), algebra=relation.algebra
    )


__all__ = ["closure", "concatenate", "union"]
