"""Sparse composition under a fixed observation of the outer tapes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum, auto
from itertools import product
from typing import cast

import torch
from torch import nn

from quivers.core.algebras import Semiring
from quivers.core.laws import SEMIRING_LAWS, TIMES_COMMUTATIVE
from quivers.core.morphisms import Morphism
from quivers.core.relations import RationalRelation
from quivers.core.relations._packed import _take
from quivers.core.relations.queries import (
    QueryWork,
    RelationLimitError,
    _PathPlan,
    _trim_paths,
    _validate_limits,
)
from quivers.core.relations.topology import EPSILON, Label, RelationTopology, Words


class _Filter(Enum):
    MATCH = auto()
    LEFT = auto()
    RIGHT = auto()


type _Configuration = tuple[int, int, _Filter, int, int]
type _ArcGroup = dict[Label, list[int]]
type _ArcIndex = dict[tuple[int, Label], _ArcGroup]
type _WeightPair = tuple[int | None, int | None]


def _arc_index(topology: RelationTopology, outer: int) -> _ArcIndex:
    index: _ArcIndex = defaultdict(dict)
    for arc_id, arc in enumerate(topology.arcs):
        index[arc.source, arc.labels[outer]].setdefault(
            arc.labels[1 - outer], []
        ).append(arc_id)
    return index


def _allowed_groups(
    arcs: _ArcIndex, state: int, word: tuple[int, ...], offset: int
) -> tuple[_ArcGroup, ...]:
    labels = (EPSILON, word[offset]) if offset < len(word) else (EPSILON,)
    return tuple(group for label in labels if (group := arcs.get((state, label))))


@dataclass(slots=True)
class _CompositionWork:
    limit: int | None
    probes: int = 0
    arcs: int = 0

    def demand(self) -> None:
        if self.limit is not None and self.probes + self.arcs == self.limit:
            raise RelationLimitError("fixed-word composition exceeds max_work")


def _moves(
    left: tuple[_ArcGroup, ...],
    right: tuple[_ArcGroup, ...],
    state: _Filter,
    work: _CompositionWork,
) -> Iterator[tuple[int | None, int | None, _Filter]]:
    if state is not _Filter.RIGHT:
        for group in left:
            for arc in group.get(EPSILON, ()):
                yield arc, None, _Filter.LEFT
    if state is not _Filter.LEFT:
        for group in right:
            for arc in group.get(EPSILON, ()):
                yield None, arc, _Filter.RIGHT
    for left_group, right_group in product(left, right):
        smaller = left_group if len(left_group) <= len(right_group) else right_group
        for label in smaller:
            work.demand()
            work.probes += 1
            # A shared symbol resets the filter. Real epsilons can pair
            # until one side starts advancing alone in the interval.
            if label is EPSILON and state is not _Filter.MATCH:
                continue
            left_arcs, right_arcs = left_group.get(label), right_group.get(label)
            if left_arcs is not None and right_arcs is not None:
                for l_arc in left_arcs:
                    for r_arc in right_arcs:
                        yield l_arc, r_arc, _Filter.MATCH


@dataclass(frozen=True, slots=True)
class _CompositionPlan:
    paths: _PathPlan
    weights: tuple[_WeightPair, ...]
    label_probes: int

    def evaluate(
        self, left: torch.Tensor, right: torch.Tensor, algebra: Semiring
    ) -> torch.Tensor:
        if left.device != right.device or left.dtype != right.dtype:
            raise ValueError("composed weights require a common dtype and device")
        if not self.weights:
            return self.paths.evaluate(torch.cat((left[:0], right[:0])), algebra)
        left_slots = tuple(len(left) if i is None else i for i, _ in self.weights)
        right_slots = tuple(len(right) if j is None else j for _, j in self.weights)
        # A missing side takes a multiplicative unit. The gathers keep a
        # live derivative path to every used original source slot.
        l_values = _take(torch.cat((left, algebra.unit_like((1,), left))), left_slots)
        r_values = _take(
            torch.cat((right, algebra.unit_like((1,), right))), right_slots
        )
        return self.paths.evaluate(algebra.tensor_op(l_values, r_values), algebra)


def _compile_composition(
    left: RelationTopology,
    right: RelationTopology,
    left_index: _ArcIndex,
    right_index: _ArcIndex,
    words: Words,
    *,
    max_states: int | None,
    max_work: int | None,
) -> _CompositionPlan:
    x, z = words
    if max_states is not None and len(left.initial) * len(right.initial) > max_states:
        raise RelationLimitError("fixed-word composition exceeds max_states")
    states: list[_Configuration] = []
    initial: list[tuple[int, int]] = []
    final: list[tuple[int, int]] = []
    weight_ids: dict[_WeightPair, int] = {}
    for l_mark, r_mark in product(left.initial, right.initial):
        pair = (l_mark.weight, r_mark.weight)
        initial.append((len(states), weight_ids.setdefault(pair, len(weight_ids))))
        states.append((l_mark.state, r_mark.state, _Filter.MATCH, 0, 0))
    state_ids = {state: i for i, state in enumerate(states)}
    arcs: list[list[tuple[int, int]]] = [[] for _ in states]
    work = _CompositionWork(max_work)
    cursor = 0
    while cursor < len(states):
        l_state, r_state, phase, i, j = states[cursor]
        if i == len(x) and j == len(z):
            left_final = left._final_slots.get(l_state)
            right_final = right._final_slots.get(r_state)
            if left_final is not None and right_final is not None:
                pair = (left_final, right_final)
                final.append((cursor, weight_ids.setdefault(pair, len(weight_ids))))
        l_groups = _allowed_groups(left_index, l_state, x, i)
        r_groups = _allowed_groups(right_index, r_state, z, j)
        for l_id, r_id, next_phase in _moves(l_groups, r_groups, phase, work):
            work.demand()
            work.arcs += 1
            l_arc = left.arcs[l_id] if l_id is not None else None
            r_arc = right.arcs[r_id] if r_id is not None else None
            destination = (
                l_state if l_arc is None else l_arc.target,
                r_state if r_arc is None else r_arc.target,
                next_phase,
                i + (l_arc is not None and l_arc.labels[0] is not EPSILON),
                j + (r_arc is not None and r_arc.labels[1] is not EPSILON),
            )
            target = state_ids.get(destination)
            if target is None:
                target = len(states)
                if max_states is not None and target == max_states:
                    raise RelationLimitError(
                        "fixed-word composition exceeds max_states"
                    )
                state_ids[destination] = target
                states.append(destination)
                arcs.append([])
            pair = (
                None if l_arc is None else l_arc.weight,
                None if r_arc is None else r_arc.weight,
            )
            # The state and two arc indices identify this edge occurrence.
            # Sharing its numerical slot never merges it with another edge.
            arcs[cursor].append((target, weight_ids.setdefault(pair, len(weight_ids))))
        cursor += 1

    path_plan = _trim_paths(arcs, initial, final, work.arcs)
    # Only the weight pairs on productive paths are needed at evaluation.
    used: dict[int, int] = {}
    slots = tuple(used.setdefault(slot, len(used)) for slot in path_plan.packed.slots)
    pairs = tuple(weight_ids)
    packed = replace(path_plan.packed, slots=slots)
    return _CompositionPlan(
        replace(path_plan, packed=packed),
        tuple(pairs[slot] for slot in used),
        work.probes,
    )


class ComposedRelation(Morphism):
    """Binary composition queried at fixed words on its two outer tapes.

    Each operand has one input tape and one output tape. The shared
    tape types must agree. The operands use the same scalar semiring,
    which must establish the semiring laws and commutative multiplication.

    A query constructs only reachable product configurations compatible
    with its two outer words. The three-state filter assigns one product
    path to each compatible pair of accepting operand paths.
    """

    def __init__(self, left: RationalRelation, right: RationalRelation) -> None:
        if not isinstance(left, RationalRelation) or not isinstance(
            right, RationalRelation
        ):
            raise TypeError("composition requires two rational relation operands")
        if any(
            len(operand.topology.tapes) != 2 or operand.topology.input_tapes != 1
            for operand in (left, right)
        ):
            raise TypeError(
                "composition requires one input and one output tape per operand"
            )
        if left.codomain != right.domain:
            raise TypeError("composition requires the same shared tape type")
        algebra = left.algebra
        if algebra is not right.algebra or not algebra.establishes(
            *SEMIRING_LAWS, TIMES_COMMUTATIVE
        ):
            raise TypeError("composition requires one established commutative semiring")
        super().__init__(left.domain, right.codomain, algebra)
        self._left = left
        self._right = right
        self._left_index = _arc_index(left.topology, 0)
        self._right_index = _arc_index(right.topology, 1)
        left_module, right_module = left.module(), right.module()
        self._module = nn.ModuleList(
            (left_module,)
            if left_module is right_module
            else (left_module, right_module)
        )

    @property
    def algebra(self) -> Semiring:
        """The common scalar semiring."""
        return cast(Semiring, super().algebra)

    def module(self) -> nn.Module:
        """The two operands' original weight sources and their parameters."""
        return self._module

    def query(
        self,
        words: Words,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
    ) -> ComposedStringQuery:
        """Compile a reusable plan for a pair of outer words.

        `max_states` limits reached configurations, including initial
        pairs. `max_work` limits the sum of shared-label probes and
        candidate arc occurrences. A productive cycle on the fixed
        words requires a closure capability and raises
        `ProductiveEpsilonCycleError`.
        """
        return ComposedStringQuery(
            self, words, max_states=max_states, max_work=max_work
        )

    def weight(self, words: Words) -> torch.Tensor:
        """Sum the weights of compatible path pairs for the two outer words."""
        return self.query(words).weight()


class ComposedStringQuery:
    """A fixed-word composition plan with live operand weights."""

    def __init__(
        self,
        relation: ComposedRelation,
        words: Words,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
    ) -> None:
        if len(words) != 2:
            raise ValueError("composition queries require two outer words")
        left, right = relation._left, relation._right
        left.topology._validate_words((words[0], ()))
        right.topology._validate_words(((), words[1]))
        _validate_limits(max_states, max_work)
        self._relation = relation
        self._plan = _compile_composition(
            left.topology,
            right.topology,
            relation._left_index,
            relation._right_index,
            words,
            max_states=max_states,
            max_work=max_work,
        )

    @property
    def work(self) -> QueryWork:
        """Reached and productive configuration and arc counts."""
        return self._plan.paths.work

    @property
    def label_probes(self) -> int:
        """Shared labels examined when matching two allowed outer-label groups."""
        return self._plan.label_probes

    def weight(self) -> torch.Tensor:
        """Read fresh operand weights and evaluate the packed plan."""
        left, right = self._relation._left, self._relation._right
        left_weights = left._weights()
        if left.module() is right.module():
            right._check_weights(left_weights)
            right_weights = left_weights
        else:
            right_weights = right._weights()
        return self._plan.evaluate(left_weights, right_weights, self._relation.algebra)


__all__ = ["ComposedRelation", "ComposedStringQuery"]
