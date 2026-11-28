"""Sparse composition of bivariate rational relations."""

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
from quivers.core.relations.topology import (
    EPSILON,
    Arc,
    Boundary,
    Label,
    RelationTopology,
    Words,
)


class _Filter(Enum):
    MATCH = auto()
    LEFT = auto()
    RIGHT = auto()


type _ProductState = tuple[int, int, _Filter]
type _Configuration = tuple[int, int, _Filter, int, int]
type _ArcGroup = dict[Label, list[int]]
type _ArcIndex = dict[tuple[int, Label], _ArcGroup]
type _WeightPair = tuple[int | None, int | None]


def _arc_index(
    topology: RelationTopology, outer: int, *, observe_outer: bool = True
) -> _ArcIndex:
    index: _ArcIndex = defaultdict(dict)
    for arc_id, arc in enumerate(topology.arcs):
        visible = arc.labels[outer] if observe_outer else EPSILON
        index[arc.source, visible].setdefault(arc.labels[1 - outer], []).append(arc_id)
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
            raise RelationLimitError("composition exceeds max_work")


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
        return self.paths.evaluate(
            _pair_weights(left, right, self.weights, algebra), algebra
        )


def _read_weight_pair(
    left: RationalRelation, right: RationalRelation
) -> tuple[torch.Tensor, torch.Tensor]:
    values = left._weights()
    if left.module() is right.module():
        right._check_weights(values)
        return values, values
    return values, right._weights()


def _pair_weights(
    left: torch.Tensor,
    right: torch.Tensor,
    pairs: tuple[_WeightPair, ...],
    algebra: Semiring,
) -> torch.Tensor:
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("composed weights require a common dtype and device")
    if not pairs:
        return torch.cat((left[:0], right[:0]))
    left_slots = tuple(len(left) if i is None else i for i, _ in pairs)
    right_slots = tuple(len(right) if j is None else j for _, j in pairs)
    # A missing side takes a multiplicative unit. The gathers keep a
    # live derivative path to every used original source slot.
    l_values = _take(torch.cat((left, algebra.unit_like((1,), left))), left_slots)
    r_values = _take(torch.cat((right, algebra.unit_like((1,), right))), right_slots)
    return algebra.tensor_op(l_values, r_values)


class _MaterializedWeights(nn.Module):
    def __init__(
        self, relation: ComposedRelation, pairs: tuple[_WeightPair, ...]
    ) -> None:
        super().__init__()
        self.sources = relation.module()
        self._relation = relation
        self._pairs = pairs

    def forward(self) -> torch.Tensor:
        left, right = self._relation._operand_weights()
        boundaries = _pair_weights(left, right, self._pairs, self._relation.algebra)
        return torch.cat((left, right, boundaries))


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


def _materialize_product(
    left: RelationTopology,
    right: RelationTopology,
    max_states: int,
    max_work: int,
) -> tuple[RelationTopology, tuple[_WeightPair, ...]]:
    if len(left.initial) * len(right.initial) > max_states:
        raise RelationLimitError("composition exceeds max_states")
    left_index = _arc_index(left, 0, observe_outer=False)
    right_index = _arc_index(right, 1, observe_outer=False)
    left_count = len(left.weight_names)
    source_slots = left_count + len(right.weight_names)
    boundary_slots: dict[_WeightPair, int] = {}
    states: list[tuple[int, _ProductState]] = []
    state_ids: dict[_ProductState, int] = {}
    initial: list[Boundary] = []
    final: list[Boundary] = []
    arcs: list[Arc] = []
    work = _CompositionWork(max_work)
    n_states = 0

    def allocate() -> int:
        nonlocal n_states
        if n_states == max_states:
            raise RelationLimitError("composition exceeds max_states")
        state = n_states
        n_states += 1
        return state

    def state_id(state: _ProductState) -> int:
        index = state_ids.get(state)
        if index is None:
            index = allocate()
            state_ids[state] = index
            states.append((index, state))
        return index

    def add_arc(source: int, target: int, labels: tuple[Label, ...], slot: int) -> None:
        work.demand()
        work.arcs += 1
        arcs.append(Arc(source, target, labels, slot))

    def boundary_slot(left_slot: int, right_slot: int) -> int:
        pair = left_slot, right_slot
        return source_slots + boundary_slots.setdefault(pair, len(boundary_slots))

    for l_mark, r_mark in product(left.initial, right.initial):
        state = state_id((l_mark.state, r_mark.state, _Filter.MATCH))
        initial.append(Boundary(state, boundary_slot(l_mark.weight, r_mark.weight)))
    for source, (l_state, r_state, phase) in states:
        l_final = left._final_slots.get(l_state)
        r_final = right._final_slots.get(r_state)
        if l_final is not None and r_final is not None:
            final.append(Boundary(source, boundary_slot(l_final, r_final)))
        l_group = left_index.get((l_state, EPSILON))
        r_group = right_index.get((r_state, EPSILON))
        for l_id, r_id, next_phase in _moves(
            (l_group,) if l_group else (), (r_group,) if r_group else (), phase, work
        ):
            work.demand()
            l_arc = None if l_id is None else left.arcs[l_id]
            r_arc = None if r_id is None else right.arcs[r_id]
            target = state_id(
                (
                    l_state if l_arc is None else l_arc.target,
                    r_state if r_arc is None else r_arc.target,
                    next_phase,
                )
            )
            if l_arc is None:
                assert r_arc is not None
                add_arc(
                    source,
                    target,
                    (EPSILON, r_arc.labels[1]),
                    left_count + r_arc.weight,
                )
            elif r_arc is None:
                add_arc(source, target, (l_arc.labels[0], EPSILON), l_arc.weight)
            else:
                # Preserve both primitive factors for a future star of a
                # cyclic product. This intermediate has one continuation.
                between = allocate()
                add_arc(
                    source, between, (l_arc.labels[0], r_arc.labels[1]), l_arc.weight
                )
                add_arc(between, target, (EPSILON, EPSILON), left_count + r_arc.weight)
    names = (
        *(f"left:{name}" for name in left.weight_names),
        *(f"right:{name}" for name in right.weight_names),
        *(f"boundary:{index}" for index in range(len(boundary_slots))),
    )
    topology = RelationTopology(
        (left.tapes[0], right.tapes[1]),
        1,
        n_states,
        tuple(arcs),
        tuple(initial),
        tuple(final),
        names,
    )
    return topology, tuple(boundary_slots)


class ComposedRelation(Morphism):
    """Binary sparse composition on a checked shared tape.

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

    def _operand_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _read_weight_pair(self._left, self._right)

    def materialize(self, *, max_states: int, max_work: int) -> RationalRelation:
        """Build a reusable sparse product topology for composition or projection.

        The limits bound the states and the sum of shared-label probes
        and arcs. They include intermediate states and arcs that retain
        the primitive factors of a matched pair. The result reads both
        original weight sources on each evaluation.
        """
        if max_states is None or max_work is None:
            raise ValueError(
                "materialized composition requires explicit state and work limits"
            )
        _validate_limits(max_states, max_work)
        topology, pairs = _materialize_product(
            self._left.topology, self._right.topology, max_states, max_work
        )
        return RationalRelation(
            topology, _MaterializedWeights(self, pairs), algebra=self.algebra
        )

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
        left, right = self._relation._operand_weights()
        return self._plan.evaluate(left, right, self._relation.algebra)


__all__ = ["ComposedRelation", "ComposedStringQuery"]
