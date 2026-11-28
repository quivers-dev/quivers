"""Packed first-overflow mass for capped fixed-word path posteriors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations._packed import _pack_incoming, _take
from quivers.core.relations.paths import (
    RelationPath,
    _markov_log_weights,
    _PathWalks,
    _probability_algebra,
    _require_normalizer,
)
from quivers.core.relations.queries import RelationLimitError
from quivers.core.relations.topology import Words

if TYPE_CHECKING:
    from quivers.core.relations import RationalRelation


type _Trace = tuple[int, tuple[int, ...], int]


class _CappedPaths:
    def __init__(
        self,
        relation: RationalRelation,
        words: Words,
        capacity: int,
        *,
        max_states: int,
        max_work: int,
        max_closure_size: int,
        max_closure_cells: int,
        max_prefix_work: int,
        max_prefix_cells: int,
    ) -> None:
        self._relation = relation
        self._words = words
        self._capacity = capacity
        topology = relation.topology
        walks = _PathWalks(
            topology,
            words,
            max_states,
            max_work,
            max_closure_size,
            max_closure_cells=max_closure_cells,
        )
        self._walks = walks
        states = len(walks._order)
        arcs = sum(len(walks._arcs[state]) for state in walks._order)
        # The packed incoming map has at most 2E + V entries including
        # padding. Each prefix layer also gathers E products and emits V
        # values. Count both the initial and last suffix layers.
        if (capacity + 2) * states > max_prefix_cells:
            raise RelationLimitError("capped path plan exceeds max_prefix_cells")
        work = (capacity + 1) * (3 * arcs + 2 * states)
        if work + len(walks._initial) + 2 * states > max_prefix_work:
            raise RelationLimitError("capped path plan exceeds max_prefix_work")
        local = {state: index for index, state in enumerate(walks._order)}
        rows: list[list[int]] = [[] for _ in local]
        sources: list[int] = []
        slots: list[int] = []
        for state in walks._order:
            for target, arc_id in walks._arcs[state]:
                rows[local[target]].append(len(slots))
                sources.append(local[state])
                slots.append(topology.arcs[arc_id].weight)
        self._incoming = _pack_incoming(rows, len(slots))
        self._sources = tuple(sources)
        self._slots = tuple(slots)
        self._starts = tuple(local[state] for state, _ in walks._initial)
        self._start_weights = tuple(slot for _, slot in walks._initial)
        self._suffixes = tuple(
            walks._backward.positions[state] for state in walks._order
        )

    def _log_snapshot(self) -> torch.Tensor:
        weights = self._relation._weights()
        algebra = _probability_algebra(weights, self._relation.algebra)
        weights = weights.to(dtype=torch.float64, copy=True)
        return _markov_log_weights(weights) if algebra is MARKOV else weights

    def _snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_weights = self._log_snapshot()
        total, suffixes = self._walks._backward.evaluate(log_weights, LOG_PROB)
        _require_normalizer(total, LOG_PROB)
        return log_weights, total, suffixes

    def sample(self, generator: torch.Generator) -> RelationPath | None:
        return self._walks._draw(
            self._log_snapshot(),
            LOG_PROB,
            generator,
            self._capacity,
            overflow_to_tail=True,
        )

    def log_mass(self) -> torch.Tensor:
        _, total, _ = self._snapshot()
        return total.new_zeros(())

    def log_prob(self, trace: _Trace | None, *, tail: bool = False) -> torch.Tensor:
        weights, total, suffixes = self._snapshot()
        if tail:
            return self._tail_log_prob(weights, total, suffixes)
        if trace is None:
            return weights.new_full((), -torch.inf)
        start, arcs, end = trace
        return self._walks.path_log_prob(self._words, start, arcs, end, weights, total)

    def _tail_log_prob(
        self, weights: torch.Tensor, total: torch.Tensor, suffixes: torch.Tensor
    ) -> torch.Tensor:
        # Every overflow walk has one first capacity+1 traversal prefix.
        # Stop edges are outside the prefix DP and live in the suffix chart.
        prefix = weights.new_full((len(self._suffixes),), -torch.inf)
        initial = weights.new_tensor(self._starts, dtype=torch.long)
        prefix = prefix.index_copy(0, initial, _take(weights, self._start_weights))
        edges = _take(weights, self._slots)
        for _ in range(self._capacity + 1):
            prefix = self._incoming.join(_take(prefix, self._sources) + edges, LOG_PROB)
        tail = LOG_PROB.join(prefix + _take(suffixes, self._suffixes), dim=0)
        if bool(torch.isneginf(tail)):
            return weights.new_full((), -torch.inf)
        score = tail - total
        if not bool(torch.isfinite(score)):
            raise OverflowError("capped tail probability is outside float64")
        return score
