"""Packed semiring evaluation of an acyclic fixed-word path graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quivers.core.algebras import Semiring


def _take(values: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    if indices == (0,) and values.numel() == 1:
        return values
    index = torch.tensor(indices, dtype=torch.long, device=values.device)
    return values.index_select(0, index)


@dataclass(frozen=True, slots=True)
class _Incoming:
    indices: tuple[int, ...]
    buckets: tuple[tuple[int, int], ...]
    state_order: tuple[int, ...]
    padded: bool

    def join(self, values: torch.Tensor, algebra: Semiring) -> torch.Tensor:
        if self.padded:
            values = torch.cat((values, algebra.zero_like((1,), values)))
        gathered = _take(values, self.indices)
        if len(self.buckets) == 1 and self.buckets[0][1] == 1:
            return gathered
        joined = [
            chunk if width == 1 else algebra.join(chunk.reshape(-1, width), dim=1)
            for chunk, (_, width) in zip(
                gathered.split(tuple(size for size, _ in self.buckets)),
                self.buckets,
                strict=True,
            )
        ]
        return _take(torch.cat(joined), self.state_order)


def _pack_incoming(rows: list[list[int]], n_values: int) -> _Incoming:
    # Each row has at least one summand. Padding to the next power of two
    # puts fewer than twice the actual number of summands into the buckets.
    by_width: dict[int, list[tuple[int, list[int]]]] = {}
    for state, row in enumerate(rows):
        width = 1 << (len(row) - 1).bit_length()
        by_width.setdefault(width, []).append((state, row))
    indices: list[int] = []
    buckets: list[tuple[int, int]] = []
    state_order = [0] * len(rows)
    padded = False
    position = 0
    for width, group in by_width.items():
        buckets.append((len(group) * width, width))
        for state, row in group:
            state_order[state] = position
            position += 1
            indices.extend(row)
            if missing := width - len(row):
                padded = True
                indices.extend([n_values] * missing)
    return _Incoming(tuple(indices), tuple(buckets), tuple(state_order), padded)


@dataclass(frozen=True, slots=True)
class _PathLayer:
    incoming: _Incoming
    initial_count: int
    outgoing_sources: tuple[int, ...]
    destinations: tuple[int, ...]
    destination_sizes: tuple[int, ...]
    final_sources: tuple[int, ...]

    @property
    def weight_sizes(self) -> tuple[int, int, int]:
        return self.initial_count, len(self.outgoing_sources), len(self.final_sources)


@dataclass(frozen=True, slots=True)
class _PackedPaths:
    layers: tuple[_PathLayer, ...]
    slots: tuple[int, ...]

    def evaluate(
        self,
        weights: torch.Tensor,
        algebra: Semiring,
        *,
        visits: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.layers:
            return algebra.join(weights[:0], dim=0)
        slots = _take(weights, self.slots)
        chunks = slots.split(tuple(sum(layer.weight_sizes) for layer in self.layers))
        pending: list[list[torch.Tensor]] = [[] for _ in self.layers]
        accepted: list[torch.Tensor] = []
        for layer, chunk, incoming in zip(self.layers, chunks, pending, strict=True):
            n_initial, n_arcs, n_final = layer.weight_sizes
            if n_initial:
                incoming.append(chunk[:n_initial])
            values = incoming[0] if len(incoming) == 1 else torch.cat(incoming)
            prefixes = layer.incoming.join(values, algebra)
            if visits is not None:
                visits.append(prefixes)
            incoming.clear()
            if n_arcs:
                arc_weights = (
                    chunk[n_initial : n_initial + n_arcs]
                    if n_initial or n_final
                    else chunk
                )
                products = algebra.tensor_op(
                    _take(prefixes, layer.outgoing_sources), arc_weights
                )
                portions = (
                    (products,)
                    if len(layer.destinations) == 1
                    else products.split(layer.destination_sizes)
                )
                for destination, portion in zip(
                    layer.destinations, portions, strict=True
                ):
                    pending[destination].append(portion)
            if n_final:
                final = chunk[-n_final:] if n_initial or n_arcs else chunk
                accepted.append(
                    algebra.tensor_op(_take(prefixes, layer.final_sources), final)
                )
        return algebra.join(torch.cat(accepted), dim=0)


def _pack_paths(
    order: list[int],
    arcs: list[tuple[tuple[int, int], ...]],
    initial: tuple[tuple[int, int], ...],
    final: list[tuple[int, int]],
    *,
    visit_order: list[int] | None = None,
) -> _PackedPaths:
    level_of = dict.fromkeys(order, 0)
    levels: list[list[int]] = []
    for state in order:
        level = level_of[state]
        if level == len(levels):
            levels.append([])
        levels[level].append(state)
        for target, _ in arcs[state]:
            level_of[target] = max(level_of[target], level + 1)
    if visit_order is not None:
        visit_order.extend(state for level in levels for state in level)

    # Row indices address the concatenation of the arc chunks from earlier
    # levels, in source-level order, followed by this level's initial marks.
    rows: dict[int, list[int]] = {state: [] for state in order}
    incoming_sizes = [0] * len(levels)
    initial_slots = dict(initial)
    final_slots = dict(final)
    layers: list[_PathLayer] = []
    slots: list[int] = []
    for level, states in enumerate(levels):
        outgoing: dict[int, list[tuple[int, int]]] = {}
        start: list[int] = []
        finish: list[tuple[int, int]] = []
        for local_id, state in enumerate(states):
            if state in initial_slots:
                rows[state].append(incoming_sizes[level] + len(start))
                start.append(initial_slots[state])
            if state in final_slots:
                finish.append((local_id, final_slots[state]))
            for target, slot in arcs[state]:
                destination = level_of[target]
                rows[target].append(incoming_sizes[destination])
                incoming_sizes[destination] += 1
                outgoing.setdefault(destination, []).append((local_id, slot))
        flat = [edge for group in outgoing.values() for edge in group]
        layers.append(
            _PathLayer(
                _pack_incoming(
                    [rows[state] for state in states],
                    incoming_sizes[level] + len(start),
                ),
                len(start),
                tuple(source for source, _ in flat),
                tuple(outgoing),
                tuple(map(len, outgoing.values())),
                tuple(source for source, _ in finish),
            )
        )
        slots.extend(start)
        slots.extend(slot for _, slot in flat)
        slots.extend(slot for _, slot in finish)
    return _PackedPaths(tuple(layers), tuple(slots))
