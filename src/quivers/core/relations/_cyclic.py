"""Topology and local star matrices for productive relation components."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from quivers.core.algebras import Star, UndecidedClosureError
from quivers.core.relations._packed import _take


def _components(
    arcs: list[tuple[tuple[int, int], ...]],
    reverse: list[list[int]],
    productive: set[int],
) -> Iterator[tuple[int, ...]]:
    # Reverse finishing order puts each source component before those it
    # reaches. A reverse search from its root can only find unassigned
    # vertices of that component.
    visiting: set[int] = set()
    finished: list[int] = []
    for root, edges in enumerate(arcs):
        if root not in productive or root in visiting:
            continue
        visiting.add(root)
        stack = [(root, iter(edges))]
        while stack:
            state, outgoing = stack[-1]
            edge = next(outgoing, None)
            if edge is None:
                finished.append(state)
                stack.pop()
            elif (target := edge[0]) not in visiting:
                visiting.add(target)
                stack.append((target, iter(arcs[target])))

    remaining = set(productive)
    for root in reversed(finished):
        if root not in remaining:
            continue
        members = [root]
        remaining.remove(root)
        for state in members:
            for source in reverse[state]:
                if source in remaining:
                    remaining.remove(source)
                    members.append(source)
        yield tuple(members)


@dataclass(frozen=True, slots=True)
class _ClosureBlock:
    order: int
    states: int
    positions: tuple[int, ...]
    slots: tuple[int, ...]

    def evaluate(self, weights: torch.Tensor, algebra: Star) -> torch.Tensor:
        auxiliary = self.order - self.states
        values = torch.cat(
            (_take(weights, self.slots), algebra.unit_like((auxiliary,), weights))
        )
        index = torch.tensor(self.positions, dtype=torch.long, device=weights.device)
        matrix = (
            algebra.zero_like((self.order**2,), weights)
            .index_copy(0, index, values)
            .reshape(self.order, self.order)
        )
        closed = algebra.star(matrix)[auxiliary:, auxiliary:]
        finite = torch.isfinite(closed)
        if not bool(finite.diagonal().all() & (finite | algebra.is_zero(closed)).all()):
            raise UndecidedClosureError(
                "productive epsilon closure did not yield usable return weights"
            )
        return closed.reshape(-1)


def _make_block(states: int, edges: list[tuple[int, int, int]]) -> _ClosureBlock:
    # One primitive occurrence per initial matrix entry. A unit-return
    # subdivision of each extra parallel arc retains its original weight
    # for the closure's numerical domain check.
    direct: list[tuple[int, int, int]] = []
    parallel: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for source, target, slot in edges:
        edge = source, target
        if edge in seen:
            parallel.append((source, target, slot))
        else:
            seen.add(edge)
            direct.append((source, target, slot))

    auxiliary = len(parallel)
    order = states + auxiliary
    positions: list[int] = []
    slots: list[int] = []
    for source, target, slot in direct:
        positions.append((source + auxiliary) * order + target + auxiliary)
        slots.append(slot)
    for index, (source, _, slot) in enumerate(parallel):
        positions.append((source + auxiliary) * order + index)
        slots.append(slot)
    positions.extend(
        index * order + target + auxiliary
        for index, (_, target, _) in enumerate(parallel)
    )
    return _ClosureBlock(order, states, tuple(positions), tuple(slots))
