"""Star matrices with one primitive occurrence per input entry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from quivers.core.algebras import Star

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _ClosureBlock:
    order: int
    states: int
    positions: tuple[int, ...]
    slots: tuple[int, ...]

    def matrix(self, weights: torch.Tensor, algebra: Star) -> torch.Tensor:
        auxiliary = self.order - self.states
        slots = torch.tensor(self.slots, dtype=torch.long, device=weights.device)
        values = torch.cat(
            (weights.index_select(0, slots), algebra.unit_like((auxiliary,), weights))
        )
        positions = torch.tensor(
            self.positions, dtype=torch.long, device=weights.device
        )
        return (
            algebra.zero_like((self.order**2,), weights)
            .index_copy(0, positions, values)
            .reshape(self.order, self.order, *algebra.element_shape)
        )

    def project(self, closed: torch.Tensor) -> torch.Tensor:
        auxiliary = self.order - self.states
        return closed[auxiliary:, auxiliary:]


def _make_block(states: int, edges: Sequence[tuple[int, int, int]]) -> _ClosureBlock:
    # Give each extra parallel arc a private unit-return subdivision. Walks
    # with original endpoints correspond bijectively and preserve weights.
    # The star's convergence check thus sees the original coefficients.
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
