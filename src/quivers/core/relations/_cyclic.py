"""Topology and local star matrices for productive relation components."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from quivers.core.algebras import Star, UndecidedClosureError
from quivers.core._closure_block import _ClosureBlock


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


def _close_block(
    block: _ClosureBlock, weights: torch.Tensor, algebra: Star
) -> torch.Tensor:
    closed = block.project(algebra.star(block.matrix(weights, algebra)))
    finite = torch.isfinite(closed)
    if not bool(finite.diagonal().all() & (finite | algebra.is_zero(closed)).all()):
        raise UndecidedClosureError(
            "productive epsilon closure did not yield usable return weights"
        )
    return closed.reshape(-1)
