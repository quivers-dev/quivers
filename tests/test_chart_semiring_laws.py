"""The chart semirings satisfy the commutative-semiring laws.

Each `ChartSemiring` is checked on a grid of carrier values against the
laws (Golan 1999): ``(S, ⊕, 0)`` and ``(S, ⊗, 1)`` are commutative
monoids, ``⊗`` distributes over ``⊕``, and ``0`` annihilates. ``plus``
over a dimension must agree with folding ``plus_pair``. A truncated
sum, which breaks distributivity, fails the same check.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)

_GRIDS: dict[str, tuple[ChartSemiring, list[float]]] = {
    "log_prob": (LOG_PROB, [-math.inf, -2.0, -0.5, 0.0, 1.0]),
    "viterbi": (VITERBI, [-math.inf, -2.0, -0.5, 0.0, 1.0]),
    "boolean": (BOOLEAN, [0.0, 1.0]),
    "counting": (COUNTING, [0.0, 1.0, 2.0, 3.0]),
}


def _t(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = float(a), float(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _failures(sr: ChartSemiring, grid: list[float]) -> list[str]:
    plus, times = sr.plus_pair, sr.times
    zero, one = _t(sr.zero), _t(sr.one)
    out: list[str] = []
    for a in map(_t, grid):
        if not _eq(plus(a, zero), a):
            out.append("additive identity")
        if not _eq(times(a, one), a):
            out.append("multiplicative identity")
        if not _eq(times(a, zero), zero):
            out.append("annihilation")
    for a, b in itertools.product(map(_t, grid), repeat=2):
        if not _eq(plus(a, b), plus(b, a)) or not _eq(times(a, b), times(b, a)):
            out.append("commutativity")
        stacked = torch.stack([a, b])
        if not _eq(sr.plus(stacked, 0), plus(a, b)):
            out.append("plus over a dimension")
    for a, b, c in itertools.product(map(_t, grid), repeat=3):
        if not _eq(plus(plus(a, b), c), plus(a, plus(b, c))):
            out.append("additive associativity")
        if not _eq(times(times(a, b), c), times(a, times(b, c))):
            out.append("multiplicative associativity")
        if not _eq(times(a, plus(b, c)), plus(times(a, b), times(a, c))):
            out.append("distributivity")
    return out


@pytest.mark.parametrize("name", sorted(_GRIDS))
def test_chart_semirings_satisfy_the_laws(name: str) -> None:
    sr, grid = _GRIDS[name]
    assert _failures(sr, grid) == []


class _TruncatedCounting(type(COUNTING)):
    """Counting with the sum capped at 1: not distributive."""

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b).clamp(max=1.0)

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        return scores.sum(dim=dim).clamp(max=1.0)


def test_the_check_rejects_a_truncated_sum() -> None:
    assert "distributivity" in _failures(_TruncatedCounting(), [0.0, 0.5, 1.0, 2.0])


def test_log_prob_addition_is_not_idempotent() -> None:
    a = _t(0.0)
    assert _eq(LOG_PROB.plus_pair(a, a), _t(math.log(2.0)))
