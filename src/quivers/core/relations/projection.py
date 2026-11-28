"""Inclusive KL projection onto a fixed n-gram family on a declared support."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import torch

type _Word = tuple[int, ...]


def _affine_basis(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    # Exact elimination on feature differences from the first point.
    pivots: dict[int, list[Fraction]] = {}
    retained: list[int] = []
    for index, first in enumerate(rows[0]):
        difference = [Fraction(row[index] - first) for row in rows[1:]]
        for pivot, previous in pivots.items():
            if coefficient := difference[pivot]:
                difference = [
                    a - coefficient * b
                    for a, b in zip(difference, previous, strict=True)
                ]
        for pivot, value in enumerate(difference):
            if value:
                pivots[pivot] = [coordinate / value for coordinate in difference]
                retained.append(index)
                break
    return tuple(retained)


def _validate_projection_settings(
    tolerance: float, max_iterations: int, max_backtracks: int
) -> None:
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if type(max_backtracks) is not int or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    if not 0 < tolerance < float("inf"):
        raise ValueError("tolerance must be positive and finite")


@dataclass(frozen=True, slots=True)
class NgramProjection:
    """An inclusive KL projection and its local optimizer diagnostics."""

    status: Literal["converged", "limit", "stalled"]
    iterations: int
    log_probabilities: torch.Tensor
    coefficients: torch.Tensor
    target_moments: torch.Tensor
    fitted_moments: torch.Tensor
    moment_residual: float
    divergence: torch.Tensor


@dataclass(frozen=True, slots=True)
class NgramFamily:
    """A fixed inclusive-KL family on a declared finite string support.

    The features count contiguous n-grams of orders one through
    `order`. Exact elimination selects an affine basis of these counts
    on the support. A base measure can modulate the family at each
    call, including a cavity distribution for EP.
    """

    words: tuple[_Word, ...]
    order: int
    features: tuple[_Word, ...] = field(init=False)
    _matrix: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 1:
            raise ValueError("n-gram order must be a positive integer")
        words = tuple(tuple(word) for word in self.words)
        if not words or len(words) != len(set(words)):
            raise ValueError("an n-gram family requires distinct support words")
        if any(
            type(symbol) is not int or symbol < 0 for word in words for symbol in word
        ):
            raise ValueError("n-gram support words require nonnegative integer symbols")
        counts = tuple(
            Counter(
                word[start : start + n]
                for n in range(1, min(len(word), self.order) + 1)
                for start in range(len(word) - n + 1)
            )
            for word in words
        )
        vocabulary = tuple(sorted({feature for row in counts for feature in row}))
        rows = tuple(tuple(row[feature] for feature in vocabulary) for row in counts)
        columns = _affine_basis(rows)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "features", tuple(vocabulary[i] for i in columns))
        object.__setattr__(
            self, "_matrix", tuple(tuple(row[i] for i in columns) for row in rows)
        )

    def project(
        self,
        tilted_log_weights: torch.Tensor,
        *,
        base_log_weights: torch.Tensor | None = None,
        tolerance: float = 1e-9,
        max_iterations: int = 64,
        max_backtracks: int = 16,
    ) -> NgramProjection:
        """Match tilted feature moments by minimizing inclusive KL.

        Both measures require positive finite weights on the declared
        support, represented in float64 log space. Each Newton step
        checks the update in the convex moment objective. The
        tolerance is an infinity-norm bound on the feature residual.
        Autograd differentiates the accepted sequence of operations.
        """
        _validate_projection_settings(tolerance, max_iterations, max_backtracks)
        tilted = tilted_log_weights
        base = (
            torch.zeros_like(tilted) if base_log_weights is None else base_log_weights
        )
        if base.shape != (len(self.words),) or tilted.shape != base.shape:
            raise ValueError("n-gram weights must have one value per support word")
        if (
            tilted.dtype != torch.float64
            or base.dtype != tilted.dtype
            or base.device != tilted.device
        ):
            raise TypeError(
                "n-gram projection requires float64 weights on a common device"
            )
        if not bool(torch.isfinite(tilted).all() & torch.isfinite(base).all()):
            raise ValueError(
                "n-gram projection requires finite log weights on the full support"
            )

        target_log_prob = tilted.log_softmax(dim=0)
        target = target_log_prob.exp()
        if not bool((target > 0).all()):
            raise ValueError(
                "tilted probabilities must be positive in the evaluation precision"
            )
        base = base - base.max()
        features = tilted.new_tensor(self._matrix)
        mean = target @ features
        coefficients = tilted.new_zeros(len(self.features))
        logits = base
        status: Literal["converged", "limit", "stalled"] = "limit"
        iteration = 0
        while True:
            log_prob = logits.log_softmax(dim=0)
            prob = log_prob.exp()
            fit = prob @ features
            difference = fit - mean
            error = difference.abs().max().item() if difference.numel() else 0.0
            if not bool((prob > 0).all() & torch.isfinite(log_prob).all()):
                status = "stalled"
                break
            if error <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break
            centered = features - fit
            hessian = centered.mT @ (prob[:, None] * centered)
            direction, info = torch.linalg.solve_ex(hessian, difference)
            decrement = (direction @ difference).item()
            if info.item() != 0 or not 0 < decrement < float("inf"):
                status = "stalled"
                break
            objective = logits.logsumexp(dim=0) - coefficients @ mean
            for backtrack in range(max_backtracks + 1):
                step = 2.0**-backtrack
                proposed = coefficients - step * direction
                update = base + features @ proposed
                new_objective = update.logsumexp(dim=0) - proposed @ mean
                if bool(
                    torch.isfinite(new_objective)
                    & (new_objective <= objective - 1e-4 * step * decrement)
                ):
                    coefficients, logits = proposed, update
                    break
            else:
                status = "stalled"
                break
            iteration += 1
        return NgramProjection(
            status,
            iteration,
            log_prob,
            coefficients,
            mean,
            fit,
            error,
            target @ (target_log_prob - log_prob),
        )


__all__ = ["NgramFamily", "NgramProjection"]
