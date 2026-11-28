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


def _projection_inputs(
    words: tuple[_Word, ...],
    tilted_log_weights: torch.Tensor,
    base_log_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tilted = tilted_log_weights
    base = torch.zeros_like(tilted) if base_log_weights is None else base_log_weights
    if base.shape != (len(words),) or tilted.shape != base.shape:
        raise ValueError("n-gram weights must have one value per support word")
    if (
        tilted.dtype != torch.float64
        or base.dtype != tilted.dtype
        or base.device != tilted.device
    ):
        raise TypeError("n-gram projection requires float64 weights on a common device")
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
    return base, target_log_prob, target


def _centered_log_moment(
    log_probabilities: torch.Tensor, shifts: torch.Tensor
) -> torch.Tensor:
    # E[shifts] = 0, so log E[exp(shifts)] = log(1 + E[exp(shifts) - 1 - shifts]).
    # Compute the nonnegative remainder in log space to retain small
    # improvements and avoid overflow when a rare word has a large shift.
    large = shifts >= 1
    ordinary = torch.where(large, 0, shifts)
    small = ordinary.abs() < 1e-3
    near_zero = torch.where(small, ordinary, 0)
    # The relative truncation error through degree six is below 4.1e-19 here.
    polynomial = 0.5 + near_zero * (
        1 / 6 + near_zero * (1 / 24 + near_zero * (1 / 120 + near_zero / 720))
    )
    log_small = 2 * near_zero.abs().log() + polynomial.log()
    log_ordinary = (ordinary.expm1() - ordinary).log()
    upper = torch.where(large, shifts, 1)
    log_large = upper + torch.log1p(-(1 + upper) * (-upper).exp())
    log_remainder = torch.where(
        large, log_large, torch.where(small, log_small, log_ordinary)
    )
    log_mean = (log_probabilities + log_remainder).logsumexp(dim=0)
    return torch.logaddexp(log_mean.new_zeros(()), log_mean)


@dataclass(frozen=True, slots=True)
class NgramProjection:
    """An inclusive KL projection and its local optimizer diagnostics.

    Iterations count accepted Newton updates. Line-search trials include
    rejected candidates and any terminal stalled search.
    """

    status: Literal["converged", "limit", "stalled"]
    iterations: int
    log_probabilities: torch.Tensor
    coefficients: torch.Tensor
    target_moments: torch.Tensor
    fitted_moments: torch.Tensor
    moment_residual: float
    divergence: torch.Tensor
    line_search_trials: int = 0


@dataclass(frozen=True, slots=True)
class PenalizedNgramProjection:
    """Finite tree-group projection with its composite optimality residual."""

    status: Literal["converged", "limit", "stalled"]
    iterations: int
    features: tuple[_Word, ...]
    log_probabilities: torch.Tensor
    coefficients: torch.Tensor
    optimality_residual: float
    divergence: torch.Tensor
    regularization: torch.Tensor


def _prefix_indices(features: tuple[_Word, ...]) -> tuple[tuple[int, ...], ...]:
    groups: dict[_Word, list[int]] = {}
    for i, feature in enumerate(features):
        for n in range(len(feature) + 1):
            groups.setdefault(feature[:n], []).append(i)
    ordered = sorted(groups, key=lambda prefix: (-len(prefix), prefix))
    return tuple(tuple(groups[prefix]) for prefix in ordered)


def _prefix_prox(
    coefficients: torch.Tensor, groups: tuple[torch.Tensor, ...], threshold: float
) -> torch.Tensor:
    for group in groups:
        values = coefficients.index_select(0, group)
        norm = torch.linalg.vector_norm(values)
        active = norm > threshold
        safe = torch.where(active, norm, torch.ones_like(norm))
        scale = torch.where(active, 1.0 - threshold / safe, 0.0)
        coefficients = coefficients.index_copy(0, group, values * scale)
    return coefficients


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
    _vocabulary: tuple[_Word, ...] = field(init=False, repr=False)
    _full_matrix: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _penalty_groups: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

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
        object.__setattr__(self, "_vocabulary", vocabulary)
        object.__setattr__(self, "_full_matrix", rows)
        object.__setattr__(self, "_penalty_groups", _prefix_indices(vocabulary))
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
        base, target_log_prob, target = _projection_inputs(
            self.words, tilted_log_weights, base_log_weights
        )
        features = base.new_tensor(self._matrix)
        mean = target @ features
        coefficients = base.new_zeros(len(self.features))
        logits = base
        status: Literal["converged", "limit", "stalled"] = "limit"
        iteration = 0
        line_search_trials = 0
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
            for backtrack in range(max_backtracks + 1):
                line_search_trials += 1
                step = 2.0**-backtrack
                proposed = coefficients - step * direction
                update = base + features @ proposed
                with torch.no_grad():
                    offset = proposed - coefficients
                    descent = -(difference @ offset).item()
                    curvature = _centered_log_moment(log_prob, centered @ offset)
                    accepted = (
                        0 < descent < float("inf")
                        and bool(torch.isfinite(update).all())
                        and curvature.item() <= (1 - 1e-4) * descent
                    )
                if accepted:
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
            line_search_trials,
        )

    def project_penalized(
        self,
        tilted_log_weights: torch.Tensor,
        *,
        penalty: float,
        base_log_weights: torch.Tensor | None = None,
        tolerance: float = 1e-9,
        max_iterations: int = 512,
    ) -> PenalizedNgramProjection:
        """Project with a positive tree-group penalty on all n-gram coefficients.

        Each group contains the features sharing a given prefix, including
        that prefix when it is a feature. The empty prefix is a group.
        The objective adds `penalty` times the sum of group norms to
        inclusive KL. The residual measures the proximal gradient
        mapping in the infinity norm.
        """
        _validate_projection_settings(tolerance, max_iterations, 0)
        if not 0 < penalty < float("inf"):
            raise ValueError("tree-group penalty must be positive and finite")
        base, target_log_prob, target = _projection_inputs(
            self.words, tilted_log_weights, base_log_weights
        )
        features = base.new_tensor(self._full_matrix)
        # A common shift of every row leaves all fitted probabilities
        # fixed and gives a global covariance operator-norm bound.
        features = features - features[0]
        radius = features.square().sum(dim=1).max().item()
        step = 1.0 / radius if 0 < radius < float("inf") else 1.0
        if not radius < float("inf") or not 0 < step * penalty < float("inf"):
            raise OverflowError("tree-group objective exceeds the evaluation precision")
        groups = tuple(
            torch.tensor(group, dtype=torch.long, device=base.device)
            for group in self._penalty_groups
        )
        coefficients = base.new_zeros(len(self._vocabulary))
        mean = target @ features
        iteration = 0
        status: Literal["converged", "limit", "stalled"] = "limit"
        while True:
            fitted = (base + features @ coefficients).log_softmax(dim=0)
            probs = fitted.exp()
            gradient = probs @ features - mean
            proposed = _prefix_prox(
                coefficients - step * gradient, groups, step * penalty
            )
            mapping = (coefficients - proposed) / step
            error = mapping.abs().max().item() if mapping.numel() else 0.0
            if not (
                bool(torch.isfinite(fitted).all() & (probs > 0).all())
                and error < float("inf")
            ):
                status = "stalled"
                break
            if error <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break
            coefficients = proposed
            iteration += 1
        norm_sum = sum(
            (
                torch.linalg.vector_norm(coefficients.index_select(0, group))
                for group in groups
            ),
            base.new_zeros(()),
        )
        return PenalizedNgramProjection(
            status,
            iteration,
            self._vocabulary,
            fitted,
            coefficients,
            error,
            target @ (target_log_prob - fitted),
            penalty * norm_sum,
        )


__all__ = ["NgramFamily", "NgramProjection", "PenalizedNgramProjection"]
