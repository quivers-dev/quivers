"""Semiring abstractions for parameterizing chart parsing.

A semiring (S, ⊕, ⊗, 0, 1) provides the algebraic structure for
chart parsing (Goodman, 1999). Different semirings yield different
parsing algorithms from the same CKY skeleton:

- **LogProbSemiring**: marginal log-probability (logsumexp / +)
- **ViterbiSemiring**: best-derivation log-probability (max / +)
- **BooleanSemiring**: recognition (or / and)
- **CountingSemiring**: derivation counting (+ / ×)

Each semiring defines:

- ``times(a, b)``: multiplicative combination (⊗)
- ``plus(scores, dim)``: additive aggregation (⊕) over a dimension
- ``zero``: additive identity (0): score for impossible derivations
- ``one``: multiplicative identity (1): score for trivial derivations

Relation to the enrichment algebras
-----------------------------------
A semiring is the structure that makes the chart's sum over
derivations of products of rule weights well defined and independent
of evaluation order (Goodman 1999). ``LogProbSemiring`` is the log
semiring ``([-∞, ∞), logsumexp, +, -∞, 0)``, isomorphic to the
probability semiring ``([0, ∞), +, ·)`` through ``exp``. Its addition
is not idempotent (``logsumexp(a, a) = a + log 2``), so it is not a
lattice join and the log semiring is not a quantale; ``ViterbiSemiring``
and ``BooleanSemiring`` are idempotent. The corresponding enrichment
algebras are the `quivers.core.algebras.Semiring` instances.

References
----------
Goodman, J. (1999). Semiring parsing. *Computational Linguistics*,
25(4), 573-606. [ACL Anthology J99-4004](https://aclanthology.org/J99-4004/)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ChartSemiring(ABC):
    """Abstract semiring for chart parsing.

    Subclasses implement the four semiring operations. All operations
    must be differentiable (where applicable) to support gradient-based
    learning.
    """

    @abstractmethod
    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplicative combination (⊗).

        Combines scores from two chart cells being joined by a rule.

        Parameters
        ----------
        a : torch.Tensor
            Left operand scores.
        b : torch.Tensor
            Right operand scores.

        Returns
        -------
        torch.Tensor
            Combined scores.
        """

    @abstractmethod
    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Additive aggregation (⊕) over a dimension.

        Marginalizes over multiple derivations of the same span.

        Parameters
        ----------
        scores : torch.Tensor
            Scores to aggregate.
        dim : int
            Dimension to aggregate over.

        Returns
        -------
        torch.Tensor
            Aggregated scores.
        """

    @abstractmethod
    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binary additive combination (⊕) of two tensors.

        Used for combining a cell with unary rule contributions.

        Parameters
        ----------
        a : torch.Tensor
            First operand.
        b : torch.Tensor
            Second operand.

        Returns
        -------
        torch.Tensor
            Combined scores.
        """

    @property
    @abstractmethod
    def zero(self) -> float:
        """Additive identity: the score of an impossible derivation."""

    @property
    @abstractmethod
    def one(self) -> float:
        """Multiplicative identity: the score of a trivial derivation."""


class LogProbSemiring(ChartSemiring):
    """Log-probability semiring: (logsumexp, +, -∞, 0).

    This is the default semiring for marginal parsing. ``times`` is
    addition in log-space (= multiplication of probabilities) and
    ``plus`` is logsumexp (= addition of probabilities in log-space).
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply probabilities by adding log-probabilities.

        Parameters
        ----------
        a : torch.Tensor
            Left operand scores.
        b : torch.Tensor
            Right operand scores.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return a + b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum probabilities over ``dim`` by logsumexp.

        Parameters
        ----------
        scores : torch.Tensor
            Scores to aggregate.
        dim : int
            Dimension to aggregate over.

        Returns
        -------
        torch.Tensor
            The aggregated scores, with ``dim`` removed.
        """
        return torch.logsumexp(scores, dim=dim)

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Sum two log-probabilities by logaddexp.

        Parameters
        ----------
        a : torch.Tensor
            First operand.
        b : torch.Tensor
            Second operand.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return torch.logaddexp(a, b)

    @property
    def zero(self) -> float:
        """Negative infinity, the log-probability of an impossible derivation."""
        return float("-inf")

    @property
    def one(self) -> float:
        """Zero, the log-probability of a certain derivation."""
        return 0.0

    def __repr__(self) -> str:
        return "LogProbSemiring()"


class ViterbiSemiring(ChartSemiring):
    """Viterbi semiring: (max, +, -∞, 0).

    Finds the single best derivation. ``plus`` takes the max over
    derivations rather than summing. Useful for Viterbi parsing
    and best-first decoding.
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply probabilities by adding log-probabilities.

        Parameters
        ----------
        a : torch.Tensor
            Left operand scores.
        b : torch.Tensor
            Right operand scores.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return a + b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Take the best score over ``dim``.

        Parameters
        ----------
        scores : torch.Tensor
            Scores to aggregate.
        dim : int
            Dimension to aggregate over.

        Returns
        -------
        torch.Tensor
            The aggregated scores, with ``dim`` removed.
        """
        return scores.max(dim=dim).values

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Take the elementwise best of two scores.

        Parameters
        ----------
        a : torch.Tensor
            First operand.
        b : torch.Tensor
            Second operand.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return torch.maximum(a, b)

    @property
    def zero(self) -> float:
        """Negative infinity, the score of an impossible derivation."""
        return float("-inf")

    @property
    def one(self) -> float:
        """Zero, the score of a trivial derivation."""
        return 0.0

    def __repr__(self) -> str:
        return "ViterbiSemiring()"


class BooleanSemiring(ChartSemiring):
    """Boolean semiring: (or, and, 0, 1).

    For recognition only: determines whether a parse exists without
    computing scores. Values are 0.0 (false) and 1.0 (true).
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Conjoin truth values in {0, 1} by multiplication.

        Parameters
        ----------
        a : torch.Tensor
            Left operand scores.
        b : torch.Tensor
            Right operand scores.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        # and in {0, 1}: min or product
        return a * b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Disjoin truth values in {0, 1} over ``dim`` by max.

        Parameters
        ----------
        scores : torch.Tensor
            Scores to aggregate.
        dim : int
            Dimension to aggregate over.

        Returns
        -------
        torch.Tensor
            The aggregated scores, with ``dim`` removed.
        """
        # or in {0, 1}: clamp of sum or max
        return scores.max(dim=dim).values

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Disjoin two truth values by elementwise max.

        Parameters
        ----------
        a : torch.Tensor
            First operand.
        b : torch.Tensor
            Second operand.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return torch.maximum(a, b)

    @property
    def zero(self) -> float:
        """``0.0``, false."""
        return 0.0

    @property
    def one(self) -> float:
        """``1.0``, true."""
        return 1.0

    def __repr__(self) -> str:
        return "BooleanSemiring()"


class CountingSemiring(ChartSemiring):
    """Counting semiring: (+, ×, 0, 1).

    Counts the number of derivations. ``plus`` sums counts and
    ``times`` multiplies them (product rule for independent choices).

    Note: Not differentiable in a useful sense for gradient-based
    learning, but useful for analysis.
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply derivation counts.

        Parameters
        ----------
        a : torch.Tensor
            Left operand scores.
        b : torch.Tensor
            Right operand scores.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return a * b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum derivation counts over ``dim``.

        Parameters
        ----------
        scores : torch.Tensor
            Scores to aggregate.
        dim : int
            Dimension to aggregate over.

        Returns
        -------
        torch.Tensor
            The aggregated scores, with ``dim`` removed.
        """
        return scores.sum(dim=dim)

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add two derivation counts.

        Parameters
        ----------
        a : torch.Tensor
            First operand.
        b : torch.Tensor
            Second operand.

        Returns
        -------
        torch.Tensor
            The combined scores.
        """
        return a + b

    @property
    def zero(self) -> float:
        """``0.0``, the count of an impossible derivation."""
        return 0.0

    @property
    def one(self) -> float:
        """``1.0``, the count of a trivial derivation."""
        return 1.0

    def __repr__(self) -> str:
        return "CountingSemiring()"


LOG_PROB = LogProbSemiring()
"""The log-probability semiring singleton, the default for chart parsing."""

VITERBI = ViterbiSemiring()
"""The Viterbi (max-product in log space) semiring singleton."""

BOOLEAN = BooleanSemiring()
"""The Boolean (or-and) semiring singleton, for recognition."""

COUNTING = CountingSemiring()
"""The counting (sum-product over naturals) semiring singleton."""


__all__ = [
    "ChartSemiring",
    "LogProbSemiring",
    "ViterbiSemiring",
    "BooleanSemiring",
    "CountingSemiring",
    "LOG_PROB",
    "VITERBI",
    "BOOLEAN",
    "COUNTING",
]
