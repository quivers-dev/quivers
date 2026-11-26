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
of evaluation order (Goodman 1999). A chart semiring is a member of
the one hierarchy of `quivers.core.algebras`: `ChartSemiring` is a
`quivers.core.algebras.Semiring`, and each chart semiring is the core
algebra of the same name read through the chart vocabulary, so its
``times``, ``plus``, ``plus_pair``, ``zero`` and ``one`` are the core
``tensor_op``, ``join``, ``join_pair``, ``zero`` and ``unit``. A sum
over no terms is the zero, a sum of impossible log weights has
derivative zero, and a best-derivation sum gives the whole derivative
of a tie to its first operand in every operation.
``LogProbSemiring`` is the log semiring
``([-∞, ∞), logsumexp, +, -∞, 0)``, isomorphic to the probability
semiring ``([0, ∞), +, ·)`` through ``exp``. Its addition is not
idempotent (``logsumexp(a, a) = a + log 2``), so it is not a lattice
join and the log semiring is not a quantale; ``ViterbiSemiring`` and
``BooleanSemiring`` are idempotent.

References
----------
Goodman, J. (1999). Semiring parsing. *Computational Linguistics*,
25(4), 573-606. [ACL Anthology J99-4004](https://aclanthology.org/J99-4004/)
"""

from __future__ import annotations

import torch

from quivers.core.algebras import (
    BooleanAlgebra,
    CountingAlgebra,
    LogProbAlgebra,
    MaxPlusAlgebra,
    Semiring,
)


class ChartSemiring(Semiring):
    """A commutative semiring read through the chart vocabulary.

    The chart names of the operations are the core operations of the
    `quivers.core.algebras.Semiring` it is: ``times`` is ``tensor_op``,
    ``plus`` is ``join``, ``plus_pair`` is ``join_pair`` and ``one`` is
    ``unit``. All operations are differentiable where the carrier
    allows, to support gradient-based learning.
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplicative combination (⊗), the core ``tensor_op``.

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
        return self.tensor_op(a, b)

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Additive aggregation (⊕) over a dimension, the core ``join``.

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
            Aggregated scores, the zero over no terms.
        """
        return self.join(scores, dim)

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binary additive combination (⊕), the core ``join_pair``.

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
        return self.join_pair(a, b)

    @property
    def one(self) -> float:
        """Multiplicative identity, the score of a trivial derivation.

        Returns
        -------
        float
            The core ``unit``.
        """
        return self.unit


class LogProbSemiring(ChartSemiring, LogProbAlgebra):
    """Log-probability semiring: (logsumexp, +, -∞, 0).

    This is the default semiring for marginal parsing. ``times`` is
    addition in log-space (= multiplication of probabilities) and
    ``plus`` is logsumexp (= addition of probabilities in log-space).
    """

    proof = LogProbAlgebra.proof
    check = LogProbAlgebra.check


class ViterbiSemiring(ChartSemiring, MaxPlusAlgebra):
    """Viterbi semiring: (max, +, -∞, 0).

    Finds the single best derivation. ``plus`` takes the max over
    derivations rather than summing, and gives the whole derivative of
    a tie to the first best operand. Useful for Viterbi parsing and
    best-first decoding.
    """

    proof = MaxPlusAlgebra.proof
    check = MaxPlusAlgebra.check


class BooleanSemiring(ChartSemiring, BooleanAlgebra):
    """Boolean semiring: (or, and, 0, 1).

    For recognition only: determines whether a parse exists without
    computing scores. Values are 0.0 (false) and 1.0 (true).
    """

    proof = BooleanAlgebra.proof
    check = BooleanAlgebra.check


class CountingSemiring(ChartSemiring, CountingAlgebra):
    """Counting semiring: (+, ×, 0, 1).

    Counts the number of derivations. ``plus`` sums counts and
    ``times`` multiplies them (product rule for independent choices).
    """

    proof = CountingAlgebra.proof
    check = CountingAlgebra.check


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
