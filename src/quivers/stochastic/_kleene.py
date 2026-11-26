"""The Kleene star of a unary-rule matrix in a chart semiring.

Unary rules derive a category over a span from another category over the
same span, so a chain of them never consumes input and may cycle. The
weight they add to a chart cell is the cell times the reflexive,
transitive closure of the unary-rule matrix ``U``,

    U* = I ⊕ U ⊕ U² ⊕ U³ ⊕ ...,

the sum over every chain of unary rules of the product of its weights
(Stolcke 1995, definition 6, where for probabilities ``U* = (I - U)⁻¹``;
Goodman 1999 for the semiring generalization). The closure is applied
once to the cell the binary and lexical rules produce; applying ``U``
repeatedly to an accumulating cell instead counts a chain once per
ordering of its prefixes and overstates every weight.

The series need not converge. For probabilities it converges exactly
when the spectral radius of ``U`` is below one; in the max-plus Viterbi
semiring when no cycle of unary rules has positive log weight; in the
Boolean semiring always. A closure that diverges has no value, and it
raises rather than return a truncated sum. The closure is the `star` of
the semiring's `quivers.core.algebras.Star` capability, so any algebra
that has the capability has a closure and no other one does.

References
----------
Stolcke, A. (1995). An efficient probabilistic context-free parsing
algorithm that computes prefix probabilities. *Computational
Linguistics* 21(2), 165-201, section 4.5.2.
<https://aclanthology.org/J95-2002/>

Goodman, J. (1999). Semiring parsing. *Computational Linguistics*
25(4), 573-605. <https://aclanthology.org/J99-4004/>
"""

from __future__ import annotations

import torch

from quivers.core.algebras import (
    DivergentClosureError,
    Star,
    UndecidedClosureError,
)
from quivers.stochastic.semiring import ChartSemiring


def kleene_star(matrix: torch.Tensor, semiring: ChartSemiring | Star) -> torch.Tensor:
    """The reflexive, transitive closure ``U*`` of a unary-rule matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        ``(C, C)`` rule weights in the semiring's representation, indexed
        ``[derived-from, derived]``: ``matrix[b, a]`` is the weight of the
        rule deriving ``a`` from ``b`` over one span; the semiring's zero
        where there is no rule.
    semiring : ChartSemiring or Star
        The semiring the weights live in.

    Returns
    -------
    torch.Tensor
        ``(C, C)`` closure in the same representation and indexing.

    Raises
    ------
    DivergentClosureError
        If the closure's series diverges in the semiring.
    UndecidedClosureError
        If rounding cannot decide whether it converges.
    TypeError
        If the semiring has no closure.
    """
    if not isinstance(semiring, Star):
        raise TypeError(
            f"no unary-rule closure is defined for {semiring!r}: it does not "
            "have the Star capability"
        )
    return semiring.star(matrix)


def apply_star(
    cell: torch.Tensor, star: torch.Tensor, semiring: ChartSemiring
) -> torch.Tensor:
    """Close a chart cell under the unary rules: ``cell ⊗ U*``.

    Parameters
    ----------
    cell : torch.Tensor
        ``(batch, C)`` weights of each category over one span.
    star : torch.Tensor
        ``(C, C)`` closure from `kleene_star`.
    semiring : ChartSemiring
        The semiring of both.

    Returns
    -------
    torch.Tensor
        ``(batch, C)``: ``out[:, a] = ⊕_b cell[:, b] ⊗ star[b, a]``.
    """
    products = semiring.times(cell.unsqueeze(2), star.unsqueeze(0))
    return semiring.plus(products, dim=1)


__all__ = [
    "DivergentClosureError",
    "UndecidedClosureError",
    "kleene_star",
    "apply_star",
]
