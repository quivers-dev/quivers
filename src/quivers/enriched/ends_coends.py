"""Ends and coends: general contraction principles for V-enriched categories.

The coend ∫^X F(X,X) computes the join (existential) over the diagonal of
matched contravariant/covariant dimensions. The end ∫_X F(X,X) computes
the meet (universal) over the diagonal.

These routines implement the (co)ends over a discrete indexing category,
where the compatibility conditions are vacuous: the coend is the join and
the end the meet over the diagonal. They are (co)ends when the join and
meet are those of a complete lattice, so the algebra must be an
`IdempotentSemiring`.
"""

from __future__ import annotations

import torch

from quivers.core.algebras import IdempotentSemiring
from quivers.core.tensor_ops import diagonal_reduce
from quivers.enriched._lattice import require_lattice


def coend(
    tensor: torch.Tensor,
    contra_dims: tuple[int, ...],
    co_dims: tuple[int, ...],
    algebra: IdempotentSemiring,
) -> torch.Tensor:
    """Compute coend ∫^X F(X,X) via diagonal extraction and join.

    For each matched pair (contra_dims[i], co_dims[i]), restricts the
    tensor to the diagonal (where the contravariant and covariant
    indices agree) and then joins (existential quantification) over
    those matched dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor representing a bifunctor F(X, Y) with both
        contravariant and covariant occurrences of X.
    contra_dims : tuple[int, ...]
        Dimension indices for the contravariant occurrences of X.
    co_dims : tuple[int, ...]
        Dimension indices for the covariant occurrences of X.
        Must be same length as contra_dims, with matching sizes.
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Returns
    -------
    torch.Tensor
        Result with matched dimension pairs removed via join.

    Raises
    ------
    TypeError
        If ``algebra`` is not an `IdempotentSemiring`.
    ValueError
        If the dimension lists differ in length or size.
    """
    algebra = require_lattice(algebra, "a (co)end")

    if len(contra_dims) != len(co_dims):
        raise ValueError(
            f"contra_dims and co_dims must have same length, "
            f"got {len(contra_dims)} and {len(co_dims)}"
        )

    if len(contra_dims) == 0:
        return tensor

    return diagonal_reduce(tensor, contra_dims, co_dims, algebra.join)


def end(
    tensor: torch.Tensor,
    contra_dims: tuple[int, ...],
    co_dims: tuple[int, ...],
    algebra: IdempotentSemiring,
) -> torch.Tensor:
    """Compute end ∫_X F(X,X) via diagonal extraction and meet.

    Dual to coend: restricts to diagonal and applies meet (universal
    quantification) over matched dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.
    contra_dims : tuple[int, ...]
        Dimension indices for contravariant occurrences.
    co_dims : tuple[int, ...]
        Dimension indices for covariant occurrences.
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Returns
    -------
    torch.Tensor
        Result with matched dimension pairs removed via meet.

    Raises
    ------
    TypeError
        If ``algebra`` is not an `IdempotentSemiring`.
    ValueError
        If the dimension lists differ in length or size.
    """
    algebra = require_lattice(algebra, "a (co)end")

    if len(contra_dims) != len(co_dims):
        raise ValueError(
            f"contra_dims and co_dims must have same length, "
            f"got {len(contra_dims)} and {len(co_dims)}"
        )

    if len(contra_dims) == 0:
        return tensor

    return diagonal_reduce(tensor, contra_dims, co_dims, algebra.meet)


__all__ = [
    "coend",
    "end",
]
