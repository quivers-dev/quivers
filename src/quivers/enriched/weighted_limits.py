"""Weighted (co)limits for V-enriched categories.

In a V-enriched category, the correct notion of limit is the
*weighted* limit. Given a V-functor D: J → C (diagram) and a
V-functor W: J → V (weight), the weighted limit {W, D} is
characterized by:

    C(X, {W, D}) ≅ [J, V](W, C(X, D-))

and the weighted colimit W ⊗_J D by:

    C(W ⊗_J D, X) ≅ [J, V](W, C(D-, X))

For a discrete indexing set J and a diagram of parallel morphisms
f_j: A → B, viewed as objects of the V-category of V-valued matrices
A → B (the power V^{A×B}, complete when V is a complete lattice), these
reduce to the pointwise end and coend

    {W, f} = ⋀_j [W(j), f_j]      (weighted limit)
    W ⊗_J f = ⋁_j W(j) ⊗ f_j      (weighted colimit)

where [-, -] is V's residuum (internal hom) [Kelly (1982), §3]. They
are characterized by g ≤ {W, f} ⟺ W(j) ⊗ g ≤ f_j for every j, and
W ⊗_J f ≤ g ⟺ W(j) ⊗ f_j ≤ g for every j, which needs V to be an
`IdempotentSemiring` with its `IdempotentSemiring.residuum`.

This module provides:

    Weight                       — a V-valued weight on a finite set
    weighted_limit_morphisms()   — compute {W, f}
    weighted_colimit_morphisms() — compute W ⊗_J f
    representable_weight()       — weight represented by an index
    terminal_weight()            — constant weight at the unit

References
----------
Kelly, G. M. (1982). *Basic Concepts of Enriched Category Theory*.
Cambridge University Press. Reprinted in *Reprints in Theory and
Applications of Categories*, 10 (2005).
[TAC reprint](https://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html)
"""

from __future__ import annotations

from collections.abc import Sequence

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.core.objects import FinSet
from quivers.core.morphisms import Morphism
from quivers.core.algebras import IdempotentSemiring
from quivers.enriched._lattice import require_lattice


class Weight(ValueModel):
    """A V-valued presheaf on a finite indexing category.

    Represents a weight functor W: J → V where J is a finite set of
    indices, and W assigns a V-value (scalar in the algebra's lattice)
    to each index.

    The tensor and algebra are held by identity. Immutable record updates
    retain the live tensor and its autograd history.

    Parameters
    ----------
    values : torch.Tensor
        A 1D tensor of shape (|J|,) with values in L (the algebra's
        lattice). W(j) = values[j].
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Raises
    ------
    didactic.ValidationError
        If ``algebra`` is not an `IdempotentSemiring` or ``values`` is
        not a one-dimensional tensor.
    """

    values: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    algebra: IdempotentSemiring = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    @dx.model_validator()
    def _check_weight(self) -> None:
        """Validate the weight tensor and enrichment algebra.

        Raises
        ------
        TypeError
            If the values are not a tensor or the algebra is not a lattice.
        ValueError
            If the weight tensor is not one-dimensional.
        """
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("weight values must be a torch.Tensor")
        if self.values.ndim != 1:
            raise ValueError(f"weight values must be 1D, got shape {self.values.shape}")
        require_lattice(self.algebra, "a weight")

    @property
    def size(self) -> int:
        """Number of indices in J."""
        return self.values.shape[0]


def weighted_limit_morphisms(
    weight: Weight,
    morphisms: Sequence[Morphism],
    algebra: IdempotentSemiring,
) -> torch.Tensor:
    """Compute a weighted limit from a family of morphisms.

    Given morphisms f_j: X → A_j and weights W(j), computes the
    weighted meet:

        result[x, ...] = ⋀_j [W(j), f_j(x, ...)]

    This is the tensor-level computation of the weighted limit
    of a cone.

    Parameters
    ----------
    weight : Weight
        The weight W.
    morphisms : Sequence[Morphism]
        The morphisms f_j comprising the cone.
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Returns
    -------
    torch.Tensor
        The weighted limit tensor.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring`.
    ValueError
        If the weight and the family differ in size or the family is
        empty.
    """
    q = require_lattice(algebra, "a weighted (co)limit")

    if weight.size != len(morphisms):
        raise ValueError(
            f"weight size {weight.size} != number of morphisms {len(morphisms)}"
        )

    n = len(morphisms)

    if n == 0:
        raise ValueError("need at least one morphism")

    # compute [W(j), f_j] for each j and meet over j
    components: list[torch.Tensor] = []

    for j in range(n):
        w_j = weight.values[j]
        f_j = morphisms[j].tensor
        hom = q.residuum(w_j, f_j)
        components.append(hom)

    # meet over all components (they should all have the same shape)
    stacked = torch.stack(components, dim=0)
    return q.meet(stacked, dim=0)


def weighted_colimit_morphisms(
    weight: Weight,
    morphisms: Sequence[Morphism],
    algebra: IdempotentSemiring,
) -> torch.Tensor:
    """Compute a weighted colimit from a family of morphisms.

    Given morphisms f_j: A_j → X and weights W(j), computes the
    weighted join:

        result[..., x] = ⋁_j W(j) ⊗ f_j(..., x)

    Parameters
    ----------
    weight : Weight
        The weight W.
    morphisms : Sequence[Morphism]
        The morphisms f_j comprising the cocone.
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Returns
    -------
    torch.Tensor
        The weighted colimit tensor.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring`.
    ValueError
        If the weight and the family differ in size or the family is
        empty.
    """
    q = require_lattice(algebra, "a weighted (co)limit")

    if weight.size != len(morphisms):
        raise ValueError(
            f"weight size {weight.size} != number of morphisms {len(morphisms)}"
        )

    n = len(morphisms)

    if n == 0:
        raise ValueError("need at least one morphism")

    components: list[torch.Tensor] = []

    for j in range(n):
        w_j = weight.values[j]
        f_j = morphisms[j].tensor
        scaled = q.tensor_op(f_j, w_j)
        components.append(scaled)

    stacked = torch.stack(components, dim=0)
    return q.join(stacked, dim=0)


def representable_weight(
    index_set: FinSet,
    represented_at: int,
    algebra: IdempotentSemiring,
) -> Weight:
    """Create a representable weight (Yoneda-style).

    The representable weight at index k is W(j) = I if j == k,
    and W(j) = ⊥ otherwise. Weighted limits with representable
    weights recover evaluation: {y(k), D} ≅ D(k).

    Parameters
    ----------
    index_set : FinSet
        The indexing set J.
    represented_at : int
        The representing index k.
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Returns
    -------
    Weight
        The representable weight at k.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring`.
    """
    q = require_lattice(algebra, "a weighted (co)limit")

    values = torch.full((index_set.cardinality,), q.zero)
    values[represented_at] = q.unit

    return Weight(values=values, algebra=q)


def terminal_weight(
    index_set: FinSet,
    algebra: IdempotentSemiring,
) -> Weight:
    """Create the terminal (constant unit) weight.

    W(j) = I for all j. Weighted limits with the terminal weight
    recover ordinary (conical) limits.

    Parameters
    ----------
    index_set : FinSet
        The indexing set J.
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Returns
    -------
    Weight
        The terminal weight.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring`.
    """
    q = require_lattice(algebra, "a weighted (co)limit")
    values = torch.full((index_set.cardinality,), q.unit)

    return Weight(values=values, algebra=q)


__all__ = [
    "Weight",
    "weighted_limit_morphisms",
    "weighted_colimit_morphisms",
    "representable_weight",
    "terminal_weight",
]
