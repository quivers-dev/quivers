"""Kan extensions along functions between finite sets.

A morphism R: A → B over an algebra V is a V-valued matrix, which is a
functor from the discrete category A to the category of V-valued
presheaves on B. For a function p: A → A' between discrete categories
and V a complete lattice, the left and right Kan extensions along p
are pointwise joins and meets over the fibres of p [Mac Lane (1998),
§X.3; Kelly (1982), §4]:

    (Lan_p R)(a', b) = ⋁{R(a, b) : p(a) = a'}
    (Ran_p R)(a', b) = ⋀{R(a, b) : p(a) = a'}

An empty fibre gives the least element ⊥ for Lan and the greatest ⊤
for Ran. This is the discrete special case of a Kan extension; the
formulas are the Kan extensions only when ⋁ and ⋀ are the join and
meet of V's order, so the functions require an `IdempotentSemiring`
algebra, whose join is idempotent and whose meet is the infimum of its
order. They satisfy the adjunctions

    Lan_p R ≤ S  ⟺  R ≤ S ∘ p        S ∘ p ≤ R  ⟺  S ≤ Ran_p R

for every S: A' → B. Marginalization over an idempotent semiring is the
left Kan extension along a projection π: A₁×...×Aₙ → A_{i₁}×...×A_{iₖ}.

This module provides the abstract `ObjectMap` (a function between
finite sets) with `Projection` and `Inclusion`, and the `left_kan` and
`right_kan` functions.

References
----------
Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd
ed.). Springer.
[doi:10.1007/978-1-4757-4721-8](https://doi.org/10.1007/978-1-4757-4721-8)

Kelly, G. M. (1982). *Basic Concepts of Enriched Category Theory*.
Cambridge University Press. Reprinted in *Reprints in Theory and
Applications of Categories*, 10 (2005).
[TAC reprint](https://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html)
"""

from __future__ import annotations
import itertools
from abc import ABC, abstractmethod
import torch
from quivers.core.objects import SetObject, ProductSet, CoproductSet
from quivers.core.morphisms import Morphism, ObservedMorphism, observed
from quivers.core.algebras import Algebra, IdempotentSemiring


class ObjectMap(ABC):
    """An abstract deterministic map between finite sets.

    Represents a function p: A → A' at the element level, used
    as the map along which to compute Kan extensions.
    """

    @property
    @abstractmethod
    def source(self) -> SetObject:
        """The source object A."""
        ...

    @property
    @abstractmethod
    def target(self) -> SetObject:
        """The target object A'."""
        ...

    @abstractmethod
    def apply(self, source_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Map a source index to a target index.

        Parameters
        ----------
        source_idx : tuple[int, ...]
            An element of A as a multi-index.

        Returns
        -------
        tuple[int, ...]
            The corresponding element of A'.
        """
        ...

    def fiber_indices(self, target_idx: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Return all source indices that map to the given target index.

        Parameters
        ----------
        target_idx : tuple[int, ...]
            An element of A'.

        Returns
        -------
        list[tuple[int, ...]]
            All a ∈ A such that p(a) = target_idx.
        """
        result: list[tuple[int, ...]] = []
        for src_idx in itertools.product(*(range(s) for s in self.source.shape)):
            if self.apply(src_idx) == target_idx:
                result.append(src_idx)
        return result


class Projection(ObjectMap):
    """Projection from a product to a subset of its components.

    Given ProductSet(A₁, ..., Aₙ), projects onto the components
    at the specified indices: π(a₁, ..., aₙ) = (a_{i₁}, ..., a_{iₖ}).

    Parameters
    ----------
    product : ProductSet
        The source product set.
    keep_indices : tuple[int, ...]
        Indices of the components to keep (0-based).
    """

    def __init__(self, product: ProductSet, keep_indices: tuple[int, ...]) -> None:
        if not isinstance(product, ProductSet):
            raise TypeError(
                f"Projection requires ProductSet, got {type(product).__name__}"
            )
        n = len(product.components)
        for idx in keep_indices:
            if not 0 <= idx < n:
                raise ValueError(f"component index {idx} out of range [0, {n})")
        self._product = product
        self._keep_indices = keep_indices
        self._drop_indices = tuple((i for i in range(n) if i not in keep_indices))
        kept = [product.components[i] for i in keep_indices]
        if len(kept) == 1:
            self._target = kept[0]
        else:
            self._target = ProductSet(components=tuple(kept))
        self._dim_offsets: list[int] = []
        offset = 0
        for comp in product.components:
            self._dim_offsets.append(offset)
            offset += comp.ndim

    @property
    def source(self) -> ProductSet:
        """The source product set."""
        return self._product

    @property
    def target(self) -> SetObject:
        """The target (projected) set."""
        return self._target

    @property
    def keep_indices(self) -> tuple[int, ...]:
        """Indices of the kept components."""
        return self._keep_indices

    @property
    def drop_indices(self) -> tuple[int, ...]:
        """Indices of the dropped (marginalized) components."""
        return self._drop_indices

    def apply(self, source_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Project by keeping only the selected component indices."""
        result: list[int] = []
        for comp_idx in self._keep_indices:
            comp = self._product.components[comp_idx]
            offset = self._dim_offsets[comp_idx]
            for d in range(comp.ndim):
                result.append(source_idx[offset + d])
        return tuple(result)


class Inclusion(ObjectMap):
    """Coproduct inclusion ι_k: Aₖ → A₁ + ... + Aₙ.

    Embeds the k-th component of a CoproductSet into the full set.

    Parameters
    ----------
    coproduct : CoproductSet
        The target coproduct set.
    component_index : int
        Which component to include (0-based).
    """

    def __init__(self, coproduct: CoproductSet, component_index: int) -> None:
        if not isinstance(coproduct, CoproductSet):
            raise TypeError(
                f"Inclusion requires CoproductSet, got {type(coproduct).__name__}"
            )
        n = len(coproduct.components)
        if not 0 <= component_index < n:
            raise ValueError(f"component_index {component_index} out of range [0, {n})")
        self._coproduct = coproduct
        self._component_index = component_index
        self._component = coproduct.components[component_index]
        self._offset = coproduct.offset(component_index)

    @property
    def source(self) -> SetObject:
        """The source component."""
        return self._component

    @property
    def target(self) -> CoproductSet:
        """The target coproduct set."""
        return self._coproduct

    def apply(self, source_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Embed into the coproduct with offset."""
        (flat_idx,) = source_idx
        return (self._offset + flat_idx,)


def _kan_algebra(morph: Morphism, algebra: Algebra | None) -> IdempotentSemiring:
    """The lattice a Kan extension of ``morph`` is computed in.

    Parameters
    ----------
    morph : Morphism
        The morphism being extended.
    algebra : Algebra or None
        The algebra requested, or None for the morphism's own.

    Returns
    -------
    IdempotentSemiring
        The algebra.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring`, or differs from
        the morphism's algebra.
    """
    q = algebra if algebra is not None else morph.algebra
    if not isinstance(q, IdempotentSemiring):
        raise TypeError(
            "Kan extensions along a function are joins and meets of a "
            f"complete lattice; {q!r} is not an IdempotentSemiring"
        )
    if not morph.algebra.is_compatible(q):
        raise TypeError(f"the morphism is over {morph.algebra.name}, not {q.name}")
    return q


def left_kan(
    morph: Morphism, along: ObjectMap, algebra: Algebra | None = None
) -> ObservedMorphism:
    """Left Kan extension of a morphism along a function.

    Computes ``(Lan_p R)(a', b) = ⋁{R(a, b) : p(a) = a'}``, which is ⊥
    over an empty fibre.

    Parameters
    ----------
    morph : Morphism
        The morphism R: A → B.
    along : ObjectMap
        The function p: A → A'.
    algebra : Algebra or None
        The algebra, an `IdempotentSemiring`; the morphism's own when
        omitted.

    Returns
    -------
    ObservedMorphism
        The left Kan extension Lan_p(R): A' → B.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring` or is not the
        morphism's.
    """
    q = _kan_algebra(morph, algebra)
    target = along.target
    codomain = morph.codomain
    result = torch.full((*target.shape, *codomain.shape), q.zero)
    source_tensor = morph.tensor.detach()
    for tgt_idx in itertools.product(*(range(s) for s in target.shape)):
        fiber = along.fiber_indices(tgt_idx)
        if not fiber:
            continue
        slices = torch.stack([source_tensor[src_idx] for src_idx in fiber])
        result[tgt_idx] = q.join(slices, dim=0)
    return observed(target, codomain, result, algebra=q)


def right_kan(
    morph: Morphism, along: ObjectMap, algebra: Algebra | None = None
) -> ObservedMorphism:
    """Right Kan extension of a morphism along a function.

    Computes ``(Ran_p R)(a', b) = ⋀{R(a, b) : p(a) = a'}``, which is ⊤
    over an empty fibre.

    Parameters
    ----------
    morph : Morphism
        The morphism R: A → B.
    along : ObjectMap
        The function p: A → A'.
    algebra : Algebra or None
        The algebra, an `IdempotentSemiring`; the morphism's own when
        omitted.

    Returns
    -------
    ObservedMorphism
        The right Kan extension Ran_p(R): A' → B.

    Raises
    ------
    TypeError
        If the algebra is not an `IdempotentSemiring` or is not the
        morphism's.
    """
    q = _kan_algebra(morph, algebra)
    target = along.target
    codomain = morph.codomain
    result = torch.full((*target.shape, *codomain.shape), q.top)
    source_tensor = morph.tensor.detach()
    for tgt_idx in itertools.product(*(range(s) for s in target.shape)):
        fiber = along.fiber_indices(tgt_idx)
        if not fiber:
            continue
        slices = torch.stack([source_tensor[src_idx] for src_idx in fiber])
        result[tgt_idx] = q.meet(slices, dim=0)
    return observed(target, codomain, result, algebra=q)


__all__ = [
    "ObjectMap",
    "Projection",
    "Inclusion",
    "left_kan",
    "right_kan",
]
