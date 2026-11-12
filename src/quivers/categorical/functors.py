"""Endofunctors on V-enriched FinSet.

A functor F maps objects to objects and morphisms to morphisms,
preserving composition and identities. This module provides:

    Functor (abstract)
    ├── IdentityFunctor  — Id: A ↦ A, f ↦ f
    ├── ComposedFunctor  — F ∘ G: applies G then F
    └── FreeMonoidFunctor — A ↦ A*, f ↦ f* (block-diagonal, componentwise)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, FinSet, FreeMonoid
from quivers.core.algebras import Algebra, Semiring
from quivers.core.tensor_ops import componentwise_lift


class Functor(ABC):
    """Abstract endofunctor on V-enriched FinSet.

    Subclasses must implement map_object, map_morphism, and map_tensor.
    """

    @abstractmethod
    def map_object(self, obj: SetObject) -> SetObject:
        """Apply the functor to an object.

        Parameters
        ----------
        obj : SetObject
            Source object.

        Returns
        -------
        SetObject
            Image object.
        """
        ...

    @abstractmethod
    def map_morphism(self, morph: Morphism) -> FunctorMorphism:
        """Apply the functor to a morphism.

        Returns a lazy FunctorMorphism that recomputes its tensor
        from the inner morphism, preserving gradient flow.

        Parameters
        ----------
        morph : Morphism
            Source morphism.

        Returns
        -------
        FunctorMorphism
            Image morphism.
        """
        ...

    @abstractmethod
    def map_tensor(self, tensor: torch.Tensor, algebra: Algebra) -> torch.Tensor:
        """Apply the functor's action on the raw tensor level.

        This is the computational core used by FunctorMorphism.tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The inner morphism's tensor.
        algebra : Algebra
            The enrichment algebra.

        Returns
        -------
        torch.Tensor
            The image tensor.
        """
        ...


class IdentityFunctor(Functor):
    """The identity endofunctor Id: C → C.

    Maps every object and morphism to itself. Needed as the source
    or target of natural transformations (e.g., monad unit η: Id ⇒ T).
    """

    def map_object(self, obj: SetObject) -> SetObject:
        """Identity on objects: A ↦ A."""
        return obj

    def map_morphism(self, morph: Morphism) -> FunctorMorphism:
        """Identity on morphisms: f ↦ f (wrapped as FunctorMorphism)."""
        return FunctorMorphism(self, morph, morph.domain, morph.codomain)

    def map_tensor(self, tensor: torch.Tensor, algebra: Algebra) -> torch.Tensor:
        """Identity on tensors."""
        return tensor

    def __repr__(self) -> str:
        return "IdentityFunctor()"


class ComposedFunctor(Functor):
    """Composition of two endofunctors: F ∘ G.

    Applies G first, then F. Preserves functoriality:
    (F ∘ G)(f) = F(G(f)).

    Parameters
    ----------
    outer : Functor
        The functor applied second (F).
    inner : Functor
        The functor applied first (G).
    """

    def __init__(self, outer: Functor, inner: Functor) -> None:
        self._outer = outer
        self._inner = inner

    @property
    def outer(self) -> Functor:
        """The outer (second-applied) functor F."""
        return self._outer

    @property
    def inner(self) -> Functor:
        """The inner (first-applied) functor G."""
        return self._inner

    def map_object(self, obj: SetObject) -> SetObject:
        """F(G(A))."""
        return self._outer.map_object(self._inner.map_object(obj))

    def map_morphism(self, morph: Morphism) -> FunctorMorphism:
        """F(G(f)) as a lazy FunctorMorphism."""
        domain = self.map_object(morph.domain)
        codomain = self.map_object(morph.codomain)
        return FunctorMorphism(self, morph, domain, codomain)

    def map_tensor(self, tensor: torch.Tensor, algebra: Algebra) -> torch.Tensor:
        """F's tensor action applied to G's tensor action."""
        inner_result = self._inner.map_tensor(tensor, algebra)
        return self._outer.map_tensor(inner_result, algebra)

    def __repr__(self) -> str:
        return f"ComposedFunctor({self._outer!r}, {self._inner!r})"


class FreeMonoidFunctor(Functor):
    """The length-bounded word functor A ↦ A^{≤L} on V-matrices.

    On objects: A ↦ FreeMonoid(generators=A, max_length=max_length)
    On morphisms: f ↦ f* (block-diagonal, componentwise on each stratum)

    The morphism action assembles a block-diagonal tensor where the k-th
    block is componentwise_lift(f, k) reshaped to 2D. This means strings
    of length k map to strings of length k via componentwise application
    of the underlying morphism.

    Words of each length are mapped to words of the same length, so the
    bound never cuts a word and the functor laws hold whenever the
    algebra is a `Semiring` (the block for length ``k`` is the
    ``k``-fold tensor power, and tensor powers commute with composition
    by distributivity). The truncated free monoid is not closed under
    concatenation, so this functor carries no monad structure.

    Parameters
    ----------
    max_length : int
        Maximum string length for the truncated free monoid.

    Raises
    ------
    ValueError
        If ``max_length`` is negative.
    """

    def __init__(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")

        self._max_length = max_length

    @property
    def max_length(self) -> int:
        """Maximum string length."""
        return self._max_length

    def map_object(self, obj: SetObject) -> FreeMonoid:
        """Apply functor to an object: A ↦ FreeMonoid(generators=A, max_length=max_length).

        For FinSet inputs, uses obj directly as generators. For other
        SetObject types (e.g., FreeMonoid, CoproductSet), creates a
        proxy FinSet with the same cardinality — this is needed for
        iterated application (e.g., the triangle identities of an
        adjunction require F(F(A)) = (A*)*).

        Parameters
        ----------
        obj : SetObject
            The generator set. Typically a FinSet, but any SetObject
            is accepted by treating it as a flat set.

        Returns
        -------
        FreeMonoid
            The free monoid on obj.
        """
        if not isinstance(obj, FinSet):
            # treat any SetObject as a flat set with its total cardinality
            obj = FinSet(name=getattr(obj, "name", repr(obj)), cardinality=obj.size)

        return FreeMonoid(generators=obj, max_length=self._max_length)

    def map_morphism(self, morph: Morphism) -> FunctorMorphism:
        """Apply functor to a morphism: f ↦ f* (block-diagonal).

        Parameters
        ----------
        morph : Morphism
            A morphism between FinSets.

        Returns
        -------
        FunctorMorphism
            Lazy image preserving gradient flow.

        Raises
        ------
        TypeError
            If the morphism's algebra is not a `Semiring`, under which
            composition, and so the functor law, fails.
        """
        if not isinstance(morph.algebra, Semiring):
            raise TypeError(
                "FreeMonoidFunctor is a functor on V-matrices for a Semiring "
                f"algebra; the morphism is over {morph.algebra!r}"
            )
        domain = self.map_object(morph.domain)
        codomain = self.map_object(morph.codomain)

        return FunctorMorphism(self, morph, domain, codomain)

    def map_tensor(self, tensor: torch.Tensor, algebra: Algebra) -> torch.Tensor:
        """Build the block-diagonal tensor for the free monoid action.

        For each k=0..max_length, computes componentwise_lift(f, k),
        reshapes the result to 2D (n_a^k × n_b^k), and assembles all
        blocks via torch.block_diag.

        Parameters
        ----------
        tensor : torch.Tensor
            The inner morphism's 2D tensor of shape (n_a, n_b).
        algebra : Algebra
            The enrichment algebra (passed to componentwise_lift).

        Returns
        -------
        torch.Tensor
            Block-diagonal tensor of shape (total_a, total_b).
        """
        n_a, n_b = tensor.shape
        blocks: list[torch.Tensor] = []

        for k in range(self._max_length + 1):
            lifted = componentwise_lift(tensor, k, algebra=algebra)

            # reshape from (n_a,)*k + (n_b,)*k to (n_a^k, n_b^k)
            rows = n_a**k if k > 0 else 1
            cols = n_b**k if k > 0 else 1
            blocks.append(lifted.reshape(rows, cols))

        return torch.block_diag(*blocks)

    def __repr__(self) -> str:
        return f"FreeMonoidFunctor(max_length={self._max_length})"


# -- module-level singletons ------------------------------------------------

IDENTITY = IdentityFunctor()
"""The `IdentityFunctor` singleton."""

# ``quivers.core.morphisms`` imports this module, so this import follows
# every class definition.
from quivers.core.morphisms import Morphism, FunctorMorphism  # noqa: E402


__all__ = [
    "Functor",
    "IdentityFunctor",
    "ComposedFunctor",
    "FreeMonoidFunctor",
    "IDENTITY",
]
