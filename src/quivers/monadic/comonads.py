"""Comonads and coKleisli categories on V-enriched FinSet.

A comonad (W, ε, δ) consists of an endofunctor W, a counit ε: W ⇒ Id,
and a comultiplication δ: W ⇒ W² satisfying:

    W(δ) ∘ δ = δ_W ∘ δ     (coassociativity)
    ε_W ∘ δ = id             (left counit)
    W(ε) ∘ δ = id            (right counit)

The coKleisli category of a comonad has the same objects but morphisms
A → B are morphisms W(A) → B in the base category, composed via:

    f =>> g = g ∘ W(f) ∘ δ_A

This module provides:

    Comonad (abstract)
    ├── CofreeComonad           — cofree comonad from a functor
    └── DiagonalComonad         — W(A) = A × A with diagonal/projection

    CoKleisliCategory — wraps a comonad for composition
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from quivers.core.objects import SetObject, ProductSet
from quivers.core.algebras import PRODUCT_FUZZY, Algebra
from quivers.core.morphisms import FunctorMorphism, Morphism, observed, identity
from quivers.categorical.functors import Functor


class Comonad(ABC):
    """Abstract comonad (W, ε, δ) on V-enriched FinSet.

    Subclasses must implement endofunctor, counit, and comultiply.
    CoKleisli composition is derived.
    """

    @property
    @abstractmethod
    def endofunctor(self) -> Functor:
        """The underlying endofunctor W."""
        ...

    @abstractmethod
    def counit(self, obj: SetObject) -> Morphism:
        """The counit component ε_A: W(A) → A.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The counit morphism ε_A.
        """
        ...

    @abstractmethod
    def comultiply(self, obj: SetObject) -> Morphism:
        """The comultiplication component δ_A: W(A) → W(W(A)).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The comultiplication morphism δ_A.
        """
        ...

    def cokleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """CoKleisli composition: f =>> g = g ∘ W(f) ∘ δ_A.

        For f: W(A) → B and g: W(B) → C, produces W(A) → C.

        Parameters
        ----------
        f : Morphism
            Left coKleisli morphism W(A) → B.
        g : Morphism
            Right coKleisli morphism W(B) → C.

        Returns
        -------
        Morphism
            Composed coKleisli morphism W(A) → C.
        """
        delta = self._comultiply_at_domain(f)
        wf = self.endofunctor.map_morphism(f)
        return delta >> wf >> g

    def _comultiply_at_domain(self, f: Morphism) -> Morphism:
        """Get δ for the appropriate object given f's domain structure.

        Subclasses can override for efficiency.

        Parameters
        ----------
        f : Morphism
            A coKleisli morphism whose domain is W(A).

        Returns
        -------
        Morphism
            The comultiplication δ_A: W(A) → W(W(A)).
        """
        raise NotImplementedError(
            "Subclasses must implement _comultiply_at_domain or override cokleisli_compose"
        )


class CoKleisliCategory:
    """The coKleisli category of a comonad.

    Objects are the same as the base category. Morphisms A → B
    in the coKleisli category are morphisms W(A) → B in the base
    category.

    Parameters
    ----------
    comonad : Comonad
        The underlying comonad.
    """

    def __init__(self, comonad: Comonad) -> None:
        self._comonad = comonad

    @property
    def comonad(self) -> Comonad:
        """The underlying comonad."""
        return self._comonad

    def identity(self, obj: SetObject) -> Morphism:
        """CoKleisli identity at object A: ε_A: W(A) → A.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The coKleisli identity morphism.
        """
        return self._comonad.counit(obj)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """CoKleisli composition: f =>> g.

        Parameters
        ----------
        f : Morphism
            Left morphism W(A) → B.
        g : Morphism
            Right morphism W(B) → C.

        Returns
        -------
        Morphism
            Composed morphism W(A) → C.
        """
        return self._comonad.cokleisli_compose(f, g)


class DiagonalComonad(Comonad):
    """Pair operations W(A) = A × A with projection and diagonal.

    ε_A: A × A → A is the first projection.
    δ_A: A × A → (A × A) × (A × A) is the diagonal on pairs.

    This historical interface name does not describe a lawful comonad
    in general. With W(f) = f × f, W(ε) after δ sends (a, b) to
    (a, a), which violates the right counit law when a differs from b.
    The methods expose the stated pair operations, not a construction
    of the comonad induced by the product adjunction.

    Parameters
    ----------
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(self, algebra: Algebra | None = None) -> None:
        self._algebra = algebra if algebra is not None else PRODUCT_FUZZY
        self._functor = _DiagonalFunctor()

    @property
    def algebra(self) -> Algebra:
        """The enrichment algebra."""
        return self._algebra

    @property
    def endofunctor(self) -> Functor:
        """W(A) = A × A."""
        return self._functor

    def counit(self, obj: SetObject) -> Morphism:
        """ε_A: A × A → A is first projection.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        ObservedMorphism
            The projection morphism.
        """
        import itertools

        source = ProductSet(components=(obj, obj))
        data = torch.full((*source.shape, *obj.shape), self._algebra.zero)
        for idx in itertools.product(*(range(s) for s in obj.shape)):
            for idx2 in itertools.product(*(range(s) for s in obj.shape)):
                data[idx + idx2 + idx] = self._algebra.unit
        return observed(source, obj, data, algebra=self._algebra)

    def comultiply(self, obj: SetObject) -> Morphism:
        """δ_A: A × A → (A × A) × (A × A) is diagonal on pairs.

        Maps (a1, a2) to ((a1, a2), (a1, a2)).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        ObservedMorphism
            The comultiplication morphism.
        """
        import itertools

        source = ProductSet(components=(obj, obj))
        target = ProductSet(components=(obj, obj, obj, obj))
        data = torch.full((*source.shape, *target.shape), self._algebra.zero)
        for idx1 in itertools.product(*(range(s) for s in obj.shape)):
            for idx2 in itertools.product(*(range(s) for s in obj.shape)):
                src = idx1 + idx2
                tgt = idx1 + idx2 + idx1 + idx2
                data[src + tgt] = self._algebra.unit
        return observed(source, target, data, algebra=self._algebra)

    def cokleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """CoKleisli composition using the diagonal comonad."""
        delta = self.comultiply(self._extract_base(f.domain))
        wf = self._functor.map_morphism(f)
        return delta >> wf >> g

    def _comultiply_at_domain(self, f: Morphism) -> Morphism:
        return self.comultiply(self._extract_base(f.domain))

    def _extract_base(self, wa: SetObject) -> SetObject:
        """Extract A from W(A) = A × A.

        Parameters
        ----------
        wa : SetObject
            Must be a ProductSet with two identical components.

        Returns
        -------
        SetObject
            The base object A.
        """
        if isinstance(wa, ProductSet) and len(wa.components) >= 2:
            return wa.components[0]
        return wa

    def __repr__(self) -> str:
        return f"DiagonalComonad({self._algebra!r})"


class CofreeComonad(Comonad):
    """The cofree comonad W(A) = A × S for a fixed store object S.

    Also known as the "store comonad" or "costate comonad" on
    finite sets.

    ε_A: A × S → A is the first projection (extract the value).
    δ_A: A × S → (A × S) × S duplicates the store component.

    Parameters
    ----------
    store : SetObject
        The fixed store/environment object S.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(self, store: SetObject, algebra: Algebra | None = None) -> None:
        self._store = store
        self._algebra = algebra if algebra is not None else PRODUCT_FUZZY
        self._functor = _StoreFunctor(store)

    @property
    def store(self) -> SetObject:
        """The store object S."""
        return self._store

    @property
    def algebra(self) -> Algebra:
        """The enrichment algebra."""
        return self._algebra

    @property
    def endofunctor(self) -> Functor:
        """W(A) = A × S."""
        return self._functor

    def counit(self, obj: SetObject) -> Morphism:
        """ε_A: A × S → A is the first projection.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        ObservedMorphism
            The extraction morphism.
        """
        import itertools

        source = ProductSet(components=(obj, self._store))
        data = torch.full((*source.shape, *obj.shape), self._algebra.zero)
        for a_idx in itertools.product(*(range(s) for s in obj.shape)):
            for s_idx in itertools.product(*(range(s) for s in self._store.shape)):
                data[a_idx + s_idx + a_idx] = self._algebra.unit
        return observed(source, obj, data, algebra=self._algebra)

    def comultiply(self, obj: SetObject) -> Morphism:
        """δ_A: A × S → (A × S) × S duplicates the store.

        Maps (a, s) to ((a, s), s).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        ObservedMorphism
            The comultiplication morphism.
        """
        import itertools

        source = ProductSet(components=(obj, self._store))
        target = ProductSet(components=(obj, self._store, self._store))
        data = torch.full((*source.shape, *target.shape), self._algebra.zero)
        for a_idx in itertools.product(*(range(s) for s in obj.shape)):
            for s_idx in itertools.product(*(range(s) for s in self._store.shape)):
                src = a_idx + s_idx
                tgt = a_idx + s_idx + s_idx
                data[src + tgt] = self._algebra.unit
        return observed(source, target, data, algebra=self._algebra)

    def _comultiply_at_domain(self, f: Morphism) -> Morphism:
        dom = f.domain
        if isinstance(dom, ProductSet) and len(dom.components) >= 2:
            base = dom.components[0]
        else:
            base = dom
        return self.comultiply(base)

    def __repr__(self) -> str:
        return f"CofreeComonad(store={self._store!r})"


class _DiagonalFunctor(Functor):
    """The diagonal functor W(A) = A × A.

    On objects: A ↦ A × A.
    On morphisms: f ↦ f × f (product morphism).
    """

    def map_object(self, obj: SetObject) -> ProductSet:
        """A ↦ A × A."""
        return ProductSet(components=(obj, obj))

    def map_morphism(self, morph: Morphism) -> FunctorMorphism:
        """f ↦ f × f (parallel product)."""
        result = morph @ morph
        return FunctorMorphism(self, morph, result.domain, result.codomain)

    def map_tensor(self, tensor: torch.Tensor, algebra: Algebra) -> torch.Tensor:
        """Compute (f ⊗ f) tensor from f's tensor."""
        n = tensor.ndim
        half = n // 2
        shape_l = list(tensor.shape) + [1] * n
        shape_r = [1] * n + list(tensor.shape)
        outer = algebra.tensor_op(tensor.reshape(shape_l), tensor.reshape(shape_r))
        dom_l = list(range(half))
        cod_l = list(range(half, n))
        dom_r = list(range(n, n + half))
        cod_r = list(range(n + half, 2 * n))
        perm = dom_l + dom_r + cod_l + cod_r
        return outer.permute(*perm)

    def __repr__(self) -> str:
        return "DiagonalFunctor()"


class _StoreFunctor(Functor):
    """The store functor W(A) = A × S for a fixed S.

    On objects: A ↦ A × S.
    On morphisms: f ↦ f × id_S.
    """

    def __init__(self, store: SetObject) -> None:
        self._store = store

    def map_object(self, obj: SetObject) -> ProductSet:
        """A ↦ A × S."""
        return ProductSet(components=(obj, self._store))

    def map_morphism(self, morph: Morphism) -> FunctorMorphism:
        """f ↦ f × id_S."""
        id_s = identity(self._store, algebra=morph.algebra)
        result = morph @ id_s
        return FunctorMorphism(self, morph, result.domain, result.codomain)

    def map_tensor(self, tensor: torch.Tensor, algebra: Algebra) -> torch.Tensor:
        """Compute (f ⊗ id_S) tensor."""
        id_tensor = algebra.identity_tensor(self._store.shape)
        n_f = tensor.ndim
        n_id = id_tensor.ndim
        shape_l = list(tensor.shape) + [1] * n_id
        shape_r = [1] * n_f + list(id_tensor.shape)
        outer = algebra.tensor_op(tensor.reshape(shape_l), id_tensor.reshape(shape_r))
        half_f = n_f // 2
        half_id = n_id // 2
        dom_f = list(range(half_f))
        cod_f = list(range(half_f, n_f))
        dom_s = list(range(n_f, n_f + half_id))
        cod_s = list(range(n_f + half_id, n_f + n_id))
        perm = dom_f + dom_s + cod_f + cod_s
        return outer.permute(*perm)

    def __repr__(self) -> str:
        return f"StoreFunctor(store={self._store!r})"


__all__ = [
    "Comonad",
    "CoKleisliCategory",
    "DiagonalComonad",
    "CofreeComonad",
]
