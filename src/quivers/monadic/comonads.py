"""Comonads and coKleisli categories on V-enriched FinSet.

A comonad (W, ε, δ) consists of an endofunctor W, a natural counit
ε: W ⇒ Id, and a natural comultiplication δ: W ⇒ W² satisfying:

    W(δ) ∘ δ = δ_W ∘ δ     (coassociativity)
    ε_W ∘ δ = id             (left counit)
    W(ε) ∘ δ = id            (right counit)

The coKleisli category of a comonad has the same objects but morphisms
A → B are morphisms W(A) → B in the base category, composed via:

    f =>> g = g ∘ W(f) ∘ δ_A

The base category here is the category of ``V``-valued matrices, which
is a category when ``V`` is a `Semiring`, so the comonads below require
a semiring algebra.

This module provides the abstract `Comonad`, the `CoKleisliCategory`
adapter, and `CofreeComonad`, the product (environment) comonad
``W(A) = A × S``.

References
----------
Uustalu, T., and Vene, V. (2008). Comonadic notions of computation.
*Electronic Notes in Theoretical Computer Science*, 203(5), 263-284.
[doi:10.1016/j.entcs.2008.05.029](https://doi.org/10.1016/j.entcs.2008.05.029)
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, ProductSet
from quivers.core.algebras import Algebra, Semiring
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
        delta = self.comultiply(self.base_of(f.domain))
        wf = self.endofunctor.map_morphism(f)
        return delta >> wf >> g

    @abstractmethod
    def base_of(self, wa: SetObject) -> SetObject:
        """Recover ``A`` from an object of the form ``W(A)``.

        Parameters
        ----------
        wa : SetObject
            An object ``W(A)``.

        Returns
        -------
        SetObject
            The object ``A``.

        Raises
        ------
        ValueError
            If ``wa`` is not of the form ``W(A)``.
        """
        ...


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


class CofreeComonad(Comonad):
    """The product (environment) comonad W(A) = A × S.

    This is the comonad of values in an environment ``S``
    [Uustalu and Vene (2008)], and it is the cofree comonad on the
    constant functor at ``S``. It is not the store (costate) comonad,
    which is ``S × (S → A)``.

    ε_A: A × S → A is the first projection (extract the value).
    δ_A: A × S → (A × S) × S duplicates the environment component.
    W(f) = f × id_S.

    Only the environment is copied, and morphisms act on it as the
    identity, so ε and δ are natural for every ``V``-valued morphism,
    not only for functions.

    Parameters
    ----------
    store : SetObject
        The environment object S.
    algebra : Semiring
        The enrichment algebra; a semiring, so that the base category
        and the coKleisli category are categories.

    Raises
    ------
    TypeError
        If ``algebra`` is not a `Semiring`.
    """

    def __init__(self, store: SetObject, algebra: Semiring) -> None:
        if not isinstance(algebra, Semiring):
            raise TypeError(
                "CofreeComonad requires a Semiring algebra, so that its "
                f"coKleisli composition is associative; got {algebra!r}"
            )
        self._store = store
        self._algebra = algebra
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
        source = ProductSet(components=(obj, self._store))
        target = ProductSet(components=(obj, self._store, self._store))
        data = torch.full((*source.shape, *target.shape), self._algebra.zero)
        for a_idx in itertools.product(*(range(s) for s in obj.shape)):
            for s_idx in itertools.product(*(range(s) for s in self._store.shape)):
                src = a_idx + s_idx
                tgt = a_idx + s_idx + s_idx
                data[src + tgt] = self._algebra.unit
        return observed(source, target, data, algebra=self._algebra)

    def base_of(self, wa: SetObject) -> SetObject:
        """Recover ``A`` from ``W(A) = A × S``.

        Parameters
        ----------
        wa : SetObject
            An object ``A × S``.

        Returns
        -------
        SetObject
            The object ``A``.

        Raises
        ------
        ValueError
            If ``wa`` is not a binary product whose second factor is
            the environment ``S``.
        """
        if (
            isinstance(wa, ProductSet)
            and len(wa.components) == 2
            and wa.components[1] == self._store
        ):
            return wa.components[0]
        raise ValueError(
            f"CofreeComonad: {wa!r} is not of the form A × {self._store!r}"
        )

    def __repr__(self) -> str:
        return f"CofreeComonad(store={self._store!r}, algebra={self._algebra!r})"


class _StoreFunctor(Functor):
    """The product functor W(A) = A × S for a fixed S.

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
    "CofreeComonad",
]
