"""Algebras and coalgebras for monads and comonads.

An algebra (A, α) for a monad (T, η, μ) consists of an object A
and a morphism α: T(A) → A satisfying:

    α ∘ η_A = id_A           (unit law)
    α ∘ μ_A = α ∘ T(α)       (associativity)

Dually, a coalgebra (A, γ) for a comonad (W, ε, δ) consists of an
object A and a morphism γ: A → W(A) satisfying:

    ε_A ∘ γ = id_A           (counit law)
    δ_A ∘ γ = W(γ) ∘ γ       (coassociativity)

The Eilenberg-Moore category of a monad T has T-algebras as objects
and algebra homomorphisms as morphisms [Mac Lane (1998), §VI.2].

This module provides:

    Algebra (abstract)
    ├── FreeAlgebra            — free T-algebra μ_A: T(T(A)) → T(A)
    └── ObservedAlgebra        — algebra from a concrete structure map

    Coalgebra (abstract)
    ├── CofreeCoalgebra        — cofree W-coalgebra δ_A: W(A) → W(W(A))
    └── ObservedCoalgebra      — coalgebra from a concrete structure map

    EilenbergMooreCategory     — the category of algebras for a monad

Monad algebras use the typeclass interface of
[`quivers.monadic.typeclasses.Monad`][quivers.monadic.typeclasses.Monad]
(``fmap_obj``, ``fmap``, ``pure``, ``join``), so they apply to every
monad instance. `ObservedAlgebra` and `ObservedCoalgebra` check their
laws when constructed and refuse a structure map that fails them.

References
----------
Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd
ed.). Springer.
[doi:10.1007/978-1-4757-4721-8](https://doi.org/10.1007/978-1-4757-4721-8)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.algebras import Algebra as VAlgebra
from quivers.core.morphisms import Morphism, identity, observed
from quivers.core.objects import SetObject
from quivers.monadic.comonads import Comonad
from quivers.monadic.typeclasses import Monad


def _close(left: torch.Tensor, right: torch.Tensor, atol: float) -> bool:
    """Tensor approximate equality, false on a shape mismatch.

    Parameters
    ----------
    left, right : torch.Tensor
        The tensors to compare.
    atol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if the shapes agree and the entries agree within ``atol``.
    """
    if left.shape != right.shape:
        return False
    return bool(torch.allclose(left, right, atol=atol))


class Algebra(ABC):
    """Abstract algebra for a monad.

    An algebra (A, α) for monad T consists of a carrier object A
    and a structure map α: T(A) → A.

    Parameters
    ----------
    monad : Monad
        The monad T.
    carrier : SetObject
        The carrier object A.
    """

    def __init__(self, monad: Monad, carrier: SetObject) -> None:
        self._monad = monad
        self._carrier = carrier

    @property
    def monad(self) -> Monad:
        """The monad T."""
        return self._monad

    @property
    def carrier(self) -> SetObject:
        """The carrier object A."""
        return self._carrier

    @abstractmethod
    def structure_map(self) -> Morphism:
        """The structure map α: T(A) → A.

        Returns
        -------
        Morphism
            The algebra structure map.
        """
        ...

    def verify_unit_law(self, atol: float = 1e-5) -> bool:
        """Verify α ∘ η_A = id_A.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if the unit law holds within tolerance.
        """
        alpha = self.structure_map()
        eta = self._monad.pure(self._carrier)
        result = (eta >> alpha).tensor
        expected = identity(self._carrier, algebra=alpha.algebra).tensor
        return _close(result, expected, atol)

    def verify_associativity(self, atol: float = 1e-5) -> bool:
        """Verify α ∘ μ_A = α ∘ T(α).

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if associativity holds within tolerance.
        """
        alpha = self.structure_map()
        t_carrier = self._monad.fmap_obj(self._carrier)
        left = (self._monad.join(self._carrier) >> alpha).tensor
        t_alpha = self._monad.fmap(t_carrier, self._carrier, alpha)
        right = (t_alpha >> alpha).tensor
        return _close(left, right, atol)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._monad!r}, carrier={self._carrier!r})"


class FreeAlgebra(Algebra):
    """The free T-algebra on an object A.

    The carrier is T(A) and the structure map is μ_A: T(T(A)) → T(A).
    Its laws are the monad's associativity and left unit law, so it is
    an algebra whenever the monad is lawful.

    Parameters
    ----------
    monad : Monad
        The monad T.
    obj : SetObject
        The base object A (the carrier will be T(A)).
    """

    def __init__(self, monad: Monad, obj: SetObject) -> None:
        super().__init__(monad, monad.fmap_obj(obj))
        self._base = obj

    @property
    def base(self) -> SetObject:
        """The base object A (carrier is T(A))."""
        return self._base

    def structure_map(self) -> Morphism:
        """μ_A: T(T(A)) → T(A)."""
        return self._monad.join(self._base)


class ObservedAlgebra(Algebra):
    """An algebra defined by an explicit structure map tensor.

    Parameters
    ----------
    monad : Monad
        The monad T.
    carrier : SetObject
        The carrier object A.
    structure_tensor : torch.Tensor
        The tensor for α: T(A) → A.
    algebra : quivers.core.algebras.Algebra
        The enrichment algebra of the structure map, the one the
        monad's morphisms are over.
    atol : float
        Absolute tolerance of the law checks.

    Raises
    ------
    ValueError
        If the structure map fails the unit law or associativity.
    """

    def __init__(
        self,
        monad: Monad,
        carrier: SetObject,
        structure_tensor: torch.Tensor,
        algebra: VAlgebra,
        atol: float = 1e-5,
    ) -> None:
        super().__init__(monad, carrier)
        t_carrier = monad.fmap_obj(carrier)
        self._morphism = observed(t_carrier, carrier, structure_tensor, algebra=algebra)
        if not self.verify_unit_law(atol):
            raise ValueError(
                "ObservedAlgebra: the structure map fails the unit law α ∘ η = id"
            )
        if not self.verify_associativity(atol):
            raise ValueError(
                "ObservedAlgebra: the structure map fails associativity "
                "α ∘ μ = α ∘ T(α)"
            )

    def structure_map(self) -> Morphism:
        """The stored structure map α: T(A) → A."""
        return self._morphism


class Coalgebra(ABC):
    """Abstract coalgebra for a comonad.

    A coalgebra (A, γ) for comonad W consists of a carrier object A
    and a structure map γ: A → W(A).

    Parameters
    ----------
    comonad : Comonad
        The comonad W.
    carrier : SetObject
        The carrier object A.
    """

    def __init__(self, comonad: Comonad, carrier: SetObject) -> None:
        self._comonad = comonad
        self._carrier = carrier

    @property
    def comonad(self) -> Comonad:
        """The comonad W."""
        return self._comonad

    @property
    def carrier(self) -> SetObject:
        """The carrier object A."""
        return self._carrier

    @abstractmethod
    def structure_map(self) -> Morphism:
        """The structure map γ: A → W(A).

        Returns
        -------
        Morphism
            The coalgebra structure map.
        """
        ...

    def verify_counit_law(self, atol: float = 1e-5) -> bool:
        """Verify ε_A ∘ γ = id_A.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if the counit law holds within tolerance.
        """
        gamma = self.structure_map()
        eps = self._comonad.counit(self._carrier)
        result = (gamma >> eps).tensor
        expected = identity(self._carrier, algebra=gamma.algebra).tensor
        return _close(result, expected, atol)

    def verify_coassociativity(self, atol: float = 1e-5) -> bool:
        """Verify δ_A ∘ γ = W(γ) ∘ γ.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if coassociativity holds within tolerance.
        """
        gamma = self.structure_map()
        delta = self._comonad.comultiply(self._carrier)
        left = (gamma >> delta).tensor
        w_gamma = self._comonad.endofunctor.map_morphism(gamma)
        right = (gamma >> w_gamma).tensor
        return _close(left, right, atol)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._comonad!r}, carrier={self._carrier!r})"


class CofreeCoalgebra(Coalgebra):
    """The cofree W-coalgebra on an object A.

    The carrier is W(A) and the structure map is δ_A: W(A) → W(W(A)).
    Its laws are the comonad's coassociativity and left counit law.

    Parameters
    ----------
    comonad : Comonad
        The comonad W.
    obj : SetObject
        The base object A (the carrier will be W(A)).
    """

    def __init__(self, comonad: Comonad, obj: SetObject) -> None:
        super().__init__(comonad, comonad.endofunctor.map_object(obj))
        self._base = obj

    @property
    def base(self) -> SetObject:
        """The base object A (carrier is W(A))."""
        return self._base

    def structure_map(self) -> Morphism:
        """δ_A: W(A) → W(W(A))."""
        return self._comonad.comultiply(self._base)


class ObservedCoalgebra(Coalgebra):
    """A coalgebra defined by an explicit structure map tensor.

    Parameters
    ----------
    comonad : Comonad
        The comonad W.
    carrier : SetObject
        The carrier object A.
    structure_tensor : torch.Tensor
        The tensor for γ: A → W(A).
    algebra : quivers.core.algebras.Algebra
        The enrichment algebra of the structure map, the one the
        comonad's morphisms are over.
    atol : float
        Absolute tolerance of the law checks.

    Raises
    ------
    ValueError
        If the structure map fails the counit law or coassociativity.
    """

    def __init__(
        self,
        comonad: Comonad,
        carrier: SetObject,
        structure_tensor: torch.Tensor,
        algebra: VAlgebra,
        atol: float = 1e-5,
    ) -> None:
        super().__init__(comonad, carrier)
        w_carrier = comonad.endofunctor.map_object(carrier)
        self._morphism = observed(carrier, w_carrier, structure_tensor, algebra=algebra)
        if not self.verify_counit_law(atol):
            raise ValueError(
                "ObservedCoalgebra: the structure map fails the counit law ε ∘ γ = id"
            )
        if not self.verify_coassociativity(atol):
            raise ValueError(
                "ObservedCoalgebra: the structure map fails coassociativity "
                "δ ∘ γ = W(γ) ∘ γ"
            )

    def structure_map(self) -> Morphism:
        """The stored structure map γ: A → W(A)."""
        return self._morphism


class EilenbergMooreCategory:
    """The Eilenberg-Moore category of a monad.

    Objects are T-algebras. Morphisms (A, α) → (B, β) are base
    morphisms f: A → B satisfying β ∘ T(f) = f ∘ α, i.e., algebra
    homomorphisms.

    Parameters
    ----------
    monad : Monad
        The underlying monad T.
    """

    def __init__(self, monad: Monad) -> None:
        self._monad = monad

    @property
    def monad(self) -> Monad:
        """The underlying monad."""
        return self._monad

    def free_algebra(self, obj: SetObject) -> FreeAlgebra:
        """Construct the free T-algebra on an object.

        Parameters
        ----------
        obj : SetObject
            The base object A.

        Returns
        -------
        FreeAlgebra
            The free algebra with carrier T(A).
        """
        return FreeAlgebra(self._monad, obj)

    def is_homomorphism(
        self,
        f: Morphism,
        source_alg: Algebra,
        target_alg: Algebra,
        atol: float = 1e-5,
    ) -> bool:
        """Check if f: A → B is an algebra homomorphism.

        Verifies β ∘ T(f) = f ∘ α.

        Parameters
        ----------
        f : Morphism
            A morphism A → B between the carriers.
        source_alg : Algebra
            The source algebra (A, α).
        target_alg : Algebra
            The target algebra (B, β).
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if f is a homomorphism within tolerance.
        """
        alpha = source_alg.structure_map()
        beta = target_alg.structure_map()
        tf = self._monad.fmap(source_alg.carrier, target_alg.carrier, f)
        left = (tf >> beta).tensor
        right = (alpha >> f).tensor
        return _close(left, right, atol)

    def __repr__(self) -> str:
        return f"EilenbergMooreCategory({self._monad!r})"


__all__ = [
    "Algebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
]
