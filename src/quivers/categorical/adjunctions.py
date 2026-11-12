"""Adjunctions between endofunctors.

An adjunction F ⊣ G consists of functors F (left adjoint) and G
(right adjoint), together with natural transformations:

    η: Id ⇒ G ∘ F    (unit)
    ε: F ∘ G ⇒ Id    (counit)

satisfying the triangle identities:

    ε_{F(A)} ∘ F(η_A) = id_{F(A)}
    G(ε_A) ∘ η_{G(A)} = id_{G(A)}

Every adjunction induces a monad T = G ∘ F with η as unit and
μ = G(ε_F) as multiplication.

This module provides the abstract `Adjunction` interface and its
triangle-identity checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, identity
from quivers.categorical.functors import (
    Functor,
)


class Adjunction(ABC):
    """Abstract adjunction F ⊣ G.

    Subclasses must implement left, right, unit_component, and
    counit_component.
    """

    @property
    @abstractmethod
    def left(self) -> Functor:
        """The left adjoint F."""
        ...

    @property
    @abstractmethod
    def right(self) -> Functor:
        """The right adjoint G."""
        ...

    @abstractmethod
    def unit_component(self, obj: SetObject) -> Morphism:
        """The unit component η_A: A → G(F(A)).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The unit morphism η_A.
        """
        ...

    @abstractmethod
    def counit_component(self, obj: SetObject) -> Morphism:
        """The counit component ε_A: F(G(A)) → A.

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

    def verify_triangle_left(self, obj: SetObject, atol: float = 1e-5) -> bool:
        """Verify left triangle identity: ε_{F(A)} ∘ F(η_A) ≈ id_{F(A)}.

        Parameters
        ----------
        obj : SetObject
            The object A.
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if the identity holds within tolerance.
        """
        fa = self.left.map_object(obj)
        eta_a = self.unit_component(obj)
        f_eta = self.left.map_morphism(eta_a)
        eps_fa = self.counit_component(fa)
        result = (f_eta >> eps_fa).tensor
        expected = identity(fa).tensor

        return torch.allclose(result, expected, atol=atol)

    def verify_triangle_right(self, obj: SetObject, atol: float = 1e-5) -> bool:
        """Verify right triangle identity: G(ε_A) ∘ η_{G(A)} ≈ id_{G(A)}.

        Parameters
        ----------
        obj : SetObject
            The object A.
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if the identity holds within tolerance.
        """
        ga = self.right.map_object(obj)
        eps_a = self.counit_component(obj)
        g_eps = self.right.map_morphism(eps_a)
        eta_ga = self.unit_component(ga)
        result = (eta_ga >> g_eps).tensor
        expected = identity(ga).tensor

        return torch.allclose(result, expected, atol=atol)

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.left!r} ⊣ {self.right!r})"


__all__ = [
    "Adjunction",
]
