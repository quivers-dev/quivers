"""Distributive laws for composing monads.

A distributive law λ: S ∘ T ⇒ T ∘ S between monads S and T allows
their composition T ∘ S to form a new monad. The component
λ_A: S(T(A)) → T(S(A)) "pushes S past T."

This module provides the abstract `DistributiveLaw` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism
from quivers.monadic.typeclasses import Monad


class DistributiveLaw(ABC):
    """Abstract distributive law λ: S ∘ T ⇒ T ∘ S.

    Subclasses must implement outer_monad, inner_monad, and distribute.
    """

    @property
    @abstractmethod
    def outer_monad(self) -> Monad:
        """The outer monad S (applied second in S ∘ T)."""
        ...

    @property
    @abstractmethod
    def inner_monad(self) -> Monad:
        """The inner monad T (applied first in S ∘ T)."""
        ...

    @abstractmethod
    def distribute(self, obj: SetObject) -> Morphism:
        """The component λ_A: S(T(A)) → T(S(A)).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The distributive morphism λ_A.
        """
        ...


__all__ = [
    "DistributiveLaw",
]
