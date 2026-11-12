"""Change of enrichment (change of base) for V-enriched categories.

A lax monoidal functor F: V → W between algebras induces a
change-of-base 2-functor from V-Cat to W-Cat. This transforms
all morphism tensors by applying F elementwise, changing the
enrichment algebra from V to W.

A subclass is a base change only if its value map is lax monoidal:
I_W ≤ F(I_V), F(a) ⊗ F(b) ≤ F(a ⊗ b), and order-preserving
[Kelly (1982), §1.10]. Thresholding [0, 1] → {0, 1} at an interior
point is not (two entries above the threshold can have a product below
it), so it is offered only as the carrier map
`quivers.core.algebra_morphisms.Threshold`, not as a base change.

This module provides:

    BaseChange (abstract)
    └── BoolToFuzzy   — inclusion {0,1} ↪ [0,1]

References
----------
Kelly, G. M. (1982). *Basic Concepts of Enriched Category Theory*.
Cambridge University Press. Reprinted in *Reprints in Theory and
Applications of Categories*, 10 (2005).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.morphisms import Morphism, ObservedMorphism, observed
from quivers.core.algebras import (
    Algebra,
    ProductFuzzyAlgebra,
    BooleanAlgebra,
    PRODUCT_FUZZY,
    BOOLEAN,
)


class BaseChange(ABC):
    """Abstract change of enrichment from algebra V to algebra W.

    Subclasses must implement source, target, and apply_to_values.
    apply_to_morphism is derived.
    """

    @property
    @abstractmethod
    def source(self) -> Algebra:
        """The source algebra V."""
        ...

    @property
    @abstractmethod
    def target(self) -> Algebra:
        """The target algebra W."""
        ...

    @abstractmethod
    def apply_to_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the value-level map V → W elementwise.

        Parameters
        ----------
        tensor : torch.Tensor
            A V-valued tensor.

        Returns
        -------
        torch.Tensor
            The corresponding W-valued tensor.
        """
        ...

    def apply_to_morphism(self, morph: Morphism) -> ObservedMorphism:
        """Transform a V-morphism to a W-morphism.

        Applies the value-level map to the morphism's tensor and
        returns an observed morphism with the target algebra.

        Parameters
        ----------
        morph : Morphism
            A morphism enriched over V.

        Returns
        -------
        ObservedMorphism
            The same morphism re-enriched over W.
        """
        new_tensor = self.apply_to_values(morph.tensor.detach())

        return observed(
            morph.domain,
            morph.codomain,
            new_tensor,
            algebra=self.target,
        )

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.source.name} → {self.target.name})"


class BoolToFuzzy(BaseChange):
    """Inclusion of the boolean algebra into the fuzzy algebra.

    The identity map {0, 1} ↪ [0, 1]. Boolean-valued tensors are
    already valid fuzzy-valued tensors, so this is the identity
    on tensor values.
    """

    @property
    def source(self) -> BooleanAlgebra:
        return BOOLEAN

    @property
    def target(self) -> ProductFuzzyAlgebra:
        return PRODUCT_FUZZY

    def apply_to_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """Identity: {0,1} values are already in [0,1]."""
        return tensor.clone()


__all__ = [
    "BaseChange",
    "BoolToFuzzy",
]
