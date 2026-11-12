"""Profunctors (bimodules) for V-enriched categories.

A V-profunctor R: A ↛ B is a V-valued functor A^op ⊗ B → V,
represented here by a tensor of shape (*A.shape, *B.shape).
The implementation treats finite weighted relations as profunctor
representations; it does not record arbitrary source and target actions.

Between discrete V-categories (finite sets), a profunctor is a V-valued
matrix, and composition uses the coend formula:

    (R ; S)(a, c) = ∫^b R(a, b) ⊗ S(b, c)

which is matrix composition over V. The coend is a join, so V must be a
complete lattice (an `IdempotentSemiring`); profunctors then compose
associatively with the identity matrices as units. This module makes the
profunctor structure explicit and provides composition via the
coend from ends_coends.py.

This module provides:

    Profunctor — explicit profunctor with contravariant/covariant structure
"""

from __future__ import annotations

import torch

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, ObservedMorphism, observed
from quivers.core.algebras import IdempotentSemiring
from quivers.enriched._lattice import require_lattice


class Profunctor:
    """A V-profunctor R: A ↛ B.

    Wraps a tensor of shape (*A.shape, *B.shape) with explicit
    contravariant (A) and covariant (B) objects.

    Parameters
    ----------
    contra : SetObject
        The contravariant (domain/source) object A.
    co : SetObject
        The covariant (codomain/target) object B.
    tensor : torch.Tensor
        The profunctor's tensor of shape (*A.shape, *B.shape).
    algebra : IdempotentSemiring
        The enrichment algebra, a complete lattice.

    Raises
    ------
    TypeError
        If ``algebra`` is not an `IdempotentSemiring`.
    ValueError
        If the tensor's shape is not ``(*contra.shape, *co.shape)``.
    """

    def __init__(
        self,
        contra: SetObject,
        co: SetObject,
        tensor: torch.Tensor,
        algebra: IdempotentSemiring,
    ) -> None:
        self._contra = contra
        self._co = co
        self._tensor = tensor
        self._algebra = require_lattice(algebra, "a profunctor")

        expected_shape = (*contra.shape, *co.shape)

        if tensor.shape != expected_shape:
            raise ValueError(
                f"tensor shape {tensor.shape} does not match expected "
                f"{expected_shape} from contra={contra!r}, co={co!r}"
            )

    @property
    def contra(self) -> SetObject:
        """The contravariant (source) object."""
        return self._contra

    @property
    def co(self) -> SetObject:
        """The covariant (target) object."""
        return self._co

    @property
    def tensor(self) -> torch.Tensor:
        """The profunctor tensor."""
        return self._tensor

    @property
    def algebra(self) -> IdempotentSemiring:
        """The enrichment algebra."""
        return self._algebra

    @classmethod
    def from_morphism(cls, morph: Morphism) -> Profunctor:
        """Wrap a finite weighted morphism as a profunctor representation.

        Parameters
        ----------
        morph : Morphism
            A morphism R: A → B.

        Returns
        -------
        Profunctor
            The profunctor R: A ↛ B with the same tensor.

        Raises
        ------
        TypeError
            If the morphism's algebra is not an `IdempotentSemiring`.
        """
        return cls(
            contra=morph.domain,
            co=morph.codomain,
            tensor=morph.tensor,
            algebra=require_lattice(morph.algebra, "a profunctor"),
        )

    def compose(self, other: Profunctor) -> Profunctor:
        """Profunctor composition via coend.

        (R ; S)(a, c) = ∫^b R(a, b) ⊗ S(b, c)

        This is equivalent to V-enriched composition via the
        algebra's compose method.

        Parameters
        ----------
        other : Profunctor
            Right profunctor S: B ↛ C.

        Returns
        -------
        Profunctor
            Composed profunctor R ; S: A ↛ C.

        Raises
        ------
        TypeError
            If the objects do not match or the algebras differ.
        """
        if not self._algebra.is_compatible(other._algebra):
            raise TypeError(
                f"cannot compose a profunctor over {self._algebra.name} with "
                f"one over {other._algebra.name}"
            )
        if self._co != other._contra:
            raise TypeError(
                f"cannot compose: R's covariant object {self._co!r} "
                f"!= S's contravariant object {other._contra!r}"
            )

        # use algebra.compose which is exactly the coend formula
        n_contract = self._co.ndim
        result_tensor = self._algebra.compose(self._tensor, other._tensor, n_contract)

        return Profunctor(
            contra=self._contra,
            co=other._co,
            tensor=result_tensor,
            algebra=self._algebra,
        )

    def to_morphism(self) -> ObservedMorphism:
        """Convert back to a morphism (same tensor, different framing).

        Returns
        -------
        ObservedMorphism
            A morphism with domain=contra, codomain=co.
        """
        return observed(
            self._contra,
            self._co,
            self._tensor,
            algebra=self._algebra,
        )

    def __repr__(self) -> str:
        return f"Profunctor({self._contra!r} ↛ {self._co!r})"


__all__ = [
    "Profunctor",
]
