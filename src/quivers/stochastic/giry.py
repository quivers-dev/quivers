"""Finite stochastic-matrix operations inspired by the Giry monad.

The mathematical Giry monad maps a measurable space to its space of
probability measures. Even for a two-element space, that space of measures
is infinite, so it is not an endofunctor on FinSet. This module instead
keeps finite index objects and represents kernels as stochastic tensors:

    G(A) = A   (at the set level, since distributions are represented
                as row-stochastic tensors, not as elements of a simplex)

This is structurally identical to the fuzzy powerset monad but with
a different composition rule: sum-product (matrix multiplication)
instead of noisy-OR.

    η_A: A → A: the Kronecker delta (deterministic distribution)
    μ_A: A → A: identity (flattening nested distributions)
    f >=> g = f >> g: Kleisli composition = matrix multiplication

FinStoch is the finite-object subcategory of probability kernels, whose
composition is matrix multiplication. The identity object map below is
a representation convention, not equality of a set with its simplex.

This module provides:

    GiryMonad: the probability monad (T, η, μ) with MarkovAlgebra
    FinStoch: the Kleisli category of GiryMonad
"""

from __future__ import annotations

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, identity
from quivers.categorical.functors import Functor, IDENTITY
from quivers.monadic.monads import KleisliCategory
from quivers.monadic.typeclasses import Monad
from quivers.core.algebras import MARKOV


class GiryMonad(Monad):
    """The identity monad on FinStoch, presenting the finite Giry monad's Kleisli category.

    On finite sets the Giry monad is the distribution monad, and its
    Kleisli category is FinStoch: stochastic matrices under matrix
    multiplication. This class works in that Kleisli category, so as a
    monad on it ``G(A) = A``, ``fmap(f) = f`` and ``η = μ = id``: it is
    the identity monad on FinStoch, and its Kleisli composition is the
    composition of FinStoch.

    At the finite-set level, G(A) = A because probability distributions
    over A are represented as tensors indexed by A (i.e., functions
    A → [0,1] that sum to 1), not as elements of a separate simplex
    object.

    The Kleisli composition uses the MarkovAlgebra (sum-product),
    yielding standard matrix multiplication of stochastic matrices.

    Mathematically a kernel A → B corresponds to a measurable map
    A → G(B), where G(B) is a space of distributions, not B itself.
    This adapter retains B as the tensor's output index object rather
    than constructing that distribution space.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.stochastic.giry import GiryMonad, FinStoch
    >>> G = GiryMonad()
    >>> A = FinSet(name="A", cardinality=3)
    >>> eta = G.unit(A)  # Kronecker delta: shape (3, 3)
    >>> finstoch = FinStoch()
    >>> # compose stochastic morphisms via finstoch.compose(f, g)
    """

    @property
    def endofunctor(self) -> Functor:
        """G = Id at the set level."""
        return IDENTITY

    # Typeclass interface
    def fmap_obj(self, A: SetObject) -> SetObject:
        """Act on an object: ``G(A) = A`` at the finite-set level.

        Parameters
        ----------
        A : SetObject
            The finite set.

        Returns
        -------
        SetObject
            ``A`` itself.
        """
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        """Act on a morphism: ``G(f) = f`` at the finite-set level.

        Parameters
        ----------
        A : SetObject
            The domain of ``f``.
        B : SetObject
            The codomain of ``f``.
        f : Morphism
            The morphism to map.

        Returns
        -------
        Morphism
            ``f`` itself.
        """
        return f

    def pure(self, A: SetObject) -> Morphism:
        """``η_A = δ``, the Kronecker delta (deterministic distribution).

        Parameters
        ----------
        A : SetObject
            The finite set.

        Returns
        -------
        Morphism
            The identity Markov kernel on ``A``.
        """
        return identity(A, algebra=MARKOV)

    def join(self, A: SetObject) -> Morphism:
        """Return the identity kernel used by this finite representation.

        This is not the general Giry multiplication on a space of
        probability measures over probability measures. Averaging
        finite kernel outputs is realized by sum-product composition.

        Parameters
        ----------
        A : SetObject
            The finite set.

        Returns
        -------
        Morphism
            The identity Markov kernel on ``A``.
        """
        return identity(A, algebra=MARKOV)

    # Aliases in the classical Eilenberg-Moore vocabulary.
    def unit(self, A: SetObject) -> Morphism:
        """The monad unit ``η_A``, an alias of `pure`.

        Parameters
        ----------
        A : SetObject
            The finite set.

        Returns
        -------
        Morphism
            The identity Markov kernel on ``A``.
        """
        return self.pure(A)

    def multiply(self, A: SetObject) -> Morphism:
        """The monad multiplication ``μ_A``, an alias of `join`.

        Parameters
        ----------
        A : SetObject
            The finite set.

        Returns
        -------
        Morphism
            The identity Markov kernel on ``A``.
        """
        return self.join(A)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition via sum-product (matrix multiplication).

        Parameters
        ----------
        f : Morphism
            The first kernel, ``A -> B``.
        g : Morphism
            The second kernel, ``B -> C``.

        Returns
        -------
        Morphism
            The composite kernel ``f >> g``, ``A -> C``.

        Raises
        ------
        ValueError
            If ``f`` or ``g`` is not over `MARKOV`, since another
            algebra would compose them by a rule other than matrix
            multiplication.
        """
        for kernel in (f, g):
            if not kernel.algebra.is_compatible(MARKOV):
                raise ValueError(
                    f"GiryMonad cannot compose a morphism over {kernel.algebra.name}; "
                    "its Kleisli composition is matrix multiplication over MARKOV"
                )
        return f >> g

    def __repr__(self) -> str:
        return "GiryMonad()"


class FinStoch(KleisliCategory):
    """The category FinStoch of finite sets and stochastic maps.

    This is the Kleisli category of the Giry monad: objects are
    finite sets, morphisms are stochastic matrices (Markov kernels),
    and composition is matrix multiplication.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.stochastic import StochasticMorphism
    >>> from quivers.stochastic.giry import FinStoch
    >>> cat = FinStoch()
    >>> A = FinSet(name="A", cardinality=3)
    >>> B = FinSet(name="B", cardinality=4)
    >>> C = FinSet(name="C", cardinality=2)
    >>> f = StochasticMorphism(A, B)
    >>> g = StochasticMorphism(B, C)
    >>> h = cat.compose(f, g)  # A → C via matrix multiplication
    """

    def __init__(self) -> None:
        super().__init__(GiryMonad())

    def __repr__(self) -> str:
        return "FinStoch()"


__all__ = [
    "GiryMonad",
    "FinStoch",
]
