"""Concrete monad instances on V-enriched FinSet.

The `Monad` typeclass itself lives in
[`quivers.monadic.typeclasses`][quivers.monadic.typeclasses]; this module provides two concrete
monad instances together with the `KleisliCategory` adapter.

- `FuzzyPowersetMonad`: the fuzzy powerset monad over an idempotent
  semiring, whose Kleisli category is the category of V-relations.
- `KleisliCategory` — wraps any `Monad` instance for
  composition.

Both concrete monads subclass `Monad` and implement the
required ``fmap_obj`` / ``fmap`` / ``pure`` / ``join`` operations.
``apply`` is inherited from `Applicative` and raises when the
internal-hom construction is not supplied per-instance.
"""

from __future__ import annotations

from quivers.categorical.functors import (
    IDENTITY,
    Functor,
)
from quivers.core.morphisms import Morphism, identity
from quivers.core.objects import SetObject
from quivers.core.algebras import Algebra, IdempotentSemiring
from quivers.monadic.typeclasses import Monad


class FuzzyPowersetMonad(Monad):
    """The fuzzy powerset monad for an idempotent semiring ``V``.

    The ``V``-powerset monad on sets sends ``A`` to ``V^A``, the
    ``V``-valued fuzzy subsets of ``A``; its Kleisli category is the
    category ``V``-Rel of ``V``-valued relations, with Kleisli
    composition the matrix composition ``⋁_b f(a, b) ⊗ g(b, c)`` and
    Kleisli identities the identity matrices [Hofmann, Seal and Tholen
    (2014)]. This class works directly in that Kleisli category: an
    object ``A`` stands for ``A``, a morphism ``A → B`` over ``V``
    stands for the Kleisli arrow ``A → V^B``, and so ``T(A) = A``,
    ``fmap(f) = f`` and ``η_A = μ_A = id_A``.

    The monad laws, and associativity of Kleisli composition, need
    ``V``-Rel to be a category, which holds when ``V`` is a semiring.
    The powerset reading further needs ``⋁`` to be idempotent, so that
    a fuzzy subset is a join of its points; on the finite carriers used
    here an `IdempotentSemiring` is what a quantale restricts to. The
    constructor therefore requires an `IdempotentSemiring`, such as
    ``BOOLEAN`` (the ordinary powerset monad), ``GODEL`` or
    ``TROPICAL``. ``PRODUCT_FUZZY`` is refused: product does not
    distribute over noisy-OR, so its Kleisli composition depends on
    bracketing.

    Parameters
    ----------
    algebra : IdempotentSemiring
        The enrichment algebra ``V``.

    Raises
    ------
    TypeError
        If ``algebra`` is not an `IdempotentSemiring`.

    References
    ----------
    Hofmann, D., Seal, G. J., and Tholen, W. (Eds.) (2014). *Monoidal
    Topology: A Categorical Approach to Order, Metric, and Topology*.
    Cambridge University Press.
    [doi:10.1017/CBO9781107517288](https://doi.org/10.1017/CBO9781107517288)
    """

    def __init__(self, algebra: IdempotentSemiring) -> None:
        if not isinstance(algebra, IdempotentSemiring):
            raise TypeError(
                "FuzzyPowersetMonad requires an IdempotentSemiring algebra, "
                f"so that its Kleisli composition is associative; got "
                f"{algebra!r}. Use BOOLEAN, GODEL, TROPICAL, MAX_PLUS, or "
                "a dual of BOOLEAN or GODEL."
            )
        self._algebra = algebra

    @property
    def algebra(self) -> Algebra:
        """The enrichment algebra."""
        return self._algebra

    @property
    def endofunctor(self) -> Functor:
        """T = Id (identity functor at set level)."""
        return IDENTITY

    # Typeclass interface
    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def pure(self, A: SetObject) -> Morphism:
        return identity(A, algebra=self._algebra)

    def join(self, A: SetObject) -> Morphism:
        return identity(A, algebra=self._algebra)

    # Eilenberg–Moore vocabulary aliases.
    def unit(self, A: SetObject) -> Morphism:
        """``η_A : A → T(A)``; alias for `pure`."""
        return self.pure(A)

    def multiply(self, A: SetObject) -> Morphism:
        """``μ_A : T(T(A)) → T(A)``; alias for `join`."""
        return self.join(A)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition, the matrix composition of ``V``-Rel.

        Parameters
        ----------
        f : Morphism
            A Kleisli arrow ``A → T(B)``, a ``V``-relation ``A → B``.
        g : Morphism
            A Kleisli arrow ``B → T(C)``, a ``V``-relation ``B → C``.

        Returns
        -------
        Morphism
            The composite ``A → T(C)``.

        Raises
        ------
        ValueError
            If ``f`` or ``g`` is not over this monad's algebra.
        """
        for arrow in (f, g):
            if not arrow.algebra.is_compatible(self._algebra):
                raise ValueError(
                    f"FuzzyPowersetMonad over {self._algebra.name} cannot "
                    f"compose a morphism over {arrow.algebra.name}"
                )
        return f >> g

    def __repr__(self) -> str:
        return f"FuzzyPowersetMonad({self._algebra!r})"


class KleisliCategory:
    """The Kleisli category of a monad.

    Objects are the same as the base category. Morphisms ``A → B`` in
    the Kleisli category are morphisms ``A → T(B)`` in the base
    category. Composition uses the underlying monad's
    `Monad.join`, falling through to a closed-form
    ``kleisli_compose`` method on the monad when available.

    Parameters
    ----------
    monad : Monad
        The underlying monad instance.
    """

    def __init__(self, monad: Monad) -> None:
        self._monad = monad

    @property
    def monad(self) -> Monad:
        return self._monad

    def identity(self, obj: SetObject) -> Morphism:
        """Kleisli identity: ``η_A : A → T(A)``."""
        return self._monad.pure(obj)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition.

        For ``f : A → T(B)`` and ``g : B → T(C)``, returns
        ``A → T(C)`` via the monad's bind / join construction.
        """
        # Defer to a closed-form kleisli_compose when the monad ships
        # one; otherwise the generic join-based construction requires
        # an internal-hom representation supplied per instance.
        kc = getattr(self._monad, "kleisli_compose", None)
        if callable(kc):
            return kc(f, g)
        raise NotImplementedError(
            f"KleisliCategory.compose: {type(self._monad).__name__} "
            "exposes no closed-form kleisli composition; supply one "
            "via the bridges in quivers.monadic.bridges"
        )


__all__ = [
    "FuzzyPowersetMonad",
    "KleisliCategory",
]
