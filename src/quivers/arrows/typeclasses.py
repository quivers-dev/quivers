"""Hughes-style arrow typeclass interfaces.

Each abstract typeclass declares the operations an arrow instance
must provide. Concrete instances (``Function``, ``VRel``,
``Stochastic``, ``Kleisli(M)``, ``Cokleisli(W)``, ``LinearMap``) live
in `quivers.arrows.instances`.

References
----------
- Hughes, J. (2000). *Generalising Monads to Arrows*. Science of
  Computer Programming, 37(1–3), 67–111.
  [doi:10.1016/S0167-6423(99)00023-4](https://doi.org/10.1016/S0167-6423(99)00023-4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quivers.core.morphisms import Morphism
from quivers.core.objects import SetObject


class Category_(ABC):
    """A category of computations.

    Provides identity at every object plus binary composition. The
    underscore disambiguates from [`quivers.dsl.ast_nodes.CategoryDecl`][quivers.dsl.ast_nodes.CategoryDecl]
    and [`quivers.stochastic.categories.Category`][quivers.stochastic.categories.Category] which use the same
    word in different senses.

    Laws:

    - ``compose(id, f) = f``                      (left identity)
    - ``compose(f, id) = f``                      (right identity)
    - ``compose(compose(f, g), h) = compose(f, compose(g, h))``  (associativity)
    """

    @abstractmethod
    def id_arr(self, A: SetObject) -> Morphism:
        """``id_A : A ⇝ A``."""

    @abstractmethod
    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """``f : A ⇝ B, g : B ⇝ C  ⊢  g ∘ f : A ⇝ C``."""


class Arrow(Category_, ABC):
    """An arrow: ``Category_`` with ``arr`` and ``first``.

    Adds:

    - `arr` — lift a pure morphism into the arrow.
    - `first` — apply on the left of a pair, threading a context.

    Defaults derivable from these and `compose`:
    ``second f = swap >>> first f >>> swap``;
    ``f *** g = first f >>> second g`` (the parallel-product combinator);
    ``f &&& g = arr (\\x. (x, x)) >>> (f *** g)``.

    Hughes 2000 §3 lists the seven arrow laws.
    """

    @abstractmethod
    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        """Lift a pure morphism: ``arr(f) : A ⇝ B`` from ``f : A → B``."""

    @abstractmethod
    def first(self, f: Morphism, C: SetObject) -> Morphism:
        """``f : A ⇝ B  ⊢  first(f, C) : (A ⊗ C) ⇝ (B ⊗ C)``."""


class ArrowChoice(Arrow, ABC):
    """An arrow with sum-elimination: ``left`` and (derivably) right.

    Adds:

    - `left_arr` — apply on the left of a coproduct.

    Derived: ``right_arr``, ``+++``, ``|||``.
    """

    @abstractmethod
    def left_arr(self, f: Morphism, C: SetObject) -> Morphism:
        """``f : A ⇝ B  ⊢  left(f, C) : (A + C) ⇝ (B + C)``."""


class ArrowApply(Arrow, ABC):
    """An arrow that can apply arrow-valued data: ``app``.

    ``ArrowApply`` instances are equivalent in expressive power to
    [`quivers.monadic.typeclasses.Monad`][quivers.monadic.typeclasses.Monad]; the bridges in
    [`quivers.monadic.bridges`][quivers.monadic.bridges] realise the conversion in both
    directions.
    """

    @abstractmethod
    def app(self, A: SetObject, B: SetObject) -> Morphism:
        """``app : Hom(A, B) ⊗ A ⇝ B``.

        ``Hom(A, B)`` is the internal hom of the arrow's underlying
        symmetric monoidal closed structure.
        """


class ArrowLoop(Arrow, ABC):
    """An arrow with feedback: ``loop``.

    Encodes traced symmetric monoidal structure on the arrow category.

    For an arrow ``f : (A ⊗ C) ⇝ (B ⊗ C)``, ``loop(f) : A ⇝ B``
    closes the loop on ``C``. `VRel` implements finite diagonal contraction,
    the compact-closed trace for weighted relations over a commutative
    semiring. Total finite functions and normalized stochastic kernels do
    not have this general trace, so `Function` and `Stochastic` are not
    registered here. A chart's closure algorithm is a separate operation;
    registering this interface does not establish a recursive chart solver.
    """

    @abstractmethod
    def loop_arr(self, f: Morphism, C: SetObject) -> Morphism:
        """``f : (A ⊗ C) ⇝ (B ⊗ C)  ⊢  loop(f) : A ⇝ B``."""


class ArrowZero(Arrow, ABC):
    """An arrow with a designated bottom at every hom: ``zero_arr``."""

    @abstractmethod
    def zero_arr(self, A: SetObject, B: SetObject) -> Morphism:
        """``zero : A ⇝ B`` — the bottom element of ``Hom(A, B)``."""


class ArrowPlus(ArrowZero, ABC):
    """An ArrowZero with pointwise alternative: ``alt_arr``.

    ``alt_arr`` plays the role of ``Alternative.alt`` lifted to the
    arrow category's hom-sets.
    """

    @abstractmethod
    def alt_arr(self, A: SetObject, B: SetObject) -> Morphism:
        """``Hom(A, B) ⊗ Hom(A, B) ⇝ Hom(A, B)``."""


__all__ = [
    "Category_",
    "Arrow",
    "ArrowChoice",
    "ArrowApply",
    "ArrowLoop",
    "ArrowZero",
    "ArrowPlus",
]
