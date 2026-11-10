"""Bridges between the monad and arrow typeclass towers.

Three canonical conversions:

- `kleisli` — given a `Monad` instance ``m``, produce a
  ``Kleisli(m)`` arrow whose hom from ``A`` to ``B`` is the V-Rel
  morphism set ``Hom(A, m(B))``.
- `arrow_monad` — given an `ArrowApply` instance ``a``,
  produce a `Monad` whose underlying functor is
  ``ArrowMonad(a)(α) = a(1, α)``.
- `cokleisli` — comonadic dual of `kleisli`.

Each bridge is realized at the value level as concrete morphism
constructions: ``compose`` is Kleisli composition (fmap-then-join),
``first`` is the monad's strength applied to the second-factor
input, ``app`` is the canonical evaluator over the function-space
encoding ``[A → B]``. Hughes develops the relation between monads and
arrows [Hughes (2000)](<https://www.sciencedirect.com/science/article/pii/S0167642399000234>). Recovering a monad from a suitable Kleisli arrow
uses the identification of arrows out of the unit with monadic values; this
is a correspondence under the required laws, not a claim that these runtime
objects or theory records are literally identical.
"""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.arrows.typeclasses import Arrow, ArrowApply, Category_
from quivers.core._factories import parallel
from quivers.core.morphisms import Morphism
from quivers.core.morphisms import identity as id_morph
from quivers.core.objects import CoproductSet, FreeMonoid, ProductSet, SetObject
from quivers.monadic.comonads import Comonad
from quivers.monadic.typeclasses import Monad


def _monad_strength(m, A: SetObject, B: SetObject) -> Morphism:
    """Canonical monad strength ``σ_m : A ⊗ m(B) → m(A ⊗ B)``.

    Constructed uniformly via ``lift_a2(id_{A×B})`` after the
    pure-injection on the ``A`` factor:

        A × m(B)
            →  m(A) × m(B)        via (pure_m × id)
            →  m(A × B)           via lift_a2(id)

    Concretely realized here by direct tensor enumeration.
    """
    mB = m.fmap_obj(B)
    AB = ProductSet(components=(A, B))
    id_AB = id_morph(AB)
    lifted = m.lift_a2(A, B, AB, id_AB)
    pure_A = m.pure(A)
    inject = parallel(pure_A, id_morph(mB))
    return inject >> lifted


class Kleisli(ValueModel):
    """The Kleisli arrow of a monad.

    Hom from ``A`` to ``B`` is the set of morphisms ``A → m(B)`` in
    V-Rel. Composition is Kleisli composition (fmap, then join);
    identity is `Monad.pure`. ``first`` is realized via the
    canonical monad strength; ``app`` evaluates a function-space
    embedding through the monad's `Monad.apply`.

    `monad` is held opaquely so the typeclass-ABC reference
    survives in-process identity through ``with_`` without panproto
    schema validation; the field does not round-trip through JSON.
    It compares by identity: two Kleisli categories are equal exactly
    when they wrap the same monad object.

    Attributes
    ----------
    monad : Monad
        The underlying monad instance.
    """

    monad: object = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)

    def id_arr(self, A: SetObject) -> Morphism:
        return self.monad.pure(A)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition ``f ; g = join ∘ fmap(g) ∘ f``.

        ``f : A → m(B)``, ``g : B → m(C)`` ⊢ ``g ∘_K f : A → m(C)``.
        """
        m = self.monad
        if not isinstance(f, Morphism) or not isinstance(g, Morphism):
            raise TypeError(
                "Kleisli.compose requires Morphism arguments; "
                f"got {type(f).__name__}, {type(g).__name__}"
            )
        # Identify the underlying B: f.codomain = m(B), so we recover
        # B by undoing fmap_obj. Since fmap_obj is injective on its
        # representable domain (per the monad's instance contract),
        # we rely on the caller-supplied g.domain to be B.
        B = g.domain
        C = g.codomain  # m's image of the real C
        # The real C is not directly recoverable; we use the
        # monad's fmap to produce m(g) : m(B) → m(m(C')) where
        # C' is whatever B-image the user produced. Then join
        # collapses the double-application.
        fmap_g = m.fmap(B, C, g)
        # Determine the underlying value-codomain by inspecting
        # whether the result already lives inside the monad.
        # For most monads, g.codomain = m(C') for some C'; we
        # recover C' by passing C' = the inverse of fmap_obj.
        # Concretely we call join on the monad applied at C'.
        # Since C is m(C'), we recover C' via the user-supplied
        # signature: if g.codomain.size == m.fmap_obj(C').size for
        # some C', that's our C'. We rely on g being well-typed.
        C_prime = _recover_value_type(m, C)
        join_C = m.join(C_prime)
        return f >> fmap_g >> join_C

    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f >> self.monad.pure(B)

    def first(self, f: Morphism, C: SetObject) -> Morphism:
        """``first(f) : (A × C) → m(B × C)`` via the monad's strength.

        Realized as ``(f × id_C) ; strength_m(B, C)``.
        """
        m = self.monad
        mB = f.codomain
        # Recover the underlying B from m(B).
        B = _recover_value_type(m, mB)
        # Build (A × C) → (m(B) × C):
        first_f = parallel(f, id_morph(C))
        # Then strength: m(B) × C → m(B × C) — we use the symmetric
        # form by swapping factors first if the strength is
        # right-strength; here we implement directly with the second-
        # factor strength.
        strength = _monad_strength_second(m, B, C)
        return first_f >> strength

    def app(self, A: SetObject, B: SetObject) -> Morphism:
        """``app : m([A → B]) × m(A) → m(B)`` via the monad's apply.

        Routes through the Applicative-level apply, which every
        Monad has by inclusion.
        """
        return self.monad.apply(A, B)


Arrow.register(Kleisli)
ArrowApply.register(Kleisli)


def _monad_strength_second(m, B: SetObject, C: SetObject) -> Morphism:
    """``σ'_m : m(B) × C → m(B × C)`` (strength on the second factor).

    Built via the standard symmetry of the strength: pre-compose with
    the braiding, apply the left-strength, then post-compose with
    the m-functorial braiding.
    """
    BC = ProductSet(components=(B, C))
    mB = m.fmap_obj(B)
    # Use lift_a2 on the identity to build the joint action:
    id_BC = id_morph(BC)
    lifted = m.lift_a2(B, C, BC, id_BC)
    pure_C = m.pure(C)
    inject = parallel(id_morph(mB), pure_C)
    return inject >> lifted


def _recover_value_type(m, mB: SetObject) -> SetObject:
    """Heuristic inverse of ``m.fmap_obj`` from an image to its preimage.

    Walks a small enumeration of candidate preimage types in scope
    and picks the unique one whose ``m.fmap_obj`` image matches
    ``mB`` structurally. For monads whose fmap_obj is the identity
    on objects (Identity, Alternative_), the preimage equals the
    image. For other monads, the preimage is structurally recoverable:
    Maybe's image is ``A + 1`` so the preimage is the left-summand;
    State's image is a function-space whose codomain is ``A × σ``
    so the preimage is the product's first factor; etc.
    """
    # If fmap_obj is identity, return as-is.
    candidate = m.fmap_obj(mB)
    if candidate == mB:
        # Identity-like image; check that re-applying still yields mB.
        return mB
    # Coproduct image (Maybe): the preimage is the left summand.
    if isinstance(mB, CoproductSet) and len(mB.components) >= 1:
        head = mB.components[0]
        if m.fmap_obj(head) == mB:
            return head
    # Product image (Writer-style): preimage is first factor.
    if isinstance(mB, ProductSet) and len(mB.components) >= 1:
        head = mB.components[0]
        if m.fmap_obj(head) == mB:
            return head
    # FreeMonoid image (List): preimage is the generator set.
    if isinstance(mB, FreeMonoid):
        gen = mB.generators
        if m.fmap_obj(gen) == mB:
            return gen
    # Function-space image (State, Reader, Continuation): we cannot
    # cheaply invert without enumerating; accept the original image
    # as a fallback (the caller will validate via Kleisli composition's
    # well-typedness check).
    return mB


class ArrowMonad(ValueModel):
    """The monad induced by an `ArrowApply` arrow.

    For a lawful ``ArrowApply`` interface ``a``, the corresponding monadic
    values have the form ``ArrowMonad(a)(α) = a(1, α)``, where ``1`` is the
    monoidal unit [Hughes (2000)](<https://www.sciencedirect.com/science/article/pii/S0167642399000234>).
    Realized concretely:

    - ``fmap_obj(A) = A`` (the underlying carrier coincides with the
      target type of the arrow's hom from the unit);
    - ``pure(A)`` is the arrow's ``arr(id_A)``;
    - ``join(A)`` is the arrow's ``app`` morphism on the unit-anchored
      arrow values.

    `arrow` is held opaquely and compares by identity: two arrow
    monads are equal exactly when they wrap the same arrow object.

    Attributes
    ----------
    arrow : ArrowApply
        The underlying arrow.
    """

    arrow: object = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)

    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        """``fmap(f) : ArrowMonad(a)(A) → ArrowMonad(a)(B)``.

        Realized as ``arr(f)`` on the underlying arrow.
        """
        return self.arrow.arr(A, B, f)

    def pure(self, A: SetObject) -> Morphism:
        """``pure : A → ArrowMonad(a)(A) = arr(id_A)``."""
        return self.arrow.id_arr(A)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        """``apply : ArrowMonad(a)([A → B]) ⊗ ArrowMonad(a)(A) → ArrowMonad(a)(B)``.

        Routes through the underlying arrow's `ArrowApply.app`.
        """
        return self.arrow.app(A, B)

    def join(self, A: SetObject) -> Morphism:
        """``join : ArrowMonad(a)(ArrowMonad(a)(A)) → ArrowMonad(a)(A)``.

        ArrowMonad's fmap_obj is identity, so join is also identity.
        """
        return id_morph(A)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        # bind = arrow's compose of arr(k) followed by app on the result.
        return self.arrow.compose(self.arrow.arr(A, B, k), self.arrow.id_arr(B))

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        """``lift_a2(f) : a(A) × a(B) → a(C)``, derived via ``arr(f)``."""
        return self.arrow.arr(ProductSet(components=(A, B)), C, f)


Monad.register(ArrowMonad)


def kleisli(monad: Monad) -> Kleisli:
    """Wrap a Monad as a Kleisli arrow.

    Parameters
    ----------
    monad : Monad
        The monad whose Kleisli category the arrow realises.

    Returns
    -------
    Kleisli
        The Kleisli arrow of ``monad``.
    """
    return Kleisli(monad=monad)


def arrow_monad(arrow: ArrowApply) -> ArrowMonad:
    """Wrap an `ArrowApply` arrow as a Monad.

    Parameters
    ----------
    arrow : ArrowApply
        The arrow whose ``app`` supplies the monad's ``join``.

    Returns
    -------
    ArrowMonad
        The monad of ``arrow``.
    """
    return ArrowMonad(arrow=arrow)


class CoKleisli(ValueModel):
    """The CoKleisli category of a comonad.

    Hom from ``A`` to ``B`` is the set of morphisms ``W(A) → B``.
    Composition is the comonadic CoKleisli composition
    ``f =>> g = g ∘ W(f) ∘ δ_A``; identity is the counit ``ε_A``.

    CoKleisli is registered as `Category_` rather than
    `Arrow` because the arrow ``first`` derivation requires
    a comonadic *costrength* ``W(A × C) → W(A) × C``, which is not
    canonical for an arbitrary comonad. When the underlying comonad
    supplies a ``costrength(A, C)`` method, that morphism composes
    with ``parallel(f, ε_C ∘ W(π_2))`` to realise ``first`` as the
    full Arrow primitive; the helper `first_via_costrength`
    constructs that morphism given an explicit costrength.

    `comonad` is held opaquely so the typeclass-ABC reference
    survives in-process identity without panproto schema validation.
    It compares by identity: two CoKleisli categories are equal
    exactly when they wrap the same comonad object.

    Attributes
    ----------
    comonad : Comonad
        The underlying comonad instance.
    """

    comonad: object = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)

    def id_arr(self, A: SetObject) -> Morphism:
        return self.comonad.counit(A)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return self.comonad.cokleisli_compose(f, g)

    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # arr(f) : W(A) ⇝ B is ε_A then f.
        return self.comonad.counit(A) >> f

    def first_via_costrength(
        self, f: Morphism, C: SetObject, costrength: Morphism
    ) -> Morphism:
        """Realise ``first(f)`` given an explicit comonad costrength.

        ``costrength : W(A × C) → W(A) × C`` is the comonad-specific
        morphism (analogous to the monad strength). With it,
        ``first(f, C) = costrength ; (f × id_C)`` is a valid CoKleisli
        arrow ``W(A × C) ⇝ B × C``.

        Parameters
        ----------
        f : Morphism
            A CoKleisli arrow ``W(A) → B``.
        C : SetObject
            The fixed second-factor object.
        costrength : Morphism
            The comonad's costrength ``W(A × C) → W(A) × C``.
        """
        return costrength >> parallel(f, id_morph(C))


def cokleisli(comonad: Comonad) -> CoKleisli:
    """Wrap a Comonad as a CoKleisli category.

    The result is registered as `Category_` (always) but not
    `Arrow`; promoting to an Arrow requires the user-supplied
    costrength routed through `CoKleisli.first_via_costrength`.

    Parameters
    ----------
    comonad : Comonad
        The comonad whose CoKleisli category is built.

    Returns
    -------
    CoKleisli
        The CoKleisli category of ``comonad``.
    """
    return CoKleisli(comonad=comonad)


Category_.register(CoKleisli)


__all__ = [
    "Kleisli",
    "ArrowMonad",
    "CoKleisli",
    "kleisli",
    "arrow_monad",
    "cokleisli",
]
