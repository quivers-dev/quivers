"""Stdlib effect instances for the typeclass hierarchy.

Each effect is a `dx.Model` carrying any effect parameters
(e.g. ``Continuation(answer)``) plus a concrete V-relation realisation
of every typeclass operation it satisfies. Effects compose with
arbitrary user-defined effects through the class-driven lifting
machinery in [`quivers.stochastic.effect_lifts`][quivers.stochastic.effect_lifts].

The constructions live in `quivers.core._factories` (coproduct
injections / case eliminators, product projections / pairings,
parallel pairs, distributivity, terminal maps). Each operation is
built as a concrete `ObservedMorphism` over the appropriate
SetObject; composition uses the underlying algebra through the
``>>`` operator, and the factory helpers use the same composition interface.

Continuation, State, Reader, and Writer carry an extra typed
parameter (the answer/state/environment/monoid). Their operations
are realised through the standard V-Rel encodings: products for
State/Writer, the function-space ``B^A`` (encoded as a finite
`FinSet`) for Reader/Continuation. Function-space cardinality
grows as ``|B|^|A|``, so large-cardinality instances should be
applied at small carriers (typical in compositional-semantics
applications).
"""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers.core._factories import (
    constant,
    coproduct_map,
    inj,
    pair,
    parallel,
)
from quivers.core.morphisms import Morphism, observed
from quivers.core.morphisms import identity as id_morph
from quivers.core.objects import (
    CoproductSet,
    FinSet,
    ProductSet,
    SetObject,
    Unit,
)
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY, Algebra
from quivers.monadic.typeclasses import (
    Foldable,
    Monad,
    MonadPlus,
    Traversable,
)


# ---------------------------------------------------------------------------
# Helpers for function-space encodings
# ---------------------------------------------------------------------------


def _function_space(domain: SetObject, codomain: SetObject) -> FinSet:
    """Encode ``[A → B]`` as a `FinSet` of cardinality ``|B|^|A|``.

    Each element is a *total function* ``A → B``, indexed by the
    flat enumeration of all such functions in row-major order over
    inputs. The bijection ``flat ↔ tuple-of-outputs`` is computed
    by `_decode_function`.
    """
    card = codomain.size**domain.size
    return FinSet(
        name=f"[{domain!s}→{codomain!s}]",
        cardinality=card,
    )


def _decode_function(
    flat_index: int, domain_size: int, codomain_size: int
) -> tuple[int, ...]:
    """Decode a flat function-space index to its output tuple."""
    outputs = [0] * domain_size
    rem = flat_index
    for i in range(domain_size - 1, -1, -1):
        outputs[i] = rem % codomain_size
        rem //= codomain_size
    return tuple(outputs)


def _function_table(
    f: Morphism, domain_size: int, codomain_size: int, partial: bool = False
) -> list[int]:
    """Read a morphism as a function between flat index sets.

    The function-space instances are monads on finite sets and
    functions, so their ``fmap`` and ``lift_a2`` accept a morphism only
    when it is a function: every row has exactly one entry equal to the
    algebra's unit and every other entry equal to its zero.

    Parameters
    ----------
    f : Morphism
        The morphism, with ``domain_size`` rows and ``codomain_size``
        columns once flattened.
    domain_size : int
        The size of the flattened domain.
    codomain_size : int
        The size of the flattened codomain.
    partial : bool
        Accept a partial function: a row that is all zero is undefined
        and reported as ``-1``.

    Returns
    -------
    list[int]
        ``table[i]`` is the flat index of the image of ``i``, or ``-1``
        where a partial function is undefined.

    Raises
    ------
    ValueError
        If some row of ``f`` is not the indicator of a single element
        (or, with ``partial``, all zero).
    """
    algebra = f.algebra
    unit = float(algebra.unit) if isinstance(algebra, Algebra) else 1.0
    zero = float(algebra.zero) if isinstance(algebra, Algebra) else 0.0
    rows = f.tensor.detach().reshape(domain_size, codomain_size)
    table: list[int] = []
    for i in range(domain_size):
        row = rows[i]
        hits = [j for j in range(codomain_size) if abs(float(row[j]) - unit) <= 1e-6]
        rest_zero = all(
            abs(float(row[j]) - zero) <= 1e-6 or j in hits for j in range(codomain_size)
        )
        if partial and not hits and rest_zero:
            table.append(-1)
            continue
        if len(hits) != 1 or not rest_zero:
            raise ValueError(
                "the function-space monad instances act on functions; row "
                f"{i} of the morphism is not the indicator of one element"
            )
        table.append(hits[0])
    return table


def _evaluation_morphism(
    domain: SetObject, codomain: SetObject, algebra: Algebra | None = None
) -> Morphism:
    """The evaluation morphism ``ev : [A → B] × A → B``.

    Sends ``(f, a)`` to ``f(a)``. Realised as an `ObservedMorphism`
    whose tensor entry at ``(f_flat, a_flat, b_flat)`` is unit iff
    ``decode(f_flat)[a_flat] == b_flat``.
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    fn_space = _function_space(domain, codomain)
    source = ProductSet(components=(fn_space, domain))
    a_size = domain.size
    b_size = codomain.size
    data = torch.full((fn_space.cardinality, a_size, b_size), q.zero)
    for f_flat in range(fn_space.cardinality):
        outputs = _decode_function(f_flat, a_size, b_size)
        for a_flat in range(a_size):
            data[f_flat, a_flat, outputs[a_flat]] = q.unit
    # Reshape from (|[A→B]|, |A|, |B|) to (*source.shape, *B.shape).
    data = data.reshape(*source.shape, *codomain.shape)
    return observed(source, codomain, data, algebra=q)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Identity(dx.Model):
    """The trivial monad: ``Id(A) = A``.

    All operations reduce to the identity in V-Rel; the laws hold
    trivially.
    """

    name: str = "Identity"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def pure(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        # apply : [A → B] × A → B is the evaluation morphism in V-Rel,
        # representing the function-space [A → B] as a FinSet of size
        # |B|^|A|. Identity's apply is therefore exactly the standard
        # finite-function evaluation.
        return _evaluation_morphism(A, B)

    def join(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        # bind(m, k) = join(fmap(k)(m)); for Identity, k itself.
        return k

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # lift_a2 for Identity is just f itself, since Id(X) = X.
        return f


Monad.register(Identity)


# ---------------------------------------------------------------------------
# Maybe (presupposition failure / partiality)
# ---------------------------------------------------------------------------


def _maybe_carrier(A: SetObject) -> CoproductSet:
    """``Maybe(A) = A + 1`` with the failure marker on the right."""
    nothing = FinSet(name=f"_nothing_{A!s}", cardinality=1)
    return CoproductSet(components=(A, nothing))


class Maybe(dx.Model):
    """The partiality monad: ``Maybe(A) = A + 1``.

    A ``MonadPlus`` instance: success injects into the left summand,
    failure into the right.
    """

    name: str = "Maybe"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return _maybe_carrier(A)

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # fmap_Maybe(f) : A + 1 → B + 1 acts as f on the left summand
        # and as the identity on the failure marker.
        nothing_A = FinSet(name=f"_nothing_{A!s}", cardinality=1)
        nothing_B = FinSet(name=f"_nothing_{B!s}", cardinality=1)
        # The failure-side branch is the unique map between the two
        # singletons (identity).
        failure_branch = (
            id_morph(nothing_A)
            if nothing_A.name == nothing_B.name
            else (constant(nothing_A, nothing_B, 0))
        )
        return coproduct_map((f, failure_branch))

    def pure(self, A: SetObject) -> Morphism:
        # pure : A → Maybe(A) is left-injection.
        nothing = FinSet(name=f"_nothing_{A!s}", cardinality=1)
        return inj((A, nothing), 0)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        # Maybe(A → B) ⊗ Maybe(A) → Maybe(B): success iff both sides
        # are success, then evaluate. Realised by case-analysis on
        # the input coproduct.
        fn_space = _function_space(A, B)
        # mf : [A→B] + 1, ma : A + 1. The combined input lives in
        # (Maybe([A→B])) ⊗ (Maybe(A)) = ([A→B]+1) ⊗ (A+1).
        ma_carrier = self.fmap_obj(A)
        mb_carrier = self.fmap_obj(B)
        mf_carrier = self.fmap_obj(fn_space)
        source = ProductSet(components=(mf_carrier, ma_carrier))
        target = mb_carrier
        # Build the result tensor explicitly: for each pair
        # (mf_flat, ma_flat), the success case is when both flat
        # indices fall in the left summand; the failure marker is
        # the last index in the codomain.
        nothing_B_flat = B.size  # failure marker = right-summand
        data = torch.full((source.size, target.size), PRODUCT_FUZZY.zero)
        # Iterate over the joint input flat space (rank-2 since both
        # carriers are CoproductSets of shape (size,)).
        for mf_flat in range(mf_carrier.size):
            mf_is_success = mf_flat < fn_space.cardinality
            for ma_flat in range(ma_carrier.size):
                ma_is_success = ma_flat < A.size
                joint = mf_flat * ma_carrier.size + ma_flat
                if mf_is_success and ma_is_success:
                    outputs = _decode_function(mf_flat, A.size, B.size)
                    data[joint, outputs[ma_flat]] = PRODUCT_FUZZY.unit
                else:
                    data[joint, nothing_B_flat] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *target.shape)
        return observed(source, target, data)

    def join(self, A: SetObject) -> Morphism:
        # join : Maybe(Maybe(A)) → Maybe(A) flattens the two failure
        # markers into one. Maybe(Maybe(A)) = (A + 1) + 1 ≅ A + 1 + 1
        # under flattening; we identify both inner-failure encodings
        # with the outer failure marker of the result.
        inner = self.fmap_obj(A)  # A + 1
        outer = self.fmap_obj(inner)  # (A + 1) + 1
        target = self.fmap_obj(A)  # A + 1
        nothing_target_flat = A.size  # failure marker in target
        data = torch.full((outer.size, target.size), PRODUCT_FUZZY.zero)
        # Outer is a CoproductSet of (inner, nothing).
        # flat indices [0, inner.size) belong to inner = A + 1.
        # within inner, [0, A.size) is success, A.size is failure.
        for outer_flat in range(outer.size):
            if outer_flat < inner.size:
                # in the inner summand: pass through with the inner's
                # success/failure structure.
                if outer_flat < A.size:
                    data[outer_flat, outer_flat] = PRODUCT_FUZZY.unit
                else:
                    data[outer_flat, nothing_target_flat] = PRODUCT_FUZZY.unit
            else:
                # outer failure: pass to target failure
                data[outer_flat, nothing_target_flat] = PRODUCT_FUZZY.unit
        return observed(outer, target, data)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        # bind(m, k) = join_B ∘ fmap(k)(m) for k : A → Maybe(B).
        return self.fmap(A, self.fmap_obj(B), k) >> self.join(B)

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # lift_a2(f) : Maybe(A) ⊗ Maybe(B) → Maybe(C) where f : A × B → C.
        # Success on both sides applies f and injects into Maybe(C);
        # otherwise produces failure.
        mA = self.fmap_obj(A)
        mB = self.fmap_obj(B)
        mC = self.fmap_obj(C)
        source = ProductSet(components=(mA, mB))
        nothing_C_flat = C.size
        # Decode the f tensor for direct value lookup.
        # f.tensor has shape (*A.shape, *B.shape, *C.shape); we
        # collapse to (A.size, B.size, C.size).
        f_flat = f.tensor.reshape(A.size, B.size, C.size)
        data = torch.full((mA.size, mB.size, mC.size), PRODUCT_FUZZY.zero)
        for ma_flat in range(mA.size):
            for mb_flat in range(mB.size):
                if ma_flat < A.size and mb_flat < B.size:
                    # Both success: distribute f's relation into the
                    # left summand of mC.
                    data[ma_flat, mb_flat, : C.size] = f_flat[ma_flat, mb_flat]
                else:
                    data[ma_flat, mb_flat, nothing_C_flat] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *mC.shape)
        return observed(source, mC, data)

    def empty(self, A: SetObject) -> Morphism:
        # empty : 1 → Maybe(A) injects into the failure summand.
        nothing = FinSet(name=f"_nothing_{A!s}", cardinality=1)
        right = inj((A, nothing), 1)
        # right has domain `nothing`; pre-compose with the unique 1 → 1.
        return id_morph(Unit) >> reshape_unit_to_nothing(nothing) >> right

    def alt(self, A: SetObject) -> Morphism:
        # alt : Maybe(A) ⊗ Maybe(A) → Maybe(A) prefers the first
        # success; falls back to the second if the first is failure.
        mA = self.fmap_obj(A)
        source = ProductSet(components=(mA, mA))
        data = torch.full((mA.size, mA.size, mA.size), PRODUCT_FUZZY.zero)
        for x_flat in range(mA.size):
            for y_flat in range(mA.size):
                if x_flat < A.size:
                    # x is success: choose x
                    data[x_flat, y_flat, x_flat] = PRODUCT_FUZZY.unit
                else:
                    # x is failure: choose y
                    data[x_flat, y_flat, y_flat] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *mA.shape)
        return observed(source, mA, data)


MonadPlus.register(Maybe)


def reshape_unit_to_nothing(nothing: FinSet) -> Morphism:
    """Trivial bijection ``1 → 1`` viewed as ``Unit → nothing``."""
    data = torch.tensor([[PRODUCT_FUZZY.unit]])
    return observed(Unit, nothing, data)


# ---------------------------------------------------------------------------
# Alternative_ (Hamblin / focus alternatives)
# ---------------------------------------------------------------------------


class Alternative_(dx.Model):
    """The Hamblin alternative monad, the powerset monad on sets.

    A set of alternatives over ``A`` is a subset of ``A``, and the
    Kleisli category of the powerset monad is Rel, the category of
    relations [Hofmann, Seal and Tholen (2014)]. The instance works in
    that Kleisli category over `BOOLEAN`: the type-level action over
    ``A`` is again ``A``, a relation ``A → B`` stands for the Kleisli
    arrow ``A → P(B)``, ``pure`` and ``join`` are identities, ``empty``
    is the empty relation and ``alt`` is union. Its morphisms are over
    `BOOLEAN`, under which composition is associative; a fuzzy variant
    over an idempotent semiring is `quivers.monadic.monads.FuzzyPowersetMonad`.

    References
    ----------
    Hofmann, D., Seal, G. J., and Tholen, W. (Eds.) (2014). *Monoidal
    Topology: A Categorical Approach to Order, Metric, and Topology*.
    Cambridge University Press.
    [doi:10.1017/CBO9781107517288](https://doi.org/10.1017/CBO9781107517288)
    """

    name: str = "Alternative"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f  # functor action is the morphism itself

    def pure(self, A: SetObject) -> Morphism:
        return id_morph(A, algebra=BOOLEAN)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        # apply : Alt([A → B]) ⊗ Alt(A) → Alt(B). At the V-Rel level,
        # this is the evaluation morphism with the V-algebra's natural
        # aggregation over alternatives (already baked into compose).
        return _evaluation_morphism(A, B, algebra=BOOLEAN)

    def join(self, A: SetObject) -> Morphism:
        return id_morph(A, algebra=BOOLEAN)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        return k

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # For Alt, lift_a2(f) is just f itself viewed as a morphism
        # A ⊗ B → C; aggregation across alternatives happens in the
        # algebra's compose.
        return f

    def empty(self, A: SetObject) -> Morphism:
        # empty : 1 → A is the constant-bottom V-relation: a single
        # row of zero entries.
        data = torch.full((1, *A.shape), BOOLEAN.zero)
        return observed(Unit, A, data, algebra=BOOLEAN)

    def alt(self, A: SetObject) -> Morphism:
        # alt : A ⊗ A → A is union: (a, a') relates to both a and a'.
        source = ProductSet(components=(A, A))
        data = torch.full((A.size, A.size, A.size), BOOLEAN.zero)
        for a in range(A.size):
            for ap in range(A.size):
                data[a, ap, a] = BOOLEAN.unit
                data[a, ap, ap] = BOOLEAN.unit
        data = data.reshape(*source.shape, *A.shape)
        return observed(source, A, data, algebra=BOOLEAN)

    def foldr(self, A: SetObject, B: SetObject) -> Morphism:
        # foldr : (A × B → B) × B × F(A) → B; for Alt where F(A)=A,
        # this is just the user-supplied step composed with the seed.
        # Since the binary step is a parameter, return the curried
        # form: source = (A → B) × B × A. We encode the (A × B → B)
        # argument via the function-space [A × B → B].
        AB = ProductSet(components=(A, B))
        step_fn = _function_space(AB, B)
        source = ProductSet(components=(step_fn, B, A))
        target = B
        # data[step_fn_flat, b_flat, a_flat, b'_flat] = unit iff
        # decode(step_fn_flat)[(a, b)] == b'.
        ab_size = A.size * B.size
        data = torch.full((step_fn.cardinality, B.size, A.size, B.size), BOOLEAN.zero)
        for fn_flat in range(step_fn.cardinality):
            outputs = _decode_function(fn_flat, ab_size, B.size)
            for b_flat in range(B.size):
                for a_flat in range(A.size):
                    ab_flat = a_flat * B.size + b_flat
                    data[fn_flat, b_flat, a_flat, outputs[ab_flat]] = BOOLEAN.unit
        data = data.reshape(*source.shape, *target.shape)
        return observed(source, target, data, algebra=BOOLEAN)

    def traverse(
        self,
        A: SetObject,
        B: SetObject,
        applicative,
        f: Morphism,
    ) -> Morphism:
        # traverse : F(A) → G(F(B)) given f : A → G(B). For Alt where
        # F(A) = A, this collapses to G's lift of f composed with G's
        # pure. Realised as f followed by Alt-singleton injection
        # within the applicative G.
        gB = applicative.fmap_obj(B)
        # We need a morphism A → G(B) followed by the unit of Alt at G(B).
        return f >> applicative.fmap(B, gB, applicative.pure(B))


MonadPlus.register(Alternative_)
Foldable.register(Alternative_)
Traversable.register(Alternative_)


# ---------------------------------------------------------------------------
# Continuation: Cont_ρ(A) = (A → ρ) → ρ
# ---------------------------------------------------------------------------


class Continuation(dx.Model):
    """The continuation monad with a typed answer.

    The carrier is the function-space encoding
    ``Cont_ρ(A) = [A → ρ] → ρ`` realised as a finite SetObject of
    cardinality ``|ρ|^(|ρ|^|A|)``. For small ``A`` and ``ρ`` this is
    tractable; for larger carriers, use the algebraic-effect handler
    of [`quivers.monadic.algebraic`][quivers.monadic.algebraic] with the
    `ContinuationSignature` (see ``ContSignature`` in this
    module).
    """

    answer: SetObject
    name: str = "Continuation"

    def _kr_object(self, A: SetObject) -> FinSet:
        """The continuation type ``K_ρ(A) = [A → ρ]`` as a FinSet."""
        return _function_space(A, self.answer)

    def fmap_obj(self, A: SetObject) -> SetObject:
        # Cont_ρ(A) = [K_ρ(A) → ρ] = [[A → ρ] → ρ]
        return _function_space(self._kr_object(A), self.answer)

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # fmap(f) : Cont_ρ(A) → Cont_ρ(B). The standard derivation:
        # fmap(f)(m) = λk_B. m(λa. k_B(f(a))).
        # Equivalently: m : [A→ρ]→ρ; pre-compose A→ρ with f to get
        # B→ρ → A→ρ; this lifts contravariantly to Cont_ρ(B) → Cont_ρ(A)
        # → wait, that's backwards. Standard Functor for Cont:
        #     fmap(f)(m) = λk. m(k ∘ f).
        # So fmap : Cont(A) → Cont(B) given f : A → B.
        # Realised as a V-relation on the function-space encodings.
        cont_A = self.fmap_obj(A)
        cont_B = self.fmap_obj(B)
        ka = self._kr_object(A)
        kb = self._kr_object(B)
        rho = self.answer
        # Pre-compute the "pre-composition" map kb → ka:
        # given kb : B → ρ and f : A → B, produce ka : A → ρ as
        # ka(a) = kb(f(a)).
        f_outputs = _function_table(f, A.size, B.size)
        precompose: dict[int, int] = {}
        for kb_flat in range(kb.cardinality):
            kb_outputs = _decode_function(kb_flat, B.size, rho.size)
            ka_outputs = tuple(kb_outputs[f_outputs[a]] for a in range(A.size))
            # Encode ka_outputs back as a flat function-space index.
            ka_flat = 0
            for v in ka_outputs:
                ka_flat = ka_flat * rho.size + v
            precompose[kb_flat] = ka_flat
        # Now build fmap(f) as a V-relation: for each m ∈ Cont(A) and
        # each kb ∈ K_ρ(B), the output is m(precompose(kb)).
        data = torch.full((cont_A.size, cont_B.size), PRODUCT_FUZZY.zero)
        for m_flat in range(cont_A.size):
            m_outputs = _decode_function(m_flat, ka.cardinality, rho.size)
            # The image m'(kb) = m(precompose[kb]) for each kb.
            new_outputs: list[int] = []
            for kb_flat in range(kb.cardinality):
                ka_flat = precompose[kb_flat]
                new_outputs.append(m_outputs[ka_flat])
            new_flat = 0
            for v in new_outputs:
                new_flat = new_flat * rho.size + v
            data[m_flat, new_flat] = PRODUCT_FUZZY.unit
        return observed(cont_A, cont_B, data)

    def pure(self, A: SetObject) -> Morphism:
        # pure_A : A → Cont_ρ(A), pure(a) = λk. k(a).
        cont_A = self.fmap_obj(A)
        ka = self._kr_object(A)
        rho = self.answer
        data = torch.full((A.size, cont_A.size), PRODUCT_FUZZY.zero)
        for a in range(A.size):
            # The output is the function-space element f_a : ka → ρ
            # given by f_a(k) = k(a) for k : A → ρ.
            # Encode f_a as flat index: outputs[ka_flat] = decode(ka_flat)[a].
            outputs: list[int] = []
            for ka_flat in range(ka.cardinality):
                k_outs = _decode_function(ka_flat, A.size, rho.size)
                outputs.append(k_outs[a])
            f_a_flat = 0
            for v in outputs:
                f_a_flat = f_a_flat * rho.size + v
            data[a, f_a_flat] = PRODUCT_FUZZY.unit
        return observed(A, cont_A, data)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        # Derive apply from lift_a2 and the function-space evaluation.
        return self.lift_a2(_function_space(A, B), A, B, _evaluation_morphism(A, B))

    def join(self, A: SetObject) -> Morphism:
        # join_A : Cont_ρ(Cont_ρ(A)) → Cont_ρ(A).
        # join(M)(k) = M(λm. m(k)) where M : Cont(Cont(A)) and k : A→ρ.
        cont_A = self.fmap_obj(A)
        cont_cont_A = self.fmap_obj(cont_A)
        ka = self._kr_object(A)
        k_contA = self._kr_object(cont_A)  # [Cont(A) → ρ]
        rho = self.answer
        # For each k : A → ρ, produce k* : Cont(A) → ρ given by
        # k*(m) = m(k). Encode this as a function from ka to k_contA.
        k_to_kstar: dict[int, int] = {}
        for k_flat in range(ka.cardinality):
            # build k*: k_contA index
            kstar_outputs: list[int] = []
            for m_flat in range(cont_A.size):
                m_outputs = _decode_function(m_flat, ka.cardinality, rho.size)
                kstar_outputs.append(m_outputs[k_flat])
            kstar_flat = 0
            for v in kstar_outputs:
                kstar_flat = kstar_flat * rho.size + v
            k_to_kstar[k_flat] = kstar_flat
        data = torch.full((cont_cont_A.size, cont_A.size), PRODUCT_FUZZY.zero)
        for M_flat in range(cont_cont_A.size):
            M_outputs = _decode_function(M_flat, k_contA.cardinality, rho.size)
            # join(M)(k) = M(k*) = M_outputs[k_to_kstar[k]]
            new_outputs: list[int] = []
            for k_flat in range(ka.cardinality):
                new_outputs.append(M_outputs[k_to_kstar[k_flat]])
            new_flat = 0
            for v in new_outputs:
                new_flat = new_flat * rho.size + v
            data[M_flat, new_flat] = PRODUCT_FUZZY.unit
        return observed(cont_cont_A, cont_A, data)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        # bind = join ∘ fmap(k)
        return self.fmap(A, self.fmap_obj(B), k) >> self.join(B)

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # liftA2(f) : Cont(A) ⊗ Cont(B) → Cont(C), realised by
        # liftA2(f)(m, n) = λk. m(λa. n(λb. k(f(a,b)))).
        # We implement directly on flat function-space tensors.
        cont_A = self.fmap_obj(A)
        cont_B = self.fmap_obj(B)
        cont_C = self.fmap_obj(C)
        kC = self._kr_object(C)
        kB = self._kr_object(B)
        kA = self._kr_object(A)
        rho = self.answer
        # Pre-compute the function f as a deterministic map A × B → C.
        f_table = _function_table(f, A.size * B.size, C.size)
        f_lookup: dict[tuple[int, int], int] = {
            (a, b): f_table[a * B.size + b]
            for a in range(A.size)
            for b in range(B.size)
        }
        # For each k : C → ρ and each a, build the kb_a : B → ρ defined
        # by kb_a(b) = k(f(a, b)). Then for each m, n, compute
        # n(kb_a) for each a, giving ka : A → ρ; then m(ka) = result.
        source = ProductSet(components=(cont_A, cont_B))
        data = torch.full((cont_A.size, cont_B.size, cont_C.size), PRODUCT_FUZZY.zero)
        for m_flat in range(cont_A.size):
            m_outs = _decode_function(m_flat, kA.cardinality, rho.size)
            for n_flat in range(cont_B.size):
                n_outs = _decode_function(n_flat, kB.cardinality, rho.size)
                # Result of liftA2(f)(m, n) is a function K_C → ρ; we
                # enumerate over k ∈ K_C.
                result_outputs: list[int] = []
                for k_flat in range(kC.cardinality):
                    k_outs = _decode_function(k_flat, C.size, rho.size)
                    # Build ka : A → ρ
                    ka_outs: list[int] = []
                    for a in range(A.size):
                        # Build kb_a : B → ρ as encoded flat
                        kb_a_outs = tuple(
                            k_outs[f_lookup[(a, b)]] for b in range(B.size)
                        )
                        kb_a_flat = 0
                        for v in kb_a_outs:
                            kb_a_flat = kb_a_flat * rho.size + v
                        ka_outs.append(n_outs[kb_a_flat])
                    ka_flat = 0
                    for v in ka_outs:
                        ka_flat = ka_flat * rho.size + v
                    result_outputs.append(m_outs[ka_flat])
                result_flat = 0
                for v in result_outputs:
                    result_flat = result_flat * rho.size + v
                data[m_flat, n_flat, result_flat] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *cont_C.shape)
        return observed(source, cont_C, data)


Monad.register(Continuation)


# ---------------------------------------------------------------------------
# State: State_σ(A) = σ → (A × σ)
# ---------------------------------------------------------------------------


def _state_carrier(state: SetObject, A: SetObject) -> FinSet:
    """``State_σ(A) = [σ → A × σ]`` as a finite function-space."""
    target = ProductSet(components=(A, state))
    return _function_space(state, target)


class State(dx.Model):
    """The state monad.

    Carrier is the function-space encoding ``σ → A × σ``. Each element
    of the carrier corresponds to a deterministic state-transition
    function pairing a result with an updated state.
    """

    state: SetObject
    name: str = "State"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return _state_carrier(self.state, A)

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # fmap(f)(s_fn)(σ) = let (a, σ') = s_fn(σ) in (f(a), σ')
        sA = self.fmap_obj(A)
        sB = self.fmap_obj(B)
        sigma = self.state
        target_A = ProductSet(components=(A, sigma))
        target_B = ProductSet(components=(B, sigma))
        # Treat f as a deterministic A → B function via the tensor.
        f_lookup = _function_table(f, A.size, B.size)
        data = torch.full((sA.cardinality, sB.cardinality), PRODUCT_FUZZY.zero)
        for s_fn_flat in range(sA.cardinality):
            s_fn_outs = _decode_function(s_fn_flat, sigma.size, target_A.size)
            # Build the new state function by applying f to the A coord.
            new_outs: list[int] = []
            for sig in range(sigma.size):
                a_sig_flat = s_fn_outs[sig]
                # decode (a, σ') from a_sig_flat
                a = a_sig_flat // sigma.size
                sig_prime = a_sig_flat % sigma.size
                b = f_lookup[a]
                new_outs.append(b * sigma.size + sig_prime)
            new_flat = 0
            for v in new_outs:
                new_flat = new_flat * target_B.size + v
            data[s_fn_flat, new_flat] = PRODUCT_FUZZY.unit
        return observed(sA, sB, data)

    def pure(self, A: SetObject) -> Morphism:
        # pure(a) = λσ. (a, σ)
        sA = self.fmap_obj(A)
        sigma = self.state
        target_A = ProductSet(components=(A, sigma))
        data = torch.full((A.size, sA.cardinality), PRODUCT_FUZZY.zero)
        for a in range(A.size):
            # The state-fn: σ ↦ (a, σ)
            outs = tuple(a * sigma.size + sig for sig in range(sigma.size))
            fn_flat = 0
            for v in outs:
                fn_flat = fn_flat * target_A.size + v
            data[a, fn_flat] = PRODUCT_FUZZY.unit
        return observed(A, sA, data)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        return self.lift_a2(_function_space(A, B), A, B, _evaluation_morphism(A, B))

    def join(self, A: SetObject) -> Morphism:
        # join(mm)(σ) = let (m, σ') = mm(σ) in m(σ')
        sA = self.fmap_obj(A)
        ssA = self.fmap_obj(sA)
        sigma = self.state
        target_A = ProductSet(components=(A, sigma))
        target_sA = ProductSet(components=(sA, sigma))
        data = torch.full((ssA.cardinality, sA.cardinality), PRODUCT_FUZZY.zero)
        for mm_flat in range(ssA.cardinality):
            mm_outs = _decode_function(mm_flat, sigma.size, target_sA.size)
            new_outs: list[int] = []
            for sig in range(sigma.size):
                m_sig_flat = mm_outs[sig]
                m = m_sig_flat // sigma.size
                sig_prime = m_sig_flat % sigma.size
                m_outs = _decode_function(m, sigma.size, target_A.size)
                new_outs.append(m_outs[sig_prime])
            new_flat = 0
            for v in new_outs:
                new_flat = new_flat * target_A.size + v
            data[mm_flat, new_flat] = PRODUCT_FUZZY.unit
        return observed(ssA, sA, data)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        return self.fmap(A, self.fmap_obj(B), k) >> self.join(B)

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # liftA2(f)(m, n)(σ) = let (a, σ') = m(σ); (b, σ'') = n(σ')
        #                    in (f(a, b), σ'')
        sA = self.fmap_obj(A)
        sB = self.fmap_obj(B)
        sC = self.fmap_obj(C)
        sigma = self.state
        target_A = ProductSet(components=(A, sigma))
        target_B = ProductSet(components=(B, sigma))
        target_C = ProductSet(components=(C, sigma))
        f_table = _function_table(f, A.size * B.size, C.size)
        f_lookup: dict[tuple[int, int], int] = {
            (a, b): f_table[a * B.size + b]
            for a in range(A.size)
            for b in range(B.size)
        }
        source = ProductSet(components=(sA, sB))
        data = torch.full(
            (sA.cardinality, sB.cardinality, sC.cardinality), PRODUCT_FUZZY.zero
        )
        for m_flat in range(sA.cardinality):
            m_outs = _decode_function(m_flat, sigma.size, target_A.size)
            for n_flat in range(sB.cardinality):
                n_outs = _decode_function(n_flat, sigma.size, target_B.size)
                new_outs: list[int] = []
                for sig in range(sigma.size):
                    a_sig = m_outs[sig]
                    a = a_sig // sigma.size
                    sig_p = a_sig % sigma.size
                    b_sig = n_outs[sig_p]
                    b = b_sig // sigma.size
                    sig_pp = b_sig % sigma.size
                    c = f_lookup[(a, b)]
                    new_outs.append(c * sigma.size + sig_pp)
                new_flat = 0
                for v in new_outs:
                    new_flat = new_flat * target_C.size + v
                data[m_flat, n_flat, new_flat] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *sC.shape)
        return observed(source, sC, data)


Monad.register(State)


# ---------------------------------------------------------------------------
# Reader: Reader_ρ(A) = ρ → A
# ---------------------------------------------------------------------------


class Reader(dx.Model):
    """The reader monad: ``Reader_ρ(A) = [ρ → A]`` as a finite function-space."""

    env: SetObject
    name: str = "Reader"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return _function_space(self.env, A)

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # fmap(f)(g) = f ∘ g, where g : ρ → A.
        rA = self.fmap_obj(A)
        rB = self.fmap_obj(B)
        f_lookup = _function_table(f, A.size, B.size)
        data = torch.full((rA.cardinality, rB.cardinality), PRODUCT_FUZZY.zero)
        for g_flat in range(rA.cardinality):
            g_outs = _decode_function(g_flat, self.env.size, A.size)
            new_outs = tuple(f_lookup[g_outs[r]] for r in range(self.env.size))
            new_flat = 0
            for v in new_outs:
                new_flat = new_flat * B.size + v
            data[g_flat, new_flat] = PRODUCT_FUZZY.unit
        return observed(rA, rB, data)

    def pure(self, A: SetObject) -> Morphism:
        # pure(a) = const(a) : ρ → A
        rA = self.fmap_obj(A)
        data = torch.full((A.size, rA.cardinality), PRODUCT_FUZZY.zero)
        for a in range(A.size):
            # const_a: ρ → A always returns a
            outs = tuple(a for _ in range(self.env.size))
            fn_flat = 0
            for v in outs:
                fn_flat = fn_flat * A.size + v
            data[a, fn_flat] = PRODUCT_FUZZY.unit
        return observed(A, rA, data)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        return self.lift_a2(_function_space(A, B), A, B, _evaluation_morphism(A, B))

    def join(self, A: SetObject) -> Morphism:
        # join(mm)(r) = mm(r)(r). mm : ρ → (ρ → A); apply twice at r.
        rA = self.fmap_obj(A)
        rrA = self.fmap_obj(rA)
        rho = self.env
        data = torch.full((rrA.cardinality, rA.cardinality), PRODUCT_FUZZY.zero)
        for mm_flat in range(rrA.cardinality):
            mm_outs = _decode_function(mm_flat, rho.size, rA.cardinality)
            new_outs: list[int] = []
            for r in range(rho.size):
                inner_flat = mm_outs[r]
                inner_outs = _decode_function(inner_flat, rho.size, A.size)
                new_outs.append(inner_outs[r])
            new_flat = 0
            for v in new_outs:
                new_flat = new_flat * A.size + v
            data[mm_flat, new_flat] = PRODUCT_FUZZY.unit
        return observed(rrA, rA, data)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        return self.fmap(A, self.fmap_obj(B), k) >> self.join(B)

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # liftA2(f)(g, h)(r) = f(g(r), h(r))
        rA = self.fmap_obj(A)
        rB = self.fmap_obj(B)
        rC = self.fmap_obj(C)
        rho = self.env
        f_table = _function_table(f, A.size * B.size, C.size)
        f_lookup: dict[tuple[int, int], int] = {
            (a, b): f_table[a * B.size + b]
            for a in range(A.size)
            for b in range(B.size)
        }
        source = ProductSet(components=(rA, rB))
        data = torch.full(
            (rA.cardinality, rB.cardinality, rC.cardinality), PRODUCT_FUZZY.zero
        )
        for g_flat in range(rA.cardinality):
            g_outs = _decode_function(g_flat, rho.size, A.size)
            for h_flat in range(rB.cardinality):
                h_outs = _decode_function(h_flat, rho.size, B.size)
                new_outs = tuple(
                    f_lookup[(g_outs[r], h_outs[r])] for r in range(rho.size)
                )
                new_flat = 0
                for v in new_outs:
                    new_flat = new_flat * C.size + v
                data[g_flat, h_flat, new_flat] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *rC.shape)
        return observed(source, rC, data)


Monad.register(Reader)


# ---------------------------------------------------------------------------
# Writer: Writer_M(A) = A × M with a chosen monoid operation on M
# ---------------------------------------------------------------------------


class Writer(dx.Model):
    """The writer monad ``Writer_M(A) = A × M`` over a finite monoid ``M``.

    The accumulator ``M`` is a finite set with an associative operation
    ``⊕`` and a two-sided unit, the element with flat index
    `unit_index`. ``pure(a) = (a, e)`` and ``join((a, m₁), m₂) =
    (a, m₁ ⊕ m₂)``; the monad laws are exactly the monoid laws of
    ``(M, ⊕, e)``.

    The operation is given as `monoid_op_tensor`, a flat 0/1 tensor
    over ``M × M × M`` in row-major ``(m₁, m₂, m)`` order whose
    ``(m₁, m₂)`` row marks ``m₁ ⊕ m₂``. Without it the operation is
    ``max`` of the flat indices, with unit index 0, the join
    semilattice of the chain ``0 < 1 < ... < |M| - 1``.

    Raises
    ------
    ValueError
        When an operation is used whose table is not a function, or
        that is not associative or does not have `unit_index` as a
        two-sided unit.
    """

    monoid: SetObject
    monoid_op_tensor: tuple[float, ...] | None = None
    unit_index: int = 0
    name: str = "Writer"

    def _op_lookup(self) -> dict[tuple[int, int], int]:
        """The monoid operation as a table, checked against the monoid laws.

        Returns
        -------
        dict[tuple[int, int], int]
            ``table[(m1, m2)]`` is the flat index of ``m1 ⊕ m2``.

        Raises
        ------
        ValueError
            If the supplied table is not a function ``M × M → M``, or
            the operation is not associative, or `unit_index` is not a
            two-sided unit.
        """
        m = self.monoid.size
        lookup: dict[tuple[int, int], int] = {}
        if self.monoid_op_tensor is None:
            for a in range(m):
                for b in range(m):
                    lookup[(a, b)] = max(a, b)
        else:
            t = list(self.monoid_op_tensor)
            if len(t) != m * m * m:
                raise ValueError(
                    f"Writer: monoid_op_tensor has {len(t)} entries, expected {m**3}"
                )
            for a in range(m):
                for b in range(m):
                    row = t[(a * m + b) * m : (a * m + b + 1) * m]
                    hits = [c for c, v in enumerate(row) if v == 1.0]
                    if len(hits) != 1 or any(
                        v != 0.0 for c, v in enumerate(row) if c != hits[0]
                    ):
                        raise ValueError(
                            f"Writer: monoid_op_tensor row ({a}, {b}) does not "
                            "mark exactly one element"
                        )
                    lookup[(a, b)] = hits[0]
        e = self.unit_index
        if not 0 <= e < m:
            raise ValueError(f"Writer: unit_index {e} is outside the monoid")
        for a in range(m):
            if lookup[(e, a)] != a or lookup[(a, e)] != a:
                raise ValueError(
                    f"Writer: element {e} is not a two-sided unit of the operation"
                )
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    if lookup[(lookup[(a, b)], c)] != lookup[(a, lookup[(b, c)])]:
                        raise ValueError(
                            "Writer: the monoid operation is not associative at "
                            f"({a}, {b}, {c})"
                        )
        return lookup

    def fmap_obj(self, A: SetObject) -> SetObject:
        return ProductSet(components=(A, self.monoid))

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # fmap(f)(a, m) = (f(a), m). Tensor product of f with id_M.
        return parallel(f, id_morph(self.monoid))

    def pure(self, A: SetObject) -> Morphism:
        # pure(a) = (a, unit). Pairs a with the constant unit_index.
        unit_const = constant(A, self.monoid, self.unit_index)
        return pair((id_morph(A), unit_const))

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        return self.lift_a2(_function_space(A, B), A, B, _evaluation_morphism(A, B))

    def join(self, A: SetObject) -> Morphism:
        # join((a, m1), m2) = (a, m1 ⊕ m2)
        # carrier: ((A × M) × M) → (A × M)
        inner = ProductSet(components=(A, self.monoid))
        outer = ProductSet(components=(inner, self.monoid))
        op = self._op_lookup()
        m = self.monoid.size
        data = torch.full((A.size, m, m, A.size, m), PRODUCT_FUZZY.zero)
        for a in range(A.size):
            for m1 in range(m):
                for m2 in range(m):
                    data[a, m1, m2, a, op[(m1, m2)]] = PRODUCT_FUZZY.unit
        data = data.reshape(*outer.shape, *inner.shape)
        return observed(outer, inner, data)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        return self.fmap(A, self.fmap_obj(B), k) >> self.join(B)

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        # liftA2(f)((a, m1), (b, m2)) = (f(a, b), m1 ⊕ m2)
        wA = self.fmap_obj(A)
        wB = self.fmap_obj(B)
        wC = self.fmap_obj(C)
        source = ProductSet(components=(wA, wB))
        op = self._op_lookup()
        f_table = _function_table(f, A.size * B.size, C.size)
        f_lookup: dict[tuple[int, int], int] = {
            (a, b): f_table[a * B.size + b]
            for a in range(A.size)
            for b in range(B.size)
        }
        m = self.monoid.size
        data = torch.full(
            (A.size, m, B.size, m, C.size, m),
            PRODUCT_FUZZY.zero,
        )
        for a in range(A.size):
            for m1 in range(m):
                for b in range(B.size):
                    for m2 in range(m):
                        c = f_lookup[(a, b)]
                        m12 = op[(m1, m2)]
                        data[a, m1, b, m2, c, m12] = PRODUCT_FUZZY.unit
        data = data.reshape(*source.shape, *wC.shape)
        return observed(source, wC, data)


Monad.register(Writer)


__all__ = [
    "Identity",
    "Maybe",
    "Alternative_",
    "Continuation",
    "State",
    "Reader",
    "Writer",
]
