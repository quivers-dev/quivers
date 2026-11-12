"""Algebras: the operations that parameterize tensor composition.

A commutative algebra Q = (L, ⊗, ⋁, ⋀, ¬, I, ⊥) defines the composition
of L-valued matrices

    (g ∘ f)(a, c) = ⋁_b f(a, b) ⊗ g(b, c)

and the identity matrix (I on the diagonal, ⊥ elsewhere). Composition
is associative exactly when the operations satisfy the commutative-
semiring laws; those algebras are instances of `Semiring`, and for them
finite sets and L-valued matrices form a category (for instance Rel
under `BooleanAlgebra`, or finite stochastic matrices under
`MarkovAlgebra`). The remaining algebras, such as the default
`ProductFuzzyAlgebra` (product with noisy-OR), fail distributivity, so
their composition is unital but depends on bracketing. ``join`` names
the implemented reduction and is a lattice supremum only for an
`IdempotentSemiring`.

The algebra also supplies marginalization and quantification, derived
from its operations.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from quivers.core._util import EPS


class CompositionRule(ABC):
    """A binary tensor contraction that promises no algebraic law.

    Defines the V-enriched composition kernel
    ``(f >> g)[i, k] = ⋁_j f[i, j] ⊗ g[j, k]`` from the two
    primitive operations ``tensor_op`` (binary ⊗) and ``join``
    (reduction ⋁). No identity element is required at this level.

    The hierarchy below this class records which laws composition
    satisfies:

    * `Magmoid`: composition is declared to satisfy no law, so the
      caller fixes every association order.
    * `Semigroupoid`: composition is associative.
    * `Algebra`: supplies ``unit`` and ``zero`` (and meet and negation),
      so composition has two-sided identities. Associativity is not
      implied.
    * `Semiring`: an `Algebra` whose operations satisfy the
      commutative-semiring laws, which makes composition associative
      and unital, so finite sets and tensors form a category.
    * `IdempotentSemiring`: a `Semiring` whose ``join`` is idempotent.

    Operations that need identity check at runtime that the
    composition rule is at least an `Algebra`; a clear
    error is raised if a non-algebra rule is fed in.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this composition rule."""
        ...

    @abstractmethod
    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Monoidal product ⊗ (elementwise)."""
        ...

    @abstractmethod
    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Join ⋁ — reduction for composition."""
        ...

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """V-enriched composition.

        Computes: result[d..., c...] = ⋁_{s...} m[d..., s...] ⊗ n[s..., c...]
        """
        if n_contract < 1:
            raise ValueError(f"n_contract must be >= 1, got {n_contract}")
        shared_m = m.shape[-n_contract:]
        shared_n = n.shape[:n_contract]
        if shared_m != shared_n:
            raise ValueError(
                f"shared dimensions do not match: "
                f"m trailing {shared_m} != n leading {shared_n}"
            )
        n_domain = m.ndim - n_contract
        n_codomain = n.ndim - n_contract
        m_expanded = m.reshape(*m.shape, *([1] * n_codomain))
        n_expanded = n.reshape(*([1] * n_domain), *n.shape)
        product = self.tensor_op(m_expanded, n_expanded)
        contract_dims = tuple(range(n_domain, n_domain + n_contract))
        return self.join(product, dim=contract_dims)

    def is_compatible(self, other: CompositionRule) -> bool:
        """Two composition rules compose if they're the same type
        or carry the same name (the latter catches custom-built
        instances of the same rule constructed independently).
        """
        if type(self) is type(other):
            return True
        return getattr(self, "name", None) == getattr(other, "name", None)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Semigroupoid(CompositionRule):
    """A composition rule whose composition is associative.

    A semigroupoid is a category without the requirement of identity
    morphisms: composition is defined for composable pairs and is
    associative [Tilson (1987)]. A `CompositionRule` belongs here
    when ``(f >> g) >> h == f >> (g >> h)`` for all composable
    tensors. Associativity of the scalar ``tensor_op`` alone does not
    suffice: matrix composition also needs ``join`` to be associative
    and commutative and ``tensor_op`` to distribute over it.

    References
    ----------
    Tilson, B. (1987). Categories as algebra: an essential ingredient
    in the theory of monoids. *Journal of Pure and Applied Algebra*,
    48(1-2), 83-198.
    [doi:10.1016/0022-4049(87)90108-3](https://doi.org/10.1016/0022-4049(87)90108-3)
    """

    pass


class Magmoid(CompositionRule):
    """A composition rule declared to satisfy no law.

    A magmoid is a quiver with a binary composition operation and no
    further axioms: neither associativity nor identities are required
    [Arnold and Dauchet (1978)]. It is to a category what a magma is to
    a monoid. ``(f >> g) >> h`` may differ from ``f >> (g >> h)``, so
    callers fix an association order explicitly, and the type records
    that composition chains may not be reordered.

    Examples include signed-dot-product compositions, top-k
    truncating compositions, and the material-implication rule of
    `material_implication`. A rule that is associative should be
    declared as a `Semigroupoid` instead.

    References
    ----------
    Arnold, A., and Dauchet, M. (1978). Théorie des magmoïdes (I).
    *RAIRO Informatique théorique*, 12(3), 235-257.
    [doi:10.1051/ita/1978120302351](https://doi.org/10.1051/ita/1978120302351)
    """

    pass


class Algebra(CompositionRule):
    """Abstract commutative algebra for V-enriched composition.

    An algebra supplies the operations ``(⊗, ⋁, ⋀, ¬)`` and the
    elements ``I`` (``unit``) and ``⊥`` (``zero``). Every shipped
    algebra satisfies ``I ⊗ a = a``, ``⊥ ⊗ a = ⊥`` and ``⋁(a, ⊥) = a``,
    so the identity tensor is a two-sided identity for composition.
    Associativity of composition is a further property, held exactly
    by the `Semiring` subclasses; an `Algebra` that is not a
    `Semiring` (``PRODUCT_FUZZY``, ``LUKASIEWICZ``, ``PROBABILITY`` and
    their duals) composes with identities but not associatively.

    Subclasses must implement the six primitive operations.
    Composition and identity are derived but overridable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this algebra."""
        ...

    # -- primitive operations ------------------------------------------------

    @abstractmethod
    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Monoidal product ⊗ (elementwise).

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand (broadcastable with a).

        Returns
        -------
        torch.Tensor
            a ⊗ b, elementwise.
        """
        ...

    @abstractmethod
    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Join ⋁ — reduction for composition and existential (∃).

        Parameters
        ----------
        t : torch.Tensor
            Input tensor with values in L.
        dim : int or tuple[int, ...]
            Dimension(s) to reduce.

        Returns
        -------
        torch.Tensor
            Reduced tensor.
        """
        ...

    @abstractmethod
    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Meet ⋀ — reduction for universal quantification (∀).

        Parameters
        ----------
        t : torch.Tensor
            Input tensor with values in L.
        dim : int or tuple[int, ...]
            Dimension(s) to reduce.

        Returns
        -------
        torch.Tensor
            Reduced tensor.
        """
        ...

    @abstractmethod
    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Complement / negation ¬.

        Parameters
        ----------
        t : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            ¬t, elementwise.
        """
        ...

    @property
    @abstractmethod
    def unit(self) -> float:
        """Unit element I of the monoidal product ⊗."""
        ...

    @property
    @abstractmethod
    def zero(self) -> float:
        """Bottom element ⊥ (identity for ⋁)."""
        ...

    # -- derived operations --------------------------------------------------

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """V-enriched composition.

        Computes: result[d..., c...] = ⋁_{s...} m[d..., s...] ⊗ n[s..., c...]

        Override for numerical stability in specific algebras.

        Parameters
        ----------
        m : torch.Tensor
            Left tensor of shape (*domain, *shared).
        n : torch.Tensor
            Right tensor of shape (*shared, *codomain).
        n_contract : int
            Number of shared dimensions to contract.

        Returns
        -------
        torch.Tensor
            Composed tensor of shape (*domain, *codomain).
        """
        if n_contract < 1:
            raise ValueError(f"n_contract must be >= 1, got {n_contract}")

        # validate shared dimensions
        shared_m = m.shape[-n_contract:]
        shared_n = n.shape[:n_contract]

        if shared_m != shared_n:
            raise ValueError(
                f"shared dimensions do not match: "
                f"m trailing {shared_m} != n leading {shared_n}"
            )

        n_domain = m.ndim - n_contract
        n_codomain = n.ndim - n_contract

        # broadcast for element-wise tensor_op
        m_expanded = m.reshape(*m.shape, *([1] * n_codomain))
        n_expanded = n.reshape(*([1] * n_domain), *n.shape)

        product = self.tensor_op(m_expanded, n_expanded)

        # join over shared dims
        contract_dims = tuple(range(n_domain, n_domain + n_contract))
        return self.join(product, dim=contract_dims)

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        """Identity morphism tensor for an object with given shape.

        Returns a tensor of shape (*obj_shape, *obj_shape) with
        the unit value on the diagonal and zero elsewhere.

        Parameters
        ----------
        obj_shape : tuple[int, ...]
            Shape of the object (e.g., (n,) for FinSet(n)).

        Returns
        -------
        torch.Tensor
            Identity tensor.
        """
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, self.zero)
        ndim = len(obj_shape)

        if ndim == 1:
            # simple case: (n, n) matrix
            n = obj_shape[0]

            for i in range(n):
                result[i, i] = self.unit

        else:
            # multi-dimensional: iterate over all index tuples
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = self.unit

        return result

    def is_compatible(self, other: Algebra) -> bool:
        """Check if two algebras are compatible for composition.

        Parameters
        ----------
        other : Algebra
            The other algebra.

        Returns
        -------
        bool
            True if morphisms from these algebras can compose.
        """
        if type(self) is type(other):
            return True
        # Two ``DualAlgebra`` instances over the same base are
        # compatible; ditto for any custom algebra that overrides
        # ``name`` to match.
        return getattr(self, "name", None) == getattr(other, "name", None)

    def dual(self) -> Algebra:
        """The role-swapped dual algebra.

        The dual uses the base ``join`` (as a binary operation) for its
        ``tensor_op``, the base ``tensor_op`` (folded as a reduction)
        for its ``join``, and swaps ``unit`` and ``zero``:

            tensor_op^op(a, b) = a ⋁ b
            join^op(a_1, ..., a_n) = a_1 ⊗ ... ⊗ a_n
            1^op = 0,    0^op = 1.

        When ``⊗`` is a t-norm ``T`` and ``⋁`` is its De Morgan dual
        t-conorm ``S(a, b) = N(T(N(a), N(b)))`` under the strong negation
        ``N(a) = 1 - a``, the swap exchanges a t-norm with its dual
        t-conorm. For ``ProductFuzzyAlgebra`` this yields the pair
        ``(⊗ = probabilistic sum, ⋁ = product reduction)``.

        Returns
        -------
        Algebra
            The dual. `BooleanAlgebra` and `GodelAlgebra` return a
            `DualIdempotentSemiring`, since swapping a distributive
            lattice's join and meet gives a distributive lattice again;
            every other base returns a `DualAlgebra`, which need not be
            a semiring even when the base is one.
        """
        return DualAlgebra(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Semiring(Algebra, Semigroupoid):
    """An algebra whose operations form a commutative semiring.

    A commutative semiring ``(L, ⊕, ⊗, 0, 1)`` has ``(L, ⊕, 0)`` and
    ``(L, ⊗, 1)`` commutative monoids, ``⊗`` distributing over ``⊕``
    on both sides, and ``0`` annihilating ``⊗`` [Golan (1999)]. Here
    ``⊕`` is the binary form of ``join``, ``0`` is ``zero`` and ``1`` is
    ``unit``. These laws make the composition
    ``(f >> g)[i, k] = ⊕_j f[i, j] ⊗ g[j, k]`` associative and unital,
    so finite sets with tensors over ``L`` form a category (the
    category of ``L``-valued matrices). Membership in this class is a
    claim that the laws hold; the test suite checks every shipped
    instance against them.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    pass


class IdempotentSemiring(Semiring):
    """A commutative semiring whose addition is idempotent.

    Additive idempotence, ``a ⊕ a = a`` for every ``a``, makes ``⊕`` the
    join of the partial order ``a ≤ b ⟺ a ⊕ b = b`` [Golan (1999)].
    On finite carriers of composition this is the structure that a
    quantale restricts to, and it is what the fuzzy powerset monad
    needs (see `quivers.monadic.monads.FuzzyPowersetMonad`).

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    pass


class DualAlgebra(Algebra):
    """The role-swapped dual of a commutative algebra.

    For a t-norm / t-conorm pair ``(T, S)`` related by the strong
    negation ``N`` (``S(a, b) = N(T(N(a), N(b)))``), the dual
    algebra carries the role-swapped pair:

        tensor_op^op = base.join         (as a binary operation)
        join^op      = base.tensor_op    (folded as a reduction)
        meet^op      = base.join
        unit^op      = base.zero
        zero^op      = base.unit
        negate^op    = base.negate

    Concretely for the built-in pairs:

    * ``PRODUCT_FUZZY.dual()``: ⊗ = probabilistic sum (``a + b - ab``),
      ⋁ = product reduction (``∏ a_i``).
    * ``LUKASIEWICZ.dual()``: ⊗ = bounded sum (``min(1, a + b)``),
      ⋁ = folded Łukasiewicz t-norm.
    * ``GODEL.dual()``: ⊗ = max, ⋁ = min.
    * ``BOOLEAN.dual()``: ⊗ = OR, ⋁ = AND.

    The dual of a semiring need not be a semiring: the dual of
    ``PRODUCT_FUZZY`` composes with identities but not associatively.
    The Boolean and Gödel duals are distributive lattices with the
    roles of join and meet exchanged, and are returned as the
    `DualIdempotentSemiring` subclass.
    """

    def __init__(self, base: Algebra) -> None:
        self._base = base
        self._name = f"Dual({base.name})"

    @property
    def base(self) -> Algebra:
        """The underlying algebra this is the dual of."""
        return self._base

    @property
    def name(self) -> str:
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Dual ⊗ = base ⋁ (as a binary reduction). The operands are
        # broadcast first so that composition's outer-product layout
        # stacks equal shapes.
        a, b = torch.broadcast_tensors(a, b)
        stacked = torch.stack([a, b], dim=-1)
        return self._base.join(stacked, dim=-1)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        # Dual ⋁ is the base ⊗ folded as a reduction. We unbind
        # the requested axes and tensor-op the chunks together.
        return self._reduce_with_op(self._base.tensor_op, t, dim)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        # Dual ⋀ = base ⋁ (the duals swap meet and join too).
        return self._base.join(t, dim=dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return self._base.negate(t)

    @property
    def unit(self) -> float:
        # unit^op = N(unit_base) — for involutive negations,
        # equals the base's zero.
        return float(self._base.zero)

    @property
    def zero(self) -> float:
        return float(self._base.unit)

    def dual(self) -> Algebra:
        """Dual of dual is the base (involution)."""
        return self._base

    @staticmethod
    def _reduce_with_op(
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        dim: int | tuple[int, ...],
    ) -> torch.Tensor:
        """Fold ``op`` along ``dim`` of ``t``. Used to lift the
        base's binary tensor_op into a reduction for the dual join."""
        return _fold_along_dim(op, t, dim)


class DualIdempotentSemiring(DualAlgebra, IdempotentSemiring):
    """The dual of a distributive-lattice algebra.

    Returned by `BooleanAlgebra.dual` and `GodelAlgebra.dual`. Swapping
    the join and meet of a distributive lattice yields a distributive
    lattice, so the dual is again an idempotent commutative semiring:
    ``max`` (or OR) distributes over ``min`` (or AND), the unit of
    ``max`` is the bottom and the bottom of the reversed order, the old
    top, annihilates ``max``.
    """


def _fold_along_dim(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    dim: int | tuple[int, ...],
) -> torch.Tensor:
    """Fold a binary tensor-op across one or more axes.

    Used both by `DualAlgebra` to lift the base tensor_op
    into a reduction, and by `CustomAlgebra` whose user-
    supplied tensor_op is binary but is reduced as a join.
    """
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)
    for d in sorted(dims, reverse=True):
        chunks = list(torch.unbind(t, dim=d))
        if not chunks:
            continue
        acc = chunks[0]
        for chunk in chunks[1:]:
            acc = op(acc, chunk)
        t = acc
    return t


class CustomAlgebra(Algebra):
    """User-defined algebra built from callable operations.

    Construct a fresh algebra by supplying the primitive operations
    as Python functions, rather than subclassing `Algebra`
    for each variant.

    The constructor only stores the operations; **the user is
    responsible for ensuring they satisfy the algebra axioms**
    (associativity, identity, distributivity of ⊗ over ⋁,
    de-Morgan duality between ⊗ and ⋁ via ``negate`` for
    involutive lattices). Basic structural axioms are
    sanity-checked at construction time against a handful of
    fixed sample inputs; serious deployments should write their
    own targeted unit tests.

    The DSL surface ``algebra name { tensor_op(a, b) = …;
    join(t) = …; unit = …; zero = …; }`` compiles to this class
    under the hood, with ``verify=False`` (user expressions are
    arithmetic and routinely violate one of the canned samples
    used by `_sanity_check`).
    """

    def __init__(
        self,
        name: str,
        tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
        unit: float,
        zero: float,
        negate: Callable[[torch.Tensor], torch.Tensor] | None = None,
        meet: (
            Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor] | None
        ) = None,
        verify: bool = True,
    ) -> None:
        if not name:
            raise ValueError("CustomAlgebra: name must be non-empty")
        self._name = str(name)
        self._tensor_op = tensor_op
        self._join = join
        self._unit = float(unit)
        self._zero = float(zero)
        self._negate = negate
        self._meet = meet
        if verify:
            self._sanity_check()

    @property
    def name(self) -> str:
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        return self._join(t, dim)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if self._meet is None:
            raise NotImplementedError(
                f"CustomAlgebra {self._name!r}: no meet supplied. "
                f"Pass meet=... to the constructor for explicit "
                f"universal-quantifier support."
            )
        return self._meet(t, dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        if self._negate is None:
            raise NotImplementedError(
                f"CustomAlgebra {self._name!r}: no negation supplied. "
                f"Pass negate=... to the constructor for explicit "
                f"complement support."
            )
        return self._negate(t)

    @property
    def unit(self) -> float:
        return self._unit

    @property
    def zero(self) -> float:
        return self._zero

    def _sanity_check(self) -> None:
        """Cheap probabilistic checks of the algebra axioms.

        Verifies on a handful of fixed sample tensors:

        * ``tensor_op(unit, a) == a`` (left identity);
        * ``tensor_op(a, unit) == a`` (right identity);
        * ``tensor_op(zero, a) == zero`` (left absorbing);
        * ``join`` returns a tensor of the same dtype.

        Raises ``ValueError`` on the first failed check so the
        construction site can fix the spec rather than silently
        ship a broken algebra.
        """
        a = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        unit_t = torch.full_like(a, self._unit)
        zero_t = torch.full_like(a, self._zero)
        left_id = self._tensor_op(unit_t, a)
        right_id = self._tensor_op(a, unit_t)
        left_zero = self._tensor_op(zero_t, a)
        if not torch.allclose(left_id, a, atol=1e-5):
            raise ValueError(
                f"CustomAlgebra {self._name!r}: tensor_op fails "
                f"left-identity check on unit={self._unit}; "
                f"expected {a.tolist()}, got {left_id.tolist()}"
            )
        if not torch.allclose(right_id, a, atol=1e-5):
            raise ValueError(
                f"CustomAlgebra {self._name!r}: tensor_op fails "
                f"right-identity check on unit={self._unit}; "
                f"expected {a.tolist()}, got {right_id.tolist()}"
            )
        if not torch.allclose(left_zero, zero_t, atol=1e-5):
            raise ValueError(
                f"CustomAlgebra {self._name!r}: tensor_op fails "
                f"left-absorbing check on zero={self._zero}; "
                f"expected {zero_t.tolist()}, got {left_zero.tolist()}"
            )

    def __repr__(self) -> str:
        return f"CustomAlgebra(name={self._name!r})"


class CustomSemigroupoid(Semigroupoid):
    """User-defined `Semigroupoid` built from callable
    operations.

    Use for composition rules whose composition is associative but
    which lack an identity element, for instance weighted shortest
    paths over a semiring without a multiplicative unit. Callers who
    want a full algebra
    (with identity, dagger, compact-closed structure) should use
    `CustomAlgebra` instead.

    Parameters
    ----------
    name : str
        Human-readable name.
    tensor_op : Callable
        Binary monoidal product.
    join : Callable
        Reduction along an axis.
    verify_associative : bool
        When ``True`` (default) checks ``tensor_op`` is associative
        on a fixed pseudo-random sample at construction time.  Set
        ``False`` to skip the smoke test.

    Raises
    ------
    ValueError
        If ``name`` is empty, or if ``verify_associative`` is set and
        ``tensor_op`` fails the associativity check.
    """

    def __init__(
        self,
        name: str,
        tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
        verify_associative: bool = True,
    ) -> None:
        if not name:
            raise ValueError("CustomSemigroupoid: name must be non-empty")
        self._name = str(name)
        self._tensor_op = tensor_op
        self._join = join
        if verify_associative:
            self._check_associativity()

    @property
    def name(self) -> str:
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        return self._join(t, dim)

    def _check_associativity(self) -> None:
        """Smoke-test ``(a ⊗ b) ⊗ c == a ⊗ (b ⊗ c)`` on a fixed
        sample. Not exhaustive — a determined user can construct
        a non-associative ``tensor_op`` that passes this check —
        but it catches gross mistakes."""
        torch.manual_seed(0)
        a = torch.rand(4)
        b = torch.rand(4)
        c = torch.rand(4)
        left = self._tensor_op(self._tensor_op(a, b), c)
        right = self._tensor_op(a, self._tensor_op(b, c))
        if not torch.allclose(left, right, atol=1e-4):
            raise ValueError(
                f"CustomSemigroupoid {self._name!r}: tensor_op fails "
                f"associativity check on a fixed sample. If you're "
                f"certain the operation is associative and the check "
                f"hit a numerical edge case, pass "
                f"verify_associative=False."
            )

    def __repr__(self) -> str:
        return f"CustomSemigroupoid(name={self._name!r})"


def semigroupoid(
    name: str,
    tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
    *,
    verify_associative: bool = True,
) -> CustomSemigroupoid:
    """Convenience constructor for `CustomSemigroupoid`.

    Parameters
    ----------
    name : str
        Human-readable name.
    tensor_op : Callable
        Binary monoidal product.
    join : Callable
        Reduction along an axis.
    verify_associative : bool
        When ``True`` (default), smoke-tests ``tensor_op`` for
        associativity on a fixed sample.

    Returns
    -------
    CustomSemigroupoid
        The composition rule built from the callables.

    Raises
    ------
    ValueError
        If ``name`` is empty, or if ``verify_associative`` is set and
        ``tensor_op`` fails the associativity check.
    """
    return CustomSemigroupoid(
        name, tensor_op, join, verify_associative=verify_associative
    )


class CustomMagmoid(Magmoid):
    """User-defined `Magmoid` built from callable operations.

    Use for composition rules whose composition is not associative,
    for instance signed-dot-product or top-k truncating rules. Callers
    must pin an association order explicitly when chaining; the
    runtime does not promise that ``(f >> g) >> h == f >> (g >> h)``.

    No associativity smoke test runs because a magmoid satisfies no
    law.

    Parameters
    ----------
    name : str
        Human-readable name.
    tensor_op : Callable
        Binary product (need not be associative).
    join : Callable
        Reduction along an axis.

    Raises
    ------
    ValueError
        If ``name`` is empty.
    """

    def __init__(
        self,
        name: str,
        tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
    ) -> None:
        if not name:
            raise ValueError("CustomMagmoid: name must be non-empty")
        self._name = str(name)
        self._tensor_op = tensor_op
        self._join = join

    @property
    def name(self) -> str:
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        return self._join(t, dim)

    def __repr__(self) -> str:
        return f"CustomMagmoid(name={self._name!r})"


def magmoid(
    name: str,
    tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
) -> CustomMagmoid:
    """Convenience constructor for `CustomMagmoid`.

    Parameters
    ----------
    name : str
        Human-readable name.
    tensor_op : Callable
        Binary product, which need not be associative.
    join : Callable
        Reduction along an axis.

    Returns
    -------
    CustomMagmoid
        The composition rule built from the callables.

    Raises
    ------
    ValueError
        If ``name`` is empty.
    """
    return CustomMagmoid(name, tensor_op, join)


def material_implication() -> CustomMagmoid:
    """Reichenbach material-implication composition as a `Magmoid`.

    The tensor product is the Reichenbach implication
    ``a → b = 1 - a + a*b`` and the join is the product reduction, so
    ``(f >> g)[i, k] = ∏_j (1 - f[i, j] + f[i, j] g[j, k])``. Neither
    the scalar implication nor this composition is associative, and
    no element is a two-sided identity, so the rule is a magmoid.

    Returns
    -------
    CustomMagmoid
        The material-implication composition rule.
    """

    def _impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 1.0 - a + a * b

    def _prod(t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.prod(dim=d)
        return result

    return CustomMagmoid("MaterialImplication", _impl, _prod)


class ProductFuzzyAlgebra(Algebra):
    """[0,1] with product t-norm and probabilistic sum (noisy-OR).

    The product t-norm does not distribute over the probabilistic
    sum (``0.5 · (0.5 ⊕ 0.5) = 0.375`` while
    ``(0.5 · 0.5) ⊕ (0.5 · 0.5) = 0.4375``), so this algebra is not a
    semiring and composition is unital but not associative: chains of
    ``>>`` depend on their bracketing. The operations are

        ⊗ = product:      a ⊗ b = a * b
        ⋁ = noisy-OR:     ⋁_i x_i = 1 - ∏_i (1 - x_i)
        ⋀ = product:      ⋀_i x_i = ∏_i x_i
        ¬ = complement:    ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    Composition uses log-space for numerical stability.
    """

    @property
    def name(self) -> str:
        return "ProductFuzzy"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Noisy-OR in log-space: 1 - exp(∑ log(1 - t))."""
        if isinstance(dim, int):
            dim = (dim,)

        t_clamped = t.clamp(min=0.0, max=1.0 - EPS)
        log_complement = torch.log1p(-t_clamped)
        sum_log = log_complement.sum(dim=dim)

        return -torch.expm1(sum_log)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Product (fuzzy AND): ∏_i t_i."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.prod(dim=d)

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """Override for log-space numerical stability.

        Computes noisy-OR contraction matching the existing
        noisy_or_contract implementation exactly.
        """
        if n_contract < 1:
            raise ValueError(f"n_contract must be >= 1, got {n_contract}")

        shared_m = m.shape[-n_contract:]
        shared_n = n.shape[:n_contract]

        if shared_m != shared_n:
            raise ValueError(
                f"shared dimensions do not match: "
                f"m trailing {shared_m} != n leading {shared_n}"
            )

        n_domain = m.ndim - n_contract
        n_codomain = n.ndim - n_contract
        n_shared = n_contract

        m_expanded = m.reshape(*m.shape, *([1] * n_codomain))
        n_expanded = n.reshape(*([1] * n_domain), *n.shape)

        product = m_expanded * n_expanded

        # log-space noisy-OR for stability
        product_clamped = product.clamp(min=0.0, max=1.0 - EPS)
        log_complement = torch.log1p(-product_clamped)

        contract_dims = tuple(range(n_domain, n_domain + n_shared))
        sum_log = log_complement.sum(dim=contract_dims)

        return -torch.expm1(sum_log)


class BooleanAlgebra(IdempotentSemiring):
    """{0, 1} with logical AND and OR.

    The enrichment for the category Rel of crisp binary relations:

        ⊗ = AND:     a ⊗ b = a ∧ b
        ⋁ = OR:      ⋁_i x_i = max_i x_i
        ⋀ = AND:     ⋀_i x_i = min_i x_i
        ¬ = NOT:     ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    Works on float tensors with values in {0.0, 1.0}. Intermediate
    fuzzy values are rounded.
    """

    @property
    def name(self) -> str:
        return "Boolean"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Logical AND via product (exact for {0,1} inputs)."""
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Logical OR via iterated max."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values

        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Logical AND via iterated min."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def dual(self) -> Algebra:
        """The dual algebra, an idempotent commutative semiring.

        Returns
        -------
        Algebra
            A `DualIdempotentSemiring` over this algebra.
        """
        return DualIdempotentSemiring(self)


# ============================================================================
# Standard MV-algebra and Gödel algebra on [0, 1]
# ============================================================================


class StandardMVAlgebra(Algebra):
    """The standard MV-algebra on [0, 1].

    The standard MV-algebra is ``[0, 1]`` with ``x ⊕ y = min(1, x + y)``
    and ``¬x = 1 - x``, the real-valued semantics of Łukasiewicz logic
    [Chang (1958); Cignoli, D'Ottaviano and Mundici (2000)]. Its derived
    product is the Łukasiewicz t-norm and its lattice meet is ``min``:

        ⊗ = Łukasiewicz t-norm:  a ⊗ b = max(a + b - 1, 0)
        ⋁ = bounded sum:         ⋁_i x_i = min(1, ∑_i x_i)
        ⋀ = min:                 ⋀_i x_i = min_i x_i
        ¬ = strong negation:     ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    The Łukasiewicz t-norm does not distribute over the bounded sum
    (``0.5 ⊗ (0.5 ⊕ 0.5) = 0.5`` while ``(0.5 ⊗ 0.5) ⊕ (0.5 ⊗ 0.5) = 0``),
    so this algebra is not a semiring and composition depends on its
    bracketing. The name follows the standard terminology: a
    "Łukasiewicz algebra" in the literature is Moisil's distinct
    Łukasiewicz-Moisil algebra.

    References
    ----------
    Chang, C. C. (1958). Algebraic analysis of many valued logics.
    *Transactions of the American Mathematical Society*, 88(2), 467-490.
    [doi:10.1090/S0002-9947-1958-0094302-9](https://doi.org/10.1090/S0002-9947-1958-0094302-9)

    Cignoli, R. L. O., D'Ottaviano, I. M. L., and Mundici, D. (2000).
    *Algebraic Foundations of Many-Valued Reasoning*. Springer.
    [doi:10.1007/978-94-015-9480-6](https://doi.org/10.1007/978-94-015-9480-6)
    """

    @property
    def name(self) -> str:
        return "Lukasiewicz"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b - 1.0).clamp(min=0.0)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        return t.sum(dim=dim).clamp(max=1.0)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0


class GodelAlgebra(IdempotentSemiring):
    """[0,1] with Gödel (min) t-norm.

    ⊗ = min,   ⋁ = max,   ⋀ = min,
    ¬ = Gödel neg (1 if a == 0 else 0),
    I = 1.0, ⊥ = 0.0.
    """

    @property
    def name(self) -> str:
        return "Godel"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.min(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return (t == 0.0).float()

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def dual(self) -> Algebra:
        """The dual algebra, an idempotent commutative semiring.

        Returns
        -------
        Algebra
            A `DualIdempotentSemiring` over this algebra.
        """
        return DualIdempotentSemiring(self)


# ============================================================================
# Tropical / max-plus / log-prob semirings
# ============================================================================


class TropicalAlgebra(IdempotentSemiring):
    """[0, ∞] with (+, min) — Lawvere metric spaces.

        ⊗ = addition (distances compose),
        ⋁ = infimum (shortest path),
        ⋀ = supremum (longest path),
        I = 0.0, ⊥ = ∞.

    Composition is the tropical / (min, +) matrix product.
    Negation is undefined.
    """

    @property
    def name(self) -> str:
        return "Tropical"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the tropical algebra"
        )

    @property
    def unit(self) -> float:
        return 0.0

    @property
    def zero(self) -> float:
        return float("inf")

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, float("inf"))
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 0.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 0.0
        return result


class MaxPlusAlgebra(IdempotentSemiring):
    """Max-plus (Viterbi) semiring on ℝ ∪ {-∞}.

    Distinct from `TropicalAlgebra` (which is min-plus,
    suited to shortest-path aggregations): the join here is ``max``
    and the tensor is ``+``. The canonical algebra for MAP decoding
    in HMMs, CRFs, and weighted automata.

        ⊗ = +,   ⋁ = max,   ⋀ = min,
        I = 0, ⊥ = -∞.
    """

    @property
    def name(self) -> str:
        return "MaxPlus"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the max-plus algebra"
        )

    @property
    def unit(self) -> float:
        return 0.0

    @property
    def zero(self) -> float:
        return -float("inf")

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, -float("inf"))
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 0.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 0.0
        return result


class LogProbAlgebra(Semiring):
    """Log-space sum-product semiring on [-∞, ∞).

    Tensor is real addition (probability multiplication in log-
    space) and join is `torch.logsumexp` (probability
    summation in log-space). ``exp`` is an isomorphism onto the
    probability semiring ``([0, ∞), +, ·)``, so the carrier must
    contain positive values: ``logsumexp(0, 0) = log 2``. The join is
    not idempotent, so this is a semiring and not a lattice. It
    supports float32 numerics for hierarchical-Bayes log-likelihood
    pipelines.

        ⊗ = +,   ⋁ = logsumexp,   ⋀ = min,
        I = 0 (log 1), ⊥ = -∞ (log 0).
    """

    @property
    def name(self) -> str:
        return "LogProb"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        # ``torch.logsumexp`` has the right value for an all-``-inf``
        # reduction but differentiates that value as ``NaN``. Impossible
        # dynamic-programming cells are represented by exactly such rows, so
        # make the empty sum an explicit constant branch. Replacing the zero
        # before ``log`` is essential: masking the result afterwards leaves
        # ``log(0)`` in the autograd graph.
        peak = t.amax(dim=dim, keepdim=True)
        empty = torch.isneginf(peak)
        shift = torch.where(empty, torch.zeros_like(peak), peak)
        total = (t - shift).exp().sum(dim=dim, keepdim=True)
        safe_total = torch.where(empty, torch.ones_like(total), total)
        combined = safe_total.log() + shift
        result = torch.where(
            empty,
            torch.full_like(combined, -float("inf")),
            combined,
        )
        for axis in sorted((d % t.ndim for d in dim), reverse=True):
            result = result.squeeze(axis)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the log-prob algebra"
        )

    @property
    def unit(self) -> float:
        return 0.0

    @property
    def zero(self) -> float:
        return -float("inf")

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, -float("inf"))
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 0.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 0.0
        return result


# ============================================================================
# Numeric sum-product semirings (Real / Probability / Counting)
# ============================================================================


class RealAlgebra(Semiring):
    """Sum-product semiring on the real numbers (ℝ, +, ·).

    The canonical numeric semiring: addition is the semiring sum (not
    a lattice join, since it is not idempotent), multiplication the
    tensor. Use when entries are
    unbounded real weights with no probability interpretation.
    """

    @property
    def name(self) -> str:
        return "Real"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.sum(dim=d)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return -t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.zeros(full_shape)
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 1.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 1.0
        return result


class BoundedSumProductAlgebra(Algebra):
    """[0, 1] with the product t-norm and the bounded sum.

    The tensor is the product t-norm ``a · b`` and the join is the
    bounded sum ``min(1, ∑_i x_i)``, the Łukasiewicz t-conorm
    [Klement, Mesiar and Pap (2000)]. Truncating the sum at 1 breaks
    distributivity (``0.5 · min(1, 1 + 1) = 0.5`` while
    ``min(1, 0.5 + 0.5)`` over the products is ``1``), so this algebra is
    not a semiring and composition depends on its bracketing. The
    probability semiring ``([0, ∞), +, ·)`` is `MarkovAlgebra`; this class
    is not a probability algebra in the measure-theoretic sense (a
    Boolean algebra carrying a measure) either.

    References
    ----------
    Klement, E. P., Mesiar, R., and Pap, E. (2000). *Triangular Norms*.
    Springer.
    [doi:10.1007/978-94-015-9540-7](https://doi.org/10.1007/978-94-015-9540-7)
    """

    @property
    def name(self) -> str:
        return "Probability"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).clamp(min=0.0, max=1.0)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.sum(dim=d)
        return result.clamp(min=0.0, max=1.0)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t).clamp(min=0.0, max=1.0)

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.zeros(full_shape)
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 1.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 1.0
        return result


class CountingAlgebra(Semiring):
    """Sum-product semiring on the non-negative integers (ℕ, +, ·).

    Counting algebra: composition counts the number of distinct
    paths through a structure. The underlying tensor is
    float-typed (PyTorch's autograd requires it) but operations
    are integer-respecting.
    """

    @property
    def name(self) -> str:
        return "Counting"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.sum(dim=d)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the counting "
            "(non-negative integer) algebra"
        )

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.zeros(full_shape)
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 1.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 1.0
        return result


# ============================================================================
# Module-level singletons
# ============================================================================

PRODUCT_FUZZY = ProductFuzzyAlgebra()
"""The `ProductFuzzyAlgebra` singleton, the default algebra of every morphism."""
BOOLEAN = BooleanAlgebra()
"""The `BooleanAlgebra` singleton, enriching crisp binary relations."""
LUKASIEWICZ = StandardMVAlgebra()
"""The `StandardMVAlgebra` singleton."""
GODEL = GodelAlgebra()
"""The `GodelAlgebra` singleton."""
TROPICAL = TropicalAlgebra()
"""The `TropicalAlgebra` singleton, the min-plus enrichment of Lawvere metric spaces."""
MAX_PLUS = MaxPlusAlgebra()
"""The `MaxPlusAlgebra` singleton, the Viterbi semiring."""
LOG_PROB = LogProbAlgebra()
"""The `LogProbAlgebra` singleton, the log-space sum-product semiring."""
REAL = RealAlgebra()
"""The `RealAlgebra` singleton."""
PROBABILITY = BoundedSumProductAlgebra()
"""The `BoundedSumProductAlgebra` singleton."""
COUNTING = CountingAlgebra()
"""The `CountingAlgebra` singleton."""

REICHENBACH = PRODUCT_FUZZY.dual()
"""The de Morgan dual of `PRODUCT_FUZZY`, the canonical probabilistic-implication
composition."""
BOOLEAN_DUAL = BOOLEAN.dual()
"""The de Morgan dual of `BOOLEAN`, which swaps its tensor and join."""
DUAL_LUKASIEWICZ = LUKASIEWICZ.dual()
"""The de Morgan dual of `LUKASIEWICZ`, which swaps its tensor and join."""
DUAL_GODEL = GODEL.dual()
"""The de Morgan dual of `GODEL`, which swaps its tensor and join."""


# ============================================================================
# Markov sum-product algebra (Kleisli composition for stochastic kernels)
# ============================================================================
#
# Defined in the core layer so that core categorical code can use it
# without importing the stochastic subpackage, which depends on core.


class MarkovAlgebra(Semiring):
    """Sum-product composition for stochastic matrices.

    Implements the composition rule of FinStoch:

        (g ∘ f)(a, c) = Σ_b f(a, b) · g(b, c)

    standard matrix multiplication on row-stochastic matrices.
    Formally:

        ⊗ = product,   ⋁ = sum,   ⋀ = product,
        ¬ = complement (1 - p),
        I = 1.0, ⊥ = 0.0.

    The operations form the probability semiring ``([0, ∞), +, ·)``;
    its sum is not idempotent, so it is a semiring and not a lattice.
    Composition is ordinary matrix multiplication, and the composite
    of row-stochastic matrices is row-stochastic.
    """

    @property
    def name(self) -> str:
        return "Markov"

    def tensor_op(self, a, b):
        return a * b

    def join(self, t, dim):
        if isinstance(dim, int):
            dim = (dim,)
        return t.sum(dim=dim)

    def meet(self, t, dim):
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.prod(dim=d)
        return result

    def negate(self, t):
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def compose(self, m, n, n_contract):
        return super().compose(m, n, n_contract)


MARKOV = MarkovAlgebra()
"""The `MarkovAlgebra` singleton, composing stochastic matrices."""


__all__ = [
    "CompositionRule",
    "Semigroupoid",
    "Magmoid",
    "Algebra",
    "Semiring",
    "IdempotentSemiring",
    "DualAlgebra",
    "DualIdempotentSemiring",
    "CustomAlgebra",
    "CustomSemigroupoid",
    "semigroupoid",
    "CustomMagmoid",
    "magmoid",
    "material_implication",
    "ProductFuzzyAlgebra",
    "BooleanAlgebra",
    "StandardMVAlgebra",
    "GodelAlgebra",
    "TropicalAlgebra",
    "MaxPlusAlgebra",
    "LogProbAlgebra",
    "RealAlgebra",
    "BoundedSumProductAlgebra",
    "CountingAlgebra",
    "MarkovAlgebra",
    "PRODUCT_FUZZY",
    "BOOLEAN",
    "LUKASIEWICZ",
    "GODEL",
    "TROPICAL",
    "MAX_PLUS",
    "LOG_PROB",
    "REAL",
    "PROBABILITY",
    "COUNTING",
    "MARKOV",
    "REICHENBACH",
    "BOOLEAN_DUAL",
    "DUAL_LUKASIEWICZ",
    "DUAL_GODEL",
]
