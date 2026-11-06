"""Algebras: enrichment algebras for V-enriched categories.

A commutative algebra Q = (L, ⊗, ⋁, ⋀, ¬, I, ⊥) provides the algebraic
structure that parameterizes composition in a V-enriched category:

    (g ∘ f)(a, c) = ⋁_b f(a, b) ⊗ g(b, c)

Different operation pairs give different finite weighted contractions:

    - BooleanAlgebra:  {0,1} with ∧, ∨         → Rel (crisp relations)
    - ProductFuzzyAlgebra: [0,1] with ×, noisy-OR → fuzzy weighted contraction

The interface does not establish categorical laws. Associative matrix
composition requires appropriate distributivity and reduction laws;
the product/noisy-OR pair fails distributivity. ``join`` names the
implemented reduction and need not be a lattice supremum.

The enrichment determines composition, identity, marginalization, and
quantification, all derived from the algebra's operations.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from quivers.core._util import clamp_probs


class CompositionRule(ABC):
    """An associative-or-loose binary tensor contraction.

    Defines the V-enriched composition kernel
    ``(f >> g)[i, k] = ⋁_j f[i, j] ⊗ g[j, k]`` from the two
    primitive operations ``tensor_op`` (binary ⊗) and ``join``
    (reduction ⋁). No identity element is required at this level.

    The hierarchy below this class is:

    * `CompositionRule` — the bare composition surface.
    * `Semigroupoid` — marks an intended associativity contract.
      Associativity of scalar ``tensor_op`` alone does not establish
      associativity of matrix composition; compatible reduction and
      distributivity laws are also needed.
    * `Algebra` — supplies ``unit`` / ``zero``, meet, and negation,
      enabling the runtime's identity and compact-style operations.
      Membership does not prove the mathematical laws of those operations.

    Operations that need identity check at runtime that the
    composition rule is at least a `Algebra`; a clear
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
    """A marker interface intended for associative composition rules.

    Semantically a `CompositionRule` with the marker
    promise of associativity. No identity, no compact-closed
    structure, no negation — those need `Algebra`.

    The marker does not check the contract. In particular, the
    material-implication operation ``1 - a + a*b`` is not associative,
    and associative scalar multiplication alone would not suffice
    without the required reduction and distributivity laws.
    """

    pass


class BilinearForm(CompositionRule):
    """A composition rule with **no** associativity guarantee.

    ``(f >> g)`` is still well-defined as a binary tensor
    contraction, but ``(f >> g) >> h`` may differ from
    ``f >> (g >> h)``. Callers fix an association order
    explicitly — the type system records that the operation
    isn't associative so downstream optimizations can't reorder
    composition chains.

    Examples include signed-dot-product compositions (sign
    flipping breaks associativity), top-k truncating compositions
    (early truncation isn't commutative with later contractions),
    and attention-style softmax-then-multiply rules.

    Sibling of `Semigroupoid` under
    `CompositionRule`: ``BilinearForm`` *opts out* of the
    associativity promise that ``Semigroupoid`` carries; a rule
    that's actually associative should be declared as
    ``Semigroupoid`` instead.
    """

    pass


class Algebra(Semigroupoid):
    """Abstract commutative algebra for V-enriched categories.

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
        """The dual algebra under de Morgan negation.

        For a commutative algebra with involution ``N`` (``negate``),
        the dual carries

            tensor_op^op(a, b) = N(N(a) ⋁ N(b))
            join^op(a_i)       = N(⋀_i N(a_i)) = N(⊗_i N(a_i))   (finite I)

        Unit and zero swap:

            1^op = 0,    0^op = 1.

        For ProductFuzzyAlgebra this yields the role-swapped pair
        ``(⊗ = noisy-OR, ⋁ = product reduction)``, which is the
        canonical Reichenbach-flavour probabilistic-implication
        composition rule.

        The default implementation requires `negate` to be a
        true involution; subclasses with non-involutive lattices
        should override.
        """
        return DualAlgebra(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class DualAlgebra(Algebra):
    """The de-Morgan dual of an involutive commutative algebra.

    For a t-norm / t-conorm pair ``(T, S)`` related by the strong
    negation ``N`` (``S(a, b) = N(T(N(a), N(b)))``), the dual
    algebra carries the role-swapped pair:

        tensor_op^op = base.join         (reducing as a binary op)
        join^op      = base.tensor_op    (reducing as a fold)
        meet^op      = base.join         (along whatever axis)
        unit^op      = base.zero
        zero^op      = base.unit
        negate^op    = base.negate       (involution self-dualizes)

    Concretely for the built-in pairs:

    * ``ProductFuzzyAlgebra.dual``: ⊗ = noisy-OR (``a + b - ab``),
      ⋁ = product (``∏ a_i``).
    * ``Lukasiewicz.dual``: ⊗ = bounded sum (``min(1, a + b)``),
      ⋁ = bounded difference / repeated Łukasiewicz t-norm.
    * ``Godel.dual``: ⊗ = max, ⋁ = min.
    * ``Boolean.dual``: ⊗ = OR, ⋁ = AND.

    Returned by `Algebra.dual`. Subclasses with
    non-involutive negation (CountingAlgebra, …) should override
    ``Algebra.dual`` to raise rather than allow dual construction
    that breaks the de-Morgan equations.
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
        # Dual ⊗ = base ⋁ (as a binary reduction).
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

    Use for composition rules that are associative under their
    ``tensor_op`` but lack an identity element — Reichenbach-style
    material implication composition, weighted shortest-path on
    a non-pointed lattice, etc. Callers who want a full algebra
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


class CustomBilinearForm(BilinearForm):
    """User-defined `BilinearForm` built from callable
    operations.

    Use for composition rules whose ``tensor_op`` is **not**
    associative — for instance signed-dot-product or top-k
    truncating rules. Callers must pin an association order
    explicitly when chaining; the runtime doesn't promise that
    ``(f >> g) >> h == f >> (g >> h)``.

    No associativity smoke test runs because the operation may be
    non-associative.

    Parameters
    ----------
    name : str
        Human-readable name.
    tensor_op : Callable
        Binary product (need not be associative).
    join : Callable
        Reduction along an axis.
    """

    def __init__(
        self,
        name: str,
        tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
    ) -> None:
        if not name:
            raise ValueError("CustomBilinearForm: name must be non-empty")
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
        return f"CustomBilinearForm(name={self._name!r})"


def bilinear_form(
    name: str,
    tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
) -> CustomBilinearForm:
    """Convenience constructor for `CustomBilinearForm`.

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
    CustomBilinearForm
        The composition rule built from the callables.
    """
    return CustomBilinearForm(name, tensor_op, join)


def material_implication() -> CustomSemigroupoid:
    """Reichenbach material implication composition as a Semigroupoid.

    Tensor product is the probabilistic implication
    ``a → b = 1 - a + a*b``; join is the product reduction.
    Associative but lacks an identity, so it's a semigroupoid,
    not an algebra.

    Returns
    -------
    CustomSemigroupoid
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

    # Material implication isn't actually associative — `(a→b)→c`
    # ≠ `a→(b→c)` — but its *composition* under join=product is
    # the well-defined Reichenbach S-implication composition. We
    # skip the associativity smoke test because the binary ⊗
    # itself isn't associative; what matters is the composition
    # algorithm is consistent.
    return CustomSemigroupoid(
        "MaterialImplication", _impl, _prod, verify_associative=False
    )


class ProductFuzzyAlgebra(Algebra):
    """[0,1] with product t-norm and probabilistic sum (noisy-OR).

    This is the enrichment for the Kleisli category of the fuzzy
    powerset monad with the product t-norm:

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

        t_clamped = clamp_probs(t)
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
        product_clamped = clamp_probs(product)
        log_complement = torch.log1p(-product_clamped)

        contract_dims = tuple(range(n_domain, n_domain + n_shared))
        sum_log = log_complement.sum(dim=contract_dims)

        return -torch.expm1(sum_log)


class BooleanAlgebra(Algebra):
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


# ============================================================================
# Łukasiewicz / Gödel t-norm pair on [0, 1]
# ============================================================================


class LukasiewiczAlgebra(Algebra):
    """[0,1] with Łukasiewicz t-norm and bounded sum.

    The Łukasiewicz t-norm is the strongest continuous t-norm:

        ⊗ = Łukasiewicz:   a ⊗ b = max(a + b - 1, 0)
        ⋁ = bounded sum:   ⋁_i x_i = min(1, ∑_i x_i)
        ⋀ = min:           ⋀_i x_i = min_i x_i
        ¬ = strong neg:    ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0
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


class GodelAlgebra(Algebra):
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


# ============================================================================
# Tropical / max-plus / log-prob semirings
# ============================================================================


class TropicalAlgebra(Algebra):
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


class MaxPlusAlgebra(Algebra):
    """Max-plus (Viterbi) semiring on (-∞, ∞].

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


class LogProbAlgebra(Algebra):
    """Log-space sum-product semiring on (-∞, 0].

    Tensor is real addition (probability multiplication in log-
    space) and join is `torch.logsumexp` (probability
    summation in log-space). It supports float32 numerics for
    hierarchical-Bayes log-likelihood pipelines.

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


class RealAlgebra(Algebra):
    """Sum-product semiring on the real numbers (ℝ, +, ·).

    The canonical numeric semiring: addition is the lattice join,
    multiplication the monoidal tensor. Use when entries are
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


class ProbabilityAlgebra(Algebra):
    """Sum-product semiring on [0, 1] with explicit clamp.

    Same operations as `RealAlgebra` but restricted to the
    unit interval.
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


class CountingAlgebra(Algebra):
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
LUKASIEWICZ = LukasiewiczAlgebra()
"""The `LukasiewiczAlgebra` singleton."""
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
PROBABILITY = ProbabilityAlgebra()
"""The `ProbabilityAlgebra` singleton."""
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
# Lives here rather than in ``quivers.core.algebras`` so the
# core categorical layer can reference it without crossing into
# the stochastic subpackage (which itself depends on core).


class MarkovAlgebra(Algebra):
    """Sum-product composition for stochastic matrices.

    Implements the composition rule of FinStoch:

        (g ∘ f)(a, c) = Σ_b f(a, b) · g(b, c)

    standard matrix multiplication on row-stochastic matrices.
    Formally:

        ⊗ = product,   ⋁ = sum,   ⋀ = product,
        ¬ = complement (1 - p),
        I = 1.0, ⊥ = 0.0.

    Not a true algebra in the lattice-theoretic sense (Σ is not
    idempotent), but the composition formula matches the algebra
    interface and composition of row-stochastic matrices yields
    row-stochastic matrices.
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
    "BilinearForm",
    "Algebra",
    "DualAlgebra",
    "CustomAlgebra",
    "CustomSemigroupoid",
    "semigroupoid",
    "CustomBilinearForm",
    "bilinear_form",
    "material_implication",
    "ProductFuzzyAlgebra",
    "BooleanAlgebra",
    "LukasiewiczAlgebra",
    "GodelAlgebra",
    "TropicalAlgebra",
    "MaxPlusAlgebra",
    "LogProbAlgebra",
    "RealAlgebra",
    "ProbabilityAlgebra",
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
