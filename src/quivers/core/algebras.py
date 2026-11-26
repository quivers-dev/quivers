"""Algebras: the operations that parameterize tensor composition.

A commutative algebra Q = (L, ⊗, ⋁, ⋀, ¬, I, ⊥) defines the composition
of L-valued matrices

    (g ∘ f)(a, c) = ⋁_b f(a, b) ⊗ g(b, c)

and the identity matrix (I on the diagonal, ⊥ elsewhere). Composition
is associative exactly when the operations satisfy the semiring laws;
those algebras are instances of `Semiring`, and for them finite sets
and L-valued matrices form a category (for instance Rel under
`BooleanAlgebra`, or finite stochastic matrices under `MarkovAlgebra`).
The remaining algebras, such as the default `ProductFuzzyAlgebra`
(product with noisy-OR), fail distributivity, so their composition is
unital but depends on bracketing. ``join`` names the implemented
reduction and is a lattice supremum only for an `IdempotentSemiring`.

The classes form one hierarchy. Each class states the laws its members
satisfy as `quivers.core.laws.Law` records, and every member reports,
for each law it claims, a `quivers.core.laws.LawClaim` whose evidence
is a written argument, a decision procedure, a test on finite data, or
a declaration.
Executors ask for a capability by type: `Selective` sums return one of
their operands, `ZeroSumFree` and `NoZeroDivisors` make support a
homomorphism, a `Semifield` divides, a `Ring` subtracts, and a
`Semimodule` is acted on by a semiring. A `Carrier` gives the zero (and
a `UnitalCarrier` the unit) in the shape, dtype and device a computation
needs, decides membership, and returns the zero for a reduction over no
terms.

References
----------
Golan, J. S. (1999). *Semirings and their Applications*. Springer.
[doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
"""

from __future__ import annotations

import math
from fractions import Fraction
from abc import abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import ClassVar

import didactic.api as dx
import torch

from quivers._positions import demand, depends_on_position
from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.core._carriers import (
    constant,
    empty_reduction,
    is_empty,
    reduced_shape,
    reduction_axes,
    require_real,
    validate_carrier,
)
from quivers.core._util import EPS
from quivers.core.laws import (
    ADDITIVE_INVERSE,
    COMPOSITION_ASSOCIATIVE,
    DISTRIBUTIVE,
    DIVISION,
    EMPTY_JOIN,
    FIRST_MAXIMIZER,
    JOIN_ITERATED,
    NO_ZERO_DIVISORS,
    OPERATIONS_CLOSED,
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    PLUS_IDEMPOTENT,
    PLUS_SELECTIVE,
    PLUS_UNIT,
    SEMIMODULE_ACTION,
    STAR_DECIDED,
    STAR_LEAST_SOLUTION,
    TIE_FIRST_MAXIMIZER,
    TIMES_ASSOCIATIVE,
    TIMES_COMMUTATIVE,
    TIMES_UNIT,
    UNIT_FORM,
    ZERO_ANNIHILATES,
    ZERO_FORM,
    ZERO_SUM_FREE,
    Citation,
    ClaimSubject,
    Law,
    LawEvidence,
    Lawful,
    Proved,
    TieRule,
)


def _pair(
    join: Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """A binary sum as a reduction over a new leading axis.

    Parameters
    ----------
    join : Callable
        The reduction.
    a : torch.Tensor
        The first operand.
    b : torch.Tensor
        The second operand, broadcastable with ``a``.

    Returns
    -------
    torch.Tensor
        ``join`` over the axis holding ``a`` then ``b``.
    """
    a, b = torch.broadcast_tensors(a, b)
    return join(torch.stack((a, b)), 0)


class Carrier(Lawful):
    """A set of values held in tensors, with a zero and a sum.

    One element of the carrier occupies the trailing ``element_shape``
    axes of a tensor; the leading axes index elements. A scalar carrier
    has ``element_shape == ()``. The zero is given in the shape, dtype
    and device a computation needs, so no computation makes it in the
    default dtype. ``join`` reduces index axes by the sum, and a
    reduction over no terms is the zero, with a derivative that is a
    number.
    """

    introduces = (OPERATIONS_CLOSED, JOIN_ITERATED, ZERO_FORM, EMPTY_JOIN)

    @property
    @abstractmethod
    def element_shape(self) -> tuple[int, ...]:
        """The trailing axes of one element.

        Returns
        -------
        tuple of int
            ``()`` for a scalar carrier.
        """
        ...

    @abstractmethod
    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The zero at every index of ``shape``.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device the zero takes.

        Returns
        -------
        torch.Tensor
            The zero with shape ``(*shape, *element_shape)``, which does
            not require a gradient.

        Raises
        ------
        TypeError
            If the dtype of ``like`` cannot hold the zero.
        """
        ...

    @abstractmethod
    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Whether each element of ``t`` lies in the carrier.

        Parameters
        ----------
        t : torch.Tensor
            Elements with trailing ``element_shape`` axes.

        Returns
        -------
        torch.Tensor
            A Boolean tensor of the index shape of ``t``.

        Raises
        ------
        TypeError
            If the entries are complex, or the carrier is not declared.
        """
        ...

    @abstractmethod
    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum ⊕ over index axes.

        Parameters
        ----------
        t : torch.Tensor
            Elements with trailing ``element_shape`` axes.
        dim : int or tuple of int
            Index axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction with the named axes removed.
        """
        ...

    def join_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The binary sum ``a ⊕ b``, with ``a`` first in declared order.

        Parameters
        ----------
        a : torch.Tensor
            The first operand.
        b : torch.Tensor
            The second operand, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            The sum, computed as `join` over a new leading axis holding
            ``a`` then ``b``.
        """
        return _pair(self.join, a, b)

    def is_zero(self, t: torch.Tensor) -> torch.Tensor:
        """Whether each element of ``t`` is exactly the zero.

        Parameters
        ----------
        t : torch.Tensor
            Elements with trailing ``element_shape`` axes.

        Returns
        -------
        torch.Tensor
            A Boolean tensor of the index shape of ``t``.
        """
        rank = len(self.element_shape)
        index = tuple(t.shape[: t.ndim - rank])
        equal = t == self.zero_like(index, t)
        for _ in range(rank):
            equal = equal.all(dim=-1)
        return equal

    def validate(self, t: torch.Tensor) -> None:
        """Require every element of ``t`` to lie in the carrier.

        Parameters
        ----------
        t : torch.Tensor
            Elements with trailing ``element_shape`` axes.

        Raises
        ------
        TypeError
            If the entries are complex, or the carrier is not declared.
        ValueError
            If an element lies outside the carrier.
        """
        validate_carrier(t, self)

    def empty_join(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """The join over normalized axes one of which has size zero.

        Parameters
        ----------
        t : torch.Tensor
            Elements with trailing ``element_shape`` axes.
        axes : tuple of int
            Normalized index axes, one of size zero.

        Returns
        -------
        torch.Tensor
            The zero of the remaining index shape in the dtype and device
            of ``t``, on the autograd graph of ``t`` so its derivative is
            zero.
        """
        rank = len(self.element_shape)
        remaining = reduced_shape(t, axes)
        index = remaining[: len(remaining) - rank]
        return empty_reduction(t, axes, self.zero_like(index, t))


class UnitalCarrier(Carrier):
    """A carrier with a product ⊗ and its unit.

    The unit, like the zero, is given in the shape, dtype and device a
    computation needs.
    """

    introduces = (UNIT_FORM,)

    @abstractmethod
    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product ⊗ of elements, broadcast over index axes.

        Parameters
        ----------
        a : torch.Tensor
            The left factor.
        b : torch.Tensor
            The right factor.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``, with ``a`` on the left.
        """
        ...

    @abstractmethod
    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The unit at every index of ``shape``.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device the unit takes.

        Returns
        -------
        torch.Tensor
            The unit with shape ``(*shape, *element_shape)``, which does
            not require a gradient.

        Raises
        ------
        TypeError
            If the dtype of ``like`` cannot hold the unit.
        """
        ...

    def identity_tensor(
        self,
        obj_shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """The identity matrix of an object of the given shape.

        Parameters
        ----------
        obj_shape : tuple of int
            The shape of the object, for instance ``(n,)`` for a finite
            set of ``n`` elements.
        dtype : torch.dtype or None
            The dtype; ``None`` means torch's default dtype, as in
            ``torch.full``.
        device : torch.device or str or None
            The device; ``None`` means torch's default device.

        Returns
        -------
        torch.Tensor
            Shape ``(*obj_shape, *obj_shape, *element_shape)``, the unit
            on the diagonal and the zero elsewhere.
        """
        like = torch.empty(0, dtype=dtype, device=device)
        full = tuple(obj_shape) + tuple(obj_shape)
        size = math.prod(obj_shape)
        diagonal = torch.eye(size, dtype=torch.bool, device=like.device).reshape(full)
        diagonal = diagonal.reshape(*full, *([1] * len(self.element_shape)))
        return torch.where(
            diagonal, self.unit_like(full, like), self.zero_like(full, like)
        )


# ============================================================================
# Composition rules
# ============================================================================


class CompositionRule(Lawful):
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
    * `OrderPreservingSemiring`: a semiring whose product need not
      commute, so composition is associative and unital but keeps the
      order of its factors.
    * `Algebra`: a scalar carrier with ``unit`` and ``zero`` (and meet
      and negation), so composition has two-sided identities.
      Associativity is not implied.
    * `Semiring`: an `Algebra` whose operations satisfy the
      commutative-semiring laws, which makes composition associative
      and unital, so finite sets and tensors form a category.
    * `IdempotentSemiring`: a `Semiring` whose ``join`` is idempotent.

    Operations that need identity check at runtime that the
    composition rule is at least an `Algebra`; a clear
    error is raised if a non-algebra rule is fed in.
    """

    @abstractmethod
    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Monoidal product ⊗ (elementwise).

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``.
        """
        ...

    @abstractmethod
    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Join ⋁, the reduction of composition.

        Parameters
        ----------
        t : torch.Tensor
            Input tensor.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction with the named axes removed.
        """
        ...

    def join_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The binary join ``a ⋁ b``, with ``a`` first in declared order.

        Parameters
        ----------
        a : torch.Tensor
            The first operand.
        b : torch.Tensor
            The second operand, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            The join over a new leading axis holding ``a`` then ``b``.
        """
        return _pair(self.join, a, b)

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """V-enriched composition of scalar-valued tensors.

        Computes ``result[d..., c...] = ⋁_{s...} m[d..., s...] ⊗ n[s..., c...]``.

        Parameters
        ----------
        m : torch.Tensor
            Left tensor of shape ``(*domain, *shared)``.
        n : torch.Tensor
            Right tensor of shape ``(*shared, *codomain)``.
        n_contract : int
            Number of shared axes to contract.

        Returns
        -------
        torch.Tensor
            Composed tensor of shape ``(*domain, *codomain)``.

        Raises
        ------
        ValueError
            If ``n_contract`` is below one or the shared axes differ.
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
        """Whether two rules compose: the same type, or the same name.

        Parameters
        ----------
        other : CompositionRule
            The other rule.

        Returns
        -------
        bool
            True when morphisms over the two rules may compose; a
            custom instance built independently matches by name.
        """
        if type(self) is type(other):
            return True
        return self.name == other.name

    def __repr__(self) -> str:
        """The class name with empty parentheses.

        Returns
        -------
        str
            The representation.
        """
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

    introduces = (COMPOSITION_ASSOCIATIVE,)


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


class OrderPreservingSemiring(Semigroupoid, UnitalCarrier):
    """A semiring whose product need not commute.

    ``(L, ⊕, 0)`` is a commutative monoid, ``(L, ⊗, 1)`` a monoid,
    ``⊗`` distributes over ``⊕`` on both sides and ``0`` annihilates
    ``⊗`` [Golan (1999)]. Matrices over it compose
    associatively and unitally, but a composite keeps the order of its
    factors: a plan over such weights may reorder sums and may not
    exchange factors. Elements may have shape, as the square matrices
    of `MatrixSemiring` do; composition contracts index axes and keeps
    each element's trailing axes.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (
        PLUS_ASSOCIATIVE,
        PLUS_COMMUTATIVE,
        PLUS_UNIT,
        TIMES_ASSOCIATIVE,
        TIMES_UNIT,
        DISTRIBUTIVE,
        ZERO_ANNIHILATES,
    )

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """Composition of element-valued tensors, keeping factor order.

        Computes ``result[d..., c...] = ⊕_{s...} m[d..., s...] ⊗ n[s..., c...]``
        with ``m``'s element on the left of every product.

        Parameters
        ----------
        m : torch.Tensor
            Shape ``(*domain, *shared, *element_shape)``.
        n : torch.Tensor
            Shape ``(*shared, *codomain, *element_shape)``.
        n_contract : int
            Number of shared index axes to contract.

        Returns
        -------
        torch.Tensor
            Shape ``(*domain, *codomain, *element_shape)``.

        Raises
        ------
        ValueError
            If ``n_contract`` is below one, the shared axes differ, or an
            operand lacks the element axes.
        """
        rank = len(self.element_shape)
        m_index = m.ndim - rank
        n_index = n.ndim - rank
        if n_contract < 1:
            raise ValueError(f"n_contract must be >= 1, got {n_contract}")
        if (
            m_index < n_contract
            or n_index < n_contract
            or tuple(m.shape[m_index:]) != self.element_shape
            or tuple(n.shape[n_index:]) != self.element_shape
        ):
            raise ValueError(
                f"operands of shapes {tuple(m.shape)} and {tuple(n.shape)} do "
                f"not hold {n_contract} shared axes of elements of shape "
                f"{self.element_shape}"
            )
        shared_m = m.shape[m_index - n_contract : m_index]
        shared_n = n.shape[:n_contract]
        if shared_m != shared_n:
            raise ValueError(
                f"shared dimensions do not match: "
                f"m trailing {shared_m} != n leading {shared_n}"
            )
        n_domain = m_index - n_contract
        n_codomain = n_index - n_contract
        m_expanded = m.reshape(
            *m.shape[:m_index], *([1] * n_codomain), *self.element_shape
        )
        n_expanded = n.reshape(*([1] * n_domain), *n.shape)
        product = self.tensor_op(m_expanded, n_expanded)
        return self.join(product, tuple(range(n_domain, n_domain + n_contract)))


class Algebra(CompositionRule, UnitalCarrier):
    """Abstract commutative algebra for V-enriched composition.

    An algebra supplies the operations ``(⊗, ⋁, ⋀, ¬)`` and the
    elements ``I`` (``unit``) and ``⊥`` (``zero``) on a scalar carrier.
    Every shipped algebra satisfies ``I ⊗ a = a``, ``⊥ ⊗ a = ⊥`` and
    ``⋁(a, ⊥) = a``, so the identity tensor is a two-sided identity for
    composition. Associativity of composition is a further property,
    held exactly by the `Semiring` subclasses; an `Algebra` that is not
    a `Semiring` (``PRODUCT_FUZZY``, ``LUKASIEWICZ``, ``PROBABILITY`` and
    their duals) composes with identities but not associatively.

    ``unit`` and ``zero`` are the scalar constants; `unit_like` and
    `zero_like` give them in a requested shape, dtype and device.
    Subclasses implement the primitive operations and `contains`.
    """

    @property
    def element_shape(self) -> tuple[int, ...]:
        """The empty shape: one element is one entry.

        Returns
        -------
        tuple of int
            ``()``.
        """
        return ()

    @abstractmethod
    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Meet ⋀, the reduction for universal quantification (∀).

        Parameters
        ----------
        t : torch.Tensor
            Input tensor with values in L.
        dim : int or tuple of int
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
        """Unit element I of the monoidal product ⊗.

        Returns
        -------
        float
            The unit.
        """
        ...

    @property
    @abstractmethod
    def zero(self) -> float:
        """Bottom element ⊥, the identity of ⋁.

        Returns
        -------
        float
            The zero.
        """
        ...

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The zero in a requested shape, dtype and device.

        Parameters
        ----------
        shape : tuple of int
            The shape.
        like : torch.Tensor
            The tensor whose dtype and device the zero takes.

        Returns
        -------
        torch.Tensor
            A constant tensor holding ``zero``.

        Raises
        ------
        TypeError
            If the dtype cannot hold the zero.
        """
        return constant(self.zero, tuple(shape), like)

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The unit in a requested shape, dtype and device.

        Parameters
        ----------
        shape : tuple of int
            The shape.
        like : torch.Tensor
            The tensor whose dtype and device the unit takes.

        Returns
        -------
        torch.Tensor
            A constant tensor holding ``unit``.

        Raises
        ------
        TypeError
            If the dtype cannot hold the unit.
        """
        return constant(self.unit, tuple(shape), like)

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


class Semiring(Algebra, OrderPreservingSemiring):
    """An algebra whose operations form a commutative semiring.

    A commutative semiring ``(L, ⊕, ⊗, 0, 1)`` has ``(L, ⊕, 0)`` and
    ``(L, ⊗, 1)`` commutative monoids, ``⊗`` distributing over ``⊕``
    on both sides, and ``0`` annihilating ``⊗`` [Golan (1999)]. Here
    ``⊕`` is the binary form of ``join``, ``0`` is ``zero`` and ``1`` is
    ``unit``. These laws make the composition
    ``(f >> g)[i, k] = ⊕_j f[i, j] ⊗ g[j, k]`` associative and unital,
    so finite sets with tensors over ``L`` form a category (the
    category of ``L``-valued matrices). Membership in this class is a
    claim that the laws hold; `claims` reports the status of each, and
    the law grid of the evidence harness checks every shipped instance.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (TIMES_COMMUTATIVE,)


class IdempotentSemiring(Semiring):
    """A commutative semiring whose addition is idempotent.

    Additive idempotence, ``a ⊕ a = a`` for every ``a``, makes ``⊕`` the
    join of the partial order ``a ≤ b ⟺ a ⊕ b = b`` [Golan (1999)].
    On finite carriers of composition this is the structure that a
    quantale restricts to, and it is what the fuzzy powerset monad
    needs (see `quivers.monadic.monads.FuzzyPowersetMonad`). In the
    shipped instances ``meet`` is the infimum of that order and `top`
    is its greatest element, so each is a bounded distributive lattice
    under ``(join, meet)``.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (PLUS_IDEMPOTENT,)

    @property
    @abstractmethod
    def top(self) -> float:
        """The greatest element of the order ``a ≤ b ⟺ a ⊕ b = b``.

        It absorbs the join (``a ⊕ ⊤ = ⊤``) and is the meet of the empty
        family.

        Returns
        -------
        float
            The top.
        """
        ...

    @abstractmethod
    def residuum(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The residuum ``[w, x]``, the greatest ``v`` with ``v ⊗ w ≤ x``.

        The order is ``a ≤ b ⟺ a ⊕ b = b``. Residuation,
        ``v ⊗ w ≤ x ⟺ v ≤ [w, x]``, makes ``[w, -]`` right adjoint to
        ``- ⊗ w``; it is the internal hom that weighted limits and
        the Yoneda lemma use.

        Parameters
        ----------
        w : torch.Tensor
            The left argument, broadcastable with ``x``.
        x : torch.Tensor
            The right argument.

        Returns
        -------
        torch.Tensor
            ``[w, x]`` elementwise.
        """
        ...


# ============================================================================
# Capabilities
# ============================================================================


class Selection(ValueModel):
    """The result of a selective reduction and the operand it selected.

    Parameters
    ----------
    value : torch.Tensor
        The selected operand at every remaining index.
    witness : torch.Tensor
        The flat position of the selected operand in row-major order of
        the reduced axes, as ``int64``.
    """

    value: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    witness: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def _select(
    t: torch.Tensor, axes: tuple[int, ...], key: torch.Tensor, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """The operand with the largest (or smallest) key, the first one on ties.

    Parameters
    ----------
    t : torch.Tensor
        The operands.
    axes : tuple of int
        Normalized, nonempty axes, none of size zero.
    key : torch.Tensor
        A key of the shape of ``t``.
    largest : bool
        Whether the largest key is selected, else the smallest.

    Returns
    -------
    tuple of torch.Tensor
        The selected operands and their flat positions in row-major order
        of ``axes``. The derivative reaches the selected operand only.
    """
    kept = t.ndim - len(axes)
    terms = math.prod(t.shape[axis] for axis in axes)
    moved = t.movedim(axes, tuple(range(kept, t.ndim)))
    flat = moved.reshape(*moved.shape[:kept], terms)
    ranked = key.movedim(axes, tuple(range(kept, t.ndim))).reshape(flat.shape)
    if ranked.dtype == torch.bool:
        ranked = ranked.to(torch.uint8)
    witness = (ranked.argmax if largest else ranked.argmin)(dim=-1, keepdim=True)
    return flat.gather(-1, witness).squeeze(-1), witness.squeeze(-1)


class Selective(Carrier):
    """A carrier whose sum returns one of its operands.

    ``a ⊕ b`` is ``a`` or ``b``: the sum selects the operand that is
    best under the order ``rank`` defines. The tie rule is one for every
    operation: the first best operand, in row-major order of the reduced
    axes (``a`` before ``b`` in `join_pair`), is the value's witness and
    receives the whole derivative. Splitting the derivative over tied
    operands is another element of the subdifferential; using it in
    one operation and not in another would make a derivative depend on
    bracketing.
    """

    introduces = (PLUS_SELECTIVE, TIE_FIRST_MAXIMIZER)
    tie_rule: ClassVar[TieRule] = FIRST_MAXIMIZER

    @abstractmethod
    def rank(self, t: torch.Tensor) -> torch.Tensor:
        """A key whose largest value marks the best operand.

        Parameters
        ----------
        t : torch.Tensor
            Elements of the carrier.

        Returns
        -------
        torch.Tensor
            A key of the same shape, strictly increasing in the order the
            sum selects by.
        """
        ...

    def select(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> Selection:
        """The best operand over index axes and its witness.

        Parameters
        ----------
        t : torch.Tensor
            Elements of the carrier.
        dim : int or tuple of int
            Index axes to reduce.

        Returns
        -------
        Selection
            The best operand and the flat position of the first one.

        Raises
        ------
        ValueError
            If a reduced axis has size zero, since nothing is selected,
            or the carrier is not scalar.
        """
        if self.element_shape:
            raise ValueError("selection is defined on scalar carriers")
        axes = reduction_axes(t, dim)
        if is_empty(t, axes):
            raise ValueError("a reduction over no terms selects no operand")
        value, witness = _select(t, axes, self.rank(t))
        return Selection(value=value, witness=witness)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The selective sum over index axes.

        Parameters
        ----------
        t : torch.Tensor
            Elements of the carrier.
        dim : int or tuple of int
            Index axes to reduce.

        Returns
        -------
        torch.Tensor
            The first best operand at every remaining index, or the zero
            for a reduction over no terms.
        """
        axes = reduction_axes(t, dim, len(self.element_shape))
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        return self.select(t, axes).value


class ZeroSumFree(Carrier):
    """A carrier in which a sum is zero only when every term is.

    With `NoZeroDivisors` this makes the support map onto the Boolean
    semiring a homomorphism: a sum is nonzero exactly when some term is
    [Golan (1999), "zerosumfree"].

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (ZERO_SUM_FREE,)


class NoZeroDivisors(UnitalCarrier):
    """A carrier in which a product is zero only when a factor is.

    Golan (1999) calls such a semiring entire. With `ZeroSumFree` it
    makes the support map a homomorphism.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (NO_ZERO_DIVISORS,)


class Semifield(NoZeroDivisors):
    """A carrier in which every element other than zero has an inverse.

    ``divide(a, b)`` is ``a ⊗ b⁻¹`` for ``b`` other than the zero, so
    ``(a ⊗ b) ⊘ b = a``. A semifield has no zero divisors: if
    ``a ⊗ b = 0`` and ``b`` is not zero then ``a = (a ⊗ b) ⊘ b = 0``.
    Executors that eliminate by division (directed spanning trees,
    weight pushing) ask for this capability.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (DIVISION,)

    @abstractmethod
    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The quotient ``a ⊘ b``.

        Parameters
        ----------
        a : torch.Tensor
            The dividend.
        b : torch.Tensor
            The divisor, broadcastable with ``a``, with no zero element.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b⁻¹``.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is the zero.
        """
        ...

    def require_nonzero(self, b: torch.Tensor) -> None:
        """Refuse a divisor holding the zero.

        Parameters
        ----------
        b : torch.Tensor
            The divisor.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is the zero.
        """
        if bool(self.is_zero(b).any()):
            raise ZeroDivisionError(f"division by the zero of {self.name}")


class Ring(Carrier):
    """A carrier whose sum has inverses.

    ``subtract(a, b)`` is ``a ⊕ (−b)``, where ``−b`` is the additive
    inverse, so ``a ⊕ (0 ⊖ a) = 0``. Algorithms that cancel terms, such
    as a determinant by elimination, need this capability; it gives up
    the relative accuracy of sums of nonnegative terms.
    """

    introduces = (ADDITIVE_INVERSE,)

    @abstractmethod
    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The difference ``a ⊖ b``.

        Parameters
        ----------
        a : torch.Tensor
            The minuend.
        b : torch.Tensor
            The subtrahend, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            ``a ⊕ (−b)``.
        """
        ...


class Semimodule(Carrier):
    """A commutative monoid acted on by an order-preserving semiring.

    A left semimodule over ``K`` is a commutative monoid ``(M, ⊕, 0)``
    with an action ``K × M → M`` that is associative, unital, and
    distributes over both sums, and in which the zero of either side
    annihilates [Golan (1999)]. Expectation carriers, whose
    second component is acted on by probabilities, are built from this
    capability; `FreeSemimodule` is the module of vectors over ``K``.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (PLUS_ASSOCIATIVE, PLUS_COMMUTATIVE, PLUS_UNIT, SEMIMODULE_ACTION)

    @property
    @abstractmethod
    def scalars(self) -> OrderPreservingSemiring:
        """The semiring that acts.

        Returns
        -------
        OrderPreservingSemiring
            The scalars.
        """
        ...

    @abstractmethod
    def act(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """The action ``k · m``.

        Parameters
        ----------
        k : torch.Tensor
            Scalars, with the scalars' element axes trailing.
        m : torch.Tensor
            Module elements with the same index shape (or one that
            broadcasts with it).

        Returns
        -------
        torch.Tensor
            ``k · m``.
        """
        ...


class DivergentClosureError(ValueError):
    """A path closure has no value in the declared carrier."""


class UndecidedClosureError(ValueError):
    """Rounding leaves open whether a path closure converges.

    The weight of a cycle lies within the bound on rounding of the value
    at which its series stops converging, so the computed numbers cannot
    tell a convergent closure from a divergent one. No value is returned.
    """


class Star(UnitalCarrier):
    """A carrier whose square matrices have a closure.

    ``star(A)`` is the least solution ``X`` of ``X = I ⊕ A X`` on the
    declared domain, the sum ``I ⊕ A ⊕ A² ⊕ ...`` of the weights of all
    paths, including the empty one [Lehmann (1977)]. Outside the domain
    the series has no value and ``star`` raises `DivergentClosureError`;
    when rounding cannot decide which side of the boundary a cycle lies
    on it raises `UndecidedClosureError`. The shipped closures eliminate
    one state after another, so they cost ``n³`` operations of the
    algebra and keep every entry on the autograd graph.

    References
    ----------
    Lehmann, D. J. (1977). Algebraic structures for transitive closure.
    *Theoretical Computer Science*, 4(1), 59-76.
    [doi:10.1016/0304-3975(77)90056-1](https://doi.org/10.1016/0304-3975(77)90056-1)
    """

    introduces = (STAR_LEAST_SOLUTION, STAR_DECIDED)

    @property
    @abstractmethod
    def star_domain(self) -> str:
        """The matrices whose closure has a value.

        Returns
        -------
        str
            The domain in words.
        """
        ...

    @abstractmethod
    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The closure ``A* = I ⊕ A ⊕ A² ⊕ ...`` of a square matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n, *element_shape)``.

        Returns
        -------
        torch.Tensor
            The closure, of the same shape.

        Raises
        ------
        DivergentClosureError
            If the matrix lies outside `star_domain`.
        UndecidedClosureError
            If rounding cannot decide whether it lies inside.
        """
        ...


# ============================================================================
# Reductions shared by the concrete algebras
# ============================================================================


def _extremum(
    t: torch.Tensor,
    dim: int | tuple[int, ...],
    *,
    largest: bool,
    empty: torch.Tensor | None,
    carrier: str,
) -> torch.Tensor:
    """The largest or smallest operand, the first one on ties.

    Parameters
    ----------
    t : torch.Tensor
        The operands.
    dim : int or tuple of int
        Axes to reduce.
    largest : bool
        Whether the largest operand is taken.
    empty : torch.Tensor or None
        The identity of the reduction, of the result's shape, or
        ``None`` when the carrier holds none.
    carrier : str
        The carrier's name, for the refusal.

    Returns
    -------
    torch.Tensor
        The reduction with the axes removed.

    Raises
    ------
    ValueError
        If a reduced axis has size zero and the carrier holds no
        identity of the reduction.
    """
    axes = reduction_axes(t, dim)
    if is_empty(t, axes):
        if empty is None:
            raise ValueError(
                f"the reduction of no terms has no value in the {carrier} carrier"
            )
        return empty_reduction(t, axes, empty)
    return _select(t, axes, t, largest)[0]


def _sum(t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """The ordinary sum over axes.

    Parameters
    ----------
    t : torch.Tensor
        The terms.
    dim : int or tuple of int
        Axes to reduce.

    Returns
    -------
    torch.Tensor
        The sum, which is zero over no terms.
    """
    return t.sum(dim=reduction_axes(t, dim))


def _product(t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """The ordinary product over axes.

    Parameters
    ----------
    t : torch.Tensor
        The factors.
    dim : int or tuple of int
        Axes to reduce.

    Returns
    -------
    torch.Tensor
        The product, which is one over no factors.
    """
    result = t
    for axis in reversed(reduction_axes(t, dim)):
        result = result.prod(dim=axis)
    return result


def _fold_along_dim(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    axes: tuple[int, ...],
) -> torch.Tensor:
    """Fold a binary operation across nonempty axes, left to right.

    Parameters
    ----------
    op : Callable
        The binary operation.
    t : torch.Tensor
        The operands.
    axes : tuple of int
        Normalized axes, none of size zero.

    Returns
    -------
    torch.Tensor
        The fold with the axes removed.
    """
    for axis in reversed(axes):
        chunks = torch.unbind(t, dim=axis)
        acc = chunks[0]
        for chunk in chunks[1:]:
            acc = op(acc, chunk)
        t = acc
    return t


def _filled(t: torch.Tensor, dim: int | tuple[int, ...], value: float) -> torch.Tensor:
    """A constant of the shape a reduction of ``t`` leaves.

    Parameters
    ----------
    t : torch.Tensor
        The operands.
    dim : int or tuple of int
        Axes to reduce.
    value : float
        The constant.

    Returns
    -------
    torch.Tensor
        The constant in the dtype and device of ``t``.
    """
    return constant(value, reduced_shape(t, reduction_axes(t, dim)), t)


def _require_carrier(t: torch.Tensor) -> torch.Tensor:
    """Refuse complex entries and return the tensor.

    Parameters
    ----------
    t : torch.Tensor
        The entries.

    Returns
    -------
    torch.Tensor
        ``t`` itself.

    Raises
    ------
    TypeError
        If the entries are complex.
    """
    require_real(t)
    return t


def _unit_interval(t: torch.Tensor) -> torch.Tensor:
    """Membership in ``[0, 1]``.

    Parameters
    ----------
    t : torch.Tensor
        The entries.

    Returns
    -------
    torch.Tensor
        Whether each entry lies in ``[0, 1]``.

    Raises
    ------
    TypeError
        If the entries are complex.
    """
    t = _require_carrier(t)
    return (t >= 0) & (t <= 1)


def _extended_below(t: torch.Tensor) -> torch.Tensor:
    """Membership in ``ℝ ∪ {-∞}``.

    Parameters
    ----------
    t : torch.Tensor
        The entries.

    Returns
    -------
    torch.Tensor
        Whether each entry is finite or minus infinity.

    Raises
    ------
    TypeError
        If the entries are complex.
    """
    t = _require_carrier(t)
    return torch.isfinite(t) | torch.isneginf(t)


def _nonnegative(t: torch.Tensor) -> torch.Tensor:
    """Membership in ``[0, ∞)``.

    Parameters
    ----------
    t : torch.Tensor
        The entries.

    Returns
    -------
    torch.Tensor
        Whether each entry is finite and nonnegative.

    Raises
    ------
    TypeError
        If the entries are complex.
    """
    t = _require_carrier(t)
    return torch.isfinite(t) & (t >= 0)


_GOLAN = Citation(
    reference="Golan, J. S. (1999). Semirings and their Applications. Springer",
    locator=(
        "the definitions of a semiring, of zerosumfree and entire semirings, and "
        "of a semifield"
    ),
    doi="10.1007/978-94-015-9333-5",
)

_PROOFS = "semantics/algebras.md"


def _proved(anchor: str) -> Proved:
    """The argument for a shipped class, written in the semantics pages.

    Parameters
    ----------
    anchor : str
        The anchor of the argument, after ``proof-``.

    Returns
    -------
    Proved
        The evidence.
    """
    return Proved(argument=f"{_PROOFS}#proof-{anchor}", sources=(_GOLAN,))


def _tests(
    *, unital: bool, tie: int | None = None, star: int | None = None
) -> Mapping[str, int]:
    """The cases the law grid evaluates for the laws about the implementation.

    Parameters
    ----------
    unital : bool
        Whether the class has a unit.
    tie : int or None
        The cases of the tie rule, for a selective class.
    star : int or None
        The cases of the closure's refusals, for a class with a closure.

    Returns
    -------
    Mapping of str to int
        Cases by law name.
    """
    cases = {ZERO_FORM.name: 20, EMPTY_JOIN.name: 12}
    if unital:
        cases[UNIT_FORM.name] = 20
    if tie is not None:
        cases[TIE_FIRST_MAXIMIZER.name] = tie
    if star is not None:
        cases[STAR_DECIDED.name] = star
    return MappingProxyType(cases)


_UNITAL_TESTS = _tests(unital=True)


_LEHMANN = Citation(
    reference=(
        "Lehmann, D. J. (1977). Algebraic structures for transitive closure. "
        "Theoretical Computer Science 4(1)"
    ),
    locator="the closure of a matrix by elimination of one state after another",
    doi="10.1016/0304-3975(77)90056-1",
)


def _proved_with_star(anchor: str) -> Proved:
    """The argument for a shipped class with a closure.

    Parameters
    ----------
    anchor : str
        The anchor of the argument, after ``proof-``.

    Returns
    -------
    Proved
        The evidence, citing the elimination of states as well.
    """
    return Proved(argument=f"{_PROOFS}#proof-{anchor}", sources=(_GOLAN, _LEHMANN))


def _square(matrix: torch.Tensor) -> int:
    """The order of a square matrix of scalars.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix.

    Returns
    -------
    int
        Its number of rows.

    Raises
    ------
    ValueError
        If the tensor is not a square matrix.
    """
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"a closure needs a square matrix; got shape {tuple(matrix.shape)}"
        )
    return int(matrix.shape[0])


def _rounding(dtype: torch.dtype, operations: int) -> float:
    """Higham's bound ``m u / (1 - m u)`` on the relative error of ``m`` roundings.

    Parameters
    ----------
    dtype : torch.dtype
        The floating dtype.
    operations : int
        The number of roundings.

    Returns
    -------
    float
        The bound.

    Raises
    ------
    UndecidedClosureError
        If the dtype has too few digits for the number of roundings.
    """
    roundoff = torch.finfo(dtype).eps / 2
    product = operations * roundoff
    if product >= 1:
        raise UndecidedClosureError(
            f"{operations} roundings in {dtype} leave no bound on the error"
        )
    return product / (1 - product)


def _largest_finite(matrix: torch.Tensor) -> float:
    """The largest magnitude among the finite entries of a matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix.

    Returns
    -------
    float
        The magnitude; ``0`` when no entry is finite.
    """
    finite = matrix.detach()[torch.isfinite(matrix.detach())]
    return float(finite.abs().max()) if finite.numel() else 0.0


def _eliminate(
    algebra: UnitalCarrier,
    matrix: torch.Tensor,
    repeat: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    unit_diagonal: bool = False,
) -> torch.Tensor:
    """The closure of a matrix by eliminating one state after another.

    At state ``k`` the loop weight ``a_kk`` gives the weight of going
    round it any number of times, ``r = a_kk*``; every entry adds the
    paths through ``k``, ``a_ij ⊕ a_ik ⊗ r ⊗ a_kj``, the row and column of
    ``k`` take ``r`` on their side, and ``a_kk`` becomes ``r``. After the
    last state the matrix is ``A*`` [Lehmann (1977)]. The work is ``n``
    rounds of ``n²`` products and sums.

    Parameters
    ----------
    algebra : UnitalCarrier
        The scalar algebra of the entries.
    matrix : torch.Tensor
        Shape ``(n, n)``.
    repeat : Callable
        The closure of one loop weight and the number of states eliminated
        before it, which bounds the roundings it carries; it raises
        outside the domain.
    unit_diagonal : bool
        Whether no cycle is better than the unit, as in an idempotent
        semiring whose loops all close to the unit. Then ``A*_ii`` is the
        unit exactly, and the diagonal is set to it rather than to a
        rounded cycle weight.

    Returns
    -------
    torch.Tensor
        ``A*``.
    """
    n = _square(matrix)
    star = matrix
    indices = torch.arange(n, device=matrix.device)
    for k in range(n):
        loop_star = repeat(star[k, k], k)
        right = algebra.tensor_op(loop_star, star[None, k, :])
        left = algebra.tensor_op(star[:, k, None], loop_star)
        through = algebra.tensor_op(star[:, k, None], right)
        combined = algebra.join_pair(star, through)
        row = indices[:, None] == k
        column = indices[None, :] == k
        star = torch.where(
            row & column,
            loop_star,
            torch.where(row, right, torch.where(column, left, combined)),
        )
    if unit_diagonal and n:
        diagonal = torch.eye(n, dtype=torch.bool, device=matrix.device)
        star = torch.where(diagonal, algebra.unit_like((n, n), star), star)
    return star


def _positive_cycle(matrix: torch.Tensor) -> bool:
    """Whether some cycle has positive score, in exact rational arithmetic.

    Floyd and Warshall's best paths over the exact rational values of the
    floating point weights; a positive cycle exists exactly when some
    state reaches itself with positive score.

    Parameters
    ----------
    matrix : torch.Tensor
        Shape ``(n, n)`` scores, ``-∞`` for absent edges.

    Returns
    -------
    bool
        Whether a cycle of positive exact score exists.
    """
    rows = matrix.detach().cpu().tolist()
    best: list[list[Fraction | None]] = [
        [None if value == -math.inf else Fraction(value) for value in row]
        for row in rows
    ]
    n = len(best)
    for k in range(n):
        for i in range(n):
            left = best[i][k]
            if left is None:
                continue
            for j in range(n):
                right = best[k][j]
                if right is None:
                    continue
                candidate = left + right
                current = best[i][j]
                if current is None or candidate > current:
                    best[i][j] = candidate
    return any((best[i][i] or 0) > 0 for i in range(n))


def _decide(
    loop: torch.Tensor,
    step: int,
    divergent: torch.Tensor,
    convergent: torch.Tensor,
    verdicts: Callable[[], list[str]],
    message: str,
) -> None:
    """Decide whether a loop weight lets the closure converge.

    ``divergent`` and ``convergent`` are the cheap tests against an a
    priori bound on rounding. When neither holds, ``verdicts`` decides
    with a finer procedure (exact arithmetic or an enclosure), unless the
    loop differs between the positions of an evaluation at many, where
    only the cheap tests run.

    Parameters
    ----------
    loop : torch.Tensor
        The loop weight.
    step : int
        The states eliminated before it.
    divergent : torch.Tensor
        Whether the loop surely makes the series diverge.
    convergent : torch.Tensor
        Whether it surely lets the series converge.
    verdicts : Callable
        The finer decision at every step: ``"convergent"``,
        ``"divergent"`` or ``"undecided"``.
    message : str
        What diverges, for the diagnostics.

    Raises
    ------
    DivergentClosureError
        If the loop makes the series diverge.
    UndecidedClosureError
        If no procedure decides.
    """
    diverges = lambda: DivergentClosureError(  # noqa: E731
        f"the closure diverges: {message}, so the series I + A + A² + ... has no value"
    )
    undecided = lambda: UndecidedClosureError(  # noqa: E731
        "a cycle weight is within the rounding bound of the boundary, so the "
        "computed numbers cannot decide whether the closure converges"
    )
    if depends_on_position(loop):
        demand(~divergent, diverges)
        demand(convergent, undecided)
        return
    if bool(divergent):
        raise diverges()
    if bool(convergent):
        return
    verdict = verdicts()
    found = verdict[step] if step < len(verdict) else "undecided"
    if found == "divergent":
        raise diverges()
    if found != "convergent":
        raise undecided()


def _cached(compute: Callable[[], list[str]]) -> Callable[[], list[str]]:
    """Compute a list of verdicts once, when it is first asked for.

    Parameters
    ----------
    compute : Callable
        The computation.

    Returns
    -------
    Callable
        The cached computation.
    """
    kept: list[list[str]] = []

    def verdicts() -> list[str]:
        """The verdicts.

        Returns
        -------
        list of str
            One per eliminated state.
        """
        if not kept:
            kept.append(compute())
        return kept[0]

    return verdicts


def _linear_verdicts(matrix: torch.Tensor) -> list[str]:
    """Decide every loop of the elimination of a nonnegative matrix exactly.

    The elimination of `_linear_star` is repeated in rational arithmetic
    on the exact values of the floating point entries.

    Parameters
    ----------
    matrix : torch.Tensor
        Shape ``(n, n)``.

    Returns
    -------
    list of str
        ``"convergent"`` for each state until one whose loop weight is at
        least one, which is ``"divergent"``.
    """
    star: list[list[Fraction]] = [
        [Fraction(value) for value in row] for row in matrix.detach().cpu().tolist()
    ]
    n = len(star)
    verdicts: list[str] = []
    for k in range(n):
        loop = star[k][k]
        if loop >= 1:
            verdicts.append("divergent")
            return verdicts
        verdicts.append("convergent")
        repeat = 1 / (1 - loop)
        new = [row[:] for row in star]
        for i in range(n):
            for j in range(n):
                if i == k and j == k:
                    new[i][j] = repeat
                elif i == k:
                    new[i][j] = repeat * star[k][j]
                elif j == k:
                    new[i][j] = star[i][k] * repeat
                else:
                    new[i][j] = star[i][j] + star[i][k] * repeat * star[k][j]
        star = new
    return verdicts


def _step(t: torch.Tensor, steps: int, toward: float) -> torch.Tensor:
    """Move every entry by units in the last place.

    Parameters
    ----------
    t : torch.Tensor
        The entries.
    steps : int
        How many units.
    toward : float
        ``-inf`` to move down, ``inf`` to move up.

    Returns
    -------
    torch.Tensor
        The moved entries.
    """
    target = torch.full_like(t, toward)
    for _ in range(steps):
        t = torch.nextafter(t, target)
    return t


def _interval_sum(
    a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """An enclosure of ``a + b``, widened only where the sum is inexact.

    An error-free transformation (Knuth's two-sum) tells an exact sum
    from a rounded one.

    Parameters
    ----------
    a : tuple of torch.Tensor
        Lower and upper bounds.
    b : tuple of torch.Tensor
        Lower and upper bounds.

    Returns
    -------
    tuple of torch.Tensor
        Lower and upper bounds of the sum.
    """

    def exact(x: torch.Tensor, y: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        """Whether a rounded sum is exact.

        Parameters
        ----------
        x : torch.Tensor
            A term.
        y : torch.Tensor
            A term.
        total : torch.Tensor
            Their rounded sum.

        Returns
        -------
        torch.Tensor
            Whether the sum is exact.
        """
        finite = torch.isfinite(x) & torch.isfinite(y)
        back = torch.where(finite, total - x, torch.zeros_like(total))
        error = torch.where(
            finite, (x - (total - back)) + (y - back), torch.zeros_like(total)
        )
        return ~finite | (error == 0)

    low = a[0] + b[0]
    high = a[1] + b[1]
    return (
        torch.where(exact(a[0], b[0], low), low, _step(low, 1, -math.inf)),
        torch.where(exact(a[1], b[1], high), high, _step(high, 1, math.inf)),
    )


def _interval_logaddexp(
    a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """An enclosure of ``log(e^a + e^b)``, which is monotone in both terms.

    The library function is taken to err by at most two units in the
    last place, and to be exact when a term is ``-∞``.

    Parameters
    ----------
    a : tuple of torch.Tensor
        Lower and upper bounds.
    b : tuple of torch.Tensor
        Lower and upper bounds.

    Returns
    -------
    tuple of torch.Tensor
        Lower and upper bounds of the log-sum.
    """
    low = torch.logaddexp(a[0], b[0])
    high = torch.logaddexp(a[1], b[1])
    exact_low = torch.isneginf(a[0]) | torch.isneginf(b[0])
    exact_high = torch.isneginf(a[1]) | torch.isneginf(b[1])
    return (
        torch.where(exact_low, low, _step(low, 2, -math.inf)),
        torch.where(exact_high, high, _step(high, 2, math.inf)),
    )


def _log_verdicts(matrix: torch.Tensor) -> list[str]:
    """Decide every loop of the log elimination with outward rounded enclosures.

    The elimination of `LogProbAlgebra.star` is repeated on intervals that
    contain the exact values: sums are widened by one unit where two-sum
    shows rounding, and each library function by two units unless its
    result is exact. A loop enclosure below zero lets the closure
    converge, one at or above zero makes it diverge, and one that
    contains zero leaves it undecided.

    Parameters
    ----------
    matrix : torch.Tensor
        Shape ``(n, n)`` log weights.

    Returns
    -------
    list of str
        One verdict per state, up to the first that is not convergent.
    """
    base = matrix.detach().to(torch.float64)
    low, high = base, base
    n = base.shape[0]
    indices = torch.arange(n, device=base.device)
    verdicts: list[str] = []
    for k in range(n):
        loop_low, loop_high = low[k, k], high[k, k]
        if float(loop_low) >= 0:
            verdicts.append("divergent")
            return verdicts
        if not float(loop_high) < 0:
            verdicts.append("undecided")
            return verdicts
        verdicts.append("convergent")
        # r(x) = -log(-expm1(x)) increases on (-inf, 0).
        if torch.isneginf(loop_low):
            repeat_low = torch.zeros_like(loop_low)
        else:
            repeat_low = _step(
                -torch.log(_step(-torch.expm1(loop_low), 2, math.inf)), 2, -math.inf
            )
        lower_mass = _step(-torch.expm1(loop_high), 2, -math.inf)
        if torch.isneginf(loop_high):
            repeat_high = torch.zeros_like(loop_high)
        elif float(lower_mass) > 0:
            repeat_high = _step(-torch.log(lower_mass), 2, math.inf)
        else:
            repeat_high = torch.full_like(loop_high, math.inf)
        repeat = (repeat_low, repeat_high)
        column = (low[:, k, None], high[:, k, None])
        row = (low[None, k, :], high[None, k, :])
        right = _interval_sum(repeat, row)
        left = _interval_sum(column, repeat)
        through = _interval_sum(column, right)
        combined = _interval_logaddexp((low, high), through)
        is_row = indices[:, None] == k
        is_column = indices[None, :] == k
        bounds = []
        for side in range(2):
            bounds.append(
                torch.where(
                    is_row & is_column,
                    repeat[side],
                    torch.where(
                        is_row,
                        right[side],
                        torch.where(is_column, left[side], combined[side]),
                    ),
                )
            )
        low, high = bounds[0], bounds[1]
    return verdicts


def _linear_star(
    algebra: UnitalCarrier, matrix: torch.Tensor, roundings: int
) -> torch.Tensor:
    """The closure ``(I - A)⁻¹`` of a nonnegative matrix, without subtraction.

    Elimination computes ``r = 1 / (1 - a_kk)`` and otherwise only sums
    and products of nonnegative numbers, so every entry is accurate to
    a small relative error and an entry that no path reaches is an exact
    zero whose derivative is kept. The closure exists exactly when every
    loop weight met in the elimination is below one (the pivots of the
    M-matrix ``I - A`` are positive). A loop weight within the relative
    rounding bound of one refuses as undecided.

    Parameters
    ----------
    algebra : UnitalCarrier
        The nonnegative algebra of the entries.
    matrix : torch.Tensor
        Shape ``(n, n)``.
    roundings : int
        Roundings each eliminated state adds to a loop weight; zero when
        the arithmetic is exact.

    Returns
    -------
    torch.Tensor
        ``(I - A)⁻¹``.

    Raises
    ------
    DivergentClosureError
        If a loop weight is at least one.
    UndecidedClosureError
        If a loop weight is within the rounding bound of one.
    """

    verdicts = _cached(lambda: _linear_verdicts(matrix))

    def repeat(loop: torch.Tensor, step: int) -> torch.Tensor:
        """The closure of one nonnegative loop weight.

        Parameters
        ----------
        loop : torch.Tensor
            The loop weight.
        step : int
            The states eliminated before it.

        Returns
        -------
        torch.Tensor
            ``1 / (1 - loop)``.
        """
        relative_error = _rounding(loop.dtype, roundings * step)
        _decide(
            loop,
            step,
            loop * (1 - relative_error) >= 1,
            loop * (1 + relative_error) < 1,
            verdicts,
            "a cycle has weight at least one",
        )
        return 1 / (1 - loop)

    return _eliminate(algebra, matrix, repeat)


# ============================================================================
# Duals
# ============================================================================


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
    `DualIdempotentSemiring` subclass. The carrier is the base's.

    Parameters
    ----------
    base : Algebra
        The algebra whose operations are swapped.
    proof : Proved or None
        The argument for the dual's laws about the carrier, given by the
        base that knows it; without one those laws are assumed.
    """

    tested = _UNITAL_TESTS

    def __init__(self, base: Algebra, proof: Proved | None = None) -> None:
        self._base = base
        self._name = f"Dual({base.name})"
        self._proof = proof

    @property
    def base(self) -> Algebra:
        """The underlying algebra this is the dual of.

        Returns
        -------
        Algebra
            The base.
        """
        return self._base

    @property
    def name(self) -> str:
        """``Dual(<base name>)``.

        Returns
        -------
        str
            The name.
        """
        return self._name

    @property
    def subject(self) -> ClaimSubject:
        """What the dual's claims are about: the dual of its base.

        Returns
        -------
        ClaimSubject
            The qualified class name with the base's name.
        """
        cls = type(self)
        return ClaimSubject(
            kind="algebra",
            name=f"{cls.__module__}.{cls.__qualname__}({self._base.name})",
        )

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for a law: the base's argument for laws about the carrier.

        Parameters
        ----------
        law : Law
            A law the dual claims.

        Returns
        -------
        LawEvidence
            The argument given at construction, or what the class declares.
        """
        if law.scope == "carrier":
            if self._proof is not None:
                return self._proof
            return self.assumption("the base gave no argument for its dual")
        return super().evidence(law)

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The base join of two operands, ``a`` first.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a ⋁_base b``.
        """
        return self._base.join_pair(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The base product folded over axes.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The fold, or the base unit over no terms.
        """
        axes = reduction_axes(t, dim)
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        return _fold_along_dim(self._base.tensor_op, t, axes)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The base join, which the dual uses as its meet.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The base join.
        """
        return self._base.join(t, dim=dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The base negation.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            The base negation of ``t``.
        """
        return self._base.negate(t)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in the base carrier.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            The base's membership.
        """
        return self._base.contains(t)

    @property
    def unit(self) -> float:
        """The base zero.

        Returns
        -------
        float
            The unit of the dual.
        """
        return float(self._base.zero)

    @property
    def zero(self) -> float:
        """The base unit.

        Returns
        -------
        float
            The zero of the dual.
        """
        return float(self._base.unit)

    def dual(self) -> Algebra:
        """The base: taking the dual twice gives it back.

        Returns
        -------
        Algebra
            The base.
        """
        return self._base


class DualIdempotentSemiring(DualAlgebra, IdempotentSemiring):
    """The dual of a distributive-lattice algebra.

    Returned by `BooleanAlgebra.dual` and `GodelAlgebra.dual`. Swapping
    the join and meet of a distributive lattice yields a distributive
    lattice, so the dual is again an idempotent commutative semiring:
    ``max`` (or OR) distributes over ``min`` (or AND), the unit of
    ``max`` is the bottom and the bottom of the reversed order, the old
    top, annihilates ``max``.

    The constructor takes the parameters of `DualAlgebra`, whose
    ``proof`` should argue that the base is a bounded distributive
    lattice under its join and product.
    """

    tested = _UNITAL_TESTS

    @property
    def top(self) -> float:
        """The greatest element of the join order, the base zero.

        Returns
        -------
        float
            The top.
        """
        return float(self._base.zero)

    def residuum(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``[w, x] = 0`` if ``w ≥ x`` (real order), else ``x``.

        The dual order reverses the reals and its tensor is ``max``, so
        ``max(v, w) ≥ x`` on the reals; its least real solution is
        ``0`` when ``w ≥ x`` and ``x`` otherwise.

        Parameters
        ----------
        w : torch.Tensor
            The left argument.
        x : torch.Tensor
            The right argument.

        Returns
        -------
        torch.Tensor
            The residuum.
        """
        w, x = torch.broadcast_tensors(torch.as_tensor(w, dtype=x.dtype), x)
        return torch.where(w >= x, torch.zeros_like(x), x)


# ============================================================================
# User-defined composition rules
# ============================================================================


class CustomAlgebra(Algebra):
    """User-defined algebra built from callable operations.

    Construct a fresh algebra from the primitive operations as Python
    functions, rather than subclassing `Algebra` for each variant.

    The constructor only stores the operations; **the user is
    responsible for ensuring they satisfy the algebra axioms**
    (associativity, identity, distributivity of ⊗ over ⋁,
    de-Morgan duality between ⊗ and ⋁ via ``negate`` for
    involutive lattices), so every law about the carrier is assumed.
    Basic structural axioms are sanity-checked at construction time
    against a handful of fixed sample inputs. A join over no terms is
    the zero whatever ``join`` does, and membership is decided by
    ``contains`` when it is given.

    The DSL surface ``algebra name { tensor_op(a, b) = …;
    join(t) = …; unit = …; zero = …; }`` compiles to this class
    under the hood, with ``verify=False`` (user expressions are
    arithmetic and routinely violate one of the canned samples
    used by `_sanity_check`).

    Parameters
    ----------
    name : str
        Human-readable name.
    tensor_op : Callable
        Binary product.
    join : Callable
        Reduction along axes, called on nonempty axes only.
    unit : float
        Unit of the product.
    zero : float
        Identity of the join.
    negate : Callable or None
        Negation, if any.
    meet : Callable or None
        Universal reduction, if any.
    verify : bool
        Whether to run the sample checks.
    contains : Callable or None
        The carrier's membership predicate, if declared.

    Raises
    ------
    ValueError
        If ``name`` is empty or a sample check fails.
    """

    tested = _UNITAL_TESTS

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
        contains: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
        self._contains = contains
        if verify:
            self._sanity_check()

    @property
    def name(self) -> str:
        """The name given at construction.

        Returns
        -------
        str
            The name.
        """
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The given product.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``.
        """
        return self._tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The given join, or the zero over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction.
        """
        axes = reduction_axes(t, dim)
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        return self._join(t, dim)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The given meet.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction.

        Raises
        ------
        NotImplementedError
            If no meet was given.
        """
        if self._meet is None:
            raise NotImplementedError(
                f"CustomAlgebra {self._name!r}: no meet supplied. "
                f"Pass meet=... to the constructor for explicit "
                f"universal-quantifier support."
            )
        return self._meet(t, dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The given negation.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            ``¬t``.

        Raises
        ------
        NotImplementedError
            If no negation was given.
        """
        if self._negate is None:
            raise NotImplementedError(
                f"CustomAlgebra {self._name!r}: no negation supplied. "
                f"Pass negate=... to the constructor for explicit "
                f"complement support."
            )
        return self._negate(t)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """The given membership predicate.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is in the carrier.

        Raises
        ------
        TypeError
            If the entries are complex or no predicate was given.
        """
        require_real(t)
        if self._contains is None:
            raise TypeError(
                f"CustomAlgebra {self._name!r}: no carrier declared. Pass "
                f"contains=... to the constructor."
            )
        return self._contains(t)

    @property
    def unit(self) -> float:
        """The given unit.

        Returns
        -------
        float
            The unit.
        """
        return self._unit

    @property
    def zero(self) -> float:
        """The given zero.

        Returns
        -------
        float
            The zero.
        """
        return self._zero

    def _sanity_check(self) -> None:
        """Cheap probabilistic checks of the algebra axioms.

        Verifies on a handful of fixed sample tensors:

        * ``tensor_op(unit, a) == a`` (left identity);
        * ``tensor_op(a, unit) == a`` (right identity);
        * ``tensor_op(zero, a) == zero`` (left absorbing).

        Raises
        ------
        ValueError
            On the first failed check, so the construction site can fix
            the specification.
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
        """The class and name.

        Returns
        -------
        str
            The representation.
        """
        return f"CustomAlgebra(name={self._name!r})"


class CustomSemigroupoid(Semigroupoid):
    """User-defined `Semigroupoid` built from callable operations.

    Use for composition rules whose composition is associative but
    which lack an identity element, for instance weighted shortest
    paths over a semiring without a multiplicative unit. Callers who
    want a full algebra (with identity, dagger, compact-closed
    structure) should use `CustomAlgebra` instead. Associativity of
    composition is assumed: the optional sample check covers the scalar
    product only, which does not suffice for it.

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
        on a fixed pseudo-random sample at construction time. Set
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
        """The name given at construction.

        Returns
        -------
        str
            The name.
        """
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The given product.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``.
        """
        return self._tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The given reduction.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction.
        """
        return self._join(t, dim)

    def _check_associativity(self) -> None:
        """Smoke-test ``(a ⊗ b) ⊗ c == a ⊗ (b ⊗ c)`` on a fixed sample.

        Not exhaustive: a determined user can construct a non-associative
        ``tensor_op`` that passes this check, but it catches gross
        mistakes. The sample comes from a local generator, so the global
        random stream is left untouched.

        Raises
        ------
        ValueError
            If the sample check fails.
        """
        gen = torch.Generator().manual_seed(0)
        a = torch.rand(4, generator=gen)
        b = torch.rand(4, generator=gen)
        c = torch.rand(4, generator=gen)
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
        """The class and name.

        Returns
        -------
        str
            The representation.
        """
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
        """The name given at construction.

        Returns
        -------
        str
            The name.
        """
        return self._name

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The given product.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``.
        """
        return self._tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The given reduction.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction.
        """
        return self._join(t, dim)

    def __repr__(self) -> str:
        """The class and name.

        Returns
        -------
        str
            The representation.
        """
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


def _implication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The Reichenbach implication ``1 - a + a b``.

    Parameters
    ----------
    a : torch.Tensor
        The antecedent.
    b : torch.Tensor
        The consequent.

    Returns
    -------
    torch.Tensor
        The implication.
    """
    return 1.0 - a + a * b


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
    return CustomMagmoid("MaterialImplication", _implication, _product)


# ============================================================================
# Built-in algebras
# ============================================================================


class ProductFuzzyAlgebra(ZeroSumFree, NoZeroDivisors, Algebra):
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

    A noisy-OR is 0 only when every term is, and a product only when a
    factor is, so support is a homomorphism from this algebra. The join
    is computed in log space, with each term below ``1 - EPS``.
    """

    proof = _proved("product-fuzzy")
    tested = _UNITAL_TESTS

    @property
    def name(self) -> str:
        """``"ProductFuzzy"``.

        Returns
        -------
        str
            The name.
        """
        return "ProductFuzzy"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product ``a b``.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Noisy-OR in log space: ``1 - exp(∑ log(1 - t))``.

        Parameters
        ----------
        t : torch.Tensor
            Operands in ``[0, 1]``.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The noisy-OR, which is 0 over no terms.
        """
        axes = reduction_axes(t, dim)
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        t_clamped = t.clamp(min=0.0, max=1.0 - EPS)
        log_complement = torch.log1p(-t_clamped)
        return -torch.expm1(log_complement.sum(dim=axes))

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The product ``∏_i t_i``, which is 1 over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return _product(t, dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The complement ``1 - t``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            The complement.
        """
        return 1.0 - t

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[0, 1]``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry lies in ``[0, 1]``.
        """
        return _unit_interval(t)

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """Composition by a noisy-OR of products, in log space.

        Parameters
        ----------
        m : torch.Tensor
            Left tensor of shape ``(*domain, *shared)``.
        n : torch.Tensor
            Right tensor of shape ``(*shared, *codomain)``.
        n_contract : int
            Number of shared axes to contract.

        Returns
        -------
        torch.Tensor
            The composite of shape ``(*domain, *codomain)``.

        Raises
        ------
        ValueError
            If ``n_contract`` is below one or the shared axes differ.
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
        product = m_expanded * n_expanded
        product_clamped = product.clamp(min=0.0, max=1.0 - EPS)
        log_complement = torch.log1p(-product_clamped)
        contract_dims = tuple(range(n_domain, n_domain + n_contract))
        return -torch.expm1(log_complement.sum(dim=contract_dims))

    def dual(self) -> Algebra:
        """The dual: probabilistic sum as product, product as join.

        Returns
        -------
        Algebra
            A `DualAlgebra` whose carrier laws are proved.
        """
        return DualAlgebra(
            self,
            proof=_proved("dual-product-fuzzy"),
        )


class BooleanAlgebra(Selective, Semifield, ZeroSumFree, Star, IdempotentSemiring):
    """{0, 1} with logical AND and OR.

    The enrichment for the category Rel of crisp binary relations:

        ⊗ = AND:     a ⊗ b = a ∧ b
        ⋁ = OR:      ⋁_i x_i = max_i x_i
        ⋀ = AND:     ⋀_i x_i = min_i x_i
        ¬ = NOT:     ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    The join selects an operand (OR of truth values is their maximum),
    and 1, the only element other than 0, is its own inverse, so the
    Boolean semiring is a selective semifield.
    """

    proof = _proved_with_star("boolean")
    tested = _tests(unital=True, tie=56, star=8)

    @property
    def star_domain(self) -> str:
        """Every matrix: no loop is better than the unit.

        Returns
        -------
        str
            The domain.
        """
        return "every square matrix over the carrier"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The closure by elimination; every loop weight closes to the unit.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            ``A*``.
        """
        return _eliminate(
            self,
            matrix,
            lambda loop, step: self.unit_like((), loop),
            unit_diagonal=True,
        )

    @property
    def name(self) -> str:
        """``"Boolean"``.

        Returns
        -------
        str
            The name.
        """
        return "Boolean"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Logical AND as a product, exact on ``{0, 1}``.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a ∧ b``.
        """
        return a * b

    def rank(self, t: torch.Tensor) -> torch.Tensor:
        """The value itself: true is better than false.

        Parameters
        ----------
        t : torch.Tensor
            Truth values.

        Returns
        -------
        torch.Tensor
            ``t``.
        """
        return t

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Logical AND as a minimum; true over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Truth values.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The conjunction.
        """
        return _extremum(
            t, dim, largest=False, empty=_filled(t, dim, self.top), carrier=self.name
        )

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Logical NOT, ``1 - t``.

        Parameters
        ----------
        t : torch.Tensor
            Truth values.

        Returns
        -------
        torch.Tensor
            The negation.
        """
        return 1.0 - t

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a ⊘ 1 = a``.

        Parameters
        ----------
        a : torch.Tensor
            The dividend.
        b : torch.Tensor
            The divisor, every element true.

        Returns
        -------
        torch.Tensor
            ``a ∧ b``, which is ``a``.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is false.
        """
        self.require_nonzero(b)
        return a * b

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``{0, 1}``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is 0 or 1.
        """
        t = _require_carrier(t)
        return (t == 0) | (t == 1)

    @property
    def unit(self) -> float:
        """``1.0``, true.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``, false.

        Returns
        -------
        float
            The zero.
        """
        return 0.0

    def dual(self) -> Algebra:
        """The dual algebra, an idempotent commutative semiring.

        Returns
        -------
        Algebra
            A `DualIdempotentSemiring` over this algebra.
        """
        return DualIdempotentSemiring(
            self,
            proof=_proved("dual-boolean"),
        )

    @property
    def top(self) -> float:
        """The greatest element of the join order.

        Returns
        -------
        float
            ``1.0``.
        """
        return 1.0

    def residuum(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``[w, x] = ¬w ∨ x``, Boolean implication.

        Parameters
        ----------
        w : torch.Tensor
            The left argument.
        x : torch.Tensor
            The right argument.

        Returns
        -------
        torch.Tensor
            The implication.
        """
        w, x = torch.broadcast_tensors(torch.as_tensor(w, dtype=x.dtype), x)
        return torch.maximum(1.0 - w, x)


class StandardMVAlgebra(ZeroSumFree, Algebra):
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
    bracketing. The product has zero divisors (``0.5 ⊗ 0.5 = 0``), so
    support is not a homomorphism from it. The name follows the
    standard terminology: a "Łukasiewicz algebra" in the literature is
    Moisil's distinct Łukasiewicz-Moisil algebra.

    References
    ----------
    Chang, C. C. (1958). Algebraic analysis of many valued logics.
    *Transactions of the American Mathematical Society*, 88(2), 467-490.
    [doi:10.1090/S0002-9947-1958-0094302-9](https://doi.org/10.1090/S0002-9947-1958-0094302-9)

    Cignoli, R. L. O., D'Ottaviano, I. M. L., and Mundici, D. (2000).
    *Algebraic Foundations of Many-Valued Reasoning*. Springer.
    [doi:10.1007/978-94-015-9480-6](https://doi.org/10.1007/978-94-015-9480-6)
    """

    proof = _proved("lukasiewicz")
    tested = _UNITAL_TESTS

    @property
    def name(self) -> str:
        """``"Lukasiewicz"``.

        Returns
        -------
        str
            The name.
        """
        return "Lukasiewicz"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The Łukasiewicz t-norm ``max(a + b - 1, 0)``.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            The t-norm.
        """
        return (a + b - 1.0).clamp(min=0.0)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The bounded sum ``min(1, ∑_i t_i)``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The bounded sum, which is 0 over no terms.
        """
        return _sum(t, dim).clamp(max=1.0)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum, which is 1 over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.
        """
        return _extremum(
            t, dim, largest=False, empty=_filled(t, dim, 1.0), carrier=self.name
        )

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The strong negation ``1 - t``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            The negation.
        """
        return 1.0 - t

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[0, 1]``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry lies in ``[0, 1]``.
        """
        return _unit_interval(t)

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0

    def dual(self) -> Algebra:
        """The dual: bounded sum as product, t-norm as join.

        Returns
        -------
        Algebra
            A `DualAlgebra` whose carrier laws are proved.
        """
        return DualAlgebra(
            self,
            proof=_proved("dual-lukasiewicz"),
        )


class GodelAlgebra(Selective, ZeroSumFree, NoZeroDivisors, Star, IdempotentSemiring):
    """[0,1] with Gödel (min) t-norm.

    ⊗ = min,   ⋁ = max,   ⋀ = min,
    ¬ = Gödel neg (1 if a == 0 else 0),
    I = 1.0, ⊥ = 0.0.

    The join selects the larger operand.
    """

    proof = _proved_with_star("godel")
    tested = _tests(unital=True, tie=672, star=6)

    @property
    def star_domain(self) -> str:
        """Every matrix: no loop is better than the unit.

        Returns
        -------
        str
            The domain.
        """
        return "every square matrix over the carrier"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The closure by elimination; every loop weight closes to the unit.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            ``A*``.
        """
        return _eliminate(
            self,
            matrix,
            lambda loop, step: self.unit_like((), loop),
            unit_diagonal=True,
        )

    @property
    def name(self) -> str:
        """``"Godel"``.

        Returns
        -------
        str
            The name.
        """
        return "Godel"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The minimum, the Gödel t-norm.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``min(a, b)``.
        """
        return torch.min(a, b)

    def rank(self, t: torch.Tensor) -> torch.Tensor:
        """The value itself: larger is better.

        Parameters
        ----------
        t : torch.Tensor
            Elements.

        Returns
        -------
        torch.Tensor
            ``t``.
        """
        return t

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum; the top over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Elements.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.
        """
        return _extremum(
            t, dim, largest=False, empty=_filled(t, dim, self.top), carrier=self.name
        )

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The Gödel negation: 1 at 0, else 0.

        Parameters
        ----------
        t : torch.Tensor
            Elements.

        Returns
        -------
        torch.Tensor
            The negation, in the dtype of ``t``.
        """
        return (t == 0).to(dtype=t.dtype)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[0, 1]``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry lies in ``[0, 1]``.
        """
        return _unit_interval(t)

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0

    def dual(self) -> Algebra:
        """The dual algebra, an idempotent commutative semiring.

        Returns
        -------
        Algebra
            A `DualIdempotentSemiring` over this algebra.
        """
        return DualIdempotentSemiring(
            self,
            proof=_proved("dual-godel"),
        )

    @property
    def top(self) -> float:
        """The greatest element of the join order.

        Returns
        -------
        float
            ``1.0``.
        """
        return 1.0

    def residuum(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``[w, x] = 1`` if ``w ≤ x``, else ``x``: the Gödel implication.

        Parameters
        ----------
        w : torch.Tensor
            The left argument.
        x : torch.Tensor
            The right argument.

        Returns
        -------
        torch.Tensor
            The implication.
        """
        w, x = torch.broadcast_tensors(torch.as_tensor(w, dtype=x.dtype), x)
        return torch.where(w <= x, torch.ones_like(x), x)


class TropicalAlgebra(Selective, ZeroSumFree, NoZeroDivisors, Star, IdempotentSemiring):
    """[0, ∞] with (+, min), the enrichment of Lawvere metric spaces.

        ⊗ = addition (distances compose),
        ⋁ = infimum (shortest path),
        ⋀ = supremum (longest path),
        I = 0.0, ⊥ = ∞.

    Composition is the tropical / (min, +) matrix product. The join
    selects the smaller operand. Negation is undefined. The carrier has
    no negative numbers, so an element other than 0 has no inverse and
    the algebra is not a semifield.
    """

    proof = _proved_with_star("tropical")
    tested = _tests(unital=True, tie=672, star=6)

    @property
    def star_domain(self) -> str:
        """Every matrix: no distance is shorter than zero.

        Returns
        -------
        str
            The domain.
        """
        return "every square matrix over the carrier"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The closure by elimination; every loop weight closes to the unit.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            ``A*``.
        """
        return _eliminate(
            self,
            matrix,
            lambda loop, step: self.unit_like((), loop),
            unit_diagonal=True,
        )

    @property
    def name(self) -> str:
        """``"Tropical"``.

        Returns
        -------
        str
            The name.
        """
        return "Tropical"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Addition of distances.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a + b``.
        """
        return a + b

    def rank(self, t: torch.Tensor) -> torch.Tensor:
        """The negated value: shorter is better.

        Parameters
        ----------
        t : torch.Tensor
            Distances.

        Returns
        -------
        torch.Tensor
            ``-t``.
        """
        return -t

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The supremum of distances; the top 0 over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Distances.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The maximum.
        """
        return _extremum(
            t, dim, largest=True, empty=_filled(t, dim, self.top), carrier=self.name
        )

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Refuse: negation is not defined here.

        Parameters
        ----------
        t : torch.Tensor
            Distances.

        Returns
        -------
        torch.Tensor
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "negation is not well-defined for the tropical algebra"
        )

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[0, ∞]``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is nonnegative or infinite.
        """
        t = _require_carrier(t)
        return t >= 0

    @property
    def unit(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The unit.
        """
        return 0.0

    @property
    def zero(self) -> float:
        """``∞``.

        Returns
        -------
        float
            The zero.
        """
        return float("inf")

    @property
    def top(self) -> float:
        """The greatest element of the join order.

        Returns
        -------
        float
            ``0.0``.
        """
        return 0.0

    def residuum(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``[w, x] = max(x - w, 0)``, truncated subtraction.

        The order reverses the reals (``a ≤ b ⟺ min(a, b) = b``), so
        ``v + w ≤ x`` there means ``v + w ≥ x`` on the reals, whose
        least real solution is ``max(x - w, 0)``; ``w = ∞`` gives ``0``.

        Parameters
        ----------
        w : torch.Tensor
            The left argument.
        x : torch.Tensor
            The right argument.

        Returns
        -------
        torch.Tensor
            The residuum.
        """
        w, x = torch.broadcast_tensors(torch.as_tensor(w, dtype=x.dtype), x)
        diff = torch.where(torch.isinf(w), torch.zeros_like(x), x - w)
        return diff.clamp(min=0.0)


class MaxPlusAlgebra(Selective, Semifield, ZeroSumFree, Star, IdempotentSemiring):
    """Max-plus (Viterbi) semifield on ℝ ∪ {-∞}.

    Distinct from `TropicalAlgebra` (which is min-plus,
    suited to shortest-path aggregations): the join here is ``max``
    and the tensor is ``+``. The canonical algebra for MAP decoding
    in HMMs, CRFs, and weighted automata.

        ⊗ = +,   ⋁ = max,   ⋀ = min,
        I = 0, ⊥ = -∞.

    The join selects the larger operand, and every finite score ``a``
    has the inverse ``-a``. `top` and the residuum live in the
    completion ``ℝ ∪ {±∞}``; ``+∞`` is not in the semiring's carrier,
    where ``-∞`` must annihilate.
    """

    proof = _proved_with_star("max-plus")
    tested = _tests(unital=True, tie=672, star=12)

    @property
    def star_domain(self) -> str:
        """The matrices without a cycle of positive score.

        Returns
        -------
        str
            The domain.
        """
        return "square matrices without a cycle of positive score"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The best path scores, by elimination with a decided loop test.

        A computed loop score above the rounding bound of its path proves
        a positive cycle, and one below minus the bound proves the loop is
        not positive. In between, the sign of the exact sum of the
        float64 or float32 weights decides: the cycles are tested in
        exact rational arithmetic on the given weights.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            ``A*``.

        Raises
        ------
        DivergentClosureError
            If a cycle has positive score.
        """
        _square(matrix)
        largest = _largest_finite(matrix)
        verdict: list[bool] = []

        def repeat(loop: torch.Tensor, step: int) -> torch.Tensor:
            """The closure of one loop score.

            Parameters
            ----------
            loop : torch.Tensor
                The loop score.
            step : int
                The states eliminated before it.

            Returns
            -------
            torch.Tensor
                The unit, when the loop does not gain.

            Raises
            ------
            DivergentClosureError
                If a cycle has positive score.
            """
            # A loop met after ``step`` eliminations is a sum of at most
            # ``2 step + 1`` entries, rounded at most ``2 step`` times.
            bound = _rounding(loop.dtype, 2 * step) * (2 * step + 1) * largest
            gains = float(loop) > bound
            if not gains and float(loop) >= -bound:
                if not verdict:
                    verdict.append(_positive_cycle(matrix))
                gains = verdict[0]
            if gains:
                raise DivergentClosureError(
                    "the best-path closure diverges: a cycle has positive score, "
                    "so a path around it can be made arbitrarily heavy"
                )
            return self.unit_like((), loop)

        return _eliminate(self, matrix, repeat, unit_diagonal=True)

    @property
    def name(self) -> str:
        """``"MaxPlus"``.

        Returns
        -------
        str
            The name.
        """
        return "MaxPlus"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Addition of scores.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a + b``.
        """
        return a + b

    def rank(self, t: torch.Tensor) -> torch.Tensor:
        """The value itself: larger is better.

        Parameters
        ----------
        t : torch.Tensor
            Scores.

        Returns
        -------
        torch.Tensor
            ``t``.
        """
        return t

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum; the top ``+∞`` of the completion over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Scores.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.
        """
        return _extremum(
            t, dim, largest=False, empty=_filled(t, dim, self.top), carrier=self.name
        )

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Refuse: negation is not defined here.

        Parameters
        ----------
        t : torch.Tensor
            Scores.

        Returns
        -------
        torch.Tensor
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "negation is not well-defined for the max-plus algebra"
        )

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The quotient ``a - b``.

        Parameters
        ----------
        a : torch.Tensor
            The dividend.
        b : torch.Tensor
            The divisor, every element finite.

        Returns
        -------
        torch.Tensor
            ``a - b``.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is ``-∞``.
        """
        self.require_nonzero(b)
        return a - b

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``ℝ ∪ {-∞}``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is finite or ``-∞``.
        """
        return _extended_below(t)

    @property
    def unit(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The unit.
        """
        return 0.0

    @property
    def zero(self) -> float:
        """``-∞``.

        Returns
        -------
        float
            The zero.
        """
        return -float("inf")

    @property
    def top(self) -> float:
        """The greatest element of the completed order.

        Returns
        -------
        float
            ``+∞``.
        """
        return float("inf")

    def residuum(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``[w, x] = x - w``, with ``[-∞, x] = +∞`` and ``[w, +∞] = +∞``.

        Parameters
        ----------
        w : torch.Tensor
            The left argument.
        x : torch.Tensor
            The right argument.

        Returns
        -------
        torch.Tensor
            The residuum.
        """
        w, x = torch.broadcast_tensors(torch.as_tensor(w, dtype=x.dtype), x)
        inf = torch.full_like(x, float("inf"))
        out = torch.where(torch.isneginf(w), inf, x - w)
        return torch.where(torch.isposinf(x), inf, out)


class _LogSumPair(torch.autograd.Function):
    """``log(e^a + e^b)`` whose derivatives are numbers at an absent operand.

    The forward pass is ``torch.logaddexp``. Its own derivative,
    ``1 / (1 + e^(b - a))``, differentiates to ``∞ / ∞`` at ``a = -∞``;
    the derivative here is ``e^(a - s)`` with ``s`` the sum, computed by
    operations whose derivatives are numbers there, so a second
    derivative is ``p (1 - p)`` at every point with a finite operand.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The log-sum of two log weights, not both ``-∞``.

        Parameters
        ----------
        a : torch.Tensor
            A log weight.
        b : torch.Tensor
            A log weight of the same shape.

        Returns
        -------
        torch.Tensor
            ``log(e^a + e^b)``.
        """
        return torch.logaddexp(a, b)

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """Keep the operands for the derivative.

        Parameters
        ----------
        ctx : FunctionCtx
            The context.
        inputs : tuple of torch.Tensor
            The operands.
        output : torch.Tensor
            The sum, unused.
        """
        del output
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The derivative ``(e^(a - s), e^(b - s))`` times the incoming one.

        Parameters
        ----------
        ctx : FunctionCtx
            The context holding the operands.
        grad : torch.Tensor
            The incoming derivative.

        Returns
        -------
        tuple of torch.Tensor
            The derivatives in each operand.
        """
        a, b = ctx.saved_tensors  # type: ignore[attr-defined]
        total = torch.logaddexp(a, b)
        return grad * torch.exp(a - total), grad * torch.exp(b - total)


class LogProbAlgebra(Semifield, ZeroSumFree, Star, Semiring):
    """Log-space sum-product semifield on [-∞, ∞).

    Tensor is real addition (probability multiplication in log-
    space) and join is `torch.logsumexp` (probability
    summation in log-space). ``exp`` is an isomorphism onto the
    probability semiring ``([0, ∞), +, ·)``, so the carrier must
    contain positive values: ``logsumexp(0, 0) = log 2``. The join is
    not idempotent, so this is a semiring and not a lattice. Division
    is subtraction of log weights.

        ⊗ = +,   ⋁ = logsumexp,   ⋀ = min,
        I = 0 (log 1), ⊥ = -∞ (log 0).

    A join of terms that are all ``-∞`` is ``-∞`` with derivative zero:
    the log of a sum of absent alternatives is ``-∞`` near them, so no
    mass moves. The binary sum agrees with the join over the pair in
    value and in first and second derivatives at every point; its
    derivative is not ``torch.logaddexp``'s, whose second derivative is
    not a number at an operand ``-∞``. The meet has no value over no terms, since the carrier
    has no greatest element.
    """

    proof = _proved_with_star("log-prob")
    tested = _tests(unital=True, star=10)

    @property
    def star_domain(self) -> str:
        """The matrices of probabilities whose spectral radius is below one.

        Returns
        -------
        str
            The domain.
        """
        return "square log-weight matrices whose weights have spectral radius below one"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The log of ``(I - A)⁻¹``, by elimination of states in log space.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``, ``-∞`` for absent edges.

        Returns
        -------
        torch.Tensor
            Log weights of all paths, including empty paths.

        Raises
        ------
        DivergentClosureError
            If an effective return weight is at least one.
        UndecidedClosureError
            If an effective return weight is within the rounding bound of one.
        ValueError
            If the matrix contains NaN or positive infinity.
        """
        _square(matrix)
        demand(
            ~(torch.isnan(matrix) | torch.isposinf(matrix)),
            lambda: ValueError(
                "log closure requires finite weights or negative infinity"
            ),
        )
        largest = _largest_finite(matrix)
        verdicts = _cached(lambda: _log_verdicts(matrix))

        def repeat(loop: torch.Tensor, step: int) -> torch.Tensor:
            """The log of ``1 / (1 - e^loop)``.

            Parameters
            ----------
            loop : torch.Tensor
                The log loop weight.
            step : int
                The states eliminated before it.

            Returns
            -------
            torch.Tensor
                The log closure of the loop.
            """
            # Each eliminated state rounds a log-sum and the log of a
            # geometric series, at most sixteen roundings of a log weight no
            # larger than the sum of ``2 step + 1`` entries.
            bound = _rounding(loop.dtype, 16 * step) * max(
                1.0, (2 * step + 1) * largest
            )
            _decide(
                loop,
                step,
                loop >= bound,
                loop < -bound,
                verdicts,
                "an effective return weight is at least one",
            )
            return -torch.log(-torch.expm1(loop))

        return _eliminate(self, matrix, repeat)

    @property
    def name(self) -> str:
        """``"LogProb"``.

        Returns
        -------
        str
            The name.
        """
        return "LogProb"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Addition of log weights.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a + b``.
        """
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The log of a sum of exponentials, stable and with zero masked.

        Parameters
        ----------
        t : torch.Tensor
            Log weights.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The reduction with the named axes removed. A reduction over
            no terms, or over terms that are all ``-∞``, is ``-∞`` with
            derivative zero.
        """
        axes = reduction_axes(t, dim)
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        # ``torch.logsumexp`` has the right value for an all-``-inf``
        # reduction but differentiates that value as ``NaN``. Impossible
        # dynamic-programming cells are represented by exactly such rows, so
        # the impossible rows are replaced by zeros before the reduction and
        # their result is a constant. Replacing them before the reduction
        # is essential: masking the result afterwards leaves the ``NaN`` of
        # the reduction in the autograd graph.
        impossible = torch.isneginf(t).all(dim=axes, keepdim=True)
        safe = torch.where(impossible, torch.zeros_like(t), t)
        total = torch.logsumexp(safe, dim=axes)
        return torch.where(
            impossible.squeeze(axes), torch.full_like(total, -float("inf")), total
        )

    def join_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The binary sum ``log(e^a + e^b)``, equal to `join` over the pair.

        Parameters
        ----------
        a : torch.Tensor
            The first log weight.
        b : torch.Tensor
            The second log weight, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            The sum. It agrees with `join` over the pair in value and in
            its first and second derivatives at every point: two absent
            operands give ``-∞`` with zero derivatives, and one gives the
            other's derivatives exactly.
        """
        a, b = torch.broadcast_tensors(a, b)
        impossible = torch.isneginf(a) & torch.isneginf(b)
        zero = torch.zeros_like(a)
        total = _LogSumPair.apply(
            torch.where(impossible, zero, a), torch.where(impossible, zero, b)
        )
        return torch.where(impossible, torch.full_like(total, -float("inf")), total)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum, refused over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Log weights.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.

        Raises
        ------
        ValueError
            If a reduced axis has size zero.
        """
        return _extremum(t, dim, largest=False, empty=None, carrier=self.name)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Refuse: negation is not defined here.

        Parameters
        ----------
        t : torch.Tensor
            Log weights.

        Returns
        -------
        torch.Tensor
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "negation is not well-defined for the log-prob algebra"
        )

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The quotient ``a - b`` of log weights.

        Parameters
        ----------
        a : torch.Tensor
            The dividend.
        b : torch.Tensor
            The divisor, every element finite.

        Returns
        -------
        torch.Tensor
            ``a - b``.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is ``-∞``.
        """
        self.require_nonzero(b)
        return a - b

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[-∞, ∞)``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is finite or ``-∞``.
        """
        return _extended_below(t)

    @property
    def unit(self) -> float:
        """``0.0``, the log of one.

        Returns
        -------
        float
            The unit.
        """
        return 0.0

    @property
    def zero(self) -> float:
        """``-∞``, the log of zero.

        Returns
        -------
        float
            The zero.
        """
        return -float("inf")


class RealAlgebra(Semifield, Ring, Semiring):
    """The field of real numbers (ℝ, +, ·).

    The canonical numeric semiring: addition is the semiring sum (not
    a lattice join, since it is not idempotent), multiplication the
    tensor. Use when entries are unbounded real weights with no
    probability interpretation. Subtraction and division are available,
    and sums may cancel, so support is not a homomorphism from it.
    """

    proof = _proved("real")
    tested = _UNITAL_TESTS

    @property
    def name(self) -> str:
        """``"Real"``.

        Returns
        -------
        str
            The name.
        """
        return "Real"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product ``a b``.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum over axes.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The sum, which is 0 over no terms.
        """
        axes = reduction_axes(t, dim)
        result = t
        for d in reversed(axes):
            result = result.sum(dim=d)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum, refused over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.

        Raises
        ------
        ValueError
            If a reduced axis has size zero.
        """
        return _extremum(t, dim, largest=False, empty=None, carrier=self.name)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The additive inverse ``-t``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            ``-t``.
        """
        return -t

    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The difference ``a - b``.

        Parameters
        ----------
        a : torch.Tensor
            The minuend.
        b : torch.Tensor
            The subtrahend.

        Returns
        -------
        torch.Tensor
            ``a - b``.
        """
        return a - b

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The quotient ``a / b``.

        Parameters
        ----------
        a : torch.Tensor
            The dividend.
        b : torch.Tensor
            The divisor, with no zero element.

        Returns
        -------
        torch.Tensor
            ``a / b``.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is 0.
        """
        self.require_nonzero(b)
        return a / b

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``ℝ``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is finite.
        """
        t = _require_carrier(t)
        return torch.isfinite(t)

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0


class BoundedSumProductAlgebra(ZeroSumFree, NoZeroDivisors, Algebra):
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

    proof = _proved("probability")
    tested = _UNITAL_TESTS

    @property
    def name(self) -> str:
        """``"Probability"``.

        Returns
        -------
        str
            The name.
        """
        return "Probability"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product, clamped to ``[0, 1]``.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return (a * b).clamp(min=0.0, max=1.0)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The bounded sum ``min(1, ∑_i t_i)``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The bounded sum, which is 0 over no terms.
        """
        return _sum(t, dim).clamp(min=0.0, max=1.0)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum, which is 1 over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.
        """
        return _extremum(
            t, dim, largest=False, empty=_filled(t, dim, 1.0), carrier=self.name
        )

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The complement ``1 - t``, clamped to ``[0, 1]``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            The complement.
        """
        return (1.0 - t).clamp(min=0.0, max=1.0)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[0, 1]``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry lies in ``[0, 1]``.
        """
        return _unit_interval(t)

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0


class CountingAlgebra(ZeroSumFree, NoZeroDivisors, Star, Semiring):
    """Sum-product semiring on the non-negative integers (ℕ, +, ·).

    Counting algebra: composition counts the number of distinct paths
    through a structure. Tensors may be floating (autograd requires
    it) or integer; either way the entries must be natural numbers. The
    naturals have no inverses, so this is not a semifield; the
    nonnegative reals are `MarkovAlgebra`.
    """

    proof = _proved_with_star("counting")
    tested = _tests(unital=True, star=10)

    @property
    def star_domain(self) -> str:
        """The matrices of counts without a cycle.

        Returns
        -------
        str
            The domain.
        """
        return "square matrices of counts whose graph has no cycle of nonzero count"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The number of paths, by elimination of states.

        Counts below ``2 / eps`` of the dtype are exact in floating point and elimination only adds and multiplies them, so
        a loop count is decided exactly.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            ``A*``.

        Raises
        ------
        DivergentClosureError
            If a cycle has a nonzero count.
        UndecidedClosureError
            If a count reaches ``2 / eps``, beyond which it is not exact.
        """
        star = _linear_star(self, matrix, 0)
        if star.is_floating_point():
            limit = 2.0 / torch.finfo(star.dtype).eps
            demand(
                star.detach() < limit,
                lambda: UndecidedClosureError(
                    f"a count of paths reaches {limit:.0f}, beyond which it is not "
                    "exact"
                ),
            )
        return star

    @property
    def name(self) -> str:
        """``"Counting"``.

        Returns
        -------
        str
            The name.
        """
        return "Counting"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product of counts.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``a b``.
        """
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum of counts.

        Parameters
        ----------
        t : torch.Tensor
            Counts.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The sum, which is 0 over no terms.
        """
        axes = reduction_axes(t, dim)
        result = t
        for d in reversed(axes):
            result = result.sum(dim=d)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum, refused over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Counts.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The minimum.

        Raises
        ------
        ValueError
            If a reduced axis has size zero.
        """
        return _extremum(t, dim, largest=False, empty=None, carrier=self.name)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Refuse: negation is not defined here.

        Parameters
        ----------
        t : torch.Tensor
            Counts.

        Returns
        -------
        torch.Tensor
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "negation is not well-defined for the counting "
            "(non-negative integer) algebra"
        )

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``ℕ``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is a finite nonnegative integer.
        """
        valid = _nonnegative(t)
        if t.is_floating_point():
            valid = valid & (t == t.floor())
        return valid

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0


class MarkovAlgebra(Semifield, ZeroSumFree, Star, Semiring):
    """Sum-product composition for stochastic matrices.

    Implements the composition rule of FinStoch:

        (g ∘ f)(a, c) = Σ_b f(a, b) · g(b, c)

    standard matrix multiplication on row-stochastic matrices.
    Formally:

        ⊗ = product,   ⋁ = sum,   ⋀ = product,
        ¬ = complement (1 - p),
        I = 1.0, ⊥ = 0.0.

    The operations form the probability semifield ``([0, ∞), +, ·)``;
    its sum is not idempotent, so it is a semiring and not a lattice.
    Composition is ordinary matrix multiplication, and the composite
    of row-stochastic matrices is row-stochastic.
    """

    proof = _proved_with_star("markov")
    tested = _tests(unital=True, star=10)

    @property
    def star_domain(self) -> str:
        """The nonnegative matrices of spectral radius below one.

        Returns
        -------
        str
            The domain.
        """
        return "square nonnegative matrices of spectral radius below one"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """``(I - A)⁻¹`` by elimination of states, without subtraction.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            ``A*``.

        Raises
        ------
        DivergentClosureError
            If the spectral radius is at least one.
        UndecidedClosureError
            If a loop weight is within the rounding bound of one.
        """
        return _linear_star(self, matrix, 4)

    @property
    def name(self) -> str:
        """``"Markov"``.

        Returns
        -------
        str
            The name.
        """
        return "Markov"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product ``a b``.

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum over axes.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The sum, which is 0 over no terms.
        """
        return _sum(t, dim)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The product over axes, which is 1 over no terms.

        Parameters
        ----------
        t : torch.Tensor
            Operands.
        dim : int or tuple of int
            Axes to reduce.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return _product(t, dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """The complement ``1 - t``.

        Parameters
        ----------
        t : torch.Tensor
            Operands.

        Returns
        -------
        torch.Tensor
            The complement.
        """
        return 1.0 - t

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The quotient ``a / b``.

        Parameters
        ----------
        a : torch.Tensor
            The dividend.
        b : torch.Tensor
            The divisor, with no zero element.

        Returns
        -------
        torch.Tensor
            ``a / b``.

        Raises
        ------
        ZeroDivisionError
            If an element of ``b`` is 0.
        """
        self.require_nonzero(b)
        return a / b

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Membership in ``[0, ∞)``.

        Parameters
        ----------
        t : torch.Tensor
            Entries.

        Returns
        -------
        torch.Tensor
            Whether each entry is finite and nonnegative.
        """
        return _nonnegative(t)

    @property
    def unit(self) -> float:
        """``1.0``.

        Returns
        -------
        float
            The unit.
        """
        return 1.0

    @property
    def zero(self) -> float:
        """``0.0``.

        Returns
        -------
        float
            The zero.
        """
        return 0.0


# ============================================================================
# Structures built from a semiring
# ============================================================================


def _inherited(structure: Lawful, base: Lawful, anchor: str) -> LawEvidence:
    """The evidence of a law that a theorem derives from the base's laws.

    Parameters
    ----------
    structure : Lawful
        The structure built from ``base``.
    base : Lawful
        The structure it is built from.
    anchor : str
        The anchor of the theorem's argument.

    Returns
    -------
    LawEvidence
        The argument when every law of the base about its carrier is
        established, else an assumption that names the base.
    """
    if all(
        claim.established for claim in base.claims() if claim.law.scope == "carrier"
    ):
        return _proved(anchor)
    return structure.assumption(f"the laws of {base.name} are not established")


class MatrixSemiring(OrderPreservingSemiring):
    """Square matrices over a scalar semiring, a semiring that does not commute.

    An element is an ``n × n`` matrix over ``base``. The sum is the
    entrywise sum, the product the matrix product of ``base``, the zero
    the zero matrix, and the unit the identity matrix. Matrices over a
    semiring form a semiring [Golan (1999)], and for
    ``n ≥ 2`` its product does not commute, so a computation over these
    weights has to keep the order of its factors.

    Parameters
    ----------
    base : OrderPreservingSemiring
        The scalar semiring of the entries.
    size : int
        The order ``n`` of the matrices.

    Raises
    ------
    ValueError
        If ``base`` is not scalar or ``size`` is negative.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    tested = _UNITAL_TESTS

    def __new__(cls, base: OrderPreservingSemiring, size: int) -> MatrixSemiring:
        """Build the matrices over a base, with a closure when the base has one.

        Parameters
        ----------
        base : OrderPreservingSemiring
            The scalar semiring of the entries.
        size : int
            The order of the matrices.

        Returns
        -------
        MatrixSemiring
            A `StarMatrixSemiring` when ``base`` is a `Star`, else a
            `MatrixSemiring`.
        """
        del size
        if isinstance(base, Star) and issubclass(StarMatrixSemiring, cls):
            return super().__new__(StarMatrixSemiring)
        return super().__new__(cls)

    def __getnewargs__(self) -> tuple[OrderPreservingSemiring, int]:
        """The constructor's arguments, for copies and pickles.

        Returns
        -------
        tuple
            The base and the order.
        """
        return (self._base, self._size)

    def __init__(self, base: OrderPreservingSemiring, size: int) -> None:
        if base.element_shape != ():
            raise ValueError("matrix entries are scalars of the base")
        if size < 0:
            raise ValueError(f"the order of a matrix is nonnegative, got {size}")
        self._base = base
        self._size = int(size)

    @property
    def base(self) -> OrderPreservingSemiring:
        """The scalar semiring of the entries.

        Returns
        -------
        OrderPreservingSemiring
            The base.
        """
        return self._base

    @property
    def size(self) -> int:
        """The order of the matrices.

        Returns
        -------
        int
            ``n``.
        """
        return self._size

    @property
    def name(self) -> str:
        """``Matrix(<base>, <n>)``.

        Returns
        -------
        str
            The name.
        """
        return f"Matrix({self._base.name}, {self._size})"

    @property
    def element_shape(self) -> tuple[int, ...]:
        """``(n, n)``.

        Returns
        -------
        tuple of int
            The shape of one matrix.
        """
        return (self._size, self._size)

    @property
    def subject(self) -> ClaimSubject:
        """What the claims are about: the construction over its base.

        Returns
        -------
        ClaimSubject
            The qualified class name with the structure's name.
        """
        cls = type(self)
        return ClaimSubject(
            kind="algebra", name=f"{cls.__module__}.{cls.__qualname__}({self.name})"
        )

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for a law; laws about the carrier hold by a theorem.

        Parameters
        ----------
        law : Law
            A law the structure claims.

        Returns
        -------
        LawEvidence
            The theorem's argument when the base's laws are established.
        """
        if law.scope == "carrier":
            return _inherited(self, self._base, "matrix")
        return super().evidence(law)

    def _check(self, t: torch.Tensor) -> None:
        """Require trailing matrix axes.

        Parameters
        ----------
        t : torch.Tensor
            Elements.

        Raises
        ------
        ValueError
            If the trailing axes are not ``(n, n)``.
        """
        if t.ndim < 2 or tuple(t.shape[-2:]) != self.element_shape:
            raise ValueError(
                f"elements of {self.name} have trailing axes {self.element_shape}, "
                f"got shape {tuple(t.shape)}"
            )

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The matrix product ``a b`` over the base, ``a`` on the left.

        Parameters
        ----------
        a : torch.Tensor
            Left matrices.
        b : torch.Tensor
            Right matrices, broadcastable over index axes.

        Returns
        -------
        torch.Tensor
            ``(a b)[i, k] = ⊕_j a[i, j] ⊗ b[j, k]``.

        Raises
        ------
        ValueError
            If an operand lacks the matrix axes.
        """
        self._check(a)
        self._check(b)
        products = self._base.tensor_op(a.unsqueeze(-1), b.unsqueeze(-3))
        return self._base.join(products, -2)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The entrywise sum over index axes.

        Parameters
        ----------
        t : torch.Tensor
            Matrices.
        dim : int or tuple of int
            Index axes to reduce.

        Returns
        -------
        torch.Tensor
            The sum, which is the zero matrix over no terms.

        Raises
        ------
        ValueError
            If ``t`` lacks the matrix axes.
        """
        self._check(t)
        axes = reduction_axes(t, dim, 2)
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        return self._base.join(t, axes)

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The zero matrix at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            Shape ``(*shape, n, n)``.
        """
        return self._base.zero_like((*shape, *self.element_shape), like)

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The identity matrix at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            Shape ``(*shape, n, n)``.
        """
        full = (*shape, *self.element_shape)
        diagonal = torch.eye(self._size, dtype=torch.bool, device=like.device)
        return torch.where(
            diagonal,
            self._base.unit_like(full, like),
            self._base.zero_like(full, like),
        )

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Whether every entry of each matrix is in the base carrier.

        Parameters
        ----------
        t : torch.Tensor
            Matrices.

        Returns
        -------
        torch.Tensor
            One truth value per matrix.

        Raises
        ------
        ValueError
            If ``t`` lacks the matrix axes.
        """
        self._check(t)
        return self._base.contains(t).all(dim=-1).all(dim=-1)


class StarMatrixSemiring(MatrixSemiring, Star):
    """Square matrices over a semiring with a closure, closed as one flat matrix.

    `MatrixSemiring` returns this class when its base is a `Star`. An
    ``n × n`` matrix ``A`` whose entries are ``k × k`` matrices is the
    ``nk × nk`` matrix ``φ(A)[(i, a), (j, b)] = A[i, j][a, b]`` of the
    base, and ``φ`` is an isomorphism of semirings from ``M_n(M_k(S))``
    onto ``M_nk(S)``. The closure of ``A`` is therefore
    ``φ⁻¹(star(φ(A)))``: it exists exactly when the base's closure of the
    flat matrix does, and it is the least solution of ``X = I ⊕ A X``
    because ``φ`` maps solutions to solutions and preserves the order of
    sums. This is the first closure whose product does not commute; it
    solves both ``X = I ⊕ A X`` and ``X = I ⊕ X A``.

    The constructor takes the parameters of `MatrixSemiring`.

    Raises
    ------
    TypeError
        If the base has no closure.
    """

    tested = _tests(unital=True, star=6)

    def __init__(self, base: OrderPreservingSemiring, size: int) -> None:
        if not isinstance(base, Star):
            raise TypeError(f"{base!r} has no closure, so its matrices have none")
        super().__init__(base, size)

    @property
    def star_domain(self) -> str:
        """The matrices whose flat matrix lies in the base's closure domain.

        Returns
        -------
        str
            The domain.
        """
        base = self._base
        assert isinstance(base, Star)
        return (
            "square matrices of blocks whose flattened matrix lies in the "
            f"domain of {base.name}: {base.star_domain}"
        )

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The closure of a matrix of blocks, through its flat matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n, k, k)``.

        Returns
        -------
        torch.Tensor
            ``A*``, of the same shape.

        Raises
        ------
        DivergentClosureError
            If the base's closure of the flat matrix diverges.
        UndecidedClosureError
            If rounding cannot decide it.
        ValueError
            If the matrix is not square in its blocks.
        """
        self._check(matrix)
        if matrix.dim() != 4 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"a closure needs a square matrix of blocks; got shape {tuple(matrix.shape)}"
            )
        base = self._base
        assert isinstance(base, Star)
        n, k = matrix.shape[0], self._size
        flat = matrix.permute(0, 2, 1, 3).reshape(n * k, n * k)
        closed = base.star(flat)
        return closed.reshape(n, k, n, k).permute(0, 2, 1, 3)


class FreeSemimodule(Semimodule):
    """Vectors of a fixed length over an order-preserving semiring.

    An element is a vector of ``n`` scalars. The sum is entrywise, the
    zero is the zero vector, and a scalar acts on the left of every
    entry. These operations make ``K^n`` a left semimodule over ``K``
    [Golan (1999)].

    Parameters
    ----------
    scalars : OrderPreservingSemiring
        The semiring that acts.
    size : int
        The length ``n`` of the vectors.

    Raises
    ------
    ValueError
        If ``size`` is negative.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    tested = _tests(unital=False)

    def __init__(self, scalars: OrderPreservingSemiring, size: int) -> None:
        if size < 0:
            raise ValueError(f"the length of a vector is nonnegative, got {size}")
        self._scalars = scalars
        self._size = int(size)

    @property
    def scalars(self) -> OrderPreservingSemiring:
        """The semiring that acts.

        Returns
        -------
        OrderPreservingSemiring
            The scalars.
        """
        return self._scalars

    @property
    def size(self) -> int:
        """The length of the vectors.

        Returns
        -------
        int
            ``n``.
        """
        return self._size

    @property
    def name(self) -> str:
        """``<scalars>^<n>``.

        Returns
        -------
        str
            The name.
        """
        return f"{self._scalars.name}^{self._size}"

    @property
    def element_shape(self) -> tuple[int, ...]:
        """``(n, *scalars.element_shape)``.

        Returns
        -------
        tuple of int
            The shape of one vector.
        """
        return (self._size, *self._scalars.element_shape)

    @property
    def subject(self) -> ClaimSubject:
        """What the claims are about: the construction over its base.

        Returns
        -------
        ClaimSubject
            The qualified class name with the structure's name.
        """
        cls = type(self)
        return ClaimSubject(
            kind="algebra", name=f"{cls.__module__}.{cls.__qualname__}({self.name})"
        )

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for a law; laws about the carrier hold by a theorem.

        Parameters
        ----------
        law : Law
            A law the structure claims.

        Returns
        -------
        LawEvidence
            The theorem's argument when the base's laws are established.
        """
        if law.scope == "carrier":
            return _inherited(self, self._scalars, "free-semimodule")
        return super().evidence(law)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The entrywise sum over index axes.

        Parameters
        ----------
        t : torch.Tensor
            Vectors.
        dim : int or tuple of int
            Index axes to reduce.

        Returns
        -------
        torch.Tensor
            The sum, which is the zero vector over no terms.
        """
        axes = reduction_axes(t, dim, len(self.element_shape))
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        return self._scalars.join(t, axes)

    def act(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """The scalar ``k`` on the left of every entry of ``m``.

        Parameters
        ----------
        k : torch.Tensor
            Scalars, with the scalars' element axes trailing.
        m : torch.Tensor
            Vectors.

        Returns
        -------
        torch.Tensor
            ``(k ⊗ m_1, ..., k ⊗ m_n)``.
        """
        rank = len(self._scalars.element_shape)
        return self._scalars.tensor_op(k.unsqueeze(k.ndim - rank), m)

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The zero vector at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            Shape ``(*shape, *element_shape)``.
        """
        return self._scalars.zero_like((*shape, self._size), like)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Whether every entry of each vector is a scalar.

        Parameters
        ----------
        t : torch.Tensor
            Vectors.

        Returns
        -------
        torch.Tensor
            One truth value per vector.
        """
        return self._scalars.contains(t).all(dim=-1)


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
MARKOV = MarkovAlgebra()
"""The `MarkovAlgebra` singleton, composing stochastic matrices."""

REICHENBACH = PRODUCT_FUZZY.dual()
"""The de Morgan dual of `PRODUCT_FUZZY`, the canonical probabilistic-implication
composition."""
BOOLEAN_DUAL = BOOLEAN.dual()
"""The de Morgan dual of `BOOLEAN`, which swaps its tensor and join."""
DUAL_LUKASIEWICZ = LUKASIEWICZ.dual()
"""The de Morgan dual of `LUKASIEWICZ`, which swaps its tensor and join."""
DUAL_GODEL = GODEL.dual()
"""The de Morgan dual of `GODEL`, which swaps its tensor and join."""


__all__ = [
    "Carrier",
    "UnitalCarrier",
    "CompositionRule",
    "Semigroupoid",
    "Magmoid",
    "OrderPreservingSemiring",
    "Algebra",
    "Semiring",
    "IdempotentSemiring",
    "Selection",
    "Selective",
    "ZeroSumFree",
    "NoZeroDivisors",
    "Semifield",
    "Ring",
    "Semimodule",
    "DivergentClosureError",
    "UndecidedClosureError",
    "Star",
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
    "MatrixSemiring",
    "StarMatrixSemiring",
    "FreeSemimodule",
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
