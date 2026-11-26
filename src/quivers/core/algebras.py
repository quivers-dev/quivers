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
satisfy as named `Law` records, and every member reports, for each law
it claims, a `LawClaim` whose status says whether the law is proved
from a written argument, checked on finite data, or only assumed.
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
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import ClassVar, Literal

import didactic.api as dx
import torch

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

type LawStatus = Literal["proved", "checked", "assumed"]
"""How a law is established: by a written argument, on finite data, or not at all."""

type LawSubject = Literal["carrier", "implementation"]
"""What a law is about: the mathematical carrier, or the tensor implementation."""


class Law(dx.Model):
    """One named law of an algebraic structure.

    A law about the carrier is an equation or implication over the
    mathematical carrier; floating-point rounding is not part of it. A
    law about the implementation states a property of the tensors the
    code returns, such as their shape, dtype, device and derivative.

    Parameters
    ----------
    name : str
        The law's name, by which classes introduce it.
    statement : str
        The law in symbols and words.
    subject : LawSubject
        ``"carrier"`` or ``"implementation"``.
    """

    name: str
    statement: str
    subject: LawSubject


class LawClaim(dx.Model):
    """A member's claim that one law holds, with its status.

    Parameters
    ----------
    law : Law
        The law.
    member : str
        The name of the member that claims it.
    status : LawStatus
        ``"proved"`` when an argument is written and cited,
        ``"checked"`` when a test on finite data establishes it, and
        ``"assumed"`` when membership in a class is the only ground.
    evidence : str
        The argument, the check, or the declaration.
    """

    law: Law
    member: str
    status: LawStatus
    evidence: str


def _law(name: str, statement: str, subject: LawSubject = "carrier") -> Law:
    """Build one entry of the catalogue.

    Parameters
    ----------
    name : str
        The name.
    statement : str
        The statement.
    subject : LawSubject
        What the law is about.

    Returns
    -------
    Law
        The law.
    """
    return Law(name=name, statement=statement, subject=subject)


LAWS: Mapping[str, Law] = MappingProxyType(
    {
        law.name: law
        for law in (
            _law(
                "operations stay in the carrier",
                "a ⊕ b, and a ⊗ b when the carrier has a product, lie in the "
                "carrier whenever a and b do",
            ),
            _law(
                "join over several axes is iterated join",
                "a reduction over several axes equals the reductions over "
                "one axis at a time, in any order",
            ),
            _law("additive associativity", "(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)"),
            _law("additive commutativity", "a ⊕ b = b ⊕ a"),
            _law("additive identity", "a ⊕ 0 = a"),
            _law("multiplicative associativity", "(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)"),
            _law("multiplicative identity", "1 ⊗ a = a = a ⊗ 1"),
            _law(
                "distributivity",
                "a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c) and (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)",
            ),
            _law("annihilation", "0 ⊗ a = 0 = a ⊗ 0"),
            _law("multiplicative commutativity", "a ⊗ b = b ⊗ a"),
            _law(
                "composition associativity",
                "(f >> g) >> h = f >> (g >> h) for composable tensors",
            ),
            _law("additive idempotence", "a ⊕ a = a"),
            _law("selectivity", "a ⊕ b is a or b"),
            _law("zero-sum-free", "a ⊕ b = 0 implies a = 0 and b = 0"),
            _law("no zero divisors", "a ⊗ b = 0 implies a = 0 or b = 0"),
            _law(
                "division inverts the product",
                "(a ⊗ b) ⊘ b = a for every b other than 0",
            ),
            _law("additive inverse", "a ⊕ (0 ⊖ a) = 0"),
            _law("action associativity", "(k ⊗ l) · m = k · (l · m)"),
            _law("action unit", "1 · m = m"),
            _law(
                "action distributes over the module sum",
                "k · (m ⊕ n) = (k · m) ⊕ (k · n)",
            ),
            _law(
                "action distributes over the scalar sum",
                "(k ⊕ l) · m = (k · m) ⊕ (l · m)",
            ),
            _law("action annihilation", "0 · m = 0 and k · 0 = 0"),
            _law(
                "zero has the requested form",
                "zero_like(shape, like) is the zero with shape (*shape, "
                "*element_shape), the dtype and device of like, and no gradient",
                "implementation",
            ),
            _law(
                "unit has the requested form",
                "unit_like(shape, like) is the unit with shape (*shape, "
                "*element_shape), the dtype and device of like, and no gradient",
                "implementation",
            ),
            _law(
                "empty join is the zero",
                "a join over an axis of size zero is the zero of the remaining "
                "shape, in the dtype and device of its input, and its "
                "derivative is a number",
                "implementation",
            ),
            _law(
                "first best receives the derivative",
                "a selective reduction returns as its witness the first best "
                "operand in row-major order of the reduced axes, and that "
                "operand receives the whole derivative, in every operation",
                "implementation",
            ),
        )
    }
)
"""The catalogue of laws, by name."""

_STRENGTH: Mapping[LawStatus, int] = MappingProxyType(
    {"assumed": 0, "checked": 1, "proved": 2}
)


class Lawful(ABC):
    """A structure whose class membership claims laws.

    A class lists in ``introduces`` the names of the laws (from `LAWS`)
    that membership in it adds; `laws` collects them along the class
    hierarchy. The status of a claim comes from declarations on the
    member's own class, never from a parent class, since a subclass may
    replace an operation: ``proof`` is an argument that establishes
    every claimed law about the carrier, and ``check`` names the test
    that establishes the laws about the implementation. A law about the
    carrier without a proof, or about the implementation without a
    check, is assumed.
    """

    introduces: ClassVar[tuple[str, ...]] = ()
    proof: ClassVar[str | None] = None
    check: ClassVar[str | None] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """The structure's name.

        Returns
        -------
        str
            A human-readable name.
        """
        ...

    @classmethod
    def laws(cls) -> tuple[Law, ...]:
        """The laws that membership in this class claims.

        Returns
        -------
        tuple of Law
            Each law once, in the order the hierarchy introduces them.
        """
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            for name in vars(klass).get("introduces", ()):
                if name not in names:
                    names.append(name)
        return tuple(LAWS[name] for name in names)

    def claim(self, law: Law) -> LawClaim:
        """The claim that one law holds, with its status.

        Parameters
        ----------
        law : Law
            A law of `laws`.

        Returns
        -------
        LawClaim
            The claim, with the status declared by the member's own class.

        Raises
        ------
        ValueError
            If the member does not claim the law.
        """
        if law not in self.laws():
            raise ValueError(f"{self.name} does not claim {law.name!r}")
        own = vars(type(self))
        ground = own.get("proof" if law.subject == "carrier" else "check")
        if isinstance(ground, str):
            status: LawStatus = "proved" if law.subject == "carrier" else "checked"
            return self._claim(law, status, ground)
        return self._claim(
            law, "assumed", f"declared by membership in {type(self).__name__}"
        )

    def claims(self) -> tuple[LawClaim, ...]:
        """The claims of every law of `laws`.

        Returns
        -------
        tuple of LawClaim
            One claim per law.
        """
        return tuple(self.claim(law) for law in self.laws())

    def establishes(self, *names: str) -> bool:
        """Whether every named law is claimed and proved or checked.

        Parameters
        ----------
        *names : str
            Names of laws in `LAWS`.

        Returns
        -------
        bool
            True when each law is claimed with a status other than
            ``"assumed"``.

        Raises
        ------
        KeyError
            If a name is not in the catalogue.
        """
        claimed = self.laws()
        for name in names:
            law = LAWS[name]
            if law not in claimed or self.claim(law).status == "assumed":
                return False
        return True

    def _claim(self, law: Law, status: LawStatus, evidence: str) -> LawClaim:
        """Build a claim of this member.

        Parameters
        ----------
        law : Law
            The law.
        status : LawStatus
            Its status.
        evidence : str
            Its ground.

        Returns
        -------
        LawClaim
            The claim.
        """
        return LawClaim(law=law, member=self.name, status=status, evidence=evidence)


def weakest(claims: tuple[LawClaim, ...]) -> LawStatus:
    """The weakest status among claims.

    Parameters
    ----------
    claims : tuple of LawClaim
        The claims; an empty tuple is proved.

    Returns
    -------
    LawStatus
        The status no claim falls below.
    """
    status: LawStatus = "proved"
    for claim in claims:
        if _STRENGTH[claim.status] < _STRENGTH[status]:
            status = claim.status
    return status


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

    introduces = (
        "operations stay in the carrier",
        "join over several axes is iterated join",
        "zero has the requested form",
        "empty join is the zero",
    )

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

    introduces = ("unit has the requested form",)

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

    introduces = ("composition associativity",)


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
    ``⊗`` [Golan (1999), chapter 1]. Matrices over it compose
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
        "additive associativity",
        "additive commutativity",
        "additive identity",
        "multiplicative associativity",
        "multiplicative identity",
        "distributivity",
        "annihilation",
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

    introduces = ("multiplicative commutativity",)


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

    introduces = ("additive idempotence",)

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

    introduces = ("selectivity", "first best receives the derivative")

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

    introduces = ("zero-sum-free",)


class NoZeroDivisors(UnitalCarrier):
    """A carrier in which a product is zero only when a factor is.

    Golan (1999) calls such a semiring entire. With `ZeroSumFree` it
    makes the support map a homomorphism.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = ("no zero divisors",)


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

    introduces = ("division inverts the product",)

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

    introduces = ("additive inverse",)

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
    annihilates [Golan (1999), chapter 14]. Expectation carriers, whose
    second component is acted on by probabilities, are built from this
    capability; `FreeSemimodule` is the module of vectors over ``K``.

    References
    ----------
    Golan, J. S. (1999). *Semirings and their Applications*. Springer.
    [doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
    """

    introduces = (
        "additive associativity",
        "additive commutativity",
        "additive identity",
        "action associativity",
        "action unit",
        "action distributes over the module sum",
        "action distributes over the scalar sum",
        "action annihilation",
    )

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


_GRID_CHECK = (
    "checked on the shipped instances by the law grid of the evidence "
    "harness (tests/evidence/test_law_grid.py)"
)


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
    proof : str or None
        An argument for the dual's laws about the carrier, given by the
        base that knows it; without one those laws are assumed.
    """

    check = _GRID_CHECK

    def __init__(self, base: Algebra, proof: str | None = None) -> None:
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

    def claim(self, law: Law) -> LawClaim:
        """The claim of a law, proved by the base's argument when given.

        Parameters
        ----------
        law : Law
            A law the dual claims.

        Returns
        -------
        LawClaim
            The claim.

        Raises
        ------
        ValueError
            If the dual does not claim the law.
        """
        if law.subject == "carrier" and self._proof is not None:
            if law not in self.laws():
                raise ValueError(f"{self.name} does not claim {law.name!r}")
            return self._claim(law, "proved", self._proof)
        return super().claim(law)

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

    check = _GRID_CHECK

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

    check = _GRID_CHECK

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

    proof = (
        "On [0, 1], 1 - (1 - a)(1 - b) and ab lie in [0, 1]; the noisy-OR "
        "over several axes is 1 minus a product over them, which is the "
        "iterated noisy-OR since the product is associative and "
        "commutative; 1 - ∏(1 - x_i) = 0 forces every x_i = 0, and ab = 0 "
        "forces a = 0 or b = 0."
    )
    check = _GRID_CHECK

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
            proof=(
                "On [0, 1], a + b - ab and ab lie in [0, 1], and the product "
                "folded over several axes is the iterated product since it "
                "is associative and commutative."
            ),
        )


class BooleanAlgebra(Selective, Semifield, ZeroSumFree, IdempotentSemiring):
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

    proof = (
        "({0, 1}, or, and, 0, 1) is the two-element Boolean algebra, a bounded "
        "distributive lattice, hence a commutative semiring with idempotent, "
        "selective sum (Golan 1999, chapter 1); or is 0 only when both "
        "operands are, and is 0 only when a factor is, and 1 is its own "
        "inverse."
    )
    check = _GRID_CHECK

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
            proof=(
                "Exchanging the join and meet of the Boolean algebra gives the "
                "Boolean algebra with its order reversed, a bounded "
                "distributive lattice, hence an idempotent commutative "
                "semiring (Golan 1999, chapter 1)."
            ),
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

    proof = (
        "On [0, 1], min(1, a + b) and max(a + b - 1, 0) lie in [0, 1]; for "
        "nonnegative terms min(1, a + b + c) = min(1, min(1, a + b) + c), so "
        "the bounded sum over several axes is iterated; min(1, a + b) = 0 "
        "forces a = b = 0."
    )
    check = _GRID_CHECK

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
            proof=(
                "On [0, 1], min(1, a + b) and max(a + b - 1, 0) lie in [0, 1], "
                "and the Łukasiewicz t-norm is associative and commutative, so "
                "its fold over several axes is iterated."
            ),
        )


class GodelAlgebra(Selective, ZeroSumFree, NoZeroDivisors, IdempotentSemiring):
    """[0,1] with Gödel (min) t-norm.

    ⊗ = min,   ⋁ = max,   ⋀ = min,
    ¬ = Gödel neg (1 if a == 0 else 0),
    I = 1.0, ⊥ = 0.0.

    The join selects the larger operand.
    """

    proof = (
        "([0, 1], max, min, 0, 1) is a bounded chain, hence a bounded "
        "distributive lattice and a commutative semiring with selective sum "
        "(Golan 1999, chapter 1); max(a, b) = 0 forces a = b = 0 and "
        "min(a, b) = 0 forces a = 0 or b = 0."
    )
    check = _GRID_CHECK

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
            proof=(
                "Exchanging max and min on [0, 1] gives the chain with its "
                "order reversed, a bounded distributive lattice, hence an "
                "idempotent commutative semiring (Golan 1999, chapter 1)."
            ),
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


class TropicalAlgebra(Selective, ZeroSumFree, NoZeroDivisors, IdempotentSemiring):
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

    proof = (
        "([0, ∞], min, +, ∞, 0) with ∞ + a = ∞ is the min-plus semiring on "
        "the extended nonnegative reals, with selective sum (Golan 1999, "
        "chapter 1); min(a, b) = ∞ forces a = b = ∞, and a + b = ∞ on "
        "[0, ∞] forces a = ∞ or b = ∞."
    )
    check = _GRID_CHECK

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


class MaxPlusAlgebra(Selective, Semifield, ZeroSumFree, IdempotentSemiring):
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

    proof = (
        "(ℝ ∪ {-∞}, max, +, -∞, 0) is the max-plus semifield, image of the "
        "tropical semifield under negation (Golan 1999, chapter 4): max "
        "selects an operand, every finite a has inverse -a, max(a, b) = -∞ "
        "forces a = b = -∞, and a + b = -∞ forces a = -∞ or b = -∞."
    )
    check = _GRID_CHECK

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


class LogProbAlgebra(Semifield, ZeroSumFree, Semiring):
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
    mass moves. The meet has no value over no terms, since the carrier
    has no greatest element.
    """

    proof = (
        "log is a semiring isomorphism from the semifield ([0, ∞), +, ·, 0, 1) "
        "onto ([-∞, ∞), logsumexp, +, -∞, 0), so every law of the former "
        "transfers (Golan 1999, chapter 4): a sum of nonnegative reals is 0 "
        "only when every term is, a product only when a factor is, and "
        "division is a - b for b > -∞."
    )
    check = _GRID_CHECK

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
        """The binary sum ``log(e^a + e^b)``, with zero masked.

        Parameters
        ----------
        a : torch.Tensor
            The first log weight.
        b : torch.Tensor
            The second log weight, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            The sum, which equals `join` over the pair; when both are
            ``-∞`` it is ``-∞`` with derivative zero.
        """
        a, b = torch.broadcast_tensors(a, b)
        impossible = torch.isneginf(a) & torch.isneginf(b)
        zero = torch.zeros_like(a)
        total = torch.logaddexp(
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

    proof = (
        "(ℝ, +, ·, 0, 1) is a field: a commutative ring in which every "
        "element other than 0 has a multiplicative inverse, hence a "
        "semifield without zero divisors."
    )
    check = _GRID_CHECK

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

    proof = (
        "On [0, 1], ab and min(1, a + b) lie in [0, 1]; for nonnegative terms "
        "min(1, a + b + c) = min(1, min(1, a + b) + c), so the bounded sum "
        "over several axes is iterated; min(1, a + b) = 0 forces a = b = 0, "
        "and ab = 0 forces a = 0 or b = 0."
    )
    check = _GRID_CHECK

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


class CountingAlgebra(ZeroSumFree, NoZeroDivisors, Semiring):
    """Sum-product semiring on the non-negative integers (ℕ, +, ·).

    Counting algebra: composition counts the number of distinct paths
    through a structure. Tensors may be floating (autograd requires
    it) or integer; either way the entries must be natural numbers. The
    naturals have no inverses, so this is not a semifield; the
    nonnegative reals are `MarkovAlgebra`.
    """

    proof = (
        "(ℕ, +, ·, 0, 1) is a commutative semiring (Golan 1999, chapter 1); "
        "a sum of naturals is 0 only when every term is, and a product only "
        "when a factor is."
    )
    check = _GRID_CHECK

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


class MarkovAlgebra(Semifield, ZeroSumFree, Semiring):
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

    proof = (
        "([0, ∞), +, ·, 0, 1) is the semifield of nonnegative reals (Golan "
        "1999, chapter 4): every positive element has a positive inverse, a "
        "sum of nonnegative reals is 0 only when every term is, and a "
        "product only when a factor is."
    )
    check = _GRID_CHECK

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


def _inherited(structure: Lawful, law: Law, base: Lawful, theorem: str) -> LawClaim:
    """A carrier claim that holds by a theorem when the base's laws do.

    Parameters
    ----------
    structure : Lawful
        The structure built from ``base``.
    law : Law
        A carrier law the structure claims.
    base : Lawful
        The structure it is built from.
    theorem : str
        The theorem that derives the law from the base's laws.

    Returns
    -------
    LawClaim
        The claim, as strong as the weakest claim of the base.
    """
    support = tuple(claim for claim in base.claims() if claim.law.subject == "carrier")
    status = weakest(support)
    evidence = f"{theorem} The laws of {base.name} it uses are {status}."
    return LawClaim(law=law, member=structure.name, status=status, evidence=evidence)


class MatrixSemiring(OrderPreservingSemiring):
    """Square matrices over a scalar semiring, a semiring that does not commute.

    An element is an ``n × n`` matrix over ``base``. The sum is the
    entrywise sum, the product the matrix product of ``base``, the zero
    the zero matrix, and the unit the identity matrix. Matrices over a
    semiring form a semiring [Golan (1999), chapter 3], and for
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

    check = _GRID_CHECK

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

    def claim(self, law: Law) -> LawClaim:
        """The claim of a law; carrier laws hold when the base's do.

        Parameters
        ----------
        law : Law
            A law the structure claims.

        Returns
        -------
        LawClaim
            The claim.

        Raises
        ------
        ValueError
            If the structure does not claim the law.
        """
        if law.subject == "carrier" and law in self.laws():
            return _inherited(
                self,
                law,
                self._base,
                "Matrices over a semiring form a semiring under the entrywise "
                "sum and the matrix product (Golan 1999, chapter 3).",
            )
        return super().claim(law)

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


class FreeSemimodule(Semimodule):
    """Vectors of a fixed length over an order-preserving semiring.

    An element is a vector of ``n`` scalars. The sum is entrywise, the
    zero is the zero vector, and a scalar acts on the left of every
    entry. These operations make ``K^n`` a left semimodule over ``K``
    [Golan (1999), chapter 14].

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

    check = _GRID_CHECK

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

    def claim(self, law: Law) -> LawClaim:
        """The claim of a law; carrier laws hold when the scalars' do.

        Parameters
        ----------
        law : Law
            A law the module claims.

        Returns
        -------
        LawClaim
            The claim.

        Raises
        ------
        ValueError
            If the module does not claim the law.
        """
        if law.subject == "carrier" and law in self.laws():
            return _inherited(
                self,
                law,
                self._scalars,
                "K^n with the entrywise sum and the left action on every entry "
                "is a left K-semimodule (Golan 1999, chapter 14).",
            )
        return super().claim(law)

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
    "LawStatus",
    "LawSubject",
    "Law",
    "LawClaim",
    "LAWS",
    "weakest",
    "Lawful",
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
