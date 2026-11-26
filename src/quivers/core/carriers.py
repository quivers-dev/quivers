"""Carriers of statistics: expectation, second-order expectation, products.

An expectation carrier pairs a weight ``p`` with a statistic ``r`` acted
on by weights [Eisner (2002); Li and Eisner (2009)]:

    <p1, r1> ⊗ <p2, r2> = <p1 p2, p1 r2 ⊕ p2 r1>,
    <p1, r1> ⊕ <p2, r2> = <p1 ⊕ p2, r1 ⊕ r2>.

Summing products over derivations gives ``<Z, ⊕_d p(d) r(d)>`` where
``r(d)`` sums the statistics of a derivation's factors; with factor
weights ``exp(θ · f)`` and statistics ``p f`` the second component is the
gradient of ``Z`` in ``θ``. The second-order carrier adds a second
statistic and their products; the product carrier runs several
semirings side by side. Each is an `OrderPreservingSemiring` of the one
hierarchy whose product commutes, with packed elements on one trailing
axis, and each gives its star from the star of its weights where that
exists.

References
----------
Eisner, J. (2002). Parameter estimation for probabilistic finite-state
transducers. *ACL*, 1-8.
[doi:10.3115/1073083.1073085](https://doi.org/10.3115/1073083.1073085)

Li, Z., and Eisner, J. (2009). First- and second-order expectation
semirings with applications to minimum-risk training on translation
forests. *EMNLP*, 40-51.
[doi:10.3115/1699510.1699517](https://doi.org/10.3115/1699510.1699517)
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence

import torch

from quivers.core._carriers import is_empty, reduction_axes
from quivers.core.algebras import (
    _UNITAL_TESTS,
    OrderPreservingSemiring,
    Semimodule,
    Semiring,
)
from quivers.core.laws import (
    TIMES_COMMUTATIVE,
    Citation,
    ClaimSubject,
    Law,
    LawEvidence,
    Lawful,
    Proved,
)

__all__ = [
    "ExpectationSemiring",
    "ProductSemiring",
    "SecondOrderExpectationSemiring",
    "ElementStar",
]

type ElementStar = Callable[[torch.Tensor], torch.Tensor]
"""The star ``p* = 1 ⊕ p ⊗ p*`` of the weights, elementwise."""

_PROOFS = "api/core/carriers.md"
"""The page that proves the laws of the carriers."""

_LI_EISNER = Citation(
    reference=(
        "Li and Eisner 2009, First- and second-order expectation semirings with "
        "applications to minimum-risk training on translation forests"
    ),
    locator="sections 3 and 4",
    doi="10.3115/1699510.1699517",
)


def _established(*structures: Lawful) -> bool:
    """Whether every law about the carrier of each structure is established.

    Parameters
    ----------
    *structures : Lawful
        The structures a carrier is built from.

    Returns
    -------
    bool
        True when each claim about a carrier is proved or decided.
    """
    return all(
        claim.established
        for structure in structures
        for claim in structure.claims()
        if claim.law.scope == "carrier"
    )


def _subject(structure: Lawful) -> ClaimSubject:
    """The subject of a construction: its class with its name.

    Parameters
    ----------
    structure : Lawful
        The construction.

    Returns
    -------
    ClaimSubject
        The subject.
    """
    cls = type(structure)
    return ClaimSubject(
        kind="algebra", name=f"{cls.__module__}.{cls.__qualname__}({structure.name})"
    )


def _size(shape: tuple[int, ...]) -> int:
    """The number of scalars of one element.

    Parameters
    ----------
    shape : tuple of int
        An element shape of rank at most one.

    Returns
    -------
    int
        Its size.

    Raises
    ------
    ValueError
        If the rank is above one.
    """
    if len(shape) > 1:
        raise ValueError(f"packed elements have rank at most one, got {shape}")
    return math.prod(shape)


class _Packed(OrderPreservingSemiring):
    """A commutative semiring whose elements are packed on one trailing axis.

    Subclasses define the element size, the product, and how the sum
    reduces each part.
    """

    introduces = (TIMES_COMMUTATIVE,)

    @property
    def element_shape(self) -> tuple[int, ...]:
        """One trailing axis of the packed size.

        Returns
        -------
        tuple of int
            ``(size,)``.
        """
        return (self._packed_size(),)

    @abstractmethod
    def _packed_size(self) -> int:
        """The number of scalars of one packed element.

        Returns
        -------
        int
            The size.
        """
        ...

    @property
    def subject(self) -> ClaimSubject:
        """What the claims are about: the construction over its parts.

        Returns
        -------
        ClaimSubject
            The qualified class name with the structure's name.
        """
        return _subject(self)

    def _check(self, t: torch.Tensor) -> None:
        """Require the packed trailing axis.

        Parameters
        ----------
        t : torch.Tensor
            Elements.

        Raises
        ------
        ValueError
            If the last axis is not the packed size.
        """
        if t.ndim < 1 or t.shape[-1] != self._packed_size():
            raise ValueError(
                f"elements of {self.name} have a trailing axis of "
                f"{self._packed_size()}, got shape {tuple(t.shape)}"
            )

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum over index axes, part by part.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        dim : int or tuple of int
            Index axes to reduce.

        Returns
        -------
        torch.Tensor
            The sum, which is the zero over no terms.

        Raises
        ------
        ValueError
            If ``t`` lacks the packed axis.
        """
        self._check(t)
        axes = reduction_axes(t, dim, 1)
        if is_empty(t, axes):
            return self.empty_join(t, axes)
        return self._join_parts(t, axes)

    @abstractmethod
    def _join_parts(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """The sum of each part over nonempty axes.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        axes : tuple of int
            Normalized index axes, none empty.

        Returns
        -------
        torch.Tensor
            The packed sum.
        """
        ...


class ExpectationSemiring(_Packed):
    """Weights paired with statistics in a semimodule (Li and Eisner 2009).

    An element is ``<p, r>``, packed as ``[p, r_1, ..., r_d]``, with ``p``
    in the weights' semiring and ``r`` in a semimodule over it of vectors
    of length ``d``. The product is ``<p1 p2, p1 · r2 ⊕ p2 · r1>``, the
    sum is componentwise, the zero is ``<0, 0>`` and the unit ``<1, 0>``.

    Parameters
    ----------
    module : Semimodule
        The statistics, vectors over a commutative semiring with scalar
        elements.

    Raises
    ------
    TypeError
        If the module's scalars are not a commutative `Semiring` with
        scalar elements, or its elements are not vectors.
    """

    tested = _UNITAL_TESTS

    def __init__(self, module: Semimodule) -> None:
        base = module.scalars
        if not isinstance(base, Semiring) or base.element_shape != ():
            raise TypeError(
                f"the weights of an expectation carrier are a commutative semiring "
                f"with scalar elements, not {base.name}"
            )
        if len(module.element_shape) != 1:
            raise TypeError(
                f"the statistics of an expectation carrier are vectors, not "
                f"elements of shape {module.element_shape}"
            )
        self._module = module
        self._base = base

    @property
    def base(self) -> Semiring:
        """The semiring of the weights.

        Returns
        -------
        Semiring
            The weights.
        """
        return self._base

    @property
    def module(self) -> Semimodule:
        """The semimodule of the statistics.

        Returns
        -------
        Semimodule
            The statistics.
        """
        return self._module

    @property
    def name(self) -> str:
        """``Expectation(<module>)``.

        Returns
        -------
        str
            The name.
        """
        return f"Expectation({self._module.name})"

    def _packed_size(self) -> int:
        """``1 + d``.

        Returns
        -------
        int
            The packed size.
        """
        return 1 + self._module.element_shape[0]

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for a law; laws about the carrier hold by a theorem.

        Parameters
        ----------
        law : Law
            A law the structure claims.

        Returns
        -------
        LawEvidence
            The argument when the laws of the weights and the statistics
            are established.
        """
        if law.scope == "carrier":
            if _established(self._base, self._module):
                return Proved(
                    argument=f"{_PROOFS}#proof-expectation", sources=(_LI_EISNER,)
                )
            return self.assumption(
                f"the laws of {self._base.name} or {self._module.name} are not "
                "established"
            )
        return super().evidence(law)

    def parts(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The weight and the statistic of each element.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            ``p`` of the index shape and ``r`` with a trailing axis of ``d``.

        Raises
        ------
        ValueError
            If ``t`` lacks the packed axis.
        """
        self._check(t)
        return t[..., 0], t[..., 1:]

    def pack(self, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Elements from weights and statistics.

        Parameters
        ----------
        p : torch.Tensor
            Weights.
        r : torch.Tensor
            Statistics, with a trailing axis of ``d``.

        Returns
        -------
        torch.Tensor
            ``<p, r>`` packed.
        """
        p, r = torch.broadcast_tensors(p.unsqueeze(-1), r)
        return torch.cat((p[..., :1], r), dim=-1)

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``<p1 p2, p1 · r2 ⊕ p2 · r1>``.

        Parameters
        ----------
        a : torch.Tensor
            Packed elements.
        b : torch.Tensor
            Packed elements, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            The product.
        """
        p1, r1 = self.parts(a)
        p2, r2 = self.parts(b)
        weight = self._base.tensor_op(p1, p2)
        statistic = self._module.join_pair(
            self._module.act(p1, r2), self._module.act(p2, r1)
        )
        return self.pack(weight, statistic)

    def _join_parts(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """The sum of weights and of statistics.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        axes : tuple of int
            Normalized index axes.

        Returns
        -------
        torch.Tensor
            The packed sum.
        """
        p, r = self.parts(t)
        return self.pack(self._base.join(p, axes), self._module.join(r, axes))

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """``<0, 0>`` at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            Shape ``(*shape, 1 + d)``.
        """
        return self.pack(
            self._base.zero_like(shape, like), self._module.zero_like(shape, like)
        )

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """``<1, 0>`` at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            Shape ``(*shape, 1 + d)``.
        """
        return self.pack(
            self._base.unit_like(shape, like), self._module.zero_like(shape, like)
        )

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Whether the weight and the statistic are in their carriers.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.

        Returns
        -------
        torch.Tensor
            One truth value per element.
        """
        p, r = self.parts(t)
        return self._base.contains(p) & self._module.contains(r)

    def star(self, t: torch.Tensor, weights_star: ElementStar) -> torch.Tensor:
        """``<p, r>* = <p*, p* · (p* · r)>``, where the star of ``p`` exists.

        It solves ``X = 1 ⊕ <p, r> ⊗ X``: the weight is ``p*`` and the
        statistic ``q`` must satisfy ``q = p · q ⊕ p* · r``, which
        ``p* · (p* · r)`` does since ``p* = 1 ⊕ p p*``.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        weights_star : ElementStar
            The star of the weights, elementwise.

        Returns
        -------
        torch.Tensor
            The packed star.
        """
        p, r = self.parts(t)
        starred = weights_star(p)
        return self.pack(
            starred, self._module.act(starred, self._module.act(starred, r))
        )


class SecondOrderExpectationSemiring(_Packed):
    """Weights with two statistics and their products (Li and Eisner 2009).

    An element is ``<p, r, s, t>`` with ``r`` of length ``m``, ``s`` of
    length ``k`` and ``t`` an ``m x k`` matrix, packed in that order with
    ``t`` row by row. The product is

        <p1 p2, p1 r2 ⊕ p2 r1, p1 s2 ⊕ p2 s1,
         p1 t2 ⊕ p2 t1 ⊕ r1 s2ᵀ ⊕ r2 s1ᵀ>,

    so summing over derivations gives ``⊕_d p(d) r(d) s(d)ᵀ`` in the last
    part, the expectation of the product of two additive statistics.

    Parameters
    ----------
    base : Semiring
        A commutative semiring with scalar elements.
    first : int
        The length ``m`` of the first statistic.
    second : int
        The length ``k`` of the second statistic.

    Raises
    ------
    TypeError
        If ``base`` is not a commutative semiring with scalar elements.
    ValueError
        If a length is negative.
    """

    tested = _UNITAL_TESTS

    def __init__(self, base: Semiring, first: int, second: int) -> None:
        if not isinstance(base, Semiring) or base.element_shape != ():
            raise TypeError(
                f"a second-order carrier is over a commutative semiring with scalar "
                f"elements, not {base.name}"
            )
        if first < 0 or second < 0:
            raise ValueError("statistic lengths are nonnegative")
        self._base = base
        self._first = int(first)
        self._second = int(second)

    @property
    def base(self) -> Semiring:
        """The semiring of every part.

        Returns
        -------
        Semiring
            The base.
        """
        return self._base

    @property
    def name(self) -> str:
        """``SecondOrder(<base>, <m>, <k>)``.

        Returns
        -------
        str
            The name.
        """
        return f"SecondOrder({self._base.name}, {self._first}, {self._second})"

    def _packed_size(self) -> int:
        """``1 + m + k + m k``.

        Returns
        -------
        int
            The packed size.
        """
        return 1 + self._first + self._second + self._first * self._second

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for a law; laws about the carrier hold by a theorem.

        Parameters
        ----------
        law : Law
            A law the structure claims.

        Returns
        -------
        LawEvidence
            The argument when the laws of the base are established.
        """
        if law.scope == "carrier":
            if _established(self._base):
                return Proved(
                    argument=f"{_PROOFS}#proof-second-order", sources=(_LI_EISNER,)
                )
            return self.assumption(f"the laws of {self._base.name} are not established")
        return super().evidence(law)

    def parts(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The four parts of each element.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.

        Returns
        -------
        tuple of torch.Tensor
            ``p``, ``r`` (``m``), ``s`` (``k``) and ``t`` (``m x k``).

        Raises
        ------
        ValueError
            If ``t`` lacks the packed axis.
        """
        self._check(t)
        m, k = self._first, self._second
        products = t[..., 1 + m + k :].reshape(*t.shape[:-1], m, k)
        return t[..., 0], t[..., 1 : 1 + m], t[..., 1 + m : 1 + m + k], products

    def pack(
        self, p: torch.Tensor, r: torch.Tensor, s: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Elements from their four parts.

        Parameters
        ----------
        p : torch.Tensor
            Weights.
        r : torch.Tensor
            First statistics, trailing ``m``.
        s : torch.Tensor
            Second statistics, trailing ``k``.
        t : torch.Tensor
            Products, trailing ``m x k``.

        Returns
        -------
        torch.Tensor
            The packed elements.
        """
        index = torch.broadcast_shapes(
            p.shape, r.shape[:-1], s.shape[:-1], t.shape[:-2]
        )
        flat = t.reshape(*t.shape[:-2], self._first * self._second)
        return torch.cat(
            (
                p.expand(index).unsqueeze(-1),
                r.expand(*index, self._first),
                s.expand(*index, self._second),
                flat.expand(*index, self._first * self._second),
            ),
            dim=-1,
        )

    def _act(self, k: torch.Tensor, v: torch.Tensor, rank: int) -> torch.Tensor:
        """A weight on every entry of a statistic.

        Parameters
        ----------
        k : torch.Tensor
            Weights.
        v : torch.Tensor
            Statistics with ``rank`` trailing axes.
        rank : int
            The rank of one statistic.

        Returns
        -------
        torch.Tensor
            ``k · v``.
        """
        return self._base.tensor_op(k.reshape(*k.shape, *([1] * rank)), v)

    def _plus(self, *terms: torch.Tensor) -> torch.Tensor:
        """The sum of several broadcastable tensors of the base.

        Parameters
        ----------
        *terms : torch.Tensor
            The terms.

        Returns
        -------
        torch.Tensor
            Their sum.
        """
        return self._base.join(torch.stack(torch.broadcast_tensors(*terms)), 0)

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product of two packed elements.

        Parameters
        ----------
        a : torch.Tensor
            Packed elements.
        b : torch.Tensor
            Packed elements, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            ``<p1 p2, p1 r2 ⊕ p2 r1, p1 s2 ⊕ p2 s1, p1 t2 ⊕ p2 t1 ⊕ r1 s2ᵀ ⊕ r2 s1ᵀ>``.
        """
        p1, r1, s1, t1 = self.parts(a)
        p2, r2, s2, t2 = self.parts(b)
        outer = self._base.tensor_op
        return self.pack(
            self._base.tensor_op(p1, p2),
            self._plus(self._act(p1, r2, 1), self._act(p2, r1, 1)),
            self._plus(self._act(p1, s2, 1), self._act(p2, s1, 1)),
            self._plus(
                self._act(p1, t2, 2),
                self._act(p2, t1, 2),
                outer(r1.unsqueeze(-1), s2.unsqueeze(-2)),
                outer(r2.unsqueeze(-1), s1.unsqueeze(-2)),
            ),
        )

    def _join_parts(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """Every part is in the base, so the sum is the base's.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        axes : tuple of int
            Normalized index axes.

        Returns
        -------
        torch.Tensor
            The packed sum.
        """
        return self._base.join(t, axes)

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """``<0, 0, 0, 0>`` at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            The zero.
        """
        return self._base.zero_like((*shape, self._packed_size()), like)

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """``<1, 0, 0, 0>`` at every index.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            The unit.
        """
        zero = self.zero_like(shape, like)
        return torch.cat(
            (self._base.unit_like((*shape, 1), like), zero[..., 1:]), dim=-1
        )

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Whether every part is in the base.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.

        Returns
        -------
        torch.Tensor
            One truth value per element.
        """
        self._check(t)
        return self._base.contains(t).all(dim=-1)

    def star(self, t: torch.Tensor, weights_star: ElementStar) -> torch.Tensor:
        """``<p*, p*² r, p*² s, p*² t ⊕ 2 p*³ r sᵀ>``, where ``p*`` exists.

        It solves ``X = 1 ⊕ <p, r, s, t> ⊗ X`` using only
        ``p* = 1 ⊕ p p*`` and commutativity; the factor 2 is ``1 ⊕ 1``.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        weights_star : ElementStar
            The star of the weights, elementwise.

        Returns
        -------
        torch.Tensor
            The packed star.
        """
        p, r, s, products = self.parts(t)
        starred = weights_star(p)
        square = self._base.tensor_op(starred, starred)
        cube = self._base.tensor_op(square, starred)
        cross = self._base.tensor_op(r.unsqueeze(-1), s.unsqueeze(-2))
        twice = self._plus(cross, cross)
        return self.pack(
            starred,
            self._act(square, r, 1),
            self._act(square, s, 1),
            self._plus(self._act(square, products, 2), self._act(cube, twice, 2)),
        )


class ProductSemiring(_Packed):
    """Several commutative semirings side by side.

    An element is a tuple of elements, one per factor, packed by
    concatenation; every operation acts factor by factor. Factors have
    elements of rank at most one and share a tensor dtype.

    Parameters
    ----------
    factors : Sequence of OrderPreservingSemiring
        The factors; each must claim a commutative product.

    Raises
    ------
    TypeError
        If there are fewer than two factors or one does not commute.
    ValueError
        If a factor's elements have rank above one.
    """

    tested = _UNITAL_TESTS

    def __init__(self, factors: Sequence[OrderPreservingSemiring]) -> None:
        if len(factors) < 2:
            raise TypeError("a product has at least two factors")
        for factor in factors:
            if TIMES_COMMUTATIVE not in factor.laws():
                raise TypeError(f"the product of {factor.name} does not commute")
        self._factors = tuple(factors)
        self._sizes = tuple(_size(factor.element_shape) for factor in factors)

    @property
    def factors(self) -> tuple[OrderPreservingSemiring, ...]:
        """The factors.

        Returns
        -------
        tuple of OrderPreservingSemiring
            The factors, in order.
        """
        return self._factors

    @property
    def name(self) -> str:
        """``Product(<factor>, ...)``.

        Returns
        -------
        str
            The name.
        """
        return f"Product({', '.join(factor.name for factor in self._factors)})"

    def _packed_size(self) -> int:
        """The sum of the factors' sizes.

        Returns
        -------
        int
            The packed size.
        """
        return sum(self._sizes)

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for a law; laws about the carrier hold by a theorem.

        Parameters
        ----------
        law : Law
            A law the structure claims.

        Returns
        -------
        LawEvidence
            The argument when the laws of every factor are established.
        """
        if law.scope == "carrier":
            if _established(*self._factors):
                return Proved(argument=f"{_PROOFS}#proof-product", sources=())
            return self.assumption("the laws of a factor are not established")
        return super().evidence(law)

    def parts(self, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The element of each factor.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.

        Returns
        -------
        tuple of torch.Tensor
            One tensor per factor, with its element axes.

        Raises
        ------
        ValueError
            If ``t`` lacks the packed axis.
        """
        self._check(t)
        found: list[torch.Tensor] = []
        start = 0
        for factor, size in zip(self._factors, self._sizes, strict=True):
            piece = t[..., start : start + size]
            found.append(piece.reshape(tuple(t.shape[:-1]) + factor.element_shape))
            start += size
        return tuple(found)

    def pack(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Elements from one element per factor.

        Parameters
        ----------
        parts : Sequence of torch.Tensor
            One tensor per factor, with its element axes.

        Returns
        -------
        torch.Tensor
            The packed elements.
        """
        flat = [
            part.reshape(
                tuple(part.shape[: part.ndim - len(factor.element_shape)]) + (size,)
            )
            for part, factor, size in zip(
                parts, self._factors, self._sizes, strict=True
            )
        ]
        index = torch.broadcast_shapes(*(piece.shape[:-1] for piece in flat))
        return torch.cat(
            [piece.expand(*index, piece.shape[-1]) for piece in flat], dim=-1
        )

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product, factor by factor.

        Parameters
        ----------
        a : torch.Tensor
            Packed elements.
        b : torch.Tensor
            Packed elements, broadcastable with ``a``.

        Returns
        -------
        torch.Tensor
            The packed product.
        """
        return self.pack(
            [
                factor.tensor_op(x, y)
                for factor, x, y in zip(
                    self._factors, self.parts(a), self.parts(b), strict=True
                )
            ]
        )

    def _join_parts(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """The sum, factor by factor.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        axes : tuple of int
            Normalized index axes.

        Returns
        -------
        torch.Tensor
            The packed sum.
        """
        return self.pack(
            [
                factor.join(part, axes)
                for factor, part in zip(self._factors, self.parts(t), strict=True)
            ]
        )

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The zero of every factor.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            The packed zero.
        """
        return self.pack([factor.zero_like(shape, like) for factor in self._factors])

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """The unit of every factor.

        Parameters
        ----------
        shape : tuple of int
            The index shape.
        like : torch.Tensor
            The tensor whose dtype and device are used.

        Returns
        -------
        torch.Tensor
            The packed unit.
        """
        return self.pack([factor.unit_like(shape, like) for factor in self._factors])

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Whether every factor's element is in its carrier.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.

        Returns
        -------
        torch.Tensor
            One truth value per element.
        """
        found = [
            factor.contains(part)
            for factor, part in zip(self._factors, self.parts(t), strict=True)
        ]
        result = found[0]
        for item in found[1:]:
            result = result & item
        return result

    def star(self, t: torch.Tensor, stars: Sequence[ElementStar]) -> torch.Tensor:
        """The star of every factor.

        Parameters
        ----------
        t : torch.Tensor
            Packed elements.
        stars : Sequence of ElementStar
            The star of each factor.

        Returns
        -------
        torch.Tensor
            The packed star, which exists where every factor's does.
        """
        return self.pack(
            [star(part) for star, part in zip(stars, self.parts(t), strict=True)]
        )
