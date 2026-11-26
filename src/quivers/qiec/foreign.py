"""What a foreign handler computes, stated as data on its declaration.

A foreign handler's clauses run in a runtime provider, so its
declaration says what they compute, and the provider builds them from
that statement and from nothing else; in particular never from the
handler's name. A search that sums its shots and a collector that
multiplies the weights of one derivation carry the semiring they work
in as a `KernelSemiring`: a member of the algebra hierarchy of
`quivers.core.algebras`, named by the subject of its law claims, with
the kernel type that holds its elements. The provider asks the member
for the laws the handler relies on and refuses a member that does not
establish them.

The elements of a kernel semiring are kernel values, so the operations
convert them to ``float64`` tensors, apply the member's own operations,
and convert back. The conversion is exact or refused: a natural number
is exact in ``float64`` below ``2 / eps``, the bound beyond which the
counting algebra refuses its own closure, and a truth value is ``0`` or
``1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import (
    BOOLEAN,
    COUNTING,
    GODEL,
    LOG_PROB,
    MARKOV,
    MAX_PLUS,
    REAL,
    TROPICAL,
    Semiring,
)
from quivers.core.laws import (
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    PLUS_UNIT,
    TIMES_ASSOCIATIVE,
    TIMES_UNIT,
    ClaimSubject,
    Law,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.types import BOOL, INT, REAL as REAL_TYPE, TypeExpr

type KernelValue = bool | int | float | tuple[KernelValue, ...]
"""An element of a kernel scalar type, or a tensor of them as nested tuples."""

KERNEL_ALGEBRAS: Mapping[str, Semiring] = MappingProxyType(
    {
        member.subject.name: member
        for member in (
            LOG_PROB,
            MAX_PLUS,
            TROPICAL,
            BOOLEAN,
            GODEL,
            COUNTING,
            REAL,
            MARKOV,
        )
    }
)
"""The shipped semirings with a scalar carrier, by the subject of their claims."""

KERNEL_CARRIERS: tuple[TypeExpr, ...] = (LOG_WEIGHT, REAL_TYPE, BOOL, INT)
"""The kernel types that hold the element of a kernel semiring."""

EXACT_COUNT = 2.0 / torch.finfo(torch.float64).eps
"""The bound below which every natural number is exact in ``float64``."""


class MissingLaw(ValueError):
    """A handler relies on a law its semiring does not establish."""


class InexactKernelValue(ValueError):
    """A kernel value has no exact image in the semiring's tensors."""


class KernelSemiring(QiecModel):
    """A semiring of the algebra hierarchy acting on one kernel type.

    Parameters
    ----------
    algebra : ClaimSubject
        The semiring, named by the subject its law claims are about; a
        key of `KERNEL_ALGEBRAS`.
    carrier : TypeExpr
        The kernel type of its elements, one of `KERNEL_CARRIERS`.

    Raises
    ------
    didactic.api.ValidationError
        If the subject names no shipped semiring with a scalar carrier,
        the carrier is not a kernel scalar type, or the semiring's zero
        or unit has no exact element of the carrier.
    """

    algebra: ClaimSubject
    carrier: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a known semiring whose zero and unit the carrier holds.

        Raises
        ------
        ValueError
            If any requirement fails.
        """
        if self.algebra.kind != "algebra" or self.algebra.name not in KERNEL_ALGEBRAS:
            raise ValueError(
                f"{self.algebra.name!r} is not a semiring with a scalar carrier; "
                f"the kernel semirings are {', '.join(sorted(KERNEL_ALGEBRAS))}"
            )
        if self.carrier not in KERNEL_CARRIERS:
            raise ValueError(
                "a kernel semiring's elements are LogWeight, Real, Bool or Int"
            )
        member = KERNEL_ALGEBRAS[self.algebra.name]
        try:
            self.value(torch.tensor(member.zero, dtype=torch.float64))
            self.value(torch.tensor(member.unit, dtype=torch.float64))
        except InexactKernelValue as error:
            raise ValueError(
                f"{member.name} has no zero and unit in the carrier: {error}"
            ) from error

    @classmethod
    def of(cls, member: Semiring, carrier: TypeExpr) -> KernelSemiring:
        """The kernel semiring of a shipped member on a carrier.

        Parameters
        ----------
        member : Semiring
            A value of `KERNEL_ALGEBRAS`.
        carrier : TypeExpr
            The kernel type of its elements.

        Returns
        -------
        KernelSemiring
            The record.
        """
        return cls(algebra=member.subject, carrier=carrier)

    @property
    def member(self) -> Semiring:
        """The semiring.

        Returns
        -------
        Semiring
            The shipped member the record names.
        """
        return KERNEL_ALGEBRAS[self.algebra.name]

    @property
    def exact(self) -> bool:
        """Whether the carrier's arithmetic is exact.

        Returns
        -------
        bool
            True for truth values and natural numbers, whose operations
            are exact within `EXACT_COUNT`; false for real and log weights,
            whose operations round.
        """
        return self.carrier in (BOOL, INT)

    @property
    def log_weights(self) -> bool:
        """Whether the elements are log weights, so a learned log weight is one.

        Returns
        -------
        bool
            True when the carrier is ``LogWeight``.
        """
        return self.carrier == LOG_WEIGHT

    def require(self, laws: Sequence[Law], what: str) -> None:
        """Require the semiring to establish the laws a handler relies on.

        Parameters
        ----------
        laws : Sequence of Law
            The laws.
        what : str
            The handler, for the diagnostic.

        Raises
        ------
        MissingLaw
            If a law is not claimed with a proved or decided status.
        """
        member = self.member
        for law in laws:
            if not member.establishes(law):
                status = (
                    member.claim(law).status.value
                    if law in member.laws()
                    else "not claimed"
                )
                raise MissingLaw(
                    f"{what} relies on {law.name!r}, which {member.name} "
                    f"does not establish ({status})"
                )

    def tensor(self, value: KernelValue) -> torch.Tensor:
        """The ``float64`` tensor of a kernel value.

        Parameters
        ----------
        value : KernelValue
            An element of the carrier, or a tensor of them.

        Returns
        -------
        torch.Tensor
            The same numbers.

        Raises
        ------
        InexactKernelValue
            If a natural number is beyond `EXACT_COUNT`, or an entry is
            not in the semiring's carrier.
        """
        if self.carrier == INT:
            _require_exact_counts(value)
        found = torch.tensor(value, dtype=torch.float64)
        if not bool(self.member.contains(found).all()):
            raise InexactKernelValue(
                f"{value!r} is not an element of {self.member.name}"
            )
        return found

    def value(self, found: torch.Tensor) -> KernelValue:
        """The kernel value of a tensor of elements.

        Parameters
        ----------
        found : torch.Tensor
            Elements of the semiring.

        Returns
        -------
        KernelValue
            A scalar of the carrier for a zero-dimensional tensor, else
            nested tuples of them.

        Raises
        ------
        InexactKernelValue
            If an entry has no exact element of the carrier: a truth value
            other than ``0`` or ``1``, or a number that is not a natural
            number below `EXACT_COUNT`.
        """
        return _kernel_value(found.tolist(), self.carrier)

    def zero(self) -> KernelValue:
        """The additive identity.

        Returns
        -------
        KernelValue
            The semiring's zero in the carrier.
        """
        return self.value(torch.tensor(self.member.zero, dtype=torch.float64))

    def unit(self) -> KernelValue:
        """The multiplicative identity.

        Returns
        -------
        KernelValue
            The semiring's unit in the carrier.
        """
        return self.value(torch.tensor(self.member.unit, dtype=torch.float64))

    def plus(self, values: Sequence[KernelValue]) -> KernelValue:
        """The sum of elements in their order.

        Parameters
        ----------
        values : Sequence of KernelValue
            Elements of one shape.

        Returns
        -------
        KernelValue
            Their sum, by the member's ``join``; the zero for none.
        """
        if not values:
            return self.zero()
        stacked = torch.stack([self.tensor(value) for value in values])
        return self.value(self.member.join(stacked, 0))

    def times(self, left: KernelValue, right: KernelValue) -> KernelValue:
        """The product of two elements, entrywise over tensors of them.

        Parameters
        ----------
        left : KernelValue
            The left factor; a scalar broadcasts over a tensor.
        right : KernelValue
            The right factor.

        Returns
        -------
        KernelValue
            ``left * right`` by the member's ``tensor_op``.
        """
        return self.value(self.member.tensor_op(self.tensor(left), self.tensor(right)))


def _require_exact_counts(value: KernelValue) -> None:
    """Require every natural number of a value to be exact in ``float64``.

    Parameters
    ----------
    value : KernelValue
        A kernel value.

    Raises
    ------
    InexactKernelValue
        If an entry's magnitude reaches `EXACT_COUNT`.
    """
    if isinstance(value, tuple):
        for entry in value:
            _require_exact_counts(entry)
        return
    if not isinstance(value, bool) and abs(value) >= EXACT_COUNT:
        raise InexactKernelValue(
            f"the count {value} reaches {EXACT_COUNT:.0f}, beyond which it is not exact"
        )


type _Listed = float | list[_Listed]
"""What ``torch.Tensor.tolist`` returns for a ``float64`` tensor."""


def _kernel_value(entry: _Listed, carrier: TypeExpr) -> KernelValue:
    """Convert listed tensor entries to kernel values.

    Parameters
    ----------
    entry : float or list
        A number, or a list of them.
    carrier : TypeExpr
        The kernel type of an element.

    Returns
    -------
    KernelValue
        The value.

    Raises
    ------
    InexactKernelValue
        If a number has no exact element of the carrier.
    """
    if isinstance(entry, list):
        return tuple(_kernel_value(item, carrier) for item in entry)
    if carrier == BOOL:
        if entry not in (0.0, 1.0):
            raise InexactKernelValue(f"{entry!r} is not a truth value")
        return entry == 1.0
    if carrier == INT:
        if not entry.is_integer() or abs(entry) >= EXACT_COUNT:
            raise InexactKernelValue(
                f"{entry!r} is not a count below {EXACT_COUNT:.0f}"
            )
        return int(entry)
    return float(entry)


SUM_LAWS: tuple[Law, ...] = (PLUS_ASSOCIATIVE, PLUS_COMMUTATIVE, PLUS_UNIT)
"""What a search relies on: the sum over a set of derivations is defined
by commutativity and associativity, whatever the enumeration order and
the nesting of choices, and an empty choice contributes the unit of the
sum."""

PRODUCT_LAWS: tuple[Law, ...] = (TIMES_ASSOCIATIVE, TIMES_UNIT)
"""What a collector relies on: the product of a derivation's weights is
taken in premise order from the unit, so it needs associativity and the
unit and not commutativity."""


class SemiringSum(QiecModel):
    """A search: resume once per alternative and sum what the shots answer.

    The handled computation answers its value with the product a
    collector inside it accumulated; a completed shot answers that
    product with the count one, an empty choice answers the zero with
    the count zero, and a choice answers the sum of its shots' weights
    with the sum of their counts. The answer is thus the semiring sum
    over the completed derivations and their number, the second being
    the counting semiring's sum of ones: a structural count that no
    weight can change.

    Parameters
    ----------
    semiring : KernelSemiring
        The semiring of the sum.
    tag : str
        Always ``"semiring-sum"``.
    """

    semiring: KernelSemiring = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["semiring-sum"] = "semiring-sum"

    @property
    def laws(self) -> tuple[Law, ...]:
        """The laws the handler relies on.

        Returns
        -------
        tuple of Law
            `SUM_LAWS`.
        """
        return SUM_LAWS


class SemiringProduct(QiecModel):
    """A collector: multiply every weight added in the scope, in order.

    Parameters
    ----------
    semiring : KernelSemiring
        The semiring of the product, applied entrywise to tensors of its
        elements.
    forkable : bool
        Whether an unrestricted resumption may capture an installation,
        each shot continuing from its own copy of the product so far;
        a search inside the scope needs it.
    tag : str
        Always ``"semiring-product"``.
    """

    semiring: KernelSemiring = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    forkable: bool = False
    tag: Literal["semiring-product"] = "semiring-product"

    @property
    def laws(self) -> tuple[Law, ...]:
        """The laws the handler relies on.

        Returns
        -------
        tuple of Law
            `PRODUCT_LAWS`.
        """
        return PRODUCT_LAWS


type EnumerationReduction = Literal["logsumexp", "sum", "mean", "table"]
"""How an enumeration aggregates its shots."""


class Enumeration(QiecModel):
    """An enumeration of a finite draw: resume once per support value.

    Parameters
    ----------
    grouped : bool
        Whether the handler answers one weight per group position.
    reduction : EnumerationReduction
        ``"logsumexp"`` for the log of the summed weights, ``"sum"`` or
        ``"mean"`` for those of the shots' values, ``"table"`` for the
        shots themselves, one per support value.
    tag : str
        Always ``"enumeration"``.
    """

    grouped: bool = False
    reduction: EnumerationReduction = "logsumexp"
    tag: Literal["enumeration"] = "enumeration"


class Settlement(QiecModel):
    """A stop of an iteration: answer its value only when the stop certifies it.

    The handled operation carries the value, whether the stop certifies
    it, and the residual of the last two iterates. A certified stop is
    answered with the value; any other is a stop by a control, which
    the provider refuses or reports as unestablished, as the caller's
    failure policy says.

    Parameters
    ----------
    semiring : KernelSemiring
        The semiring of the value, which fixes its metric.
    tag : str
        Always ``"settlement"``.
    """

    semiring: KernelSemiring = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["settlement"] = "settlement"


type ForeignSemantics = SemiringSum | SemiringProduct | Enumeration | Settlement
"""What a foreign handler computes."""


__all__ = [
    "EXACT_COUNT",
    "Enumeration",
    "EnumerationReduction",
    "ForeignSemantics",
    "InexactKernelValue",
    "KERNEL_ALGEBRAS",
    "KERNEL_CARRIERS",
    "KernelSemiring",
    "KernelValue",
    "MissingLaw",
    "PRODUCT_LAWS",
    "SUM_LAWS",
    "SemiringProduct",
    "SemiringSum",
    "Settlement",
]
