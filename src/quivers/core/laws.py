"""Laws, the evidence for them, and the structures that claim them.

A `Law` is a named statement about a carrier, or about the tensors an
implementation returns. A `LawClaim` says that a law holds of a subject
and carries its evidence; the claim's status is read from the kind of
evidence, so the two cannot disagree:

- `Proved`: a written argument, with the published results it uses;
- `Decided`: a procedure that decides the law on the stated domain,
  such as every tuple of a finite carrier;
- `Tested`: finitely many points of a larger domain, which can refute a
  law and cannot establish it;
- `Assumed`: a declaration and nothing more.

Proved and decided claims are *established*. A structure of the algebra
hierarchy is `Lawful`: its class lists the laws it introduces, and it
reports a claim for each law it has, with evidence declared by its own
class and never inherited, since a subclass may replace an operation.
`Requirement` pairs a law with the subject it must hold of, which is
what a plan asks for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Literal

import didactic.api as dx


def _nonempty(text: str, what: str) -> None:
    """Require a text field to say something.

    Parameters
    ----------
    text : str
        The text.
    what : str
        What it is, named in the diagnostic.

    Raises
    ------
    ValueError
        If the text is empty or only whitespace.
    """
    if not text.strip():
        raise ValueError(f"{what} must not be empty")


type LawScope = Literal["carrier", "implementation"]
"""What a law is about: the mathematical carrier, or the tensors of an implementation."""


class Citation(dx.Model):
    """A published source and the place in it that is relied on.

    Parameters
    ----------
    reference : str
        Authors, year and title.
    locator : str
        The section, theorem, equation, page, or the notion relied on.
    doi : str or None
        The digital object identifier, when the source has one.
    """

    reference: str
    locator: str
    doi: str | None = None

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a reference and a locator.

        Raises
        ------
        ValueError
            If either is empty.
        """
        _nonempty(self.reference, "a citation's reference")
        _nonempty(self.locator, "a citation's locator")


class Law(dx.Model):
    """An equation or property that an executor may rely on.

    A law about the carrier is a statement over the mathematical carrier;
    a floating point implementation satisfies no algebraic law exactly,
    and its rounding is a separate matter. A law about the
    implementation states a property of the tensors the code returns,
    such as their shape, dtype, device and derivative.

    Parameters
    ----------
    name : str
        A short name, unique among laws.
    statement : str
        The statement, quantified over the carrier.
    scope : LawScope
        ``"carrier"`` or ``"implementation"``.
    """

    name: str
    statement: str
    scope: LawScope = "carrier"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a name and a statement.

        Raises
        ------
        ValueError
            If either is empty.
        """
        _nonempty(self.name, "a law's name")
        _nonempty(self.statement, "a law's statement")


PLUS_ASSOCIATIVE = Law(
    name="plus-associative", statement="(a + b) + c = a + (b + c) for all a, b, c"
)
"""Associativity of the sum."""

PLUS_COMMUTATIVE = Law(name="plus-commutative", statement="a + b = b + a for all a, b")
"""Commutativity of the sum, which licenses reordering a finite sum."""

PLUS_UNIT = Law(name="plus-unit", statement="0 + a = a + 0 = a for all a")
"""Zero is the unit of the sum."""

TIMES_ASSOCIATIVE = Law(
    name="times-associative", statement="(a * b) * c = a * (b * c) for all a, b, c"
)
"""Associativity of the product."""

TIMES_COMMUTATIVE = Law(
    name="times-commutative", statement="a * b = b * a for all a, b"
)
"""Commutativity of the product, which licenses elimination orders."""

TIMES_UNIT = Law(name="times-unit", statement="1 * a = a * 1 = a for all a")
"""One is the unit of the product."""

DISTRIBUTIVE = Law(
    name="distributive",
    statement="a * (b + c) = a * b + a * c and (a + b) * c = a * c + b * c",
)
"""Distributivity, which licenses merging pending contributions."""

ZERO_ANNIHILATES = Law(name="zero-annihilates", statement="0 * a = a * 0 = 0 for all a")
"""Zero annihilates the product."""

PLUS_IDEMPOTENT = Law(name="plus-idempotent", statement="a + a = a for all a")
"""Idempotence, which licenses dropping a repeated contribution."""

PLUS_SELECTIVE = Law(name="plus-selective", statement="a + b is a or b for all a, b")
"""Selectivity, which a best derivation and its witness need."""

SUPERIOR = Law(
    name="superior",
    statement=(
        "under the declared total order, a * b is no better than a and no better "
        "than b, and the product is monotone in each argument"
    ),
)
"""Superiority (Knuth 1977), which licenses stopping best-first at the goal."""

STAR_LEAST_SOLUTION = Law(
    name="star-least-solution",
    statement="star(A) is the least solution X of X = I + A X on the declared domain",
)
"""The star of a matrix solves its linear system, which linear cycles need."""

DIVISION = Law(
    name="division", statement="(a * b) / b = a for all a and every nonzero b"
)
"""Division by nonzero elements, which normalizing needs."""

ADDITIVE_INVERSE = Law(name="additive-inverse", statement="a + (-a) = 0 for all a")
"""An additive inverse, which subtraction needs."""

ZERO_SUM_FREE = Law(name="zero-sum-free", statement="a + b = 0 implies a = b = 0")
"""No sum of nonzero elements is zero."""

NO_ZERO_DIVISORS = Law(
    name="no-zero-divisors", statement="a * b = 0 implies a = 0 or b = 0"
)
"""No product of nonzero elements is zero."""

SEMIMODULE_ACTION = Law(
    name="semimodule-action",
    statement=(
        "the action of the weights on the statistics is associative with the "
        "product, unital, distributes over the sum of the weights and over the "
        "sum of the statistics, and either zero annihilates it"
    ),
)
"""The action that carries expected statistics."""

OMEGA_CONTINUOUS = Law(
    name="omega-continuous",
    statement=(
        "the natural order is a complete partial order on increasing chains, and "
        "sum and product preserve their suprema"
    ),
)
"""Continuity, which gives a least fixed point of a monotone polynomial system."""

REAL_EMBEDDING = Law(
    name="real-embedding",
    statement=(
        "to_reals is an injective homomorphism of semirings into the nonnegative "
        "reals that preserves the natural order, and from_reals inverts it on its "
        "image"
    ),
)
"""An embedding into the nonnegative reals, whose completion ``[0, inf]`` gives a
monotone polynomial system its least fixed point and its certificate."""

FOLD_LINEAR = Law(
    name="fold-linear",
    statement="h(x + y) = h(x) + h(y) and h(k x) = k h(x) for weights k",
)
"""Linearity of a fold of the detailed meaning over the weights."""

OPERATIONS_CLOSED = Law(
    name="operations-closed",
    statement=(
        "a + b, and a * b and the action when the carrier has them, lie in the "
        "carrier whenever their operands do"
    ),
)
"""The operations do not leave the carrier."""

JOIN_ITERATED = Law(
    name="join-iterated",
    statement=(
        "a sum over several axes equals the sums over one axis at a time, in any order"
    ),
)
"""A reduction over several axes is iterated, which a contraction plan needs."""

COMPOSITION_ASSOCIATIVE = Law(
    name="composition-associative",
    statement="(f >> g) >> h = f >> (g >> h) for composable matrices",
)
"""Associativity of matrix composition."""

ZERO_FORM = Law(
    name="zero-form",
    statement=(
        "zero_like(shape, like) is the zero with shape (*shape, *element_shape), "
        "the dtype and device of like, and no gradient, and a dtype that cannot "
        "hold it is refused"
    ),
    scope="implementation",
)
"""The zero is made in the shape, dtype and device a computation needs."""

UNIT_FORM = Law(
    name="unit-form",
    statement=(
        "unit_like(shape, like) is the unit with shape (*shape, *element_shape), "
        "the dtype and device of like, and no gradient, and a dtype that cannot "
        "hold it is refused"
    ),
    scope="implementation",
)
"""The unit is made in the shape, dtype and device a computation needs."""

EMPTY_JOIN = Law(
    name="empty-join",
    statement=(
        "a sum over an axis of size zero is the zero of the remaining shape, in "
        "the dtype and device of its input, and its derivative is zero"
    ),
    scope="implementation",
)
"""A reduction over no terms is the zero."""

TIE_FIRST_MAXIMIZER = Law(
    name="tie-first-maximizer",
    statement=(
        "a selective sum returns as its witness the first best operand in "
        "row-major order of the reduced axes, and that operand receives the whole "
        "derivative, in every operation"
    ),
    scope="implementation",
)
"""The one tie rule of a selective sum, as its implementation keeps it."""

STAR_DECIDED = Law(
    name="star-decided",
    statement=(
        "star returns a value exactly on its declared domain: for a matrix outside "
        "it raises DivergentClosureError, and when rounding cannot decide which "
        "side of the boundary a cycle lies on it raises UndecidedClosureError"
    ),
    scope="implementation",
)
"""The closure refuses where it has no value and where rounding cannot decide."""

SEMIRING_LAWS: tuple[Law, ...] = (
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    PLUS_UNIT,
    TIMES_ASSOCIATIVE,
    TIMES_UNIT,
    DISTRIBUTIVE,
    ZERO_ANNIHILATES,
)
"""The laws of a semiring whose product need not commute."""

LAWS: Mapping[str, Law] = MappingProxyType(
    {
        law.name: law
        for law in (
            *SEMIRING_LAWS,
            TIMES_COMMUTATIVE,
            PLUS_IDEMPOTENT,
            PLUS_SELECTIVE,
            SUPERIOR,
            STAR_LEAST_SOLUTION,
            STAR_DECIDED,
            DIVISION,
            ADDITIVE_INVERSE,
            ZERO_SUM_FREE,
            NO_ZERO_DIVISORS,
            SEMIMODULE_ACTION,
            OMEGA_CONTINUOUS,
            REAL_EMBEDDING,
            FOLD_LINEAR,
            OPERATIONS_CLOSED,
            JOIN_ITERATED,
            COMPOSITION_ASSOCIATIVE,
            ZERO_FORM,
            UNIT_FORM,
            EMPTY_JOIN,
            TIE_FIRST_MAXIMIZER,
        )
    }
)
"""The catalogue of laws, by name."""


class TieRule(dx.Model):
    """The element of the subdifferential that a selection returns at a tie.

    Parameters
    ----------
    name : str
        A short name.
    statement : str
        Which operand receives the gradient.
    """

    name: str
    statement: str

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a name and a statement.

        Raises
        ------
        ValueError
            If either is empty.
        """
        _nonempty(self.name, "a tie rule's name")
        _nonempty(self.statement, "a tie rule's statement")


FIRST_MAXIMIZER = TieRule(
    name="first-maximizer",
    statement=(
        "the first maximizer in declared order receives the whole subgradient, and "
        "the value carries it as the witness"
    ),
)
"""The one tie rule of every selective sum."""


class ClaimSubject(dx.Model):
    """What a law claim is about.

    Parameters
    ----------
    kind : str
        ``"algebra"``, ``"computation"``, ``"interpretation"``,
        ``"operation"``, ``"presentation"`` or ``"estimator"``.
    name : str
        The subject's qualified Python name or stable identifier text.
    """

    kind: Literal[
        "algebra",
        "computation",
        "interpretation",
        "operation",
        "presentation",
        "estimator",
    ]
    name: str

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a name.

        Raises
        ------
        ValueError
            If the name is empty.
        """
        _nonempty(self.name, "a claim subject's name")


class LawEvidence(dx.TaggedUnion, discriminator="tag"):
    """Why a law is believed: an argument, a decision, a test, or a declaration."""


class Proved(LawEvidence):
    """Evidence that is a written argument.

    Parameters
    ----------
    argument : str
        Where the argument is written, as a documentation path and
        anchor.
    sources : tuple of Citation
        The published results the argument uses.
    tag : str
        Always ``"proved"``.
    """

    argument: str
    sources: tuple[Citation, ...] = ()
    tag: Literal["proved"] = "proved"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the place of the argument.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.argument, "the place of an argument")


class Decided(LawEvidence):
    """Evidence from a procedure that decides the law on the carrier.

    Parameters
    ----------
    procedure : str
        ``"exhaustive"`` (every tuple of a finite carrier),
        ``"polynomial-identity"`` (a normal form of polynomial terms) or
        ``"real-closed-field"`` (quantifier elimination over the reals).
    domain : str
        The carrier the decision covers.
    check : str
        The test that runs the procedure again.
    tag : str
        Always ``"decided"``.
    """

    procedure: Literal["exhaustive", "polynomial-identity", "real-closed-field"]
    domain: str
    check: str
    tag: Literal["decided"] = "decided"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the domain and the check.

        Raises
        ------
        ValueError
            If either is empty.
        """
        _nonempty(self.domain, "the domain of a decision")
        _nonempty(self.check, "the check of a decision")


class Tested(LawEvidence):
    """Evidence from tests on finitely many points of a larger domain.

    A test can refute a law and cannot establish one on an infinite
    domain, so tested laws license no exact or certified result.

    Parameters
    ----------
    check : str
        The test.
    cases : int
        How many points it evaluates, at least one.
    domain : str
        Where the points come from.
    tag : str
        Always ``"tested"``.
    """

    check: str
    cases: int
    domain: str
    tag: Literal["tested"] = "tested"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a check, a domain, and at least one case.

        Raises
        ------
        ValueError
            If the check or the domain is empty or there is no case.
        """
        _nonempty(self.check, "the check of a test")
        _nonempty(self.domain, "the domain of a test")
        if self.cases < 1:
            raise ValueError("a test evaluates at least one case")


class Assumed(LawEvidence):
    """A law that is declared and neither proved, decided nor tested.

    Parameters
    ----------
    declared_by : str
        Who or what declares it.
    reason : str
        Why it is assumed.
    tag : str
        Always ``"assumed"``.
    """

    declared_by: str
    reason: str
    tag: Literal["assumed"] = "assumed"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the declarer and the reason.

        Raises
        ------
        ValueError
            If either is empty.
        """
        _nonempty(self.declared_by, "the declarer of an assumption")
        _nonempty(self.reason, "the reason of an assumption")


class LawStatus(str, Enum):
    """The status a law claim has, fixed by its evidence."""

    PROVED = "proved"
    DECIDED = "decided"
    TESTED = "tested"
    ASSUMED = "assumed"


_STRENGTH: Mapping[LawStatus, int] = MappingProxyType(
    {
        LawStatus.ASSUMED: 0,
        LawStatus.TESTED: 1,
        LawStatus.DECIDED: 2,
        LawStatus.PROVED: 3,
    }
)


class LawClaim(dx.Model):
    """A claim that a law holds of a subject, with its evidence.

    The status is read from the evidence, so the two cannot disagree.
    QIEC equality evidence and registrations in a type class are not law
    evidence: they prove index equalities and record declarations.

    Parameters
    ----------
    law : Law
        The law.
    subject : ClaimSubject
        What it holds of.
    evidence : LawEvidence
        Why it is believed.
    """

    law: Law
    subject: ClaimSubject
    evidence: LawEvidence

    @property
    def status(self) -> LawStatus:
        """The status of the claim.

        Returns
        -------
        LawStatus
            Proved, decided, tested or assumed, by the kind of evidence.
        """
        tag: str = self.evidence.tag  # type: ignore[attr-defined]
        return LawStatus(tag)

    @property
    def established(self) -> bool:
        """Whether the claim is proved or decided.

        Returns
        -------
        bool
            True when an argument or a decision procedure stands behind
            the claim. Only established claims license exact, certified
            and probabilistic results.
        """
        return self.status in {LawStatus.PROVED, LawStatus.DECIDED}


class Requirement(dx.Model):
    """A law that a plan needs to hold of a subject.

    Parameters
    ----------
    law : Law
        The law.
    subject : ClaimSubject
        What it must hold of.
    """

    law: Law
    subject: ClaimSubject


def requirements(laws: Iterable[Law], subject: ClaimSubject) -> tuple[Requirement, ...]:
    """Pair laws with the subject they must hold of.

    Parameters
    ----------
    laws : Iterable of Law
        The laws.
    subject : ClaimSubject
        The subject.

    Returns
    -------
    tuple of Requirement
        One requirement per law, in order.
    """
    return tuple(Requirement(law=law, subject=subject) for law in laws)


def weakest(claims: Iterable[LawClaim]) -> LawStatus:
    """The weakest status among claims.

    Parameters
    ----------
    claims : Iterable of LawClaim
        The claims; none at all is proved.

    Returns
    -------
    LawStatus
        The status no claim falls below, in the order assumed, tested,
        decided, proved.
    """
    status = LawStatus.PROVED
    for claim in claims:
        if _STRENGTH[claim.status] < _STRENGTH[status]:
            status = claim.status
    return status


LAW_GRID = "tests/evidence/test_law_grid.py::test_cell"
"""The test that checks the laws of the shipped structures."""

TEST_DOMAINS: Mapping[str, str] = MappingProxyType(
    {
        ZERO_FORM.name: (
            "dtypes float64, float32, int64 and bool, and index shapes (), (0,), "
            "(3,), (2, 0) and (0, 2, 3), on the CPU"
        ),
        UNIT_FORM.name: (
            "dtypes float64, float32, int64 and bool, and index shapes (), (0,), "
            "(3,), (2, 0) and (0, 2, 3), on the CPU"
        ),
        EMPTY_JOIN.name: (
            "six index shapes with an axis of size zero, reduced over one or two "
            "axes, in float64 and float32"
        ),
        STAR_DECIDED.name: (
            "every matrix of the row's closure cases inside and outside the domain, "
            "of orders zero to three, in float64 and float32"
        ),
        TIE_FIRST_MAXIMIZER.name: (
            "every pair, triple and four-tuple of the row's samples, ties included, "
            "in float64 and float32"
        ),
    }
)
"""Where the tests of the laws about the implementation take their cases."""


class Lawful(ABC):
    """A structure whose class membership claims laws.

    A class lists in ``introduces`` the laws (from `LAWS`) that
    membership in it adds; `laws` collects them along the class
    hierarchy. The evidence of a claim comes from declarations on the
    member's own class, never from a parent class, since a subclass may
    replace an operation: ``proof`` is the argument for every claimed law
    about the carrier, and ``tested`` gives, for each law about the
    implementation that the law grid checks, the number of cases it
    evaluates. A law without either is assumed.
    """

    introduces: ClassVar[tuple[Law, ...]] = ()
    proof: ClassVar[Proved | None] = None
    tested: ClassVar[Mapping[str, int] | None] = None

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

    @property
    def subject(self) -> ClaimSubject:
        """What the structure's claims are about.

        Returns
        -------
        ClaimSubject
            The qualified name of the structure's class.
        """
        cls = type(self)
        return ClaimSubject(kind="algebra", name=f"{cls.__module__}.{cls.__qualname__}")

    @classmethod
    def laws(cls) -> tuple[Law, ...]:
        """The laws that membership in this class claims.

        Returns
        -------
        tuple of Law
            Each law once, in the order the hierarchy introduces them.
        """
        found: list[Law] = []
        for klass in reversed(cls.__mro__):
            for law in vars(klass).get("introduces", ()):
                if law not in found:
                    found.append(law)
        return tuple(found)

    def evidence(self, law: Law) -> LawEvidence:
        """The evidence for one claimed law.

        Parameters
        ----------
        law : Law
            A law of `laws`.

        Returns
        -------
        LawEvidence
            The evidence the member's own class declares.
        """
        own = vars(type(self))
        if law.scope == "carrier":
            proof = own.get("proof")
            if isinstance(proof, Proved):
                return proof
        else:
            tested = own.get("tested")
            if isinstance(tested, Mapping) and law.name in tested:
                return Tested(
                    check=f"{LAW_GRID}[{self.name}, {law.name}]",
                    cases=int(tested[law.name]),
                    domain=TEST_DOMAINS[law.name],
                )
        return self.assumption("membership in the class is the only ground")

    def assumption(self, reason: str) -> Assumed:
        """Evidence that a law is only declared.

        Parameters
        ----------
        reason : str
            Why it is assumed.

        Returns
        -------
        Assumed
            The declaration by the member's class.
        """
        cls = type(self)
        return Assumed(
            declared_by=f"{cls.__module__}.{cls.__qualname__}", reason=reason
        )

    def claim(self, law: Law) -> LawClaim:
        """The claim that one law holds, with its evidence.

        Parameters
        ----------
        law : Law
            A law of `laws`.

        Returns
        -------
        LawClaim
            The claim.

        Raises
        ------
        ValueError
            If the member does not claim the law.
        """
        if law not in self.laws():
            raise ValueError(f"{self.name} does not claim {law.name!r}")
        return LawClaim(law=law, subject=self.subject, evidence=self.evidence(law))

    def claims(self) -> tuple[LawClaim, ...]:
        """The claims of every law of `laws`.

        Returns
        -------
        tuple of LawClaim
            One claim per law.
        """
        return tuple(self.claim(law) for law in self.laws())

    def establishes(self, *laws: Law) -> bool:
        """Whether every given law is claimed and established.

        Parameters
        ----------
        *laws : Law
            Laws of the catalogue.

        Returns
        -------
        bool
            True when each law is claimed with a proved or decided status,
            read from its evidence as `LawClaim.status` reads it. The
            evidence of a structure is fixed by its class and its parts, so
            the answer is kept on the structure for the laws' names.
        """
        memo: dict[tuple[str, ...], bool] = vars(self).setdefault("_established", {})
        key = tuple(law.name for law in laws)
        found = memo.get(key)
        if found is None:
            claimed = self.laws()
            found = all(
                law in claimed and isinstance(self.evidence(law), (Proved, Decided))
                for law in laws
            )
            memo[key] = found
        return found


__all__ = [
    "LawScope",
    "Citation",
    "Law",
    "PLUS_ASSOCIATIVE",
    "PLUS_COMMUTATIVE",
    "PLUS_UNIT",
    "TIMES_ASSOCIATIVE",
    "TIMES_COMMUTATIVE",
    "TIMES_UNIT",
    "DISTRIBUTIVE",
    "ZERO_ANNIHILATES",
    "PLUS_IDEMPOTENT",
    "PLUS_SELECTIVE",
    "SUPERIOR",
    "STAR_LEAST_SOLUTION",
    "STAR_DECIDED",
    "DIVISION",
    "ADDITIVE_INVERSE",
    "ZERO_SUM_FREE",
    "NO_ZERO_DIVISORS",
    "SEMIMODULE_ACTION",
    "OMEGA_CONTINUOUS",
    "REAL_EMBEDDING",
    "FOLD_LINEAR",
    "OPERATIONS_CLOSED",
    "JOIN_ITERATED",
    "COMPOSITION_ASSOCIATIVE",
    "ZERO_FORM",
    "UNIT_FORM",
    "EMPTY_JOIN",
    "TIE_FIRST_MAXIMIZER",
    "SEMIRING_LAWS",
    "LAWS",
    "TieRule",
    "FIRST_MAXIMIZER",
    "ClaimSubject",
    "LawEvidence",
    "Proved",
    "Decided",
    "Tested",
    "Assumed",
    "LawStatus",
    "LawClaim",
    "Requirement",
    "requirements",
    "weakest",
    "LAW_GRID",
    "TEST_DOMAINS",
    "Lawful",
]
