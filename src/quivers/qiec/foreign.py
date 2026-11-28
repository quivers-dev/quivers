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

The elements of a kernel semiring are numbers of the machine, so the
operations apply the member's own operations to them: a real or log weight
in its own dtype, keeping its gradient, and a truth value or a count in
``float64`` and back. The conversion is exact or refused: a natural number
is exact in ``float64`` below ``2 / eps``, the bound beyond which the
counting algebra refuses its own closure, and a truth value is ``0`` or
``1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

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
from quivers._positions import demand
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import ParameterId, PresentationId
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.numbers import NumberDomain, machine_value
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL as REAL_TYPE,
    STRING,
    IndexVariable,
    TypeApplication,
    TypeExpr,
)

if TYPE_CHECKING:
    from quivers.qiec._span_finiteness import SpanOrder
    from quivers.qiec.rules import RulePresentation
    from quivers.stochastic.planning import Plan


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

type KernelValue = bool | int | float
"""A literal element of a kernel scalar type, as elaboration writes it."""

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
            self.value(torch.tensor(member.zero, dtype=torch.float64), NumberDomain())
            self.value(torch.tensor(member.unit, dtype=torch.float64), NumberDomain())
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

    def tensor(self, value: torch.Tensor) -> torch.Tensor:
        """An element as the member's operations take it.

        Parameters
        ----------
        value : torch.Tensor
            An element of the carrier, or a tensor of them, as the
            machine holds it.

        Returns
        -------
        torch.Tensor
            The same numbers: a real or log weight in its own dtype, so
            its gradient is kept; a truth value or a count in ``float64``,
            exact below `EXACT_COUNT`.

        Raises
        ------
        InexactKernelValue
            If the value is not a tensor of the carrier's kind, a count
            reaches `EXACT_COUNT`, or an entry is not in the semiring's
            carrier.
        """
        if not isinstance(value, torch.Tensor):
            raise InexactKernelValue(f"{value!r} is not a number of the machine")
        if self.carrier in (BOOL, INT):
            if value.is_floating_point():
                raise InexactKernelValue(f"{value!r} is not a truth value or a count")
            found = value.to(torch.float64)
            if self.carrier == INT:
                demand(
                    found.abs() < EXACT_COUNT,
                    lambda: InexactKernelValue(
                        f"a count reaches {EXACT_COUNT:.0f}, beyond which it is "
                        "not exact"
                    ),
                )
        else:
            if not value.is_floating_point():
                raise InexactKernelValue(f"{value!r} is not a real number")
            found = value
        member = self.member
        demand(
            member.contains(found),
            lambda: InexactKernelValue(f"an entry is not an element of {member.name}"),
        )
        return found

    def value(self, found: torch.Tensor, numbers: NumberDomain) -> torch.Tensor:
        """The element of the carrier a tensor of the member holds.

        Parameters
        ----------
        found : torch.Tensor
            Elements of the semiring.
        numbers : NumberDomain
            The run's numbers.

        Returns
        -------
        torch.Tensor
            The elements in the carrier's dtype: ``bool`` for truth values,
            ``int64`` for counts, and the run's real dtype otherwise, which
            keeps the gradient.

        Raises
        ------
        InexactKernelValue
            If an entry has no exact element of the carrier: a truth value
            other than ``0`` or ``1``, or a number that is not a natural
            number below `EXACT_COUNT`.
        """
        if self.carrier == BOOL:
            demand(
                (found == 0) | (found == 1),
                lambda: InexactKernelValue("an entry is not a truth value"),
            )
            return found != 0
        if self.carrier == INT:
            demand(
                (found == torch.round(found)) & (found.abs() < EXACT_COUNT),
                lambda: InexactKernelValue(
                    f"an entry is not a count below {EXACT_COUNT:.0f}"
                ),
            )
            return found.to(torch.int64)
        return found.to(numbers.real_dtype)

    def zero(self) -> KernelValue:
        """The additive identity, as a literal of the carrier.

        Returns
        -------
        KernelValue
            The semiring's zero.
        """
        return self._literal(self.member.zero)

    def unit(self) -> KernelValue:
        """The multiplicative identity, as a literal of the carrier.

        Returns
        -------
        KernelValue
            The semiring's unit.
        """
        return self._literal(self.member.unit)

    def _literal(self, element: float) -> KernelValue:
        """A constant of the member as a literal of the carrier.

        Parameters
        ----------
        element : float
            The constant.

        Returns
        -------
        KernelValue
            The literal.
        """
        found = self.value(torch.tensor(element, dtype=torch.float64), NumberDomain())
        return found.item()

    def plus(
        self, values: Sequence[torch.Tensor], numbers: NumberDomain
    ) -> torch.Tensor:
        """The sum of elements in their order.

        Parameters
        ----------
        values : Sequence of torch.Tensor
            Elements of one shape.
        numbers : NumberDomain
            The run's numbers.

        Returns
        -------
        torch.Tensor
            Their sum, by the member's ``join``; the zero for none.
        """
        if not values:
            zero = machine_value(self.carrier, self.zero(), numbers)
            assert isinstance(zero, torch.Tensor)
            return zero
        stacked = torch.stack([self.tensor(value) for value in values])
        return self.value(self.member.join(stacked, 0), numbers)

    def times(
        self, left: torch.Tensor, right: torch.Tensor, numbers: NumberDomain
    ) -> torch.Tensor:
        """The product of two elements, entrywise over tensors of them.

        Parameters
        ----------
        left : torch.Tensor
            The left factor; a scalar broadcasts over a tensor.
        right : torch.Tensor
            The right factor.
        numbers : NumberDomain
            The run's numbers.

        Returns
        -------
        torch.Tensor
            ``left * right`` by the member's ``tensor_op``.
        """
        return self.value(
            self.member.tensor_op(self.tensor(left), self.tensor(right)), numbers
        )


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


type EnumerationReduction = Literal[
    "logsumexp", "sum", "mean", "table", "log_expectation"
]
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
        shots themselves, one per support value. ``"log_expectation"``
        aggregates scalar log scores, with ``LogWeight`` as both the
        scope's input and output type.
    tag : str
        Always ``"enumeration"``.
    """

    grouped: bool = False
    reduction: EnumerationReduction = "logsumexp"
    tag: Literal["enumeration"] = "enumeration"

    @dx.model_validator()
    def _validate_reduction(self) -> None:
        """Require one scalar answer for a log expectation.

        Raises
        ------
        ValueError
            If a log expectation is configured to preserve group positions.
        """
        if self.reduction == "log_expectation" and self.grouped:
            raise ValueError("log expectation enumeration is ungrouped")


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


DEDUCTION_OUTCOME_KEYS = frozenset({"item", "category", "form"})


class DeductionLinearGoal(QiecModel):
    """Solve the full-span weight of one checked lexical linear system.

    Parameters
    ----------
    presentation
        The nominal span rules. The native adapter admits unary cycles
        and binary rules whose two spans concatenate. Lexical spans
        have width one and the presentation has no other axiom widths.
    lexicon
        The ordered pairs of token and atomic category. Their live log
        weights occupy the first slots of the operation's weight vector.
    parameters
        The distinct nominal rule-weight slots, in their vector order.
    start
        The requested final category, or None to sum all full-span items.
    """

    presentation: QiecModel = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    lexicon: tuple[tuple[str, str], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[ParameterId, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    start: str | None
    tag: Literal["deduction-linear-goal"] = "deduction-linear-goal"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Check the lexical metadata and its exact plan before attachment.

        Raises
        ------
        ValueError
            If the checked recurrence, weight addresses, or lexical inputs
            do not admit the native span plan.
        """
        from quivers.qiec.rules import RulePresentation

        presentation = self.presentation
        if not isinstance(presentation, RulePresentation) or (
            presentation.id != PresentationId.derive(presentation.name)
        ):
            raise ValueError(
                "a linear deduction goal needs its nominal rule presentation"
            )
        if presentation.computed_weight_rules:
            raise ValueError(
                "linear deduction goals cannot execute computed rule weights"
            )
        expected = tuple(
            dict.fromkeys(
                rule.weight
                for rule in presentation.rules
                if isinstance(rule.weight, ParameterId)
            )
        )
        if expected != self.parameters:
            raise ValueError(
                "linear deduction rule-weight addresses do not match its rules"
            )
        prepared = self.plan
        assert prepared.span is not None
        categories = set(prepared.span.categories)
        if (
            not isinstance(self.lexicon, tuple)
            or not self.lexicon
            or any(
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not all(isinstance(field, str) for field in entry)
                or (entry[1],) not in categories
                for entry in self.lexicon
            )
            or (self.start is not None and (self.start,) not in categories)
        ):
            raise ValueError(
                "linear deduction lexical categories do not match its rules"
            )

    @cached_property
    def plan(self) -> Plan:
        """Select the existing exact span executor for the checked recurrence.

        Returns
        -------
        Plan
            The rule and accuracy contract shared by successive runs.

        Raises
        ------
        ValueError
            If the presentation cannot use the checked linear span closure.
        """
        from quivers.qiec.linear_deduction import prepare_linear_goal
        from quivers.qiec.rules import RulePresentation

        assert isinstance(self.presentation, RulePresentation)
        return prepare_linear_goal(self.presentation)


class DeductionOutcomes(QiecModel):
    """A finite lexical LogProb chart returning typed weighted outcomes.

    The operation receives a token tensor, the corresponding lexical
    entries' category and optional form values in lexicon order, the log
    weights of those entries and the listed rule parameters, and the
    structural key and normalization choices. Lexical occurrences have
    positive width one. Ground rules may supply fixed ordered spans, including empty spans.
    Other rules concatenate contiguous spans, and any edge preserving
    width advances in an acyclic graph on category heads. The pair of
    span width and category rank increases strictly along every edge.

    Parameters
    ----------
    presentation
        The checked `RulePresentation`. Its model base in the annotation
        keeps the rule checker's dependency on effects acyclic at import.
    words
        One word per lexical row, preserving distinct occurrences.
    parameters
        Rule parameters supplied after the lexical weights, in order.
    span
        The name of the single lexical judgment and corresponding item
        constructor, whose fields are the endpoints, category, and any form.
    start
        The start category, or ``None`` to accept all full-span items.
    item_type
        The runtime constructor family for the span and its data fields.
    item_constructors
        Source names and constructor identities of that family.
    empty, entry
        The constructors of the result family. An entry has a key of
        ``item_type``, its log weight as ``Real``, and the remaining entries.
    """

    presentation: QiecModel = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: tuple[str, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[ParameterId, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    span: str
    start: str | None = None
    item_type: TypeApplication = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    item_constructors: tuple[tuple[str, ConstructorId], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    empty: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    entry: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["deduction-outcomes"] = "deduction-outcomes"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Verify that the carried presentation satisfies the finite span contract.

        Raises
        ------
        ValueError
            If the presentation, lexical slot, rule spans or weight address
            mapping fall outside this fragment.
        """
        from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
        from quivers.qiec.rules import RulePresentation

        presentation = self.presentation
        if not isinstance(presentation, RulePresentation):
            raise ValueError("deduction outcomes require a checked rule presentation")
        if presentation.conditions or presentation.equations:
            raise ValueError(
                "deduction outcomes require unconditional rules without proof equations"
            )
        if len(presentation.judgments) != 1 or len(presentation.axioms) != 1:
            raise ValueError("deduction outcomes require one lexical span judgment")
        (form,) = presentation.judgments
        (slot,) = presentation.axioms
        sorts = tuple(
            binder.sort for binder in form.indices if isinstance(binder, IndexBinder)
        )
        if (
            form.name != self.span
            or len(sorts) not in (3, 4)
            or sorts[:2] != (NAT, NAT)
            or not all(isinstance(sort, UserIndexSort) for sort in sorts[2:])
            or slot.judgment.family != form.id
            or tuple(field.type for field in slot.values) != (STRING,)
        ):
            raise ValueError("deduction outcomes require one lexical span judgment")
        bindings = tuple(
            index.name
            for index in slot.judgment.arguments
            if isinstance(index, IndexVariable)
        )
        if len(bindings) != len(sorts) or bindings != tuple(
            binder.name for binder in slot.telescope
        ):
            raise ValueError(
                "deduction outcome axioms must bind ordered span arguments"
            )
        names = dict(self.item_constructors)
        if (
            self.span not in names
            or len(names) != len(self.item_constructors)
            or len(set(names.values())) != len(names)
            or self.entry == self.empty
        ):
            raise ValueError("deduction outcomes require distinct typed constructors")
        if presentation.computed_weight_rules:
            raise ValueError("deduction outcomes cannot execute computed rule weights")
        parameters = {
            rule.weight
            for rule in presentation.rules
            if isinstance(rule.weight, ParameterId)
        }
        if (
            len(set(self.parameters)) != len(self.parameters)
            or set(self.parameters) != parameters
            or slot.weight is not None
        ):
            raise ValueError(
                "deduction outcomes require exactly the rule weight addresses"
            )
        _ = self.span_order

    @cached_property
    def span_order(self) -> SpanOrder:
        """Certify finite grounding of this presentation.

        Returns
        -------
        SpanOrder
            The strict endpoint-width and category-rank order.
        """
        from quivers.qiec._span_finiteness import certify_span_order
        from quivers.qiec.rules import RulePresentation

        assert isinstance(self.presentation, RulePresentation)
        return certify_span_order(self.presentation)

    @property
    def unary_height(self) -> int:
        """Measure chains using only unary transitions.

        Returns
        -------
        int
            The longest chain of unary category-head transitions.
        """
        return self.span_order.unary_height


class DeductionProofs(QiecModel):
    """Return supported full-span proofs under one exact proof budget.

    Each result entry binds its proof's indexed family arguments. The
    entry holds that proof, its log weight, and the next entry. The
    operation accepts a structural proof limit and a normalization
    choice. Different derivations keep their own weights.

    Parameters
    ----------
    chart
        The existing lexical chart descriptor and its result constructors.
        Its lexical span syntax and grammar parameters remain unchanged.
    """

    chart: DeductionOutcomes = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["deduction-proofs"] = "deduction-proofs"

    @cached_property
    def proof_presentation(self) -> RulePresentation:
        """Rename the chart's proof declarations into the calling module.

        Returns
        -------
        RulePresentation
            The same rule, axiom and judgment identities at renamed sorts,
            with proof constructors named by the deduction.
        """
        from quivers.qiec._lexical_proof_types import export_lexical_proofs
        from quivers.qiec.rules import RulePresentation

        presentation = self.chart.presentation
        assert isinstance(presentation, RulePresentation)
        return export_lexical_proofs(presentation, presentation.name)


class DeductionBooleanOutcomes(QiecModel):
    """Push finite lexical proof mass to a Boolean interpretation.

    An input truth table contains one axiom result per distinct lexical word
    in first occurrence order, followed by each rule's truth table in
    presentation order. Each rule table enumerates premise outcomes left to
    right, with false before true. An arity bound of two keeps every rule
    table at most four entries.

    Parameters
    ----------
    chart : DeductionOutcomes
        The shared weighted span presentation and its Boolean result row
        constructors.
    """

    chart: DeductionOutcomes = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["deduction-boolean-outcomes"] = "deduction-boolean-outcomes"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require ordinary nullary, unary or binary proof constructors.

        Raises
        ------
        ValueError
            If an operation needs rule data fields or more than two premises.
        """
        from quivers.qiec.rules import RulePresentation

        chart = self.chart
        if not isinstance(chart, DeductionOutcomes):
            raise ValueError("Boolean proof outcomes require a checked lexical chart")
        presentation = chart.presentation
        assert isinstance(presentation, RulePresentation)
        if any(rule.arity > 2 or rule.values for rule in presentation.rules):
            raise ValueError(
                "packed Boolean proof outcomes require rules of arity at most "
                "two without value fields"
            )

    @cached_property
    def lexical_words(self) -> tuple[str, ...]:
        """Count one axiom evaluation for every distinct lexical word.

        Returns
        -------
        tuple of str
            The words in first lexical occurrence order.
        """
        return tuple(dict.fromkeys(self.chart.words))

    @cached_property
    def table_size(self) -> int:
        """Count entries of the axiom and ordered rule tables.

        Returns
        -------
        int
            The length of the input truth vector.
        """
        presentation = self.chart.presentation
        from quivers.qiec.rules import RulePresentation

        assert isinstance(presentation, RulePresentation)
        return len(self.lexical_words) + sum(
            1 << rule.arity for rule in presentation.rules
        )


class DeductionOutcomeGrouping(QiecModel):
    """Group a typed list of finite log masses by exact structural equality.

    Parameters
    ----------
    key_type
        The checked type of each key. The module verifies that the type
        admits exact structural equality before the reducer can run.
    empty, entry
        The constructors of the typed row list. An entry contains a key,
        its log mass, and the rest of the list.
    """

    key_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    empty: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    entry: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["deduction-outcome-grouping"] = "deduction-outcome-grouping"


class DeductionOutcomeLaw(QiecModel):
    """Normalize checked deduction outcome rows as one finite sampleable.

    Parameters
    ----------
    layout : DeductionOutcomeGrouping
        The original key type and the two constructors of the weighted rows.
    """

    layout: DeductionOutcomeGrouping = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["deduction-outcome-law"] = "deduction-outcome-law"


type ForeignSemantics = (
    SemiringSum
    | SemiringProduct
    | Enumeration
    | Settlement
    | DeductionLinearGoal
    | DeductionOutcomes
    | DeductionProofs
    | DeductionBooleanOutcomes
    | DeductionOutcomeGrouping
    | DeductionOutcomeLaw
)
"""What a foreign handler computes."""


__all__ = [
    "DeductionBooleanOutcomes",
    "DeductionLinearGoal",
    "DeductionOutcomeGrouping",
    "DeductionOutcomeLaw",
    "DeductionOutcomes",
    "DeductionProofs",
    "EXACT_COUNT",
    "Enumeration",
    "EnumerationReduction",
    "ForeignSemantics",
    "InexactKernelValue",
    "KERNEL_ALGEBRAS",
    "KERNEL_CARRIERS",
    "KernelValue",
    "KernelSemiring",
    "MissingLaw",
    "PRODUCT_LAWS",
    "SUM_LAWS",
    "SemiringProduct",
    "SemiringSum",
    "Settlement",
]
