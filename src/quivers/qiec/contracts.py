"""Contract records: what a query asks for and what a run may claim.

A query names its observable, the laws it relies on, how recursion is
solved, how accurate the answer must be, what the work costs, and what
provenance is kept. Each of these is a record here, and every record
crosses the QIEC wire codec.

Three rules hold by construction and are checked again by
`report_refusals`:

- a resource budget is never an error bound: budgets are counts of work,
  and no guarantee accepts one as its bound;
- a residual is never a certificate: a run that stopped on a control
  claims a bound only through a certificate, and a certificate is an
  enclosure, a duality gap, an established argument, or a composition of
  certified parts;
- exhaustion never returns a value labelled exact.

`compose` is the composition law of accuracy guarantees: the composite
takes the weakest guarantee of its parts, and error bounds propagate
through declared Lipschitz constants. Local tolerances alone give a
heuristic composite. The proof is in the API page of this module.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import Enum
from types import MappingProxyType
from fractions import Fraction
from math import gcd, isfinite
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, HeldValue
from quivers.core.laws import (
    DIVISION,
    FOLD_LINEAR,
    NO_ZERO_DIVISORS,
    PLUS_SELECTIVE,
    SEMIMODULE_ACTION,
    SEMIRING_LAWS,
    STAR_LEAST_SOLUTION,
    SUPERIOR,
    ZERO_SUM_FREE,
    Citation,
    ClaimSubject,
    Law,
    LawClaim,
    Requirement,
    TieRule,
    requirements,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import ComputationId, FamilyId, SiteId, StableId
from quivers.qiec.types import TypeExpr


# ---------------------------------------------------------------------------
# identities


class PresentationId(StableId):
    """Stable identity of a rule presentation."""

    namespace = "presentation"


class InterpretationId(StableId):
    """Stable identity of an interpretation of a rule presentation."""

    namespace = "interpretation"


class OccurrenceId(StableId):
    """Stable identity of one occurrence of an axiom or a weight."""

    namespace = "occurrence"


class ParameterId(StableId):
    """Stable identity of a trainable parameter."""

    namespace = "parameter"


# ---------------------------------------------------------------------------
# exact numbers


class Rational(QiecModel):
    """An exact nonnegative or signed rational number in lowest terms.

    Bounds are exact so that adding them, as the composition law does,
    never rounds a bound down.

    Parameters
    ----------
    numerator : int
        The numerator.
    denominator : int
        The denominator, positive and coprime to the numerator.

    Raises
    ------
    didactic.api.ValidationError
        If the denominator is not positive or the fraction is not in
        lowest terms.
    """

    numerator: int
    denominator: int = 1

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require lowest terms with a positive denominator.

        Raises
        ------
        ValueError
            If the denominator is not positive or shares a factor with
            the numerator.
        """
        if self.denominator <= 0:
            raise ValueError("a rational needs a positive denominator")
        if gcd(self.numerator, self.denominator) != 1:
            raise ValueError(
                f"{self.numerator}/{self.denominator} is not in lowest terms"
            )

    @classmethod
    def of(cls, value: Fraction | int | float) -> Rational:
        """Build the rational equal to a number.

        Parameters
        ----------
        value : Fraction or int or float
            The number. A float converts exactly, digit for digit of its
            binary value.

        Returns
        -------
        Rational
            The same number in lowest terms.

        Raises
        ------
        ValueError
            If the value is a boolean or a float that is not finite.
        """
        if isinstance(value, bool):
            raise ValueError("a boolean is not a number here")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("a bound is finite; an infinite bound is no bound")
        exact = Fraction(value)
        return cls(numerator=exact.numerator, denominator=exact.denominator)

    @property
    def fraction(self) -> Fraction:
        """The number as a `fractions.Fraction`.

        Returns
        -------
        Fraction
            The same number.
        """
        return Fraction(self.numerator, self.denominator)


ZERO = Rational(numerator=0)
"""The rational zero."""


def _nonnegative(value: Rational, what: str) -> None:
    """Require a rational to be at least zero.

    Parameters
    ----------
    value : Rational
        The number.
    what : str
        What it is, named in the diagnostic.

    Raises
    ------
    ValueError
        If the number is negative.
    """
    if value.numerator < 0:
        raise ValueError(f"{what} must not be negative, got {value.fraction}")


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


# ---------------------------------------------------------------------------
# laws and their evidence


# ---------------------------------------------------------------------------
# derivative semantics


class Condition(QiecModel):
    """A hypothesis under which a result holds.

    Parameters
    ----------
    name : str
        A short name, unique among conditions.
    statement : str
        The hypothesis.
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
        _nonempty(self.name, "a condition's name")
        _nonempty(self.statement, "a condition's statement")


SPECTRAL_RADIUS_BELOW_ONE = Condition(
    name="spectral-radius-below-one",
    statement=(
        "the spectral radius of the Jacobian of the system at its least fixed point "
        "is below one, so the implicit derivative (I - DF)^-1 dF exists"
    ),
)
"""The hypothesis of the implicit derivative of a least fixed point."""


class Unbiased(QiecModel):
    """An estimate whose expectation is the quantity estimated.

    Parameters
    ----------
    tag : str
        Always ``"unbiased"``.
    """

    tag: Literal["unbiased"] = "unbiased"


class BiasBound(QiecModel):
    """An estimate whose bias is at most a declared positive bound.

    Parameters
    ----------
    bound : Rational
        The bound on the norm of the bias, positive.
    tag : str
        Always ``"bias-bound"``.
    """

    bound: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["bias-bound"] = "bias-bound"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a positive bound; a zero bound is `Unbiased`.

        Raises
        ------
        ValueError
            If the bound is not positive.
        """
        if self.bound.numerator <= 0:
            raise ValueError("a bias bound is positive; write Unbiased for zero")


class BiasUnknown(QiecModel):
    """An estimate with a bias of unknown size.

    Parameters
    ----------
    tag : str
        Always ``"bias-unknown"``.
    """

    tag: Literal["bias-unknown"] = "bias-unknown"


type Bias = Unbiased | BiasBound | BiasUnknown
"""What is known about the expectation of a derivative estimate."""


def _bias_rank(bias: Bias) -> tuple[int, Fraction]:
    """Order biases from strongest to weakest.

    Parameters
    ----------
    bias : Bias
        The bias.

    Returns
    -------
    tuple of int and Fraction
        A key that increases as the statement gets weaker: unbiased,
        then bounds by size, then unknown.
    """
    if isinstance(bias, Unbiased):
        return (0, Fraction(0))
    if isinstance(bias, BiasBound):
        return (1, bias.bound.fraction)
    return (2, Fraction(0))


class SiteEstimator(QiecModel):
    """The estimator that handles one random site under differentiation.

    Parameters
    ----------
    site : SiteId
        The site.
    estimator : str
        Its estimator, such as ``"enumerate"``, ``"score"``,
        ``"measure"`` or ``"reparam"``, or a declared estimator such as
        ``"straight-through"``.
    """

    site: SiteId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    estimator: str

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require an estimator name.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.estimator, "an estimator's name")


class DerivativeSemantics(QiecModel):
    """What a derivative is the derivative of, and how it was obtained.

    The record is a product of independent components, each ordered from
    strong to weak, and one semantics is at least as strong as another
    when it is in every component (`at_least`). The forms of a derivative
    report are its points:

    - exact at regular points (`exact`), almost everywhere when every
      primitive is analytic and every decision is a sign test;
    - a selected branch at ties (`selected_branch`), by one tie rule;
    - implicit (`implicit`), under stated conditions;
    - unrolled (`unrolled`): the derivative of the first ``k`` iterates
      and of nothing else;
    - an estimator (`estimated`) per random site, with its bias;
    - none (`none`), with the reason.

    A composite holds several at once, and `kind` names its weakest.

    Parameters
    ----------
    missing : tuple of str
        Reasons why there is no derivative. When any is given, every
        other component is at its strongest value.
    almost_everywhere : bool
        Whether the regular points have full measure.
    tie_rule : TieRule or None
        The rule of every selection in the computation, or ``None`` when
        it makes no selection.
    implicit_conditions : tuple of Condition
        Hypotheses of implicit derivatives, sorted and without repeats.
    unrolled : tuple of int
        Iteration counts of the unrolled loops the derivative passes
        through, sorted, one entry per loop.
    estimators : tuple of SiteEstimator
        The estimator of each random site, sorted, one per site.
    bias : Bias
        What is known about the expectation of the estimate.

    Raises
    ------
    didactic.api.ValidationError
        If a missing derivative carries other components, a loop has
        fewer than one iteration, a site has two estimators, or a bias
        is declared without an estimator.
    """

    missing: tuple[str, ...] = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    almost_everywhere: bool = False
    tie_rule: TieRule | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    implicit_conditions: tuple[Condition, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    unrolled: tuple[int, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    estimators: tuple[SiteEstimator, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    bias: Bias = dx.field(default=Unbiased(), opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Bring the components to their canonical order and check them.

        Raises
        ------
        ValueError
            If a missing derivative carries other components, a loop has
            fewer than one iteration, a site has two estimators, or a
            bias is declared without an estimator.
        """
        self._normalize_field("missing", tuple(sorted(set(self.missing))))
        self._normalize_field(
            "implicit_conditions",
            tuple(
                sorted(
                    set(self.implicit_conditions),
                    key=lambda item: (item.name, item.statement),
                )
            ),
        )
        self._normalize_field("unrolled", tuple(sorted(self.unrolled)))
        self._normalize_field(
            "estimators",
            tuple(
                sorted(
                    set(self.estimators),
                    key=lambda item: (item.site.digest, item.estimator),
                )
            ),
        )
        if any(not reason.strip() for reason in self.missing):
            raise ValueError("a missing derivative names its reason")
        if self.missing and (
            self.almost_everywhere
            or self.tie_rule is not None
            or self.implicit_conditions
            or self.unrolled
            or self.estimators
            or not isinstance(self.bias, Unbiased)
        ):
            raise ValueError("a missing derivative carries no other component")
        if any(count < 1 for count in self.unrolled):
            raise ValueError("an unrolled loop runs at least one iteration")
        sites = [item.site for item in self.estimators]
        if len(set(sites)) != len(sites):
            raise ValueError("a random site has one estimator")
        if not isinstance(self.bias, Unbiased) and not self.estimators:
            raise ValueError("only an estimator has a bias")

    @classmethod
    def exact(cls, *, almost_everywhere: bool = False) -> DerivativeSemantics:
        """The exact derivative at regular points.

        Parameters
        ----------
        almost_everywhere : bool
            Whether the regular points have full measure.

        Returns
        -------
        DerivativeSemantics
            The strongest semantics.
        """
        return cls(almost_everywhere=almost_everywhere)

    @classmethod
    def selected_branch(
        cls, rule: TieRule, *, almost_everywhere: bool = False
    ) -> DerivativeSemantics:
        """The exact derivative at regular points, and a rule's element at ties.

        Parameters
        ----------
        rule : TieRule
            The tie rule of every selection.
        almost_everywhere : bool
            Whether the regular points have full measure.

        Returns
        -------
        DerivativeSemantics
            The semantics.
        """
        return cls(tie_rule=rule, almost_everywhere=almost_everywhere)

    @classmethod
    def implicit(cls, *conditions: Condition) -> DerivativeSemantics:
        """The implicit derivative under hypotheses.

        Parameters
        ----------
        conditions : Condition
            The hypotheses.

        Returns
        -------
        DerivativeSemantics
            The semantics.
        """
        return cls(implicit_conditions=conditions)

    @classmethod
    def unrolled_iterations(cls, *iterations: int) -> DerivativeSemantics:
        """The derivative of unrolled loops and of nothing else.

        Parameters
        ----------
        iterations : int
            The iteration count of each loop.

        Returns
        -------
        DerivativeSemantics
            The semantics.
        """
        return cls(unrolled=iterations)

    @classmethod
    def estimated(
        cls, *estimators: SiteEstimator, bias: Bias = Unbiased()
    ) -> DerivativeSemantics:
        """An estimate of the derivative from estimators at random sites.

        Parameters
        ----------
        estimators : SiteEstimator
            The estimator of each site.
        bias : Bias
            What is known about the expectation of the estimate.

        Returns
        -------
        DerivativeSemantics
            The semantics.
        """
        return cls(estimators=estimators, bias=bias)

    @classmethod
    def none(cls, *reasons: str) -> DerivativeSemantics:
        """No derivative.

        Parameters
        ----------
        reasons : str
            Why there is none.

        Returns
        -------
        DerivativeSemantics
            The weakest semantics.

        Raises
        ------
        ValueError
            If no reason is given.
        """
        if not reasons:
            raise ValueError("a missing derivative names its reason")
        return cls(missing=reasons)

    @property
    def kind(self) -> str:
        """The weakest form the semantics holds.

        Returns
        -------
        str
            ``"none"``, ``"unrolled"``, ``"estimator"``, ``"implicit"``,
            ``"selected-branch"`` or ``"exact-at-regular-points"``, the
            first that applies.
        """
        if self.missing:
            return "none"
        if self.unrolled:
            return "unrolled"
        if self.estimators:
            return "estimator"
        if self.implicit_conditions:
            return "implicit"
        if self.tie_rule is not None:
            return "selected-branch"
        return "exact-at-regular-points"

    def at_least(self, request: DerivativeSemantics) -> bool:
        """Whether this semantics is at least as strong as a request.

        Parameters
        ----------
        request : DerivativeSemantics
            The weakest semantics that is acceptable.

        Returns
        -------
        bool
            True when the request asks for no derivative, or when in
            every component this one is at least as strong: no missing
            derivative, regular points of full measure if asked, a tie
            rule only if the request has the same one, and conditions,
            unrolled loops and estimators that the request admits, with a
            bias no larger than the request's.
        """
        if request.missing:
            return True
        if self.missing:
            return False
        if request.almost_everywhere and not self.almost_everywhere:
            return False
        if self.tie_rule is not None and self.tie_rule != request.tie_rule:
            return False
        if not set(self.implicit_conditions) <= set(request.implicit_conditions):
            return False
        if Counter(self.unrolled) - Counter(request.unrolled):
            return False
        if not set(self.estimators) <= set(request.estimators):
            return False
        return _bias_rank(self.bias) <= _bias_rank(request.bias)

    def meet(self, other: DerivativeSemantics) -> DerivativeSemantics:
        """The strongest semantics that both this one and another reach.

        Parameters
        ----------
        other : DerivativeSemantics
            The other semantics.

        Returns
        -------
        DerivativeSemantics
            The componentwise weakest: the reasons of both when either
            has none, regular points of full measure only if both have
            them, the tie rule of either, the union of conditions and of
            estimators, the larger count of each unrolled loop length,
            and the larger bias.

        Raises
        ------
        ValueError
            If the two use different tie rules, or estimate one site by
            two estimators. Neither has a meet: mixing tie rules makes a
            result depend on bracketing.
        """
        if self.missing or other.missing:
            return DerivativeSemantics.none(*self.missing, *other.missing)
        unrolled = Counter(self.unrolled) | Counter(other.unrolled)
        return DerivativeSemantics(
            almost_everywhere=self.almost_everywhere and other.almost_everywhere,
            tie_rule=_one_tie_rule(self.tie_rule, other.tie_rule),
            implicit_conditions=(*self.implicit_conditions, *other.implicit_conditions),
            unrolled=tuple(unrolled.elements()),
            estimators=_one_estimator_per_site(self.estimators, other.estimators),
            bias=max(self.bias, other.bias, key=_bias_rank),
        )


def _one_tie_rule(first: TieRule | None, second: TieRule | None) -> TieRule | None:
    """The tie rule of two parts that may each select.

    Parameters
    ----------
    first, second : TieRule or None
        The rules of the parts.

    Returns
    -------
    TieRule or None
        The rule of either part, or ``None`` when neither selects.

    Raises
    ------
    ValueError
        If the parts use different rules.
    """
    if first is not None and second is not None and first != second:
        raise ValueError(
            f"two tie rules, {first.name!r} and {second.name!r}, in one computation"
        )
    return first if first is not None else second


def _one_estimator_per_site(
    first: tuple[SiteEstimator, ...], second: tuple[SiteEstimator, ...]
) -> tuple[SiteEstimator, ...]:
    """The estimators of two parts, one per site.

    Parameters
    ----------
    first, second : tuple of SiteEstimator
        The estimators of the parts.

    Returns
    -------
    tuple of SiteEstimator
        Their union.

    Raises
    ------
    ValueError
        If one site has two estimators.
    """
    chosen: dict[SiteId, SiteEstimator] = {}
    for item in (*first, *second):
        known = chosen.setdefault(item.site, item)
        if known != item:
            raise ValueError(
                f"site {item.site} is estimated by {known.estimator!r} and "
                f"{item.estimator!r}"
            )
    return tuple(chosen.values())


class DerivativeEntry(QiecModel):
    """The derivative semantics of one output with respect to one input.

    Parameters
    ----------
    output : str
        The output.
    wrt : ParameterId or OccurrenceId
        The parameter or the occurrence differentiated.
    semantics : DerivativeSemantics
        What the derivative is.
    """

    output: str
    wrt: ParameterId | OccurrenceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    semantics: DerivativeSemantics = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


NOT_DIFFERENTIATED = DerivativeSemantics.none("not differentiated")
"""The semantics of a run that computes no derivative."""


class DerivativeReport(QiecModel):
    """The derivative semantics of every output and input of a run.

    Parameters
    ----------
    entries : tuple of DerivativeEntry
        One entry per output and input, without repeats.

    Raises
    ------
    didactic.api.ValidationError
        If an output and input pair has two entries.
    """

    entries: tuple[DerivativeEntry, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require one entry per output and input.

        Raises
        ------
        ValueError
            If a pair repeats.
        """
        pairs = [(entry.output, entry.wrt) for entry in self.entries]
        if len(set(pairs)) != len(pairs):
            raise ValueError("a derivative report has one entry per output and input")

    @property
    def summary(self) -> DerivativeSemantics:
        """The strongest semantics every entry reaches.

        Returns
        -------
        DerivativeSemantics
            The meet of the entries, or `NOT_DIFFERENTIATED` when there
            is none.
        """
        if not self.entries:
            return NOT_DIFFERENTIATED
        summary = self.entries[0].semantics
        for entry in self.entries[1:]:
            summary = summary.meet(entry.semantics)
        return summary


# ---------------------------------------------------------------------------
# metrics and guarantees


class AbsoluteError(QiecModel):
    """The distance ``|computed - target|``.

    Parameters
    ----------
    tag : str
        Always ``"absolute"``.
    """

    tag: Literal["absolute"] = "absolute"

    @property
    def triangle_inequality(self) -> bool:
        """Whether the triangle inequality holds.

        Returns
        -------
        bool
            True: it is a metric.
        """
        return True


class AbsoluteLogError(QiecModel):
    """The distance ``|log computed - log target|`` on positive weights.

    Since the logarithm is a semiring isomorphism from the nonnegative
    reals onto the log weights, a relative error of probabilities is an
    absolute error of log weights.

    Parameters
    ----------
    tag : str
        Always ``"absolute-log"``.
    """

    tag: Literal["absolute-log"] = "absolute-log"

    @property
    def triangle_inequality(self) -> bool:
        """Whether the triangle inequality holds.

        Returns
        -------
        bool
            True: it is a metric.
        """
        return True


class RelativeError(QiecModel):
    """The divergence ``|computed - target| / |target|``.

    It fails the triangle inequality: with target 1, a midpoint of 100
    and a computed value of 101, it is 100 against 0.01 plus 99.

    Parameters
    ----------
    tag : str
        Always ``"relative"``.
    """

    tag: Literal["relative"] = "relative"

    @property
    def triangle_inequality(self) -> bool:
        """Whether the triangle inequality holds.

        Returns
        -------
        bool
            False.
        """
        return False


class TotalVariation(QiecModel):
    """The total variation distance between two distributions.

    Parameters
    ----------
    tag : str
        Always ``"total-variation"``.
    """

    tag: Literal["total-variation"] = "total-variation"

    @property
    def triangle_inequality(self) -> bool:
        """Whether the triangle inequality holds.

        Returns
        -------
        bool
            True: it is a metric.
        """
        return True


class KLDivergence(QiecModel):
    """The Kullback-Leibler divergence in a stated direction.

    It fails the triangle inequality: for Bernoulli laws with means 0.1,
    0.5 and 0.9 it is 1.758 from the first to the third against 0.368
    plus 0.511 through the second.

    Parameters
    ----------
    direction : str
        ``"target-to-computed"`` for ``D(target || computed)`` or
        ``"computed-to-target"`` for ``D(computed || target)``.
    tag : str
        Always ``"kl"``.
    """

    direction: Literal["target-to-computed", "computed-to-target"]
    tag: Literal["kl"] = "kl"

    @property
    def triangle_inequality(self) -> bool:
        """Whether the triangle inequality holds.

        Returns
        -------
        bool
            False.
        """
        return False


class ObjectiveGap(QiecModel):
    """The gap between an objective at the computed point and its optimum.

    Parameters
    ----------
    tag : str
        Always ``"objective-gap"``.
    """

    tag: Literal["objective-gap"] = "objective-gap"

    @property
    def triangle_inequality(self) -> bool:
        """Whether the triangle inequality holds.

        Returns
        -------
        bool
            False: it compares objective values, not the points.
        """
        return False


type Metric = (
    AbsoluteError
    | AbsoluteLogError
    | RelativeError
    | TotalVariation
    | KLDivergence
    | ObjectiveGap
)
"""How the distance between a computed value and its target is measured.

Every metric is read with the computed value first and the target
second.
"""


class Enclosure(QiecModel):
    """An interval that holds both the target and the computed value.

    The endpoints come from outward rounded interval arithmetic, such as
    a lower iterate and a pre-fixed point of a monotone system. The bound
    it certifies is the width.

    Parameters
    ----------
    lower : Rational
        The lower endpoint.
    upper : Rational
        The upper endpoint, at least the lower one.
    conditions : tuple of str
        What the enclosure rests on.
    tag : str
        Always ``"enclosure"``.
    """

    lower: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    upper: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conditions: tuple[str, ...] = ()
    tag: Literal["enclosure"] = "enclosure"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require ordered endpoints.

        Raises
        ------
        ValueError
            If the upper endpoint is below the lower one.
        """
        if self.upper.fraction < self.lower.fraction:
            raise ValueError("an enclosure's upper endpoint is not below its lower")

    @property
    def bound(self) -> Fraction:
        """The bound the enclosure certifies.

        Returns
        -------
        Fraction
            The width of the interval.
        """
        return self.upper.fraction - self.lower.fraction


class DualityGap(QiecModel):
    """A primal value and a dual bound of an optimization.

    The optimum lies between them, so the gap bounds how far the primal
    value is from it.

    Parameters
    ----------
    primal : Rational
        The objective of a feasible point.
    dual : Rational
        A bound on the optimum, on the other side.
    conditions : tuple of str
        What the dual bound rests on, such as exact local solves.
    tag : str
        Always ``"duality-gap"``.
    """

    primal: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    dual: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conditions: tuple[str, ...] = ()
    tag: Literal["duality-gap"] = "duality-gap"

    @property
    def bound(self) -> Fraction:
        """The bound the gap certifies.

        Returns
        -------
        Fraction
            ``|dual - primal|``.
        """
        return abs(self.dual.fraction - self.primal.fraction)


class Argument(QiecModel):
    """A bound that an established claim proves.

    Parameters
    ----------
    value : Rational
        The bound, at least zero.
    claim : LawClaim
        The theorem that gives the bound, such as an a posteriori bound
        of a contraction with a proved rate. It must be established.
    conditions : tuple of str
        The hypotheses of the theorem that the run checked.
    tag : str
        Always ``"argument"``.
    """

    value: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    claim: LawClaim = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conditions: tuple[str, ...] = ()
    tag: Literal["argument"] = "argument"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative bound.

        Raises
        ------
        ValueError
            If the bound is negative.
        """
        _nonnegative(self.value, "an argument's bound")

    @property
    def bound(self) -> Fraction:
        """The bound the argument certifies.

        Returns
        -------
        Fraction
            The stated bound.
        """
        return self.value.fraction


class ComposedTerm(QiecModel):
    """One part of a composition: its bound and its Lipschitz constant.

    Parameters
    ----------
    lipschitz : Rational
        The constant of the operation in the part's argument.
    bound : Rational
        The part's bound.
    """

    lipschitz: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    bound: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require nonnegative numbers.

        Raises
        ------
        ValueError
            If either is negative.
        """
        _nonnegative(self.lipschitz, "a Lipschitz constant")
        _nonnegative(self.bound, "a part's bound")


class Composition(QiecModel):
    """A bound obtained by the composition law from certified parts.

    Parameters
    ----------
    operation : str
        The operation that combines the parts.
    terms : tuple of ComposedTerm
        The parts that are not exact.
    conditions : tuple of str
        The conditions of the parts and of the constants.
    tag : str
        Always ``"composition"``.
    """

    operation: str
    terms: tuple[ComposedTerm, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conditions: tuple[str, ...] = ()
    tag: Literal["composition"] = "composition"

    @property
    def bound(self) -> Fraction:
        """The bound the composition certifies.

        Returns
        -------
        Fraction
            The sum of each constant times its part's bound.
        """
        return sum(
            (term.lipschitz.fraction * term.bound.fraction for term in self.terms),
            Fraction(0),
        )


type Certificate = Enclosure | DualityGap | Argument | Composition
"""Evidence for a certified bound. A residual is not one of them."""


class Exact(QiecModel):
    """The mathematical value, up to the rounding stated in `Numerics`.

    Parameters
    ----------
    tag : str
        Always ``"exact"``.
    """

    tag: Literal["exact"] = "exact"


class Certified(QiecModel):
    """A bound that holds on every run, up to the stated rounding.

    In a contract it is the requested bound and carries no certificate.
    In a report it is the attained bound and must carry the certificate
    that shows it.

    Parameters
    ----------
    bound : Rational
        The bound on the distance to the target, at least zero.
    conditions : tuple of str
        The hypotheses under which it holds.
    certificate : Certificate or None
        The evidence, in a report.
    tag : str
        Always ``"certified"``.
    """

    bound: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conditions: tuple[str, ...] = ()
    certificate: Certificate | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["certified"] = "certified"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative bound.

        Raises
        ------
        ValueError
            If the bound is negative.
        """
        _nonnegative(self.bound, "a certified bound")


class Probabilistic(QiecModel):
    """A bound that fails with probability at most ``delta``.

    Parameters
    ----------
    epsilon : Rational
        The bound, at least zero.
    delta : Rational
        The probability of failure, strictly between zero and one.
    assumptions : tuple of str
        What the probability rests on.
    tag : str
        Always ``"probabilistic"``.
    """

    epsilon: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    delta: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    assumptions: tuple[str, ...] = ()
    tag: Literal["probabilistic"] = "probabilistic"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative bound and a probability in (0, 1).

        Raises
        ------
        ValueError
            If the bound is negative, or ``delta`` is zero (a certified
            bound is the right record) or at least one (no statement).
        """
        _nonnegative(self.epsilon, "a probabilistic bound")
        if not 0 < self.delta.fraction < 1:
            raise ValueError("the failure probability lies strictly between 0 and 1")


class Heuristic(QiecModel):
    """No guarantee.

    Parameters
    ----------
    reason : str
        Why there is none.
    tag : str
        Always ``"heuristic"``.
    """

    reason: str
    tag: Literal["heuristic"] = "heuristic"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the reason.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.reason, "the reason of a heuristic")


type Guarantee = Exact | Certified | Probabilistic | Heuristic
"""What is promised about the distance between a result and its target."""


def _guarantee_meets(attained: Guarantee, requested: Guarantee) -> bool:
    """Whether an attained guarantee is at least a requested one.

    Parameters
    ----------
    attained : Guarantee
        What a run reached.
    requested : Guarantee
        What its contract asks for.

    Returns
    -------
    bool
        Exact meets everything; a certified bound meets a larger or equal
        certified or probabilistic bound; a probabilistic bound meets one
        with a larger or equal bound and failure probability whose
        assumptions include its own; everything meets a heuristic.
    """
    if isinstance(requested, Heuristic) or isinstance(attained, Exact):
        return True
    if isinstance(requested, Exact):
        return False
    if isinstance(attained, Certified):
        limit = (
            requested.bound if isinstance(requested, Certified) else requested.epsilon
        )
        return attained.bound.fraction <= limit.fraction
    if isinstance(attained, Probabilistic) and isinstance(requested, Probabilistic):
        return (
            attained.epsilon.fraction <= requested.epsilon.fraction
            and attained.delta.fraction <= requested.delta.fraction
            and set(attained.assumptions) <= set(requested.assumptions)
        )
    return False


# ---------------------------------------------------------------------------
# the accuracy contract


class Numerics(QiecModel):
    """Precision of a computation.

    Parameters
    ----------
    dtype : str
        The element type.
    domain : str
        ``"linear"`` or ``"log"`` weights.
    reduction : str
        ``"plain"`` or ``"compensated"`` summation.
    rounding : Rational or None
        A bound on the rounding error in the target's metric, from the
        operation count and the condition number, or ``None`` when no
        bound is known. In a contract it is the largest acceptable.
    """

    dtype: Literal["bool", "int64", "float16", "bfloat16", "float32", "float64"]
    domain: Literal["linear", "log"] = "linear"
    reduction: Literal["plain", "compensated"] = "plain"
    rounding: Rational | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative rounding bound.

        Raises
        ------
        ValueError
            If the bound is negative.
        """
        if self.rounding is not None:
            _nonnegative(self.rounding, "a rounding bound")


class Target(QiecModel):
    """The mathematical quantity a result approximates, and its metric.

    Parameters
    ----------
    quantity : str
        The quantity, such as the observable of the query or the least
        fixed point of its system.
    metric : Metric
        How distance to it is measured.
    """

    quantity: str
    metric: Metric = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the quantity.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.quantity, "a target's quantity")


class Control(QiecModel):
    """A method parameter. It is recorded and never read as a guarantee.

    Parameters
    ----------
    name : str
        Its name, such as ``"beam"``, ``"damping"`` or ``"tolerance"``.
    value : Rational or str
        Its value.
    """

    name: str
    value: Rational | str = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a name.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.name, "a control's name")


class ResourceBudget(QiecModel):
    """Limits on the work of a run. A budget is never an error bound.

    Each limit is ``None`` when there is none.

    Parameters
    ----------
    iterations : int or None
        Iterations of a solver.
    scalar_work : int or None
        Scalar sums, products and divisions.
    states : int or None
        Product states created.
    elements : int or None
        Elements live at once.
    bytes : int or None
        Bytes live at once.
    seconds : Rational or None
        Wall time.
    """

    iterations: int | None = None
    scalar_work: int | None = None
    states: int | None = None
    elements: int | None = None
    bytes: int | None = None
    seconds: Rational | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require nonnegative limits.

        Raises
        ------
        ValueError
            If a limit is negative.
        """
        for name in ("iterations", "scalar_work", "states", "elements", "bytes"):
            limit = getattr(self, name)
            if limit is not None and limit < 0:
                raise ValueError(f"the {name} budget must not be negative")
        if self.seconds is not None:
            _nonnegative(self.seconds, "the time budget")


UNLIMITED = ResourceBudget()
"""A budget without limits."""


class FailurePolicy(str, Enum):
    """What a run does when it cannot meet its contract.

    Neither policy truncates silently, switches method, or returns zero.
    """

    REFUSE = "refuse"
    RETURN_UNESTABLISHED = "return-unestablished"


class AccuracyContract(QiecModel):
    """How accurate a result must be, and what happens when it is not.

    The fields are separate on purpose: a method parameter is not a
    guarantee, a budget is not an error bound, and the derivative has
    its own semantics.

    Parameters
    ----------
    numerics : Numerics
        Precision.
    target : Target
        The quantity and its metric.
    guarantee : Guarantee
        The weakest acceptable guarantee.
    controls : tuple of Control
        Method parameters.
    budget : ResourceBudget
        Limits on work.
    on_failure : FailurePolicy
        Refuse, or return the value wrapped as unestablished.
    gradient : DerivativeSemantics
        The weakest acceptable derivative semantics, or
        `NOT_DIFFERENTIATED`.
    """

    numerics: Numerics = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    target: Target = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    guarantee: Guarantee = dx.field(
        default=Exact(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    controls: tuple[Control, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    budget: ResourceBudget = dx.field(
        default=UNLIMITED, opaque=True, extras=OPAQUE_BY_VALUE
    )
    on_failure: FailurePolicy = dx.field(
        default=FailurePolicy.REFUSE, opaque=True, extras=OPAQUE_BY_VALUE
    )
    gradient: DerivativeSemantics = dx.field(
        default=NOT_DIFFERENTIATED, opaque=True, extras=OPAQUE_BY_VALUE
    )


# ---------------------------------------------------------------------------
# recursion


class Acyclic(QiecModel):
    """The derived hypergraph has no cycle, found by parent counts.

    The solution is unique, and one relaxation per hyperedge computes it.

    Parameters
    ----------
    tag : str
        Always ``"acyclic"``.
    """

    tag: Literal["acyclic"] = "acyclic"

    @property
    def finite_method(self) -> bool:
        """Whether a finite procedure computes the exact value.

        Returns
        -------
        bool
            True.
        """
        return True

    @property
    def finite_derivations(self) -> bool:
        """Whether every judgment has finitely many derivations.

        Returns
        -------
        bool
            True.
        """
        return True

    def laws(self) -> tuple[Law, ...]:
        """The laws the solution needs beyond the observable's.

        Returns
        -------
        tuple of Law
            None.
        """
        return ()

    def admits(self, gradient: DerivativeSemantics) -> bool:
        """Whether a requested derivative is available.

        Parameters
        ----------
        gradient : DerivativeSemantics
            The request.

        Returns
        -------
        bool
            True: reverse accumulation of the computation is exact.
        """
        return True


class LinearCyclic(QiecModel):
    """Each strongly connected component has at most one premise inside it.

    The least solution is ``A* b`` in an algebra with a star.

    Parameters
    ----------
    tag : str
        Always ``"linear-cyclic"``.
    """

    tag: Literal["linear-cyclic"] = "linear-cyclic"

    @property
    def finite_method(self) -> bool:
        """Whether a finite procedure computes the exact value.

        Returns
        -------
        bool
            True: elimination computes the star.
        """
        return True

    @property
    def finite_derivations(self) -> bool:
        """Whether every judgment has finitely many derivations.

        Returns
        -------
        bool
            False.
        """
        return False

    def laws(self) -> tuple[Law, ...]:
        """The laws the solution needs beyond the observable's.

        Returns
        -------
        tuple of Law
            The star law.
        """
        return (STAR_LEAST_SOLUTION,)

    def admits(self, gradient: DerivativeSemantics) -> bool:
        """Whether a requested derivative is available.

        Parameters
        ----------
        gradient : DerivativeSemantics
            The request.

        Returns
        -------
        bool
            True: the derivative of ``A*`` is ``A* dA A*`` on the open
            domain of convergence.
        """
        return True


class MonotonePolynomial(QiecModel):
    """A nonnegative polynomial system ``x = f(x)`` with a least fixed point.

    Kleene and Newton iterates increase to the least fixed point, and a
    pre-fixed point bounds it from above. No finite procedure computes it
    exactly in general, so the result is at best certified.

    Parameters
    ----------
    tag : str
        Always ``"monotone-polynomial"``.
    """

    tag: Literal["monotone-polynomial"] = "monotone-polynomial"

    @property
    def finite_method(self) -> bool:
        """Whether a finite procedure computes the exact value.

        Returns
        -------
        bool
            False.
        """
        return False

    @property
    def finite_derivations(self) -> bool:
        """Whether every judgment has finitely many derivations.

        Returns
        -------
        bool
            False.
        """
        return False

    def laws(self) -> tuple[Law, ...]:
        """The laws the solution needs beyond the observable's.

        Returns
        -------
        tuple of Law
            None of their own: the least fixed point lies in the
            omega-continuous completion of the weights, and the executor
            that solves the system states the laws it uses, an embedding in
            the nonnegative reals for certified Newton or an idempotent sum
            with a star for exact Newton.
        """
        return ()

    def admits(self, gradient: DerivativeSemantics) -> bool:
        """Whether a requested derivative is available.

        Parameters
        ----------
        gradient : DerivativeSemantics
            The request.

        Returns
        -------
        bool
            True when the request asks for no derivative, admits the
            implicit derivative under its spectral condition, or admits
            an unrolled one.
        """
        return (
            bool(gradient.missing)
            or SPECTRAL_RADIUS_BELOW_ONE in gradient.implicit_conditions
            or bool(gradient.unrolled)
        )


class Bounded(QiecModel):
    """The model restricts its derivations by a bound that is part of it.

    A bound on constructor depth, length or height changes the model and
    its name; an execution budget does not.

    Parameters
    ----------
    measure : str
        ``"constructor-depth"``, ``"length"`` or ``"height"``.
    limit : int
        The bound, at least zero.
    tag : str
        Always ``"bounded"``.
    """

    measure: Literal["constructor-depth", "length", "height"]
    limit: int
    tag: Literal["bounded"] = "bounded"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative bound.

        Raises
        ------
        ValueError
            If the bound is negative.
        """
        if self.limit < 0:
            raise ValueError("a model bound is at least zero")

    @property
    def finite_method(self) -> bool:
        """Whether a finite procedure computes the exact value.

        Returns
        -------
        bool
            True.
        """
        return True

    @property
    def finite_derivations(self) -> bool:
        """Whether every judgment has finitely many derivations.

        Returns
        -------
        bool
            True.
        """
        return True

    def laws(self) -> tuple[Law, ...]:
        """The laws the solution needs beyond the observable's.

        Returns
        -------
        tuple of Law
            None.
        """
        return ()

    def admits(self, gradient: DerivativeSemantics) -> bool:
        """Whether a requested derivative is available.

        Parameters
        ----------
        gradient : DerivativeSemantics
            The request.

        Returns
        -------
        bool
            True: the bounded model is finite and acyclic.
        """
        return True


class Unrolled(QiecModel):
    """The first ``iterations`` iterates of a recursive system.

    In an ordered algebra with increasing iterates they bound the
    solution from below. They are not the solution, and their derivative
    is the derivative of the iterates and of nothing else.

    Parameters
    ----------
    iterations : int
        How many iterates, at least one.
    tag : str
        Always ``"unrolled"``.
    """

    iterations: int
    tag: Literal["unrolled"] = "unrolled"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require at least one iterate.

        Raises
        ------
        ValueError
            If there is none.
        """
        if self.iterations < 1:
            raise ValueError("an unrolled system runs at least one iterate")

    @property
    def finite_method(self) -> bool:
        """Whether a finite procedure computes the exact value.

        Returns
        -------
        bool
            False: the iterates are not the solution.
        """
        return False

    @property
    def finite_derivations(self) -> bool:
        """Whether every judgment has finitely many derivations.

        Returns
        -------
        bool
            False.
        """
        return False

    def laws(self) -> tuple[Law, ...]:
        """The laws the solution needs beyond the observable's.

        Returns
        -------
        tuple of Law
            None.
        """
        return ()

    def admits(self, gradient: DerivativeSemantics) -> bool:
        """Whether a requested derivative is available.

        Parameters
        ----------
        gradient : DerivativeSemantics
            The request.

        Returns
        -------
        bool
            True when the request asks for no derivative or admits a loop
            unrolled this many times.
        """
        return bool(gradient.missing) or self.iterations in gradient.unrolled


type Recursion = Acyclic | LinearCyclic | MonotonePolynomial | Bounded | Unrolled
"""How a recursive system is solved, and what that solution is."""


# ---------------------------------------------------------------------------
# retention


class NoRetention(QiecModel):
    """Keep no provenance beyond the values.

    Parameters
    ----------
    tag : str
        Always ``"none"``.
    """

    tag: Literal["none"] = "none"

    @property
    def keeps_forest(self) -> bool:
        """Whether the packed forest is kept.

        Returns
        -------
        bool
            False.
        """
        return False


class ParentCounts(QiecModel):
    """Keep a count of pending parents per item, and no edges.

    Parameters
    ----------
    tag : str
        Always ``"parent-counts"``.
    """

    tag: Literal["parent-counts"] = "parent-counts"

    @property
    def keeps_forest(self) -> bool:
        """Whether the packed forest is kept.

        Returns
        -------
        bool
            False.
        """
        return False


class JudgmentPattern(QiecModel):
    """The judgments of one family whose evidence is kept.

    Parameters
    ----------
    family : FamilyId
        The family of the judgments.
    """

    family: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class RequestedEvidence(QiecModel):
    """Keep the evidence of the judgments that match patterns.

    Parameters
    ----------
    patterns : tuple of JudgmentPattern
        The judgments, at least one pattern and none repeated.
    tag : str
        Always ``"requested"``.
    """

    patterns: tuple[JudgmentPattern, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    tag: Literal["requested"] = "requested"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require distinct patterns.

        Raises
        ------
        ValueError
            If there is none or one repeats.
        """
        if not self.patterns:
            raise ValueError("requested evidence names at least one pattern")
        if len(set(self.patterns)) != len(self.patterns):
            raise ValueError("a pattern of requested evidence repeats")

    @property
    def keeps_forest(self) -> bool:
        """Whether the packed forest is kept.

        Returns
        -------
        bool
            False.
        """
        return False


class PackedForest(QiecModel):
    """Keep the derived hypergraph with occurrences and grounded instances.

    Its tree unfoldings from a judgment correspond one to one to the
    derivations of the judgment when it is acyclic, so every linear fold
    can be computed from it.

    Parameters
    ----------
    tag : str
        Always ``"packed-forest"``.
    """

    tag: Literal["packed-forest"] = "packed-forest"

    @property
    def keeps_forest(self) -> bool:
        """Whether the packed forest is kept.

        Returns
        -------
        bool
            True.
        """
        return True


type Retention = NoRetention | ParentCounts | RequestedEvidence | PackedForest
"""Which provenance a run keeps beside the autograd tape."""


# ---------------------------------------------------------------------------
# observables


class _Observable(QiecModel):
    """Properties shared by every observable.

    An observable is a map from the detailed meaning of a judgment to
    its result. The properties below say what it needs and gives; the
    defaults are those of a semiring observable.
    """

    @property
    def differentiable(self) -> bool:
        """Whether the result has a derivative in the weights.

        Returns
        -------
        bool
            True.
        """
        return True

    @property
    def selects(self) -> bool:
        """Whether the result selects among alternatives.

        Returns
        -------
        bool
            False.
        """
        return False

    @property
    def needs_forest(self) -> bool:
        """Whether the result needs the packed forest.

        Returns
        -------
        bool
            False.
        """
        return False

    @property
    def needs_finite_derivations(self) -> bool:
        """Whether a guaranteed result needs finitely many derivations.

        Returns
        -------
        bool
            False.
        """
        return False

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws of the algebra.
        """
        return requirements(SEMIRING_LAWS, algebra)


class Support(_Observable):
    """Whether the detailed meaning has a derivation of nonzero weight.

    Support is a homomorphism to the Boolean algebra when no sum of
    nonzero weights and no product of nonzero weights is zero.

    Parameters
    ----------
    tag : str
        Always ``"support"``.
    """

    tag: Literal["support"] = "support"

    @property
    def differentiable(self) -> bool:
        """Whether the result has a derivative in the weights.

        Returns
        -------
        bool
            False: it is a truth value.
        """
        return False

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws, no zero sums and no zero divisors.
        """
        return requirements((*SEMIRING_LAWS, ZERO_SUM_FREE, NO_ZERO_DIVISORS), algebra)


class Weight(_Observable):
    """The sum of the weights of all derivations.

    Parameters
    ----------
    tag : str
        Always ``"weight"``.
    """

    tag: Literal["weight"] = "weight"


class Best(_Observable):
    """The best derivation's weight and the derivation.

    Parameters
    ----------
    tag : str
        Always ``"best"``.
    """

    tag: Literal["best"] = "best"

    @property
    def selects(self) -> bool:
        """Whether the result selects among alternatives.

        Returns
        -------
        bool
            True.
        """
        return True

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws and a selective sum.
        """
        return requirements((*SEMIRING_LAWS, PLUS_SELECTIVE), algebra)


class KBest(_Observable):
    """The ``k`` best derivations.

    Parameters
    ----------
    k : int
        How many, at least one.
    tag : str
        Always ``"k-best"``.
    """

    k: int
    tag: Literal["k-best"] = "k-best"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require at least one derivation.

        Raises
        ------
        ValueError
            If ``k`` is below one.
        """
        if self.k < 1:
            raise ValueError("k-best asks for at least one derivation")

    @property
    def selects(self) -> bool:
        """Whether the result selects among alternatives.

        Returns
        -------
        bool
            True.
        """
        return True

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws, a selective sum and superior rules.
        """
        return requirements((*SEMIRING_LAWS, PLUS_SELECTIVE, SUPERIOR), algebra)


class Outcome(_Observable):
    """The weight of each outcome: the pushforward along a key.

    Parameters
    ----------
    key : ComputationId
        The computation that maps a meaning to its outcome, with a
        decidable equality on outcomes.
    tag : str
        Always ``"outcome"``.
    """

    key: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["outcome"] = "outcome"


class Statistic(_Observable):
    """The weight and the expectation of a statistic, in a semimodule.

    Parameters
    ----------
    statistic : ComputationId
        The computation of the statistic of a derivation.
    tag : str
        Always ``"statistic"``.
    """

    statistic: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["statistic"] = "statistic"

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws and the semimodule action.
        """
        return requirements((*SEMIRING_LAWS, SEMIMODULE_ACTION), algebra)


class Marginal(_Observable):
    """The derivative of the weight in each occurrence weight.

    Parameters
    ----------
    normalized : bool
        Whether the marginals are divided by the total weight.
    tag : str
        Always ``"marginal"``.
    """

    normalized: bool = False
    tag: Literal["marginal"] = "marginal"

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws, and division when normalized.
        """
        laws = (*SEMIRING_LAWS, DIVISION) if self.normalized else SEMIRING_LAWS
        return requirements(laws, algebra)


class Sample(_Observable):
    """Derivations drawn in proportion to their weight.

    Parameters
    ----------
    n : int
        How many draws, at least one.
    tag : str
        Always ``"sample"``.
    """

    n: int
    tag: Literal["sample"] = "sample"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require at least one draw.

        Raises
        ------
        ValueError
            If ``n`` is below one.
        """
        if self.n < 1:
            raise ValueError("a sample has at least one draw")

    @property
    def differentiable(self) -> bool:
        """Whether the result has a derivative in the weights.

        Returns
        -------
        bool
            False: a draw has none. The gradient of an expectation is a
            statistic with an estimator.
        """
        return False

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws and division, which normalizing needs.
        """
        return requirements((*SEMIRING_LAWS, DIVISION), algebra)


class Fold(_Observable):
    """A fold of the detailed meaning that is linear in the weights.

    Parameters
    ----------
    fold : ComputationId
        The fold.
    tag : str
        Always ``"fold"``.
    """

    fold: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["fold"] = "fold"

    @property
    def needs_forest(self) -> bool:
        """Whether the result needs the packed forest.

        Returns
        -------
        bool
            True.
        """
        return True

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            The semiring laws, and linearity of the fold.
        """
        return (
            *requirements(SEMIRING_LAWS, algebra),
            Requirement(
                law=FOLD_LINEAR,
                subject=ClaimSubject(kind="computation", name=str(self.fold)),
            ),
        )


class NonlinearFold(_Observable):
    """Any fold of the detailed meaning, linear or not.

    No aggregate summarizes it in general: the alternatives -1 and 1 sum
    to 0 and have squares summing to 2. It needs every derivation, so a
    guaranteed result needs finitely many and the packed forest.

    Parameters
    ----------
    fold : ComputationId
        The fold.
    tag : str
        Always ``"nonlinear-fold"``.
    """

    fold: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["nonlinear-fold"] = "nonlinear-fold"

    @property
    def needs_forest(self) -> bool:
        """Whether the result needs the packed forest.

        Returns
        -------
        bool
            True.
        """
        return True

    @property
    def needs_finite_derivations(self) -> bool:
        """Whether a guaranteed result needs finitely many derivations.

        Returns
        -------
        bool
            True.
        """
        return True


class Forest(_Observable):
    """The packed forest itself.

    Parameters
    ----------
    tag : str
        Always ``"forest"``.
    """

    tag: Literal["forest"] = "forest"

    @property
    def differentiable(self) -> bool:
        """Whether the result has a derivative in the weights.

        Returns
        -------
        bool
            False: it is a structure.
        """
        return False

    @property
    def needs_forest(self) -> bool:
        """Whether the result needs the packed forest.

        Returns
        -------
        bool
            True.
        """
        return True

    def requirements(self, algebra: ClaimSubject) -> tuple[Requirement, ...]:
        """The laws the result needs.

        Parameters
        ----------
        algebra : ClaimSubject
            The weight algebra.

        Returns
        -------
        tuple of Requirement
            None: the forest is the derivations, not their weights.
        """
        return ()


type Observable = (
    Support
    | Weight
    | Best
    | KBest
    | Outcome
    | Statistic
    | Marginal
    | Sample
    | Fold
    | NonlinearFold
    | Forest
)
"""What a query observes of the detailed meaning of a judgment."""


class QuerySpec(QiecModel):
    """What a query asks of a presentation under an interpretation.

    Parameters
    ----------
    presentation : PresentationId
        The rule presentation.
    interpretation : InterpretationId
        The interpretation of its rules.
    observable : Observable
        What is observed.
    input_types : tuple of TypeExpr
        The types of the inputs.
    differentiate_with_respect_to : tuple of ParameterId or OccurrenceId
        The parameters and occurrences differentiated, in the order of
        the gradient, without repeats.
    batch_axes : tuple of str
        Names of batch axes, without repeats.
    retention : Retention
        The provenance kept.

    Raises
    ------
    didactic.api.ValidationError
        If a differentiated input or a batch axis repeats.
    """

    presentation: PresentationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    interpretation: InterpretationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    observable: Observable = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    input_types: tuple[TypeExpr, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    differentiate_with_respect_to: tuple[ParameterId | OccurrenceId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    batch_axes: tuple[str, ...] = ()
    retention: Retention = dx.field(
        default=NoRetention(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require distinct inputs of the gradient and distinct axes.

        Raises
        ------
        ValueError
            If either repeats.
        """
        wrt = self.differentiate_with_respect_to
        if len(set(wrt)) != len(wrt):
            raise ValueError("an input of the gradient repeats")
        if len(set(self.batch_axes)) != len(self.batch_axes):
            raise ValueError("a batch axis repeats")


# ---------------------------------------------------------------------------
# complexity


class Phase(str, Enum):
    """A phase of a computation whose work is bounded and counted apart."""

    CONSTRUCTION = "construction"
    EXECUTION = "execution"
    DIFFERENTIATION = "differentiation"
    REFIT = "refit"


class ComputationModel(QiecModel):
    """The model of computation a bound is stated in.

    Parameters
    ----------
    name : str
        A short name.
    description : str
        What costs one unit.
    """

    name: str
    description: str

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a name and a description.

        Raises
        ------
        ValueError
            If either is empty.
        """
        _nonempty(self.name, "a model's name")
        _nonempty(self.description, "a model's description")


ALGEBRA_OPERATIONS = ComputationModel(
    name="algebra-operations",
    description=(
        "one unit per scalar operation of the weight algebra, counted inside "
        "tensor kernels"
    ),
)
"""Algebra operations at unit cost on scalars."""

WORD_RAM_EXPECTED_HASHING = ComputationModel(
    name="word-ram-expected-hashing",
    description="word RAM for indices, with hashing in expected constant time",
)
"""A word RAM with expected constant time hashing."""


class ComplexityParameter(QiecModel):
    """A named size that bounds are stated in.

    Parameters
    ----------
    name : str
        An identifier, such as ``V``, ``E`` or ``width``.
    meaning : str
        What it counts.
    """

    name: str
    meaning: str

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require an identifier that is not a function name, and a meaning.

        Raises
        ------
        ValueError
            If the name is not an identifier or is a known function, or
            the meaning is empty.
        """
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError(f"{self.name!r} is not an identifier")
        if self.name in BOUND_FUNCTIONS:
            raise ValueError(f"{self.name!r} names a function of bounds")
        _nonempty(self.meaning, "a parameter's meaning")


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
"""An identifier in a bound."""

BOUND_FUNCTIONS = frozenset(
    {"log", "min", "max", "sqrt", "exp", "poly", "polylog", "BM", "omega"}
)
"""Function and constant names a bound may use besides its parameters.

``BM(n)`` is the cost of a Boolean matrix product and ``omega`` the
exponent of matrix multiplication.
"""


class ProvedOptimal(QiecModel):
    """The bound is optimal: a lower bound matches it in a stated model.

    Parameters
    ----------
    lower_bound : str
        The lower bound, in the parameters of the claim, without the
        asymptotic symbol. It must equal the bound's expression.
    model : ComputationModel
        The model of the lower bound.
    sources : tuple of Citation
        Where it is proved, at least one.
    tag : str
        Always ``"proved-optimal"``.
    """

    lower_bound: str
    model: ComputationModel = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sources: tuple[Citation, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["proved-optimal"] = "proved-optimal"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the lower bound and a source.

        Raises
        ------
        ValueError
            If either is missing.
        """
        _nonempty(self.lower_bound, "a lower bound")
        if not self.sources:
            raise ValueError("an optimality proof cites its source")


class Conditional(QiecModel):
    """The bound is optimal if a stated hypothesis holds.

    Parameters
    ----------
    hypothesis : str
        The hypothesis, such as a conjecture of fine-grained complexity.
    sources : tuple of Citation
        Where the conditional lower bound is proved, at least one.
    tag : str
        Always ``"conditional"``.
    """

    hypothesis: str
    sources: tuple[Citation, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["conditional"] = "conditional"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the hypothesis and a source.

        Raises
        ------
        ValueError
            If either is missing.
        """
        _nonempty(self.hypothesis, "a hypothesis")
        if not self.sources:
            raise ValueError("a conditional bound cites its source")


class BestKnown(QiecModel):
    """The bound is the best published, with no matching lower bound.

    Parameters
    ----------
    sources : tuple of Citation
        Where it is published, at least one.
    tag : str
        Always ``"best-known"``.
    """

    sources: tuple[Citation, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["best-known"] = "best-known"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a source.

        Raises
        ------
        ValueError
            If there is none.
        """
        if not self.sources:
            raise ValueError("a best known bound cites its source")


class Unresolved(QiecModel):
    """Whether the bound can be improved is open: a research obligation.

    Parameters
    ----------
    obligation : str
        What remains to be shown.
    tag : str
        Always ``"unresolved"``.
    """

    obligation: str
    tag: Literal["unresolved"] = "unresolved"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the obligation.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.obligation, "an obligation")


class Deficient(QiecModel):
    """A better bound is known, and a reviewed reason keeps this one.

    Parameters
    ----------
    better_bound : str
        The better bound, in the parameters of the claim.
    justification : str
        Why this bound is kept, and in which regime of parameters.
    reviewed_by : str
        Who reviewed the justification.
    sources : tuple of Citation
        Where the better bound is published, at least one.
    tag : str
        Always ``"deficient"``.
    """

    better_bound: str
    justification: str
    reviewed_by: str
    sources: tuple[Citation, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["deficient"] = "deficient"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the better bound, the justification, a reviewer and a source.

        Raises
        ------
        ValueError
            If any is missing.
        """
        _nonempty(self.better_bound, "a better bound")
        _nonempty(self.justification, "a justification")
        _nonempty(self.reviewed_by, "a reviewer")
        if not self.sources:
            raise ValueError("a deficient bound cites the better one")


type OptimalityStatus = ProvedOptimal | Conditional | BestKnown | Unresolved | Deficient
"""How a bound compares with what is known to be possible."""


class ComplexityBound(QiecModel):
    """An upper bound on one resource of one phase.

    Parameters
    ----------
    phase : Phase
        The phase.
    resource : str
        ``"time"`` or ``"space"``.
    expression : str
        The bound in the parameters of the claim, without the asymptotic
        symbol: ``V + E`` stands for ``O(V + E)``.
    status : OptimalityStatus
        How it compares with what is known.
    """

    phase: Phase = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    resource: Literal["time", "space"]
    expression: str
    status: OptimalityStatus = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require an expression.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.expression, "a bound")


class PhaseInapplicable(QiecModel):
    """A phase that the query class does not have.

    Parameters
    ----------
    phase : Phase
        The phase.
    reason : str
        Why it does not apply.
    """

    phase: Phase = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    reason: str

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the reason.

        Raises
        ------
        ValueError
            If it is empty.
        """
        _nonempty(self.reason, "the reason a phase does not apply")


class ComplexityClaim(QiecModel):
    """Bounds on the work of a query class, with their optimality status.

    Parameters
    ----------
    query : str
        The query class.
    parameters : tuple of ComplexityParameter
        The sizes the bounds are stated in.
    model : ComputationModel
        The model of the bounds.
    bounds : tuple of ComplexityBound
        A time and a space bound for every phase that applies.
    inapplicable : tuple of PhaseInapplicable
        The phases that do not apply.
    conditions : tuple of str
        What the bounds need of the algebra and the input.
    """

    query: str
    parameters: tuple[ComplexityParameter, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    model: ComputationModel = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    bounds: tuple[ComplexityBound, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    inapplicable: tuple[PhaseInapplicable, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    conditions: tuple[str, ...] = ()

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require the query and distinct parameter names.

        Raises
        ------
        ValueError
            If the query is empty or a parameter name repeats.
        """
        _nonempty(self.query, "a claim's query class")
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("a parameter of a complexity claim repeats")


# ---------------------------------------------------------------------------
# work


class WorkField(str, Enum):
    """A counter of a work report."""

    ITERATIONS = "iterations"
    SCALAR_SUMS = "scalar_sums"
    SCALAR_PRODUCTS = "scalar_products"
    SCALAR_DIVISIONS = "scalar_divisions"
    STAR_PIVOTS = "star_pivots"
    EXACT_OPERATIONS = "exact_operations"
    KEY_COMPARISONS = "key_comparisons"
    HASH_OPERATIONS = "hash_operations"
    PRODUCT_STATES = "product_states"
    ARCS_VISITED = "arcs_visited"
    ELEMENTS_ALLOCATED = "elements_allocated"
    PEAK_LIVE_ELEMENTS = "peak_live_elements"
    PEAK_BYTES = "peak_bytes"
    ELEMENTS_SAVED_FOR_BACKWARD = "elements_saved_for_backward"
    PROVENANCE_RECORDS = "provenance_records"
    NODES_VALIDATED = "nodes_validated"
    NODES_INSTANTIATED = "nodes_instantiated"
    CACHE_HITS = "cache_hits"
    CACHE_INVALIDATIONS = "cache_invalidations"


def measuring(fields: Iterable[WorkField]) -> tuple[WorkField, ...]:
    """Counters as a work report lists them: in the order of `WorkField`, each once.

    Parameters
    ----------
    fields : Iterable of WorkField
        The counters.

    Returns
    -------
    tuple of WorkField
        The listing.
    """
    chosen = set(fields)
    return tuple(field for field in WorkField if field in chosen)


class WorkReport(QiecModel):
    """Counters of the work of one phase of a run.

    Counts are elements inside tensor operations, not kernel launches.

    Parameters
    ----------
    phase : Phase
        The phase counted.
    iterations : int
        Solver iterations.
    scalar_sums : int
        Scalar sums.
    scalar_products : int
        Scalar products.
    scalar_divisions : int
        Scalar divisions outside closures of one element.
    star_pivots : int
        Pivots of star computations: closures of one element, each one
        operation of the algebra, whatever it computes inside.
    exact_operations : int
        Rational or interval updates of an exact or enclosed decision,
        run when rounding alone cannot decide.
    key_comparisons : int
        Comparisons of keys.
    hash_operations : int
        Hash computations and lookups.
    product_states : int
        Product states created.
    arcs_visited : int
        Arcs visited.
    elements_allocated : int
        Elements allocated.
    peak_live_elements : int
        Elements live at once, at most.
    peak_bytes : int
        Bytes live at once, at most.
    elements_saved_for_backward : int
        Elements saved for the backward pass.
    provenance_records : int
        Provenance records kept.
    nodes_validated : int
        Graph nodes validated.
    nodes_instantiated : int
        Graph nodes instantiated.
    cache_hits : int
        Cache hits.
    cache_invalidations : int
        Cache invalidations.
    seconds : Rational or None
        Wall time, when measured.
    measured : tuple of WorkField
        The counters the producer counted, in the order of `WorkField`,
        each once; `measuring` builds it. A counter outside it is not a
        count of zero but no count at all, so a cost claim is never
        checked against it.
    """

    phase: Phase = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    iterations: int = 0
    scalar_sums: int = 0
    scalar_products: int = 0
    scalar_divisions: int = 0
    star_pivots: int = 0
    exact_operations: int = 0
    key_comparisons: int = 0
    hash_operations: int = 0
    product_states: int = 0
    arcs_visited: int = 0
    elements_allocated: int = 0
    peak_live_elements: int = 0
    peak_bytes: int = 0
    elements_saved_for_backward: int = 0
    provenance_records: int = 0
    nodes_validated: int = 0
    nodes_instantiated: int = 0
    cache_hits: int = 0
    cache_invalidations: int = 0
    seconds: Rational | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    measured: tuple[WorkField, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require nonnegative counters, each nonzero one measured.

        Raises
        ------
        ValueError
            If a counter or the time is negative, a counter is nonzero
            without being measured, or the measured counters are not in
            order or repeat.
        """
        if self.measured != measuring(self.measured):
            raise ValueError("measured counters are listed once each, in order")
        for name in _COUNTERS:
            if getattr(self, name) < 0:
                raise ValueError(f"the counter {name} must not be negative")
            if getattr(self, name) and WorkField(name) not in self.measured:
                raise ValueError(f"the counter {name} is nonzero but not measured")
        if self.seconds is not None:
            _nonnegative(self.seconds, "a measured time")

    @property
    def scalar_work(self) -> int:
        """Scalar sums, products and divisions.

        Returns
        -------
        int
            Their total.
        """
        return self.scalar_sums + self.scalar_products + self.scalar_divisions


_COUNTERS = (
    "iterations",
    "scalar_sums",
    "scalar_products",
    "scalar_divisions",
    "star_pivots",
    "exact_operations",
    "key_comparisons",
    "hash_operations",
    "product_states",
    "arcs_visited",
    "elements_allocated",
    "peak_live_elements",
    "peak_bytes",
    "elements_saved_for_backward",
    "provenance_records",
    "nodes_validated",
    "nodes_instantiated",
    "cache_hits",
    "cache_invalidations",
)
"""The integer counters of a work report."""


class UnmeasuredWork(ValueError):
    """A cost claim bounds a counter the work report did not count.

    Parameters
    ----------
    field : WorkField
        The counter.
    reason : str
        What was being checked.
    """

    def __init__(self, field: WorkField, reason: str) -> None:
        super().__init__(f"{field.value} is not measured: {reason}")
        self.field = field
        self.reason = reason


TIME_FIELDS: frozenset[WorkField] = frozenset(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.SCALAR_DIVISIONS,
        WorkField.STAR_PIVOTS,
        WorkField.EXACT_OPERATIONS,
        WorkField.KEY_COMPARISONS,
        WorkField.HASH_OPERATIONS,
    }
)
"""The counters whose sum is the time of the algebra-operations model."""

SPACE_FIELDS: Mapping[Phase, frozenset[WorkField]] = MappingProxyType(
    {
        Phase.CONSTRUCTION: frozenset({WorkField.PEAK_LIVE_ELEMENTS}),
        Phase.EXECUTION: frozenset({WorkField.PEAK_LIVE_ELEMENTS}),
        Phase.DIFFERENTIATION: frozenset(
            {WorkField.PEAK_LIVE_ELEMENTS, WorkField.ELEMENTS_SAVED_FOR_BACKWARD}
        ),
        Phase.REFIT: frozenset({WorkField.PEAK_LIVE_ELEMENTS}),
    }
)
"""The counters whose largest is the space of a phase in the model."""


def bounded_fields(bound: ComplexityBound) -> frozenset[WorkField]:
    """The counters a bound of the algebra-operations model bounds.

    Parameters
    ----------
    bound : ComplexityBound
        The bound.

    Returns
    -------
    frozenset of WorkField
        The time counters for a time bound, the space counters of its
        phase for a space bound.
    """
    if bound.resource == "time":
        return TIME_FIELDS
    return SPACE_FIELDS[bound.phase]


def measured_total(bound: ComplexityBound, report: WorkReport) -> int:
    """The measured quantity a bound bounds: the time spent, or the space held.

    Parameters
    ----------
    bound : ComplexityBound
        The bound.
    report : WorkReport
        The work of the bound's phase.

    Returns
    -------
    int
        The sum of the time counters, or the largest space counter.

    Raises
    ------
    UnmeasuredWork
        If a counter the bound bounds is not measured.
    ValueError
        If the report is of another phase.
    """
    if report.phase is not bound.phase:
        raise ValueError(
            f"a {bound.phase.value} bound is checked against {report.phase.value} work"
        )
    fields = bounded_fields(bound)
    for field in sorted(fields, key=lambda item: item.value):
        if field not in report.measured:
            raise UnmeasuredWork(
                field,
                f"the {bound.phase.value} {bound.resource} bound {bound.expression}",
            )
    values = [getattr(report, field.value) for field in fields]
    return sum(values) if bound.resource == "time" else max(values)


def _exceeded(budget: ResourceBudget, work: tuple[WorkReport, ...]) -> tuple[str, ...]:
    """The limits of a budget that the work of a run went past.

    Parameters
    ----------
    budget : ResourceBudget
        The limits.
    work : tuple of WorkReport
        The work of each phase.

    Returns
    -------
    tuple of str
        The names of the limits exceeded. Counts add over phases; live
        elements and bytes are compared phase by phase; time counts only
        when every phase measured it.
    """
    totals = {
        "iterations": sum(report.iterations for report in work),
        "scalar_work": sum(report.scalar_work for report in work),
        "states": sum(report.product_states for report in work),
        "elements": max((report.peak_live_elements for report in work), default=0),
        "bytes": max((report.peak_bytes for report in work), default=0),
    }
    exceeded = [
        name
        for name, total in totals.items()
        if (limit := getattr(budget, name)) is not None and total > limit
    ]
    times = [report.seconds for report in work]
    if budget.seconds is not None and work and all(t is not None for t in times):
        spent = sum((t.fraction for t in times if t is not None), Fraction(0))
        if spent > budget.seconds.fraction:
            exceeded.append("seconds")
    return tuple(exceeded)


# ---------------------------------------------------------------------------
# attainment and composition


class Attainment(QiecModel):
    """What a result reached: its guarantee, metric, rounding and derivative.

    Parameters
    ----------
    guarantee : Guarantee
        The guarantee reached. A certified bound carries its certificate.
    metric : Metric
        The metric of the guarantee.
    rounding : Rational or None
        A bound on the rounding error in the metric, or ``None`` when
        unknown.
    derivative : DerivativeSemantics
        The derivative reached.
    """

    guarantee: Guarantee = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    metric: Metric = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rounding: Rational | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    derivative: DerivativeSemantics = dx.field(
        default=NOT_DIFFERENTIATED, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative rounding bound.

        Raises
        ------
        ValueError
            If it is negative.
        """
        if self.rounding is not None:
            _nonnegative(self.rounding, "a rounding bound")

    @property
    def total_bound(self) -> Fraction | None:
        """The bound with rounding included, when both parts are known.

        Returns
        -------
        Fraction or None
            The certified or exact bound plus the rounding bound, or
            ``None`` when the guarantee has no bound that holds on every
            run or the rounding is unknown.
        """
        if self.rounding is None:
            return None
        if isinstance(self.guarantee, Exact):
            return self.rounding.fraction
        if isinstance(self.guarantee, Certified):
            return self.guarantee.bound.fraction + self.rounding.fraction
        return None


class ArgumentBound(QiecModel):
    """What an operation promises about one of its arguments.

    Parameters
    ----------
    metric : Metric
        The metric of the argument the constant is stated in.
    lipschitz : Rational or None
        A constant ``L`` with ``d(g(.., x, ..), g(.., y, ..)) <= L d(x, y)``
        for arguments that differ only here, on a region that holds the
        computed and the target values; ``None`` when there is none.
    affine : bool
        Whether the operation is affine in this argument, which keeps
        unbiased estimates unbiased.
    """

    metric: Metric = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    lipschitz: Rational | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    affine: bool = False

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative constant.

        Raises
        ------
        ValueError
            If it is negative.
        """
        if self.lipschitz is not None:
            _nonnegative(self.lipschitz, "a Lipschitz constant")


class OperationBound(QiecModel):
    """What an operation that combines results promises about them.

    Parameters
    ----------
    name : str
        The operation.
    arguments : tuple of ArgumentBound
        One per argument.
    metric : Metric
        The metric of the result.
    rounding : Rational or None
        A bound on the operation's own rounding error, or ``None``.
    derivative : DerivativeSemantics
        The derivative of the operation itself, such as a tie rule for a
        maximum or none for a threshold.
    conditions : tuple of str
        Where the constants hold.
    """

    name: str
    arguments: tuple[ArgumentBound, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    metric: Metric = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rounding: Rational | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    derivative: DerivativeSemantics = dx.field(
        default=DerivativeSemantics(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    conditions: tuple[str, ...] = ()

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a name and a nonnegative rounding bound.

        Raises
        ------
        ValueError
            If the name is empty or the bound negative.
        """
        _nonempty(self.name, "an operation's name")
        if self.rounding is not None:
            _nonnegative(self.rounding, "a rounding bound")


def _conditions(guarantee: Guarantee) -> tuple[str, ...]:
    """The hypotheses a guarantee rests on.

    Parameters
    ----------
    guarantee : Guarantee
        The guarantee.

    Returns
    -------
    tuple of str
        Its conditions or assumptions; none for an exact or heuristic one.
    """
    if isinstance(guarantee, Certified):
        return guarantee.conditions
    if isinstance(guarantee, Probabilistic):
        return guarantee.assumptions
    return ()


def _error_bound(guarantee: Guarantee) -> Fraction:
    """The bound of a guarantee that has one.

    Parameters
    ----------
    guarantee : Guarantee
        An exact, certified or probabilistic guarantee.

    Returns
    -------
    Fraction
        Zero for exact, the bound for certified, ``epsilon`` for
        probabilistic.

    Raises
    ------
    ValueError
        If the guarantee is heuristic.
    """
    if isinstance(guarantee, Exact):
        return Fraction(0)
    if isinstance(guarantee, Certified):
        return guarantee.bound.fraction
    if isinstance(guarantee, Probabilistic):
        return guarantee.epsilon.fraction
    raise ValueError("a heuristic has no bound")


def _ordered_union(groups: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    """The distinct strings of several tuples, in first appearance order.

    Parameters
    ----------
    groups : Iterable of tuple of str
        The tuples.

    Returns
    -------
    tuple of str
        Their union.
    """
    return tuple(dict.fromkeys(item for group in groups for item in group))


def compose(parts: tuple[Attainment, ...], operation: OperationBound) -> Attainment:
    """The attainment of an operation applied to results.

    This is the composition law: the guarantee of the composite is the
    weakest of its parts, and a bound propagates through each part's
    Lipschitz constant. A part with a bound but no constant in its own
    metric makes the composite heuristic, and so does a heuristic part
    unless the operation is constant in it.

    Parameters
    ----------
    parts : tuple of Attainment
        The attainment of each argument.
    operation : OperationBound
        What the operation promises.

    Returns
    -------
    Attainment
        The composite. Its bound is ``sum_i L_i e_i`` over the parts that
        are not exact, its rounding ``r + sum_i L_i r_i`` when every term
        is known, its failure probability the sum of the parts', and its
        derivative the weakest of the parts' and the operation's, with an
        unbiased estimate kept only through affine arguments.

    Raises
    ------
    ValueError
        If the number of parts is not the number of arguments, or the
        parts use different tie rules or estimate one site twice.
    """
    if len(parts) != len(operation.arguments):
        raise ValueError(
            f"{operation.name} takes {len(operation.arguments)} arguments, "
            f"got {len(parts)}"
        )
    pairs = tuple(zip(parts, operation.arguments, strict=True))
    usable = tuple(
        argument.lipschitz is not None and part.metric == argument.metric
        for part, argument in pairs
    )
    triangle = operation.metric.triangle_inequality
    derivative = _composed_derivative(parts, operation)
    rounding = None
    if (
        triangle
        and operation.rounding is not None
        and all(usable)
        and all(part.rounding is not None for part in parts)
    ):
        rounding = Rational.of(
            operation.rounding.fraction
            + sum(
                (_lipschitz(argument) * _rounding(part) for part, argument in pairs),
                Fraction(0),
            )
        )
    guarantee = _composed_guarantee(pairs, usable, operation, triangle)
    return Attainment(
        guarantee=guarantee,
        metric=operation.metric,
        rounding=rounding,
        derivative=derivative,
    )


def _lipschitz(argument: ArgumentBound) -> Fraction:
    """The Lipschitz constant of an argument that has one.

    Parameters
    ----------
    argument : ArgumentBound
        The argument.

    Returns
    -------
    Fraction
        The constant.

    Raises
    ------
    ValueError
        If there is none.
    """
    if argument.lipschitz is None:
        raise ValueError("the argument has no Lipschitz constant")
    return argument.lipschitz.fraction


def _rounding(part: Attainment) -> Fraction:
    """The rounding bound of a part that has one.

    Parameters
    ----------
    part : Attainment
        The part.

    Returns
    -------
    Fraction
        The bound.

    Raises
    ------
    ValueError
        If it is unknown.
    """
    if part.rounding is None:
        raise ValueError("the part has no rounding bound")
    return part.rounding.fraction


def _composed_guarantee(
    pairs: tuple[tuple[Attainment, ArgumentBound], ...],
    usable: tuple[bool, ...],
    operation: OperationBound,
    triangle: bool,
) -> Guarantee:
    """The guarantee of a composite, by the composition law.

    Parameters
    ----------
    pairs : tuple of tuple of Attainment and ArgumentBound
        Each part with what the operation promises about it.
    usable : tuple of bool
        Whether each part has a constant in its own metric.
    operation : OperationBound
        The operation.
    triangle : bool
        Whether the result's metric satisfies the triangle inequality.

    Returns
    -------
    Guarantee
        Exact when every part is exact. Otherwise heuristic when a part
        cannot be bounded through the operation, when the result's metric
        has no triangle inequality, or when the failure probabilities sum
        to one or more; else certified when no part is probabilistic, and
        probabilistic when one is.
    """
    if all(isinstance(part.guarantee, Exact) for part, _ in pairs):
        return Exact()
    if not triangle:
        return Heuristic(
            reason=(
                f"the metric of {operation.name} has no triangle inequality, so "
                "the errors of its parts do not add"
            )
        )
    terms: list[ComposedTerm] = []
    delta = Fraction(0)
    for (part, argument), bounded in zip(pairs, usable, strict=True):
        guarantee = part.guarantee
        if isinstance(guarantee, Exact):
            continue
        if not bounded:
            return Heuristic(
                reason=(
                    f"{operation.name} has no Lipschitz constant in the "
                    "metric of a part that is not exact"
                )
            )
        constant = _lipschitz(argument)
        if isinstance(guarantee, Heuristic):
            if constant != 0:
                return Heuristic(
                    reason=f"a part of {operation.name} is heuristic: {guarantee.reason}"
                )
            continue
        if isinstance(guarantee, Probabilistic):
            delta += guarantee.delta.fraction
        terms.append(
            ComposedTerm(
                lipschitz=Rational.of(constant),
                bound=Rational.of(_error_bound(guarantee)),
            )
        )
    conditions = _ordered_union(
        (*(_conditions(part.guarantee) for part, _ in pairs), operation.conditions)
    )
    certificate = Composition(
        operation=operation.name, terms=tuple(terms), conditions=conditions
    )
    bound = Rational.of(certificate.bound)
    if delta == 0:
        return Certified(bound=bound, conditions=conditions, certificate=certificate)
    if delta >= 1:
        return Heuristic(
            reason=(
                f"the failure probabilities of the parts of {operation.name} sum "
                f"to {delta}, so the union bound says nothing"
            )
        )
    return Probabilistic(
        epsilon=bound, delta=Rational.of(delta), assumptions=conditions
    )


def _composed_derivative(
    parts: tuple[Attainment, ...], operation: OperationBound
) -> DerivativeSemantics:
    """The derivative of a composite.

    Parameters
    ----------
    parts : tuple of Attainment
        The parts.
    operation : OperationBound
        The operation.

    Returns
    -------
    DerivativeSemantics
        The weakest of the parts' derivatives and the operation's, with
        the unrolled loops of all parts counted together. The estimate is
        unbiased only when every estimated part is unbiased and enters an
        affine argument; otherwise its bias is unknown.

    Raises
    ------
    ValueError
        If the parts use different tie rules or estimate one site twice.
    """
    semantics = [part.derivative for part in parts] + [operation.derivative]
    missing = tuple(reason for item in semantics for reason in item.missing)
    if missing:
        return DerivativeSemantics.none(*missing)
    tie_rule: TieRule | None = None
    estimators: tuple[SiteEstimator, ...] = ()
    for item in semantics:
        tie_rule = _one_tie_rule(tie_rule, item.tie_rule)
        estimators = _one_estimator_per_site(estimators, item.estimators)
    unbiased = all(
        isinstance(part.derivative.bias, Unbiased)
        and (not part.derivative.estimators or argument.affine)
        for part, argument in zip(parts, operation.arguments, strict=True)
    ) and isinstance(operation.derivative.bias, Unbiased)
    return DerivativeSemantics(
        almost_everywhere=all(item.almost_everywhere for item in semantics),
        tie_rule=tie_rule,
        implicit_conditions=tuple(
            condition for item in semantics for condition in item.implicit_conditions
        ),
        unrolled=tuple(count for item in semantics for count in item.unrolled),
        estimators=estimators,
        bias=Unbiased() if unbiased or not estimators else BiasUnknown(),
    )


# ---------------------------------------------------------------------------
# reports


class Completed(QiecModel):
    """The method ran to its end.

    Parameters
    ----------
    tag : str
        Always ``"completed"``.
    """

    tag: Literal["completed"] = "completed"

    @property
    def early(self) -> bool:
        """Whether the run stopped before its method ended.

        Returns
        -------
        bool
            False.
        """
        return False


class BudgetExhausted(QiecModel):
    """The run stopped because a resource limit was reached.

    Parameters
    ----------
    resource : str
        The limit, a field of `ResourceBudget`.
    limit : Rational
        Its value.
    tag : str
        Always ``"budget-exhausted"``.
    """

    resource: Literal[
        "iterations", "scalar_work", "states", "elements", "bytes", "seconds"
    ]
    limit: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["budget-exhausted"] = "budget-exhausted"

    @property
    def early(self) -> bool:
        """Whether the run stopped before its method ended.

        Returns
        -------
        bool
            True.
        """
        return True


class ControlReached(QiecModel):
    """The run stopped because a control was reached, such as a residual.

    A residual below a tolerance is a diagnostic, not a certificate.

    Parameters
    ----------
    control : str
        The control.
    value : Rational
        The diagnostic when it stopped.
    tag : str
        Always ``"control-reached"``.
    """

    control: str
    value: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["control-reached"] = "control-reached"

    @property
    def early(self) -> bool:
        """Whether the run stopped before its method ended.

        Returns
        -------
        bool
            True.
        """
        return True


type StopReason = Completed | BudgetExhausted | ControlReached
"""Why a run stopped."""


def attained_guarantee(
    stop: StopReason, method: Guarantee, certificate: Certificate | None
) -> Guarantee:
    """The guarantee a run may claim.

    Parameters
    ----------
    stop : StopReason
        Why the run stopped.
    method : Guarantee
        The guarantee of the method when it runs to its end.
    certificate : Certificate or None
        A certificate the run computed, if any.

    Returns
    -------
    Guarantee
        The method's guarantee when it completed. A run that stopped
        early claims the bound of its certificate, and without one it
        claims nothing: neither the budget nor the residual is a bound.
    """
    if not stop.early:
        return method
    if certificate is not None:
        return Certified(
            bound=Rational.of(certificate.bound),
            conditions=certificate.conditions,
            certificate=certificate,
        )
    return Heuristic(reason=f"stopped early by {stop.tag} without a certificate")


class ContractField(str, Enum):
    """A requirement of an accuracy contract that a run can fail."""

    TARGET = "target"
    GUARANTEE = "guarantee"
    NUMERICS = "numerics"
    GRADIENT = "gradient"
    BUDGET = "budget"


def unmet_fields(
    contract: AccuracyContract, attainment: Attainment, work: tuple[WorkReport, ...]
) -> tuple[ContractField, ...]:
    """The requirements of a contract that a run did not meet.

    Parameters
    ----------
    contract : AccuracyContract
        The contract.
    attainment : Attainment
        What the run reached.
    work : tuple of WorkReport
        Its work.

    Returns
    -------
    tuple of ContractField
        In the order of `ContractField`: the target's metric if it is
        another one, the guarantee if it is weaker, the rounding if it is
        unknown or above the contract's, the derivative if it is weaker,
        and the budget if the work exceeded it.
    """
    unmet: list[ContractField] = []
    if attainment.metric != contract.target.metric:
        unmet.append(ContractField.TARGET)
    if not _guarantee_meets(attainment.guarantee, contract.guarantee):
        unmet.append(ContractField.GUARANTEE)
    limit = contract.numerics.rounding
    if limit is not None and (
        attainment.rounding is None or attainment.rounding.fraction > limit.fraction
    ):
        unmet.append(ContractField.NUMERICS)
    if not attainment.derivative.at_least(contract.gradient):
        unmet.append(ContractField.GRADIENT)
    if _exceeded(contract.budget, work):
        unmet.append(ContractField.BUDGET)
    return tuple(unmet)


class AccuracyReport(QiecModel):
    """What a run reached, why it stopped, and what it cost.

    A derivative report of a run is the ``derivatives`` field, and its
    summary is the derivative of the attainment.

    Parameters
    ----------
    attainment : Attainment
        What the result reached.
    stop : StopReason
        Why the run stopped.
    unmet : tuple of ContractField
        The requirements not met.
    derivatives : DerivativeReport
        The derivative of each output and input.
    work : tuple of WorkReport
        The work of each phase, one report per phase.
    claims : tuple of LawClaim
        Every law claim the run used.

    Raises
    ------
    didactic.api.ValidationError
        If a phase has two work reports, or the attainment's derivative
        is not the summary of the derivative report.
    """

    attainment: Attainment = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    stop: StopReason = dx.field(
        default=Completed(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    unmet: tuple[ContractField, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    derivatives: DerivativeReport = dx.field(
        default=DerivativeReport(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    work: tuple[WorkReport, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    claims: tuple[LawClaim, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require one work report per phase and a consistent derivative.

        Raises
        ------
        ValueError
            If a phase repeats or the summary of the derivative report
            is not the attainment's derivative.
        """
        phases = [report.phase for report in self.work]
        if len(set(phases)) != len(phases):
            raise ValueError("a phase has one work report")
        if self.derivatives.summary != self.attainment.derivative:
            raise ValueError(
                "the attainment's derivative is the summary of the derivative report"
            )


class Unestablished(QiecModel):
    """A value whose contract was not met, with the report that says why.

    It is not a weight, a tensor or a number, and nothing that expects
    one accepts it.

    Parameters
    ----------
    value : HeldValue
        The value the run computed.
    report : AccuracyReport
        What it reached, with the unmet requirements.
    """

    value: HeldValue = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    report: AccuracyReport = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class ContractNotMet(ValueError):
    """A run did not meet its contract, and the contract says to refuse.

    Parameters
    ----------
    report : AccuracyReport
        What the run reached.
    """

    def __init__(self, report: AccuracyReport) -> None:
        names = ", ".join(field.value for field in report.unmet)
        super().__init__(f"the contract is not met: {names}")
        self.report = report


def settle[T](
    contract: AccuracyContract, value: T, report: AccuracyReport
) -> T | Unestablished:
    """Apply a contract's failure policy to a run's result.

    Parameters
    ----------
    contract : AccuracyContract
        The contract.
    value : T
        The value the run computed.
    report : AccuracyReport
        Its report.

    Returns
    -------
    T or Unestablished
        The value when the report meets the contract; otherwise the value
        wrapped as unestablished, when the contract says to return it.

    Raises
    ------
    InadmissibleReport
        If the report is inconsistent with the contract.
    ContractNotMet
        If the contract is not met and says to refuse.
    """
    found = report_refusals(contract, report)
    if found:
        raise InadmissibleReport(found)
    if not report.unmet:
        return value
    if contract.on_failure is FailurePolicy.REFUSE:
        raise ContractNotMet(report)
    return Unestablished(value=value, report=report)


# ---------------------------------------------------------------------------
# admissibility


class RefusalCode(str, Enum):
    """Why a combination of contract records is inadmissible."""

    MISSING_CLAIM = "missing-claim"
    CLAIM_NOT_ESTABLISHED = "claim-not-established"
    EXACT_WITHOUT_FINITE_METHOD = "exact-without-finite-method"
    DERIVATIVE_UNAVAILABLE = "derivative-unavailable"
    GRADIENT_WITHOUT_INPUTS = "gradient-without-inputs"
    INPUTS_WITHOUT_GRADIENT = "inputs-without-gradient"
    OBSERVABLE_NOT_DIFFERENTIABLE = "observable-not-differentiable"
    SELECTION_WITHOUT_TIE_RULE = "selection-without-tie-rule"
    FOREST_NOT_RETAINED = "forest-not-retained"
    INFINITE_DERIVATIONS = "infinite-derivations"
    REQUEST_WITH_CERTIFICATE = "request-with-certificate"
    DUPLICATE_CONTROL = "duplicate-control"
    EARLY_STOP_LABELLED_EXACT = "early-stop-labelled-exact"
    CERTIFIED_WITHOUT_CERTIFICATE = "certified-without-certificate"
    BOUND_BELOW_CERTIFICATE = "bound-below-certificate"
    ARGUMENT_NOT_ESTABLISHED = "argument-not-established"
    CERTIFICATE_METRIC = "certificate-metric"
    UNMET_MISREPORTED = "unmet-misreported"
    USED_CLAIM_NOT_ESTABLISHED = "used-claim-not-established"
    EXACT_FOR_INEXACT_RECURSION = "exact-for-inexact-recursion"
    INCOMPLETE_CLAIM = "incomplete-claim"
    UNDECLARED_PARAMETER = "undeclared-parameter"
    LOWER_BOUND_MISMATCH = "lower-bound-mismatch"


class Refusal(QiecModel):
    """One reason a combination is inadmissible.

    Parameters
    ----------
    code : RefusalCode
        The kind of reason.
    reason : str
        The reason, with the records it concerns.
    """

    code: RefusalCode = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    reason: str


class InadmissibleContract(ValueError):
    """A query, its recursion, its contract and its claims do not fit.

    Parameters
    ----------
    refusals : tuple of Refusal
        Every reason.
    """

    def __init__(self, refusals: tuple[Refusal, ...]) -> None:
        super().__init__(
            "; ".join(f"{item.code.value}: {item.reason}" for item in refusals)
        )
        self.refusals = refusals


class InadmissibleReport(ValueError):
    """A report claims what its run cannot show, or contradicts its contract.

    Parameters
    ----------
    refusals : tuple of Refusal
        Every reason.
    """

    def __init__(self, refusals: tuple[Refusal, ...]) -> None:
        super().__init__(
            "; ".join(f"{item.code.value}: {item.reason}" for item in refusals)
        )
        self.refusals = refusals


def admissibility(
    query: QuerySpec,
    recursion: Recursion,
    contract: AccuracyContract,
    *,
    algebra: ClaimSubject,
    claims: tuple[LawClaim, ...],
    uses: tuple[Requirement, ...] = (),
) -> tuple[Refusal, ...]:
    """Every reason a query, its recursion, contract and claims do not fit.

    Parameters
    ----------
    query : QuerySpec
        The query.
    recursion : Recursion
        How its system is solved.
    contract : AccuracyContract
        Its accuracy contract.
    algebra : ClaimSubject
        Its weight algebra.
    claims : tuple of LawClaim
        The law claims available.
    uses : tuple of Requirement
        Further laws the executor relies on, such as commutativity of
        the product for an elimination order.

    Returns
    -------
    tuple of Refusal
        Empty when the combination is admissible.
    """
    found: list[Refusal] = []
    observable = query.observable
    guaranteed = not isinstance(contract.guarantee, Heuristic)
    wanted = contract.gradient
    needed = (
        *observable.requirements(algebra),
        *requirements(recursion.laws(), algebra),
        *uses,
    )
    for requirement in dict.fromkeys(needed):
        matching = [
            claim
            for claim in claims
            if claim.law == requirement.law and claim.subject == requirement.subject
        ]
        if not matching:
            found.append(
                Refusal(
                    code=RefusalCode.MISSING_CLAIM,
                    reason=(
                        f"no claim that {requirement.law.name} holds of "
                        f"{requirement.subject.name}"
                    ),
                )
            )
        if matching and guaranteed and not any(c.established for c in matching):
            found.append(
                Refusal(
                    code=RefusalCode.CLAIM_NOT_ESTABLISHED,
                    reason=(
                        f"{requirement.law.name} of {requirement.subject.name} is "
                        f"only {matching[0].status.value}, which licenses no "
                        "guaranteed result"
                    ),
                )
            )
    if isinstance(contract.guarantee, Exact) and not recursion.finite_method:
        found.append(
            Refusal(
                code=RefusalCode.EXACT_WITHOUT_FINITE_METHOD,
                reason=f"no finite method computes a {recursion.tag} solution exactly",
            )
        )
    if not recursion.admits(wanted):
        found.append(
            Refusal(
                code=RefusalCode.DERIVATIVE_UNAVAILABLE,
                reason=(
                    f"a {recursion.tag} solution has no {wanted.kind} derivative "
                    "of the kind requested"
                ),
            )
        )
    if not wanted.missing and not query.differentiate_with_respect_to:
        found.append(
            Refusal(
                code=RefusalCode.GRADIENT_WITHOUT_INPUTS,
                reason="a derivative is requested with respect to nothing",
            )
        )
    if wanted.missing and query.differentiate_with_respect_to:
        found.append(
            Refusal(
                code=RefusalCode.INPUTS_WITHOUT_GRADIENT,
                reason="inputs are differentiated but the contract requests no derivative",
            )
        )
    if query.differentiate_with_respect_to and not observable.differentiable:
        found.append(
            Refusal(
                code=RefusalCode.OBSERVABLE_NOT_DIFFERENTIABLE,
                reason=f"a {observable.tag} result has no derivative",
            )
        )
    if observable.selects and not wanted.missing and wanted.tie_rule is None:
        found.append(
            Refusal(
                code=RefusalCode.SELECTION_WITHOUT_TIE_RULE,
                reason=f"a {observable.tag} derivative needs the tie rule it accepts",
            )
        )
    if observable.needs_forest and not query.retention.keeps_forest:
        found.append(
            Refusal(
                code=RefusalCode.FOREST_NOT_RETAINED,
                reason=f"a {observable.tag} result needs the packed forest",
            )
        )
    if (
        observable.needs_finite_derivations
        and guaranteed
        and not recursion.finite_derivations
    ):
        found.append(
            Refusal(
                code=RefusalCode.INFINITE_DERIVATIONS,
                reason=(
                    f"a guaranteed {observable.tag} result enumerates derivations, "
                    f"and a {recursion.tag} system can have infinitely many"
                ),
            )
        )
    if (
        isinstance(contract.guarantee, Certified)
        and contract.guarantee.certificate is not None
    ):
        found.append(
            Refusal(
                code=RefusalCode.REQUEST_WITH_CERTIFICATE,
                reason="a requested bound carries no certificate; a run shows it",
            )
        )
    names = [control.name for control in contract.controls]
    if len(set(names)) != len(names):
        found.append(
            Refusal(
                code=RefusalCode.DUPLICATE_CONTROL,
                reason="a control is set twice",
            )
        )
    return tuple(found)


def check_admissible(
    query: QuerySpec,
    recursion: Recursion,
    contract: AccuracyContract,
    *,
    algebra: ClaimSubject,
    claims: tuple[LawClaim, ...],
    uses: tuple[Requirement, ...] = (),
) -> None:
    """Refuse a query, recursion, contract and claims that do not fit.

    Parameters
    ----------
    query : QuerySpec
        The query.
    recursion : Recursion
        How its system is solved.
    contract : AccuracyContract
        Its accuracy contract.
    algebra : ClaimSubject
        Its weight algebra.
    claims : tuple of LawClaim
        The law claims available.
    uses : tuple of Requirement
        Further laws the executor relies on.

    Raises
    ------
    InadmissibleContract
        With every reason, if there is one.
    """
    found = admissibility(
        query, recursion, contract, algebra=algebra, claims=claims, uses=uses
    )
    if found:
        raise InadmissibleContract(found)


def _certificate_metric(certificate: Certificate, metric: Metric) -> bool:
    """Whether a certificate bounds distances in a metric.

    Parameters
    ----------
    certificate : Certificate
        The certificate.
    metric : Metric
        The metric of the attainment.

    Returns
    -------
    bool
        An enclosure bounds absolute errors, of values or of log values;
        a duality gap bounds an objective gap; an argument and a
        composition state their bound in the metric of the attainment.
    """
    if isinstance(certificate, Enclosure):
        return isinstance(metric, (AbsoluteError, AbsoluteLogError))
    if isinstance(certificate, DualityGap):
        return isinstance(metric, ObjectiveGap)
    return True


def report_refusals(
    contract: AccuracyContract,
    report: AccuracyReport,
    recursion: Recursion | None = None,
) -> tuple[Refusal, ...]:
    """Every reason a report claims what its run cannot show.

    Parameters
    ----------
    contract : AccuracyContract
        The contract of the run.
    report : AccuracyReport
        Its report.
    recursion : Recursion or None
        How the run solved its system, when it has one.

    Returns
    -------
    tuple of Refusal
        Empty when the report is admissible.
    """
    found: list[Refusal] = []
    attained = report.attainment.guarantee
    if report.stop.early and isinstance(attained, Exact):
        found.append(
            Refusal(
                code=RefusalCode.EARLY_STOP_LABELLED_EXACT,
                reason=f"a run stopped by {report.stop.tag} is not exact",
            )
        )
    claimed = attained.bound.fraction if isinstance(attained, Certified) else None
    certificate = attained.certificate if isinstance(attained, Certified) else None
    if claimed is not None and certificate is None:
        found.append(
            Refusal(
                code=RefusalCode.CERTIFIED_WITHOUT_CERTIFICATE,
                reason="an attained bound carries the certificate that shows it",
            )
        )
    if claimed is not None and certificate is not None and claimed < certificate.bound:
        found.append(
            Refusal(
                code=RefusalCode.BOUND_BELOW_CERTIFICATE,
                reason=(
                    f"the bound {claimed} is below the {certificate.bound} its "
                    f"{certificate.tag} shows"
                ),
            )
        )
    if isinstance(certificate, Argument) and not certificate.claim.established:
        found.append(
            Refusal(
                code=RefusalCode.ARGUMENT_NOT_ESTABLISHED,
                reason=(
                    f"the argument {certificate.claim.law.name} is only "
                    f"{certificate.claim.status.value}"
                ),
            )
        )
    if certificate is not None and not _certificate_metric(
        certificate, report.attainment.metric
    ):
        found.append(
            Refusal(
                code=RefusalCode.CERTIFICATE_METRIC,
                reason=(
                    f"a {certificate.tag} does not bound the "
                    f"{report.attainment.metric.tag} metric"
                ),
            )
        )
    if not isinstance(attained, Heuristic):
        weak = [claim for claim in report.claims if not claim.established]
        if weak:
            found.append(
                Refusal(
                    code=RefusalCode.USED_CLAIM_NOT_ESTABLISHED,
                    reason=(
                        f"the run used {weak[0].law.name}, only "
                        f"{weak[0].status.value}, for a guaranteed result"
                    ),
                )
            )
    if (
        recursion is not None
        and isinstance(attained, Exact)
        and not recursion.finite_method
    ):
        found.append(
            Refusal(
                code=RefusalCode.EXACT_FOR_INEXACT_RECURSION,
                reason=f"a {recursion.tag} solution is not exact",
            )
        )
    expected = unmet_fields(contract, report.attainment, report.work)
    if report.unmet != expected:
        found.append(
            Refusal(
                code=RefusalCode.UNMET_MISREPORTED,
                reason=(
                    f"the unmet requirements are {[f.value for f in expected]}, "
                    f"reported {[f.value for f in report.unmet]}"
                ),
            )
        )
    return tuple(found)


def check_report(
    contract: AccuracyContract,
    report: AccuracyReport,
    recursion: Recursion | None = None,
) -> None:
    """Refuse a report that claims what its run cannot show.

    Parameters
    ----------
    contract : AccuracyContract
        The contract of the run.
    report : AccuracyReport
        Its report.
    recursion : Recursion or None
        How the run solved its system, when it has one.

    Raises
    ------
    InadmissibleReport
        With every reason, if there is one.
    """
    found = report_refusals(contract, report, recursion)
    if found:
        raise InadmissibleReport(found)


def _identifiers(expression: str) -> set[str]:
    """The names a bound expression uses, other than functions.

    Parameters
    ----------
    expression : str
        The bound.

    Returns
    -------
    set of str
        Its identifiers outside `BOUND_FUNCTIONS`.
    """
    return set(_IDENTIFIER.findall(expression)) - BOUND_FUNCTIONS


def _normal(expression: str) -> str:
    """A bound expression without whitespace.

    Parameters
    ----------
    expression : str
        The bound.

    Returns
    -------
    str
        The same bound with every space removed, for comparison.
    """
    return "".join(expression.split())


def complexity_refusals(claim: ComplexityClaim) -> tuple[Refusal, ...]:
    """Every reason a complexity claim is incomplete or unsupported.

    Parameters
    ----------
    claim : ComplexityClaim
        The claim.

    Returns
    -------
    tuple of Refusal
        Empty when every phase has a time and a space bound or is
        declared inapplicable (not both), every bound uses only the
        declared parameters, and every optimality proof has a lower bound
        equal to its bound.
    """
    found: list[Refusal] = []
    declared = {parameter.name for parameter in claim.parameters}
    inapplicable = Counter(item.phase for item in claim.inapplicable)
    present = Counter((bound.phase, bound.resource) for bound in claim.bounds)
    for phase in Phase:
        resources = [present[(phase, resource)] for resource in ("time", "space")]
        bounded = resources == [1, 1]
        absent = resources == [0, 0] and inapplicable[phase] == 1
        if bounded == absent or (bounded and inapplicable[phase]):
            found.append(
                Refusal(
                    code=RefusalCode.INCOMPLETE_CLAIM,
                    reason=(
                        f"the {phase.value} phase needs one time and one space "
                        "bound, or one declaration that it does not apply"
                    ),
                )
            )
    for bound in claim.bounds:
        status = bound.status
        expressions = [bound.expression]
        if isinstance(status, ProvedOptimal):
            expressions.append(status.lower_bound)
        if isinstance(status, Deficient):
            expressions.append(status.better_bound)
        unknown = sorted(
            set().union(*(_identifiers(item) for item in expressions)) - declared
        )
        if unknown:
            found.append(
                Refusal(
                    code=RefusalCode.UNDECLARED_PARAMETER,
                    reason=(
                        f"the {bound.phase.value} {bound.resource} bound uses "
                        f"{unknown}, which the claim does not declare"
                    ),
                )
            )
        if isinstance(status, ProvedOptimal) and _normal(status.lower_bound) != _normal(
            bound.expression
        ):
            found.append(
                Refusal(
                    code=RefusalCode.LOWER_BOUND_MISMATCH,
                    reason=(
                        f"the lower bound {status.lower_bound} does not match the "
                        f"{bound.phase.value} {bound.resource} bound {bound.expression}"
                    ),
                )
            )
    return tuple(found)


def check_complexity(claim: ComplexityClaim) -> None:
    """Refuse an incomplete or unsupported complexity claim.

    Parameters
    ----------
    claim : ComplexityClaim
        The claim.

    Raises
    ------
    InadmissibleContract
        With every reason, if there is one.
    """
    found = complexity_refusals(claim)
    if found:
        raise InadmissibleContract(found)


__all__ = [
    "ALGEBRA_OPERATIONS",
    "AbsoluteError",
    "AbsoluteLogError",
    "AccuracyContract",
    "AccuracyReport",
    "Acyclic",
    "Argument",
    "ArgumentBound",
    "Attainment",
    "BOUND_FUNCTIONS",
    "Best",
    "BestKnown",
    "Bias",
    "BiasBound",
    "BiasUnknown",
    "Bounded",
    "BudgetExhausted",
    "Certificate",
    "Certified",
    "Completed",
    "ComplexityBound",
    "ComplexityClaim",
    "ComplexityParameter",
    "ComposedTerm",
    "Composition",
    "ComputationModel",
    "Condition",
    "Conditional",
    "ContractField",
    "ContractNotMet",
    "Control",
    "ControlReached",
    "Deficient",
    "DerivativeEntry",
    "DerivativeReport",
    "DerivativeSemantics",
    "DualityGap",
    "Enclosure",
    "Exact",
    "FailurePolicy",
    "Fold",
    "Forest",
    "Guarantee",
    "Heuristic",
    "InadmissibleContract",
    "InadmissibleReport",
    "InterpretationId",
    "JudgmentPattern",
    "KBest",
    "KLDivergence",
    "LinearCyclic",
    "Marginal",
    "Metric",
    "MonotonePolynomial",
    "NOT_DIFFERENTIATED",
    "NoRetention",
    "NonlinearFold",
    "Numerics",
    "ObjectiveGap",
    "Observable",
    "OccurrenceId",
    "OperationBound",
    "OptimalityStatus",
    "Outcome",
    "PackedForest",
    "ParameterId",
    "ParentCounts",
    "Phase",
    "PhaseInapplicable",
    "PresentationId",
    "Probabilistic",
    "ProvedOptimal",
    "QuerySpec",
    "Rational",
    "Recursion",
    "Refusal",
    "RefusalCode",
    "RelativeError",
    "RequestedEvidence",
    "ResourceBudget",
    "Retention",
    "SPECTRAL_RADIUS_BELOW_ONE",
    "Sample",
    "SiteEstimator",
    "Statistic",
    "StopReason",
    "Support",
    "Target",
    "TotalVariation",
    "UNLIMITED",
    "Unbiased",
    "Unestablished",
    "Unresolved",
    "Unrolled",
    "WORD_RAM_EXPECTED_HASHING",
    "Weight",
    "SPACE_FIELDS",
    "TIME_FIELDS",
    "UnmeasuredWork",
    "WorkField",
    "WorkReport",
    "bounded_fields",
    "measured_total",
    "measuring",
    "ZERO",
    "admissibility",
    "attained_guarantee",
    "check_admissible",
    "check_complexity",
    "check_report",
    "complexity_refusals",
    "compose",
    "report_refusals",
    "settle",
    "unmet_fields",
]
