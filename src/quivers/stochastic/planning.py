"""Execution plans for queries over rule presentations.

`plan` checks a query against its recursion contract, accuracy contract,
retention and the law claims of its algebra, analyses the presentation,
and selects an algorithm; the plan records the algorithm, the
assumptions it rests on, the alternatives considered, and the cost claim.
Nothing is chosen silently: a query no executor here answers is refused
with the reason.

The executor for acyclic systems is the two-pass evaluation of Eisner
(2023, sections 5 and 6) on an indexed chart. The first pass discovers
every item and counts, for each, the rule instances that conclude it,
without storing them. The second pass computes values in an order in
which an item's value is final before it is used: an item is ready when
every instance concluding it has contributed. Each instance is produced
once, by the last of its premises to become final, so space is linear in
the items beside the autograd tape. A recomputing backward pass computes
gradients in the same space by enumerating each item's incoming
instances again in reverse order.

References
----------
Eisner, J. (2023). Time-and-space-efficient weighted deduction.
*Transactions of the ACL* 11, 960-973.
[doi:10.1162/tacl_a_00588](https://doi.org/10.1162/tacl_a_00588)
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Mapping, Sequence
from enum import Enum
from fractions import Fraction
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import OrderPreservingSemiring
from quivers.core.laws import (
    DISTRIBUTIVE,
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    TIMES_ASSOCIATIVE,
    TIMES_UNIT,
    Citation,
    ClaimSubject,
    Law,
    LawClaim,
    requirements,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import (
    ALGEBRA_OPERATIONS,
    AccuracyContract,
    Acyclic,
    Best,
    BestKnown,
    Bounded,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    InadmissibleContract,
    Fold,
    Forest,
    Marginal,
    NonlinearFold,
    NoRetention,
    PackedForest,
    OccurrenceId,
    ParameterId,
    ParentCounts,
    Phase,
    PhaseInapplicable,
    ProvedOptimal,
    QuerySpec,
    Rational,
    Recursion,
    Retention,
    Statistic,
    Support,
    Weight,
    WorkReport,
    check_admissible,
)
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import IndexBinder, NatSort
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    Derivation,
    Ground,
    Item,
    PresentationAnalysis,
    Rule,
    RulePresentation,
    evaluate_index,
)
from quivers.qiec.types import IndexLiteral, IndexSum, IndexTerm, IndexVariable
from quivers.stochastic.forest import DerivationForest, build_forest
from quivers.stochastic.indexed import ChartWork, IndexedChart, generic_joinable
from quivers.stochastic.joins import agm_bound, body_query

__all__ = [
    "Alternative",
    "Evaluation",
    "MeasureCertificate",
    "MeasureComponent",
    "Plan",
    "PlanRecord",
    "PlanRefusal",
    "FOREST_ALGORITHM",
    "FOREST_WORK",
    "PlanRefused",
    "RECOMPUTED_TWO_PASS_WORK",
    "RuleJoin",
    "TWO_PASS_ALGORITHM",
    "TWO_PASS_LAWS",
    "TWO_PASS_WORK",
    "acyclicity_certificate",
    "rule_joins",
    "plan",
]


# ---------------------------------------------------------------------------
# refusals


class PlanRefusal(str, Enum):
    """Why no plan, or no result, is given."""

    INADMISSIBLE = "inadmissible"
    RECURSION_WITHOUT_EXECUTOR = "recursion-without-executor"
    OPEN_PREMISES = "open-premises"
    OBSERVABLE_WITHOUT_EXECUTOR = "observable-without-executor"
    RETENTION_WITHOUT_EXECUTOR = "retention-without-executor"
    ALGEBRA = "algebra"
    CYCLE = "cycle"


class PlanRefused(ValueError):
    """A query is refused, with the reason.

    Parameters
    ----------
    code : PlanRefusal
        The kind of reason.
    reason : str
        The reason.
    """

    def __init__(self, code: PlanRefusal, reason: str) -> None:
        super().__init__(f"{code.value}: {reason}")
        self.code = code
        self.reason = reason


# ---------------------------------------------------------------------------
# a measure that proves acyclicity


type Linear = dict[str, Fraction]
"""A linear form: coefficients by variable, the constant under ``""``."""


def _linear(term: IndexTerm) -> Linear | None:
    """The linear form of a natural-number index term.

    Parameters
    ----------
    term : IndexTerm
        The term.

    Returns
    -------
    Linear or None
        Its form, or ``None`` when it is not linear in natural numbers.
    """
    if isinstance(term, IndexVariable):
        return {term.name: Fraction(1)}
    if isinstance(term, IndexLiteral) and isinstance(term.value, int):
        return {"": Fraction(term.value)}
    if isinstance(term, IndexSum):
        form: Linear = {"": Fraction(term.constant)}
        for summand in term.terms:
            form[summand.name] = form.get(summand.name, Fraction(0)) + 1
        return form
    return None


def _combine(*pairs: tuple[Fraction, Linear]) -> Linear:
    """A weighted sum of linear forms.

    Parameters
    ----------
    *pairs : tuple of Fraction and Linear
        Coefficients and forms.

    Returns
    -------
    Linear
        The sum.
    """
    found: Linear = {}
    for coefficient, form in pairs:
        for name, value in form.items():
            found[name] = found.get(name, Fraction(0)) + coefficient * value
    return found


def _infeasible(constraints: list[Linear]) -> bool:
    """Whether ``form >= 0`` for every form has no real solution.

    Fourier-Motzkin elimination: each variable is eliminated by combining
    every constraint where it is positive with every one where it is
    negative; the system has no real solution exactly when a constant
    constraint below zero appears.

    Parameters
    ----------
    constraints : list of Linear
        Constraints ``form >= 0``.

    Returns
    -------
    bool
        True when no real point satisfies them all.
    """
    system = [dict(form) for form in constraints]
    names = sorted({name for form in system for name in form if name})
    for name in names:
        positive = [form for form in system if form.get(name, 0) > 0]
        negative = [form for form in system if form.get(name, 0) < 0]
        rest = [form for form in system if form.get(name, 0) == 0]
        for upper in positive:
            for lower in negative:
                a, b = upper[name], -lower[name]
                combined = _combine((b, upper), (a, lower))
                combined.pop(name, None)
                rest.append(combined)
        system = rest
    return any(
        all(value == 0 for key, value in form.items() if key) and form.get("", 0) < 0
        for form in system
    )


def _proves(goal: Linear, facts: list[Linear]) -> bool:
    """Whether ``goal >= 1`` follows from ``fact >= 0`` and nonnegative variables.

    The goal has integer coefficients over natural-number variables, so
    ``goal >= 1`` fails only where ``goal <= 0``; that case is shown to have
    no real solution.

    Parameters
    ----------
    goal : Linear
        The form.
    facts : list of Linear
        Known constraints ``fact >= 0``.

    Returns
    -------
    bool
        Whether the implication holds.
    """
    names = {name for form in (goal, *facts) for name in form if name}
    nonnegative = [{name: Fraction(1)} for name in names]
    return _infeasible([*facts, *nonnegative, _combine((Fraction(-1), goal))])


class MeasureComponent(QiecModel):
    """Measures that prove one cyclic component of forms acyclic.

    Each alternative gives, for every form of the component, coefficients
    ``a_p`` of the measure ``sum_p a_p x_p`` over its arguments (zero for
    an argument that is not a natural number). Every alternative is
    proved: if every premise of the component has measure at least one,
    so has the conclusion, and the conclusion's measure exceeds each such
    premise's. One alternative applies to a run when every item of the
    component's forms has measure at least one; then no item of the
    component derives itself.

    Parameters
    ----------
    families : tuple of FamilyId
        The forms of the component.
    alternatives : tuple of tuple of tuple of int
        The proved measures, each with one coefficient vector per form.
    """

    families: tuple[FamilyId, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    alternatives: tuple[tuple[tuple[int, ...], ...], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    def applies(self, items: Sequence[Item]) -> bool:
        """Whether some alternative gives every item of the component measure one or more.

        Parameters
        ----------
        items : Sequence of Item
            The items of a run.

        Returns
        -------
        bool
            True when one alternative holds on all of them.
        """
        position = {family: index for index, family in enumerate(self.families)}
        relevant = [item for item in items if item.family in position]
        for alternative in self.alternatives:
            if all(
                _weighted(alternative[position[item.family]], item.arguments) >= 1
                for item in relevant
            ):
                return True
        return False


def _weighted(coefficients: Sequence[int], arguments: Sequence[Ground]) -> int:
    """The measure of ground arguments.

    Parameters
    ----------
    coefficients : Sequence of int
        The coefficients.
    arguments : Sequence of Ground
        The arguments; a weighted one is a natural number.

    Returns
    -------
    int
        ``sum_p a_p x_p``.
    """
    total = 0
    for coefficient, value in zip(coefficients, arguments, strict=True):
        if coefficient:
            assert isinstance(value, int)
            total += coefficient * value
    return total


class MeasureCertificate(QiecModel):
    """Measures for every cyclic component of the form graph.

    Each rule of a cyclic component strictly increases a measure from its
    premises in the component to its conclusion, given that those
    premises have measure at least one, and keeps the conclusion's
    measure at least one. When a run's items satisfy the condition of one
    alternative per component, no item derives itself, so the derived
    hypergraph is acyclic. A run reports whether the certificate applies;
    the two passes refuse a cycle in any case.

    Parameters
    ----------
    components : tuple of MeasureComponent
        One per cyclic component.
    """

    components: tuple[MeasureComponent, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    def applies(self, items: Sequence[Item]) -> bool:
        """Whether every component has an alternative that holds on the items.

        Parameters
        ----------
        items : Sequence of Item
            The items of a run.

        Returns
        -------
        bool
            True when the certificate proves the run acyclic.
        """
        return all(component.applies(items) for component in self.components)


def _measure_of(
    judgment_arguments: Sequence[IndexTerm], coefficients: Sequence[int]
) -> Linear | None:
    """The measure of a pattern as a linear form of its variables.

    Parameters
    ----------
    judgment_arguments : Sequence of IndexTerm
        The pattern's arguments.
    coefficients : Sequence of int
        The measure's coefficients.

    Returns
    -------
    Linear or None
        The form, or ``None`` when a weighted argument is not linear.
    """
    parts: list[tuple[Fraction, Linear]] = []
    for coefficient, argument in zip(coefficients, judgment_arguments, strict=True):
        if coefficient == 0:
            continue
        form = _linear(argument)
        if form is None:
            return None
        parts.append((Fraction(coefficient), form))
    return _combine(*parts)


def acyclicity_certificate(presentation: RulePresentation) -> MeasureCertificate | None:
    """Find measures that prove the derived hypergraph acyclic.

    For each cyclic component of the form graph, measures with
    coefficients in ``{-1, 0, 1}`` on the natural-number arguments are
    tried in a fixed order, at most 4096 combinations per component; the
    first that `MeasureCertificate` describes is kept.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    MeasureCertificate or None
        A certificate covering every cyclic component, or ``None`` when
        one has no measure of this form.
    """
    analysis = presentation.analysis
    forms = presentation.judgments_by_id
    found: list[MeasureComponent] = []
    for component in analysis.components:
        if not component.cyclic:
            continue
        members = component.families
        choices: list[list[tuple[int, ...]]] = []
        for family in members:
            natural = [
                isinstance(binder, IndexBinder) and isinstance(binder.sort, NatSort)
                for binder in forms[family].indices
            ]
            options: list[tuple[int, ...]] = [
                vector
                for vector in itertools.product((1, -1, 0), repeat=len(natural))
                if any(value > 0 for value in vector)
                and all(
                    value == 0 or is_nat
                    for value, is_nat in zip(vector, natural, strict=True)
                )
            ]
            choices.append(options)
        proved: list[tuple[tuple[int, ...], ...]] = []
        for count, combination in enumerate(itertools.product(*choices)):
            if count >= 4096:
                break
            if _certifies(presentation, dict(zip(members, combination, strict=True))):
                proved.append(tuple(combination))
        if not proved:
            return None
        found.append(MeasureComponent(families=members, alternatives=tuple(proved)))
    return MeasureCertificate(components=tuple(found))


def _certifies(
    presentation: RulePresentation, measures: Mapping[FamilyId, tuple[int, ...]]
) -> bool:
    """Whether measures satisfy the two obligations on every rule.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    measures : Mapping of FamilyId to tuple of int
        A measure per form of one component.

    Returns
    -------
    bool
        True when, for every rule concluding a form of the component,
        the conclusion's measure is at least one and exceeds each inside
        premise's, given that each inside premise's is at least one.
    """
    for rule in presentation.rules:
        if rule.conclusion.family not in measures:
            continue
        conclusion = _measure_of(
            rule.conclusion.arguments, measures[rule.conclusion.family]
        )
        if conclusion is None:
            return False
        inside: list[Linear] = []
        for premise in rule.premises:
            family = premise.judgment.family
            if family in measures:
                form = _measure_of(premise.judgment.arguments, measures[family])
                if form is None:
                    return False
                inside.append(form)
        if not inside:
            continue
        facts = [
            _combine((Fraction(1), form), (Fraction(-1), {"": Fraction(1)}))
            for form in inside
        ]
        if not _proves(conclusion, facts):
            return False
        for form in inside:
            if not _proves(
                _combine((Fraction(1), conclusion), (Fraction(-1), form)), facts
            ):
                return False
    return True


# ---------------------------------------------------------------------------
# records


class Alternative(QiecModel):
    """An algorithm considered and why it was not selected.

    Parameters
    ----------
    algorithm : str
        The algorithm.
    reason : str
        Why not.
    """

    algorithm: str
    reason: str


_EISNER = Citation(
    reference="Eisner 2023, Time-and-Space-Efficient Weighted Deduction",
    locator="sections 5 and 6",
    doi="10.1162/tacl_a_00588",
)
_READING = Citation(
    reference="every instance is enumerated and every item read, by an adversary argument",
    locator="complexity claims, Q10",
)


def _two_pass_work(backward: Literal["autograd", "recompute"]) -> ComplexityClaim:
    """The cost claim of two-pass evaluation with one backward pass.

    Parameters
    ----------
    backward : {"autograd", "recompute"}
        The backward pass.

    Returns
    -------
    ComplexityClaim
        The claim.
    """
    final_space = ProvedOptimal(
        lower_bound="V", model=ALGEBRA_OPERATIONS, sources=(_READING,)
    )
    if backward == "autograd":
        derivative_space = ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="V + P E",
            status=BestKnown(sources=(_EISNER,)),
        )
        tape = (
            "the autograd tape holds every product and sum of the run, P E "
            "elements of the weight's size, beside the V values"
        )
    else:
        derivative_space = ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="V + P D",
            status=BestKnown(sources=(_EISNER,)),
        )
        tape = (
            "the recomputing pass holds the V values and cotangents and, while "
            "one item is visited, the P D elements of that item's own sum"
        )
    return ComplexityClaim(
        query="value of every item of an acyclic presentation, and its gradient",
        parameters=(
            ComplexityParameter(name="V", meaning="items"),
            ComplexityParameter(name="E", meaning="grounded rule instances and inputs"),
            ComplexityParameter(
                name="J", meaning="candidates the keyed lookups return, at least E"
            ),
            ComplexityParameter(name="P", meaning="the most premises of one rule"),
            ComplexityParameter(
                name="D", meaning="the most instances and inputs concluding one item"
            ),
        ),
        model=ALGEBRA_OPERATIONS,
        bounds=(
            ComplexityBound(
                phase=Phase.EXECUTION,
                resource="time",
                expression="V + P E + J",
                status=BestKnown(sources=(_EISNER,)),
            ),
            ComplexityBound(
                phase=Phase.EXECUTION,
                resource="space",
                expression="V",
                status=final_space,
            ),
            ComplexityBound(
                phase=Phase.DIFFERENTIATION,
                resource="time",
                expression="V + P E + J",
                status=BestKnown(sources=(_EISNER,)),
            ),
            derivative_space,
        ),
        inapplicable=(
            PhaseInapplicable(
                phase=Phase.CONSTRUCTION,
                reason="join plans and indexes are built from the presentation alone",
            ),
            PhaseInapplicable(
                phase=Phase.REFIT,
                reason="a refit is one more run of both passes at the new weights",
            ),
        ),
        conditions=(
            "the derived hypergraph is acyclic",
            "operations of the algebra cost one each; the count of operations "
            "does not depend on the precision",
            "J counts every candidate the keyed lookups return, including those "
            "not yet final or failing a repeated variable or side condition; "
            "a run reports J, and the record bounds each rule's part of it by the "
            "AGM bound of its body",
            "space excludes the presentation, the inputs and the indexes of the "
            "chart, which hold O(V) entries per keyed premise pattern",
            tape,
        ),
    )


TWO_PASS_WORK = _two_pass_work("autograd")
"""The cost of two-pass evaluation differentiated by autograd."""

RECOMPUTED_TWO_PASS_WORK = _two_pass_work("recompute")
"""The cost of two-pass evaluation with the recomputing backward pass."""

FOREST_WORK = ComplexityClaim(
    query="the packed forest of a presentation, and folds over its derivations",
    parameters=(
        ComplexityParameter(name="V", meaning="items"),
        ComplexityParameter(name="E", meaning="grounded rule instances and inputs"),
        ComplexityParameter(
            name="J", meaning="candidates the keyed lookups return, at least E"
        ),
        ComplexityParameter(name="P", meaning="the most premises of one rule"),
        ComplexityParameter(
            name="H", meaning="the bound on height, one for an acyclic forest"
        ),
        ComplexityParameter(
            name="L", meaning="the declared limit on the derivations enumerated"
        ),
        ComplexityParameter(
            name="K",
            meaning="the items, at each height, that derivations of the goal use",
        ),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.CONSTRUCTION,
            resource="time",
            expression="V + P E + J",
            status=BestKnown(sources=(_EISNER,)),
        ),
        ComplexityBound(
            phase=Phase.CONSTRUCTION,
            resource="space",
            expression="V + P E",
            status=ProvedOptimal(
                lower_bound="P E, the forest being the output",
                model=ALGEBRA_OPERATIONS,
                sources=(_READING,),
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="H (V + P E) for a linear fold; P L K for a nonlinear one",
            status=BestKnown(sources=(_EISNER,)),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="H V for a linear fold; L K for a nonlinear one",
            status=BestKnown(sources=(_EISNER,)),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.DIFFERENTIATION,
            reason="a fold is differentiated by autograd through its own operations",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="the forest depends on the inputs only; a refit folds it again",
        ),
    ),
    conditions=(
        "a goal with more than L derivations is refused, exactly counted first",
        "each listed item has at most L derivations, since each extends to "
        "distinct derivations of the goal",
        "a forest with a cycle is unfolded only within a bound on height",
    ),
)
"""The cost of keeping the packed forest and folding over it."""


class RuleJoin(QiecModel):
    """How one rule's premises are joined, and the bound that gives.

    Parameters
    ----------
    rule : ConstructorId
        The rule.
    name : str
        Its name.
    algorithm : str
        ``"generic join"``, ``"keyed premise at a time"`` or ``"constant"``
        for a rule without premises.
    rho : Rational
        The weight of a least fractional edge cover of the body's
        variables by its premises, ``rho*``; zero for a constant.
    bound : str
        The bound on the join's lookups and candidates over a run.
    reason : str
        Why the algorithm applies.
    """

    rule: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    algorithm: Literal["generic join", "keyed premise at a time", "constant"]
    rho: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    bound: str
    reason: str


def rule_joins(presentation: RulePresentation) -> tuple[RuleJoin, ...]:
    """The join each rule of a presentation gets on an indexed chart.

    A rule whose patterns are relations of their own variables is joined
    by Generic Join from each trigger. Over a run, each premise's trigger
    joins visit no more than Generic Join over the whole body below the
    trigger's values, so the rule's lookups and candidates are
    ``O(m N + m^2 n N^rho)`` for ``m`` premises, ``n`` variables and
    ``N`` the most items of a premise's form. Any other rule looks its
    premises up one at a time.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    tuple of RuleJoin
        One per rule, in order.
    """
    found: list[RuleJoin] = []
    for rule in presentation.rules:
        m = len(rule.premises)
        if not m:
            found.append(
                RuleJoin(
                    rule=rule.id,
                    name=rule.name,
                    algorithm="constant",
                    rho=Rational(numerator=0),
                    bound="1",
                    reason="a rule without premises has one ground instance",
                )
            )
            continue
        if not generic_joinable(rule):
            found.append(
                RuleJoin(
                    rule=rule.id,
                    name=rule.name,
                    algorithm="keyed premise at a time",
                    rho=Rational.of(m),
                    bound=f"N^{m}",
                    reason="a pattern is a sum of two variables, which is not a "
                    "relation of its items alone",
                )
            )
            continue
        query = body_query(rule)
        cover = agm_bound(
            query.variables,
            [atom.variables for atom in query.atoms],
            [2] * m,
        )
        n = len(query.variables)
        rho = cover.rho
        found.append(
            RuleJoin(
                rule=rule.id,
                name=rule.name,
                algorithm="generic join",
                rho=Rational.of(rho),
                bound=f"{m} N + {m * m * n} N^({rho})",
                reason="every pattern is a relation of its variables; the AGM "
                "bound of the body holds for every trigger's join below its values",
            )
        )
    return tuple(found)


class PlanRecord(QiecModel):
    """What a plan computes, how, under which assumptions, and at what cost.

    Parameters
    ----------
    query : QuerySpec
        The query.
    recursion : Recursion
        Its recursion contract.
    accuracy : AccuracyContract
        Its accuracy contract.
    retention : Retention
        The provenance kept.
    algebra : ClaimSubject
        The weight algebra.
    claims : tuple of LawClaim
        The law claims the plan relies on.
    algorithm : str
        The selected algorithm.
    assumptions : tuple of str
        What the algorithm rests on.
    structure : PresentationAnalysis
        The analysis of the presentation.
    certificate : MeasureCertificate or None
        A static proof of acyclicity, when one was found.
    complexity : ComplexityClaim
        The cost claim.
    alternatives : tuple of Alternative
        Algorithms considered and not selected.
    backward : str
        ``"autograd"`` or ``"recompute"``.
    joins : tuple of RuleJoin
        The join and bound of each rule.
    """

    query: QuerySpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    recursion: Recursion = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    accuracy: AccuracyContract = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    retention: Retention = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    algebra: ClaimSubject = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    claims: tuple[LawClaim, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    algorithm: str
    assumptions: tuple[str, ...] = ()
    structure: PresentationAnalysis = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    certificate: MeasureCertificate | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    complexity: ComplexityClaim = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    alternatives: tuple[Alternative, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    backward: Literal["autograd", "recompute"] = "autograd"
    joins: tuple[RuleJoin, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


type InputId = ParameterId | OccurrenceId
"""A rule weight's parameter, or an input occurrence."""


class Evaluation(ValueModel):
    """The values of a run, in the order they became final.

    Parameters
    ----------
    values : dict of Item to torch.Tensor
        The value of every item.
    order : tuple of Item
        The items in the order their values became final.
    work : ChartWork
        The chart's counters.
    certified : bool or None
        Whether every item of a form with a measure has measure at least
        one, so the plan's certificate proves this run acyclic; ``None``
        without a certificate.
    sums : int
        Sums of the algebra computed.
    products : int
        Products of the algebra computed.
    counts : int
        Parent counts kept during the run, one per item.
    """

    values: dict[Item, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    order: tuple[Item, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    work: ChartWork = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    certified: bool | None = None
    sums: int = 0
    products: int = 0
    counts: int = 0

    def report(self) -> WorkReport:
        """The run's counters as a work report of its execution.

        Algebra operations are counted once each, whatever the size of an
        element; lookups are hash operations and candidates key
        comparisons.

        Returns
        -------
        WorkReport
            The report.
        """
        return WorkReport(
            phase=Phase.EXECUTION,
            scalar_sums=self.sums,
            scalar_products=self.products,
            hash_operations=self.work.lookups,
            key_comparisons=self.work.candidates,
            provenance_records=self.counts,
        )

    def value(self, item: Item) -> torch.Tensor:
        """The value of one item.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        torch.Tensor
            Its value.

        Raises
        ------
        KeyError
            If the item was not derived.
        """
        return self.values[item]


# ---------------------------------------------------------------------------
# the plan


class Plan(ValueModel):
    """A selected algorithm for a query, ready to run.

    Parameters
    ----------
    record : PlanRecord
        What it computes and how.
    presentation : RulePresentation
        The presentation.
    algebra : OrderPreservingSemiring
        The weight algebra.
    """

    record: PlanRecord = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    algebra: OrderPreservingSemiring = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def item(self, family: FamilyId, arguments: tuple[Ground, ...]) -> Item:
        """An item of the presentation.

        Parameters
        ----------
        family : FamilyId
            Its form.
        arguments : tuple of Ground
            Its arguments.

        Returns
        -------
        Item
            The item.
        """
        return Item(family=family, arguments=arguments)

    @property
    def height(self) -> int | None:
        """The bound on the height of derivations, under bounded recursion.

        Returns
        -------
        int or None
            The bound, or ``None`` for an acyclic system.
        """
        recursion = self.record.recursion
        return recursion.limit if isinstance(recursion, Bounded) else None

    def forest(self, axioms: Sequence[AxiomOccurrence]) -> DerivationForest:
        """The packed forest over input occurrences.

        Parameters
        ----------
        axioms : Sequence of AxiomOccurrence
            The input occurrences.

        Returns
        -------
        DerivationForest
            The forest; fold it with `height` as its bound.

        Raises
        ------
        PlanRefused
            If the plan does not keep the packed forest.
        """
        if not isinstance(self.record.retention, PackedForest):
            raise PlanRefused(
                PlanRefusal.RETENTION_WITHOUT_EXECUTOR,
                f"the plan keeps {self.record.retention.tag} provenance, not the forest",
            )
        return build_forest(self.presentation, axioms)

    def _two_pass(self) -> None:
        """Refuse the two-pass run of a system bounded by height.

        Raises
        ------
        PlanRefused
            If the recursion is bounded: its meaning counts only the
            derivations within the bound, which the forest unfolds.
        """
        if not isinstance(self.record.recursion, Acyclic):
            raise PlanRefused(
                PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
                "a system bounded by height is folded on its forest within the bound",
            )

    def run(
        self,
        axioms: Sequence[AxiomOccurrence],
        inputs: Mapping[InputId, torch.Tensor],
        *,
        like: torch.Tensor,
    ) -> Evaluation:
        """Run both passes.

        Parameters
        ----------
        axioms : Sequence of AxiomOccurrence
            The input occurrences.
        inputs : Mapping of InputId to torch.Tensor
            The weight of each rule's parameter and of each occurrence;
            a missing weight is the unit, and an occurrence of a weighted
            slot takes the slot's parameter.
        like : torch.Tensor
            The dtype and device of units and zeros.

        Returns
        -------
        Evaluation
            The value of every item.

        Raises
        ------
        PlanRefused
            If the derived hypergraph has a cycle, or the recursion is
            bounded.
        """
        self._two_pass()
        return _evaluate(self, axioms, inputs, like)

    def gradient(
        self,
        axioms: Sequence[AxiomOccurrence],
        inputs: Mapping[InputId, torch.Tensor],
        goal: Item,
        *,
        like: torch.Tensor,
        adjoint: torch.Tensor | None = None,
    ) -> dict[InputId, torch.Tensor]:
        """The gradient of a goal's value in every input weight.

        With ``backward == "autograd"`` the run is differentiated by
        autograd, whose tape holds every instance. With ``"recompute"`` the
        run keeps only values and their order, and a reverse sweep
        enumerates each item's incoming instances again and differentiates
        each item's sum on its own.

        Parameters
        ----------
        axioms : Sequence of AxiomOccurrence
            The input occurrences.
        inputs : Mapping of InputId to torch.Tensor
            The input weights.
        goal : Item
            The item differentiated.
        like : torch.Tensor
            The dtype and device of units and zeros.
        adjoint : torch.Tensor or None
            The cotangent of the goal's value; ones by default.

        Returns
        -------
        dict of InputId to torch.Tensor
            The gradient of each input, zero for an input the goal does
            not use.

        Raises
        ------
        PlanRefused
            If the derived hypergraph has a cycle.
        KeyError
            If the goal is not derived.
        """
        self._two_pass()
        if self.record.backward == "autograd":
            leaves = {
                key: value.detach().requires_grad_() for key, value in inputs.items()
            }
            evaluation = _evaluate(self, axioms, leaves, like)
            value = evaluation.value(goal)
            if not value.requires_grad:
                return {key: torch.zeros_like(leaf) for key, leaf in leaves.items()}
            cotangent = torch.ones_like(value) if adjoint is None else adjoint
            keys = list(leaves)
            grads = torch.autograd.grad(
                value, [leaves[key] for key in keys], cotangent, allow_unused=True
            )
            return {
                key: torch.zeros_like(leaves[key]) if grad is None else grad
                for key, grad in zip(keys, grads, strict=True)
            }
        with torch.no_grad():
            detached = {key: value.detach() for key, value in inputs.items()}
            evaluation = _evaluate(self, axioms, detached, like)
        return _recompute(self, axioms, detached, evaluation, goal, like, adjoint)

    def best_derivation(
        self,
        axioms: Sequence[AxiomOccurrence],
        evaluation: Evaluation,
        goal: Item,
        inputs: Mapping[InputId, torch.Tensor],
        *,
        like: torch.Tensor,
    ) -> Derivation:
        """A derivation of the goal whose weight is its value, in a selective algebra.

        Each item takes the first of its terms, in the order instances are
        enumerated, whose weight equals its value: the first maximizer.

        Parameters
        ----------
        axioms : Sequence of AxiomOccurrence
            The input occurrences.
        evaluation : Evaluation
            A run over a selective algebra.
        goal : Item
            The item.
        inputs : Mapping of InputId to torch.Tensor
            The input weights of the run.
        like : torch.Tensor
            The dtype and device of units and zeros.

        Returns
        -------
        Derivation
            The witness.

        Raises
        ------
        PlanRefused
            If the observable is not `Best` or no term attains a value.
        """
        if not isinstance(self.record.query.observable, Best):
            raise PlanRefused(
                PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR,
                "a witness is a derivation of a best value; the query is not Best",
            )
        return _witness(self, axioms, evaluation, goal, inputs, like)


# ---------------------------------------------------------------------------
# planning


TWO_PASS_LAWS: tuple[Law, ...] = (
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    TIMES_ASSOCIATIVE,
    TIMES_UNIT,
    DISTRIBUTIVE,
)
"""The laws two-pass evaluation uses: sums are regrouped and reordered as
instances arrive, products of sums are expanded in premise order, and the
unit stands for a rule or input without a weight."""

TWO_PASS_ALGORITHM = "two-pass evaluation by parent counts on an indexed chart"
"""The executor without provenance."""

FOREST_ALGORITHM = (
    "packed forest on an indexed chart; linear folds by dynamic programming over "
    "it, nonlinear folds by enumeration within a declared limit"
)
"""The executor that keeps the packed forest."""

_SUPPORTED_OBSERVABLES = (Weight, Support, Best, Marginal, Statistic)
"""The observables a two-pass run answers: values of items in the algebra,
their support, best values with witnesses, and their derivatives."""

_FOREST_OBSERVABLES = (*_SUPPORTED_OBSERVABLES, Fold, NonlinearFold, Forest)
"""The observables a packed forest answers besides: folds of the
derivations, linear or not, and the forest itself."""


def plan(
    query: QuerySpec,
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    recursion: Recursion,
    accuracy: AccuracyContract,
    retention: Retention,
    *,
    claims: Sequence[LawClaim] = (),
) -> Plan:
    """Select and record an algorithm for a query.

    With no retention or parent counts, the executor is two-pass
    evaluation of an acyclic system. With a packed forest, the executor
    builds the forest, computes linear folds over it by dynamic
    programming and nonlinear ones by enumeration within a declared
    limit; it also admits recursion bounded by the height of
    derivations, which a forest with cycles unfolds within.

    Parameters
    ----------
    query : QuerySpec
        The query; its presentation must be ``presentation``.
    presentation : RulePresentation
        The presentation.
    algebra : OrderPreservingSemiring
        The weight algebra, whose own law claims are used.
    recursion : Recursion
        The recursion contract.
    accuracy : AccuracyContract
        The accuracy contract.
    retention : Retention
        The provenance to keep; it must be the query's.
    claims : Sequence of LawClaim
        Further law claims the result may rest on, such as the linearity
        of a fold; they are checked with the algebra's and recorded.

    Returns
    -------
    Plan
        The plan.

    Raises
    ------
    PlanRefused
        If the combination is inadmissible, or no executor here answers
        the recursion, the observable or the retention, or the
        presentation has premises with hypotheses.
    """
    if query.presentation != presentation.id or query.retention != retention:
        raise PlanRefused(
            PlanRefusal.INADMISSIBLE,
            "the query names another presentation or another retention",
        )
    try:
        check_admissible(
            query,
            recursion,
            accuracy,
            algebra=algebra.subject,
            claims=(*algebra.claims(), *claims),
            uses=requirements(TWO_PASS_LAWS, algebra.subject),
        )
    except InadmissibleContract as error:
        raise PlanRefused(PlanRefusal.INADMISSIBLE, str(error)) from error
    kept = isinstance(retention, PackedForest)
    bounded = isinstance(recursion, Bounded) and recursion.measure == "height"
    if not isinstance(recursion, Acyclic) and not (kept and bounded):
        raise PlanRefused(
            PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
            f"the two-pass executor solves acyclic systems, and the forest "
            f"executor those and systems bounded by height, not {recursion.tag}",
        )
    if any(
        premise.hypotheses for rule in presentation.rules for premise in rule.premises
    ):
        raise PlanRefused(
            PlanRefusal.OPEN_PREMISES,
            "a premise with hypotheses is derived open; a chart holds closed items",
        )
    supported = _FOREST_OBSERVABLES if kept else _SUPPORTED_OBSERVABLES
    if not isinstance(query.observable, supported):
        raise PlanRefused(
            PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR,
            f"a {query.observable.tag} result needs an executor not in this plan",
        )
    if not isinstance(retention, (NoRetention, ParentCounts, PackedForest)):
        raise PlanRefused(
            PlanRefusal.RETENTION_WITHOUT_EXECUTOR,
            f"{retention.tag} provenance needs evidence restricted to patterns, "
            "which the packed forest holds in full",
        )
    certificate = acyclicity_certificate(presentation)
    analysis = presentation.analysis
    assumptions = [
        "sums are reordered and products distributed over them, by the claimed "
        "semiring laws; products keep the premise order, the rule weight last",
        "each instance is produced once, by the last of its premises to become "
        "final, at the first premise that item fills",
    ]
    if certificate is not None:
        assumptions.append(
            "the derived hypergraph is acyclic by the recorded measures when "
            "every item of a measured form has measure at least one, which each "
            "run reports; a cycle is refused during the run in any case"
        )
    elif analysis.recursion() != Acyclic():
        assumptions.append(
            "the form graph has cycles and no measure was found; a cycle in the "
            "derived hypergraph is refused during the run"
        )
    if kept:
        assumptions.append(
            "every input occurrence and every grounded instance is an edge of its "
            "conclusion, so the forest's unfoldings are the derivations one to one"
        )
        assumptions.append(
            "a nonlinear fold builds every derivation; a goal with more than the "
            "declared limit is refused before any is built"
        )
        if bounded:
            assumptions.append(
                f"only derivations of height at most {recursion.limit} are "
                "unfolded, an input having height one"
            )
    budget = accuracy.budget
    backward: Literal["autograd", "recompute"] = (
        "recompute"
        if budget.elements is not None or budget.bytes is not None
        else "autograd"
    )
    record = PlanRecord(
        query=query,
        recursion=recursion,
        accuracy=accuracy,
        retention=retention,
        algebra=algebra.subject,
        claims=(*algebra.claims(), *claims),
        algorithm=FOREST_ALGORITHM if kept else TWO_PASS_ALGORITHM,
        assumptions=tuple(assumptions),
        structure=analysis,
        certificate=certificate,
        complexity=FOREST_WORK
        if kept
        else TWO_PASS_WORK
        if backward == "autograd"
        else RECOMPUTED_TWO_PASS_WORK,
        alternatives=(
            Alternative(
                algorithm=TWO_PASS_ALGORITHM if kept else FOREST_ALGORITHM,
                reason="the observable needs the derivations, which only the forest keeps"
                if kept
                else "stores every instance; the query needs values only",
            ),
            Alternative(
                algorithm="enumeration of derivations",
                reason="exponential in general; the observable survives aggregation",
            ),
            Alternative(
                algorithm="dense contraction",
                reason="materializes every grounded instance of the rule bodies",
            ),
        ),
        backward=backward,
        joins=rule_joins(presentation),
    )
    return Plan(record=record, presentation=presentation, algebra=algebra)


def _constants(plan: Plan, chart: IndexedChart) -> list[tuple[Rule, Item]]:
    """The instances of rules without premises, and their conclusions.

    Validation makes the conclusion of such a rule ground, so each has at
    most one instance, used when its side condition holds.

    Parameters
    ----------
    plan : Plan
        The plan.
    chart : IndexedChart
        The chart, whose table interns the conclusions.

    Returns
    -------
    list of tuple of Rule and Item
        Each rule without premises whose condition holds, with its item.
    """
    found: list[tuple[Rule, Item]] = []
    for rule in plan.presentation.rules:
        if rule.premises or not chart.holds(rule, {}):
            continue
        arguments = tuple(evaluate_index(a, {}) for a in rule.conclusion.arguments)
        found.append((rule, chart.items.intern(rule.conclusion.family, arguments)))
    return found


# ---------------------------------------------------------------------------
# the two passes


def _weight(
    plan: Plan,
    key: InputId | None,
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    """The weight of a rule's parameter or an occurrence, the unit when absent.

    Parameters
    ----------
    plan : Plan
        The plan.
    key : InputId or None
        The parameter or occurrence.
    inputs : Mapping of InputId to torch.Tensor
        The input weights.
    like : torch.Tensor
        The dtype and device of the unit.

    Returns
    -------
    torch.Tensor
        The weight.
    """
    if key is not None and key in inputs:
        return inputs[key]
    return plan.algebra.unit_like((), like)


def _product(
    algebra: OrderPreservingSemiring, factors: Sequence[torch.Tensor]
) -> torch.Tensor:
    """The product of factors, in order.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    factors : Sequence of torch.Tensor
        The factors: the premises' values in order, then the rule weight.

    Returns
    -------
    torch.Tensor
        ``f_0 ⊗ f_1 ⊗ ...``.
    """
    total = factors[0]
    for factor in factors[1:]:
        total = algebra.tensor_op(total, factor)
    return total


def _accumulate(
    algebra: OrderPreservingSemiring, current: torch.Tensor | None, term: torch.Tensor
) -> torch.Tensor:
    """Add a term to a running sum.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    current : torch.Tensor or None
        The sum so far, or ``None``.
    term : torch.Tensor
        The term.

    Returns
    -------
    torch.Tensor
        ``current ⊕ term``.
    """
    if current is None:
        return term
    return algebra.join(torch.stack(torch.broadcast_tensors(current, term)), 0)


def _axiom_item(plan: Plan, chart: IndexedChart, occurrence: AxiomOccurrence) -> Item:
    """The item of an input occurrence.

    Parameters
    ----------
    plan : Plan
        The plan.
    chart : IndexedChart
        The chart, whose table interns the item.
    occurrence : AxiomOccurrence
        The occurrence.

    Returns
    -------
    Item
        Its item.

    Raises
    ------
    PlanRefused
        If the occurrence names no slot of the presentation.
    """
    slot = plan.presentation.constructors_by_id.get(occurrence.slot)
    if not isinstance(slot, AxiomSlot):
        raise PlanRefused(PlanRefusal.INADMISSIBLE, "an occurrence names no slot")
    binding = dict(occurrence.binding)
    arguments = tuple(
        evaluate_index(argument, binding) for argument in slot.judgment.arguments
    )
    return chart.items.intern(slot.judgment.family, arguments)


def _occurrence_weight(
    plan: Plan,
    occurrence: AxiomOccurrence,
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    """The weight of an input occurrence.

    Parameters
    ----------
    plan : Plan
        The plan.
    occurrence : AxiomOccurrence
        The occurrence.
    inputs : Mapping of InputId to torch.Tensor
        The input weights.
    like : torch.Tensor
        The dtype and device of the unit.

    Returns
    -------
    torch.Tensor
        Its own weight, else its slot's parameter, else the unit.
    """
    if occurrence.occurrence in inputs:
        return inputs[occurrence.occurrence]
    slot = plan.presentation.constructors_by_id[occurrence.slot]
    return _weight(plan, slot.weight, inputs, like)


def _discover(
    plan: Plan, chart: IndexedChart, axioms: Sequence[AxiomOccurrence]
) -> tuple[list[Item], dict[Item, int], dict[Item, list[AxiomOccurrence]]]:
    """The first pass: every item and how many instances conclude it.

    Parameters
    ----------
    plan : Plan
        The plan.
    chart : IndexedChart
        An empty chart.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.

    Returns
    -------
    tuple
        The items in discovery order, the count of instances concluding
        each, and the occurrences of each input item.
    """
    leaves: dict[Item, list[AxiomOccurrence]] = {}
    counts: dict[Item, int] = {}
    discovered: list[Item] = []
    seen: set[Item] = set()
    queue: deque[Item] = deque()
    for occurrence in axioms:
        item = _axiom_item(plan, chart, occurrence)
        leaves.setdefault(item, []).append(occurrence)
        if item not in seen:
            seen.add(item)
            discovered.append(item)
            counts[item] = 0
            queue.append(item)
    for _, item in _constants(plan, chart):
        counts[item] = counts.get(item, 0) + 1
        if item not in seen:
            seen.add(item)
            discovered.append(item)
            queue.append(item)
    popped: set[Item] = set()
    by_premise = plan.presentation.rules_by_premise
    while queue:
        item = queue.popleft()
        popped.add(item)
        chart.add(item)
        for rule, position in by_premise.get(item.family, ()):
            for binding, _ in chart.instances(
                rule, position, item, popped.__contains__
            ):
                conclusion = chart.items.intern(
                    rule.conclusion.family,
                    tuple(
                        evaluate_index(a, binding) for a in rule.conclusion.arguments
                    ),
                )
                counts[conclusion] = counts.get(conclusion, 0) + 1
                if conclusion not in seen:
                    seen.add(conclusion)
                    discovered.append(conclusion)
                    queue.append(conclusion)
    return discovered, counts, leaves


def _evaluate(
    plan: Plan,
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Evaluation:
    """Both passes.

    Parameters
    ----------
    plan : Plan
        The plan.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.
    inputs : Mapping of InputId to torch.Tensor
        The input weights.
    like : torch.Tensor
        The dtype and device of units and zeros.

    Returns
    -------
    Evaluation
        The values.

    Raises
    ------
    PlanRefused
        If an item lies on a cycle.
    """
    algebra = plan.algebra
    discovery = IndexedChart(plan.presentation)
    discovered, remaining, leaves = _discover(plan, discovery, axioms)
    chart = IndexedChart(plan.presentation)
    chart.items = discovery.items
    first = dict(discovery.work)
    del discovery
    operations = {"sums": 0, "products": 0}
    sums: dict[Item, torch.Tensor | None] = {}

    def add(item: Item, term: torch.Tensor) -> None:
        """Add a term to an item's sum.

        Parameters
        ----------
        item : Item
            The item.
        term : torch.Tensor
            The term.
        """
        current = sums.get(item)
        if current is not None:
            operations["sums"] += 1
        sums[item] = _accumulate(algebra, current, term)

    for item, occurrences in leaves.items():
        for occurrence in occurrences:
            add(item, _occurrence_weight(plan, occurrence, inputs, like))
    for rule, item in _constants(plan, chart):
        add(item, _weight(plan, rule.weight, inputs, like))
        remaining[item] -= 1
    ready: deque[Item] = deque(item for item in discovered if remaining[item] == 0)
    final: set[Item] = set()
    order: list[Item] = []
    values: dict[Item, torch.Tensor] = {}
    by_premise = plan.presentation.rules_by_premise
    while ready:
        item = ready.popleft()
        current = sums.get(item)
        values[item] = algebra.zero_like((), like) if current is None else current
        final.add(item)
        order.append(item)
        chart.add(item)
        for rule, position in by_premise.get(item.family, ()):
            weight = _weight(plan, rule.weight, inputs, like)
            for binding, premises in chart.instances(
                rule, position, item, final.__contains__
            ):
                conclusion = chart.items.intern(
                    rule.conclusion.family,
                    tuple(
                        evaluate_index(a, binding) for a in rule.conclusion.arguments
                    ),
                )
                operations["products"] += len(premises)
                add(
                    conclusion,
                    _product(algebra, [*(values[p] for p in premises), weight]),
                )
                remaining[conclusion] -= 1
                if remaining[conclusion] == 0:
                    ready.append(conclusion)
    if len(final) != len(discovered):
        stuck = [item for item in discovered if item not in final][:3]
        raise PlanRefused(
            PlanRefusal.CYCLE,
            f"{len(discovered) - len(final)} items lie on or above a cycle, "
            f"for instance {stuck}",
        )
    work = ChartWork(**{name: first[name] + chart.work[name] for name in first})
    certificate = plan.record.certificate
    certified = None if certificate is None else certificate.applies(discovered)
    return Evaluation(
        values=values,
        order=tuple(order),
        work=work,
        certified=certified,
        sums=operations["sums"],
        products=operations["products"],
        counts=len(discovered),
    )


def _terms(
    plan: Plan,
    chart: IndexedChart,
    item: Item,
    leaves: Mapping[Item, list[AxiomOccurrence]],
) -> list[tuple[Rule | AxiomOccurrence, dict[str, Ground], tuple[Item, ...]]]:
    """The terms of an item's sum: its occurrences, then its incoming instances.

    Parameters
    ----------
    plan : Plan
        The plan.
    chart : IndexedChart
        A chart holding every final item.
    item : Item
        The item.
    leaves : Mapping of Item to list of AxiomOccurrence
        The occurrences of each input item.

    Returns
    -------
    list
        For an occurrence ``(occurrence, {}, ())``; for an instance
        ``(rule, binding, premises)``.
    """
    found: list[tuple[Rule | AxiomOccurrence, dict[str, Ground], tuple[Item, ...]]] = [
        (occurrence, {}, ()) for occurrence in leaves.get(item, [])
    ]
    for rule in plan.presentation.rules_by_conclusion.get(item.family, ()):
        for binding, premises in chart.incoming(rule, item, lambda _: True):
            found.append((rule, binding, premises))
    return found


def _rebuild(
    plan: Plan, axioms: Sequence[AxiomOccurrence], evaluation: Evaluation
) -> tuple[IndexedChart, dict[Item, list[AxiomOccurrence]]]:
    """A chart of every final item, for enumerating incoming instances.

    Parameters
    ----------
    plan : Plan
        The plan.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.
    evaluation : Evaluation
        A run.

    Returns
    -------
    tuple of IndexedChart and dict
        The chart and the occurrences of each input item.
    """
    chart = IndexedChart(plan.presentation)
    for item in evaluation.order:
        chart.add(item)
    leaves: dict[Item, list[AxiomOccurrence]] = {}
    for occurrence in axioms:
        leaves.setdefault(_axiom_item(plan, chart, occurrence), []).append(occurrence)
    return chart, leaves


def _recompute(
    plan: Plan,
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    evaluation: Evaluation,
    goal: Item,
    like: torch.Tensor,
    adjoint: torch.Tensor | None,
) -> dict[InputId, torch.Tensor]:
    """The reverse sweep that recomputes each item's sum on its own.

    Items are visited in the reverse of the order their values became
    final, so an item's cotangent is complete when it is visited: every
    item that uses it became final later. Each item's terms are rebuilt
    from the final values of its premises, its sum is differentiated by
    autograd in isolation, and the cotangents of the premises and inputs
    are accumulated. Space is linear in the items.

    Parameters
    ----------
    plan : Plan
        The plan.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.
    inputs : Mapping of InputId to torch.Tensor
        The input weights, detached.
    evaluation : Evaluation
        The run without a tape.
    goal : Item
        The item differentiated.
    like : torch.Tensor
        The dtype and device of units and zeros.
    adjoint : torch.Tensor or None
        The cotangent of the goal's value; ones by default.

    Returns
    -------
    dict of InputId to torch.Tensor
        The gradient of each input.
    """
    algebra = plan.algebra
    chart, leaves = _rebuild(plan, axioms, evaluation)
    goal_value = evaluation.value(goal)
    cotangents: dict[Item, torch.Tensor] = {
        goal: torch.ones_like(goal_value) if adjoint is None else adjoint
    }
    gradients: dict[InputId, torch.Tensor] = {
        key: torch.zeros_like(value) for key, value in inputs.items()
    }
    for item in reversed(evaluation.order):
        cotangent = cotangents.pop(item, None)
        if cotangent is None:
            continue
        terms = _terms(plan, chart, item, leaves)
        if not terms:
            continue
        premise_leaves: dict[Item, torch.Tensor] = {}
        input_leaves: dict[InputId, torch.Tensor] = {}

        def leaf_of(key: InputId) -> torch.Tensor:
            """The leaf standing for one input weight in this item's sum.

            Parameters
            ----------
            key : InputId
                The input.

            Returns
            -------
            torch.Tensor
                Its leaf.
            """
            if key not in input_leaves:
                input_leaves[key] = inputs[key].detach().requires_grad_()
            return input_leaves[key]

        pieces: list[torch.Tensor] = []
        for source, _, premises in terms:
            if isinstance(source, AxiomOccurrence):
                slot = plan.presentation.constructors_by_id[source.slot]
                key = (
                    source.occurrence
                    if source.occurrence in inputs
                    else slot.weight
                    if slot.weight in inputs
                    else None
                )
                pieces.append(
                    algebra.unit_like((), like) if key is None else leaf_of(key)
                )
                continue
            weight = (
                leaf_of(source.weight)
                if source.weight is not None and source.weight in inputs
                else algebra.unit_like((), like)
            )
            factors = []
            for premise in premises:
                if premise not in premise_leaves:
                    premise_leaves[premise] = (
                        evaluation.value(premise).detach().requires_grad_()
                    )
                factors.append(premise_leaves[premise])
            factors.append(weight)
            pieces.append(_product(algebra, factors))
        total = pieces[0]
        for piece in pieces[1:]:
            total = _accumulate(algebra, total, piece)
        sources = [*premise_leaves.values(), *input_leaves.values()]
        if not sources:
            continue
        grads = torch.autograd.grad(total, sources, cotangent, allow_unused=True)
        for premise, grad in zip(
            premise_leaves, grads[: len(premise_leaves)], strict=True
        ):
            if grad is not None:
                cotangents[premise] = (
                    grad if premise not in cotangents else cotangents[premise] + grad
                )
        for key, grad in zip(input_leaves, grads[len(premise_leaves) :], strict=True):
            if grad is not None:
                gradients[key] = gradients[key] + grad
    return gradients


def _witness(
    plan: Plan,
    axioms: Sequence[AxiomOccurrence],
    evaluation: Evaluation,
    goal: Item,
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Derivation:
    """A best derivation by following, at each item, the first term attaining it.

    Parameters
    ----------
    plan : Plan
        The plan.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.
    evaluation : Evaluation
        A run over a selective algebra.
    goal : Item
        The item.
    inputs : Mapping of InputId to torch.Tensor
        The input weights.
    like : torch.Tensor
        The dtype and device of units.

    Returns
    -------
    Derivation
        The witness.

    Raises
    ------
    PlanRefused
        If no term of an item attains its value.
    """
    algebra = plan.algebra
    chart, leaves = _rebuild(plan, axioms, evaluation)

    def build(item: Item) -> Derivation:
        """The witness of one item.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        Derivation
            Its best derivation.

        Raises
        ------
        PlanRefused
            If no term attains the item's value.
        """
        target = evaluation.value(item)
        for source, binding, premises in _terms(plan, chart, item, leaves):
            if isinstance(source, AxiomOccurrence):
                term = _occurrence_weight(plan, source, inputs, like)
                if bool(torch.equal(term.detach(), target.detach())):
                    return Derivation(
                        constructor=source.slot,
                        item=item,
                        binding=source.binding,
                        values=source.values,
                        occurrence=source.occurrence,
                    )
                continue
            weight = _weight(plan, source.weight, inputs, like)
            term = _product(algebra, [*(evaluation.value(p) for p in premises), weight])
            if bool(torch.equal(term.detach(), target.detach())):
                return Derivation(
                    constructor=source.id,
                    item=item,
                    binding=tuple(sorted(binding.items())),
                    premises=tuple(build(premise) for premise in premises),
                )
        raise PlanRefused(
            PlanRefusal.ALGEBRA,
            "no term attains the value; the algebra does not select",
        )

    return build(goal)
