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
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import (
    DivergentClosureError,
    OrderPreservingSemiring,
    RealEmbedding,
    Star,
    UndecidedClosureError,
)
from quivers.core.elimination import ELIMINATION_LAWS, fiber_join
from quivers.core.laws import (
    PLUS_IDEMPOTENT,
    REAL_EMBEDDING,
    STAR_LEAST_SOLUTION,
    TIMES_COMMUTATIVE,
    NO_ZERO_DIVISORS,
    ZERO_SUM_FREE,
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
    NOT_DIFFERENTIATED,
    SPECTRAL_RADIUS_BELOW_ONE,
    UNLIMITED,
    AbsoluteError,
    AbsoluteLogError,
    AccuracyReport,
    Attainment,
    BudgetExhausted,
    Certified,
    Completed,
    DerivativeEntry,
    DerivativeReport,
    DerivativeSemantics,
    Enclosure,
    Exact,
    Heuristic,
    FailurePolicy,
    MonotonePolynomial,
    Unestablished,
    attained_guarantee,
    settle,
    unmet_fields,
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
    LinearCyclic,
    Marginal,
    NoRetention,
    NonlinearFold,
    PackedForest,
    OccurrenceId,
    ParameterId,
    Phase,
    PhaseInapplicable,
    ProvedOptimal,
    QuerySpec,
    Rational,
    Recursion,
    RequestedEvidence,
    Retention,
    Statistic,
    Support,
    Unresolved,
    Weight,
    WorkField,
    WorkReport,
    check_admissible,
    measuring,
)
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import IndexBinder, NatSort
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    ComputedRuleWeight,
    Derivation,
    Ground,
    Item,
    PresentationAnalysis,
    Rule,
    RulePresentation,
    evaluate_index,
)
from quivers.qiec.types import (
    IndexConstructor,
    IndexLiteral,
    IndexSum,
    IndexTerm,
    IndexVariable,
)
from quivers.stochastic.forest import DerivationForest, ForestEdge, build_forest
from quivers.stochastic.indexed import ChartWork, IndexedChart, generic_joinable
from quivers.stochastic.recursion import (
    Component,
    CyclicRefusal,
    CyclicRefused,
    NewtonComponent,
    PolynomialSolution,
    solve,
    solve_polynomial,
)
from quivers.stochastic.joins import (
    BodyQuery,
    JoinWork,
    agm_bound,
    body_query,
    eliminate,
    elimination_order,
    relations,
)
from quivers.stochastic.span import (
    SpanGrammar,
    SpanRefused,
    span_grammar,
    span_inside,
    span_recognize,
)

__all__ = [
    "Alternative",
    "CYCLIC_MEASURES",
    "EVALUATION_MEASURES",
    "Evaluation",
    "MeasureCertificate",
    "MeasureComponent",
    "Plan",
    "PlanRecord",
    "PlanRefusal",
    "CYCLIC_ALGORITHM",
    "NEWTON_ALGORITHM",
    "NEWTON_MEASURES",
    "NEWTON_WORK",
    "SPAN_ALGORITHM",
    "SPAN_MEASURES",
    "SPAN_RECOGNITION_ALGORITHM",
    "SPAN_RECOGNITION_WORK",
    "SPAN_WORK",
    "CYCLIC_WORK",
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
    "with_requested_evidence",
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
    ELIMINATION_FRAGMENT = "elimination-fragment"
    CYCLE = "cycle"
    NONLINEAR_CYCLE = "nonlinear-cycle"
    MIXED_SIDES = "mixed-sides"
    DIVERGENT = "divergent"


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


def with_requested_evidence(claim: ComplexityClaim) -> ComplexityClaim:
    """A two-pass cost claim that also keeps the evidence of requested items.

    Parameters
    ----------
    claim : ComplexityClaim
        The claim without evidence.

    Returns
    -------
    ComplexityClaim
        The claim with ``R``, the edges and premise references of the
        requested items, added to the execution space.
    """
    bounds = tuple(
        bound.with_(expression=f"{bound.expression} + R")
        if bound.phase is Phase.EXECUTION and bound.resource == "space"
        else bound
        for bound in claim.bounds
    )
    return claim.with_(
        parameters=(
            *claim.parameters,
            ComplexityParameter(
                name="R",
                meaning="edges and premise references of the requested items",
            ),
        ),
        bounds=bounds,
        conditions=(
            *claim.conditions,
            "R counts only what is requested; the rest of the forest is not kept",
        ),
    )


TWO_PASS_WORK = _two_pass_work("autograd")
"""The cost of two-pass evaluation differentiated by autograd."""

RECOMPUTED_TWO_PASS_WORK = _two_pass_work("recompute")
"""The cost of two-pass evaluation with the recomputing backward pass."""

_LEHMANN = Citation(
    reference="Lehmann 1977, Algebraic structures for transitive closure",
    locator="closed semirings; the closure is the least solution",
    doi="10.1016/0304-3975(77)90056-1",
)

CYCLIC_WORK = ComplexityClaim(
    query="value of every item of a linear cyclic presentation, and its gradient",
    parameters=(
        ComplexityParameter(name="V", meaning="items"),
        ComplexityParameter(name="E", meaning="grounded rule instances and inputs"),
        ComplexityParameter(
            name="J", meaning="candidates the keyed lookups return, at least E"
        ),
        ComplexityParameter(name="P", meaning="the most premises of one rule"),
        ComplexityParameter(
            name="S", meaning="the sizes S_i of the strongly connected components"
        ),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="V + P E + J + sum_i S_i^3",
            status=BestKnown(sources=(_EISNER, _LEHMANN)),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="V + P E + max_i S_i^2",
            status=BestKnown(sources=(_EISNER, _LEHMANN)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="V + P E + J + sum_i S_i^3",
            status=BestKnown(sources=(_EISNER, _LEHMANN)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="V + P E + sum_i S_i^3",
            status=BestKnown(sources=(_EISNER,)),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="join plans and indexes are built from the presentation alone",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="a refit is one more solve at the new weights",
        ),
    ),
    conditions=(
        "the edges are kept to build each component's system",
        "a component's closure costs S^3 operations by elimination of states; "
        "a ring would allow S^omega, and between them the bound is unresolved",
        "the autograd tape holds the eliminations of every closure",
        "a closure outside the algebra's domain is refused, never truncated",
    ),
)
"""The cost of solving a linear cyclic system component by component."""


_VALIANT = Citation(
    reference="Valiant 1975, General context-free recognition in less than cubic time",
    locator="recognition by Boolean matrix products in O(BM(n)) for a fixed grammar",
    doi="10.1016/S0022-0000(75)80046-8",
)
_SPAN_PARAMETERS = (
    ComplexityParameter(name="n", meaning="positions of the input"),
    ComplexityParameter(name="N", meaning="categories"),
    ComplexityParameter(
        name="P", meaning="distinct premise pairs of the binary rule instances"
    ),
    ComplexityParameter(name="R", meaning="binary rule instances"),
    ComplexityParameter(name="C", meaning="side conditions evaluated"),
    ComplexityParameter(
        name="Z",
        meaning=(
            "cells next to an empty span with a nullable category, each closed "
            "on its own"
        ),
    ),
)

SPAN_WORK = ComplexityClaim(
    query="value of every category over every span of a presentation in span form",
    parameters=(
        *_SPAN_PARAMETERS,
        ComplexityParameter(
            name="M",
            meaning="largest closure order, with auxiliary states for parallel unary steps",
        ),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="P n^3 + R n^2 + N^2 n^2 + M^3 + n (R + M^3) + Z M^3 + C",
            status=Unresolved(
                obligation=(
                    "for real weights the inside values are ring computations, so "
                    "Valiant's reduction to matrix products applies in exact "
                    "arithmetic, but its subtractions break the relative accuracy "
                    "of nonnegative sums; in a selective algebra a truly subcubic "
                    "schedule would give one for all-pairs shortest paths"
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="N n^2 + P n^2 + R n + M^2",
            status=Unresolved(obligation="as for time"),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="P n^3 + R n^2 + N^2 n^2 + M^3 + n (R + M^3) + Z M^3 + C",
            status=Unresolved(obligation="as for execution"),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="P n^3 + R n^2 + N^2 n^2 + M^3 + n (R + M^3) + Z M^3",
            status=Unresolved(
                obligation="the autograd tape holds every product of every length"
            ),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason=(
                "the span form is read off the presentation in its size times the "
                "categories its variables range over"
            ),
        ),
        PhaseInapplicable(
            phase=Phase.REFIT, reason="a refit is one more run at the new weights"
        ),
    ),
    conditions=(
        "the split points of a premise pair are summed before its rules, by right "
        "distributivity, so the premises keep their order and the rule weight "
        "comes last",
        "a side condition reading the split point makes its rule a pair of its "
        "own; one reading the span masks its rule",
        "the unary closure is the algebra's star when the unary rules have a "
        "cycle and a sum in topological order of N^3 operations otherwise",
        "in a logarithmic star, retain each parallel unary step as a separate "
        "matrix entry by inserting one auxiliary state per extra step",
        "inputs over empty spans make categories nullable: their weights at each "
        "position solve a polynomial system by components, acyclic or linear by "
        "the star, and a binary rule with an empty premise is a step of its cell "
        "over one span; an empty first premise needs a commutative product",
    ),
)
"""The cost of the span executor."""

SPAN_RECOGNITION_WORK = ComplexityClaim(
    query="which category derives which span of a presentation in span form",
    parameters=_SPAN_PARAMETERS,
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="N^2 n^3 + R n^2 + N^3 log(N) + Z N^3 log(N) + n R N + C",
            status=Unresolved(
                obligation=(
                    "Valiant's divide-and-conquer recognizes a string of a fixed "
                    "grammar in O(BM(n)), below this cubic schedule; whether it "
                    "pays at the lengths here is open and it is not implemented"
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="N n^2 + N^2 n",
            status=Unresolved(obligation="as for time"),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="the span form is read off the presentation",
        ),
        PhaseInapplicable(
            phase=Phase.DIFFERENTIATION, reason="a support has no derivative"
        ),
        PhaseInapplicable(
            phase=Phase.REFIT, reason="a refit is one more run at the new weights"
        ),
    ),
    conditions=(
        "the support of a sum is the union of the supports and that of a product "
        "the intersection: the algebra establishes zero-sum-free and "
        "no-zero-divisors",
        "the products over split points are matrix products of 0-1 matrices, "
        "counted as their multiplications",
    ),
)
"""The cost of span recognition by Boolean matrix products."""


_EY = Citation(
    reference=(
        "Etessami and Yannakakis 2009, Recursive Markov chains, stochastic grammars, "
        "and monotone systems of nonlinear equations"
    ),
    locator="section 6, the decomposed Newton's method and its monotone convergence",
    doi="10.1145/1462153.1462154",
)
_EKL = Citation(
    reference=(
        "Esparza, Kiefer and Luttenberger 2010, Computing the least fixed point of "
        "positive polynomial systems"
    ),
    locator=(
        "SIAM Journal on Computing 39(6); section 6, the number of iterations of "
        "the decomposed method, k_f + i n 2^n for i bits"
    ),
    doi="10.1137/090749591",
)

NEWTON_WORK = ComplexityClaim(
    query="least fixed point of every item of a monotone polynomial presentation",
    parameters=(
        ComplexityParameter(name="V", meaning="items"),
        ComplexityParameter(name="E", meaning="grounded rule instances and inputs"),
        ComplexityParameter(
            name="J", meaning="candidates the keyed lookups return, at least E"
        ),
        ComplexityParameter(name="P", meaning="the most premises of one rule"),
        ComplexityParameter(
            name="S", meaning="the size of the largest strongly connected component"
        ),
        ComplexityParameter(name="K", meaning="Newton steps over all components"),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="V + P E + J + K (S^3 + P^2 E)",
            status=Unresolved(
                obligation=(
                    "the steps K grow with the bits asked for, linearly after a "
                    "threshold k_f that is only shown to exist, and up to n 2^n "
                    "steps per bit in a component of n items; a bound polynomial "
                    "in the bits for every system was not checked"
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="V + P E + S^2",
            status=BestKnown(sources=(_EY, _EKL)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="V + P E + J + V S^2",
            status=BestKnown(sources=(_EY,)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="V + P E + V S",
            status=BestKnown(sources=(_EY,)),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="join plans and indexes are built from the presentation alone",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="a refit is one more solve at the new weights",
        ),
    ),
    conditions=(
        "the edges are kept to build each component's polynomials",
        "each step solves one linear system of the component's Jacobian",
        "the derivative is implicit, (I - J)^-1 dF at the solution, one inverse "
        "per component; the tape holds the polynomials at the solution only",
        "a least fixed point certified infinite is refused, never truncated",
    ),
)
"""The cost of solving a monotone polynomial system component by component."""

NEWTON_MEASURES: tuple[WorkField, ...] = measuring(
    {
        WorkField.ITERATIONS,
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.SCALAR_DIVISIONS,
        WorkField.STAR_PIVOTS,
        WorkField.EXACT_OPERATIONS,
        WorkField.HASH_OPERATIONS,
        WorkField.KEY_COMPARISONS,
    }
)
"""The counters a Newton run counts: its steps, the sums and products of
its floating evaluations, its outward rounded operations, and the chart's
lookups and candidates."""


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
        ``"generic join"``, ``"keyed premise at a time"``, ``"variable elimination"``
        or ``"constant"``
        for a rule without premises.
    rho : Rational
        The weight of a least fractional edge cover of the body's
        variables by its premises, ``rho*``; zero for a constant.
    bound : str
        The bound on the join's lookups and candidates over a run.
    reason : str
        Why the algorithm applies.
    order : tuple of str
        The variables summed out, in the selected elimination order.
    """

    rule: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    algorithm: Literal[
        "generic join", "keyed premise at a time", "variable elimination", "constant"
    ]
    rho: Rational = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    bound: str
    reason: str
    order: tuple[str, ...] = ()


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


EVALUATION_MEASURES: tuple[WorkField, ...] = measuring(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.SCALAR_DIVISIONS,
        WorkField.STAR_PIVOTS,
        WorkField.EXACT_OPERATIONS,
        WorkField.HASH_OPERATIONS,
        WorkField.KEY_COMPARISONS,
        WorkField.PROVENANCE_RECORDS,
    }
)
"""The counters a two-pass evaluation counts: it divides, closes and decides
nothing exactly."""

CYCLIC_MEASURES: tuple[WorkField, ...] = measuring(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.SCALAR_DIVISIONS,
        WorkField.STAR_PIVOTS,
        WorkField.EXACT_OPERATIONS,
        WorkField.HASH_OPERATIONS,
        WorkField.KEY_COMPARISONS,
    }
)
"""The counters a cyclic run counts: the sums and products of building,
closing and applying each component's system, the closures' pivots and
exact decisions as each closure reports them, and the chart's lookups and
candidates. A closure of one element is one pivot, whatever it divides
inside, and nothing outside the closures divides."""


SPAN_MEASURES: tuple[WorkField, ...] = measuring(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.SCALAR_DIVISIONS,
        WorkField.STAR_PIVOTS,
        WorkField.EXACT_OPERATIONS,
        WorkField.HASH_OPERATIONS,
        WorkField.KEY_COMPARISONS,
        WorkField.ELEMENTS_ALLOCATED,
        WorkField.PEAK_LIVE_ELEMENTS,
    }
)
"""The counters a span run counts: its sums and products, the closures'
pivots and exact decisions, and the elements it forms and holds; it looks
up no item and compares no key, and divides nothing outside closures."""


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
    pivots : int
        Closures of one element, one per state a cyclic run eliminates.
    exact : int
        Operations of the closures' exact or enclosed decisions.
    evidence : dict of Item to tuple of ForestEdge, or None
        Under requested evidence, the edges of every derived item of a
        requested form: its input occurrences, then its rule instances;
        ``None`` otherwise.
    components : tuple of Component
        Under linear cyclic recursion, each strongly connected component
        and the side it was solved on; none otherwise.
    measures : tuple of WorkField
        The counters the run counted: `EVALUATION_MEASURES` for two-pass
        runs, `CYCLIC_MEASURES` for cyclic ones and `SPAN_MEASURES` for
        span runs.
    allocated : int
        Elements of the tensors a span run formed.
    peak : int
        Elements a span run held at once.
    iterations : int
        Newton steps of a monotone polynomial run.
    lower : dict of Item to float
        A certified lower bound of every item's real value, in a monotone
        polynomial run; empty otherwise.
    upper : dict of Item to float
        A certified upper bound, ``inf`` where none was certified.
    newton : tuple of NewtonComponent
        Each component of a monotone polynomial run, with its conditioning.
    accuracy : AccuracyReport or None
        What a run of an inexact method reached, which its contract
        settles; ``None`` for exact methods.
    join_work : tuple of JoinWork
        A counter per rule evaluated by variable elimination, in execution order.
    """

    values: dict[Item, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    order: tuple[Item, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    work: ChartWork = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    certified: bool | None = None
    sums: int = 0
    products: int = 0
    counts: int = 0
    pivots: int = 0
    exact: int = 0
    evidence: dict[Item, tuple[ForestEdge, ...]] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    components: tuple[Component, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    measures: tuple[WorkField, ...] = dx.field(
        default=EVALUATION_MEASURES, opaque=True, extras=OPAQUE_BY_VALUE
    )
    allocated: int = 0
    peak: int = 0
    iterations: int = 0
    lower: dict[Item, float] = dx.field(
        default={}, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    upper: dict[Item, float] = dx.field(
        default={}, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    newton: tuple[NewtonComponent, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    accuracy: AccuracyReport | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    join_work: tuple[JoinWork, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def retained(self) -> int:
        """The size of the requested evidence: one per edge and premise reference.

        Returns
        -------
        int
            ``sum_e (1 + |premises(e)|)`` over the kept edges; zero without
            requested evidence.
        """
        if self.evidence is None:
            return 0
        return sum(
            1 + len(edge.premises) for edges in self.evidence.values() for edge in edges
        )

    def report(self) -> WorkReport:
        """The run's counters as a work report of its execution.

        Algebra operations are counted once each, whatever the size of an
        element; lookups are hash operations and candidates key
        comparisons. Only the counters in `measures` are declared
        measured.

        Returns
        -------
        WorkReport
            The report.
        """
        measured = set(self.measures)

        def kept(field: WorkField, count: int) -> int:
            """A count when the run measured it, else nothing.

            Parameters
            ----------
            field : WorkField
                The counter.
            count : int
                What the run tallied for it.

            Returns
            -------
            int
                The count, or zero for a counter not measured.
            """
            return count if field in measured else 0

        return WorkReport(
            phase=Phase.EXECUTION,
            iterations=kept(WorkField.ITERATIONS, self.iterations),
            scalar_sums=kept(WorkField.SCALAR_SUMS, self.sums),
            scalar_products=kept(WorkField.SCALAR_PRODUCTS, self.products),
            star_pivots=kept(WorkField.STAR_PIVOTS, self.pivots),
            exact_operations=kept(WorkField.EXACT_OPERATIONS, self.exact),
            hash_operations=kept(WorkField.HASH_OPERATIONS, self.work.lookups),
            key_comparisons=kept(
                WorkField.KEY_COMPARISONS,
                self.work.candidates
                + sum(w.iterated + w.probes for w in self.join_work),
            ),
            provenance_records=kept(
                WorkField.PROVENANCE_RECORDS, self.counts + self.retained
            ),
            elements_allocated=kept(WorkField.ELEMENTS_ALLOCATED, self.allocated),
            peak_live_elements=kept(WorkField.PEAK_LIVE_ELEMENTS, self.peak),
            measured=self.measures,
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


@dataclass(frozen=True)
class _EliminatedRule:
    query: BodyQuery
    order: tuple[str, ...]
    families: tuple[FamilyId, ...]


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
    span : SpanGrammar or None
        The presentation's span form, when a span executor is selected.
    eliminations : tuple or None
        The family order and each family's rule queries and elimination orders,
        when the partition executor is selected.
    """

    record: PlanRecord = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    algebra: OrderPreservingSemiring = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    span: SpanGrammar | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    eliminations: tuple[tuple[FamilyId, tuple[_EliminatedRule, ...]], ...] | None = (
        dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    )

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
        if isinstance(self.record.recursion, Bounded):
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
        """Run the plan's executor.

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
            The value of every item. A run of an inexact method is
            returned only when it meets its accuracy contract.

        Raises
        ------
        PlanRefused
            If the derived hypergraph has a cycle, the recursion is
            bounded, or the method is inexact and the contract may return
            an unestablished value, which `settled` returns.
        ContractNotMet
            If an inexact run does not meet a contract that refuses.
        """
        self._two_pass()
        self._returns_values()
        evaluation = _run(self, axioms, inputs, like)
        if evaluation.accuracy is None:
            return evaluation
        settled = settle(self.record.accuracy, evaluation, evaluation.accuracy)
        assert isinstance(settled, Evaluation)
        return settled

    def settled(
        self,
        axioms: Sequence[AxiomOccurrence],
        inputs: Mapping[InputId, torch.Tensor],
        *,
        like: torch.Tensor,
    ) -> Evaluation | Unestablished:
        """Run the plan's executor and settle the result by its contract.

        Parameters
        ----------
        axioms : Sequence of AxiomOccurrence
            The input occurrences.
        inputs : Mapping of InputId to torch.Tensor
            The input weights.
        like : torch.Tensor
            The dtype and device of units and zeros.

        Returns
        -------
        Evaluation or Unestablished
            The values when the run meets the contract, or an exact
            method's values; otherwise the values wrapped as unestablished,
            when the contract says to return them.

        Raises
        ------
        PlanRefused
            If the derived hypergraph has a cycle, or the recursion is
            bounded.
        ContractNotMet
            If the run does not meet a contract that refuses.
        """
        self._two_pass()
        evaluation = _run(self, axioms, inputs, like)
        if evaluation.accuracy is None:
            return evaluation
        return settle(self.record.accuracy, evaluation, evaluation.accuracy)

    def _returns_values(self) -> None:
        """Refuse a plain run of a plan that may return an unestablished value.

        Raises
        ------
        PlanRefused
            If the method is inexact and the contract returns values that
            fail it.
        """
        if (
            not self.record.recursion.finite_method
            and self.record.accuracy.on_failure is FailurePolicy.RETURN_UNESTABLISHED
        ):
            raise PlanRefused(
                PlanRefusal.INADMISSIBLE,
                "the contract may return an unestablished value, which settled "
                "and settled_gradient return",
            )

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
        autograd, whose tape holds the selected executor's operations. With ``"recompute"`` the
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
            not use; for an inexact method, only when the run meets its
            contract.

        Raises
        ------
        PlanRefused
            If the derived hypergraph has a cycle, or the method is
            inexact and the contract may return an unestablished value,
            which `settled_gradient` returns.
        KeyError
            If the goal is not derived.
        ContractNotMet
            If an inexact run does not meet a contract that refuses.
        """
        self._returns_values()
        found = self.settled_gradient(axioms, inputs, goal, like=like, adjoint=adjoint)
        assert isinstance(found, dict)
        return found

    def settled_gradient(
        self,
        axioms: Sequence[AxiomOccurrence],
        inputs: Mapping[InputId, torch.Tensor],
        goal: Item,
        *,
        like: torch.Tensor,
        adjoint: torch.Tensor | None = None,
    ) -> dict[InputId, torch.Tensor] | Unestablished:
        """The gradient of a goal's value, settled by the contract.

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
        dict of InputId to torch.Tensor or Unestablished
            The gradient, as `gradient` computes it; for an inexact method
            that does not meet its contract, wrapped as unestablished when
            the contract says to return it.

        Raises
        ------
        PlanRefused
            If the derived hypergraph has a cycle.
        KeyError
            If the goal is not derived.
        ContractNotMet
            If the run does not meet a contract that refuses.
        """
        self._two_pass()
        if self.record.backward == "autograd":
            leaves = {
                key: value.detach().requires_grad_() for key, value in inputs.items()
            }
            evaluation = _run(self, axioms, leaves, like)
            value = evaluation.value(goal)
            if not value.requires_grad:
                found = {key: torch.zeros_like(leaf) for key, leaf in leaves.items()}
            else:
                cotangent = torch.ones_like(value) if adjoint is None else adjoint
                keys = list(leaves)
                grads = torch.autograd.grad(
                    value, [leaves[key] for key in keys], cotangent, allow_unused=True
                )
                found = {
                    key: torch.zeros_like(leaves[key]) if grad is None else grad
                    for key, grad in zip(keys, grads, strict=True)
                }
            if evaluation.accuracy is None:
                return found
            return settle(self.record.accuracy, found, evaluation.accuracy)
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
            If the observable is not `Best`, the recursion is cyclic, or no
            term attains a value.
        """
        if not isinstance(self.record.query.observable, Best):
            raise PlanRefused(
                PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR,
                "a witness is a derivation of a best value; the query is not Best",
            )
        if isinstance(self.record.recursion, LinearCyclic):
            raise PlanRefused(
                PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR,
                "the best value of a cyclic system is its closure in the selective "
                "algebra; following the terms that attain it may cycle",
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

SPAN_ALGORITHM = (
    "span values by length: split points summed per premise pair, then the "
    "rules, then the unary closure"
)
"""The executor for presentations in span form."""

SPAN_RECOGNITION_ALGORITHM = (
    "span recognition by length with Boolean matrix products over the split points"
)
"""The observer plan of the support of a presentation in span form."""

NEWTON_ALGORITHM = (
    "strongly connected components of the derived hypergraph, each polynomial "
    "component solved by Newton's method with a certified enclosure"
)
"""The executor for monotone polynomial systems."""

CYCLIC_ALGORITHM = (
    "strongly connected components of the derived hypergraph, each linear "
    "component solved by the algebra's closure"
)
"""The executor for linear cyclic systems."""

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
    executor: Literal["auto", "eliminate"] = "auto",
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
    executor : {"auto", "eliminate"}
        The default selects among the chart, span and forest executors.
        ``"eliminate"`` requires the exact family-acyclic weight fragment
        and sums out rule variables before combining their conclusions.

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
    if presentation.computed_weight_rules:
        raise PlanRefused(
            PlanRefusal.INADMISSIBLE,
            "the weight planner has no executable attachment for computed rule weights",
        )
    if executor not in ("auto", "eliminate"):
        raise PlanRefused(PlanRefusal.INADMISSIBLE, f"unknown executor {executor!r}")
    if executor == "eliminate":
        return _plan_elimination(
            query, presentation, algebra, recursion, accuracy, retention, claims
        )
    try:
        check_admissible(
            query,
            recursion,
            accuracy,
            algebra=algebra.subject,
            claims=(*algebra.claims(), *claims),
            uses=requirements(
                (*TWO_PASS_LAWS, *_newton_laws(algebra))
                if isinstance(recursion, MonotonePolynomial)
                else TWO_PASS_LAWS,
                algebra.subject,
            ),
        )
    except InadmissibleContract as error:
        raise PlanRefused(PlanRefusal.INADMISSIBLE, str(error)) from error
    kept = isinstance(retention, PackedForest)
    bounded = isinstance(recursion, Bounded) and recursion.measure == "height"
    cyclic = isinstance(recursion, LinearCyclic)
    polynomial = isinstance(recursion, MonotonePolynomial)
    if polynomial and kept:
        raise PlanRefused(
            PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
            "a monotone polynomial system is solved by Newton's method, without a "
            "forest of its infinitely many derivations",
        )
    if cyclic and (kept or not isinstance(algebra, Star)):
        raise PlanRefused(
            PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
            "a linear cyclic system is solved by the algebra's closure, without "
            "a forest of its infinitely many derivations",
        )
    if (
        not isinstance(recursion, Acyclic)
        and not (kept and bounded)
        and not cyclic
        and not polynomial
    ):
        raise PlanRefused(
            PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
            f"the executors here solve acyclic, linear cyclic, monotone polynomial "
            f"and height-bounded systems, not {recursion.tag}",
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
    unknown = (
        [
            pattern.family
            for pattern in retention.patterns
            if pattern.family not in presentation.judgments_by_id
        ]
        if isinstance(retention, RequestedEvidence)
        else []
    )
    if unknown:
        raise PlanRefused(
            PlanRefusal.INADMISSIBLE,
            f"requested evidence names {len(unknown)} forms the presentation lacks",
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
    if isinstance(retention, RequestedEvidence):
        assumptions.append(
            "the evidence of a requested item is every edge concluding it, kept "
            "when the first pass produces it, so it is the packed forest "
            "restricted to the requested items"
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
    if cyclic:
        assumptions.append(
            "each strongly connected component of a run is linear, one premise "
            "of an edge inside it; it is solved on the right as M* b when every "
            "edge's premise in it comes last and its rule has no weight, on the "
            "left as b N* when it comes first, and otherwise only when the "
            "product is established to commute; each run checks this"
        )
    budget = accuracy.budget
    backward: Literal["autograd", "recompute"] = (
        "recompute"
        if (budget.elements is not None or budget.bytes is not None)
        and not cyclic
        and not polynomial
        else "autograd"
    )
    span, declined = _span_form(query, presentation, retention, recursion, backward)
    if span is not None and cyclic and budget != UNLIMITED:
        raise PlanRefused(
            PlanRefusal.INADMISSIBLE,
            "the cyclic span executor does not enforce a general resource budget",
        )
    recognizes = (
        span is not None
        and isinstance(query.observable, Support)
        and algebra.establishes(ZERO_SUM_FREE, NO_ZERO_DIVISORS)
    )
    if span is not None:
        if span.unary_cyclic and not cyclic:
            raise PlanRefused(
                PlanRefusal.CYCLE,
                "the unary rules have a cycle; the system is linear cyclic, not "
                "acyclic",
            )
        assumptions.append(
            "the presentation is in span form: the spans of each length are final "
            "once the shorter ones are, so they are computed length by length"
        )
    if polynomial:
        assumptions.append(
            "each strongly connected component of a run is a polynomial system with "
            "nonnegative coefficients in the reals of the algebra's embedding; "
            "Newton's method from zero keeps a certified lower bound and a "
            "pre-fixed point bounds it from above, both in outward rounded "
            "arithmetic, and the bounds of a component feed those above it"
        )
    algorithm = (
        NEWTON_ALGORITHM
        if polynomial
        else FOREST_ALGORITHM
        if kept
        else SPAN_RECOGNITION_ALGORITHM
        if recognizes
        else SPAN_ALGORITHM
        if span is not None
        else CYCLIC_ALGORITHM
        if cyclic
        else TWO_PASS_ALGORITHM
    )
    complexity = (
        NEWTON_WORK
        if polynomial
        else SPAN_RECOGNITION_WORK
        if recognizes
        else SPAN_WORK
        if span is not None
        else CYCLIC_WORK
        if cyclic
        else _complexity(kept, backward, retention)
    )
    record = PlanRecord(
        query=query,
        recursion=recursion,
        accuracy=accuracy,
        retention=retention,
        algebra=algebra.subject,
        claims=(*algebra.claims(), *claims),
        algorithm=algorithm,
        assumptions=tuple(assumptions),
        structure=analysis,
        certificate=certificate,
        complexity=complexity,
        alternatives=(
            Alternative(
                algorithm=SPAN_ALGORITHM if span is None else TWO_PASS_ALGORITHM,
                reason=declined
                if span is None
                else "enumerates every grounded instance one at a time; the span "
                "form lets a length's cells be computed together",
            ),
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
    return Plan(record=record, presentation=presentation, algebra=algebra, span=span)


_ELIMINATION_ALGORITHM = "variable elimination in judgment-family order"

_ELIMINATION_MEASURES = measuring(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.KEY_COMPARISONS,
        WorkField.PROVENANCE_RECORDS,
    }
)

_ELIMINATION_COMPLEXITY = ComplexityClaim(
    query="exact weight without provenance on an acyclic family graph",
    parameters=(
        ComplexityParameter(name="S", meaning="work of compiling the rule queries"),
        ComplexityParameter(name="G", meaning="the size of the presentation"),
        ComplexityParameter(
            name="I", meaning="input occurrences and item scans by rules"
        ),
        ComplexityParameter(
            name="J", meaning="work of the selected bag and output joins"
        ),
        ComplexityParameter(name="A", meaning="all sums and products in the run"),
        ComplexityParameter(name="V", meaning="items in the result"),
        ComplexityParameter(
            name="M", meaning="largest live set of factor and trie rows"
        ),
        ComplexityParameter(name="P", meaning="elements per algebra operand"),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=tuple(
        ComplexityBound(
            phase=phase,
            resource=resource,
            expression=expression,
            status=Unresolved(
                obligation="tighter bounds across rules and order choices for the family DAG"
            ),
        )
        for phase, time, space in (
            (Phase.CONSTRUCTION, "S", "S"),
            (Phase.EXECUTION, "G + I + J + A", "G + V + M"),
            (Phase.DIFFERENTIATION, "G + I + J + A", "G + V + M + P A"),
            (Phase.REFIT, "G + I + J + A", "G + V + M"),
        )
        for resource, expression in (("time", time), ("space", space))
    ),
    conditions=(
        "expected constant-time hashing and fixed-size patterns for data complexity",
        "S includes graph analysis, greedy order selection and full-body edge covers",
        "J includes trie construction, cover reporting, partial bindings and the free output",
        "execution and refit space exclude the autograd tape",
        "a run counts algebra operations, independent of tensor shape",
        "the differentiation space assumes a bounded number of saved operand tensors per operation",
    ),
)


def _elimination_pattern(term: IndexTerm) -> bool:
    if isinstance(term, IndexConstructor):
        return all(_elimination_pattern(argument) for argument in term.arguments)
    if isinstance(term, IndexSum):
        return len(term.terms) <= 1 and term.constant >= 0
    return isinstance(term, (IndexVariable, IndexLiteral))


def _plan_elimination(
    query: QuerySpec,
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    recursion: Recursion,
    accuracy: AccuracyContract,
    retention: Retention,
    claims: Sequence[LawClaim],
) -> Plan:
    if not isinstance(query.observable, Weight):
        raise PlanRefused(
            PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR,
            f"variable elimination answers weight, but the query asks for {query.observable.tag}",
        )
    if not isinstance(retention, NoRetention):
        raise PlanRefused(
            PlanRefusal.RETENTION_WITHOUT_EXECUTOR,
            f"variable elimination keeps no provenance, but the query asks for {retention.tag}",
        )
    if not isinstance(recursion, Acyclic):
        raise PlanRefused(
            PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
            "variable elimination requires an acyclic judgment-family graph",
        )
    if accuracy.budget != UNLIMITED:
        raise PlanRefused(
            PlanRefusal.INADMISSIBLE,
            "variable elimination does not enforce resource limits",
        )
    missing = [law.name for law in ELIMINATION_LAWS if not algebra.establishes(law)]
    if missing:
        raise PlanRefused(
            PlanRefusal.ALGEBRA,
            f"variable elimination requires established laws {missing}",
        )
    try:
        check_admissible(
            query,
            recursion,
            accuracy,
            algebra=algebra.subject,
            claims=(*algebra.claims(), *claims),
            uses=requirements(ELIMINATION_LAWS, algebra.subject),
        )
    except InadmissibleContract as error:
        raise PlanRefused(PlanRefusal.INADMISSIBLE, str(error)) from error

    dependencies: dict[FamilyId, dict[FamilyId, None]] = {
        family.id: {} for family in presentation.judgments
    }
    for rule in presentation.rules:
        dependencies[rule.conclusion.family].update(
            (premise.judgment.family, None) for premise in rule.premises
        )
    try:
        families = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        cycle = " -> ".join(
            presentation.judgments_by_id[family].name for family in error.args[1]
        )
        raise PlanRefused(
            PlanRefusal.CYCLE,
            f"variable elimination found a judgment-family cycle: {cycle}",
        ) from error

    orders: dict[FamilyId, list[_EliminatedRule]] = {family: [] for family in families}
    records: list[RuleJoin] = []
    for rule in presentation.rules:
        if any(premise.hypotheses for premise in rule.premises):
            raise PlanRefused(
                PlanRefusal.OPEN_PREMISES, f"{rule.name!r} has open premises"
            )
        judgments = [premise.judgment for premise in rule.premises] + [rule.conclusion]
        if rule.side_condition is not None or not all(
            _elimination_pattern(argument)
            for judgment in judgments
            for argument in judgment.arguments
        ):
            raise PlanRefused(
                PlanRefusal.ELIMINATION_FRAGMENT,
                f"variable elimination requires independent flat patterns and no "
                f"side condition in {rule.name!r}",
            )
        body = body_query(rule)
        unbound = {binder.name for binder in rule.telescope} - set(body.variables)
        if unbound:
            raise PlanRefused(
                PlanRefusal.ELIMINATION_FRAGMENT,
                f"{rule.name!r} leaves {sorted(unbound)} unbound by its premises",
            )
        order = elimination_order(body)
        premises = tuple(dict.fromkeys(atom.family for atom in body.atoms))
        orders[rule.conclusion.family].append(_EliminatedRule(body, order, premises))
        edges = [atom.variables for atom in body.atoms]
        rho = agm_bound(body.variables, edges, [2] * len(edges)).rho
        records.append(
            RuleJoin(
                rule=rule.id,
                name=rule.name,
                algorithm="variable elimination" if edges else "constant",
                rho=Rational.of(rho),
                bound="sum of the chosen bag and free-output join bounds"
                if edges
                else "1",
                reason="premise families have final values before the rule runs",
                order=order,
            )
        )
    record = PlanRecord(
        query=query,
        recursion=recursion,
        accuracy=accuracy,
        retention=retention,
        algebra=algebra.subject,
        claims=(*algebra.claims(), *claims),
        algorithm=_ELIMINATION_ALGORITHM,
        assumptions=(
            "every rule uses only earlier judgment families",
            "one factor for each ordered premise, using the sum of its item's derivations",
            "distributivity permits summing a binding variable before multiplying by "
            "factors that do not contain it",
        ),
        structure=presentation.analysis,
        complexity=_ELIMINATION_COMPLEXITY,
        alternatives=(
            Alternative(
                algorithm=TWO_PASS_ALGORITHM,
                reason="joins every grounded rule instance before aggregating its weight",
            ),
        ),
        joins=tuple(records),
    )
    return Plan(
        record=record,
        presentation=presentation,
        algebra=algebra,
        eliminations=tuple((family, tuple(orders[family])) for family in families),
    )


def _span_form(
    query: QuerySpec,
    presentation: RulePresentation,
    retention: Retention,
    recursion: Recursion,
    backward: Literal["autograd", "recompute"],
) -> tuple[SpanGrammar | None, str]:
    """The span form of a presentation, when the span executor answers the query.

    Parameters
    ----------
    query : QuerySpec
        The query.
    presentation : RulePresentation
        The presentation.
    retention : Retention
        The provenance kept.
    recursion : Recursion
        The recursion contract.
    backward : {"autograd", "recompute"}
        The backward pass the accuracy contract asks for.

    Returns
    -------
    tuple of SpanGrammar or None and str
        The span form, or ``None`` with the reason the span executor is
        not selected.
    """
    if not isinstance(retention, NoRetention):
        return None, "keeps no provenance, which the query asks for"
    if not isinstance(query.observable, (Weight, Support, Marginal)):
        return (
            None,
            f"answers values, supports and marginals, not {query.observable.tag}",
        )
    if not isinstance(recursion, (Acyclic, LinearCyclic)):
        return None, f"solves acyclic and unary cyclic systems, not {recursion.tag}"
    if backward != "autograd":
        return None, "its gradient is by autograd, and the contract bounds memory"
    try:
        return span_grammar(presentation), ""
    except SpanRefused as refused:
        return None, f"the presentation is not in span form: {refused.reason}"


def _complexity(
    kept: bool,
    backward: Literal["autograd", "recompute"],
    retention: Retention,
) -> ComplexityClaim:
    """The cost claim of the selected executor.

    Parameters
    ----------
    kept : bool
        Whether the packed forest is kept.
    backward : {"autograd", "recompute"}
        The backward pass of a two-pass plan.
    retention : Retention
        The provenance kept.

    Returns
    -------
    ComplexityClaim
        The claim.
    """
    if kept:
        return FOREST_WORK
    claim = TWO_PASS_WORK if backward == "autograd" else RECOMPUTED_TWO_PASS_WORK
    if isinstance(retention, RequestedEvidence):
        return with_requested_evidence(claim)
    return claim


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
    key: InputId | ComputedRuleWeight | None,
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    """The weight of a rule's parameter or an occurrence, the unit when absent.

    Parameters
    ----------
    plan : Plan
        The plan.
    key : InputId, ComputedRuleWeight, or None
        The parameter or occurrence. A computed weight needs an executable attachment.
    inputs : Mapping of InputId to torch.Tensor
        The input weights.
    like : torch.Tensor
        The dtype and device of the unit.

    Returns
    -------
    torch.Tensor
        The weight.

    Raises
    ------
    PlanRefused
        If a rule has a computed weight.
    """
    if isinstance(key, ComputedRuleWeight):
        raise PlanRefused(
            PlanRefusal.INADMISSIBLE,
            "the weight planner has no executable attachment for computed rule weights",
        )
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


def _axiom_item(
    plan: Plan, occurrence: AxiomOccurrence, chart: IndexedChart | None = None
) -> Item:
    """The item of an input occurrence.

    Parameters
    ----------
    plan : Plan
        The plan.
    occurrence : AxiomOccurrence
        The occurrence.
    chart : IndexedChart or None
        The chart that interns the item, when the executor uses one.

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
    if chart is None:
        return Item(family=slot.judgment.family, arguments=arguments)
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


type Kept = dict[Item, list[ForestEdge]]
"""The edges of the requested items, by item."""


def _keep(kept: Kept, item: Item, edge: ForestEdge) -> None:
    """Keep one edge of a requested item.

    Every edge is kept, each alternative being part of the item's evidence.

    Parameters
    ----------
    kept : Kept
        The evidence so far.
    item : Item
        The item.
    edge : ForestEdge
        One of its edges.
    """
    kept.setdefault(item, []).append(edge)


def _requested(plan: Plan) -> frozenset[FamilyId]:
    """The forms whose evidence the plan keeps.

    Parameters
    ----------
    plan : Plan
        The plan.

    Returns
    -------
    frozenset of FamilyId
        The requested forms; none without requested evidence.
    """
    retention = plan.record.retention
    if isinstance(retention, RequestedEvidence):
        return frozenset(pattern.family for pattern in retention.patterns)
    return frozenset()


def _discover(
    plan: Plan, chart: IndexedChart, axioms: Sequence[AxiomOccurrence]
) -> tuple[list[Item], dict[Item, int], dict[Item, list[AxiomOccurrence]], Kept]:
    """The first pass: every item, how many instances conclude it, and requested evidence.

    Each instance is enumerated once here, so keeping it when its
    conclusion is requested keeps each edge of a requested item once.

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
        each, the occurrences of each input item, and the edges of each
        requested item.
    """
    requested = _requested(plan)
    kept: Kept = {}
    leaves: dict[Item, list[AxiomOccurrence]] = {}
    counts: dict[Item, int] = {}
    discovered: list[Item] = []
    seen: set[Item] = set()
    queue: deque[Item] = deque()
    for occurrence in axioms:
        item = _axiom_item(plan, occurrence, chart)
        leaves.setdefault(item, []).append(occurrence)
        if item.family in requested:
            _keep(
                kept,
                item,
                ForestEdge(
                    constructor=occurrence.slot,
                    binding=occurrence.binding,
                    occurrence=occurrence.occurrence,
                    values=occurrence.values,
                ),
            )
        if item not in seen:
            seen.add(item)
            discovered.append(item)
            counts[item] = 0
            queue.append(item)
    for rule, item in _constants(plan, chart):
        counts[item] = counts.get(item, 0) + 1
        if item.family in requested:
            _keep(kept, item, ForestEdge(constructor=rule.id))
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
            for binding, premises in chart.instances(
                rule, position, item, popped.__contains__
            ):
                conclusion = chart.items.intern(
                    rule.conclusion.family,
                    tuple(
                        evaluate_index(a, binding) for a in rule.conclusion.arguments
                    ),
                )
                counts[conclusion] = counts.get(conclusion, 0) + 1
                if conclusion.family in requested:
                    _keep(
                        kept,
                        conclusion,
                        ForestEdge(
                            constructor=rule.id,
                            binding=tuple(sorted(binding.items())),
                            premises=premises,
                        ),
                    )
                if conclusion not in seen:
                    seen.add(conclusion)
                    discovered.append(conclusion)
                    queue.append(conclusion)
    return discovered, counts, leaves, kept


def _run(
    plan: Plan,
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Evaluation:
    """Run the plan's executor.

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
        If the run has no value under the plan.
    """
    if plan.eliminations is not None:
        return _run_elimination(plan, plan.eliminations, axioms, inputs, like)
    if plan.span is not None:
        return _run_span(plan, plan.span, axioms, inputs, like)
    if isinstance(plan.record.recursion, MonotonePolynomial):
        return _solve_monotone(plan, axioms, inputs, like)
    if isinstance(plan.record.recursion, LinearCyclic):
        return _solve_cyclic(plan, axioms, inputs, like)
    return _evaluate(plan, axioms, inputs, like)


def _run_elimination(
    plan: Plan,
    stages: tuple[tuple[FamilyId, tuple[_EliminatedRule, ...]], ...],
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Evaluation:
    """Finalize each family's sum before joining any rule that uses it."""
    values: dict[Item, torch.Tensor] = {}
    by_family: dict[FamilyId, dict[Item, None]] = {family: {} for family, _ in stages}
    reports: list[JoinWork] = []
    sums = 0

    def add(item: Item, term: torch.Tensor) -> None:
        nonlocal sums
        current = values.get(item)
        sums += current is not None
        values[item] = _accumulate(plan.algebra, current, term)
        by_family[item.family][item] = None

    for occurrence in axioms:
        add(
            _axiom_item(plan, occurrence),
            _occurrence_weight(plan, occurrence, inputs, like),
        )
    for family, rules in stages:
        for stage in rules:
            body = stage.query
            weight = _weight(plan, body.rule.weight, inputs, like)
            if not body.atoms:
                arguments = tuple(
                    evaluate_index(a, {}) for a in body.rule.conclusion.arguments
                )
                add(Item(family=family, arguments=arguments), weight)
                continue
            items = itertools.chain.from_iterable(by_family[f] for f in stage.families)
            result = eliminate(
                body,
                relations(body, items),
                plan.presentation,
                plan.algebra,
                values,
                like=like,
                rule_weight=weight,
                order=stage.order,
            )
            reports.append(result.work)
            for item, term in result.values.items():
                add(item, term)
    return Evaluation(
        values=values,
        order=tuple(itertools.chain.from_iterable(by_family.values())),
        work=ChartWork(),
        sums=sums + sum(work.sums for work in reports),
        products=sum(work.products for work in reports),
        measures=_ELIMINATION_MEASURES,
        join_work=tuple(reports),
    )


def _run_span(
    plan: Plan,
    grammar: SpanGrammar,
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Evaluation:
    """Run the span executor, or the recognition plan for a support.

    The items reported are the cells some derivation reaches when every
    input and rule is present, which recognition with every weight
    present finds; a cell whose derivations all weigh zero is reported
    with the zero, as the two-pass executor reports it.

    Parameters
    ----------
    plan : Plan
        The plan.
    grammar : SpanGrammar
        Its span form.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.
    inputs : Mapping of InputId to torch.Tensor
        The input weights.
    like : torch.Tensor
        The dtype and device of units and zeros.

    Returns
    -------
    Evaluation
        The value of every derivable cell, by length; Booleans under the
        recognition plan.

    Raises
    ------
    PlanRefused
        If an input names no slot or spans no position.
    """
    algebra = plan.algebra
    size = len(grammar.categories)
    placed: list[tuple[int, int, int, torch.Tensor]] = []
    for occurrence in axioms:
        slot = plan.presentation.constructors_by_id.get(occurrence.slot)
        if not isinstance(slot, AxiomSlot):
            raise PlanRefused(PlanRefusal.INADMISSIBLE, "an occurrence names no slot")
        binding = dict(occurrence.binding)
        item = Item(
            family=slot.judgment.family,
            arguments=tuple(
                evaluate_index(argument, binding)
                for argument in slot.judgment.arguments
            ),
        )
        start, end = grammar.positions(item)
        placed.append(
            (
                start,
                end,
                grammar.category(item),
                _occurrence_weight(plan, occurrence, inputs, like),
            )
        )
    for constant in grammar.constants:
        item = Item(
            family=grammar.family,
            arguments=tuple(
                evaluate_index(argument, constant.binding)
                for argument in constant.rule.conclusion.arguments
            ),
        )
        start, end = grammar.positions(item)
        placed.append(
            (
                start,
                end,
                constant.result,
                _weight(plan, constant.rule.weight, inputs, like),
            )
        )
    for start, end, _, _ in placed:
        if end < start:
            raise PlanRefused(
                PlanRefusal.INADMISSIBLE,
                f"an input ends at {end} before its start {start}; a span runs forward",
            )
    length = max((end for _, end, _, _ in placed), default=0)
    by_length: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
    for start, end, category, weight in placed:
        by_length.setdefault(end - start, []).append((start, category, weight))
    binary = [_weight(plan, rule.rule.weight, inputs, like) for rule in grammar.binary]
    unary = [_weight(plan, rule.rule.weight, inputs, like) for rule in grammar.unary]
    structural = {
        width: _support_seed(length - width + 1, size, entries, None, like)
        for width, entries in by_length.items()
    }
    support, _ = span_recognize(
        grammar,
        structural,
        length,
        [True] * len(grammar.binary),
        [True] * len(grammar.unary),
        device=like.device,
        empty=_empty_categories(by_length.get(0, ()), length, None),
    )
    cyclic = isinstance(plan.record.recursion, LinearCyclic)
    if plan.record.algorithm == SPAN_RECOGNITION_ALGORITHM:
        seeds = {
            width: _support_seed(length - width + 1, size, entries, algebra, like)
            for width, entries in by_length.items()
        }
        chart, work = span_recognize(
            grammar,
            seeds,
            length,
            [not bool(algebra.is_zero(weight).all()) for weight in binary],
            [not bool(algebra.is_zero(weight).all()) for weight in unary],
            device=like.device,
            empty=_empty_categories(by_length.get(0, ()), length, algebra),
        )
    else:
        seeds = {
            width: _value_seed(algebra, length - width + 1, size, entries, like)
            for width, entries in by_length.items()
        }
        try:
            chart, work = span_inside(
                grammar,
                algebra,
                seeds,
                length,
                binary,
                unary,
                like=like,
                empty=_empty_categories(by_length.get(0, ()), length, None),
                cyclic=cyclic,
            )
        except SpanRefused as refused:
            raise PlanRefused(_SPAN_CODES[refused.code], refused.reason) from refused
        except (DivergentClosureError, UndecidedClosureError) as refused:
            raise PlanRefused(PlanRefusal.DIVERGENT, str(refused)) from refused
    values: dict[Item, torch.Tensor] = {}
    order: list[Item] = []
    for width in range(0 if 0 in support else 1, length + 1):
        reached = support[width]
        for start in range(length - width + 1):
            for category in range(size):
                if bool(reached[start, category]):
                    item = grammar.item(category, start, start + width)
                    values[item] = chart[width][start, category]
                    order.append(item)
    certificate = plan.record.certificate
    return Evaluation(
        values=values,
        order=tuple(order),
        work=ChartWork(conditions=work.conditions),
        certified=None if certificate is None else certificate.applies(order),
        sums=work.sums,
        products=work.products,
        pivots=work.pivots,
        exact=work.exact,
        measures=SPAN_MEASURES,
        allocated=work.allocated,
        peak=work.peak,
    )


_SPAN_CODES = {
    "form": PlanRefusal.INADMISSIBLE,
    "cycle": PlanRefusal.CYCLE,
    "nonlinear": PlanRefusal.NONLINEAR_CYCLE,
    "sides": PlanRefusal.MIXED_SIDES,
    "algebra": PlanRefusal.ALGEBRA,
}
"""The plan's refusal for each refusal of the span executor."""


def _empty_categories(
    entries: Sequence[tuple[int, int, torch.Tensor]],
    length: int,
    algebra: OrderPreservingSemiring | None,
) -> list[set[int]]:
    """The categories of the inputs over each empty span.

    Parameters
    ----------
    entries : Sequence of tuple
        Each input over an empty span: its position, category and weight.
    length : int
        The last position.
    algebra : OrderPreservingSemiring or None
        The algebra whose zero is absent; ``None`` counts every input.

    Returns
    -------
    list of set of int
        One set per position, all empty without such inputs.
    """
    found: list[set[int]] = [set() for _ in range(length + 1)] if entries else []
    for position, category, weight in entries:
        if algebra is None or not bool(algebra.is_zero(weight).all()):
            found[position].add(category)
    return found


def _support_seed(
    rows: int,
    size: int,
    entries: Sequence[tuple[int, int, torch.Tensor]],
    algebra: OrderPreservingSemiring | None,
    like: torch.Tensor,
) -> torch.Tensor:
    """The inputs of one length present in each cell.

    Parameters
    ----------
    rows : int
        The cells' starts.
    size : int
        The categories.
    entries : Sequence of tuple
        Each input's start, category and weight.
    algebra : OrderPreservingSemiring or None
        The algebra whose zero is absent; ``None`` counts every input as
        present, its structural support.
    like : torch.Tensor
        The device.

    Returns
    -------
    torch.Tensor
        ``(rows, size)`` Booleans.
    """
    present = torch.zeros((rows, size), dtype=torch.bool, device=like.device)
    for start, category, weight in entries:
        if algebra is None or not bool(algebra.is_zero(weight).all()):
            present[start, category] = True
    return present


def _value_seed(
    algebra: OrderPreservingSemiring,
    rows: int,
    size: int,
    entries: Sequence[tuple[int, int, torch.Tensor]],
    like: torch.Tensor,
) -> torch.Tensor:
    """The sum of the inputs of one length in each cell.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    rows : int
        The cells' starts.
    size : int
        The categories.
    entries : Sequence of tuple
        Each input's start, category and weight.
    like : torch.Tensor
        The dtype and device.

    Returns
    -------
    torch.Tensor
        ``(rows, size, *element_shape)``: each cell's inputs summed, the
        zero where there is none.
    """
    fibers: list[list[int]] = [[] for _ in range(rows * size)]
    for position, (start, category, _) in enumerate(entries):
        fibers[start * size + category].append(position)
    stacked = torch.stack([weight for _, _, weight in entries])
    summed = fiber_join(algebra, stacked, fibers, algebra.zero_like((1,), like), axis=0)
    return summed.reshape(rows, size, *stacked.shape[1:])


def _newton_laws(algebra: OrderPreservingSemiring) -> tuple[Law, ...]:
    """The laws Newton's method uses in an algebra.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.

    Returns
    -------
    tuple of Law
        A commutative product, and the real embedding for certified Newton
        or, without one, an idempotent sum with a star for exact Newton.
    """
    if isinstance(algebra, RealEmbedding):
        return (TIMES_COMMUTATIVE, REAL_EMBEDDING)
    return (TIMES_COMMUTATIVE, PLUS_IDEMPOTENT, STAR_LEAST_SOLUTION)


def _rounding(algebra: OrderPreservingSemiring) -> int:
    """The rounding of an algebra's embedding in the reals.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra, a `RealEmbedding` under a monotone polynomial plan.

    Returns
    -------
    int
        Units in the last place.
    """
    assert isinstance(algebra, RealEmbedding)
    return algebra.embedding_rounding


def _exact_newton(
    plan: Plan, forest: DerivationForest, solution: PolynomialSolution
) -> Evaluation:
    """The evaluation and report of exact Newton in an idempotent algebra.

    Parameters
    ----------
    plan : Plan
        The plan.
    forest : DerivationForest
        The derived hypergraph.
    solution : PolynomialSolution
        The solution.

    Returns
    -------
    Evaluation
        The values and the report: exact when every component took its
        steps, without a guarantee when the budget ended one first.
    """
    contract = plan.record.accuracy
    stop = (
        BudgetExhausted(
            resource="iterations", limit=Rational.of(contract.budget.iterations or 0)
        )
        if solution.stopped == "budget"
        else Completed()
    )
    guarantee = attained_guarantee(stop, Exact(), None)
    wanted = contract.gradient
    derivative = (
        NOT_DIFFERENTIATED
        if wanted.missing
        else DerivativeSemantics.none(
            "the least fixed point of an idempotent system is piecewise in its "
            "weights; no implicit derivative is claimed"
        )
    )
    derivatives = (
        DerivativeReport()
        if wanted.missing
        else DerivativeReport(
            entries=tuple(
                DerivativeEntry(
                    output="least fixed point", wrt=key, semantics=derivative
                )
                for key in plan.record.query.differentiate_with_respect_to
            )
        )
    )
    work = WorkReport(
        phase=Phase.EXECUTION,
        iterations=solution.iterations,
        scalar_sums=solution.sums,
        scalar_products=solution.products,
        hash_operations=forest.work.lookups,
        key_comparisons=forest.work.candidates,
        measured=NEWTON_MEASURES,
    )
    attainment = Attainment(
        guarantee=guarantee,
        metric=contract.target.metric,
        rounding=Rational.of(0),
        derivative=derivative,
    )
    report = AccuracyReport(
        attainment=attainment,
        stop=stop,
        unmet=unmet_fields(contract, attainment, (work,)),
        derivatives=derivatives,
        work=(work,),
        claims=tuple(
            plan.algebra.claim(law)
            for law in (*TWO_PASS_LAWS, *_newton_laws(plan.algebra))
        ),
    )
    return Evaluation(
        values=solution.values,
        order=tuple(solution.values),
        work=forest.work,
        sums=solution.sums,
        products=solution.products,
        iterations=solution.iterations,
        measures=NEWTON_MEASURES,
        newton=solution.components,
        accuracy=report,
    )


def _solve_monotone(
    plan: Plan,
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Evaluation:
    """Solve a monotone polynomial system by decomposed Newton, and report it.

    The report's guarantee is the certified bound of the widest interval
    in the contract's metric, with that interval as its certificate; a
    component whose upper end is not certified leaves the run without a
    guarantee, which names the conditioning there.

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
        The values, their intervals, and the accuracy report.

    Raises
    ------
    PlanRefused
        If a component's least fixed point is certified infinite.
    """
    forest = build_forest(plan.presentation, axioms)
    contract = plan.record.accuracy
    try:
        solution = solve_polynomial(
            forest,
            plan.algebra,
            inputs,
            like=like,
            budget=contract.budget.iterations,
        )
    except CyclicRefused as refused:
        raise PlanRefused(_CYCLIC_CODES[refused.code], refused.reason) from refused
    if solution.decided or not isinstance(plan.algebra, RealEmbedding):
        return _exact_newton(plan, forest, solution)
    metric = contract.target.metric
    logarithmic = not isinstance(metric, AbsoluteError)
    widest: tuple[float, float] | None = None
    certified = all(part.certified for part in solution.components)
    if certified:
        for item in solution.values:
            low, high = solution.lower[item], solution.upper[item]
            if logarithmic:
                if high == 0:
                    continue
                if low == 0:
                    certified = False
                    break
                ends = (
                    math.nextafter(math.nextafter(math.log(low), -math.inf), -math.inf),
                    math.nextafter(math.nextafter(math.log(high), math.inf), math.inf),
                )
            else:
                ends = (low, high)
            if widest is None or ends[1] - ends[0] > widest[1] - widest[0]:
                widest = ends
    if certified and widest is None:
        widest = (0.0, 0.0)
    stop = (
        BudgetExhausted(
            resource="iterations", limit=Rational.of(contract.budget.iterations or 0)
        )
        if solution.stopped == "budget"
        else Completed()
    )
    worst = max(
        (part for part in solution.components if not part.certified),
        key=lambda part: part.condition or math.inf,
        default=None,
    )
    if certified and widest is not None:
        certificate = Enclosure(
            lower=Rational.of(widest[0]),
            upper=Rational.of(widest[1]),
            conditions=(
                "every item lies in its interval, a certified lower bound of "
                "Newton's iteration and a pre-fixed point; the widest is shown",
                "the embedding's maps within "
                f"{_rounding(plan.algebra)} units in the last place",
            ),
        )
        method: Certified | Heuristic = Certified(
            bound=Rational.of(math.nextafter(widest[1] - widest[0], math.inf)),
            conditions=certificate.conditions,
            certificate=certificate,
        )
    else:
        method = Heuristic(
            reason="no pre-fixed point was certified above the estimate"
            + (
                f"; the spectral radius there is about {worst.spectral} and the "
                f"condition number {worst.condition}"
                if worst is not None
                else ""
            )
        )
    guarantee = attained_guarantee(
        stop, method, method.certificate if isinstance(method, Certified) else None
    )
    implicit = all(part.implicit for part in solution.components)
    wanted = contract.gradient
    if wanted.missing:
        derivative = NOT_DIFFERENTIATED
        derivatives = DerivativeReport()
    else:
        derivative = (
            DerivativeSemantics.implicit(SPECTRAL_RADIUS_BELOW_ONE)
            if implicit
            else DerivativeSemantics.none(
                "the spectral radius at the solution is not certified below one"
            )
        )
        derivatives = DerivativeReport(
            entries=tuple(
                DerivativeEntry(
                    output="least fixed point", wrt=key, semantics=derivative
                )
                for key in plan.record.query.differentiate_with_respect_to
            )
        )
    work = WorkReport(
        phase=Phase.EXECUTION,
        iterations=solution.iterations,
        scalar_sums=solution.sums,
        scalar_products=solution.products,
        exact_operations=solution.exact,
        hash_operations=forest.work.lookups,
        key_comparisons=forest.work.candidates,
        measured=NEWTON_MEASURES,
    )
    attainment = Attainment(
        guarantee=guarantee,
        metric=metric
        if isinstance(metric, (AbsoluteError, AbsoluteLogError))
        else AbsoluteLogError(),
        rounding=Rational.of(0),
        derivative=derivative,
    )
    used = tuple(
        plan.algebra.claim(law) for law in (*TWO_PASS_LAWS, *_newton_laws(plan.algebra))
    )
    report = AccuracyReport(
        attainment=attainment,
        stop=stop,
        unmet=unmet_fields(contract, attainment, (work,)),
        derivatives=derivatives,
        work=(work,),
        claims=used,
    )
    return Evaluation(
        values=solution.values,
        order=tuple(solution.values),
        work=forest.work,
        sums=solution.sums,
        products=solution.products,
        exact=solution.exact,
        iterations=solution.iterations,
        measures=NEWTON_MEASURES,
        lower=solution.lower,
        upper=solution.upper,
        newton=solution.components,
        accuracy=report,
    )


_CYCLIC_CODES = {
    CyclicRefusal.NONLINEAR: PlanRefusal.NONLINEAR_CYCLE,
    CyclicRefusal.SIDES: PlanRefusal.MIXED_SIDES,
    CyclicRefusal.DIVERGENT: PlanRefusal.DIVERGENT,
    CyclicRefusal.UNDECIDED: PlanRefusal.DIVERGENT,
}
"""The plan's refusal for each refusal of the cyclic solver."""


def _solve_cyclic(
    plan: Plan,
    axioms: Sequence[AxiomOccurrence],
    inputs: Mapping[InputId, torch.Tensor],
    like: torch.Tensor,
) -> Evaluation:
    """Solve a linear cyclic system component by component.

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
        The least solution, in component order.

    Raises
    ------
    PlanRefused
        If a component is not linear, mixes the sides of a product that
        does not commute, or has no closure.
    """
    forest = build_forest(plan.presentation, axioms)
    try:
        values, parts, closures = solve(forest, plan.algebra, inputs, like=like)
    except CyclicRefused as refused:
        raise PlanRefused(_CYCLIC_CODES[refused.code], refused.reason) from refused
    requested = _requested(plan)
    return Evaluation(
        values=values,
        order=tuple(item for part in parts for item in part.items),
        work=forest.work,
        counts=0,
        sums=closures.sums,
        products=closures.products,
        pivots=closures.pivots,
        exact=closures.exact,
        components=parts,
        measures=CYCLIC_MEASURES,
        evidence=None
        if not requested
        else {
            item: edges
            for item, edges in forest.edges.items()
            if item.family in requested
        },
    )


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
    discovered, remaining, leaves, kept = _discover(plan, discovery, axioms)
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
        evidence=None
        if not _requested(plan)
        else {item: tuple(edges) for item, edges in kept.items()},
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
        leaves.setdefault(_axiom_item(plan, occurrence, chart), []).append(occurrence)
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
                if isinstance(source.weight, ParameterId) and source.weight in inputs
                else _weight(plan, source.weight, inputs, like)
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
