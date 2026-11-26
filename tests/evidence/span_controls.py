"""Checks and deliberately wrong versions of span plans.

A presentation in span form is planned with the span executor when no
provenance is kept. The attachment grammar is checked against its
enumerated derivations (O1), in values and derivatives; a grammar of one
category with ``X -> X X`` against the Catalan numbers; a cycle of unary
rules against its geometric sum (O6); matrices that do not commute
against the products of every tree in premise order; rules of weight
exactly zero against the exact polynomial; and the support plan against
the existence of a derivation. The counters are checked on growing
inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import torch

from quivers.core.algebras import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    MARKOV,
    MatrixSemiring,
    OrderPreservingSemiring,
)
from quivers.qiec.contracts import (
    Acyclic,
    InterpretationId,
    LinearCyclic,
    NoRetention,
    OccurrenceId,
    ParameterId,
    ParentCounts,
    Phase,
    PresentationId,
    QuerySpec,
    Recursion,
    Support,
    Weight,
    measured_total,
)
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    Item,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
)
from quivers.qiec.types import IndexLiteral, IndexVariable
from quivers.stochastic.planning import (
    SPAN_ALGORITHM,
    SPAN_RECOGNITION_ALGORITHM,
    SPAN_WORK,
    TWO_PASS_ALGORITHM,
    Evaluation,
    Plan,
    PlanRefusal,
    PlanRefused,
    plan,
)
from tests.evidence import graphs, scaling
from tests.evidence import presentations as fixture
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.counters import saved_elements
from tests.evidence.mutations import Mutation, Rewrite
from tests.evidence.oracles import derivations as oracle

LIKE = torch.zeros((), dtype=torch.float64)
"""The dtype and device of scalar runs."""

type Inputs = dict[ParameterId | OccurrenceId, torch.Tensor]
"""Input weights."""


def _require(condition: bool, what: str) -> None:
    """Fail a check with an infinite deviation.

    Parameters
    ----------
    condition : bool
        What has to hold.
    what : str
        The failure.

    Raises
    ------
    Mismatch
        If the condition does not hold.
    """
    if not condition:
        raise Mismatch("value", math.inf, what)


def _plan(
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    *,
    recursion: Recursion | None = None,
    support: bool = False,
    parents: bool = False,
) -> Plan:
    """A plan of a presentation that keeps no provenance.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    algebra : OrderPreservingSemiring
        The algebra.
    recursion : Recursion or None
        The recursion; acyclic by default.
    support : bool
        Whether the query asks for the support.
    parents : bool
        Whether parent counts are kept instead, which the two-pass
        executor answers.

    Returns
    -------
    Plan
        The plan.
    """
    retention = ParentCounts() if parents else NoRetention()
    query = QuerySpec(
        presentation=presentation.id,
        interpretation=InterpretationId.derive(presentation.name, "weight"),
        observable=Support() if support else Weight(),
        retention=retention,
    )
    return plan(
        query,
        presentation,
        algebra,
        Acyclic() if recursion is None else recursion,
        graphs.contract(),
        retention,
    )


# -- the attachment grammar ------------------------------------------------


def _inputs(transform: Callable[[Fraction], float], **changes: Fraction) -> Inputs:
    """The attachment grammar's weights under a transform.

    Parameters
    ----------
    transform : Callable
        Applied to each weight.
    **changes : Fraction
        Rule weights replaced, by rule name.

    Returns
    -------
    Inputs
        The weights.
    """
    rules = {**fixture.WEIGHTS, **changes}
    inputs: Inputs = {
        ParameterId.derive(fixture.NAME, name): torch.tensor(
            transform(value), dtype=torch.float64
        )
        for name, value in rules.items()
    }
    for name, value in fixture.LEAF_WEIGHTS.items():
        inputs[fixture.occurrence_id(name)] = torch.tensor(
            transform(value), dtype=torch.float64
        )
    return inputs


def _log(value: Fraction) -> float:
    """A logarithm, ``-inf`` at zero.

    Parameters
    ----------
    value : Fraction
        The value.

    Returns
    -------
    float
        Its logarithm.
    """
    return -math.inf if value == 0 else math.log(value)


def _trees(goal: tuple[str, int, int]) -> list[oracle.Tree]:
    """The derivations of an item of the attachment grammar, by the oracle.

    Parameters
    ----------
    goal : tuple of str, int and int
        The item's category and span.

    Returns
    -------
    list of Tree
        Every derivation.
    """
    return oracle.derivations(
        fixture.oracle_axioms(), fixture.oracle_rules(), goal, fixture.oracle_sides()
    )


def _item(category: str, start: int, end: int) -> Item:
    """An item of the attachment grammar.

    Parameters
    ----------
    category : str
        The category.
    start, end : int
        The span.

    Returns
    -------
    Item
        The item.
    """
    return Item(family=fixture.SPAN, arguments=(category, start, end))


_CELLS = (("S", 0, 5), ("VP", 1, 5), ("NP", 2, 5), ("PP", 3, 5), ("NP", 0, 1))
"""Items of the attachment grammar checked one by one."""


def check_selected() -> None:
    """Require the span executor for span form without provenance, and only then.

    Raises
    ------
    Mismatch
        If another executor is selected or the alternative is not recorded.
    """
    spanned = _plan(fixture.PRESENTATION, LOG_PROB).record
    _require(spanned.algorithm == SPAN_ALGORITHM, f"selected {spanned.algorithm}")
    _require(spanned.complexity == SPAN_WORK, "the span plan's claim is another")
    _require(
        any(item.algorithm == TWO_PASS_ALGORITHM for item in spanned.alternatives),
        "the two-pass executor is not recorded as considered",
    )
    kept = _plan(fixture.PRESENTATION, LOG_PROB, parents=True).record
    _require(kept.algorithm == TWO_PASS_ALGORITHM, "parent counts are not two-pass")
    other = plan(
        graphs.query(),
        graphs.PRESENTATION,
        LOG_PROB,
        Acyclic(),
        graphs.contract(),
        NoRetention(),
    ).record
    _require(other.algorithm == TWO_PASS_ALGORITHM, "a graph is planned by spans")
    _require(
        any("not in span form" in item.reason for item in other.alternatives),
        "the reason the span executor is declined is not recorded",
    )
    recognized = _plan(fixture.PRESENTATION, BOOLEAN, support=True).record
    _require(
        recognized.algorithm == SPAN_RECOGNITION_ALGORITHM,
        f"a support is planned by {recognized.algorithm}",
    )


def check_attachment() -> None:
    """Require every checked item's log weight to be the oracle's sum over its trees.

    Raises
    ------
    Mismatch
        If a weight differs, or the run is not certified acyclic.
    """
    evaluation = _plan(fixture.PRESENTATION, LOG_PROB).run(
        fixture.OCCURRENCES, _inputs(_log), like=LIKE
    )
    weights = fixture.oracle_weights()
    for category, start, end in _CELLS:
        expected = oracle.total(
            _trees((category, start, end)), weights, Fraction(0), Fraction(1)
        )
        compare_number(
            float(evaluation.value(_item(category, start, end))),
            _log(expected),
            Accuracy(domain="log", operations=64),
            f"{category}({start}, {end})",
        )
    _require(evaluation.certified is True, "the span measure did not apply")


def check_attachment_gradients() -> None:
    """Require the derivative in each log weight to be its expected uses (O7).

    Raises
    ------
    Mismatch
        If a derivative differs.
    """
    trees = _trees(fixture.GOAL)
    weights = fixture.oracle_weights()
    gradient = _plan(fixture.PRESENTATION, LOG_PROB).gradient(
        fixture.OCCURRENCES, _inputs(_log), _item(*fixture.GOAL), like=LIKE
    )
    keys: list[tuple[ParameterId | OccurrenceId, str]] = [
        (ParameterId.derive(fixture.NAME, name), name) for name in fixture.WEIGHTS
    ]
    keys += [(fixture.occurrence_id(name), name) for name in fixture.LEAF_WEIGHTS]
    for key, name in keys:
        compare_number(
            float(gradient[key]),
            float(oracle.logarithmic_gradient(trees, weights, name)),
            Accuracy(domain="linear", operations=256),
            f"d log Z / d log {name}",
        )


def check_exact_zero() -> None:
    """Require a rule of weight exactly zero to keep its cells and its derivative.

    With ``vp-pp`` at zero the goal's weight is that of the other trees,
    and since each tree uses the rule at most once the weight is linear in
    it, so its derivative is the weight of the trees that use it. In the
    linear algebra the derivative is that number; in the log algebra the
    cell ``VP(1, 5)`` keeps the derivations through the other rules, and
    a cell all of whose derivations use the rule is reported with ``-inf``
    and a derivative of zero, not a number of no meaning.

    Raises
    ------
    Mismatch
        If a value or a derivative differs, a zero cell is dropped, or a
        value is not in float64.
    """
    trees = _trees(fixture.GOAL)
    zero = {**fixture.oracle_weights(), "vp-pp": Fraction(0)}
    one = {**fixture.oracle_weights(), "vp-pp": Fraction(1)}
    at_zero = oracle.total(trees, zero, Fraction(0), Fraction(1))
    slope = oracle.total(trees, one, Fraction(0), Fraction(1)) - at_zero
    current = _plan(fixture.PRESENTATION, MARKOV)
    inputs = _inputs(float, **{"vp-pp": Fraction(0)})
    goal = _item(*fixture.GOAL)
    evaluation = current.run(fixture.OCCURRENCES, inputs, like=LIKE)
    compare_number(
        float(evaluation.value(goal)),
        float(at_zero),
        Accuracy(domain="linear", operations=64),
        "goal at a zero rule",
    )
    gradient = current.gradient(fixture.OCCURRENCES, inputs, goal, like=LIKE)
    compare_number(
        float(gradient[ParameterId.derive(fixture.NAME, "vp-pp")]),
        float(slope),
        Accuracy(domain="linear", operations=64),
        "derivative in a rule of weight zero",
    )
    logged = _plan(fixture.PRESENTATION, LOG_PROB).run(
        fixture.OCCURRENCES, _inputs(_log, **{"pp": Fraction(0)}), like=LIKE
    )
    _require(_item("PP", 3, 5) in logged.values, "an impossible cell is dropped")
    silent = _inputs(_log)
    silent[fixture.occurrence_id("w2")] = torch.tensor(-math.inf, dtype=torch.float64)
    muted = _plan(fixture.PRESENTATION, LOG_PROB).run(
        fixture.OCCURRENCES, silent, like=LIKE
    )
    _require(_item("NP", 2, 3) in muted.values, "an input of weight zero is dropped")
    value = logged.value(_item("PP", 3, 5))
    _require(value.dtype == torch.float64, f"an impossible cell in {value.dtype}")
    _require(float(value) == -math.inf, f"an impossible cell weighs {float(value)}")
    gradient = _plan(fixture.PRESENTATION, LOG_PROB).gradient(
        fixture.OCCURRENCES,
        _inputs(_log, **{"pp": Fraction(0)}),
        _item("VP", 1, 3),
        like=LIKE,
    )
    for key, found in gradient.items():
        _require(not bool(torch.isnan(found).any()), f"the derivative in {key} is NaN")


def check_recognition() -> None:
    """Require the support plan to find a derivation exactly when one exists.

    Raises
    ------
    Mismatch
        If a support differs from the oracle's.
    """
    current = _plan(fixture.PRESENTATION, BOOLEAN, support=True)
    inputs: Inputs = {}
    full = current.run(
        fixture.OCCURRENCES, inputs, like=torch.zeros((), dtype=torch.bool)
    )
    _require(bool(full.value(_item(*fixture.GOAL))), "the sentence is not recognized")
    missing = [
        o for o in fixture.OCCURRENCES if o.occurrence != fixture.occurrence_id("w3")
    ]
    axioms = [row for row in fixture.oracle_axioms() if row[0] != "w3"]
    reached = oracle.derivations(
        axioms, fixture.oracle_rules(), fixture.GOAL, fixture.oracle_sides()
    )
    partial = current.run(missing, inputs, like=torch.zeros((), dtype=torch.bool))
    found = partial.values.get(_item(*fixture.GOAL))
    _require(
        (found is not None and bool(found)) == bool(reached),
        f"without the preposition the support is {found}, the oracle {bool(reached)}",
    )


# -- one category ------------------------------------------------------------

_NAME = "catalan"
_SORT = UserIndexSort(name="One", constructors=("X",))
_FAMILY = FamilyId.derive(_NAME, "span")


def _pattern(start: str, end: str) -> Judgment:
    """``X(start, end)``.

    Parameters
    ----------
    start, end : str
        The position variables.

    Returns
    -------
    Judgment
        The pattern.
    """
    return Judgment(
        family=_FAMILY,
        arguments=(
            IndexLiteral(value="X", sort=_SORT),
            IndexVariable(name=start, sort=NAT),
            IndexVariable(name=end, sort=NAT),
        ),
    )


_PAIR = ParameterId.derive(_NAME, "pair")
_WORD = AxiomSlot(
    id=ConstructorId.derive(_NAME, "word"),
    name="word",
    telescope=(IndexBinder(name="i", sort=NAT), IndexBinder(name="j", sort=NAT)),
    judgment=_pattern("i", "j"),
)
CATALAN = RulePresentation(
    id=PresentationId.derive(_NAME),
    name=_NAME,
    index_sorts=(_SORT,),
    judgments=(
        JudgmentDecl(
            id=_FAMILY,
            name="span",
            indices=(
                IndexBinder(name="c", sort=_SORT, refinable=True),
                IndexBinder(name="i", sort=NAT, refinable=True),
                IndexBinder(name="j", sort=NAT, refinable=True),
            ),
        ),
    ),
    rules=(
        Rule(
            id=ConstructorId.derive(_NAME, "pair"),
            name="pair",
            telescope=tuple(IndexBinder(name=v, sort=NAT) for v in "ikj"),
            premises=(
                Premise(judgment=_pattern("i", "k")),
                Premise(judgment=_pattern("k", "j")),
            ),
            conclusion=_pattern("i", "j"),
            weight=_PAIR,
        ),
    ),
    axioms=(_WORD,),
)
"""``X(i, j) <- X(i, k), X(k, j)``: every binary bracketing of the words."""


def word(position: int) -> OccurrenceId:
    """The identity of the word at a position.

    Parameters
    ----------
    position : int
        The position.

    Returns
    -------
    OccurrenceId
        Its identity.
    """
    return OccurrenceId.derive(_NAME, f"w{position}")


def words(length: int) -> list[AxiomOccurrence]:
    """One word at each position.

    Parameters
    ----------
    length : int
        The number of words.

    Returns
    -------
    list of AxiomOccurrence
        The occurrences.
    """
    return [
        AxiomOccurrence(
            slot=_WORD.id, occurrence=word(p), binding=(("i", p), ("j", p + 1))
        )
        for p in range(length)
    ]


def goal(length: int) -> Item:
    """``X(0, length)``.

    Parameters
    ----------
    length : int
        The number of words.

    Returns
    -------
    Item
        The item.
    """
    return Item(family=_FAMILY, arguments=("X", 0, length))


def check_catalan() -> None:
    """Require the count and the weight of the bracketings of ``n`` words.

    There are ``C(n - 1) = binom(2n - 2, n - 1) / n`` bracketings, each
    with ``n - 1`` pairs, so in the linear algebra the weight with pair
    weight ``1/2`` and word weight ``1/3`` is ``C(n - 1) 2^(1 - n) 3^-n``.

    Raises
    ------
    Mismatch
        If a count or a weight differs.
    """
    for length in range(1, 8):
        count = math.comb(2 * length - 2, length - 1) // length
        counted = _plan(CATALAN, COUNTING).run(words(length), {}, like=LIKE)
        compare_number(
            float(counted.value(goal(length))),
            float(count),
            Accuracy(domain="linear", operations=4 * length**3),
            f"bracketings of {length} words",
        )
        inputs: Inputs = {_PAIR: torch.tensor(0.5, dtype=torch.float64)}
        inputs.update(
            {word(p): torch.tensor(1 / 3, dtype=torch.float64) for p in range(length)}
        )
        weighed = _plan(CATALAN, MARKOV).run(words(length), inputs, like=LIKE)
        compare_number(
            float(weighed.value(goal(length))),
            float(Fraction(count) * Fraction(1, 2) ** (length - 1) / 3**length),
            Accuracy(domain="linear", operations=4 * length**3),
            f"weight of {length} words",
        )


def check_matrix_order() -> None:
    """Require the premises' order and the rule weight last over matrices.

    The value of ``X(i, j)`` is the sum over splits of ``v(i, k) v(k, j) R``
    with the matrices of the words at the leaves, computed here by that
    recursion with matrix products.

    Raises
    ------
    Mismatch
        If an entry differs.
    """
    length = 4
    generator = torch.Generator().manual_seed(7)
    leaves = [
        torch.rand(2, 2, dtype=torch.float64, generator=generator)
        for _ in range(length)
    ]
    rule = torch.tensor([[0.3, 0.6], [0.1, 0.2]], dtype=torch.float64)
    table: dict[tuple[int, int], torch.Tensor] = {
        (p, p + 1): leaves[p] for p in range(length)
    }
    for width in range(2, length + 1):
        for start in range(length - width + 1):
            end = start + width
            table[(start, end)] = sum(
                (
                    table[(start, split)] @ table[(split, end)] @ rule
                    for split in range(start + 1, end)
                ),
                torch.zeros(2, 2, dtype=torch.float64),
            )
    inputs: Inputs = {_PAIR: rule}
    inputs.update({word(p): leaves[p] for p in range(length)})
    evaluation = _plan(CATALAN, MatrixSemiring(MARKOV, 2)).run(
        words(length), inputs, like=torch.zeros((2, 2), dtype=torch.float64)
    )
    deviation = float((evaluation.value(goal(length)) - table[(0, length)]).abs().max())
    if deviation > 1e-12:
        raise Mismatch("value", deviation, "matrix products out of premise order")


# -- unary rules -------------------------------------------------------------

_UNAME = "chains"
_CATEGORY = UserIndexSort(name="SA", constructors=("S", "A"))
_UFAMILY = FamilyId.derive(_UNAME, "span")


def _labelled(label: str) -> Judgment:
    """``label(i, j)``.

    Parameters
    ----------
    label : str
        The category.

    Returns
    -------
    Judgment
        The pattern.
    """
    return Judgment(
        family=_UFAMILY,
        arguments=(
            IndexLiteral(value=label, sort=_CATEGORY),
            IndexVariable(name="i", sort=NAT),
            IndexVariable(name="j", sort=NAT),
        ),
    )


def _unary(name: str, source: str, target: str) -> Rule:
    """A unary rule ``target(i, j) <- source(i, j)``.

    Parameters
    ----------
    name : str
        The rule.
    source, target : str
        Its categories.

    Returns
    -------
    Rule
        The rule.
    """
    return Rule(
        id=ConstructorId.derive(_UNAME, name),
        name=name,
        telescope=tuple(IndexBinder(name=v, sort=NAT) for v in "ij"),
        premises=(Premise(judgment=_labelled(source)),),
        conclusion=_labelled(target),
        weight=ParameterId.derive(_UNAME, name),
    )


_UWORD = AxiomSlot(
    id=ConstructorId.derive(_UNAME, "word"),
    name="word",
    telescope=(IndexBinder(name="i", sort=NAT), IndexBinder(name="j", sort=NAT)),
    judgment=_labelled("A"),
)
CHAINS = RulePresentation(
    id=PresentationId.derive(_UNAME),
    name=_UNAME,
    index_sorts=(_CATEGORY,),
    judgments=(
        JudgmentDecl(
            id=_UFAMILY,
            name="span",
            indices=(
                IndexBinder(name="c", sort=_CATEGORY, refinable=True),
                IndexBinder(name="i", sort=NAT, refinable=True),
                IndexBinder(name="j", sort=NAT, refinable=True),
            ),
        ),
    ),
    rules=(_unary("up", "A", "S"), _unary("down", "S", "A")),
    axioms=(_UWORD,),
)
"""``S <- A`` and ``A <- S`` over one span: a cycle of unary rules."""

_UP, _DOWN, _W = Fraction(1, 2), Fraction(3, 5), Fraction(2, 7)
"""The rule weights and the word's, with ``up down < 1``."""


def check_unary_cycle() -> None:
    """Require the geometric sum of a unary cycle (O6), and its refusal as acyclic.

    From the word ``A = w``: ``S = w u / (1 - u d)`` and ``A = w / (1 - u d)``.

    Raises
    ------
    Mismatch
        If a value differs or the acyclic contract is admitted.
    """
    occurrence = AxiomOccurrence(
        slot=_UWORD.id,
        occurrence=OccurrenceId.derive(_UNAME, "w"),
        binding=(("i", 0), ("j", 1)),
    )
    inputs: Inputs = {
        ParameterId.derive(_UNAME, "up"): torch.tensor(float(_UP), dtype=torch.float64),
        ParameterId.derive(_UNAME, "down"): torch.tensor(
            float(_DOWN), dtype=torch.float64
        ),
        occurrence.occurrence: torch.tensor(float(_W), dtype=torch.float64),
    }
    evaluation = _plan(CHAINS, MARKOV, recursion=LinearCyclic()).run(
        [occurrence], inputs, like=LIKE
    )
    loop = 1 - _UP * _DOWN
    for label, expected in (("S", _W * _UP / loop), ("A", _W / loop)):
        compare_number(
            float(evaluation.value(Item(family=_UFAMILY, arguments=(label, 0, 1)))),
            float(expected),
            Accuracy(domain="linear", operations=64),
            f"{label} over a unary cycle",
        )
    _refused(lambda: _plan(CHAINS, MARKOV), PlanRefusal.CYCLE, "a cycle as acyclic")


def _refused(
    attempt: Callable[[], Evaluation | Plan], code: PlanRefusal, what: str
) -> None:
    """Require an attempt to be refused for a reason.

    Parameters
    ----------
    attempt : Callable
        The attempt.
    code : PlanRefusal
        The reason.
    what : str
        A description.

    Raises
    ------
    Mismatch
        If it is admitted or refused for another reason.
    """
    try:
        attempt()
    except PlanRefused as refused:
        _require(refused.code is code, f"{what}: refused as {refused.code.value}")
        return
    raise Mismatch("refusal", math.inf, f"{what} was admitted")


def check_refusals() -> None:
    """Require an input over no position to be refused.

    Raises
    ------
    Mismatch
        If it is admitted.
    """
    empty = AxiomOccurrence(
        slot=_WORD.id, occurrence=word(0), binding=(("i", 1), ("j", 1))
    )
    _refused(
        lambda: _plan(CATALAN, MARKOV).run([empty], {}, like=LIKE),
        PlanRefusal.INADMISSIBLE,
        "an input over no position",
    )


# -- work ----------------------------------------------------------------------


def _claim(length: int) -> float:
    """The span claim's time for one category and one rule.

    Parameters
    ----------
    length : int
        The words.

    Returns
    -------
    float
        ``P n^3 + R n^2 + N^2 n^2 + N^3 + C`` at ``P = R = N = 1, C = 0``.
    """
    return float(length**3 + 2 * length**2 + 1)


def span_time(length: int) -> float:
    """The measured time of the bracketings of ``length`` words.

    Parameters
    ----------
    length : int
        The words.

    Returns
    -------
    float
        The time counters, as the claim's execution bound reads them.
    """
    current = _plan(CATALAN, COUNTING)
    report = current.run(words(length), {}, like=LIKE).report()
    bound = next(
        bound
        for bound in current.record.complexity.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "time"
    )
    return float(measured_total(bound, report))


def span_saved(length: int) -> float:
    """Elements kept for the backward pass of the bracketings of ``length`` words.

    Parameters
    ----------
    length : int
        The words.

    Returns
    -------
    float
        Elements saved by autograd.
    """
    inputs: Inputs = {_PAIR: torch.tensor(0.5, dtype=torch.float64, requires_grad=True)}
    with saved_elements() as sizes:
        evaluation = _plan(CATALAN, MARKOV).run(words(length), inputs, like=LIKE)
    torch.autograd.grad(evaluation.value(goal(length)), [inputs[_PAIR]])
    return float(sum(sizes))


def check_work() -> None:
    """Require the span counters within the claim on growing inputs.

    Raises
    ------
    Mismatch
        If the work or the tape breaks its bound, or a control does not.
    """
    for name, part, work, constant in (
        ("span time over length", "forward", span_time, 1.0),
        ("span tape over length", "saved", span_saved, 2.0),
    ):
        scaling.check_family(
            scaling.Family(
                name=name,
                executor="span",
                question="Q3",
                varies="length",
                part=part,
                sizes=(4, 8, 16, 32, 64),
                work=work,
                claim=_claim,
                constant=constant,
            )
        )


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="span", question="Q3", run=run)
    for name, run in (
        ("span/selected", check_selected),
        ("span/attachment", check_attachment),
        ("span/attachment-gradients", check_attachment_gradients),
        ("span/exact-zero", check_exact_zero),
        ("span/recognition", check_recognition),
        ("span/catalan", check_catalan),
        ("span/matrix-order", check_matrix_order),
        ("span/unary-cycle", check_unary_cycle),
        ("span/refusals", check_refusals),
        ("span/work", check_work),
    )
}
"""Every check of span plans by name."""

SPAN = "quivers.stochastic.span"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="span-split-sum-replaced-by-max",
        defect_class="aggregation",
        provenance="the sum over split points keeps only the largest product",
        target=f"{SPAN}:span_inside",
        also_in=("quivers.stochastic.planning",),
        rewrites=(
            Rewrite(
                old="summed = count.plus(algebra, products, 1)",
                new="summed = products.amax(dim=1)",
            ),
        ),
        rejected_by=("span/catalan",),
        min_effect=0.5,
    ),
    Mutation(
        name="span-premises-exchanged",
        defect_class="order",
        provenance="the right premise is multiplied first",
        target=f"{SPAN}:span_inside",
        also_in=("quivers.stochastic.planning",),
        rewrites=(
            Rewrite(
                old=(
                    "                left.index_select(2, lefts.to(left.device)),\n"
                    "                right.index_select(2, rights.to(right.device)),"
                ),
                new=(
                    "                right.index_select(2, rights.to(right.device)),\n"
                    "                left.index_select(2, lefts.to(left.device)),"
                ),
            ),
        ),
        rejected_by=("span/matrix-order",),
        min_effect=0.078,
    ),
    Mutation(
        name="fiber-padded-with-the-number-zero",
        defect_class="empty reduction",
        provenance="the padding of a fiber is the number zero, not the algebra's",
        target="quivers.core.elimination:fiber_join",
        also_in=(
            "quivers.stochastic.span",
            "quivers.stochastic.planning",
            "quivers.enriched.day_convolution",
        ),
        rewrites=(
            Rewrite(
                old="padded = torch.cat([values, zero.to(values.dtype)], dim=axis)",
                new=(
                    "padded = torch.cat([values, torch.zeros_like(zero).to("
                    "values.dtype)], dim=axis)"
                ),
            ),
        ),
        rejected_by=("span/attachment",),
        min_effect=3.7,
    ),
    Mutation(
        name="span-support-from-numbers",
        defect_class="support",
        provenance="the cells reported are those of nonzero weight",
        target="quivers.stochastic.planning:_run_span",
        rewrites=(
            Rewrite(
                old="_support_seed(length - width + 1, size, entries, None, like)",
                new="_support_seed(length - width + 1, size, entries, algebra, like)",
            ),
        ),
        rejected_by=("span/exact-zero",),
        min_effect=1.0,
    ),
)
"""Every wrong version of span plans and the checks that reject it."""
