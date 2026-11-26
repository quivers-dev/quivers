"""Checks and deliberately wrong versions of two-pass plans.

The checks compare plans with the derivation oracle (O1) and its
gradients (O7) on the attachment grammar, with path sums and matrix
products computed by hand on input graphs, with refusals of what a plan
does not answer, and with linear work on chain, diamond and skip graphs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import ClassVar
from fractions import Fraction

import torch

from quivers.core.algebras import (
    COUNTING,
    LOG_PROB,
    MAX_PLUS,
    REAL,
    MarkovAlgebra,
    MatrixSemiring,
    RealAlgebra,
    OrderPreservingSemiring,
)
from quivers.qiec.contracts import (
    Acyclic,
    Best,
    Bounded,
    InterpretationId,
    JudgmentPattern,
    LinearCyclic,
    NonlinearFold,
    NoRetention,
    Outcome,
    OccurrenceId,
    ParameterId,
    Phase,
    QuerySpec,
    RequestedEvidence,
    UnmeasuredWork,
    WorkField,
    measured_total,
)
from quivers.qiec.identifiers import ComputationId, FamilyId
from quivers.qiec.rules import Ground, Item, RulePresentation
from quivers.stochastic.forest import DerivationForest, ForestEdge, build_forest
from quivers.stochastic.indexed import IndexedChart
from quivers.stochastic.planning import Plan, PlanRefusal, PlanRefused, plan
from tests.evidence import graphs, scaling
from tests.evidence import presentations as fixture
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.counters import saved_elements
from tests.evidence.counters import measure
from tests.evidence.mutations import Mutation, Rewrite
from tests.evidence.oracles import derivations as oracle

LIKE = torch.zeros((), dtype=torch.float64)
"""The dtype and device of scalar runs."""

type Inputs = dict[ParameterId | OccurrenceId, torch.Tensor]
"""Input weights of a run."""


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


def _attachment_inputs() -> Inputs:
    """Log weights of the attachment grammar's rules and words.

    Returns
    -------
    Inputs
        The input weights.
    """
    inputs: Inputs = {
        ParameterId.derive(fixture.NAME, name): torch.tensor(
            math.log(value), dtype=torch.float64
        )
        for name, value in fixture.WEIGHTS.items()
    }
    for name, value in fixture.LEAF_WEIGHTS.items():
        inputs[fixture.occurrence_id(name)] = torch.tensor(
            math.log(value), dtype=torch.float64
        )
    return inputs


def _attachment_plan(
    algebra: OrderPreservingSemiring = LOG_PROB,
    *,
    best: bool = False,
    recompute: bool = False,
) -> Plan:
    """A plan of the attachment grammar.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    best : bool
        Whether the query is for the best derivation.
    recompute : bool
        Whether to select the recomputing backward pass.

    Returns
    -------
    Plan
        The plan.
    """
    query = QuerySpec(
        presentation=fixture.PRESENTATION.id,
        interpretation=InterpretationId.derive(fixture.NAME, "weight"),
        observable=Best() if best else graphs.query().observable,
    )
    return plan(
        query,
        fixture.PRESENTATION,
        algebra,
        Acyclic(),
        graphs.contract(recompute=recompute),
        NoRetention(),
    )


def _trees() -> list[oracle.Tree]:
    """The goal's derivations by the oracle.

    Returns
    -------
    list of Tree
        Every derivation.
    """
    return oracle.derivations(
        fixture.oracle_axioms(),
        fixture.oracle_rules(),
        fixture.GOAL,
        fixture.oracle_sides(),
    )


def _goal() -> Item:
    """The attachment grammar's goal item.

    Returns
    -------
    Item
        ``span(S, 0, 5)``.
    """
    return Item(family=fixture.SPAN, arguments=fixture.GOAL)


def check_weight() -> None:
    """Require the goal's log weight to be the oracle's, and the measure to apply.

    Raises
    ------
    Mismatch
        If the weight differs or the span measure does not apply.
    """
    evaluation = _attachment_plan().run(
        fixture.OCCURRENCES, _attachment_inputs(), like=LIKE
    )
    expected = oracle.total(
        _trees(), fixture.oracle_weights(), Fraction(0), Fraction(1)
    )
    compare_number(
        float(evaluation.value(_goal())),
        math.log(expected),
        Accuracy(domain="log", operations=64),
        "goal weight",
    )
    _require(evaluation.certified is True, "the span measure did not apply")


def check_best() -> None:
    """Require the best weight and its witness to be the oracle's best tree.

    Raises
    ------
    Mismatch
        If the best tree is not unique, or the weight or witness differs.
    """
    current = _attachment_plan(MAX_PLUS, best=True)
    inputs = _attachment_inputs()
    evaluation = current.run(fixture.OCCURRENCES, inputs, like=LIKE)
    weights = fixture.oracle_weights()
    ranked = sorted(
        _trees(),
        key=lambda tree: oracle.total([tree], weights, Fraction(0), Fraction(1)),
        reverse=True,
    )
    best = ranked[0]
    top = oracle.total([best], weights, Fraction(0), Fraction(1))
    _require(
        len(ranked) < 2
        or oracle.total([ranked[1]], weights, Fraction(0), Fraction(1)) < top,
        "the fixture's best tree is tied",
    )
    compare_number(
        float(evaluation.value(_goal())),
        math.log(top),
        Accuracy(domain="log", operations=64),
        "best weight",
    )
    witness = current.best_derivation(
        fixture.OCCURRENCES, evaluation, _goal(), inputs, like=LIKE
    )
    _require(fixture.shape(witness) == best, "the witness is not the best tree")


def check_gradients() -> None:
    """Require both backward passes to give the oracle's expected counts (O7).

    The derivative of the log weight in a log input weight is the
    expected number of its uses under the normalized derivation weights.

    Raises
    ------
    Mismatch
        If a derivative differs from the oracle's.
    """
    inputs = _attachment_inputs()
    trees = _trees()
    weights = fixture.oracle_weights()
    keys: list[tuple[ParameterId | OccurrenceId, str]] = [
        (ParameterId.derive(fixture.NAME, name), name) for name in fixture.WEIGHTS
    ]
    keys += [(fixture.occurrence_id(name), name) for name in fixture.LEAF_WEIGHTS]
    for recompute in (False, True):
        gradient = _attachment_plan(recompute=recompute).gradient(
            fixture.OCCURRENCES, inputs, _goal(), like=LIKE
        )
        for key, name in keys:
            compare_number(
                float(gradient[key]),
                oracle.logarithmic_gradient(trees, weights, name),
                Accuracy(domain="linear", operations=256),
                f"d log Z / d log {name}, recompute={recompute}",
            )


def _graph_plan(
    algebra: OrderPreservingSemiring = REAL,
    *,
    recompute: bool = False,
    presentation: RulePresentation = graphs.PRESENTATION,
) -> Plan:
    """A plan of a reachability presentation.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    recompute : bool
        Whether to select the recomputing backward pass.
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    Plan
        The plan.
    """
    return plan(
        graphs.query(),
        presentation,
        algebra,
        Acyclic(),
        graphs.contract(recompute=recompute),
        NoRetention(),
    )


def check_repeated_premise() -> None:
    """Require one instance when one item fills three premises of a rule (M3).

    ``triple(0) <- reach(0), reach(0), reach(0)`` weighs ``w^3`` when
    ``reach(0)`` weighs ``w``, and its derivative is ``3 w^2``.

    Raises
    ------
    Mismatch
        If the weight or the derivative differs.
    """
    start = OccurrenceId.derive(graphs.NAME, "start")
    inputs: Inputs = {start: torch.tensor(3.0, dtype=torch.float64)}
    goal = Item(family=graphs.TRIPLE, arguments=(0,))
    found = float(
        _graph_plan().run(graphs.occurrences([]), inputs, like=LIKE).value(goal)
    )
    compare_number(found, 27.0, Accuracy(domain="linear", operations=4), "triple(0)")
    for recompute in (False, True):
        gradient = _graph_plan(recompute=recompute).gradient(
            graphs.occurrences([]), inputs, goal, like=LIKE
        )
        compare_number(
            float(gradient[start]),
            27.0,
            Accuracy(domain="linear", operations=8),
            f"d triple(0) / d w, recompute={recompute}",
        )


def check_repeated_premise_keyed() -> None:
    """Require one instance from a keyed chart when one item fills three premises.

    Raises
    ------
    Mismatch
        If ``triple(0) <- reach(0), reach(0), reach(0)`` is produced other
        than once.
    """
    for join in ("generic", "keyed"):
        chart = IndexedChart(graphs.PRESENTATION, join=join)
        item = graphs.node(0)
        chart.add(item)
        found = [
            premises
            for position in range(3)
            for _, premises in chart.instances(
                graphs.TRIPLED, position, item, lambda other: other == item
            )
        ]
        if len(found) != 1:
            raise Mismatch(
                "value", float(abs(len(found) - 1)), f"{join}: {len(found)} instances"
            )


def _evidence_plan(family: FamilyId) -> Plan:
    """A reachability plan that keeps the evidence of one form.

    Parameters
    ----------
    family : FamilyId
        The requested form.

    Returns
    -------
    Plan
        The plan.
    """
    evidence = RequestedEvidence(patterns=(JudgmentPattern(family=family),))
    return plan(
        graphs.query().with_(retention=evidence),
        graphs.PRESENTATION,
        REAL,
        Acyclic(),
        graphs.contract(),
        evidence,
    )


def _root(edge: ForestEdge) -> tuple[str, tuple[tuple[Ground, ...], ...]]:
    """An edge as the oracle names a tree's root: its name and its children's items.

    Parameters
    ----------
    edge : ForestEdge
        The edge.

    Returns
    -------
    tuple
        The rule or occurrence name and the premise items.
    """
    if edge.occurrence is not None:
        if edge.occurrence == OccurrenceId.derive(graphs.NAME, "start"):
            return "start", ()
        index = next(i for i in range(64) if graphs.edge_id(i) == edge.occurrence)
        return f"edge{index}", ()
    names = {graphs.REACH: "reach", graphs.EDGE: "edge"}
    return "step", tuple((names[p.family], *p.arguments) for p in edge.premises)


def check_requested_evidence() -> None:
    """Require requested evidence to be the forest restricted to the requested items (O1).

    Over parallel edges, every reach item keeps exactly the edges the
    packed forest has for it, in the same order, and these are the
    distinct roots of the oracle's derivations of it: one rule instance or
    input occurrence per alternative, whatever lies below. No edge item is
    kept.

    Raises
    ------
    Mismatch
        If an item, an edge or an alternative differs.
    """
    edges = [(0, 1), (0, 1), (1, 2), (1, 2), (0, 2)]
    inputs: Inputs = {**graphs.edge_weights(edges, [1.0] * len(edges))}
    evaluation = _evidence_plan(graphs.REACH).run(
        graphs.occurrences(edges), inputs, like=LIKE
    )
    evidence = evaluation.evidence
    _require(evidence is not None, "no evidence was kept")
    assert evidence is not None
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    requested = {item for item in forest.items if item.family == graphs.REACH}
    _require(set(evidence) == requested, "the kept items are not the requested ones")
    for item in requested:
        _require(
            evidence[item] == forest.edges[item],
            f"{item.arguments}: {len(evidence[item])} edges for {len(forest.edges[item])}",
        )
        (node,) = item.arguments
        assert isinstance(node, int)
        expected = sorted(
            {
                (name, tuple(child[1] for child in children))
                for name, _, _, children in oracle_graph(edges, node)
            }
        )
        found = sorted(_root(edge) for edge in evidence[item])
        if found != expected:
            raise Mismatch(
                "value",
                float(abs(len(found) - len(expected)) or 1),
                f"reach({node}): {len(found)} alternatives for the oracle's {len(expected)}",
            )


def oracle_graph(edges: Sequence[tuple[int, int]], goal: int) -> list[oracle.Tree]:
    """The oracle's derivations of ``reach(goal)`` over a graph.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The edges.
    goal : int
        The node.

    Returns
    -------
    list of Tree
        Every derivation.
    """
    i, j = oracle.Variable("i"), oracle.Variable("j")
    rules: tuple[oracle.RuleRow, ...] = (
        ("step", (("reach", i), ("edge", i, j)), ("reach", j), "step"),
    )
    axioms: list[oracle.AxiomRow] = [("start", ("reach", 0))]
    axioms += [(f"edge{index}", ("edge", a, b)) for index, (a, b) in enumerate(edges)]
    return oracle.derivations(axioms, rules, ("reach", goal))


def check_requested_memory() -> None:
    """Require the evidence kept to be bounded by what is requested.

    Requesting ``reach`` on ``0 -> 1 -> 2`` beside a growing chain that
    ``0`` does not reach keeps the same seven entries at every size,
    while the packed forest grows with the edges.

    Raises
    ------
    Mismatch
        If the kept evidence grows, or is not counted in the report.
    """
    current = _evidence_plan(graphs.REACH)
    for size in (8, 32, 128):
        edges = graphs.chain(2) + [(1000 + i, 1001 + i) for i in range(size)]
        evaluation = current.run(graphs.occurrences(edges), {}, like=LIKE)
        _require(evaluation.retained == 7, f"{evaluation.retained} kept at {size}")
        _require(
            evaluation.report().provenance_records == evaluation.counts + 7,
            "the report does not count the evidence",
        )
        forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
        _require(forest.size > size, "the forest does not grow")


class _CountingReals(RealAlgebra):
    """The reals, counting the products and sums a plan asks of them."""

    proof = RealAlgebra.proof
    tested = RealAlgebra.tested
    counted: ClassVar[dict[str, int]] = {"products": 0, "sums": 0}

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Count and forward a product.

        Parameters
        ----------
        a, b : torch.Tensor
            The factors.

        Returns
        -------
        torch.Tensor
            Their product.
        """
        self.counted["products"] += 1
        return super().tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Count and forward a sum of two terms.

        Parameters
        ----------
        t : torch.Tensor
            The stacked terms.
        dim : int or tuple of int
            The axis.

        Returns
        -------
        torch.Tensor
            Their sum.
        """
        self.counted["sums"] += 1
        return super().join(t, dim)


class _CountingMarkov(MarkovAlgebra):
    """Probabilities with a closure, counting every product and sum, the closure's too."""

    proof = MarkovAlgebra.proof
    tested = MarkovAlgebra.tested
    counted: ClassVar[dict[str, int]] = {"products": 0, "sums": 0}

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Count and forward products, one per element.

        Parameters
        ----------
        a, b : torch.Tensor
            The factors.

        Returns
        -------
        torch.Tensor
            Their products.
        """
        result = super().tensor_op(a, b)
        self.counted["products"] += result.numel()
        return result

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Count and forward sums of two terms, one per element.

        Parameters
        ----------
        t : torch.Tensor
            The stacked terms.
        dim : int or tuple of int
            The axis.

        Returns
        -------
        torch.Tensor
            Their sums.
        """
        result = super().join(t, dim)
        self.counted["sums"] += result.numel()
        return result


def _check_cyclic_counts() -> None:
    """Require a cyclic plan's measured counters to be independent counts.

    The counting algebra sees every product and sum, the matrix closure's
    included; the five states of the ring are five pivots, and no loop
    nears the boundary, so no exact decision runs.

    Raises
    ------
    Mismatch
        If a measured counter differs from its independent count, or the
        closure's work is below ``n`` rounds of ``n^2`` for a component of
        ``n``.
    """
    _CountingMarkov.counted.update(products=0, sums=0)
    edges = [(i, (i + 1) % 5) for i in range(5)]
    inputs: Inputs = {**graphs.edge_weights(edges, [0.5] * len(edges))}
    current = plan(
        graphs.query().with_(presentation=graphs.LEFT.id),
        graphs.LEFT,
        _CountingMarkov(),
        LinearCyclic(),
        graphs.contract(),
        NoRetention(),
    )
    report = current.run(graphs.occurrences(edges), inputs, like=LIKE).report()
    counted = _CountingMarkov.counted
    for field, count in (
        (WorkField.SCALAR_PRODUCTS, counted["products"]),
        (WorkField.SCALAR_SUMS, counted["sums"]),
        (WorkField.STAR_PIVOTS, 5),
        (WorkField.EXACT_OPERATIONS, 0),
        (WorkField.SCALAR_DIVISIONS, 0),
    ):
        _require(
            field in report.measured, f"a cyclic run does not measure {field.value}"
        )
        found = getattr(report, field.value)
        if found != count:
            raise Mismatch(
                "value",
                float(abs(found - count)) or 1.0,
                f"cyclic {field.value}: reported {found}, counted {count}",
            )
    _require(
        report.scalar_products >= 5**3 and report.scalar_sums >= 5**3,
        "the closure did fewer than n rounds of n^2 products and sums",
    )
    _require(
        report.scalar_divisions == 0,
        "a cyclic run divided outside its closures",
    )


def check_measured_counts() -> None:
    """Require every measured counter of a plan's report to be an independent count.

    On diamonds, the algebra counts its own products and sums, the items
    are the parent counts, and saved tensors are counted by a hook on
    the autograd graph. A counter the report does not measure is not
    compared, and a space bound, which no counter here measures, is
    refused by name.

    Raises
    ------
    Mismatch
        If a measured counter differs from its independent count, or the
        space bound is checked against an unmeasured counter.
    """
    algebra = _CountingReals()
    _CountingReals.counted.update(products=0, sums=0)
    edges = graphs.diamonds(3)
    inputs: Inputs = {
        key: value.requires_grad_()
        for key, value in graphs.edge_weights(edges, [0.5] * len(edges)).items()
    }
    current = _graph_plan(algebra)
    with saved_elements() as sizes:
        evaluation = current.run(graphs.occurrences(edges), inputs, like=LIKE)
    report = evaluation.report()
    independent = {
        WorkField.SCALAR_PRODUCTS: _CountingReals.counted["products"],
        WorkField.SCALAR_SUMS: _CountingReals.counted["sums"],
        WorkField.PROVENANCE_RECORDS: len(evaluation.order),
        WorkField.ELEMENTS_SAVED_FOR_BACKWARD: sum(sizes),
        WorkField.SCALAR_DIVISIONS: 0,
        WorkField.STAR_PIVOTS: 0,
    }
    _require(sum(sizes) > 0, "the run saved nothing for its gradient")
    for field, count in independent.items():
        if field not in report.measured:
            continue
        found = getattr(report, field.value)
        if found != count:
            raise Mismatch(
                "value",
                float(abs(found - count)) or 1.0,
                f"{field.value}: reported {found}, counted {count}",
            )
    space = next(
        bound
        for bound in current.record.complexity.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "space"
    )
    try:
        measured_total(space, report)
    except UnmeasuredWork as unmeasured:
        _require(
            unmeasured.field is WorkField.PEAK_LIVE_ELEMENTS,
            f"refused for {unmeasured.field.value}",
        )
    else:
        raise Mismatch(
            "refusal", math.inf, "a space bound was checked against no count"
        )
    time = next(
        bound
        for bound in current.record.complexity.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "time"
    )
    _require(
        measured_total(time, report)
        == report.scalar_work + report.hash_operations + report.key_comparisons,
        "the time total is not the sum of the time counters",
    )
    _check_cyclic_counts()


def check_constant_rule() -> None:
    """Require a rule without premises to be an instance, summed with the input.

    In the seeded presentation ``reach(0)`` has an input of weight 3 and
    the rule ``reach(0) <-`` of weight 4, so it weighs 7, and node 1 one
    edge of weight 2 further; the rule's derivative is 2.

    Raises
    ------
    Mismatch
        If a weight or the derivative differs.
    """
    edges = graphs.chain(1)
    origin = ParameterId.derive(graphs.NAME, "origin")
    inputs: Inputs = {
        **graphs.edge_weights(edges, (2.0,)),
        OccurrenceId.derive(graphs.NAME, "start"): torch.tensor(
            3.0, dtype=torch.float64
        ),
        origin: torch.tensor(4.0, dtype=torch.float64),
    }
    for recompute in (False, True):
        current = _graph_plan(recompute=recompute, presentation=graphs.SEEDED)
        evaluation = current.run(graphs.occurrences(edges), inputs, like=LIKE)
        compare_number(
            float(evaluation.value(graphs.node(1))),
            14.0,
            Accuracy(domain="linear", operations=8),
            "node 1",
        )
        gradient = current.gradient(
            graphs.occurrences(edges), inputs, graphs.node(1), like=LIKE
        )
        compare_number(
            float(gradient[origin]),
            2.0,
            Accuracy(domain="linear", operations=8),
            f"d node 1 / d origin, recompute={recompute}",
        )


def check_alternatives_are_summed() -> None:
    """Require two unequal paths to a node to be summed, not maximized (M5).

    Raises
    ------
    Mismatch
        If the node's weight is not the sum of its paths.
    """
    edges = graphs.diamonds(1)
    inputs: Inputs = {**graphs.edge_weights(edges, (2.0, 3.0, 5.0, 7.0))}
    evaluation = _graph_plan().run(graphs.occurrences(edges), inputs, like=LIKE)
    compare_number(
        float(evaluation.value(graphs.node(3))),
        2.0 * 5.0 + 3.0 * 7.0,
        Accuracy(domain="linear", operations=8),
        "two paths",
    )


def check_zero_weight_gradient() -> None:
    """Require derivatives at an edge whose weight is exactly zero (M17).

    On the chain ``0 -> 1 -> 2 -> 3`` with the middle edge at zero, node 3
    weighs zero, and its derivative in the middle edge is the product of
    the other two, ``2 * 5``.

    Raises
    ------
    Mismatch
        If that derivative differs from ten under either backward pass.
    """
    edges = graphs.chain(3)
    inputs: Inputs = {**graphs.edge_weights(edges, (2.0, 0.0, 5.0))}
    for recompute in (False, True):
        gradient = _graph_plan(recompute=recompute).gradient(
            graphs.occurrences(edges), inputs, graphs.node(3), like=LIKE
        )
        compare_number(
            float(gradient[graphs.edge_id(1)]),
            10.0,
            Accuracy(domain="linear", operations=8),
            f"derivative at a zero edge, recompute={recompute}",
        )


def check_premise_order() -> None:
    """Require products in premise order, then the rule weight, over matrices (M18).

    On the chain ``0 -> 1 -> 2 -> 3`` with edge matrices ``A, B, C`` and the
    step rule's matrix ``R``, a derivation's factors read children first,
    then the rule, so node 3 is ``A R B R C R``.

    Raises
    ------
    Mismatch
        If node 3's value is another product.
    """
    matrices = [
        torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 2.0], [1.0, 0.0]], dtype=torch.float64),
    ]
    rule = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    edges = graphs.chain(3)
    inputs: Inputs = {
        graphs.edge_id(index): matrix for index, matrix in enumerate(matrices)
    }
    inputs[ParameterId.derive(graphs.NAME, "step")] = rule
    like = torch.zeros((2, 2), dtype=torch.float64)
    evaluation = _graph_plan(MatrixSemiring(COUNTING, 2)).run(
        graphs.occurrences(edges), inputs, like=like
    )
    expected = matrices[0] @ rule @ matrices[1] @ rule @ matrices[2] @ rule
    found = evaluation.value(graphs.node(3))
    deviation = float((found - expected).abs().max())
    if deviation:
        raise Mismatch("value", deviation, f"node 3 is {found.tolist()}")


def _square_query() -> QuerySpec:
    """A query for a fold that does not commute with sums: squaring.

    Returns
    -------
    QuerySpec
        The query.
    """
    return graphs.query().with_(
        observable=NonlinearFold(fold=ComputationId.derive(graphs.NAME, "square"))
    )


def _attempts() -> dict[PlanRefusal, Callable[[], Plan | DerivationForest]]:
    """One plan the planner has to refuse per reason.

    Returns
    -------
    dict of PlanRefusal to Callable
        Attempts by the refusal they have to meet.
    """
    lambek = QuerySpec(
        presentation=fixture.LAMBEK.id,
        interpretation=InterpretationId.derive("lambek", "weight"),
        observable=graphs.query().observable,
    )
    return {
        PlanRefusal.INADMISSIBLE: lambda: plan(
            _square_query(),
            graphs.PRESENTATION,
            REAL,
            Acyclic(),
            graphs.contract(),
            NoRetention(),
        ),
        PlanRefusal.RECURSION_WITHOUT_EXECUTOR: lambda: plan(
            graphs.query(),
            graphs.PRESENTATION,
            REAL,
            Bounded(measure="height", limit=4),
            graphs.contract(),
            NoRetention(),
        ),
        PlanRefusal.OPEN_PREMISES: lambda: plan(
            lambek, fixture.LAMBEK, REAL, Acyclic(), graphs.contract(), NoRetention()
        ),
        PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR: lambda: plan(
            graphs.query().with_(
                observable=Outcome(key=ComputationId.derive(graphs.NAME, "key"))
            ),
            graphs.PRESENTATION,
            REAL,
            Acyclic(),
            graphs.contract(),
            NoRetention(),
        ),
        PlanRefusal.RETENTION_WITHOUT_EXECUTOR: lambda: _graph_plan().forest(
            graphs.occurrences([])
        ),
    }


def check_refusals() -> None:
    """Require each query a plan does not answer to be refused for its reason (M35).

    A fold that does not commute with sums cannot be applied to the
    aggregate a two-pass run computes: with alternatives ``-1`` and ``1``
    the square of the sum is zero and the sum of the squares two. So
    without the packed forest it is refused, as are bounded recursion
    without the forest, open premises, a pushforward, a forest from a plan
    that does not keep one, and a run whose inputs make the derived
    hypergraph cyclic.

    Raises
    ------
    Mismatch
        If one is admitted or refused for another reason.
    """
    for code, attempt in _attempts().items():
        try:
            attempt()
        except PlanRefused as refused:
            _require(
                refused.code is code,
                f"{code.value} refused as {refused.code.value}: {refused.reason}",
            )
            continue
        raise Mismatch("refusal", math.inf, f"{code.value} was admitted")
    try:
        _graph_plan().run(graphs.occurrences([(0, 1), (1, 0)]), {}, like=LIKE)
    except PlanRefused as refused:
        _require(
            refused.code is PlanRefusal.CYCLE, f"a cycle refused as {refused.code}"
        )
        return
    raise Mismatch("refusal", math.inf, "a cycle returned values")


def check_certificate() -> None:
    """Require the span measure for the grammar and no measure for reachability.

    Reachability's cycle of forms is decided by its input graph, so no
    measure on its arguments proves it acyclic.

    Raises
    ------
    Mismatch
        If either certificate differs.
    """
    _require(
        _attachment_plan().record.certificate is not None,
        "no measure for the attachment grammar",
    )
    _require(_graph_plan().record.certificate is None, "a measure for reachability")


def _work(edges: Sequence[tuple[int, int]]) -> float:
    """The work of one run: scalar operations, lookups, candidates and instances.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The graph.

    Returns
    -------
    float
        The work.
    """
    current = _graph_plan(LOG_PROB)
    inputs: Inputs = {**graphs.edge_weights(edges, [0.0] * len(edges))}
    counted: dict[str, int] = {}

    def body() -> torch.Tensor:
        """One run, recording the chart's counters.

        Returns
        -------
        torch.Tensor
            The weight of the start node.
        """
        evaluation = current.run(graphs.occurrences(edges), inputs, like=LIKE)
        work = evaluation.work
        counted["chart"] = work.lookups + work.candidates + work.instances
        return evaluation.value(graphs.node(0))

    return float(measure(body).scalar + counted["chart"])


def _family(name: str, build: Callable[[int], list[tuple[int, int]]]) -> scaling.Family:
    """A family of graphs whose work has to stay linear in the edges.

    Parameters
    ----------
    name : str
        The family.
    build : Callable[[int], list of tuple of int and int]
        The graph at a size.

    Returns
    -------
    scaling.Family
        The family.
    """
    return scaling.Family(
        name=f"two-pass plan on {name}",
        executor="plan",
        question="Q1",
        varies="edges",
        part="forward",
        sizes=(8, 16, 32, 64, 128),
        work=lambda size: _work(build(size)),
        claim=lambda size: float(len(build(size)) + 1),
        constant=40.0,
    )


def _diamonds(size: int) -> list[tuple[int, int]]:
    """Diamonds with ``size`` edges.

    Parameters
    ----------
    size : int
        A multiple of four.

    Returns
    -------
    list of tuple of int and int
        The edges.
    """
    return graphs.diamonds(size // 4)


def _skips(size: int) -> list[tuple[int, int]]:
    """A skip graph with about ``size`` edges.

    Parameters
    ----------
    size : int
        Twice the last node.

    Returns
    -------
    list of tuple of int and int
        The edges.
    """
    return graphs.skips(size // 2)


def check_work() -> None:
    """Require linear work on chains, diamonds and skip graphs.

    Diamonds have exponentially many paths and skip graphs a Fibonacci
    number of them; the work grows with the edges alone. Each family's
    control of the next order has to fail.

    Raises
    ------
    Mismatch
        If a family breaks its bound, or its control does not.
    """
    for name, build in (
        ("chains", graphs.chain),
        ("diamonds", _diamonds),
        ("skip graphs", _skips),
    ):
        scaling.check_family(_family(name, build))


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="plan", question="Q1", run=run)
    for name, run in (
        ("plan/weight", check_weight),
        ("plan/best", check_best),
        ("plan/gradients", check_gradients),
        ("plan/repeated-premise", check_repeated_premise),
        ("plan/repeated-premise-keyed", check_repeated_premise_keyed),
        ("plan/constant-rule", check_constant_rule),
        ("plan/requested-evidence", check_requested_evidence),
        ("plan/requested-memory", check_requested_memory),
        ("plan/measured-counts", check_measured_counts),
        ("plan/alternatives", check_alternatives_are_summed),
        ("plan/zero-gradient", check_zero_weight_gradient),
        ("plan/premise-order", check_premise_order),
        ("plan/refusals", check_refusals),
        ("plan/certificate", check_certificate),
        ("plan/work", check_work),
    )
}
"""Every check of plans by name."""

PLAN = "quivers.stochastic.planning"
INDEXED = "quivers.stochastic.indexed"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="plan-repeated-premise-at-every-position",
        defect_class="repeated premise",
        provenance="an instance is produced at every premise its last item fills",
        target=f"{INDEXED}:IndexedChart._intersect",
        rewrites=(
            Rewrite(
                old="if trigger >= 0 and position < trigger and node == item:",
                new="if False:",
            ),
        ),
        rejected_by=("plan/repeated-premise",),
        min_effect=1.0,
    ),
    Mutation(
        name="plan-keyed-repeated-premise-at-every-position",
        defect_class="repeated premise",
        provenance="the keyed join produces an instance at every premise its item fills",
        target=f"{INDEXED}:IndexedChart._extend",
        rewrites=(
            Rewrite(
                old="if trigger >= 0 and step.premise < trigger and candidate == item:",
                new="if False:",
            ),
        ),
        rejected_by=("plan/repeated-premise-keyed",),
        min_effect=1.0,
    ),
    Mutation(
        name="plan-tape-marked-measured",
        defect_class="evidence",
        provenance="a report declares the tape measured without counting it",
        target=f"{PLAN}:Evaluation.report",
        rewrites=(
            Rewrite(
                old="measured=self.measures,",
                new=(
                    "measured=measuring((*self.measures, "
                    "WorkField.ELEMENTS_SAVED_FOR_BACKWARD)),"
                ),
            ),
        ),
        rejected_by=("plan/measured-counts",),
        min_effect=1.0,
    ),
    Mutation(
        name="plan-requested-alternative-dropped",
        defect_class="omitted alternative",
        provenance="a requested item keeps only its first edge",
        target=f"{PLAN}:_keep",
        rewrites=(
            Rewrite(
                old="kept.setdefault(item, []).append(edge)",
                new="if item not in kept:\n        kept[item] = [edge]",
            ),
        ),
        rejected_by=("plan/requested-evidence",),
        min_effect=1.0,
    ),
    Mutation(
        name="plan-rules-without-premises-dropped",
        defect_class="omitted alternative",
        provenance="a rule no premise triggers is never used",
        target=f"{PLAN}:_constants",
        rewrites=(Rewrite(old="if rule.premises or not", new="if True or not"),),
        rejected_by=("plan/constant-rule",),
        min_effect=1.0,
    ),
    Mutation(
        name="plan-sum-replaced-by-max",
        defect_class="aggregation",
        provenance="alternatives are combined by a maximum instead of the sum",
        target=f"{PLAN}:_accumulate",
        rewrites=(
            Rewrite(
                old="return algebra.join(torch.stack(torch.broadcast_tensors(current, term)), 0)",
                new="return torch.maximum(current, term)",
            ),
        ),
        rejected_by=("plan/alternatives", "plan/weight"),
        min_effect=0.1,
    ),
    Mutation(
        name="plan-support-from-current-numbers",
        defect_class="support",
        provenance="a term that is zero at the current weights is dropped as absent",
        target=f"{PLAN}:_accumulate",
        rewrites=(
            Rewrite(
                old="if current is None:",
                new=(
                    "if bool(algebra.is_zero(term).all()):\n"
                    "        return term.detach() if current is None else current\n"
                    "    if current is None:"
                ),
            ),
        ),
        rejected_by=("plan/zero-gradient",),
        min_effect=1.0,
    ),
    Mutation(
        name="plan-premises-multiplied-in-reverse",
        defect_class="order",
        provenance="factors are multiplied from the last premise to the first",
        target=f"{PLAN}:_product",
        rewrites=(
            Rewrite(
                old="total = algebra.tensor_op(total, factor)",
                new="total = algebra.tensor_op(factor, total)",
            ),
        ),
        rejected_by=("plan/premise-order",),
        min_effect=0.5,
    ),
    Mutation(
        name="plan-observable-not-checked",
        defect_class="observer",
        provenance="an observable no executor answers is planned as a weight",
        target=f"{PLAN}:plan",
        also_in=("tests.evidence.plan_controls",),
        rewrites=(
            Rewrite(
                old="if not isinstance(query.observable, supported):",
                new="if False:",
            ),
        ),
        rejected_by=("plan/refusals",),
        min_effect=1.0,
    ),
)
"""Every wrong version of plans and the checks that reject it."""
