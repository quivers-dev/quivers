"""Checks and deliberately wrong versions of two-pass plans.

The checks compare plans with the derivation oracle (O1) and its
gradients (O7) on the attachment grammar, with path sums and matrix
products computed by hand on input graphs, with refusals of what a plan
does not answer, and with linear work on chain, diamond and skip graphs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import torch

from quivers.core.algebras import (
    COUNTING,
    LOG_PROB,
    MAX_PLUS,
    REAL,
    MatrixSemiring,
    OrderPreservingSemiring,
)
from quivers.qiec.contracts import (
    Acyclic,
    Best,
    Bounded,
    InterpretationId,
    JudgmentPattern,
    NonlinearFold,
    NoRetention,
    Outcome,
    OccurrenceId,
    ParameterId,
    QuerySpec,
    RequestedEvidence,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.rules import Item, RulePresentation
from quivers.stochastic.planning import Plan, PlanRefusal, PlanRefused, plan
from tests.evidence import graphs, scaling
from tests.evidence import presentations as fixture
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
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


def _attempts() -> dict[PlanRefusal, Callable[[], Plan]]:
    """One plan the planner has to refuse per reason.

    Returns
    -------
    dict of PlanRefusal to Callable
        Attempts by the refusal they have to meet.
    """
    evidence = RequestedEvidence(patterns=(JudgmentPattern(family=graphs.REACH),))
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
        PlanRefusal.RETENTION_WITHOUT_EXECUTOR: lambda: plan(
            graphs.query().with_(retention=evidence),
            graphs.PRESENTATION,
            REAL,
            Acyclic(),
            graphs.contract(),
            evidence,
        ),
    }


def check_refusals() -> None:
    """Require each query a plan does not answer to be refused for its reason (M35).

    A fold that does not commute with sums cannot be applied to the
    aggregate a two-pass run computes: with alternatives ``-1`` and ``1``
    the square of the sum is zero and the sum of the squares two. So
    without the packed forest it is refused, as are bounded recursion
    without the forest, open premises, a pushforward, evidence restricted
    to patterns, and a run whose inputs make the derived hypergraph
    cyclic.

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
        ("plan/constant-rule", check_constant_rule),
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
        target=f"{INDEXED}:IndexedChart._extend",
        rewrites=(
            Rewrite(
                old="if trigger >= 0 and step.premise < trigger and candidate == item:",
                new="if False:",
            ),
        ),
        rejected_by=("plan/repeated-premise",),
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
