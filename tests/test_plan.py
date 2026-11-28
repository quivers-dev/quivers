"""Tests for plans: what a plan records, reports and refuses."""

from __future__ import annotations

import math
from itertools import product

import pytest
import torch

from quivers.core.algebras import LOG_PROB, REAL, RealAlgebra
from quivers.qiec.contracts import (
    InterpretationId,
    QuerySpec,
    Weight,
    ALGEBRA_OPERATIONS,
    Acyclic,
    BestKnown,
    NoRetention,
    OccurrenceId,
    ParentCounts,
    ParameterId,
    Phase,
    PresentationId,
    ProvedOptimal,
    Retention,
    Support,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import AxiomOccurrence, AxiomSlot, Item
from quivers.qiec.types import IndexLiteral, IndexSum, IndexVariable
from quivers.stochastic.indexed import index_variables, join_plans
from quivers.stochastic.planning import (
    RECOMPUTED_TWO_PASS_WORK,
    TWO_PASS_WORK,
    Plan,
    PlanRefusal,
    PlanRefused,
    acyclicity_certificate,
    plan,
)
from tests.evidence import bodies, graphs
from tests.evidence import presentations as fixture

LIKE = torch.zeros((), dtype=torch.float64)


class _Unproved(RealAlgebra):
    """The reals with no evidence of their own, so every law is only assumed."""


def _plan(*, recompute: bool = False, retention: Retention | None = None) -> Plan:
    """A plan of the reachability presentation.

    Parameters
    ----------
    recompute : bool
        Whether to set a memory budget.
    retention : Retention or None
        The provenance kept; none by default.

    Returns
    -------
    Plan
        The plan.
    """
    kept = NoRetention() if retention is None else retention
    return plan(
        graphs.query().with_(retention=kept),
        graphs.PRESENTATION,
        REAL,
        Acyclic(),
        graphs.contract(recompute=recompute),
        kept,
    )


def test_record_names_the_algorithm_and_what_it_rests_on() -> None:
    """The record holds the query, contracts, laws, structure and alternatives."""
    record = _plan().record
    assert (
        record.algorithm == "two-pass evaluation by parent counts on an indexed chart"
    )
    assert record.query == graphs.query()
    assert record.recursion == Acyclic()
    assert record.accuracy == graphs.contract()
    assert record.algebra == REAL.subject
    assert record.claims == REAL.claims()
    assert record.structure == graphs.PRESENTATION.analysis
    assert record.alternatives
    assert any("premise order" in text for text in record.assumptions)
    assert record.certificate is None


def test_record_cost_counts_the_tape_under_autograd() -> None:
    """The derivative's space is ``V + P E`` with autograd and ``V + P D`` recomputed."""
    autograd = _plan().record
    recomputed = _plan(recompute=True).record
    assert autograd.backward == "autograd"
    assert recomputed.backward == "recompute"
    assert autograd.complexity == TWO_PASS_WORK
    assert recomputed.complexity == RECOMPUTED_TWO_PASS_WORK

    assert TWO_PASS_WORK.model == ALGEBRA_OPERATIONS
    spaces = {
        name: next(
            bound
            for bound in claim.bounds
            if bound.phase is Phase.DIFFERENTIATION and bound.resource == "space"
        )
        for name, claim in (
            ("autograd", TWO_PASS_WORK),
            ("recompute", RECOMPUTED_TWO_PASS_WORK),
        )
    }
    assert spaces["autograd"].expression == "V + P E"
    assert spaces["recompute"].expression == "V + P D"
    assert isinstance(spaces["autograd"].status, BestKnown)
    execution = next(
        bound
        for bound in TWO_PASS_WORK.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "space"
    )
    assert isinstance(execution.status, ProvedOptimal)


def test_parent_counts_are_admitted_and_reported() -> None:
    """Parent counts are the provenance a run keeps: one per item."""
    edges = graphs.diamonds(2)
    current = _plan(retention=ParentCounts())
    evaluation = current.run(
        graphs.occurrences(edges),
        {**graphs.edge_weights(edges, [1.0] * len(edges))},
        like=LIKE,
    )
    report = evaluation.report()
    assert report.phase is Phase.EXECUTION
    assert report.provenance_records == len(evaluation.order)
    assert report.hash_operations == evaluation.work.lookups
    assert report.key_comparisons == evaluation.work.candidates
    # eight step instances multiply the weight by two premises, and seven
    # tripled ones by three
    assert report.scalar_products == 2 * 8 + 3 * 7
    assert float(evaluation.value(graphs.node(6))) == 4.0


@pytest.mark.parametrize("algebra", [REAL, LOG_PROB])
def test_family_elimination_preserves_labelled_proofs(algebra) -> None:
    """Each ordered premise can independently select either axiom occurrence."""
    triple = Item(family=graphs.TRIPLE, arguments=(0,))
    goal = Item(family=graphs.EDGE, arguments=(0, 0))
    base_id = ParameterId.derive("family-elimination", "base")
    constant = graphs.TRIPLED.with_(
        id=ConstructorId.derive("family-elimination", "base"),
        name="base",
        telescope=(),
        premises=(),
        conclusion=graphs.TRIPLED.conclusion.with_(
            arguments=(IndexLiteral(value=0, sort=NAT),)
        ),
        weight=base_id,
    )
    finish = graphs.TRIPLED.with_(
        id=ConstructorId.derive("family-elimination", "finish"),
        name="finish",
        premises=(
            graphs.TRIPLED.premises[0].with_(judgment=graphs.TRIPLED.conclusion),
        ),
        conclusion=graphs.TRIPLED.conclusion.with_(
            family=graphs.EDGE, arguments=graphs.TRIPLED.conclusion.arguments * 2
        ),
        weight=None,
    )
    presentation = graphs.PRESENTATION.with_(rules=(finish, constant, graphs.TRIPLED))
    (first,) = graphs.occurrences([])
    second = first.with_(occurrence=OccurrenceId.derive("family-elimination", "second"))
    weights = LIKE.new_tensor([[0.0, 2.0], [3.0, 1.0], [5.0, 3.0], [7.0, 11.0]])
    if algebra is LOG_PROB:
        weights = weights.log()
    weights.requires_grad_()
    parameters = (first.occurrence, second.occurrence, graphs.TRIPLED.weight, base_id)
    inputs = dict(zip(parameters, weights.unbind(), strict=True))
    occurrences = (first, second)
    args = (
        graphs.query(),
        presentation,
        algebra,
        Acyclic(),
        graphs.contract(),
        NoRetention(),
    )
    selected = plan(*args, executor="eliminate")
    chart = plan(*args)
    masses = weights.exp() if algebra is LOG_PROB else weights
    # Index triples are ordered proofs. The two occurrences have the same
    # judgment but distinct identities. Their eight products are all terms.
    exact = masses[3] + sum(
        masses[2] * masses[left] * masses[middle] * masses[right]
        for left, middle, right in product(range(2), repeat=3)
    )
    if algebra is LOG_PROB:
        exact = exact.log()
    result = selected.run(occurrences, inputs, like=LIKE)
    assert result.order.index(graphs.node(0)) < result.order.index(triple)
    assert result.order.index(triple) < result.order.index(goal)
    assert result.report().provenance_records == 0
    assert [join.algorithm for join in selected.record.joins] == [
        "variable elimination",
        "constant",
        "variable elimination",
    ]
    assert [work.outputs for work in result.join_work] == [1, 1]
    for engine in (selected, chart):
        value = engine.run(occurrences, inputs, like=LIKE).value(goal)
        torch.testing.assert_close(value, exact)
        derivative = engine.gradient(occurrences, inputs, goal, like=LIKE)
        (oracle,) = torch.autograd.grad(exact.sum(), weights, retain_graph=True)
        torch.testing.assert_close(
            torch.stack([derivative[key] for key in parameters]), oracle
        )
    # The term with the zero occurrence is structurally present and has
    # a nonzero first derivative in the real rather than log coordinates.
    if algebra is REAL:
        assert oracle[0, 0].item() == 135.0
    empty = selected.run([], {base_id: LIKE.new_tensor(7.0)}, like=LIKE)
    assert set(empty.values) == {triple, goal}
    assert empty.value(goal).item() == 7.0


def test_family_elimination_avoids_the_cubic_star() -> None:
    """Three independent partners are summed before their masses are multiplied."""
    slots = tuple(
        AxiomSlot(
            id=ConstructorId.derive("family-elimination", family),
            name=family,
            telescope=tuple(IndexBinder(name=name, sort=NAT) for name in ("x", "y")),
            judgment=bodies.judgment(family, "x", "y"),
        )
        for family in ("r1", "r2", "r3")
    )
    presentation = bodies.PRESENTATION.with_(rules=(bodies.STAR,), axioms=slots)
    args = (
        graphs.query().with_(presentation=presentation.id),
        presentation,
        REAL,
        Acyclic(),
        graphs.contract(),
        NoRetention(),
    )
    selected = plan(*args, executor="eliminate")
    chart = plan(*args)
    assert selected.record.joins[0].order == ("y1", "y2", "y3")
    size = 8
    occurrences = tuple(
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive("family-elimination", slot.name, str(y)),
            binding=(("x", 0), ("y", y)),
        )
        for slot in slots
        for y in range(size)
    )
    result = selected.run(occurrences, {}, like=LIKE)
    chart_result = chart.run(occurrences, {}, like=LIKE)
    assert result.values == chart_result.values
    (goal,) = bodies.items("hub", [(0,)])
    assert result.value(goal).item() == size**3
    (join,) = result.join_work
    assert join.outputs == 1
    assert join.iterated <= 4 * size
    assert result.report().scalar_sums + result.report().scalar_products <= 4 * size
    assert chart_result.work.instances >= size**3


def test_family_elimination_refuses_unsupported_contracts() -> None:
    """An explicit executor cannot silently switch to a chart or a forest."""
    presentation = graphs.PRESENTATION.with_(rules=(graphs.TRIPLED,))
    query = graphs.query()
    no_retention = NoRetention()
    with pytest.raises(PlanRefused, match="support") as caught:
        plan(
            query.with_(observable=Support()),
            presentation,
            REAL,
            Acyclic(),
            graphs.contract(),
            no_retention,
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR
    kept = ParentCounts()
    with pytest.raises(PlanRefused, match="provenance") as caught:
        plan(
            query.with_(retention=kept),
            presentation,
            REAL,
            Acyclic(),
            graphs.contract(),
            kept,
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.RETENTION_WITHOUT_EXECUTOR
    with pytest.raises(PlanRefused, match="established laws") as caught:
        plan(
            query,
            presentation,
            _Unproved(),
            Acyclic(),
            graphs.contract(),
            no_retention,
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.ALGEBRA
    with pytest.raises(PlanRefused, match="resource limits") as caught:
        plan(
            query,
            presentation,
            REAL,
            Acyclic(),
            graphs.contract(recompute=True),
            no_retention,
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.INADMISSIBLE
    with pytest.raises(PlanRefused, match="reach -> reach") as caught:
        plan(
            query,
            graphs.PRESENTATION,
            REAL,
            Acyclic(),
            graphs.contract(),
            no_retention,
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.CYCLE


@pytest.mark.parametrize("rule", [bodies.FIRST, bodies.SUMMED])
def test_family_elimination_checks_rule_fragment(rule) -> None:
    """The separate rule-elimination solver can use a wider rule fragment."""
    presentation = bodies.PRESENTATION.with_(rules=(rule,))
    with pytest.raises(PlanRefused, match=rule.name) as caught:
        plan(
            graphs.query().with_(presentation=presentation.id),
            presentation,
            REAL,
            Acyclic(),
            graphs.contract(),
            NoRetention(),
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.ELIMINATION_FRAGMENT


def test_family_elimination_refuses_unbound_rule_index() -> None:
    """A rule row identifies a grounding only if its premises bind the telescope."""
    indexed = graphs.TRIPLED.with_(
        telescope=(*graphs.TRIPLED.telescope, IndexBinder(name="unbound", sort=NAT))
    )
    presentation = graphs.PRESENTATION.with_(rules=(indexed,))
    with pytest.raises(
        PlanRefused, match=r"'unbound'.*unbound by its premises"
    ) as caught:
        plan(
            graphs.query(),
            presentation,
            REAL,
            Acyclic(),
            graphs.contract(),
            NoRetention(),
            executor="eliminate",
        )
    assert caught.value.code is PlanRefusal.ELIMINATION_FRAGMENT


def test_another_presentation_is_refused() -> None:
    """A query naming another presentation is refused before planning."""
    query = graphs.query().with_(presentation=PresentationId.derive("other"))
    with pytest.raises(PlanRefused) as caught:
        plan(
            query,
            graphs.PRESENTATION,
            REAL,
            Acyclic(),
            graphs.contract(),
            NoRetention(),
        )
    assert caught.value.code is PlanRefusal.INADMISSIBLE


def test_constant_goal_has_zero_gradient() -> None:
    """A goal no input weight reaches has a zero gradient, not an error."""
    start = OccurrenceId.derive(graphs.NAME, "start")
    gradient = _plan().gradient(
        graphs.occurrences([(0, 1)]),
        {start: torch.tensor(2.0, dtype=torch.float64)},
        graphs.node(1),
        like=LIKE,
    )
    assert float(gradient[start]) == 1.0
    gradient = _plan().gradient(graphs.occurrences([]), {}, graphs.node(0), like=LIKE)
    assert gradient == {}


def test_span_measure_is_found_and_applies() -> None:
    """``j - i`` is among the measures proving the span rules acyclic."""
    certificate = acyclicity_certificate(fixture.PRESENTATION)
    assert certificate is not None
    (component,) = certificate.components
    assert ((0, -1, 1),) in component.alternatives
    assert certificate.applies([Item(family=fixture.SPAN, arguments=("S", 0, 5))])
    # an empty span has measure zero under j - i, which then does not apply
    alone = component.with_(alternatives=(((0, -1, 1),),))
    assert alone.applies([Item(family=fixture.SPAN, arguments=("S", 0, 5))])
    assert not alone.applies([Item(family=fixture.SPAN, arguments=("S", 2, 2))])


def test_join_plans_key_each_premise_on_bound_variables() -> None:
    """The step rule looks up its other premise on the shared variable."""
    plans = join_plans(graphs.PRESENTATION)
    from_reach = plans[(graphs.STEP.id, 0)]
    (step,) = from_reach.steps
    assert step.premise == 1
    assert step.keyed == (0,)
    from_edge = plans[(graphs.STEP.id, 1)]
    (step,) = from_edge.steps
    assert step.premise == 0
    assert step.keyed == (0,)


def test_index_variables() -> None:
    """Variables of literals, variables and sums."""
    assert index_variables(IndexLiteral(value=3, sort=NAT)) == set()
    assert index_variables(IndexVariable(name="i", sort=NAT)) == {"i"}
    total = IndexSum(
        terms=(IndexVariable(name="i", sort=NAT), IndexVariable(name="k", sort=NAT)),
        constant=1,
    )
    assert index_variables(total) == {"i", "k"}


def test_log_domain_run_matches_the_real_one() -> None:
    """Log weights give the log of the real path sum."""
    edges = graphs.skips(6)
    weights = [0.5 + 0.1 * index for index in range(len(edges))]
    real = _plan().run(
        graphs.occurrences(edges),
        {**graphs.edge_weights(edges, weights)},
        like=LIKE,
    )
    log = plan(
        graphs.query(),
        graphs.PRESENTATION,
        LOG_PROB,
        Acyclic(),
        graphs.contract(),
        NoRetention(),
    ).run(
        graphs.occurrences(edges),
        {**graphs.edge_weights(edges, [math.log(w) for w in weights])},
        like=LIKE,
    )
    assert torch.allclose(
        log.value(graphs.node(6)).exp(), real.value(graphs.node(6)), rtol=1e-12
    )


def test_an_algebra_without_the_laws_is_refused() -> None:
    """An algebra whose laws are only assumed is not planned for an exact result."""
    with pytest.raises(PlanRefused) as caught:
        plan(
            graphs.query(),
            graphs.PRESENTATION,
            _Unproved(),
            Acyclic(),
            graphs.contract(),
            NoRetention(),
        )
    assert caught.value.code is PlanRefusal.INADMISSIBLE
    assert "claim-not-established: distributive" in caught.value.reason


def test_record_states_each_rules_join_and_bound() -> None:
    """Each rule's join algorithm and the cover number of its body are recorded."""
    joins = {join.name: join for join in _plan().record.joins}
    assert joins["step"].algorithm == "generic join"
    assert joins["step"].rho.fraction == 1
    assert joins["tripled"].algorithm == "generic join"
    spans = plan(
        QuerySpec(
            presentation=fixture.PRESENTATION.id,
            interpretation=InterpretationId.derive(fixture.NAME, "weight"),
            observable=Weight(),
        ),
        fixture.PRESENTATION,
        LOG_PROB,
        Acyclic(),
        graphs.contract(),
        NoRetention(),
    ).record
    assert all(join.algorithm == "generic join" for join in spans.joins)
    assert all(join.rho.fraction == 2 for join in spans.joins)
