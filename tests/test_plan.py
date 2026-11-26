"""Tests for plans: what a plan records, reports and refuses."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.core.algebras import LOG_PROB, REAL, RealAlgebra
from quivers.qiec.contracts import (
    ALGEBRA_OPERATIONS,
    Acyclic,
    BestKnown,
    NoRetention,
    OccurrenceId,
    ParentCounts,
    Phase,
    PresentationId,
    ProvedOptimal,
    Retention,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.rules import Item
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
from tests.evidence import graphs
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
