"""Tests for packed forests and the plans that keep them."""

from __future__ import annotations

import pytest
import torch

from quivers.core.algebras import REAL
from quivers.core.laws import FOLD_LINEAR, ClaimSubject, LawClaim, Proved
from quivers.qiec.contracts import (
    Acyclic,
    Bounded,
    Fold,
    Forest,
    NonlinearFold,
    Observable,
    OccurrenceId,
    PackedForest,
    ParameterId,
    Recursion,
)
from quivers.qiec.identifiers import ComputationId
from quivers.stochastic.forest import (
    ForestRefusal,
    ForestRefused,
    build_forest,
    edge_weight,
    forest_fold,
)
from quivers.stochastic.planning import (
    FOREST_ALGORITHM,
    FOREST_WORK,
    Plan,
    PlanRefusal,
    PlanRefused,
    plan,
)
from tests.evidence import graphs

LIKE = torch.zeros((), dtype=torch.float64)
SQUARE = ComputationId.derive(graphs.NAME, "square")


def _forest_plan(observable: Observable, recursion: Recursion | None = None) -> Plan:
    """A plan that keeps the forest.

    Parameters
    ----------
    observable : Observable
        The observable.
    recursion : Recursion or None
        The recursion; acyclic by default.

    Returns
    -------
    Plan
        The plan.
    """
    return plan(
        graphs.query().with_(observable=observable, retention=PackedForest()),
        graphs.PRESENTATION,
        REAL,
        Acyclic() if recursion is None else recursion,
        graphs.contract(),
        PackedForest(),
    )


def test_a_kept_forest_selects_the_forest_executor() -> None:
    """The record names the forest executor and its cost."""
    record = _forest_plan(NonlinearFold(fold=SQUARE)).record
    assert record.algorithm == FOREST_ALGORITHM
    assert record.complexity == FOREST_WORK
    assert any("one to one" in text for text in record.assumptions)


def test_bounded_recursion_folds_on_the_forest_only() -> None:
    """Under a height bound the forest is built and the two-pass run refused."""
    current = _forest_plan(Forest(), Bounded(measure="height", limit=4))
    assert current.height == 4
    edges = [(0, 1), (1, 0)]
    forest = current.forest(graphs.occurrences(edges))
    assert not forest.acyclic
    assert forest.count(graphs.node(1), height=current.height) == 2
    with pytest.raises(PlanRefused) as caught:
        current.run(graphs.occurrences(edges), {}, like=LIKE)
    assert caught.value.code is PlanRefusal.RECURSION_WITHOUT_EXECUTOR


def test_bounded_recursion_without_the_forest_is_refused() -> None:
    """A height bound needs the forest."""
    with pytest.raises(PlanRefused) as caught:
        plan(
            graphs.query(),
            graphs.PRESENTATION,
            REAL,
            Bounded(measure="height", limit=4),
            graphs.contract(),
            graphs.query().retention,
        )
    assert caught.value.code is PlanRefusal.RECURSION_WITHOUT_EXECUTOR


def test_a_fold_needs_a_claim_of_linearity() -> None:
    """A linear fold is planned only with a proved claim that it is linear."""
    fold = ComputationId.derive(graphs.NAME, "fold")
    with pytest.raises(PlanRefused) as caught:
        _forest_plan(Fold(fold=fold))
    assert caught.value.code is PlanRefusal.INADMISSIBLE
    claim = LawClaim(
        law=FOLD_LINEAR,
        subject=ClaimSubject(kind="computation", name=str(fold)),
        evidence=Proved(argument="a sum of products of edge values over derivations"),
    )
    current = plan(
        graphs.query().with_(observable=Fold(fold=fold), retention=PackedForest()),
        graphs.PRESENTATION,
        REAL,
        Acyclic(),
        graphs.contract(),
        PackedForest(),
        claims=(claim,),
    )
    assert claim in current.record.claims


def test_two_pass_plans_do_not_build_forests() -> None:
    """A plan without the forest refuses to build one."""
    current = plan(
        graphs.query(),
        graphs.PRESENTATION,
        REAL,
        Acyclic(),
        graphs.contract(),
        graphs.query().retention,
    )
    with pytest.raises(PlanRefused):
        current.forest(graphs.occurrences([]))


def test_forest_fold_agrees_with_the_two_pass_run() -> None:
    """On skip graphs the forest's linear fold is the two-pass value."""
    edges = graphs.skips(8)
    weights = [0.5 + 0.05 * index for index in range(len(edges))]
    inputs: dict[ParameterId | OccurrenceId, torch.Tensor] = {
        **graphs.edge_weights(edges, weights)
    }
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    assert forest.acyclic
    folded = forest_fold(
        forest,
        graphs.node(8),
        REAL,
        lambda edge: edge_weight(forest, edge, REAL, inputs, like=LIKE),
        like=LIKE,
    )
    value = (
        plan(
            graphs.query(),
            graphs.PRESENTATION,
            REAL,
            Acyclic(),
            graphs.contract(),
            graphs.query().retention,
        )
        .run(graphs.occurrences(edges), inputs, like=LIKE)
        .value(graphs.node(8))
    )
    assert torch.allclose(folded, value, rtol=1e-12)
    assert forest.size >= len(edges)


def test_an_absent_goal_has_no_derivations() -> None:
    """An item the forest lacks counts zero and folds to zero."""
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences([(0, 1)]))
    assert forest.count(graphs.node(5)) == 0
    assert (
        float(
            forest_fold(forest, graphs.node(5), REAL, lambda edge: LIKE + 1, like=LIKE)
        )
        == 0.0
    )
    cyclic = build_forest(graphs.PRESENTATION, graphs.occurrences([(0, 1), (1, 0)]))
    with pytest.raises(ForestRefused) as caught:
        forest_fold(cyclic, graphs.node(1), REAL, lambda edge: LIKE + 1, like=LIKE)
    assert caught.value.code is ForestRefusal.UNBOUNDED
