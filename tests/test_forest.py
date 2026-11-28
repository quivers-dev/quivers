"""Tests for packed forests and the plans that keep them."""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest
import torch

from quivers.core.algebras import LOG_PROB, REAL
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
    PresentationId,
    Recursion,
)
from quivers.qiec.identifiers import ComputationId, ConstructorId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    Derivation,
    HostInterpretation,
    Item,
    Judgment,
    JudgmentDecl,
    Premise,
    PremiseValue,
    Rule,
    RulePresentation,
    closed_values,
)
from quivers.qiec.types import IndexLiteral
from quivers.stochastic import forest as forest_module
from quivers.stochastic.forest import (
    DerivationForest,
    ForestEdge,
    ForestRefusal,
    ForestRefused,
    build_forest,
    edge_weight,
    fold_derivations,
    fold_derivations_many,
    forest_fold,
    forest_folds,
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


def _squared_forest(
    depth: int,
) -> tuple[DerivationForest, tuple[AxiomOccurrence, ...], tuple[Item, ...]]:
    """Each level chooses two proofs of the previous level independently."""
    identity = PresentationId.derive("global-proof-budget")
    family = FamilyId.derive(identity, "node")
    items = tuple(Item(family=family, arguments=(level,)) for level in range(depth + 1))
    judgments = tuple(
        Judgment(family=family, arguments=(IndexLiteral(value=level, sort=NAT),))
        for level in range(depth + 1)
    )
    slot = AxiomSlot(
        id=ConstructorId.derive(identity, "base"), name="base", judgment=judgments[0]
    )
    presentation = RulePresentation(
        id=identity,
        name="global_proof_budget",
        judgments=(
            JudgmentDecl(
                id=family,
                name="node",
                indices=(IndexBinder(name="level", sort=NAT, refinable=True),),
            ),
        ),
        rules=tuple(
            Rule(
                id=ConstructorId.derive(identity, level),
                name=f"square{level}",
                conclusion=judgment,
                premises=(Premise(judgment=prior), Premise(judgment=prior)),
            )
            for level, (prior, judgment) in enumerate(
                zip(judgments[:-1], judgments[1:], strict=True)
            )
        ),
        axioms=(slot,),
    )
    occurrences = tuple(
        AxiomOccurrence(
            slot=slot.id, occurrence=OccurrenceId.derive(slot.id, row), binding=()
        )
        for row in (0, 1)
    )
    return build_forest(presentation, occurrences), occurrences, items


def test_combined_proof_budget_counts_distinct_goals_and_shares_their_subproofs() -> (
    None
):
    forest, occurrences, (base, smaller, larger) = _squared_forest(2)
    absent = Item(family=base.family, arguments=(9,))
    labels = {row.occurrence: index for index, row in enumerate(occurrences)}
    calls: Counter[Item] = Counter()

    def observe(
        node: Derivation, children: tuple[PremiseValue[tuple[int, ...]], ...]
    ) -> tuple[int, ...]:
        calls[node.item] += 1
        if node.occurrence is not None:
            return (labels[node.occurrence],)
        return tuple(label for subtree in closed_values(children) for label in subtree)

    interpretation = HostInterpretation(
        presentation=forest.presentation,
        operations={
            identity: observe for identity in forest.presentation.constructors_by_id
        },
    )
    logs = torch.tensor([-0.4, -1.1], dtype=LIKE.dtype, requires_grad=True)
    weights = {
        row.occurrence: log for row, log in zip(occurrences, logs.unbind(), strict=True)
    }
    goals = (larger, smaller, larger, absent)
    with pytest.raises(
        ForestRefused, match="20 derivations exceed the declared limit of 19"
    ) as failed:
        fold_derivations_many(
            forest, goals, LOG_PROB, weights, interpretation, limit=19, like=LIKE
        )
    assert failed.value.code == ForestRefusal.TOO_MANY
    assert not calls

    accepted = fold_derivations_many(
        forest, goals, LOG_PROB, weights, interpretation, limit=20, like=LIKE
    )
    assert tuple(accepted) == (larger, smaller, absent)
    assert accepted[absent] == []
    assert calls == {base: 2, smaller: 4, larger: 16}
    assert forest.count(smaller) == 4
    assert [label for _, _, label in accepted[smaller]] == list(
        product(range(2), repeat=2)
    )
    assert [label for _, _, label in accepted[larger]] == list(
        product(range(2), repeat=4)
    )
    actual = torch.stack(
        [weight for goal in (smaller, larger) for _, weight, _ in accepted[goal]]
    )
    oracle = torch.stack(
        [
            logs[list(choice)].sum()
            for width in (2, 4)
            for choice in product(range(2), repeat=width)
        ]
    )
    torch.testing.assert_close(actual, oracle)
    torch.testing.assert_close(
        torch.autograd.grad(actual.logsumexp(0), logs)[0],
        torch.autograd.grad(oracle.logsumexp(0), logs)[0],
    )

    calls.clear()
    single = fold_derivations(
        forest, smaller, LOG_PROB, weights, interpretation, limit=4, like=LIKE
    )
    assert calls == {base: 2, smaller: 4}
    assert [proof for proof, _, _ in single] == [
        proof for proof, _, _ in accepted[smaller]
    ]
    torch.testing.assert_close(
        torch.stack([weight for _, weight, _ in single]), oracle[:4]
    )


def test_exponential_budget_refuses_before_building_any_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The last item has 2 ** (2 ** 20) proofs over a 21-item forest.
    forest, _, items = _squared_forest(20)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("a proof or callback was reached before the budget refusal")

    interpretation = HostInterpretation(
        presentation=forest.presentation,
        operations={
            identity: forbidden for identity in forest.presentation.constructors_by_id
        },
    )
    count = forest_module._counts

    def count_once(part, height, *, cap=None):
        calls.append(cap)
        result = count(part, height, cap=cap)
        assert set(result.values()) <= {2, 4, 16, 21}
        return result

    monkeypatch.setattr(forest_module, "Derivation", forbidden)
    monkeypatch.setattr(forest_module, "_counts", count_once)
    with pytest.raises(ForestRefused, match="at least 21 derivations") as failed:
        fold_derivations_many(
            forest, (items[-1],), LOG_PROB, {}, interpretation, limit=20, like=LIKE
        )
    assert failed.value.code == ForestRefusal.TOO_MANY
    assert calls == [21]


def test_zero_mass_proofs_still_count_as_structural_work() -> None:
    forest, occurrences, (goal,) = _squared_forest(0)
    interpretation = HostInterpretation(
        presentation=forest.presentation,
        operations={occurrences[0].slot: lambda node, children: None},
    )
    weights = {occurrences[0].occurrence: LIKE.new_tensor(-float("inf"))}
    with pytest.raises(
        ForestRefused, match="2 derivations exceed the declared limit of 1"
    ):
        fold_derivations_many(
            forest, (goal,), LOG_PROB, weights, interpretation, limit=1, like=LIKE
        )
    rows = fold_derivations_many(
        forest, (goal,), LOG_PROB, weights, interpretation, limit=2, like=LIKE
    )[goal]
    assert [float(weight) for _, weight, _ in rows] == [-float("inf"), 0.0]


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


def test_log_folds_share_premise_values_and_keep_occurrence_multiplicity() -> None:
    """Three uses of one ambiguous item give independent derivation choices."""
    edges = [(0, 1), (0, 1), (0, 2), (0, 3), (0, 3)]
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    weights = LIKE.new_tensor([-0.4, -1.6, -0.8, -torch.inf, -torch.inf])
    weights.requires_grad_()
    inputs: dict[ParameterId | OccurrenceId, torch.Tensor] = {
        graphs.edge_id(index): weight for index, weight in enumerate(weights.unbind())
    }
    first = graphs.node(1)
    tripled = Item(family=graphs.TRIPLE, arguments=(1,))
    other = graphs.node(2)
    impossible = graphs.node(3)
    goals = (first, tripled, other, impossible)
    visited: list[ForestEdge] = []

    def weight(edge: ForestEdge) -> torch.Tensor:
        """Read a weight and count one visit to its edge."""
        visited.append(edge)
        return edge_weight(forest, edge, LOG_PROB, inputs, like=LIKE)

    folded = forest_folds(forest, goals, LOG_PROB, weight, like=LIKE)
    one_proof = weights[2]
    two_proofs = torch.logsumexp(weights[:2], 0)
    eight_proofs = torch.logsumexp(
        torch.stack([sum(path) for path in product(weights[:2], repeat=3)]), 0
    )
    expected = torch.stack((two_proofs, eight_proofs, one_proof))
    actual = torch.stack([folded[goal] for goal in goals[:-1]])
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.isneginf(folded[impossible])

    # A zero chart row with two -inf inputs adjoins finite alternatives.
    score = LOG_PROB.join(torch.stack([folded[goal] for goal in goals]), 0)
    expected_score = torch.logsumexp(expected, 0)
    (gradient,) = torch.autograd.grad(score, weights, retain_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected_score, weights)
    assert torch.allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.equal(gradient[-2:], weights.new_zeros(2))

    part = forest.below_all(goals)
    assert len(visited) == sum(len(incoming) for incoming in part.edges.values())
    assert len(visited) == len(set(visited))


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
