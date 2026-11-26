"""Tests for rule-body joins: queries, covers, records and claims."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from quivers.core.algebras import LOG_PROB, REAL, RealAlgebra
from quivers.qiec.contracts import Conditional, Phase, ProvedOptimal
from quivers.stochastic.joins import (
    ELIMINATION_LAWS,
    ELIMINATION_WORK,
    GENERIC_JOIN_WORK,
    JoinCounter,
    JoinRefusal,
    JoinRefused,
    agm_bound,
    body_query,
    eliminate,
    elimination_order,
    enumerate_aggregate,
    generic_join,
    relations,
)
from tests.evidence import bodies
from tests.evidence import presentations as attachment

LIKE = torch.zeros((), dtype=torch.float64)


class _Unproved(RealAlgebra):
    """The reals with no evidence of their own, so every law is only assumed."""


def test_body_query_of_the_triangle() -> None:
    """Atoms are the premises over their variables; the conclusion's are free."""
    query = body_query(bodies.TRIANGLE)
    assert query.variables == ("a", "b", "c")
    assert [atom.variables for atom in query.atoms] == [
        ("a", "b"),
        ("b", "c"),
        ("a", "c"),
    ]
    assert query.free == ("a", "b", "c")
    assert query.filtered == ()
    star = body_query(bodies.STAR)
    assert star.free == ("x",)
    first = body_query(bodies.FIRST)
    assert first.filtered == ("a", "c")


def test_relations_flatten_literal_patterns() -> None:
    """A premise with a category literal keeps only items of that category."""
    rule = attachment.PRESENTATION.rules[0]
    query = body_query(rule)
    items = [
        attachment.Item(family=attachment.SPAN, arguments=("NP", 0, 1)),
        attachment.Item(family=attachment.SPAN, arguments=("VP", 1, 2)),
        attachment.Item(family=attachment.SPAN, arguments=("V", 1, 2)),
    ]
    tables = relations(query, items)
    sizes = [len(table.tuples) for table in tables]
    assert sum(sizes) >= 1
    for table in tables:
        for values, item in table.tuples.items():
            assert len(values) == len(table.atom.variables)
            assert item.family == attachment.SPAN


def test_sum_of_two_variables_is_refused() -> None:
    """A premise ``u(i + k)`` is not a relation of its items alone."""
    with pytest.raises(JoinRefused) as caught:
        body_query(bodies.SUMMED)
    assert caught.value.code is JoinRefusal.NOT_FLAT


def test_agm_bound_of_paths_and_triangles() -> None:
    """A path of two atoms has cover weight 2; a triangle 3/2; an empty relation bounds 0."""
    path = agm_bound(("a", "b", "c"), (("a", "b"), ("b", "c")), (10, 10))
    assert path.rho == 2
    assert path.value == pytest.approx(100.0)
    triangle = agm_bound(
        ("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")), (16, 16, 16)
    )
    assert triangle.rho == Fraction(3, 2)
    assert triangle.value == pytest.approx(64.0)
    empty = agm_bound(("a", "b"), (("a", "b"),), (0,))
    assert empty.value == 0.0
    with pytest.raises(JoinRefused):
        agm_bound(("a", "z"), (("a",),), (3,))


def test_generic_join_counts_its_work() -> None:
    """The counters see every output, and the hub's triangles are 3n + 1."""
    query = body_query(bodies.TRIANGLE)
    counter = JoinCounter()
    found = list(
        generic_join(
            query, relations(query, bodies.hub(5)), bodies.PRESENTATION, work=counter
        )
    )
    assert len(found) == 3 * 5 + 1
    assert counter.report().outputs == len(found)


def test_elimination_records_its_order_and_widths() -> None:
    """The star's order removes each leaf variable with bag ``{x, y}``."""
    query = body_query(bodies.STAR)
    assert elimination_order(query) == ("y1", "y2", "y3")
    result = eliminate(
        query,
        relations(query, bodies.star(3)),
        bodies.PRESENTATION,
        LOG_PROB,
        {},
        like=LIKE,
    )
    assert result.algorithm == "variable elimination"
    assert [step.bag for step in result.steps] == [
        ("x", "y1"),
        ("x", "y2"),
        ("x", "y3"),
    ]
    assert result.width == 1
    assert result.fractional_width.fraction == 1
    # log weights at zero: x = 0 has 3 * 3 * 3 instances
    hub = next(item for item in result.values if item.arguments == (0,))
    assert float(result.values[hub]) == pytest.approx(
        float(torch.log(torch.tensor(27.0)))
    )


def test_elimination_takes_a_given_order() -> None:
    """A given order is used, and one naming the free variable is refused."""
    query = body_query(bodies.STAR)
    tables = relations(query, bodies.star(3))
    result = eliminate(
        query,
        tables,
        bodies.PRESENTATION,
        REAL,
        {},
        like=LIKE,
        order=("y3", "y1", "y2"),
    )
    assert result.order == ("y3", "y1", "y2")
    with pytest.raises(JoinRefused) as caught:
        eliminate(
            query, tables, bodies.PRESENTATION, REAL, {}, like=LIKE, order=("x", "y1")
        )
    assert caught.value.code is JoinRefusal.ORDER


def test_algebras_without_the_laws_are_refused() -> None:
    """An algebra whose laws are only assumed is neither eliminated nor enumerated."""
    query = body_query(bodies.STAR)
    tables = relations(query, bodies.star(2))
    for compute in (eliminate, enumerate_aggregate):
        with pytest.raises(JoinRefused) as caught:
            compute(query, tables, bodies.PRESENTATION, _Unproved(), {}, like=LIKE)
        assert caught.value.code is JoinRefusal.LAWS


def test_rule_weight_multiplies_the_aggregate() -> None:
    """The rule weight scales every conclusion."""
    query = body_query(bodies.STAR)
    tables = relations(query, bodies.star(2))
    plain = eliminate(query, tables, bodies.PRESENTATION, REAL, {}, like=LIKE)
    scaled = eliminate(
        query,
        tables,
        bodies.PRESENTATION,
        REAL,
        {},
        like=LIKE,
        rule_weight=torch.tensor(3.0, dtype=torch.float64),
    )
    for item, value in plain.values.items():
        assert float(scaled.values[item]) == pytest.approx(3.0 * float(value))


def test_claims_state_status_and_conditions() -> None:
    """Generic Join is worst-case optimal; elimination is conditional on ETH."""
    execution = next(
        bound
        for bound in GENERIC_JOIN_WORK.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "time"
    )
    assert isinstance(execution.status, ProvedOptimal)
    assert any("not instance optimal" in text for text in GENERIC_JOIN_WORK.conditions)
    eliminated = next(
        bound
        for bound in ELIMINATION_WORK.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "time"
    )
    assert isinstance(eliminated.status, Conditional)
    assert any("NP-hard" in text for text in ELIMINATION_WORK.conditions)
    assert {law.name for law in ELIMINATION_LAWS} >= {
        "times-commutative",
        "distributive",
    }
