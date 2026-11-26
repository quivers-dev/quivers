"""Checks and deliberately wrong versions of rule-body joins.

Joins are compared with a brute-force enumeration of every choice of one
tuple per atom, in every variable order; aggregates with the sums of the
enumerated products; work with the AGM bound on the triangle with a hub
and on a skewed triangle, where a plan that joins two atoms first is of
the next order, and with linear work on the star, where listing the
instances is cubic.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from quivers.core.algebras import (
    COUNTING,
    MAX_PLUS,
    REAL,
    MatrixSemiring,
    OrderPreservingSemiring,
)
from quivers.qiec.rules import Binding, Ground, Item, Rule, evaluate_index, match_index
from quivers.stochastic.indexed import IndexedChart, side_condition_holds
from quivers.stochastic.joins import (
    BodyQuery,
    Elimination,
    JoinCounter,
    JoinRefusal,
    JoinRefused,
    agm_bound,
    body_query,
    eliminate,
    enumerate_aggregate,
    generic_join,
    relations,
)
from tests.evidence import bodies, scaling
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.mutations import Mutation, Rewrite

LIKE = torch.zeros((), dtype=torch.float64)
"""The dtype and device of scalar weights."""

type Instance = tuple[tuple[tuple[str, Ground], ...], tuple[Item, ...]]
"""An instance as a sorted binding and its premise items."""


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


def brute_force(rule: Rule, items: Sequence[Item]) -> list[Instance]:
    """Every instance of a rule body, by trying every choice of premise items.

    Parameters
    ----------
    rule : Rule
        The rule.
    items : Sequence of Item
        The items.

    Returns
    -------
    list of Instance
        Every instance, sorted.
    """
    choices = [
        [
            item
            for item in dict.fromkeys(items)
            if item.family == premise.judgment.family
        ]
        for premise in rule.premises
    ]
    found: list[Instance] = []
    for chosen in itertools.product(*choices):
        binding: Binding = {}
        if not all(
            match_index(pattern, value, binding)
            for item, premise in zip(chosen, rule.premises, strict=True)
            for pattern, value in zip(
                premise.judgment.arguments, item.arguments, strict=True
            )
        ):
            continue
        if not side_condition_holds(bodies.PRESENTATION, rule, binding):
            continue
        found.append((tuple(sorted(binding.items())), tuple(chosen)))
    return sorted(found, key=repr)


def _random_items(seed: int, size: int) -> list[Item]:
    """Random tuples of every binary and unary form on a small domain.

    Parameters
    ----------
    seed : int
        The seed.
    size : int
        Tuples per form.

    Returns
    -------
    list of Item
        The items.
    """
    generator = random.Random(seed)
    found: list[Item] = []
    for name in ("r", "s", "t", "r1", "r2", "r3"):
        found += bodies.items(
            name,
            [(generator.randrange(4), generator.randrange(4)) for _ in range(size)],
        )
    for name in ("u", "w"):
        found += bodies.items(name, [(generator.randrange(4),) for _ in range(size)])
    return found


_RULES = (bodies.TRIANGLE, bodies.STAR, bodies.FIRST, bodies.SEPARATE)
"""The bodies compared with the brute force."""


def _listed(
    query: BodyQuery, items: Sequence[Item], order: Sequence[str]
) -> list[Instance]:
    """The instances Generic Join lists in one order, sorted.

    Parameters
    ----------
    query : BodyQuery
        The query.
    items : Sequence of Item
        The items.
    order : Sequence of str
        The variable order.

    Returns
    -------
    list of Instance
        The instances.
    """
    return sorted(
        (
            (tuple(sorted(binding.items())), premises)
            for binding, premises in generic_join(
                query, relations(query, items), bodies.PRESENTATION, order=order
            )
        ),
        key=repr,
    )


def check_instances() -> None:
    """Require Generic Join to list exactly the brute force's instances (O1).

    Every variable order of every body is tried on random inputs.

    Raises
    ------
    Mismatch
        If an instance is missing or extra in some order.
    """
    for seed in range(6):
        items = _random_items(seed, 6)
        for rule in _RULES:
            query = body_query(rule)
            expected = brute_force(rule, items)
            for order in itertools.permutations(query.variables):
                found = _listed(query, items, order)
                if found != expected:
                    raise Mismatch(
                        "value",
                        float(abs(len(found) - len(expected)) or 1),
                        f"{rule.name} in order {order}, seed {seed}: "
                        f"{len(found)} instances for {len(expected)}",
                    )


def check_agm() -> None:
    """Require the output within the AGM bound and the triangle's cover of 3/2.

    Raises
    ------
    Mismatch
        If an output exceeds its bound, a cover fails to cover, or the
        triangle's least cover is not 3/2 at equal sizes.
    """
    for seed in range(6):
        items = _random_items(seed, 8)
        for rule in _RULES:
            query = body_query(rule)
            tables = relations(query, items)
            bound = agm_bound(
                query.variables,
                [atom.variables for atom in query.atoms],
                [len(table.tuples) for table in tables],
            )
            for name in query.variables:
                covered = sum(
                    weight.fraction
                    for weight, atom in zip(bound.cover, query.atoms, strict=True)
                    if name in atom.variables
                )
                _require(covered >= 1, f"{rule.name}: {name} covered {covered}")
            size = len(brute_force(rule, items))
            _require(
                size <= bound.value * (1 + 1e-9),
                f"{rule.name}: {size} instances above the bound {bound.value}",
            )
    triangle = body_query(bodies.TRIANGLE)
    equal = agm_bound(
        triangle.variables, [atom.variables for atom in triangle.atoms], [100] * 3
    )
    _require(equal.rho == 1.5, f"the triangle's cover weighs {equal.rho}")


def _weights(items: Sequence[Item], seed: int) -> dict[Item, torch.Tensor]:
    """Random positive weights.

    Parameters
    ----------
    items : Sequence of Item
        The items.
    seed : int
        The seed.

    Returns
    -------
    dict of Item to torch.Tensor
        A weight per item.
    """
    generator = random.Random(seed)
    return {
        item: torch.tensor(generator.uniform(0.5, 2.0), dtype=torch.float64)
        for item in items
    }


def _expected(
    rule: Rule,
    items: Sequence[Item],
    weights: dict[Item, torch.Tensor],
    algebra: OrderPreservingSemiring,
    like: torch.Tensor,
) -> dict[Item, torch.Tensor]:
    """The aggregate of a body by summing the brute force's products in premise order.

    Parameters
    ----------
    rule : Rule
        The rule.
    items : Sequence of Item
        The items.
    weights : dict of Item to torch.Tensor
        The weights.
    algebra : OrderPreservingSemiring
        The algebra.
    like : torch.Tensor
        The dtype and device.

    Returns
    -------
    dict of Item to torch.Tensor
        The sum for each conclusion.
    """
    totals: dict[Item, torch.Tensor] = {}
    for binding, premises in brute_force(rule, items):
        values = dict(binding)
        conclusion = Item(
            family=rule.conclusion.family,
            arguments=tuple(
                evaluate_index(a, values) for a in rule.conclusion.arguments
            ),
        )
        product = algebra.unit_like((), like)
        for item in premises:
            product = algebra.tensor_op(product, weights[item])
        if conclusion in totals:
            product = algebra.join(torch.stack([totals[conclusion], product]), 0)
        totals[conclusion] = product
    return totals


def _compare(
    found: dict[Item, torch.Tensor],
    expected: dict[Item, torch.Tensor],
    what: str,
) -> None:
    """Require two aggregates to have the same conclusions and close values.

    Parameters
    ----------
    found : dict of Item to torch.Tensor
        The computed aggregate.
    expected : dict of Item to torch.Tensor
        The reference.
    what : str
        A description.

    Raises
    ------
    Mismatch
        If a conclusion is missing or extra, or a value differs.
    """
    if set(found) != set(expected):
        raise Mismatch(
            "value",
            float(len(set(found) ^ set(expected))),
            f"{what}: {len(found)} conclusions for {len(expected)}",
        )
    for item, value in expected.items():
        compare_number(
            float(found[item]),
            float(value),
            Accuracy(domain="linear", operations=256, magnitude=abs(float(value)) + 1),
            f"{what} at {item.arguments}",
        )


def check_aggregates() -> None:
    """Require elimination and enumeration to give the brute force's sums (O1).

    Over the reals and over max-plus, on random inputs and on the
    star, for every body including the one with a side condition.

    Raises
    ------
    Mismatch
        If a sum differs.
    """
    cases = [(seed, _random_items(seed, 6)) for seed in range(6)]
    cases.append((99, bodies.star(3)))
    for seed, items in cases:
        weights = _weights(items, seed)
        for rule in (bodies.TRIANGLES, bodies.STAR, bodies.FIRST, bodies.SEPARATE):
            query = body_query(rule)
            tables = relations(query, items)
            for algebra in (REAL, MAX_PLUS):
                expected = _expected(rule, items, weights, algebra, LIKE)
                for compute in (eliminate, enumerate_aggregate):
                    result = compute(
                        query, tables, bodies.PRESENTATION, algebra, weights, like=LIKE
                    )
                    _compare(
                        result.values,
                        expected,
                        f"{compute.__name__} of {rule.name} over "
                        f"{algebra.subject.name}, seed {seed}",
                    )


def check_empty_factor() -> None:
    """Require a body with an empty premise relation to conclude nothing.

    ``total() <- u(a), w(b)`` with ``w`` empty sums out ``a`` to a factor
    of no variables and finds none for ``b``; no conclusion results.

    Raises
    ------
    Mismatch
        If a conclusion is produced.
    """
    items = bodies.items("u", [(0,), (1,)])
    query = body_query(bodies.SEPARATE)
    result = eliminate(
        query, relations(query, items), bodies.PRESENTATION, REAL, {}, like=LIKE
    )
    if result.values:
        raise Mismatch(
            "value", 1.0, f"an empty body concluded {list(result.values.values())}"
        )


def check_refusals_and_order() -> None:
    """Require refusals, and products in premise order when the product does not commute.

    A premise that is a sum of two variables is not a relation of its
    items; elimination is refused for matrices, whose aggregate by
    enumeration multiplies the premises in order and then the rule
    weight; an order of other variables is refused.

    Raises
    ------
    Mismatch
        If a refusal is missing, or the matrices are multiplied out of order.
    """
    attempts: dict[
        JoinRefusal,
        Callable[[], BodyQuery | Elimination | list[tuple[Binding, tuple[Item, ...]]]],
    ] = {
        JoinRefusal.NOT_FLAT: lambda: body_query(bodies.SUMMED),
        JoinRefusal.LAWS: lambda: eliminate(
            body_query(bodies.STAR),
            relations(body_query(bodies.STAR), bodies.star(2)),
            bodies.PRESENTATION,
            MatrixSemiring(COUNTING, 2),
            {},
            like=torch.zeros((2, 2), dtype=torch.float64),
        ),
        JoinRefusal.ORDER: lambda: list(
            generic_join(
                body_query(bodies.TRIANGLE),
                relations(body_query(bodies.TRIANGLE), bodies.hub(2)),
                bodies.PRESENTATION,
                order=("a", "b"),
            )
        ),
    }
    for code, attempt in attempts.items():
        try:
            attempt()
        except JoinRefused as refused:
            _require(refused.code is code, f"{code.value} refused as {refused.code}")
            continue
        raise Mismatch("refusal", math.inf, f"{code.value} was admitted")
    algebra = MatrixSemiring(COUNTING, 2)
    like = torch.zeros((2, 2), dtype=torch.float64)
    items = (
        bodies.items("r", [(0, 1)])
        + bodies.items("s", [(1, 2)])
        + bodies.items("t", [(0, 2)])
    )
    matrices = [
        torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 2.0], [1.0, 0.0]], dtype=torch.float64),
    ]
    weights = dict(zip(items, matrices, strict=True))
    query = body_query(bodies.TRIANGLES)
    rule = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    result = enumerate_aggregate(
        query,
        relations(query, items),
        bodies.PRESENTATION,
        algebra,
        weights,
        like=like,
        rule_weight=rule,
    )
    (value,) = result.values.values()
    expected = matrices[0] @ matrices[1] @ matrices[2] @ rule
    deviation = float((value - expected).abs().max())
    if deviation:
        raise Mismatch("value", deviation, f"the product is {value.tolist()}")


def _join_work(items: Sequence[Item]) -> float:
    """The work of listing the triangles by Generic Join.

    Parameters
    ----------
    items : Sequence of Item
        The input.

    Returns
    -------
    float
        Values iterated, probes, and bindings extended.
    """
    query = body_query(bodies.TRIANGLE)
    counter = JoinCounter()
    for _ in generic_join(
        query, relations(query, items), bodies.PRESENTATION, work=counter
    ):
        pass
    work = counter.report()
    return float(work.iterated + work.probes + work.tuples)


def _chart_work(items: Sequence[Item], join: Literal["generic", "keyed"]) -> float:
    """The work of listing the triangles on an indexed chart as items arrive.

    Each item is admitted in turn and completes the instances in which
    it is the last premise admitted, as a plan's passes find them.

    Parameters
    ----------
    items : Sequence of Item
        The input.
    join : {"generic", "keyed"}
        Generic Join from each trigger, or premises looked up one at a time.

    Returns
    -------
    float
        Lookups and candidates.
    """
    chart = IndexedChart(bodies.PRESENTATION, join=join)
    admitted: set[Item] = set()
    rule = bodies.TRIANGLE
    for item in items:
        admitted.add(item)
        chart.add(item)
        for position, premise in enumerate(rule.premises):
            if premise.judgment.family == item.family:
                for _ in chart.instances(rule, position, item, admitted.__contains__):
                    pass
    work = chart.counters()
    return float(work.lookups + work.candidates)


def _agm(items: Sequence[Item]) -> float:
    """The AGM bound of the triangle on an input.

    Parameters
    ----------
    items : Sequence of Item
        The input.

    Returns
    -------
    float
        The bound.
    """
    query = body_query(bodies.TRIANGLE)
    tables = relations(query, items)
    return agm_bound(
        query.variables,
        [atom.variables for atom in query.atoms],
        [len(table.tuples) for table in tables],
    ).value


_SIZES = (8, 16, 32, 64, 128)


def check_triangle_work() -> None:
    """Require Generic Join within the AGM bound, alone and on the chart, and keyed lookups above it.

    On the hub and the skewed triangle, Generic Join over the whole body,
    and the chart that runs it from each arriving item, stay within a
    constant of ``N^{3/2}`` (their controls of the next order fail). The
    chart that looks premises up one at a time does ``N^2`` work, which
    the same bound rejects.

    Raises
    ------
    Mismatch
        If Generic Join breaks the bound, or the keyed chart does not.
    """
    for name, build in (("hub", bodies.hub), ("skew", bodies.skew)):
        for label, work in (
            ("generic join", _join_work),
            ("the chart's generic join", lambda items: _chart_work(items, "generic")),
        ):
            scaling.check_family(
                scaling.Family(
                    name=f"{label} on the {name} triangle",
                    executor="joins",
                    question="Q5",
                    varies="tuples",
                    part="forward",
                    sizes=_SIZES,
                    work=lambda size, build=build, work=work: work(build(size)),
                    claim=lambda size, build=build: _agm(build(size)),
                    constant=16.0,
                )
            )
        keyed = [_chart_work(build(size), "keyed") for size in _SIZES]
        claim = [_agm(build(size)) for size in _SIZES]
        try:
            scaling.accept(_SIZES, keyed, claim, 16.0, f"keyed chart on {name}")
        except Mismatch:
            continue
        raise Mismatch(
            "support", math.inf, f"the keyed chart meets the AGM bound on {name}"
        )


def _elimination_work(size: int) -> float:
    """The work of the star's aggregate by elimination.

    Parameters
    ----------
    size : int
        The degree of the hub.

    Returns
    -------
    float
        Values iterated, probes, bindings, products and sums.
    """
    items = bodies.star(size)
    query = body_query(bodies.STAR)
    result = eliminate(
        query, relations(query, items), bodies.PRESENTATION, REAL, {}, like=LIKE
    )
    work = result.work
    return float(work.iterated + work.probes + work.tuples + work.products + work.sums)


def _enumeration_work(size: int) -> float:
    """The work of the star's aggregate by listing its instances.

    Parameters
    ----------
    size : int
        The degree of the hub.

    Returns
    -------
    float
        Values iterated, probes, bindings, products and sums.
    """
    items = bodies.star(size)
    query = body_query(bodies.STAR)
    result = enumerate_aggregate(
        query, relations(query, items), bodies.PRESENTATION, REAL, {}, like=LIKE
    )
    work = result.work
    return float(work.iterated + work.probes + work.tuples + work.products + work.sums)


def check_star_work() -> None:
    """Require the star's aggregate in linear work by elimination, and cubic by listing.

    Raises
    ------
    Mismatch
        If elimination is not linear in the tuples, its recorded width is
        not one, or listing the instances meets the linear bound.
    """
    sizes = (4, 8, 16, 32, 64)
    family = scaling.Family(
        name="elimination on the star",
        executor="joins",
        question="Q6",
        varies="tuples",
        part="forward",
        sizes=sizes,
        work=_elimination_work,
        claim=lambda size: float(len(bodies.star(size))),
        constant=16.0,
    )
    scaling.check_family(family)
    query = body_query(bodies.STAR)
    result = eliminate(
        query,
        relations(query, bodies.star(4)),
        bodies.PRESENTATION,
        REAL,
        {},
        like=LIKE,
    )
    _require(result.width == 1, f"the star's width is {result.width}")
    _require(
        result.fractional_width.fraction == 1,
        f"the star's fractional width is {result.fractional_width}",
    )
    listed = [_enumeration_work(size) for size in sizes]
    claim = [float(len(bodies.star(size))) for size in sizes]
    try:
        scaling.accept(sizes, listed, claim, 16.0, "listing the star")
    except Mismatch:
        return
    raise Mismatch("support", math.inf, "listing the star's instances is linear")


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="joins", question=question, run=run)
    for name, question, run in (
        ("joins/instances", "Q5", check_instances),
        ("joins/agm", "Q5", check_agm),
        ("joins/aggregates", "Q6", check_aggregates),
        ("joins/empty-factor", "Q6", check_empty_factor),
        ("joins/refusals", "Q6", check_refusals_and_order),
        ("joins/triangle-work", "Q5", check_triangle_work),
        ("joins/star-work", "Q6", check_star_work),
    )
}
"""Every check of joins by name."""

JOINS = "quivers.stochastic.joins"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="join-last-value-omitted",
        defect_class="omitted alternative",
        provenance="the last value of an intersection is not tried",
        target=f"{JOINS}:_descend",
        rewrites=(
            Rewrite(
                old="for value in list(sets[smallest]):",
                new="for value in list(sets[smallest])[:-1]:",
            ),
        ),
        rejected_by=("joins/instances",),
        min_effect=1.0,
    ),
    Mutation(
        name="join-first-value-omitted",
        defect_class="omitted alternative",
        provenance="the first value of an intersection is not tried",
        target=f"{JOINS}:_descend",
        rewrites=(
            Rewrite(
                old="for value in list(sets[smallest]):",
                new="for value in list(sets[smallest])[1:]:",
            ),
        ),
        rejected_by=("joins/instances",),
        min_effect=1.0,
    ),
    Mutation(
        name="elimination-last-value-omitted",
        defect_class="omitted alternative",
        provenance="the last value of a factor join is not summed",
        target=f"{JOINS}:_descend_factors",
        rewrites=(
            Rewrite(
                old="for value in list(sets[smallest]):",
                new="for value in list(sets[smallest])[:-1]:",
            ),
        ),
        rejected_by=("joins/aggregates",),
        min_effect=1.0,
    ),
    Mutation(
        name="elimination-empty-factor-as-unit",
        defect_class="omitted alternative",
        provenance="an empty factor of no variables is skipped as if it were the unit",
        target=f"{JOINS}:_join_factors",
        rewrites=(
            Rewrite(
                old="if () not in factor.table:\n                return",
                new="if () not in factor.table:\n                continue",
            ),
        ),
        rejected_by=("joins/empty-factor",),
        min_effect=1.0,
    ),
)
"""Every wrong version of joins and the checks that reject it."""
