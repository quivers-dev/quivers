"""Checks and deliberately wrong versions of the work an agenda run reports.

Each measured counter of a run's report is compared with an independent
count: pops with an agenda that counts its own, products with a
semiring that counts its own (the chart's and the queue's sums included),
the tape with a hook on the autograd graph, and provenance with the
recorded applications. A run with a rule weight
function does not claim its products, one without the tape measured
does not claim it, and a chart or queue that reports no work leaves its
counters unmeasured. A priority agenda's comparisons are compared with a
fresh heap replaying its pushes and pops. The agenda's own work on chains, diamonds and skip
graphs is recorded by the ledger entry ``agenda-work-quadratic``.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Sequence

import torch

from quivers.qiec.contracts import WorkField
from quivers.stochastic.agenda import (
    COMPONENT_FIELDS,
    AgendaResult,
    Chart,
    FIFOAgenda,
    InferenceRule,
    Item,
    LIFOAgenda,
    Pattern,
    PriorityQueueAgenda,
    make_wildcard,
    match,
    run_agenda,
)
from quivers.stochastic.semiring import LOG_PROB, ChartSemiring
from tests.evidence import graphs, scaling
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Mismatch
from tests.evidence.counters import forwarding, saved_elements
from tests.evidence.mutations import Mutation, Rewrite

X, Y = make_wildcard("x"), make_wildcard("y")

STEP = InferenceRule(
    name="step", premises=(("reach", X), ("edge", X, Y)), conclusion=("reach", Y)
)
"""``reach(y) <- reach(x), edge(x, y)``."""


def _weighted(name: str) -> InferenceRule:
    """The step rule with a weight function that multiplies by itself.

    Parameters
    ----------
    name : str
        The rule's name.

    Returns
    -------
    InferenceRule
        The rule.
    """
    return STEP.with_(
        name=name,
        weight_fn=lambda bindings, weights, semiring: semiring.times(
            weights[0], weights[1]
        ),
    )


class _CountingAgenda(FIFOAgenda):
    """A first-in-first-out agenda that counts its pops."""

    def __init__(self) -> None:
        super().__init__()
        self.pops = 0

    def pop(self) -> tuple[Item, torch.Tensor]:
        """Count and forward a pop.

        Returns
        -------
        tuple of Item and torch.Tensor
            The item and its pending update.
        """
        self.pops += 1
        return super().pop()


class _PlainChart(Chart):
    """A chart that keeps its items in a dictionary and reports no work."""

    def __init__(self) -> None:
        self._store: dict[Item, torch.Tensor] = {}

    def lookup(self, pattern: Pattern) -> Iterable[tuple[Item, torch.Tensor]]:
        """The stored items that match a pattern.

        Parameters
        ----------
        pattern : Pattern
            The pattern.

        Returns
        -------
        Iterable of tuple of Item and torch.Tensor
            The matches.
        """
        return [
            (item, weight)
            for item, weight in self._store.items()
            if match(pattern, item) is not None
        ]

    def insert_or_aggregate(
        self, item: Item, weight: torch.Tensor, semiring: ChartSemiring
    ) -> bool:
        """Aggregate a weight into an item.

        Parameters
        ----------
        item : Item
            The item.
        weight : torch.Tensor
            The weight.
        semiring : ChartSemiring
            The semiring.

        Returns
        -------
        bool
            Whether the stored weight changed.
        """
        existing = self._store.get(item)
        if existing is None:
            self._store[item] = weight
            return True
        merged = semiring.plus(torch.stack([existing, weight]), dim=0)
        self._store[item] = merged
        return bool((merged != existing).any())

    def get(self, item: Item) -> torch.Tensor | None:
        """The stored weight of an item.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        torch.Tensor or None
            Its weight, if stored.
        """
        return self._store.get(item)

    def items(self) -> Iterable[tuple[Item, torch.Tensor]]:
        """Every stored item and weight.

        Returns
        -------
        Iterable of tuple of Item and torch.Tensor
            The items.
        """
        return self._store.items()


class _LoggingHeap(PriorityQueueAgenda):
    """A priority agenda that logs the priorities it pushes and its pops."""

    def __init__(self) -> None:
        super().__init__(lambda item, weight: float(weight))
        self.log: list[float | None] = []

    def push(self, item: Item, weight: torch.Tensor) -> None:
        """Log and forward a push.

        Parameters
        ----------
        item : Item
            The item.
        weight : torch.Tensor
            Its weight.
        """
        self.log.append(-float(weight))
        super().push(item, weight)

    def pop(self) -> tuple[Item, torch.Tensor]:
        """Log and forward a pop.

        Returns
        -------
        tuple of Item and torch.Tensor
            The next item and its pending weight.
        """
        self.log.append(None)
        return super().pop()


class _CountedKey:
    """A heap key ordered by priority, then push order, counting comparisons."""

    count = 0

    def __init__(self, priority: float, order: int) -> None:
        self.priority = priority
        self.order = order

    def __lt__(self, other: _CountedKey) -> bool:
        """Count one comparison and order the keys.

        Parameters
        ----------
        other : _CountedKey
            Another key.

        Returns
        -------
        bool
            Whether this key comes first.
        """
        _CountedKey.count += 1
        return (self.priority, self.order) < (other.priority, other.order)


def _replayed(log: Sequence[float | None]) -> int:
    """The comparisons a fresh heap makes for the same pushes and pops.

    Parameters
    ----------
    log : Sequence of float or None
        Each push's key, or ``None`` for a pop, in order.

    Returns
    -------
    int
        The comparisons counted by the keys.
    """
    _CountedKey.count = 0
    heap: list[_CountedKey] = []
    for order, entry in enumerate(log, start=1):
        if entry is None:
            heapq.heappop(heap)
        else:
            heapq.heappush(heap, _CountedKey(entry, order))
    return _CountedKey.count


def _axioms(
    edges: Sequence[tuple[int, int]], *, grad: bool = False
) -> list[tuple[Item, torch.Tensor]]:
    """The start and the edges as axioms with log weights.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The edges.
    grad : bool
        Whether the weights require gradients.

    Returns
    -------
    list of tuple of Item and torch.Tensor
        The axioms.
    """
    found = [(("reach", 0), torch.tensor(0.0, requires_grad=grad))]
    found += [
        (("edge", a, b), torch.tensor(-0.5, requires_grad=grad)) for a, b in edges
    ]
    return found


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


def _equal(found: int, count: int, what: str) -> None:
    """Require a reported counter to equal its independent count.

    Parameters
    ----------
    found : int
        The reported counter.
    count : int
        The independent count.
    what : str
        The counter.

    Raises
    ------
    Mismatch
        If they differ.
    """
    if found != count:
        raise Mismatch(
            "value", float(abs(found - count)) or 1.0, f"{what}: {found} for {count}"
        )


def check_counts() -> None:
    """Require every measured counter of an agenda run to be an independent count.

    Raises
    ------
    Mismatch
        If a measured counter differs from its independent count, or a
        counter is claimed that the run cannot count.
    """
    edges = graphs.diamonds(3)
    semiring, counted = forwarding(LOG_PROB)
    agenda = _CountingAgenda()
    result: AgendaResult = run_agenda(
        _axioms(edges, grad=True),
        [STEP],
        semiring=semiring,
        agenda=agenda,
        measure_tape=True,
    )
    report = result.work
    # Hooks on saved tensors nest with the innermost winning, so the
    # independent count of the tape is taken on a second, identical run.
    with saved_elements() as sizes:
        run_agenda(_axioms(edges, grad=True), [STEP], semiring=LOG_PROB)
    _equal(report.iterations, agenda.pops, "pops")
    _equal(report.scalar_products, counted["products"], "products")
    _equal(report.elements_saved_for_backward, sum(sizes), "saved elements")
    _equal(
        report.provenance_records,
        sum(1 + len(premises) for _, _, premises, _ in result.applications)
        + len(result.axiom_occurrences),
        "provenance",
    )
    _require(sum(sizes) > 0, "the run saved nothing")
    _equal(report.scalar_sums, counted["sums"], "sums")
    _require(counted["sums"] > 0, "the run summed nothing")
    stacked, stack_counted = forwarding(LOG_PROB)
    lifo = run_agenda(
        _axioms(edges), [STEP], semiring=stacked, agenda=LIFOAgenda()
    ).work
    _equal(lifo.scalar_sums, stack_counted["sums"], "sums in the chart")
    _require(stack_counted["sums"] > 0, "the chart aggregated nothing")
    plain_chart = run_agenda(
        _axioms(edges), [STEP], semiring=LOG_PROB, chart=_PlainChart()
    ).work
    for field in (
        WorkField.SCALAR_SUMS,
        WorkField.HASH_OPERATIONS,
        WorkField.KEY_COMPARISONS,
    ):
        _require(
            field not in plain_chart.measured,
            f"{field.value} is claimed with a chart that reports no work",
        )
    logged = _LoggingHeap()
    heap = run_agenda(_axioms(edges), [STEP], semiring=LOG_PROB, agenda=logged).work
    for field in COMPONENT_FIELDS:
        _require(field in heap.measured, f"a heap run does not measure {field.value}")
    queue = logged.work_counts()[WorkField.KEY_COMPARISONS]
    _equal(queue, _replayed(logged.log), "heap comparisons")
    _require(queue > 0, "the heap compared nothing")
    plain = run_agenda(_axioms(edges), [STEP], semiring=LOG_PROB).work
    _require(
        WorkField.ELEMENTS_SAVED_FOR_BACKWARD not in plain.measured,
        "the tape is claimed without being measured",
    )
    weighted = run_agenda(_axioms(edges), [_weighted("step")], semiring=LOG_PROB).work
    _require(
        WorkField.SCALAR_PRODUCTS not in weighted.measured,
        "products are claimed under a weight function",
    )


def _work(edges: Sequence[tuple[int, int]]) -> float:
    """The work of an agenda run on a graph: pops, hashing and comparisons.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The graph.

    Returns
    -------
    float
        The work.
    """
    report = run_agenda(_axioms(edges), [STEP], semiring=LOG_PROB).work
    return float(report.iterations + report.hash_operations + report.key_comparisons)


LINEAR_SIZES = (8, 16, 32, 64, 128)
"""The sizes of the families."""

LINEAR_CONSTANT = 16.0
"""The constant of the linear bound on the agenda's own work per edge."""

WITHIN = f"within {LINEAR_CONSTANT:g} per edge"
"""The fact of a family whose work meets the linear bound."""


def _families() -> tuple[tuple[str, Callable[[int], list[tuple[int, int]]]], ...]:
    """The chain, diamond and skip families.

    Returns
    -------
    tuple of tuple of str and Callable
        Each family's name and its graph at a size.
    """
    return (
        ("chains", graphs.chain),
        ("diamonds", lambda size: graphs.diamonds(size // 4)),
        ("skip graphs", lambda size: graphs.skips(size // 2)),
    )


def linear_facts() -> dict[str, str]:
    """Whether the agenda's own work per edge meets the linear bound, per family.

    Returns
    -------
    dict of str to str
        For each family, `WITHIN` when the linear bound accepts its work,
        else its work per edge at each size, to two decimals.
    """
    facts: dict[str, str] = {}
    for name, build in _families():
        work = [_work(build(size)) for size in LINEAR_SIZES]
        edges = [float(len(build(size)) + 1) for size in LINEAR_SIZES]
        try:
            scaling.accept(LINEAR_SIZES, work, edges, LINEAR_CONSTANT, name)
        except Mismatch:
            facts[f"{name}, work per edge"] = " ".join(
                f"{count / bound:.2f}" for count, bound in zip(work, edges, strict=True)
            )
            continue
        facts[f"{name}, work per edge"] = WITHIN
    return facts


def check_linear() -> None:
    """Require the agenda's own work linear on chains, diamonds and skip graphs.

    The agenda looks up a premise's form by its pattern alone, so this
    fails today; the ledger entry ``agenda-work-quadratic`` records the
    measured work, and removing that entry with its repair registers
    this check.

    Raises
    ------
    Mismatch
        If a family breaks the linear bound, or its control does not.
    """
    for name, build in _families():
        scaling.check_family(
            scaling.Family(
                name=f"agenda on {name}",
                executor="agenda",
                question="Q1",
                varies="edges",
                part="forward",
                sizes=LINEAR_SIZES,
                work=lambda size, build=build: _work(build(size)),
                claim=lambda size, build=build: float(len(build(size)) + 1),
                constant=LINEAR_CONSTANT,
            )
        )


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="agenda", question="Q1", run=run)
    for name, run in (("agenda/work-counts", check_counts),)
}
"""Every check of agenda work reports by name."""

AGENDA = "quivers.stochastic.agenda"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="agenda-products-counted-twice",
        defect_class="evidence",
        provenance="each product of a firing is counted twice",
        target=f"{AGENDA}:_conclusion_weight",
        rewrites=(
            Rewrite(
                old="work.products += acc.numel()",
                new="work.products += 2 * acc.numel()",
            ),
        ),
        rejected_by=("agenda/work-counts",),
        min_effect=1.0,
    ),
    Mutation(
        name="agenda-chart-sums-under-reported",
        defect_class="evidence",
        provenance="the chart does not count the sums of its aggregation",
        target=f"{AGENDA}:HashChart.insert_or_aggregate",
        rewrites=(
            Rewrite(
                old="self._work.sums += merged.numel()",
                new="self._work.sums += 0",
            ),
        ),
        rejected_by=("agenda/work-counts",),
        min_effect=1.0,
    ),
    Mutation(
        name="agenda-heap-comparisons-not-counted",
        defect_class="evidence",
        provenance="a priority agenda's heap entries do not count their comparisons",
        target=f"{AGENDA}:_HeapEntry.__lt__",
        rewrites=(
            Rewrite(old="self.work.comparisons += 1", new="self.work.comparisons += 0"),
        ),
        rejected_by=("agenda/work-counts",),
        min_effect=1.0,
    ),
    Mutation(
        name="agenda-pops-not-counted",
        defect_class="evidence",
        provenance="pops are not counted",
        target=f"{AGENDA}:_drain",
        rewrites=(Rewrite(old="work.pops += 1", new="work.pops += 0"),),
        rejected_by=("agenda/work-counts",),
        min_effect=1.0,
    ),
)
"""Every wrong version of agenda work reports and the checks that reject it."""
