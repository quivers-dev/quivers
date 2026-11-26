"""Checks and deliberately wrong versions of the work an agenda run reports.

Each measured counter of a run's report is compared with an independent
count: pops with an agenda that counts its own, products with a
semiring that counts its own (the chart's and the queue's sums included),
the tape with a hook on the autograd graph, and provenance with the
recorded applications. A run with a rule weight
function does not claim its products, one without the tape measured
does not claim it, and a chart or queue that reports no work leaves its
counters unmeasured. A priority agenda's comparisons are compared with a
fresh heap replaying its pushes and pops. A run's work is linear on
chains, diamonds and skip graphs.
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
    Bindings,
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
from quivers.stochastic.semiring import LOG_PROB, VITERBI, ChartSemiring
from tests.evidence import graphs, scaling
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Mismatch
from tests.evidence.checks.deduction import AGENDAS
from tests.evidence.counters import CallCounter, forwarding, saved_elements
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

    def lookup(
        self, pattern: Pattern, bindings: Bindings | None = None
    ) -> Iterable[tuple[Item, torch.Tensor]]:
        """The stored items that match a pattern under bindings.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        bindings : Bindings or None
            Values of some of its wildcards.

        Returns
        -------
        Iterable of tuple of Item and torch.Tensor
            The matches.
        """
        return [
            (item, weight)
            for item, weight in self._store.items()
            if match(pattern, item, bindings) is not None
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


def _families() -> tuple[
    tuple[str, Callable[[int], list[tuple[int, int]]], Callable[[int], int]], ...
]:
    """The chain, diamond and skip families, each with its exact work in edges.

    Returns
    -------
    tuple of tuple of str, Callable and Callable
        Each family's name, its graph at a size, and its work as a
        function of its number of edges (see `check_linear`).
    """
    return (
        ("chains", graphs.chain, lambda edges: 23 * edges + 9),
        (
            "diamonds",
            lambda size: graphs.diamonds(size // 4),
            lambda edges: 21 * edges + 8,
        ),
        (
            "skip graphs",
            lambda size: graphs.skips(size // 2),
            lambda edges: 19 * edges + 12,
        ),
    )


def check_linear() -> None:
    """Require a run's work to be exactly its derived count on chains, diamonds and skips.

    A run of ``reach(y) <- reach(x), edge(x, y)`` with a first-in-first-out
    queue, a hash chart and ``P`` pops, ``A`` applications, ``E`` edges,
    ``V`` reached nodes and ``U = E + 1 + A`` pushes counts:

    - ``P`` pops;
    - in the chart, ``3P + A`` hash operations: per pop one lookup and
      two accesses inserting a new item, and per application one lookup
      of the premise the popped item does not fill;
    - in the queue, ``2U + P`` hash operations: per push a lookup and a
      store of the pending entry, and per pop the removal;
    - in the agenda's tries, ``V + 2E + P`` hash operations: each reached
      node enters the trie of ``reach`` under one key and each edge the
      trie of ``edge`` under two, and each pop descends one key;
    - ``P + A - d_0`` comparisons: each pop is matched against the one
      premise of its form, and each application found when its ``reach``
      premise arrives costs one value of the intersection, while the
      ``d_0`` edges from node 0 are found when they arrive, after
      ``reach(0)``, with no variable left to intersect.

    On these families every item is popped once, as its contributions all
    arrive while it is pending (a node's predecessors enter the queue
    before it), and every edge is reached, so ``P = E + V`` and
    ``A = E``, and the total is ``15 E + 8 V + 2 - d_0``: ``23 E + 9`` on
    chains (``V = E + 1``, ``d_0 = 1``), ``21 E + 8`` on diamonds
    (``V = 3E/4 + 1``, ``d_0 = 2``) and ``19 E + 12`` on skip graphs
    (``E = 2L - 1``, ``V = L + 1``, ``d_0 = 2``). The bound is that count
    with constant one, so one operation more per edge fails; its ratio
    must not grow and the control of the next order has to fail.

    Raises
    ------
    Mismatch
        If a family's work differs from its count, or its control does not.
    """
    for name, build, count in _families():
        scaling.check_family(
            scaling.Family(
                name=f"agenda on {name}",
                executor="agenda",
                question="Q1",
                varies="edges",
                part="forward",
                sizes=LINEAR_SIZES,
                work=lambda size, build=build: _work(build(size)),
                claim=lambda size, build=build, count=count: float(
                    count(len(build(size)))
                ),
                constant=1.0,
            )
        )


def _items_popped(edges: Sequence[tuple[int, int]]) -> int:
    """The distinct items of a reachability run: the edges and the reached nodes.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The graph, every node reachable from 0.

    Returns
    -------
    int
        ``E + V``.
    """
    return len(edges) + len({0} | {target for _, target in edges})


def check_dispatch() -> None:
    """Require each item to be matched only against the premise of its form.

    Along a line of ``n`` edges, ``2n + 1`` items arrive, each filling one
    premise of the one rule, so the matches are exactly ``2n + 1``. Fewer
    would leave an item untested against a premise it fills.

    Raises
    ------
    Mismatch
        If the matches differ from the items.
    """
    for length in (20, 40, 80):
        edges = graphs.chain(length)
        with CallCounter({"match": match}) as calls:
            run_agenda(_axioms(edges), [STEP], semiring=LOG_PROB)
        _equal(calls.counts["match"], _items_popped(edges), f"matches at {length}")


X3, Y3, Z3 = make_wildcard("x"), make_wildcard("y"), make_wildcard("z")

TRIANGLE = InferenceRule(
    name="triangle",
    premises=(("R", X3, Y3), ("S", Y3, Z3), ("T", Z3, X3)),
    conclusion=("triangle",),
)
"""A triangle over three relations."""


def _hub(size: int) -> list[tuple[Item, torch.Tensor]]:
    """Three relations in which ``0`` is joined to ``size - 1`` others each way.

    Parameters
    ----------
    size : int
        The degree of the hub.

    Returns
    -------
    list of tuple of Item and torch.Tensor
        The axioms.
    """
    found: list[tuple[Item, torch.Tensor]] = []
    for relation in "RST":
        found += [((relation, 0, k), torch.tensor(0.0)) for k in range(size)]
        found += [((relation, k, 0), torch.tensor(0.0)) for k in range(1, size)]
    return found


def _hub_work(size: int, discipline: str) -> float:
    """The comparisons of a triangle run over a hub.

    Parameters
    ----------
    size : int
        The degree of the hub.
    discipline : str
        The agenda.

    Returns
    -------
    float
        Matches and values of the intersection.
    """
    report = run_agenda(
        _hub(size), [TRIANGLE], semiring=VITERBI, agenda=AGENDAS[discipline]()
    ).work
    return float(report.key_comparisons)


def _hub_bound(size: int) -> float:
    """The bound ``m N + m^2 n AGM`` of a triangle with ``m = n = 3`` over a hub.

    Parameters
    ----------
    size : int
        The degree of the hub.

    Returns
    -------
    float
        ``3 N + 27 N^{3/2}`` for relations of ``N = 2 size - 1`` tuples.
    """
    tuples = 2 * size - 1
    return 3 * tuples + 27 * tuples**1.5


def _hub_check(discipline: str) -> Callable[[], None]:
    """The check of a triangle over a hub under one agenda.

    Parameters
    ----------
    discipline : str
        The agenda.

    Returns
    -------
    Callable[[], None]
        The check.
    """

    def check() -> None:
        """Require a triangle over a hub within its AGM bound.

        Generic Join from each arriving tuple does at most ``m N + m^2 n
        AGM`` comparisons over a run, ``N`` the tuples of a relation; a
        join one premise at a time does ``N^2`` on a hub, which the same
        bound rejects through the control of the next order.

        Raises
        ------
        Mismatch
            If the work breaks the bound, or its control does not.
        """
        scaling.check_family(
            scaling.Family(
                name=f"triangle over a hub, {discipline}",
                executor=f"agenda-{discipline}",
                question="Q6",
                varies="hub degree",
                part="forward",
                sizes=LINEAR_SIZES,
                work=lambda size: _hub_work(size, discipline),
                claim=_hub_bound,
                constant=1.0,
            )
        )

    return check


CHECKS: dict[str, Check] = {
    **{
        name: Check(name=name, executor="agenda", question="Q1", run=run)
        for name, run in (
            ("agenda/work-counts", check_counts),
            ("agenda/work-linear", check_linear),
            ("agenda/dispatch", check_dispatch),
        )
    },
    **{
        f"agenda/hub/{discipline}": Check(
            name=f"agenda/hub/{discipline}",
            executor=f"agenda-{discipline}",
            question="Q6",
            run=_hub_check(discipline),
        )
        for discipline in AGENDAS
    },
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
        name="agenda-dispatch-to-every-premise",
        defect_class="graph work",
        provenance="an arriving item is matched against every premise of every rule",
        target=f"{AGENDA}:_RuleJoins.matches",
        rewrites=(
            Rewrite(
                old="candidates = [*self._by_form.get(form, ()), *self._any] if form else self._any",
                new=(
                    "candidates = [(r, p) for r, rule in enumerate(self.rules) "
                    "for p in range(len(rule.premises))]"
                ),
            ),
        ),
        rejected_by=("agenda/dispatch",),
        min_effect=1.0,
    ),
    Mutation(
        name="intersection-iterates-the-first-set",
        defect_class="graph work",
        provenance="the intersection iterates the first premise's values, not the fewest",
        target="quivers.stochastic.tries:intersect",
        also_in=(
            AGENDA,
            "quivers.stochastic.joins",
            "quivers.stochastic.indexed",
        ),
        rewrites=(
            Rewrite(
                old="smallest = min(range(len(sets)), key=lambda index: len(sets[index]))",
                new="smallest = 0",
            ),
        ),
        rejected_by=("agenda/hub/fifo",),
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
