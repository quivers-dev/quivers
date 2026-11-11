"""Agenda-based weighted-deduction engine.

A general framework for evaluating weighted deductive systems by
agenda-driven semi-naïve enumeration. Subsumes CKY, Earley,
Viterbi, A* parsing, Knuth's algorithm, semi-naïve Datalog
evaluation, and bidirectional MLTT type-checking under one
runtime, parameterized by an `Item` algebra, a list of
arity-n `InferenceRule` hyperedges, a `ChartSemiring`,
an `Agenda` data structure, a priority function, and a
goal predicate.

The chart maps each item to its accumulated weight. A rule combines premise
weights and contributes to its conclusion. On a complete lattice with a
monotone rule operator, this has a least fixed-point interpretation. A finite
acyclic system can be evaluated in dependency order; cyclic systems need
additional convergence or termination assumptions.

The agenda chooses which pending deduction to examine next. Idempotent
addition makes repeated identical contributions harmless, but does not alone
prove termination or strategy independence. Correct scheduling must reach the
required deductions and respect the selected aggregation and stopping rule.

References
----------
Parsing as deduction is developed in [Pereira and Warren (1983)](<https://aclanthology.org/P83-1021/>); [Shieber et al. (1995)](<https://www.sciencedirect.com/science/article/pii/074310669500035I>). Weighted interpretations are developed in
[Goodman (1999)](<https://aclanthology.org/J99-4004/>), and the hypergraph perspective in
[Klein and Manning (2001)](<https://aclanthology.org/W01-1812/>). Priority methods are treated in
[Knuth (1977)](<https://www.sciencedirect.com/science/article/abs/pii/0020019077900023>); [Nederhof (2003)](<https://aclanthology.org/J03-1006/>); [Klein and Manning (2003)](<https://aclanthology.org/N03-1016/>). For
weighted dynamic programs, complexity analysis, and program transformations,
see [Eisner et al. (2005)](<https://aclanthology.org/H05-1036/>); [McAllester (2002)](<https://doi.org/10.1145/581771.581774>); [Eisner and Blatz (2007)](<https://web.stanford.edu/group/cslipublications/cslipublications/FG/2006/FG.pdf>).
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from quivers.stochastic.semiring import ChartSemiring, LOG_PROB
from quivers.structural.encoder import Encoder


# ---------------------------------------------------------------------------
# Items, patterns, and rule structure
# ---------------------------------------------------------------------------


Item = tuple[Any, ...]
"""An item is a tuple of (constructor_name, *arguments).

The constructor name disambiguates item shapes; the remaining
slots are the item's payload. Arguments may be other items
(structural), strings (atoms), integers (positions / cardinals),
or `Wildcard` placeholders in patterns.
"""


@dataclass(frozen=True)
class Wildcard:
    """Pattern-position wildcard.

    A wildcard matches any value in the corresponding slot and
    binds it to ``name`` for use in the conclusion. Two wildcards
    with the same ``name`` in a single rule's premise list must
    bind to equal values (variable-sharing).
    """

    name: str

    def __repr__(self) -> str:
        return f"_{self.name}"


def make_wildcard(name: str) -> Wildcard:
    """Build a fresh wildcard with the given variable name.

    Parameters
    ----------
    name : str
        The pattern variable the wildcard binds.

    Returns
    -------
    Wildcard
        The wildcard.
    """
    return Wildcard(name)


Pattern = tuple[Any, ...]
"""A pattern is an item-shaped tuple that may contain `Wildcard`
positions. A pattern matches an item if (a) the constructor name
matches, (b) the arities match, (c) each non-wildcard slot is
structurally equal to the corresponding item slot, and (d) all
wildcards with the same name bind to equal values."""


Bindings = dict[str, Any]
"""A bindings environment maps wildcard variable names to the
values they captured during pattern-matching."""


@dataclass(frozen=True)
class InferenceRule:
    """A weighted inference rule, that is, an arity-n hyperedge.

    Parameters
    ----------
    name : str
        Rule identifier (used for diagnostics and provenance).
    premises : tuple of Pattern
        Patterns the rule's antecedents must match.
    conclusion : Pattern
        Pattern of the item the rule produces. Wildcards must
        appear in the premises (so they are bound by the time
        the conclusion is constructed).
    weight_fn : Callable, optional
        Function ``(bindings, premise_weights, semiring) -> Weight``
        producing the conclusion's weight from the matched
        premises. Defaults to the semiring product of premise
        weights (the standard semiring-parsing aggregation).
    side_condition : Callable, optional
        Predicate ``bindings -> bool``. The rule fires only when
        this returns ``True``. Used for guards like adjacency
        ``i < k < j`` in CKY-style rules.
    """

    name: str
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    weight_fn: Callable[[Bindings, tuple, ChartSemiring], Any] | None = None
    side_condition: Callable[[Bindings], bool] | None = None


def instantiate(pattern: Pattern, bindings: Bindings) -> Item:
    """Substitute wildcards in a pattern with their bound values.

    The pattern may be a bare `Wildcard` (treated as "the
    entire item is the wildcard"), a structural tuple, or a leaf
    value (returned unchanged). Recursion runs over tuple
    children.

    Parameters
    ----------
    pattern : Pattern
        The pattern to instantiate.
    bindings : Bindings
        Values for the pattern's wildcards.

    Returns
    -------
    Item
        A concrete item with no wildcards.

    Raises
    ------
    KeyError
        If a wildcard in the pattern has no binding.
    """
    if isinstance(pattern, Wildcard):
        return bindings[pattern.name]
    if not isinstance(pattern, tuple):
        return pattern
    out: list[Any] = []
    for p in pattern:
        if isinstance(p, Wildcard):
            out.append(bindings[p.name])
        elif isinstance(p, tuple):
            out.append(instantiate(p, bindings))
        else:
            out.append(p)
    return tuple(out)


def match(
    pattern: Pattern, item: Item, bindings: Bindings | None = None
) -> Bindings | None:
    """Match an item against a pattern.

    Wildcards bind the values at their positions; a wildcard
    already bound in ``bindings`` (or earlier in the pattern) must
    see an equal value. Tuples match slot by slot when their
    lengths agree, and other values match by equality.

    Parameters
    ----------
    pattern : Pattern
        The pattern to match.
    item : Item
        The item to match against.
    bindings : Bindings or None
        Bindings to extend; ``None`` starts from no bindings.

    Returns
    -------
    Bindings or None
        The extended bindings, or ``None`` when the item does not
        match.
    """
    if bindings is None:
        bindings = {}
    if isinstance(pattern, Wildcard):
        existing = bindings.get(pattern.name)
        if existing is None:
            return {**bindings, pattern.name: item}
        if existing == item:
            return bindings
        return None
    if not isinstance(pattern, tuple) or not isinstance(item, tuple):
        if pattern == item:
            return bindings
        return None
    if len(pattern) != len(item):
        return None
    out = dict(bindings)
    for p, v in zip(pattern, item):
        if isinstance(p, Wildcard):
            existing = out.get(p.name)
            if existing is None:
                out[p.name] = v
            elif existing != v:
                return None
        elif isinstance(p, tuple) and isinstance(v, tuple):
            sub = match(p, v, out)
            if sub is None:
                return None
            out = sub
        elif p != v:
            return None
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


class Chart(ABC):
    """A K-valued presheaf on the item algebra.

    Categorically, a chart is the K-presheaf
    :math:`C : I^{\\mathrm{op}} \\to K` produced by the deduction
    system on an input. The chart is the least pre-fixed point of
    the rule-system functor; operationally, the agenda engine
    populates it bottom-up.

    Concrete subclasses choose a storage strategy: dense
    tensor-indexed (for CKY-style spans), sparse hash-indexed
    (for Datalog atoms), or term-structural (for MLTT
    judgments).
    """

    @abstractmethod
    def lookup(self, pattern: Pattern) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate (item, weight) pairs matching a pattern.

        Parameters
        ----------
        pattern : Pattern
            The pattern to match stored items against.

        Returns
        -------
        Iterable of (Item, torch.Tensor)
            The matching items with their weights.
        """
        ...

    @abstractmethod
    def insert_or_aggregate(
        self,
        item: Item,
        weight: torch.Tensor,
        semiring: ChartSemiring,
    ) -> bool:
        """Insert or aggregate an item's weight.

        Parameters
        ----------
        item : Item
            The item to store.
        weight : torch.Tensor
            The weight to insert, or to aggregate with the stored
            weight under ``semiring.plus``.
        semiring : ChartSemiring
            The semiring whose ``plus`` aggregates weights.

        Returns
        -------
        bool
            ``True`` if the item's stored weight changed (so it
            should be re-enqueued for downstream firings);
            ``False`` if the weight was unchanged (the inference was
            redundant under the semiring's idempotent join).
        """
        ...

    @abstractmethod
    def get(self, item: Item) -> torch.Tensor | None:
        """Return the chart's weight at an item, or ``None`` if absent.

        Parameters
        ----------
        item : Item
            The item to look up.

        Returns
        -------
        torch.Tensor or None
            The stored weight, or ``None`` if the item is absent.
        """
        ...

    @abstractmethod
    def items(self) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate all (item, weight) pairs in the chart.

        Returns
        -------
        Iterable of (Item, torch.Tensor)
            Every stored item with its weight.
        """
        ...


class HashChart(Chart):
    """Dictionary-backed chart for arbitrary item algebras.

    Stores items as keys in a Python dict; supports linear-scan
    pattern matching. Suitable for Datalog atoms, MLTT
    judgments, and any deduction whose items don't have a dense
    integer encoding. Falls back to O(|chart| · |pattern|) match
    cost per lookup; faster specializations should override
    `lookup` with an indexed structure.

    Weights are stored as `torch.Tensor` values; autograd
    flows through ``insert_or_aggregate``'s
    ``semiring.plus(torch.stack([…]))`` reduction, so the chart's
    final values carry gradients with respect to any
    ``requires_grad=True`` rule-weight parameters that fed into
    the agenda.

    Parameters
    ----------
    tolerance : float
        The smallest change in an item's weight that counts as an
        update; ``0.0`` treats any change as an update.
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        self._store: dict[Item, torch.Tensor] = {}
        self._tolerance: float = float(tolerance)

    def lookup(self, pattern: Pattern) -> Iterable[tuple[Item, torch.Tensor]]:
        """Scan the stored items for those matching a pattern.

        Parameters
        ----------
        pattern : Pattern
            The pattern to match stored items against.

        Returns
        -------
        Iterable of (Item, torch.Tensor)
            The matching items with their weights.
        """
        for item, w in self._store.items():
            b = match(pattern, item)
            if b is not None:
                yield item, w

    def insert_or_aggregate(
        self,
        item: Item,
        weight: torch.Tensor,
        semiring: ChartSemiring,
    ) -> bool:
        """Aggregate ``weight`` into the chart at ``item``.

        For non-idempotent semirings (LogProb, Counting, Inside)
        with cyclic rule graphs, every re-derivation of an item via
        a new path increases its weight via `semiring.plus`,
        even when the item itself is already in the chart. With an
        unconstrained cycle weight, this sequence is unbounded:
        either the cycle is contractive (cycle log-weight :math:`<
        0`), in which case the chart's weight converges to a
        finite Kleene-star limit; or the cycle is non-contractive
        (:math:`\\ge 0`), in which case the chart total is :math:`+
        \\infty` and the model is mathematically ill-posed.

        The chart makes this distinction observable by terminating
        the agenda when the per-item update falls below the
        tolerance. The default tolerance ``0.0`` treats any change
        as an update; positive tolerances expose convergent cyclic
        fixed points while still routing divergent systems through
        the agenda's ``max_iterations`` safety net.

        Parameters
        ----------
        item : Item
            The item to store.
        weight : torch.Tensor
            The weight to insert or aggregate.
        semiring : ChartSemiring
            The semiring whose ``plus`` aggregates weights.

        Returns
        -------
        bool
            ``True`` if the stored weight changed by more than the
            tolerance (or the item is new).
        """
        if item not in self._store:
            self._store[item] = weight
            return True
        existing = self._store[item]
        stacked = torch.stack([existing, weight])
        merged = semiring.plus(stacked, dim=0)
        if self._tolerance > 0.0:
            delta = (merged.detach() - existing.detach()).abs().max()
            changed = bool(float(delta) > self._tolerance)
        else:
            changed = not torch.equal(merged, existing)
        self._store[item] = merged
        return changed

    def get(self, item: Item) -> torch.Tensor | None:
        """Return the stored weight at an item.

        Parameters
        ----------
        item : Item
            The item to look up.

        Returns
        -------
        torch.Tensor or None
            The stored weight, or ``None`` if the item is absent.
        """
        return self._store.get(item)

    def items(self) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate all stored (item, weight) pairs.

        Returns
        -------
        Iterable of (Item, torch.Tensor)
            Every stored item with its weight.
        """
        return self._store.items()


# ---------------------------------------------------------------------------
# Chart as first-class differentiable value
# ---------------------------------------------------------------------------


class ChartView:
    """A first-class, differentiable view onto a chart.

    Wraps a `Chart` produced by an agenda run and exposes
    the user-facing presheaf-evaluation operations:

    * `weight`: query the aggregated weight at a single
      ground item; returns a differentiable `torch.Tensor`.
    * `enumerate`: enumerate items matching a pattern with
      their weights.
    * `derivations`: extract the derivation forest at an
      item (under the derivation semiring; for the basic
      Boolean / LOG_PROB / Viterbi semirings, returns the
      flat list of matched derivations as a placeholder).
    * `goal_weight`: return the weight at the goal items
      identified at run time.

    Categorically, the chart is the K-presheaf
    :math:`I^{\\mathrm{op}} \\to K` produced by the deduction
    system; ``ChartView``'s methods are presheaf evaluations.
    Gradients flow through these operations because the
    underlying tensors carry ``requires_grad`` from the rule
    weights.

    Parameters
    ----------
    result : AgendaResult
        The agenda run to view.
    """

    def __init__(self, result: "AgendaResult") -> None:
        self._result = result

    @property
    def chart(self) -> Chart:
        """The underlying K-presheaf."""
        return self._result.chart

    @property
    def attached_loss(self) -> torch.Tensor | None:
        """The sum of rule-attached and chart-attached loss values
        fired during this deduction's run, or ``None`` if no losses
        were declared at those sites."""
        return self._result.attached_loss

    @property
    def semiring(self) -> ChartSemiring:
        """The semiring the chart was computed over."""
        return self._result.semiring

    def weight(self, item: Item) -> torch.Tensor:
        """Return the aggregated weight at a concrete ``item``.

        Parameters
        ----------
        item : Item
            A ground item.

        Returns
        -------
        torch.Tensor
            The item's differentiable weight.

        Raises
        ------
        KeyError
            If the item was never derived.
        """
        w = self._result.chart.get(item)
        if w is None:
            raise KeyError(f"item {item!r} not in chart")
        return w

    def try_weight(
        self,
        item: Item,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the weight at ``item``, or a fallback if it is absent.

        Parameters
        ----------
        item : Item
            A ground item.
        default : torch.Tensor or None
            The value to return for an absent item; ``None`` means
            the semiring's zero.

        Returns
        -------
        torch.Tensor
            The item's weight, ``default``, or the semiring's zero.
        """
        w = self._result.chart.get(item)
        if w is not None:
            return w
        if default is not None:
            return default
        return torch.tensor(
            float(self.semiring.zero)
            if hasattr(self.semiring.zero, "__float__")
            else 0.0,
            dtype=torch.get_default_dtype(),
        )

    def enumerate(self, pattern: Pattern) -> list[tuple[Item, torch.Tensor]]:
        """Enumerate (item, weight) pairs matching ``pattern``.

        Wildcards in ``pattern`` (instances of `Wildcard`)
        match any slot value.

        Parameters
        ----------
        pattern : Pattern
            The pattern to match.

        Returns
        -------
        list of (Item, torch.Tensor)
            The matching items with their differentiable weights.
        """
        return list(self._result.chart.lookup(pattern))

    def embedding(self, item: Item) -> torch.Tensor:
        """Return the vector embedding of ``item`` under the deduction's
        attached encoder.

        Requires the originating `DeductionSystem` to carry an
        ``_item_encoder`` attribute (set by the DSL compiler when
        the deduction's body declares ``encoder C``). Items are
        converted to [`quivers.structural.Term`][quivers.structural.Term] form on the fly.

        Parameters
        ----------
        item : Item
            The item to embed.

        Returns
        -------
        torch.Tensor
            The encoder's embedding of the item.

        Raises
        ------
        RuntimeError
            If the run carries no encoder.
        """
        encoder = getattr(self._result, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                "chart has no attached encoder; declare "
                "`signature ... encoder ...` in the deduction block"
            )
        return encoder(item)

    def derivations(self, item: Item) -> list[Item]:
        """Return the items of the chart in which ``item`` was derived.

        The chart records only the aggregate weight at each item,
        so this returns the chart's item set as a flat enumeration
        when ``item`` is in the chart. The full derivation forest
        (a tree-structured object whose leaves are axioms and whose
        internal nodes are rule firings) is recoverable by running
        the agenda with a derivation semiring.

        Parameters
        ----------
        item : Item
            The item whose derivations to return.

        Returns
        -------
        list of Item
            Every item in the chart, or an empty list when ``item``
            was never derived.
        """
        # In the current implementation we expose the set of
        # items that contributed to the item's weight; a
        # derivation-semiring run is required for the full tree.
        # For the common case where the user wants "every item
        # in the chart," returning the item list is the closest
        # correct answer.
        if item not in (it for it, _ in self._result.chart.items()):
            return []
        return [it for it, _ in self._result.chart.items()]

    def goal_weight(self) -> torch.Tensor:
        """Aggregate weight over all goal items.

        If a single goal item was identified, returns its weight
        directly; otherwise aggregates via the semiring's
        ``plus`` reduction (the standard parse-as-marginal
        computation).

        Returns
        -------
        torch.Tensor
            The aggregated goal weight, or the semiring's zero when
            no goal item was derived.
        """
        if not self._result.goal_items:
            return torch.tensor(
                float(self.semiring.zero)
                if hasattr(self.semiring.zero, "__float__")
                else 0.0,
                dtype=torch.get_default_dtype(),
            )
        if len(self._result.goal_items) == 1:
            return self._result.goal_items[0][1]
        weights = torch.stack([w for _, w in self._result.goal_items])
        return self.semiring.plus(weights, dim=0)

    @property
    def goal_items(self) -> list[tuple[Item, torch.Tensor]]:
        """List of (item, weight) pairs matching the goal predicate."""
        return self._result.goal_items

    def __repr__(self) -> str:
        return (
            f"ChartView(items={sum(1 for _ in self._result.chart.items())},"
            f" goal={len(self._result.goal_items)},"
            f" semiring={self._result.semiring.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# Agendas
# ---------------------------------------------------------------------------


class Agenda(ABC):
    """A queue of items pending further inference.

    The agenda discipline (FIFO / LIFO / priority) determines evaluation
    order. Correctness requires an appropriate schedule and stopping rule;
    idempotent aggregation alone does not establish it. Semiring parsing
    [Goodman (1999)](<https://aclanthology.org/J99-4004/>) and bottom-up complexity analysis
    [McAllester (2002)](<https://doi.org/10.1145/581771.581774>) explain the relevant background.
    """

    @abstractmethod
    def push(self, item: Item, weight: torch.Tensor) -> None:
        """Add an item and its weight to the agenda.

        Parameters
        ----------
        item : Item
            The derived item.
        weight : torch.Tensor
            The weight contributed by this derivation.
        """

    @abstractmethod
    def pop(self) -> tuple[Item, torch.Tensor]:
        """Remove and return the next item under this discipline.

        Returns
        -------
        tuple of (Item, torch.Tensor)
            The next item and its pending weight.
        """

    @abstractmethod
    def empty(self) -> bool:
        """Whether no items are pending.

        Returns
        -------
        bool
            ``True`` when the agenda holds no items.
        """

    def bind_semiring(self, semiring: ChartSemiring) -> None:
        """Give the agenda access to the engine's semiring.

        `run_agenda` calls this once, before pushing the axioms.
        Disciplines that can exploit the binding override it; for
        instance, `FIFOAgenda` merges pending contributions
        to the same item via ``semiring.plus``. The default ignores
        the binding.

        Parameters
        ----------
        semiring : ChartSemiring
            The engine's semiring.
        """


class FIFOAgenda(Agenda):
    """First-in-first-out agenda (semi-naïve Datalog discipline).

    The agenda processes items in the order they were derived.
    Correct for any monotone semiring; the canonical choice for
    Datalog and Earley parsing.

    Once a semiring is bound (`bind_semiring`, which the engine does
    on entry), a contribution pushed for an item that is already
    pending merges into that item's single queue entry via
    ``semiring.plus``. Semiring distributivity makes the merge exact:
    firing a rule once on the merged contribution pushes the same
    total mass downstream as firing it once per contribution. Without
    merging, a cyclic rule graph re-enqueues every contribution
    separately, and the number of pending entries grows geometrically
    with derivation depth even when the weighted fixed point
    converges.
    """

    def __init__(self) -> None:
        self._queue: deque[tuple[Item, torch.Tensor]] = deque()
        self._order: deque[Item] = deque()
        self._pending: dict[Item, torch.Tensor] = {}
        self._semiring: ChartSemiring | None = None

    def bind_semiring(self, semiring: ChartSemiring) -> None:
        """Bind the semiring that merges pending contributions.

        Parameters
        ----------
        semiring : ChartSemiring
            The engine's semiring.
        """
        self._semiring = semiring

    def push(self, item: Item, weight: torch.Tensor) -> None:
        """Add an item and its weight to the agenda.

        Parameters
        ----------
        item : Item
            The derived item.
        weight : torch.Tensor
            The weight contributed by this derivation.
        """
        if self._semiring is None:
            self._queue.append((item, weight))
            return
        existing = self._pending.get(item)
        if existing is None:
            self._pending[item] = weight
            self._order.append(item)
            return
        stacked = torch.stack([existing, weight])
        self._pending[item] = self._semiring.plus(stacked, dim=0)

    def pop(self) -> tuple[Item, torch.Tensor]:
        """Remove and return the next item under this discipline.

        Returns
        -------
        tuple of (Item, torch.Tensor)
            The next item and its pending weight.
        """
        # Entries pushed before a semiring was bound drain first;
        # they were pushed earliest, so this preserves FIFO order.
        if self._queue:
            return self._queue.popleft()
        item = self._order.popleft()
        return item, self._pending.pop(item)

    def empty(self) -> bool:
        """Whether no items are pending.

        Returns
        -------
        bool
            ``True`` when the agenda holds no items.
        """
        return not self._queue and not self._order


class LIFOAgenda(Agenda):
    """Last-in-first-out agenda (depth-first proof-search discipline).

    The agenda processes items in reverse-derivation order.
    Suitable for goal-directed search like Agda elaboration and
    Twelf-style MLTT type-checking, where depth-first proof
    construction is the natural strategy.
    """

    def __init__(self) -> None:
        self._stack: list[tuple[Item, torch.Tensor]] = []

    def push(self, item: Item, weight: torch.Tensor) -> None:
        """Add an item and its weight to the agenda.

        Parameters
        ----------
        item : Item
            The derived item.
        weight : torch.Tensor
            The weight contributed by this derivation.
        """
        self._stack.append((item, weight))

    def pop(self) -> tuple[Item, torch.Tensor]:
        """Remove and return the next item under this discipline.

        Returns
        -------
        tuple of (Item, torch.Tensor)
            The next item and its pending weight.
        """
        return self._stack.pop()

    def empty(self) -> bool:
        """Whether no items are pending.

        Returns
        -------
        bool
            ``True`` when the agenda holds no items.
        """
        return not self._stack


class PriorityQueueAgenda(Agenda):
    """Priority-queue agenda parameterized by a priority function.

    Items are processed in decreasing priority order. Used by
    Knuth's algorithm (priority = current best score),
    A* parsing (priority = g + h with admissible h), and any
    best-first chart algorithm.

    Parameters
    ----------
    priority_fn : Callable
        ``(item, weight) -> float``. Higher priorities are
        processed first. The chart's monotone-priority property
        (Nederhof 2003) ensures correctness when the semiring is
        superior.
    """

    def __init__(self, priority_fn: Callable[[Item, torch.Tensor], float]) -> None:
        self._priority_fn = priority_fn
        self._heap: list[tuple[float, int, Item, torch.Tensor]] = []
        self._counter = 0  # tie-breaker for deterministic ordering

    def push(self, item: Item, weight: torch.Tensor) -> None:
        """Add an item and its weight to the agenda.

        Parameters
        ----------
        item : Item
            The derived item.
        weight : torch.Tensor
            The weight contributed by this derivation.
        """
        # heapq is a min-heap; negate priority for max-heap behavior.
        priority = -float(self._priority_fn(item, weight))
        self._counter += 1
        heapq.heappush(self._heap, (priority, self._counter, item, weight))

    def pop(self) -> tuple[Item, torch.Tensor]:
        """Remove and return the next item under this discipline.

        Returns
        -------
        tuple of (Item, torch.Tensor)
            The next item and its pending weight.
        """
        _, _, item, weight = heapq.heappop(self._heap)
        return item, weight

    def empty(self) -> bool:
        """Whether no items are pending.

        Returns
        -------
        bool
            ``True`` when the agenda holds no items.
        """
        return not self._heap


# ---------------------------------------------------------------------------
# The agenda engine
# ---------------------------------------------------------------------------


@dataclass
class AgendaResult:
    """The chart produced by an agenda-driven deduction.

    Attributes
    ----------
    chart : Chart
        The final K-presheaf on the item algebra.
    semiring : ChartSemiring
        The semiring used for weight aggregation.
    goal_items : list of (Item, torch.Tensor)
        Items in the chart matching the goal predicate, with
        their final aggregated weights.
    iterations : int
        Number of agenda steps consumed (diagnostic).
    """

    chart: Chart
    semiring: ChartSemiring
    goal_items: list[tuple[Item, torch.Tensor]] = field(default_factory=list)
    iterations: int = 0
    encoder: Encoder | None = None
    # Sum of all rule-attached + chart-attached loss values fired
    # during this run, populated by `DeductionSystem.run` when a
    # loss registry is attached. `None` if no losses fired.
    attached_loss: torch.Tensor | None = None


def run_agenda(
    axioms: Iterable[tuple[Item, torch.Tensor]],
    rules: Iterable[InferenceRule],
    semiring: ChartSemiring | None = None,
    agenda: Agenda | None = None,
    goal: Callable[[Item], bool] | None = None,
    max_iterations: int = 100_000,
    chart: Chart | None = None,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> AgendaResult:
    """Run the agenda-driven deduction engine to fixed point.

    The intended result is the closure of the input axioms under the
    weighted rules. A least fixed-point interpretation requires an ordered
    setting in which the rule operator and its iteration are well defined.
    The iteration bound may raise before closure is reached; idempotence
    alone does not guarantee convergence or independence from scheduling.

    The algorithm follows the parsing-as-deduction pattern
    [Shieber et al. (1995)](<https://www.sciencedirect.com/science/article/pii/074310669500035I>), with semiring aggregation
    [Goodman (1999)](<https://aclanthology.org/J99-4004/>):

    1. Push every axiom onto the agenda.
    2. While the agenda is non-empty:
       a. Pop an item.
       b. Insert-or-aggregate into the chart.
       c. If the chart's weight changed, fire every rule for which
          the popped item is a premise: scan the chart for matching
          siblings, compute the conclusion's weight via the rule's
          ``weight_fn``, and push the conclusion onto the agenda.
    3. Return the chart with all items that match ``goal``.

    Parameters
    ----------
    axioms : iterable of (Item, weight)
        Initial items with their input-derived weights.
    rules : iterable of InferenceRule
        The deduction system's hyperedges.
    semiring : ChartSemiring, optional
        Weight aggregation. Defaults to log-prob.
    agenda : Agenda, optional
        Evaluation strategy. Defaults to FIFO (semi-naïve).
    goal : callable, optional
        Predicate selecting result items. Defaults to "every item".
    max_iterations : int, optional
        Safety bound on agenda steps; raises if exceeded.
    chart : Chart, optional
        Initial chart (defaults to empty `HashChart`).
    rule_callback : callable, optional
        Called as ``(rule_name, antecedents, conclusion, weight)``
        after each rule firing.

    Returns
    -------
    AgendaResult
        The final chart, the goal items, and the step count.

    Raises
    ------
    RuntimeError
        If the agenda runs for more than ``max_iterations`` steps.
    """
    semiring = semiring or LOG_PROB
    agenda = agenda or FIFOAgenda()
    agenda.bind_semiring(semiring)
    chart = chart or HashChart()
    rules = tuple(rules)

    for item, w in axioms:
        agenda.push(item, w)

    iterations = 0
    while not agenda.empty():
        if iterations >= max_iterations:
            raise RuntimeError(
                f"agenda exceeded max_iterations={max_iterations}; "
                f"likely a non-terminating deduction"
            )
        iterations += 1
        item, weight = agenda.pop()
        changed = chart.insert_or_aggregate(item, weight, semiring)
        if not changed:
            continue
        # Fire every rule for which this item could be a premise.
        for rule in rules:
            for premise_idx, premise_pattern in enumerate(rule.premises):
                bindings = match(premise_pattern, item)
                if bindings is None:
                    continue
                # Collect all sibling premises by matching the
                # remaining premise patterns against the chart.
                _fire_rule_with_premise(
                    rule,
                    premise_idx,
                    item,
                    weight,
                    bindings,
                    chart,
                    semiring,
                    agenda,
                    rule_callback,
                )

    goal_items: list[tuple[Item, torch.Tensor]] = []
    if goal is not None:
        for item, w in chart.items():
            if goal(item):
                goal_items.append((item, w))
    else:
        goal_items = list(chart.items())

    return AgendaResult(
        chart=chart,
        semiring=semiring,
        goal_items=goal_items,
        iterations=iterations,
    )


def _fire_rule_with_premise(
    rule: InferenceRule,
    fixed_idx: int,
    fixed_item: Item,
    fixed_weight: torch.Tensor,
    bindings: Bindings,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> None:
    """Fire a rule where one premise is the popped item.

    Recursively scans the chart for matches to the remaining
    premise patterns under the running bindings. When all
    premises are matched, instantiates the conclusion, computes
    its weight via the rule's ``weight_fn`` (or the semiring
    product of premise weights if no ``weight_fn`` is given),
    checks the side condition, and pushes onto the agenda.
    """
    _fire_remaining_premises(
        rule=rule,
        accumulated=[(fixed_item, fixed_weight)],
        bindings=bindings,
        chart=chart,
        semiring=semiring,
        agenda=agenda,
        remaining_indices=[i for i in range(len(rule.premises)) if i != fixed_idx],
        fixed_idx=fixed_idx,
        fixed_pair=(fixed_item, fixed_weight),
        rule_callback=rule_callback,
    )


def _fire_remaining_premises(
    *,
    rule: InferenceRule,
    accumulated: list,
    bindings: Bindings,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    remaining_indices: list[int],
    fixed_idx: int,
    fixed_pair,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> None:
    """Recursively match remaining premise patterns.

    On each call, picks the next remaining premise and scans the
    chart for matching items consistent with the running
    bindings. Recurses until all premises are matched, then
    fires.
    """
    if not remaining_indices:
        _fire(
            rule,
            bindings,
            fixed_idx,
            fixed_pair,
            chart,
            semiring,
            agenda,
            rule_callback,
        )
        return
    next_idx = remaining_indices[0]
    next_pattern = rule.premises[next_idx]
    for sibling, sib_w in chart.lookup(next_pattern):
        sub_bindings = match(next_pattern, sibling, bindings)
        if sub_bindings is None:
            continue
        _fire_remaining_premises(
            rule=rule,
            accumulated=accumulated + [(sibling, sib_w)],
            bindings=sub_bindings,
            chart=chart,
            semiring=semiring,
            agenda=agenda,
            remaining_indices=remaining_indices[1:],
            fixed_idx=fixed_idx,
            fixed_pair=fixed_pair,
            rule_callback=rule_callback,
        )


def _fire(
    rule: InferenceRule,
    bindings: Bindings,
    fixed_idx: int,
    fixed_pair,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> None:
    """All premises matched; instantiate and push the conclusion.

    Collects every premise's weight from the chart at the matched
    items, computes the conclusion's weight via the rule's
    ``weight_fn`` (defaulting to the semiring product), and
    pushes the new item onto the agenda. The pushed item will
    only update the chart if its aggregated weight increases the
    chart's value (the idempotent-merge property).
    """
    if rule.side_condition is not None and not rule.side_condition(bindings):
        return
    # Gather premise (item, weight) pairs in declaration order.
    antecedents: list[tuple[Item, torch.Tensor]] = []
    for i, premise_pattern in enumerate(rule.premises):
        if i == fixed_idx:
            antecedents.append(fixed_pair)
            continue
        try:
            premise_item = instantiate(premise_pattern, bindings)
        except KeyError:
            return  # premise has unresolved wildcards, so it can't fire
        w = chart.get(premise_item)
        if w is None:
            return
        antecedents.append((premise_item, w))

    premise_weights = tuple(w for _, w in antecedents)
    # Compute the conclusion's weight.
    if rule.weight_fn is not None:
        conclusion_weight = rule.weight_fn(
            bindings,
            premise_weights,
            semiring,
        )
    elif not premise_weights:
        conclusion_weight = torch.tensor(
            float(semiring.one) if hasattr(semiring.one, "__float__") else 0.0,
            dtype=torch.get_default_dtype(),
        )
    else:
        acc = premise_weights[0]
        for w in premise_weights[1:]:
            acc = semiring.times(acc, w)
        conclusion_weight = acc
    try:
        conclusion_item = instantiate(rule.conclusion, bindings)
    except KeyError:
        return
    agenda.push(conclusion_item, conclusion_weight)
    if rule_callback is not None:
        rule_callback(
            rule.name,
            antecedents,
            conclusion_item,
            conclusion_weight,
        )


# ---------------------------------------------------------------------------
# Public deduction-system entry
# ---------------------------------------------------------------------------


def _full_extent_goal(
    goal: Callable[[Item], bool], input_value: object
) -> Callable[[Item], bool]:
    """Restrict a goal predicate to spans covering the whole input.

    A span-shaped chart holds an item for every substring the rules
    derive, and the goal category may head several of them; the system's
    answer is the weight of the derivations of the *whole* input, so a
    goal span must run from position zero to the input's length. An item
    that is not a span, and an input that is not a sequence, are judged
    by the predicate alone.

    Parameters
    ----------
    goal : Callable[[Item], bool]
        The declared goal predicate.
    input_value : object
        The input the system is run on.

    Returns
    -------
    Callable[[Item], bool]
        The predicate, narrowed to full-extent spans when the input is a
        sequence.
    """
    if not isinstance(input_value, (list, tuple)):
        return goal
    extent = len(input_value)

    def full_extent(item: Item) -> bool:
        """Whether an item is a goal covering the whole input.

        Parameters
        ----------
        item : Item
            The chart item.

        Returns
        -------
        bool
            The declared predicate's answer, and for a span item also
            that it runs from zero to the input's length.
        """
        if not goal(item):
            return False
        if (
            isinstance(item, tuple)
            and len(item) >= 4
            and item[0] == "span"
            and isinstance(item[1], int)
            and isinstance(item[2], int)
        ):
            return item[1] == 0 and item[2] == extent
        return True

    return full_extent


@dataclass
class DeductionSystem:
    """A weighted deductive system parameterized over its components.

    The system is parameterized by:

    - An item algebra (implicit in the patterns the rules use).
    - A list of arity-n `InferenceRule` hyperedges.
    - A `ChartSemiring` for weight aggregation.
    - An axiom injector ``In -> [(Item, Weight)]`` producing the
      input's lexical / boundary items.
    - A goal predicate ``Item -> bool`` selecting the result items.
    - (Optional) a chart constructor and an agenda strategy.

    The same data structure subsumes CKY (FIFO agenda, span items,
    Boolean / inside semiring), Viterbi (priority agenda with the
    current weight as priority, max-times semiring), A* parsing
    (priority agenda with an admissible heuristic, tropical
    semiring), MLTT type-checking (LIFO agenda, judgment items,
    Boolean semiring), and weighted Datalog (FIFO, atoms, any
    naturally-ordered semiring).
    """

    rules: tuple[InferenceRule, ...]
    semiring: ChartSemiring
    axiom_injector: Callable[[Any], list[tuple[Item, torch.Tensor]]]
    goal: Callable[[Item], bool]
    agenda_factory: Callable[[], Agenda] = FIFOAgenda
    chart_factory: Callable[[], Chart] = HashChart
    max_iterations: int = 100_000
    tolerance: float = 0.0

    def run(self, input_value: Any) -> AgendaResult:
        """Run the deduction system on an input value.

        Injects the input's axioms, runs the agenda to fixed point,
        attaches the item encoder when one is set, and sums any
        rule-attached and chart-attached losses.

        Parameters
        ----------
        input_value : Any
            The input the axiom injector reads.

        Returns
        -------
        AgendaResult
            The completed chart, its goal items, and any attached
            loss.
        """
        axioms = self.axiom_injector(input_value)
        registry = getattr(self, "_loss_registry", None)
        deduction_name = getattr(self, "_deduction_name", None)
        rule_loss_acc: list[torch.Tensor] = []

        def _rule_callback(
            rule_name: str,
            antecedents: list[tuple[Item, torch.Tensor]],
            conclusion: Item,
            conclusion_w: torch.Tensor,
        ) -> None:
            if registry is None or deduction_name is None:
                return
            env = {
                "rule": rule_name,
                "deduction": deduction_name,
                "antecedents": list(antecedents),
                "conclusion": conclusion,
                "weight": conclusion_w,
            }
            val = registry.evaluate_on(
                "rule",
                target=rule_name,
                env=env,
                rule_deduction=deduction_name,
            )
            rule_loss_acc.append(val)

        # If the user supplied a positive ``tolerance``, propagate
        # it to the chart so its aggregation step terminates on
        # convergence. The default chart_factory is ``HashChart``,
        # whose constructor accepts the tolerance; user-supplied
        # alternative chart factories can ignore the argument.
        try:
            chart_inst = self.chart_factory(tolerance=self.tolerance)
        except TypeError:
            chart_inst = self.chart_factory()
        result = run_agenda(
            axioms=axioms,
            rules=self.rules,
            semiring=self.semiring,
            agenda=self.agenda_factory(),
            goal=_full_extent_goal(self.goal, input_value),
            max_iterations=self.max_iterations,
            chart=chart_inst,
            rule_callback=(_rule_callback if registry is not None else None),
        )
        # Propagate any attached item-encoder to the result.
        comp = getattr(self, "_item_encoder", None)
        if comp is not None:
            result.encoder = comp
        # Evaluate chart-attached losses on the completed chart.
        if registry is not None and deduction_name is not None:
            chart_env = {
                "deduction": deduction_name,
                "chart": result.chart,
                "goal_items": result.goal_items,
            }
            chart_loss = registry.evaluate_on(
                "chart",
                target=deduction_name,
                env=chart_env,
            )
            losses = rule_loss_acc + [chart_loss]
        else:
            losses = rule_loss_acc
        if losses:
            total = losses[0]
            for v in losses[1:]:
                total = total + v
            result.attached_loss = total
        return result

    def __call__(self, input_value: Any) -> ChartView:
        """Run the deduction and return a `ChartView`.

        Convenience for the user-facing presheaf-evaluation API:
        the chart's weights are differentiable tensors, and the
        view exposes ``weight``, ``enumerate``, ``derivations``,
        and ``goal_weight`` methods for downstream programs.

        Parameters
        ----------
        input_value : Any
            The input the axiom injector reads.

        Returns
        -------
        ChartView
            A view onto the completed chart.
        """
        return ChartView(self.run(input_value))

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        """Yield every learnable parameter owned by this system.

        Walks the optional ``_axiom_module`` (lexicon log-weights)
        and ``_rule_module`` (per-rule log-weights keyed by category bindings)
        submodules attached by the compiler. The ``recurse`` flag
        is the standard `torch.nn.Module.parameters` signature
        so user code can pass a ``DeductionSystem`` anywhere a
        ``nn.Module`` parameter iterator is expected.

        Parameters
        ----------
        recurse : bool
            Whether to include parameters of nested submodules.

        Returns
        -------
        Iterable of torch.nn.Parameter
            The learnable parameters.
        """
        for attr in ("_axiom_module", "_rule_module"):
            mod = getattr(self, attr, None)
            if mod is not None and hasattr(mod, "parameters"):
                yield from mod.parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterable[tuple[str, torch.nn.Parameter]]:
        """Yield ``(name, parameter)`` pairs over all learnable parameters.

        Parameters
        ----------
        prefix : str
            A prefix prepended to every parameter name.
        recurse : bool
            Whether to include parameters of nested submodules.

        Returns
        -------
        Iterable of (str, torch.nn.Parameter)
            The named learnable parameters.
        """
        for attr in ("_axiom_module", "_rule_module"):
            mod = getattr(self, attr, None)
            if mod is not None and hasattr(mod, "named_parameters"):
                sub_prefix = f"{prefix}.{attr}" if prefix else attr
                for n, p in mod.named_parameters(
                    prefix=sub_prefix,
                    recurse=recurse,
                ):
                    yield n, p


# ---------------------------------------------------------------------------
# Strategy factories: concrete parsers as agenda specializations
# ---------------------------------------------------------------------------


def cky_agenda() -> Agenda:
    """The CKY (bottom-up sweep) agenda: semi-naïve FIFO.

    For context-free + Boolean / inside semirings, FIFO order
    suffices to reach the chart's fixed point in
    :math:`O(n^3 \\cdot |R|)` time (McAllester 2002).

    Returns
    -------
    Agenda
        A fresh `FIFOAgenda`.
    """
    return FIFOAgenda()


def earley_agenda() -> Agenda:
    """The Earley (predict / scan / complete) agenda: FIFO.

    Earley's algorithm is identical to FIFO agenda-driven
    deduction over the predict / scan / complete items
    (Pereira & Warren 1983).

    Returns
    -------
    Agenda
        A fresh `FIFOAgenda`.
    """
    return FIFOAgenda()


def viterbi_agenda(priority_fn: Callable[[Item, torch.Tensor], float]) -> Agenda:
    """Viterbi-style best-first agenda.

    Priorities are the current best weight in the
    :math:`(\\max, \\times)` semiring; the agenda is a priority
    queue. Equivalent to Knuth's algorithm when the semiring is
    superior (Nederhof 2003).

    Parameters
    ----------
    priority_fn : Callable[[Item, torch.Tensor], float]
        The priority of an item given its weight; higher pops first.

    Returns
    -------
    Agenda
        A `PriorityQueueAgenda` ordered by ``priority_fn``.
    """
    return PriorityQueueAgenda(priority_fn)


def astar_agenda(
    g_plus_h: Callable[[Item, torch.Tensor], float],
) -> Agenda:
    """Priority agenda for an A* parsing strategy [Klein and Manning (2003)](<https://aclanthology.org/N03-1016/>).

    The queue pops the highest numeric priority first. If ``g`` and ``h``
    are costs to minimize, pass a priority such as ``-(g + h)``. A valid
    A* algorithm also needs an appropriate heuristic, update policy, and
    termination condition; constructing this queue alone does not prove
    optimality or enumerate every item in increasing true cost.

    Parameters
    ----------
    g_plus_h : Callable[[Item, torch.Tensor], float]
        The priority ``g + h`` of an item given its weight; higher pops first.

    Returns
    -------
    Agenda
        A `PriorityQueueAgenda` ordered by ``g_plus_h``.
    """
    return PriorityQueueAgenda(g_plus_h)


def knuth_agenda() -> Agenda:
    """Knuth's best-first hyperpath search.

    Priority is the current chart weight (the item's best-so-far
    score). For a superior semiring, this is Dijkstra's algorithm
    on AND-OR hypergraphs (Knuth 1977; Nederhof 2003).

    Returns
    -------
    Agenda
        A `PriorityQueueAgenda` ordered by the item's weight.
    """

    def _priority(_item: Item, weight: torch.Tensor) -> float:
        return float(weight) if weight.numel() == 1 else float(weight.sum())

    return PriorityQueueAgenda(_priority)


def depth_first_agenda() -> Agenda:
    """LIFO agenda: depth-first proof search (Agda / Twelf style).

    Returns
    -------
    Agenda
        A fresh `LIFOAgenda`.
    """
    return LIFOAgenda()


def semi_naive_agenda() -> Agenda:
    """Semi-naïve Datalog evaluation (McAllester 2002): FIFO.

    Returns
    -------
    Agenda
        A fresh `FIFOAgenda`.
    """
    return FIFOAgenda()


__all__ = [
    "Item",
    "Wildcard",
    "make_wildcard",
    "Pattern",
    "Bindings",
    "InferenceRule",
    "instantiate",
    "match",
    "Chart",
    "HashChart",
    "ChartView",
    "Agenda",
    "FIFOAgenda",
    "LIFOAgenda",
    "PriorityQueueAgenda",
    "AgendaResult",
    "run_agenda",
    "DeductionSystem",
    "cky_agenda",
    "earley_agenda",
    "viterbi_agenda",
    "astar_agenda",
    "knuth_agenda",
    "depth_first_agenda",
    "semi_naive_agenda",
]
