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
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from quivers.stochastic.productions import ProductionGroup, ProductionWeights
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


class CorpusChart:
    """The charts of a corpus of independent sentences.

    Parameters
    ----------
    charts : tuple[ChartView, ...]
        One chart per sentence, in corpus order.
    semiring : ChartSemiring
        The semiring the charts were computed in.
    """

    def __init__(self, charts: tuple[ChartView, ...], semiring: ChartSemiring) -> None:
        self.charts = charts
        self.semiring = semiring

    def goal_weight(self) -> torch.Tensor:
        """The corpus's goal weight: the semiring product over sentences.

        Returns
        -------
        torch.Tensor
            Under ``LogProb``, the sum of the sentences' log inside
            weights, the corpus's log-likelihood.
        """
        total = self.charts[0].goal_weight()
        for chart in self.charts[1:]:
            total = self.semiring.times(total, chart.goal_weight())
        return total

    def __repr__(self) -> str:
        """A one-line summary.

        Returns
        -------
        str
            The sentence count.
        """
        return f"CorpusChart(sentences={len(self.charts)})"


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


type RuleCallback = Callable[
    [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor], None
]
"""A hook called once per rule application with its name, antecedents
(each with its chart weight), conclusion, and contribution."""


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
    rule_callback: RuleCallback | None = None,
) -> AgendaResult:
    """Run the agenda-driven deduction engine to fixed point.

    The value of an item is the semiring sum, over its derivations, of
    the semiring product of the weights in the derivation
    [Goodman (1999)](<https://aclanthology.org/J99-4004/>): an axiom's
    weight, and for each rule application the product of its
    antecedents' values (combined by the rule's ``weight_fn`` when it
    has one). Equivalently, the chart solves the equations
    ``value(c) = axiom(c) ⊕ ⨁_{applications e → c} ⊗_j value(a_j)``;
    on a finite acyclic system the solution is unique and equals the
    sum over derivations, each derivation counted exactly once.

    The engine is the weighted agenda algorithm of
    [Eisner, Goldlust, and Smith (2005)](<https://doi.org/10.3115/1220575.1220611>),
    figure 3:

    1. Push every axiom onto the agenda.
    2. While the agenda is non-empty:
       a. Pop an item ``a`` with its pending update ``Δ``.
       b. Record ``old = chart[a]`` and aggregate ``chart[a] := old ⊕ Δ``.
       c. If the chart's weight changed, then for every rule and every
          premise position ``i`` that ``a`` matches, and every way of
          matching the other premises against the chart, push onto
          the conclusion the product whose ``j``-th factor is ``old``
          when ``j < i`` and premise ``j`` is ``a`` itself, ``Δ`` when
          ``j = i``, and ``chart[a_j]`` otherwise.
    3. Return the chart with all items that match ``goal``.

    The case split in 2c is what keeps an item that fills several
    premises of one rule from being counted once per premise: when
    ``a`` fills the premise positions ``S``, the increments over
    ``i ∈ S`` sum, by distributivity, to exactly the terms of
    ``⊗_{j ∈ S} (old ⊕ Δ)`` that contain at least one ``Δ``. A rule
    ``NP, NP ⊢ S`` with one ``NP`` axiom of weight ``w`` thus gives
    ``S`` the weight ``w ⊗ w``, not ``(w ⊗ w) ⊕ (w ⊗ w)``.

    A least fixed-point reading of a cyclic system requires an ordered
    setting in which the iteration converges; the iteration bound may
    raise before closure is reached.

    Parameters
    ----------
    axioms : iterable of (Item, weight)
        Initial items with their input-derived weights.
    rules : iterable of InferenceRule
        The deduction system's hyperedges. A rule's ``weight_fn`` must
        distribute over ``⊕`` in each premise weight (for instance, a
        rule weight times the product of the premise weights), as the
        algorithm propagates increments rather than totals.
    semiring : ChartSemiring, optional
        Weight aggregation. Defaults to log-prob.
    agenda : Agenda, optional
        Evaluation strategy. Defaults to FIFO (semi-naive).
    goal : callable, optional
        Predicate selecting result items. Defaults to "every item".
    max_iterations : int, optional
        Safety bound on agenda steps; raises if exceeded.
    chart : Chart, optional
        Initial chart (defaults to empty `HashChart`).
    rule_callback : callable, optional
        Called as ``(rule_name, antecedents, conclusion, weight)`` once
        per rule application (a rule together with one tuple of
        antecedent items), after the chart reaches its fixed point.
        ``antecedents`` carries each antecedent's final chart weight
        and ``weight`` is the application's contribution to the
        conclusion computed from those weights.

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
    applications: dict[tuple[int, tuple[Item, ...]], _Application] | None = (
        {} if rule_callback is not None else None
    )

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
        item, delta = agenda.pop()
        old = chart.get(item)
        changed = chart.insert_or_aggregate(item, delta, semiring)
        if not changed:
            continue
        update = _Update(item=item, delta=delta, old=old)
        # Fire every rule for which this item could be a premise.
        for rule_index, rule in enumerate(rules):
            for premise_idx, premise_pattern in enumerate(rule.premises):
                bindings = match(premise_pattern, item)
                if bindings is None:
                    continue
                _fire_remaining_premises(
                    rule=rule,
                    rule_index=rule_index,
                    bindings=bindings,
                    chart=chart,
                    semiring=semiring,
                    agenda=agenda,
                    remaining_indices=[
                        i for i in range(len(rule.premises)) if i != premise_idx
                    ],
                    fixed_idx=premise_idx,
                    update=update,
                    applications=applications,
                )

    if rule_callback is not None and applications is not None:
        for application in applications.values():
            weights = tuple(_chart_weight(chart, a) for a in application.antecedents)
            rule_callback(
                application.rule.name,
                list(zip(application.antecedents, weights)),
                application.conclusion,
                _conclusion_weight(
                    application.rule, application.bindings, weights, semiring
                ),
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


def _chart_weight(chart: Chart, item: Item) -> torch.Tensor:
    """The chart weight of an item the run has derived.

    Parameters
    ----------
    chart : Chart
        The completed chart.
    item : Item
        An antecedent of a recorded rule application.

    Returns
    -------
    torch.Tensor
        The item's weight.

    Raises
    ------
    RuntimeError
        If the item is not in the chart; every recorded antecedent was
        matched against the chart, and a chart never drops an item.
    """
    w = chart.get(item)
    if w is None:
        raise RuntimeError(f"recorded antecedent {item!r} is not in the chart")
    return w


class _Update:
    """The chart update being propagated: an item, its increment, its old value.

    Parameters
    ----------
    item : Item
        The popped item.
    delta : torch.Tensor
        The increment aggregated into the item's chart weight.
    old : torch.Tensor or None
        The item's chart weight before the increment, or ``None`` if
        the item was absent (the semiring's zero).
    """

    __slots__ = ("item", "delta", "old")

    def __init__(self, *, item: Item, delta: torch.Tensor, old: torch.Tensor | None):
        self.item = item
        self.delta = delta
        self.old = old


class _Application:
    """One rule application: a rule with one tuple of antecedent items.

    Parameters
    ----------
    rule : InferenceRule
        The rule applied.
    bindings : Bindings
        The pattern variables the antecedents bind.
    antecedents : tuple of Item
        The antecedent items in premise order.
    conclusion : Item
        The item the application derives.
    """

    __slots__ = ("rule", "bindings", "antecedents", "conclusion")

    def __init__(
        self,
        *,
        rule: InferenceRule,
        bindings: Bindings,
        antecedents: tuple[Item, ...],
        conclusion: Item,
    ) -> None:
        self.rule = rule
        self.bindings = bindings
        self.antecedents = antecedents
        self.conclusion = conclusion


def _fire_remaining_premises(
    *,
    rule: InferenceRule,
    rule_index: int,
    bindings: Bindings,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    remaining_indices: list[int],
    fixed_idx: int,
    update: _Update,
    applications: dict[tuple[int, tuple[Item, ...]], _Application] | None,
) -> None:
    """Match the remaining premise patterns against the chart, then fire.

    Picks the next remaining premise, scans the chart for items
    consistent with the running bindings, and recurses until every
    premise is matched.

    Parameters
    ----------
    rule : InferenceRule
        The rule being fired.
    rule_index : int
        The rule's position in the system, which identifies its
        applications.
    bindings : Bindings
        The pattern variables bound so far.
    chart : Chart
        The chart, already holding the popped item's new weight.
    semiring : ChartSemiring
        The semiring of the run.
    agenda : Agenda
        The agenda that receives the conclusion's increment.
    remaining_indices : list of int
        The premise positions still to match.
    fixed_idx : int
        The premise position the popped item fills.
    update : _Update
        The popped item, its increment, and its old weight.
    applications : dict or None
        The rule applications seen so far, recorded for the rule
        callback; ``None`` when no callback is set.
    """
    if not remaining_indices:
        _fire(
            rule=rule,
            rule_index=rule_index,
            bindings=bindings,
            fixed_idx=fixed_idx,
            update=update,
            chart=chart,
            semiring=semiring,
            agenda=agenda,
            applications=applications,
        )
        return
    next_idx = remaining_indices[0]
    next_pattern = rule.premises[next_idx]
    for sibling, _ in chart.lookup(next_pattern):
        sub_bindings = match(next_pattern, sibling, bindings)
        if sub_bindings is None:
            continue
        _fire_remaining_premises(
            rule=rule,
            rule_index=rule_index,
            bindings=sub_bindings,
            chart=chart,
            semiring=semiring,
            agenda=agenda,
            remaining_indices=remaining_indices[1:],
            fixed_idx=fixed_idx,
            update=update,
            applications=applications,
        )


def _conclusion_weight(
    rule: InferenceRule,
    bindings: Bindings,
    premise_weights: tuple[torch.Tensor, ...],
    semiring: ChartSemiring,
) -> torch.Tensor:
    """The weight a rule application contributes to its conclusion.

    Parameters
    ----------
    rule : InferenceRule
        The rule applied.
    bindings : Bindings
        The pattern variables the antecedents bind.
    premise_weights : tuple of torch.Tensor
        One weight per premise, in premise order.
    semiring : ChartSemiring
        The semiring of the run.

    Returns
    -------
    torch.Tensor
        The rule's ``weight_fn`` applied to the premise weights, or the
        semiring product of the premise weights when it has none.
    """
    if rule.weight_fn is not None:
        return rule.weight_fn(bindings, premise_weights, semiring)
    if not premise_weights:
        return torch.tensor(
            float(semiring.one) if hasattr(semiring.one, "__float__") else 0.0,
            dtype=torch.get_default_dtype(),
        )
    acc = premise_weights[0]
    for w in premise_weights[1:]:
        acc = semiring.times(acc, w)
    return acc


def _fire(
    *,
    rule: InferenceRule,
    rule_index: int,
    bindings: Bindings,
    fixed_idx: int,
    update: _Update,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    applications: dict[tuple[int, tuple[Item, ...]], _Application] | None,
) -> None:
    """Push one rule application's increment onto the agenda.

    Premise ``fixed_idx`` contributes the popped increment; an earlier
    premise filled by the popped item itself contributes the item's old
    weight; every other premise contributes its current chart weight
    (Eisner, Goldlust, and Smith 2005, figure 3, line 12). A factor
    whose weight is the semiring's zero (an absent item) makes the
    increment zero, and nothing is pushed.

    Parameters
    ----------
    rule : InferenceRule
        The rule being fired.
    rule_index : int
        The rule's position in the system.
    bindings : Bindings
        The bindings of every premise pattern.
    fixed_idx : int
        The premise position the popped item fills.
    update : _Update
        The popped item, its increment, and its old weight.
    chart : Chart
        The chart, already holding the popped item's new weight.
    semiring : ChartSemiring
        The semiring of the run.
    agenda : Agenda
        The agenda that receives the conclusion's increment.
    applications : dict or None
        The rule applications seen so far, extended with this one;
        ``None`` when no callback is set.
    """
    if rule.side_condition is not None and not rule.side_condition(bindings):
        return
    antecedents: list[Item] = []
    factors: list[torch.Tensor] = []
    complete = True
    for i, premise_pattern in enumerate(rule.premises):
        if i == fixed_idx:
            antecedents.append(update.item)
            factors.append(update.delta)
            continue
        premise_item = instantiate(premise_pattern, bindings)
        antecedents.append(premise_item)
        if i < fixed_idx and premise_item == update.item:
            w = update.old
        else:
            w = chart.get(premise_item)
        if w is None:
            complete = False
            continue
        factors.append(w)
    conclusion_item = instantiate(rule.conclusion, bindings)
    if applications is not None:
        key = (rule_index, tuple(antecedents))
        if key not in applications:
            applications[key] = _Application(
                rule=rule,
                bindings=bindings,
                antecedents=tuple(antecedents),
                conclusion=conclusion_item,
            )
    if not complete:
        return
    agenda.push(
        conclusion_item,
        _conclusion_weight(rule, bindings, tuple(factors), semiring),
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
    productions: ProductionWeights | None = None
    production_refusal: str | None = None

    def production_groups(self) -> tuple[ProductionGroup, ...]:
        """The system's productions, grouped by left-hand side.

        Returns
        -------
        tuple[ProductionGroup, ...]
            One group per left-hand side, in order of its first
            production; rules first, then lexicon entries.

        Raises
        ------
        ValueError
            If the system has no fixed production groups: it was not
            compiled from a grammar, or a rule stands for one production
            per binding of a category variable.
        """
        if self.productions is None:
            raise ValueError(
                "the deduction has no fixed production groups: "
                + (self.production_refusal or "it was not compiled from a grammar")
            )
        return self.productions.groups

    def production_log_probs(
        self, probabilities: Mapping[str, torch.Tensor] | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Each production's log-probability, by left-hand side.

        Parameters
        ----------
        probabilities : Mapping[str, torch.Tensor] | None
            Production probabilities to read instead of the learned
            weights: one vector per left-hand side with several
            productions, keyed by its label, in production order.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            For each left-hand side, the log-probability of each of its
            productions (a rule by name, a lexicon entry by its word in
            double quotes); the probabilities of a group sum to one.
            The tensors carry gradients to the learned weights.

        Raises
        ------
        ValueError
            If the system does not normalize its weights per left-hand
            side (``normalize=lhs``), or the probabilities do not fit its
            groups.
        """
        return self._normalized().log_probs(probabilities)

    def _normalized(self) -> ProductionWeights:
        """The production weights of a grammar normalized per left-hand side.

        Returns
        -------
        ProductionWeights
            The system's production weights.

        Raises
        ------
        ValueError
            If the system does not normalize its weights per left-hand
            side.
        """
        if self.productions is None or not self.productions.normalize:
            raise ValueError(
                "the deduction's weights are free log-weights, not production "
                "probabilities; declare normalize=lhs to normalize them per "
                "left-hand side"
            )
        return self.productions

    def run(
        self,
        input_value: Any,
        probabilities: Mapping[str, torch.Tensor] | None = None,
    ) -> AgendaResult:
        """Run the deduction system on an input value.

        Injects the input's axioms, runs the agenda to fixed point,
        attaches the item encoder when one is set, and sums any
        rule-attached and chart-attached losses.

        Parameters
        ----------
        input_value : Any
            The input the axiom injector reads.
        probabilities : Mapping[str, torch.Tensor] | None
            Production probabilities the run reads instead of the
            learned weights, for a grammar normalized per left-hand
            side: one vector per left-hand side with several
            productions, keyed by its label (`production_groups`).

        Returns
        -------
        AgendaResult
            The completed chart, its goal items, and any attached
            loss.

        Raises
        ------
        ValueError
            If probabilities are given to a system that is not
            normalized per left-hand side, or do not fit its groups.
        """
        productions = self.productions
        # A normalized grammar computes its production table once per run,
        # so every axiom and firing reads one consistent table.
        if probabilities is not None:
            normalized = self._normalized()
            scope = normalized.active(normalized.table(probabilities))
        elif productions is not None and productions.normalize:
            scope = productions.active(productions.table())
        else:
            scope = nullcontext()
        with scope:
            return self._run(input_value)

    def _run(self, input_value: Any) -> AgendaResult:
        """Run the deduction under the weights already in scope.

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

    def __call__(
        self,
        input_value: Any,
        probabilities: Mapping[str, torch.Tensor] | None = None,
    ) -> ChartView:
        """Run the deduction and return a `ChartView`.

        Convenience for the user-facing presheaf-evaluation API:
        the chart's weights are differentiable tensors, and the
        view exposes ``weight``, ``enumerate``, ``derivations``,
        and ``goal_weight`` methods for downstream programs.

        Parameters
        ----------
        input_value : Any
            The input the axiom injector reads.
        probabilities : Mapping[str, torch.Tensor] | None
            Production probabilities to read instead of the learned
            weights, as for `run`.

        Returns
        -------
        ChartView
            A view onto the completed chart.

        Raises
        ------
        ValueError
            If the probabilities do not fit the system, as for `run`.
        """
        return ChartView(self.run(input_value, probabilities))

    def corpus(
        self,
        sentences: Sequence[Any],
        probabilities: Mapping[str, torch.Tensor] | None = None,
    ) -> CorpusChart:
        """Run the deduction on every sentence of a corpus.

        The sentences are independent draws, so the corpus's goal weight
        is the semiring product of the sentences' goal weights: under
        ``LogProb``, the sum of their log inside weights.

        Parameters
        ----------
        sentences : Sequence[Any]
            The inputs, one per sentence.
        probabilities : Mapping[str, torch.Tensor] | None
            Production probabilities every sentence reads instead of the
            learned weights, as for `run`.

        Returns
        -------
        CorpusChart
            The sentences' charts and the corpus's goal weight.

        Raises
        ------
        ValueError
            If the corpus is empty, or the probabilities do not fit the
            system.
        """
        if len(sentences) == 0:
            raise ValueError("a corpus has at least one sentence")
        views = tuple(self(sentence, probabilities) for sentence in sentences)
        return CorpusChart(views, self.semiring)

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
