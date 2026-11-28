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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Protocol, runtime_checkable

import didactic.api as dx
import torch

from quivers._positions import anywhere, depends_on_position
from quivers._values import HeldValue, OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.core.laws import PLUS_IDEMPOTENT, PLUS_SELECTIVE
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import Phase, WorkField, WorkReport, measuring
from quivers.qiec.rules import RulePresentation
from quivers.stochastic.tries import Trie, TrieCounter, descend, insert, intersect
from quivers.stochastic.productions import ProductionGroup, ProductionWeights
from quivers.stochastic.semiring import ChartSemiring, LOG_PROB
from quivers.structural.encoder import Encoder
from quivers.structural.losses import LossRegistry
from quivers.structural.signature import Signature


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


class Wildcard(dx.Model):
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
    return Wildcard(name=name)


Pattern = tuple[Any, ...] | Wildcard
"""A pattern is an item-shaped tuple with optional `Wildcard` positions,
or a `Wildcard` matching the whole item. A tuple pattern matches an item
if (a) the constructor name
matches, (b) the arities match, (c) each non-wildcard slot is
structurally equal to the corresponding item slot, and (d) all
wildcards with the same name bind to equal values."""


Bindings = dict[str, Any]
"""A bindings environment maps wildcard variable names to the
values they captured during pattern-matching."""


class InferenceRule(QiecModel):
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
    premises: tuple[tuple[HeldValue, ...] | Wildcard, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    conclusion: tuple[HeldValue, ...] | Wildcard = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    weight_fn: (
        Callable[[Bindings, tuple[torch.Tensor, ...], ChartSemiring], torch.Tensor]
        | None
    ) = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    side_condition: Callable[[Bindings], bool] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


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
    see an equal value, whatever value it is bound to, ``None``
    included. Tuples match slot by slot when their
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
        if pattern.name not in bindings:
            return {**bindings, pattern.name: item}
        if bindings[pattern.name] == item:
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
            if p.name not in out:
                out[p.name] = v
            elif out[p.name] != v:
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


def _ground(pattern: Pattern, bindings: Bindings) -> bool:
    """Whether a pattern has no wildcard left once bindings are substituted.

    Parameters
    ----------
    pattern : Pattern
        The pattern.
    bindings : Bindings
        Values of some wildcards.

    Returns
    -------
    bool
        True when every wildcard in it is bound.
    """
    if isinstance(pattern, Wildcard):
        return pattern.name in bindings
    if isinstance(pattern, tuple):
        return all(_ground(element, bindings) for element in pattern)
    return True


@runtime_checkable
class ReportsWork(Protocol):
    """A chart or agenda queue that reports the work it has done.

    The counts cover everything since the component was made; a run takes
    the difference across it. The counters it does not return are not
    measured, and a component that does not implement this leaves all its
    work unmeasured in the run's report: a declared absence, not a count
    of zero.
    """

    def work_counts(self) -> Mapping[WorkField, int]:
        """The work done since the component was made.

        Returns
        -------
        Mapping of WorkField to int
            A count for each counter it measures, of those in
            `COMPONENT_FIELDS`.
        """
        ...


COMPONENT_FIELDS: tuple[WorkField, ...] = measuring(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.HASH_OPERATIONS,
        WorkField.KEY_COMPARISONS,
    }
)
"""The counters a chart or a queue may contribute to a run's report."""


class _ComponentWork:
    """Counts a chart or queue keeps of its own operations."""

    def __init__(self) -> None:
        self.sums = 0
        self.hashes = 0
        self.comparisons = 0

    def counts(self) -> dict[WorkField, int]:
        """Every counter a chart or queue may contribute, as counted.

        Returns
        -------
        dict of WorkField to int
            Sums, products (none), hash operations and comparisons.
        """
        return {
            WorkField.SCALAR_SUMS: self.sums,
            WorkField.SCALAR_PRODUCTS: 0,
            WorkField.HASH_OPERATIONS: self.hashes,
            WorkField.KEY_COMPARISONS: self.comparisons,
        }


class Chart(ABC):
    """A finite map from derived items to their weights in a semiring K.

    A chart assigns each item it holds a weight in the semiring's
    carrier; an item it does not hold has the semiring's zero. A
    completed chart solves the deduction system's equations
    ``value(c) = axiom(c) ⊕ ⨁_{applications e → c} ⊗_j value(a_j)``;
    the agenda engine fills it bottom-up.

    Concrete subclasses choose a storage strategy: dense
    tensor-indexed (for CKY-style spans), sparse hash-indexed
    (for Datalog atoms), or term-structural (for MLTT
    judgments).
    """

    @abstractmethod
    def lookup(
        self, pattern: Pattern, bindings: Bindings | None = None
    ) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate the (item, weight) pairs that match a pattern under bindings.

        Parameters
        ----------
        pattern : Pattern
            The pattern to match stored items against.
        bindings : Bindings or None
            Values of some of the pattern's wildcards; an item matches
            when it matches the pattern with those wildcards fixed.

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

    def propagated(
        self,
        before: torch.Tensor,
        item: Item,
        weight: torch.Tensor,
        semiring: ChartSemiring,
    ) -> torch.Tensor:
        """The increment of an update that its consequences take.

        Parameters
        ----------
        before : torch.Tensor
            The item's weight before the update.
        item : Item
            The item, whose stored weight the update changed.
        weight : torch.Tensor
            The increment.
        semiring : ChartSemiring
            The semiring.

        Returns
        -------
        torch.Tensor
            The increment itself.
        """
        del before, item, semiring
        return weight

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

    Stores items as keys in a Python dict, with indexes of the items of
    each length keyed on the positions a lookup finds bound. Suitable for
    Datalog atoms, MLTT judgments, and any deduction whose items don't
    have a dense integer encoding. A lookup costs its key and the items
    that agree with it on the bound positions, each matched against the
    whole pattern, so bindings pushed into it select the candidates.

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
        self._work = _ComponentWork()
        self._indexes: dict[
            tuple[int, tuple[int, ...]], dict[tuple[HeldValue, ...], list[Item]]
        ] = {}

    def work_counts(self) -> Mapping[WorkField, int]:
        """The chart's work since it was made.

        Returns
        -------
        Mapping of WorkField to int
            Sums of aggregation, one per element merged; dictionary
            accesses; the items a lookup matched against its pattern; and
            no products.
        """
        return self._work.counts()

    def lookup(
        self, pattern: Pattern, bindings: Bindings | None = None
    ) -> Iterable[tuple[Item, torch.Tensor]]:
        """The stored items that match a pattern under bindings, by a keyed index.

        A tuple pattern of length ``L`` is looked up in the index of the
        items of length ``L`` keyed on the positions whose pattern is
        ground once the bindings are substituted; with no such position
        the key is empty and the index holds every item of that length.
        Indexes are built on first use and kept up to date as items
        arrive, so a lookup costs its key and the items that agree with
        it, each then matched against the whole pattern.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        bindings : Bindings or None
            Values of some of its wildcards.

        Returns
        -------
        Iterable of (Item, torch.Tensor)
            The matching items with their weights.
        """
        fixed = {} if bindings is None else bindings
        if isinstance(pattern, Wildcard) and pattern.name in fixed:
            self._work.hashes += 1
            value = fixed[pattern.name]
            stored = self._store.get(value)
            return [] if stored is None else [(value, stored)]
        if not isinstance(pattern, tuple):
            candidates: Iterable[Item] = list(self._store)
        else:
            positions = tuple(
                index
                for index, element in enumerate(pattern)
                if _ground(element, fixed)
            )
            key = tuple(instantiate(pattern[index], fixed) for index in positions)
            self._work.hashes += 1
            candidates = self._index(len(pattern), positions).get(key, [])
        found: list[tuple[Item, torch.Tensor]] = []
        for item in candidates:
            self._work.comparisons += 1
            if match(pattern, item, fixed) is not None:
                found.append((item, self._store[item]))
        return found

    def _index(
        self, length: int, positions: tuple[int, ...]
    ) -> dict[tuple[HeldValue, ...], list[Item]]:
        """The index of the items of one length keyed on some positions.

        Parameters
        ----------
        length : int
            The items' length.
        positions : tuple of int
            The positions of the key.

        Returns
        -------
        dict of tuple to list of Item
            The items by their values at the positions, built on first use.
        """
        found = self._indexes.get((length, positions))
        if found is None:
            found = {}
            for item in self._store:
                if isinstance(item, tuple) and len(item) == length:
                    self._work.hashes += 1
                    found.setdefault(tuple(item[i] for i in positions), []).append(item)
            self._indexes[(length, positions)] = found
        return found

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
            tolerance, the item is new, or the update is the zero and
            carries a derivative in an algebra that is not declared
            idempotent: such an update leaves the number unchanged and
            moves its derivative, which its consequences need.
        """
        self._work.hashes += 1
        if item not in self._store:
            self._store[item] = weight
            self._work.hashes += 1
            self._indexed(item)
            return True
        existing = self._store[item]
        stacked = torch.stack([existing, weight])
        merged = semiring.plus(stacked, dim=0)
        self._work.sums += merged.numel()
        self._work.hashes += 2
        self._store[item] = merged
        return anywhere(self._changed(existing, merged) | _carries(weight, semiring))

    def _indexed(self, item: Item) -> None:
        """Add a new item to every index of its length.

        Parameters
        ----------
        item : Item
            The item, new to the chart.
        """
        if not isinstance(item, tuple):
            return
        for (length, positions), index in self._indexes.items():
            if len(item) == length:
                self._work.hashes += 1
                index.setdefault(tuple(item[i] for i in positions), []).append(item)

    def _changed(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Whether a stored weight changed by more than the tolerance.

        Parameters
        ----------
        before : torch.Tensor
            The weight before an update.
        after : torch.Tensor
            The weight after it.

        Returns
        -------
        torch.Tensor
            A scalar Boolean.
        """
        if self._tolerance > 0.0:
            return (after.detach() - before.detach()).abs().amax() > self._tolerance
        return (after != before).any()

    def propagated(
        self,
        before: torch.Tensor,
        item: Item,
        weight: torch.Tensor,
        semiring: ChartSemiring,
    ) -> torch.Tensor:
        """The increment of an update that its consequences take.

        Parameters
        ----------
        before : torch.Tensor
            The item's weight before the update.
        item : Item
            The item.
        weight : torch.Tensor
            The increment.
        semiring : ChartSemiring
            The semiring, whose zero is the increment of no change.

        Returns
        -------
        torch.Tensor
            The increment. Over many parameter values, it is zero at the
            values where the stored weight did not change, whose agenda
            would not have gone on from here.
        """
        self._work.hashes += 1
        changed = self._changed(before, self._store[item]) | _carries(weight, semiring)
        if not depends_on_position(changed):
            return weight
        return torch.where(changed, weight, semiring.zero_like((), weight))

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
        self._work.hashes += 1
        return self._store.get(item)

    def items(self) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate all stored (item, weight) pairs.

        Returns
        -------
        Iterable of (Item, torch.Tensor)
            Every stored item with its weight.
        """
        return self._store.items()


def _carries(weight: torch.Tensor, semiring: ChartSemiring) -> torch.Tensor:
    """Whether an update is the zero and still carries a derivative.

    An update equal to the semiring's zero leaves the number unchanged,
    but at a parameter value where a weight is zero its derivative need
    not be: ``3 (1 + 2 t)`` at ``t = 0`` has derivative 6. Such an update
    has to reach the conclusions. In an algebra that establishes an
    idempotent, selective sum the zero is never selected over a stored
    weight, so its derivative is zero by the tie rule and it may stop.
    A zero that does not require a derivative is absent for every
    parameter value and stops.

    Parameters
    ----------
    weight : torch.Tensor
        The update.
    semiring : ChartSemiring
        The semiring.

    Returns
    -------
    torch.Tensor
        A Boolean of the update's shape.
    """
    if not weight.requires_grad or semiring.establishes(
        PLUS_IDEMPOTENT, PLUS_SELECTIVE
    ):
        return torch.zeros_like(weight, dtype=torch.bool)
    return semiring.is_zero(weight)


# ---------------------------------------------------------------------------
# Chart as first-class differentiable value
# ---------------------------------------------------------------------------


class ChartView:
    """A differentiable view onto a completed chart.

    Wraps the `Chart` an agenda run produced, a map from derived items to
    their weights, and exposes lookups into it:

    * `weight`: the weight of one ground item, a differentiable
      `torch.Tensor`.
    * `enumerate`: the items matching a pattern, with their weights.
    * `derivations`: the rule applications that derive an item, the
      item's incoming hyperedges in the derivation hypergraph.
    * `goal_weight`: the semiring sum of the goal items' weights.

    Gradients flow through these lookups because the stored tensors
    carry ``requires_grad`` from the rule and axiom weights.

    Parameters
    ----------
    result : AgendaResult
        The agenda run to view.
    """

    def __init__(self, result: "AgendaResult") -> None:
        self._result = result

    @property
    def chart(self) -> Chart:
        """The underlying chart."""
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
            The item's weight, ``default``, or the semiring's zero in the
            dtype and device of the run's weights.
        """
        w = self._result.chart.get(item)
        if w is not None:
            return w
        if default is not None:
            return default
        return self.semiring.zero_like((), self._result.like)

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
        ``item_encoder`` field (set by the DSL compiler when
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

    def derivations(self, item: Item) -> list[tuple[int, str, tuple[Item, ...]]]:
        """The rule applications that derive ``item``.

        These are the item's incoming hyperedges in the derivation
        hypergraph [Klein and Manning (2001)](<https://aclanthology.org/W01-1812/>):
        each derivation of ``item`` is an axiom of it, or one of these
        applications together with a derivation of each antecedent.

        Parameters
        ----------
        item : Item
            The item whose derivations to return.

        Returns
        -------
        list of (int, str, tuple of Item)
            Each application's rule position, rule name and antecedent
            items, in premise order; empty when no rule derives ``item``.
            The axiom occurrences of ``item`` are in
            ``AgendaResult.axiom_occurrences``.
        """
        return [
            (index, rule_name, antecedents)
            for index, rule_name, antecedents, conclusion in self._result.applications
            if conclusion == item
        ]

    def goal_weight(self) -> torch.Tensor:
        """Aggregate weight over all goal items.

        If a single goal item was identified, returns its weight
        directly; otherwise aggregates via the semiring's
        ``plus`` reduction (the standard parse-as-marginal
        computation).

        Returns
        -------
        torch.Tensor
            The aggregated goal weight, or the semiring's zero in the
            dtype and device of the run's weights when no goal item was
            derived.
        """
        if not self._result.goal_items:
            return self.semiring.zero_like((), self._result.like)
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
        self._work = _ComponentWork()

    def work_counts(self) -> Mapping[WorkField, int]:
        """The queue's work since it was made.

        Returns
        -------
        Mapping of WorkField to int
            Sums merging pending contributions, one per element, and
            dictionary accesses; a queue in order compares and multiplies
            nothing.
        """
        return self._work.counts()

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
        self._work.hashes += 1
        existing = self._pending.get(item)
        if existing is None:
            self._work.hashes += 1
            self._pending[item] = weight
            self._order.append(item)
            return
        stacked = torch.stack([existing, weight])
        merged = self._semiring.plus(stacked, dim=0)
        self._work.sums += merged.numel()
        self._work.hashes += 1
        self._pending[item] = merged

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
        self._work.hashes += 1
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

    def work_counts(self) -> Mapping[WorkField, int]:
        """The queue's work: none of the counted kinds.

        Returns
        -------
        Mapping of WorkField to int
            A stack sums, hashes, compares and multiplies nothing.
        """
        return dict.fromkeys(COMPONENT_FIELDS, 0)

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


class _HeapEntry:
    """An entry of a priority agenda's heap that counts its comparisons.

    Entries order by priority, then by the order they were pushed, so no
    two compare equal; each comparison the heap makes is one call of
    ``__lt__``, which the queue's counter records.

    Parameters
    ----------
    priority : float
        The negated priority, for a min-heap.
    order : int
        The push's position, the tie-breaker.
    item : Item
        The item.
    weight : torch.Tensor
        Its pending weight.
    work : _ComponentWork
        The queue's counts.
    """

    __slots__ = ("item", "order", "priority", "weight", "work")

    def __init__(
        self,
        priority: float,
        order: int,
        item: Item,
        weight: torch.Tensor,
        work: _ComponentWork,
    ) -> None:
        self.priority = priority
        self.order = order
        self.item = item
        self.weight = weight
        self.work = work

    def __lt__(self, other: _HeapEntry) -> bool:
        """Count one comparison and order by priority, then push order.

        Parameters
        ----------
        other : _HeapEntry
            Another entry of the same heap.

        Returns
        -------
        bool
            Whether this entry comes first.
        """
        self.work.comparisons += 1
        return (self.priority, self.order) < (other.priority, other.order)


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
        self._heap: list[_HeapEntry] = []
        self._counter = 0  # tie-breaker for deterministic ordering
        self._work = _ComponentWork()

    def work_counts(self) -> Mapping[WorkField, int]:
        """The queue's work since it was made.

        Returns
        -------
        Mapping of WorkField to int
            The comparisons its heap made, counted by its entries; it
            sums, hashes and multiplies nothing.
        """
        return self._work.counts()

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
        heapq.heappush(
            self._heap, _HeapEntry(priority, self._counter, item, weight, self._work)
        )

    def pop(self) -> tuple[Item, torch.Tensor]:
        """Remove and return the next item under this discipline.

        Returns
        -------
        tuple of (Item, torch.Tensor)
            The next item and its pending weight.
        """
        entry = heapq.heappop(self._heap)
        return entry.item, entry.weight

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


class AgendaResult(QiecModel):
    """The chart produced by an agenda-driven deduction.

    Attributes
    ----------
    chart : Chart
        The completed chart: each derived item's weight.
    semiring : ChartSemiring
        The semiring used for weight aggregation.
    goal_items : list of (Item, torch.Tensor)
        Items in the chart matching the goal predicate, with
        their final aggregated weights.
    iterations : int
        Number of agenda steps consumed (diagnostic).
    applications : list of (int, str, tuple of Item, Item)
        Every rule application of the run: the rule's position in the
        system, its name, its antecedent items in premise order, and its
        conclusion. Two rules of one name are told apart by position.
    axiom_occurrences : list of Item
        The item of every axiom occurrence, in the order given, so equal
        axioms are two occurrences.
    like : torch.Tensor
        An empty tensor of the dtype and device of the run's axiom
        weights, which the zeros and units the result makes take; the
        default dtype when the run had no axiom.
    work : WorkReport
        The run's work: its pops and recorded applications; its pattern
        matches, sums, products and hash operations together with the
        chart's and the queue's, each measured when the run and both
        report it (`ReportsWork`); products only when no rule has a
        ``weight_fn``, whose products are its own; and the elements saved
        for the backward pass when measured.
    """

    chart: Chart = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    semiring: ChartSemiring = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    goal_items: list[tuple[tuple[HeldValue, ...], torch.Tensor]] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    iterations: int = 0
    applications: list[
        tuple[int, str, tuple[tuple[HeldValue, ...], ...], tuple[HeldValue, ...]]
    ] = dx.field(default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY)
    axiom_occurrences: list[tuple[HeldValue, ...]] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    like: torch.Tensor = dx.field(
        default_factory=lambda: torch.empty(0), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    encoder: Encoder | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    # Sum of all rule-attached + chart-attached loss values fired
    # during this run, populated by `DeductionSystem.run` when a
    # loss registry is attached. `None` if no losses fired.
    attached_loss: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    work: WorkReport = dx.field(
        default_factory=lambda: WorkReport(phase=Phase.EXECUTION),
        opaque=True,
        extras=OPAQUE_BY_VALUE,
    )


class _AgendaWork:
    """The counts an agenda run keeps of its own operations."""

    def __init__(self) -> None:
        self.pops = 0
        self.matches = 0
        self.hashes = 0
        self.products = 0


def _since(
    component: Chart | Agenda, before: Mapping[WorkField, int] | None
) -> dict[WorkField, int] | None:
    """The work a chart or queue did since earlier counts.

    Parameters
    ----------
    component : Chart or Agenda
        The component.
    before : Mapping of WorkField to int or None
        Its counts before the run, ``None`` when it reports no work.

    Returns
    -------
    dict of WorkField to int or None
        The difference for each counter it measures, or ``None`` when it
        reports no work.
    """
    if before is None or not isinstance(component, ReportsWork):
        return None
    after = component.work_counts()
    return {field: count - before.get(field, 0) for field, count in after.items()}


def _agenda_report(
    work: _AgendaWork,
    *,
    applications: int,
    occurrences: int,
    products_measured: bool,
    saved: int | None,
    components: Sequence[Mapping[WorkField, int] | None],
) -> WorkReport:
    """The work report of an agenda run.

    The run's own counts are its pops, its pattern matches, its recorded
    provenance, its products when every product is its own, and the tape
    when measured; it sums and hashes nothing itself. The chart's and the
    queue's reports add their sums, products, hash operations and
    comparisons. Each of those is measured only when the run and every
    component measured it; a component that reports no work leaves them
    unmeasured.

    Parameters
    ----------
    work : _AgendaWork
        The run's own counts.
    applications : int
        Applications recorded, each counting one plus its premises.
    occurrences : int
        The axiom occurrences.
    products_measured : bool
        Whether every product went through the run's own calls, which
        holds when no rule has a ``weight_fn``.
    saved : int or None
        Elements saved for the backward pass, or ``None`` when not
        measured.
    components : Sequence of Mapping of WorkField to int, or None
        The chart's and the queue's work during the run, ``None`` for one
        that reports none.

    Returns
    -------
    WorkReport
        The report, with exactly the counters measured.
    """
    own = {
        WorkField.SCALAR_SUMS: 0,
        WorkField.SCALAR_PRODUCTS: work.products,
        WorkField.HASH_OPERATIONS: work.hashes,
        WorkField.KEY_COMPARISONS: work.matches,
    }
    measured = {WorkField.ITERATIONS, WorkField.PROVENANCE_RECORDS}
    if saved is not None:
        measured.add(WorkField.ELEMENTS_SAVED_FOR_BACKWARD)
    totals: dict[WorkField, int] = {}
    for field, count in own.items():
        if field is WorkField.SCALAR_PRODUCTS and not products_measured:
            continue
        if all(report is not None and field in report for report in components):
            measured.add(field)
            totals[field] = count + sum(
                report[field] for report in components if report is not None
            )
    return WorkReport(
        phase=Phase.EXECUTION,
        iterations=work.pops,
        scalar_sums=totals.get(WorkField.SCALAR_SUMS, 0),
        scalar_products=totals.get(WorkField.SCALAR_PRODUCTS, 0),
        hash_operations=totals.get(WorkField.HASH_OPERATIONS, 0),
        key_comparisons=totals.get(WorkField.KEY_COMPARISONS, 0),
        provenance_records=applications + occurrences,
        elements_saved_for_backward=0 if saved is None else saved,
        measured=measuring(measured),
    )


def run_agenda(
    axioms: Iterable[tuple[Item, torch.Tensor]],
    rules: Iterable[InferenceRule],
    semiring: ChartSemiring | None = None,
    agenda: Agenda | None = None,
    goal: Callable[[Item], bool] | None = None,
    max_iterations: int = 100_000,
    chart: Chart | None = None,
    rule_callback: RuleCallback | None = None,
    *,
    measure_tape: bool = False,
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

    measure_tape : bool, optional
        Whether to count the elements the run saves for the backward
        pass, by a hook on the autograd graph that costs time.

    Returns
    -------
    AgendaResult
        The final chart, the goal items, the step count, and the work
        report of the run.

    Raises
    ------
    RuntimeError
        If the agenda runs for more than ``max_iterations`` steps.
    """
    semiring = semiring or LOG_PROB
    agenda = agenda or FIFOAgenda()
    agenda.bind_semiring(semiring)
    if chart is None:
        chart = HashChart()
    rules = tuple(rules)
    applications: dict[tuple[int, tuple[Item, ...]], _Application] = {}

    chart_before = chart.work_counts() if isinstance(chart, ReportsWork) else None
    queue_before = agenda.work_counts() if isinstance(agenda, ReportsWork) else None
    occurrences: list[Item] = []
    weights_seen: list[torch.Tensor] = []
    for item, w in axioms:
        occurrences.append(item)
        weights_seen.append(w)
        agenda.push(item, w)
    like = (
        torch.empty(0, dtype=weights_seen[0].dtype, device=weights_seen[0].device)
        if weights_seen
        else torch.empty(0)
    )

    work = _AgendaWork()
    joins = _RuleJoins(rules, work)
    for present, _ in list(chart.items()):
        joins.add(present, joins.matches(present))
    saved: list[int] = []
    with _tape(saved) if measure_tape else nullcontext():
        _drain(
            agenda,
            chart,
            rules,
            semiring,
            applications,
            work,
            max_iterations,
            joins,
        )
    iterations = work.pops
    if rule_callback is not None:
        for application in applications.values():
            weights = tuple(_chart_weight(chart, a) for a in application.antecedents)
            rule_callback(
                application.rule.name,
                list(zip(application.antecedents, weights)),
                application.conclusion,
                _conclusion_weight(
                    application.rule,
                    application.bindings,
                    weights,
                    semiring,
                    like,
                    work,
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
        applications=[
            (index, a.rule.name, a.antecedents, a.conclusion)
            for (index, _), a in applications.items()
        ],
        axiom_occurrences=occurrences,
        like=like,
        work=_agenda_report(
            work,
            applications=sum(
                1 + len(application.antecedents)
                for application in applications.values()
            ),
            occurrences=len(occurrences),
            products_measured=all(rule.weight_fn is None for rule in rules),
            saved=sum(saved) if measure_tape else None,
            components=(_since(chart, chart_before), _since(agenda, queue_before)),
        ),
    )


def _pattern_variables(pattern: Pattern) -> list[str]:
    """The wildcard names of a pattern, in order of first occurrence.

    Parameters
    ----------
    pattern : Pattern
        The pattern.

    Returns
    -------
    list of str
        Each name once.
    """
    if isinstance(pattern, Wildcard):
        return [pattern.name]
    found: list[str] = []
    if isinstance(pattern, tuple):
        for element in pattern:
            for name in _pattern_variables(element):
                if name not in found:
                    found.append(name)
    return found


type Form = tuple[HeldValue, int]
"""The head and length of a tuple item or pattern."""


def _form(value: Pattern) -> Form | None:
    """The form a pattern requires of its items, or the form of an item.

    Parameters
    ----------
    value : Pattern
        A pattern or an item.

    Returns
    -------
    Form or None
        The ground head and the length of a tuple, or ``None`` when the
        value is not a tuple or its head is a wildcard, so its items may
        have any form.
    """
    if isinstance(value, tuple) and value and _ground(value[0], {}):
        return (value[0], len(value))
    return None


class _RuleJoins:
    """The premises each form of item fills, and tries of their matches.

    Each rule premise is filed under the form its pattern requires, or
    among the premises any item may fill. An item is matched only against
    the premises of its form and those; the bindings of each match are a
    tuple of that premise's relation. For a rule and the premise an
    arriving item fills, every other premise has a trie of its relation
    keyed in the order that binds the arriving premise's variables
    first, kept up to date as items arrive, and the other premises are
    joined by Generic Join.

    Parameters
    ----------
    rules : tuple of InferenceRule
        The rules.
    work : _AgendaWork
        The run's counts.
    """

    def __init__(self, rules: tuple[InferenceRule, ...], work: _AgendaWork) -> None:
        self.rules = rules
        self.work = work
        self._by_form: dict[Form, list[tuple[int, int]]] = {}
        self._any: list[tuple[int, int]] = []
        for rule_index, rule in enumerate(rules):
            for position, pattern in enumerate(rule.premises):
                form = _form(pattern)
                target = (
                    self._any if form is None else self._by_form.setdefault(form, [])
                )
                target.append((rule_index, position))
        self._relations: dict[tuple[int, int], list[tuple[Item, Bindings]]] = {}
        self._tries: dict[
            tuple[int, int, int], tuple[tuple[str, ...], Trie[HeldValue, Item]]
        ] = {}
        self._orders: dict[tuple[int, int], tuple[str, ...]] = {}

    def matches(self, item: Item) -> list[tuple[int, int, Bindings]]:
        """The premises an item fills, with the bindings of each.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        list of tuple of int, int and Bindings
            The rule, the premise and the bindings of each match.
        """
        form = _form(item)
        candidates = [*self._by_form.get(form, ()), *self._any] if form else self._any
        found: list[tuple[int, int, Bindings]] = []
        for rule_index, position in candidates:
            self.work.matches += 1
            bindings = match(self.rules[rule_index].premises[position], item)
            if bindings is not None:
                found.append((rule_index, position, bindings))
        return found

    def add(self, item: Item, matched: Sequence[tuple[int, int, Bindings]]) -> None:
        """Record an item that has entered the chart in the relations it fills.

        Parameters
        ----------
        item : Item
            The item, new to the chart.
        matched : Sequence of tuple of int, int and Bindings
            Its matches.
        """
        for rule_index, position, bindings in matched:
            self._relations.setdefault((rule_index, position), []).append(
                (item, bindings)
            )
            for trigger in range(len(self.rules[rule_index].premises)):
                key = (rule_index, trigger, position)
                if trigger != position and key in self._tries:
                    variables, root = self._tries[key]
                    self._insert(root, variables, bindings, item)

    def _insert(
        self,
        root: Trie[HeldValue, Item],
        variables: tuple[str, ...],
        bindings: Bindings,
        item: Item,
    ) -> None:
        """Add one tuple to a trie, counting its accesses.

        Parameters
        ----------
        root : Trie
            The trie.
        variables : tuple of str
            Its variables in order.
        bindings : Bindings
            The tuple's bindings.
        item : Item
            The item it stands for.
        """
        path = [bindings[name] for name in variables] or [()]
        self.work.hashes += len(path)
        insert(root, path, item)

    def _order(self, rule_index: int, trigger: int) -> tuple[str, ...]:
        """The variable order of a rule's join from one premise.

        Parameters
        ----------
        rule_index : int
            The rule.
        trigger : int
            The premise the arriving item fills.

        Returns
        -------
        tuple of str
            The premise's variables, then the others in order of first
            occurrence.
        """
        key = (rule_index, trigger)
        if key not in self._orders:
            rule = self.rules[rule_index]
            order = _pattern_variables(rule.premises[trigger])
            for pattern in rule.premises:
                for name in _pattern_variables(pattern):
                    if name not in order:
                        order.append(name)
            self._orders[key] = tuple(order)
        return self._orders[key]

    def _trie(
        self, rule_index: int, trigger: int, position: int
    ) -> tuple[tuple[str, ...], Trie[HeldValue, Item]]:
        """The trie of one premise's relation in a trigger's order, built on first use.

        Parameters
        ----------
        rule_index : int
            The rule.
        trigger : int
            The premise the arriving item fills.
        position : int
            The premise of the trie.

        Returns
        -------
        tuple of tuple of str and Trie
            The premise's variables in order, and the trie.
        """
        key = (rule_index, trigger, position)
        found = self._tries.get(key)
        if found is None:
            order = self._order(rule_index, trigger)
            names = set(_pattern_variables(self.rules[rule_index].premises[position]))
            variables = tuple(name for name in order if name in names)
            root: Trie[HeldValue, Item] = {}
            for item, bindings in self._relations.get((rule_index, position), ()):
                self._insert(root, variables, bindings, item)
            found = (variables, root)
            self._tries[key] = found
        return found

    def instances(
        self, rule_index: int, trigger: int, bindings: Bindings
    ) -> Iterator[Bindings]:
        """The bindings of every instance an arriving item completes at one premise.

        Parameters
        ----------
        rule_index : int
            The rule.
        trigger : int
            The premise the arriving item fills.
        bindings : Bindings
            The bindings of its match.

        Yields
        ------
        Bindings
            Each instance's bindings of every premise's variables.
        """
        rule = self.rules[rule_index]
        others = [p for p in range(len(rule.premises)) if p != trigger]
        tries = [self._trie(rule_index, trigger, p) for p in others]
        nodes: list[Trie[HeldValue, Item] | Item] = []
        for variables, root in tries:
            path = [bindings[name] for name in variables if name in bindings] or (
                [()] if not variables else []
            )
            self.work.hashes += max(len(path), 1)
            node = descend(root, path)
            if node is None:
                return
            nodes.append(node)
        order = self._order(rule_index, trigger)
        free = [name for name in order if name not in bindings]
        containing = [
            [index for index, (variables, _) in enumerate(tries) if name in variables]
            for name in free
        ]
        counter = TrieCounter()
        counted = 0
        for full, _ in intersect(free, containing, nodes, dict(bindings), counter):
            seen = counter.iterated + counter.probes
            self.work.matches += seen - counted
            counted = seen
            yield full
        self.work.matches += counter.iterated + counter.probes - counted


def _drain(
    agenda: Agenda,
    chart: Chart,
    rules: tuple[InferenceRule, ...],
    semiring: ChartSemiring,
    applications: dict[tuple[int, tuple[Item, ...]], _Application],
    work: _AgendaWork,
    max_iterations: int,
    joins: _RuleJoins,
) -> None:
    """Pop and fire until the agenda is empty.

    Parameters
    ----------
    agenda : Agenda
        The agenda.
    chart : Chart
        The chart.
    rules : tuple of InferenceRule
        The rules.
    semiring : ChartSemiring
        The semiring.
    applications : dict
        The applications seen so far, extended in place.
    work : _AgendaWork
        The counts, extended in place.
    max_iterations : int
        The bound on pops.
    joins : _RuleJoins
        The premises each item fills and the tries of their relations.

    Raises
    ------
    RuntimeError
        If the agenda runs for more than ``max_iterations`` steps.
    """
    while not agenda.empty():
        if work.pops >= max_iterations:
            raise RuntimeError(
                f"agenda exceeded max_iterations={max_iterations}; "
                f"likely a non-terminating deduction"
            )
        work.pops += 1
        item, delta = agenda.pop()
        old = chart.get(item)
        changed = chart.insert_or_aggregate(item, delta, semiring)
        if not changed:
            continue
        if old is not None:
            delta = chart.propagated(old, item, delta, semiring)
        update = _Update(item=item, delta=delta, old=old)
        matched = joins.matches(item)
        if old is None:
            joins.add(item, matched)
        for rule_index, premise_idx, bindings in matched:
            rule = rules[rule_index]
            for full in joins.instances(rule_index, premise_idx, bindings):
                _fire(
                    rule=rule,
                    rule_index=rule_index,
                    bindings=full,
                    fixed_idx=premise_idx,
                    update=update,
                    chart=chart,
                    semiring=semiring,
                    agenda=agenda,
                    applications=applications,
                    work=work,
                )


@contextmanager
def _tape(saved: list[int]) -> Iterator[None]:
    """Count the elements saved for the backward pass inside the block.

    Parameters
    ----------
    saved : list of int
        Receives one entry per saved tensor.

    Yields
    ------
    None
        While the hook is installed.
    """

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Count a saved tensor and keep it.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        saved.append(tensor.numel())
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        """Return a saved tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


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


def _conclusion_weight(
    rule: InferenceRule,
    bindings: Bindings,
    premise_weights: tuple[torch.Tensor, ...],
    semiring: ChartSemiring,
    like: torch.Tensor,
    work: _AgendaWork,
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
    like : torch.Tensor
        A tensor of the run's dtype and device, which the unit of an
        application without premises takes.
    work : _AgendaWork
        The run's counts; a ``weight_fn``'s products are its own.

    Returns
    -------
    torch.Tensor
        The rule's ``weight_fn`` applied to the premise weights, or the
        semiring product of the premise weights when it has none.
    """
    if rule.weight_fn is not None:
        return rule.weight_fn(bindings, premise_weights, semiring)
    if not premise_weights:
        return semiring.unit_like((), like)
    acc = premise_weights[0]
    for w in premise_weights[1:]:
        acc = semiring.times(acc, w)
        work.products += acc.numel()
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
    applications: dict[tuple[int, tuple[Item, ...]], _Application],
    work: _AgendaWork,
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
    applications : dict
        The rule applications seen so far, extended with this one.
    work : _AgendaWork
        The run's counts.
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
        _conclusion_weight(
            rule, bindings, tuple(factors), semiring, update.delta, work
        ),
    )


# ---------------------------------------------------------------------------
# Public deduction-system entry
# ---------------------------------------------------------------------------


class DeductionSystem(QiecModel):
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

    Attributes
    ----------
    name : str
        The declaration name used to look up an attached loss registry.
    modules : dict[str, torch.nn.Module]
        Owned parameter modules, keyed by their stable parameter-name prefixes.
    item_signature : Signature | None
        The signature of structured items, when declared.
    item_encoder : Encoder | None
        The encoder used by chart embedding queries.
    losses : dict[str, LossRegistry]
        Shared loss registries keyed by deduction name. Compilation can attach
        a registry after another program has captured this system.
    goal_extent : Callable or None
        The positions of the input an item covers, declared for an item
        type whose items cover stretches of the input, ``None`` for an
        item that covers none. A goal of such a type has to cover the
        whole input. Without a declaration the goal is the predicate
        alone.
    mappable : bool
        Whether the chart can be filled at many parameter values in one
        pass: the order of the agenda, the rules that fire, and the axioms
        depend on the items alone, and the weights are computed by tensor
        operations. The DSL compiler declares it of the systems it builds.
    presentation : RulePresentation | None
        The rule presentation the rules are derived from: each rule is
        the rule of the presentation at its position, spelled in the
        chart representation. ``None`` for a system built from rules
        directly.
    """

    rules: tuple[InferenceRule, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    semiring: ChartSemiring = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    axiom_injector: Callable[[HeldValue], list[tuple[Item, torch.Tensor]]] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    goal: Callable[[Item], bool] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    agenda_factory: Callable[[], Agenda] = dx.field(
        default=FIFOAgenda, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    chart_factory: Callable[..., Chart] = dx.field(
        default=HashChart, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    max_iterations: int = 100_000
    tolerance: float = 0.0
    mappable: bool = False
    productions: ProductionWeights | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    production_refusal: str | None = None
    goal_extent: Callable[[Item], tuple[int, int] | None] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    name: str = ""
    modules: dict[str, torch.nn.Module] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    item_signature: Signature | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    item_encoder: Encoder | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    losses: dict[str, LossRegistry] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    presentation: RulePresentation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

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

    def _goal_for(self, input_value: HeldValue) -> Callable[[Item], bool]:
        """The goal predicate of one run.

        Parameters
        ----------
        input_value : HeldValue
            The input the system is run on.

        Returns
        -------
        Callable[[Item], bool]
            The declared goal, narrowed by the declared extent to items
            covering the whole input when the input is a sequence.
        """
        extent = self.goal_extent
        goal = self.goal
        if extent is None or not isinstance(input_value, (list, tuple)):
            return goal
        whole = (0, len(input_value))

        def covering(item: Item) -> bool:
            """Whether an item is a goal covering the whole input.

            Parameters
            ----------
            item : Item
                The chart item.

            Returns
            -------
            bool
                The declared predicate's answer, and, for an item that
                covers a stretch of the input, that it covers all of it.
            """
            if not goal(item):
                return False
            covered = extent(item)
            return covered is None or covered == whole

        return covering

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
        if self.presentation is not None and self.presentation.computed_weight_rules:
            raise ValueError("computed rule weights require checked QVR execution")
        axioms = self.axiom_injector(input_value)
        registry = self.losses.get(self.name)
        deduction_name = self.name
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

        chart_inst = self.chart_factory(tolerance=self.tolerance)
        result = run_agenda(
            axioms=axioms,
            rules=self.rules,
            semiring=self.semiring,
            agenda=self.agenda_factory(),
            goal=self._goal_for(input_value),
            max_iterations=self.max_iterations,
            chart=chart_inst,
            rule_callback=(_rule_callback if registry is not None else None),
        )
        # Propagate any attached item-encoder to the result.
        comp = self.item_encoder
        if comp is not None:
            result = result.with_(encoder=comp)
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
            result = result.with_(attached_loss=total)
        return result

    def __call__(
        self,
        input_value: Any,
        probabilities: Mapping[str, torch.Tensor] | None = None,
    ) -> ChartView:
        """Run the deduction and return a `ChartView`.

        The chart's weights are differentiable tensors, and the
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
            mod = self.modules.get(attr)
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
            mod = self.modules.get(attr)
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
    "COMPONENT_FIELDS",
    "ReportsWork",
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
