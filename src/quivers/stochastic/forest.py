"""Packed forests of derivations, with occurrence-sensitive provenance.

A packed forest is the derived hypergraph of a run: every item, and for
each item every grounded rule instance concluding it and every input
occurrence of it, as an edge to its premise items. Two occurrences of
one item stay two edges, so the derivations a forest unfolds into are
the derivations of the presentation, occurrence identities included,
one to one.

Three kinds of observation are computed from a forest.

- A linear proof fold, the sum over derivations of the product of a
  value per edge, by dynamic programming over the forest in premise
  order; an expectation or other carrier makes it a statistic.
- A nonlinear fold, the sum over derivations of any function of the
  derivation and its weight, by enumerating the derivations. It needs
  them all, so a declared limit on their number is checked before any
  is built, and a forest with more is refused, never truncated.
- An expected proof embedding, the weighted mean of a fold of each
  derivation into vectors, by enumeration in which each shared
  sub-derivation is folded once.

A cyclic forest has infinitely many derivations; it is unfolded only
under a declared bound on their height.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from enum import Enum

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import OrderPreservingSemiring
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import (
    OccurrenceId,
    ParameterId,
    Phase,
    WorkField,
    WorkReport,
    measuring,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    ComputedRuleWeight,
    Derivation,
    Ground,
    HostInterpretation,
    Item,
    PremiseValue,
    Rule,
    RulePresentation,
    evaluate_index,
)
from quivers.stochastic.indexed import ChartWork, IndexedChart
from quivers.structural.encoder import Encoder
from quivers.structural.signature import DataLeaf, Signature

__all__ = [
    "DerivationForest",
    "ForestEdge",
    "ForestRefusal",
    "ForestRefused",
    "build_forest",
    "edge_weight",
    "encoder_interpretation",
    "expected_embedding",
    "fold_derivations",
    "fold_derivations_many",
    "forest_fold",
    "forest_folds",
    "nonlinear_fold",
]


class ForestRefusal(str, Enum):
    """Why a forest is not unfolded."""

    TOO_MANY = "too-many-derivations"
    TOO_MANY_NODES = "too-many-proof-nodes"
    UNBOUNDED = "unbounded"
    MISSING = "missing"


class ForestRefused(ValueError):
    """A forest is not unfolded, with the reason.

    Parameters
    ----------
    code : ForestRefusal
        The kind of reason.
    reason : str
        The reason.
    """

    def __init__(self, code: ForestRefusal, reason: str) -> None:
        super().__init__(f"{code.value}: {reason}")
        self.code = code
        self.reason = reason


class ForestEdge(QiecModel):
    """One way an item is derived: a rule instance or an input occurrence.

    Parameters
    ----------
    constructor : ConstructorId
        The rule, or the input's slot.
    binding : tuple of tuple of str and Ground
        The rule's or slot's variables.
    premises : tuple of Item
        The premise items in premise order; none for an input.
    occurrence : OccurrenceId or None
        The input occurrence; ``None`` for a rule instance.
    values : tuple of str or int or float or bool
        The occurrence's values; none for a rule instance.
    """

    constructor: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    binding: tuple[tuple[str, Ground], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    premises: tuple[Item, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    occurrence: OccurrenceId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    values: tuple[str | int | float | bool, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class DerivationForest(ValueModel):
    """The derived hypergraph of a run.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    items : tuple of Item
        Every item, in the order discovered.
    edges : dict of Item to tuple of ForestEdge
        The edges concluding each item: its input occurrences, then its
        rule instances, in the order found.
    order : tuple of Item or None
        The items in an order in which every premise precedes its
        conclusions, or ``None`` when the hypergraph has a cycle.
    work : ChartWork
        The counters of the chart that found the edges.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    items: tuple[Item, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    edges: dict[Item, tuple[ForestEdge, ...]] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    order: tuple[Item, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    work: ChartWork = dx.field(
        default=ChartWork(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def acyclic(self) -> bool:
        """Whether the hypergraph has no cycle.

        Returns
        -------
        bool
            True when every item has finitely many derivations.
        """
        return self.order is not None

    @property
    def size(self) -> int:
        """The number of edges and premise references, the forest's size.

        Returns
        -------
        int
            ``sum_e (1 + |premises(e)|)``.
        """
        return sum(
            1 + len(edge.premises) for edges in self.edges.values() for edge in edges
        )

    def report(self) -> WorkReport:
        """The work of building the forest.

        Returns
        -------
        WorkReport
            The chart's lookups and candidates, and the forest's size as
            its provenance, each measured.
        """
        chart = self.work.report(Phase.CONSTRUCTION)
        return chart.with_(
            provenance_records=self.size,
            measured=measuring((*chart.measured, WorkField.PROVENANCE_RECORDS)),
        )

    def count(self, goal: Item, *, height: int | None = None) -> int:
        """The number of derivations of an item, exactly.

        Parameters
        ----------
        goal : Item
            The item.
        height : int or None
            The greatest height of a derivation counted; required when
            the forest has a cycle. An input has height one.

        Returns
        -------
        int
            The count, zero for an item not in the forest.

        Raises
        ------
        ForestRefused
            If the forest has a cycle and no height is given.
        """
        if goal not in self.edges:
            return 0
        counts = _counts(self.below(goal), height)
        return counts.get((goal, height), 0)

    def below(self, goal: Item) -> DerivationForest:
        """The part of the forest that derivations of one item use.

        Parameters
        ----------
        goal : Item
            The item.

        Returns
        -------
        DerivationForest
            The items reachable from ``goal`` through premises, with
            their edges, in the same orders.
        """
        return self.below_all((goal,))

    def below_all(self, goals: Sequence[Item]) -> DerivationForest:
        """The part of the forest used by derivations of any of the goals.

        Parameters
        ----------
        goals : Sequence of Item
            The requested items.

        Returns
        -------
        DerivationForest
            The items reachable from the goals through premises, with their
            edges, in the same orders.
        """
        reached = set(goals)
        stack = list(reached)
        while stack:
            item = stack.pop()
            for edge in self.edges.get(item, ()):
                for premise in edge.premises:
                    if premise not in reached:
                        reached.add(premise)
                        stack.append(premise)
        return DerivationForest(
            presentation=self.presentation,
            items=tuple(item for item in self.items if item in reached),
            edges={item: self.edges[item] for item in reached if item in self.edges},
            order=None
            if self.order is None
            else tuple(item for item in self.order if item in reached),
        )


type Key = tuple[Item, int | None]
"""An item and the bound on the height of its derivations, ``None`` for none."""


def _unbounded() -> ForestRefused:
    """The refusal of a cyclic forest without a bound on height.

    Returns
    -------
    ForestRefused
        The refusal.
    """
    return ForestRefused(
        ForestRefusal.UNBOUNDED,
        "the forest has a cycle, so its items have infinitely many derivations; "
        "give a bound on their height",
    )


def _levels(forest: DerivationForest, height: int | None) -> list[list[Key]]:
    """The keys to compute, in an order in which premises come first.

    Without a height, one level of the items in order; with one, a level
    per height from one up, each holding every item.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    height : int or None
        The bound on height.

    Returns
    -------
    list of list of Key
        The levels.

    Raises
    ------
    ForestRefused
        If the forest has a cycle and no height is given.
    """
    if height is None:
        if forest.order is None:
            raise _unbounded()
        return [[(item, None) for item in forest.order]]
    return [[(item, level) for item in forest.items] for level in range(1, height + 1)]


def _below(key: Key) -> int | None:
    """The height bound of a key's premises.

    Parameters
    ----------
    key : Key
        The key.

    Returns
    -------
    int or None
        One less, or ``None`` without a bound.
    """
    return None if key[1] is None else key[1] - 1


def _counts(
    forest: DerivationForest,
    height: int | None,
    *,
    cap: int | None = None,
    sizes: dict[Key, int] | None = None,
) -> dict[Key, int]:
    """Count derivations of every item, optionally saturating the count.

    Over nonnegative integers, truncation at a positive cap preserves
    addition and multiplication followed by truncation. A product of a
    saturated count with zero is still zero. Each operand is thus at most
    the cap, even when the exact count has exponentially many bits.

    To count unfolded node occurrences as well, compose a premise count
    and node sum `(c, s)` with the partial Cartesian product `(p, q)` as
    `(p*c, p*s + q*c)`. This accounts for each premise position, even
    when positions share an item. Add one root node per combination.
    Truncation is sound for both coordinates since these formulas use
    only addition and multiplication of nonnegative integers.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    height : int or None
        The bound on height.
    cap : int or None
        An optional positive saturation threshold for counts.
    sizes : dict of Key to int or None
        An optional empty table to receive total unfolded node counts,
        with the same cap if supplied.

    Returns
    -------
    dict of Key to int
        The count, limited by the cap when supplied, of each key with at
        least one derivation.

    Raises
    ------
    ForestRefused
        If the forest has a cycle and no height is given.
    """
    counts: dict[Key, int] = {}
    for level in _levels(forest, height):
        for key in level:
            total = 0
            nodes = 0
            below = _below(key)
            for edge in forest.edges.get(key[0], ()):
                product = 1
                children = 0
                if edge.occurrence is None:
                    if below is not None and below < 0:
                        continue
                    for premise in edge.premises:
                        child = (premise, below)
                        alternatives = counts.get(child, 0)
                        if sizes is not None:
                            children = children * alternatives + product * sizes.get(
                                child, 0
                            )
                            if cap is not None:
                                children = min(children, cap)
                        product *= alternatives
                        if cap is not None:
                            product = min(product, cap)
                total += product
                if sizes is not None:
                    nodes += children + product
                if cap is not None:
                    total = min(total, cap)
                    nodes = min(nodes, cap)
                    if total == cap and (sizes is None or nodes == cap):
                        break
            if total:
                counts[key] = total
                if sizes is not None:
                    sizes[key] = nodes
    return counts


def _useful(
    forest: DerivationForest,
    goals: Sequence[Item],
    height: int | None,
    counts: Mapping[Key, int],
) -> set[Key]:
    """The keys whose derivations some derivation of a requested goal uses.

    A key is useful when it is the goal's, or a premise of a live edge of
    a useful key, one whose premises all have derivations within the
    bound. Every derivation of a useful key then extends to a derivation
    of the goal, and distinct ones to distinct ones, so no useful key has
    more derivations than the goal.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goals : Sequence of Item
        The requested items.
    height : int or None
        The bound on height.
    counts : Mapping of Key to int
        The counts.

    Returns
    -------
    set of Key
        The useful keys.
    """
    useful = {(goal, height) for goal in goals if (goal, height) in counts}
    stack = list(useful)
    while stack:
        key = stack.pop()
        below = _below(key)
        for edge in forest.edges.get(key[0], ()):
            if edge.occurrence is not None:
                continue
            premises = [(premise, below) for premise in edge.premises]
            if not all(premise in counts for premise in premises):
                continue
            for premise in premises:
                if premise not in useful:
                    useful.add(premise)
                    stack.append(premise)
    return useful


# ---------------------------------------------------------------------------
# building


def build_forest(
    presentation: RulePresentation, axioms: Sequence[AxiomOccurrence]
) -> DerivationForest:
    """The packed forest of a presentation over input occurrences.

    Items are discovered from the inputs, and each instance is found once,
    by the last of its premises to be discovered, and stored as an edge of
    its conclusion. Rules without premises are edges of their ground
    conclusions. The forest is then ordered so premises precede
    conclusions, when it has no cycle.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation; no premise may have hypotheses.
    axioms : Sequence of AxiomOccurrence
        The input occurrences, each kept as its own edge.

    Returns
    -------
    DerivationForest
        The forest.

    Raises
    ------
    ForestRefused
        If an occurrence names no slot of the presentation.
    """
    chart = IndexedChart(presentation)
    edges: dict[Item, list[ForestEdge]] = {}
    discovered: list[Item] = []
    seen: set[Item] = set()
    queue: deque[Item] = deque()

    def reach(item: Item) -> None:
        """Note an item, queueing it when new.

        Parameters
        ----------
        item : Item
            The item.
        """
        if item not in seen:
            seen.add(item)
            discovered.append(item)
            queue.append(item)

    for occurrence in axioms:
        slot = presentation.constructors_by_id.get(occurrence.slot)
        if not isinstance(slot, AxiomSlot):
            raise ForestRefused(ForestRefusal.MISSING, "an occurrence names no slot")
        binding = dict(occurrence.binding)
        item = chart.items.intern(
            slot.judgment.family,
            tuple(evaluate_index(a, binding) for a in slot.judgment.arguments),
        )
        _add_input(
            edges,
            item,
            ForestEdge(
                constructor=slot.id,
                binding=occurrence.binding,
                occurrence=occurrence.occurrence,
                values=occurrence.values,
            ),
        )
        reach(item)
    for rule in presentation.rules:
        if rule.premises or not chart.holds(rule, {}):
            continue
        item = chart.items.intern(
            rule.conclusion.family,
            tuple(evaluate_index(a, {}) for a in rule.conclusion.arguments),
        )
        edges.setdefault(item, []).append(ForestEdge(constructor=rule.id))
        reach(item)
    admitted: set[Item] = set()
    by_premise = presentation.rules_by_premise
    while queue:
        item = queue.popleft()
        admitted.add(item)
        chart.add(item)
        for rule, position in by_premise.get(item.family, ()):
            for binding, premises in chart.instances(
                rule, position, item, admitted.__contains__
            ):
                conclusion = chart.items.intern(
                    rule.conclusion.family,
                    tuple(
                        evaluate_index(a, binding) for a in rule.conclusion.arguments
                    ),
                )
                edges.setdefault(conclusion, []).append(
                    ForestEdge(
                        constructor=rule.id,
                        binding=tuple(sorted(binding.items())),
                        premises=premises,
                    )
                )
                reach(conclusion)
    frozen = {item: tuple(found) for item, found in edges.items()}
    return DerivationForest(
        presentation=presentation,
        items=tuple(discovered),
        edges=frozen,
        order=_topological(discovered, frozen),
        work=chart.counters(),
    )


def _add_input(
    edges: dict[Item, list[ForestEdge]], item: Item, edge: ForestEdge
) -> None:
    """Add an input occurrence as its own edge of its item.

    Two occurrences of one item, even of one slot at one binding, are two
    edges: each is a distinct derivation.

    Parameters
    ----------
    edges : dict of Item to list of ForestEdge
        The edges so far.
    item : Item
        The occurrence's item.
    edge : ForestEdge
        Its edge.
    """
    edges.setdefault(item, []).append(edge)


def _topological(
    items: Sequence[Item], edges: Mapping[Item, tuple[ForestEdge, ...]]
) -> tuple[Item, ...] | None:
    """Order items so every premise precedes its conclusions, if possible.

    Parameters
    ----------
    items : Sequence of Item
        The items.
    edges : Mapping of Item to tuple of ForestEdge
        The edges of each item.

    Returns
    -------
    tuple of Item or None
        The order, or ``None`` when there is a cycle.
    """
    waiting = {item: 0 for item in items}
    users: dict[Item, list[Item]] = {}
    for item in items:
        for edge in edges.get(item, ()):
            for premise in edge.premises:
                waiting[item] += 1
                users.setdefault(premise, []).append(item)
    ready = deque(item for item in items if waiting[item] == 0)
    order: list[Item] = []
    while ready:
        item = ready.popleft()
        order.append(item)
        for user in users.get(item, ()):
            waiting[user] -= 1
            if waiting[user] == 0:
                ready.append(user)
    return tuple(order) if len(order) == len(items) else None


# ---------------------------------------------------------------------------
# weights and linear folds


type InputId = ParameterId | OccurrenceId
"""A rule's parameter, or an input occurrence."""


def edge_weight(
    forest: DerivationForest,
    edge: ForestEdge,
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """The weight of an edge, as a plan weighs it.

    An input occurrence takes its own weight, else its slot's parameter;
    a rule instance takes its rule's parameter; a missing weight is the
    unit.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    edge : ForestEdge
        The edge.
    algebra : OrderPreservingSemiring
        The algebra.
    inputs : Mapping of InputId to torch.Tensor
        The weights.
    like : torch.Tensor
        The dtype and device of the unit.

    Returns
    -------
    torch.Tensor
        The weight.

    Raises
    ------
    ForestRefused
        If the rule's weight needs an executable attachment.
    """
    constructor = forest.presentation.constructors_by_id[edge.constructor]
    assert isinstance(constructor, (Rule, AxiomSlot))
    if isinstance(constructor.weight, ComputedRuleWeight):
        raise ForestRefused(
            ForestRefusal.MISSING,
            "the forest has no executable attachment for this computed rule weight",
        )
    if edge.occurrence is not None and edge.occurrence in inputs:
        return inputs[edge.occurrence]
    if constructor.weight is not None and constructor.weight in inputs:
        return inputs[constructor.weight]
    return algebra.unit_like((), like)


def _join(
    algebra: OrderPreservingSemiring, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """The algebra's sum of two values.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    left, right : torch.Tensor
        The values.

    Returns
    -------
    torch.Tensor
        ``left ⊕ right``.
    """
    return algebra.join(torch.stack(torch.broadcast_tensors(left, right)), 0)


def forest_fold(
    forest: DerivationForest,
    goal: Item,
    algebra: OrderPreservingSemiring,
    value: Callable[[ForestEdge], torch.Tensor],
    *,
    like: torch.Tensor,
    height: int | None = None,
) -> torch.Tensor:
    """A linear proof fold: the sum over derivations of the product of edge values.

    Each derivation's product is read as its factors are: the premises'
    values in order, then the edge's own. Dynamic programming over the
    forest computes it in time linear in the forest's size, times the
    height bound when one is given.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goal : Item
        The item.
    algebra : OrderPreservingSemiring
        The algebra of the values.
    value : Callable[[ForestEdge], torch.Tensor]
        The value of each edge.
    like : torch.Tensor
        The dtype and device of the zero.
    height : int or None
        The bound on height, required for a cyclic forest.

    Returns
    -------
    torch.Tensor
        The fold, zero when the goal has no derivation.

    Raises
    ------
    ForestRefused
        If the forest has a cycle and no height is given.
    """
    return forest_folds(forest, (goal,), algebra, value, like=like, height=height)[goal]


def forest_folds(
    forest: DerivationForest,
    goals: Sequence[Item],
    algebra: OrderPreservingSemiring,
    value: Callable[[ForestEdge], torch.Tensor],
    *,
    like: torch.Tensor,
    height: int | None = None,
) -> dict[Item, torch.Tensor]:
    """Evaluate linear proof folds sharing one table for all requested items.

    For each incoming edge, the premise occurrences independently choose
    derivations of their items. Distributivity factors this sum of products
    into the product of the already-computed premise sums and the edge's
    value. Repeated occurrences of an item each contribute a factor.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goals : Sequence of Item
        The requested items.
    algebra : OrderPreservingSemiring
        The algebra of the values.
    value : Callable[[ForestEdge], torch.Tensor]
        The value of each edge.
    like : torch.Tensor
        The dtype and device of the zero.
    height : int or None
        The bound on height, required for a cyclic forest.

    Returns
    -------
    dict of Item to torch.Tensor
        The requested folds, with zero for any goal with no derivation.

    Raises
    ------
    ForestRefused
        If the forest has a cycle and no height is given.
    """
    part = forest.below_all(goals)

    values: dict[Key, torch.Tensor] = {}
    for level in _levels(part, height):
        for key in level:
            below = _below(key)
            total: torch.Tensor | None = None
            for edge in part.edges.get(key[0], ()):
                if edge.occurrence is None and below is not None and below < 0:
                    continue
                premises = [(premise, below) for premise in edge.premises]
                if not all(premise in values for premise in premises):
                    continue
                weight = value(edge)
                term = weight
                if premises:
                    first, *rest = premises
                    term = values[first]
                    for premise in rest:
                        term = algebra.tensor_op(term, values[premise])
                    term = algebra.tensor_op(term, weight)
                total = term if total is None else _join(algebra, total, term)
            if total is not None:
                values[key] = total
    return {
        goal: values[(goal, height)]
        if (goal, height) in values
        else algebra.zero_like((), like)
        for goal in goals
    }


# ---------------------------------------------------------------------------
# enumeration


type Listed[V] = list[tuple[Derivation, torch.Tensor, V]]
"""Derivations of one item, each with its weight and its fold."""


def fold_derivations[V](
    forest: DerivationForest,
    goal: Item,
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    interpretation: HostInterpretation[V],
    *,
    limit: int,
    like: torch.Tensor,
    height: int | None = None,
) -> Listed[V]:
    """Every derivation of an item, with its weight and its fold.

    Derivations are built bottom up and shared: a derivation of a premise
    is built and folded once and reused by every derivation above it.
    The exact budget decision happens before any derivation is built.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goal : Item
        The item.
    algebra : OrderPreservingSemiring
        The algebra of the weights.
    inputs : Mapping of InputId to torch.Tensor
        The weights.
    interpretation : HostInterpretation
        The fold of a derivation; its operations receive each node and
        its premises' folds.
    limit : int
        The most derivations of the goal to build, declared by the caller.
    like : torch.Tensor
        The dtype and device of the unit.
    height : int or None
        The bound on height, required for a cyclic forest.

    Returns
    -------
    list of tuple of Derivation, torch.Tensor and V
        Every derivation of the goal, its weight and its fold.

    Raises
    ------
    ForestRefused
        If the goal has more derivations than ``limit``, or the forest
        has a cycle and no height is given.
    """
    return fold_derivations_many(
        forest,
        (goal,),
        algebra,
        inputs,
        interpretation,
        limit=limit,
        like=like,
        height=height,
    )[goal]


def fold_derivations_many[V](
    forest: DerivationForest,
    goals: Sequence[Item],
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    interpretation: HostInterpretation[V],
    *,
    limit: int,
    like: torch.Tensor,
    height: int | None = None,
    node_limit: int | None = None,
) -> dict[Item, Listed[V]]:
    """List several goals' derivations under one exact global proof budget.

    Count structural proofs over the common subforest in one pass,
    saturating integers at one above the larger of the active limits.
    Every useful subtree extends to
    a requested proof, so each list in a successful run has at most
    ``limit`` entries. Build and fold each sub-derivation once across all
    goals. This sharing applies to host interpretations whose operations
    may be reused. The count includes zero-weight structural proofs.

    Parameters
    ----------
    forest : DerivationForest
        The common forest of the requested goals.
    goals : Sequence of Item
        The roots to list. A repeated goal is counted and returned once,
        at its first position.
    algebra : OrderPreservingSemiring
        The algebra of each derivation's weight.
    inputs : Mapping of InputId to torch.Tensor
        The rule and occurrence weights.
    interpretation : HostInterpretation
        The fold performed on each listed derivation.
    limit : int
        The maximum total number of proofs of the distinct goals.
    like : torch.Tensor
        The dtype and device of the unit.
    height : int or None
        The bound on height, required for a cyclic forest.
    node_limit : int or None
        An optional global limit on the sum of the unfolded proof sizes
        of those goals. Repeated subtree occurrences each count, even
        when the listing stores their derivations once.

    Returns
    -------
    dict of Item to Listed
        Every requested goal's derivations, with weights and folds.

    Raises
    ------
    ForestRefused
        If there are more proofs or unfolded nodes than the limits allow,
        or the forest has an unbounded cycle.
    ValueError
        If a proof budget is negative.
    """
    if limit < 0:
        raise ValueError("the proof budget must be nonnegative")
    if node_limit is not None and node_limit < 0:
        raise ValueError("the proof node budget must be nonnegative")
    goals = tuple(dict.fromkeys(goals))
    if not goals:
        return {}
    part = forest.below_all(goals)
    cap = max(limit, node_limit or 0) + 1
    sizes: dict[Key, int] = {}
    if node_limit is None:
        counts = _counts(part, height, cap=cap)
        total = sum(counts.get((goal, height), 0) for goal in goals)
    else:
        counts = _counts(part, height, cap=cap, sizes=sizes)
        total = 0
        for goal in goals:
            total += counts.get((goal, height), 0)
            if total > limit:
                break
    if total > limit:
        raise ForestRefused(
            ForestRefusal.TOO_MANY,
            f"at least {total} derivations exceed the declared limit of {limit}; "
            "none is returned",
        )
    if node_limit is not None:
        nodes = 0
        for goal in goals:
            nodes += sizes.get((goal, height), 0)
            if nodes > node_limit:
                raise ForestRefused(
                    ForestRefusal.TOO_MANY_NODES,
                    f"at least {nodes} unfolded proof nodes exceed the declared "
                    f"node limit of {node_limit}; none is returned",
                )
    useful = _useful(part, goals, height, counts)
    operations = interpretation.operations
    listed: dict[Key, Listed[V]] = {}
    for level in _levels(part, height):
        for key in level:
            if key not in useful:
                continue
            item = key[0]
            below = _below(key)
            found: Listed[V] = []
            for edge in part.edges.get(item, ()):
                weight = edge_weight(part, edge, algebra, inputs, like=like)
                if edge.occurrence is not None:
                    node = Derivation(
                        constructor=edge.constructor,
                        item=item,
                        binding=edge.binding,
                        values=edge.values,
                        occurrence=edge.occurrence,
                    )
                    found.append((node, weight, operations[edge.constructor](node, ())))
                    continue
                premises = [(premise, below) for premise in edge.premises]
                if not all(premise in listed for premise in premises):
                    continue
                for choice in _choices([listed[premise] for premise in premises]):
                    node = Derivation(
                        constructor=edge.constructor,
                        item=item,
                        binding=edge.binding,
                        premises=tuple(derivation for derivation, _, _ in choice),
                    )
                    product = weight
                    if choice:
                        product = choice[0][1]
                        for _, factor, _ in choice[1:]:
                            product = algebra.tensor_op(product, factor)
                        product = algebra.tensor_op(product, weight)
                    folds: tuple[PremiseValue[V], ...] = tuple(
                        folded for _, _, folded in choice
                    )
                    found.append(
                        (node, product, operations[edge.constructor](node, folds))
                    )
            listed[key] = found
    return {goal: listed.get((goal, height), []) for goal in goals}


def _choices[T](children: Sequence[list[T]]) -> list[list[T]]:
    """Every choice of one element per list, the last list varying fastest.

    Parameters
    ----------
    children : Sequence of list
        The lists.

    Returns
    -------
    list of list
        The choices.
    """
    choices: list[list[T]] = [[]]
    for options in children:
        choices = [[*prefix, option] for prefix in choices for option in options]
    return choices


def _weights_only(forest: DerivationForest) -> HostInterpretation[None]:
    """The interpretation that folds every derivation to nothing.

    Parameters
    ----------
    forest : DerivationForest
        The forest.

    Returns
    -------
    HostInterpretation
        One constant operation per rule and slot.
    """
    return HostInterpretation(
        presentation=forest.presentation,
        operations={
            identity: _nothing for identity in forest.presentation.constructors_by_id
        },
    )


def _nothing(node: Derivation, premises: tuple[PremiseValue[None], ...]) -> None:
    """Fold a node to nothing.

    Parameters
    ----------
    node : Derivation
        The node.
    premises : tuple of PremiseValue
        Its premises' folds.
    """
    del node, premises


def nonlinear_fold(
    forest: DerivationForest,
    goal: Item,
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    observe: Callable[[Derivation, torch.Tensor], torch.Tensor],
    *,
    limit: int,
    like: torch.Tensor,
    height: int | None = None,
) -> torch.Tensor:
    """The sum over derivations of any function of a derivation and its weight.

    No aggregate summarizes such a fold: with two derivations of weights
    ``-1`` and ``1``, squaring gives ``2`` here and ``0`` on their sum. So
    every derivation is built, which the declared limit bounds.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goal : Item
        The item.
    algebra : OrderPreservingSemiring
        The algebra of the weights and of the sum.
    inputs : Mapping of InputId to torch.Tensor
        The weights.
    observe : Callable[[Derivation, torch.Tensor], torch.Tensor]
        The function of a derivation and its weight.
    limit : int
        The most derivations to build.
    like : torch.Tensor
        The dtype and device of the zero.
    height : int or None
        The bound on height, required for a cyclic forest.

    Returns
    -------
    torch.Tensor
        The sum, zero when the goal has no derivation.

    Raises
    ------
    ForestRefused
        If the goal has more derivations than ``limit``, or the forest
        has a cycle and no height is given.
    """
    terms = [
        (derivation, weight)
        for derivation, weight, _ in fold_derivations(
            forest,
            goal,
            algebra,
            inputs,
            _weights_only(forest),
            limit=limit,
            like=like,
            height=height,
        )
    ]
    total = algebra.zero_like((), like)
    for derivation, weight in terms:
        total = _join(algebra, total, observe(derivation, weight))
    return total


def expected_embedding(
    forest: DerivationForest,
    goal: Item,
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    interpretation: HostInterpretation[torch.Tensor],
    *,
    probability: Callable[[torch.Tensor], torch.Tensor],
    limit: int,
    like: torch.Tensor,
    height: int | None = None,
) -> torch.Tensor:
    """The mean of a fold of the derivations into vectors, weighted by their weights.

    ``sum_d p(d) phi(d) / sum_d p(d)`` with ``p(d)`` the derivation's
    weight read as a nonnegative real by ``probability`` (``exp`` for log
    weights) and ``phi`` the interpretation's fold. A fold that is not
    linear, such as a neural encoder, has no aggregate over the forest, so
    the derivations are enumerated within the declared limit; each shared
    sub-derivation is folded once.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goal : Item
        The item.
    algebra : OrderPreservingSemiring
        The algebra of the weights.
    inputs : Mapping of InputId to torch.Tensor
        The weights.
    interpretation : HostInterpretation of torch.Tensor
        The embedding of a derivation.
    probability : Callable[[torch.Tensor], torch.Tensor]
        A derivation's weight as a nonnegative real.
    limit : int
        The most derivations to build.
    like : torch.Tensor
        The dtype and device of the unit.
    height : int or None
        The bound on height, required for a cyclic forest.

    Returns
    -------
    torch.Tensor
        The expected embedding.

    Raises
    ------
    ForestRefused
        If the goal has no derivation or more than ``limit``, or the
        forest has a cycle and no height is given.
    """
    listed = fold_derivations(
        forest,
        goal,
        algebra,
        inputs,
        interpretation,
        limit=limit,
        like=like,
        height=height,
    )
    if not listed:
        raise ForestRefused(ForestRefusal.MISSING, "the goal has no derivation")
    masses = torch.stack([probability(weight) for _, weight, _ in listed])
    embeddings = torch.stack([embedding for _, _, embedding in listed])
    shape = (len(listed),) + (1,) * (embeddings.dim() - 1)
    return (masses.reshape(shape) * embeddings).sum(0) / masses.sum()


def encoder_interpretation(
    encoder: Encoder, presentation: RulePresentation
) -> HostInterpretation[torch.Tensor]:
    """The fold of an encoder over the derivations of its signature's presentation.

    Each rule of `presentation_of` the encoder's signature is a
    constructor; its operation is the encoder's `Encoder.step` on the
    premises' embeddings and the node's values, interleaved by the kinds
    of the constructor's argument sorts. On every derivation it equals
    the encoder applied to the derivation's term.

    Parameters
    ----------
    encoder : Encoder
        The encoder.
    presentation : RulePresentation
        The presentation of the encoder's signature.

    Returns
    -------
    HostInterpretation of torch.Tensor
        The fold.

    Raises
    ------
    ValueError
        If a rule or slot of the presentation is not a constructor of the
        signature.
    """
    signature = encoder.signature
    operations: dict[ConstructorId, Callable[..., torch.Tensor]] = {}
    for identity, constructor in presentation.constructors_by_id.items():
        if constructor.name not in signature.constructors:
            raise ValueError(
                f"{constructor.name!r} is not a constructor of the signature"
            )
        operations[identity] = _step(encoder, signature, constructor.name)
    return HostInterpretation(presentation=presentation, operations=operations)


def _step(
    encoder: Encoder, signature: Signature, op: str
) -> Callable[[Derivation, tuple[PremiseValue[torch.Tensor], ...]], torch.Tensor]:
    """The operation of one constructor: the encoder's step at a node.

    Parameters
    ----------
    encoder : Encoder
        The encoder.
    signature : Signature
        Its signature.
    op : str
        The constructor.

    Returns
    -------
    Callable
        The operation.
    """
    kinds = [signature.sorts[sort].kind for sort in signature.constructors[op].domain]

    def operation(
        node: Derivation, premises: tuple[PremiseValue[torch.Tensor], ...]
    ) -> torch.Tensor:
        """Embed a node from its premises' embeddings and its values.

        Parameters
        ----------
        node : Derivation
            The node.
        premises : tuple of PremiseValue
            The premises' embeddings.

        Returns
        -------
        torch.Tensor
            The node's embedding.
        """
        children = iter(premises)
        values = iter(node.values)
        arguments: list[torch.Tensor | DataLeaf] = []
        for kind in kinds:
            if kind == "object":
                child = next(children)
                assert isinstance(child, torch.Tensor)
                arguments.append(child)
            else:
                arguments.append(next(values))
        return encoder.step(op, tuple(arguments))

    return operation
