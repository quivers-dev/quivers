"""An indexed chart of a rule presentation, with bindings pushed into lookups.

Items of each judgment form are indexed on the argument positions a
premise finds bound when it is matched: when a rule fires on one item,
the variables that item binds turn the other premises' patterns into
lookup keys, and only the items with those arguments are visited. The
indexes are built on first use and kept up to date as items arrive, so a
lookup costs its key and its answers, not a scan of the family.

A join plan fixes, for each rule and each premise that triggers it, the
order in which the other premises are looked up and the positions each
lookup is keyed on: next comes the premise with the most bound
positions, ties in premise order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._deduction_guards import _matched_value
from quivers.qiec.contracts import Phase, WorkField, WorkReport, measuring
from quivers.stochastic.tries import Trie, TrieCounter, descend, insert, intersect
from quivers.qiec.evaluator import Evaluator
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.rules import (
    Binding,
    Ground,
    Item,
    ItemTable,
    MatchError,
    Rule,
    RulePresentation,
    evaluate_index,
    match_index,
)
from quivers.qiec.types import IndexConstructor, IndexSum, IndexTerm, IndexVariable

__all__ = [
    "ChartWork",
    "IndexedChart",
    "TrieNode",
    "JoinStep",
    "RuleJoinPlan",
    "flat_pattern",
    "generic_joinable",
    "index_variables",
    "index_variables_of",
    "join_order",
    "join_plans",
    "side_condition_holds",
]


def index_variables(term: IndexTerm) -> set[str]:
    """The variable names of an index term.

    Parameters
    ----------
    term : IndexTerm
        The term.

    Returns
    -------
    set of str
        Its variables.
    """
    if isinstance(term, IndexVariable):
        return {term.name}
    if isinstance(term, IndexConstructor):
        found: set[str] = set()
        for item in term.arguments:
            found |= index_variables(item)
        return found
    if isinstance(term, IndexSum):
        return {item.name for item in term.terms}
    return set()


def flat_pattern(term: IndexTerm) -> bool:
    """Whether matching a term alone binds all its variables.

    Parameters
    ----------
    term : IndexTerm
        The term.

    Returns
    -------
    bool
        False when a sum, here or nested, has two variables.
    """
    if isinstance(term, IndexSum):
        return len({item.name for item in term.terms}) <= 1
    if isinstance(term, IndexConstructor):
        return all(flat_pattern(argument) for argument in term.arguments)
    return True


def _ordered_variables(arguments: Sequence[IndexTerm]) -> list[str]:
    """The variables of a pattern in order of first occurrence.

    Parameters
    ----------
    arguments : Sequence of IndexTerm
        The pattern's arguments.

    Returns
    -------
    list of str
        Each variable once.
    """
    found: list[str] = []
    for argument in arguments:
        for name in sorted(index_variables(argument)):
            if name not in found:
                found.append(name)
    return found


def generic_joinable(rule: Rule) -> bool:
    """Whether every premise, and the conclusion, is a relation of its own variables.

    Parameters
    ----------
    rule : Rule
        The rule.

    Returns
    -------
    bool
        True when no pattern of the rule has a sum of two variables and
        no premise has hypotheses.
    """
    judgments = [premise.judgment for premise in rule.premises] + [rule.conclusion]
    return all(not premise.hypotheses for premise in rule.premises) and all(
        flat_pattern(argument)
        for judgment in judgments
        for argument in judgment.arguments
    )


def join_order(rule: Rule, trigger: int) -> tuple[str, ...]:
    """The variable order of a rule's join from one trigger.

    The variables the trigger binds come first, in the order they occur
    in its pattern (the conclusion's for trigger ``-1``), then the others
    in the order they first occur in the premises.

    Parameters
    ----------
    rule : Rule
        The rule.
    trigger : int
        The triggering premise, or ``-1`` for the conclusion.

    Returns
    -------
    tuple of str
        The order.
    """
    source = (
        rule.conclusion.arguments
        if trigger < 0
        else rule.premises[trigger].judgment.arguments
    )
    order = _ordered_variables(source)
    for premise in rule.premises:
        for name in _ordered_variables(premise.judgment.arguments):
            if name not in order:
                order.append(name)
    return tuple(order)


type TrieNode = Trie[Ground, Item]
"""A level of a premise's trie; its leaves are items. A premise without
variables keeps its item under the key ``()``."""


def index_variables_of(arguments: Sequence[IndexTerm]) -> set[str]:
    """The variables of a pattern.

    Parameters
    ----------
    arguments : Sequence of IndexTerm
        The pattern's arguments.

    Returns
    -------
    set of str
        Its variables.
    """
    found: set[str] = set()
    for argument in arguments:
        found |= index_variables(argument)
    return found


class JoinStep(ValueModel):
    """One lookup of a join plan.

    Parameters
    ----------
    premise : int
        The premise looked up.
    keyed : tuple of int
        The argument positions whose values are known before the lookup
        and form its key.
    """

    premise: int
    keyed: tuple[int, ...] = ()


class RuleJoinPlan(ValueModel):
    """The order of lookups when one premise of a rule triggers it.

    Parameters
    ----------
    rule : ConstructorId
        The rule.
    trigger : int
        The premise matched against the new item, or ``-1`` when the
        conclusion is matched first, as in a backward pass.
    steps : tuple of JoinStep
        The other premises, in lookup order.
    """

    rule: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    trigger: int
    steps: tuple[JoinStep, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


def _plan(
    rule: Rule, bound: set[str], start: Sequence[int], trigger: int
) -> RuleJoinPlan:
    """Order the remaining premises by the positions they find bound.

    Parameters
    ----------
    rule : Rule
        The rule.
    bound : set of str
        The variables bound before the first lookup.
    start : Sequence of int
        The premises to order.
    trigger : int
        The trigger recorded in the plan.

    Returns
    -------
    RuleJoinPlan
        The plan.
    """
    known = set(bound)
    left = list(start)
    steps: list[JoinStep] = []
    while left:

        def keyed(index: int) -> tuple[int, ...]:
            """The positions of a premise determined by the known variables.

            Parameters
            ----------
            index : int
                The premise.

            Returns
            -------
            tuple of int
                Its keyed positions.
            """
            arguments = rule.premises[index].judgment.arguments
            return tuple(
                position
                for position, argument in enumerate(arguments)
                if index_variables(argument) <= known
            )

        best = max(left, key=lambda index: (len(keyed(index)), -index))
        steps.append(JoinStep(premise=best, keyed=keyed(best)))
        left.remove(best)
        for argument in rule.premises[best].judgment.arguments:
            known |= index_variables(argument)
    return RuleJoinPlan(rule=rule.id, trigger=trigger, steps=tuple(steps))


def join_plans(
    presentation: RulePresentation,
) -> dict[tuple[ConstructorId, int], RuleJoinPlan]:
    """The join plan of every rule for every trigger, and from its conclusion.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    dict of tuple of ConstructorId and int to RuleJoinPlan
        Plans by rule and trigger; trigger ``-1`` starts from the
        conclusion's variables.
    """
    plans: dict[tuple[ConstructorId, int], RuleJoinPlan] = {}
    for rule in presentation.rules:
        positions = range(len(rule.premises))
        for trigger in positions:
            bound: set[str] = set()
            for argument in rule.premises[trigger].judgment.arguments:
                bound |= index_variables(argument)
            others = [index for index in positions if index != trigger]
            plans[(rule.id, trigger)] = _plan(rule, bound, others, trigger)
        conclusion: set[str] = set()
        for argument in rule.conclusion.arguments:
            conclusion |= index_variables(argument)
        plans[(rule.id, -1)] = _plan(rule, conclusion, list(positions), -1)
    return plans


def side_condition_holds(
    presentation: RulePresentation, rule: Rule, binding: Binding
) -> bool:
    """Evaluate a rule's side condition with the QIEC evaluator.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation, which holds the condition's computation.
    rule : Rule
        The rule.
    binding : Binding
        Its variables.

    Returns
    -------
    bool
        Whether the condition holds; true without one.

    Raises
    ------
    MatchError
        If the condition reads a variable the binding lacks.
    """
    condition = rule.side_condition
    if condition is None:
        return True
    computation = presentation.conditions_by_id[condition.computation]
    environment = {}
    for name, parameter in zip(
        condition.arguments, computation.parameters, strict=True
    ):
        if name not in binding:
            raise MatchError(f"the side condition of {rule.name!r} reads {name!r}")
        environment[parameter] = _matched_value(
            computation, rule, name, parameter, binding[name]
        )
    return bool(Evaluator().evaluate(computation.body, environment))


class ChartWork(ValueModel):
    """Counters of the work of a chart.

    Parameters
    ----------
    lookups : int
        Index lookups.
    candidates : int
        Items returned by lookups and matched against a pattern.
    instances : int
        Rule instances enumerated.
    conditions : int
        Side conditions evaluated.
    index_entries : int
        Entries added to indexes.
    """

    lookups: int = 0
    candidates: int = 0
    instances: int = 0
    conditions: int = 0
    index_entries: int = 0

    def report(self, phase: Phase) -> WorkReport:
        """The counters as a work report.

        Parameters
        ----------
        phase : Phase
            The phase they counted.

        Returns
        -------
        WorkReport
            Lookups as hash operations and candidates as key
            comparisons, the two measured.
        """
        return WorkReport(
            phase=phase,
            hash_operations=self.lookups,
            key_comparisons=self.candidates,
            measured=measuring({WorkField.HASH_OPERATIONS, WorkField.KEY_COMPARISONS}),
        )


class IndexedChart:
    """Items of a presentation indexed on the argument positions lookups use.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    join : {"generic", "keyed"}
        How a rule's other premises are found from a trigger. ``"generic"``
        runs Generic Join over tries of the premises in the trigger's
        variable order, for every rule that `generic_joinable` admits,
        and looks premises up one at a time for the others; ``"keyed"``
        looks premises up one at a time for every rule.
    """

    def __init__(
        self,
        presentation: RulePresentation,
        *,
        join: Literal["generic", "keyed"] = "generic",
    ) -> None:
        self.presentation = presentation
        self.join = join
        self._tries: dict[
            tuple[ConstructorId, int, int], tuple[tuple[str, ...], TrieNode]
        ] = {}
        self._tries_by_family: dict[FamilyId, list[tuple[ConstructorId, int, int]]] = {}
        self._orders: dict[tuple[ConstructorId, int], tuple[str, ...]] = {}
        self.items = ItemTable()
        self._members: dict[FamilyId, list[Item]] = {}
        self._indexes: dict[
            tuple[FamilyId, tuple[int, ...]], dict[tuple[Ground, ...], list[Item]]
        ] = {}
        self.plans = join_plans(presentation)
        self.work = {
            "lookups": 0,
            "candidates": 0,
            "instances": 0,
            "conditions": 0,
            "index_entries": 0,
        }

    def counters(self) -> ChartWork:
        """The work so far.

        Returns
        -------
        ChartWork
            The counters.
        """
        return ChartWork(**self.work)

    def add(self, item: Item) -> None:
        """Index an item under every index of its family.

        Parameters
        ----------
        item : Item
            The item, added once.
        """
        self._members.setdefault(item.family, []).append(item)
        for key in self._tries_by_family.get(item.family, ()):
            variables, root = self._tries[key]
            self._insert(key, variables, root, item)
        for (family, positions), index in self._indexes.items():
            if family == item.family:
                key = tuple(item.arguments[p] for p in positions)
                index.setdefault(key, []).append(item)
                self.work["index_entries"] += 1

    def lookup(
        self, family: FamilyId, positions: tuple[int, ...], key: tuple[Ground, ...]
    ) -> list[Item]:
        """The items of a family with given values at given positions.

        Parameters
        ----------
        family : FamilyId
            The judgment form.
        positions : tuple of int
            The argument positions.
        key : tuple of Ground
            Their values.

        Returns
        -------
        list of Item
            The items, in the order they were added.
        """
        self.work["lookups"] += 1
        if not positions:
            return self._members.get(family, [])
        index = self._indexes.get((family, positions))
        if index is None:
            index = {}
            for item in self._members.get(family, []):
                index.setdefault(
                    tuple(item.arguments[p] for p in positions), []
                ).append(item)
                self.work["index_entries"] += 1
            self._indexes[(family, positions)] = index
        return index.get(key, [])

    def holds(self, rule: Rule, binding: Binding) -> bool:
        """Evaluate a rule's side condition, counting the evaluation.

        Parameters
        ----------
        rule : Rule
            The rule.
        binding : Binding
            Its variables.

        Returns
        -------
        bool
            Whether the condition holds; true without one.

        Raises
        ------
        MatchError
            If the condition reads a variable the binding lacks.
        """
        if rule.side_condition is not None:
            self.work["conditions"] += 1
        return side_condition_holds(self.presentation, rule, binding)

    def instances(
        self,
        rule: Rule,
        trigger: int,
        item: Item,
        admitted: Callable[[Item], bool],
    ) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
        """The instances of a rule that ``item`` completes.

        ``item`` fills premise ``trigger`` and no earlier premise; every
        other premise is an admitted item. So when items are admitted one
        at a time and each new item triggers every position it fits, each
        instance is produced once, by the last of its premises to be
        admitted, at the first position that item fills.

        Parameters
        ----------
        rule : Rule
            The rule.
        trigger : int
            The premise ``item`` fills.
        item : Item
            The newly admitted item.
        admitted : Callable[[Item], bool]
            Whether an item is admitted; ``item`` itself must be.

        Yields
        ------
        tuple of Binding and tuple of Item
            The binding and the premise items, in premise order.
        """
        premise = rule.premises[trigger]
        binding: Binding = {}
        for pattern, value in zip(
            premise.judgment.arguments, item.arguments, strict=True
        ):
            if not match_index(pattern, value, binding):
                return
        if self.method(rule) == "generic":
            yield from self._generic(rule, trigger, item, binding, admitted)
            return
        plan = self.plans[(rule.id, trigger)]
        chosen: dict[int, Item] = {trigger: item}
        yield from self._extend(
            rule, plan.steps, 0, binding, chosen, trigger, item, admitted
        )

    def incoming(
        self, rule: Rule, item: Item, admitted: Callable[[Item], bool]
    ) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
        """The instances of a rule that conclude ``item`` from admitted premises.

        Parameters
        ----------
        rule : Rule
            The rule.
        item : Item
            The conclusion.
        admitted : Callable[[Item], bool]
            Whether a premise item is admitted.

        Yields
        ------
        tuple of Binding and tuple of Item
            The binding and the premise items.
        """
        if rule.conclusion.family != item.family:
            return
        binding: Binding = {}
        for pattern, value in zip(
            rule.conclusion.arguments, item.arguments, strict=True
        ):
            if not match_index(pattern, value, binding):
                return
        if self.method(rule) == "generic":
            yield from self._generic(rule, -1, item, binding, admitted)
            return
        plan = self.plans[(rule.id, -1)]
        yield from self._extend(rule, plan.steps, 0, binding, {}, -1, item, admitted)

    def method(self, rule: Rule) -> Literal["generic", "keyed"]:
        """How this chart joins a rule's premises.

        Parameters
        ----------
        rule : Rule
            The rule.

        Returns
        -------
        {"generic", "keyed"}
            ``"generic"`` when the chart runs Generic Join and the rule
            admits it.
        """
        if self.join == "generic" and generic_joinable(rule):
            return "generic"
        return "keyed"

    def _trie(
        self, rule: Rule, trigger: int, position: int
    ) -> tuple[tuple[str, ...], TrieNode]:
        """The trie of one premise in the order of one trigger, built on first use.

        Parameters
        ----------
        rule : Rule
            The rule.
        trigger : int
            The trigger, or ``-1``.
        position : int
            The premise.

        Returns
        -------
        tuple of tuple of str and TrieNode
            The premise's variables in the trigger's order, and the trie
            of the items of its form that match it.
        """
        key = (rule.id, trigger, position)
        found = self._tries.get(key)
        if found is not None:
            return found
        order = self._order(rule, trigger)
        names = index_variables_of(rule.premises[position].judgment.arguments)
        variables = tuple(name for name in order if name in names)
        root: TrieNode = {}
        self._tries[key] = (variables, root)
        family = rule.premises[position].judgment.family
        self._tries_by_family.setdefault(family, []).append(key)
        for member in self._members.get(family, []):
            self._insert(key, variables, root, member)
        return variables, root

    def _order(self, rule: Rule, trigger: int) -> tuple[str, ...]:
        """The variable order of a rule's join from one trigger, cached.

        Parameters
        ----------
        rule : Rule
            The rule.
        trigger : int
            The trigger, or ``-1``.

        Returns
        -------
        tuple of str
            `join_order` of the rule and trigger.
        """
        key = (rule.id, trigger)
        if key not in self._orders:
            self._orders[key] = join_order(rule, trigger)
        return self._orders[key]

    def _insert(
        self,
        key: tuple[ConstructorId, int, int],
        variables: tuple[str, ...],
        root: TrieNode,
        item: Item,
    ) -> None:
        """Add an item to one premise's trie when it matches the premise.

        Parameters
        ----------
        key : tuple of ConstructorId, int and int
            The rule, trigger and premise of the trie.
        variables : tuple of str
            The trie's variables in order.
        root : TrieNode
            The trie.
        item : Item
            The item.
        """
        rule = self.presentation.constructors_by_id[key[0]]
        assert isinstance(rule, Rule)
        pattern = rule.premises[key[2]].judgment.arguments
        binding: Binding = {}
        if not all(
            match_index(term, value, binding)
            for term, value in zip(pattern, item.arguments, strict=True)
        ):
            return
        self.work["index_entries"] += 1
        insert(root, [binding[name] for name in variables] or [()], item)

    def _generic(
        self,
        rule: Rule,
        trigger: int,
        item: Item,
        binding: Binding,
        admitted: Callable[[Item], bool],
    ) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
        """The instances from a trigger by Generic Join over the other premises.

        The trigger's variables are bound; every other premise's trie is
        entered along the bound prefix of its variables, and the
        remaining variables are bound one at a time, each to the values
        in the intersection of the tries that contain it, iterating the
        smallest. A complete binding determines each premise's item.

        Parameters
        ----------
        rule : Rule
            The rule.
        trigger : int
            The premise ``item`` fills, or ``-1`` when it is the conclusion.
        item : Item
            The trigger's item, or the conclusion.
        binding : Binding
            The variables the trigger binds.
        admitted : Callable[[Item], bool]
            Whether an item is admitted.

        Yields
        ------
        tuple of Binding and tuple of Item
            The binding and the premise items, in premise order.
        """
        order = self._order(rule, trigger)
        others = [p for p in range(len(rule.premises)) if p != trigger]
        tries = [self._trie(rule, trigger, p) for p in others]
        nodes: list[TrieNode | Item] = []
        for variables, root in tries:
            self.work["lookups"] += 1
            path = [binding[name] for name in variables if name in binding] or (
                [()] if not variables else []
            )
            node = descend(root, path)
            if node is None:
                return
            nodes.append(node)
        free = [name for name in order if name not in binding]
        containing = [
            [index for index, (variables, _) in enumerate(tries) if name in variables]
            for name in free
        ]
        yield from self._intersect(
            rule, trigger, item, admitted, free, containing, nodes, others, binding
        )

    def _intersect(
        self,
        rule: Rule,
        trigger: int,
        item: Item,
        admitted: Callable[[Item], bool],
        free: list[str],
        containing: list[list[int]],
        nodes: list[TrieNode | Item],
        others: list[int],
        binding: Binding,
    ) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
        """Complete the instances by the intersection step and test each.

        Parameters
        ----------
        rule : Rule
            The rule.
        trigger : int
            The trigger, or ``-1``.
        item : Item
            The trigger's item, or the conclusion.
        admitted : Callable[[Item], bool]
            Whether an item is admitted.
        free : list of str
            The variables the trigger leaves free, in order.
        containing : list of list of int
            For each free variable, the other premises containing it, as
            indexes into ``others``.
        nodes : list
            Each other premise's trie below the bound variables.
        others : list of int
            The positions of the other premises.
        binding : Binding
            The variables the trigger binds.

        Yields
        ------
        tuple of Binding and tuple of Item
            Complete instances.
        """
        counter = TrieCounter()
        counted = 0
        for full, leaves in intersect(free, containing, nodes, binding, counter):
            seen = counter.iterated + counter.probes
            self.work["candidates"] += seen - counted
            counted = seen
            chosen: dict[int, Item] = {}
            if trigger >= 0:
                chosen[trigger] = item
            rejected = False
            for position, node in zip(others, leaves, strict=True):
                assert isinstance(node, Item)
                if trigger >= 0 and position < trigger and node == item:
                    rejected = True
                    break
                if not admitted(node):
                    rejected = True
                    break
                chosen[position] = node
            if rejected or not self.holds(rule, full):
                continue
            if trigger < 0:
                conclusion = tuple(
                    evaluate_index(argument, full)
                    for argument in rule.conclusion.arguments
                )
                if conclusion != item.arguments:
                    continue
            self.work["instances"] += 1
            yield full, tuple(chosen[p] for p in range(len(rule.premises)))
        self.work["candidates"] += counter.iterated + counter.probes - counted

    def _extend(
        self,
        rule: Rule,
        steps: tuple[JoinStep, ...],
        depth: int,
        binding: Binding,
        chosen: dict[int, Item],
        trigger: int,
        item: Item,
        admitted: Callable[[Item], bool],
    ) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
        """Look up the remaining premises one step at a time.

        Parameters
        ----------
        rule : Rule
            The rule.
        steps : tuple of JoinStep
            The plan's steps.
        depth : int
            The next step.
        binding : Binding
            The variables bound so far.
        chosen : dict of int to Item
            The premises chosen so far.
        trigger : int
            The trigger, or ``-1``.
        item : Item
            The trigger's item, or the conclusion.
        admitted : Callable[[Item], bool]
            Whether an item is admitted.

        Yields
        ------
        tuple of Binding and tuple of Item
            Complete instances.
        """
        if depth == len(steps):
            if not self.holds(rule, binding):
                return
            if trigger < 0:
                conclusion = tuple(
                    evaluate_index(argument, binding)
                    for argument in rule.conclusion.arguments
                )
                if conclusion != item.arguments:
                    return
            self.work["instances"] += 1
            yield (
                dict(binding),
                tuple(chosen[index] for index in range(len(rule.premises))),
            )
            return
        step = steps[depth]
        premise = rule.premises[step.premise].judgment
        key = tuple(evaluate_index(premise.arguments[p], binding) for p in step.keyed)
        for candidate in self.lookup(premise.family, step.keyed, key):
            self.work["candidates"] += 1
            if not admitted(candidate):
                continue
            if trigger >= 0 and step.premise < trigger and candidate == item:
                continue
            extended = dict(binding)
            if not all(
                match_index(pattern, value, extended)
                for pattern, value in zip(
                    premise.arguments, candidate.arguments, strict=True
                )
            ):
                continue
            chosen[step.premise] = candidate
            yield from self._extend(
                rule, steps, depth + 1, extended, chosen, trigger, item, admitted
            )
            del chosen[step.premise]
