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

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
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
    "JoinStep",
    "RuleJoinPlan",
    "join_plans",
]


def _variables(term: IndexTerm) -> set[str]:
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
            found |= _variables(item)
        return found
    if isinstance(term, IndexSum):
        return {item.name for item in term.terms}
    return set()


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
                if _variables(argument) <= known
            )

        best = max(left, key=lambda index: (len(keyed(index)), -index))
        steps.append(JoinStep(premise=best, keyed=keyed(best)))
        left.remove(best)
        for argument in rule.premises[best].judgment.arguments:
            known |= _variables(argument)
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
                bound |= _variables(argument)
            others = [index for index in positions if index != trigger]
            plans[(rule.id, trigger)] = _plan(rule, bound, others, trigger)
        conclusion: set[str] = set()
        for argument in rule.conclusion.arguments:
            conclusion |= _variables(argument)
        plans[(rule.id, -1)] = _plan(rule, conclusion, list(positions), -1)
    return plans


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


class IndexedChart:
    """Items of a presentation indexed on the argument positions lookups use.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    """

    def __init__(self, presentation: RulePresentation) -> None:
        self.presentation = presentation
        self.items = ItemTable()
        self._members: dict[FamilyId, list[Item]] = {}
        self._indexes: dict[
            tuple[FamilyId, tuple[int, ...]], dict[tuple[Ground, ...], list[Item]]
        ] = {}
        self.plans = join_plans(presentation)
        self._conditions = {item.id: item for item in presentation.conditions}
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
        """Evaluate a rule's side condition with the QIEC evaluator.

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
        condition = rule.side_condition
        if condition is None:
            return True
        self.work["conditions"] += 1
        computation = self._conditions[condition.computation]
        environment = {}
        for name, parameter in zip(
            condition.arguments, computation.parameters, strict=True
        ):
            if name not in binding:
                raise MatchError(f"the side condition of {rule.name!r} reads {name!r}")
            environment[parameter] = binding[name]
        return bool(Evaluator().evaluate(computation.body, environment))

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
        plan = self.plans[(rule.id, -1)]
        yield from self._extend(rule, plan.steps, 0, binding, {}, -1, item, admitted)

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
            if trigger >= 0 and step.premise < trigger and candidate is item:
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
