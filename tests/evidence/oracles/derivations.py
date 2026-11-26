"""Enumeration of every derivation tree of a weighted deduction system.

A system is plain data. An item is a nested tuple of strings, integers,
and ``None``; a pattern is an item that may hold `Variable` leaves. A
rule is ``(name, premises, conclusion, weight)`` where ``weight`` names
the rule's weight parameter or is ``None`` for weight one. An axiom is
``(occurrence, item)``; two axioms with equal items and different
occurrence names are two axioms.

A tree is enumerated with the names of its leaves, so multiplicities,
nonlinear folds, and the statistics of single occurrences are sums over
an explicit list. Nothing is shared between trees and no weight is
aggregated before the list is complete.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction

import torch

type Term = str | int | None | Variable | tuple[Term, ...]
"""A ground item or a pattern."""

type Binding = dict[str, Term]
"""Values of pattern variables."""

type RuleRow = tuple[str, tuple[Term, ...], Term, str | None]
"""A rule: its name, premise patterns, conclusion pattern, weight name."""

type AxiomRow = tuple[str, Term]
"""An axiom: its occurrence name and its item."""

type SideCondition = Callable[[Mapping[str, Term]], bool]
"""A predicate on the variables a rule binds."""

type Tree = tuple[str, Term, str | None, tuple[Tree, ...]]
"""A derivation: rule or axiom name, item, leaf or rule weight, children."""

type Instance = tuple[int, tuple[Term, ...], Term]
"""A grounded rule: its index, premise items, conclusion item."""


class Variable(str):
    """A named pattern variable."""

    __slots__ = ()


class CyclicDerivations(ValueError):
    """The item has infinitely many derivations and no height bound was given."""


class TooManyItems(ValueError):
    """The system derives more items than the stated limit."""


_MISSING = ("unbound",)


def bind(pattern: Term, item: Term, binding: Binding) -> Binding | None:
    """Extend a binding so that a pattern equals an item.

    Parameters
    ----------
    pattern : Term
        The pattern.
    item : Term
        The ground item.
    binding : Binding
        The variables bound so far.

    Returns
    -------
    Binding or None
        The extended binding, or ``None`` when no extension exists.
    """
    if isinstance(pattern, Variable):
        if str(pattern) in binding:
            return binding if _same(binding[str(pattern)], item) else None
        return {**binding, str(pattern): item}
    if isinstance(pattern, tuple):
        if not isinstance(item, tuple) or len(pattern) != len(item):
            return None
        current: Binding | None = binding
        for part, value in zip(pattern, item, strict=True):
            if current is None:
                return None
            current = bind(part, value, current)
        return current
    return binding if _same(pattern, item) else None


def _same(left: Term, right: Term) -> bool:
    """Whether two ground terms are one term, with their types.

    Parameters
    ----------
    left : Term
        The first term.
    right : Term
        The second term.

    Returns
    -------
    bool
        Equality of both type and value at every leaf.
    """
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    return type(left) is type(right) and left == right


def substitute(pattern: Term, binding: Mapping[str, Term]) -> Term:
    """Replace every variable of a pattern by its value.

    Parameters
    ----------
    pattern : Term
        The pattern.
    binding : Mapping[str, Term]
        The values.

    Returns
    -------
    Term
        The ground item.
    """
    if isinstance(pattern, Variable):
        return binding[str(pattern)]
    if isinstance(pattern, tuple):
        return tuple(substitute(part, binding) for part in pattern)
    return pattern


def ground(
    axioms: Sequence[AxiomRow],
    rules: Sequence[RuleRow],
    sides: Mapping[str, SideCondition] | None = None,
    *,
    limit: int = 400,
) -> list[Instance]:
    """Every grounded rule whose premises are derivable.

    Parameters
    ----------
    axioms : Sequence of AxiomRow
        The axioms.
    rules : Sequence of RuleRow
        The rules.
    sides : Mapping[str, SideCondition] or None
        Side conditions by rule name.
    limit : int
        The largest number of items allowed.

    Returns
    -------
    list of Instance
        The grounded rules in rule order.


    Raises
    ------
    TooManyItems
        If more than ``limit`` items are derivable.
    """
    conditions = sides or {}
    items: dict[Term, None] = dict.fromkeys(item for _, item in axioms)
    found: dict[Instance, None] = {}
    while True:
        before = (len(items), len(found))
        for index, (name, premises, conclusion, _) in enumerate(rules):
            condition = conditions.get(name)
            for chosen, binding in _matches(premises, tuple(items), {}):
                if condition is not None and not condition(binding):
                    continue
                result = substitute(conclusion, binding)
                found[(index, chosen, result)] = None
                items[result] = None
                if len(items) > limit:
                    raise TooManyItems(f"more than {limit} items are derivable")
        if (len(items), len(found)) == before:
            return list(found)


def _matches(
    premises: Sequence[Term], items: Sequence[Term], binding: Binding
) -> Iterator[tuple[tuple[Term, ...], Binding]]:
    """Every choice of one item per premise under one binding.

    Parameters
    ----------
    premises : Sequence of Term
        The premise patterns.
    items : Sequence of Term
        The items.
    binding : Binding
        The variables bound so far.

    Yields
    ------
    tuple of (tuple of Term, Binding)
        The items in premise order and the complete binding.
    """
    if not premises:
        yield (), binding
        return
    for item in items:
        extended = bind(premises[0], item, binding)
        if extended is None:
            continue
        for rest, complete in _matches(premises[1:], items, extended):
            yield (item, *rest), complete


def derivations(
    axioms: Sequence[AxiomRow],
    rules: Sequence[RuleRow],
    goal: Term,
    sides: Mapping[str, SideCondition] | None = None,
    *,
    height: int | None = None,
) -> list[Tree]:
    """Every derivation tree of an item.

    Parameters
    ----------
    axioms : Sequence of AxiomRow
        The axioms.
    rules : Sequence of RuleRow
        The rules.
    goal : Term
        The derived item.
    sides : Mapping[str, SideCondition] or None
        Side conditions by rule name.
    height : int or None
        The greatest height of a tree. It is a restriction of the model
        and is required when the item lies above a cycle.

    Returns
    -------
    list of Tree
        The trees, each once.


    Raises
    ------
    CyclicDerivations
        If the item has infinitely many derivations and no height bound
        was given.
    """
    incoming: dict[Term, list[Instance]] = {}
    for instance in ground(axioms, rules, sides):
        incoming.setdefault(instance[2], []).append(instance)
    if height is None:
        _refuse_cycles(goal, incoming)
    leaves: dict[Term, list[str]] = {}
    for occurrence, item in axioms:
        leaves.setdefault(item, []).append(occurrence)

    def trees(item: Term, budget: int | None) -> Iterator[Tree]:
        """The trees of one item within a height.

        Parameters
        ----------
        item : Term
            The item.
        budget : int or None
            The remaining height.

        Yields
        ------
        Tree
            A derivation of the item.
        """
        for occurrence in leaves.get(item, ()):
            yield (occurrence, item, occurrence, ())
        if budget == 0:
            return
        lower = None if budget is None else budget - 1
        for index, premises, _ in incoming.get(item, ()):
            name, _, _, weight = rules[index]
            branches = [list(trees(premise, lower)) for premise in premises]
            for children in itertools.product(*branches):
                yield (name, item, weight, children)

    return list(trees(goal, height))


def _refuse_cycles(goal: Term, incoming: Mapping[Term, Sequence[Instance]]) -> None:
    """Raise when an item is an ancestor of itself below a goal.

    Parameters
    ----------
    goal : Term
        The goal item.
    incoming : Mapping[Term, Sequence of Instance]
        The grounded rules by conclusion.


    Raises
    ------
    CyclicDerivations
        If the derivations of the goal are infinite.
    """
    state: dict[Term, str] = {}

    def visit(item: Term) -> None:
        """Walk the premises of one item.

        Parameters
        ----------
        item : Term
            The item.


        Raises
        ------
        CyclicDerivations
            If the item is below itself.
        """
        if state.get(item) == "done":
            return
        if state.get(item) == "open":
            raise CyclicDerivations(f"{item!r} derives itself")
        state[item] = "open"
        for _, premises, _ in incoming.get(item, ()):
            for premise in premises:
                visit(premise)
        state[item] = "done"

    visit(goal)


def factors(tree: Tree) -> list[str]:
    """The weight names a tree multiplies, in premise order.

    Parameters
    ----------
    tree : Tree
        The derivation.

    Returns
    -------
    list of str
        One name per leaf and per weighted rule application.
    """
    _, _, weight, children = tree
    inner = [name for child in children for name in factors(child)]
    return inner if weight is None else [*inner, weight]


def total[N: (Fraction, float, torch.Tensor)](
    trees: Sequence[Tree], weight_of: Mapping[str, N], zero: N, one: N
) -> N:
    """The sum over trees of the product of their weights.

    Parameters
    ----------
    trees : Sequence of Tree
        The derivations.
    weight_of : Mapping[str, N]
        The weight of each name.
    zero : N
        The additive unit.
    one : N
        The multiplicative unit.

    Returns
    -------
    N
        The weight.
    """
    answer = zero
    for tree in trees:
        product = one
        for name in factors(tree):
            product = product * weight_of[name]
        answer = answer + product
    return answer


def best(trees: Sequence[Tree], weight_of: Mapping[str, Fraction]) -> Fraction:
    """The largest product of weights over the trees.

    Parameters
    ----------
    trees : Sequence of Tree
        The derivations.
    weight_of : Mapping[str, Fraction]
        The weight of each name.

    Returns
    -------
    Fraction
        The largest product; zero without a derivation.
    """
    return max(
        (total([tree], weight_of, Fraction(0), Fraction(1)) for tree in trees),
        default=Fraction(0),
    )


def outcome_masses(
    trees: Sequence[Tree],
    weight_of: Mapping[str, Fraction],
    outcome: Callable[[Tree], Term],
) -> dict[Term, Fraction]:
    """The total weight of the trees with each outcome.

    Parameters
    ----------
    trees : Sequence of Tree
        The derivations.
    weight_of : Mapping[str, Fraction]
        The weight of each name.
    outcome : Callable[[Tree], Term]
        The outcome of one derivation.

    Returns
    -------
    dict[Term, Fraction]
        The mass of each outcome.
    """
    masses: dict[Term, Fraction] = {}
    for tree in trees:
        key = outcome(tree)
        weight = total([tree], weight_of, Fraction(0), Fraction(1))
        masses[key] = masses.get(key, Fraction(0)) + weight
    return masses


def fold(
    tree: Tree, algebra: Callable[[str, tuple[Fraction, ...]], Fraction]
) -> Fraction:
    """Interpret one tree by a function of names and child values.

    Parameters
    ----------
    tree : Tree
        The derivation.
    algebra : Callable[[str, tuple of Fraction], Fraction]
        The value of an application from the values of its premises.

    Returns
    -------
    Fraction
        The interpretation.
    """
    name, _, _, children = tree
    return algebra(name, tuple(fold(child, algebra) for child in children))


def nonlinear_fold(
    trees: Sequence[Tree],
    weight_of: Mapping[str, Fraction],
    algebra: Callable[[str, tuple[Fraction, ...]], Fraction],
    observer: Callable[[Fraction], Fraction],
) -> Fraction:
    """The weighted sum over trees of an observer of their interpretation.

    Parameters
    ----------
    trees : Sequence of Tree
        The derivations.
    weight_of : Mapping[str, Fraction]
        The weight of each name.
    algebra : Callable[[str, tuple of Fraction], Fraction]
        The interpretation.
    observer : Callable[[Fraction], Fraction]
        Any function of one interpretation, linear or not.

    Returns
    -------
    Fraction
        The sum.
    """
    answer = Fraction(0)
    for tree in trees:
        weight = total([tree], weight_of, Fraction(0), Fraction(1))
        answer += weight * observer(fold(tree, algebra))
    return answer


def logarithmic_gradient(
    trees: Sequence[Tree], weight_of: Mapping[str, Fraction], name: str
) -> Fraction:
    """The derivative of the log weight in the log of one named weight.

    The derivative is the expected number of uses of the name under the
    normalized weights of the trees.

    Parameters
    ----------
    trees : Sequence of Tree
        The derivations.
    weight_of : Mapping[str, Fraction]
        The weight of each name.
    name : str
        The name differentiated by.

    Returns
    -------
    Fraction
        The derivative.
    """
    mass = Fraction(0)
    used = Fraction(0)
    for tree in trees:
        weight = total([tree], weight_of, Fraction(0), Fraction(1))
        mass += weight
        used += weight * factors(tree).count(name)
    return used / mass
