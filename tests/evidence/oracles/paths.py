"""Enumeration of the accepting paths of a weighted machine.

A machine is a tuple of arcs with one label per tape and ``None`` for no
symbol, with marked initial and final states. Weights are names; their
values are given with the query. Arcs with equal labels and different
names are different arcs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction

import didactic.api as dx
import torch

type Label = str | None
"""One tape's symbol, or ``None`` for no symbol."""

type Strings = tuple[tuple[str, ...], ...]
"""One string per tape."""


class Arc(dx.Model):
    """One arc of a machine.

    Parameters
    ----------
    name : str
        The arc's name.
    source : int
        The state it leaves.
    target : int
        The state it enters.
    labels : tuple of (str or None)
        One label per tape.
    weight : str
        The name of its weight.
    """

    name: str
    source: int
    target: int
    labels: tuple[str | None, ...]
    weight: str


class Mark(dx.Model):
    """An initial or final state.

    Parameters
    ----------
    state : int
        The state.
    weight : str
        The name of its weight.
    """

    state: int
    weight: str


class Machine(dx.Model):
    """A weighted machine over several tapes.

    Parameters
    ----------
    arcs : tuple of Arc
        The arcs.
    initial : tuple of Mark
        The initial states.
    final : tuple of Mark
        The final states.
    """

    arcs: tuple[Arc, ...]
    initial: tuple[Mark, ...]
    final: tuple[Mark, ...]


class Path(dx.Model):
    """One accepting path.

    Parameters
    ----------
    start : Mark
        The initial state.
    arcs : tuple of Arc
        The arcs in order.
    end : Mark
        The final state.
    """

    start: Mark
    arcs: tuple[Arc, ...]
    end: Mark

    def factors(self) -> list[str]:
        """The weight names along the path.

        Returns
        -------
        list of str
            The names in path order.
        """
        return [self.start.weight, *(arc.weight for arc in self.arcs), self.end.weight]

    def strings(self) -> Strings:
        """The string the path writes on each tape.

        Returns
        -------
        Strings
            One string per tape; no tapes when the path has no arc.
        """
        if not self.arcs:
            return ()
        return tuple(
            tuple(label for arc in self.arcs if (label := arc.labels[tape]) is not None)
            for tape in range(len(self.arcs[0].labels))
        )


class CyclicMachine(ValueError):
    """The machine has infinitely many accepting paths and no bound was given."""


def _useful(machine: Machine, arcs: Sequence[Arc]) -> list[Arc]:
    """The arcs on some accepting path.

    Parameters
    ----------
    machine : Machine
        The machine.
    arcs : Sequence of Arc
        The arcs considered.

    Returns
    -------
    list of Arc
        The arcs whose source is reachable and whose target reaches a
        final state.
    """
    forward = {mark.state for mark in machine.initial}
    backward = {mark.state for mark in machine.final}
    changed = True
    while changed:
        changed = False
        for arc in arcs:
            if arc.source in forward and arc.target not in forward:
                forward.add(arc.target)
                changed = True
            if arc.target in backward and arc.source not in backward:
                backward.add(arc.source)
                changed = True
    return [arc for arc in arcs if arc.source in forward and arc.target in backward]


def _has_cycle(arcs: Sequence[Arc]) -> bool:
    """Whether a set of arcs contains a directed cycle.

    Parameters
    ----------
    arcs : Sequence of Arc
        The arcs.

    Returns
    -------
    bool
        Whether some state reaches itself.
    """
    remaining = list(arcs)
    while True:
        sources = {arc.source for arc in remaining}
        kept = [arc for arc in remaining if arc.target in sources]
        if len(kept) == len(remaining):
            return bool(remaining)
        remaining = kept


def accepting_paths(machine: Machine, *, max_arcs: int | None = None) -> list[Path]:
    """Every accepting path of a machine.

    Parameters
    ----------
    machine : Machine
        The machine.
    max_arcs : int or None
        The greatest number of arcs. It is a restriction of the model and
        is required for a cyclic machine.

    Returns
    -------
    list of Path
        The paths, each once.

    Raises
    ------
    CyclicMachine
        If the machine has infinitely many accepting paths and no bound
        was given.
    """
    arcs = _useful(machine, machine.arcs)
    if max_arcs is None and _has_cycle(arcs):
        raise CyclicMachine("the accepting paths are infinite")

    def walk(state: int, taken: tuple[Arc, ...]) -> Iterator[tuple[int, tuple[Arc, ...]]]:
        """The walks that extend a prefix.

        Parameters
        ----------
        state : int
            The state reached.
        taken : tuple of Arc
            The arcs so far.

        Yields
        ------
        tuple of (int, tuple of Arc)
            A reached state and the arcs taken to it.
        """
        yield state, taken
        if max_arcs is not None and len(taken) == max_arcs:
            return
        for arc in arcs:
            if arc.source == state:
                yield from walk(arc.target, (*taken, arc))

    return [
        Path(start=start, arcs=taken, end=end)
        for start in machine.initial
        for state, taken in walk(start.state, ())
        for end in machine.final
        if end.state == state
    ]


def total[N: (Fraction, float, torch.Tensor)](
    paths: Sequence[Path], weight_of: Mapping[str, N], zero: N, one: N
) -> N:
    """The sum over paths of the product of their weights.

    Parameters
    ----------
    paths : Sequence of Path
        The paths.
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
    for path in paths:
        product = one
        for name in path.factors():
            product = product * weight_of[name]
        answer = answer + product
    return answer


def best(paths: Sequence[Path], weight_of: Mapping[str, Fraction]) -> Fraction:
    """The largest product of weights over the paths.

    Parameters
    ----------
    paths : Sequence of Path
        The paths.
    weight_of : Mapping[str, Fraction]
        The weight of each name.

    Returns
    -------
    Fraction
        The largest product; zero without a path.
    """
    return max(
        (total([path], weight_of, Fraction(0), Fraction(1)) for path in paths),
        default=Fraction(0),
    )


def relation(
    machine: Machine, weight_of: Mapping[str, Fraction], *, tapes: int
) -> dict[Strings, Fraction]:
    """The weight of every string tuple an acyclic machine relates.

    Parameters
    ----------
    machine : Machine
        The machine, whose accepting paths are finite.
    weight_of : Mapping[str, Fraction]
        The weight of each name.
    tapes : int
        The number of tapes.

    Returns
    -------
    dict[Strings, Fraction]
        The table, complete for the machine.
    """
    table: dict[Strings, Fraction] = {}
    for path in accepting_paths(machine):
        key = path.strings() or ((),) * tapes
        weight = total([path], weight_of, Fraction(0), Fraction(1))
        table[key] = table.get(key, Fraction(0)) + weight
    return table


def compose_relations(
    left: Mapping[Strings, Fraction], right: Mapping[Strings, Fraction]
) -> dict[Strings, Fraction]:
    """The relation that sums over the string two relations share.

    Parameters
    ----------
    left : Mapping[Strings, Fraction]
        A table of pairs ``(x, y)``.
    right : Mapping[Strings, Fraction]
        A table of pairs ``(y, z)``.

    Returns
    -------
    dict[Strings, Fraction]
        The table of pairs ``(x, z)``.
    """
    table: dict[Strings, Fraction] = {}
    for (x, shared), first in left.items():
        for (other, z), second in right.items():
            if shared == other:
                table[(x, z)] = table.get((x, z), Fraction(0)) + first * second
    return table


def conditional(table: Mapping[Strings, Fraction]) -> dict[Strings, Fraction]:
    """Normalize a table of pairs over the second string.

    Parameters
    ----------
    table : Mapping[Strings, Fraction]
        A table of pairs ``(x, y)``.

    Returns
    -------
    dict[Strings, Fraction]
        ``p(y | x)`` for every ``x`` of nonzero mass.
    """
    mass: dict[tuple[str, ...], Fraction] = {}
    for (x, _), weight in table.items():
        mass[x] = mass.get(x, Fraction(0)) + weight
    return {
        (x, y): weight / mass[x] for (x, y), weight in table.items() if mass[x] != 0
    }
