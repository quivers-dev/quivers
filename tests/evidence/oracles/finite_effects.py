"""The law of a program that draws finitely many values.

A program is a function of a `Draws` object. It asks for a value with
``draws.choose(address, probabilities)`` and may multiply its weight
with ``draws.score(address, weight)``. The oracle runs the program once
for every sequence of values the program can see. Every draw has a name
of its own, so a draw that is shared or repeated is visible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import didactic.api as dx


class RepeatedAddress(ValueError):
    """Two draws of one run have the same name."""


class Draws:
    """One run of a program along a prescribed sequence of values.

    Parameters
    ----------
    prescribed : Sequence of int
        The values of the first draws; later draws take value zero and
        are recorded as open.
    """

    def __init__(self, prescribed: Sequence[int]) -> None:
        self._prescribed = list(prescribed)
        self.addresses: list[str] = []
        self.values: list[int] = []
        self.supports: list[int] = []
        self.weight = Fraction(1)

    def choose(self, address: str, probabilities: Sequence[Fraction]) -> int:
        """Draw an index.

        Parameters
        ----------
        address : str
            The name of the draw.
        probabilities : Sequence of Fraction
            The weight of each index.

        Returns
        -------
        int
            The index of this run.


        Raises
        ------
        RepeatedAddress
            If the run already drew under this name.
        """
        if address in self.addresses:
            raise RepeatedAddress(address)
        position = len(self.values)
        value = self._prescribed[position] if position < len(self._prescribed) else 0
        self.addresses.append(address)
        self.values.append(value)
        self.supports.append(len(probabilities))
        self.weight *= probabilities[value]
        return value

    def score(self, weight: Fraction) -> None:
        """Multiply the weight of the run.

        Parameters
        ----------
        weight : Fraction
            The factor.
        """
        self.weight *= weight


class Outcome(dx.Model):
    """One run of a program.

    Parameters
    ----------
    result : int
        What the program returned.
    addresses : tuple of str
        The draws in order.
    values : tuple of int
        Their values.
    numerator : int
        The numerator of the weight.
    denominator : int
        The denominator of the weight.
    """

    result: int
    addresses: tuple[str, ...]
    values: tuple[int, ...]
    numerator: int
    denominator: int

    @property
    def weight(self) -> Fraction:
        """The weight of the run.

        Returns
        -------
        Fraction
            The weight.
        """
        return Fraction(self.numerator, self.denominator)


def runs(program: Callable[[Draws], int]) -> list[Outcome]:
    """Every run of a program with nonzero or zero weight.

    Parameters
    ----------
    program : Callable[[Draws], int]
        The program.

    Returns
    -------
    list of Outcome
        One outcome per sequence of values.
    """
    found: list[Outcome] = []
    pending: list[tuple[int, ...]] = [()]
    while pending:
        prefix = pending.pop()
        draws = Draws(prefix)
        result = program(draws)
        found.append(
            Outcome(
                result=result,
                addresses=tuple(draws.addresses),
                values=tuple(draws.values),
                numerator=draws.weight.numerator,
                denominator=draws.weight.denominator,
            )
        )
        for position in range(len(prefix), len(draws.values)):
            for value in range(1, draws.supports[position]):
                pending.append((*draws.values[:position], value))
    return found


def law(program: Callable[[Draws], int]) -> dict[int, Fraction]:
    """The total weight of each result of a program.

    Parameters
    ----------
    program : Callable[[Draws], int]
        The program.

    Returns
    -------
    dict[int, Fraction]
        The weights, not normalized.
    """
    answer: dict[int, Fraction] = {}
    for outcome in runs(program):
        answer[outcome.result] = (
            answer.get(outcome.result, Fraction(0)) + outcome.weight
        )
    return answer


def evidence(program: Callable[[Draws], int]) -> Fraction:
    """The total weight of all runs of a program.

    Parameters
    ----------
    program : Callable[[Draws], int]
        The program.

    Returns
    -------
    Fraction
        The total.
    """
    return sum(law(program).values(), Fraction(0))


def joint(program: Callable[[Draws], int], fixed: Mapping[str, int]) -> Fraction:
    """The total weight of the runs that agree with fixed draws.

    Parameters
    ----------
    program : Callable[[Draws], int]
        The program.
    fixed : Mapping[str, int]
        Values of some draws, by name.

    Returns
    -------
    Fraction
        The weight of the runs whose named draws have these values.
    """
    answer = Fraction(0)
    for outcome in runs(program):
        drawn = dict(zip(outcome.addresses, outcome.values, strict=True))
        if all(drawn.get(name) == value for name, value in fixed.items()):
            answer += outcome.weight
    return answer
