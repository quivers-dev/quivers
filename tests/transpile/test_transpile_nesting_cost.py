"""Transpiling a deeply nested expression costs work linear in its depth.

A let that sums ``n`` builtin calls is a left-nested chain ``n`` levels
deep. Each level adds a fixed number of IR records, so the number of
`didactic` model constructions the transpile performs grows by the same
amount per added term. A constructor that re-encodes and re-decodes a
nested child instead doubles the work at every level, which made a
twelve-term let take most of a minute and a 44-term one never finish.
The tests count constructions rather than time, so they hold on any
machine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

import didactic.api as dx
import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile

_NAMES = ("exp", "log", "sqrt", "tanh", "sin", "cos", "floor", "ceil", "erf", "atan")


def _chain(terms: int) -> str:
    """A program whose one let sums ``terms`` builtin calls.

    Parameters
    ----------
    terms : int
        How many calls the let adds, at most ``2 * len(_NAMES)``.

    Returns
    -------
    str
        The QVR source.
    """
    calls = " + ".join(f"{name}(0.4)" for name in (_NAMES * 2)[:terms])
    return f"""object Step : FinSet 3
object Val : Real 1
program p : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Step <- Normal(loc=mu, scale=1.0)
    let c = {calls}
    score s = mu * c
    return mu
export p
"""


class _Counter:
    """Counts `didactic` model constructions while installed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The fixture the counting constructor is installed through.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        monkeypatch.setattr(dx.Model, "__init__", self._counted(dx.Model.__init__))

    def _counted[**P](self, function: Callable[P, None]) -> Callable[P, None]:
        """function, counting each call.

        Parameters
        ----------
        function : Callable[P, None]
            The constructor to count.

        Returns
        -------
        Callable[P, None]
            The constructor, incrementing the count before each call.
        """

        def counting(*args: P.args, **kwargs: P.kwargs) -> None:
            self.count += 1
            function(*args, **kwargs)

        return counting


@pytest.fixture
def counter(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Counter]:
    yield _Counter(monkeypatch)


def _constructions(counter: _Counter, terms: int) -> int:
    """The model constructions one transpile of a ``terms``-term chain makes.

    Parameters
    ----------
    counter : _Counter
        The installed counter.
    terms : int
        The chain's length.

    Returns
    -------
    int
        Constructions during the transpile alone, parsing excluded.
    """
    module = parse(_chain(terms))
    counter.count = 0
    transpile(module, target="pyro")
    return counter.count


def test_each_added_term_costs_the_same_work(counter: _Counter) -> None:
    # Linear growth adds the same work per four terms at every depth;
    # doubling per level multiplies it, by 2**4 = 16 here. A tolerance of
    # twice the first increment admits small bookkeeping differences and
    # nothing exponential.
    short, middle, long = (_constructions(counter, n) for n in (4, 8, 12))
    assert long - middle <= 2 * (middle - short), (short, middle, long)


class _Expr(dx.TaggedUnion, discriminator="kind"):
    """An expression tree of the shape a let chain lowers to."""


class _Lit(_Expr):
    """A literal leaf."""

    value: float
    kind: Literal["lit"] = "lit"


class _Call(_Expr):
    """A call holding its arguments in a tuple field."""

    func: str
    args: tuple[_Expr, ...]
    kind: Literal["call"] = "call"


def test_a_nested_tuple_field_is_built_once_per_node(counter: _Counter) -> None:
    # A left-nested chain of calls of depth n has 2n + 1 nodes, so building
    # it constructs 2n + 1 models and reading its spine constructs none.
    for depth in (4, 8, 12):
        counter.count = 0
        node: _Expr = _Lit(value=0.0)
        for index in range(depth):
            node = _Call(func="f", args=(node, _Lit(value=float(index))))
        assert counter.count == 2 * depth + 1
        counter.count = 0
        spine: _Expr = node
        while isinstance(spine, _Call):
            spine = spine.args[0]
        assert counter.count == 0
