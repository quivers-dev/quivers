"""Problems stated as plain data, and their references from the oracles.

A problem holds rational weights by name. An executor reads it through
an adapter; the reference is computed by an oracle from the same data.
The operation counts that set the tolerances are the pinned numbers of
`tests.evidence.catalogue`, which `operation_bound` derives again.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from tests.evidence.compare import ELEMENTARY, Accuracy
from tests.evidence.oracles import derivations as tree_oracle
from tests.evidence.oracles.derivations import (
    AxiomRow,
    RuleRow,
    SideCondition,
    Term,
    Tree,
    Variable,
)


class DeductionProblem(ValueModel):
    """A weighted deduction with one goal item.

    Parameters
    ----------
    name : str
        The problem.
    axioms : tuple of AxiomRow
        Occurrence names and items.
    rules : tuple of RuleRow
        Rule names, premises, conclusions, and weight names.
    goal : Term
        The goal item.
    weights : dict of str to str
        The parameter of every axiom occurrence and rule weight.
    values : dict of str to str
        The rational value of every parameter.
    sides : dict of str to SideCondition
        Side conditions by rule name.
    height : int or None
        A height bound that is part of the model, for a cyclic system.
    """

    name: str
    axioms: tuple[AxiomRow, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    rules: tuple[RuleRow, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    goal: Term = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    weights: dict[str, str]
    values: dict[str, str]
    sides: dict[str, SideCondition] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    height: int | None = None

    def rational(self) -> dict[str, Fraction]:
        """The exact weight of every occurrence and rule.

        Returns
        -------
        dict of str to Fraction
            The weights by occurrence or rule weight name.
        """
        return {
            name: Fraction(self.values[parameter])
            for name, parameter in self.weights.items()
        }


def _v(name: str) -> Variable:
    """A pattern variable.

    Parameters
    ----------
    name : str
        Its name.

    Returns
    -------
    Variable
        The variable.
    """
    return Variable(name)


REPEATED_PREMISE = DeductionProblem(
    name="repeated_premise",
    axioms=(("np", ("NP",)),),
    rules=(("pair", (("NP",), ("NP",)), ("S",), "r"),),
    goal=("S",),
    weights={"np": "p", "r": "q"},
    values={"p": "3/7", "q": "5/11"},
)
"""One item fills both premises of a rule."""

THREE_EQUAL = DeductionProblem(
    name="three_equal_premises",
    axioms=(("one", ("A",)), ("two", ("B",))),
    rules=(
        ("from_b", (("B",),), ("A",), "r"),
        ("triple", (("A",), ("A",), ("A",)), ("S",), None),
    ),
    goal=("S",),
    weights={"one": "p", "two": "q", "r": "p"},
    values={"p": "2/5", "q": "3/4"},
)
"""One item fills three premises and receives two updates; a parameter is shared."""

DIAMOND = DeductionProblem(
    name="diamond",
    axioms=(("start", ("n", 0)),),
    rules=tuple(
        row
        for i in range(3)
        for row in (
            (f"straight{i}", (("n", i),), ("n", i + 1), f"s{i}"),
            (f"out{i}", (("n", i),), ("m", i), f"o{i}"),
            (f"back{i}", (("m", i),), ("n", i + 1), None),
        )
    ),
    goal=("n", 3),
    weights={
        "start": "w",
        **{f"s{i}": "a" if i < 2 else "b" for i in range(3)},
        **{f"o{i}": "b" if i < 2 else "c" for i in range(3)},
    },
    values={"w": "7/9", "a": "1/3", "b": "2/3", "c": "5/8"},
)
"""Two routes at each of three layers, with shared parameters."""

SKIP = DeductionProblem(
    name="skip",
    axioms=(("start", ("n", 0)),),
    rules=tuple(
        (f"step{i}_{d}", (("n", i),), ("n", i + d), f"d{d}")
        for i in range(6)
        for d in (1, 2)
        if i + d <= 6
    ),
    goal=("n", 6),
    weights={"start": "w", "d1": "a", "d2": "b"},
    values={"w": "1", "a": "3/5", "b": "1/4"},
)
"""Steps of one and two along a line."""

REACH = DeductionProblem(
    name="reach",
    axioms=(
        ("start", ("reach", 0)),
        *((f"e{i}", ("edge", i, i + 1)) for i in range(4)),
        ("e02", ("edge", 0, 2)),
        ("e14", ("edge", 1, 4)),
    ),
    rules=(
        (
            "step",
            (("reach", _v("i")), ("edge", _v("i"), _v("j"))),
            ("reach", _v("j")),
            "r",
        ),
    ),
    goal=("reach", 4),
    weights={
        "start": "w",
        "r": "r",
        **{f"e{i}": f"e{i}" for i in range(4)},
        "e02": "f",
        "e14": "f",
    },
    values={
        "w": "4/5",
        "r": "1/2",
        "e0": "1/3",
        "e1": "2/3",
        "e2": "1/7",
        "e3": "6/7",
        "f": "3/8",
    },
)
"""Reachability with a variable shared by two premises."""

AMBIGUOUS = DeductionProblem(
    name="ambiguous",
    axioms=tuple((f"w{i}", ("span", i, i + 1)) for i in range(5)),
    rules=(
        (
            "join",
            (("span", _v("i"), _v("k")), ("span", _v("k"), _v("j"))),
            ("span", _v("i"), _v("j")),
            "r",
        ),
    ),
    goal=("span", 0, 5),
    weights={"r": "r", **{f"w{i}": f"w{i % 2}" for i in range(5)}},
    values={"r": "2/9", "w0": "3/5", "w1": "4/7"},
)
"""Every binary bracketing of five words."""

NESTED = DeductionProblem(
    name="nested",
    axioms=(("a", ("has", ("f", "x"))), ("b", ("has", ("g", ("f", "x"))))),
    rules=(
        ("strip", (("has", ("g", _v("t"))),), ("has", _v("t")), "r"),
        ("wrap", (("has", ("f", _v("t"))),), ("done", _v("t")), None),
    ),
    goal=("done", "x"),
    weights={"a": "p", "b": "q", "r": "r"},
    values={"p": "1/6", "q": "5/6", "r": "3/10"},
)
"""Items with constructors inside items."""

HIGH_ARITY = DeductionProblem(
    name="high_arity",
    axioms=(
        ("a0", ("p", 0)),
        ("a1", ("p", 1)),
        ("b0", ("q", 0)),
        ("b1", ("q", 1)),
    ),
    rules=(
        (
            "four",
            (("p", _v("x")), ("q", _v("x")), ("p", _v("y")), ("q", _v("y"))),
            ("g",),
            "r",
        ),
    ),
    goal=("g",),
    weights={"a0": "a", "a1": "b", "b0": "c", "b1": "a", "r": "r"},
    values={"a": "2/7", "b": "3/5", "c": "1/9", "r": "7/4"},
)
"""One rule with four premises and two shared variables."""

ORDERED_SIDE = DeductionProblem(
    name="side_condition",
    axioms=tuple((f"x{i}", ("x", i)) for i in range(3)),
    rules=(("pair", (("x", _v("i")), ("x", _v("j"))), ("y",), None),),
    goal=("y",),
    weights={f"x{i}": f"x{i}" for i in range(3)},
    values={"x0": "1/2", "x1": "1/3", "x2": "1/5"},
    sides={"pair": lambda binding: str(binding["i"]) < str(binding["j"])},
)
"""A side condition that keeps one order of each pair."""

EQUAL_PROOFS = DeductionProblem(
    name="equal_proofs",
    axioms=(("a", ("a",)), ("b", ("b",))),
    rules=(
        ("left", (("a",),), ("out",), None),
        ("right", (("b",),), ("out",), None),
    ),
    goal=("out",),
    weights={"a": "p", "b": "p"},
    values={"p": "3/8"},
)
"""Two proofs of one outcome with equal weight."""

EXACT_ZERO = DeductionProblem(
    name="exact_zero",
    axioms=(("a", ("a",)), ("b", ("b",))),
    rules=(
        ("left", (("a",),), ("out",), None),
        ("right", (("b",),), ("out",), None),
    ),
    goal=("out",),
    weights={"a": "p", "b": "q"},
    values={"p": "0", "q": "2/3"},
)
"""A contribution whose current weight is zero arrives first."""

TINY = DeductionProblem(
    name="tiny",
    axioms=(("a", ("a",)), ("b", ("b",))),
    rules=(
        ("left", (("a",),), ("out",), None),
        ("right", (("b",),), ("out",), None),
    ),
    goal=("out",),
    weights={"a": "p", "b": "q"},
    values={"p": f"1/{10**450}", "q": f"1/{3 * 10**450}"},
)
"""Weights near the smallest representable number."""

IMPOSSIBLE = DeductionProblem(
    name="impossible",
    axioms=(("a", ("a",)),),
    rules=(("step", (("b",),), ("out",), None),),
    goal=("out",),
    weights={"a": "p"},
    values={"p": "1/2"},
)
"""A goal with no derivation."""

EDIT = DeductionProblem(
    name="edit_lattice",
    axioms=(("start", ("at", 0, 0)),),
    rules=tuple(
        (f"{move}{i}{j}", (("at", i, j),), ("at", i + di, j + dj), weight)
        for i in range(3)
        for j in range(4)
        for move, di, dj, weight in (
            ("delete", 1, 0, "del"),
            ("insert", 0, 1, "ins"),
            ("match", 1, 1, "same" if "ab"[i : i + 1] == "aba"[j : j + 1] else "sub"),
        )
        if i + di <= 2 and j + dj <= 3
    ),
    goal=("at", 2, 3),
    weights={"start": "w", "del": "d", "ins": "d", "same": "m", "sub": "s"},
    values={"w": "1", "d": "1/10", "m": "1/2", "s": "1/20"},
)
"""Every alignment of ``ab`` with ``aba``."""

DEDUCTIONS: dict[str, DeductionProblem] = {
    problem.name: problem
    for problem in (
        REPEATED_PREMISE,
        THREE_EQUAL,
        DIAMOND,
        SKIP,
        REACH,
        AMBIGUOUS,
        NESTED,
        HIGH_ARITY,
        ORDERED_SIDE,
        EQUAL_PROOFS,
        EXACT_ZERO,
        TINY,
        IMPOSSIBLE,
        EDIT,
    )
}
"""The acyclic deduction problems by name."""


@lru_cache(maxsize=None)
def trees_of(name: str) -> tuple[Tree, ...]:
    """Every derivation of a problem's goal.

    Parameters
    ----------
    name : str
        The problem.

    Returns
    -------
    tuple of Tree
        The derivations.
    """
    problem = DEDUCTIONS[name]
    return tuple(
        tree_oracle.derivations(
            problem.axioms,
            problem.rules,
            problem.goal,
            problem.sides,
            height=problem.height,
        )
    )


@lru_cache(maxsize=None)
def operation_bound(name: str) -> int:
    """Roundings on the longest path from an input of a problem to its goal.

    Each grounded rule application rounds once per premise and once for
    its weight, and contributes one sum; in the logarithmic domain a sum
    is at most three elementary functions and three arithmetic
    operations. A path visits a grounded application at most once, so
    their total is a bound. The factor two covers the conversion of the
    input to a logarithm and of the reference back.

    Parameters
    ----------
    name : str
        The problem.

    Returns
    -------
    int
        The bound.
    """
    problem = DEDUCTIONS[name]
    per_sum = 3 * ELEMENTARY + 3
    instances = tree_oracle.ground(problem.axioms, problem.rules, problem.sides)
    applications = sum(len(premises) + 1 + per_sum for _, premises, _ in instances)
    return 2 * (applications + per_sum * len(problem.axioms))


def logarithm(value: Fraction) -> float:
    """The logarithm of a nonnegative rational.

    Parameters
    ----------
    value : Fraction
        The rational.

    Returns
    -------
    float
        Its natural logarithm; minus infinity at zero.
    """
    if value == 0:
        return -math.inf
    # An integer logarithm of each part is accurate where the quotient
    # itself would leave the range of float64.
    if abs(value.numerator).bit_length() + value.denominator.bit_length() > 1000:
        return math.log(value.numerator) - math.log(value.denominator)
    return math.log(float(value))


def magnitude(problem: DeductionProblem) -> float:
    """The largest magnitude of a logarithm in a run of a problem.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.

    Returns
    -------
    float
        The largest magnitude over weights and over products along a
        derivation, at least one.
    """
    rational = problem.rational()
    sizes = [1.0]
    sizes += [abs(logarithm(w)) for w in rational.values() if w != 0]
    for tree in trees_of(problem.name):
        sizes.append(
            sum(
                abs(logarithm(rational[name]))
                for name in tree_oracle.factors(tree)
                if rational[name] != 0
            )
        )
    return max(sizes)


def accuracy_of(
    name: str, domain: str, operations: Mapping[str, int], *, backward: bool = False
) -> Accuracy:
    """The accuracy a run of a problem has to reach.

    Reverse accumulation performs at most four operations for each
    forward operation (Baur and Strassen 1983), so a gradient counts four
    times the roundings, plus the forward ones. The derivative of a log
    weight is a ratio of two such sums and stays a relative error.

    Parameters
    ----------
    name : str
        The problem.
    domain : str
        ``"log"`` or ``"linear"``.
    operations : Mapping[str, int]
        The pinned operation counts.
    backward : bool
        Whether the query is a gradient.

    Returns
    -------
    Accuracy
        The accuracy.
    """
    problem = DEDUCTIONS[name]
    count = operations[name] * (5 if backward else 1)
    size = magnitude(problem)
    if backward:
        return Accuracy(domain="linear", operations=count, condition=size)
    if domain == "log":
        return Accuracy(domain="log", operations=count, magnitude=size)
    return Accuracy(domain="linear", operations=count)
