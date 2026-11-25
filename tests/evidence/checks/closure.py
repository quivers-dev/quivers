"""Checks of matrix closures against exact inverses and enumerated paths."""

from __future__ import annotations

import math
from fractions import Fraction
from functools import partial

import torch

from quivers.categorical.traced import KleeneStarTrace
from quivers.core.algebras import (
    BooleanAlgebra,
    CountingAlgebra,
    LogProbAlgebra,
    MarkovAlgebra,
    MaxPlusAlgebra,
    RealAlgebra,
    Semiring,
)
from quivers.core.morphisms import observed
from quivers.core.objects import CoproductSet, FinSet
from quivers.stochastic._kleene import DivergentClosureError, kleene_star
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)
from tests.evidence.checks._registry import add
from tests.evidence.compare import (
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_tensor,
)
from tests.evidence.oracles import closed_forms, exact
from tests.evidence.problems import logarithm

MATRICES: dict[str, tuple[tuple[str, ...], ...]] = {
    "empty": (),
    "one_loop": (("1/2",),),
    "acyclic": (("0", "1/2", "1/5"), ("0", "0", "3/4"), ("0", "0", "0")),
    "two_cycle": (("1/4", "1/2"), ("1/3", "1/5")),
    "dense": (
        ("1/10", "1/5", "1/20", "1/8"),
        ("1/4", "1/16", "1/9", "1/7"),
        ("1/6", "1/11", "1/13", "1/5"),
        ("1/30", "2/5", "1/3", "1/12"),
    ),
    "disconnected": (("1/2", "0", "0"), ("0", "0", "0"), ("0", "0", "3/4")),
    "near_one": ((f"{2**20 - 1}/{2**20}",),),
    "long_cycle": (
        ("0", "9/10", "0", "0"),
        ("0", "0", "9/10", "0"),
        ("0", "0", "0", "9/10"),
        ("9/10", "0", "0", "0"),
    ),
}
"""Nonnegative matrices whose powers sum to a finite value."""

TINY_EDGE = -1000.0
"""A log weight whose exponential is zero in float64."""

DIVERGENT: dict[str, tuple[tuple[str, ...], ...]] = {
    "at_one": (("1",),),
    "above_one": (("3/2",),),
    "cycle_at_one": (("0", "2"), ("1/2", "0")),
}
"""Nonnegative matrices whose powers do not sum to a finite value."""

CLOSURES: dict[str, ChartSemiring] = {
    "logprob": LOG_PROB,
    "counting": COUNTING,
    "viterbi": VITERBI,
    "boolean": BOOLEAN,
}
"""The semirings a closure is defined in."""

TRACES: dict[str, Semiring] = {
    "boolean": BooleanAlgebra(),
    "counting": CountingAlgebra(),
    "markov": MarkovAlgebra(),
    "logprob": LogProbAlgebra(),
    "maxplus": MaxPlusAlgebra(),
}
"""The algebras of the coproduct trace."""


def rational(rows: tuple[tuple[str, ...], ...]) -> exact.Matrix:
    """Read a matrix of rational strings.

    Parameters
    ----------
    rows : tuple of tuple of str
        The rows.

    Returns
    -------
    exact.Matrix
        The matrix.
    """
    return [[Fraction(value) for value in row] for row in rows]


def encode(matrix: exact.Matrix, log: bool) -> torch.Tensor:
    """A rational matrix as a float64 tensor.

    Parameters
    ----------
    matrix : exact.Matrix
        The matrix.
    log : bool
        Whether the entries are logarithms.

    Returns
    -------
    torch.Tensor
        The tensor, of shape ``(n, n)`` also for ``n = 0``.
    """
    size = len(matrix)
    values = [logarithm(x) if log else float(x) for row in matrix for x in row]
    return torch.tensor(values, dtype=torch.float64).reshape(size, size)


def norm(matrix: exact.Matrix) -> float:
    """The largest row sum of a nonnegative matrix.

    Parameters
    ----------
    matrix : exact.Matrix
        The matrix.

    Returns
    -------
    float
        The norm, at least one.
    """
    return max([1.0, *(float(sum(row)) for row in matrix)])


def star_accuracy(matrix: exact.Matrix, log: bool) -> Accuracy:
    """The accuracy of a closure.

    Eliminating one state updates every entry with one product of three
    factors, one geometric sum, and one addition, and an entry is updated
    once per state. The condition number is the norm of ``(I - A)^-1``.

    Parameters
    ----------
    matrix : exact.Matrix
        The matrix.
    log : bool
        Whether the entries are logarithms.

    Returns
    -------
    Accuracy
        The accuracy.
    """
    size = max(1, len(matrix))
    star = exact.linear_star(matrix)
    sizes = [
        abs(logarithm(x)) for m in (matrix, star) for row in m for x in row if x != 0
    ]
    return Accuracy(
        domain="log" if log else "linear",
        operations=size * (5 * ELEMENTARY + 5) * (1 if log else size),
        magnitude=max([1.0, *sizes]) if log else 1.0,
        condition=norm(star),
    )


def closure_reference(matrix: exact.Matrix, semiring: str) -> torch.Tensor:
    """The closure of a matrix in a semiring, from exact arithmetic.

    Parameters
    ----------
    matrix : exact.Matrix
        The matrix.
    semiring : str
        The semiring's name.

    Returns
    -------
    torch.Tensor
        The closure.
    """
    if semiring == "viterbi":
        return encode(exact.simple_path_star(matrix), True)
    star = exact.linear_star(matrix)
    if semiring == "boolean":
        return encode([[Fraction(int(x != 0)) for x in row] for row in star], False)
    return encode(star, semiring == "logprob")


def check_star(name: str, semiring: str) -> None:
    """Compare a closure with the exact one.

    Parameters
    ----------
    name : str
        The matrix.
    semiring : str
        The semiring's name.
    """
    matrix = rational(MATRICES[name])
    log = semiring in ("logprob", "viterbi")
    entries = (
        [[Fraction(int(x != 0)) for x in row] for row in matrix]
        if semiring == "boolean"
        else matrix
    )
    compare_tensor(
        kleene_star(encode(entries, log), CLOSURES[semiring]),
        closure_reference(matrix, semiring),
        star_accuracy(matrix, log),
        f"{semiring} closure of {name}",
    )


def check_tiny_edge() -> None:
    """Require a path through an edge below the range of float64 to keep its weight."""
    matrix = torch.tensor(
        [[-math.inf, TINY_EDGE], [-math.inf, -math.inf]], dtype=torch.float64
    )
    compare_tensor(
        kleene_star(matrix, LOG_PROB),
        torch.tensor([[0.0, TINY_EDGE], [-math.inf, 0.0]], dtype=torch.float64),
        Accuracy(domain="log", operations=100, magnitude=-TINY_EDGE),
        "closure with a tiny edge",
    )


def check_star_gradient(name: str, semiring: str) -> None:
    """Compare the derivative of a closure with ``A* dA A*``.

    Parameters
    ----------
    name : str
        The matrix.
    semiring : str
        ``"logprob"`` or ``"counting"``.
    """
    matrix = rational(MATRICES[name])
    star = exact.linear_star(matrix)
    log = semiring == "logprob"
    size = len(matrix)
    entries = encode(matrix, log).requires_grad_()
    target = kleene_star(entries, CLOSURES[semiring])[0, size - 1]
    (found,) = torch.autograd.grad(target, entries)
    reference = {}
    for k in range(size):
        for m in range(size):
            value = star[0][k] * star[m][size - 1]
            if log:
                value = value * matrix[k][m] / star[0][size - 1]
            reference[f"{k},{m}"] = value
    forward = star_accuracy(matrix, log)
    compare_gradients(
        {f"{k},{m}": found[k, m] for k in range(size) for m in range(size)},
        reference,
        Accuracy(
            domain="linear",
            operations=5 * forward.operations,
            condition=forward.condition * forward.condition * forward.magnitude,
        ),
        f"gradient of {semiring} closure of {name}",
    )


def check_divergence(name: str, semiring: str) -> None:
    """Require a refusal for a closure that has no finite value.

    Parameters
    ----------
    name : str
        The matrix.
    semiring : str
        The semiring's name.

    Raises
    ------
    Mismatch
        If a number is returned.
    """
    matrix = encode(rational(DIVERGENT[name]), semiring in ("logprob", "viterbi"))
    try:
        value = kleene_star(matrix, CLOSURES[semiring])
    except DivergentClosureError:
        return
    raise Mismatch("refusal", math.inf, f"{semiring} {name} returned {value.tolist()}")


def check_near_boundary() -> None:
    """Compare a closure near its boundary with the geometric closed form."""
    for power in (10, 20, 30, 40):
        ratio = 1 - Fraction(1, 2**power)
        value = kleene_star(
            torch.tensor([[logarithm(ratio)]], dtype=torch.float64), LOG_PROB
        )
        compare_tensor(
            value,
            torch.tensor(
                [[logarithm(closed_forms.geometric(ratio))]], dtype=torch.float64
            ),
            Accuracy(
                domain="log",
                operations=5 * ELEMENTARY,
                magnitude=1.0,
                condition=2.0**power,
            ),
            f"closure of 1 - 2^-{power}",
        )


def check_empty(semiring: str) -> None:
    """Require a closure over no states to keep its shape and precision.

    Parameters
    ----------
    semiring : str
        The semiring's name.
    """
    for dtype in (torch.float32, torch.float64):
        value = kleene_star(torch.zeros(0, 0, dtype=dtype), CLOSURES[semiring])
        compare_tensor(
            value,
            torch.zeros(0, 0, dtype=dtype),
            Accuracy(domain="linear", operations=1),
            f"{semiring} closure over no states",
        )


#: A matrix from ``A + U`` to ``B + U`` with two external and two internal states.
FEEDBACK = (
    ("1/8", "0", "1/4", "1/2"),
    ("1/3", "1/5", "0", "1/6"),
    ("1/7", "1/2", "1/4", "1/3"),
    ("2/5", "0", "1/6", "1/9"),
)
#: An acyclic whole-number matrix for the counting algebra.
FEEDBACK_COUNTS = (
    ("1", "0", "2", "1"),
    ("0", "3", "0", "1"),
    ("2", "1", "0", "4"),
    ("1", "5", "0", "0"),
)


def check_trace(algebra: str) -> None:
    """Compare a trace with ``f_AB + f_AU (f_UU)* f_UB``.

    Parameters
    ----------
    algebra : str
        The algebra's name.
    """
    rows = rational(FEEDBACK_COUNTS if algebra == "counting" else FEEDBACK)
    log = algebra in ("logprob", "maxplus")
    inner = [row[2:] for row in rows[2:]]
    star = (
        exact.simple_path_star(inner) if algebra == "maxplus" else exact.linear_star(inner)
    )
    wanted: exact.Matrix = []
    for a in range(2):
        line = []
        for b in range(2):
            terms = [rows[a][b]]
            for u in range(2):
                for v in range(2):
                    terms.append(rows[a][2 + u] * star[u][v] * rows[2 + v][b])
            if algebra == "maxplus":
                line.append(max(terms))
            elif algebra == "boolean":
                line.append(Fraction(int(any(x != 0 for x in terms))))
            else:
                line.append(sum(terms, Fraction(0)))
        wanted.append(line)
    entries = (
        [[Fraction(int(x != 0)) for x in row] for row in rows]
        if algebra == "boolean"
        else rows
    )
    outer = FinSet(name="A", cardinality=2)
    loop = FinSet(name="U", cardinality=2)
    whole = CoproductSet(components=(outer, loop))
    morphism = observed(whole, whole, encode(entries, log), algebra=TRACES[algebra])
    found = KleeneStarTrace(TRACES[algebra]).trace(morphism, loop, outer, outer).tensor
    compare_tensor(
        found,
        encode(wanted, log),
        Accuracy(
            domain="log" if log else "linear",
            operations=8 * (5 * ELEMENTARY + 5),
            magnitude=8.0 if log else 1.0,
            condition=norm(exact.linear_star(inner)) ** 2,
        ),
        f"{algebra} trace",
    )


def check_trace_refuses_signed() -> None:
    """Require the signed real algebra to be refused by name.

    Raises
    ------
    Mismatch
        If a trace is built or the diagnostic differs.
    """
    algebra = RealAlgebra()
    wanted = f"no coproduct Kleene-star trace is declared for {algebra!r}"
    try:
        KleeneStarTrace(algebra)
    except TypeError as refusal:
        if str(refusal) == wanted:
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, "a trace over the signed reals was built")


def _register() -> None:
    """Register every check of this module."""
    for semiring in CLOSURES:
        for name in MATRICES:
            if name == "empty":
                continue
            # The linear representation of a whole-number algebra keeps
            # the same entries; Boolean entries are zero and one.
            add(
                f"closure/star/{semiring}/{name}",
                "kleene-star",
                "Q7",
                partial(check_star, name, semiring),
            )
        if semiring != "viterbi":
            add(
                f"closure/empty/{semiring}",
                "kleene-star",
                "Q7",
                partial(check_empty, semiring),
            )
    for name in ("one_loop", "acyclic", "two_cycle", "dense", "near_one", "long_cycle"):
        add(
            f"closure/gradient/logprob/{name}",
            "kleene-star",
            "Q1g",
            partial(check_star_gradient, name, "logprob"),
        )
    for name in ("one_loop", "two_cycle", "dense", "near_one"):
        add(
            f"closure/gradient/counting/{name}",
            "kleene-star",
            "Q1g",
            partial(check_star_gradient, name, "counting"),
        )
    for name in DIVERGENT:
        for semiring in ("logprob", "counting"):
            add(
                f"closure/divergence/{semiring}/{name}",
                "kleene-star",
                "Q8",
                partial(check_divergence, name, semiring),
            )
    add(
        "closure/divergence/viterbi/above_one",
        "kleene-star",
        "Q8",
        partial(check_divergence, "above_one", "viterbi"),
    )
    add("closure/near_boundary", "kleene-star", "Q8", check_near_boundary)
    add("closure/tiny_edge", "kleene-star", "Q8", check_tiny_edge)
    for algebra in TRACES:
        add(f"closure/trace/{algebra}", "kleene-star-trace", "Q8", partial(check_trace, algebra))
    add("closure/trace/refuses_signed", "kleene-star-trace", "Q8", check_trace_refuses_signed)


_register()
