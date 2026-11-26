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
    Star,
)
from quivers.core.morphisms import observed
from quivers.core.objects import CoproductSet, FinSet
from quivers.stochastic._kleene import (
    DivergentClosureError,
    UndecidedClosureError,
    kleene_star,
)
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)
from tests.evidence.checks._registry import add
from tests.evidence.compare import (
    differentiate,
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_tensor,
)
from tests.evidence.oracles import closed_forms, exact, gradients
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
    (found,) = differentiate(target, [entries], "closure")
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
        exact.simple_path_star(inner)
        if algebra == "maxplus"
        else exact.linear_star(inner)
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


def check_trace_gradient() -> None:
    """Compare a trace's gradient with the one of its matrix formula.

    The entry from the first input to the first output is
    ``f_AB + f_AU (f_UU)* f_UB``; its derivative in an entry of the
    feedback block is the product of the weights of the walks into and
    out of it.
    """
    rows = rational(FEEDBACK)
    inner = [row[2:] for row in rows[2:]]
    star = exact.linear_star(inner)
    outer = FinSet(name="A", cardinality=2)
    loop = FinSet(name="U", cardinality=2)
    whole = CoproductSet(components=(outer, loop))
    entries = encode(rows, False).requires_grad_()
    algebra = TRACES["markov"]
    morphism = observed(whole, whole, entries, algebra=algebra)
    value = KleeneStarTrace(algebra).trace(morphism, loop, outer, outer).tensor[0, 0]
    (found,) = differentiate(value, [entries], "trace")
    into = [
        sum((rows[0][2 + u] * star[u][v] for u in range(2)), Fraction(0))
        for v in range(2)
    ]
    out = [
        sum((star[v][w] * rows[2 + w][0] for w in range(2)), Fraction(0))
        for v in range(2)
    ]
    reference: dict[str, Fraction] = {"0,0": Fraction(1)}
    for u in range(2):
        reference[f"0,{2 + u}"] = out[u]
        reference[f"{2 + u},0"] = into[u]
        for v in range(2):
            reference[f"{2 + u},{2 + v}"] = into[u] * out[v]
    compare_gradients(
        {
            key: found[int(key.split(",")[0]), int(key.split(",")[1])]
            for key in reference
        },
        reference,
        Accuracy(
            domain="linear",
            operations=40 * (5 * ELEMENTARY + 5),
            condition=norm(star) ** 2,
        ),
        "gradient of the markov trace",
    )


def check_trace_refuses_signed() -> None:
    """Require the signed real algebra to be refused by name.


    Raises
    ------
    Mismatch
        If a trace is built or the diagnostic differs.
    """
    algebra = RealAlgebra()
    wanted = (
        f"no coproduct Kleene-star trace is declared for {algebra!r}: it does not "
        "establish a closure"
    )
    try:
        KleeneStarTrace(algebra)
    except TypeError as refusal:
        if str(refusal) == wanted:
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, "a trace over the signed reals was built")


class Forwarded(ChartSemiring, Star):
    """A semiring that forwards every operation, its closure too, to the log semiring."""

    @property
    def name(self) -> str:
        """A name of its own.

        Returns
        -------
        str
            The name.
        """
        return "ForwardedLogProb"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward a product.

        Parameters
        ----------
        a : torch.Tensor
            The first factor.
        b : torch.Tensor
            The second factor.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return LOG_PROB.tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Forward a reduction.

        Parameters
        ----------
        t : torch.Tensor
            The terms.
        dim : int or tuple of int
            The axes.

        Returns
        -------
        torch.Tensor
            The sum.
        """
        return LOG_PROB.join(t, dim)

    def join_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward a binary sum.

        Parameters
        ----------
        a : torch.Tensor
            The first term.
        b : torch.Tensor
            The second term.

        Returns
        -------
        torch.Tensor
            The sum.
        """
        return LOG_PROB.join_pair(a, b)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Forward a meet.

        Parameters
        ----------
        t : torch.Tensor
            The terms.
        dim : int or tuple of int
            The axes.

        Returns
        -------
        torch.Tensor
            The meet.
        """
        return LOG_PROB.meet(t, dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Forward a negation.

        Parameters
        ----------
        t : torch.Tensor
            The values.

        Returns
        -------
        torch.Tensor
            The negation.
        """
        return LOG_PROB.negate(t)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Forward membership.

        Parameters
        ----------
        t : torch.Tensor
            The values.

        Returns
        -------
        torch.Tensor
            Membership in the log carrier.
        """
        return LOG_PROB.contains(t)

    @property
    def zero(self) -> float:
        """The forwarded zero.

        Returns
        -------
        float
            The zero.
        """
        return LOG_PROB.zero

    @property
    def unit(self) -> float:
        """The forwarded unit.

        Returns
        -------
        float
            The unit.
        """
        return LOG_PROB.unit

    @property
    def star_domain(self) -> str:
        """The forwarded domain.

        Returns
        -------
        str
            The domain of the log semiring's closure.
        """
        return LOG_PROB.star_domain

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """Forward a closure.

        Parameters
        ----------
        matrix : torch.Tensor
            The matrix.

        Returns
        -------
        torch.Tensor
            The closure.
        """
        return LOG_PROB.star(matrix)


#: A cycle whose weights, as exact numbers of float64, sum to zero, while
#: their sum from left to right is the spacing of float64 near two fifths.
ROUNDED = (0.2, 0.4, -0.2, -0.4)


def rounded_cycle(weights: tuple[float, ...]) -> torch.Tensor:
    """A max-plus matrix with one cycle.

    Parameters
    ----------
    weights : tuple of float
        The weights along the cycle.

    Returns
    -------
    torch.Tensor
        The matrix.
    """
    size = len(weights)
    matrix = torch.full((size, size), -math.inf, dtype=torch.float64)
    for state, weight in enumerate(weights):
        matrix[state, (state + 1) % size] = weight
    return matrix


def check_rounded_zero_cycle() -> None:
    """Compare the closure of a cycle whose exact score is zero.

    The best path from ``i`` to ``j`` goes forward along the cycle, so its
    score is the exact sum of the weights between them; going round adds
    zero. Rounding makes one order of the sum positive, which must not
    make the closure divergent.
    """
    size = len(ROUNDED)
    wanted = torch.tensor(
        [
            [
                float(
                    sum(
                        (
                            Fraction(ROUNDED[(i + step) % size])
                            for step in range((j - i) % size)
                        ),
                        Fraction(0),
                    )
                )
                for j in range(size)
            ]
            for i in range(size)
        ],
        dtype=torch.float64,
    )
    compare_tensor(
        kleene_star(rounded_cycle(ROUNDED), VITERBI),
        wanted,
        Accuracy(domain="log", operations=4 * size, magnitude=2.0),
        "best-path closure of a cycle of exact score zero",
    )


def check_zero_entry_gradient() -> None:
    """Compare the derivative of a closure at entries that are zero now.

    The weight of the walks from 0 to 1 in ``[[a, b], [0, 0]]`` is
    ``b / (1 - a)``; at ``a = b = 0`` its derivative in the entry from 0
    to 1 is the one the gradient oracle declares.
    """
    case = gradients.BOUNDARY_CASES[0]
    for semiring in ("counting",):
        leaf = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
        value = kleene_star(leaf, CLOSURES[semiring])[0, 1]
        (found,) = torch.autograd.grad(value, leaf)
        compare_gradients(
            {"a": found[0, 0], "b": found[0, 1]},
            dict(zip(("a", "b"), case.gradient, strict=True)),
            Accuracy(domain="linear", operations=8),
            f"{semiring} closure at zero entries",
        )


def check_undecided_cycle() -> None:
    """Require a cycle whose weight rounding cannot place to be refused as undecided.

    The cycle 2, 0, 1 has log weights ``1``, ``-2^-60`` and ``-1``;
    elimination of state 0 rounds the partial sum ``1 - 2^-60`` to ``1``,
    so the cycle's computed log weight is ``0``.
    The exact one is ``-2^-60``: the closure converges, and no
    computation in float64 with these roundings can tell. The closure
    has to refuse as undecided, not return a number and not call the
    cycle divergent.
    """
    matrix = torch.full((3, 3), -math.inf, dtype=torch.float64)
    matrix[2, 0], matrix[0, 1], matrix[1, 2] = 1.0, -(2.0**-60), -1.0
    try:
        value = kleene_star(matrix, LOG_PROB)
    except UndecidedClosureError:
        return
    except DivergentClosureError as refusal:
        raise Mismatch(
            "refusal", math.inf, "a convergent cycle was refused"
        ) from refusal
    raise Mismatch(
        "refusal", math.inf, f"an undecided closure returned {value.tolist()}"
    )


def check_forwarded_closure() -> None:
    """Require a semiring with the closure capability to have a closure.

    The forwarding semiring is of no class the library names; it has the
    capability, so its closure is the log semiring's.
    """
    matrix = torch.tensor([[-1.0]], dtype=torch.float64)
    compare_tensor(
        kleene_star(matrix, Forwarded()),
        torch.tensor([[-math.log(1 - math.exp(-1))]], dtype=torch.float64),
        Accuracy(domain="log", operations=12),
        "closure in a forwarding semiring",
    )


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
    add("closure/rounded_zero_cycle", "kleene-star", "Q8", check_rounded_zero_cycle)
    add("closure/zero_entry_gradient", "kleene-star", "Q1g", check_zero_entry_gradient)
    add("closure/forwarded", "kleene-star", "Q7", check_forwarded_closure)
    add("closure/undecided", "kleene-star", "Q8", check_undecided_cycle)
    for algebra in TRACES:
        add(
            f"closure/trace/{algebra}",
            "kleene-star-trace",
            "Q8",
            partial(check_trace, algebra),
        )
    add("closure/trace/gradient", "kleene-star-trace", "Q1g", check_trace_gradient)
    add(
        "closure/trace/refuses_signed",
        "kleene-star-trace",
        "Q8",
        check_trace_refuses_signed,
    )


_register()
