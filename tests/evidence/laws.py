"""The law grid: every law a member of the algebra hierarchy claims, checked.

A row is a shipped member of the hierarchy of `quivers.core.algebras`
(the chart semirings of `quivers.stochastic.semiring` included) and a
column is a law of `quivers.core.algebras.LAWS`. A cell is the check of
one law the member claims. It runs in float64 and float32 over every
pair and triple of the member's samples and over batches with axes of
size zero, so a law that holds on numbers and fails on an empty batch is
rejected.

The operations of every member are also compared with a reference
written here in plain Python numbers, independently of the library: a
sum replaced by a maximum, or a product whose factors are exchanged,
satisfies every law and is caught only there.

Every law has a lawless member: a member that claims the law, whose
check has to reject it. A check of a law about the implementation turns
an exception into a rejection, so a reduction that raises over no terms
fails as clearly as one that returns a wrong type.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from collections.abc import Callable, Sequence
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.core._carriers import reduction_axes
from quivers.core.laws import FIRST_MAXIMIZER, LAWS
from quivers.core.algebras import (
    BOOLEAN,
    BOOLEAN_DUAL,
    COUNTING,
    DUAL_GODEL,
    DUAL_LUKASIEWICZ,
    GODEL,
    LOG_PROB,
    LUKASIEWICZ,
    MARKOV,
    MAX_PLUS,
    PROBABILITY,
    PRODUCT_FUZZY,
    REAL,
    REICHENBACH,
    TROPICAL,
    Carrier,
    CompositionRule,
    CountingAlgebra,
    CustomAlgebra,
    CustomSemigroupoid,
    FreeSemimodule,
    GodelAlgebra,
    Lawful,
    MarkovAlgebra,
    MatrixSemiring,
    MaxPlusAlgebra,
    NoZeroDivisors,
    RealAlgebra,
    Ring,
    Selective,
    Semifield,
    Star,
    DivergentClosureError,
    Semimodule,
    StandardMVAlgebra,
    TropicalAlgebra,
    UnitalCarrier,
    ZeroSumFree,
)
from quivers.stochastic import semiring as chart
from tests.evidence.compare import Accuracy, Mismatch, compare_tensor
from tests.evidence.oracles import exact
from tests.evidence.mutations import Mutation, Rewrite

type Element = float | list[Element]
"""One carrier element as plain Python numbers: a number, or nested lists."""

type Plus = Callable[[Sequence[Element]], Element]
"""The sum of a finite family of elements."""

type Times = Callable[[Element, Element], Element]
"""A binary product, or a scalar acting on a module element."""

DTYPES = (torch.float64, torch.float32)
"""The precisions every cell runs in."""

FORM_DTYPES = (torch.float64, torch.float32, torch.int64, torch.bool)
"""The dtypes in which the zero and the unit are asked for."""

SHAPES: tuple[tuple[int, ...], ...] = ((), (0,), (3,), (2, 0), (0, 2, 3))
"""Index shapes of the zero and the unit, with axes of size zero."""

INFINITY = math.inf

CLAMP = 1.0 - 1e-7
"""The largest term of a noisy-OR, which the product t-norm algebra declares."""


class Reference(ValueModel):
    """The operations of a member in plain Python numbers.

    Parameters
    ----------
    plus : Plus
        The sum of a finite family.
    times : Times or None
        The product, or the action of a scalar on a module element.
    zero : Element
        The zero.
    unit : Element or None
        The unit, ``None`` for a module or a rule without one.
    second : Callable or None
        The matrix of second derivatives of ``a ⊕ b`` in ``(a, b)``, for
        a scalar sum that has one.
    """

    plus: Plus = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    times: Times | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    zero: Element | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    unit: Element | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    second: Callable[[float, float], list[list[float]]] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class Member(ValueModel):
    """One row of the law grid.

    Parameters
    ----------
    name : str
        The row.
    structure : Lawful
        The member.
    samples : tuple of Element
        Elements of the carrier, the zero and the unit among them when
        the member has them.
    reference : Reference
        Its operations in plain numbers.
    domain : str
        ``"exact"`` when every operation on the samples is exact in both
        precisions, ``"log"`` for a log-domain tolerance, ``"linear"``
        for a relative one.
    operations : int
        Roundings on the longest path of a law's two sides.
    magnitude : float
        The largest magnitude of a logarithm that occurs, for ``"log"``.
    best : str
        ``"largest"`` or ``"smallest"``: the operand a selective sum
        keeps; ``""`` for a member that does not select.
    scalars : tuple of Element
        Samples of the acting semiring of a module; empty otherwise.
    scalar_reference : Reference or None
        The operations of the acting semiring of a module.
    """

    name: str
    structure: Lawful = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    samples: tuple[Element, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    reference: Reference = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    domain: Literal["exact", "log", "linear"] = "exact"
    operations: int = 0
    magnitude: float = 1.0
    best: Literal["largest", "smallest", ""] = ""
    scalars: tuple[Element, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    scalar_reference: Reference | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


# -- references -------------------------------------------------------------


def _total(values: Sequence[Element]) -> float:
    """The ordinary sum.

    Parameters
    ----------
    values : Sequence of Element
        Numbers.

    Returns
    -------
    float
        Their sum.
    """
    return float(sum(_number(v) for v in values))


def _number(value: Element) -> float:
    """A scalar element as a number.

    Parameters
    ----------
    value : Element
        The element.

    Returns
    -------
    float
        The number.
    """
    assert isinstance(value, float | int)
    return float(value)


def _log_total(values: Sequence[Element]) -> float:
    """The log of a sum of exponentials, minus infinity for no terms.

    Parameters
    ----------
    values : Sequence of Element
        Log weights.

    Returns
    -------
    float
        ``log Σ exp(v)``.
    """
    finite = [_number(v) for v in values if _number(v) != -INFINITY]
    if not finite:
        return -INFINITY
    peak = max(finite)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in finite))


def _noisy_or(values: Sequence[Element]) -> float:
    """The noisy-OR with every term at most `CLAMP`.

    Parameters
    ----------
    values : Sequence of Element
        Numbers in ``[0, 1]``.

    Returns
    -------
    float
        ``1 - ∏ (1 - min(v, CLAMP))``.
    """
    return 1.0 - math.prod(1.0 - min(_number(v), CLAMP) for v in values)


def _scalar(
    plus: Callable[[list[float]], float],
    times: Callable[[float, float], float] | None,
    zero: float | None,
    unit: float | None,
) -> Reference:
    """A reference of a scalar carrier.

    Parameters
    ----------
    plus : Callable
        The sum of a list of numbers.
    times : Callable or None
        The product.
    zero : float or None
        The zero.
    unit : float or None
        The unit.

    Returns
    -------
    Reference
        The reference.
    """

    def total(values: Sequence[Element]) -> Element:
        """The sum.

        Parameters
        ----------
        values : Sequence of Element
            Numbers.

        Returns
        -------
        Element
            Their sum.
        """
        return plus([_number(v) for v in values])

    def product(a: Element, b: Element) -> Element:
        """The product.

        Parameters
        ----------
        a : Element
            A number.
        b : Element
            A number.

        Returns
        -------
        Element
            Their product.
        """
        assert times is not None
        return times(_number(a), _number(b))

    return Reference(
        plus=total, times=None if times is None else product, zero=zero, unit=unit
    )


def _fold(
    times: Callable[[float, float], float], empty: float
) -> Callable[[list[float]], float]:
    """The left fold of a product, for the sum of a dual.

    Parameters
    ----------
    times : Callable
        The product.
    empty : float
        The value over no terms.

    Returns
    -------
    Callable
        The fold.
    """

    def fold(values: list[float]) -> float:
        """Fold the values.

        Parameters
        ----------
        values : list of float
            Numbers.

        Returns
        -------
        float
            The fold.
        """
        result = empty
        for position, value in enumerate(values):
            result = value if position == 0 else times(result, value)
        return result

    return fold


def _bounded(values: list[float]) -> float:
    """The bounded sum ``min(1, Σ v)``.

    Parameters
    ----------
    values : list of float
        Numbers in ``[0, 1]``.

    Returns
    -------
    float
        The bounded sum.
    """
    return min(1.0, math.fsum(values))


def _lukasiewicz(a: float, b: float) -> float:
    """The Łukasiewicz t-norm.

    Parameters
    ----------
    a : float
        A number in ``[0, 1]``.
    b : float
        A number in ``[0, 1]``.

    Returns
    -------
    float
        ``max(a + b - 1, 0)``.
    """
    return max(a + b - 1.0, 0.0)


def _maximum(values: list[float]) -> float:
    """The maximum, minus infinity for no terms.

    Parameters
    ----------
    values : list of float
        Numbers.

    Returns
    -------
    float
        The maximum.
    """
    return max(values, default=-INFINITY)


def _minimum(values: list[float]) -> float:
    """The minimum, infinity for no terms.

    Parameters
    ----------
    values : list of float
        Numbers.

    Returns
    -------
    float
        The minimum.
    """
    return min(values, default=INFINITY)


def _add(a: float, b: float) -> float:
    """Addition.

    Parameters
    ----------
    a : float
        A number.
    b : float
        A number.

    Returns
    -------
    float
        ``a + b``.
    """
    return a + b


def _multiply(a: float, b: float) -> float:
    """Multiplication.

    Parameters
    ----------
    a : float
        A number.
    b : float
        A number.

    Returns
    -------
    float
        ``a b``.
    """
    return a * b


def _log_sum_second(a: float, b: float) -> list[list[float]]:
    """The second derivatives of ``log(e^a + e^b)``.

    With ``p = e^a / (e^a + e^b)`` the matrix is ``p (1 - p)`` times
    ``[[1, -1], [-1, 1]]``; an absent operand makes ``p`` zero or one,
    and two absent operands leave the sum ``-∞`` near them, so the
    matrix is zero.

    Parameters
    ----------
    a : float
        A log weight.
    b : float
        A log weight.

    Returns
    -------
    list of list of float
        The matrix.
    """
    if a == -INFINITY or b == -INFINITY:
        return [[0.0, 0.0], [0.0, 0.0]]
    peak = max(a, b)
    share = math.exp(a - peak) / (math.exp(a - peak) + math.exp(b - peak))
    curve = share * (1.0 - share)
    return [[curve, -curve], [-curve, curve]]


def _flat_second(a: float, b: float) -> list[list[float]]:
    """The second derivatives of a sum that is linear or piecewise linear.

    Parameters
    ----------
    a : float
        An operand.
    b : float
        An operand.

    Returns
    -------
    list of list of float
        The zero matrix.
    """
    del a, b
    return [[0.0, 0.0], [0.0, 0.0]]


PRODUCT_FUZZY_REFERENCE = _scalar(_noisy_or, _multiply, 0.0, 1.0)
LUKASIEWICZ_REFERENCE = _scalar(_bounded, _lukasiewicz, 0.0, 1.0)
REFERENCES: dict[str, Reference] = {
    "ProductFuzzy": PRODUCT_FUZZY_REFERENCE,
    "Boolean": _scalar(lambda v: max(v, default=0.0), _multiply, 0.0, 1.0),
    "Lukasiewicz": LUKASIEWICZ_REFERENCE,
    "Godel": _scalar(lambda v: max(v, default=0.0), min, 0.0, 1.0),
    "Tropical": _scalar(_minimum, _add, INFINITY, 0.0),
    "MaxPlus": _scalar(_maximum, _add, -INFINITY, 0.0).with_(second=_flat_second),
    "LogProb": _scalar(_log_total, _add, -INFINITY, 0.0).with_(second=_log_sum_second),
    "Real": _scalar(math.fsum, _multiply, 0.0, 1.0).with_(second=_flat_second),
    "Probability": _scalar(
        lambda v: min(1.0, max(0.0, math.fsum(v))),
        lambda a, b: min(1.0, max(0.0, a * b)),
        0.0,
        1.0,
    ),
    "Counting": _scalar(math.fsum, _multiply, 0.0, 1.0).with_(second=_flat_second),
    "Markov": _scalar(math.fsum, _multiply, 0.0, 1.0).with_(second=_flat_second),
    "Dual(ProductFuzzy)": _scalar(
        _fold(_multiply, 1.0), lambda a, b: _noisy_or([a, b]), 1.0, 0.0
    ),
    "Dual(Boolean)": _scalar(_fold(_multiply, 1.0), lambda a, b: max(a, b), 1.0, 0.0),
    "Dual(Lukasiewicz)": _scalar(
        _fold(_lukasiewicz, 1.0), lambda a, b: _bounded([a, b]), 1.0, 0.0
    ),
    "Dual(Godel)": _scalar(_fold(min, 1.0), lambda a, b: max(a, b), 1.0, 0.0),
    "Bounded": _scalar(lambda v: max(v, default=0.0), min, 0.0, 1.0),
    "BestDistance": _scalar(_maximum, _add, None, None),
}
"""The reference of every scalar member, by the member's name."""


def matrix_reference(base: Reference, size: int) -> Reference:
    """The reference of square matrices over a scalar reference.

    Parameters
    ----------
    base : Reference
        The entries' operations.
    size : int
        The order.

    Returns
    -------
    Reference
        Entrywise sum, matrix product, zero and identity matrices.
    """
    assert base.times is not None
    entry_times = base.times
    indices = range(size)

    def entry(m: Element, i: int, j: int) -> Element:
        """One entry.

        Parameters
        ----------
        m : Element
            A matrix.
        i : int
            The row.
        j : int
            The column.

        Returns
        -------
        Element
            The entry.
        """
        assert isinstance(m, list)
        row = m[i]
        assert isinstance(row, list)
        return row[j]

    def plus(values: Sequence[Element]) -> Element:
        """The entrywise sum.

        Parameters
        ----------
        values : Sequence of Element
            Matrices.

        Returns
        -------
        Element
            Their sum.
        """
        return [
            [base.plus([entry(m, i, j) for m in values]) for j in indices]
            for i in indices
        ]

    def times(a: Element, b: Element) -> Element:
        """The matrix product, ``a`` on the left.

        Parameters
        ----------
        a : Element
            A matrix.
        b : Element
            A matrix.

        Returns
        -------
        Element
            ``a b``.
        """
        return [
            [
                base.plus(
                    [entry_times(entry(a, i, k), entry(b, k, j)) for k in indices]
                )
                for j in indices
            ]
            for i in indices
        ]

    assert base.zero is not None and base.unit is not None
    zero: Element = [[base.zero for _ in indices] for _ in indices]
    unit: Element = [
        [base.unit if i == j else base.zero for j in indices] for i in indices
    ]
    return Reference(plus=plus, times=times, zero=zero, unit=unit)


def vector_reference(scalars: Reference, size: int) -> Reference:
    """The reference of vectors over a reference, with the left action.

    Parameters
    ----------
    scalars : Reference
        The operations of the acting semiring.
    size : int
        The length.

    Returns
    -------
    Reference
        Entrywise sum, the action on every entry, and the zero vector.
    """
    assert scalars.times is not None and scalars.zero is not None
    scalar_times = scalars.times
    indices = range(size)

    def component(m: Element, i: int) -> Element:
        """One entry.

        Parameters
        ----------
        m : Element
            A vector.
        i : int
            The position.

        Returns
        -------
        Element
            The entry.
        """
        assert isinstance(m, list)
        return m[i]

    def plus(values: Sequence[Element]) -> Element:
        """The entrywise sum.

        Parameters
        ----------
        values : Sequence of Element
            Vectors.

        Returns
        -------
        Element
            Their sum.
        """
        return [scalars.plus([component(m, i) for m in values]) for i in indices]

    def act(k: Element, m: Element) -> Element:
        """The scalar on the left of every entry.

        Parameters
        ----------
        k : Element
            A scalar.
        m : Element
            A vector.

        Returns
        -------
        Element
            ``k · m``.
        """
        return [scalar_times(k, component(m, i)) for i in indices]

    return Reference(plus=plus, times=act, zero=[scalars.zero for _ in indices])


# -- members ---------------------------------------------------------------

UNIT_INTERVAL = (0.0, 0.25, 0.5, 1.0)
LOG_WEIGHTS = (-INFINITY, -1.0, 0.0, 1.5)
SCORES = (-INFINITY, -1.0, 0.0, 2.0)
COUNTS = (0.0, 1.0, 2.0, 3.0)

#: Two by two matrices over the naturals that do not commute.
COUNT_MATRICES: tuple[Element, ...] = (
    [[0.0, 0.0], [0.0, 0.0]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 1.0], [0.0, 1.0]],
    [[1.0, 0.0], [1.0, 1.0]],
    [[0.0, 2.0], [1.0, 0.0]],
)
SCORE_MATRICES: tuple[Element, ...] = (
    [[-INFINITY, -INFINITY], [-INFINITY, -INFINITY]],
    [[0.0, -INFINITY], [-INFINITY, 0.0]],
    [[0.0, -1.0], [-INFINITY, 2.0]],
    [[1.0, -INFINITY], [0.0, 0.0]],
)


def _row(
    name: str,
    structure: Lawful,
    samples: Sequence[Element],
    reference: Reference,
    **extra: str | int | float | tuple[Element, ...] | Reference,
) -> Member:
    """Build a member.

    Parameters
    ----------
    name : str
        The row.
    structure : Lawful
        The member.
    samples : Sequence of Element
        Samples of its carrier.
    reference : Reference
        Its operations.
    **extra : str or int or float or tuple or Reference
        Further fields of `Member`.

    Returns
    -------
    Member
        The row.
    """
    return Member(
        name=name,
        structure=structure,
        samples=tuple(samples),
        reference=reference,
        **extra,  # type: ignore[arg-type]
    )


def _semigroupoid_join(t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """The maximum over axes, without an identity.

    Parameters
    ----------
    t : torch.Tensor
        Scores.
    dim : int or tuple of int
        Axes.

    Returns
    -------
    torch.Tensor
        The maximum.
    """
    return t.amax(dim=dim)


def _bounded_join(t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """The maximum over axes of a custom lattice.

    Parameters
    ----------
    t : torch.Tensor
        Values.
    dim : int or tuple of int
        Axes.

    Returns
    -------
    torch.Tensor
        The maximum.
    """
    return t.amax(dim=dim)


def _unit_interval(t: torch.Tensor) -> torch.Tensor:
    """Membership in ``[0, 1]``.

    Parameters
    ----------
    t : torch.Tensor
        Entries.

    Returns
    -------
    torch.Tensor
        Whether each lies in ``[0, 1]``.
    """
    return (t >= 0) & (t <= 1)


BOUNDED = CustomAlgebra(
    "Bounded",
    tensor_op=torch.minimum,
    join=_bounded_join,
    unit=1.0,
    zero=0.0,
    contains=_unit_interval,
)
"""A user-defined lattice on ``[0, 1]``, whose laws are only assumed."""

BEST_DISTANCE = CustomSemigroupoid("BestDistance", torch.add, _semigroupoid_join)
"""A user-defined rule without an identity, whose associativity is assumed."""

LOG_REFERENCE = REFERENCES["LogProb"]
COUNT_REFERENCE = REFERENCES["Counting"]

MEMBERS: dict[str, Member] = {
    member.name: member
    for member in (
        _row(
            "core/ProductFuzzy",
            PRODUCT_FUZZY,
            UNIT_INTERVAL,
            REFERENCES["ProductFuzzy"],
            domain="linear",
            operations=48,
        ),
        _row(
            "core/Boolean", BOOLEAN, (0.0, 1.0), REFERENCES["Boolean"], best="largest"
        ),
        _row("core/Lukasiewicz", LUKASIEWICZ, UNIT_INTERVAL, REFERENCES["Lukasiewicz"]),
        _row("core/Godel", GODEL, UNIT_INTERVAL, REFERENCES["Godel"], best="largest"),
        _row(
            "core/Tropical",
            TROPICAL,
            (0.0, 1.0, 2.5, INFINITY),
            REFERENCES["Tropical"],
            best="smallest",
        ),
        _row("core/MaxPlus", MAX_PLUS, SCORES, REFERENCES["MaxPlus"], best="largest"),
        _row(
            "core/LogProb",
            LOG_PROB,
            LOG_WEIGHTS,
            LOG_REFERENCE,
            domain="log",
            operations=48,
            magnitude=6.0,
        ),
        _row("core/Real", REAL, (-2.0, 0.0, 0.5, 1.0, 3.0), REFERENCES["Real"]),
        _row("core/Probability", PROBABILITY, UNIT_INTERVAL, REFERENCES["Probability"]),
        _row("core/Counting", COUNTING, COUNTS, COUNT_REFERENCE),
        _row("core/Markov", MARKOV, (0.0, 0.5, 1.0, 2.0), REFERENCES["Markov"]),
        _row(
            "dual/ProductFuzzy",
            REICHENBACH,
            UNIT_INTERVAL,
            REFERENCES["Dual(ProductFuzzy)"],
            domain="linear",
            operations=48,
        ),
        _row("dual/Boolean", BOOLEAN_DUAL, (0.0, 1.0), REFERENCES["Dual(Boolean)"]),
        _row(
            "dual/Lukasiewicz",
            DUAL_LUKASIEWICZ,
            UNIT_INTERVAL,
            REFERENCES["Dual(Lukasiewicz)"],
        ),
        _row("dual/Godel", DUAL_GODEL, UNIT_INTERVAL, REFERENCES["Dual(Godel)"]),
        _row(
            "chart/LogProb",
            chart.LOG_PROB,
            LOG_WEIGHTS,
            LOG_REFERENCE,
            domain="log",
            operations=48,
            magnitude=6.0,
        ),
        _row(
            "chart/Viterbi",
            chart.VITERBI,
            SCORES,
            REFERENCES["MaxPlus"],
            best="largest",
        ),
        _row(
            "chart/Boolean",
            chart.BOOLEAN,
            (0.0, 1.0),
            REFERENCES["Boolean"],
            best="largest",
        ),
        _row("chart/Counting", chart.COUNTING, COUNTS, COUNT_REFERENCE),
        _row("custom/Bounded", BOUNDED, UNIT_INTERVAL, REFERENCES["Bounded"]),
        _row(
            "custom/BestDistance",
            BEST_DISTANCE,
            (-1.0, 0.0, 2.0),
            REFERENCES["BestDistance"],
        ),
        _row(
            "matrix/Counting",
            MatrixSemiring(COUNTING, 2),
            COUNT_MATRICES,
            matrix_reference(COUNT_REFERENCE, 2),
        ),
        _row(
            "matrix/MaxPlus",
            MatrixSemiring(MAX_PLUS, 2),
            SCORE_MATRICES,
            matrix_reference(REFERENCES["MaxPlus"], 2),
        ),
        _row(
            "vector/Counting",
            FreeSemimodule(COUNTING, 3),
            ([0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 0.0, 1.0]),
            vector_reference(COUNT_REFERENCE, 3),
            scalars=COUNTS,
            scalar_reference=COUNT_REFERENCE,
        ),
        _row(
            "vector/LogProb",
            FreeSemimodule(LOG_PROB, 2),
            ([-INFINITY, -INFINITY], [0.0, -1.0], [1.5, -INFINITY]),
            vector_reference(LOG_REFERENCE, 2),
            scalars=LOG_WEIGHTS,
            scalar_reference=LOG_REFERENCE,
            domain="log",
            operations=48,
            magnitude=6.0,
        ),
        _row(
            "vector/Matrix",
            FreeSemimodule(MatrixSemiring(COUNTING, 2), 2),
            (
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 1.0]], [[0.0, 2.0], [1.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            ),
            vector_reference(matrix_reference(COUNT_REFERENCE, 2), 2),
            scalars=COUNT_MATRICES,
            scalar_reference=matrix_reference(COUNT_REFERENCE, 2),
        ),
    )
}
"""The rows of the law grid."""

ASSUMED: frozenset[str] = frozenset({"custom/Bounded", "custom/BestDistance"})
"""Rows whose laws about the carrier are only assumed, by design."""


# -- tensors of samples ------------------------------------------------------


def element_shape(member: Member) -> tuple[int, ...]:
    """The shape of one element of a member.

    Parameters
    ----------
    member : Member
        The row.

    Returns
    -------
    tuple of int
        The element shape; ``()`` for a rule that is not a carrier.
    """
    structure = member.structure
    return structure.element_shape if isinstance(structure, Carrier) else ()


def samples(member: Member, dtype: torch.dtype) -> torch.Tensor:
    """The samples of a member as one tensor.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    torch.Tensor
        Shape ``(N, *element_shape)``.
    """
    return torch.tensor(list(member.samples), dtype=dtype)


def scalar_samples(member: Member, dtype: torch.dtype) -> torch.Tensor:
    """The samples of the acting semiring of a module.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    torch.Tensor
        Shape ``(N, *scalars.element_shape)``.
    """
    return torch.tensor(list(member.scalars), dtype=dtype)


def tuples(values: torch.Tensor, arity: int) -> list[torch.Tensor]:
    """Every tuple of samples, one tensor per position.

    Parameters
    ----------
    values : torch.Tensor
        Samples, shape ``(N, *E)``.
    arity : int
        The length of the tuples.

    Returns
    -------
    list of torch.Tensor
        ``arity`` tensors of shape ``(N**arity, *E)``.
    """
    grids = torch.meshgrid(*(torch.arange(values.shape[0]),) * arity, indexing="ij")
    return [values[grid.reshape(-1)] for grid in grids]


def empty_batches(values: torch.Tensor) -> list[torch.Tensor]:
    """Batches of no elements, with one and with two index axes.

    Parameters
    ----------
    values : torch.Tensor
        Samples, shape ``(N, *E)``.

    Returns
    -------
    list of torch.Tensor
        Shapes ``(0, *E)`` and ``(2, 0, *E)``.
    """
    rest = tuple(values.shape[1:])
    return [values.new_empty((0, *rest)), values.new_empty((2, 0, *rest))]


def accuracy(member: Member, dtype: torch.dtype) -> Accuracy:
    """The accuracy of a cell.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    Accuracy
        Exact (no rounding allowed) for an exact row.
    """
    name = "float64" if dtype == torch.float64 else "float32"
    if member.domain == "exact":
        return Accuracy(domain="linear", operations=0, dtype=name)
    return Accuracy(
        domain=member.domain,
        operations=member.operations,
        magnitude=member.magnitude,
        dtype=name,
    )


def agree(
    found: torch.Tensor,
    wanted: torch.Tensor,
    member: Member,
    what: str,
) -> None:
    """Require two tensors to agree within the cell's accuracy.

    Parameters
    ----------
    found : torch.Tensor
        One side.
    wanted : torch.Tensor
        The other, whose shape, dtype and device are required.
    member : Member
        The row.
    what : str
        A description.

    Raises
    ------
    Mismatch
        If they differ.
    """
    if wanted.dtype in DTYPES:
        compare_tensor(
            found, wanted, accuracy(member, wanted.dtype), f"{member.name}: {what}"
        )
    else:
        compare_tensor(
            found,
            wanted,
            Accuracy(domain="linear", operations=0),
            f"{member.name}: {what}",
        )


def as_tensor(elements: Sequence[Element], like: torch.Tensor) -> torch.Tensor:
    """Plain elements as a tensor in the dtype and device of ``like``.

    Parameters
    ----------
    elements : Sequence of Element
        The elements.
    like : torch.Tensor
        The tensor whose dtype and device are taken.

    Returns
    -------
    torch.Tensor
        Shape ``(len(elements), *E)``.
    """
    return torch.tensor(list(elements), dtype=like.dtype, device=like.device)


def equation(
    member: Member,
    dtype: torch.dtype,
    arity: int,
    left: Callable[..., torch.Tensor],
    right: Callable[..., torch.Tensor],
    what: str,
    values: torch.Tensor | None = None,
) -> None:
    """Require an equation on every tuple of samples and on empty batches.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    arity : int
        The number of variables.
    left : Callable
        One side, of ``arity`` tensors.
    right : Callable
        The other side.
    what : str
        The law.
    values : torch.Tensor or None
        The samples, when not the member's own.

    Raises
    ------
    Mismatch
        If the sides differ.
    """
    pool = samples(member, dtype) if values is None else values
    agree(left(*tuples(pool, arity)), right(*tuples(pool, arity)), member, what)
    for empty in empty_batches(pool):
        found = left(*([empty] * arity))
        wanted = right(*([empty] * arity))
        agree(found, wanted, member, f"{what} on a batch of shape {tuple(empty.shape)}")
        if tuple(found.shape) != tuple(empty.shape):
            raise Mismatch(
                "shape",
                math.inf,
                f"{member.name}: {what} on {tuple(empty.shape)} gave {tuple(found.shape)}",
            )


def is_element(t: torch.Tensor, element: Element, rank: int) -> torch.Tensor:
    """Whether each element of ``t`` equals a plain element exactly.

    Parameters
    ----------
    t : torch.Tensor
        Elements, shape ``(*index, *E)``.
    element : Element
        The element.
    rank : int
        The length of ``E``.

    Returns
    -------
    torch.Tensor
        A Boolean tensor of the index shape.
    """
    equal = t == torch.tensor(element, dtype=t.dtype, device=t.device)
    for _ in range(rank):
        equal = equal.all(dim=-1)
    return equal


def unit_tensor(structure: UnitalCarrier, like: torch.Tensor) -> torch.Tensor:
    """The unit at every index of a batch.

    Parameters
    ----------
    structure : UnitalCarrier
        The member.
    like : torch.Tensor
        A batch of elements.

    Returns
    -------
    torch.Tensor
        The unit, of the shape of ``like``.
    """
    rank = len(structure.element_shape)
    return structure.unit_like(tuple(like.shape[: like.ndim - rank]), like)


def zero_tensor(structure: Carrier, like: torch.Tensor) -> torch.Tensor:
    """The zero at every index of a batch.

    Parameters
    ----------
    structure : Carrier
        The member.
    like : torch.Tensor
        A batch of elements.

    Returns
    -------
    torch.Tensor
        The zero, of the shape of ``like``.
    """
    rank = len(structure.element_shape)
    return structure.zero_like(tuple(like.shape[: like.ndim - rank]), like)


def _carrier(member: Member) -> Carrier:
    """The member as a carrier.

    Parameters
    ----------
    member : Member
        The row.

    Returns
    -------
    Carrier
        The member.
    """
    structure = member.structure
    assert isinstance(structure, Carrier)
    return structure


def _unital(member: Member) -> UnitalCarrier:
    """The member as a carrier with a product.

    Parameters
    ----------
    member : Member
        The row.

    Returns
    -------
    UnitalCarrier
        The member.
    """
    structure = member.structure
    assert isinstance(structure, UnitalCarrier)
    return structure


def _module(member: Member) -> Semimodule:
    """The member as a module.

    Parameters
    ----------
    member : Member
        The row.

    Returns
    -------
    Semimodule
        The member.
    """
    structure = member.structure
    assert isinstance(structure, Semimodule)
    return structure


# -- the checks of the laws -------------------------------------------------


def check_closure(member: Member, dtype: torch.dtype) -> None:
    """Sums, products and actions of samples lie in the carrier.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a result lies outside the carrier.
    """
    structure = _carrier(member)
    a, b = tuples(samples(member, dtype), 2)
    results = {"sum": structure.join_pair(a, b)}
    if isinstance(structure, UnitalCarrier):
        results["product"] = structure.tensor_op(a, b)
    if isinstance(structure, Semimodule):
        k, m = _scalar_module_pairs(member, dtype)
        results["action"] = structure.act(k, m)
    for kind, result in results.items():
        if not bool(structure.contains(result).all()):
            raise Mismatch(
                "support", math.inf, f"{member.name}: a {kind} leaves the carrier"
            )


def _scalar_module_pairs(
    member: Member, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every pair of a scalar sample and a module sample.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    tuple of torch.Tensor
        Scalars and module elements, aligned.
    """
    k = scalar_samples(member, dtype)
    m = samples(member, dtype)
    left = torch.arange(k.shape[0]).repeat_interleave(m.shape[0])
    right = torch.arange(m.shape[0]).repeat(k.shape[0])
    return k[left], m[right]


def check_iterated_join(member: Member, dtype: torch.dtype) -> None:
    """A join over two axes is the join over one axis after the other.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If the reductions differ.
    """
    structure = _carrier(member)
    pool = samples(member, dtype)
    n = pool.shape[0]
    grid = torch.stack(torch.meshgrid(*(torch.arange(n),) * 3, indexing="ij"), -1)
    index = (grid * torch.tensor([n * n, n, 1])).sum(-1) % n
    cube = pool[index]
    for t in (cube, cube[:, :0], cube[:0]):
        both = structure.join(t, (0, 1))
        agree(both, structure.join(structure.join(t, 1), 0), member, "join over (0, 1)")
        agree(both, structure.join(structure.join(t, 0), 0), member, "join over (1, 0)")
        agree(
            structure.join(t, (0, 1, 2)),
            structure.join(structure.join(structure.join(t, 2), 1), 0),
            member,
            "join over three axes",
        )


def check_additive_associativity(member: Member, dtype: torch.dtype) -> None:
    """``(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    plus = _carrier(member).join_pair
    equation(
        member,
        dtype,
        3,
        lambda a, b, c: plus(plus(a, b), c),
        lambda a, b, c: plus(a, plus(b, c)),
        "additive associativity",
    )


def check_additive_commutativity(member: Member, dtype: torch.dtype) -> None:
    """``a ⊕ b = b ⊕ a``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    plus = _carrier(member).join_pair
    equation(
        member,
        dtype,
        2,
        lambda a, b: plus(a, b),
        lambda a, b: plus(b, a),
        "additive commutativity",
    )


def check_additive_identity(member: Member, dtype: torch.dtype) -> None:
    """``a ⊕ 0 = a = 0 ⊕ a``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    structure = _carrier(member)
    plus = structure.join_pair
    equation(
        member,
        dtype,
        1,
        lambda a: plus(a, zero_tensor(structure, a)),
        lambda a: a,
        "right additive identity",
    )
    equation(
        member,
        dtype,
        1,
        lambda a: plus(zero_tensor(structure, a), a),
        lambda a: a,
        "left additive identity",
    )


def check_multiplicative_associativity(member: Member, dtype: torch.dtype) -> None:
    """``(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    times = _unital(member).tensor_op
    equation(
        member,
        dtype,
        3,
        lambda a, b, c: times(times(a, b), c),
        lambda a, b, c: times(a, times(b, c)),
        "multiplicative associativity",
    )


def check_multiplicative_identity(member: Member, dtype: torch.dtype) -> None:
    """``1 ⊗ a = a = a ⊗ 1``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    structure = _unital(member)
    times = structure.tensor_op
    equation(
        member,
        dtype,
        1,
        lambda a: times(unit_tensor(structure, a), a),
        lambda a: a,
        "left multiplicative identity",
    )
    equation(
        member,
        dtype,
        1,
        lambda a: times(a, unit_tensor(structure, a)),
        lambda a: a,
        "right multiplicative identity",
    )


def check_distributivity(member: Member, dtype: torch.dtype) -> None:
    """``a ⊗ (b ⊕ c) = a⊗b ⊕ a⊗c`` and ``(a ⊕ b) ⊗ c = a⊗c ⊕ b⊗c``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    structure = _unital(member)
    times, plus = structure.tensor_op, structure.join_pair
    equation(
        member,
        dtype,
        3,
        lambda a, b, c: times(a, plus(b, c)),
        lambda a, b, c: plus(times(a, b), times(a, c)),
        "left distributivity",
    )
    equation(
        member,
        dtype,
        3,
        lambda a, b, c: times(plus(a, b), c),
        lambda a, b, c: plus(times(a, c), times(b, c)),
        "right distributivity",
    )


def check_annihilation(member: Member, dtype: torch.dtype) -> None:
    """``0 ⊗ a = 0 = a ⊗ 0``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    structure = _unital(member)
    times = structure.tensor_op
    equation(
        member,
        dtype,
        1,
        lambda a: times(zero_tensor(structure, a), a),
        lambda a: zero_tensor(structure, a),
        "left annihilation",
    )
    equation(
        member,
        dtype,
        1,
        lambda a: times(a, zero_tensor(structure, a)),
        lambda a: zero_tensor(structure, a),
        "right annihilation",
    )


def check_multiplicative_commutativity(member: Member, dtype: torch.dtype) -> None:
    """``a ⊗ b = b ⊗ a``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    times = _unital(member).tensor_op
    equation(
        member,
        dtype,
        2,
        lambda a, b: times(a, b),
        lambda a, b: times(b, a),
        "multiplicative commutativity",
    )


def _matrices(pool: torch.Tensor, rows: int, columns: int, offset: int) -> torch.Tensor:
    """A matrix of samples, taken cyclically.

    Parameters
    ----------
    pool : torch.Tensor
        Samples, shape ``(N, *E)``.
    rows : int
        Rows.
    columns : int
        Columns.
    offset : int
        Where to start in the samples.

    Returns
    -------
    torch.Tensor
        Shape ``(rows, columns, *E)``.
    """
    index = (torch.arange(rows * columns) * 3 + offset) % pool.shape[0]
    return pool[index].reshape(rows, columns, *pool.shape[1:])


def check_composition_associativity(member: Member, dtype: torch.dtype) -> None:
    """``(f >> g) >> h = f >> (g >> h)``, also through an empty axis.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    rule = member.structure
    assert isinstance(rule, CompositionRule)
    pool = samples(member, dtype)
    # A composite through no states is a sum of no terms, which only a
    # rule with a zero has.
    for inner in (3, 0) if isinstance(rule, Carrier) else (3,):
        f = _matrices(pool, 2, inner, 0)
        g = _matrices(pool, inner, 2, 1)
        h = _matrices(pool, 2, 2, 2)
        agree(
            rule.compose(rule.compose(f, g, 1), h, 1),
            rule.compose(f, rule.compose(g, h, 1), 1),
            member,
            f"composition associativity through {inner} states",
        )


def check_additive_idempotence(member: Member, dtype: torch.dtype) -> None:
    """``a ⊕ a = a``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    plus = _carrier(member).join_pair
    equation(member, dtype, 1, lambda a: plus(a, a), lambda a: a, "idempotence")


def check_selectivity(member: Member, dtype: torch.dtype) -> None:
    """``a ⊕ b`` is ``a`` or ``b`` exactly.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a sum is neither operand.
    """
    a, b = tuples(samples(member, dtype), 2)
    result = _carrier(member).join_pair(a, b)
    if not bool(((result == a) | (result == b)).all()):
        raise Mismatch("value", math.inf, f"{member.name}: a sum is neither operand")


def _first_best(values: Sequence[float], best: str) -> int:
    """The position of the first best number.

    Parameters
    ----------
    values : Sequence of float
        Numbers.
    best : str
        ``"largest"`` or ``"smallest"``.

    Returns
    -------
    int
        The position.
    """
    target = max(values) if best == "largest" else min(values)
    return list(values).index(target)


def check_tie_rule(member: Member, dtype: torch.dtype) -> int:
    """The first best operand is the witness and receives the derivative.

    Checked on every pair (ties included) through the binary sum, on every
    triple through a reduction over one axis, and on every square of
    samples through a reduction over two axes, whose witness counts in
    row-major order.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    int
        The number of sums and reductions checked.

    Raises
    ------
    Mismatch
        If a derivative or a witness differs from the first best operand.
    """
    structure = member.structure
    assert isinstance(structure, Selective)
    if structure.tie_rule != FIRST_MAXIMIZER:
        raise Mismatch("value", math.inf, f"{member.name}: another tie rule")
    pool = samples(member, dtype)
    a, b = (part.clone().requires_grad_() for part in tuples(pool, 2))
    grads = torch.autograd.grad(structure.join_pair(a, b).sum(), (a, b))
    cases = a.shape[0]
    for row, (x, y) in enumerate(zip(a.tolist(), b.tolist(), strict=True)):
        first = _first_best((x, y), member.best)
        wanted = (float(first == 0), float(first == 1))
        found = (float(grads[0][row]), float(grads[1][row]))
        if found != wanted:
            raise Mismatch(
                "value",
                max(abs(f - w) for f, w in zip(found, wanted, strict=True)),
                f"{member.name}: derivative {found} of the sum of {x} and {y}, "
                f"declared {wanted}",
            )
    for shape, axes in (((3,), (1,)), ((2, 2), (1, 2))):
        size = math.prod(shape)
        terms = torch.stack(tuples(pool, size), dim=1).reshape(-1, *shape)
        leaf = terms.clone().requires_grad_()
        value = structure.join(leaf, axes)
        (found,) = torch.autograd.grad(value.sum(), (leaf,))
        witness = structure.select(terms, axes).witness
        cases += terms.shape[0]
        for row, flat in enumerate(terms.reshape(-1, size).tolist()):
            first = _first_best(flat, member.best)
            derivative = found.reshape(-1, size)[row].tolist()
            wanted = [float(position == first) for position in range(size)]
            if derivative != wanted or int(witness[row]) != first:
                raise Mismatch(
                    "value",
                    max(abs(f - w) for f, w in zip(derivative, wanted, strict=True)),
                    f"{member.name}: reduction of {flat} gave derivative "
                    f"{derivative} and witness {int(witness[row])}, declared the "
                    f"operand at {first}",
                )
    return cases


def check_zero_sum_free(member: Member, dtype: torch.dtype) -> None:
    """A sum is the zero exactly when both terms are.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a sum of terms, not both zero, is the zero.
    """
    structure = _carrier(member)
    zero = member.reference.zero
    assert zero is not None
    rank = len(structure.element_shape)
    a, b = tuples(samples(member, dtype), 2)
    found = is_element(structure.join_pair(a, b), zero, rank)
    wanted = is_element(a, zero, rank) & is_element(b, zero, rank)
    if not torch.equal(found, wanted):
        raise Mismatch("value", math.inf, f"{member.name}: a sum cancels to the zero")


def check_no_zero_divisors(member: Member, dtype: torch.dtype) -> None:
    """A product is the zero exactly when a factor is.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a product of factors other than the zero is the zero.
    """
    structure = _unital(member)
    zero = member.reference.zero
    assert zero is not None
    rank = len(structure.element_shape)
    a, b = tuples(samples(member, dtype), 2)
    found = is_element(structure.tensor_op(a, b), zero, rank)
    wanted = is_element(a, zero, rank) | is_element(b, zero, rank)
    if not torch.equal(found, wanted):
        raise Mismatch("value", math.inf, f"{member.name}: a product has zero divisors")


def check_division(member: Member, dtype: torch.dtype) -> None:
    """``(a ⊗ b) ⊘ b = a`` for every ``b`` other than zero; ``a ⊘ 0`` refuses.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a quotient differs, or a division by the zero is not refused.
    """
    structure = member.structure
    assert isinstance(structure, Semifield)
    zero = member.reference.zero
    assert zero is not None
    pool = samples(member, dtype)
    nonzero = pool[~is_element(pool, zero, 0)]
    equation(
        member,
        dtype,
        2,
        lambda x, y: structure.divide(structure.tensor_op(x, y), y),
        lambda x, y: x,
        "division inverts the product",
        values=nonzero,
    )
    try:
        structure.divide(pool, zero_tensor(structure, pool))
    except ZeroDivisionError:
        return
    raise Mismatch("refusal", math.inf, f"{member.name}: division by the zero")


def check_additive_inverse(member: Member, dtype: torch.dtype) -> None:
    """``a ⊕ (0 ⊖ a) = 0`` and ``(a ⊖ b) ⊕ b = a``.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    structure = member.structure
    assert isinstance(structure, Ring)
    plus = structure.join_pair
    equation(
        member,
        dtype,
        1,
        lambda a: plus(a, structure.subtract(zero_tensor(structure, a), a)),
        lambda a: zero_tensor(structure, a),
        "additive inverse",
    )
    equation(
        member,
        dtype,
        2,
        lambda a, b: plus(structure.subtract(a, b), b),
        lambda a, b: a,
        "subtraction",
    )


def _module_tuples(
    member: Member, dtype: torch.dtype, scalars: int, elements: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Every combination of scalar and module samples.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    scalars : int
        Scalars per combination.
    elements : int
        Module elements per combination.

    Returns
    -------
    tuple of lists of torch.Tensor
        Aligned scalars and module elements.
    """
    k = scalar_samples(member, dtype)
    m = samples(member, dtype)
    sizes = (k.shape[0],) * scalars + (m.shape[0],) * elements
    grids = torch.meshgrid(*(torch.arange(n) for n in sizes), indexing="ij")
    flat = [grid.reshape(-1) for grid in grids]
    return (
        [k[index] for index in flat[:scalars]],
        [m[index] for index in flat[scalars:]],
    )


def _module_equation(
    member: Member,
    dtype: torch.dtype,
    counts: tuple[int, int],
    left: Callable[[list[torch.Tensor], list[torch.Tensor]], torch.Tensor],
    right: Callable[[list[torch.Tensor], list[torch.Tensor]], torch.Tensor],
    what: str,
) -> None:
    """Require an equation of an action on every combination of samples.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    counts : tuple of int
        Scalars and module elements per combination.
    left : Callable
        One side.
    right : Callable
        The other side.
    what : str
        The law.
    """
    k, m = _module_tuples(member, dtype, *counts)
    agree(left(k, m), right(k, m), member, what)
    k_empty = [x[:0] for x in k]
    m_empty = [x[:0] for x in m]
    agree(
        left(k_empty, m_empty),
        right(k_empty, m_empty),
        member,
        f"{what} on no elements",
    )


def check_action_associativity(member: Member, dtype: torch.dtype) -> None:
    """``(k ⊗ l) · m = k · (l · m)``.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    """
    module = _module(member)
    times = module.scalars.tensor_op
    _module_equation(
        member,
        dtype,
        (2, 1),
        lambda k, m: module.act(times(k[0], k[1]), m[0]),
        lambda k, m: module.act(k[0], module.act(k[1], m[0])),
        "action associativity",
    )


def check_action_unit(member: Member, dtype: torch.dtype) -> None:
    """``1 · m = m``.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    """
    module = _module(member)
    _module_equation(
        member,
        dtype,
        (1, 1),
        lambda k, m: module.act(unit_tensor(module.scalars, k[0]), m[0]),
        lambda k, m: m[0],
        "action unit",
    )


def check_action_over_module_sum(member: Member, dtype: torch.dtype) -> None:
    """``k · (m ⊕ n) = k·m ⊕ k·n``.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    """
    module = _module(member)
    plus = module.join_pair
    _module_equation(
        member,
        dtype,
        (1, 2),
        lambda k, m: module.act(k[0], plus(m[0], m[1])),
        lambda k, m: plus(module.act(k[0], m[0]), module.act(k[0], m[1])),
        "action over the module sum",
    )


def check_action_over_scalar_sum(member: Member, dtype: torch.dtype) -> None:
    """``(k ⊕ l) · m = k·m ⊕ l·m``.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    """
    module = _module(member)
    plus = module.join_pair
    scalar_plus = module.scalars.join_pair
    _module_equation(
        member,
        dtype,
        (2, 1),
        lambda k, m: module.act(scalar_plus(k[0], k[1]), m[0]),
        lambda k, m: plus(module.act(k[0], m[0]), module.act(k[1], m[0])),
        "action over the scalar sum",
    )


def check_action_annihilation(member: Member, dtype: torch.dtype) -> None:
    """``0 · m = 0`` and ``k · 0 = 0``.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    """
    module = _module(member)
    _module_equation(
        member,
        dtype,
        (1, 1),
        lambda k, m: module.act(zero_tensor(module.scalars, k[0]), m[0]),
        lambda k, m: zero_tensor(module, m[0]),
        "zero scalar",
    )
    _module_equation(
        member,
        dtype,
        (1, 1),
        lambda k, m: module.act(k[0], zero_tensor(module, m[0])),
        lambda k, m: zero_tensor(module, m[0]),
        "zero element",
    )


def check_semimodule_action(member: Member, dtype: torch.dtype) -> None:
    """The action is associative, unital, additive in both sums, and annihilated.

    Parameters
    ----------
    member : Member
        A module's row.
    dtype : torch.dtype
        The precision.
    """
    check_action_associativity(member, dtype)
    check_action_unit(member, dtype)
    check_action_over_module_sum(member, dtype)
    check_action_over_scalar_sum(member, dtype)
    check_action_annihilation(member, dtype)


def _representable(element: Element, dtype: torch.dtype) -> bool:
    """Whether a dtype holds every number of a plain element exactly.

    Parameters
    ----------
    element : Element
        The element.
    dtype : torch.dtype
        The dtype.

    Returns
    -------
    bool
        True for floating dtypes, and for integer and Boolean ones holding
        each finite integer within range.
    """
    if isinstance(element, list):
        return all(_representable(part, dtype) for part in element)
    if dtype.is_floating_point:
        return True
    if not math.isfinite(element) or not float(element).is_integer():
        return False
    if dtype == torch.bool:
        return element in (0.0, 1.0)
    return True


def _check_form(
    member: Member,
    name: str,
    make: Callable[[tuple[int, ...], torch.Tensor], torch.Tensor],
    element: Element,
) -> int:
    """A constant has the requested shape, dtype, device and value.

    Parameters
    ----------
    member : Member
        The row.
    name : str
        ``"zero"`` or ``"unit"``.
    make : Callable
        ``zero_like`` or ``unit_like``.
    element : Element
        The reference value.

    Returns
    -------
    int
        The number of dtypes and shapes asked for.

    Raises
    ------
    Mismatch
        If the constant has another form or value, or a dtype that cannot
        hold it is not refused.
    """
    structure = _carrier(member)
    rank = len(structure.element_shape)
    cases = 0
    for dtype in FORM_DTYPES:
        like = torch.empty(0, dtype=dtype)
        if dtype.is_floating_point:
            like = like.requires_grad_()
        for shape in SHAPES:
            cases += 1
            if not _representable(element, dtype):
                try:
                    make(shape, like)
                except TypeError:
                    continue
                raise Mismatch(
                    "refusal",
                    math.inf,
                    f"{member.name}: {name} in {dtype} is not refused",
                )
            try:
                value = make(shape, like)
            except Exception as failure:
                raise Mismatch(
                    "refusal", math.inf, f"{member.name}: {name} raised {failure!r}"
                ) from failure
            wanted_shape = (*shape, *structure.element_shape)
            if tuple(value.shape) != wanted_shape:
                raise Mismatch(
                    "shape",
                    math.inf,
                    f"{member.name}: {name} of shape {tuple(value.shape)} for {wanted_shape}",
                )
            if value.dtype != dtype or value.device != like.device:
                raise Mismatch(
                    "dtype",
                    math.inf,
                    f"{member.name}: {name} in {value.dtype} for {dtype}",
                )
            if value.requires_grad:
                raise Mismatch(
                    "value", math.inf, f"{member.name}: {name} requires a gradient"
                )
            if not bool(is_element(value, element, rank).all()):
                raise Mismatch(
                    "value", math.inf, f"{member.name}: {name} has another value"
                )
            if not bool(structure.contains(value).all()):
                raise Mismatch(
                    "support", math.inf, f"{member.name}: {name} is not in the carrier"
                )
    return cases


def check_zero_form(member: Member, dtype: torch.dtype) -> int:
    """The zero in every requested shape, dtype and device.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        Unused: every dtype of `FORM_DTYPES` is asked for.

    Returns
    -------
    int
        The number of cases.
    """
    del dtype
    zero = member.reference.zero
    assert zero is not None
    return _check_form(member, "zero", _carrier(member).zero_like, zero)


def check_unit_form(member: Member, dtype: torch.dtype) -> int:
    """The unit in every requested shape, dtype and device.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        Unused: every dtype of `FORM_DTYPES` is asked for.

    Returns
    -------
    int
        The number of cases.
    """
    del dtype
    unit = member.reference.unit
    assert unit is not None
    return _check_form(member, "unit", _unital(member).unit_like, unit)


def check_empty_join(member: Member, dtype: torch.dtype) -> int:
    """A join over an axis of size zero is the zero with a defined derivative.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    int
        The number of reductions checked.

    Raises
    ------
    Mismatch
        If the result has another shape, dtype, value or derivative, or
        the join raises.
    """
    structure = _carrier(member)
    zero = member.reference.zero
    assert zero is not None
    rest = structure.element_shape
    cases: tuple[tuple[tuple[int, ...], int | tuple[int, ...]], ...] = (
        ((0,), 0),
        ((3, 0), 1),
        ((3, 0), -1),
        ((3, 0), (0, 1)),
        ((0, 3), 1),
        ((2, 0, 3), (1, 2)),
    )
    for index, dim in cases:
        leaf = torch.zeros((*index, *rest), dtype=dtype, requires_grad=True)
        try:
            value = structure.join(leaf, dim)
            axes = reduction_axes(leaf, dim, len(rest))
            remaining = tuple(n for axis, n in enumerate(index) if axis not in axes)
            wanted = torch.tensor(zero, dtype=dtype).expand((*remaining, *rest))
            agree(
                value, wanted.contiguous(), member, f"join of shape {index} over {dim}"
            )
            if not value.requires_grad:
                raise Mismatch(
                    "missing",
                    math.inf,
                    f"{member.name}: join of {index} over {dim} has no derivative",
                )
            (gradient,) = torch.autograd.grad(
                value, (leaf,), grad_outputs=torch.ones_like(value), allow_unused=True
            )
        except Mismatch:
            raise
        except Exception as failure:
            raise Mismatch(
                "refusal",
                math.inf,
                f"{member.name}: join of {index} over {dim} raised {failure!r}",
            ) from failure
        if gradient is None or bool(torch.isnan(gradient).any()):
            raise Mismatch(
                "not-a-number",
                math.inf,
                f"{member.name}: the derivative of a join over no terms is not a number",
            )
    return len(cases)


# -- closures ---------------------------------------------------------------


class ClosureCases(ValueModel):
    """The matrices whose closure a row is checked on.

    Parameters
    ----------
    kind : str
        ``"idempotent"`` when the least solution is the sum of the first
        ``n + 1`` powers, ``"linear"`` when it is ``(I - A)⁻¹``, ``"log"``
        when it is the logarithm of that of the exponentiated matrix.
    convergent : tuple of Element
        Square matrices inside the closure's domain, one of order zero.
    divergent : tuple of Element
        Square matrices outside it.
    exhaustive : bool
        Whether the samples are the whole carrier, so leastness is
        decided over every matrix of order two.
    """

    kind: Literal["idempotent", "linear", "log"]
    convergent: tuple[Element, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    divergent: tuple[Element, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    exhaustive: bool = False


_LOG = math.log
_NONE = -INFINITY
_BOOLEAN_CLOSURES = ClosureCases(
    kind="idempotent",
    convergent=(
        [],
        [[0.0, 1.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    ),
    exhaustive=True,
)
_SCORE_CLOSURES = ClosureCases(
    kind="idempotent",
    convergent=(
        [],
        [[-1.0, 0.0], [_NONE, _NONE]],
        [[0.0, -1.0], [-1.0, _NONE]],
        [[_NONE, 2.0, _NONE], [_NONE, _NONE, -1.0], [-1.5, _NONE, _NONE]],
    ),
    divergent=([[1.0]], [[_NONE, 2.0], [-1.0, _NONE]]),
)
_LOG_CLOSURES = ClosureCases(
    kind="log",
    convergent=(
        [],
        [[_LOG(0.5), _LOG(0.25)], [_NONE, _LOG(0.25)]],
        [[_NONE, _LOG(0.5)], [_LOG(0.5), _NONE]],
    ),
    divergent=([[_LOG(1.5)]],),
)
_COUNT_CLOSURES = ClosureCases(
    kind="linear",
    convergent=(
        [],
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0, 2.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
    ),
    divergent=([[1.0]], [[0.0, 1.0], [1.0, 0.0]]),
)
CLOSURE_CASES: dict[str, ClosureCases] = {
    "core/Boolean": _BOOLEAN_CLOSURES,
    "chart/Boolean": _BOOLEAN_CLOSURES,
    "core/Godel": ClosureCases(
        kind="idempotent",
        convergent=([], [[0.5, 0.25], [1.0, 0.0]], [[0.0, 1.0], [0.25, 0.5]]),
    ),
    "core/Tropical": ClosureCases(
        kind="idempotent",
        convergent=(
            [],
            [[0.0, 1.0], [2.5, INFINITY]],
            [
                [INFINITY, 1.0, INFINITY],
                [INFINITY, INFINITY, 2.5],
                [1.0, 0.0, INFINITY],
            ],
        ),
    ),
    "core/MaxPlus": _SCORE_CLOSURES,
    "chart/Viterbi": _SCORE_CLOSURES,
    "lawless/no-loops": _SCORE_CLOSURES,
    "core/LogProb": _LOG_CLOSURES,
    "chart/LogProb": _LOG_CLOSURES,
    "core/Counting": _COUNT_CLOSURES,
    "chart/Counting": _COUNT_CLOSURES,
    "core/Markov": ClosureCases(
        kind="linear",
        convergent=([], [[0.5, 0.25], [0.0, 0.5]], [[0.0, 0.5], [0.5, 0.0]]),
        divergent=([[1.5]], [[0.0, 2.0], [0.5, 0.0]]),
    ),
}
"""The closure cases of every row with a closure."""


def _matrix(rows: Element, dtype: torch.dtype) -> torch.Tensor:
    """A square matrix of plain numbers as a tensor.

    Parameters
    ----------
    rows : Element
        Rows of numbers, or ``[]`` for the matrix of order zero.
    dtype : torch.dtype
        The precision.

    Returns
    -------
    torch.Tensor
        Shape ``(n, n)``.
    """
    assert isinstance(rows, list)
    if not rows:
        return torch.zeros(0, 0, dtype=dtype)
    return torch.tensor(rows, dtype=dtype)


def reference_star(member: Member, rows: Element, kind: str) -> list[list[float]]:
    """The least solution of ``X = I + A X``, from the reference operations.

    Parameters
    ----------
    member : Member
        The row.
    rows : Element
        The matrix.
    kind : str
        How the least solution is computed.

    Returns
    -------
    list of list of float
        The closure.
    """
    assert isinstance(rows, list)
    n = len(rows)
    entry: list[list[float]] = [[_number(x) for x in row] for row in rows]  # type: ignore[union-attr]
    if kind == "linear":
        star = exact.linear_star([[Fraction(x) for x in row] for row in entry])
        return [[float(x) for x in row] for row in star]
    if kind == "log":
        weights = [
            [Fraction(math.exp(x)) if x != -INFINITY else Fraction(0) for x in row]
            for row in entry
        ]
        star = exact.linear_star(weights)
        return [[math.log(x) if x > 0 else -INFINITY for x in row] for row in star]
    reference = member.reference
    assert reference.times is not None and reference.zero is not None
    assert reference.unit is not None
    times = reference.times
    zero, unit = _number(reference.zero), _number(reference.unit)
    power = [[unit if i == j else zero for j in range(n)] for i in range(n)]
    total = [row[:] for row in power]
    for _ in range(n):
        power = [
            [
                _number(
                    reference.plus([times(power[i][k], entry[k][j]) for k in range(n)])
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        total = [
            [_number(reference.plus([total[i][j], power[i][j]])) for j in range(n)]
            for i in range(n)
        ]
    return total


def check_star(member: Member, dtype: torch.dtype) -> None:
    """The closure is the least solution of ``X = I ⊕ A X``, and refuses outside.

    Parameters
    ----------
    member : Member
        A row with a closure.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a closure differs from the least solution, is not a solution
        on either side, or a divergent closure is not refused.
    """
    structure = member.structure
    assert isinstance(structure, Star) and isinstance(structure, CompositionRule)
    cases = CLOSURE_CASES[member.name]
    for rows in cases.convergent:
        a = _matrix(rows, dtype)
        n = a.shape[0]
        found = structure.star(a)
        wanted = torch.tensor(reference_star(member, rows, cases.kind), dtype=dtype)
        agree(found, wanted.reshape(n, n), member, f"closure of {rows}")
        unit = structure.identity_tensor((n,), dtype=dtype)
        agree(
            found,
            structure.join_pair(unit, structure.compose(a, found, 1)),
            member,
            f"left fixed point of the closure of {rows}",
        )
        agree(
            found,
            structure.join_pair(unit, structure.compose(found, a, 1)),
            member,
            f"right fixed point of the closure of {rows}",
        )
        if cases.exhaustive and n == 2:
            _check_least(member, a, found, dtype)
    for rows in cases.divergent:
        try:
            value = structure.star(_matrix(rows, dtype))
        except DivergentClosureError:
            continue
        raise Mismatch(
            "refusal",
            math.inf,
            f"{member.name}: the closure of {rows} gave {value.tolist()}",
        )


def _check_least(
    member: Member, a: torch.Tensor, found: torch.Tensor, dtype: torch.dtype
) -> None:
    """Decide leastness over every matrix of order two of a finite carrier.

    Parameters
    ----------
    member : Member
        The row, whose samples are its whole carrier.
    a : torch.Tensor
        The matrix.
    found : torch.Tensor
        Its closure.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If some solution is not above the closure.
    """
    structure = member.structure
    assert isinstance(structure, Star) and isinstance(structure, CompositionRule)
    unit = structure.identity_tensor((2,), dtype=dtype)
    for entries in itertools.product(member.samples, repeat=4):
        x = torch.tensor(entries, dtype=dtype).reshape(2, 2)
        if torch.equal(x, structure.join_pair(unit, structure.compose(a, x, 1))):
            if not torch.equal(structure.join_pair(found, x), x):
                raise Mismatch(
                    "value",
                    math.inf,
                    f"{member.name}: the solution {x.tolist()} lies below the closure",
                )


LAW_CHECKS: dict[str, Callable[[Member, torch.dtype], int | None]] = {
    "operations-closed": check_closure,
    "join-iterated": check_iterated_join,
    "plus-associative": check_additive_associativity,
    "plus-commutative": check_additive_commutativity,
    "plus-unit": check_additive_identity,
    "times-associative": check_multiplicative_associativity,
    "times-unit": check_multiplicative_identity,
    "distributive": check_distributivity,
    "zero-annihilates": check_annihilation,
    "times-commutative": check_multiplicative_commutativity,
    "composition-associative": check_composition_associativity,
    "plus-idempotent": check_additive_idempotence,
    "plus-selective": check_selectivity,
    "tie-first-maximizer": check_tie_rule,
    "zero-sum-free": check_zero_sum_free,
    "no-zero-divisors": check_no_zero_divisors,
    "division": check_division,
    "additive-inverse": check_additive_inverse,
    "semimodule-action": check_semimodule_action,
    "star-least-solution": check_star,
    "zero-form": check_zero_form,
    "unit-form": check_unit_form,
    "empty-join": check_empty_join,
}
"""The check of every law of the catalogue."""


# -- the reference of the operations ----------------------------------------


def check_declared(member: Member, dtype: torch.dtype) -> None:
    """The operations are the ones the reference defines.

    Compares the binary sum on every pair, the sum over an axis of three
    terms, the product or the action on every pair, composition of two
    small matrices of samples, and the zero and the unit.

    Parameters
    ----------
    member : Member
        The row.
    dtype : torch.dtype
        The precision.
    """
    structure = member.structure
    reference = member.reference
    pool = samples(member, dtype)
    a, b = tuples(pool, 2)
    plain_a, plain_b = a.tolist(), b.tolist()
    assert isinstance(structure, Carrier | CompositionRule)
    agree(
        structure.join_pair(a, b),
        as_tensor(
            [reference.plus([x, y]) for x, y in zip(plain_a, plain_b, strict=True)], a
        ),
        member,
        "binary sum",
    )
    terms = torch.stack(tuples(pool, 3), dim=1)
    agree(
        structure.join(terms, 1),
        as_tensor([reference.plus(row) for row in terms.tolist()], terms),
        member,
        "sum of three terms",
    )
    if isinstance(structure, Semimodule):
        assert reference.times is not None
        k, m = _scalar_module_pairs(member, dtype)
        agree(
            structure.act(k, m),
            as_tensor(
                [
                    reference.times(x, y)
                    for x, y in zip(k.tolist(), m.tolist(), strict=True)
                ],
                m,
            ),
            member,
            "action",
        )
    elif reference.times is not None and isinstance(
        structure, UnitalCarrier | CompositionRule
    ):
        agree(
            structure.tensor_op(a, b),
            as_tensor(
                [reference.times(x, y) for x, y in zip(plain_a, plain_b, strict=True)],
                a,
            ),
            member,
            "product",
        )
    if isinstance(structure, CompositionRule):
        assert reference.times is not None
        times = reference.times
        f = _matrices(pool, 2, 3, 0)
        g = _matrices(pool, 3, 2, 1)
        plain_f, plain_g = f.tolist(), g.tolist()
        wanted = [
            [
                reference.plus([times(plain_f[i][j], plain_g[j][k]) for j in range(3)])
                for k in range(2)
            ]
            for i in range(2)
        ]
        agree(
            structure.compose(f, g, 1),
            torch.tensor(wanted, dtype=dtype),
            member,
            "composition",
        )
    if isinstance(structure, Carrier) and reference.zero is not None:
        agree(
            structure.zero_like((), pool),
            torch.tensor(reference.zero, dtype=dtype),
            member,
            "zero",
        )
    if isinstance(structure, UnitalCarrier) and reference.unit is not None:
        agree(
            structure.unit_like((), pool),
            torch.tensor(reference.unit, dtype=dtype),
            member,
            "unit",
        )


def check_second(member: Member, dtype: torch.dtype) -> None:
    """The binary sum and the sum over a pair agree to second order.

    On every pair of samples, the value, the first derivatives and the
    matrix of second derivatives of ``join_pair(a, b)`` and of ``join``
    over the stacked pair are numbers, and the second derivatives are
    the reference's formula.

    Parameters
    ----------
    member : Member
        A row whose reference has second derivatives.
    dtype : torch.dtype
        The precision.

    Raises
    ------
    Mismatch
        If a derivative is not a number or differs from the formula.
    """
    structure = _carrier(member)
    second = member.reference.second
    assert second is not None
    pool = samples(member, dtype)
    for a, b in itertools.product(pool.tolist(), repeat=2):
        point = torch.tensor([a, b], dtype=dtype)
        wanted = torch.tensor(second(a, b), dtype=dtype)
        for what, function in (
            ("binary sum", lambda t: structure.join_pair(t[0], t[1])),
            ("sum over the pair", lambda t: structure.join(t, 0)),
        ):
            leaf = point.clone().requires_grad_()
            (first,) = torch.autograd.grad(function(leaf), (leaf,), create_graph=True)
            if bool(torch.isnan(first).any()):
                raise Mismatch(
                    "not-a-number", math.inf, f"{member.name}: {what} at {(a, b)}"
                )
            found = torch.autograd.functional.hessian(function, point)
            if bool(torch.isnan(found).any()):
                raise Mismatch(
                    "not-a-number",
                    math.inf,
                    f"{member.name}: second derivative of the {what} at {(a, b)}",
                )
            agree(found, wanted, member, f"second derivative of the {what} at {(a, b)}")


# -- cells ------------------------------------------------------------------


def claimed(member: Member) -> tuple[str, ...]:
    """The laws a member claims.

    Parameters
    ----------
    member : Member
        The row.

    Returns
    -------
    tuple of str
        Their names.
    """
    return tuple(law.name for law in member.structure.laws())


def run(member: Member, law: str, dtype: torch.dtype = torch.float64) -> None:
    """Check one cell.

    Parameters
    ----------
    member : Member
        The row.
    law : str
        The law, ``"declared operations"`` for the reference.
    dtype : torch.dtype
        The precision.
    """
    if law == "declared operations":
        check_declared(member, dtype)
        return
    if law == "second derivatives":
        check_second(member, dtype)
        return
    LAW_CHECKS[law](member, dtype)


def cell(name: str) -> tuple[Member, str]:
    """The row and the law of a cell's name.

    Parameters
    ----------
    name : str
        ``"<row>/<law>"``, the row itself holding one slash.

    Returns
    -------
    tuple of (Member, str)
        The row and the law.
    """
    kind, row, law = name.split("/", 2)
    return MEMBERS[f"{kind}/{row}"], law


# -- lawless members --------------------------------------------------------


class _Averaging(RealAlgebra):
    """The mean as the sum: not associative."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The mean over axes.

        Parameters
        ----------
        t : torch.Tensor
            Numbers.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The mean.
        """
        return t.mean(dim=reduction_axes(t, dim))


class _Weighted(RealAlgebra):
    """A sum weighting the i-th term by i + 1: not commutative."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The weighted sum over axes.

        Parameters
        ----------
        t : torch.Tensor
            Numbers.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The weighted sum.
        """
        result = t
        for axis in reversed(reduction_axes(t, dim)):
            weights = torch.arange(1, result.shape[axis] + 1, dtype=t.dtype)
            shape = [1] * result.ndim
            shape[axis] = -1
            result = (result * weights.reshape(shape)).sum(dim=axis)
        return result


class _Offset(CountingAlgebra):
    """A sum plus one: zero is no identity."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum plus one.

        Parameters
        ----------
        t : torch.Tensor
            Counts.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The sum plus one.
        """
        return t.sum(dim=reduction_axes(t, dim)) + 1


class _Collapsing(RealAlgebra):
    """A reduction over several axes that adds one."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum, plus one over several axes.

        Parameters
        ----------
        t : torch.Tensor
            Numbers.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The reduction.
        """
        axes = reduction_axes(t, dim)
        return t.sum(dim=axes) + (len(axes) > 1)


class _Halving(CountingAlgebra):
    """A product that leaves the naturals."""

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product plus a half.

        Parameters
        ----------
        a : torch.Tensor
            Counts.
        b : torch.Tensor
            Counts.

        Returns
        -------
        torch.Tensor
            ``a b + 1/2``.
        """
        return a * b + 0.5


class _Squaring(RealAlgebra):
    """``a ⊗ b = a² b``: not associative."""

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a² b``.

        Parameters
        ----------
        a : torch.Tensor
            Numbers.
        b : torch.Tensor
            Numbers.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return a * a * b


class _Doubling(RealAlgebra):
    """``a ⊗ b = 2 a b``: one is no identity."""

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``2 a b``.

        Parameters
        ----------
        a : torch.Tensor
            Numbers.
        b : torch.Tensor
            Numbers.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return 2 * a * b


class _Truncated(CountingAlgebra):
    """Counting with the sum capped at one: not distributive."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum capped at one.

        Parameters
        ----------
        t : torch.Tensor
            Counts.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The capped sum.
        """
        return t.sum(dim=reduction_axes(t, dim)).clamp(max=1.0)


class _Absorbing(RealAlgebra):
    """``0 ⊗ b = b``: zero does not annihilate."""

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product, but ``b`` where ``a`` is zero.

        Parameters
        ----------
        a : torch.Tensor
            Numbers.
        b : torch.Tensor
            Numbers.

        Returns
        -------
        torch.Tensor
            The product.
        """
        a, b = torch.broadcast_tensors(a, b)
        return torch.where(a == 0, b, a * b)


class _Signed(RealAlgebra):
    """``a ⊗ b = a |b|``: not commutative."""

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a |b|``.

        Parameters
        ----------
        a : torch.Tensor
            Numbers.
        b : torch.Tensor
            Numbers.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return a * b.abs()


def _squared_sum(t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """The square of a sum: composition by it is not associative.

    Parameters
    ----------
    t : torch.Tensor
        Numbers.
    dim : int or tuple of int
        Axes.

    Returns
    -------
    torch.Tensor
        ``(Σ t)²``.
    """
    return t.sum(dim=dim) ** 2


class _Summing(MaxPlusAlgebra):
    """Max-plus with the log-sum as its sum: not idempotent."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The log-sum over axes.

        Parameters
        ----------
        t : torch.Tensor
            Scores.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            ``log Σ exp(t)``.
        """
        return torch.logsumexp(t, dim=reduction_axes(t, dim))


class _Middle(GodelAlgebra):
    """The midpoint of the largest and the smallest: not selective."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The midpoint over axes.

        Parameters
        ----------
        t : torch.Tensor
            Values.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            ``(max + min) / 2``.
        """
        axes = reduction_axes(t, dim)
        return (t.amax(dim=axes) + t.amin(dim=axes)) / 2


class _Splitting(MaxPlusAlgebra):
    """A maximum that splits the derivative of a tie."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """``amax``, whose derivative is shared among tied operands.

        Parameters
        ----------
        t : torch.Tensor
            Scores.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The maximum.
        """
        return t.amax(dim=reduction_axes(t, dim))


class _CancellingReal(ZeroSumFree, RealAlgebra):
    """The reals declared zero-sum-free, which they are not."""


class _DividingStandardMV(NoZeroDivisors, StandardMVAlgebra):
    """The Łukasiewicz t-norm declared free of zero divisors, which it is not."""


class _Subtracting(MarkovAlgebra):
    """Division replaced by subtraction."""

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a - b``.

        Parameters
        ----------
        a : torch.Tensor
            Numbers.
        b : torch.Tensor
            Numbers.

        Returns
        -------
        torch.Tensor
            The difference.
        """
        self.require_nonzero(b)
        return a - b


class _Adding(RealAlgebra):
    """Subtraction replaced by addition."""

    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a + b``.

        Parameters
        ----------
        a : torch.Tensor
            Numbers.
        b : torch.Tensor
            Numbers.

        Returns
        -------
        torch.Tensor
            The sum.
        """
        return a + b


class _TwiceModule(FreeSemimodule):
    """``k · m = 2 k m``: neither associative nor unital."""

    def act(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``2 k m``.

        Parameters
        ----------
        k : torch.Tensor
            Scalars.
        m : torch.Tensor
            Vectors.

        Returns
        -------
        torch.Tensor
            The action.
        """
        return 2 * k.unsqueeze(-1) * m


class _SquaredElement(FreeSemimodule):
    """``k · m = k m²``: not additive in the module."""

    def act(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``k m²``.

        Parameters
        ----------
        k : torch.Tensor
            Scalars.
        m : torch.Tensor
            Vectors.

        Returns
        -------
        torch.Tensor
            The action.
        """
        return k.unsqueeze(-1) * m * m


class _SquaredScalar(FreeSemimodule):
    """``k · m = k² m``: not additive in the scalar."""

    def act(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``k² m``.

        Parameters
        ----------
        k : torch.Tensor
            Scalars.
        m : torch.Tensor
            Vectors.

        Returns
        -------
        torch.Tensor
            The action.
        """
        return (k * k).unsqueeze(-1) * m


class _ShiftedModule(FreeSemimodule):
    """``k · m = k m + 1``: the zero is not fixed."""

    def act(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``k m + 1``.

        Parameters
        ----------
        k : torch.Tensor
            Scalars.
        m : torch.Tensor
            Vectors.

        Returns
        -------
        torch.Tensor
            The action.
        """
        return k.unsqueeze(-1) * m + 1


class _DefaultZero(RealAlgebra):
    """A zero made in the default dtype."""

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """Zeros of the default dtype.

        Parameters
        ----------
        shape : tuple of int
            The shape.
        like : torch.Tensor
            Ignored.

        Returns
        -------
        torch.Tensor
            Zeros.
        """
        del like
        return torch.zeros(shape)


class _ScalarUnit(RealAlgebra):
    """A unit that ignores the requested shape."""

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """One number.

        Parameters
        ----------
        shape : tuple of int
            Ignored.
        like : torch.Tensor
            The dtype and device.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``()``.
        """
        del shape
        return torch.ones((), dtype=like.dtype, device=like.device)


class _RawTropical(TropicalAlgebra):
    """A minimum that raises over no terms."""

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The minimum, one axis after another.

        Parameters
        ----------
        t : torch.Tensor
            Distances.
        dim : int or tuple of int
            Axes.

        Returns
        -------
        torch.Tensor
            The minimum.
        """
        result = t
        for axis in reversed(reduction_axes(t, dim)):
            result = result.min(dim=axis).values
        return result


class _NoLoops(MaxPlusAlgebra):
    """A closure that keeps only the empty path."""

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The identity matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            The matrix.

        Returns
        -------
        torch.Tensor
            The identity, whatever the matrix.
        """
        return self.identity_tensor((matrix.shape[0],), dtype=matrix.dtype)


def _lawless(name: str, structure: Lawful, like: str) -> Member:
    """A lawless member with the samples and reference of a shipped row.

    Parameters
    ----------
    name : str
        Its row.
    structure : Lawful
        The member.
    like : str
        The shipped row whose samples and reference it takes.

    Returns
    -------
    Member
        The row.
    """
    return MEMBERS[like].with_(name=f"lawless/{name}", structure=structure)


LAWLESS: dict[str, Member] = {
    "operations-closed": _lawless("halving", _Halving(), "core/Counting"),
    "join-iterated": _lawless("collapsing", _Collapsing(), "core/Real"),
    "plus-associative": _lawless("averaging", _Averaging(), "core/Real"),
    "plus-commutative": _lawless("weighted", _Weighted(), "core/Real"),
    "plus-unit": _lawless("offset", _Offset(), "core/Counting"),
    "times-associative": _lawless("squaring", _Squaring(), "core/Real"),
    "times-unit": _lawless("doubling", _Doubling(), "core/Real"),
    "distributive": _lawless("truncated", _Truncated(), "core/Counting"),
    "zero-annihilates": _lawless("absorbing", _Absorbing(), "core/Real"),
    "times-commutative": _lawless("signed", _Signed(), "core/Real"),
    "composition-associative": _lawless(
        "squared",
        CustomSemigroupoid(
            "Squared", torch.mul, _squared_sum, verify_associative=False
        ),
        "core/Real",
    ),
    "plus-idempotent": _lawless("summing", _Summing(), "core/MaxPlus"),
    "plus-selective": _lawless("middle", _Middle(), "core/Godel"),
    "tie-first-maximizer": _lawless("splitting", _Splitting(), "core/MaxPlus"),
    "zero-sum-free": _lawless("cancelling", _CancellingReal(), "core/Real").with_(
        samples=(-1.0, 0.0, 1.0)
    ),
    "no-zero-divisors": _lawless("dividing", _DividingStandardMV(), "core/Lukasiewicz"),
    "division": _lawless("subtracting", _Subtracting(), "core/Markov"),
    "additive-inverse": _lawless("adding", _Adding(), "core/Real"),
    "semimodule-action": _lawless(
        "twice", _TwiceModule(COUNTING, 3), "vector/Counting"
    ),
    "zero-form": _lawless("default-zero", _DefaultZero(), "core/Real"),
    "unit-form": _lawless("scalar-unit", _ScalarUnit(), "core/Real"),
    "empty-join": _lawless("raw-tropical", _RawTropical(), "core/Tropical"),
    "star-least-solution": _lawless("no-loops", _NoLoops(), "core/MaxPlus"),
}
"""A member that claims each law and breaks it."""

ACTION_LAWLESS: tuple[Member, ...] = (
    _lawless("twice", _TwiceModule(COUNTING, 3), "vector/Counting"),
    _lawless("squared-element", _SquaredElement(COUNTING, 3), "vector/Counting"),
    _lawless("squared-scalar", _SquaredScalar(COUNTING, 3), "vector/Counting"),
    _lawless("shifted", _ShiftedModule(COUNTING, 3), "vector/Counting"),
)
"""Modules that break each part of the semimodule action."""


ROWS: tuple[str, ...] = tuple(MEMBERS)
"""The rows of the grid."""

CELLS: tuple[str, ...] = tuple(
    f"{row}/{law}"
    for row in ROWS
    for law in (
        *claimed(MEMBERS[row]),
        "declared operations",
        *(("second derivatives",) if MEMBERS[row].reference.second else ()),
    )
)
"""Every cell: each claimed law of each row, its declared operations, and
the second derivatives of its sum where the reference has them."""

assert set(LAW_CHECKS) <= set(LAWS)


# -- deliberately wrong library code ----------------------------------------

ALGEBRAS = "quivers.core.algebras"
CARRIERS = "quivers.core._carriers"

LAW_MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="selection-replaced-by-sum",
        defect_class="aggregation",
        provenance="a best-score sum computed as a partition",
        target=f"{ALGEBRAS}:Selective.join",
        rewrites=(
            Rewrite(
                old="return self.select(t, axes).value",
                new="return torch.logsumexp(t, dim=axes)",
            ),
        ),
        rejected_by=(
            "core/MaxPlus/plus-idempotent",
            "core/MaxPlus/plus-selective",
            "core/MaxPlus/declared operations",
        ),
        min_effect=0.34,
    ),
    Mutation(
        name="log-sum-replaced-by-maximum",
        defect_class="aggregation",
        provenance="a partition reduction computed as a best score",
        target=f"{ALGEBRAS}:LogProbAlgebra.join",
        rewrites=(
            Rewrite(
                old="total = torch.logsumexp(safe, dim=axes)",
                new="total = safe.amax(dim=axes)",
            ),
        ),
        rejected_by=("core/LogProb/declared operations",),
        min_effect=0.5,
    ),
    Mutation(
        name="pair-sum-by-logaddexp",
        defect_class="gradient path",
        provenance=(
            "a binary log-sum by torch.logaddexp, whose second derivative is not "
            "a number at an absent operand"
        ),
        target=f"{ALGEBRAS}:LogProbAlgebra.join_pair",
        rewrites=(
            Rewrite(
                old="total = _LogSumPair.apply(",
                new="total = torch.logaddexp(",
            ),
        ),
        rejected_by=(
            "core/LogProb/second derivatives",
            "chart/LogProb/second derivatives",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="matrix-product-reversed",
        defect_class="order",
        provenance="the factors of a product that does not commute exchanged",
        target=f"{ALGEBRAS}:MatrixSemiring.tensor_op",
        rewrites=(
            Rewrite(
                old="self._base.tensor_op(a.unsqueeze(-1), b.unsqueeze(-3))",
                new="self._base.tensor_op(b.unsqueeze(-1), a.unsqueeze(-3))",
            ),
        ),
        rejected_by=("matrix/Counting/declared operations",),
        min_effect=0.5,
    ),
    Mutation(
        name="composition-factors-exchanged",
        defect_class="order",
        provenance="a contraction that multiplies the right factor first",
        target=f"{ALGEBRAS}:OrderPreservingSemiring.compose",
        rewrites=(
            Rewrite(
                old="product = self.tensor_op(m_expanded, n_expanded)",
                new="product = self.tensor_op(n_expanded, m_expanded)",
            ),
        ),
        rejected_by=("matrix/Counting/declared operations",),
        min_effect=0.5,
    ),
    Mutation(
        name="empty-join-in-the-default-type",
        defect_class="empty reduction",
        provenance="the zero of a reduction over no terms made in the default type",
        target=f"{CARRIERS}:empty_reduction",
        rewrites=(
            Rewrite(
                old="return fill + nothing",
                new="return (fill + nothing).to(torch.get_default_dtype())",
            ),
        ),
        also_in=(ALGEBRAS,),
        rejected_by=(
            "core/LogProb/empty-join",
            "matrix/MaxPlus/empty-join",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="empty-join-not-a-number",
        defect_class="empty reduction",
        provenance="a reduction over no terms that is not a number",
        target=f"{CARRIERS}:empty_reduction",
        rewrites=(
            Rewrite(
                old="nothing = t.sum(dim=axes).to(dtype=t.dtype)",
                new='nothing = t.sum(dim=axes).to(dtype=t.dtype) * float("nan")',
            ),
        ),
        also_in=(ALGEBRAS,),
        rejected_by=(
            "core/Tropical/empty-join",
            "vector/Counting/empty-join",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="empty-join-loses-an-axis",
        defect_class="empty reduction",
        provenance="a reduction over no terms that returns one number",
        target=f"{CARRIERS}:empty_reduction",
        rewrites=(
            Rewrite(old="return fill + nothing", new="return (fill + nothing).sum()"),
        ),
        also_in=(ALGEBRAS,),
        rejected_by=("core/Tropical/empty-join",),
        min_effect=1.0,
    ),
    Mutation(
        name="tie-derivative-split",
        defect_class="gradient path",
        provenance="a maximum whose derivative is shared among tied operands",
        target=f"{ALGEBRAS}:_select",
        rewrites=(
            Rewrite(
                old="return flat.gather(-1, witness).squeeze(-1), witness.squeeze(-1)",
                new="return flat.amax(dim=-1), witness.squeeze(-1)",
            ),
        ),
        rejected_by=(
            "core/MaxPlus/tie-first-maximizer",
            "chart/Viterbi/tie-first-maximizer",
        ),
        min_effect=0.25,
    ),
    Mutation(
        name="constant-in-the-default-type",
        defect_class="scalar coercion",
        provenance="a zero or unit made in the default dtype and device",
        target=f"{CARRIERS}:constant",
        rewrites=(
            Rewrite(
                old="return torch.full(shape, value, dtype=like.dtype, device=like.device)",
                new="return torch.full(shape, value, device=like.device)",
            ),
        ),
        also_in=(ALGEBRAS,),
        rejected_by=(
            "core/Real/zero-form",
            "core/Boolean/unit-form",
        ),
        min_effect=1.0,
    ),
)
"""Wrong algebra code, each rejected by named cells of the law grid."""


def every_claim() -> list[tuple[str, str]]:
    """Every pair of a row and a law it claims.

    Returns
    -------
    list of (str, str)
        The pairs.
    """
    return [(row, law) for row in ROWS for law in claimed(MEMBERS[row])]
