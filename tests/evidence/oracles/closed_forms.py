"""Values known in closed form.

Each function states the formula it evaluates. The argument is rational,
and a result that is not rational is returned as a float computed from
the exact argument.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction


class Divergent(ValueError):
    """The series has no finite value."""


def geometric(ratio: Fraction) -> Fraction:
    """The sum of all powers of a nonnegative ratio.

    Parameters
    ----------
    ratio : Fraction
        The ratio.

    Returns
    -------
    Fraction
        ``1 / (1 - ratio)``.

    Raises
    ------
    Divergent
        If the ratio is at least one.
    """
    if ratio >= 1:
        raise Divergent("the geometric series diverges")
    return 1 / (1 - ratio)


def least_quadratic_root(branch: Fraction) -> Fraction:
    """The least nonnegative root of ``x = p x^2 + q`` with ``p + q = 1``.

    Parameters
    ----------
    branch : Fraction
        The weight ``p``, strictly between zero and one.

    Returns
    -------
    Fraction
        ``min(1, q / p)``.
    """
    return min(Fraction(1), (1 - branch) / branch)


def two_state_star(
    stay: Fraction, forth: Fraction, back: Fraction, rest: Fraction
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """The inverse of ``I - A`` for ``A = [[stay, forth], [back, rest]]``.

    Parameters
    ----------
    stay : Fraction
        The weight from the first state to itself.
    forth : Fraction
        The weight from the first state to the second.
    back : Fraction
        The weight from the second state to the first.
    rest : Fraction
        The weight from the second state to itself.

    Returns
    -------
    tuple of tuple of Fraction
        The inverse by Cramer's rule.

    Raises
    ------
    Divergent
        If the series of powers of the nonnegative matrix diverges.
    """
    scale = (1 - stay) * (1 - rest) - forth * back
    if stay >= 1 or rest >= 1 or scale <= 0:
        raise Divergent("the series of matrix powers diverges")
    return (
        ((1 - rest) / scale, forth / scale),
        (back / scale, (1 - stay) / scale),
    )


def beta_binomial(heads: int, tails: int, a: Fraction, b: Fraction) -> Fraction:
    """The probability of one sequence of coin flips under a Beta prior.

    Parameters
    ----------
    heads : int
        The number of ones.
    tails : int
        The number of zeros.
    a : Fraction
        The first shape.
    b : Fraction
        The second shape.

    Returns
    -------
    Fraction
        The ratio of rising factorials
        ``(a)_heads (b)_tails / (a + b)_(heads + tails)``.
    """
    answer = Fraction(1)
    for i in range(heads):
        answer *= a + i
    for i in range(tails):
        answer *= b + i
    for i in range(heads + tails):
        answer /= a + b + i
    return answer


def dirichlet_categorical(
    counts: Sequence[int], concentration: Sequence[Fraction]
) -> Fraction:
    """The probability of one sequence of categories under a Dirichlet prior.

    Parameters
    ----------
    counts : Sequence of int
        How often each category occurs.
    concentration : Sequence of Fraction
        The concentration of each category.

    Returns
    -------
    Fraction
        ``prod_k (a_k)_(n_k) / (sum a)_(sum n)`` with rising factorials.
    """
    answer = Fraction(1)
    for count, shape in zip(counts, concentration, strict=True):
        for i in range(count):
            answer *= shape + i
    whole = sum(concentration, Fraction(0))
    for i in range(sum(counts)):
        answer /= whole + i
    return answer


def two_state_hmm(
    start: Fraction, stay: Fraction, emit: Fraction, symbols: Sequence[int]
) -> Fraction:
    """The likelihood of a symmetric two-state chain by the power of a matrix.

    The chain starts in the first state with probability ``start``, keeps
    its state with probability ``stay``, and emits its own index with
    probability ``emit``. In the basis of the sum and the difference of
    the two states one symbol acts by a two-by-two matrix.

    Parameters
    ----------
    start : Fraction
        The probability of the first state before the first symbol.
    stay : Fraction
        The probability of no change of state.
    emit : Fraction
        The probability that a state emits its own index.
    symbols : Sequence of int
        The observed symbols, zero or one.

    Returns
    -------
    Fraction
        The likelihood.
    """
    both, difference = Fraction(1), 2 * start - 1
    drift = 2 * stay - 1
    tilt = 2 * emit - 1
    for symbol in symbols:
        sign = 1 if symbol == 0 else -1
        both, difference = (
            (both + sign * tilt * drift * difference) / 2,
            (sign * tilt * both + drift * difference) / 2,
        )
    return both


def edit_distance(left: str, right: str) -> int:
    """The unit-cost edit distance by its recursive definition.

    Parameters
    ----------
    left : str
        The first string.
    right : str
        The second string.

    Returns
    -------
    int
        The least number of insertions, deletions, and substitutions.
    """
    if not left or not right:
        return len(left) + len(right)
    return min(
        edit_distance(left[1:], right) + 1,
        edit_distance(left, right[1:]) + 1,
        edit_distance(left[1:], right[1:]) + (left[0] != right[0]),
    )


def delannoy(left: int, right: int) -> int:
    """The number of alignments of two strings by three moves.

    Parameters
    ----------
    left : int
        The length of the first string.
    right : int
        The length of the second string.

    Returns
    -------
    int
        ``sum_k C(left, k) C(right, k) 2^k``.
    """
    return sum(
        math.comb(left, k) * math.comb(right, k) * 2**k for k in range(left + 1)
    )


def catalan(size: int) -> int:
    """The number of binary trees with a given number of inner nodes.

    Parameters
    ----------
    size : int
        The number of inner nodes.

    Returns
    -------
    int
        ``C(2 size, size) / (size + 1)``.
    """
    return math.comb(2 * size, size) // (size + 1)


def fibonacci(size: int) -> int:
    """The number of ways to write a number as ordered ones and twos.

    Parameters
    ----------
    size : int
        The number.

    Returns
    -------
    int
        The count, by the sum of binomials on a diagonal.
    """
    return sum(math.comb(size - k, k) for k in range(size // 2 + 1))
