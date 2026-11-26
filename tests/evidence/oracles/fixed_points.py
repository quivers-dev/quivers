"""Least fixed points by Kleene iteration in exact arithmetic (O6 for idempotent systems).

A system is one polynomial per variable, each a list of monomials
``(coefficient, variables)``, over a semiring given by its sum, product
and zero on exact values. Kleene's iterates ``0, f(0), f(f(0)), ...``
increase to the least fixed point; in the max-plus and min-plus
semirings with no improving cycle they reach it after finitely many
steps, which the iteration detects by an iterate equal to its image.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

type Value = Fraction | None
"""An exact weight; ``None`` is the zero of the semiring (an infinity)."""

type Monomial = tuple[Fraction, tuple[int, ...]]
"""A coefficient and the variables it multiplies, with repeats."""


class NotReached(ValueError):
    """Kleene's iteration did not reach a fixed point within its steps."""


def least(
    system: Sequence[Sequence[Monomial]],
    plus: Callable[[Value, Value], Value],
    times: Callable[[Value, Value], Value],
    steps: int = 1000,
) -> list[Value]:
    """The least fixed point, when Kleene's iteration reaches it.

    Parameters
    ----------
    system : Sequence of Sequence of Monomial
        The polynomials.
    plus : Callable
        The sum.
    times : Callable
        The product.
    steps : int
        The most iterations.

    Returns
    -------
    list of Value
        The fixed point.

    Raises
    ------
    NotReached
        If no iterate equals its image within ``steps``.
    """
    point: list[Value] = [None] * len(system)
    for _ in range(steps):
        image: list[Value] = []
        for monomials in system:
            total: Value = None
            for coefficient, variables in monomials:
                term: Value = coefficient
                for variable in variables:
                    term = times(term, point[variable])
                total = plus(total, term)
            image.append(total)
        if image == point:
            return point
        point = image
    raise NotReached(f"no fixed point within {steps} iterations")


def max_plus(a: Value, b: Value) -> Value:
    """``max`` with ``None`` as minus infinity.

    Parameters
    ----------
    a, b : Value
        The operands.

    Returns
    -------
    Value
        The larger.
    """
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def plus_times(a: Value, b: Value) -> Value:
    """``+`` with ``None`` absorbing, as the product of max-plus and min-plus.

    Parameters
    ----------
    a, b : Value
        The operands.

    Returns
    -------
    Value
        The sum, ``None`` when either is.
    """
    if a is None or b is None:
        return None
    return a + b


def min_plus(a: Value, b: Value) -> Value:
    """``min`` with ``None`` as plus infinity.

    Parameters
    ----------
    a, b : Value
        The operands.

    Returns
    -------
    Value
        The smaller.
    """
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)
