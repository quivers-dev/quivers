"""Enclosures of least fixed points, decided in exact arithmetic (O8).

A system is a list of polynomials, one per variable, each a list of
monomials ``(coefficient, variables)`` with nonnegative rational
coefficients. By Knaster and Tarski the least fixed point is the least
pre-fixed point, so a point ``u`` with ``f(u) <= u`` bounds it from above;
the check evaluates ``f`` at the floats of ``u`` exactly, with no
rounding at all, so it is stronger than interval arithmetic with
outward rounding. A lower end is compared with a least fixed point known
in closed form.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

type Monomial = tuple[Fraction, tuple[int, ...]]
"""A coefficient and the variables it multiplies, with repeats."""

type System = Sequence[Sequence[Monomial]]
"""One polynomial per variable."""


def image(system: System, point: Sequence[Fraction]) -> list[Fraction]:
    """The polynomials at a point, exactly.

    Parameters
    ----------
    system : System
        The polynomials.
    point : Sequence of Fraction
        The variables' values.

    Returns
    -------
    list of Fraction
        Each polynomial's value.
    """
    found: list[Fraction] = []
    for monomials in system:
        total = Fraction(0)
        for coefficient, variables in monomials:
            term = coefficient
            for variable in variables:
                term *= point[variable]
            total += term
        found.append(total)
    return found


def pre_fixed(system: System, point: Sequence[float]) -> bool:
    """Whether ``f(u) <= u`` holds exactly at the floats of ``u``.

    Parameters
    ----------
    system : System
        The polynomials.
    point : Sequence of float
        ``u``, finite and nonnegative.

    Returns
    -------
    bool
        Whether every polynomial is at most its variable.
    """
    exact = [Fraction(value) for value in point]
    return all(
        value <= bound for value, bound in zip(image(system, exact), exact, strict=True)
    )


def encloses(lower: float, upper: float, exact: Fraction) -> bool:
    """Whether an interval of floats holds an exact value.

    Parameters
    ----------
    lower : float
        The lower end.
    upper : float
        The upper end.
    exact : Fraction
        The value.

    Returns
    -------
    bool
        ``lower <= exact <= upper``, decided exactly.
    """
    return Fraction(lower) <= exact <= Fraction(upper)
