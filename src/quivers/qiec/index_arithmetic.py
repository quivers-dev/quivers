"""Exact addition and cancellation for natural-number tensor extents.

The static language represents nonnegative affine expressions. These helpers
never introduce subtraction whose result has no representable Nat proof.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction

from quivers.qiec.kinds import NAT
from quivers.qiec.types import (
    IndexLiteral,
    IndexSum,
    IndexTerm,
    IndexVariable,
    add_indices,
    render_static,
)


def _coefficients(term: IndexTerm) -> tuple[Counter[IndexVariable], int]:
    """Expose the canonical nonnegative affine coefficients of a Nat term.

    Parameters
    ----------
    term : IndexTerm
        Natural-number dimension expression.

    Returns
    -------
    tuple[Counter[IndexVariable], int]
        Nominal variable multiplicities and the constant part.
    """
    canonical = add_indices(term)
    if isinstance(canonical, IndexLiteral):
        return Counter(), int(canonical.value)
    if isinstance(canonical, IndexVariable):
        return Counter({canonical: 1}), 0
    assert isinstance(canonical, IndexSum)
    return Counter(canonical.terms), canonical.constant


def solve_index_equation(
    left: IndexTerm, right: IndexTerm, unknown: IndexVariable
) -> IndexTerm | None:
    """Solve a Nat equality when cancellation leaves a nonnegative affine term.

    Parameters
    ----------
    left, right : IndexTerm
        Natural-number terms equated by a checked shape relation.
    unknown : IndexVariable
        Nonrigid Nat variable to solve; every other variable remains symbolic.

    Returns
    -------
    IndexTerm or None
        The exact solution, or None when it requires unproved subtraction,
        nonintegral coefficients, or does not determine this variable.

    Raises
    ------
    TypeError
        If an argument is not Nat, or the unknown is a rigid skolem.
    """
    if unknown.sort != NAT or unknown.identity is not None:
        raise TypeError("only a nonrigid Nat variable may be solved")
    coefficients, constant = _coefficients(left)
    other, right_constant = _coefficients(right)
    coefficients.subtract(other)
    constant -= right_constant
    divisor = coefficients.pop(unknown, 0)
    if not divisor:
        return None
    numerator = -constant
    if numerator % divisor or numerator // divisor < 0:
        return None
    terms: list[IndexTerm] = [IndexLiteral(value=numerator // divisor, sort=NAT)]
    for variable, coefficient in coefficients.items():
        numerator = -coefficient
        if numerator % divisor or numerator // divisor < 0:
            return None
        terms.extend([variable] * (numerator // divisor))
    return add_indices(*terms)


def infer_index_extents(
    equations: Sequence[tuple[IndexTerm, int]],
    given: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Infer extents forced by exact affine shape equations.

    Parameters
    ----------
    equations : Sequence[tuple[IndexTerm, int]]
        A checked Nat dimension and the observed integer axis length.
    given : Mapping[str, int] or None
        Previously fixed nonnegative integer extents.

    Returns
    -------
    dict[str, int]
        Supplied and uniquely determined extents. Underdetermined variables
        remain absent; no equal-size or free-variable heuristic is applied.

    Raises
    ------
    TypeError
        If evidence is not an integer, a term is not Nat, or a rigid variable
        is used as a named extent.
    ValueError
        If evidence is negative, equations disagree, or a forced solution is
        negative or nonintegral.
    """
    known = dict(given or {})
    for name, value in known.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"extent {name!r} must be an integer")
        if value < 0:
            raise ValueError(f"extent {name!r} must be nonnegative")
    forms = []
    names: set[str] = set()
    for term, extent in equations:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError("observed shape extents must be integers")
        if extent < 0:
            raise ValueError("observed shape extents must be nonnegative")
        coefficients, constant = _coefficients(term)
        row = Counter()
        for variable, coefficient in coefficients.items():
            if variable.identity is not None:
                raise TypeError("a rigid index cannot be inferred by its display name")
            if variable.name in known:
                constant += coefficient * known[variable.name]
            else:
                row[variable.name] += coefficient
                names.add(variable.name)
        residual = extent - constant
        if residual < 0:
            raise ValueError(
                f"natural-number shape equation for {render_static(term)} has no nonnegative solution"
            )
        forms.append((row, residual))
        if residual == 0:
            # A sum of nonnegative terms can vanish only term by term.
            forms.extend((Counter({name: 1}), 0) for name in row)
    # Preserve the common one-variable path as linear-time integer checks.
    # Coupled rows then enter the same exact solver with these values fixed.
    for form, residual in forms:
        if len(form) != 1:
            continue
        name, coefficient = next(iter(form.items()))
        if residual % coefficient:
            raise ValueError(f"extent {name!r} has no nonnegative integer solution")
        value = residual // coefficient
        if name in known and known[name] != value:
            raise ValueError(f"inconsistent natural-number shape equations for {name}")
        known[name] = value
    columns = sorted(names - known.keys())
    forms = [
        (
            form,
            residual
            - sum(
                coefficient * known[name]
                for name, coefficient in form.items()
                if name in known
            ),
        )
        for form, residual in forms
    ]
    if not columns:
        if any(value for _, value in forms):
            raise ValueError(
                "inconsistent natural-number shape equations for closed dimensions"
            )
        return known
    rows = [
        [*(Fraction(form[name]) for name in columns), Fraction(value)]
        for form, value in forms
    ]
    pivots = []
    next_row = 0
    for column in range(len(columns)):
        pivot = next((i for i in range(next_row, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[next_row], rows[pivot] = rows[pivot], rows[next_row]
        scale = rows[next_row][column]
        rows[next_row] = [value / scale for value in rows[next_row]]
        for i in range(len(rows)):
            if i != next_row and rows[i][column]:
                scale = rows[i][column]
                rows[i] = [
                    value - scale * basis
                    for value, basis in zip(rows[i], rows[next_row], strict=True)
                ]
        pivots.append((next_row, column))
        next_row += 1
    if any(not any(row[:-1]) and row[-1] for row in rows):
        raise ValueError(
            f"inconsistent natural-number shape equations for {', '.join(columns) or 'closed dimensions'}"
        )
    for row_index, column in pivots:
        row = rows[row_index]
        if any(value for j, value in enumerate(row[:-1]) if j != column):
            continue
        value = row[-1]
        if value.denominator != 1 or value < 0:
            raise ValueError(
                f"extent {columns[column]!r} has no nonnegative integer solution"
            )
        known[columns[column]] = int(value)
    return known


__all__ = ["infer_index_extents", "solve_index_equation"]
