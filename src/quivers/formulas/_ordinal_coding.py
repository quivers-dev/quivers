"""Integer category codes of ordinal and categorical responses, as brms reads them.

brms codes an ordinal response by positive integers: the categories are
``1, ..., max(y)``, whether or not each is observed, and the number of
thresholds is ``max(y) - 1`` (``extract_nthres``). A family with an extra
category (``hurdle_cumulative``) adds the code zero below them. A categorical
response takes the sorted distinct values as its categories. The compiled
program indexes categories from zero, so this module maps authored integer
values to positions and back.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quivers.formulas.family import Family

type CategoryCoding = Literal["ordinal", "hurdle", "nominal"]
"""How a family reads integer response values as categories."""

_ZERO_CODED: ContextVar[frozenset[str]] = ContextVar(
    "quivers_zero_coded_columns", default=frozenset()
)


def family_category_coding(family: Family) -> CategoryCoding | None:
    """Read how a family codes an integer response as categories.

    Parameters
    ----------
    family : Family
        Response family.

    Returns
    -------
    {"ordinal", "hurdle", "nominal"} or None
        The coding of an ordinal family (also as mixture components), of
        ``hurdle_cumulative``, and of unordered categories; None for a family
        whose response is not a category.
    """
    components = family.mixture.components if family.mixture is not None else (family,)
    kinds = {
        component.ordinal.kind
        for component in components
        if component.ordinal is not None
    }
    if "hurdle_cumulative" in kinds:
        return "hurdle"
    if kinds:
        return "ordinal"
    if family.unordered_categories:
        return "nominal"
    return None


@contextmanager
def zero_coded_columns(frame: pd.DataFrame) -> Iterator[None]:
    """Scope the columns a refit reads as zero-based category codes.

    A refit on a subset of the rows (a cross-validation fold, an update) must
    read an integer response as the original fit read it. A response whose
    original values hold a zero has the codes ``0, ..., K - 1``, also on a
    subset where the zero is absent.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows of the original fit.

    Yields
    ------
    None
        Control, with the numeric columns of ``frame`` whose smallest value
        is zero readable by `is_zero_coded`.
    """
    names = frozenset(
        str(name)
        for name in frame.columns
        if frame[name].dtype.kind in "iuf"
        and len(frame) > 0
        and float(np.nanmin(frame[name].to_numpy(dtype=np.float64))) == 0.0
    )
    token = _ZERO_CODED.set(names)
    try:
        yield
    finally:
        _ZERO_CODED.reset(token)


def is_zero_coded(name: str) -> bool:
    """Report whether a column is read as zero-based codes in this scope.

    Parameters
    ----------
    name : str
        Response column name.

    Returns
    -------
    bool
        True inside a refit whose original rows hold a zero in the column.
    """
    return name in _ZERO_CODED.get()


def _whole(values: np.ndarray) -> np.ndarray | None:
    """Read a response vector as exact integers when it holds only those.

    Parameters
    ----------
    values : numpy.ndarray
        Evaluated response.

    Returns
    -------
    numpy.ndarray or None
        Int64 values, or None for a response that is not a finite vector of
        whole numbers in the Int64 range.
    """
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
        return None
    numeric = values.astype(np.float64)
    if (
        not np.isfinite(numeric).all()
        or np.any(np.abs(numeric) >= 2**53)
        or not np.equal(numeric, np.round(numeric)).all()
    ):
        return None
    return values.astype(np.int64)


def integer_category_values(
    values: np.ndarray, coding: CategoryCoding, thresholds: int | None = None
) -> tuple[int, ...]:
    """Derive the integer value of each category from an integer response.

    Parameters
    ----------
    values : numpy.ndarray
        Evaluated response before any category coding.
    coding : {"ordinal", "hurdle", "nominal"}
        The family's reading of integer values.
    thresholds : int or None
        Number of thresholds declared by ``thres()``, or None to take it
        from the response as brms does.

    Returns
    -------
    tuple[int, ...]
        Category values in category order. Empty when the response keeps the
        zero-based codes ``0, ..., K - 1`` of this package, or is not a
        vector of whole numbers.

    Raises
    ------
    ValueError
        If a declared threshold count has fewer categories than the largest
        response.

    Notes
    -----
    Ordinal: a response whose smallest value is at least one has the
    categories ``1, ..., max(y)``, or ``1, ..., thresholds + 1``. brms
    refuses a zero for these families, and this package reads a response
    holding a zero as the codes ``0, ..., K - 1``.

    Hurdle: nonnegative values have the categories ``0, ..., max(y)``, or
    ``0, ..., thresholds + 1``.

    Nominal: values other than the contiguous codes ``0, ..., K - 1`` have
    their sorted distinct values as categories.

    Cost is linear in the number of rows.
    """
    whole = _whole(np.asarray(values))
    if whole is None:
        return ()
    low, high = int(whole.min()), int(whole.max())
    if coding == "nominal":
        distinct = np.unique(whole)
        if low == 0 and high == distinct.size - 1:
            return ()
        return tuple(int(value) for value in distinct)
    first = 0 if coding == "hurdle" else 1
    if low < first:
        return ()
    last = high if thresholds is None else thresholds + 1
    if high > last:
        raise ValueError(
            f"thres() declares the categories {first}, ..., {last}; response "
            f"codes must be integers among them, and the response holds {high}"
        )
    return tuple(range(first, last + 1))


def category_positions(
    values: np.ndarray, category_values: tuple[int, ...]
) -> np.ndarray:
    """Map integer response values to zero-based category positions.

    Parameters
    ----------
    values : numpy.ndarray
        Integer response values.
    category_values : tuple[int, ...]
        Integer value of each category, in category order.

    Returns
    -------
    numpy.ndarray
        Int64 positions into ``category_values``.

    Raises
    ------
    ValueError
        If a value is not a whole number or is not one of the categories.
    """
    whole = _whole(np.asarray(values))
    known = np.asarray(category_values, dtype=np.int64)
    if whole is None or not np.isin(whole, known).all():
        raise ValueError(
            "response values must be integers among the categories "
            f"{list(category_values)!r}"
        )
    return np.searchsorted(known, whole).astype(np.int64)


__all__: list[str] = []
