"""Shared response carriers and exact row totals for scalar/vector workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    from quivers.formulas.family import Family


def response_dtype(formula: Formula, family: Family | None = None) -> torch.dtype:
    """Read the response's scalar carrier from the shared semantic catalog.

    Parameters
    ----------
    formula : Formula
        Retained response event layout.
    family : Family or None
        Response family whose observed law declares its element type.

    Returns
    -------
    torch.dtype
        Canonical Int64/Bool carriers, or the established real precision.
    """
    from quivers.qiec.families import family as semantic_family
    from quivers.qiec.types import BOOL, INT

    element = semantic_family(family.observe_family).element if family is not None else None
    if element == INT or family is None and formula.response_layout.carrier == "integer":
        return torch.int64
    if element == BOOL:
        return torch.bool
    return torch.float64 if formula.response_layout.event_shape else torch.float32


def _response_values_tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Validate exact integer carriers before the shared native tensor boundary.

    Parameters
    ----------
    values : numpy.ndarray
        Stored numeric response values with untouched integer source precision.
    dtype : torch.dtype
        Canonical semantic response carrier.

    Returns
    -------
    torch.Tensor
        Response values without fractional truncation or large-count rounding.

    Raises
    ------
    ValueError
        If an integer/Boolean observation cannot inhabit its declared carrier.
    """
    if dtype in {torch.int64,torch.bool}:
        bound = 1 if dtype == torch.bool else 2**53
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or np.any(values > bound) or np.any(values < (0 if dtype == torch.bool else -bound)) or np.any(values != np.floor(values)):
            raise ValueError("Boolean responses require exact zero/one values" if dtype == torch.bool else "integer responses require finite integral values within exact Float64 scoring capacity 2**53")
    return torch.as_tensor(values.copy(),dtype=dtype)


def response_tensor(formula: Formula, family: Family | None = None) -> torch.Tensor:
    """Retain rows and event coordinates using the semantic law's scalar carrier.

    Parameters
    ----------
    formula : Formula
        Evaluated scalar or vector response and its carrier metadata.
    family : Family or None
        Response family selecting its canonical semantic count/Boolean carrier.

    Returns
    -------
    torch.Tensor
        Int64 integer responses, Boolean binary responses, Float64 real vector
        events, or Float32 real scalar observations. Count validation precedes
        conversion and never silently truncates fractional observations.
    """
    return _response_values_tensor(formula.response_values,response_dtype(formula,family))


def response_point(formula: Formula, family: Family) -> torch.Tensor:
    """Build one retained replay point without copying every training response.

    Parameters
    ----------
    formula : Formula
        Fitted response rows and event coordinates.
    family : Family
        Response law's semantic scalar carrier.

    Returns
    -------
    torch.Tensor
        One scalar/event point preserving the fitted response carrier.
    """
    return _response_values_tensor(formula.response_values[:1],response_dtype(formula,family))[0]


def count_totals(formula: Formula) -> torch.Tensor:
    """Validate count events and sum exact row totals within native capacity.

    Parameters
    ----------
    formula : Formula
        Matrix-valued response, with one named component per trailing entry.

    Returns
    -------
    torch.Tensor
        Exact Int64 row totals, including zero-trial rows.

    Raises
    ------
    ValueError
        If an event is negative, fractional, nonfinite, or exceeds capacity.
    """
    values = np.asarray(formula.response_values)
    if (
        values.ndim != 2
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 2**53)
        or np.any(values != np.floor(values))
    ):
        raise ValueError(
            "vector count responses require finite nonnegative integers at most 2**53"
        )
    totals = np.zeros(values.shape[0], dtype=np.int64)
    for column in values.T:
        totals += column.astype(np.int64)
        if np.any(totals > 2**53):
            raise ValueError("vector count row totals must be at most 2**53")
    return torch.as_tensor(totals)


def validate_simplex(formula: Formula) -> None:
    """Check interior compositional response events before native compilation.

    Parameters
    ----------
    formula : Formula
        Matrix-valued real response in retained component order.

    Raises
    ------
    ValueError
        If a row is not a finite, strictly positive simplex point.
    """
    values = np.asarray(formula.response_values)
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or not np.allclose(values.sum(-1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "compositional responses require finite positive components summing to one per row"
        )


def resolve_trials(family: Family, trials: int | str | None) -> int | str | None:
    """Apply scalar defaults while retaining inferred vector trial totals.

    Parameters
    ----------
    family : Family
        Typed response-family metadata.
    trials : int or str or None
        Explicit constant/column or an omitted trial specification.

    Returns
    -------
    int or str or None
        One trial for scalar families by default, or inferred totals for
        vector count families.
    """
    vector_counts = family.response is not None and family.response.kind in {
        "multinomial",
        "dirichlet_multinomial",
    }
    return 1 if trials is None and not vector_counts else trials
