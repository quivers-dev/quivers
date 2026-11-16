"""Shared response carriers and exact row totals for scalar/vector workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    from quivers.formulas.family import Family


def response_tensor(formula: Formula, family: Family | None = None) -> torch.Tensor:
    """Retain observation rows and declared event coordinates without flattening.

    Parameters
    ----------
    formula : Formula
        Evaluated scalar or vector response and its carrier metadata.
    family : Family or None
        Optional native response metadata selecting canonical count events.

    Returns
    -------
    torch.Tensor
        Int64 vector counts, Float64 real vector events, or existing Float32
        scalar observations.
    """
    dtype = torch.float32
    if formula.response_layout.event_shape:
        counts = (
            family is not None
            and family.response is not None
            and family.response.kind in {"multinomial", "dirichlet_multinomial"}
        )
        dtype = (
            torch.long
            if counts or formula.response_layout.carrier == "integer"
            else torch.float64
        )
    return torch.as_tensor(formula.response_values.copy(), dtype=dtype)


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
