"""Shared response carriers and exact row totals for scalar/vector workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from quivers._count_capacity import _MAX_EXACT_COUNT
from quivers.formulas._count_data import checked_integer_data
import torch

from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    from quivers.formulas.family import Family


def normalize_response_values(
    values: np.ndarray,
    *,
    component_names: tuple[str, ...] = (),
    category_levels: tuple[str, ...] = (),
) -> np.ndarray:
    """Decode an evaluated response under its retained event coordinates.

    Parameters
    ----------
    values : numpy.ndarray
        Evaluated response before any integer or floating carrier conversion.
    component_names : tuple[str, ...]
        Explicit vector component labels, or empty for a scalar response.
    category_levels : tuple[str, ...]
        Fitted categorical level order, or empty for a numeric response.

    Returns
    -------
    numpy.ndarray
        Scalar category codes or numeric row/event values in Int64 or Float64;
        Boolean values become the integers zero and one.

    Raises
    ------
    ValueError
        If indicators, event dimensions, labels, or original integer values
        cannot inhabit the declared response representation.
    """
    values = np.asarray(values)
    if category_levels:
        if (
            values.ndim != 2
            or values.shape[1] != len(category_levels)
            or not np.all((values == 0) | (values == 1))
            or not np.all(values.sum(-1) == 1)
        ):
            raise ValueError(
                "categorical responses require one indicator per row in the declared level order"
            )
        values = values.argmax(-1).astype(np.int64)
    if component_names:
        if values.ndim != 2 or values.shape[1] != len(component_names):
            raise ValueError(
                "cbind response arguments must each produce one numeric column"
            )
        if len(component_names) < 2 or len(set(component_names)) != len(
            component_names
        ):
            raise ValueError(
                "cbind responses require at least two distinct component labels"
            )
    else:
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("vector responses require explicit cbind component labels")
    if values.dtype.kind == "b":
        # A Boolean response is the integer carrier of zero and one.
        values = values.astype(np.int64)
    if values.dtype.kind not in "iuf":
        raise ValueError("responses must have a real or integer numeric carrier")
    integer = values.dtype.kind in "iu"
    if integer and values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError("integer responses exceed the Int64 carrier")
    return values.astype(np.int64 if integer else np.float64)


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
    from quivers.qiec.types import BOOL, INT

    element = family.native.element if family is not None else None
    if (
        element == INT
        or family is None
        and formula.response_layout.carrier == "integer"
    ):
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
    if dtype in {torch.int64, torch.bool}:
        checked_integer_data(
            values,
            signed=dtype != torch.bool,
            allow_bool=True,
            maximum=1 if dtype == torch.bool else _MAX_EXACT_COUNT,
            error=(
                "Boolean responses require exact zero/one values"
                if dtype == torch.bool
                else "integer responses require finite integral values within exact Float64 scoring capacity 2**53"
            ),
        )
    return torch.as_tensor(values.copy(), dtype=dtype)


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
    return _response_values_tensor(
        formula.response_values, response_dtype(formula, family)
    )


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
    return _response_values_tensor(
        formula.response_values[:1], response_dtype(formula, family)
    )[0]


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
    error = "vector count responses require finite nonnegative integers at most 2**53"
    if values.ndim != 2:
        raise ValueError(error)
    checked_integer_data(values, error=error)
    totals = np.zeros(values.shape[0], dtype=np.int64)
    for column in values.T:
        totals += column.astype(np.int64)
        if np.any(totals > _MAX_EXACT_COUNT):
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
    vector_counts = family.vector_response and family.trials_parameter is not None
    return 1 if trials is None and not vector_counts else trials


def expected_response(
    distribution: torch.distributions.Distribution,
    family: Family,
    *,
    categories: int = 0,
    decision: bool = False,
    mc_samples: int | None = None,
) -> torch.Tensor:
    """Read the declared response measurement from one native conditional law.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        Native law produced by the existing program replay.
    family : Family
        Retained response-family semantics and category ordering.
    categories : int
        Actual ordered or categorical response extent, zero for numeric events.
    decision : bool
        Return Wiener lower/upper probabilities rather than reaction-time means.
    mc_samples : int or None
        Explicit Monte Carlo expectation budget for certified finite means.

    Returns
    -------
    torch.Tensor
        Response means, category probabilities, or decision probabilities.

    Raises
    ------
    ValueError, TypeError
        If the requested measurement is unavailable, undefined, or nonfinite.
    """
    from quivers.inference._response import category_probabilities, response_mean

    if family.circular:
        raise TypeError(
            "a circular response expectation requires an angular convention"
        )
    if decision or family.decision_event is not None:
        from quivers.continuous._wiener import Wiener

        if not isinstance(distribution, Wiener):
            raise TypeError("the response has no binary decision probabilities")
        value = (
            distribution.choice_probs if decision else distribution.mean_response_time
        )
    elif isinstance(distribution, torch.distributions.Categorical):
        value = distribution.probs
    elif family.response_ordinal() is not None:
        value = category_probabilities(distribution, categories)
    else:
        value = response_mean(distribution, mc_samples)
    if not torch.isfinite(value).all():
        raise ValueError(
            f"{family.observe_family} has an undefined or nonfinite response mean at a posterior draw"
        )
    return value
