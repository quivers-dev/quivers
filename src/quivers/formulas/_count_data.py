"""Original-array integer validation shared by response and trial adapters."""

from __future__ import annotations

import numpy as np

from quivers._count_capacity import _MAX_EXACT_COUNT


def checked_integer_data(
    values: np.ndarray,
    *,
    error: str,
    signed: bool = False,
    allow_bool: bool = False,
    maximum: int = _MAX_EXACT_COUNT,
) -> np.ndarray:
    """Validate original integer values before any lossy host conversion.

    Parameters
    ----------
    values : numpy.ndarray
        Original array retaining the supplied integer precision.
    error : str
        Role-specific diagnostic for invalid values.
    signed : bool
        Permit negative integers within the same exact arithmetic capacity.
    allow_bool : bool
        Permit Boolean source arrays where the response carrier allows them.
    maximum : int
        Inclusive magnitude bound, one for a Boolean response.

    Returns
    -------
    numpy.ndarray
        The same array, without allocation or numeric conversion.

    Raises
    ------
    ValueError
        If the dtype, finiteness, integrality, or bounds fail.
    """
    if (
        values.dtype.kind not in ("biuf" if allow_bool else "iuf")
        or not np.isfinite(values).all()
        or np.any(values < (-maximum if signed else 0))
        or np.any(values > maximum)
        or np.any(values != np.floor(values))
    ):
        raise ValueError(error)
    return values
