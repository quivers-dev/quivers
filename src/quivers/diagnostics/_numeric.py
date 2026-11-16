"""Real retained-sample conversion shared by empirical posterior diagnostics."""

from __future__ import annotations

import torch

from quivers._count_capacity import _MAX_EXACT_COUNT


def real_samples(values: torch.Tensor, *, label: str) -> torch.Tensor:
    """Detach finite real samples, preserving original integer capacity checks.

    Parameters
    ----------
    values : torch.Tensor
        Real samples of arbitrary shape, including Boolean or integer draws.
    label : str
        Diagnostic name used in validation errors.

    Returns
    -------
    torch.Tensor
        Float64 values on the original device, without an autograd graph.

    Raises
    ------
    TypeError
        If values are not a real tensor.
    ValueError
        If values are nonfinite or an original integer exceeds the exact
        Float64 range. Missing values are not deleted or imputed.
    """
    if not isinstance(values, torch.Tensor) or values.is_complex():
        raise TypeError(f"{label} requires a real sample tensor")
    values = values.detach()
    if not values.is_floating_point():
        integers = values.to(torch.int64)
        unsigned_overflow = values.dtype == torch.uint64 and bool((integers < 0).any())
        if unsigned_overflow or bool(
            ((integers < -_MAX_EXACT_COUNT) | (integers > _MAX_EXACT_COUNT)).any()
        ):
            raise ValueError(f"{label} integer inputs must lie within +/-2**53")
    result = values.to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(
            f"{label} requires finite samples; missing values are not deleted"
        )
    return result


__all__: list[str] = []
