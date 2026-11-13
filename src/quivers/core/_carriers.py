"""Carrier validation for built-in support and closure operations."""

import torch

from quivers.core.algebras import (
    Algebra,
    BooleanAlgebra,
    BoundedSumProductAlgebra,
    CountingAlgebra,
    GodelAlgebra,
    LogProbAlgebra,
    MarkovAlgebra,
    MaxPlusAlgebra,
    ProductFuzzyAlgebra,
)


def validate_carrier(tensor: torch.Tensor, algebra: Algebra) -> None:
    """Check entries against a supported built-in algebra's carrier.

    Parameters
    ----------
    tensor : torch.Tensor
        Entries to check without modifying their values.
    algebra : Algebra
        Built-in algebra whose carrier is required.

    Raises
    ------
    TypeError
        If the algebra has no carrier check or the tensor is complex.
    ValueError
        If any entry lies outside the carrier.
    """
    if tensor.is_complex():
        raise TypeError("carrier entries must be real")
    kind = type(algebra)
    if kind in (LogProbAlgebra, MaxPlusAlgebra):
        valid = torch.isfinite(tensor) | torch.isneginf(tensor)
    elif kind is BooleanAlgebra:
        valid = (tensor == 0) | (tensor == 1)
    elif kind in (ProductFuzzyAlgebra, GodelAlgebra, BoundedSumProductAlgebra):
        valid = (tensor >= 0) & (tensor <= 1)
    elif kind in (CountingAlgebra, MarkovAlgebra):
        valid = torch.isfinite(tensor) & (tensor >= 0)
        if kind is CountingAlgebra and tensor.is_floating_point():
            valid = valid & (tensor == tensor.floor())
    else:
        raise TypeError(f"no built-in carrier check for {algebra!r}")
    if not bool(valid.all()):
        raise ValueError(f"entries lie outside the {algebra.name} carrier")
