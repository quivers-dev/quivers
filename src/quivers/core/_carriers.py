"""Carrier mechanics shared by the algebras.

The functions here know nothing about any particular algebra. They
normalize the axes of a reduction, build constants with the dtype and
device of a tensor, return the value of a reduction over no terms, and
check membership through the carrier's own predicate.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch

__all__: list[str] = []


class _Members(Protocol):
    """Anything with a name and an elementwise membership predicate."""

    @property
    def name(self) -> str:
        """The carrier's name."""
        ...

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Return whether each element of ``t`` is in the carrier.

        Parameters
        ----------
        t : torch.Tensor
            The elements.

        Returns
        -------
        torch.Tensor
            A Boolean tensor of the index shape of ``t``.
        """
        ...


def require_real(t: torch.Tensor) -> None:
    """Refuse complex entries.

    Parameters
    ----------
    t : torch.Tensor
        The entries.

    Raises
    ------
    TypeError
        If the entries are complex.
    """
    if t.is_complex():
        raise TypeError("carrier entries must be real")


def reduction_axes(
    t: torch.Tensor, dim: int | tuple[int, ...], element_rank: int = 0
) -> tuple[int, ...]:
    """Normalize the axes of a reduction over the index axes of ``t``.

    The last ``element_rank`` axes of ``t`` hold one carrier element and
    are never reduced; negative axes count from the last index axis.

    Parameters
    ----------
    t : torch.Tensor
        The tensor.
    dim : int or tuple of int
        The axes to reduce.
    element_rank : int
        The number of trailing axes of one element.

    Returns
    -------
    tuple of int
        The distinct nonnegative axes, sorted.

    Raises
    ------
    ValueError
        If no axis is named, an axis repeats, or the tensor has fewer
        axes than one element.
    IndexError
        If an axis is out of range.
    """
    index_rank = t.ndim - element_rank
    if index_rank < 0:
        raise ValueError(
            f"a tensor of {t.ndim} axes holds no element of rank {element_rank}"
        )
    named = (dim,) if isinstance(dim, int) else tuple(dim)
    if not named:
        raise ValueError("a reduction names at least one axis")
    axes: list[int] = []
    for axis in named:
        if not -index_rank <= axis < index_rank:
            raise IndexError(f"axis {axis} is out of range for {index_rank} index axes")
        axes.append(axis % index_rank)
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes {named} repeat")
    return tuple(sorted(axes))


def is_empty(t: torch.Tensor, axes: tuple[int, ...]) -> bool:
    """Return whether a reduction over ``axes`` has no terms.

    Parameters
    ----------
    t : torch.Tensor
        The tensor.
    axes : tuple of int
        Normalized axes.

    Returns
    -------
    bool
        True when one of the axes has size zero.
    """
    return any(t.shape[axis] == 0 for axis in axes)


def reduced_shape(t: torch.Tensor, axes: tuple[int, ...]) -> tuple[int, ...]:
    """The shape of ``t`` with ``axes`` removed.

    Parameters
    ----------
    t : torch.Tensor
        The tensor.
    axes : tuple of int
        Normalized axes.

    Returns
    -------
    tuple of int
        The remaining sizes in order.
    """
    return tuple(size for axis, size in enumerate(t.shape) if axis not in axes)


def representable(value: float, dtype: torch.dtype) -> bool:
    """Return whether a dtype holds a real number exactly.

    Parameters
    ----------
    value : float
        The number.
    dtype : torch.dtype
        The dtype.

    Returns
    -------
    bool
        True for floating dtypes, which hold every constant an algebra
        declares, and for integer and Boolean dtypes holding the finite
        integer ``value`` within their range.
    """
    if dtype.is_floating_point:
        return True
    if dtype.is_complex or not math.isfinite(value) or not value.is_integer():
        return False
    if dtype == torch.bool:
        return value in (0.0, 1.0)
    info = torch.iinfo(dtype)
    return info.min <= value <= info.max


def constant(value: float, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
    """A tensor filled with one real number in the dtype and device of ``like``.

    Parameters
    ----------
    value : float
        The number.
    shape : tuple of int
        The shape.
    like : torch.Tensor
        The tensor whose dtype and device are used.

    Returns
    -------
    torch.Tensor
        The constant, which does not require a gradient.

    Raises
    ------
    TypeError
        If the dtype of ``like`` cannot hold ``value`` exactly.
    """
    if not representable(value, like.dtype):
        raise TypeError(f"{like.dtype} cannot hold the constant {value}")
    return torch.full(shape, value, dtype=like.dtype, device=like.device)


def empty_reduction(
    t: torch.Tensor, axes: tuple[int, ...], fill: torch.Tensor
) -> torch.Tensor:
    """The value of a reduction over no terms, connected to ``t``.

    The result is ``fill`` plus the sum of ``t`` over ``axes``, which is
    an exact zero of the dtype of ``t`` since the sum has no terms. The
    sum keeps the result on the autograd graph of ``t`` when ``t``
    requires a gradient, so a gradient with respect to ``t`` exists and
    is zero.

    Parameters
    ----------
    t : torch.Tensor
        The tensor, with a size-zero axis among ``axes``.
    axes : tuple of int
        Normalized, nonempty axes.
    fill : torch.Tensor
        The identity of the reduction, with the shape of the result.

    Returns
    -------
    torch.Tensor
        ``fill`` in the dtype and device of ``t``.
    """
    if not t.requires_grad:
        return fill
    nothing = t.sum(dim=axes).to(dtype=t.dtype)
    return fill + nothing


def validate_carrier(tensor: torch.Tensor, carrier: _Members) -> None:
    """Check that every element of a tensor lies in a carrier.

    Parameters
    ----------
    tensor : torch.Tensor
        The entries, which are not modified.
    carrier : _Members
        The carrier, which decides membership by its own predicate.

    Raises
    ------
    TypeError
        If the entries are complex, or the carrier declares no membership
        predicate.
    ValueError
        If any element lies outside the carrier.
    """
    require_real(tensor)
    if not bool(carrier.contains(tensor).all()):
        raise ValueError(f"entries lie outside the {carrier.name} carrier")
