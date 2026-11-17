"""Shared validation and axis partition for independent product events."""

import operator
from typing import SupportsIndex, cast


def independent_ndims(value: object, batch_rank: int | None = None) -> int:
    """Validate a count of trailing batch axes without numeric coercion.

    Parameters
    ----------
    value : object
        Actual integer count; Boolean and fractional values are invalid.
    batch_rank : int or None
        Available batch rank, when known.

    Returns
    -------
    int
        Nonnegative count no larger than the supplied rank.

    Raises
    ------
    TypeError, ValueError
        If the count is not an integer or consumes unavailable axes.
    """
    if isinstance(value, bool):
        raise TypeError("Independent: reinterpreted_batch_ndims must be an integer")
    try:
        count = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError(
            "Independent: reinterpreted_batch_ndims must be an integer"
        ) from error
    if count < 0 or (batch_rank is not None and count > batch_rank):
        raise ValueError(
            f"Independent reinterprets {count} batch axes of a base with "
            f"{batch_rank if batch_rank is not None else 'unknown rank'}; "
            "the count must be nonnegative and cannot exceed the base batch rank"
        )
    return count


def independent_axes[A](
    batch: tuple[A, ...], event: tuple[A, ...], count: object
) -> tuple[tuple[A, ...], tuple[A, ...]]:
    """Move only the requested trailing batch axes into the event.

    Parameters
    ----------
    batch, event : tuple
        Existing independent batch and intrinsic event axes.
    count : object
        Valid integer axis count.

    Returns
    -------
    tuple[tuple, tuple]
        Remaining batch axes and the joined event axes.

    Raises
    ------
    TypeError, ValueError
        If the count is invalid for the base batch rank.
    """
    size = independent_ndims(count, len(batch))
    return (batch[:-size], (*batch[-size:], *event)) if size else (batch, event)


def reserved_parameter_axes[A](
    shape: tuple[A, ...], parameter_rank: int, outer_rank: int, reserved: int
) -> tuple[int, tuple[A, ...]]:
    """Partition a parameter by structural axis positions before comparing sizes.

    Parameters
    ----------
    shape : tuple
        Full parameter axes, including enclosing rows and intrinsic parameters.
    parameter_rank : int
        The semantic family's intrinsic numeric parameter rank.
    outer_rank : int
        Explicit enclosing batch rank.
    reserved : int
        Minimum trailing base batch rank required by Independent.

    Returns
    -------
    tuple[int, tuple]
        Number of enclosing axes consumed and all retained parameter axes.
        Equal numerical extents never select between these roles.
    """
    leading = max(len(shape) - parameter_rank, 0)
    consumed = min(outer_rank, max(leading - reserved, 0))
    return consumed, shape[consumed:]
