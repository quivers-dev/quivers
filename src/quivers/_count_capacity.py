"""Dependency-free exact-integer limits shared by count backends."""

_MAX_EXACT_COUNT = 2**53


def _check_scalar_count_capacity(
    value: int | float, *, label: str, signed: bool
) -> None:
    """Reject source counts outside exact Float64 integer representation.

    Parameters
    ----------
    value : int or float
        Numeric value before conversion to floating-point arithmetic.
    label : str
        Input role included in the error message.
    signed : bool
        Check both bounds for observations. Infinite observations remain outside
        support; trial parameters instead reject an infinite upper bound.

    Raises
    ------
    ValueError
        If a finite observation or an upper-bounded trial exceeds capacity.
    """
    exceeds = (
        value not in (float("inf"), -float("inf"))
        and (value > _MAX_EXACT_COUNT or value < -_MAX_EXACT_COUNT)
        if signed
        else value > _MAX_EXACT_COUNT
    )
    if exceeds:
        raise ValueError(f"{label} exceed the exact float64 integer capacity 2**53")


__all__ = []
