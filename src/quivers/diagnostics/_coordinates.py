"""Dependency-free dimension checks shared by retained and ArviZ summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def axis_labels(
    dimension: str, size: int, coords: Mapping[str, Sequence[object]] | None
) -> tuple[object, ...]:
    """Resolve labels without coercing their values or parsing variable names.

    Parameters
    ----------
    dimension : str
        Named event, chain, or draw dimension.
    size : int
        Checked axis extent.
    coords : Mapping[str, Sequence[object]] or None
        Optional labels; absent dimensions use positional integer labels.

    Returns
    -------
    tuple[object, ...]
        Axis labels in their original order and carriers.

    Raises
    ------
    ValueError
        If a provided label sequence has the wrong length or is a bare string.
    """
    labels = coords.get(dimension) if coords is not None else None
    if labels is None:
        return tuple(range(size))
    if isinstance(labels, str | bytes) or len(labels) != size:
        raise ValueError(f"coordinate {dimension!r} requires {size} labels")
    return tuple(labels)


def event_dimensions(
    shapes: Mapping[str, Sequence[int]],
    *,
    coords: Mapping[str, Sequence[object]] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Validate named event axes and fill ArviZ's positional default names.

    Parameters
    ----------
    shapes : Mapping[str, Sequence[int]]
        Event shapes keyed by original variable names.
    coords : Mapping[str, Sequence[object]] or None
        Labels for named event axes.
    dims : Mapping[str, Sequence[str]] or None
        Leading named axes per variable; missing axes receive default names.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Complete event-axis names for each variable.

    Raises
    ------
    ValueError, TypeError
        If dimension rank, names, shared extents, or coordinate lengths disagree.
    """
    result = {}
    extents: dict[str, int] = {}
    for variable, shape in shapes.items():
        declared = dims.get(variable, ()) if dims is not None else ()
        if isinstance(declared, str | bytes):
            raise TypeError("dimensions must be sequences of axis names")
        authored = tuple(declared)
        if len(authored) > len(shape):
            raise ValueError(f"variable {variable!r} has only {len(shape)} event axes")
        names = authored + tuple(
            f"{variable}_dim_{axis}" for axis in range(len(shape) - len(authored))
        )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("dimension names must be nonempty strings")
        if len(set(names)) != len(names) or set(names) & {"chain", "draw"}:
            raise ValueError(
                "event dimension names must be distinct from chain and draw"
            )
        for name, extent in zip(names, shape, strict=True):
            if extents.setdefault(name, extent) != extent:
                raise ValueError(f"dimension {name!r} has incompatible extents")
            axis_labels(name, extent, coords)
        result[variable] = names
    return result


__all__: list[str] = []
