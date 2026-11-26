"""Unit weights of a score whose plates are subsampled.

A score ``S = sum_u w(u)`` is a sum over its units, so a uniform subset of
the units, each weight scaled by the number of units over the number kept,
estimates it without bias. A subsampled plate keeps a subset of the units
when no unit has elements of the plate both inside and outside the subset.
A unit that has both is split: its weight is not a sum over its elements,
so no subset of them estimates it.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from quivers.continuous.program_steps import (
    Environment,
    UnitFactorization,
)


def units_subsampled(
    factorization: UnitFactorization, subsets: Mapping[str, torch.Tensor]
) -> bool:
    """Whether a subsampled plate concerns a score.

    Parameters
    ----------
    factorization : UnitFactorization
        The score's units.
    subsets : Mapping[str, torch.Tensor]
        The positions kept of each subsampled plate.

    Returns
    -------
    bool
        ``True`` when a subsampled plate is a unit axis, or has elements
        in the units.
    """
    return any(
        name in subsets
        for name in (
            *factorization.axes,
            *(item.plate for item in factorization.members),
        )
    )


def unit_coefficients(
    factorization: UnitFactorization,
    env: Environment,
    subsets: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """The coefficient of each unit's weight under subsampled plates.

    Parameters
    ----------
    factorization : UnitFactorization
        The score's units.
    env : Environment
        The run's values.
    subsets : Mapping[str, torch.Tensor]
        The positions kept of each subsampled plate.

    Returns
    -------
    torch.Tensor
        ``(units,)``. With a subsampled unit axis, the number of positions
        of the axis over the number kept for a unit kept, and zero for any
        other, the product of these over the subsampled axes. Otherwise
        the number of units with elements of the subsampled plate over the
        number of those kept, zero for a unit not kept, and one for a unit
        with no element of the plate.

    Raises
    ------
    ValueError
        If a unit has elements of a subsampled plate inside and outside
        the subset, or the subsets of two plates would each select the
        units.
    """
    units = 1
    for size in factorization.sizes:
        units *= size
    inner = tuple(
        dict.fromkeys(
            item.plate for item in factorization.members if item.plate in subsets
        )
    )
    counts = {
        plate: _membership(factorization, plate, env, subsets, units) for plate in inner
    }
    axes = [name for name in factorization.axes if name in subsets]
    if axes:
        coefficients = torch.ones(units)
        positions = torch.unravel_index(torch.arange(units), factorization.sizes)
        for axis, size, along in zip(
            factorization.axes, factorization.sizes, positions, strict=True
        ):
            if axis not in subsets:
                continue
            index = subsets[axis].cpu()
            kept = torch.isin(along, index)
            coefficients = coefficients * kept * (size / int(index.numel()))
        for plate, (_, outside) in counts.items():
            _refuse_split(
                factorization, plate, (coefficients > 0) & (outside > 0), kept_by=axes
            )
        return coefficients
    if len(inner) > 1:
        raise ValueError(
            f"the score of the block over {factorization.latent!r} has rows or "
            f"inner units in the subsampled plates {list(inner)}, and each "
            "subset would select the block's units. How often a unit is in both "
            "subsets depends on how the subsets are drawn together, which their "
            "indices do not say; subsample one of the plates"
            + (
                f", or a unit axis {list(factorization.axes)}"
                if factorization.axes
                else ""
            )
        )
    ((plate, (inside, outside)),) = counts.items()
    _refuse_split(factorization, plate, (inside > 0) & (outside > 0), kept_by=None)
    governed = (inside + outside) > 0
    kept = inside > 0
    factor = int(governed.sum()) / int(kept.sum())
    return torch.where(governed, kept * factor, torch.ones(units))


def _membership(
    factorization: UnitFactorization,
    plate: str,
    env: Environment,
    subsets: Mapping[str, torch.Tensor],
    units: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """How many elements of a plate each unit has inside and outside a subset.

    Parameters
    ----------
    factorization : UnitFactorization
        The score's units.
    plate : str
        The plate.
    env : Environment
        The run's values.
    subsets : Mapping[str, torch.Tensor]
        The positions kept of each subsampled plate.
    units : int
        The number of units.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The number inside and the number outside, each ``(units,)``.
    """
    inside = torch.zeros(units, dtype=torch.long)
    outside = torch.zeros(units, dtype=torch.long)
    index = subsets[plate].cpu()
    for item in factorization.members:
        if item.plate != plate:
            continue
        held = item.units(env).cpu()
        within = torch.isin(item.positions(env).cpu(), index)
        inside = inside + torch.bincount(held[within], minlength=units)
        outside = outside + torch.bincount(held[~within], minlength=units)
    return inside, outside


def _refuse_split(
    factorization: UnitFactorization,
    plate: str,
    split: torch.Tensor,
    kept_by: list[str] | None,
) -> None:
    """Refuse a subset that keeps part of a unit.

    Parameters
    ----------
    factorization : UnitFactorization
        The score's units.
    plate : str
        The subsampled plate.
    split : torch.Tensor
        Whether the subset splits each unit, ``(units,)``.
    kept_by : list[str] or None
        The subsampled unit axes that keep the unit, when they select the
        units.

    Raises
    ------
    ValueError
        If a unit is split.
    """
    if not bool(split.any()):
        return
    unit = int(split.nonzero()[0])
    if factorization.axes:
        position = torch.unravel_index(torch.tensor(unit), factorization.sizes)
        where = "the unit " + ", ".join(
            f"{axis}={int(index)}"
            for axis, index in zip(factorization.axes, position, strict=True)
        )
    else:
        where = f"the one unit of the block over {factorization.latent!r}"
    held = (
        "has rows or inner units inside the subset and outside it"
        if kept_by is None
        else f"is kept by the subset of {kept_by}, and some of its rows or inner "
        f"units are outside the subset of {plate!r}"
    )
    raise ValueError(
        f"the subset of plate {plate!r} splits a unit: {where} {held}. They share "
        f"one {factorization.latent!r}, so their joint score is not a sum over "
        f"them and a part of them does not estimate it. Subsample whole units"
    )


__all__ = ["unit_coefficients", "units_subsampled"]
