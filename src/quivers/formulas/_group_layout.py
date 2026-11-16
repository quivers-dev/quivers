"""One immutable coordinate layout for emitted Gaussian group blocks."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx
import torch
from collections.abc import Mapping, Sequence

from quivers._values import OPAQUE_BY_VALUE, ValueModel


class GroupMemberLayout(ValueModel):
    """Locate one named coefficient within its joint Gaussian block.

    Parameters
    ----------
    prefix, slope : str
        Predictor namespace and expanded slope label.
    axis : str or None
        Component axis; None denotes a scalar coefficient.
    width, offset : int
        Component count and first coordinate in the joint block.
    coefficient : str
        Retained deterministic coefficient binding.
    raw_site : str
        Sample site supplying this member's coordinates.
    scale_site : str or None
        Actual emitted conditional scale site for noncentered reconstruction.
    """

    prefix: str
    slope: str
    axis: str | None
    width: int
    offset: int
    coefficient: str
    raw_site: str
    scale_site: str | None = None


class GroupBlockLayout(ValueModel):
    """Share Gaussian coefficient slices between compilation and prediction.

    Parameters
    ----------
    group, identity : str
        Original grouping factor and correlation-block identity.
    coordinate_axis : str or None
        Joint finite coordinate axis, absent for one scalar coefficient.
    parameterization : {"centered", "noncentered"}
        Whether raw sites contain coefficients or standard-normal innovations.
    members : tuple[GroupMemberLayout, ...]
        Ordered nonoverlapping coefficient slices.
    correlation_site : str or None
        Actual emitted Cholesky site; None denotes an independent block.
    """

    group: str
    identity: str
    coordinate_axis: str | None
    parameterization: Literal["centered", "noncentered"]
    members: tuple[GroupMemberLayout, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    correlation_site: str | None = None

    @property
    def width(self) -> int:
        """Return the joint coordinate count.

        Returns
        -------
        int
            Sum of the retained member widths.
        """
        return sum(member.width for member in self.members)

    @property
    def raw_sites(self) -> dict[str, int]:
        """Read the number of coordinates per group at each raw site.

        Returns
        -------
        dict[str, int]
            Sum of member widths at each retained sample-site name. Joint
            centered members share one site; independent members need not.
        """
        sites: dict[str, int] = {}
        for member in self.members:
            sites[member.raw_site] = sites.get(member.raw_site, 0) + member.width
        return sites


def _group_coefficients(
    draws: Mapping[str, torch.Tensor],
    layouts: Sequence[GroupBlockLayout],
    levels: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Reconstruct all Gaussian block coefficients over retained sample axes.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Original constrained sites with chain and draw axes.
    layouts : Sequence[GroupBlockLayout]
        Exact block, slice, scale, and Cholesky metadata from emission.
    levels : Mapping[str, Sequence[str]]
        Authored group labels in retained level order.

    Returns
    -------
    dict[tuple[str, str, str], torch.Tensor]
        Prefix, group, and slope to arrays shaped chain, draw, level and,
        for an explicitly indexed member, component. No model replay occurs.

    Raises
    ------
    ValueError
        If required coordinate metadata or retained shapes disagree.
    """
    result = {}
    for layout in layouts:
        count = len(levels[layout.group])
        if (
            layout.parameterization == "centered"
            and layout.correlation_site is not None
        ):
            raw = draws[layout.members[0].raw_site]
            coefficients = raw.reshape(*raw.shape[:2], count, layout.width)
        elif layout.correlation_site is not None:
            innovations, scales = [], []
            for member in layout.members:
                if member.scale_site is None:
                    raise ValueError(
                        "noncentered coefficients require retained scale-site metadata"
                    )
                raw = draws[member.raw_site]
                innovations.append(raw.reshape(*raw.shape[:2], count, member.width))
                scale = draws[member.scale_site]
                scales.append(scale.reshape(*scale.shape[:2], member.width))
            innovation = torch.cat(innovations, -1)
            scale = torch.cat(scales, -1)
            coefficients = (
                innovation @ draws[layout.correlation_site].transpose(-1, -2)
            ) * scale.unsqueeze(-2)
        else:
            coefficients = None
        for member in layout.members:
            if coefficients is not None:
                value = coefficients[..., member.offset : member.offset + member.width]
            else:
                raw = draws[member.raw_site]
                value = raw.reshape(*raw.shape[:2], count, member.width)
                if layout.parameterization == "noncentered":
                    if member.scale_site is None:
                        raise ValueError(
                            "noncentered coefficients require retained scale-site metadata"
                        )
                    scale = draws[member.scale_site]
                    value = value * scale.reshape(*scale.shape[:2], 1, member.width)
            result[(member.prefix, layout.group, member.slope)] = (
                value if member.axis is not None else value.squeeze(-1)
            )
    return result


__all__ = []
