"""One immutable coordinate layout for emitted Gaussian group blocks."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

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
    """

    prefix: str
    slope: str
    axis: str | None
    width: int
    offset: int
    coefficient: str
    raw_site: str


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
    """

    group: str
    identity: str
    coordinate_axis: str | None
    parameterization: Literal["centered", "noncentered"]
    members: tuple[GroupMemberLayout, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

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


__all__ = []
