"""One immutable coordinate layout for emitted Gaussian group blocks."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._group_covariance import GroupCovarianceBasis


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
    raw_site : str or None
        Sample site supplying this member's coordinates; None denotes a
        deterministic zero-rank deviation.
    scale_site : str or None
        Actual emitted conditional scale site for noncentered reconstruction.
    mean_site : str or None
        Optional affine hypermean, retained outside the supported deviation
        subspace for grouped threshold log-spacing profiles.
    """

    prefix: str
    slope: str
    axis: str | None
    width: int
    offset: int
    coefficient: str
    raw_site: str | None
    scale_site: str | None = None
    mean_site: str | None = None


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
    level_covariance : GroupCovarianceBasis or None
        Known covariance support over labeled levels. Its raw coordinates
        index this support rather than physical levels. Centered coordinates
        carry Gaussian deviations; noncentered coordinates are standardized.
    """

    group: str
    identity: str
    coordinate_axis: str | None
    parameterization: Literal["centered", "noncentered"]
    members: tuple[GroupMemberLayout, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    correlation_site: str | None = None
    level_covariance: GroupCovarianceBasis | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
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
            if member.raw_site is None:
                continue
            sites[member.raw_site] = sites.get(member.raw_site, 0) + member.width
        return sites


__all__ = []
