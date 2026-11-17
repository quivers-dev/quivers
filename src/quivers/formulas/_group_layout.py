"""One immutable coordinate layout for emitted Gaussian group blocks."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx
import torch
from collections.abc import Mapping, Sequence

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
        if layout.level_covariance is not None:
            coefficients = _covariance_coefficients(draws, layout, sample_ndims=2)
            for member in layout.members:
                value = coefficients[..., member.offset : member.offset + member.width]
                result[(member.prefix, layout.group, member.slope)] = (
                    value if member.axis is not None else value.squeeze(-1)
                )
            continue
        if (
            layout.parameterization == "centered"
            and layout.correlation_site is not None
        ):
            raw_site = layout.members[0].raw_site
            assert raw_site is not None
            raw = draws[raw_site]
            coefficients = raw.reshape(*raw.shape[:2], count, layout.width)
        elif layout.correlation_site is not None:
            innovations, scales = [], []
            for member in layout.members:
                if member.scale_site is None:
                    raise ValueError(
                        "noncentered coefficients require retained scale-site metadata"
                    )
                assert member.raw_site is not None
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
                assert member.raw_site is not None
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


def _covariance_coefficients(
    draws: Mapping[str, torch.Tensor],
    layout: GroupBlockLayout,
    *,
    sample_ndims: int,
) -> torch.Tensor:
    """Decode physical coefficients from one explicit level-support basis.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sample bindings, optionally carrying retained sample axes.
    layout : GroupBlockLayout
        Covariance-aware Gaussian block with scale and optional mean metadata.
    sample_ndims : int
        Number of leading retained sample axes; zero denotes one replay draw.

    Returns
    -------
    torch.Tensor
        Sample axes followed by physical level and coefficient coordinates.

    Raises
    ------
    ValueError
        If the layout lacks its level covariance or Gaussian scale metadata.
    """
    basis = layout.level_covariance
    if basis is None:
        raise ValueError("group covariance reconstruction requires a retained basis")
    scales, means = [], []
    for member in layout.members:
        if member.scale_site is None:
            raise ValueError("group covariance requires retained scale metadata")
        scale = draws[member.scale_site]
        shape = scale.shape[:sample_ndims]
        scales.append(scale.reshape(*shape, member.width))
        means.append(
            draws[member.mean_site].reshape(*shape, member.width)
            if member.mean_site is not None
            else torch.zeros_like(scales[-1])
        )
    scale = torch.cat(scales, -1).to(
        dtype=torch.promote_types(scales[0].dtype, torch.float64)
    )
    mean = torch.cat(means, -1).to(scale).unsqueeze(-2)
    projection, _, factor = basis.tensors(like=scale)
    if basis.rank == 0:
        return mean.expand(*scale.shape[:-1], len(basis.levels), layout.width)
    site = layout.members[0].raw_site
    if site is None:
        raise ValueError("positive-rank group covariance requires a raw sample site")
    raw = draws[site].reshape(*scale.shape[:-1], basis.rank, layout.width).to(scale)
    if layout.parameterization == "centered":
        return projection @ raw + mean
    coefficient = raw
    if layout.correlation_site is not None:
        coefficient = coefficient @ draws[layout.correlation_site].to(
            coefficient
        ).transpose(-1, -2)
    return (factor @ coefficient) * scale.unsqueeze(-2) + mean


__all__ = []


def _encode_covariance_coefficients(
    coefficients: torch.Tensor,
    draws: Mapping[str, torch.Tensor],
    layout: GroupBlockLayout,
    *,
    parameterization: Literal["centered", "noncentered"] | None = None,
) -> torch.Tensor:
    """Encode one physical Gaussian block in its declared support coordinates.

    Parameters
    ----------
    coefficients : torch.Tensor
        Physical level by coefficient matrix for one posterior draw.
    draws : Mapping[str, torch.Tensor]
        Hyperparameter bindings used to standardize noncentered coordinates.
    layout : GroupBlockLayout
        Exact support, scale, mean and correlation metadata.
    parameterization : {"centered", "noncentered"} or None
        Optional innovation-space request, used by covariance-aware donors.

    Returns
    -------
    torch.Tensor
        Support by coefficient matrix, including an empty rank-zero matrix.

    Raises
    ------
    ValueError
        If the layout is not covariance-aware or lacks required scale metadata.
    """
    basis = layout.level_covariance
    if basis is None:
        raise ValueError("coefficient encoding requires a group covariance basis")
    scales, means = [], []
    for member in layout.members:
        if member.scale_site is None:
            raise ValueError("group covariance requires scale metadata")
        scales.append(draws[member.scale_site].reshape(member.width))
        means.append(
            draws[member.mean_site].reshape(member.width)
            if member.mean_site is not None
            else torch.zeros_like(scales[-1])
        )
    scale, mean = torch.cat(scales), torch.cat(means)
    projection, covariance, _ = basis.tensors(like=coefficients)
    value = projection.T @ (coefficients - mean)
    if (parameterization or layout.parameterization) == "centered" or not basis.rank:
        return value
    coordinate_factor = torch.tensor(
        basis.coordinate_factor, dtype=coefficients.dtype, device=coefficients.device
    ).reshape(basis.rank, basis.rank)
    value = torch.linalg.solve_triangular(coordinate_factor, value, upper=False)
    value = value / scale
    if layout.correlation_site is not None:
        value = torch.linalg.solve_triangular(
            draws[layout.correlation_site].to(value), value.T, upper=False
        ).T
    return value
