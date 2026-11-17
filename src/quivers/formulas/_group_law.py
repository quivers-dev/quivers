"""The law of one block of group-level coefficients, and its planning.

A block is a matrix ``B`` of coefficients over the levels of one grouping
factor (rows) and the block's coefficient coordinates (columns). Its law is
the product of independent factors:

- the level covariance ``V`` between grouping levels, the identity or a known
  matrix on its support;
- the coefficient correlation ``R`` between coordinates, the identity, one
  LKJ factor, or one LKJ factor per level of a ``by`` variable;
- the scale ``sd`` of each coordinate, optionally indexed by the ``by`` level
  of a grouping level and by a component axis;
- a per-level Student-t mixing multiplier ``dfm``;
- a per-level weight on the standard-normal prior of the innovations;
- the row map carrying a level-indexed coefficient to the response rows.

With standard-normal innovations ``Z``,

    B = diag(dfm) . L_V Z . (diag(sd[by]) L_R[by])^T + mean.

The emitter, the coordinate layout read by prediction, the coefficient
reconstruction, and the per-factor structure read by postprocessing are all
functions of these laws.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._group_covariance import GroupCovarianceBasis
from quivers.formulas._group_layout import (
    GroupBlockLayout,
    GroupMemberLayout,
    _covariance_coefficients,
)

__all__: list[str] = []


class LevelCovariance(dx.TaggedUnion, discriminator="kind"):
    """The covariance between the levels of a grouping factor."""


class IdentityCovariance(LevelCovariance):
    """Independent, identically scaled grouping levels."""

    kind: Literal["identity"] = "identity"


class KnownCovariance(LevelCovariance):
    """A known covariance between grouping levels, on its support.

    Parameters
    ----------
    basis : GroupCovarianceBasis
        Projection from support coordinates to the physical levels and the
        factored support covariance.
    """

    basis: GroupCovarianceBasis = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    kind: Literal["known"] = "known"


class ByLevels(ValueModel):
    """The ``by`` variable partitioning the levels of a grouping factor.

    Parameters
    ----------
    variable : str
        Authored ``by`` variable.
    levels : tuple[str, ...]
        Its levels, in scale-site order.
    codes : tuple[int, ...]
        The ``by`` level of each grouping level, in level order.
    axis : str
        Finite axis over the ``by`` levels.
    input : str
        Runtime input holding ``codes`` over the levels in use.
    """

    variable: str
    levels: tuple[str, ...]
    codes: tuple[int, ...]
    axis: str
    input: str


class StudentMixing(ValueModel):
    """The scale mixture giving a grouping factor Student-t coefficients.

    Parameters
    ----------
    df_site : str
        Degrees-of-freedom site.
    mixing_site : str
        Per-level inverse-chi-square mixing site.
    multiplier : str
        Binding of the per-level multiplier ``sqrt(df * mixing)``.
    """

    df_site: str
    mixing_site: str
    multiplier: str


class PriorWeight(ValueModel):
    """Per-level weights on the standard-normal prior of the innovations.

    Parameters
    ----------
    input : str
        Runtime input holding one weight per level in use.
    weights : tuple[float, ...]
        Training weights, in level order.
    """

    input: str
    weights: tuple[float, ...]


class GroupLevelLaw(ValueModel):
    """The factors of a block's law that index the grouping levels.

    Parameters
    ----------
    group : str
        Authored grouping factor.
    axis : str
        Finite axis over its levels.
    size : int
        Number of levels.
    covariance : LevelCovariance
        Covariance between the levels.
    by : ByLevels or None
        Partition of the levels selecting scales and correlations.
    mixing : StudentMixing or None
        Student-t scale mixture; None for Gaussian coefficients.
    prior_weight : PriorWeight or None
        Weights on the innovations' prior; None for unit weights.
    """

    group: str
    axis: str
    size: int
    covariance: LevelCovariance = dx.field(
        default_factory=IdentityCovariance, opaque=True, extras=OPAQUE_BY_VALUE
    )
    by: ByLevels | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_VALUE)
    mixing: StudentMixing | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prior_weight: PriorWeight | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def basis(self) -> GroupCovarianceBasis | None:
        """Return the support basis of a known level covariance.

        Returns
        -------
        GroupCovarianceBasis or None
            The basis, or None for independent levels.
        """
        covariance = self.covariance
        return covariance.basis if isinstance(covariance, KnownCovariance) else None


class CoefficientCorrelation(dx.TaggedUnion, discriminator="kind"):
    """The correlation between the coefficient coordinates of a block."""

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the Cholesky-factor sites of the correlation.

        Returns
        -------
        tuple[str, ...]
            Sample sites, empty for independent coordinates.
        """
        return ()


class IndependentCoordinates(CoefficientCorrelation):
    """Uncorrelated coefficient coordinates."""

    kind: Literal["independent"] = "independent"


class LKJCorrelation(CoefficientCorrelation):
    """One LKJ-distributed correlation shared by every grouping level.

    Parameters
    ----------
    site : str
        Cholesky-factor site.
    """

    site: str
    kind: Literal["lkj"] = "lkj"

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the Cholesky-factor site.

        Returns
        -------
        tuple[str, ...]
            The one sample site.
        """
        return (self.site,)


class ByLKJCorrelation(CoefficientCorrelation):
    """One LKJ-distributed correlation per level of the ``by`` variable.

    Parameters
    ----------
    by_sites : tuple[str, ...]
        Cholesky-factor sites in ``by``-level order.
    """

    by_sites: tuple[str, ...]
    kind: Literal["by_lkj"] = "by_lkj"

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the Cholesky-factor sites.

        Returns
        -------
        tuple[str, ...]
            One sample site per ``by`` level.
        """
        return self.by_sites


class GroupScale(ValueModel):
    """The standard deviation of one member, over two optional axes.

    Parameters
    ----------
    site : str
        Standard-deviation site.
    by_axis : str or None
        Axis over the ``by`` levels, when each has its own scale.
    component_axis : str or None
        Axis over the member's components, when each has its own scale.
    prior : str or None
        Default prior; None for the configured group-level scale prior.
    lower : float or None
        Lower bound imposed on an explicitly configured prior; None leaves
        a configured prior as written.
    """

    site: str
    by_axis: str | None = None
    component_axis: str | None = None
    prior: str | None = None
    lower: float | None = None


class GroupMean(ValueModel):
    """The affine mean of one member over its components.

    Parameters
    ----------
    site : str
        Mean site.
    prior : str
        Default prior.
    """

    site: str
    prior: str


class MemberBinders(ValueModel):
    """The private bindings one member's emission introduces.

    Parameters
    ----------
    level, component, cell : str
        Variables ranging over the grouping levels, the member's components,
        and the flat level-by-component cells of its raw site.
    matrix : str or None
        Binding of a flat raw site read over levels and components.
    cell_scale, cell_mean : str or None
        Bindings of the scale and mean of each flat cell of a centered raw
        site.
    level_major : bool
        Whether the coefficient binding is indexed by level before
        component.
    """

    level: str = "random_group"
    component: str = "category_logit"
    cell: str = "random_cell"
    matrix: str | None = None
    cell_scale: str | None = None
    cell_mean: str | None = None
    level_major: bool = False


class RowMap(dx.TaggedUnion, discriminator="kind"):
    """The map from a level-indexed coefficient to the response rows."""


class SingleLevelRows(RowMap):
    """Each row reads the coefficient of its one grouping level.

    Parameters
    ----------
    index : str
        Runtime input holding each row's level.
    covariate : str or None
        Runtime input multiplying the coefficient; None for an intercept.
    """

    index: str
    covariate: str | None = None
    kind: Literal["single"] = "single"


class MembershipRows(RowMap):
    """Each row sums weighted coefficients over its member levels.

    Parameters
    ----------
    index : str
        Runtime input of the factor's single-level row index, which the
        member indices replace.
    indices : tuple[str, ...]
        Runtime inputs holding each row's level for every member.
    covariates : tuple[str, ...]
        Runtime inputs holding the product of each member's weight and
        covariate.
    """

    index: str
    indices: tuple[str, ...]
    covariates: tuple[str, ...]
    kind: Literal["membership"] = "membership"


class GroupMemberLaw(ValueModel):
    """One named coefficient of a block.

    Parameters
    ----------
    prefix, slope : str
        Predictor namespace and coefficient label.
    axis : str or None
        Component axis; None for a scalar coefficient.
    width, offset : int
        Component count and first coordinate within the block.
    coefficient : str
        Binding of the coefficient over the grouping levels.
    raw_site : str or None
        Sample site carrying the member's coordinates; None for a block
        whose level covariance has rank zero.
    scale : GroupScale
        Its standard deviation.
    rows : RowMap
        Its map to the response rows.
    mean : GroupMean or None
        Affine mean of the coefficient; None for a zero mean.
    binders : MemberBinders
        Private bindings of the member's emission.
    cell_axis : str or None
        Flat axis over the levels and components, level-major, indexing the
        raw site of a member with a component axis that owns its site.
    """

    prefix: str
    slope: str
    axis: str | None
    width: int
    offset: int
    coefficient: str
    raw_site: str | None
    scale: GroupScale = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rows: RowMap = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    mean: GroupMean | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_VALUE)
    binders: MemberBinders = dx.field(
        default_factory=MemberBinders, opaque=True, extras=OPAQUE_BY_VALUE
    )
    cell_axis: str | None = None

    @property
    def mean_site(self) -> str | None:
        """Return the site of the member's affine mean.

        Returns
        -------
        str or None
            Mean site, or None for a zero mean.
        """
        return self.mean.site if self.mean is not None else None

    @property
    def key(self) -> tuple[str, str]:
        """Return the member's identity within its grouping factor.

        Returns
        -------
        tuple[str, str]
            Predictor namespace and coefficient label.
        """
        return self.prefix, self.slope


class GroupBlockLaw(ValueModel):
    """The joint law of one block of group-level coefficients.

    Parameters
    ----------
    identity : str
        Block identity, unique within the grouping factor.
    level : GroupLevelLaw
        Factors over the grouping levels.
    members : tuple[GroupMemberLaw, ...]
        Coefficients in coordinate order.
    correlation : CoefficientCorrelation
        Correlation between the coordinates.
    parameterization : {"centered", "noncentered"}
        Whether the raw sites carry coefficients or standard-normal
        innovations.
    placement : {"joint", "inline"}
        Whether the block is emitted ahead of the predictors or where its
        one member's predictor is emitted.
    coordinate_axis : str or None
        Axis over the block's coordinates; None for one scalar coefficient.
    """

    identity: str
    level: GroupLevelLaw = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    members: tuple[GroupMemberLaw, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    correlation: CoefficientCorrelation = dx.field(
        default_factory=IndependentCoordinates, opaque=True, extras=OPAQUE_BY_VALUE
    )
    parameterization: Literal["centered", "noncentered"] = "noncentered"
    placement: Literal["joint", "inline"] = "joint"
    coordinate_axis: str | None = None

    @property
    def group(self) -> str:
        """Return the authored grouping factor.

        Returns
        -------
        str
            Grouping-factor name.
        """
        return self.level.group

    @property
    def width(self) -> int:
        """Return the number of coefficient coordinates.

        Returns
        -------
        int
            Sum of the member widths.
        """
        return sum(member.width for member in self.members)

    @property
    def layout(self) -> GroupBlockLayout:
        """Return the coordinate layout read by prediction and postprocessing.

        Returns
        -------
        GroupBlockLayout
            Raw sites, scale sites, and coordinate slices of the block.
        """
        correlation = self.correlation
        return GroupBlockLayout(
            group=self.group,
            identity=self.identity,
            coordinate_axis=self.coordinate_axis,
            parameterization=self.parameterization,
            members=tuple(
                GroupMemberLayout(
                    prefix=member.prefix,
                    slope=member.slope,
                    axis=member.axis,
                    width=member.width,
                    offset=member.offset,
                    coefficient=member.coefficient,
                    raw_site=member.raw_site,
                    scale_site=member.scale.site,
                    mean_site=member.mean_site,
                )
                for member in self.members
            ),
            correlation_site=correlation.site
            if isinstance(correlation, LKJCorrelation)
            else None,
            level_covariance=self.level.basis,
        )


def _rename(names: Mapping[str, str], value: str | None) -> str | None:
    """Rename an optional name, keeping names outside the mapping.

    Parameters
    ----------
    names : Mapping[str, str]
        Local names and their replacements.
    value : str or None
        Optional local name.

    Returns
    -------
    str or None
        The replacement, the name itself when it has none, or None.
    """
    return None if value is None else names.get(value, value)


def renamed_law(
    law: GroupBlockLaw,
    names: Mapping[str, str],
    objects: Mapping[str, str],
    prefix: Callable[[str], str],
    identity: str,
) -> GroupBlockLaw:
    """Carry a law into another namespace of bindings and objects.

    Parameters
    ----------
    law : GroupBlockLaw
        Law in its local namespace.
    names : Mapping[str, str]
        Local sites, bindings and inputs and their replacements.
    objects : Mapping[str, str]
        Local axes and their replacements.
    prefix : Callable[[str], str]
        Map from a local predictor namespace to its replacement.
    identity : str
        Block identity in the new namespace.

    Returns
    -------
    GroupBlockLaw
        The same law with every identity, axis, site and input renamed.
    """
    level = law.level
    by, mixing, weight = level.by, level.mixing, level.prior_weight
    correlation = law.correlation
    if isinstance(correlation, LKJCorrelation):
        correlation = correlation.with_(
            site=names.get(correlation.site, correlation.site)
        )
    elif isinstance(correlation, ByLKJCorrelation):
        correlation = correlation.with_(
            by_sites=tuple(names.get(site, site) for site in correlation.by_sites)
        )
    members = []
    for member in law.members:
        rows = member.rows
        if isinstance(rows, SingleLevelRows):
            rows = rows.with_(
                index=names.get(rows.index, rows.index),
                covariate=_rename(names, rows.covariate),
            )
        elif isinstance(rows, MembershipRows):
            rows = rows.with_(
                index=names.get(rows.index, rows.index),
                indices=tuple(names.get(name, name) for name in rows.indices),
                covariates=tuple(names.get(name, name) for name in rows.covariates),
            )
        members.append(
            member.with_(
                prefix=prefix(member.prefix),
                axis=_rename(objects, member.axis),
                coefficient=names.get(member.coefficient, member.coefficient),
                raw_site=_rename(names, member.raw_site),
                mean=None
                if member.mean is None
                else member.mean.with_(
                    site=names.get(member.mean.site, member.mean.site)
                ),
                binders=member.binders.with_(
                    matrix=_rename(names, member.binders.matrix),
                    cell_scale=_rename(names, member.binders.cell_scale),
                    cell_mean=_rename(names, member.binders.cell_mean),
                ),
                cell_axis=_rename(objects, member.cell_axis),
                scale=member.scale.with_(
                    site=names.get(member.scale.site, member.scale.site),
                    by_axis=_rename(objects, member.scale.by_axis),
                    component_axis=_rename(objects, member.scale.component_axis),
                ),
                rows=rows,
            )
        )
    return law.with_(
        identity=identity,
        coordinate_axis=_rename(objects, law.coordinate_axis),
        correlation=correlation,
        members=tuple(members),
        level=level.with_(
            axis=objects.get(level.axis, level.axis),
            by=None
            if by is None
            else by.with_(
                axis=objects.get(by.axis, by.axis), input=names.get(by.input, by.input)
            ),
            mixing=None
            if mixing is None
            else mixing.with_(
                df_site=names.get(mixing.df_site, mixing.df_site),
                mixing_site=names.get(mixing.mixing_site, mixing.mixing_site),
                multiplier=names.get(mixing.multiplier, mixing.multiplier),
            ),
            prior_weight=None
            if weight is None
            else weight.with_(input=names.get(weight.input, weight.input)),
        ),
    )


def option_prior_sites(
    laws: Iterable[GroupBlockLaw],
) -> tuple[tuple[str, str, str, str | None, float | None], ...]:
    """List the overrideable sites the level factors of the blocks add.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Emitted group blocks.

    Returns
    -------
    tuple[tuple[str, str, str, str or None, float or None], ...]
        Site, prior class (``df`` or ``cor``), group, coefficient label (the
        ``by`` level of a correlation), and lower bound: each Student-t
        degrees-of-freedom site once, then every per-``by``-level
        correlation.
    """
    laws = tuple(laws)
    degrees: dict[str, tuple[str, str, str, str | None, float | None]] = {}
    for law in laws:
        mixing = law.level.mixing
        if mixing is not None:
            degrees.setdefault(
                mixing.df_site, (mixing.df_site, "df", law.group, None, 1.0)
            )
    sites = list(degrees.values())
    for law in laws:
        correlation, by = law.correlation, law.level.by
        if isinstance(correlation, ByLKJCorrelation) and by is not None:
            for label, site in zip(by.levels, correlation.by_sites, strict=True):
                sites.append((site, "cor", law.group, label, None))
    return tuple(sites)


def group_coefficients(
    draws: Mapping[str, torch.Tensor],
    laws: Sequence[GroupBlockLaw],
    levels: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Reconstruct the coefficients of every block from its retained sites.

    Each block is rebuilt from the factors its law declares: innovations
    are correlated by the Cholesky factor of the block, or of each level's
    ``by`` level; scaled by each member's standard deviation, read at the
    level's ``by`` level when the scale has a ``by`` axis; and multiplied by
    the Student-t mixing multiplier ``sqrt(df * mixing)`` of each level. A
    known level covariance is decoded from its support coordinates. Cost is
    O(draws x levels x width^2).

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with chain and draw axes.
    laws : Sequence[GroupBlockLaw]
        Laws of the emitted blocks.
    levels : Mapping[str, Sequence[str]]
        Authored group labels in retained level order.

    Returns
    -------
    dict[tuple[str, str, str], torch.Tensor]
        Prefix, group, and slope to arrays shaped chain, draw, level and,
        for a member with a component axis, component.

    Raises
    ------
    ValueError
        If the retained ``by`` codes disagree with the level count, or a
        block without a known covariance lacks a raw site.
    """
    result = {}
    for law in laws:
        level = law.level
        count = len(levels[law.group])
        codes = None
        if level.by is not None:
            codes = torch.tensor(level.by.codes, dtype=torch.long)
            if codes.numel() != count:
                raise ValueError(
                    f"grouping factor {law.group!r} has {count} levels but "
                    f"{codes.numel()} by-level codes"
                )
        correlation = law.correlation
        if level.basis is not None:
            coefficients = _covariance_coefficients(draws, law.layout, sample_ndims=2)
        else:
            raws, scales = [], []
            for member in law.members:
                if member.raw_site is None:
                    raise ValueError(
                        f"group block {law.identity!r} has no retained raw site"
                    )
                raw = draws[member.raw_site]
                raws.append(raw)
                scale = draws[member.scale.site]
                scale = scale.reshape(*scale.shape[:2], -1, member.width)
                scales.append(
                    scale[:, :, codes]
                    if member.scale.by_axis is not None and codes is not None
                    else scale
                )
            if law.parameterization == "centered":
                if len({member.raw_site for member in law.members}) == 1:
                    coefficients = raws[0].reshape(*raws[0].shape[:2], count, law.width)
                else:
                    coefficients = torch.cat(
                        [
                            raw.reshape(*raw.shape[:2], count, member.width)
                            for raw, member in zip(raws, law.members, strict=True)
                        ],
                        -1,
                    )
            else:
                innovation = torch.cat(
                    [
                        raw.reshape(*raw.shape[:2], count, member.width)
                        for raw, member in zip(raws, law.members, strict=True)
                    ],
                    -1,
                )
                if isinstance(correlation, LKJCorrelation):
                    innovation = innovation @ draws[correlation.site].transpose(-1, -2)
                elif isinstance(correlation, ByLKJCorrelation):
                    if codes is None:
                        raise ValueError(
                            "per-by-level correlations require retained by-level codes"
                        )
                    factors = torch.stack(
                        [draws[site] for site in correlation.by_sites], dim=2
                    )
                    innovation = (
                        factors[:, :, codes] @ innovation.unsqueeze(-1)
                    ).squeeze(-1)
                coefficients = innovation * torch.cat(
                    [scale.expand(*scale.shape[:2], count, -1) for scale in scales], -1
                )
        if level.mixing is not None:
            df, udf = draws[level.mixing.df_site], draws[level.mixing.mixing_site]
            multiplier = (df.unsqueeze(-1) * udf.reshape(*udf.shape[:2], count)).sqrt()
            coefficients = coefficients * multiplier.unsqueeze(-1)
        for member in law.members:
            value = coefficients[..., member.offset : member.offset + member.width]
            result[(member.prefix, law.group, member.slope)] = (
                value if member.axis is not None else value.squeeze(-1)
            )
    return result
