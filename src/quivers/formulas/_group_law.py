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

`plan_group_blocks` builds every block of a model from its predictors. The
emitter, the coordinate layout read by prediction, and the per-factor
structure read by postprocessing are all functions of the planned laws.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._group_covariance import GroupCovariance, GroupCovarianceBasis
from quivers.formulas._group_layout import (
    GroupBlockLayout,
    GroupLevelStructure,
    GroupMemberLayout,
)
from quivers.formulas._grouping import GroupLevelOptions
from quivers.formulas._grouping_emission import (
    all_group_options,
    by_axis,
    by_correlation_sites,
    by_input,
    joint_group_blocks,
    member_index,
    member_input,
    student_sites,
    weight_input,
)
from quivers.formulas.formula import Formula, RandomTerm, _qvr_name

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
    """

    site: str
    by_axis: str | None = None
    component_axis: str | None = None


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
    mean_site : str or None
        Affine mean of the coefficient; None for a zero mean.
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
    mean_site: str | None = None

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


def coefficient_binding(prefix: str, group: str, slope: str) -> str:
    """Name the binding of one coefficient over its grouping levels.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    group : str
        Grouping-factor name.
    slope : str
        Coefficient label.

    Returns
    -------
    str
        ``alpha_<group>`` for an intercept, ``beta_<group>_<slope>``
        otherwise, within the predictor namespace.
    """
    qgroup = _qvr_name(group)
    if slope == "Intercept":
        return f"{prefix}alpha_{qgroup}"
    return f"{prefix}beta_{qgroup}_{_qvr_name(slope)}"


def scale_site(prefix: str, group: str, slope: str) -> str:
    """Name the standard-deviation site of one coefficient.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    group : str
        Grouping-factor name.
    slope : str
        Coefficient label.

    Returns
    -------
    str
        Sample-site name.
    """
    return f"{prefix}sigma_{_qvr_name(group)}_{_qvr_name(slope)}"


def innovation_site(prefix: str, group: str, slope: str) -> str:
    """Name the standard-normal innovation site of one coefficient.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    group : str
        Grouping-factor name.
    slope : str
        Coefficient label.

    Returns
    -------
    str
        Sample-site name.
    """
    return f"{prefix}z_{_qvr_name(group)}_{_qvr_name(slope)}"


def row_map(formula: Formula, term: RandomTerm, prefix: str) -> RowMap:
    """Build the map from one coefficient's levels to the response rows.

    Parameters
    ----------
    formula : Formula
        Predictor owning the coefficient.
    term : RandomTerm
        Group-level coefficient.
    prefix : str
        Predictor namespace.

    Returns
    -------
    RowMap
        A weighted sum over members for a multi-membership factor, the row's
        one level and covariate otherwise.
    """
    index = f"{_qvr_name(term.group)}_idx"
    membership = formula.grouping.membership(term.group)
    if membership is not None:
        return MembershipRows(
            index=index,
            indices=tuple(
                member_index(term.group, member) for member in range(membership.size)
            ),
            covariates=tuple(
                member_input(prefix, term.group, term.slope, member)
                for member in range(membership.size)
            ),
        )
    if term.slope == "Intercept":
        return SingleLevelRows(index=index)
    return SingleLevelRows(
        index=index,
        covariate=prefix
        + (
            f"random_{term.column.qvr_name}"
            if term.column is not None
            else _qvr_name(term.slope)
        ),
    )


def _level_law(
    group: str,
    option: GroupLevelOptions | None,
    covariance: LevelCovariance,
) -> GroupLevelLaw:
    """Build the level factors of a block from its ``gr()`` options.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    option : GroupLevelOptions or None
        Level-coded options of the block.
    covariance : LevelCovariance
        Covariance between the levels.

    Returns
    -------
    GroupLevelLaw
        Level axis, covariance, and the declared ``by``, mixing, and prior
        weight factors.

    Raises
    ------
    ValueError
        If the options partition the levels without naming a ``by`` variable.
    """
    by = mixing = weight = None
    if option is not None and option.by_levels:
        if option.options.by is None:
            raise ValueError(f"gr({group}) has by levels without a by variable")
        by = ByLevels(
            variable=option.options.by,
            levels=option.by_levels,
            codes=option.by_codes,
            axis=by_axis(group),
            input=by_input(group),
        )
    if option is not None and option.options.student:
        df, udf, dfm = student_sites(group)
        mixing = StudentMixing(df_site=df, mixing_site=udf, multiplier=dfm)
    if option is not None and option.prior_weights:
        weight = PriorWeight(input=weight_input(group), weights=option.prior_weights)
    return GroupLevelLaw(
        group=group,
        axis=_qvr_name(group),
        covariance=covariance,
        by=by,
        mixing=mixing,
        prior_weight=weight,
    )


def member_axis(
    prefix: str,
    term: RandomTerm,
    component_axes: Mapping[str, tuple[str, int]],
    transition_axis: tuple[str, int] | None,
) -> tuple[str, int] | None:
    """Read the component axis of one group-level coefficient.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : RandomTerm
        Group-level coefficient.
    component_axes : Mapping[str, tuple[str, int]]
        Axis and width of each predictor with one coefficient per component.
    transition_axis : tuple[str, int] or None
        Axis and width of category-specific coefficients.

    Returns
    -------
    tuple[str, int] or None
        Axis name and width, or None for a scalar coefficient.

    Raises
    ------
    ValueError
        If a category-specific coefficient has no transition axis.
    """
    axis = component_axes.get(prefix)
    if axis is not None:
        return axis
    if term.column is not None and term.column.category_specific:
        if transition_axis is None:
            raise ValueError(
                f"category-specific coefficient {term.slope!r} requires a "
                "response with transitions"
            )
        return transition_axis
    return None


def plan_joint_block(
    group: str,
    identity: str,
    members: Sequence[tuple[str, RandomTerm, Formula]],
    axes: Sequence[tuple[str, int] | None],
    *,
    option: GroupLevelOptions | None,
    basis: GroupCovarianceBasis | None,
    correlate: bool,
    parameterization: Literal["centered", "noncentered"],
) -> GroupBlockLaw:
    """Build the law of one block emitted ahead of the predictors.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    identity : str
        Block identity.
    members : Sequence[tuple[str, RandomTerm, Formula]]
        Predictor namespace, coefficient, and owning predictor of each
        member, in coordinate order.
    axes : Sequence[tuple[str, int] or None]
        Component axis and width of each member.
    option : GroupLevelOptions or None
        Level-coded ``gr()`` options of the block.
    basis : GroupCovarianceBasis or None
        Support basis of a known level covariance.
    correlate : bool
        Whether the members share a correlation identifier.
    parameterization : {"centered", "noncentered"}
        Coordinates of the raw sites.

    Returns
    -------
    GroupBlockLaw
        The planned joint law.

    Raises
    ------
    ValueError
        If the axes do not align with the members.
    """
    if len(axes) != len(members):
        raise ValueError("group member axes must align with every coefficient")
    token = _qvr_name(identity)
    width = sum(axis[1] if axis is not None else 1 for axis in axes)
    covariance: LevelCovariance = (
        KnownCovariance(basis=basis) if basis is not None else IdentityCovariance()
    )
    level = _level_law(group, option, covariance)
    correlation: CoefficientCorrelation
    if basis is not None:
        correlation = (
            LKJCorrelation(site=f"cor_{token}")
            if correlate and width > 1
            else IndependentCoordinates()
        )
        shared_raw = f"group_covariance_{token}" if basis.rank else None
    else:
        correlation = (
            ByLKJCorrelation(by_sites=by_correlation_sites(identity, option))
            if option is not None and level.by is not None
            else LKJCorrelation(site=f"cor_{token}")
        )
        shared_raw = (
            f"group_coefficients_{token}" if parameterization == "centered" else None
        )
    laws, offset = [], 0
    for (prefix, term, formula), axis in zip(members, axes, strict=True):
        raw = (
            shared_raw
            if basis is not None or parameterization == "centered"
            else innovation_site(prefix, group, term.slope)
        )
        laws.append(
            GroupMemberLaw(
                prefix=prefix,
                slope=term.slope,
                axis=axis[0] if axis is not None else None,
                width=axis[1] if axis is not None else 1,
                offset=offset,
                coefficient=coefficient_binding(prefix, group, term.slope),
                raw_site=raw,
                scale=GroupScale(
                    site=scale_site(prefix, group, term.slope),
                    by_axis=level.by.axis
                    if level.by is not None and axis is None
                    else None,
                    component_axis=axis[0] if axis is not None else None,
                ),
                rows=row_map(formula, term, prefix),
            )
        )
        offset += laws[-1].width
    return GroupBlockLaw(
        identity=identity,
        level=level,
        members=tuple(laws),
        correlation=correlation,
        parameterization=parameterization,
        placement="joint",
        coordinate_axis=f"FormulaCorrelation_{token}",
    )


def plan_inline_block(
    prefix: str,
    term: RandomTerm,
    formula: Formula,
    axis: tuple[str, int] | None,
    *,
    parameterization: Literal["centered", "noncentered"],
) -> GroupBlockLaw:
    """Build the law of one independent coefficient emitted with its predictor.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : RandomTerm
        Group-level coefficient.
    formula : Formula
        Predictor owning the coefficient.
    axis : tuple[str, int] or None
        Component axis and width; None for a scalar coefficient.
    parameterization : {"centered", "noncentered"}
        Coordinates of the raw site.

    Returns
    -------
    GroupBlockLaw
        The planned one-member law.
    """
    group = term.group
    coefficient = coefficient_binding(prefix, group, term.slope)
    # A coefficient over a component axis reads none of the gr() options.
    option = formula.grouping.option(group) if axis is None else None
    level = _level_law(group, option, IdentityCovariance())
    if parameterization == "noncentered":
        raw = innovation_site(prefix, group, term.slope)
    elif axis is None:
        raw = coefficient
    else:
        raw = f"centered_{coefficient}"
    return GroupBlockLaw(
        identity=f"independent_{prefix}{_qvr_name(group)}_{_qvr_name(term.slope)}",
        level=level,
        members=(
            GroupMemberLaw(
                prefix=prefix,
                slope=term.slope,
                axis=axis[0] if axis is not None else None,
                width=axis[1] if axis is not None else 1,
                offset=0,
                coefficient=coefficient,
                raw_site=raw,
                scale=GroupScale(
                    site=scale_site(prefix, group, term.slope),
                    by_axis=level.by.axis if level.by is not None else None,
                    component_axis=axis[0] if axis is not None else None,
                ),
                rows=row_map(formula, term, prefix),
            ),
        ),
        correlation=IndependentCoordinates(),
        parameterization=parameterization,
        placement="inline",
        coordinate_axis=axis[0] if axis is not None else None,
    )


def plan_group_blocks(
    predictors: Iterable[tuple[str, str, Formula]],
    *,
    levels: Mapping[str, Sequence[str]],
    covariances: Mapping[str, GroupCovariance],
    parameterization: Literal["centered", "noncentered"],
    component_axes: Mapping[str, tuple[str, int]],
    transition_axis: tuple[str, int] | None,
    supplied: Collection[tuple[str, str, str]] = (),
) -> tuple[GroupBlockLaw, ...]:
    """Plan the law of every block of group-level coefficients of a model.

    Terms sharing a correlation identifier, and every term of a factor with
    a known level covariance, form joint blocks; a joint block of one scalar
    coordinate without a known covariance is an independent coefficient. The
    remaining terms are independent one-member blocks. Runs in time linear
    in the number of group terms.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Predicted parameters as (name, coefficient prefix, design).
    levels : Mapping[str, Sequence[str]]
        Labels of each grouping factor with a known covariance, in the
        order of its level axis.
    covariances : Mapping[str, GroupCovariance]
        Known level covariance by grouping factor.
    parameterization : {"centered", "noncentered"}
        Coordinates of the raw sites.
    component_axes : Mapping[str, tuple[str, int]]
        Axis and width of each predictor with one coefficient per component.
    transition_axis : tuple[str, int] or None
        Axis and width of category-specific coefficients.
    supplied : Collection[tuple[str, str, str]]
        Members, as (prefix, group, slope), whose block another program
        emits.

    Returns
    -------
    tuple[GroupBlockLaw, ...]
        Joint blocks in first-appearance order, then independent blocks in
        predictor and term order.
    """
    predictors = tuple(predictors)
    options = all_group_options(predictors)
    owners = {
        (prefix, term.group, term.slope): formula
        for _, prefix, formula in predictors
        for term in formula.random_terms
    }
    bases: dict[str, GroupCovarianceBasis] = {}
    laws: list[GroupBlockLaw] = []
    planned: set[tuple[str, str, str]] = set(supplied)
    for (group, identity), members in joint_group_blocks(
        predictors, covariances, supplied
    ).items():
        axes = [
            member_axis(prefix, term, component_axes, transition_axis)
            for prefix, term in members
        ]
        width = sum(axis[1] if axis is not None else 1 for axis in axes)
        if width < 2 and group not in covariances:
            continue
        if group in covariances and group not in bases:
            bases[group] = covariances[group].basis(levels[group])
        laws.append(
            plan_joint_block(
                group,
                identity,
                [
                    (prefix, term, owners[(prefix, group, term.slope)])
                    for prefix, term in members
                ],
                axes,
                option=options.get(group),
                basis=bases.get(group),
                correlate=any(term.correlation_id is not None for _, term in members),
                parameterization=parameterization,
            )
        )
        planned.update((prefix, group, term.slope) for prefix, term in members)
    for _, prefix, formula in predictors:
        for term in formula.random_terms:
            if (prefix, term.group, term.slope) in planned:
                continue
            laws.append(
                plan_inline_block(
                    prefix,
                    term,
                    formula,
                    member_axis(prefix, term, component_axes, transition_axis),
                    parameterization=parameterization,
                )
            )
    return tuple(laws)


def group_level_structures(
    laws: Iterable[GroupBlockLaw],
) -> dict[str, GroupLevelStructure]:
    """Describe the per-level structure of each grouping factor's blocks.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Planned group blocks.

    Returns
    -------
    dict[str, GroupLevelStructure]
        Grouping factors with a ``by`` partition or Student-t mixing mapped
        to the sites and codes their blocks declare.
    """
    result: dict[str, GroupLevelStructure] = {}
    for law in laws:
        level = law.level
        if level.by is None and level.mixing is None:
            continue
        existing = result.get(law.group, GroupLevelStructure())
        correlation = law.correlation
        sites = dict(existing.by_correlation_sites)
        if isinstance(correlation, ByLKJCorrelation):
            sites[law.identity] = correlation.by_sites
        result[law.group] = GroupLevelStructure(
            by=level.by.variable if level.by is not None else existing.by,
            by_levels=level.by.levels if level.by is not None else existing.by_levels,
            by_codes=level.by.codes if level.by is not None else existing.by_codes,
            by_correlation_sites=sites,
            df_site=level.mixing.df_site
            if level.mixing is not None
            else existing.df_site,
            mixing_site=level.mixing.mixing_site
            if level.mixing is not None
            else existing.mixing_site,
        )
    return result
