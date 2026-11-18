"""Plan the law of every block of group-level coefficients of a model.

`plan_group_blocks` builds one `GroupBlockLaw` per block from the predictors,
the known level covariances and the component axes of the response. Site and
binding names are fixed here; the emitter and the readers take them from the
laws.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal

from quivers.formulas._group_covariance import GroupCovariance, GroupCovarianceBasis
from quivers.formulas._group_law import (
    ByLevels,
    ByLKJCorrelation,
    CoefficientCorrelation,
    GroupBlockLaw,
    GroupLevelLaw,
    GroupMean,
    GroupMemberLaw,
    GroupScale,
    IdentityCovariance,
    IndependentCoordinates,
    KnownCovariance,
    LevelCovariance,
    LKJCorrelation,
    MembershipRows,
    PriorWeight,
    RowMap,
    SingleLevelRows,
    StudentMixing,
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


def cell_axis(prefix: str, group: str, slope: str) -> str:
    """Name the flat level-by-component axis of one coefficient's raw site.

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
        Axis name.
    """
    return f"{prefix}FormulaRandomCell_{_qvr_name(group)}_{_qvr_name(slope)}"


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
    size: int,
    option: GroupLevelOptions | None,
    covariance: LevelCovariance,
) -> GroupLevelLaw:
    """Build the level factors of a block from its ``gr()`` options.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    size : int
        Number of levels.
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
        size=size,
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
    level = _level_law(
        group, len(members[0][2].group_levels[group]), option, covariance
    )
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
                cell_axis=cell_axis(prefix, group, term.slope)
                if axis is not None and raw != shared_raw
                else None,
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
    option = formula.grouping.option(group)
    level = _level_law(
        group, len(formula.group_levels[group]), option, IdentityCovariance()
    )
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
                cell_axis=cell_axis(prefix, group, term.slope)
                if axis is not None
                else None,
            ),
        ),
        correlation=IndependentCoordinates(),
        parameterization=parameterization,
        placement="inline",
        coordinate_axis=axis[0] if axis is not None else None,
    )


def admit_block(
    law: GroupBlockLaw, *, component_valued: Collection[str], family: str
) -> None:
    """Refuse a block whose level factors its carrier cannot express.

    Parameters
    ----------
    law : GroupBlockLaw
        Planned block.
    component_valued : Collection[str]
        Namespaces of the predictors with one coefficient per component.
    family : str
        Response family, named in the refusal.

    Raises
    ------
    ValueError
        For a ``by`` partition, Student-t mixing, or prior weights with
        centered coordinates; for a ``by`` partition or prior weights with a
        known level covariance, whose innovations live in support
        coordinates that no level owns; and for a ``by`` partition or prior
        weights on a predictor with one coefficient per component.
    """
    level = law.level
    indexed = level.by is not None or level.prior_weight is not None
    if law.parameterization == "centered" and (indexed or level.mixing is not None):
        raise ValueError(
            f"gr({law.group}) options by, dist, and pw use brms's noncentered "
            "innovations; use reparameterize='noncentered'"
        )
    if level.basis is not None and indexed:
        raise ValueError(
            f"gr({law.group}) options by, dist, and pw are not supported together "
            "with a known level covariance"
        )
    if indexed and any(member.prefix in component_valued for member in law.members):
        raise ValueError(
            f"gr({law.group}) options by, dist, and pw are supported for scalar "
            f"group coefficients, not for {family} predictors that "
            "carry one coefficient per response category"
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


def plan_threshold_block(
    group: str,
    size: int,
    axis: str,
    width: int,
    basis: GroupCovarianceBasis | None,
    *,
    cell_axis: str,
    parameterization: Literal["centered", "noncentered"],
) -> GroupBlockLaw:
    """Build the law of the grouped log spacings of ordinal thresholds.

    Parameters
    ----------
    group : str
        Grouping factor of the thresholds.
    size : int
        Number of levels.
    axis : str
        Axis over the free log-spacing coordinates.
    width : int
        Number of free log-spacing coordinates.
    basis : GroupCovarianceBasis or None
        Support basis of a known level covariance; None for independent
        levels.
    cell_axis : str
        Flat axis over levels and spacing coordinates.
    parameterization : {"centered", "noncentered"}
        Coordinates of the raw site.

    Returns
    -------
    GroupBlockLaw
        One member over the spacing axis with an affine mean.
    """
    identity = f"thresholds_{_qvr_name(group)}"
    if basis is not None:
        raw = f"group_covariance_{identity}" if basis.rank else None
    elif parameterization == "centered":
        raw = "cutpoint_log_spacing_value"
    else:
        raw = "cutpoint_log_spacing_noise"
    return GroupBlockLaw(
        identity=identity,
        level=GroupLevelLaw(
            group=group,
            axis=_qvr_name(group),
            size=size,
            covariance=KnownCovariance(basis=basis)
            if basis is not None
            else IdentityCovariance(),
        ),
        members=(
            GroupMemberLaw(
                prefix="thresholds_",
                slope="log_spacing",
                axis=axis,
                width=width,
                offset=0,
                coefficient="cutpoint_log_spacing_matrix",
                raw_site=raw,
                scale=GroupScale(
                    site="cutpoint_log_spacing_scale",
                    component_axis=axis,
                    prior="HalfNormal(0.5)",
                    lower=0.0,
                ),
                rows=SingleLevelRows(index=f"{_qvr_name(group)}_idx"),
                mean=GroupMean(
                    site="cutpoint_log_spacing_mean", prior="Normal(0.0, 1.0)"
                ),
                cell_axis=cell_axis if basis is None else None,
            ),
        ),
        correlation=IndependentCoordinates(),
        parameterization=parameterization,
        placement="inline",
        coordinate_axis=axis,
    )
