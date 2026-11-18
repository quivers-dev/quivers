"""Emit and bind brms multi-membership (``mm()``) group-level contributions.

The ordinary group emitter gathers a level-indexed coefficient ``r`` at each
row's level, ``r[J[n]]``, and multiplies it by the row's covariate. For a
multi-membership factor the gather is replaced by

    sum_k r[J_k[n]] * C_k[n],    C_k = W_k * Z_k,

one gather per member ``k`` with its own index input ``J_k`` and the product
``C_k`` of its weight and covariate. This costs O(rows times members) per
coefficient and keeps the group coordinates, correlation blocks, priors, and
new-level policies of the shared Gaussian group layout unchanged.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import TYPE_CHECKING

from formulae.terms.terms import Term
import numpy as np
import pandas as pd
import torch

from quivers.dsl.ast_nodes import (
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.formulas._group_law import GroupLevelLaw
from quivers.formulas._grouping import (
    GroupLevelOptions,
    Membership,
    member_codes,
    member_weights,
)
from quivers.formulas.formula import Formula, RandomTerm, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas._group_prediction import GroupRemap

__all__: list[str] = []


def joint_group_blocks(
    predictors: Iterable[tuple[str, str, Formula]],
    covariances: Collection[str],
    supplied: Collection[tuple[str, str, str]] = (),
) -> dict[tuple[str, str], list[tuple[str, RandomTerm]]]:
    """Collect the group terms that share one joint coefficient distribution.

    A term joins a block through a correlation identifier, or stands as a
    block of its own when its grouping factor has a known level covariance.
    Other terms are independent and belong to no block. Runs in time linear
    in the number of group terms.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Predicted parameters as (name, coefficient prefix, design).
    covariances : Collection[str]
        Grouping factors with a known level covariance.
    supplied : Collection[tuple[str, str, str]]
        Members, as (prefix, group, slope), whose block is already emitted.

    Returns
    -------
    dict[tuple[str, str], list[tuple[str, RandomTerm]]]
        Members of each block as (prefix, term), keyed by grouping factor and
        block identity, in first-appearance order.
    """
    blocks: dict[tuple[str, str], list[tuple[str, RandomTerm]]] = {}
    for _, prefix, predictor in predictors:
        for term in predictor.random_terms:
            if (prefix, term.group, term.slope) in supplied:
                continue
            if term.correlation_id is None:
                if term.group not in covariances:
                    continue
                identity = (
                    prefix
                    + f"independent_{_qvr_name(term.group)}_{_qvr_name(term.slope)}"
                )
            else:
                identity = (
                    f"shared_{term.correlation_id}"
                    if term.shared
                    else prefix + term.correlation_id
                )
            blocks.setdefault((term.group, identity), []).append((prefix, term))
    return blocks


def member_index(group: str, member: int) -> str:
    """Name the row-level code input of one member.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    member : int
        Zero-based member position.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{_qvr_name(group)}_idx_member{member + 1}"


def member_input(prefix: str, group: str, slope: str, member: int) -> str:
    """Name the weighted covariate input of one member and coefficient.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    group : str
        Grouping-factor name.
    slope : str
        Coefficient label.
    member : int
        Zero-based member position.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{prefix}{_qvr_name(group)}_{_qvr_name(slope)}_member{member + 1}"


def _tensor(element: str, *axes: str) -> QiecTypeApplication:
    """Declare a tensor over named finite axes.

    Parameters
    ----------
    element : str
        Element carrier.
    *axes : str
        Axis names.

    Returns
    -------
    QiecTypeApplication
        The checked tensor type.
    """
    return QiecTypeApplication(
        constructor="Tensor",
        static_arguments=(QiecTypeName(name=element),),
        indices=(
            QiecShapeIndex(dimensions=tuple(QiecIndexName(name=axis) for axis in axes)),
        ),
    )


def membership_runtime_parameters(
    predictors: Iterable[tuple[str, str, Formula]],
) -> tuple[QiecValueParameter, ...]:
    """Declare member codes and weighted covariates of every predictor.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        Integer row codes per member and Real row covariates per member and
        coefficient.
    """
    parameters: dict[str, QiecValueParameter] = {}
    for _, prefix, formula in predictors:
        for term in formula.random_terms:
            membership = formula.grouping.membership(term.group)
            if membership is None:
                continue
            for k in range(membership.size):
                name = member_index(term.group, k)
                parameters.setdefault(
                    name,
                    QiecValueParameter(name=name, type_expr=_tensor("Int", "Resp")),
                )
                name = member_input(prefix, term.group, term.slope, k)
                parameters.setdefault(
                    name,
                    QiecValueParameter(name=name, type_expr=_tensor("Real", "Resp")),
                )
    return tuple(parameters.values())


def _covariate_inputs(
    prefix: str,
    term: RandomTerm,
    membership: Membership,
    weights: np.ndarray,
    values: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Bind ``C_k = W_k * Z_k`` for one coefficient.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : RandomTerm
        Group-level coefficient.
    membership : Membership
        Its factor.
    weights, values : numpy.ndarray
        Weights and covariates, shape ``(rows, members)``.

    Returns
    -------
    dict[str, torch.Tensor]
        One Double row vector per member.
    """
    product = np.asarray(weights, dtype=np.float64) * np.asarray(
        values, dtype=np.float64
    )
    return {
        member_input(prefix, term.group, term.slope, k): torch.as_tensor(
            product[:, k].copy()
        )
        for k in range(membership.size)
    }


def membership_observations(formula: Formula, prefix: str) -> dict[str, torch.Tensor]:
    """Bind training member codes and weighted covariates.

    Parameters
    ----------
    formula : Formula
        One predictor's evaluated terms.
    prefix : str
        Predictor namespace.

    Returns
    -------
    dict[str, torch.Tensor]
        Runtime bindings.

    Raises
    ------
    ValueError
        If a multi-membership coefficient has no retained covariate.
    """
    values: dict[str, torch.Tensor] = {}
    for term in formula.random_terms:
        membership = formula.grouping.membership(term.group)
        if membership is None:
            continue
        covariate = formula.grouping.covariate(term.group, term.slope)
        if covariate is None:
            raise ValueError(
                f"multi-membership coefficient {term.slope!r} has no member covariate"
            )
        for k, codes in enumerate(membership.codes):
            values[member_index(term.group, k)] = torch.tensor(codes, dtype=torch.long)
        values.update(
            _covariate_inputs(
                prefix, term, membership, membership.weights, covariate.values
            )
        )
    return values


def membership_prediction_bindings(
    formula: Formula,
    design_terms: Mapping[str, Term],
    frame: pd.DataFrame,
    prefix: str,
    groups: Mapping[str, GroupRemap] | None,
) -> dict[str, torch.Tensor]:
    """Bind fresh member codes and weighted covariates.

    Parameters
    ----------
    formula : Formula
        Fitted predictor.
    design_terms : Mapping[str, Term]
        Retained formulae group-level expressions by term name.
    frame : pandas.DataFrame
        Fresh rows.
    prefix : str
        Predictor namespace.
    groups : Mapping[str, GroupRemap] or None
        Compact prediction axes with member codes, or None to code fresh
        members against the fitted levels.

    Returns
    -------
    dict[str, torch.Tensor]
        Runtime bindings.

    Raises
    ------
    ValueError
        If members, weights, or covariates are invalid.
    """
    values: dict[str, torch.Tensor] = {}
    rows = len(frame)
    for term in formula.random_terms:
        membership = formula.grouping.membership(term.group)
        if membership is None:
            continue
        spec = membership.spec
        if groups is not None:
            codes = tuple(
                tuple(int(code) for code in indices.tolist())
                for indices in groups[term.group].member_indices
            )
        else:
            levels = formula.group_levels[term.group]
            codes = member_codes(
                frame, spec, {label: index for index, label in enumerate(levels)}
            )
        for k, member in enumerate(codes):
            values[member_index(term.group, k)] = torch.tensor(member, dtype=torch.long)
        covariate = formula.grouping.covariate(term.group, term.slope)
        assert covariate is not None
        if covariate.term == "":
            columns = np.ones((rows, membership.size))
        else:
            evaluated = np.asarray(
                design_terms[covariate.term].eval_new_data(frame),
                dtype=np.float64,
            ).reshape(rows, -1)
            if covariate.per_member:
                columns = evaluated
            else:
                # The slope's position among its term's exploded columns.
                assert term.column is not None
                siblings = [
                    other.column.name
                    for other in formula.random_terms
                    if other.group == term.group
                    and other.column is not None
                    and other.column.term == covariate.term
                ]
                position = siblings.index(term.column.name)
                if evaluated.shape[1] != len(siblings):
                    raise ValueError(
                        f"new_data changed the design width of {covariate.term!r}"
                    )
                columns = np.repeat(
                    evaluated[:, position : position + 1], membership.size, axis=1
                )
            if columns.shape[1] != membership.size or not np.isfinite(columns).all():
                raise ValueError(
                    f"new_data must give finite member covariates for {covariate.term!r}"
                )
        values.update(
            _covariate_inputs(
                prefix, term, membership, member_weights(frame, spec), columns
            )
        )
    return values


def by_axis(group: str) -> str:
    """Name the finite axis of a grouping factor's ``by`` levels.

    Parameters
    ----------
    group : str
        Grouping-factor name.

    Returns
    -------
    str
        Axis name.
    """
    return f"FormulaBy_{_qvr_name(group)}"


def by_input(group: str) -> str:
    """Name the input holding each grouping level's ``by`` level.

    Parameters
    ----------
    group : str
        Grouping-factor name.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{_qvr_name(group)}_by"


def weight_input(group: str) -> str:
    """Name the input holding each grouping level's prior weight.

    Parameters
    ----------
    group : str
        Grouping-factor name.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{_qvr_name(group)}_prior_weight"


def student_sites(group: str) -> tuple[str, str, str]:
    """Name the Student-t degrees of freedom, level mixing scales, and multiplier.

    Parameters
    ----------
    group : str
        Grouping-factor name.

    Returns
    -------
    tuple[str, str, str]
        brms's ``df``, ``udf``, and ``dfm``.
    """
    name = _qvr_name(group)
    return f"df_{name}", f"udf_{name}", f"dfm_{name}"


def all_group_options(
    predictors: Iterable[tuple[str, str, Formula]],
) -> dict[str, GroupLevelOptions]:
    """Merge the ``gr()`` options of every predictor by grouping factor.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Returns
    -------
    dict[str, GroupLevelOptions]
        Options by grouping factor; a factor that is Student-t in one
        predictor shares its degrees of freedom with every such predictor.

    Raises
    ------
    ValueError
        If predictors give one factor different ``by`` variables.
    """
    merged: dict[str, GroupLevelOptions] = {}
    for _, _, formula in predictors:
        for option in formula.grouping.options:
            existing = merged.get(option.group)
            if existing is None:
                merged[option.group] = option
                continue
            if (
                option.by_levels
                and existing.by_levels
                and (option.by_levels, option.by_codes)
                != (existing.by_levels, existing.by_codes)
            ):
                raise ValueError(
                    "Each grouping factor can only be associated with one 'by' "
                    f"variable; {option.group!r} has several"
                )
            merged[option.group] = existing.with_(
                options=existing.options.with_(
                    student=existing.options.student or option.options.student
                ),
                by_levels=existing.by_levels or option.by_levels,
                by_codes=existing.by_codes or option.by_codes,
                prior_weights=existing.prior_weights or option.prior_weights,
            )
    return merged


def option_runtime_parameters(
    predictors: Iterable[tuple[str, str, Formula]],
) -> tuple[QiecValueParameter, ...]:
    """Declare per-level ``by`` codes and prior weights.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        Integer ``by`` codes and Real weights over the grouping axis.
    """
    parameters = []
    for group, option in all_group_options(predictors).items():
        if option.by_levels:
            parameters.append(
                QiecValueParameter(
                    name=by_input(group), type_expr=_tensor("Int", _qvr_name(group))
                )
            )
        if option.prior_weights:
            parameters.append(
                QiecValueParameter(
                    name=weight_input(group),
                    type_expr=_tensor("Real", _qvr_name(group)),
                )
            )
    return tuple(parameters)


def level_runtime_parameters(
    levels: Iterable[GroupLevelLaw],
) -> tuple[QiecValueParameter, ...]:
    """Declare the ``by`` codes and prior weights the level laws read.

    Parameters
    ----------
    levels : Iterable[GroupLevelLaw]
        Level factors of emitted blocks.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        One Integer ``by`` input and one Real weight input per level axis
        that declares the factor.
    """
    parameters: dict[str, QiecValueParameter] = {}
    for level in levels:
        if level.by is not None:
            parameters.setdefault(
                level.by.input,
                QiecValueParameter(
                    name=level.by.input, type_expr=_tensor("Int", level.axis)
                ),
            )
        if level.prior_weight is not None:
            parameters.setdefault(
                level.prior_weight.input,
                QiecValueParameter(
                    name=level.prior_weight.input,
                    type_expr=_tensor("Real", level.axis),
                ),
            )
    return tuple(parameters.values())


def level_observations(levels: Iterable[GroupLevelLaw]) -> dict[str, torch.Tensor]:
    """Bind the training ``by`` codes and prior weights of the level laws.

    Parameters
    ----------
    levels : Iterable[GroupLevelLaw]
        Level factors of emitted blocks.

    Returns
    -------
    dict[str, torch.Tensor]
        Runtime bindings over the training levels.
    """
    values: dict[str, torch.Tensor] = {}
    for level in levels:
        if level.by is not None:
            values[level.by.input] = torch.tensor(level.by.codes, dtype=torch.long)
        if level.prior_weight is not None:
            values[level.prior_weight.input] = torch.tensor(
                level.prior_weight.weights, dtype=torch.float64
            )
    return values


def option_observations(
    predictors: Iterable[tuple[str, str, Formula]],
) -> dict[str, torch.Tensor]:
    """Bind training ``by`` codes and prior weights.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Returns
    -------
    dict[str, torch.Tensor]
        Runtime bindings over the training levels.
    """
    values: dict[str, torch.Tensor] = {}
    for group, option in all_group_options(predictors).items():
        if option.by_levels:
            values[by_input(group)] = torch.tensor(option.by_codes, dtype=torch.long)
        if option.prior_weights:
            values[weight_input(group)] = torch.tensor(
                option.prior_weights, dtype=torch.float64
            )
    return values


def compact_option(
    option: GroupLevelOptions, frame: pd.DataFrame, remap: GroupRemap
) -> GroupLevelOptions:
    """Code one factor's ``gr()`` options on its compact prediction levels.

    A fitted level keeps its training ``by`` level. A new level takes the
    ``by`` level of the fresh rows that name it, which must be a fitted
    ``by`` level. Prior weights only shape the fitted prior, so prediction
    levels carry unit weights.

    Parameters
    ----------
    option : GroupLevelOptions
        Options coded on the fitted levels.
    frame : pandas.DataFrame
        Fresh rows.
    remap : GroupRemap
        Compact prediction axis of the factor.

    Returns
    -------
    GroupLevelOptions
        The options coded on the compact levels.

    Raises
    ------
    ValueError
        If a new level has no, several, or an unfitted ``by`` level.
    """
    group = option.group
    codes: list[int] = []
    if option.by_levels:
        by = option.options.by
        assert by is not None
        position = {label: index for index, label in enumerate(option.by_levels)}
        rows = remap.row_indices.tolist()
        for compact, fitted in enumerate(remap.fitted_indices):
            if fitted >= 0:
                codes.append(option.by_codes[fitted])
                continue
            if by not in frame.columns:
                raise ValueError(f"new_data is missing predictors {[by]}")
            labels = {
                str(value)
                for value, row in zip(frame[by].tolist(), rows, strict=True)
                if row == compact
            }
            if len(labels) != 1 or not labels <= position.keys():
                raise ValueError(
                    f"a new level of {group!r} needs one fitted level of {by!r}, "
                    f"got {sorted(labels)}"
                )
            codes.append(position[labels.pop()])
    return option.with_(
        by_codes=tuple(codes),
        prior_weights=(1.0,) * len(remap.fitted_indices)
        if option.prior_weights
        else (),
    )


def option_prediction_bindings(
    predictors: Iterable[tuple[str, str, Formula]],
    frame: pd.DataFrame,
    groups: Mapping[str, GroupRemap],
) -> dict[str, torch.Tensor]:
    """Bind ``by`` codes and prior weights on the compact prediction levels.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.
    frame : pandas.DataFrame
        Fresh rows.
    groups : Mapping[str, GroupRemap]
        Compact prediction axes.

    Returns
    -------
    dict[str, torch.Tensor]
        Runtime bindings over the compact levels.

    Raises
    ------
    ValueError
        If a new level has no, several, or an unfitted ``by`` level.
    """
    values: dict[str, torch.Tensor] = {}
    for group, option in all_group_options(predictors).items():
        remap = groups.get(group)
        if remap is None:
            continue
        compact = compact_option(option, frame, remap)
        if compact.by_levels:
            values[by_input(group)] = torch.tensor(compact.by_codes, dtype=torch.long)
        if compact.prior_weights:
            values[weight_input(group)] = torch.tensor(
                compact.prior_weights, dtype=torch.float64
            )
    return values


def by_correlation_sites(identity: str, option: GroupLevelOptions) -> tuple[str, ...]:
    """Name the correlation Cholesky factor of each ``by`` level of a block.

    Parameters
    ----------
    identity : str
        Correlation-block identity.
    option : GroupLevelOptions
        The block's grouping options.

    Returns
    -------
    tuple[str, ...]
        One LKJ site per ``by`` level, in level order.
    """
    return tuple(
        f"cor_{_qvr_name(identity)}_by{index + 1}"
        for index in range(len(option.by_levels))
    )
