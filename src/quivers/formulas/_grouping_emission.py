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

from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from quivers.dsl.ast_nodes import (
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
)
from quivers.formulas._grouping import (
    GroupLevelOptions,
    Membership,
)
from quivers.formulas.formula import Formula, RandomTerm, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas._group_prediction import GroupRemap

__all__: list[str] = []


def joint_group_blocks(
    predictors: Iterable[tuple[str, str, Formula]],
    known: Callable[[RandomTerm], bool],
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
    known : Callable[[RandomTerm], bool]
        Whether a term has a known level covariance.
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
                if not known(term):
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


def member_covariates(
    prefix: str,
    group: str,
    slope: str,
    membership: Membership,
    weights: np.ndarray,
    values: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Bind ``C_k = W_k * Z_k`` for one coefficient.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    group : str
        Grouping-factor name.
    slope : str
        Coefficient label.
    membership : Membership
        The factor.
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
        member_input(prefix, group, slope, k): torch.as_tensor(product[:, k].copy())
        for k in range(membership.size)
    }


def by_axis(group: str, variable: str) -> str:
    """Name the finite axis of a ``by`` variable's levels.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    variable : str
        Authored ``by`` variable.

    Returns
    -------
    str
        Axis name.
    """
    return f"FormulaBy_{_qvr_name(group)}_{_qvr_name(variable)}"


def by_input(group: str, variable: str) -> str:
    """Name the input holding each grouping level's ``by`` level.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    variable : str
        Authored ``by`` variable.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{_qvr_name(group)}_by_{_qvr_name(variable)}"


def weight_input(group: str, column: str) -> str:
    """Name the input holding each grouping level's prior weight.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    column : str
        Column of the weights.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{_qvr_name(group)}_prior_weight_{_qvr_name(column)}"


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


def merged_group_option(
    options: Iterable[GroupLevelOptions],
) -> GroupLevelOptions | None:
    """Merge the ``gr()`` options the terms of one block declare.

    The coefficients of a block have one law, so an option any member
    declares applies to the block, as in brms.

    Parameters
    ----------
    options : Iterable[GroupLevelOptions]
        Options of one grouping factor, coded on the same levels.

    Returns
    -------
    GroupLevelOptions or None
        The merged options, or None when no member declares any.

    Raises
    ------
    ValueError
        If members give one block different ``by`` variables or prior
        weights.
    """
    merged: GroupLevelOptions | None = None
    for option in options:
        if merged is None:
            merged = option
            continue
        for label, first, second in (
            (
                "'by' variable",
                (merged.by_levels, merged.by_codes),
                (option.by_levels, option.by_codes),
            ),
            (
                "vector of prior weights",
                (merged.prior_weights,),
                (option.prior_weights,),
            ),
        ):
            if any(first) and any(second) and first != second:
                raise ValueError(
                    "Each block of group-level effects can only be associated "
                    f"with one {label}; a block of {option.group!r} has several"
                )
        merged = GroupLevelOptions(
            options=merged.options.with_(
                by=merged.options.by or option.options.by,
                by_columns=merged.options.by_columns or option.options.by_columns,
                student=merged.options.student or option.options.student,
                prior_weight=merged.options.prior_weight or option.options.prior_weight,
            ),
            by_levels=merged.by_levels or option.by_levels,
            by_codes=merged.by_codes or option.by_codes,
            prior_weights=merged.prior_weights or option.prior_weights,
        )
    return merged


def compact_option(
    option: GroupLevelOptions, frame: pd.DataFrame, remap: GroupRemap
) -> GroupLevelOptions:
    """Code one factor's ``gr()`` options on its compact prediction levels.

    A fitted level keeps its training ``by`` level. A new level takes the
    ``by`` level of the fresh rows that name it, which must be a fitted
    ``by`` level. Prior weights shape only the fitted prior, so prediction
    levels declare none.

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
        position = {label: index for index, label in enumerate(option.by_levels)}
        rows = [
            indices.tolist() for indices in remap.member_indices or (remap.row_indices,)
        ]
        columns = option.options.by_columns
        for compact, fitted in enumerate(remap.fitted_indices):
            if fitted >= 0:
                codes.append(option.by_codes[fitted])
                continue
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"new_data is missing predictors {missing}")
            labels = {
                str(value)
                for column, coded in zip(columns, rows, strict=True)
                for value, row in zip(frame[column].tolist(), coded, strict=True)
                if row == compact
            }
            if len(labels) != 1 or not labels <= position.keys():
                raise ValueError(
                    f"a new level of {group!r} needs one fitted level of "
                    f"{option.options.by!r}, got {sorted(labels)}"
                )
            codes.append(position[labels.pop()])
    return option.with_(by_codes=tuple(codes), prior_weights=())


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
