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
from quivers.formulas._group_law import GroupBlockLaw, law_inputs
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


def merged_group_option(
    options: Iterable[GroupLevelOptions],
) -> GroupLevelOptions | None:
    """Merge the ``gr()`` options the members of one block declare.

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
        If members give one factor different ``by`` variables or prior
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
                    "Each grouping factor can only be associated with one "
                    f"{label}; {option.group!r} has several"
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


def law_runtime_parameters(
    laws: Iterable[GroupBlockLaw],
) -> tuple[QiecValueParameter, ...]:
    """Declare the runtime inputs that the blocks of a program read.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Laws of the blocks.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        One checked tensor per input.
    """
    return tuple(
        QiecValueParameter(name=item.name, type_expr=_tensor(item.element, *item.axes))
        for item in law_inputs(laws)
    )


def law_observations(laws: Iterable[GroupBlockLaw]) -> dict[str, torch.Tensor]:
    """Bind the inputs whose values the laws of a program retain.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Laws of the blocks.

    Returns
    -------
    dict[str, torch.Tensor]
        ``by`` codes, prior weights, and matrices of known level covariances.
        The row data of multi-membership factors is bound with the design.
    """
    return {
        item.name: item.value for item in law_inputs(laws) if item.value is not None
    }


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
