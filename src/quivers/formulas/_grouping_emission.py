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
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    LetExprBinOp,
    LetExprIndex,
    LetExprLiteral,
    ProgramStep,
    SampleStep,
    ScoreStep,
    Statement,
    TypeName,
    LetExprNode,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.formulas._ast import _add, _call, _factor, _index, _let, _mul, _sub, _var
from quivers.formulas._category_formulas import component_valued_prefixes
from quivers.formulas._grouping import (
    GroupLevelOptions,
    Membership,
    member_codes,
    member_weights,
)
from quivers.formulas.formula import Formula, RandomTerm, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas._group_prediction import GroupRemap
    from quivers.formulas.compile import FormulaToQVRModule

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


def append_group_options(
    lens: FormulaToQVRModule,
    predictors: Iterable[tuple[str, str, Formula]],
    statements: list[Statement],
    steps: list[ProgramStep],
) -> None:
    """Declare ``by`` axes and emit Student-t mixing scales once per factor.

    For ``dist = "student"`` brms draws ``df ~ gamma(2, 0.1)`` restricted to
    ``df > 1``, one ``udf_j ~ inv_chi_square(df)`` per level, and multiplies
    every Gaussian group coefficient of level ``j`` by
    ``dfm_j = sqrt(df * udf_j)``, a scale mixture whose marginal law is the
    multivariate Student-t with ``df`` degrees of freedom.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.
    statements : list[Statement]
        Module declarations receiving the ``by`` axes.
    steps : list[ProgramStep]
        Program steps receiving the Student-t draws.
    """
    predictors = tuple(predictors)
    check_group_options(lens, predictors)
    for group, option in all_group_options(predictors).items():
        if option.by_levels:
            lens._append_finite_object(
                statements, by_axis(group), len(option.by_levels)
            )
        if not option.options.student:
            continue
        df, udf, dfm = student_sites(group)
        qgroup = _qvr_name(group)
        steps.append(lens._prior_draw(df, "Gamma(2.0, 0.1)", lower=1.0))
        steps.append(_let(df + "_half", _mul(_var(df), LetExprLiteral(value=0.5))))
        steps.append(
            SampleStep(
                vars=(udf,),
                morphism="InverseGamma",
                args=(
                    DrawArgNamed(
                        parameter="concentration", value=DrawArgName(text=df + "_half")
                    ),
                    DrawArgNamed(parameter="rate", value=DrawArgScalar(value=0.5)),
                ),
                index=TypeName(name=qgroup),
            )
        )
        steps.append(
            _let(
                dfm,
                _factor(
                    (("student_level", qgroup),),
                    body=_call(
                        "sqrt",
                        _mul(_var(df), _index(udf, _var("student_level"))),
                    ),
                ),
            )
        )


def scale_index(option: GroupLevelOptions | None) -> TypeName | None:
    """Index a group-level standard deviation by the ``by`` levels.

    Parameters
    ----------
    option : GroupLevelOptions or None
        The coefficient's grouping options.

    Returns
    -------
    TypeName or None
        The ``by`` axis, or None for one standard deviation.
    """
    if option is None or not option.by_levels:
        return None
    return TypeName(name=by_axis(option.group))


def level_scale(
    option: GroupLevelOptions | None, sigma: str, level: LetExprNode
) -> LetExprNode:
    """Read a coefficient's standard deviation at one grouping level.

    Parameters
    ----------
    option : GroupLevelOptions or None
        The coefficient's grouping options.
    sigma : str
        Standard-deviation site.
    level : LetExprNode
        Grouping level.

    Returns
    -------
    LetExprNode
        ``sigma``, or ``sigma[by[level]]`` with ``by``.
    """
    if option is None or not option.by_levels:
        return _var(sigma)
    return LetExprIndex(
        array=_var(sigma), indices=(_index(by_input(option.group), level),)
    )


def mixing(option: GroupLevelOptions | None, level: LetExprNode) -> LetExprNode | None:
    """Read the Student-t multiplier ``dfm`` at one grouping level.

    Parameters
    ----------
    option : GroupLevelOptions or None
        The coefficient's grouping options.
    level : LetExprNode
        Grouping level.

    Returns
    -------
    LetExprNode or None
        ``dfm[level]`` for Student-t effects, None otherwise.
    """
    if option is None or not option.options.student:
        return None
    return _index(student_sites(option.group)[2], level)


_LOG_ROOT_TWO_PI = 0.9189385332046727


def prior_weight_score(
    option: GroupLevelOptions | None, name: str, innovations: tuple[str, ...]
) -> ScoreStep | None:
    """Weight the standard-normal prior of each level's innovations, as brms ``pw``.

    brms adds ``PW_j * std_normal_lpdf(z_j)`` per level. The innovations keep
    their unit-weight standard-normal draws, so this factor adds
    ``(PW_j - 1) * std_normal_lpdf(z_j)``.

    Parameters
    ----------
    option : GroupLevelOptions or None
        The block's grouping options.
    name : str
        Binder of the score.
    innovations : tuple[str, ...]
        Standard-normal level-indexed sites of the block.

    Returns
    -------
    ScoreStep or None
        The log-density factor, or None without prior weights.
    """
    if option is None or not option.prior_weights or not innovations:
        return None
    level = _var("weighted_level")
    density = _add(
        *(
            _sub(
                _mul(
                    LetExprLiteral(value=-0.5),
                    _mul(_index(site, level), _index(site, level)),
                ),
                LetExprLiteral(value=_LOG_ROOT_TWO_PI),
            )
            for site in innovations
        )
    )
    value = _call(
        "sum",
        _factor(
            (("weighted_level", _qvr_name(option.group)),),
            body=_mul(
                _sub(
                    _index(weight_input(option.group), level), LetExprLiteral(value=1.0)
                ),
                density,
            ),
        ),
    )
    return ScoreStep(name=name, value=value)


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


def option_prediction_bindings(
    predictors: Iterable[tuple[str, str, Formula]],
    frame: pd.DataFrame,
    groups: Mapping[str, GroupRemap],
) -> dict[str, torch.Tensor]:
    """Bind ``by`` codes and prior weights on the compact prediction levels.

    A fitted level keeps its training ``by`` level. A new level takes the
    ``by`` level of the fresh rows that name it, which must be a fitted
    ``by`` level. Prior weights only shape the fitted prior, so prediction
    levels carry unit weights.

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
        count = len(remap.fitted_indices)
        if option.by_levels:
            by = option.options.by
            assert by is not None
            position = {label: index for index, label in enumerate(option.by_levels)}
            codes = []
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
            values[by_input(group)] = torch.tensor(codes, dtype=torch.long)
        if option.prior_weights:
            values[weight_input(group)] = torch.ones(count, dtype=torch.float64)
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


def by_correlations(
    lens: FormulaToQVRModule,
    option: GroupLevelOptions,
    identity: str,
    dimension: str,
    innovations: str,
    steps: list[ProgramStep],
) -> LetExprNode:
    """Correlate each level's innovations with the factor of its ``by`` level.

    brms's ``scale_r_cor_by`` multiplies the innovations of level ``j`` by
    ``L[by_j]``, one LKJ Cholesky factor per ``by`` level. Each factor has its
    own site and the configured correlation prior; the product costs
    O(levels times by levels times the block dimension squared).

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    option : GroupLevelOptions
        The block's grouping options.
    identity : str
        Correlation-block identity.
    dimension : str
        Finite axis of the block's coordinates.
    innovations : str
        Binding of the coordinate-by-level standard-normal innovations.
    steps : list[ProgramStep]
        Program steps receiving the correlation draws.

    Returns
    -------
    LetExprNode
        The correlated innovations, coordinate by level.

    Raises
    ------
    ValueError
        If the correlation prior is not an LKJ Cholesky distribution.
    """
    qgroup = _qvr_name(option.group)
    level, row, column = _var("random_group"), _var("group_coordinate"), _var("k")
    terms = []
    for index, site in enumerate(by_correlation_sites(identity, option)):
        draw = lens._prior_draw(
            site, lens._correlation_prior, index=TypeName(name=dimension)
        )
        if draw.morphism != "LKJCholesky":
            raise ValueError("correlation priors must use LKJCholesky")
        steps.append(draw)
        product = _call(
            "sum",
            _factor(
                (("k", dimension),),
                body=_mul(
                    _index(site, row, column), _index(innovations, column, level)
                ),
            ),
        )
        terms.append(
            _call(
                "select",
                LetExprBinOp(
                    op="==",
                    left=_index(by_input(option.group), level),
                    right=LetExprLiteral(value=index, integral=True),
                ),
                product,
                LetExprLiteral(value=0.0),
            )
        )
    return _factor(
        (("group_coordinate", dimension), ("random_group", qgroup)), body=_add(*terms)
    )


def check_group_options(
    lens: FormulaToQVRModule, predictors: Iterable[tuple[str, str, Formula]]
) -> None:
    """Refuse ``gr()`` options outside the parameterizations brms defines.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler and its configuration.
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Raises
    ------
    ValueError
        For centered coordinates, a known level covariance, or a predictor
        with one coefficient per response category together with ``by``,
        ``dist = "student"``, or ``pw``.
    """
    predictors = tuple(predictors)
    component_valued = component_valued_prefixes(lens._family, lens._distributional)
    for group in all_group_options(predictors):
        if lens._reparameterize == "centered":
            raise ValueError(
                f"gr({group}) options by, dist, and pw use brms's noncentered "
                "innovations; use reparameterize='noncentered'"
            )
        if group in lens._group_covariances:
            raise ValueError(
                f"gr({group}) options by, dist, and pw are not supported together "
                "with a known level covariance"
            )
        if any(
            option.group == group
            for _, prefix, formula in predictors
            if prefix in component_valued
            for option in formula.grouping.options
        ):
            raise ValueError(
                f"gr({group}) options by, dist, and pw are supported for scalar "
                f"group coefficients, not for {lens._family.name} predictors that "
                "carry one coefficient per response category"
            )
