"""brms ``default_prior``, ``validate_prior`` and ``prior_summary`` as typed tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict, Unpack

from narwhals.typing import IntoDataFrame

from quivers.formulas import _joint_fit
from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._brms_priors import (
    DefaultPriorPolicy,
    PriorTable,
    prior_table,
    user_selection,
)
from quivers.formulas._data_options import Knots, formula_data_options
from quivers.formulas._fit import BayesianFit, _formula_lens, _refuse_joint_syntax
from quivers.formulas._joint_fit import MultivariateFit
from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._multivariate import MultivariateFormula, ResponseModel
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas._priors import PriorInput, normalize_priors
from quivers.formulas.family import Family


class PriorTableOptions(TypedDict, total=False):
    """Keywords of `default_prior` and `validate_prior`.

    Each key has the meaning of the `get_prior` keyword of the same name.
    """

    family: str | Family
    weights: str | None
    decision: str | None
    data2: Mapping[str, AutocorMatrix] | None
    distributional: Mapping[str, str | NonlinearExpression] | None
    nonlinear: Mapping[str, str | NonlinearExpression] | None
    fixed_parameters: Mapping[str, float] | None
    fixed_prior: str
    random_scale_prior: str
    correlation_prior: str
    group_covariances: Mapping[str, GroupCovariance] | None
    reparameterize: Literal["centered", "noncentered"]
    binomial_trials: int | str | None
    thresholds_by: str | None
    mixture_components: int
    predictor_name: str | None
    default_priors: DefaultPriorPolicy
    center: bool | None
    decomp: Literal["none", "QR"]
    knots: Knots | None
    drop_unused_levels: bool


def _table(
    formula: str | ResponseModel | MultivariateFormula,
    data: IntoDataFrame,
    priors: PriorInput | None,
    options: PriorTableOptions,
) -> PriorTable:
    """Emit the model once and tabulate its effective priors.

    Parameters
    ----------
    formula : str or ResponseModel or MultivariateFormula
        Single-response formula or joint specification.
    data : IntoDataFrame
        Training rows.
    priors : PriorInput or None
        User rules.
    options : PriorTableOptions
        Ordinary preparation keywords.

    Returns
    -------
    PriorTable
        One row per overrideable site.

    Raises
    ------
    ValueError
        If a rule is unknown, unused, ambiguous, or invalid for its target.
    """
    rules = normalize_priors(priors)
    policy = options.get("default_priors", "quivers")
    if isinstance(formula, MultivariateFormula):
        # brms knots and drop_unused_levels reach every response's design.
        with formula_data_options(
            options.get("knots"), options.get("drop_unused_levels", True)
        ):
            module, targets = _joint_fit.joint_module(
                formula.with_default_priors(policy),
                data,
                fixed_prior=options.get("fixed_prior", "Normal(0.0, 5.0)"),
                random_scale_prior=options.get("random_scale_prior", "HalfNormal(1.0)"),
                correlation_prior=options.get("correlation_prior", "LKJCholesky(1.0)"),
                group_covariances=options.get("group_covariances"),
                priors=rules,
                reparameterize=options.get("reparameterize", "noncentered"),
                data2=options.get("data2"),
                caller="default_prior",
            )
        return prior_table(
            module, targets, user_selection(rules, targets), policy=policy
        )
    _refuse_joint_syntax(formula, "default_prior")
    lens, parsed = _formula_lens(formula, data=data, priors=rules, **options)
    module, _ = lens.forward(parsed)
    targets = lens.prior_targets
    return prior_table(module, targets, user_selection(rules, targets), policy=policy)


def default_prior(
    formula: str | ResponseModel | MultivariateFormula,
    *,
    data: IntoDataFrame,
    **options: Unpack[PriorTableOptions],
) -> PriorTable:
    """Tabulate the default prior of every parameter, as ``brms::default_prior``.

    Parameters
    ----------
    formula : str or ResponseModel or MultivariateFormula
        brms-style formula or joint specification.
    data : IntoDataFrame
        Rows that scale data-dependent defaults.
    **options : PriorTableOptions
        Preparation keywords of `get_prior`, including ``default_priors``,
        ``center``, and ``decomp``.

    Returns
    -------
    PriorTable
        brms's columns (``prior, class, coef, group, resp, dpar, nlpar, lb,
        ub, tag, source``) plus the generated site, every source ``default``.

    Raises
    ------
    ValueError
        If the formula or a preparation keyword is invalid.
    """
    return _table(formula, data, None, options)


def validate_prior(
    priors: PriorInput,
    formula: str | ResponseModel | MultivariateFormula,
    *,
    data: IntoDataFrame,
    **options: Unpack[PriorTableOptions],
) -> PriorTable:
    """Check user priors against a model and return the complete prior table.

    Parameters
    ----------
    priors : PriorInput
        User rules: typed `PriorSpec` values or exact site names.
    formula : str or ResponseModel or MultivariateFormula
        brms-style formula or joint specification.
    data : IntoDataFrame
        Training rows.
    **options : PriorTableOptions
        Preparation keywords of `get_prior`.

    Returns
    -------
    PriorTable
        Every site with its effective prior; rows a user rule sets have
        source ``user`` and carry the rule's tag.

    Raises
    ------
    ValueError
        If a rule matches no parameter, is completely shadowed, conflicts
        with another rule, or is invalid for its target (bounds outside the
        domain, a shrinkage prior on an unsupported class, a constant of the
        wrong width).
    """
    return _table(formula, data, priors, options)


def prior_summary(fit: BayesianFit | MultivariateFit) -> PriorTable:
    """Tabulate the priors a fitted model used, as brms ``prior_summary``.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted single-response or joint model.

    Returns
    -------
    PriorTable
        The effective prior of every site; rows set by a rule given to
        `fit` have source ``user``.

    Raises
    ------
    ValueError
        If a single-response fit retains no emission.
    """
    if isinstance(fit, BayesianFit):
        return fit.prior_summary()
    targets = fit.emission.prior_targets
    policies = {
        response.default_priors for response in fit.specification.responses.values()
    }
    return prior_table(
        fit.emission.module,
        targets,
        user_selection(fit.priors, targets),
        policy="brms" if policies == {"brms"} else "quivers",
    )


__all__ = ["PriorTableOptions", "default_prior", "prior_summary", "validate_prior"]
