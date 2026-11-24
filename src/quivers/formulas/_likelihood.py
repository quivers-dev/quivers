"""Bind recorded fresh outcomes to the existing prediction and replay paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.typing import IntoDataFrame
import numpy as np
import torch

from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._group_prediction import NewLevelPolicy
from quivers.formulas._prediction import (
    PredictionContext,
    prepare_prediction,
    _set_trial_rows,
)
from quivers.formulas._response import (
    normalize_response_values,
    response_tensor,
    validate_simplex,
)
from quivers.formulas._ordinal_coding import category_positions
from quivers.formulas._response_modifiers import response_input_name
from quivers.formulas._response_roles import (
    bind_roles,
    resolved_threshold_groups,
    validate_event_response,
)
from quivers.formulas.formula import ResponseModifiers

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def prepare_likelihood(
    fit: BayesianFit,
    new_data: IntoDataFrame,
    predictor_data: torch.Tensor | None,
    *,
    trials: int | str | None = None,
    re_formula: str | None = None,
    allow_new_levels: bool = False,
    sample_new_levels: NewLevelPolicy = "uncertainty",
    seed: int = 0,
    incl_autocor: bool = True,
    data2: Mapping[str, AutocorMatrix] | None = None,
) -> PredictionContext:
    """Prepare recorded rows with fitted encodings and raw sampling-law inputs.

    Parameters
    ----------
    fit : BayesianFit
        Fit retaining its response evaluator, event layout, and prediction plan.
    new_data : IntoDataFrame
        Fresh predictors, outcomes, and any authored observation-event columns.
    predictor_data : torch.Tensor or None
        Neural inputs aligned with the original fresh rows, before selection.
    trials : int or str or None
        Explicit trial override; vector counts otherwise infer their row totals.
    re_formula : str or None
        Existing varying-effect selector.
    allow_new_levels : bool
        Existing policy for fresh grouping identities.
    sample_new_levels : NewLevelPolicy
        Existing joint group-effect prediction policy.
    seed : int
        Seed for that policy's separate random stream.
    incl_autocor : bool
        Score with an autocorrelation term's structure (brms ``incl_autocor``).
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        Matrices of ``sar`` and ``fcor`` terms over the rows of ``new_data``
        (brms ``newdata2``).

    Returns
    -------
    PredictionContext
        Prepared program and actual outcomes, ready for the shared trace scorer.

    Raises
    ------
    ValueError
        If rows, recorded outcomes, trials, or observation-event inputs are invalid.
    """
    plan = fit.prediction_plan
    if plan is None:
        raise ValueError(
            "fresh likelihood rows require a retained formula prediction plan"
        )
    frame = nw.from_native(new_data, eager_only=True).to_pandas()
    if len(frame) < 1 or not frame.columns.is_unique:
        raise ValueError("new_data requires nonempty rows and unique column names")
    # Raw likelihoods do not depend on fitting weights, including their columns.
    sources = {
        role: source
        for role, source in fit.formula.modifiers.sources
        if role not in {"weights", "weights_scale"}
    }
    if trials is not None and "trials" in sources:
        sources["trials"] = trials
    frame, bound = bind_roles(
        frame, sources, threshold_groups=fit.formula.modifiers.threshold_groups
    )
    modifiers = ResponseModifiers(**bound)
    if predictor_data is not None and modifiers.subset is not None:
        if predictor_data.shape[:1] != (modifiers.subset.size,):
            raise ValueError(
                "predictor_data must align with fresh rows before subset selection"
            )
        predictor_data = predictor_data[list(modifiers.subset.positions)]
    evaluated = plan.designs["mu"].response
    if evaluated is None:
        raise ValueError("the retained formula has no response evaluator")
    evaluated_values = np.asarray(evaluated.term.term.eval_new_data(frame))
    # Integer categories evaluate to a value vector and not to factor
    # indicators: brms thres() and cat() codes, and brms's integer coding.
    integer_categories = fit.formula.response_layout.category_values
    values = normalize_response_values(
        evaluated_values,
        component_names=fit.formula.response_layout.component_names,
        category_levels=()
        if (fit.formula.modifiers.thresholds is not None or integer_categories)
        and evaluated_values.ndim == 1
        else fit.formula.response_layout.category_levels,
    )
    if integer_categories:
        values = category_positions(values, integer_categories)
    if values.shape[:1] != (len(frame),):
        raise ValueError("fresh response evaluation must retain every included row")
    if not np.isfinite(values).all():
        raise ValueError("fresh responses must be finite")
    validate_event_response(values, modifiers.censoring, modifiers.truncation)
    if modifiers.threshold_groups is not None:
        # brms thres(n, gr): a fresh response lies among its group's categories.
        resolved_threshold_groups(values, modifiers.threshold_groups)
    # This immutable view shares fitted encodings; it does not reevaluate designs.
    response = fit.formula.with_(response_values=values, modifiers=modifiers)
    if (
        response.response_layout.event_shape
        and fit.family.vector_response
        and fit.family.trials_parameter is None
    ):
        validate_simplex(response)
    outcome = response_tensor(response, fit.family)
    if not torch.isfinite(outcome).all():
        raise ValueError("fresh responses must fit finite native response carriers")
    counts: torch.Tensor | None = None
    specification: int | str | None = None
    if fit.family.trials_parameter is not None:
        from quivers.formulas._fit import _response_trials

        retained = sources.get("trials")
        specification = (
            trials
            if trials is not None
            else "formula_trials"
            if modifiers.trials is not None
            else fit.binomial_trials
        )
        # A source trials modifier binds the existing compiler-owned alias.
        counts = _response_trials(
            response, frame, specification, caller="log_likelihood"
        )
        prediction_trials = trials if trials is not None else retained
        if not isinstance(prediction_trials, (str, int)) or isinstance(
            prediction_trials, bool
        ):
            prediction_trials = None
    else:
        prediction_trials = trials
    context = prepare_prediction(
        fit,
        frame,
        predictor_data,
        re_formula=re_formula,
        allow_new_levels=allow_new_levels,
        sample_new_levels=sample_new_levels,
        seed=seed,
        trials=prediction_trials,
        require_trials=False,
        incl_autocor=incl_autocor,
        data2=data2,
    )
    observations = dict(context.observations)
    if counts is not None:
        _set_trial_rows(fit, observations, counts, specification=specification)
    observations.update(
        {
            column.qvr_name: torch.as_tensor(
                column.data.copy(), dtype=torch.bool if role == "decision" else None
            )
            for role, column in modifiers.columns()
            if role != "trials"
        }
    )
    observations[response_input_name(fit.formula)] = outcome
    return context.with_(observations=observations)
