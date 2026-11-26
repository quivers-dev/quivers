"""Regression variance diagnostics through retained formula replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import narwhals as nw
import numpy as np
import torch

from quivers.diagnostics.regression import bayesian_r2
from quivers.formulas._response_modifiers import ExpectationOptions

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def _responses(fit: BayesianFit, options: ExpectationOptions) -> torch.Tensor:
    """Evaluate original responses with the retained formula transform."""
    data = options.get("new_data")
    if data is None:
        values = fit.formula.response_values
    else:
        if fit.prediction_plan is None:
            raise ValueError("new-data R-squared requires retained response evaluators")
        frame = nw.from_native(data, eager_only=True).to_pandas()
        response = fit.prediction_plan.designs["mu"].response
        if response is None:
            raise ValueError(
                "new-data R-squared requires a retained response transform"
            )
        term = response.term.term
        try:
            values = np.asarray(term.eval_new_data(frame))
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(
                "R-squared needs new-data response values; supply their columns or observed explicitly"
            ) from error
        if fit.formula.response_layout.category_levels:
            width = len(fit.formula.response_layout.category_levels)
            if (
                values.shape != (len(frame), width)
                or not np.all((values == 0) | (values == 1))
                or not np.all(values.sum(-1) == 1)
            ):
                raise ValueError(
                    "R-squared responses require the fitted category encoding"
                )
            values = values.argmax(-1)
        elif not fit.formula.response_layout.event_shape and values.ndim == 2:
            values = values[:, 0]
    return torch.as_tensor(values.copy())


def formula_r2(
    fit: BayesianFit,
    method: Literal["auto", "model", "residual"],
    observed: torch.Tensor | None,
    options: ExpectationOptions,
) -> torch.Tensor:
    """Reuse prediction bindings, posterior draws, and response distributions."""
    if method not in {"auto", "model", "residual"}:
        raise ValueError("R-squared method must be auto, model, or residual")
    if observed is not None and not isinstance(observed, torch.Tensor):
        raise TypeError("R-squared observed responses must be a real tensor")
    if fit.family.unordered_categories:
        raise ValueError(
            "bayes_R2 is not defined for unordered categorical models, as in "
            "brms: unordered categories have no residual"
        )
    if method == "auto":
        method = (
            (fit.family if fit.residual is None else fit.residual).r_squared
            if observed is None and fit.formula.modifiers.truncation is None
            else "residual"
        )
    if method == "residual":
        if observed is None:
            censoring = fit.formula.modifiers.censoring
            if (
                options.get("new_data") is None
                and censoring is not None
                and bool(np.any(censoring.status.data != 0))
            ):
                raise ValueError(
                    "residual R-squared needs actual responses, not censoring bounds; "
                    "supply observed or use a model-variance diagnostic"
                )
            observed = _responses(fit, options)
        expected = fit.posterior_epred(**options)
        if fit.family.response_ordinal() is not None:
            codes = torch.arange(
                expected.shape[-1], device=expected.device, dtype=expected.dtype
            )
            expected = (expected * codes).sum(-1)
        return bayesian_r2(expected, observed=observed.to(expected.device))
    from quivers.formulas._prediction import prepare_prediction
    from quivers.formulas._posterior import (
        collect_draws,
        replay,
        response_distribution,
        row_binding,
    )
    from quivers.formulas.formula import _qvr_name
    from quivers.inference._posterior import draw_count

    if observed is not None:
        raise ValueError("observed responses are used by method='residual'")
    if options.get("mc_samples") is not None:
        raise ValueError(
            "model R-squared uses analytic moments; mc_samples applies to residual means"
        )
    context = prepare_prediction(
        fit,
        options.get("new_data"),
        options.get("predictor_data"),
        re_formula=options.get("re_formula"),
        allow_new_levels=options.get("allow_new_levels", False),
        sample_new_levels=options.get("sample_new_levels", "uncertainty"),
        seed=options.get("seed", 0),
        trials=options.get("trials"),
    )
    shape = draw_count(fit.posterior, options.get("num_samples"))
    draws = fit.draws(
        num_samples=options.get("num_samples"), seed=options.get("seed", 0)
    )
    response = _qvr_name(fit.formula.response_name)
    event = fit.formula.response_layout.event_shape

    def moments():
        for recorded in replay(fit, draws, shape, context=context):
            law = response_distribution(recorded, response)
            try:
                if fit.family.decision_event is not None:
                    from quivers.continuous._wiener import Wiener

                    assert isinstance(law, Wiener)
                    mean, variance = law.mean_response_time, law.variance_response_time
                elif fit.family.response_ordinal() is not None:
                    from quivers.formulas._posterior import category_probabilities

                    probs = category_probabilities(law, len(fit.category_labels))
                    codes = torch.arange(
                        probs.shape[-1], device=probs.device, dtype=probs.dtype
                    )
                    mean = (probs * codes).sum(-1)
                    variance = (probs * (codes - mean.unsqueeze(-1)).square()).sum(-1)
                else:
                    mean, variance = law.mean, law.variance
            except NotImplementedError as error:
                raise ValueError(
                    "model R-squared requires analytic conditional response moments; use method='residual'"
                ) from error
            yield torch.stack(
                (
                    row_binding(mean, context.rows, event),
                    row_binding(variance, context.rows, event),
                ),
                dim=-1,
            )

    paired = collect_draws(moments(), shape)
    return bayesian_r2(paired[..., 0], residual_variance=paired[..., 1])
