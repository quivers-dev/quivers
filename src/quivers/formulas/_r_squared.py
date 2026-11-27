"""Regression variance diagnostics through retained formula replay."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from typing import TYPE_CHECKING, Literal

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame
import torch
from torch.distributions import Distribution

from quivers.continuous._wiener import Wiener
from quivers.diagnostics.regression import bayesian_r2
from quivers.formulas._posterior import (
    category_probabilities,
    collect_draws,
    row_binding,
)
from quivers.formulas._response import ResponseReadout
from quivers.formulas._response_modifiers import ExpectationOptions

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def _responses(fit: BayesianFit, options: ExpectationOptions) -> torch.Tensor:
    """Evaluate original responses with the retained formula transform."""
    data = options.get("new_data")
    fitted_response = fit._response_data
    if data is None:
        values = fitted_response.values
    else:
        if fit.prediction_plan is None:
            raise ValueError("new-data R-squared requires retained response evaluators")
        frame = nw.from_native(data, eager_only=True).to_pandas()
        response = fitted_response.evaluator
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
        layout = fitted_response.layout
        if layout.category_levels:
            width = len(layout.category_levels)
            if (
                values.shape != (len(frame), width)
                or not np.all((values == 0) | (values == 1))
                or not np.all(values.sum(-1) == 1)
            ):
                raise ValueError(
                    "R-squared responses require the fitted category encoding"
                )
            values = values.argmax(-1)
        elif not layout.event_shape and values.ndim == 2:
            values = values[:, 0]
    return torch.as_tensor(values.copy())


type ResponseLaws = Callable[
    [], tuple[Generator[Distribution, None, None], tuple[int, int], int]
]
"""Replay a response's law once per draw, with the chain and draw counts and
the number of rows."""


def law_moments(
    readout: ResponseReadout, law: Distribution
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the conditional mean and variance of a replayed response law.

    Parameters
    ----------
    readout : ResponseReadout
        The measurements exposed by the response law.
    law : torch.distributions.Distribution
        The response law at one draw.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean and variance on the row axes: of the response time for a
        decision family, of the category code for an ordinal family.

    Raises
    ------
    ValueError
        If the law has no analytic moments.
    """
    try:
        if readout.decision:
            assert isinstance(law, Wiener)
            return law.mean_response_time, law.variance_response_time
        if readout.ordinal:
            probs = category_probabilities(law, readout.categories)
            codes = torch.arange(
                probs.shape[-1], device=probs.device, dtype=probs.dtype
            )
            mean = (probs * codes).sum(-1)
            return mean, (probs * (codes - mean.unsqueeze(-1)).square()).sum(-1)
        return law.mean, law.variance
    except NotImplementedError as error:
        raise ValueError(
            "model R-squared requires analytic conditional response moments; use method='residual'"
        ) from error


def formula_r2(
    fit: BayesianFit,
    method: Literal["auto", "model", "residual"],
    observed: torch.Tensor | None,
    *,
    new_data: IntoDataFrame | None,
    mc_samples: int | None,
    expected: Callable[[], torch.Tensor],
    laws: ResponseLaws,
) -> torch.Tensor:
    """Compute one response's Bayesian R-squared draws.

    The residual form divides the variance of the conditional means over the
    rows by itself plus the variance of the residuals, observed minus mean
    [@gelman2019rsquared]. The model form replaces the residual variance by
    the mean of each row's conditional variance. The response decides the
    default and the observations; the fit replays them.

    Parameters
    ----------
    fit : BayesianFit
        The response: its family, residual law, formula, and recorded events.
    method : {"auto", "model", "residual"}
        Requested form.
    observed : torch.Tensor or None
        Explicit observed responses for the residual form.
    new_data : IntoDataFrame or None
        Rows of the request, or None for the training rows.
    mc_samples : int or None
        Monte Carlo budget of the conditional means, refused by the model
        form.
    expected : Callable[[], torch.Tensor]
        Conditional means on ``(chain, draw, row)``.
    laws : ResponseLaws
        The response law at each draw, for the model form.

    Returns
    -------
    torch.Tensor
        Ratios on ``(chain, draw)``.

    Raises
    ------
    ValueError
        If the form is unknown or undefined for the response, or the
        residual form has only censoring bounds to compare with.
    TypeError
        If ``observed`` is not a tensor.
    """
    if method not in {"auto", "model", "residual"}:
        raise ValueError("R-squared method must be auto, model, or residual")
    if observed is not None and not isinstance(observed, torch.Tensor):
        raise TypeError("R-squared observed responses must be a real tensor")
    readout = fit._response_readout
    fitted_response = fit._response_data
    if readout.unordered:
        raise ValueError(
            "bayes_R2 is not defined for unordered categorical models, as in "
            "brms: unordered categories have no residual"
        )
    if method == "auto":
        method = (
            (readout if fit.residual is None else fit.residual).r_squared
            if observed is None and fitted_response.modifiers.truncation is None
            else "residual"
        )
    if method == "residual":
        if observed is None:
            censoring = fitted_response.modifiers.censoring
            if (
                new_data is None
                and censoring is not None
                and bool(np.any(censoring.status.data != 0))
            ):
                raise ValueError(
                    "residual R-squared needs actual responses, not censoring bounds; "
                    "supply observed or use a model-variance diagnostic"
                )
            observed = _responses(fit, ExpectationOptions(new_data=new_data))
        means = expected()
        if readout.ordinal:
            codes = torch.arange(
                means.shape[-1], device=means.device, dtype=means.dtype
            )
            means = (means * codes).sum(-1)
        return bayesian_r2(means, observed=observed.to(means.device))
    if observed is not None:
        raise ValueError("observed responses are used by method='residual'")
    if mc_samples is not None:
        raise ValueError(
            "model R-squared uses analytic moments; mc_samples applies to residual means"
        )
    event = fitted_response.layout.event_shape
    replayed, shape, rows = laws()

    def moments() -> Iterator[torch.Tensor]:
        # The replay is closed where it runs, so its gradient mode is restored
        # in the scope it was entered from, also when a draw's law has no
        # analytic moments.
        try:
            for law in replayed:
                mean, variance = law_moments(readout, law)
                yield torch.stack(
                    (
                        row_binding(mean, rows, event),
                        row_binding(variance, rows, event),
                    ),
                    dim=-1,
                )
        finally:
            replayed.close()

    paired = collect_draws(moments(), shape)
    return bayesian_r2(paired[..., 0], residual_variance=paired[..., 1])
