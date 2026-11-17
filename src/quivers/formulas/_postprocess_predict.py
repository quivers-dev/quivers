"""brms ``predict``, ``fitted``, ``residuals`` and ``predictive_error`` helpers.

brms wraps its draw-level prediction functions in summary forms
[@burkner2017brms]: ``predict`` summarizes ``posterior_predict``, ``fitted``
summarizes ``posterior_epred`` (``scale = "response"``) or
``posterior_linpred`` (``scale = "linear"``), and ``residuals`` summarizes
``predictive_error``, the draws of observed minus predicted responses. The
helpers here hold the parts shared by single-response and joint fits: the
summary of a draw tensor, the residual arithmetic, and the selection of
explicit posterior draws (brms ``draw_ids``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from quivers.diagnostics.posterior_tables import (
    RowLabel,
    SummaryTable,
    posterior_summary,
    posterior_table,
)
from quivers.formulas._fit_types import Fit
from quivers.inference._posterior import constrained_draws
from quivers.inference.prior import PriorDraws
from quivers.inference.registry import LatentRegistry

type PredictionMethod = Literal[
    "posterior_predict", "posterior_epred", "posterior_linpred"
]
"""Draw-level prediction function behind a residual or k-fold prediction."""

#: brms's aliases of prediction methods (R/posterior_predict.R ``validate_pp_method``).
_METHOD_ALIASES: dict[str, PredictionMethod] = {
    "posterior_predict": "posterior_predict",
    "predict": "posterior_predict",
    "pp": "posterior_predict",
    "posterior_epred": "posterior_epred",
    "fitted": "posterior_epred",
    "pp_expect": "posterior_epred",
    "posterior_linpred": "posterior_linpred",
}


def prediction_method(method: str) -> PredictionMethod:
    """Validate a prediction method name, accepting brms's aliases.

    Parameters
    ----------
    method : str
        ``posterior_predict`` (or ``predict``, ``pp``), ``posterior_epred``
        (or ``fitted``, ``pp_expect``), or ``posterior_linpred``.

    Returns
    -------
    PredictionMethod
        The canonical method name.

    Raises
    ------
    ValueError
        If the method is unknown.
    """
    if method not in _METHOD_ALIASES:
        raise ValueError(
            f"posterior predictive method {method!r} is not supported; expected one "
            f"of {sorted(_METHOD_ALIASES)}"
        )
    return _METHOD_ALIASES[method]


def summarize_prediction(
    draws: torch.Tensor,
    *,
    categories: Sequence[RowLabel],
    discrete_table: bool,
    probs: Sequence[float],
    robust: bool,
) -> SummaryTable:
    """Summarize prediction draws as brms's summary forms do.

    Parameters
    ----------
    draws : torch.Tensor
        Draws ``(chain, draw, row)`` or ``(chain, draw, row, component)``.
    categories : Sequence[str or int]
        Category labels of an ordinal or categorical response, else empty.
    discrete_table : bool
        Tabulate category frequencies (``predict`` of an ordinal or
        categorical response) instead of summarizing values.
    probs : Sequence[float]
        Quantile probabilities.
    robust : bool
        Median and MAD instead of mean and SD.

    Returns
    -------
    SummaryTable
        ``posterior_table`` frequencies ``P(Y = <category>)`` for discrete
        tables, otherwise ``posterior_summary`` statistics, three-way with a
        component axis for vector-valued draws.
    """
    if discrete_table:
        return posterior_table(draws, levels=[str(label) for label in categories])
    return posterior_summary(
        draws,
        probs=probs,
        robust=robust,
        components=tuple(categories)
        if draws.ndim == 4 and len(categories) == draws.shape[-1]
        else None,
    )


def predictive_errors(
    observed: torch.Tensor,
    predicted: torch.Tensor,
    *,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    """Return observed minus predicted draws, optionally Pearson-scaled.

    Parameters
    ----------
    observed : torch.Tensor
        Observed responses on row (and event) axes.
    predicted : torch.Tensor
        Prediction draws ``(chain, draw, row[, event])``.
    scale : torch.Tensor or None
        Per-row predictive standard deviations for Pearson residuals
        ``(y - yrep) / SD(yrep)``; None for ordinary residuals ``y - yrep``.

    Returns
    -------
    torch.Tensor
        Float64 residual draws with the prediction's shape.

    Raises
    ------
    ValueError
        If observed values do not align with the prediction rows.
    """
    target = observed.detach().to(torch.float64)
    values = predicted.detach().to(torch.float64)
    if tuple(target.shape) != tuple(values.shape[2:]):
        raise ValueError(
            f"observed responses have shape {tuple(target.shape)}; predictions "
            f"have row shape {tuple(values.shape[2:])}"
        )
    errors = target - values
    if scale is not None:
        errors = errors / scale.to(torch.float64)
    return errors


def latent_draws(
    fit: Fit, *, num_samples: int | None, seed: int
) -> dict[str, torch.Tensor]:
    """Return constrained draws of the sampled latent sites alone.

    `BayesianFit.draws` and `MultivariateFit.draws` add derived sites (a
    reported intercept under centering, QR coefficients, constants, equated
    mixture parameters). A posterior that replays the program holds only the
    sites the program samples, so a view of selected draws starts here.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    num_samples : int or None
        Draws per chain, or fresh guide draws.
    seed : int
        Seed for variational draws, the stream `draws` uses.

    Returns
    -------
    dict[str, torch.Tensor]
        Latent site draws on ``(chain, draw, *event)`` axes.

    Raises
    ------
    ValueError
        If posterior sites or shapes do not match the fitted program.
    """
    registry = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    return constrained_draws(
        fit.posterior,
        torch.zeros(1, 1, dtype=torch.long),
        num_samples=num_samples,
        seed=seed,
        registry=registry,
    )


def selected_draws(
    draws: Mapping[str, torch.Tensor], draw_ids: Sequence[int]
) -> PriorDraws:
    """Select explicit pooled posterior draws as one independent layout axis.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained draws ``(chain, draw, *event)`` of the latent sites, as
        `latent_draws` returns them.
    draw_ids : Sequence[int]
        0-based positions in the chain-major pooled draw order, the order of
        brms's ``draw_ids`` shifted to 0-based indexing.

    Returns
    -------
    PriorDraws
        The selected draws with axes ``(1, len(draw_ids), *event)``; a
        container for replay, not a claim that the draws are prior draws.

    Raises
    ------
    ValueError
        If the selection is empty, repeated, or out of range.
    """
    ids = tuple(draw_ids)
    total = next(iter(draws.values())).shape[0] * next(iter(draws.values())).shape[1]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(type(i) is not int or not 0 <= i < total for i in ids)
    ):
        raise ValueError(f"draw_ids must be distinct 0-based positions below {total}")
    index = torch.tensor(ids, dtype=torch.long)
    return PriorDraws(
        samples={
            name: value.flatten(0, 1)
            .index_select(0, index.to(value.device))
            .unsqueeze(0)
            for name, value in draws.items()
        },
        num_samples=len(ids),
    )


__all__ = [
    "PredictionMethod",
    "latent_draws",
    "prediction_method",
    "predictive_errors",
    "selected_draws",
    "summarize_prediction",
]
