"""Posterior and posterior-predictive model averaging, as brms.

brms ``posterior_average`` and ``pp_average`` mix the draws of several fits
in proportion to model weights: with weights :math:`w_k` and a draw budget
:math:`S`, model :math:`k` contributes :math:`S_k` draws, the largest-remainder
rounding of :math:`w_k S`, chosen at random without replacement. The pooled
draws are draws of the mixture :math:`\\sum_k w_k p(\\cdot \\mid y, M_k)`
[@hoeting1999bayesian; @yao2018stacking]. Weights are stacking, pseudo-BMA or
information-criterion weights (`model_weights`), Bayesian model averaging
weights from bridge-sampled marginal likelihoods (`post_prob`), or explicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, Literal, Unpack, cast

import didactic.api as dx
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.diagnostics.posterior_tables import (
    SummaryTable,
    posterior_summary as summary_table,
)
from quivers.formulas._evidence import match_responses
from quivers.formulas._postprocess_methods import (
    JointPredictionOptions,
    SinglePredictionOptions,
    model_weights,
)
from quivers.formulas._postprocess_predict import PredictionMethod
from quivers.formulas._response_modifiers import PredictionOptions
from quivers.formulas._fit_types import Fit

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit

type WeightMethod = Literal["stacking", "pseudobma", "loo", "waic", "kfold", "bma"]
"""brms ``model_weights`` methods."""


class PosteriorAverage(ValueModel):
    """Model-averaged posterior draws, as brms ``posterior_average``.

    Parameters
    ----------
    draws : dict[str, torch.Tensor]
        Pooled draws ``(S,)`` of each brms-named variable, model by model.
    weights : dict[str, float]
        Normalized model weights.
    ndraws : dict[str, int]
        Draws contributed by each model.
    """

    draws: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    weights: dict[str, float]
    ndraws: dict[str, int]

    def to_frame(self) -> pd.DataFrame:
        """Return the draws as brms's data frame, one column per variable.

        Returns
        -------
        pandas.DataFrame
            One row per pooled draw.
        """
        return pd.DataFrame({name: value.numpy() for name, value in self.draws.items()})


class PredictionAverage(ValueModel):
    """Model-averaged predictions, as brms ``pp_average``.

    Parameters
    ----------
    draws : torch.Tensor
        Pooled prediction draws ``(S, row[, event])``, model by model.
    weights : dict[str, float]
        Normalized model weights.
    ndraws : dict[str, int]
        Draws contributed by each model.
    summary : SummaryTable or None
        brms's summary of the pooled draws, when requested.
    """

    draws: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    weights: dict[str, float]
    ndraws: dict[str, int]
    summary: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def round_largest_remainder(values: Sequence[float]) -> list[int]:
    """Round to integers preserving the rounded total, as brms.

    Parameters
    ----------
    values : Sequence[float]
        Nonnegative reals.

    Returns
    -------
    list[int]
        Floors, with one added to the largest fractional parts until the
        total equals ``round(sum(values))``.
    """
    total = round(sum(values))
    floors = [math.floor(v) for v in values]
    order = sorted(
        range(len(values)), key=lambda i: values[i] - floors[i], reverse=True
    )
    for i in order[: total - sum(floors)]:
        floors[i] += 1
    return floors


def averaging_weights(
    models: Mapping[str, Fit],
    weights: WeightMethod | Sequence[float],
    *,
    seed: int,
) -> dict[str, float]:
    """Return normalized model weights, as brms ``validate_weights``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit]
        Named fits.
    weights : str or Sequence[float]
        A `model_weights` method, ``"bma"`` for posterior model
        probabilities, or one nonnegative weight per model.
    seed : int
        Seed of pseudo-BMA bootstrap weights and bridge-sampling proposals.

    Returns
    -------
    dict[str, float]
        Weights summing to one.

    Raises
    ------
    ValueError
        If explicit weights have the wrong length, are negative, or sum to
        zero, or the method is unknown.
    """
    if isinstance(weights, str):
        return model_weights(models, weights=weights, seed=seed)
    values = [float(w) for w in weights]
    if len(values) != len(models):
        raise ValueError("numeric weights must have one value per model")
    if any(not math.isfinite(w) or w < 0 for w in values) or sum(values) <= 0:
        raise ValueError("numeric weights must be nonnegative with a positive sum")
    total = sum(values)
    return {name: w / total for name, w in zip(models, values, strict=True)}


def _allocation(
    models: Mapping[str, Fit],
    weights: WeightMethod | Sequence[float],
    ndraws: int | None,
    seed: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return model weights and the draws each model contributes.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit]
        Named fits, at least one.
    weights : str or Sequence[float]
        Weighting method or explicit weights.
    ndraws : int or None
        Total draws; the first model's draw count by default.
    seed : int
        Weighting seed.

    Returns
    -------
    tuple[dict[str, float], dict[str, int]]
        Normalized weights and per-model draw counts.

    Raises
    ------
    ValueError
        If no model is given or ``ndraws`` is not a positive integer.
    """
    if not models:
        raise ValueError("model averaging requires at least one model")
    first = next(iter(models.values()))
    total = first.ndraws() if ndraws is None else ndraws
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("ndraws must be a positive integer")
    normalized = averaging_weights(models, weights, seed=seed)
    counts = round_largest_remainder([w * total for w in normalized.values()])
    return normalized, dict(zip(models, counts, strict=True))


def _draw_ids(available: int, count: int, generator: torch.Generator) -> list[int]:
    """Choose pooled draw positions without replacement.

    Parameters
    ----------
    available : int
        Draws of the model.
    count : int
        Draws to choose.
    generator : torch.Generator
        Random stream.

    Returns
    -------
    list[int]
        Sorted 0-based draw positions.

    Raises
    ------
    ValueError
        If more draws are requested than the model has.
    """
    if count > available:
        raise ValueError(
            f"a model contributes {count} draws but has only {available}; lower ndraws"
        )
    return sorted(torch.randperm(available, generator=generator)[:count].tolist())


def posterior_average(
    models: Mapping[str, Fit],
    *,
    variable: str | Sequence[str] | None = None,
    weights: WeightMethod | Sequence[float] = "stacking",
    ndraws: int | None = None,
    missing: float | Mapping[str, float] | None = None,
    seed: int = 0,
) -> PosteriorAverage:
    """Mix the posterior draws of several fits, as brms ``posterior_average``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit]
        Named fits.
    variable : str or Sequence[str] or None
        brms variable names (see ``variables``); None takes every variable
        common to all models (every variable of any model with ``missing``).
    weights : str or Sequence[float]
        Weighting method of `averaging_weights`, or explicit weights.
    ndraws : int or None
        Total draws; the first model's draw count by default.
    missing : float or Mapping[str, float] or None
        Value of a variable in models that lack it; None requires every
        variable in every model.
    seed : int
        Seed of the draw selection and of the weights.

    Returns
    -------
    PosteriorAverage
        Pooled draws, weights and draw counts.

    Raises
    ------
    ValueError
        If a variable is absent from the required models, ``missing`` lacks
        a value for one, or the weights or draw counts are invalid.
    """
    named = {name: fit.variables() for name, fit in models.items()}
    every = list(dict.fromkeys(v for draws in named.values() for v in draws))
    if missing is None:
        common = [v for v in every if all(v in draws for draws in named.values())]
        requested = common if variable is None else _names(variable)
        absent = [v for v in requested if v not in common]
        if absent:
            raise ValueError(
                f"variables {absent} are not in all models; give missing= to fill them"
            )
        fill: dict[str, float] = {}
    else:
        requested = every if variable is None else _names(variable)
        unknown = [v for v in requested if v not in every]
        if unknown:
            raise ValueError(f"variables {unknown} are in none of the models")
        if isinstance(missing, Mapping):
            needed = {
                v for draws in named.values() for v in requested if v not in draws
            }
            lacking = sorted(needed - set(missing))
            if lacking:
                raise ValueError(f"missing has no value for variables {lacking}")
            fill = {k: float(v) for k, v in missing.items()}
        else:
            fill = {v: float(missing) for v in requested}
    normalized, counts = _allocation(models, weights, ndraws, seed)
    generator = torch.Generator().manual_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {v: [] for v in requested}
    for name, fit in models.items():
        count = counts[name]
        if count == 0:
            continue
        draws = named[name]
        ids = torch.tensor(_draw_ids(fit.ndraws(), count, generator))
        for v in requested:
            if v in draws:
                pooled = draws[v].reshape(-1, *draws[v].shape[2:]).double()
                parts[v].append(pooled[ids])
            else:
                parts[v].append(torch.full((count,), fill[v], dtype=torch.float64))
    return PosteriorAverage(
        draws={v: torch.cat(p) for v, p in parts.items()},
        weights=normalized,
        ndraws=counts,
    )


def _names(variable: str | Sequence[str]) -> list[str]:
    """Return requested variable names as a list.

    Parameters
    ----------
    variable : str or Sequence[str]
        One name or several.

    Returns
    -------
    list[str]
        The names.
    """
    return [variable] if isinstance(variable, str) else list(variable)


def pp_average(
    models: Mapping[str, Fit],
    *,
    weights: WeightMethod | Sequence[float] = "stacking",
    method: PredictionMethod = "posterior_predict",
    ndraws: int | None = None,
    summary: bool = True,
    probs: Sequence[float] = (0.025, 0.975),
    robust: bool = False,
    resp: str | None = None,
    **options: Unpack[PredictionOptions],
) -> PredictionAverage:
    """Mix the predictions of several fits, as brms ``pp_average``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit]
        Named fits of the same response data.
    weights : str or Sequence[float]
        Weighting method of `averaging_weights`, or explicit weights.
    method : {"posterior_predict", "posterior_epred", "posterior_linpred"}
        Prediction method.
    ndraws : int or None
        Total draws; the first model's draw count by default.
    summary : bool
        Also summarize the pooled draws.
    probs : Sequence[float]
        Quantile probabilities of the summary.
    robust : bool
        Median and MAD instead of mean and SD.
    resp : str or None
        Response of joint fits; required when a joint fit has several.
    **options : PredictionOptions
        Prediction keywords such as ``new_data`` and ``re_formula``; ``seed``
        also seeds the draw selection and the weights.

    Returns
    -------
    PredictionAverage
        Pooled draws, weights, draw counts and the optional summary.

    Raises
    ------
    ValueError
        If the fits model different responses, ``draw_ids`` or ``num_samples``
        is given, a joint response is ambiguous, or the weights are invalid.
    """
    if "num_samples" in options:
        raise ValueError("pp_average chooses draws itself; give ndraws instead")
    if method not in {"posterior_predict", "posterior_epred", "posterior_linpred"}:
        raise ValueError(
            "method must be posterior_predict, posterior_epred, or posterior_linpred"
        )
    match_responses(tuple(models.values()))
    seed = options.get("seed", 0)
    normalized, counts = _allocation(models, weights, ndraws, seed)
    generator = torch.Generator().manual_seed(seed)
    parts: list[torch.Tensor] = []
    request = dict(options)
    for name, fit in models.items():
        count = counts[name]
        if count == 0:
            continue
        ids = _draw_ids(fit.ndraws(), count, generator)
        if hasattr(fit, "response_fits"):
            joint = cast("MultivariateFit", fit)
            selected = resp
            if selected is None:
                if len(joint.response_names) != 1:
                    raise ValueError(
                        "pp_average of a joint fit with several responses requires resp"
                    )
                selected = joint.response_names[0]
            values = joint._pp_prediction(
                method, selected, cast(JointPredictionOptions, request), draw_ids=ids
            )[selected]
        else:
            if resp is not None:
                raise ValueError("resp applies only to joint fits")
            values = cast("BayesianFit", fit)._pp_prediction(
                method, cast(SinglePredictionOptions, request), draw_ids=ids
            )
        parts.append(values.reshape(-1, *values.shape[2:]).double())
    pooled = torch.cat(parts)
    table = (
        summary_table(pooled.unsqueeze(0), probs=probs, robust=robust)
        if summary
        else None
    )
    return PredictionAverage(
        draws=pooled, weights=normalized, ndraws=counts, summary=table
    )


__all__ = [
    "PosteriorAverage",
    "PredictionAverage",
    "averaging_weights",
    "posterior_average",
    "pp_average",
    "round_largest_remainder",
]
