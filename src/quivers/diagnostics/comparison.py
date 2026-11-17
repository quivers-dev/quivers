"""Model comparison and PSIS-LOO via ArviZ.

Thin wrappers around the canonical ArviZ entry points with
quivers-typed inputs. No information-criterion or PSIS math lives
here; ArviZ is the source of truth.
"""

from __future__ import annotations

from typing import Literal, Mapping

import arviz as az
import xarray as xr

from quivers.diagnostics.posterior import PosteriorAnalysis
import torch


def compare(
    fits: Mapping[str, xr.DataTree | PosteriorAnalysis],
    *,
    method: Literal["stacking", "BB-pseudo-BMA", "pseudo-BMA"] = "stacking",
    var_name: str | None = None,
    reference: str | None = None,
    round_to: int | Literal["auto", "none", "None"] | None = "auto",
) -> object:
    """Rank candidate models by expected log predictive density.

    Delegates to ArviZ, which computes PSIS-LOO on each fit's
    log-likelihood group and combines the results into a ranked
    comparison table.

    Parameters
    ----------
    fits : Mapping[str, xr.DataTree or PosteriorAnalysis]
        Per-model retained analyses or DataTrees. Each supplies raw observation
        log likelihoods. Retained analyses are checked for matching response
        values and observation units before their PSIS-LOO estimates are ranked.
        Variational analyses use relative efficiency one.
    method : "stacking", "BB-pseudo-BMA", or "pseudo-BMA"
        Model-weight estimator. Default "stacking" follows Yao,
        Vehtari, Simpson, and Gelman (2018) [Yao et al. (2018)](<https://doi.org/10.1214/17-BA1091>).
    var_name : str, optional
        Observed variable in the log-likelihood group; required
        when a fit carries multiple log-likelihood variables.
    reference : str, optional
        Fit name used as the reference for ELPD differences.
        Default is the top-ranked model.
    round_to : int, "auto", "none", "None", or None, optional
        ArviZ rounding policy. Default "auto" retains display
        rounding. An integer specifies decimal places, while
        "none" or "None" preserves full numeric precision. None
        uses ArviZ's configured default.

    Returns
    -------
    pandas.DataFrame
        ArviZ ranking table with one row per model.
    """
    analyses = [fit for fit in fits.values() if isinstance(fit, PosteriorAnalysis)]
    if analyses:
        # Matching response values catch accidental comparison of different
        # datasets; callers still choose a common predictive unit/design.
        selected = [fit._likelihood_name(var_name) for fit in analyses]
        reference_analysis, reference_name = analyses[0], selected[0]
        reference_scores = reference_analysis.log_likelihood[reference_name]
        for fit, name in zip(analyses[1:], selected[1:]):
            if fit.log_likelihood[name].shape[2:] != reference_scores.shape[2:]:
                raise ValueError("model comparisons require matching observation units")
            reference_data = reference_analysis.observed_data.get(reference_name)
            values = fit.observed_data.get(name)
            if reference_data is None or values is None:
                raise ValueError("model comparisons require observed response data")
            if not torch.equal(reference_data, values):
                raise ValueError(
                    "model comparisons require the same observed responses"
                )
    prepared = {
        name: fit.loo(var_name=var_name, pointwise=True)
        if isinstance(fit, PosteriorAnalysis)
        else fit
        for name, fit in fits.items()
    }
    return az.compare(
        prepared,
        method=method,
        var_name=var_name,
        reference=reference,
        round_to=round_to,
    )


def _require_log_likelihood_variable(fit: xr.DataTree, var_name: str | None) -> None:
    """Validate only the variable selection needed before ArviZ LOO."""
    if "log_likelihood" not in fit.children:
        raise ValueError("loo: missing log_likelihood group")

    variables = list(fit["log_likelihood"].dataset.data_vars)
    if not variables:
        raise ValueError("loo: log_likelihood group contains no variables")
    if var_name is None and len(variables) > 1:
        names = ", ".join(repr(name) for name in variables)
        raise ValueError(
            "loo: var_name is required when log_likelihood contains "
            f"multiple variables: {names}"
        )
    if var_name is not None and var_name not in variables:
        names = ", ".join(repr(name) for name in variables)
        raise ValueError(
            f"loo: log_likelihood variable {var_name!r} was not found; "
            f"available variables: {names}"
        )


def loo(
    fit: xr.DataTree,
    *,
    pointwise: bool | None = None,
    var_name: str | None = None,
) -> object:
    """Estimate expected log predictive density with PSIS-LOO.

    This thin wrapper delegates all computation to ArviZ. It
    preserves ArviZ's canonical log scale and, with pointwise=True,
    returns its per-observation elpd_i and pareto_k arrays unchanged,
    including nonfinite Pareto estimates. PSIS follows Vehtari, Simpson,
    Gelman, Yao, and Gabry (2024) [Vehtari et al. (2024)](<https://www.jmlr.org/papers/v25/19-556.html>); its use for
    leave-one-out evaluation follows Vehtari, Gelman, and Gabry (2017)
    [Vehtari et al. (2017)](<https://link.springer.com/article/10.1007/s11222-016-9696-4>).

    Parameters
    ----------
    fit : xr.DataTree
        Fit containing posterior and log-likelihood groups.
    pointwise : bool, optional
        Whether to include one elpd_i and pareto_k value per
        observation. None uses ArviZ's configured default.
    var_name : str, optional
        Log-likelihood variable to evaluate. Required when the
        log-likelihood group contains multiple variables.

    Returns
    -------
    arviz.ELPDData
        ArviZ's canonical PSIS-LOO result.
    """
    _require_log_likelihood_variable(fit, var_name)
    return az.loo(fit, pointwise=pointwise, var_name=var_name)


__all__ = ["compare", "loo"]
