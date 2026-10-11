"""Model comparison and PSIS-LOO via ArviZ.

Thin wrappers around the canonical ArviZ entry points with
quivers-typed inputs. No information-criterion or PSIS math lives
here; ArviZ is the source of truth.
"""

from __future__ import annotations

from typing import Literal, Mapping

import arviz as az
import xarray as xr


def compare(
    fits: Mapping[str, xr.DataTree],
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
    fits : Mapping[str, xr.DataTree]
        Per-model fits produced by quivers.diagnostics.to_datatree.
        Every fit must carry a log-likelihood group.
    method : "stacking", "BB-pseudo-BMA", or "pseudo-BMA"
        Model-weight estimator. Default "stacking" follows Yao,
        Vehtari, Simpson, and Gelman (2018).
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
    return az.compare(
        dict(fits),
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
    including nonfinite Pareto estimates.

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
