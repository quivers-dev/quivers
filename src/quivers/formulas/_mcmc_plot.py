"""brms ``mcmc_plot``, ``plot`` and ``pairs`` through ArviZ.

brms draws MCMC diagnostics with bayesplot's ``mcmc_*`` functions on its
brms-named draws. Here each bayesplot type maps onto the ArviZ plot that
shows the same quantity, drawn from a data tree of the fit's brms-named draws
(``variables``) and sampler statistics. Types without an ArviZ counterpart
are refused with the reason.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import TYPE_CHECKING

import arviz_plots as azp
import numpy as np
import xarray as xr

from quivers.diagnostics.arviz_io import _ARVIZ_SAMPLE_STATS
from quivers.inference.mcmc.driver import MCMCResult
from quivers.formulas._fit_types import Fit

if TYPE_CHECKING:
    pass

type PlotResult = azp.PlotCollection | azp.PlotMatrix
"""An ArviZ plot."""

#: bayesplot ``mcmc_*`` types and their ArviZ counterparts: function and options.
MCMC_PLOTS: dict[str, tuple[str, dict[str, str | bool]]] = {
    "acf": ("plot_autocorr", {}),
    "acf_bar": ("plot_autocorr", {}),
    "areas": ("plot_ridge", {}),
    "areas_ridges": ("plot_ridge", {}),
    "combo": ("plot_trace_dist", {}),
    "dens": ("plot_dist", {"kind": "kde"}),
    "dens_chains": ("plot_dist", {"kind": "kde"}),
    "dens_overlay": ("plot_dist", {"kind": "kde"}),
    "hist": ("plot_dist", {"kind": "hist"}),
    "hist_by_chain": ("plot_dist", {"kind": "hist"}),
    "intervals": ("plot_forest", {}),
    "neff": ("plot_ess", {}),
    "neff_hist": ("plot_ess", {}),
    "nuts_energy": ("plot_energy", {}),
    "pairs": ("plot_pair", {}),
    "parcoord": ("plot_parallel", {}),
    "rank_ecdf": ("plot_rank", {}),
    "rank_hist": ("plot_rank_dist", {"kind": "hist"}),
    "rank_overlay": ("plot_rank_dist", {}),
    "rhat": ("plot_convergence_dist", {}),
    "rhat_hist": ("plot_convergence_dist", {"kind": "hist"}),
    "scatter": ("plot_pair", {}),
    "hex": ("plot_pair", {}),
    "trace": ("plot_trace", {}),
    "trace_highlight": ("plot_trace", {}),
    "violin": ("plot_dist", {"kind": "kde"}),
}

#: bayesplot types without an ArviZ counterpart, and why.
NOT_PROVIDED: dict[str, str] = {
    "nuts_acceptance": "ArviZ has no acceptance-statistic plot; the values are in nuts_params()",
    "nuts_divergence": "ArviZ has no divergence-by-log-posterior plot; the values are in nuts_params()",
    "nuts_stepsize": "ArviZ has no step-size plot; the values are in nuts_params()",
    "nuts_treedepth": "ArviZ has no tree-depth plot; the values are in nuts_params()",
    "recover_hist": "recovery plots need known true parameter values, which a fit does not hold",
    "recover_intervals": "recovery plots need known true parameter values, which a fit does not hold",
    "recover_scatter": "recovery plots need known true parameter values, which a fit does not hold",
}


def _safe(name: str) -> str:
    """Return an ArviZ-safe variable name for a brms name.

    Parameters
    ----------
    name : str
        brms variable name, possibly with brackets and commas.

    Returns
    -------
    str
        The name with brackets and commas replaced by dots.
    """
    return re.sub(r"[\[\],]", ".", name).rstrip(".")


def mcmc_datatree(fit: Fit, variable: str | Sequence[str] | None = None) -> xr.DataTree:
    """Return an ArviZ data tree of brms-named draws and sampler statistics.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    variable : str or Sequence[str] or None
        brms variable names, or regular expressions matched against them;
        None for every variable.

    Returns
    -------
    xarray.DataTree
        ``posterior`` with one scalar variable per brms name (brackets
        replaced by dots) and ``sample_stats`` with ``energy`` and
        ``diverging`` of NUTS chains.

    Raises
    ------
    ValueError
        If no variable matches.
    """
    named = fit.variables()
    if variable is None:
        chosen = list(named)
    else:
        patterns = [variable] if isinstance(variable, str) else list(variable)
        chosen = [
            n
            for n in named
            if any(n == p or re.fullmatch(p, n) is not None for p in patterns)
        ]
    if not chosen:
        raise ValueError(f"no variable matches {variable!r}")
    posterior = xr.Dataset(
        {
            _safe(n): (
                ("chain", "draw"),
                named[n].double().numpy().reshape(named[n].shape[:2]),
            )
            for n in chosen
            if named[n].ndim == 2
        },
        coords={
            "chain": np.arange(named[chosen[0]].shape[0]),
            "draw": np.arange(named[chosen[0]].shape[1]),
        },
    )
    groups = {"posterior": posterior}
    result = fit.posterior
    if isinstance(result, MCMCResult):
        stats = {
            _ARVIZ_SAMPLE_STATS[key]: (("chain", "draw"), value.double().numpy())
            for key, value in result.sample_stats.items()
            if key in _ARVIZ_SAMPLE_STATS and value.ndim == 2
        }
        if stats:
            groups["sample_stats"] = xr.Dataset(stats, coords=posterior.coords)
    return xr.DataTree.from_dict(groups)


def mcmc_plot(
    fit: Fit,
    type: str = "intervals",
    *,
    variable: str | Sequence[str] | None = None,
    backend: str,
    **kwargs: str | int | float | bool,
) -> PlotResult:
    """Draw an MCMC diagnostic plot, as brms ``mcmc_plot``.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    type : str
        bayesplot ``mcmc_*`` type without prefix (`MCMC_PLOTS`).
    variable : str or Sequence[str] or None
        brms variable names or regular expressions; None for all.
    backend : str
        ArviZ plotting backend (``"matplotlib"``, ``"bokeh"``, ``"plotly"``,
        or ``"none"`` to build without drawing).
    **kwargs : str or int or float or bool
        Further ArviZ keyword arguments.

    Returns
    -------
    arviz_plots.PlotCollection or arviz_plots.PlotMatrix
        The plot.

    Raises
    ------
    ValueError
        If the type is unknown or has no ArviZ counterpart.
    """
    if type in NOT_PROVIDED:
        raise ValueError(
            f"mcmc_plot type {type!r} is not provided: {NOT_PROVIDED[type]}"
        )
    if type not in MCMC_PLOTS:
        raise ValueError(
            f"unknown mcmc_plot type {type!r}; types are {', '.join(sorted(MCMC_PLOTS))}"
        )
    name, defaults = MCMC_PLOTS[type]
    options: dict[str, str | int | float | bool] = {**defaults, **kwargs}
    function = getattr(azp, name)
    tree = mcmc_datatree(fit, variable)
    return function(tree, backend=backend, **options)


__all__ = ["MCMC_PLOTS", "NOT_PROVIDED", "mcmc_datatree", "mcmc_plot"]
