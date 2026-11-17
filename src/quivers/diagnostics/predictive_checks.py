"""Posterior-predictive checks (PPCs) and calibration diagnostics.

ArviZ ships `arviz.plot_ppc`, `arviz.loo_pit`, and a
broad family of PPC visualisations.  This module exposes a small
typed surface that delegates to those, and adds a registry of
common test statistics that map cleanly onto quivers' plate-and-
group structure (per-Verb mean, per-Subject sd, etc.).
"""

from __future__ import annotations

from typing import Callable, Mapping

import arviz as az
import numpy as np
import xarray as xr

from quivers.diagnostics._ppc_data import STATISTICS


def posterior_predictive_check(
    idata: xr.DataTree,
    *,
    observed_name: str,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    by: str | None = None,
) -> Mapping[str, float | np.ndarray | str]:
    """Compute a posterior-predictive p-value (PPP-value) for a
    user-chosen test statistic.

    Parameters
    ----------
    idata : xr.DataTree
        Fit produced by [`quivers.diagnostics.to_datatree`][quivers.diagnostics.to_datatree],
        with both ``observed_data`` and ``posterior_predictive``
        groups populated.
    observed_name : str
        Name of the observed site (must appear in both groups).
    statistic : str or callable
        Either a key into `STATISTICS` or a user-supplied
        ``Callable[[np.ndarray], float]``.
    by : str, optional
        If given, computes the statistic per group along the named
        dim (e.g. ``by="Verb"``); the PPP-value becomes a numpy
        array of shape ``(|by|,)``.

    Returns
    -------
    Mapping[str, float or numpy.ndarray]
        ``{"observed": T(y), "predictive_mean": E[T(y_rep)],
        "ppp": Pr(T(y_rep) >= T(y))}``.  The PPP-value is the
        canonical [posterior-predictive p-value](https://en.wikipedia.org/wiki/Posterior_predictive_p-value);
        values near 0 or 1 indicate model mis-fit on the chosen
        statistic.
    """
    if callable(statistic):
        stat_fn = statistic
        stat_name = getattr(statistic, "__name__", "user_statistic")
    else:
        if statistic not in STATISTICS:
            raise ValueError(
                f"unknown statistic {statistic!r}; choices are "
                f"{sorted(STATISTICS)} or a user-supplied callable"
            )
        stat_fn = STATISTICS[statistic]
        stat_name = statistic

    observed = idata["observed_data"][observed_name].values
    pp = idata["posterior_predictive"][observed_name].values
    # pp shape: (chain, draw, *site_shape).  Flatten chains for stat
    # computation across all draws.
    chains, draws = pp.shape[0], pp.shape[1]
    pp_flat = pp.reshape(chains * draws, *pp.shape[2:])

    if by is None:
        t_obs = stat_fn(observed)
        t_rep = np.array([stat_fn(pp_flat[i]) for i in range(pp_flat.shape[0])])
        return {
            "statistic": stat_name,
            "observed": t_obs,
            "predictive_mean": float(t_rep.mean()),
            "ppp": float((t_rep >= t_obs).mean()),
        }

    # Per-group statistic.  Locate the `by` dim in the observed array.
    dims = idata["observed_data"][observed_name].dims
    if by not in dims:
        raise ValueError(
            f"posterior_predictive_check: dim {by!r} not found in "
            f"observed_data/{observed_name} dims {dims}"
        )
    group_axis = dims.index(by)
    observed_groups = np.split(observed, observed.shape[group_axis], axis=group_axis)
    pp_axis = group_axis + 2  # +2 for the leading (chain, draw) axes
    pp_groups = np.split(pp_flat, pp_flat.shape[pp_axis - 1], axis=pp_axis - 1)
    t_obs = np.array([stat_fn(g.squeeze(group_axis)) for g in observed_groups])
    t_rep = np.array(
        [
            np.array([stat_fn(g[i].squeeze(pp_axis - 2)) for i in range(g.shape[0])])
            for g in pp_groups
        ]
    )  # (groups, n_draws)
    return {
        "statistic": stat_name,
        "by": by,
        "observed": t_obs,
        "predictive_mean": t_rep.mean(axis=1),
        "ppp": (t_rep >= t_obs[:, None]).mean(axis=1),
    }


def loo_pit(
    idata: xr.DataTree,
    *,
    observed_name: str,
) -> xr.Dataset:
    """Leave-one-out probability-integral-transform calibration.

    Delegates to `arviz.loo_pit`. The canonical use of the PIT values
    is `arviz.plot_loo_pit` for the calibration diagnostic plot.

    Parameters
    ----------
    idata : xr.DataTree
        Fit produced by [`quivers.diagnostics.to_datatree`][quivers.diagnostics.to_datatree],
        with ``posterior_predictive``, ``log_likelihood``, and
        ``observed_data`` groups populated.
    observed_name : str
        Name of the observed site, which must appear in those groups.

    Returns
    -------
    xr.Dataset
        The LOO-PIT value at each observed data point, under the
        variable ``observed_name``.
    """
    return az.loo_pit(idata, var_names=observed_name)


__all__ = ["STATISTICS", "loo_pit", "posterior_predictive_check"]
