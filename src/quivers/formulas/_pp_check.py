"""Posterior and prior predictive checks of fits, as brms ``pp_check``.

brms ``pp_check`` draws replicated data, selects observed responses and
covariates, and hands them to a bayesplot ``ppc_*`` or ``ppd_*`` function.
`pp_check` here performs the same preparation through the fit's own
prediction methods and returns the data behind the requested plot
(the ``ppc_*`` functions of `quivers.diagnostics`) as a `PPCheck`, which keeps the observed
values, the replicated draws, and (for leave-one-out types) the PSIS log
weights. Where ArviZ implements a counterpart plot, `PPCheck.plot` draws it
from `PPCheck.to_datatree` with the backend the caller names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import arviz_plots as azp
import didactic.api as dx
import narwhals as nw
import numpy as np
import pandas as pd
import torch
import xarray as xr
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.diagnostics import _ppc_data as data_functions
from quivers.diagnostics.elpd import loo_expectation, psis_weights
from quivers.formulas._brms_effects import model_frame
from quivers.formulas._postprocess_methods import (
    JointPredictionOptions,
    SinglePredictionOptions,
)
from quivers.formulas._postprocess_predict import PredictionMethod
from quivers.formulas._response import response_tensor
from quivers.formulas._fit_types import Fit

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit

type Prefix = Literal["ppc", "ppd"]
"""Posterior predictive check (``ppc``) or predictive distribution (``ppd``)."""

type StatisticSpec = (
    data_functions.Statistic | tuple[data_functions.Statistic, data_functions.Statistic]
)
"""One statistic, or two for ``stat_2d``."""

#: bayesplot ppc types and the table each is drawn from.
PPC_TYPES: dict[str, str] = {
    "bars": "bars",
    "bars_grouped": "bars",
    "boxplot": "long",
    "dens": "long",
    "dens_overlay": "long",
    "dens_overlay_grouped": "long",
    "ecdf_overlay": "long",
    "ecdf_overlay_grouped": "long",
    "error_binned": "error_binned",
    "error_hist": "error",
    "error_hist_grouped": "error",
    "error_scatter": "error",
    "error_scatter_avg": "error_avg",
    "error_scatter_avg_grouped": "error_avg",
    "error_scatter_avg_vs_x": "error_avg_x",
    "freqpoly": "long",
    "freqpoly_grouped": "long",
    "hist": "long",
    "intervals": "intervals",
    "intervals_grouped": "intervals",
    "km_overlay": "km",
    "km_overlay_grouped": "km",
    "loo_intervals": "loo_intervals",
    "loo_pit": "loo_pit",
    "loo_pit_overlay": "loo_pit",
    "loo_pit_qq": "loo_pit",
    "loo_ribbon": "loo_intervals",
    "pit_ecdf": "pit_ecdf",
    "pit_ecdf_grouped": "pit_ecdf",
    "ribbon": "intervals",
    "ribbon_grouped": "intervals",
    "rootogram": "rootogram",
    "scatter": "scatter",
    "scatter_avg": "scatter_avg",
    "scatter_avg_grouped": "scatter_avg",
    "stat": "stat",
    "stat_2d": "stat",
    "stat_freqpoly": "stat",
    "stat_freqpoly_grouped": "stat",
    "stat_grouped": "stat",
    "violin_grouped": "long",
}

#: bayesplot ppd types (no observed data).
PPD_TYPES: frozenset[str] = frozenset(
    {
        "boxplot",
        "dens",
        "dens_overlay",
        "ecdf_overlay",
        "freqpoly",
        "freqpoly_grouped",
        "hist",
        "intervals",
        "intervals_grouped",
        "ribbon",
        "ribbon_grouped",
        "stat",
        "stat_2d",
        "stat_freqpoly",
        "stat_freqpoly_grouped",
        "stat_grouped",
    }
)

#: Types brms draws with every posterior draw by default (others use 10).
ALL_DRAW_TYPES: frozenset[str] = frozenset(
    {
        "error_scatter_avg",
        "error_scatter_avg_vs_x",
        "intervals",
        "intervals_grouped",
        "loo_intervals",
        "loo_pit",
        "loo_pit_overlay",
        "loo_pit_qq",
        "loo_ribbon",
        "loo_pit_ecdf",
        "pit_ecdf",
        "pit_ecdf_grouped",
        "ribbon",
        "ribbon_grouped",
        "rootogram",
        "scatter_avg",
        "scatter_avg_grouped",
        "stat",
        "stat_2d",
        "stat_freqpoly_grouped",
        "stat_grouped",
        "violin_grouped",
    }
)

#: ArviZ counterpart of a type: plot function and keyword arguments.
ARVIZ_PLOTS: dict[str, tuple[str, dict[str, str]]] = {
    "dens": ("plot_ppc_dist", {"kind": "kde"}),
    "dens_overlay": ("plot_ppc_dist", {"kind": "kde"}),
    "hist": ("plot_ppc_dist", {"kind": "hist"}),
    "ecdf_overlay": ("plot_ppc_dist", {"kind": "ecdf"}),
    "stat": ("plot_ppc_tstat", {}),
    "intervals": ("plot_ppc_interval", {}),
    "ribbon": ("plot_ppc_interval", {}),
    "bars": ("plot_ppc_rootogram", {}),
    "rootogram": ("plot_ppc_rootogram", {}),
    "loo_pit": ("plot_loo_pit", {}),
    "loo_pit_overlay": ("plot_loo_pit", {}),
    "loo_pit_ecdf": ("plot_loo_pit", {}),
    "loo_intervals": ("plot_loo_interval", {}),
    "loo_ribbon": ("plot_loo_interval", {}),
    "pit_ecdf": ("plot_ppc_pit", {}),
    "km_overlay": ("plot_ppc_censored", {}),
}


class PPCheck(ValueModel):
    """The data behind one predictive check, as brms ``pp_check`` prepares it.

    Parameters
    ----------
    type : str
        bayesplot type without its prefix.
    prefix : {"ppc", "ppd"}
        Posterior predictive check or predictive distribution.
    data : pandas.DataFrame
        bayesplot's table for the type (see the ``ppc_*`` functions of `quivers.diagnostics`).
    y : torch.Tensor or None
        Observed responses ``(N,)``; None for ``ppd`` types.
    yrep : torch.Tensor
        Predictive draws ``(S, N)`` (expected values for ``error_binned``).
    draw_ids : tuple[int, ...]
        Pooled 0-based draws used.
    group : tuple or None
        Group label per observation.
    x : tuple or None
        Covariate per observation.
    status : torch.Tensor or None
        Event indicator of censored responses (``km_overlay``).
    log_weights : torch.Tensor or None
        Normalized PSIS log weights ``(S, N)`` of leave-one-out types.
    log_likelihood : torch.Tensor or None
        Pointwise log-likelihoods ``(S, N)`` of leave-one-out types.
    pit : torch.Tensor or None
        PIT or LOO-PIT values per observation.
    """

    type: str
    prefix: Prefix
    data: pd.DataFrame = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    y: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    yrep: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    draw_ids: tuple[int, ...] = ()
    group: tuple[str | int | float | bool, ...] | None = None
    x: tuple[str | int | float | bool, ...] | None = None
    status: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_weights: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_likelihood: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    pit: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def to_datatree(self) -> xr.DataTree:
        """Return an ArviZ data tree of the check.

        Returns
        -------
        xarray.DataTree
            ``posterior_predictive`` (one chain), ``observed_data`` for
            ``ppc`` types, ``log_likelihood`` for leave-one-out types, and
            ``constant_data`` with the event indicator of censored responses.
        """
        draws = self.yrep.double().numpy()[None]
        groups: dict[str, xr.Dataset] = {
            "posterior_predictive": xr.Dataset(
                {"y": (("chain", "draw", "obs"), draws)},
                coords={"chain": [0], "draw": np.arange(draws.shape[1])},
            )
        }
        if self.y is not None:
            groups["observed_data"] = xr.Dataset(
                {"y": (("obs",), self.y.double().numpy())}
            )
        if self.log_likelihood is not None:
            groups["log_likelihood"] = xr.Dataset(
                {
                    "y": (
                        ("chain", "draw", "obs"),
                        self.log_likelihood.double().numpy()[None],
                    )
                },
                coords={"chain": [0], "draw": np.arange(draws.shape[1])},
            )
        if self.status is not None:
            # ArviZ's censored check reads 1 for a censored observation.
            groups["constant_data"] = xr.Dataset(
                {"y": (("obs",), 1.0 - self.status.double().numpy())}
            )
        return xr.DataTree.from_dict(groups)

    def plot(
        self, *, backend: str, **kwargs: str | int | float | bool
    ) -> azp.PlotCollection:
        """Draw the check with its ArviZ counterpart.

        Parameters
        ----------
        backend : str
            ArviZ plotting backend (``"matplotlib"``, ``"bokeh"``,
            ``"plotly"``, or ``"none"`` to build the plot without drawing).
        **kwargs : str or int or float or bool
            Further keyword arguments of the ArviZ function.

        Returns
        -------
        arviz_plots.PlotCollection
            The plot.

        Raises
        ------
        ValueError
            If ArviZ has no counterpart of the type; its data remain in `data`.
        """
        if self.type not in ARVIZ_PLOTS or (
            self.prefix == "ppd"
            and self.type not in {"dens", "dens_overlay", "hist", "ecdf_overlay"}
        ):
            raise ValueError(
                f"ArviZ has no counterpart of {self.prefix}_{self.type}; plot the "
                "table in PPCheck.data"
            )
        name, defaults = ARVIZ_PLOTS[self.type]
        options: dict[str, str | int | float | bool] = {**defaults, **kwargs}
        if self.prefix == "ppd":
            # The predictive distribution alone: ArviZ's marginal plot.
            name = "plot_dist"
            options.setdefault("group", "posterior_predictive")
        function = getattr(azp, name)
        return function(self.to_datatree(), backend=backend, **options)


def _joint(fit: Fit) -> bool:
    """Return whether a fit is a joint multivariate fit.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.

    Returns
    -------
    bool
        True for a `MultivariateFit`.
    """
    return hasattr(fit, "response_fits")


def _select_draws(
    fit: Fit, type: str, ndraws: int | None, draw_ids: Sequence[int] | None, seed: int
) -> list[int]:
    """Choose draws as brms ``pp_check`` does.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    type : str
        Plot type.
    ndraws : int or None
        Requested draws; brms's default is every draw for summary types and
        ten random draws otherwise.
    draw_ids : Sequence[int] or None
        Explicit 0-based draws.
    seed : int
        Seed of the random selection.

    Returns
    -------
    list[int]
        0-based pooled draws.

    Raises
    ------
    ValueError
        If both are given or ``ndraws`` is invalid.
    """
    available = fit.ndraws()
    if draw_ids is not None:
        if ndraws is not None:
            raise ValueError("give ndraws or draw_ids, not both")
        return list(draw_ids)
    if ndraws is None:
        if type in ALL_DRAW_TYPES:
            return list(range(available))
        ndraws = min(10, available)
    if (
        isinstance(ndraws, bool)
        or not isinstance(ndraws, int)
        or not 0 < ndraws <= available
    ):
        raise ValueError(f"ndraws must be an integer in [1, {available}]")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(available, generator=generator)[:ndraws].tolist()


def _predict(
    fit: Fit,
    resp: str | None,
    method: PredictionMethod,
    newdata: IntoDataFrame | None,
    draw_ids: list[int],
    seed: int,
) -> torch.Tensor:
    """Return prediction draws ``(S, N)`` at the selected draws.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    resp : str or None
        Joint response.
    method : {"posterior_predict", "posterior_epred", "posterior_linpred"}
        Prediction method.
    newdata : IntoDataFrame or None
        Prediction rows.
    draw_ids : list[int]
        0-based pooled draws.
    seed : int
        Prediction seed.

    Returns
    -------
    torch.Tensor
        Draws ``(S, N)``.

    Raises
    ------
    ValueError
        If predictions have an event axis (a multicolumn response).
    """
    if _joint(fit):
        assert resp is not None
        options: JointPredictionOptions = {"new_data": newdata, "seed": seed}
        values = cast("MultivariateFit", fit)._pp_prediction(
            method, resp, options, draw_ids=draw_ids
        )[resp]
    else:
        single: SinglePredictionOptions = {"new_data": newdata, "seed": seed}
        values = cast("BayesianFit", fit)._pp_prediction(
            method, single, draw_ids=draw_ids
        )
    if values.ndim != 3:
        raise ValueError("pp_check is not implemented for multicolumn responses")
    return values.reshape(-1, values.shape[-1]).double()


def _view(fit: Fit, resp: str | None) -> BayesianFit:
    """Return the single-response view of the checked response.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    resp : str or None
        Joint response.

    Returns
    -------
    BayesianFit
        The response view.
    """
    if _joint(fit):
        assert resp is not None
        return cast("MultivariateFit", fit)._response_fit(resp)
    return cast("BayesianFit", fit)


def _observed(view: BayesianFit, newdata: IntoDataFrame | None) -> torch.Tensor:
    """Return observed responses in the fit's response coding.

    Parameters
    ----------
    view : BayesianFit
        Response view.
    newdata : IntoDataFrame or None
        Prediction rows holding the response, or None for training rows.

    Returns
    -------
    torch.Tensor
        Responses ``(N,)``; categories as positions in ``category_labels``.

    Raises
    ------
    ValueError
        If new rows lack the response or hold unknown categories.
    """
    if newdata is None:
        values = response_tensor(view.formula, view.family).double()
        return values.reshape(values.shape[0], -1)[:, 0] if values.ndim > 1 else values
    frame = nw.from_native(newdata, eager_only=True).to_pandas()
    name = view.formula.response_name
    if name not in frame.columns:
        raise ValueError(f"newdata must hold the response {name!r} for ppc types")
    column = frame[name]
    labels = view.category_labels
    if labels:
        index = {str(label): i for i, label in enumerate(labels)}
        unknown = sorted({str(v) for v in column.tolist()} - set(index))
        if unknown:
            raise ValueError(f"newdata holds unknown response categories {unknown}")
        return torch.tensor([float(index[str(v)]) for v in column.tolist()])
    return torch.as_tensor(column.to_numpy(np.float64))


def _status(view: BayesianFit, newdata: IntoDataFrame | None) -> np.ndarray | None:
    """Return brms's censoring codes of the training rows.

    Parameters
    ----------
    view : BayesianFit
        Response view.
    newdata : IntoDataFrame or None
        Prediction rows; censoring is read only for training rows.

    Returns
    -------
    numpy.ndarray or None
        Codes -1 (left), 0 (none), 1 (right), 2 (interval), or None.
    """
    censoring = view.formula.modifiers.censoring
    if censoring is None or newdata is not None:
        return None
    return np.asarray(censoring.status.data, dtype=np.float64).reshape(-1)


def pp_check(
    fit: Fit,
    type: str = "dens_overlay",
    *,
    ndraws: int | None = None,
    prefix: Prefix = "ppc",
    group: str | None = None,
    x: str | None = None,
    newdata: IntoDataFrame | None = None,
    resp: str | None = None,
    draw_ids: Sequence[int] | None = None,
    stat: StatisticSpec | None = None,
    prob: float = 0.5,
    prob_outer: float = 0.9,
    bins: int | None = None,
    freq: bool = True,
    style: Literal["standing", "hanging", "suspended"] = "standing",
    seed: int = 0,
) -> PPCheck:
    """Prepare a posterior or prior predictive check, as brms ``pp_check``.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model; a prior view (``prior_view``) checks the prior
        predictive distribution.
    type : str
        bayesplot type without prefix (`PPC_TYPES`, or `PPD_TYPES` for
        ``prefix="ppd"``).
    ndraws : int or None
        Random draws; every draw for summary types and ten otherwise.
    prefix : {"ppc", "ppd"}
        Compare with observed data, or show the predictive distribution only.
    group : str or None
        Grouping variable; required by ``*_grouped`` types.
    x : str or None
        Covariate of ``intervals``, ``ribbon`` and ``error_scatter_avg_vs_x``.
    newdata : IntoDataFrame or None
        Rows to predict, holding the response for ``ppc`` types.
    resp : str or None
        Response of a joint fit; required when it has several.
    draw_ids : Sequence[int] or None
        Explicit 0-based draws.
    stat : str or Callable or tuple or None
        Test statistic; ``"mean"`` (``("mean", "sd")`` for ``stat_2d``) by
        default.
    prob, prob_outer : float
        Interval probabilities of interval types; ``prob_outer`` is the
        interval of ``bars`` and ``rootogram``.
    bins : int or None
        Bins of ``error_binned``.
    freq : bool
        Counts instead of proportions for ``bars``.
    style : {"standing", "hanging", "suspended"}
        Rootogram style.
    seed : int
        Seed of the draw selection and predictions.

    Returns
    -------
    PPCheck
        The table behind the plot with the inputs it was computed from.

    Raises
    ------
    ValueError
        If the type is unknown for the prefix, a required ``group`` or ``x``
        is missing or not a data variable, a joint response is ambiguous, or
        no uncensored response remains.

    Notes
    -----
    Cost is one prediction at ``S`` draws over ``N`` rows, plus one
    log-likelihood evaluation and PSIS smoothing, O(S N log S), for
    leave-one-out types.
    """
    if prefix not in ("ppc", "ppd"):
        raise ValueError("prefix must be 'ppc' or 'ppd'")
    valid = set(PPC_TYPES) if prefix == "ppc" else set(PPD_TYPES)
    if type not in valid:
        raise ValueError(
            f"type {type!r} is not a valid {prefix} type; valid types are "
            f"{', '.join(sorted(valid))}"
        )
    if _joint(fit):
        joint = cast("MultivariateFit", fit)
        if resp is None:
            if len(joint.response_names) != 1:
                raise ValueError(
                    "pp_check of a joint fit with several responses requires resp"
                )
            resp = joint.response_names[0]
        elif resp not in joint.response_names:
            raise ValueError(f"unknown response {resp!r}")
    elif resp is not None:
        raise ValueError("resp applies only to joint fits")
    view = _view(fit, resp)
    if view.family.response is not None and view.family.response.kind != "categorical":
        raise ValueError(
            "pp_check is not implemented for multicolumn response families"
        )
    rows = (
        model_frame(fit)
        if newdata is None
        else nw.from_native(newdata, eager_only=True).to_pandas()
    )
    grouped = type.endswith("_grouped")
    if grouped and group is None:
        raise ValueError(f"argument group is required for ppc type {type!r}")
    for name, value in (("group", group), ("x", x)):
        if value is not None and value not in rows.columns:
            raise ValueError(
                f"{name} variable {value!r} could not be found in the data"
            )
    if type == "error_scatter_avg_vs_x" and x is None:
        raise ValueError("error_scatter_avg_vs_x requires x")
    kind = PPC_TYPES[type]
    method: PredictionMethod = (
        "posterior_epred" if type == "error_binned" else "posterior_predict"
    )
    if type == "error_binned" and (
        view.family.ordinal is not None or view.category_labels
    ):
        raise ValueError("error_binned is not available for polytomous models")
    ids = _select_draws(fit, type, ndraws, draw_ids, seed)
    yrep = _predict(fit, resp, method, newdata, ids, seed)
    y = _observed(view, newdata) if prefix == "ppc" else None
    n = yrep.shape[1]
    group_values = (
        None if group is None else np.asarray(rows[group].tolist(), dtype=object)
    )
    x_values = None if x is None else np.asarray(rows[x].tolist(), dtype=object)
    if group_values is not None and group_values.shape[0] != n:
        raise ValueError("group values do not align with the predicted rows")
    keep = np.ones(n, dtype=bool)
    if y is not None:
        keep &= ~torch.isnan(y).numpy()
    status_values = None
    codes = _status(view, newdata)
    if codes is not None:
        if kind == "km":
            keep &= np.isin(codes, (0.0, 1.0))
            status_values = 1.0 - codes
        else:
            keep &= codes == 0.0
    elif kind == "km":
        status_values = np.ones(n)
    if not keep.any():
        raise ValueError("no valid responses found to include")
    yrep_np = yrep.numpy()[:, keep]
    y_np = None if y is None else y.numpy()[keep]
    g = None if group_values is None else group_values[keep]
    xv = None if x_values is None else x_values[keep]
    status_kept = None if status_values is None else status_values[keep]
    log_weights = log_lik = pit = None
    if kind in {"loo_pit", "loo_intervals"}:
        log_lik_full = _log_likelihood(fit, resp, ids, seed)
        log_lik = log_lik_full[:, keep]
        weights = psis_weights(torch.from_numpy(log_lik).unsqueeze(0), r_eff=1.0)
        log_weights = weights.log_weights.reshape(log_lik.shape).double().numpy()
    if kind == "long":
        frame = data_functions.ppc_data(y_np, yrep_np, group=g)
    elif kind == "stat":
        default: StatisticSpec = ("mean", "sd") if type == "stat_2d" else "mean"
        chosen = stat if stat is not None else default
        if type == "stat_2d" and not isinstance(chosen, tuple):
            raise ValueError("stat_2d requires two statistics")
        frame = data_functions.ppc_stat_data(y_np, yrep_np, stat=chosen, group=g)
    elif kind == "intervals":
        frame = data_functions.ppc_intervals_data(
            y_np,
            yrep_np,
            x=None if xv is None else xv.astype(np.float64),
            group=g,
            prob=prob,
            prob_outer=prob_outer,
        )
    else:
        assert y_np is not None
        if kind == "bars":
            frame = data_functions.ppc_bars_data(
                y_np, yrep_np, group=g, prob=prob_outer, freq=freq
            )
        elif kind == "error":
            frame = data_functions.ppc_error_data(y_np, yrep_np, group=g)
        elif kind == "error_avg":
            frame = data_functions.ppc_scatter_avg_data(
                y_np, y_np[None, :] - yrep_np, group=g
            )
        elif kind == "error_avg_x":
            assert xv is not None
            frame = data_functions.ppc_scatter_avg_data(
                xv.astype(np.float64), y_np[None, :] - yrep_np
            )
        elif kind == "error_binned":
            frame = data_functions.ppc_error_binned_data(y_np, yrep_np, bins=bins)
        elif kind == "scatter":
            frame = data_functions.ppc_scatter_data(y_np, yrep_np)
        elif kind == "scatter_avg":
            frame = data_functions.ppc_scatter_avg_data(y_np, yrep_np, group=g)
        elif kind == "rootogram":
            frame = data_functions.ppc_rootogram_data(
                y_np, yrep_np, style=style, prob=prob_outer
            )
        elif kind == "km":
            assert status_kept is not None
            frame = data_functions.km_data(y_np, yrep_np, status=status_kept)
        elif kind == "pit_ecdf":
            pit = data_functions.pit_values(y_np, yrep_np)
            frame = data_functions.pit_ecdf_data(y_np, yrep_np)
        elif kind == "loo_pit":
            assert log_weights is not None
            pit = data_functions.loo_pit_values(y_np, yrep_np, log_weights)
            frame = _loo_pit_frame(pit, qq=type == "loo_pit_qq")
        else:
            assert log_weights is not None and log_lik is not None
            frame = _loo_interval_frame(y_np, yrep_np, log_lik, prob, prob_outer)
    return PPCheck(
        type=type,
        prefix=prefix,
        data=frame,
        y=None if y_np is None else torch.from_numpy(y_np),
        yrep=torch.from_numpy(np.ascontiguousarray(yrep_np)),
        draw_ids=tuple(ids),
        group=None if g is None else tuple(g.tolist()),
        x=None if xv is None else tuple(xv.tolist()),
        status=None if status_kept is None else torch.from_numpy(status_kept),
        log_weights=None if log_weights is None else torch.from_numpy(log_weights),
        log_likelihood=None if log_lik is None else torch.from_numpy(log_lik),
        pit=None
        if pit is None
        else torch.from_numpy(np.asarray(pit, dtype=np.float64)),
    )


def _log_likelihood(
    fit: Fit, resp: str | None, ids: list[int], seed: int
) -> np.ndarray:
    """Return pointwise log-likelihoods ``(S, N)`` at the selected draws.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    resp : str or None
        Joint response.
    ids : list[int]
        0-based pooled draws.
    seed : int
        Seed of variational draws.

    Returns
    -------
    numpy.ndarray
        Log-likelihoods.
    """
    view = fit._pp_view(ids, seed)
    if _joint(fit):
        scores = cast("MultivariateFit", view).log_likelihood(resp=resp)
        value = scores[cast(str, resp)]
    else:
        (value,) = cast("BayesianFit", view).log_likelihood().values()
    return value.reshape(-1, value.shape[-1]).double().numpy()


def _loo_pit_frame(pit: np.ndarray, *, qq: bool) -> pd.DataFrame:
    """Return LOO-PIT values, with uniform quantiles for ``loo_pit_qq``.

    Parameters
    ----------
    pit : numpy.ndarray
        LOO-PIT values.
    qq : bool
        Add the theoretical uniform quantiles of ``geom_qq``
        (``ppoints``) for the sorted values.

    Returns
    -------
    pandas.DataFrame
        ``y_id`` and ``pit``; for ``qq`` the sorted ``sample`` and ``theoretical``.
    """
    frame = pd.DataFrame({"y_id": np.arange(1, pit.shape[0] + 1), "pit": pit})
    if qq:
        n = pit.shape[0]
        a = 3.0 / 8.0 if n <= 10 else 0.5
        frame["sample"] = np.sort(pit)
        frame["theoretical"] = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    return frame


def _loo_interval_frame(
    y: np.ndarray,
    yrep: np.ndarray,
    log_lik: np.ndarray,
    prob: float,
    prob_outer: float,
) -> pd.DataFrame:
    """Return PSIS leave-one-out predictive intervals (``ppc_loo_intervals``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    log_lik : numpy.ndarray
        Pointwise log-likelihoods ``(S, N)``.
    prob, prob_outer : float
        Interval probabilities.

    Returns
    -------
    pandas.DataFrame
        bayesplot's ``y_id``, ``y_obs``, ``x``, ``ll``, ``l``, ``m``, ``h``,
        ``hh``: loo ``E_loo`` weighted quantiles.
    """
    inner, outer = sorted((prob, prob_outer))
    alpha = [(1 - outer) / 2, (1 - inner) / 2]
    probs = (alpha[0], alpha[1], 0.5, 1 - alpha[1], 1 - alpha[0])
    weights = psis_weights(torch.from_numpy(log_lik).unsqueeze(0), r_eff=1.0)
    expectation = loo_expectation(
        torch.from_numpy(yrep).unsqueeze(0), weights, kind="quantile", probs=probs
    )
    values = expectation.table.values.double().numpy().reshape(y.shape[0], -1)
    columns = values[:, -len(probs) :]
    n = y.shape[0]
    return pd.DataFrame(
        {
            "y_id": np.arange(1, n + 1),
            "y_obs": y,
            "x": np.arange(1, n + 1),
            "ll": columns[:, 0],
            "l": columns[:, 1],
            "m": columns[:, 2],
            "h": columns[:, 3],
            "hh": columns[:, 4],
        }
    )


__all__ = [
    "ALL_DRAW_TYPES",
    "ARVIZ_PLOTS",
    "PPC_TYPES",
    "PPD_TYPES",
    "PPCheck",
    "pp_check",
]
