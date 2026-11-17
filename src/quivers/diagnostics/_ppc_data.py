"""The data behind graphical posterior predictive checks, as bayesplot computes it.

brms ``pp_check`` [@burkner2017brms] hands observed responses ``y`` and
predictive draws ``yrep`` (an ``S x N`` matrix) to a bayesplot ``ppc_*``
function [@gabry2019visualization]. Each plot is drawn from a data frame that
bayesplot computes first: the long table of observed and replicated values,
test statistics of every replicate, predictive quantile intervals, predictive
errors and their binned averages, frequency tables of discrete values, the
rootogram table [@kleiber2016visualizing], the leave-one-out probability
integral transform (LOO-PIT) under PSIS weights, Kaplan-Meier curves, and the
empirical CDF of PIT values [@sailynoja2022graphical]. The functions here
return those tables with bayesplot's column names and row order, so they can
be plotted with any library or checked numerically. Quantiles are R's type 7.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import numpy as np
import pandas as pd

type Statistic = str | Callable[[np.ndarray], float]
"""A test statistic: a name of `STATISTICS` or a function of one vector."""

type GroupValues = Sequence[str | int | float | bool] | np.ndarray
"""Group labels, one per observation."""

type Cell = str | int | float | bool | None
"""One cell of a table row."""

STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
    "sd": lambda v: float(np.std(v, ddof=1)),
    "var": lambda v: float(np.var(v, ddof=1)),
    "min": lambda v: float(np.min(v)),
    "max": lambda v: float(np.max(v)),
    "mad": lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))),
    "IQR": lambda v: float(np.subtract(*np.quantile(v, [0.75, 0.25]))),
    "q05": lambda v: float(np.quantile(v, 0.05)),
    "q95": lambda v: float(np.quantile(v, 0.95)),
}
"""The built-in scalar test statistics ``T(y) -> scalar``, keyed by the name
that `posterior_predictive_check` and the ``ppc_*`` tables accept. They follow
R's conventions: ``sd`` and ``var`` divide by n - 1, ``mad`` is scaled by
1.4826, and quantiles are type 7."""

_Y = "italic(y)"
_YREP = "italic(y)[rep]"


def _statistic(stat: Statistic) -> Callable[[np.ndarray], float]:
    """Resolve a statistic.

    Parameters
    ----------
    stat : str or Callable
        Name or function.

    Returns
    -------
    Callable[[numpy.ndarray], float]
        The function.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    if callable(stat):
        return stat
    if stat not in STATISTICS:
        raise ValueError(
            f"unknown statistic {stat!r}; choices are {sorted(STATISTICS)}"
        )
    return STATISTICS[stat]


def _check(
    y: np.ndarray | None, yrep: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray]:
    """Validate observed values and an ``S x N`` prediction matrix.

    Parameters
    ----------
    y : numpy.ndarray or None
        Observed values ``(N,)``, or None for predictive-distribution plots.
    yrep : numpy.ndarray
        Draws ``(S, N)``.

    Returns
    -------
    tuple[numpy.ndarray or None, numpy.ndarray]
        Float arrays.

    Raises
    ------
    ValueError
        If shapes disagree or values are not finite.
    """
    pred = np.asarray(yrep, dtype=np.float64)
    if pred.ndim != 2 or not np.isfinite(pred).all():
        raise ValueError("predictions must be a finite (draws, observations) matrix")
    if y is None:
        return None, pred
    obs = np.asarray(y, dtype=np.float64).reshape(-1)
    if obs.shape[0] != pred.shape[1] or not np.isfinite(obs).all():
        raise ValueError("y must be finite with one value per prediction column")
    return obs, pred


def _groups(group: GroupValues | None, n: int) -> np.ndarray | None:
    """Validate group labels.

    Parameters
    ----------
    group : Sequence or numpy.ndarray or None
        Labels.
    n : int
        Observations.

    Returns
    -------
    numpy.ndarray or None
        Object array of labels.

    Raises
    ------
    ValueError
        If the length differs from the observations.
    """
    if group is None:
        return None
    labels = np.asarray(group, dtype=object).reshape(-1)
    if labels.shape[0] != n:
        raise ValueError("group must have one label per observation")
    return labels


def _factor_levels(labels: np.ndarray) -> list:
    """Return R's ``levels(factor(x))``: sorted distinct values.

    Parameters
    ----------
    labels : numpy.ndarray
        Values.

    Returns
    -------
    list
        Sorted distinct values.
    """
    return sorted(set(labels.tolist()), key=lambda v: (str(type(v)), v))


def _rep_label(i: int, kind: str = "rep") -> str:
    """Return bayesplot's replicate label.

    Parameters
    ----------
    i : int
        1-based replicate.
    kind : str
        ``rep`` or ``pred``.

    Returns
    -------
    str
        ``italic(y)[rep] ( i )``.
    """
    return f"italic(y)[{kind}] ( {i} )"


def _melt(pred: np.ndarray, kind: str = "rep") -> pd.DataFrame:
    """Melt predictions as bayesplot ``melt_predictions`` (replicate fastest).

    Parameters
    ----------
    pred : numpy.ndarray
        Draws ``(S, N)``.
    kind : str
        Label kind.

    Returns
    -------
    pandas.DataFrame
        ``y_id``, ``y_name``, ``rep_id``, ``rep_label``, ``value``.
    """
    s, n = pred.shape
    y_id = np.repeat(np.arange(1, n + 1), s)
    rep_id = np.tile(np.arange(1, s + 1), n)
    return pd.DataFrame(
        {
            "y_id": y_id,
            "y_name": [str(i) for i in y_id],
            "rep_id": rep_id.astype(float),
            "rep_label": [_rep_label(i, kind) for i in rep_id],
            "value": pred.T.reshape(-1),
        }
    )


def ppc_data(
    y: np.ndarray | None, yrep: np.ndarray, *, group: GroupValues | None = None
) -> pd.DataFrame:
    """Return the long table of observed and replicated values (``ppc_data``).

    Parameters
    ----------
    y : numpy.ndarray or None
        Observed values, or None for ``ppd_data``.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    group : Sequence or None
        Group label per observation.

    Returns
    -------
    pandas.DataFrame
        Replicated rows, replicate fastest, then the observed rows, with
        ``is_y`` and ``is_y_label`` when ``y`` is given; ``group`` first.
    """
    obs, pred = _check(y, yrep)
    labels = _groups(group, pred.shape[1])
    if obs is None:
        data = _melt(pred, "pred")
    else:
        data = _melt(pred)
        n = obs.shape[0]
        observed = pd.DataFrame(
            {
                "y_id": np.arange(1, n + 1),
                "y_name": [str(i) for i in range(1, n + 1)],
                "rep_id": np.full(n, np.nan),
                "rep_label": _Y,
                "value": obs,
            }
        )
        data = pd.concat([data, observed], ignore_index=True)
        data["is_y"] = data["rep_id"].isna()
        data["is_y_label"] = np.where(data["is_y"], _Y, _YREP)
        data = pd.DataFrame(
            data[
                ["y_id", "y_name", "rep_id", "rep_label", "is_y", "is_y_label", "value"]
            ]
        )
    if labels is not None:
        data.insert(0, "group", labels[data["y_id"].to_numpy() - 1])
    return data


def ppc_stat_data(
    y: np.ndarray | None,
    yrep: np.ndarray,
    *,
    stat: Statistic | tuple[Statistic, Statistic] = "mean",
    group: GroupValues | None = None,
) -> pd.DataFrame:
    """Return test statistics of the observed and each replicated data set.

    Parameters
    ----------
    y : numpy.ndarray or None
        Observed values, or None for ``ppd_stat_data``.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    stat : str or Callable or tuple
        One statistic, or two for ``stat_2d``.
    group : Sequence or None
        Group label per observation; statistics are computed within groups.

    Returns
    -------
    pandas.DataFrame
        ``group`` (when given), ``variable`` (``y``, ``yrep_<s>`` or
        ``ypred_<s>``), ``value`` and, for two statistics, ``value2``; rows by
        sorted group, then variable.
    """
    obs, pred = _check(y, yrep)
    labels = _groups(group, pred.shape[1])
    functions = (
        (_statistic(stat[0]), _statistic(stat[1]))
        if isinstance(stat, tuple)
        else (_statistic(stat),)
    )
    kind = "yrep" if obs is not None else "ypred"
    series = ([("y", obs)] if obs is not None else []) + [
        (f"{kind}_{s + 1}", pred[s]) for s in range(pred.shape[0])
    ]
    levels = [None] if labels is None else _factor_levels(labels)
    rows = []
    for level in levels:
        mask = slice(None) if labels is None else labels == level
        for name, values in series:
            row: dict[str, Cell] = {} if labels is None else {"group": level}
            row["variable"] = name
            row["value"] = functions[0](values[mask])
            if len(functions) == 2:
                row["value2"] = functions[-1](values[mask])
            rows.append(row)
    return pd.DataFrame(rows)


def ppc_intervals_data(
    y: np.ndarray | None,
    yrep: np.ndarray,
    *,
    x: Sequence[float] | np.ndarray | None = None,
    group: GroupValues | None = None,
    prob: float = 0.5,
    prob_outer: float = 0.9,
) -> pd.DataFrame:
    """Return predictive quantile intervals per observation (``ppc_intervals_data``).

    Parameters
    ----------
    y : numpy.ndarray or None
        Observed values, or None for ``ppd_intervals_data``.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    x : Sequence[float] or None
        Horizontal coordinate; the observation index by default.
    group : Sequence or None
        Group label per observation.
    prob, prob_outer : float
        Inner and outer interval probabilities.

    Returns
    -------
    pandas.DataFrame
        ``y_id``, ``y_obs``, ``group``, ``x``, ``outer_width``,
        ``inner_width``, ``ll``, ``l``, ``m``, ``h``, ``hh``.

    Raises
    ------
    ValueError
        If the probabilities are outside (0, 1].
    """
    if not (0 < prob < 1 and 0 < prob_outer <= 1):
        raise ValueError("prob must be in (0, 1) and prob_outer in (0, 1]")
    obs, pred = _check(y, yrep)
    n = pred.shape[1]
    labels = _groups(group, n)
    inner, outer = sorted((prob, prob_outer))
    alpha = [(1 - inner) / 2, (1 - outer) / 2]
    probs = sorted([*alpha, 0.5, *(1 - a for a in alpha)])
    quantiles = np.quantile(pred, probs, axis=0)
    coordinate = np.arange(1, n + 1) if x is None else np.asarray(x, dtype=np.float64)
    data: dict[str, np.ndarray | float] = {"y_id": np.arange(1, n + 1)}
    if obs is not None:
        data["y_obs"] = obs
    if labels is not None:
        data["group"] = labels
    data["x"] = coordinate
    data["outer_width"] = outer
    data["inner_width"] = inner
    for name, values in zip(("ll", "l", "m", "h", "hh"), quantiles, strict=True):
        data[name] = values
    return pd.DataFrame(data)


def predictive_errors(y: np.ndarray, yrep: np.ndarray) -> np.ndarray:
    """Return predictive errors ``y - yrep`` (rstantools ``predictive_error``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.

    Returns
    -------
    numpy.ndarray
        Errors ``(S, N)``.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    return obs[None, :] - pred


def ppc_error_data(
    y: np.ndarray, yrep: np.ndarray, *, group: GroupValues | None = None
) -> pd.DataFrame:
    """Return the long table of predictive errors (``ppc_error_data``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    group : Sequence or None
        Group label per observation.

    Returns
    -------
    pandas.DataFrame
        ``group``, ``y_id``, ``y_name``, ``y_obs``, ``rep_id``,
        ``rep_label``, ``value``.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    data = _melt(obs[None, :] - pred)
    data.insert(2, "y_obs", obs[data["y_id"].to_numpy() - 1])
    labels = _groups(group, pred.shape[1])
    if labels is not None:
        data.insert(0, "group", labels[data["y_id"].to_numpy() - 1])
    return data


def _bins(n: int) -> int:
    """Return bayesplot's default number of bins.

    Parameters
    ----------
    n : int
        Observations.

    Returns
    -------
    int
        ``floor(N / 2)``, 10, or ``floor(sqrt(N))``.
    """
    if n <= 10:
        return n // 2
    if n < 100:
        return 10
    return math.floor(math.sqrt(n))


def _bin_errors(ey: np.ndarray, r: np.ndarray, bins: int, rep_id: int) -> pd.DataFrame:
    """Bin errors by expected value, as bayesplot ``bin_errors``.

    Parameters
    ----------
    ey : numpy.ndarray
        Expected values.
    r : numpy.ndarray
        Errors.
    bins : int
        Requested bins.
    rep_id : int
        Replicate.

    Returns
    -------
    pandas.DataFrame
        ``ey_bar``, ``err_bar``, ``se2``, ``bin``, ``rep_id`` per bin.
    """
    n = ey.shape[0]
    break_ids = [math.floor(n * i / bins) for i in range(1, bins)]
    if any(b == 0 for b in break_ids):
        bins = 1
    if bins == 1:
        breaks = [-math.inf, (ey.min() + ey.max()) / 2, math.inf]
    else:
        ordered = np.sort(ey)
        breaks = [-math.inf]
        for b in break_ids:
            low, high = float(ordered[b - 1]), float(ordered[b])
            if high - low == 0:
                low = -math.inf if low == ey.min() else float(ey[ey < low].max())
            breaks.append((low + high) / 2)
        breaks = list(dict.fromkeys([*breaks, math.inf]))
    edges = np.asarray(breaks)
    binned = np.searchsorted(edges, ey, side="left")
    rows = []
    for i in range(1, len(breaks)):
        mark = binned == i
        count = int(mark.sum())
        sd = float(np.std(r[mark], ddof=1)) if count > 1 else 0.0
        rows.append(
            {
                "ey_bar": float(np.mean(ey[mark])) if count else math.nan,
                "err_bar": float(np.mean(r[mark])) if count else math.nan,
                "se2": 2 * sd / math.sqrt(count) if count else math.nan,
                "bin": float(i),
                "rep_id": rep_id,
            }
        )
    return pd.DataFrame(rows)


def ppc_error_binned_data(
    y: np.ndarray, epred: np.ndarray, *, bins: int | None = None
) -> pd.DataFrame:
    """Return binned errors per draw (``ppc_error_binned``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    epred : numpy.ndarray
        Expected values ``(S, N)`` (brms uses ``posterior_epred``).
    bins : int or None
        Bins; bayesplot's default by sample size.

    Returns
    -------
    pandas.DataFrame
        ``ey_bar``, ``err_bar``, ``se2``, ``bin``, ``rep_id``.
    """
    obs, pred = _check(y, epred)
    assert obs is not None
    count = _bins(obs.shape[0]) if bins is None else bins
    return pd.concat(
        [
            _bin_errors(pred[s], obs - pred[s], count, s + 1)
            for s in range(pred.shape[0])
        ],
        ignore_index=True,
    )


def ppc_bars_data(
    y: np.ndarray,
    yrep: np.ndarray,
    *,
    group: GroupValues | None = None,
    prob: float = 0.9,
    freq: bool = True,
) -> pd.DataFrame:
    """Return frequency tables of discrete values (``ppc_bars_data``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed whole numbers.
    yrep : numpy.ndarray
        Draws ``(S, N)`` of whole numbers.
    group : Sequence or None
        Group label per observation.
    prob : float
        Interval probability of the replicated counts.
    freq : bool
        Counts; False gives proportions within each data set and group.

    Returns
    -------
    pandas.DataFrame
        ``group`` (when given), ``x``, ``y_obs``, ``l``, ``m``, ``h``. As in
        bayesplot, replicate quantiles are taken over the replicates in
        which a value occurs.

    Raises
    ------
    ValueError
        If values are not whole numbers.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    if not (np.all(obs == np.round(obs)) and np.all(pred == np.round(pred))):
        raise ValueError("bars require discrete y and yrep")
    labels = _groups(group, obs.shape[0])
    levels = [None] if labels is None else _factor_levels(labels)
    lower = (1 - prob) / 2
    probs = sorted([lower, 0.5, 1 - lower])
    rows = []
    for level in levels:
        mask = np.ones(obs.shape[0], bool) if labels is None else labels == level
        observed: dict[float, float] = {}
        values, counts = np.unique(obs[mask], return_counts=True)
        total = counts.sum()
        for v, c in zip(values, counts, strict=True):
            observed[float(v)] = float(c) if freq else float(c / total)
        replicated: dict[float, list[float]] = {}
        for s in range(pred.shape[0]):
            values, counts = np.unique(pred[s, mask], return_counts=True)
            total = counts.sum()
            for v, c in zip(values, counts, strict=True):
                replicated.setdefault(float(v), []).append(
                    float(c) if freq else float(c / total)
                )
        keys = sorted(replicated) + sorted(set(observed) - set(replicated))
        for key in keys:
            row: dict[str, Cell] = {} if labels is None else {"group": level}
            row["x"] = key
            row["y_obs"] = observed.get(key, math.nan)
            if key in replicated:
                l, m, h = np.quantile(replicated[key], probs)
            else:
                l = m = h = math.nan
            row.update({"l": float(l), "m": float(m), "h": float(h)})
            rows.append(row)
    data = pd.DataFrame(rows)
    return data.sort_values("x", kind="stable").reset_index(drop=True)


def ppc_rootogram_data(
    y: np.ndarray,
    yrep: np.ndarray,
    *,
    style: str = "standing",
    prob: float = 0.9,
) -> pd.DataFrame:
    """Return the rootogram table of count data (``ppc_rootogram``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed counts.
    yrep : numpy.ndarray
        Draws ``(S, N)`` of counts.
    style : {"standing", "hanging", "suspended"}
        Rootogram style [@kleiber2016visualizing].
    prob : float
        Interval probability of the replicated square-root frequencies.

    Returns
    -------
    pandas.DataFrame
        ``xpos`` (0 to the largest count), ``ypos``, ``ty`` (square-root
        observed frequency), ``tyexp`` (square root of the mean replicated
        frequency), ``tylower``, ``tyupper``.

    Raises
    ------
    ValueError
        If values are not nonnegative whole numbers or the style is unknown.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    if style not in {"standing", "hanging", "suspended"}:
        raise ValueError("style must be standing, hanging, or suspended")
    if (
        np.any(obs < 0)
        or np.any(pred < 0)
        or not (np.all(obs == np.round(obs)) and np.all(pred == np.round(pred)))
    ):
        raise ValueError("rootograms require counts")
    top = int(max(obs.max(), pred.max()))
    xpos = np.arange(top + 1)
    table = np.stack(
        [
            np.bincount(pred[s].astype(int), minlength=top + 1)
            for s in range(pred.shape[0])
        ]
    ).astype(float)
    tyexp = np.sqrt(table.mean(axis=0))
    alpha = (1 - prob) / 2
    lower, upper = np.sqrt(np.quantile(table, [alpha, 1 - alpha], axis=0))
    ty = np.sqrt(np.bincount(obs.astype(int), minlength=top + 1).astype(float))
    if style == "suspended":
        ty = tyexp - ty
    ypos = ty / 2
    if style == "hanging":
        ypos = tyexp - ypos
    return pd.DataFrame(
        {
            "xpos": xpos,
            "ypos": ypos,
            "ty": ty,
            "tyexp": tyexp,
            "tylower": lower,
            "tyupper": upper,
        }
    )


def ppc_scatter_data(y: np.ndarray, yrep: np.ndarray) -> pd.DataFrame:
    """Return observed values against each replicate (``ppc_scatter_data``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.

    Returns
    -------
    pandas.DataFrame
        ``y_id``, ``y_name``, ``y_obs``, ``rep_id``, ``rep_label``, ``value``.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    data = _melt(pred)
    data.insert(2, "y_obs", obs[data["y_id"].to_numpy() - 1])
    return data


def ppc_scatter_avg_data(
    y: np.ndarray, yrep: np.ndarray, *, group: GroupValues | None = None
) -> pd.DataFrame:
    """Return observed values against the mean replicate (``ppc_scatter_avg_data``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values (or a covariate for ``error_scatter_avg_vs_x``).
    yrep : numpy.ndarray
        Draws ``(S, N)`` (or predictive errors).
    group : Sequence or None
        Group label per observation.

    Returns
    -------
    pandas.DataFrame
        ``group``, ``y_id``, ``y_name``, ``y_obs``, ``rep_id`` (missing),
        ``rep_label``, ``value`` (the mean over draws).
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    data = ppc_scatter_data(obs, pred.mean(axis=0, keepdims=True))
    data["rep_id"] = np.nan
    data["rep_label"] = "mean(italic(y)[rep]))"
    labels = _groups(group, obs.shape[0])
    if labels is not None:
        data.insert(0, "group", labels[data["y_id"].to_numpy() - 1])
    return data


def loo_pit_values(
    y: np.ndarray, yrep: np.ndarray, log_weights: np.ndarray
) -> np.ndarray:
    """Return LOO-PIT values from PSIS log weights (rstantools ``loo_pit``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    log_weights : numpy.ndarray
        Normalized PSIS log weights ``(S, N)``.

    Returns
    -------
    numpy.ndarray
        :math:`\\sum_s w_{s i} 1[y^{rep}_{s i} \\le y_i]` per observation.

    Raises
    ------
    ValueError
        If the weights do not match the draws.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    weights = np.asarray(log_weights, dtype=np.float64)
    if weights.shape != pred.shape:
        raise ValueError("log weights must match the predictions")
    masked = np.where(pred <= obs[None, :], weights, -np.inf)
    top = masked.max(axis=0)
    finite = np.isfinite(top)
    out = np.zeros(obs.shape[0])
    out[finite] = np.exp(
        top[finite] + np.log(np.exp(masked[:, finite] - top[finite]).sum(axis=0))
    )
    return out


def km_data(
    y: np.ndarray, yrep: np.ndarray, *, status: Sequence[float] | np.ndarray
) -> pd.DataFrame:
    """Return Kaplan-Meier curves of observed and replicated data (``ppc_km_overlay``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed times.
    yrep : numpy.ndarray
        Draws ``(S, N)``, all treated as events.
    status : Sequence[float]
        Event indicator of each observed time (one event, zero right censored).

    Returns
    -------
    pandas.DataFrame
        ``strata`` (``rep_label=italic(y)`` first), ``time``, ``n_risk``,
        ``n_event``, ``n_censor``, ``surv`` at each distinct time, as
        ``survival::survfit``.

    Raises
    ------
    ValueError
        If a status is not zero or one.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    events = np.asarray(status, dtype=np.float64).reshape(-1)
    if events.shape != obs.shape or not np.isin(events, (0.0, 1.0)).all():
        raise ValueError("status must hold 0 or 1 for every observation")
    curves = [(f"rep_label={_Y}", obs, events)] + [
        (f"rep_label={_rep_label(s + 1)}", pred[s], np.ones(pred.shape[1]))
        for s in range(pred.shape[0])
    ]
    rows = []
    for name, times, indicator in curves:
        survival = 1.0
        for t in np.unique(times):
            at = times == t
            risk = int((times >= t).sum())
            died = int(indicator[at].sum())
            censored = int(at.sum()) - died
            survival *= 1.0 - died / risk
            rows.append(
                {
                    "strata": name,
                    "time": float(t),
                    "n_risk": risk,
                    "n_event": died,
                    "n_censor": censored,
                    "surv": survival,
                }
            )
    return pd.DataFrame(rows)


def pit_values(y: np.ndarray, yrep: np.ndarray) -> np.ndarray:
    """Return posterior predictive PIT values, as bayesplot ``ppc_pit_ecdf``.

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.

    Returns
    -------
    numpy.ndarray
        The share of draws with :math:`y^{rep}_{s i} \\le y_i`, per observation.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    return (obs[None, :] >= pred).mean(axis=0)


def pit_ecdf_data(
    y: np.ndarray, yrep: np.ndarray, *, K: int | None = None
) -> pd.DataFrame:
    """Return PIT values' empirical CDF on a uniform grid (``ppc_pit_ecdf``).

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yrep : numpy.ndarray
        Draws ``(S, N)``.
    K : int or None
        Grid size; ``min(S + 1, N)`` by default.

    Returns
    -------
    pandas.DataFrame
        ``x`` (``k / K``), ``grid`` (``seq(0, 1, length.out = K)``) and
        ``ecdf`` of the PIT values :math:`\\Pr(y^{rep}_i \\le y_i)` at the grid.
        bayesplot's simultaneous bands (``adjust_gamma``) are not included;
        ArviZ's ``plot_ppc_pit`` draws its own simultaneous bands.
    """
    obs, pred = _check(y, yrep)
    assert obs is not None
    pit = pit_values(obs, pred)
    size = min(pred.shape[0] + 1, obs.shape[0]) if K is None else K
    grid = np.linspace(0.0, 1.0, size)
    ecdf = (pit[None, :] <= grid[:, None]).mean(axis=1)
    return pd.DataFrame(
        {"x": np.arange(1, size + 1) / size, "grid": grid, "ecdf": ecdf}
    )


__all__ = [
    "STATISTICS",
    "km_data",
    "loo_pit_values",
    "pit_ecdf_data",
    "pit_values",
    "ppc_bars_data",
    "ppc_data",
    "ppc_error_binned_data",
    "ppc_error_data",
    "ppc_intervals_data",
    "ppc_rootogram_data",
    "ppc_scatter_avg_data",
    "ppc_scatter_data",
    "ppc_stat_data",
    "predictive_errors",
]
