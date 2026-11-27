"""brms ``summary``, ``fixef``, ``ranef``, ``coef``, ``VarCorr`` and ``vcov`` tables.

Every function here consumes draws already renamed to brms parameter names
(`quivers.formulas._postprocess_names.single_variables` or
`MultivariateFit.variables`), so single-response and joint fits share one
implementation. Statistics follow `quivers.diagnostics.posterior_tables`;
convergence columns reuse `quivers.diagnostics.rhat`, `ess` and `mcse`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._numeric import probabilities
from quivers.diagnostics.convergence import ess, mcse, rhat
from quivers.diagnostics.posterior_tables import (
    SummaryTable,
    _percent,
    _statistics,
    posterior_summary,
)
from quivers.formulas._postprocess_names import named_draw_classes

#: Population-coefficient classes of brms (R/summary.R ``fixef_pars``).
_FIXED = re.compile(r"^b(|s|cs|sp|mo|me|mi|m)_")

type DrawKind = Literal["mcmc", "variational", "prior"]
"""Provenance of retained draws."""

type _ThresholdSign = Literal["thres_minus_eta", "eta_minus_thres", "plus"]


class FitSummary(ValueModel):
    """brms's ``summary`` of a fitted model, section by section.

    Parameters
    ----------
    formula : str
        Authored formula text.
    family : str
        Family description.
    nobs : int
        Number of observations used in fitting.
    ngrps : Mapping[str, int]
        Number of levels of each grouping factor.
    algorithm : str
        ``"sampling"`` for recorded chains, ``"meanfield"`` or the guide's
        name for variational fits, ``"prior"`` for prior draws.
    chains, iterations, warmup : int
        Chains, retained draws per chain, and warmup iterations per chain.
    total_ndraws : int
        Retained draws pooled over chains.
    fixed : SummaryTable or None
        Population coefficients, named without their ``b_`` prefix.
    spec_pars : SummaryTable or None
        Unregressed family parameters (``sigma``, ``shape``, ...).
    random : Mapping[str, SummaryTable]
        Per grouping factor, ``sd(<coef>)`` and ``cor(<coef>,<coef>)`` rows.
    splines : SummaryTable or None
        Smoothing standard deviations of ``s()`` and ``t2()`` terms,
        ``sds(<label>_<penalty>)``.
    rescor : SummaryTable or None
        Residual correlations of a joint fit, ``rescor(<a>,<b>)``.
    other : SummaryTable or None
        Native QVR sites without a brms counterpart.
    divergences : int
        Divergent transitions after warmup, zero without recorded chains.
    """

    formula: str
    family: str
    nobs: int
    ngrps: Mapping[str, int] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    algorithm: str
    chains: int
    iterations: int
    warmup: int
    total_ndraws: int
    fixed: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    spec_pars: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    random: Mapping[str, SummaryTable] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    splines: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    rescor: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    other: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    divergences: int = 0


def summary_columns(
    probs: tuple[float, float], *, mcmc: bool, mc_se: bool
) -> tuple[str, ...]:
    """Return brms's ``summary`` column labels.

    Parameters
    ----------
    probs : tuple[float, float]
        Lower and upper interval probabilities.
    mcmc : bool
        Include ``Rhat``, ``Bulk_ESS`` and ``Tail_ESS``.
    mc_se : bool
        Include the ``MCSE`` column.

    Returns
    -------
    tuple[str, ...]
        Column labels in brms's order.
    """
    mass = _percent(100 * (probs[1] - probs[0]))
    return (
        "Estimate",
        *(("MCSE",) if mc_se else ()),
        "Est.Error",
        f"l-{mass}% CI",
        f"u-{mass}% CI",
        *(("Rhat", "Bulk_ESS", "Tail_ESS") if mcmc else ()),
    )


def _section(
    names: Mapping[str, torch.Tensor],
    selected: Sequence[str],
    rows: Sequence[str],
    *,
    probs: tuple[float, float],
    robust: bool,
    mc_se: bool,
    mcmc: bool,
) -> SummaryTable | None:
    """Summarize selected variables with brms's ``summary`` columns.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws on chain and draw axes.
    selected : Sequence[str]
        Variables in output order.
    rows : Sequence[str]
        Printed row labels, one per selected variable.
    probs : tuple[float, float]
        Interval probabilities.
    robust : bool
        Use the median and MAD.
    mc_se : bool
        Add the Monte Carlo standard error of the estimate.
    mcmc : bool
        Add convergence diagnostics for recorded chains.

    Returns
    -------
    SummaryTable or None
        The section, or None when nothing is selected.
    """
    if not selected:
        return None
    rows_out: list[torch.Tensor] = []
    for name in selected:
        draws = names[name].detach().to(torch.float64)
        statistics = _statistics(draws.flatten(0, 1).unsqueeze(-1), probs, robust)[:, 0]
        values = [statistics[0]]
        if mc_se:
            values.append(
                mcse(draws, method="quantile", prob=0.5)
                if robust
                else mcse(draws, method="mean")
            )
        values.extend([statistics[1], statistics[2], statistics[3]])
        if mcmc:
            values.extend(
                [rhat(draws), ess(draws, method="bulk"), ess(draws, method="tail")]
            )
        rows_out.append(torch.stack([value.to(torch.float64) for value in values]))
    return SummaryTable(
        rows=tuple(rows),
        columns=summary_columns(probs, mcmc=mcmc, mc_se=mc_se),
        values=torch.stack(rows_out),
    )


def interval_probs(prob: float) -> tuple[float, float]:
    """Return brms's central interval bounds for a probability mass.

    Parameters
    ----------
    prob : float
        Central mass in the open unit interval.

    Returns
    -------
    tuple[float, float]
        ``((1 - prob) / 2, 1 - (1 - prob) / 2)``.

    Raises
    ------
    ValueError
        If the mass is outside the open unit interval.
    """
    if isinstance(prob, bool) or not math.isfinite(prob) or not 0 < prob < 1:
        raise ValueError("prob must lie strictly between zero and one")
    return ((1 - prob) / 2, 1 - (1 - prob) / 2)


def _paired(name: str, prefix: str) -> str:
    """Rename ``<prefix>__a__b`` to brms's printed ``<label>(a,b)`` form.

    Parameters
    ----------
    name : str
        Variable such as ``cor_g__Intercept__x``.
    prefix : str
        The class prefix including its group, such as ``cor_g``.

    Returns
    -------
    str
        For instance ``cor(Intercept,x)``.
    """
    parts = name[len(prefix) + 2 :].split("__")
    return f"{prefix.split('_', 1)[0]}({','.join(parts)})"


def fit_summary(
    names: Mapping[str, torch.Tensor],
    *,
    dpars: frozenset[str],
    kind: DrawKind,
    algorithm: str,
    formula: str,
    family: str,
    nobs: int,
    ngrps: Mapping[str, int],
    warmup: int,
    divergences: int,
    prob: float,
    robust: bool,
    mc_se: bool,
) -> FitSummary:
    """Assemble brms's ``summary`` sections from brms-named draws.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws on chain and draw axes.
    dpars : frozenset[str]
        Family parameter names that form ``spec_pars``.
    kind : {"mcmc", "variational", "prior"}
        Draw provenance; only recorded chains receive Rhat and ESS columns.
    algorithm : str
        Fitting algorithm label.
    formula, family : str
        Model description.
    nobs : int
        Observation count.
    ngrps : Mapping[str, int]
        Levels per grouping factor.
    warmup : int
        Warmup iterations per chain.
    divergences : int
        Post-warmup divergent transitions.
    prob : float
        Central interval mass.
    robust : bool
        Median and MAD instead of mean and SD.
    mc_se : bool
        Add the Monte Carlo standard error column.

    Returns
    -------
    FitSummary
        Sections in brms's order.

    Raises
    ------
    ValueError
        If no draws are given.
    """
    if not names:
        raise ValueError("summary requires at least one parameter")
    probs = interval_probs(prob)
    first = next(iter(names.values()))
    chains, draws = int(first.shape[0]), int(first.shape[1])
    mcmc = kind == "mcmc"
    options = {"probs": probs, "robust": robust, "mc_se": mc_se, "mcmc": mcmc}
    sections = named_draw_classes(names)
    fixed = sections.get("fixed", ())
    special: list[str] = []
    other: list[str] = []
    for name in sections.get("other", ()):
        base = name.split("_", 1)[0]
        (special if base in dpars or name in dpars else other).append(name)
    random = {
        key.removeprefix("random:"): table
        for key, value in sections.items()
        if key.startswith("random:")
        and (
            table := _section(
                names,
                value,
                [
                    _paired(name, name.split("__", 1)[0])
                    if name.startswith("cor_")
                    else f"sd({name.split('__', 1)[1]})"
                    for name in value
                ],
                **options,
            )
        )
        is not None
    }
    rescor = sections.get("rescor", ())
    splines = sections.get("splines", ())
    return FitSummary(
        formula=formula,
        family=family,
        nobs=nobs,
        ngrps=dict(ngrps),
        algorithm=algorithm,
        chains=chains,
        iterations=draws,
        warmup=warmup,
        total_ndraws=chains * draws,
        fixed=_section(
            names, fixed, [_FIXED.sub("", name) for name in fixed], **options
        ),
        spec_pars=_section(names, special, special, **options),
        random=random,
        splines=_section(
            names, splines, [f"sds({name[4:]})" for name in splines], **options
        ),
        rescor=_section(
            names, rescor, [_paired(name, "rescor") for name in rescor], **options
        ),
        other=_section(names, other, other, **options),
        divergences=divergences,
    )


def fixef_draws(
    names: Mapping[str, torch.Tensor], pars: Sequence[str] | None
) -> dict[str, torch.Tensor]:
    """Select population coefficients under brms's ``fixef`` labels.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws.
    pars : Sequence[str] or None
        Coefficient labels to keep, matched after the class prefix.

    Returns
    -------
    dict[str, torch.Tensor]
        Labels without the ``b_`` prefix mapped to ``(chain, draw)`` draws.
    """
    selected = {
        _FIXED.sub("", name): value
        for name, value in names.items()
        if _FIXED.match(name)
    }
    if pars is not None:
        wanted = set(pars)
        selected = {name: value for name, value in selected.items() if name in wanted}
    return selected


def fixef_table(
    names: Mapping[str, torch.Tensor],
    *,
    probs: Sequence[float],
    robust: bool,
    pars: Sequence[str] | None,
) -> SummaryTable:
    """Summarize population coefficients as brms ``fixef(summary = TRUE)``.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws.
    probs : Sequence[float]
        Quantile probabilities.
    robust : bool
        Median and MAD instead of mean and SD.
    pars : Sequence[str] or None
        Optional coefficient selection.

    Returns
    -------
    SummaryTable
        One row per coefficient.

    Raises
    ------
    ValueError
        If no population coefficient is selected.
    """
    selected = fixef_draws(names, pars)
    if not selected:
        raise ValueError("the model has no selected population-level coefficients")
    return posterior_summary(selected, probs=probabilities(probs), robust=robust)


def level_array(
    coefficients: Sequence[tuple[str, torch.Tensor]],
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Stack per-level coefficient draws into brms's draws x levels x coefs array.

    Parameters
    ----------
    coefficients : Sequence[tuple[str, torch.Tensor]]
        Coefficient labels with ``(chain, draw, level)`` draws.

    Returns
    -------
    tuple[tuple[str, ...], torch.Tensor]
        Labels and an array ``(chain, draw, level, coef)``.
    """
    return tuple(label for label, _ in coefficients), torch.stack(
        [value for _, value in coefficients], dim=-1
    )


def coef_arrays(
    fixed: Mapping[str, torch.Tensor],
    varying: Mapping[str, Sequence[tuple[str, torch.Tensor]]],
    *,
    threshold_signs: Mapping[str, _ThresholdSign],
) -> dict[str, tuple[tuple[str, ...], torch.Tensor]]:
    """Combine population and group coefficients as brms ``coef``.

    Parameters
    ----------
    fixed : Mapping[str, torch.Tensor]
        ``fixef`` labels mapped to ``(chain, draw)`` draws.
    varying : Mapping[str, Sequence[tuple[str, torch.Tensor]]]
        Per group, ``ranef`` labels with ``(chain, draw, level)`` draws.
    threshold_signs : Mapping[str, _ThresholdSign]
        Threshold directions keyed by population coefficient label. A
        cumulative threshold subtracts the group's linear predictor.

    Returns
    -------
    dict[str, tuple[tuple[str, ...], torch.Tensor]]
        Per group, coefficient labels and ``(chain, draw, level, coef)``
        draws: group coefficients first, then population-only ones, which
        repeat over levels. A group-only coefficient has population zero.
    """
    result: dict[str, tuple[tuple[str, ...], torch.Tensor]] = {}

    def stem(label: str) -> str:
        """Drop a trailing threshold index such as ``[2]``.

        Parameters
        ----------
        label : str
            Coefficient label.

        Returns
        -------
        str
            The label before its first ``[``.
        """
        return label.split("[", 1)[0]

    fixed_names = tuple(fixed)
    fixed_stems = {stem(name) for name in fixed_names}
    ranef_names = {label for members in varying.values() for label, _ in members}
    population = dict(fixed)
    missing_fixed = [name for name in ranef_names if name not in fixed]
    template = next(iter(fixed.values()), None)
    for name in missing_fixed:
        if stem(name) in fixed:
            population[name] = fixed[stem(name)]
        elif name not in fixed_stems:
            members = next(
                value
                for values in varying.values()
                for label, value in values
                if label == name
            )
            population[name] = torch.zeros_like(
                members[..., 0] if template is None else template
            )
    removed = {stem(name) for name in missing_fixed}
    population = {
        name: value
        for name, value in population.items()
        if not (name in fixed and name in removed)
    }
    threshold = re.compile(r"(^|_)Intercept\[[0-9]+\]$")
    for group, members in varying.items():
        local = dict(members)
        count = next(iter(local.values())).shape[2]
        missing_varying = [name for name in fixed_names if name not in local]
        local_stems = {stem(name) for name in local}
        added: dict[str, torch.Tensor] = {}
        for name in missing_varying:
            if stem(name) in local:
                added[name] = local[stem(name)]
            elif name not in local_stems:
                added[name] = torch.zeros(
                    *fixed[name].shape[:2], count, dtype=fixed[name].dtype
                )
        dropped = {stem(name) for name in missing_varying}
        combined = {
            name: value for name, value in local.items() if name not in dropped
        } | added
        labels: list[str] = []
        values: list[torch.Tensor] = []
        for name, value in combined.items():
            base = population[name].unsqueeze(-1)
            if threshold.search(name):
                sign = threshold_signs.get(name, "plus")
                value = (
                    base - value
                    if sign == "thres_minus_eta"
                    else value - base
                    if sign == "eta_minus_thres"
                    else base + value
                )
            else:
                value = base + value
            labels.append(name)
            values.append(value)
        result[group] = (tuple(labels), torch.stack(values, dim=-1))
    return result


class VarCorrBlock(ValueModel):
    """Standard deviations, correlations and covariances of one block.

    Parameters
    ----------
    sd : SummaryTable or None
        Summary of the standard deviations, one row per coefficient.
    cor, cov : SummaryTable or None
        Three-way summaries ``(coef, statistic, coef)``; None when the
        block has a single coefficient and no correlation.
    sd_draws : torch.Tensor
        Draws ``(chain, draw, coef)`` of the standard deviations.
    cor_draws, cov_draws : torch.Tensor or None
        Draws ``(chain, draw, coef, coef)``.
    labels : tuple[str, ...]
        Coefficient labels.
    """

    labels: tuple[str, ...]
    sd: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    cor: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    cov: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    sd_draws: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    cor_draws: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    cov_draws: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def varcorr_block(
    sd: Sequence[tuple[str, torch.Tensor]],
    cor: Mapping[tuple[str, str], torch.Tensor] | None,
    *,
    summary: bool,
    probs: Sequence[float],
    robust: bool,
) -> VarCorrBlock:
    """Build one ``VarCorr`` block from scale and correlation draws.

    Parameters
    ----------
    sd : Sequence[tuple[str, torch.Tensor]]
        Coefficient labels with ``(chain, draw)`` standard deviations.
    cor : Mapping[tuple[str, str], torch.Tensor] or None
        Pairwise correlation draws; absent pairs are zero, as in brms.
        None when the block estimates no correlation.
    summary : bool
        Also compute summary tables.
    probs : Sequence[float]
        Quantile probabilities.
    robust : bool
        Median and MAD instead of mean and SD.

    Returns
    -------
    VarCorrBlock
        Draws, and summaries when requested.
    """
    labels = tuple(label for label, _ in sd)
    scales = torch.stack([value for _, value in sd], dim=-1).to(torch.float64)
    levels = probabilities(probs)
    correlation = covariance = None
    if cor is not None:
        size = len(labels)
        correlation = torch.zeros(*scales.shape, size, dtype=torch.float64)
        for k in range(size):
            correlation[..., k, k] = 1.0
        for (a, b), value in cor.items():
            j, k = labels.index(a), labels.index(b)
            correlation[..., j, k] = correlation[..., k, j] = value.to(torch.float64)
        covariance = scales.unsqueeze(-1) * correlation * scales.unsqueeze(-2)
    if not summary:
        return VarCorrBlock(
            labels=labels,
            sd_draws=scales,
            cor_draws=correlation,
            cov_draws=covariance,
        )
    return VarCorrBlock(
        labels=labels,
        sd=posterior_summary(scales, probs=levels, robust=robust, labels=labels),
        cor=posterior_summary(
            correlation, probs=levels, robust=robust, labels=labels, components=labels
        )
        if correlation is not None
        else None,
        cov=posterior_summary(
            covariance, probs=levels, robust=robust, labels=labels, components=labels
        )
        if covariance is not None
        else None,
        sd_draws=scales,
        cor_draws=correlation,
        cov_draws=covariance,
    )


def group_varcorr(
    names: Mapping[str, torch.Tensor],
) -> dict[
    str,
    tuple[list[tuple[str, torch.Tensor]], dict[tuple[str, str], torch.Tensor] | None],
]:
    """Collect group scales and correlations from brms-named draws.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws with ``sd_<g>__<coef>`` and ``cor_<g>__<a>__<b>``.

    Returns
    -------
    dict
        Per group, the scale list and the correlation pairs (None when the
        group estimates no correlation).
    """
    result: dict[
        str,
        tuple[
            list[tuple[str, torch.Tensor]], dict[tuple[str, str], torch.Tensor] | None
        ],
    ] = {}
    for name, value in names.items():
        if name.startswith("sd_") and "__" in name:
            group, coef = name[3:].split("__", 1)
            result.setdefault(group, ([], None))[0].append((coef, value))
    for name, value in names.items():
        if name.startswith("cor_") and name.count("__") == 2:
            prefix, a, b = name.split("__")
            group = prefix[4:]
            if group not in result:
                continue
            scales, pairs = result[group]
            if pairs is None:
                pairs = {}
                result[group] = (scales, pairs)
            pairs[(a, b)] = value
    return result


def vcov_table(
    names: Mapping[str, torch.Tensor],
    *,
    correlation: bool,
    pars: Sequence[str] | None,
) -> SummaryTable:
    """Return brms's ``vcov``: the draw covariance of population coefficients.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws.
    correlation : bool
        Return the correlation matrix instead of the covariance.
    pars : Sequence[str] or None
        Optional coefficient selection.

    Returns
    -------
    SummaryTable
        Square matrix with coefficient labels on both axes; the sample
        covariance divides by S - 1, as R's ``cov``.

    Raises
    ------
    ValueError
        If no population coefficient is selected.
    """
    selected = fixef_draws(names, pars)
    if not selected:
        raise ValueError("the model has no selected population-level coefficients")
    labels = tuple(selected)
    matrix = torch.stack(
        [value.detach().to(torch.float64).flatten() for value in selected.values()]
    )
    result = torch.cov(matrix)
    if correlation:
        result = torch.corrcoef(matrix)
    return SummaryTable(
        rows=labels, columns=labels, values=result.reshape(len(labels), -1)
    )


__all__ = [
    "FitSummary",
    "VarCorrBlock",
    "coef_arrays",
    "fit_summary",
    "fixef_draws",
    "fixef_table",
    "group_varcorr",
    "interval_probs",
    "level_array",
    "summary_columns",
    "varcorr_block",
    "vcov_table",
]
