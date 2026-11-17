"""Typed expected-log-predictive-density estimates and their comparisons.

`ELPDEstimate` is the counterpart of a ``loo`` object of the R ``loo``
package [@vehtari2017loo] as brms stores it: one estimate of the
expected log pointwise predictive density (ELPD) from PSIS-LOO, WAIC,
k-fold cross-validation, or subsampled PSIS-LOO, with its pointwise terms
and diagnostics. ArviZ computes PSIS, WAIC weights, comparisons, and
stacking weights; this module converts ArviZ's results into retained
tensors and implements what ArviZ lacks: the difference estimator of
subsampled PSIS-LOO [@magnusson2020leave], PSIS-weighted expectations with
precomputed weights, and the leave-one-out R-squared of brms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Literal

import arviz as az
from arviz_stats.utils import ELPDData
import didactic.api as dx
import numpy as np
import torch
import xarray as xr

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics.posterior import waic
from quivers.diagnostics.posterior_tables import SummaryTable

type Criterion = Literal["loo", "waic", "kfold", "loo_subsample"]
"""Predictive criterion of an `ELPDEstimate`."""

#: Row labels of each criterion's estimates table, as the loo package prints them.
_LABELS: dict[str, tuple[str, str, str]] = {
    "loo": ("elpd_loo", "p_loo", "looic"),
    "loo_subsample": ("elpd_loo", "p_loo", "looic"),
    "waic": ("elpd_waic", "p_waic", "waic"),
    "kfold": ("elpd_kfold", "p_kfold", "kfoldic"),
}


def response_hash(values: Sequence[torch.Tensor]) -> str:
    """Hash observed responses to check that criteria score the same data.

    Parameters
    ----------
    values : Sequence[torch.Tensor]
        Observed response tensors in a fixed order.

    Returns
    -------
    str
        SHA-1 digest of the Float64 values, their shapes, and NaN positions.
        brms stores the same check as a hash of its response data.
    """
    digest = hashlib.sha1()
    for value in values:
        array = value.detach().to(torch.float64).cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(np.nan_to_num(array, nan=np.inf).tobytes())
        digest.update(np.isnan(array).tobytes())
    return digest.hexdigest()


class ELPDEstimate(ValueModel):
    """One ELPD estimate with pointwise terms, as a brms ``loo`` object.

    Parameters
    ----------
    criterion : {"loo", "waic", "kfold", "loo_subsample"}
        Estimator that produced the values.
    elpd, se_elpd : float
        ELPD estimate and its standard error over observations.
    p, se_p : float
        Effective number of parameters and its standard error.
    pointwise_elpd, pointwise_p : torch.Tensor
        Float64 terms per scored unit. Subsampled estimates hold exact
        terms on the subsample and NaN elsewhere.
    pareto_k : torch.Tensor or None
        Pareto shape diagnostic per unit for PSIS estimates (NaN where not
        computed), None for WAIC and k-fold.
    num_samples : int
        Pooled posterior draws behind the estimate.
    good_k : float or None
        Pareto shape threshold, ``min(1 - 1 / log10(S), 0.7)``.
    responses : tuple[str, ...]
        Scored response names.
    response_hash : str
        Hash of the scored responses, checked before comparisons.
    moment_matched : tuple[int, ...]
        Units whose terms were improved by moment matching.
    refitted : tuple[int, ...]
        Units whose terms come from exact refits.
    subsample : tuple[int, ...]
        Units scored exactly by subsampled PSIS-LOO.
    subsampling_se, subsampling_se_p : float or None
        Standard errors of the ELPD and of ``p`` from subsampling alone.
    approximation : torch.Tensor or None
        Approximate pointwise ELPD of every unit for subsampled PSIS-LOO.
    approximation_method : str or None
        ``"plpd"`` or ``"lpd"`` for subsampled PSIS-LOO.
    folds : tuple[int, ...]
        Zero-based fold of every observation for k-fold estimates.
    joint : str or None
        k-fold predictive unit: ``"obs"``, ``"fold"`` or ``"group"``.
    """

    criterion: Criterion
    elpd: float
    se_elpd: float
    p: float
    se_p: float
    pointwise_elpd: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    pointwise_p: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    pareto_k: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    num_samples: int
    good_k: float | None = None
    responses: tuple[str, ...]
    response_hash: str
    moment_matched: tuple[int, ...] = ()
    refitted: tuple[int, ...] = ()
    subsample: tuple[int, ...] = ()
    subsampling_se: float | None = None
    subsampling_se_p: float | None = None
    approximation: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    approximation_method: str | None = None
    folds: tuple[int, ...] = ()
    joint: str | None = None

    @property
    def num_observations(self) -> int:
        """Number of scored units.

        Returns
        -------
        int
            Length of the pointwise terms.
        """
        return int(self.pointwise_elpd.numel())

    @property
    def ic(self) -> float:
        """Deviance-scale criterion ``-2 * elpd`` (looic, waic, kfoldic).

        Returns
        -------
        float
            The information criterion.
        """
        return -2 * self.elpd

    @property
    def se_ic(self) -> float:
        """Standard error of the deviance-scale criterion.

        Returns
        -------
        float
            ``2 * se_elpd``.
        """
        return 2 * self.se_elpd

    def estimates(self) -> SummaryTable:
        """Return the loo package's estimates table.

        Returns
        -------
        SummaryTable
            Rows ``elpd_<c>``, ``p_<c>``, ``<c>ic`` and columns
            ``Estimate``, ``SE`` (and ``subsampling SE`` for subsampled
            estimates).
        """
        rows = _LABELS[self.criterion]
        values = [[self.elpd, self.se_elpd], [self.p, self.se_p], [self.ic, self.se_ic]]
        columns: tuple[str, ...] = ("Estimate", "SE")
        if self.criterion == "loo_subsample":
            assert self.subsampling_se is not None
            columns = (*columns, "subsampling SE")
            values[0].append(self.subsampling_se)
            assert self.subsampling_se_p is not None
            values[1].append(self.subsampling_se_p)
            values[2].append(2 * self.subsampling_se)
        return SummaryTable(
            rows=rows,
            columns=columns,
            values=torch.tensor(values, dtype=torch.float64),
        )

    def bad_observations(self, threshold: float | None = None) -> tuple[int, ...]:
        """List units whose Pareto shape exceeds a threshold.

        Parameters
        ----------
        threshold : float or None
            Shape threshold; None uses `good_k`.

        Returns
        -------
        tuple[int, ...]
            Zero-based unit positions, as loo's ``pareto_k_ids``.

        Raises
        ------
        ValueError
            If the estimate has no Pareto diagnostics.
        """
        if self.pareto_k is None:
            raise ValueError(f"{self.criterion} estimates have no Pareto diagnostics")
        limit = self.good_k if threshold is None else threshold
        if limit is None:
            raise ValueError("select a Pareto threshold")
        return tuple(int(i) for i in torch.nonzero(self.pareto_k > limit).flatten())

    def to_arviz(self) -> ELPDData:
        """Convert to ArviZ's ``ELPDData`` for ArviZ comparisons.

        Returns
        -------
        arviz_stats.utils.ELPDData
            The same estimate, pointwise terms and diagnostics.
        """
        pointwise = xr.DataArray(self.pointwise_elpd.cpu().numpy(), dims=("__obs__",))
        pareto = (
            xr.DataArray(self.pareto_k.cpu().numpy(), dims=("__obs__",))
            if self.pareto_k is not None
            else None
        )
        kind = "loo" if self.criterion in {"loo", "loo_subsample"} else self.criterion
        if self.criterion == "kfold":
            kind = "loo_kfold"
        result = ELPDData(
            kind=kind,
            elpd=self.elpd,
            se=self.se_elpd,
            p=self.p,
            n_samples=self.num_samples,
            n_data_points=self.num_observations,
            scale="log",
            warning=bool(
                self.pareto_k is not None
                and self.good_k is not None
                and bool((self.pareto_k > self.good_k).any())
            ),
            good_k=self.good_k if self.good_k is not None else 0.7,
            elpd_i=pointwise,
        )
        # ELPDData declares its optional diagnostics without None in their types.
        setattr(result, "pareto_k", pareto)
        if self.criterion == "loo_subsample":
            assert self.approximation is not None
            setattr(result, "subsampling_se", self.subsampling_se)
            setattr(result, "subsample_size", len(self.subsample))
            setattr(result, "loo_subsample_observations", np.asarray(self.subsample))
            setattr(
                result,
                "elpd_loo_approx",
                xr.DataArray(self.approximation.cpu().numpy(), dims=("__obs__",)),
            )
        if self.criterion == "kfold":
            setattr(result, "n_folds", len(set(self.folds)) if self.folds else None)
        return result


def _standard_error(values: torch.Tensor) -> float:
    """Return ``sqrt(n * var(values))`` over finite pointwise terms.

    Parameters
    ----------
    values : torch.Tensor
        Pointwise terms.

    Returns
    -------
    float
        The loo package's standard error of a sum, NaN for fewer than two units.
    """
    finite = values[~torch.isnan(values)]
    if finite.numel() < 2:
        return math.nan
    return math.sqrt(finite.numel() * float(finite.var(correction=1)))


def pointwise_lpd(log_likelihood: torch.Tensor) -> torch.Tensor:
    """Return the log pointwise predictive density over pooled draws.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Scores ``(chain, draw, unit)``.

    Returns
    -------
    torch.Tensor
        Float64 ``log(mean_s exp(ll_s))`` per unit.
    """
    pooled = log_likelihood.detach().to(torch.float64).flatten(0, 1)
    return torch.logsumexp(pooled, dim=0) - math.log(pooled.shape[0])


def from_arviz(
    result: ELPDData,
    log_likelihood: torch.Tensor,
    *,
    criterion: Criterion,
    responses: tuple[str, ...],
    hashed: str,
    moment_matched: tuple[int, ...] = (),
    refitted: tuple[int, ...] = (),
) -> ELPDEstimate:
    """Convert a pointwise ArviZ PSIS-LOO result to an `ELPDEstimate`.

    Parameters
    ----------
    result : arviz_stats.utils.ELPDData
        Pointwise ArviZ result.
    log_likelihood : torch.Tensor
        The scores ``(chain, draw, unit)`` behind the result, for ``p_i``.
    criterion : {"loo", "waic", "kfold", "loo_subsample"}
        Estimator label.
    responses : tuple[str, ...]
        Scored response names.
    hashed : str
        Response hash.
    moment_matched, refitted : tuple[int, ...]
        Units changed by moment matching or exact refits.

    Returns
    -------
    ELPDEstimate
        The estimate with ``p_i = lpd_i - elpd_i`` and its standard error.

    Raises
    ------
    ValueError
        If the result has no pointwise terms.
    """
    if result.elpd_i is None:
        raise ValueError("an ELPD estimate requires pointwise ArviZ terms")
    pointwise = torch.as_tensor(
        np.asarray(result.elpd_i.values, dtype=np.float64).reshape(-1)
    )
    p_i = pointwise_lpd(log_likelihood).reshape(-1).cpu() - pointwise
    pareto = (
        torch.as_tensor(
            np.asarray(result.pareto_k.values, dtype=np.float64).reshape(-1)
        )
        if result.pareto_k is not None
        else None
    )
    return ELPDEstimate(
        criterion=criterion,
        elpd=float(result.elpd),
        # loo's standard error uses the unbiased variance of the terms;
        # ArviZ's divides by n.
        se_elpd=_standard_error(pointwise),
        p=float(p_i.sum()),
        se_p=_standard_error(p_i),
        pointwise_elpd=pointwise,
        pointwise_p=p_i,
        pareto_k=pareto,
        num_samples=int(result.n_samples),
        good_k=float(result.good_k),
        responses=responses,
        response_hash=hashed,
        moment_matched=moment_matched,
        refitted=refitted,
    )


def waic_estimate(
    log_likelihood: torch.Tensor, *, responses: tuple[str, ...], hashed: str
) -> ELPDEstimate:
    """Estimate WAIC with the loo package's conventions.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Finite scores ``(chain, draw, unit)``.
    responses : tuple[str, ...]
        Scored response names.
    hashed : str
        Response hash.

    Returns
    -------
    ELPDEstimate
        ``elpd_waic_i = lpd_i - p_waic_i`` with ``p_waic_i`` the sample
        variance (divisor S - 1) of the pointwise log-likelihood
        [@vehtari2017loo].
    """
    result = waic(log_likelihood)
    pointwise = result.elpd_i.reshape(-1).to(torch.float64)
    variance = result.variance_i.reshape(-1).to(torch.float64)
    return ELPDEstimate(
        criterion="waic",
        elpd=result.elpd_waic,
        se_elpd=result.se,
        p=result.p_waic,
        se_p=result.se_p_waic,
        pointwise_elpd=pointwise,
        pointwise_p=variance,
        num_samples=result.num_samples,
        responses=responses,
        response_hash=hashed,
    )


class PSISWeights(ValueModel):
    """Pareto smoothed importance weights for leave-one-out, as loo's ``psis``.

    Parameters
    ----------
    log_weights : torch.Tensor
        Float64 normalized smoothed log weights ``(chain, draw, unit)``;
        each unit's weights sum to one.
    pareto_k : torch.Tensor
        Pareto shape diagnostic per unit [@vehtari2024psis].
    r_eff : float
        Relative efficiency used for the tail length.
    good_k : float
        Threshold ``min(1 - 1 / log10(S), 0.7)`` for S pooled draws.
    """

    log_weights: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    pareto_k: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    r_eff: float
    good_k: float


def good_k(num_samples: int) -> float:
    """Return the sample-size dependent Pareto shape threshold.

    Parameters
    ----------
    num_samples : int
        Pooled draw count S.

    Returns
    -------
    float
        ``min(1 - 1 / log10(S), 0.7)`` [@vehtari2024psis].
    """
    return min(1 - 1 / math.log10(num_samples), 0.7) if num_samples > 1 else 0.7


#: Fewest pooled draws for which PSIS has a tail of five importance ratios.
PSIS_MIN_DRAWS = 25


def require_scores(log_likelihood: torch.Tensor, *, criterion: str) -> None:
    """Refuse an importance-sampling criterion over missing pointwise scores.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Scores ``(chain, draw, unit)``.
    criterion : str
        Criterion named in the error.

    Raises
    ------
    ValueError
        If a score is missing (``nan``), as the scores of a missing ``mi()``
        response are.
    """
    missing = log_likelihood.isnan().any(dim=0).any(dim=0)
    if bool(missing.any()):
        raise ValueError(
            f"{criterion} requires finite log likelihoods at every posterior "
            f"draw; {int(missing.sum())} of {missing.numel()} units have "
            "missing scores (rows with a missing mi() response). brms "
            "reports NA for them too: score the responses without missing "
            "values through resp, or the complete rows through "
            "log_likelihood(new_data=...)"
        )


def require_psis_draws(count: int) -> None:
    """Refuse Pareto smoothing with too few draws for a tail fit.

    PSIS fits a generalized Pareto distribution to the largest fifth of the
    importance ratios of each unit and needs at least five of them
    [@vehtari2024psis].

    Parameters
    ----------
    count : int
        Pooled posterior draws.

    Raises
    ------
    ValueError
        If fewer than 25 draws are pooled.
    """
    if count < PSIS_MIN_DRAWS:
        raise ValueError(
            "Pareto-smoothed importance sampling fits a generalized Pareto "
            "tail to the largest fifth of the importance ratios and needs at "
            f"least five of them; {count} pooled draws are fewer than "
            f"{PSIS_MIN_DRAWS}, so request more draws"
        )


def psis_weights(log_likelihood: torch.Tensor, *, r_eff: float) -> PSISWeights:
    """Smooth leave-one-out importance ratios with ArviZ's PSIS.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Scores ``(chain, draw, unit)``; the raw log ratios are their negatives.
    r_eff : float
        Relative efficiency of the draws, one for independent draws.

    Returns
    -------
    PSISWeights
        Normalized smoothed log weights and Pareto diagnostics.

    Raises
    ------
    ValueError
        If the scores are not three-dimensional, the efficiency is not
        positive, fewer than 25 draws are pooled, or a score is missing.
    """
    if log_likelihood.ndim != 3:
        raise ValueError("PSIS requires scores shaped (chain, draw, unit)")
    if not r_eff > 0:
        raise ValueError("relative efficiency must be positive")
    require_psis_draws(log_likelihood.shape[0] * log_likelihood.shape[1])
    require_scores(log_likelihood, criterion="PSIS")
    # ArviZ's psislw takes log-likelihoods and smooths their negatives.
    scores = xr.DataArray(
        log_likelihood.detach().to(torch.float64).cpu().numpy(),
        dims=("chain", "draw", "__obs__"),
    )
    log_weights, pareto_k = scores.azstats.psislw(r_eff=r_eff, dim=["chain", "draw"])
    weights = torch.as_tensor(
        np.ascontiguousarray(
            log_weights.transpose("chain", "draw", "__obs__").values, dtype=np.float64
        )
    )
    count = log_likelihood.shape[0] * log_likelihood.shape[1]
    return PSISWeights(
        log_weights=weights,
        pareto_k=torch.as_tensor(np.asarray(pareto_k.values, dtype=np.float64)),
        r_eff=float(r_eff),
        good_k=good_k(count),
    )


class LooExpectation(ValueModel):
    """PSIS-weighted leave-one-out expectations, as brms ``loo_predict``.

    Parameters
    ----------
    table : SummaryTable
        One row per unit and one column per statistic (``mean``, ``var``,
        ``sd``, ``median`` or ``q<100 p>``), with a component axis for
        vector-valued draws.
    pareto_k : torch.Tensor
        Function-specific Pareto shape per unit (and component).
    """

    table: SummaryTable = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    pareto_k: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def loo_expectation(
    draws: torch.Tensor,
    weights: PSISWeights,
    *,
    kind: Literal["mean", "var", "sd", "median", "quantile"],
    probs: Sequence[float] = (0.5,),
    labels: Sequence[str | int] | None = None,
    components: Sequence[str | int] | None = None,
) -> LooExpectation:
    """Compute leave-one-out expectations of draws with ArviZ.

    Parameters
    ----------
    draws : torch.Tensor
        Draws ``(chain, draw, unit)`` or ``(chain, draw, unit, component)``
        aligned with the weights' units.
    weights : PSISWeights
        Leave-one-out weights of the same units.
    kind : {"mean", "var", "sd", "median", "quantile"}
        Weighted statistic. The weighted variance divides by
        ``1 - sum(w^2)`` and quantiles interpolate the weighted cumulative
        distribution, both as loo's ``E_loo``.
    probs : Sequence[float]
        Probabilities for ``kind="quantile"``.
    labels, components : Sequence or None
        Unit and component labels; 0-based positions by default.

    Returns
    -------
    LooExpectation
        Statistics and ArviZ's function-specific Pareto diagnostics.

    Raises
    ------
    ValueError
        If draws and weights disagree or probabilities are invalid.

    Notes
    -----
    Cost is O(S U) for S pooled draws and U units, plus a sort per unit
    for quantiles. The estimates assume that PSIS is reliable for the unit
    (its Pareto shape below `PSISWeights.good_k`).
    """
    if tuple(draws.shape[:3]) != tuple(weights.log_weights.shape):
        raise ValueError("expectation draws must align with the PSIS weights")
    values = draws.detach().to(torch.float64).cpu().numpy()
    vector = values.ndim == 4
    dims = ("chain", "draw", "__obs__", "__component__")[: values.ndim]
    h_draws = xr.DataArray(values, dims=dims)
    log_weights = xr.DataArray(
        weights.log_weights.cpu().numpy(), dims=("chain", "draw", "__obs__")
    )
    pareto = xr.DataArray(weights.pareto_k.cpu().numpy(), dims=("__obs__",))
    if vector:
        log_weights = log_weights.broadcast_like(h_draws)
        pareto = pareto.broadcast_like(h_draws.isel(chain=0, draw=0, drop=True))
    tree = xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset({"h": h_draws}),
            "log_likelihood": xr.Dataset({"h": -log_weights}),
            "posterior": xr.Dataset({"h": h_draws}),
        }
    )
    levels = tuple(probs)
    if kind == "quantile":
        if not levels or any(not 0 < p < 1 for p in levels):
            raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
        # ArviZ supplies the tail diagnostic; the quantiles follow loo's E_loo,
        # which brms uses, rather than ArviZ's effective-size quantile.
        _, khat = az.loo_expectations(
            tree,
            var_name="h",
            kind="quantile",
            probs=list(levels),
            log_weights=log_weights,
            pareto_k=pareto,
        )
        columns = tuple(f"q{_percent_label(p)}" for p in levels)
        weights_array = np.exp(weights.log_weights.cpu().numpy()).reshape(
            -1, values.shape[2]
        )
        flat = values.reshape(-1, *values.shape[2:])
        statistic = np.empty((values.shape[2], len(levels), *values.shape[3:]))
        for unit in range(values.shape[2]):
            for index in np.ndindex(*values.shape[3:]):
                statistic[(unit, slice(None), *index)] = weighted_quantile(
                    flat[(slice(None), unit, *index)], weights_array[:, unit], levels
                )
    else:
        result, khat = az.loo_expectations(
            tree, var_name="h", kind=kind, log_weights=log_weights, pareto_k=pareto
        )
        columns = (kind,)
        statistic = np.expand_dims(np.asarray(result.values), 1)
    rows = tuple(range(values.shape[2])) if labels is None else tuple(labels)
    names = (
        (tuple(range(values.shape[3])) if components is None else tuple(components))
        if vector
        else ()
    )
    return LooExpectation(
        table=SummaryTable(
            rows=rows,
            columns=columns,
            components=names,
            values=torch.as_tensor(np.ascontiguousarray(statistic, dtype=np.float64)),
        ),
        pareto_k=torch.as_tensor(np.asarray(khat.values, dtype=np.float64)),
    )


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probs: Sequence[float]
) -> np.ndarray:
    """Interpolate quantiles of a weighted sample as loo's ``E_loo``.

    Parameters
    ----------
    values : numpy.ndarray
        Draws ``(S,)``.
    weights : numpy.ndarray
        Nonnegative weights ``(S,)``.
    probs : Sequence[float]
        Probabilities in the open unit interval.

    Returns
    -------
    numpy.ndarray
        Linear interpolation of the normalized cumulative weights at each
        probability, from the largest draw below it; R's type-7 quantile
        when all weights are equal.
    """
    if np.all(weights == weights[0]):
        return np.quantile(values, probs)
    order = np.argsort(values, kind="stable")
    x = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    result = np.empty(len(probs))
    for j, prob in enumerate(probs):
        index = int(np.argmax(cumulative >= prob))
        if index == 0:
            result[j] = x[0]
        else:
            low, below = cumulative[index - 1], x[index - 1]
            result[j] = below + (x[index] - below) * (prob - low) / (
                cumulative[index] - low
            )
    return result


def _percent_label(prob: float) -> str:
    """Format ``100 p`` as R's ``as.character`` does.

    Parameters
    ----------
    prob : float
        Probability.

    Returns
    -------
    str
        At most 15 significant digits.
    """
    return format(100 * prob, ".15g")


def loo_r2_draws(
    observed: torch.Tensor,
    loo_mean: torch.Tensor,
    generator: torch.Generator,
    num_draws: int,
) -> torch.Tensor:
    """Draw brms's leave-one-out R-squared by the Bayesian bootstrap.

    Parameters
    ----------
    observed : torch.Tensor
        Observed responses ``(unit,)``.
    loo_mean : torch.Tensor
        Leave-one-out expected responses ``(unit,)``.
    generator : torch.Generator
        Stream of the Dirichlet weights.
    num_draws : int
        Bootstrap draws, the posterior draw count S in brms.

    Returns
    -------
    torch.Tensor
        ``1 - var_w(e) / var_w(y)`` per Dirichlet weight vector, with
        ``e = loo_mean - y`` and weighted variances scaled by ``N / (N - 1)``,
        clipped to ``[-1, 1]`` as brms does [@gelman2019rsquared].

    Raises
    ------
    ValueError
        If fewer than two units are given.
    """
    y = observed.detach().to(torch.float64).reshape(-1)
    error = loo_mean.detach().to(torch.float64).reshape(-1) - y
    count = y.numel()
    if count < 2:
        raise ValueError("loo R-squared requires at least two observations")
    exponential = torch.empty(num_draws, count, dtype=torch.float64).exponential_(
        generator=generator
    )
    weights = exponential / exponential.sum(1, keepdim=True)
    scale = count / (count - 1)
    var_y = scale * ((weights * y**2).sum(1) - (weights * y).sum(1) ** 2)
    var_e = scale * ((weights * error**2).sum(1) - (weights * error).sum(1) ** 2)
    return (1 - var_e / var_y).clamp(-1.0, 1.0)


def subsample_estimate(
    exact: torch.Tensor,
    approximation: torch.Tensor,
    subsample: Sequence[int],
    lpd: torch.Tensor,
    pareto_k: torch.Tensor,
    *,
    num_samples: int,
    responses: tuple[str, ...],
    hashed: str,
    method: Literal["plpd", "lpd"],
) -> ELPDEstimate:
    """Combine exact subsample PSIS-LOO terms with an approximation of all units.

    The difference estimator of [@magnusson2020leave] under simple random
    sampling without replacement estimates the total ELPD as
    ``sum_i a_i + N mean_j (e_j - a_j)`` over subsample units ``j``, with
    ``a`` the approximation and ``e`` the exact PSIS-LOO terms, and its
    variance with the finite-population correction ``1 - m / N``.

    Parameters
    ----------
    exact : torch.Tensor
        Exact PSIS-LOO terms of the subsample, ``(m,)``.
    approximation : torch.Tensor
        Approximate terms of every unit, ``(N,)``.
    subsample : Sequence[int]
        Zero-based subsample units, aligned with ``exact``.
    lpd : torch.Tensor
        Log predictive densities of the subsample, ``(m,)``, for ``p_loo``.
    pareto_k : torch.Tensor
        Pareto shapes of the subsample, ``(m,)``.
    num_samples : int
        Pooled draw count.
    responses : tuple[str, ...]
        Scored response names.
    hashed : str
        Response hash.
    method : {"plpd", "lpd"}
        Approximation behind ``approximation``.

    Returns
    -------
    ELPDEstimate
        The subsampled estimate with total and subsampling-only standard
        errors; ``p_loo`` uses the simple random sampling estimator.

    Raises
    ------
    ValueError
        If the subsample has fewer than two units, repeats a unit, or
        terms are not finite.
    """
    units = tuple(subsample)
    total = approximation.numel()
    m = len(units)
    if m < 2 or len(set(units)) != m or any(not 0 <= u < total for u in units):
        raise ValueError("a subsample needs at least two distinct valid units")
    e = exact.detach().to(torch.float64).reshape(-1).cpu()
    a_all = approximation.detach().to(torch.float64).reshape(-1).cpu()
    index = torch.tensor(units, dtype=torch.long)
    a = a_all[index]
    if not bool(torch.isfinite(e).all() and torch.isfinite(a_all).all()):
        raise ValueError("subsampled PSIS-LOO requires finite pointwise terms")
    difference = e - a
    estimate = float(a_all.sum() + total * difference.mean())
    variance = float(difference.var(correction=1))
    sub_variance = total**2 * (1 - m / total) * variance / m
    # Total variance of the pointwise ELPD distribution (Magnusson et al., 2020).
    sigma2 = (
        float((a_all**2).sum())
        + total * float((e**2 - a**2).mean())
        - (
            (total * float(difference.mean())) ** 2
            - sub_variance
            + 2 * float(a_all.sum()) * estimate
            - float(a_all.sum()) ** 2
        )
        / total
    )
    p_terms = lpd.detach().to(torch.float64).reshape(-1).cpu() - e
    p = total * float(p_terms.mean())
    p_variance = total * float(p_terms.var(correction=1))
    p_sub_variance = total**2 * (1 - m / total) * float(p_terms.var(correction=1)) / m
    pointwise = torch.full((total,), math.nan, dtype=torch.float64)
    pointwise[index] = e
    p_pointwise = torch.full((total,), math.nan, dtype=torch.float64)
    p_pointwise[index] = p_terms
    pareto = torch.full((total,), math.nan, dtype=torch.float64)
    pareto[index] = pareto_k.detach().to(torch.float64).reshape(-1).cpu()
    return ELPDEstimate(
        criterion="loo_subsample",
        elpd=estimate,
        se_elpd=math.sqrt(max(sigma2, 0.0)),
        p=p,
        se_p=math.sqrt(p_variance),
        pointwise_elpd=pointwise,
        pointwise_p=p_pointwise,
        pareto_k=pareto,
        num_samples=num_samples,
        good_k=good_k(num_samples),
        responses=responses,
        response_hash=hashed,
        subsample=units,
        subsampling_se=math.sqrt(sub_variance),
        subsampling_se_p=math.sqrt(p_sub_variance),
        approximation=a_all,
        approximation_method=method,
    )


def kfold_estimate(
    elpd: torch.Tensor,
    lpd: torch.Tensor,
    *,
    num_samples: int,
    folds: tuple[int, ...],
    joint: Literal["obs", "fold", "group"],
    responses: tuple[str, ...],
    hashed: str,
) -> ELPDEstimate:
    """Assemble brms's k-fold criterion from held-out and full-data scores.

    Parameters
    ----------
    elpd : torch.Tensor
        Held-out log predictive density per predictive unit.
    lpd : torch.Tensor
        Full-data log predictive density of the same units.
    num_samples : int
        Pooled draws of the full-data fit.
    folds : tuple[int, ...]
        Zero-based fold of every observation.
    joint : {"obs", "fold", "group"}
        Predictive unit.
    responses : tuple[str, ...]
        Scored response names.
    hashed : str
        Response hash.

    Returns
    -------
    ELPDEstimate
        ``elpd_kfold``, ``p_kfold = lpd - elpd_kfold`` and their standard
        errors ``sqrt(n var)`` over units, as brms's ``kfold``.
    """
    e = elpd.detach().to(torch.float64).reshape(-1).cpu()
    p_i = lpd.detach().to(torch.float64).reshape(-1).cpu() - e
    return ELPDEstimate(
        criterion="kfold",
        elpd=float(e.sum()),
        se_elpd=_standard_error(e),
        p=float(p_i.sum()),
        se_p=_standard_error(p_i),
        pointwise_elpd=e,
        pointwise_p=p_i,
        num_samples=num_samples,
        responses=responses,
        response_hash=hashed,
        folds=folds,
        joint=joint,
    )


def compare_estimates(estimates: Mapping[str, ELPDEstimate]) -> SummaryTable:
    """Rank models by ELPD with the loo package's ``loo_compare`` layout.

    Parameters
    ----------
    estimates : Mapping[str, ELPDEstimate]
        Named estimates of one criterion on the same responses.

    Returns
    -------
    SummaryTable
        Rows ordered from best to worst; columns ``elpd_diff``, ``se_diff``,
        ``elpd_<c>``, ``se_elpd_<c>``, ``p_<c>``, ``se_p_<c>``, ``<c>ic``,
        ``se_<c>ic``. Differences are ``elpd_model - elpd_best`` (zero or
        negative) with paired standard errors ``sqrt(n var(diff_i))`` as loo,
        and ArviZ's difference estimator when an estimate is subsampled.

    Raises
    ------
    ValueError
        If fewer than two models are given, criteria differ, or the models
        score different responses or unit counts.
    """
    if len(estimates) < 2:
        raise ValueError("loo_compare requires at least two models")
    criteria = {estimate.criterion for estimate in estimates.values()}
    kinds = {"loo" if c == "loo_subsample" else c for c in criteria}
    if len(kinds) != 1:
        raise ValueError(f"cannot compare different criteria {sorted(criteria)}")
    if len({e.response_hash for e in estimates.values()}) != 1:
        raise ValueError("models must score the same response values")
    if len({e.num_observations for e in estimates.values()}) != 1:
        raise ValueError("models have different numbers of observations")
    labels = _LABELS[next(iter(criteria))]
    order = sorted(estimates, key=lambda name: -estimates[name].elpd)
    best = estimates[order[0]]
    subsampled = "loo_subsample" in criteria
    table = (
        az.compare(
            {name: estimate.to_arviz() for name, estimate in estimates.items()},
            round_to="none",
        )
        if subsampled
        else None
    )
    rows: list[str] = []
    values: list[list[float]] = []
    for name in order:
        estimate = estimates[name]
        if table is not None:
            # ArviZ's difference estimator for subsampled terms.
            difference = -float(table.loc[name, "elpd_diff"])
            se_difference = float(table.loc[name, "dse"])
        else:
            # loo_compare: paired pointwise differences, sqrt(n var) with n - 1.
            terms = estimate.pointwise_elpd - best.pointwise_elpd
            difference = float(terms.sum())
            se_difference = 0.0 if name == order[0] else _standard_error(terms)
        rows.append(name)
        values.append(
            [
                difference,
                se_difference,
                estimate.elpd,
                estimate.se_elpd,
                estimate.p,
                estimate.se_p,
                estimate.ic,
                estimate.se_ic,
            ]
        )
    return SummaryTable(
        rows=tuple(rows),
        columns=(
            "elpd_diff",
            "se_diff",
            labels[0],
            f"se_{labels[0]}",
            labels[1],
            f"se_{labels[1]}",
            labels[2],
            f"se_{labels[2]}",
        ),
        values=torch.tensor(values, dtype=torch.float64),
    )


def stacking_weights(
    estimates: Mapping[str, ELPDEstimate],
    *,
    method: Literal["stacking", "pseudobma"],
    bayesian_bootstrap: bool,
    seed: int,
) -> dict[str, float]:
    """Return ArviZ's stacking or pseudo-BMA weights, as loo_model_weights.

    Parameters
    ----------
    estimates : Mapping[str, ELPDEstimate]
        PSIS-LOO estimates of the candidate models on the same responses.
    method : {"stacking", "pseudobma"}
        Stacking of predictive distributions [@yao2018stacking] or pseudo-BMA
        weights.
    bayesian_bootstrap : bool
        Regularize pseudo-BMA weights with the Bayesian bootstrap (loo's
        ``BB = TRUE``, its default).
    seed : int
        Seed of the Bayesian bootstrap.

    Returns
    -------
    dict[str, float]
        Normalized weights in input order.

    Raises
    ------
    ValueError
        If the estimates are not comparable PSIS-LOO estimates.
    """
    if any(e.criterion not in {"loo", "loo_subsample"} for e in estimates.values()):
        raise ValueError("stacking and pseudo-BMA weights require PSIS-LOO estimates")
    if len({e.response_hash for e in estimates.values()}) != 1:
        raise ValueError("models must score the same response values")
    arviz_method = (
        "stacking"
        if method == "stacking"
        else "BB-pseudo-BMA"
        if bayesian_bootstrap
        else "pseudo-BMA"
    )
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        table = az.compare(
            {name: estimate.to_arviz() for name, estimate in estimates.items()},
            method=arviz_method,
            round_to="none",
        )
    finally:
        np.random.set_state(state)
    return {name: float(table.loc[name, "weight"]) for name in estimates}


def information_weights(estimates: Mapping[str, ELPDEstimate]) -> dict[str, float]:
    """Return Akaike-type weights ``exp(-ic_diff / 2)`` as brms ``model_weights``.

    Parameters
    ----------
    estimates : Mapping[str, ELPDEstimate]
        Estimates of one criterion (loo, waic or kfold).

    Returns
    -------
    dict[str, float]
        Normalized weights in input order.

    Raises
    ------
    ValueError
        If criteria differ.
    """
    if len({e.criterion for e in estimates.values()}) != 1:
        raise ValueError("information-criterion weights need one criterion")
    ics = {name: estimate.ic for name, estimate in estimates.items()}
    best = min(ics.values())
    raw = {name: math.exp(-(value - best) / 2) for name, value in ics.items()}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


__all__ = [
    "Criterion",
    "ELPDEstimate",
    "LooExpectation",
    "PSISWeights",
    "compare_estimates",
    "from_arviz",
    "good_k",
    "information_weights",
    "kfold_estimate",
    "loo_expectation",
    "loo_r2_draws",
    "pointwise_lpd",
    "psis_weights",
    "require_psis_draws",
    "require_scores",
    "response_hash",
    "stacking_weights",
    "subsample_estimate",
    "waic_estimate",
    "weighted_quantile",
]
