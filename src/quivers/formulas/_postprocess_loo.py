"""Leave-one-out and k-fold criteria of formula fits, as brms computes them.

brms delegates these criteria to the R ``loo`` package [@vehtari2017loo].
Here ArviZ computes PSIS-LOO, moment matching [@paananen2021implicitly],
WAIC weights, comparisons and stacking; `quivers.diagnostics.elpd` holds the
estimators ArviZ lacks. This module adapts single-response and joint fits:
it collects their pointwise log-likelihoods, evaluates them at moved
posterior draws for moment matching, scores subsamples of rows, and runs
k-fold refits through the fits' existing refit callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import math
from typing import TYPE_CHECKING, Literal, cast

import arviz as az
from arviz_stats.utils import ELPDData
import didactic.api as dx
import narwhals as nw
import numpy as np
import torch
import xarray as xr
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics.cross_validation import (
    CrossValidationResult,
    FoldPlan,
    cross_validate,
    fold_plan,
)
from quivers.diagnostics.elpd import (
    ELPDEstimate,
    PSISWeights,
    from_arviz,
    good_k,
    kfold_estimate,
    pointwise_lpd,
    psis_weights,
    require_psis_draws,
    require_scores,
    response_hash,
    subsample_estimate,
    waic_estimate,
)
from quivers.diagnostics.posterior import PosteriorAnalysis
from quivers.formulas._group_prediction import NewLevelPolicy
from quivers.formulas._autocor import autocor_binding, autocor_joint
from quivers.inference.mcmc.kernel import PotentialFn
from quivers.inference.prior import PriorDraws
from quivers.inference.registry import LatentRegistry

# Pathfinder draws report variational provenance.
from quivers.inference.pathfinder import draw_provenance
from quivers.formulas._fit_types import Fit

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit

type FoldSpec = (
    Literal["random", "stratified", "grouped", "loo"] | Sequence[int | str] | None
)
"""brms ``folds``: a fold type, an explicit fold label per row, or None."""

type JointUnit = Literal["obs", "fold", "group"]
"""brms ``joint``: the k-fold predictive unit."""


class CriterionScores(ValueModel):
    """Pointwise scores of one criterion request.

    Parameters
    ----------
    analysis : PosteriorAnalysis
        Retained draws, scores and outcomes.
    name : str
        Log-likelihood variable of the scored responses.
    responses : tuple[str, ...]
        Scored response names.
    hashed : str
        Hash of the scored responses.
    """

    analysis: PosteriorAnalysis = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    name: str
    responses: tuple[str, ...]
    hashed: str

    @property
    def log_likelihood(self) -> torch.Tensor:
        """Return the scores ``(chain, draw, unit)``.

        Returns
        -------
        torch.Tensor
            Pointwise log-likelihoods.
        """
        return self.analysis.log_likelihood[self.name]


def criterion_scores(
    fit: Fit,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    num_samples: int | None,
    seed: int,
) -> CriterionScores:
    """Collect the pointwise log-likelihoods a criterion scores.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection, with the units of
        `MultivariateFit.log_likelihood`; must be None for a single response.
    num_samples : int or None
        Draws per chain.
    seed : int
        Seed for variational draws.

    Returns
    -------
    CriterionScores
        Scores, response names and response hash.

    Raises
    ------
    ValueError
        If ``resp`` is given for a single-response fit, or scores have more
        than one observation axis.
    """
    if joint:
        joint_fit = cast("MultivariateFit", fit)
        analysis = joint_fit.analysis(resp=resp, num_samples=num_samples, seed=seed)
        responses = joint_fit._selection(resp)
    else:
        if resp is not None:
            raise ValueError("resp applies only to joint (multivariate) fits")
        analysis = cast("BayesianFit", fit).analysis(num_samples=num_samples, seed=seed)
        responses = tuple(analysis.log_likelihood)
    (name,) = tuple(analysis.log_likelihood)
    scores = analysis.log_likelihood[name]
    if scores.ndim != 3:
        raise ValueError("criteria require one observation axis per scored unit")
    return CriterionScores(
        analysis=analysis,
        name=name,
        responses=responses,
        hashed=response_hash(list(analysis.observed_data.values())),
    )


def relative_efficiency(analysis: PosteriorAnalysis) -> float:
    """Estimate ArviZ's relative efficiency of the retained draws.

    Parameters
    ----------
    analysis : PosteriorAnalysis
        Retained draws.

    Returns
    -------
    float
        One for a single chain, variational or prior draws; otherwise the
        mean ESS of the posterior means divided by the pooled draw count, the
        value ArviZ's ``loo`` uses.
    """
    first = next(iter(analysis.log_likelihood.values()))
    count = first.shape[0] * first.shape[1]
    if analysis.kind != "mcmc" or first.shape[0] == 1 or not analysis.samples:
        return 1.0
    tree = analysis.to_datatree()
    ess = tree["posterior"].dataset.azstats.ess(method="mean")
    return float(
        np.hstack([np.asarray(ess[v].values).ravel() for v in ess.data_vars]).mean()
        / count
    )


def _arviz_loo(scores: CriterionScores, r_eff: float) -> ELPDData:
    """Run ArviZ PSIS-LOO with an explicit relative efficiency.

    Parameters
    ----------
    scores : CriterionScores
        Pointwise scores.
    r_eff : float
        Relative efficiency.

    Returns
    -------
    arviz_stats.utils.ELPDData
        Pointwise PSIS-LOO result with smoothed log weights.

    Raises
    ------
    ValueError
        If fewer than 25 draws are pooled, or a pointwise score is missing.
    """
    pooled = scores.log_likelihood
    require_psis_draws(pooled.shape[0] * pooled.shape[1])
    require_scores(pooled, criterion="PSIS-LOO")
    result = az.loo(
        scores.analysis.to_datatree(),
        var_name=scores.name,
        pointwise=True,
        reff=r_eff,
    )
    if not isinstance(result, ELPDData):
        raise TypeError("ArviZ loo returned no pointwise ELPDData")
    return result


def unconstrained_draws(
    fit: Fit, draws: Mapping[str, torch.Tensor]
) -> tuple[LatentRegistry, torch.Tensor]:
    """Map constrained draws to the flat unconstrained sampler coordinates.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    draws : Mapping[str, torch.Tensor]
        Constrained draws ``(chain, draw, *event)``.

    Returns
    -------
    tuple[LatentRegistry, torch.Tensor]
        The latent registry and Float64 draws ``(chain, draw, D)``.
    """
    registry = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    unconstrained, _ = registry.to_unconstrained(dict(draws))
    return registry, registry.flatten_unconstrained(unconstrained).to(torch.float64)


def constrained_view(
    fit: Fit, registry: LatentRegistry, upars: torch.Tensor, *, joint: bool
) -> Fit:
    """Return the fit replaying given unconstrained draws as its posterior.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    registry : LatentRegistry
        Its latent registry.
    upars : torch.Tensor
        Unconstrained draws ``(S, D)``.
    joint : bool
        Whether the fit is a `MultivariateFit`.

    Returns
    -------
    BayesianFit or MultivariateFit
        A view whose retained draws are the constrained images of ``upars``
        on one layout axis.
    """
    sites = registry.unflatten_unconstrained(upars.to(torch.get_default_dtype()))
    constrained, _ = registry.to_constrained(sites)
    draws = PriorDraws(
        samples={name: value.unsqueeze(0) for name, value in constrained.items()},
        num_samples=upars.shape[0],
    )
    if joint:
        joint_fit = cast("MultivariateFit", fit)
        return joint_fit.with_(
            posterior=draws, response_fits=dict(joint_fit.response_fits)
        )
    return cast("BayesianFit", fit).with_(posterior=draws)


def view_log_likelihood(
    view: Fit, *, joint: bool, resp: str | Sequence[str] | None
) -> torch.Tensor:
    """Score the training rows at a view's retained draws.

    Parameters
    ----------
    view : BayesianFit or MultivariateFit
        A fit or a draw view of one.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.

    Returns
    -------
    torch.Tensor
        Scores ``(chain, draw, unit)``.
    """
    if joint:
        scores = cast("MultivariateFit", view).log_likelihood(resp=resp)
    else:
        scores = cast("BayesianFit", view).log_likelihood()
    (value,) = scores.values()
    return value


def moment_match(
    fit: Fit,
    scores: CriterionScores,
    original: ELPDData,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    num_samples: int | None,
    seed: int,
    r_eff: float,
    k_threshold: float,
    max_iters: int,
    split: bool,
    cov: bool,
) -> tuple[ELPDData, tuple[int, ...]]:
    """Improve PSIS-LOO terms with ArviZ's implicitly adaptive importance sampling.

    Moment matching [@paananen2021implicitly] moves the posterior draws in
    the unconstrained space by affine maps that match the importance-weighted
    moments of each problematic leave-one-out posterior, and reweights the
    moved draws by the ratio of the full-data posterior density to the
    density of the moved proposal. The full-data density is the fit's own
    Jacobian-corrected log joint, the potential its sampler used.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    scores : CriterionScores
        Retained scores of the original estimate.
    original : arviz_stats.utils.ELPDData
        Pointwise PSIS-LOO result.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.
    num_samples : int or None
        Draws per chain behind the scores.
    seed : int
        Seed of variational draws behind the scores.
    r_eff : float
        Relative efficiency.
    k_threshold : float
        Units with Pareto shape above it are moment matched.
    max_iters : int
        Maximum moment-matching iterations per unit.
    split : bool
        Transform half of the draws and combine with multiple importance
        sampling, as loo's default.
    cov : bool
        Also match covariances.

    Returns
    -------
    tuple[arviz_stats.utils.ELPDData, tuple[int, ...]]
        The updated estimate and the units whose Pareto shape decreased.

    Notes
    -----
    Each iteration for a unit evaluates the full log joint and the unit's
    log-likelihood at S draws: O(S (C_joint + C_lik)) with C the cost of one
    program evaluation, at most ``max_iters`` times per problematic unit.
    """
    draws = fit.draws(num_samples=num_samples, seed=seed)
    registry, upars = unconstrained_draws(fit, draws)
    chains, count = upars.shape[0], upars.shape[1]
    potential = PotentialFn(
        fit.program,
        registry,
        torch.zeros(1, 1, dtype=torch.long),
        dict(fit.observations),
    )

    def flat(values: xr.DataArray) -> torch.Tensor:
        """Flatten an ArviZ ``(chain, draw, parameter)`` array to ``(S, D)``.

        Parameters
        ----------
        values : xarray.DataArray
            Unconstrained draws.

        Returns
        -------
        torch.Tensor
            Float64 draws, chain-major.
        """
        dims = [d for d in values.dims if d != "unconstrained_parameter"]
        array = np.asarray(
            values.transpose(*dims, "unconstrained_parameter").values,
            dtype=np.float64,
        )
        return torch.as_tensor(array.reshape(-1, array.shape[-1]))

    def sample_array(values: xr.DataArray, flat_values: torch.Tensor) -> xr.DataArray:
        """Restore the sample dimensions of ArviZ's unconstrained draws.

        Parameters
        ----------
        values : xarray.DataArray
            Draws whose sample dimensions are kept.
        flat_values : torch.Tensor
            One value per draw, chain-major.

        Returns
        -------
        xarray.DataArray
            Values on the draws' sample dimensions.
        """
        dims = [d for d in values.dims if d != "unconstrained_parameter"]
        return xr.DataArray(
            flat_values.to(torch.float64)
            .reshape([values.sizes[d] for d in dims])
            .cpu()
            .numpy(),
            dims=dims,
        )

    def log_prob(values: xr.DataArray) -> xr.DataArray:
        """Evaluate the Jacobian-corrected log joint at each draw.

        Parameters
        ----------
        values : xarray.DataArray
            Unconstrained draws ``(chain, draw, parameter)``.

        Returns
        -------
        xarray.DataArray
            Log densities ``(chain, draw)``.
        """
        rows = flat(values)
        with torch.no_grad():
            densities = torch.stack(
                [
                    potential.log_density(row.to(torch.get_default_dtype()))
                    for row in rows
                ]
            )
        return sample_array(values, densities)

    def log_lik_i(values: xr.DataArray, i: int) -> xr.DataArray:
        """Score unit ``i`` at each draw.

        Parameters
        ----------
        values : xarray.DataArray
            Unconstrained draws ``(chain, draw, parameter)``.
        i : int
            Unit position.

        Returns
        -------
        xarray.DataArray
            Log-likelihoods ``(chain, draw)``.
        """
        view = constrained_view(fit, registry, flat(values), joint=joint)
        scored = view_log_likelihood(view, joint=joint, resp=resp)[0, :, int(i)]
        return sample_array(values, scored)

    before = np.asarray(original.pareto_k.values, dtype=np.float64).reshape(-1)
    tree = scores.analysis.to_datatree()
    layout = (chains, count)
    if chains == 1:
        # ArviZ's moment matching squeezes singleton axes of each unit's
        # scores and weights, which drops a lone chain axis. The draws, scores
        # and smoothed weights enter ArviZ as two consecutive halves instead;
        # every quantity it computes pools the draws, and the relative
        # efficiency is passed explicitly.
        if count % 2:
            raise ValueError(
                "moment matching a single chain requires an even number of draws"
            )
        layout = (2, count // 2)
        tree = PosteriorAnalysis(
            samples={},
            log_likelihood={scores.name: scores.log_likelihood.reshape(*layout, -1)},
            # A two-axis score container for ArviZ only, with no draws or
            # sampler statistics; it claims no convergence diagnostics.
            kind="mcmc",
        ).to_datatree()
        original = copy.copy(original)
        weights = original.log_weights
        if isinstance(weights, xr.Dataset):
            weights = weights[scores.name]
        other = [d for d in weights.dims if d not in ("chain", "draw")]
        values = np.asarray(weights.transpose("chain", "draw", *other).values)
        setattr(
            original,
            "log_weights",
            xr.DataArray(
                values.reshape(*layout, *values.shape[2:]),
                dims=("chain", "draw", *other),
            ),
        )
    result = az.loo_moment_match(
        tree,
        original,
        log_prob,
        log_lik_i,
        upars=xr.DataArray(
            upars.cpu().numpy().reshape(*layout, -1),
            dims=("chain", "draw", "unconstrained_parameter"),
        ),
        var_name=scores.name,
        reff=r_eff,
        max_iters=max_iters,
        k_threshold=k_threshold,
        split=split,
        cov=cov,
        pointwise=True,
    )
    after = np.asarray(result.pareto_k.values, dtype=np.float64).reshape(-1)
    improved = tuple(
        int(i) for i in np.nonzero((before > k_threshold) & (after < before))[0]
    )
    return result, improved


def loo_estimate(
    fit: Fit,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    moment_match_options: Mapping[str, int | bool] | None,
    reloo_factory: Callable[[Mapping[str, object], int], Mapping[str, object]] | None,
    k_threshold: float,
    num_samples: int | None,
    seed: int,
) -> ELPDEstimate:
    """Estimate PSIS-LOO, optionally with moment matching and exact refits.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.
    moment_match_options : Mapping or None
        ``max_iters``, ``split`` and ``cov`` of moment matching, or None to
        skip it.
    reloo_factory : Callable or None
        Fold-specific fit settings for exact refits of the units still above
        the threshold (brms ``reloo``), or None to skip them.
    k_threshold : float
        Pareto shape above which units are moment matched or refitted.
    num_samples : int or None
        Draws per chain.
    seed : int
        Seed for variational draws.

    Returns
    -------
    ELPDEstimate
        PSIS-LOO estimate with the units changed by each correction.
    """
    scores = criterion_scores(
        fit, joint=joint, resp=resp, num_samples=num_samples, seed=seed
    )
    r_eff = relative_efficiency(scores.analysis)
    result = _arviz_loo(scores, r_eff)
    matched: tuple[int, ...] = ()
    if moment_match_options is not None:
        options = dict(moment_match_options)
        result, matched = moment_match(
            fit,
            scores,
            result,
            joint=joint,
            resp=resp,
            num_samples=num_samples,
            seed=seed,
            r_eff=r_eff,
            k_threshold=k_threshold,
            max_iters=int(options.get("max_iters", 30)),
            split=bool(options.get("split", True)),
            cov=bool(options.get("cov", True)),
        )
    refitted: tuple[int, ...] = ()
    if reloo_factory is not None:
        pareto = np.asarray(result.pareto_k.values, dtype=np.float64).reshape(-1)
        refitted = tuple(int(i) for i in np.nonzero(~(pareto <= k_threshold))[0])
        if refitted:
            callback = (
                cast("MultivariateFit", fit).refit_callback(
                    method_factory=reloo_factory, resp=resp
                )
                if joint
                else cast("BayesianFit", fit).refit_callback(
                    method_factory=reloo_factory
                )
            )
            result = cast(
                ELPDData,
                scores.analysis.reloo(
                    callback,
                    var_name=scores.name,
                    k_threshold=k_threshold,
                    seed=seed,
                ),
            )
    return from_arviz(
        result,
        scores.log_likelihood,
        criterion="loo",
        responses=scores.responses,
        hashed=scores.hashed,
        moment_matched=matched,
        refitted=refitted,
    )


def waic_from_fit(
    fit: Fit,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    num_samples: int | None,
    seed: int,
) -> ELPDEstimate:
    """Estimate WAIC from the fit's pointwise scores.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.
    num_samples : int or None
        Draws per chain.
    seed : int
        Seed for variational draws.

    Returns
    -------
    ELPDEstimate
        WAIC estimate.
    """
    scores = criterion_scores(
        fit, joint=joint, resp=resp, num_samples=num_samples, seed=seed
    )
    return waic_estimate(
        scores.log_likelihood, responses=scores.responses, hashed=scores.hashed
    )


def psis_from_fit(
    fit: Fit,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    num_samples: int | None,
    seed: int,
) -> PSISWeights:
    """Return the PSIS leave-one-out weights of the fit's pointwise scores.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.
    num_samples : int or None
        Draws per chain.
    seed : int
        Seed for variational draws.

    Returns
    -------
    PSISWeights
        Smoothed weights with ArviZ's relative efficiency.
    """
    scores = criterion_scores(
        fit, joint=joint, resp=resp, num_samples=num_samples, seed=seed
    )
    return psis_weights(
        scores.log_likelihood, r_eff=relative_efficiency(scores.analysis)
    )


def point_view(fit: Fit, *, joint: bool, num_samples: int | None, seed: int) -> Fit:
    """Return the fit replaying its posterior mean in unconstrained coordinates.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    num_samples : int or None
        Draws per chain.
    seed : int
        Seed for variational draws.

    Returns
    -------
    BayesianFit or MultivariateFit
        A one-draw view at the unconstrained posterior mean mapped to the
        constrained support, a valid point for every site.
    """
    draws = fit.draws(num_samples=num_samples, seed=seed)
    registry, upars = unconstrained_draws(fit, draws)
    point = upars.flatten(0, 1).mean(0, keepdim=True)
    return constrained_view(fit, registry, point, joint=joint)


def training_rows(fit: Fit, *, joint: bool) -> tuple[IntoDataFrame, tuple[int, ...]]:
    """Return the original rows and the positions of the scored units.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model with a retained refit recipe.
    joint : bool
        Whether the fit is a `MultivariateFit`.

    Returns
    -------
    tuple[IntoDataFrame, tuple[int, ...]]
        The original frame and, per scored unit, its original row position.

    Raises
    ------
    ValueError
        If the fit retains no original rows, or the joint fit has ``mi()``
        responses whose scored units are not original rows.
    """
    recipe = fit.refit_recipe
    if recipe is None:
        raise ValueError("this fit retains no original rows; refit it with fit()")
    frame = recipe.data
    size = len(nw.from_native(frame, eager_only=True))
    if joint:
        joint_fit = cast("MultivariateFit", fit)
        if joint_fit.missing.responses or joint_fit.missing.inputs:
            raise ValueError(
                "row subsampling is not defined for joint fits with mi() terms"
            )
        selections = {
            joint_fit.emission.responses[name].formula.modifiers.subset
            for name in joint_fit.response_names
        }
        selection = next(iter(selections)) if len(selections) == 1 else None
        if len(selections) != 1:
            raise ValueError("responses with different subset() rows: select one resp")
    else:
        selection = cast("BayesianFit", fit).formula.modifiers.subset
    return frame, selection.positions if selection is not None else tuple(range(size))


def loo_subsample_estimate(
    fit: Fit,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    observations: int | Sequence[int],
    approximation: Literal["plpd", "lpd"],
    num_samples: int | None,
    seed: int,
) -> ELPDEstimate:
    """Estimate PSIS-LOO from a row subsample with the difference estimator.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model with retained original rows.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.
    observations : int or Sequence[int]
        Subsample size drawn by simple random sampling without replacement,
        or explicit 0-based unit positions.
    approximation : {"plpd", "lpd"}
        ``plpd`` scores every row at the posterior mean (brms's default; one
        replay of all rows); ``lpd`` uses each row's log predictive density
        over all draws, which scores every row at every draw.
    num_samples : int or None
        Draws per chain.
    seed : int
        Seed of the subsample and of variational draws.

    Returns
    -------
    ELPDEstimate
        Subsampled PSIS-LOO estimate [@magnusson2019bayesian;
        @magnusson2020leave].

    Raises
    ------
    ValueError
        If the subsample is not smaller than the number of units, or the
        fit models residual dependence between rows.

    Notes
    -----
    With ``plpd`` the cost is one replay of all N rows plus S replays of
    the m subsampled rows, O(N + S m), instead of O(S N) for full PSIS-LOO.
    """
    if (joint and cast("MultivariateFit", fit).data2) or (
        not joint
        and (
            autocor_joint(cast("BayesianFit", fit).family)
            or autocor_binding(cast("BayesianFit", fit).family) is not None
        )
    ):
        raise ValueError(
            "loo_subsample requires conditionally independent rows; this fit "
            "models residual dependence between rows"
        )
    frame, positions = training_rows(fit, joint=joint)
    total = len(positions)
    if isinstance(observations, bool):
        raise TypeError("observations must be a count or a sequence of rows")
    if isinstance(observations, int):
        if not 2 <= observations < total:
            raise ValueError(
                f"subsample size must lie in [2, {total - 1}]; use loo for all rows"
            )
        generator = torch.Generator().manual_seed(seed)
        units = tuple(
            sorted(
                int(i)
                for i in torch.randperm(total, generator=generator)[:observations]
            )
        )
    else:
        units = tuple(int(i) for i in observations)
    if len(set(units)) != len(units) or any(not 0 <= u < total for u in units):
        raise ValueError("subsample units must be distinct positions below N")
    if not 2 <= len(units) < total:
        raise ValueError("the subsample must hold at least two and fewer than N units")
    if approximation == "plpd":
        view = point_view(fit, joint=joint, num_samples=num_samples, seed=seed)
        approx = view_log_likelihood(view, joint=joint, resp=resp)[0, 0]
        outcomes = criterion_outcomes(fit, joint=joint, resp=resp)
    elif approximation == "lpd":
        full = criterion_scores(
            fit, joint=joint, resp=resp, num_samples=num_samples, seed=seed
        )
        approx = pointwise_lpd(full.log_likelihood)
        outcomes = list(full.analysis.observed_data.values())
    else:
        raise ValueError("approximation must be 'plpd' or 'lpd'")
    rows = nw.from_native(frame, eager_only=True)[
        [positions[u] for u in units], :
    ].to_native()
    if joint:
        joint_fit = cast("MultivariateFit", fit)
        subsample = joint_fit.log_likelihood(
            resp=resp, new_data=rows, num_samples=num_samples, seed=seed
        )
        analysis_kind_fit: Fit = joint_fit
    else:
        single = cast("BayesianFit", fit)
        subsample = single.log_likelihood(
            new_data=rows, num_samples=num_samples, seed=seed
        )
        analysis_kind_fit = single
    (scores,) = subsample.values()
    draws = analysis_kind_fit.draws(num_samples=num_samples, seed=seed)
    analysis = PosteriorAnalysis(
        samples=draws,
        log_likelihood={"subsample": scores},
        kind=draw_provenance(fit.posterior),
    )
    weights = psis_weights(scores, r_eff=relative_efficiency(analysis))
    pooled = scores.detach().to(torch.float64)
    exact = torch.logsumexp(weights.log_weights + pooled, dim=(0, 1))
    return subsample_estimate(
        exact,
        approx,
        units,
        pointwise_lpd(scores),
        weights.pareto_k,
        num_samples=scores.shape[0] * scores.shape[1],
        responses=cast("MultivariateFit", fit)._selection(resp)
        if joint
        else (cast("BayesianFit", fit).formula.response_name,),
        hashed=response_hash(outcomes),
        method=approximation,
    )


def criterion_outcomes(
    fit: Fit, *, joint: bool, resp: str | Sequence[str] | None
) -> list[torch.Tensor]:
    """Return the observed responses a criterion hashes.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.

    Returns
    -------
    list[torch.Tensor]
        Observed response tensors in selection order.
    """
    if joint:
        return list(cast("MultivariateFit", fit).observed(resp=resp).values())
    single = cast("BayesianFit", fit)
    analysis_outcome = single.formula.response_values
    return [torch.as_tensor(np.asarray(analysis_outcome))]


class KFoldResult(ValueModel):
    """A brms ``kfold`` result: the criterion, its refits and its partition.

    Parameters
    ----------
    estimate : ELPDEstimate
        The k-fold criterion.
    cross_validation : CrossValidationResult
        Held-out scores, the partition, and the refits when saved.
    data : IntoDataFrame
        Original rows of the fitted model.
    positions : tuple[int, ...]
        Original row position of every partitioned unit.
    fold_type : str
        ``random``, ``group``, ``stratified``, ``grouped``, ``loo`` or ``custom``.
    joint_fit : bool
        Whether the refits are joint fits.
    resp : str or tuple[str, ...] or None
        Joint response selection of the scores.
    """

    estimate: ELPDEstimate = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    cross_validation: CrossValidationResult = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    data: IntoDataFrame = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    positions: tuple[int, ...]
    fold_type: str
    joint_fit: bool
    resp: str | tuple[str, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _group_labels(
    frame: IntoDataFrame, positions: tuple[int, ...], group: str
) -> tuple[str, ...]:
    """Read a grouping column at the partitioned rows.

    Parameters
    ----------
    frame : IntoDataFrame
        Original rows.
    positions : tuple[int, ...]
        Partitioned row positions.
    group : str
        Column name.

    Returns
    -------
    tuple[str, ...]
        String labels, one per partitioned row.

    Raises
    ------
    ValueError
        If the column is absent or has missing values.
    """
    table = nw.from_native(frame, eager_only=True)
    if group not in table.columns:
        raise ValueError(f"group {group!r} names no column of the fitted data")
    values = table[group].to_list()
    labels = [values[position] for position in positions]
    if any(label is None for label in labels):
        raise ValueError(f"group {group!r} has missing values")
    return tuple(str(label) for label in labels)


def kfold_plan(
    size: int,
    *,
    K: int,
    folds: FoldSpec,
    groups: tuple[str, ...] | None,
    seed: int,
) -> tuple[FoldPlan, str]:
    """Build brms's k-fold partition from its arguments.

    Parameters
    ----------
    size : int
        Number of partitioned units.
    K : int
        Number of folds for random, stratified and grouped partitions.
    folds : FoldSpec
        None (random, or one fold per group level when ``groups`` is given),
        ``"loo"``, ``"stratified"``, ``"grouped"``, or one label per unit.
    groups : tuple[str, ...] or None
        Group labels per unit.
    seed : int
        Partition seed.

    Returns
    -------
    tuple[FoldPlan, str]
        The partition and brms's fold type label.

    Raises
    ------
    ValueError
        If a grouped type lacks groups or labels have the wrong length.
    """
    if folds is None and groups is None:
        return fold_plan(size, folds=K, seed=seed), "random"
    if folds is None:
        assert groups is not None
        levels = sorted(set(groups))
        return (
            FoldPlan(assignments=tuple(levels.index(g) for g in groups), seed=seed),
            "group",
        )
    if isinstance(folds, str):
        if folds == "loo":
            return (
                FoldPlan(
                    assignments=tuple(range(size)),
                    seed=seed,
                    omission_groups=groups if groups is not None else (),
                ),
                "loo",
            )
        if folds in {"stratified", "grouped"}:
            if groups is None:
                raise ValueError(f"fold type {folds!r} requires a group")
            plan = (
                fold_plan(size, folds=K, strata=groups, seed=seed)
                if folds == "stratified"
                else fold_plan(size, folds=K, groups=groups, seed=seed)
            )
            return plan, folds
        if folds == "random":
            return fold_plan(size, folds=K, seed=seed), "random"
        raise ValueError(f"unknown fold type {folds!r}")
    labels = list(folds)
    if len(labels) != size:
        raise ValueError(f"folds must give one label per unit ({size})")
    levels = sorted(set(labels), key=lambda value: (str(type(value)), value))
    return (
        FoldPlan(assignments=tuple(levels.index(v) for v in labels), seed=seed),
        "custom",
    )


def kfold_from_fit(
    fit: Fit,
    *,
    joint: bool,
    resp: str | Sequence[str] | None,
    K: int,
    Ksub: int | Sequence[int] | None,
    folds: FoldSpec,
    group: str | None,
    joint_unit: JointUnit | bool,
    save_fits: bool,
    method_factory: Callable[[Mapping[str, object], int], Mapping[str, object]],
    num_samples: int | None,
    seed: int,
    allow_new_levels: bool,
    sample_new_levels: NewLevelPolicy,
) -> KFoldResult:
    """Run brms's ``kfold`` through the fit's refit callback.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model with a retained refit recipe.
    joint : bool
        Whether the fit is a `MultivariateFit`.
    resp : str or Sequence[str] or None
        Joint response selection.
    K : int
        Number of folds.
    Ksub : int or Sequence[int] or None
        Number of randomly selected folds, or explicit 0-based folds to
        evaluate; None evaluates all.
    folds : FoldSpec
        Fold type or explicit labels.
    group : str or None
        Grouping column for grouped, stratified and leave-one-group-out folds.
    joint_unit : {"obs", "fold", "group"} or bool
        Predictive unit; True means ``"fold"`` and False ``"obs"`` as brms.
    save_fits : bool
        Retain each refit for `kfold_predict`.
    method_factory : Callable
        Fold-specific fit settings and seed.
    num_samples : int or None
        Draws per chain of held-out and full-data scoring.
    seed : int
        Partition and refit seed.
    allow_new_levels : bool
        Score held-out rows with new group levels (True in brms's k-fold).
    sample_new_levels : NewLevelPolicy
        New-level policy (``"gaussian"`` in brms's k-fold).

    Returns
    -------
    KFoldResult
        The criterion, held-out scores and partition.

    Raises
    ------
    ValueError
        If arguments are inconsistent.

    Notes
    -----
    Cost is one refit per evaluated fold plus held-out and full-data scoring.
    """
    unit: JointUnit = (
        ("fold" if joint_unit else "obs")
        if isinstance(joint_unit, bool)
        else joint_unit
    )
    if unit not in {"obs", "fold", "group"}:
        raise ValueError("joint must be 'obs', 'fold', 'group' or Boolean")
    frame, positions = training_rows(fit, joint=joint)
    size = len(positions)
    groups = _group_labels(frame, positions, group) if group is not None else None
    if unit == "group" and groups is None:
        raise ValueError("joint='group' requires a group")
    plan, fold_type = kfold_plan(size, K=K, folds=folds, groups=groups, seed=seed)
    count = plan.num_folds
    if Ksub is None:
        chosen = tuple(range(count))
    elif isinstance(Ksub, bool):
        raise TypeError("Ksub must be a fold count or a sequence of folds")
    elif isinstance(Ksub, int):
        if not 1 <= Ksub <= count:
            raise ValueError(f"Ksub must lie in [1, {count}]")
        generator = torch.Generator().manual_seed(seed + 1)
        chosen = tuple(
            sorted(int(i) for i in torch.randperm(count, generator=generator)[:Ksub])
        )
    else:
        chosen = tuple(sorted(set(int(i) for i in Ksub)))
        if not chosen or any(not 0 <= i < count for i in chosen):
            raise ValueError(f"Ksub folds must be 0-based fold ids below {count}")
    if joint:
        joint_fit = cast("MultivariateFit", fit)
        callback = joint_fit.refit_callback(
            method_factory=method_factory,
            num_samples=num_samples,
            resp=resp,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
        )
    else:
        callback = cast("BayesianFit", fit).refit_callback(
            method_factory=method_factory,
            num_samples=num_samples,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
        )
    units: dict[str, Literal["observation", "fold", "group"]] = {
        "obs": "observation",
        "fold": "fold",
        "group": "group",
    }
    result = cross_validate(
        plan,
        callback,
        fold_ids=chosen,
        joint=units[unit],
        groups=groups if unit == "group" else None,
        save_fits=save_fits,
    )
    if joint:
        full = cast("MultivariateFit", fit).log_likelihood(
            resp=resp,
            num_samples=num_samples,
            seed=seed,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
        )
    else:
        full = cast("BayesianFit", fit).log_likelihood(
            num_samples=num_samples,
            seed=seed,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
        )
    (scores,) = full.values()
    pooled = scores.detach().to(torch.float64)
    order = sorted(range(len(result.units)), key=lambda k: result.units[k][1])
    elpd = result.log_predictive_density
    lpd = torch.stack(
        [
            torch.logsumexp(pooled[..., list(rows)].sum(-1).flatten(), 0)
            - math.log(pooled.shape[0] * pooled.shape[1])
            for _, rows in result.units
        ]
    )
    if unit == "obs":
        elpd, lpd = elpd[order], lpd[order]
    estimate = kfold_estimate(
        elpd,
        lpd,
        num_samples=pooled.shape[0] * pooled.shape[1],
        folds=plan.assignments,
        joint=unit,
        responses=tuple(full),
        hashed=response_hash(criterion_outcomes(fit, joint=joint, resp=resp)),
    )
    return KFoldResult(
        estimate=estimate,
        cross_validation=result,
        data=frame,
        positions=positions,
        fold_type=fold_type,
        joint_fit=joint,
        resp=resp if isinstance(resp, str) or resp is None else tuple(resp),
    )


class KFoldPrediction(ValueModel):
    """Held-out observations and predictions of k-fold refits, as ``kfold_predict``.

    Parameters
    ----------
    observed : torch.Tensor
        Observed responses of the predicted units, in original row order.
    predicted : torch.Tensor
        Draws ``(chain, draw, unit[, event])`` from each unit's own refit.
    positions : tuple[int, ...]
        Original row position of each predicted unit.
    """

    observed: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    predicted: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    positions: tuple[int, ...]


__all__ = [
    "CriterionScores",
    "KFoldPrediction",
    "KFoldResult",
    "criterion_outcomes",
    "criterion_scores",
    "good_k",
    "kfold_from_fit",
    "kfold_plan",
    "loo_estimate",
    "loo_subsample_estimate",
    "moment_match",
    "point_view",
    "psis_from_fit",
    "relative_efficiency",
    "training_rows",
    "waic_from_fit",
]
