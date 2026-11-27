"""brms post-processing methods of single-response and joint formula fits.

`SinglePostprocess` and `JointPostprocess` are bases of `BayesianFit` and
`MultivariateFit`. They add brms's summary forms (``summary``, ``fixef``,
``ranef``, ``coef``, ``VarCorr``, ``vcov``, ``posterior_summary``,
``posterior_interval``, ``predict``, ``fitted``, ``residuals``,
``predictive_error``, ``predictive_interval``), its accessors (``ndraws``,
``nchains``, ``niterations``, ``nobs``, ``ngrps``, ``rhat``,
``neff_ratio``, ``nuts_params``, ``log_posterior``) and its leave-one-out
family (``loo`` with moment matching and exact refits, ``waic``, ``psis``,
``loo_moment_match``, ``loo_subsample``, ``loo_predict``, ``loo_epred``,
``loo_linpred``, ``loo_predictive_interval``, ``loo_R2``, ``kfold``,
``kfold_predict``, ``add_criterion``). Every method replays the fit's own
draws through its existing prediction and scoring methods.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack, cast

import narwhals as nw
import pandas as pd
import torch
from narwhals.typing import IntoDataFrame

from quivers.diagnostics.convergence import ess, rhat
from quivers.diagnostics.elpd import (
    ELPDEstimate,
    LooExpectation,
    PSISWeights,
    compare_estimates,
    information_weights,
    stacking_weights,
    loo_expectation,
    loo_r2_draws,
    psis_weights,
)
from quivers.diagnostics.posterior_tables import (
    SummaryTable,
    posterior_interval as interval_table,
    posterior_summary as summary_table,
    predictive_interval as predictive_table,
)
from quivers.formulas._group_prediction import NewLevelPolicy
from quivers.formulas._postprocess_loo import (
    KFoldPrediction,
    KFoldResult,
    criterion_scores,
    kfold_from_fit,
    loo_estimate,
    loo_subsample_estimate,
    relative_efficiency,
    waic_from_fit,
)
from quivers.formulas._name_scope import level_coefficients
from quivers.formulas._postprocess_names import (
    coefficient_labels,
    name_scope,
    single_variables,
)
from quivers.formulas._postprocess_predict import (
    PredictionMethod,
    latent_draws,
    prediction_method,
    predictive_errors,
    selected_draws,
    summarize_prediction,
)
from quivers.formulas._postprocess_summary import (
    FitSummary,
    VarCorrBlock,
    _ThresholdSign,
    coef_arrays,
    fit_summary,
    fixef_draws,
    fixef_table,
    group_varcorr,
    level_array,
    varcorr_block,
    vcov_table,
)
from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._r_squared import _responses, evaluate_responses, formula_r2
from quivers.formulas._fit_controls import SamplerControl, mcmc_kernel, sampler_control
from quivers.formulas._initial_values import chain_initial_values
from quivers.inference.mcmc.hmc import _HamiltonianKernel

# Bayesian model averaging weights by bridge sampling.
from quivers.formulas._evidence import post_prob
from quivers.formulas._response_modifiers import ExpectationOptions
from quivers.inference._posterior import draw_count
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.prior import PriorDraws
from quivers.inference.pathfinder import PathfinderDraws, draw_provenance
from quivers.formulas._fit_types import Fit, JointFit, SingleFit

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit
    from quivers.formulas.family import OrdinalSpec

type ResponseSelection = str | Sequence[str] | None
"""Joint response selection: one label, several labels, or every response."""

type MethodFactory = Callable[[Mapping[str, object], int], Mapping[str, object]]
"""Fold-specific fit settings from the original settings and a fold seed."""

#: brms's NUTS statistic names (bayesplot ``nuts_params``) of the native statistics.
_NUTS_NAMES: dict[str, str] = {
    "accept_prob": "accept_stat__",
    "step_size": "stepsize__",
    "tree_depth": "treedepth__",
    "num_steps": "n_leapfrog__",
    "diverged": "divergent__",
    "energy": "energy__",
}


def _ordinal_threshold_sign(ordinal: OrdinalSpec) -> _ThresholdSign:
    """Return the direction of an ordinal threshold on the linear predictor.

    Parameters
    ----------
    ordinal : OrdinalSpec
        The response's ordinal likelihood.

    Returns
    -------
    _ThresholdSign
        Threshold minus predictor, or predictor minus threshold for adjacent
        categories and continuation ratios.
    """
    if ordinal.kind in {"cratio", "acat"}:
        return "eta_minus_thres"
    return "thres_minus_eta"


class JointPredictionOptions(TypedDict, total=False):
    """Prediction keywords forwarded to `MultivariateFit` prediction methods.

    Each key has the meaning of the keyword of the same name on
    `MultivariateFit.posterior_predict`.
    """

    new_data: IntoDataFrame | None
    re_formula: str | None
    allow_new_levels: bool
    sample_new_levels: NewLevelPolicy
    num_samples: int | None
    seed: int
    data2: Mapping[str, AutocorMatrix] | None
    incl_autocor: bool


class SinglePredictionOptions(TypedDict, total=False):
    """Prediction keywords forwarded to `BayesianFit` prediction methods.

    Each key has the meaning of the keyword of the same name on
    `BayesianFit.posterior_predict`.
    """

    new_data: IntoDataFrame | None
    predictor_data: torch.Tensor | None
    trials: int | str | None
    re_formula: str | None
    allow_new_levels: bool
    sample_new_levels: NewLevelPolicy
    num_samples: int | None
    seed: int
    incl_autocor: bool


def _kind(
    posterior: MCMCResult | PriorDraws | Guide,
) -> Literal["mcmc", "variational", "prior"]:
    """Name the provenance of retained draws.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws or Guide
        Retained posterior.

    Returns
    -------
    {"mcmc", "variational", "prior"}
        Draw provenance.
    """
    # Pathfinder draws report variational provenance.
    return draw_provenance(posterior)


def _algorithm(posterior: MCMCResult | PriorDraws | Guide) -> str:
    """Name the fitting algorithm as brms's summary does.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws or Guide
        Retained posterior.

    Returns
    -------
    str
        ``"sampling"`` for recorded chains, ``"meanfield"`` for a mean-field
        Normal guide, the guide class name for other guides, ``"pathfinder"``
        for Pathfinder draws, ``"prior"`` for prior draws.
    """
    if isinstance(posterior, MCMCResult):
        return "sampling"
    # brms names Pathfinder fits "pathfinder".
    if isinstance(posterior, PathfinderDraws):
        return "pathfinder"
    if isinstance(posterior, PriorDraws):
        return "prior"
    name = type(posterior).__name__
    return "meanfield" if name == "AutoNormalGuide" else name


def _select(
    names: Mapping[str, torch.Tensor], variable: Sequence[str] | None
) -> dict[str, torch.Tensor]:
    """Select named draws, keeping brms's order.

    Parameters
    ----------
    names : Mapping[str, torch.Tensor]
        brms-named draws.
    variable : Sequence[str] or None
        Names to keep, or None for all.

    Returns
    -------
    dict[str, torch.Tensor]
        Selected draws in requested order.

    Raises
    ------
    ValueError
        If a name is unknown.
    """
    if variable is None:
        return dict(names)
    wanted = [variable] if isinstance(variable, str) else list(variable)
    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise ValueError(f"unknown variables {unknown!r}; see variables()")
    return {name: names[name] for name in wanted}


class _PostprocessCore:
    """Response-independent brms post-processing shared by both fit classes.

    Subclasses implement `_pp_named`, `_pp_levels`, `_pp_dpars`,
    `_pp_threshold_sign`, `_pp_describe`, `_pp_residual` and `_pp_nobs`.
    """

    def _pp_posterior(self) -> MCMCResult | PriorDraws | Guide:
        """Return the retained posterior.

        Returns
        -------
        MCMCResult or PriorDraws or Guide
            The fit's posterior.
        """
        return cast("BayesianFit", self).posterior

    def _pp_named(self, num_samples: int | None, seed: int) -> dict[str, torch.Tensor]:
        """Return brms-named draws.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            brms-named draws.
        """
        raise TypeError(f"{type(self).__name__} does not name its parameters")

    def _pp_levels(
        self, num_samples: int | None, seed: int
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Return per-level group coefficients with brms ``ranef`` labels.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, list[tuple[str, torch.Tensor]]]
            Per group, labels and ``(chain, draw, level)`` draws.
        """
        raise TypeError(f"{type(self).__name__} has no group coefficients")

    def _pp_dpars(self) -> frozenset[str]:
        """Return the family parameter names forming brms's ``spec_pars``.

        Returns
        -------
        frozenset[str]
            Auxiliary parameter names.
        """
        return frozenset()

    def _pp_threshold_signs(
        self, fixed: Mapping[str, torch.Tensor]
    ) -> dict[str, _ThresholdSign]:
        """Return directions of the population ordinal coefficients.

        Parameters
        ----------
        fixed : Mapping[str, torch.Tensor]
            Population coefficient labels and draws.

        Returns
        -------
        dict[str, _ThresholdSign]
            Threshold directions keyed by coefficient label.
        """
        return {}

    def _pp_describe(self) -> tuple[str, str]:
        """Describe the formula and family for the summary header.

        Returns
        -------
        tuple[str, str]
            Formula text and family description.
        """
        return type(self).__name__, ""

    def _pp_residual(
        self, names: Mapping[str, torch.Tensor]
    ) -> (
        tuple[
            list[tuple[str, torch.Tensor]], dict[tuple[str, str], torch.Tensor] | None
        ]
        | None
    ):
        """Return brms's ``residual__`` block of ``VarCorr``.

        Parameters
        ----------
        names : Mapping[str, torch.Tensor]
            brms-named draws.

        Returns
        -------
        tuple or None
            Residual scales and correlations, or None without an unregressed
            Gaussian-type ``sigma``.
        """
        return None

    def _pp_nobs(self, resp: str | None) -> int:
        """Return the number of observations.

        Parameters
        ----------
        resp : str or None
            Joint response, or None.

        Returns
        -------
        int
            Observation count.
        """
        raise TypeError(f"{type(self).__name__} has no observation count")

    def ndraws(self, *, num_samples: int | None = None) -> int:
        """Return the number of pooled posterior draws, as brms ``ndraws``.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain under the convention of `draws`; None counts every
            retained MCMC draw or the 100 draws a guide gives by default.

        Returns
        -------
        int
            Chains times draws per chain.
        """
        chains, draws = draw_count(self._pp_posterior(), num_samples)
        return chains * draws

    def nchains(self) -> int:
        """Return the number of chains, as brms ``nchains``.

        Returns
        -------
        int
            Recorded chains; one pseudo-chain for guides and prior draws.
        """
        return draw_count(self._pp_posterior(), None)[0]

    def niterations(self) -> int:
        """Return the retained draws per chain, as brms ``niterations``.

        Returns
        -------
        int
            Post-warmup draws per chain (100 by default for a guide).
        """
        return draw_count(self._pp_posterior(), None)[1]

    def nwarmup(self) -> int:
        """Return the warmup iterations per chain.

        Returns
        -------
        int
            MCMC warmup iterations; zero for guides and prior draws.
        """
        posterior = self._pp_posterior()
        return posterior.num_warmup if isinstance(posterior, MCMCResult) else 0

    def nvariables(self, *, num_samples: int | None = None, seed: int = 0) -> int:
        """Return the number of brms-named parameters, as brms ``nvariables``.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        int
            Count of `variables` entries.
        """
        return len(self._pp_named(num_samples, seed))

    def posterior_summary(
        self,
        variable: str | Sequence[str] | None = None,
        *,
        probs: Sequence[float] = (0.025, 0.975),
        robust: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> SummaryTable:
        """Summarize named parameters, as brms ``posterior_summary``.

        Parameters
        ----------
        variable : str or Sequence[str] or None
            brms parameter names (see `variables`); None summarizes all.
        probs : Sequence[float]
            Quantile probabilities.
        robust : bool
            Median and MAD instead of mean and SD.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        SummaryTable
            Columns ``Estimate``, ``Est.Error``, ``Q<100 p>``.

        Raises
        ------
        ValueError
            If a name is unknown.
        """
        names = _select(self._pp_named(num_samples, seed), variable)
        return summary_table(names, probs=probs, robust=robust)

    def posterior_interval(
        self,
        variable: str | Sequence[str] | None = None,
        *,
        prob: float = 0.95,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> SummaryTable:
        """Compute central intervals of named parameters, as brms ``posterior_interval``.

        Parameters
        ----------
        variable : str or Sequence[str] or None
            brms parameter names; None selects all.
        prob : float
            Central mass.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        SummaryTable
            Columns ``<100 a>%`` and ``<100 (1 - a)>%``.

        Raises
        ------
        ValueError
            If a name is unknown or the mass is invalid.
        """
        names = _select(self._pp_named(num_samples, seed), variable)
        return interval_table(names, prob=prob)

    def summary(
        self,
        *,
        prob: float = 0.95,
        robust: bool = False,
        mc_se: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> FitSummary:
        """Summarize the fit section by section, as brms ``summary``.

        Parameters
        ----------
        prob : float
            Central interval mass, 0.95 as in brms.
        robust : bool
            Median and MAD instead of mean and SD.
        mc_se : bool
            Add the Monte Carlo standard error of each estimate.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        FitSummary
            Population effects, family parameters, group-level standard
            deviations and correlations, residual correlations and native
            sites. Recorded chains add ``Rhat``, ``Bulk_ESS`` and ``Tail_ESS``;
            variational and prior draws have no convergence columns.
        """
        posterior = self._pp_posterior()
        formula, family = self._pp_describe()
        return fit_summary(
            {
                name: value
                for name, value in self._pp_named(num_samples, seed).items()
                if not (name.startswith("r_") and name.endswith("]"))
            },
            dpars=self._pp_dpars(),
            kind=_kind(posterior),
            algorithm=_algorithm(posterior),
            formula=formula,
            family=family,
            nobs=self._pp_nobs(None),
            ngrps=self.ngrps(),
            warmup=self.nwarmup(),
            divergences=int(posterior.divergence_counts.sum())
            if isinstance(posterior, MCMCResult)
            else 0,
            prob=prob,
            robust=robust,
            mc_se=mc_se,
        )

    def ngrps(self) -> dict[str, int]:
        """Return the number of levels of each grouping factor, as brms ``ngrps``.

        Returns
        -------
        dict[str, int]
            Grouping factor to level count.
        """
        levels = cast("BayesianFit", self).group_levels
        return {group: len(values) for group, values in levels.items()}

    def fixef(
        self,
        *,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        pars: Sequence[str] | None = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> SummaryTable | dict[str, torch.Tensor]:
        """Return population-level effects, as brms ``fixef``.

        Parameters
        ----------
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        pars : Sequence[str] or None
            Coefficient labels to keep.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        SummaryTable or dict[str, torch.Tensor]
            One row per coefficient, or labels mapped to ``(chain, draw)``
            draws. Labels drop brms's ``b_`` prefix, so a distributional
            coefficient is ``sigma_x``.

        Raises
        ------
        ValueError
            If no coefficient is selected.
        """
        names = self._pp_named(num_samples, seed)
        if summary:
            return fixef_table(names, probs=probs, robust=robust, pars=pars)
        selected = fixef_draws(names, pars)
        if not selected:
            raise ValueError("the model has no selected population-level coefficients")
        return selected

    def ranef(
        self,
        *,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        pars: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, SummaryTable] | dict[str, tuple[tuple[str, ...], torch.Tensor]]:
        """Return group-level effects of every level, as brms ``ranef``.

        Parameters
        ----------
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        pars : Sequence[str] or None
            Coefficient labels to keep, such as ``"Intercept"`` or
            ``"sigma_Intercept"``.
        groups : Sequence[str] or None
            Grouping factors to keep.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict
            Per grouping factor, a three-way table (level, statistic,
            coefficient) as brms's array; or, without summary, coefficient
            labels and draws ``(chain, draw, level, coefficient)``.

        Raises
        ------
        ValueError
            If the model has no group-level effects.
        """
        levels = self._pp_levels(num_samples, seed)
        if not levels:
            raise ValueError("the model does not contain group-level effects")
        group_levels = cast("BayesianFit", self).group_levels
        result: dict[str, tuple[tuple[str, ...], torch.Tensor]] = {}
        for group, members in levels.items():
            if groups is not None and group not in groups:
                continue
            kept = [
                (label, value)
                for label, value in members
                if pars is None or label in pars
            ]
            if kept:
                result[group] = level_array(kept)
        if not summary:
            return result
        return {
            group: summary_table(
                values,
                probs=probs,
                robust=robust,
                labels=group_levels[group],
                components=labels,
            )
            for group, (labels, values) in result.items()
        }

    def coef(
        self,
        *,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, SummaryTable] | dict[str, tuple[tuple[str, ...], torch.Tensor]]:
        """Return population plus group coefficients per level, as brms ``coef``.

        Parameters
        ----------
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict
            Per grouping factor, a three-way table (level, statistic,
            coefficient), or labels and draws ``(chain, draw, level, coef)``.
            Ordinal thresholds ``Intercept[k]`` combine with group intercepts
            with the family's sign, as brms.

        Raises
        ------
        ValueError
            If the model has no group-level effects.
        """
        levels = self._pp_levels(num_samples, seed)
        if not levels:
            raise ValueError(
                "no group-level effects detected; use fixef for population effects"
            )
        fixed = fixef_draws(self._pp_named(num_samples, seed), None)
        arrays = coef_arrays(
            fixed, levels, threshold_signs=self._pp_threshold_signs(fixed)
        )
        if not summary:
            return arrays
        group_levels = cast("BayesianFit", self).group_levels
        return {
            group: summary_table(
                values,
                probs=probs,
                robust=robust,
                labels=group_levels[group],
                components=labels,
            )
            for group, (labels, values) in arrays.items()
        }

    def varcorr(
        self,
        *,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, VarCorrBlock]:
        """Return group and residual standard deviations and correlations, as brms ``VarCorr``.

        Parameters
        ----------
        summary : bool
            Also compute summary tables.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, VarCorrBlock]
            Per grouping factor, and ``residual__`` for an unregressed
            residual ``sigma`` (with residual correlations of a joint fit).

        Raises
        ------
        ValueError
            If the model has neither group effects nor a residual ``sigma``.
        """
        names = self._pp_named(num_samples, seed)
        blocks = group_varcorr(names)
        residual = self._pp_residual(names)
        if residual is not None:
            blocks["residual__"] = residual
        if not blocks:
            raise ValueError("the model does not contain covariance matrices")
        return {
            group: varcorr_block(sd, cor, summary=summary, probs=probs, robust=robust)
            for group, (sd, cor) in blocks.items()
        }

    def vcov(
        self,
        *,
        correlation: bool = False,
        pars: Sequence[str] | None = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> SummaryTable:
        """Return the draw covariance of population effects, as brms ``vcov``.

        Parameters
        ----------
        correlation : bool
            Return correlations instead of covariances.
        pars : Sequence[str] or None
            Coefficient labels to keep.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        SummaryTable
            Square matrix labeled by coefficient on both axes.
        """
        return vcov_table(
            self._pp_named(num_samples, seed), correlation=correlation, pars=pars
        )

    def rhat(self, variable: str | Sequence[str] | None = None) -> dict[str, float]:
        """Return the rank-normalized split R-hat per parameter, as brms ``rhat``.

        Parameters
        ----------
        variable : str or Sequence[str] or None
            brms parameter names; None selects all.

        Returns
        -------
        dict[str, float]
            The larger of the bulk and tail statistics [@vehtari2021rhat].

        Raises
        ------
        ValueError
            If the draws are not recorded chains.
        """
        self._require_chains("rhat")
        return {
            name: float(rhat(value))
            for name, value in _select(self._pp_named(None, 0), variable).items()
        }

    def neff_ratio(
        self, variable: str | Sequence[str] | None = None
    ) -> dict[str, float]:
        """Return effective sample size ratios, as brms ``neff_ratio``.

        Parameters
        ----------
        variable : str or Sequence[str] or None
            brms parameter names; None selects all.

        Returns
        -------
        dict[str, float]
            ``min(Bulk_ESS, Tail_ESS) / ndraws`` per parameter.

        Raises
        ------
        ValueError
            If the draws are not recorded chains.
        """
        self._require_chains("neff_ratio")
        total = self.ndraws()
        return {
            name: min(
                float(ess(value, method="bulk")), float(ess(value, method="tail"))
            )
            / total
            for name, value in _select(self._pp_named(None, 0), variable).items()
        }

    def control_params(
        self, pars: Sequence[str] | None = None
    ) -> dict[str, float | int | str]:
        """Return the sampler's control settings, as brms ``control_params``.

        brms returns the ``control`` list Stan ran with. The settings here are
        those of the fit's kernel: the ``control`` argument of `fit` where it
        was given and the kernel's defaults elsewhere.

        Parameters
        ----------
        pars : Sequence[str] or None
            Names of the settings to return; None returns all of them.

        Returns
        -------
        dict[str, float | int | str]
            ``adapt_delta`` (target mean acceptance statistic),
            ``max_treedepth`` (NUTS only), ``stepsize`` (the step size before
            adaptation; the adapted values are the ``stepsize__`` rows of
            `nuts_params`), and ``metric``.

        Raises
        ------
        ValueError
            If the fit holds no MCMC draws or no retained fit arguments, was
            fitted without a Hamiltonian kernel, or a name is unknown.
        """
        if not isinstance(self._pp_posterior(), MCMCResult):
            raise ValueError("control parameters require a model fitted by MCMC")
        recipe = cast("BayesianFit", self).refit_recipe
        if recipe is None:
            raise ValueError("control parameters require the fit's retained arguments")
        method = str(recipe.method["method"])
        if method not in {"nuts", "hmc"}:
            raise ValueError(
                f"method {method!r} has no step size or metric; control "
                "parameters belong to NUTS and HMC fits"
            )
        control = (
            sampler_control(
                cast(
                    "SamplerControl | Mapping[str, float | int | str] | None",
                    recipe.method.get("control"),
                )
            )
            or SamplerControl()
        )
        kernel = mcmc_kernel(cast(Literal["nuts", "hmc"], method), control)
        settings: dict[str, float | int | str] = {
            "adapt_delta": control.adapt_delta,
            "stepsize": float(cast(_HamiltonianKernel, kernel).step_size),
            "metric": control.metric,
        }
        if method == "nuts":
            settings["max_treedepth"] = control.max_treedepth
        if pars is None:
            return settings
        unknown = [name for name in pars if name not in settings]
        if unknown:
            raise ValueError(
                f"unknown control parameters {unknown!r}; this fit has "
                f"{sorted(settings)}"
            )
        return {name: settings[name] for name in pars}

    def inits(self) -> list[dict[str, torch.Tensor]]:
        """Return the initial values of each chain, as brms ``inits``.

        brms returns one named list per chain with the initial value of each
        Stan parameter, given by the user or generated. The values here are
        on the constrained scale and keyed by latent site, the names of
        `draws`. The driver derives a chain's initial point from the seed
        and the chain index before any other draw, so the fit's retained
        arguments determine it, and this method reads it from a rerun of
        those arguments that stops before sampling.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            One mapping per chain, in chain order.

        Raises
        ------
        ValueError
            If the fit holds no MCMC draws or no retained fit arguments.

        Notes
        -----
        Cost is one preparation and compilation of the model.
        """
        if not isinstance(self._pp_posterior(), MCMCResult):
            raise ValueError("initial values require a model fitted by MCMC")
        recipe = cast("BayesianFit", self).refit_recipe
        if recipe is None:
            raise ValueError("initial values require the fit's retained arguments")
        return chain_initial_values(recipe)

    def nuts_params(self) -> pd.DataFrame:
        """Return per-draw sampler statistics, as brms ``nuts_params``.

        Returns
        -------
        pandas.DataFrame
            Long format with columns ``Chain``, ``Iteration`` (1-based),
            ``Parameter`` and ``Value``; parameters use Stan's names
            (``accept_stat__``, ``stepsize__``, ``treedepth__``,
            ``n_leapfrog__``, ``divergent__``, ``energy__``) where the native
            statistic has one, and the native name otherwise.

        Raises
        ------
        ValueError
            If the draws are not recorded chains.
        """
        posterior = self._require_chains("nuts_params")
        frames = []
        for name, values in posterior.sample_stats.items():
            chains, draws = values.shape
            frames.append(
                pd.DataFrame(
                    {
                        "Chain": torch.arange(1, chains + 1)
                        .repeat_interleave(draws)
                        .numpy(),
                        "Iteration": torch.arange(1, draws + 1).repeat(chains).numpy(),
                        "Parameter": _NUTS_NAMES.get(name, name),
                        "Value": values.detach()
                        .to(torch.float64)
                        .flatten()
                        .cpu()
                        .numpy(),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def log_posterior(self) -> pd.DataFrame:
        """Return the log posterior density per draw, as brms ``log_posterior``.

        Returns
        -------
        pandas.DataFrame
            Columns ``Chain``, ``Iteration`` (1-based) and ``Value``: the
            sampler's Jacobian-corrected unconstrained log joint density, the
            counterpart of Stan's ``lp__``. Stan drops constant terms, so
            values differ from Stan's by a constant.

        Raises
        ------
        ValueError
            If the draws are not recorded chains.
        """
        posterior = self._require_chains("log_posterior")
        values = posterior.log_densities.detach().to(torch.float64)
        chains, draws = values.shape
        return pd.DataFrame(
            {
                "Chain": torch.arange(1, chains + 1).repeat_interleave(draws).numpy(),
                "Iteration": torch.arange(1, draws + 1).repeat(chains).numpy(),
                "Value": values.flatten().cpu().numpy(),
            }
        )

    def _require_chains(self, method: str) -> MCMCResult:
        """Return the recorded chains or refuse a chain diagnostic.

        Parameters
        ----------
        method : str
            Requesting method, for the error.

        Returns
        -------
        MCMCResult
            The recorded chains.

        Raises
        ------
        ValueError
            If the posterior is a guide or prior draws.
        """
        posterior = self._pp_posterior()
        if not isinstance(posterior, MCMCResult):
            raise ValueError(
                f"{method} requires recorded MCMC chains; this fit has "
                f"{_kind(posterior)} draws"
            )
        return posterior

    def criterion(self, name: str) -> ELPDEstimate | torch.Tensor:
        """Return a criterion stored by `add_criterion`.

        Parameters
        ----------
        name : str
            ``loo``, ``waic``, ``kfold``, ``loo_subsample``, ``bayes_R2`` or
            ``loo_R2``.

        Returns
        -------
        ELPDEstimate or torch.Tensor
            The stored estimate, or stored R-squared draws.

        Raises
        ------
        KeyError
            If the criterion is not stored.
        """
        stored = cast("BayesianFit", self).criteria
        if name not in stored:
            raise KeyError(
                f"criterion {name!r} is not stored; call add_criterion first"
            )
        return stored[name]


class SinglePostprocess(_PostprocessCore):
    """brms post-processing methods of a single-response `BayesianFit`."""

    def _pp_fit(self) -> BayesianFit:
        """Return this object as a `BayesianFit`.

        Returns
        -------
        BayesianFit
            The fit.
        """
        return cast("BayesianFit", self)

    def _pp_named(self, num_samples: int | None, seed: int) -> dict[str, torch.Tensor]:
        """Return brms-named draws of the single-response fit.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            brms-named draws.
        """
        return single_variables(self._pp_fit(), num_samples=num_samples, seed=seed)

    def variables(
        self, *, num_samples: int | None = None, seed: int = 0
    ) -> dict[str, torch.Tensor]:
        """Return draws under brms parameter names, as brms ``variables``/``as_draws``.

        Population coefficients are ``b_<coef>`` (``b_<dpar>_<coef>`` for a
        distributional or nonlinear parameter), cumulative thresholds
        ``b_Intercept[k]``, unregressed family parameters by name
        (``sigma``), group standard deviations ``sd_<group>__<coef>``,
        correlations ``cor_<group>__<a>__<b>`` and per-level coefficients
        ``r_<group>[<level>,<coef>]``. Native sites without a brms
        counterpart keep their QVR names. The log density ``lp__`` is
        returned by `log_posterior`.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain under the convention of `draws`.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            brms-named draws on chain and draw axes.
        """
        return self._pp_named(num_samples, seed)

    def _pp_levels(
        self, num_samples: int | None, seed: int
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Return per-level group coefficients with brms labels.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, list[tuple[str, torch.Tensor]]]
            Per group, labels (``<dpar>_<coef>`` outside ``mu``) and draws.
        """
        fit = self._pp_fit()
        draws = fit.draws(num_samples=num_samples, seed=seed)
        shape = draw_count(fit.posterior, num_samples)
        labels = coefficient_labels(fit)
        plan = fit.prediction_plan
        outputs = {
            item.prefix: item.output
            for item in (plan.predictors if plan is not None else ())
        }
        result: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for parameter, prefix, predictor in fit._predictor_items():
            if not predictor.random_terms:
                continue
            for item in level_coefficients(
                fit._random_effects(draws, shape, predictor, prefix),
                name_scope(fit, parameter, prefix, predictor, labels),
                outputs.get(prefix),
            ):
                result.setdefault(item.group, []).append((item.name, item.values))
        return result

    def _pp_dpars(self) -> frozenset[str]:
        """Return the family's auxiliary parameter names.

        Returns
        -------
        frozenset[str]
            Auxiliary parameter names.
        """
        return frozenset(aux.name for aux in self._pp_fit().family.aux_params)

    def _pp_threshold_signs(
        self, fixed: Mapping[str, torch.Tensor]
    ) -> dict[str, _ThresholdSign]:
        ordinal = self._pp_fit().family.ordinal
        if ordinal is None:
            return {}
        return dict.fromkeys(fixed, _ordinal_threshold_sign(ordinal))

    def _pp_describe(self) -> tuple[str, str]:
        """Describe the formula and family.

        Returns
        -------
        tuple[str, str]
            Formula text and ``family(link)``.
        """
        fit = self._pp_fit()
        return (
            fit.formula.formula,
            f"{fit.family.name}({fit.family.location_link.name})",
        )

    def _pp_residual(
        self, names: Mapping[str, torch.Tensor]
    ) -> (
        tuple[
            list[tuple[str, torch.Tensor]], dict[tuple[str, str], torch.Tensor] | None
        ]
        | None
    ):
        """Return the residual ``sigma`` block when it is not regressed.

        Parameters
        ----------
        names : Mapping[str, torch.Tensor]
            brms-named draws.

        Returns
        -------
        tuple or None
            The response's ``sigma`` draws, or None.
        """
        fit = self._pp_fit()
        residual = fit.family.residual_scale
        if (
            residual is None
            or residual.name not in names
            or residual.name in fit.distributional
        ):
            return None
        return [(fit.formula.response_name, names[residual.name])], None

    def _pp_nobs(self, resp: str | None) -> int:
        """Return the number of fitted rows.

        Parameters
        ----------
        resp : str or None
            Must be None.

        Returns
        -------
        int
            Rows of the response.

        Raises
        ------
        ValueError
            If ``resp`` is given.
        """
        if resp is not None:
            raise ValueError("resp applies only to joint (multivariate) fits")
        return int(self._pp_fit().formula.response_values.shape[0])

    def nobs(self) -> int:
        """Return the number of observations used in fitting, as brms ``nobs``.

        Returns
        -------
        int
            Fitted rows.
        """
        return self._pp_nobs(None)

    def _pp_view(self, draw_ids: Sequence[int] | None, seed: int) -> BayesianFit:
        """Return the fit, or a view of selected posterior draws.

        Parameters
        ----------
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        seed : int
            Seed for variational draws.

        Returns
        -------
        BayesianFit
            The fit itself, or a view replaying only the selected draws.
        """
        fit = self._pp_fit()
        if draw_ids is None:
            return fit
        needed = max(draw_ids) + 1 if draw_ids else 1
        source = latent_draws(
            fit,
            num_samples=None if isinstance(fit.posterior, MCMCResult) else needed,
            seed=seed,
        )
        return fit.with_(posterior=selected_draws(source, draw_ids))

    def _pp_prediction(
        self,
        method: PredictionMethod,
        options: SinglePredictionOptions,
        *,
        draw_ids: Sequence[int] | None,
        parameter: str = "mu",
        transform: bool = False,
    ) -> torch.Tensor:
        """Return draws of one prediction method.

        Parameters
        ----------
        method : PredictionMethod
            Prediction function.
        options : SinglePredictionOptions
            Prediction keywords.
        draw_ids : Sequence[int] or None
            Explicit pooled draws.
        parameter : str
            Predicted parameter of ``posterior_linpred``.
        transform : bool
            Inverse link of ``posterior_linpred``.

        Returns
        -------
        torch.Tensor
            Draws ``(chain, draw, row[, event])``.
        """
        view = self._pp_view(draw_ids, options.get("seed", 0))
        request = dict(options)
        if draw_ids is not None:
            request.pop("num_samples", None)
        if method == "posterior_predict":
            return view.posterior_predict(**cast(SinglePredictionOptions, request))
        if method == "posterior_epred":
            return view.posterior_epred(**cast(SinglePredictionOptions, request))
        return view.posterior_linpred(
            parameter=parameter,
            transform=transform,
            **cast(SinglePredictionOptions, request),
        )

    def _pp_categorical(self) -> bool:
        """Report whether responses are ordinal or categorical codes.

        Returns
        -------
        bool
            True for ordinal and categorical families.
        """
        fit = self._pp_fit()
        return fit.family.category_response

    def predict(
        self,
        *,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[SinglePredictionOptions],
    ) -> SummaryTable | torch.Tensor:
        """Summarize posterior predictive draws, as brms ``predict``.

        Parameters
        ----------
        summary : bool
            Summarize; False returns `posterior_predict` draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        draw_ids : Sequence[int] or None
            0-based positions of the pooled draws to use (brms ``draw_ids``
            minus one); None uses every draw.
        **options : SinglePredictionOptions
            `posterior_predict` keywords such as ``new_data``.

        Returns
        -------
        SummaryTable or torch.Tensor
            ``Estimate``, ``Est.Error`` and quantiles per row; for ordinal
            and categorical responses, category frequencies
            ``P(Y = <category>)`` labeled by the fitted categories.
        """
        draws = self._pp_prediction("posterior_predict", options, draw_ids=draw_ids)
        if not summary:
            return draws
        return summarize_prediction(
            draws,
            categories=self._pp_fit().category_labels,
            discrete_table=self._pp_categorical(),
            probs=probs,
            robust=robust,
        )

    def fitted(
        self,
        *,
        scale: Literal["response", "linear"] = "response",
        parameter: str | None = None,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[SinglePredictionOptions],
    ) -> SummaryTable | torch.Tensor:
        """Summarize expected values or linear predictors, as brms ``fitted``.

        Parameters
        ----------
        scale : {"response", "linear"}
            Expected responses (`posterior_epred`) or linear predictors
            (`posterior_linpred`).
        parameter : str or None
            A distributional or nonlinear parameter (brms ``dpar``/``nlpar``);
            on the response scale its inverse-link value is returned.
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : SinglePredictionOptions
            Prediction keywords such as ``new_data``.

        Returns
        -------
        SummaryTable or torch.Tensor
            Per-row statistics; category probabilities of ordinal and
            categorical responses form a three-way table with a category axis.

        Raises
        ------
        ValueError
            If the scale is unknown.
        """
        if scale not in {"response", "linear"}:
            raise ValueError("scale must be 'response' or 'linear'")
        fit = self._pp_fit()
        if parameter is None and scale == "response":
            draws = self._pp_prediction("posterior_epred", options, draw_ids=draw_ids)
            labels = fit.category_labels
        else:
            name = parameter or "mu"
            draws = self._pp_prediction(
                "posterior_linpred",
                options,
                draw_ids=draw_ids,
                parameter=name,
                transform=scale == "response",
            )
            labels = fit.predictor_labels(name)
        if not summary:
            return draws
        return summarize_prediction(
            draws, categories=labels, discrete_table=False, probs=probs, robust=robust
        )

    def predictive_error(
        self,
        *,
        method: str = "posterior_predict",
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[SinglePredictionOptions],
    ) -> torch.Tensor:
        """Return draws of observed minus predicted responses, as brms ``predictive_error``.

        Parameters
        ----------
        method : str
            ``posterior_predict`` (default), ``posterior_epred`` or
            ``posterior_linpred``, or one of brms's aliases.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : SinglePredictionOptions
            Prediction keywords; ``new_data`` must hold the response.

        Returns
        -------
        torch.Tensor
            Float64 draws ``(chain, draw, row)``.

        Raises
        ------
        ValueError
            For ordinal or categorical responses, where errors are undefined.
        """
        return self._pp_errors(method, draw_ids, options, pearson=False)

    def _pp_errors(
        self,
        method: str,
        draw_ids: Sequence[int] | None,
        options: SinglePredictionOptions,
        *,
        pearson: bool,
    ) -> torch.Tensor:
        """Compute ordinary or Pearson predictive errors.

        Parameters
        ----------
        method : str
            Prediction method or alias.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        options : SinglePredictionOptions
            Prediction keywords.
        pearson : bool
            Divide by the per-row predictive standard deviation.

        Returns
        -------
        torch.Tensor
            Error draws.

        Raises
        ------
        ValueError
            For ordinal or categorical responses.
        """
        if self._pp_categorical():
            raise ValueError(
                "predictive errors are not defined for ordinal or categorical models"
            )
        canonical = prediction_method(method)
        predicted = self._pp_prediction(canonical, options, draw_ids=draw_ids)
        observed = _responses(self._pp_fit(), cast(ExpectationOptions, options))
        scale = None
        if pearson:
            reference = self._pp_prediction(
                "posterior_predict", options, draw_ids=draw_ids
            )
            pooled = reference.detach().to(torch.float64).flatten(0, 1)
            scale = pooled.std(dim=0, correction=1)
        return predictive_errors(observed, predicted, scale=scale)

    def residuals(
        self,
        *,
        method: str = "posterior_predict",
        type: Literal["ordinary", "pearson"] = "ordinary",
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[SinglePredictionOptions],
    ) -> SummaryTable | torch.Tensor:
        """Summarize predictive errors, as brms ``residuals``.

        Parameters
        ----------
        method : str
            Prediction method or alias behind the errors.
        type : {"ordinary", "pearson"}
            ``y - yrep``, or that divided by the predictive standard
            deviation of each row (the ``Est.Error`` of `predict`).
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : SinglePredictionOptions
            Prediction keywords; ``new_data`` must hold the response.

        Returns
        -------
        SummaryTable or torch.Tensor
            Per-row statistics, or error draws.

        Raises
        ------
        ValueError
            If the type is unknown or the response is ordinal or categorical.
        """
        if type not in {"ordinary", "pearson"}:
            raise ValueError("type must be 'ordinary' or 'pearson'")
        draws = self._pp_errors(method, draw_ids, options, pearson=type == "pearson")
        if not summary:
            return draws
        return summary_table(draws, probs=probs, robust=robust)

    def predictive_interval(
        self, *, prob: float = 0.9, **options: Unpack[SinglePredictionOptions]
    ) -> SummaryTable:
        """Compute central posterior predictive intervals, as brms ``predictive_interval``.

        Parameters
        ----------
        prob : float
            Central mass, 0.9 as in brms.
        **options : SinglePredictionOptions
            `posterior_predict` keywords.

        Returns
        -------
        SummaryTable
            Lower and upper bounds per row.
        """
        return predictive_table(self._pp_fit().posterior_predict(**options), prob=prob)

    def loo(
        self,
        *,
        moment_match: bool = False,
        reloo: bool = False,
        k_threshold: float = 0.7,
        moment_match_args: Mapping[str, int | bool] | None = None,
        method_factory: MethodFactory | None = None,
        use_stored: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Estimate PSIS-LOO, as brms ``loo``.

        Parameters
        ----------
        moment_match : bool
            Moment-match units with Pareto shape above ``k_threshold``
            [@paananen2021implicitly].
        reloo : bool
            Refit the model once per unit still above ``k_threshold``.
        k_threshold : float
            Pareto shape threshold, 0.7 as in brms.
        moment_match_args : Mapping or None
            ``max_iters`` (30), ``split`` (True) and ``cov`` (True).
        method_factory : MethodFactory or None
            Fit settings of each exact refit; None keeps the original
            settings with the refit's own seed.
        use_stored : bool
            Return a ``loo`` criterion stored by `add_criterion` when no
            correction is requested.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and refits.

        Returns
        -------
        ELPDEstimate
            PSIS-LOO estimate with Pareto diagnostics.
        """
        if (
            use_stored
            and not (moment_match or reloo)
            and "loo" in self._pp_fit().criteria
        ):
            return cast(ELPDEstimate, self._pp_fit().criteria["loo"])
        return loo_estimate(
            self._pp_fit(),
            joint=False,
            resp=None,
            moment_match_options=(moment_match_args or {}) if moment_match else None,
            reloo_factory=(method_factory or keep_settings) if reloo else None,
            k_threshold=k_threshold,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_moment_match(
        self,
        loo: ELPDEstimate | None = None,
        *,
        k_threshold: float = 0.7,
        max_iters: int = 30,
        split: bool = True,
        cov: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Moment-match problematic PSIS-LOO units, as brms ``loo_moment_match``.

        Parameters
        ----------
        loo : ELPDEstimate or None
            An estimate whose response hash must match this fit; None
            recomputes PSIS-LOO.
        k_threshold : float
            Units above this Pareto shape are moment matched.
        max_iters : int
            Maximum iterations per unit.
        split : bool
            Split moment matching with multiple importance sampling.
        cov : bool
            Also match covariances.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        ELPDEstimate
            Updated estimate.

        Raises
        ------
        ValueError
            If ``loo`` scores different responses.
        """
        result = loo_estimate(
            self._pp_fit(),
            joint=False,
            resp=None,
            moment_match_options={"max_iters": max_iters, "split": split, "cov": cov},
            reloo_factory=None,
            k_threshold=k_threshold,
            num_samples=num_samples,
            seed=seed,
        )
        if loo is not None and loo.response_hash != result.response_hash:
            raise ValueError("response values used in loo and in this fit differ")
        return result

    def waic(self, *, num_samples: int | None = None, seed: int = 0) -> ELPDEstimate:
        """Estimate WAIC, as brms ``waic``.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        ELPDEstimate
            WAIC estimate.
        """
        return waic_from_fit(
            self._pp_fit(), joint=False, resp=None, num_samples=num_samples, seed=seed
        )

    def psis(self, *, num_samples: int | None = None, seed: int = 0) -> PSISWeights:
        """Return PSIS leave-one-out weights, as brms ``psis``.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        PSISWeights
            Smoothed log weights and Pareto diagnostics.
        """
        scores = criterion_scores(
            self._pp_fit(), joint=False, resp=None, num_samples=num_samples, seed=seed
        )
        return psis_weights(
            scores.log_likelihood, r_eff=relative_efficiency(scores.analysis)
        )

    def loo_subsample(
        self,
        *,
        observations: int | Sequence[int] = 400,
        approximation: Literal["plpd", "lpd"] = "plpd",
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Estimate PSIS-LOO from a row subsample, as brms ``loo_subsample``.

        Parameters
        ----------
        observations : int or Sequence[int]
            Subsample size (400 as in loo) or explicit 0-based rows.
        approximation : {"plpd", "lpd"}
            Approximation of every row's term: the log density at the
            posterior mean (default) or the log predictive density.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed of the subsample and variational draws.

        Returns
        -------
        ELPDEstimate
            Subsampled estimate with total and subsampling standard errors.
        """
        return loo_subsample_estimate(
            self._pp_fit(),
            joint=False,
            resp=None,
            observations=observations,
            approximation=approximation,
            num_samples=num_samples,
            seed=seed,
        )

    def _pp_loo_expectation(
        self,
        method: PredictionMethod,
        *,
        type: Literal["mean", "var", "quantile"],
        probs: Sequence[float],
        num_samples: int | None,
        seed: int,
    ) -> LooExpectation:
        """Weight prediction draws by PSIS leave-one-out weights.

        Parameters
        ----------
        method : PredictionMethod
            Prediction function.
        type : {"mean", "var", "quantile"}
            Weighted statistic.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and responses.

        Returns
        -------
        LooExpectation
            Weighted statistics per row.
        """
        weights = self.psis(num_samples=num_samples, seed=seed)
        draws = self._pp_prediction(
            method, {"num_samples": num_samples, "seed": seed}, draw_ids=None
        )
        return loo_expectation(draws, weights, kind=type, probs=probs)

    def loo_predict(
        self,
        *,
        type: Literal["mean", "var", "quantile"] = "mean",
        probs: Sequence[float] = (0.5,),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> LooExpectation:
        """Compute leave-one-out predictive statistics, as brms ``loo_predict``.

        Parameters
        ----------
        type : {"mean", "var", "quantile"}
            Weighted statistic of `posterior_predict` draws.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and responses.

        Returns
        -------
        LooExpectation
            Statistics per training row with function-specific Pareto shapes.
        """
        return self._pp_loo_expectation(
            "posterior_predict",
            type=type,
            probs=probs,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_epred(
        self,
        *,
        type: Literal["mean", "var", "quantile"] = "mean",
        probs: Sequence[float] = (0.5,),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> LooExpectation:
        """Compute leave-one-out expected responses, as brms ``loo_epred``.

        Parameters
        ----------
        type : {"mean", "var", "quantile"}
            Weighted statistic of `posterior_epred` draws.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        LooExpectation
            Statistics per training row.
        """
        return self._pp_loo_expectation(
            "posterior_epred",
            type=type,
            probs=probs,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_linpred(
        self,
        *,
        type: Literal["mean", "var", "quantile"] = "mean",
        probs: Sequence[float] = (0.5,),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> LooExpectation:
        """Compute leave-one-out linear predictors, as brms ``loo_linpred``.

        Parameters
        ----------
        type : {"mean", "var", "quantile"}
            Weighted statistic of `posterior_linpred` draws.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        LooExpectation
            Statistics per training row.
        """
        return self._pp_loo_expectation(
            "posterior_linpred",
            type=type,
            probs=probs,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_predictive_interval(
        self, *, prob: float = 0.9, num_samples: int | None = None, seed: int = 0
    ) -> LooExpectation:
        """Compute leave-one-out predictive intervals, as brms ``loo_predictive_interval``.

        Parameters
        ----------
        prob : float
            Central mass, 0.9 as in brms.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and responses.

        Returns
        -------
        LooExpectation
            Weighted quantiles ``(1 - prob) / 2`` and ``1 - (1 - prob) / 2``.

        Raises
        ------
        ValueError
            If the mass is not strictly inside (0, 1).
        """
        if not 0 < prob < 1:
            raise ValueError("prob must lie strictly between zero and one")
        alpha = (1 - prob) / 2
        return self.loo_predict(
            type="quantile",
            probs=(alpha, 1 - alpha),
            num_samples=num_samples,
            seed=seed,
        )

    def loo_r2(
        self,
        *,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> SummaryTable | torch.Tensor:
        """Compute the leave-one-out R-squared, as brms ``loo_R2``.

        Parameters
        ----------
        summary : bool
            Summarize; False returns the bootstrap draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed of the Dirichlet weights and variational draws.

        Returns
        -------
        SummaryTable or torch.Tensor
            A ``R2`` row, or S bootstrap draws.

        Raises
        ------
        ValueError
            For unordered categorical responses.
        """
        fit = self._pp_fit()
        if fit.family.unordered_categories:
            raise ValueError("loo_R2 is not defined for unordered categorical models")
        if "loo_R2" in fit.criteria:
            values = cast(torch.Tensor, fit.criteria["loo_R2"])
        else:
            weights = self.psis(num_samples=num_samples, seed=seed)
            expected = fit.posterior_epred(num_samples=num_samples, seed=seed)
            if expected.ndim == 4:
                scores = torch.arange(expected.shape[-1], dtype=torch.float64)
                expected = (expected.to(torch.float64) * scores).sum(-1)
            loo_mean = loo_expectation(expected, weights, kind="mean").table.values[
                :, 0
            ]
            observed = _responses(fit, ExpectationOptions())
            values = loo_r2_draws(
                observed,
                loo_mean,
                torch.Generator().manual_seed(seed),
                expected.shape[0] * expected.shape[1],
            )
        if not summary:
            return values
        return summary_table({"R2": values.reshape(1, -1)}, probs=probs, robust=robust)

    def kfold(
        self,
        *,
        K: int = 10,
        Ksub: int | Sequence[int] | None = None,
        folds: Literal["random", "stratified", "grouped", "loo"]
        | Sequence[int | str]
        | None = None,
        group: str | None = None,
        joint: Literal["obs", "fold", "group"] | bool = "obs",
        save_fits: bool = False,
        method_factory: MethodFactory | None = None,
        num_samples: int | None = None,
        seed: int = 0,
        allow_new_levels: bool = True,
        sample_new_levels: NewLevelPolicy = "gaussian",
    ) -> KFoldResult:
        """Run k-fold cross-validation, as brms ``kfold``.

        Parameters
        ----------
        K : int
            Number of folds.
        Ksub : int or Sequence[int] or None
            Number of randomly chosen folds, or explicit 0-based folds.
        folds : str or Sequence or None
            None (random folds, or one fold per level of ``group``),
            ``"loo"`` (one fold per row; with ``group``, each fold omits the
            row's whole group), ``"stratified"``, ``"grouped"``, or one fold
            label per row.
        group : str or None
            Grouping column.
        joint : {"obs", "fold", "group"} or bool
            Predictive unit: rows, whole folds, or groups within folds.
        save_fits : bool
            Retain the refits for `kfold_predict`.
        method_factory : MethodFactory or None
            Fit settings per fold; None keeps the original settings with the
            fold's seed.
        num_samples : int or None
            Draws per chain of held-out scoring.
        seed : int
            Partition and refit seed.
        allow_new_levels : bool
            Score held-out rows with new group levels, True as in brms.
        sample_new_levels : NewLevelPolicy
            New-level policy, ``"gaussian"`` as in brms's k-fold.

        Returns
        -------
        KFoldResult
            The ``kfold`` criterion and the refits.
        """
        return kfold_from_fit(
            self._pp_fit(),
            joint=False,
            resp=None,
            K=K,
            Ksub=Ksub,
            folds=folds,
            group=group,
            joint_unit=joint,
            save_fits=save_fits,
            method_factory=method_factory or keep_settings,
            num_samples=num_samples,
            seed=seed,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
        )

    def add_criterion(
        self,
        criterion: str | Sequence[str],
        *,
        overwrite: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
        kfold_options: Mapping[str, object] | None = None,
    ) -> SingleFit:
        """Return the fit with criteria stored, as brms ``add_criterion``.

        Parameters
        ----------
        criterion : str or Sequence[str]
            Any of ``loo``, ``waic``, ``kfold``, ``loo_subsample``,
            ``bayes_R2`` and ``loo_R2``.
        overwrite : bool
            Recompute criteria already stored.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and resampling.
        kfold_options : Mapping or None
            Keywords of `kfold`.

        Returns
        -------
        BayesianFit
            A fit sharing this fit's model with the criteria stored; they
            persist through `save_fit` and `load_fit`.

        Raises
        ------
        ValueError
            If a criterion is unknown.
        """
        names = [criterion] if isinstance(criterion, str) else list(criterion)
        stored = dict(self._pp_fit().criteria)
        for name in names:
            if name in stored and not overwrite:
                continue
            stored[name] = self._pp_compute_criterion(
                name, num_samples=num_samples, seed=seed, kfold_options=kfold_options
            )
        return self._pp_fit().with_(criteria=stored)

    def _pp_compute_criterion(
        self,
        name: str,
        *,
        num_samples: int | None,
        seed: int,
        kfold_options: Mapping[str, object] | None,
    ) -> ELPDEstimate | torch.Tensor:
        """Compute one storable criterion.

        Parameters
        ----------
        name : str
            Criterion name.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed.
        kfold_options : Mapping or None
            Keywords of `kfold`.

        Returns
        -------
        ELPDEstimate or torch.Tensor
            Estimate, or R-squared draws.

        Raises
        ------
        ValueError
            If the name is unknown.
        """
        fit = self._pp_fit()
        if name == "loo":
            return self.loo(use_stored=False, num_samples=num_samples, seed=seed)
        if name == "waic":
            return self.waic(num_samples=num_samples, seed=seed)
        if name == "loo_subsample":
            return self.loo_subsample(num_samples=num_samples, seed=seed)
        if name == "kfold":
            options = dict(kfold_options or {})
            options.setdefault("seed", seed)
            return self.kfold(**cast(dict, options)).estimate
        if name == "bayes_R2":
            return fit.bayes_r2(num_samples=num_samples, seed=seed).flatten(0, 1)
        if name == "loo_R2":
            return cast(
                torch.Tensor,
                self.loo_r2(summary=False, num_samples=num_samples, seed=seed),
            )
        raise ValueError(
            f"unknown criterion {name!r}; expected loo, waic, kfold, loo_subsample, "
            "bayes_R2 or loo_R2"
        )


class JointPostprocess(_PostprocessCore):
    """brms post-processing methods of a joint `MultivariateFit`.

    Per-response results are dictionaries keyed by response, where brms
    returns arrays with a response axis. Criteria score the units of
    `MultivariateFit.log_likelihood`: one row's joint event under residual
    correlation, or the per-row sum over the selected responses.
    """

    def _pp_fit(self) -> MultivariateFit:
        """Return this object as a `MultivariateFit`.

        Returns
        -------
        MultivariateFit
            The fit.
        """
        return cast("MultivariateFit", self)

    def _criterion_key(self, name: str, responses: tuple[str, ...]) -> str:
        """Key a criterion by its responses in the fit's authored order.

        Parameters
        ----------
        name : str
            Criterion name.
        responses : tuple[str, ...]
            Selected response labels.

        Returns
        -------
        str
            The criterion name for the whole fit, or the name and selection.
        """
        if responses == self._pp_fit().response_names:
            return name
        return f"{name}:{json.dumps(responses, separators=(',', ':'))}"

    def _stored_criterion(
        self, name: str, responses: tuple[str, ...]
    ) -> ELPDEstimate | torch.Tensor | None:
        """Read a stored criterion of exactly the selected responses.

        Parameters
        ----------
        name : str
            Criterion name.
        responses : tuple[str, ...]
            Selected response labels.

        Returns
        -------
        ELPDEstimate or torch.Tensor or None
            The matching stored value, if present.
        """
        stored = self._pp_fit().criteria
        value = stored.get(self._criterion_key(name, responses))
        if value is None:
            old = stored.get(name)
            if isinstance(old, ELPDEstimate) and old.responses == responses:
                return old
        if isinstance(value, ELPDEstimate) and value.responses != responses:
            return None
        return value

    def criterion(
        self, name: str, *, resp: ResponseSelection = None
    ) -> ELPDEstimate | torch.Tensor:
        """Return a criterion stored by `add_criterion`.

        Parameters
        ----------
        name : str
            Criterion name.
        resp : str or Sequence[str] or None
            Response selection. None prefers the whole-fit criterion if it is
            stored, then the most recently stored selection of this criterion.

        Returns
        -------
        ELPDEstimate or torch.Tensor
            The stored estimate or R-squared draws.

        Raises
        ------
        KeyError
            If the criterion is not stored for the selection.
        """
        fit = self._pp_fit()
        if resp is None:
            stored = fit.criteria
            value = stored.get(name)
            if value is not None:
                return value
            for key in reversed(tuple(stored)):
                if key.startswith(f"{name}:"):
                    return stored[key]
        else:
            value = self._stored_criterion(name, fit._selection(resp))
            if value is not None:
                return value
        raise KeyError(f"criterion {name!r} is not stored; call add_criterion first")

    def _pp_named(self, num_samples: int | None, seed: int) -> dict[str, torch.Tensor]:
        """Return the joint fit's brms-named draws.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, torch.Tensor]
            `MultivariateFit.variables`.
        """
        return self._pp_fit().variables(num_samples=num_samples, seed=seed)

    def _pp_threshold_signs(
        self, fixed: Mapping[str, torch.Tensor]
    ) -> dict[str, _ThresholdSign]:
        fit = self._pp_fit()
        signs: dict[str, _ThresholdSign] = {}
        for name, response in fit.emission.responses.items():
            ordinal = response.lens._family.ordinal
            if ordinal is None:
                continue
            prefix = f"{name}_" if len(fit.response_names) > 1 else ""
            signs.update(
                dict.fromkeys(
                    (
                        label
                        for label in fixed
                        if label.startswith(f"{prefix}Intercept[")
                    ),
                    _ordinal_threshold_sign(ordinal),
                )
            )
        return signs

    def _pp_levels(
        self, num_samples: int | None, seed: int
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Return per-level group coefficients labeled as brms's ``ranef``.

        Parameters
        ----------
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, list[tuple[str, torch.Tensor]]]
            Per group, labels ``<resp>_<coef>``, ``<dpar>_<resp>_<coef>`` or
            ``<resp>_<nlpar>_<coef>`` and ``(chain, draw, level)`` draws.
        """
        fit = self._pp_fit()
        result: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for name in fit.response_names:
            view = fit._response_fit(name)
            labels = coefficient_labels(view)
            plan = view.prediction_plan
            outputs = {
                item.prefix: item.output
                for item in (plan.predictors if plan is not None else ())
            }
            for parameter, prefix, predictor in view._predictor_items():
                if not predictor.random_terms:
                    continue
                for item in level_coefficients(
                    fit.random_effects(
                        resp=name,
                        parameter=parameter,
                        num_samples=num_samples,
                        seed=seed,
                    ),
                    name_scope(
                        view, parameter, prefix, predictor, labels, response=name
                    ),
                    outputs.get(prefix),
                ):
                    result.setdefault(item.group, []).append((item.name, item.values))
        return result

    def _pp_dpars(self) -> frozenset[str]:
        """Return the auxiliary parameter names of every response family.

        Returns
        -------
        frozenset[str]
            Auxiliary parameter names.
        """
        fit = self._pp_fit()
        return frozenset(
            aux.name
            for name in fit.response_names
            for aux in fit.prepared[name].family.aux_params
        )

    def _pp_describe(self) -> tuple[str, str]:
        """Describe the joint formula and its families.

        Returns
        -------
        tuple[str, str]
            Response formulas joined by ``+`` and their families.
        """
        fit = self._pp_fit()
        formulas = " + ".join(
            fit.specification.responses[name].formula for name in fit.response_names
        )
        families = ", ".join(
            fit.prepared[name].family.name for name in fit.response_names
        )
        return formulas, families

    def _pp_residual(
        self, names: Mapping[str, torch.Tensor]
    ) -> (
        tuple[
            list[tuple[str, torch.Tensor]], dict[tuple[str, str], torch.Tensor] | None
        ]
        | None
    ):
        """Return residual scales and correlations, as brms's ``residual__``.

        Parameters
        ----------
        names : Mapping[str, torch.Tensor]
            brms-named draws.

        Returns
        -------
        tuple or None
            ``sigma_<resp>`` of each response with an unregressed sigma and,
            under residual correlation, ``rescor__<a>__<b>``; None when any
            response regresses sigma or no response has one.
        """
        fit = self._pp_fit()
        residuals = {
            name: scale.name
            for name in fit.response_names
            if (scale := fit.prepared[name].family.residual_scale) is not None
        }
        if any(
            scale in fit.emission.responses[name].lens._distributional
            for name, scale in residuals.items()
        ):
            return None
        # brms omits the response from names of a one-response model.
        single = len(fit.response_names) == 1
        labelled = {
            name: scale if single else f"{scale}_{name}"
            for name, scale in residuals.items()
        }
        scales = [
            (name, names[label]) for name, label in labelled.items() if label in names
        ]
        if not scales:
            return None
        if not fit.rescor:
            return scales, None
        labels = {label for label, _ in scales}
        pairs = {
            (a, b): names[f"rescor__{a}__{b}"]
            for a in labels
            for b in labels
            if f"rescor__{a}__{b}" in names
        }
        return scales, pairs

    def _pp_nobs(self, resp: str | None) -> int:
        """Return the number of observations of the data or of one response.

        Parameters
        ----------
        resp : str or None
            One response, or None for every original row.

        Returns
        -------
        int
            Row count, as brms ``nobs``.

        Raises
        ------
        ValueError
            If the fit retains no original rows.
        """
        fit = self._pp_fit()
        if resp is not None:
            return int(fit.observed(resp=resp)[resp].shape[0])
        if fit.preparation_data is None:
            raise ValueError("this fit retains no original rows")
        return len(nw.from_native(fit.preparation_data, eager_only=True))

    def nobs(self, *, resp: str | None = None) -> int:
        """Return the number of observations, as brms ``nobs``.

        Parameters
        ----------
        resp : str or None
            One response, counting its included rows; None counts the rows of
            the data.

        Returns
        -------
        int
            Row count.
        """
        return self._pp_nobs(resp)

    def _pp_view(self, draw_ids: Sequence[int] | None, seed: int) -> MultivariateFit:
        """Return the fit, or a view of selected posterior draws.

        Parameters
        ----------
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        seed : int
            Seed for variational draws.

        Returns
        -------
        MultivariateFit
            The fit, or a view replaying the selected draws.
        """
        fit = self._pp_fit()
        if draw_ids is None:
            return fit
        needed = max(draw_ids) + 1 if draw_ids else 1
        source = latent_draws(
            fit,
            num_samples=None if isinstance(fit.posterior, MCMCResult) else needed,
            seed=seed,
        )
        return fit.with_(
            posterior=selected_draws(source, draw_ids),
            response_fits=dict(fit.response_fits),
        )

    def _pp_prediction(
        self,
        method: PredictionMethod,
        resp: ResponseSelection,
        options: JointPredictionOptions,
        *,
        draw_ids: Sequence[int] | None,
        parameter: str = "mu",
        transform: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return draws of one prediction method per selected response.

        Parameters
        ----------
        method : PredictionMethod
            Prediction function.
        resp : str or Sequence[str] or None
            Selected responses.
        options : JointPredictionOptions
            Prediction keywords.
        draw_ids : Sequence[int] or None
            Explicit pooled draws.
        parameter : str
            Predicted parameter of ``posterior_linpred``.
        transform : bool
            Inverse link of ``posterior_linpred``.

        Returns
        -------
        dict[str, torch.Tensor]
            Draws keyed by response.
        """
        view = self._pp_view(draw_ids, options.get("seed", 0))
        request = dict(options)
        if draw_ids is not None:
            request.pop("num_samples", None)
        typed = cast(JointPredictionOptions, request)
        if method == "posterior_predict":
            return view.posterior_predict(resp=resp, **typed)
        if method == "posterior_epred":
            return view.posterior_epred(resp=resp, **typed)
        return view.posterior_linpred(
            resp=resp, parameter=parameter, transform=transform, **typed
        )

    def _pp_categorical(self, name: str) -> bool:
        """Report whether a response is ordinal or categorical.

        Parameters
        ----------
        name : str
            Response.

        Returns
        -------
        bool
            True for ordinal and categorical families.
        """
        return self._pp_fit().prepared[name].family.category_response

    def predict(
        self,
        *,
        resp: ResponseSelection = None,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[JointPredictionOptions],
    ) -> dict[str, SummaryTable] | dict[str, torch.Tensor]:
        """Summarize joint posterior predictive draws, as brms ``predict``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : JointPredictionOptions
            `MultivariateFit.posterior_predict` keywords.

        Returns
        -------
        dict
            Per response, a summary table (category frequencies for ordinal
            and categorical responses), or the draws.
        """
        draws = self._pp_prediction(
            "posterior_predict", resp, options, draw_ids=draw_ids
        )
        if not summary:
            return draws
        fit = self._pp_fit()
        return {
            name: summarize_prediction(
                value,
                categories=fit._predictor_readout(name).category_labels,
                discrete_table=self._pp_categorical(name),
                probs=probs,
                robust=robust,
            )
            for name, value in draws.items()
        }

    def fitted(
        self,
        *,
        resp: ResponseSelection = None,
        scale: Literal["response", "linear"] = "response",
        parameter: str | None = None,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[JointPredictionOptions],
    ) -> dict[str, SummaryTable] | dict[str, torch.Tensor]:
        """Summarize expected values or linear predictors, as brms ``fitted``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        scale : {"response", "linear"}
            Expected responses or linear predictors.
        parameter : str or None
            Distributional or nonlinear parameter of every selected response.
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : JointPredictionOptions
            Prediction keywords.

        Returns
        -------
        dict
            Per response, a summary table or the draws.

        Raises
        ------
        ValueError
            If the scale is unknown.
        """
        if scale not in {"response", "linear"}:
            raise ValueError("scale must be 'response' or 'linear'")
        fit = self._pp_fit()
        if parameter is None and scale == "response":
            draws = self._pp_prediction(
                "posterior_epred", resp, options, draw_ids=draw_ids
            )
            labels = {
                name: fit._predictor_readout(name).category_labels for name in draws
            }
        else:
            name_ = parameter or "mu"
            draws = self._pp_prediction(
                "posterior_linpred",
                resp,
                options,
                draw_ids=draw_ids,
                parameter=name_,
                transform=scale == "response",
            )
            labels = {
                name: fit._predictor_readout(name).predictor_labels(name_)
                for name in draws
            }
        if not summary:
            return draws
        return {
            name: summarize_prediction(
                value,
                categories=labels[name],
                discrete_table=False,
                probs=probs,
                robust=robust,
            )
            for name, value in draws.items()
        }

    def _pp_errors(
        self,
        method: str,
        resp: ResponseSelection,
        draw_ids: Sequence[int] | None,
        options: JointPredictionOptions,
        *,
        pearson: bool,
    ) -> dict[str, torch.Tensor]:
        """Compute ordinary or Pearson predictive errors per response.

        Parameters
        ----------
        method : str
            Prediction method or alias.
        resp : str or Sequence[str] or None
            Selected responses.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        options : JointPredictionOptions
            Prediction keywords.
        pearson : bool
            Divide by the predictive standard deviation of each row.

        Returns
        -------
        dict[str, torch.Tensor]
            Error draws keyed by response.

        Raises
        ------
        ValueError
            For ordinal or categorical responses.
        """
        fit = self._pp_fit()
        names = fit._selection(resp)
        if any(self._pp_categorical(name) for name in names):
            raise ValueError(
                "predictive errors are not defined for ordinal or categorical models"
            )
        canonical = prediction_method(method)
        predicted = self._pp_prediction(canonical, names, options, draw_ids=draw_ids)
        reference = (
            self._pp_prediction("posterior_predict", names, options, draw_ids=draw_ids)
            if pearson
            else {}
        )
        result: dict[str, torch.Tensor] = {}
        for name in names:
            observed = evaluate_responses(
                fit._response_data(name), options.get("new_data")
            )
            scale = (
                reference[name]
                .detach()
                .to(torch.float64)
                .flatten(0, 1)
                .std(dim=0, correction=1)
                if pearson
                else None
            )
            result[name] = predictive_errors(observed, predicted[name], scale=scale)
        return result

    def predictive_error(
        self,
        *,
        resp: ResponseSelection = None,
        method: str = "posterior_predict",
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[JointPredictionOptions],
    ) -> dict[str, torch.Tensor]:
        """Return draws of observed minus predicted responses, as brms ``predictive_error``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        method : str
            Prediction method or alias.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : JointPredictionOptions
            Prediction keywords; ``new_data`` must hold the responses.

        Returns
        -------
        dict[str, torch.Tensor]
            Error draws keyed by response.
        """
        return self._pp_errors(method, resp, draw_ids, options, pearson=False)

    def residuals(
        self,
        *,
        resp: ResponseSelection = None,
        method: str = "posterior_predict",
        type: Literal["ordinary", "pearson"] = "ordinary",
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        draw_ids: Sequence[int] | None = None,
        **options: Unpack[JointPredictionOptions],
    ) -> dict[str, SummaryTable] | dict[str, torch.Tensor]:
        """Summarize predictive errors per response, as brms ``residuals``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        method : str
            Prediction method or alias.
        type : {"ordinary", "pearson"}
            Residual type.
        summary : bool
            Summarize; False returns the draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        draw_ids : Sequence[int] or None
            0-based pooled draw positions.
        **options : JointPredictionOptions
            Prediction keywords.

        Returns
        -------
        dict
            Per response, a summary table or error draws.

        Raises
        ------
        ValueError
            If the type is unknown.
        """
        if type not in {"ordinary", "pearson"}:
            raise ValueError("type must be 'ordinary' or 'pearson'")
        draws = self._pp_errors(
            method, resp, draw_ids, options, pearson=type == "pearson"
        )
        if not summary:
            return draws
        return {
            name: summary_table(value, probs=probs, robust=robust)
            for name, value in draws.items()
        }

    def predictive_interval(
        self,
        *,
        resp: ResponseSelection = None,
        prob: float = 0.9,
        **options: Unpack[JointPredictionOptions],
    ) -> dict[str, SummaryTable]:
        """Compute central predictive intervals per response, as brms ``predictive_interval``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        prob : float
            Central mass, 0.9 as in brms.
        **options : JointPredictionOptions
            Prediction keywords.

        Returns
        -------
        dict[str, SummaryTable]
            Lower and upper bounds per row, keyed by response.
        """
        draws = self._pp_fit().posterior_predict(resp=resp, **options)
        return {
            name: predictive_table(value, prob=prob) for name, value in draws.items()
        }

    def loo(
        self,
        *,
        resp: ResponseSelection = None,
        moment_match: bool = False,
        reloo: bool = False,
        k_threshold: float = 0.7,
        moment_match_args: Mapping[str, int | bool] | None = None,
        method_factory: MethodFactory | None = None,
        use_stored: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Estimate PSIS-LOO over joint units, as brms ``loo``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Scored responses, with the units of `log_likelihood`.
        moment_match : bool
            Moment-match units above ``k_threshold``.
        reloo : bool
            Refit once per unit still above ``k_threshold``.
        k_threshold : float
            Pareto shape threshold, 0.7 as in brms.
        moment_match_args : Mapping or None
            ``max_iters``, ``split`` and ``cov``.
        method_factory : MethodFactory or None
            Fit settings of each refit; None keeps the original settings.
        use_stored : bool
            Return a stored ``loo`` criterion for the selected responses
            when no correction is requested.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and refits.

        Returns
        -------
        ELPDEstimate
            PSIS-LOO estimate.
        """
        fit = self._pp_fit()
        if use_stored and not (moment_match or reloo):
            stored = self._stored_criterion("loo", fit._selection(resp))
            if isinstance(stored, ELPDEstimate):
                return stored
        return loo_estimate(
            fit,
            joint=True,
            resp=resp,
            moment_match_options=(moment_match_args or {}) if moment_match else None,
            reloo_factory=(method_factory or keep_settings) if reloo else None,
            k_threshold=k_threshold,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_moment_match(
        self,
        loo: ELPDEstimate | None = None,
        *,
        resp: ResponseSelection = None,
        k_threshold: float = 0.7,
        max_iters: int = 30,
        split: bool = True,
        cov: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Moment-match problematic joint units, as brms ``loo_moment_match``.

        Parameters
        ----------
        loo : ELPDEstimate or None
            An estimate whose response hash must match; None recomputes.
        resp : str or Sequence[str] or None
            Scored responses.
        k_threshold : float
            Pareto shape threshold.
        max_iters : int
            Maximum iterations per unit.
        split : bool
            Split moment matching.
        cov : bool
            Also match covariances.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        ELPDEstimate
            Updated estimate.

        Raises
        ------
        ValueError
            If ``loo`` scores different responses.
        """
        result = loo_estimate(
            self._pp_fit(),
            joint=True,
            resp=resp,
            moment_match_options={"max_iters": max_iters, "split": split, "cov": cov},
            reloo_factory=None,
            k_threshold=k_threshold,
            num_samples=num_samples,
            seed=seed,
        )
        if loo is not None and loo.response_hash != result.response_hash:
            raise ValueError("response values used in loo and in this fit differ")
        return result

    def waic(
        self,
        *,
        resp: ResponseSelection = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Estimate WAIC over joint units, as brms ``waic``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Scored responses.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        ELPDEstimate
            WAIC estimate.
        """
        return waic_from_fit(
            self._pp_fit(), joint=True, resp=resp, num_samples=num_samples, seed=seed
        )

    def psis(
        self,
        *,
        resp: ResponseSelection = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> PSISWeights:
        """Return PSIS leave-one-out weights of joint units, as brms ``psis``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Scored responses.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        PSISWeights
            Smoothed log weights and Pareto diagnostics.
        """
        scores = criterion_scores(
            self._pp_fit(), joint=True, resp=resp, num_samples=num_samples, seed=seed
        )
        return psis_weights(
            scores.log_likelihood, r_eff=relative_efficiency(scores.analysis)
        )

    def loo_subsample(
        self,
        *,
        resp: ResponseSelection = None,
        observations: int | Sequence[int] = 400,
        approximation: Literal["plpd", "lpd"] = "plpd",
        num_samples: int | None = None,
        seed: int = 0,
    ) -> ELPDEstimate:
        """Estimate PSIS-LOO from a row subsample, as brms ``loo_subsample``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Scored responses.
        observations : int or Sequence[int]
            Subsample size or explicit 0-based rows.
        approximation : {"plpd", "lpd"}
            Approximation of every row's term.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed of the subsample and variational draws.

        Returns
        -------
        ELPDEstimate
            Subsampled estimate.
        """
        return loo_subsample_estimate(
            self._pp_fit(),
            joint=True,
            resp=resp,
            observations=observations,
            approximation=approximation,
            num_samples=num_samples,
            seed=seed,
        )

    def _pp_loo_expectation(
        self,
        method: PredictionMethod,
        resp: ResponseSelection,
        *,
        type: Literal["mean", "var", "quantile"],
        probs: Sequence[float],
        num_samples: int | None,
        seed: int,
    ) -> dict[str, LooExpectation]:
        """Weight each response's draws by the joint units' PSIS weights.

        Parameters
        ----------
        method : PredictionMethod
            Prediction function.
        resp : str or Sequence[str] or None
            Selected responses; the weights use their joint units.
        type : {"mean", "var", "quantile"}
            Weighted statistic.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and responses.

        Returns
        -------
        dict[str, LooExpectation]
            Weighted statistics keyed by response.
        """
        weights = self.psis(resp=resp, num_samples=num_samples, seed=seed)
        draws = self._pp_prediction(
            method, resp, {"num_samples": num_samples, "seed": seed}, draw_ids=None
        )
        return {
            name: loo_expectation(value, weights, kind=type, probs=probs)
            for name, value in draws.items()
        }

    def loo_predict(
        self,
        *,
        resp: ResponseSelection = None,
        type: Literal["mean", "var", "quantile"] = "mean",
        probs: Sequence[float] = (0.5,),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, LooExpectation]:
        """Compute leave-one-out predictive statistics, as brms ``loo_predict``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        type : {"mean", "var", "quantile"}
            Weighted statistic.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and responses.

        Returns
        -------
        dict[str, LooExpectation]
            Statistics per row, keyed by response.
        """
        return self._pp_loo_expectation(
            "posterior_predict",
            resp,
            type=type,
            probs=probs,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_epred(
        self,
        *,
        resp: ResponseSelection = None,
        type: Literal["mean", "var", "quantile"] = "mean",
        probs: Sequence[float] = (0.5,),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, LooExpectation]:
        """Compute leave-one-out expected responses, as brms ``loo_epred``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        type : {"mean", "var", "quantile"}
            Weighted statistic.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, LooExpectation]
            Statistics per row, keyed by response.
        """
        return self._pp_loo_expectation(
            "posterior_epred",
            resp,
            type=type,
            probs=probs,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_linpred(
        self,
        *,
        resp: ResponseSelection = None,
        type: Literal["mean", "var", "quantile"] = "mean",
        probs: Sequence[float] = (0.5,),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, LooExpectation]:
        """Compute leave-one-out linear predictors, as brms ``loo_linpred``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        type : {"mean", "var", "quantile"}
            Weighted statistic.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws.

        Returns
        -------
        dict[str, LooExpectation]
            Statistics per row, keyed by response.
        """
        return self._pp_loo_expectation(
            "posterior_linpred",
            resp,
            type=type,
            probs=probs,
            num_samples=num_samples,
            seed=seed,
        )

    def loo_predictive_interval(
        self,
        *,
        resp: ResponseSelection = None,
        prob: float = 0.9,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, LooExpectation]:
        """Compute leave-one-out predictive intervals, as brms ``loo_predictive_interval``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        prob : float
            Central mass.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed for variational draws and responses.

        Returns
        -------
        dict[str, LooExpectation]
            Weighted quantiles per row, keyed by response.

        Raises
        ------
        ValueError
            If the mass is not strictly inside (0, 1).
        """
        if not 0 < prob < 1:
            raise ValueError("prob must lie strictly between zero and one")
        alpha = (1 - prob) / 2
        return self.loo_predict(
            resp=resp,
            type="quantile",
            probs=(alpha, 1 - alpha),
            num_samples=num_samples,
            seed=seed,
        )

    def bayes_r2(
        self,
        *,
        resp: ResponseSelection = None,
        method: Literal["auto", "model", "residual"] = "auto",
        observed: Mapping[str, torch.Tensor] | None = None,
        mc_samples: int | None = None,
        **options: Unpack[JointPredictionOptions],
    ) -> dict[str, torch.Tensor]:
        """Compute the Bayesian R-squared of each response, as brms ``bayes_R2``.

        Each response is read as the single-response fit reads it
        (`BayesianFit.bayes_r2`): the residual form divides the variance of
        the conditional means over the rows by itself plus the variance of
        the residuals [@gelman2019rsquared]; the model form uses the mean
        conditional variance of each row's law instead, under residual
        correlation the marginal law of the response's coordinate. An
        ordinal response is scored by its category codes.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses, or None for every response.
        method : {"auto", "model", "residual"}
            Requested form; ``auto`` is each response's own default.
        observed : Mapping[str, torch.Tensor] or None
            Explicit observed responses by label, for the residual form.
        mc_samples : int or None
            Monte Carlo budget of conditional means without a closed form.
        **options
            Prediction keywords of `MultivariateFit.posterior_epred`.

        Returns
        -------
        dict[str, torch.Tensor]
            ``R2<resp>`` mapped to ratios on ``(chain, draw)`` axes.

        Raises
        ------
        ValueError
            If a selected response is unordered categorical, has several
            columns, or is censored without explicit observations under the
            residual form.

        Notes
        -----
        Cost is one posterior replay per draw for the conditional means, and
        one more for the model form.
        """
        fit = self._pp_fit()
        new_data = options.get("new_data")
        values: dict[str, torch.Tensor] = {}
        for name in fit._selection(resp):
            data = fit._response_data(name)
            readout = fit._response_readout(name)
            dependence = fit.emission.responses[name].lens._residual

            def expected(
                name: str = name, ordinal: bool = readout.ordinal
            ) -> torch.Tensor:
                means = fit.posterior_epred(
                    resp=name, mc_samples=mc_samples, **options
                )[name]
                if not ordinal and means.ndim != 3:
                    raise ValueError(
                        f"bayes_R2 is defined for one value per row; response {name!r} "
                        f"has conditional means of shape {tuple(means.shape[2:])}"
                    )
                return means

            values[f"R2{name}"] = formula_r2(
                data,
                readout,
                dependence,
                method,
                None if observed is None else observed.get(name),
                new_data=new_data,
                mc_samples=mc_samples,
                recorded=lambda: evaluate_responses(data, new_data),
                expected=expected,
                laws=lambda name=name: fit._response_laws(name, options),
            )
        return values

    def loo_r2(
        self,
        *,
        resp: ResponseSelection = None,
        summary: bool = True,
        robust: bool = False,
        probs: Sequence[float] = (0.025, 0.975),
        num_samples: int | None = None,
        seed: int = 0,
    ) -> SummaryTable | dict[str, torch.Tensor]:
        """Compute leave-one-out R-squared per response, as brms ``loo_R2``.

        Each response uses its own pointwise log-likelihood, as brms does.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Selected responses.
        summary : bool
            Summarize; False returns the bootstrap draws.
        robust : bool
            Median and MAD instead of mean and SD.
        probs : Sequence[float]
            Quantile probabilities.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed of the Dirichlet weights and variational draws.

        Returns
        -------
        SummaryTable or dict[str, torch.Tensor]
            Rows ``R2<resp>``, or draws keyed by ``R2<resp>``.

        Raises
        ------
        ValueError
            For unordered categorical responses.
        """
        fit = self._pp_fit()
        values: dict[str, torch.Tensor] = {}
        for index, name in enumerate(fit._selection(resp)):
            if fit._response_readout(name).unordered:
                raise ValueError(
                    "loo_R2 is not defined for unordered categorical models"
                )
            weights = self.psis(resp=name, num_samples=num_samples, seed=seed)
            expected = fit.posterior_epred(
                resp=name, num_samples=num_samples, seed=seed
            )[name]
            if expected.ndim == 4:
                scores = torch.arange(expected.shape[-1], dtype=torch.float64)
                expected = (expected.to(torch.float64) * scores).sum(-1)
            loo_mean = loo_expectation(expected, weights, kind="mean").table.values[
                :, 0
            ]
            values[f"R2{name}"] = loo_r2_draws(
                evaluate_responses(fit._response_data(name), None),
                loo_mean,
                torch.Generator().manual_seed(seed + index),
                expected.shape[0] * expected.shape[1],
            )
        if not summary:
            return values
        return summary_table(
            {name: value.reshape(1, -1) for name, value in values.items()},
            probs=probs,
            robust=robust,
        )

    def kfold(
        self,
        *,
        resp: ResponseSelection = None,
        K: int = 10,
        Ksub: int | Sequence[int] | None = None,
        folds: Literal["random", "stratified", "grouped", "loo"]
        | Sequence[int | str]
        | None = None,
        group: str | None = None,
        joint: Literal["obs", "fold", "group"] | bool = "obs",
        save_fits: bool = False,
        method_factory: MethodFactory | None = None,
        num_samples: int | None = None,
        seed: int = 0,
        allow_new_levels: bool = True,
        sample_new_levels: NewLevelPolicy = "gaussian",
    ) -> KFoldResult:
        """Run k-fold cross-validation of the joint model, as brms ``kfold``.

        Parameters
        ----------
        resp : str or Sequence[str] or None
            Scored responses.
        K : int
            Number of folds.
        Ksub : int or Sequence[int] or None
            Number of random folds, or explicit 0-based folds.
        folds : str or Sequence or None
            Fold type or labels, as `BayesianFit.kfold`.
        group : str or None
            Grouping column.
        joint : {"obs", "fold", "group"} or bool
            Predictive unit.
        save_fits : bool
            Retain the refits.
        method_factory : MethodFactory or None
            Fit settings per fold.
        num_samples : int or None
            Draws per chain of held-out scoring.
        seed : int
            Partition and refit seed.
        allow_new_levels : bool
            Score new group levels.
        sample_new_levels : NewLevelPolicy
            New-level policy.

        Returns
        -------
        KFoldResult
            The criterion and the refits.
        """
        return kfold_from_fit(
            self._pp_fit(),
            joint=True,
            resp=resp,
            K=K,
            Ksub=Ksub,
            folds=folds,
            group=group,
            joint_unit=joint,
            save_fits=save_fits,
            method_factory=method_factory or keep_settings,
            num_samples=num_samples,
            seed=seed,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
        )

    def add_criterion(
        self,
        criterion: str | Sequence[str],
        *,
        resp: ResponseSelection = None,
        overwrite: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
        kfold_options: Mapping[str, object] | None = None,
    ) -> JointFit:
        """Return the joint fit with criteria stored, as brms ``add_criterion``.

        Parameters
        ----------
        criterion : str or Sequence[str]
            Any of ``loo``, ``waic``, ``kfold``, ``loo_subsample``,
            ``bayes_R2`` and ``loo_R2``.
        resp : str or Sequence[str] or None
            Responses the criterion scores, as brms's ``resp``; None scores
            every response. At a row with an ``mi()`` response, the likelihood
            scores the recorded outcomes given the completed values.
        overwrite : bool
            Recompute stored criteria.
        num_samples : int or None
            Draws per chain.
        seed : int
            Seed.
        kfold_options : Mapping or None
            Keywords of `kfold`.

        Returns
        -------
        MultivariateFit
            A fit sharing this model with the criteria stored.

        Raises
        ------
        ValueError
            If a criterion is unknown.
        """
        fit = self._pp_fit()
        names = [criterion] if isinstance(criterion, str) else list(criterion)
        stored = dict(fit.criteria)
        for name in names:
            options = dict(kfold_options or {}) if name == "kfold" else {}
            selected = fit._selection(
                cast(ResponseSelection, options.get("resp", resp))
            )
            key = self._criterion_key(name, selected)
            legacy = stored.get(name)
            if (
                isinstance(legacy, ELPDEstimate)
                and legacy.responses != fit.response_names
                and legacy.responses
                and set(legacy.responses) <= set(fit.response_names)
            ):
                stored.setdefault(self._criterion_key(name, legacy.responses), legacy)
                del stored[name]
            if key in stored and not overwrite:
                continue
            if name == "loo":
                stored[key] = self.loo(
                    resp=resp, use_stored=False, num_samples=num_samples, seed=seed
                )
            elif name == "waic":
                stored[key] = self.waic(resp=resp, num_samples=num_samples, seed=seed)
            elif name == "loo_subsample":
                stored[key] = self.loo_subsample(
                    resp=resp, num_samples=num_samples, seed=seed
                )
            elif name == "kfold":
                options.setdefault("resp", resp)
                options.setdefault("seed", seed)
                stored[key] = self.kfold(**cast(dict, options)).estimate
            elif name == "loo_R2":
                draws = cast(
                    dict,
                    self.loo_r2(
                        resp=resp, summary=False, num_samples=num_samples, seed=seed
                    ),
                )
                stored[key] = torch.stack(list(draws.values()), dim=-1)
            elif name == "bayes_R2":
                ratios = self.bayes_r2(resp=resp, num_samples=num_samples, seed=seed)
                stored[key] = torch.stack(
                    [value.flatten(0, 1) for value in ratios.values()], dim=-1
                )
            else:
                raise ValueError(
                    f"unknown criterion {name!r}; expected loo, waic, kfold, "
                    "loo_subsample, bayes_R2 or loo_R2"
                )
        return fit.with_(criteria=stored, response_fits=dict(fit.response_fits))


def keep_settings(settings: Mapping[str, object], seed: int) -> Mapping[str, object]:
    """Keep a fit's settings with a refit-specific seed.

    Parameters
    ----------
    settings : Mapping[str, object]
        Canonical fit settings.
    seed : int
        Fold seed.

    Returns
    -------
    Mapping[str, object]
        The settings with ``seed`` replaced, as brms's ``update`` in
        ``kfold`` and ``reloo``.
    """
    return {**settings, "seed": seed}


def kfold_predict(
    result: KFoldResult,
    *,
    method: str = "posterior_predict",
    resp: str | None = None,
    num_samples: int | None = None,
    seed: int = 0,
) -> KFoldPrediction:
    """Predict held-out rows from their own k-fold refits, as brms ``kfold_predict``.

    Parameters
    ----------
    result : KFoldResult
        A `kfold` result computed with ``save_fits=True``.
    method : str
        ``posterior_predict`` (default), ``posterior_epred`` or
        ``posterior_linpred``, or one of brms's aliases.
    resp : str or None
        Response of a joint fit; None for a single response.
    num_samples : int or None
        Draws per chain of each refit.
    seed : int
        Prediction seed.

    Returns
    -------
    KFoldPrediction
        Observed responses and predictions of every held-out row, ordered by
        original row; new group levels are sampled from the fitted Gaussian
        group law, as brms.

    Raises
    ------
    ValueError
        If the refits were not saved or draw counts differ between folds.
    """
    cv = result.cross_validation
    if not cv.fits:
        raise ValueError("kfold_predict requires kfold(save_fits=True)")
    canonical = prediction_method(method)
    observed: dict[int, torch.Tensor] = {}
    predicted: dict[int, torch.Tensor] = {}
    for fold, fitted in cv.fits.items():
        rows = cv.plan.split(fold).test
        positions = [result.positions[row] for row in rows]
        frame = nw.from_native(result.data, eager_only=True)[positions, :].to_native()
        options = {
            "new_data": frame,
            "allow_new_levels": True,
            "sample_new_levels": "gaussian",
            "num_samples": num_samples,
            "seed": seed,
        }
        if result.joint_fit:
            joint_fit = cast("MultivariateFit", fitted)
            name = resp if resp is not None else joint_fit.response_names[0]
            call = getattr(joint_fit, canonical)
            draws = call(resp=name, **options)[name]
            truth = evaluate_responses(joint_fit._response_data(name), frame)
        else:
            single = cast("BayesianFit", fitted)
            draws = getattr(single, canonical)(**options)
            truth = _responses(single, ExpectationOptions(new_data=frame))
        for index, row in enumerate(rows):
            predicted[row] = draws[:, :, index]
            observed[row] = truth[index]
    order = sorted(predicted)
    shapes = {tuple(predicted[row].shape[:2]) for row in order}
    if len(shapes) != 1:
        raise ValueError("refits have different draw counts; set num_samples")
    return KFoldPrediction(
        observed=torch.stack([observed[row] for row in order]),
        predicted=torch.stack([predicted[row] for row in order], dim=2),
        positions=tuple(result.positions[row] for row in order),
    )


type ModelInput = Fit | ELPDEstimate
"""A fit with stored or computable criteria, or a precomputed estimate."""


def _estimate(
    model: ModelInput, criterion: Literal["loo", "waic", "kfold"], *, compute: bool
) -> ELPDEstimate:
    """Return a model's criterion, stored or freshly computed.

    Parameters
    ----------
    model : BayesianFit or MultivariateFit or ELPDEstimate
        A fit or an estimate.
    criterion : {"loo", "waic", "kfold"}
        Requested criterion.
    compute : bool
        Compute PSIS-LOO or WAIC when not stored; k-fold estimates must be
        stored, since they refit the model.

    Returns
    -------
    ELPDEstimate
        The estimate.

    Raises
    ------
    ValueError
        If the criterion is unavailable or an estimate has another criterion.
    """
    if isinstance(model, ELPDEstimate):
        kind = "loo" if model.criterion == "loo_subsample" else model.criterion
        if kind != criterion:
            raise ValueError(f"expected a {criterion} estimate, got {model.criterion}")
        return model
    stored = model.criteria.get(criterion)
    if isinstance(stored, ELPDEstimate):
        return stored
    if not compute or criterion == "kfold":
        raise ValueError(
            f"model has no stored {criterion!r} criterion; call add_criterion first"
        )
    return model.loo() if criterion == "loo" else model.waic()


def loo_compare(
    models: Mapping[str, ModelInput],
    *,
    criterion: Literal["loo", "waic", "kfold"] = "loo",
) -> SummaryTable:
    """Compare models by a stored criterion, as brms ``loo_compare``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit or ELPDEstimate]
        Named fits holding the criterion from `add_criterion`, or estimates.
    criterion : {"loo", "waic", "kfold"}
        Criterion to compare.

    Returns
    -------
    SummaryTable
        The loo package's comparison table, best model first.

    Raises
    ------
    ValueError
        If a fit has no stored criterion, or the models score different data.
    """
    return compare_estimates(
        {
            name: _estimate(model, criterion, compute=False)
            for name, model in models.items()
        }
    )


def loo_model_weights(
    models: Mapping[str, ModelInput],
    *,
    method: Literal["stacking", "pseudobma"] = "stacking",
    bayesian_bootstrap: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Return stacking or pseudo-BMA weights, as brms ``loo_model_weights``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit or ELPDEstimate]
        Named fits or PSIS-LOO estimates; fits without a stored ``loo`` are
        evaluated with `loo`.
    method : {"stacking", "pseudobma"}
        Weighting method.
    bayesian_bootstrap : bool
        Bayesian-bootstrap regularization of pseudo-BMA weights (loo's ``BB``).
    seed : int
        Seed of the Bayesian bootstrap.

    Returns
    -------
    dict[str, float]
        Weights summing to one.
    """
    return stacking_weights(
        {name: _estimate(model, "loo", compute=True) for name, model in models.items()},
        method=method,
        bayesian_bootstrap=bayesian_bootstrap,
        seed=seed,
    )


def model_weights(
    models: Mapping[str, ModelInput],
    *,
    weights: Literal[
        "stacking", "pseudobma", "loo", "waic", "kfold", "bma"
    ] = "stacking",
    seed: int = 0,
) -> dict[str, float]:
    """Return model weights, as brms ``model_weights``.

    Parameters
    ----------
    models : Mapping[str, BayesianFit or MultivariateFit or ELPDEstimate]
        Named fits or estimates.
    weights : {"stacking", "pseudobma", "loo", "waic", "kfold", "bma"}
        Stacking or pseudo-BMA weights of PSIS-LOO, or Akaike-type weights
        ``exp(-ic_diff / 2)`` of an information criterion. k-fold weights
        require stored ``kfold`` criteria. ``"bma"`` gives posterior model
        probabilities from bridge-sampled marginal likelihoods (`post_prob`).
    seed : int
        Seed of the pseudo-BMA Bayesian bootstrap and of bridge sampling.

    Returns
    -------
    dict[str, float]
        Weights summing to one.

    Raises
    ------
    ValueError
        If the weighting method is unknown.
    """
    if weights in {"stacking", "pseudobma"}:
        return loo_model_weights(
            models, method=cast(Literal["stacking", "pseudobma"], weights), seed=seed
        )
    # Posterior model probabilities by bridge sampling.
    if weights == "bma":
        fits = {
            name: model
            for name, model in models.items()
            if not isinstance(model, ELPDEstimate)
        }
        if len(fits) != len(models):
            raise ValueError("bma weights require fits, not ELPD estimates")
        return post_prob(fits, seed=seed)
    if weights not in {"loo", "waic", "kfold"}:
        raise ValueError(
            "weights must be 'stacking', 'pseudobma', 'loo', 'waic', 'kfold' or 'bma'"
        )
    return information_weights(
        {
            name: _estimate(
                model, cast(Literal["loo", "waic", "kfold"], weights), compute=True
            )
            for name, model in models.items()
        }
    )


__all__ = [
    "JointPostprocess",
    "JointPredictionOptions",
    "SinglePostprocess",
    "SinglePredictionOptions",
    "keep_settings",
    "kfold_predict",
    "loo_compare",
    "loo_model_weights",
    "model_weights",
]
