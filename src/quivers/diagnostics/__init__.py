"""Posterior diagnostics: convergence estimators and the ArviZ adapter.

Quivers' inference layer produces [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult]
records and [`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide] fits.
This package holds two kinds of analysis over them.

The convergence estimators (`rhat`, `ess`, `mcse`, `summary`) take any
``(num_chains, num_draws, *event_shape)`` tensor and need only torch.
`MCMCResult` reports its diagnostics through the same functions, so
derived quantities and draws from custom sampling loops are diagnosed
with the estimators a run reports.

The simulation-based calibration tools (`sbc`, `sbc_ranks`,
`sbc_uniformity`, `sbc_coverage`) check an inference procedure against
the program it targets; they also need only torch, and are bound on
first access because they build on the inference layer.

Retained uncertainty, covariance, contrast-region evidence, and regression
variance ratios also need only torch. They describe supplied or retained
draws with explicit provenance; they do not infer convergence diagnostics
for a variational approximation.

The ArviZ adapters convert quivers fits into the
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html)
data model of [ArviZ](https://python.arviz.org/) and wrap the
canonical ArviZ entry points (`arviz.compare`, `arviz.loo`,
`arviz.plot_ppc`) with shape-aware, quivers-typed signatures; ArviZ
owns those analytics. They need the ``diagnostics`` extra and are
bound on first access, which also keeps this package importable by
the MCMC driver that the adapter itself imports.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
from types import FunctionType, GenericAlias, UnionType
from typing import TYPE_CHECKING, TypeAliasType

from quivers.diagnostics.convergence import (
    ConvergenceSummary,
    EssMethod,
    McseMethod,
    ess,
    mcse,
    rhat,
    summary,
)

if TYPE_CHECKING:
    import numpy as np

    from quivers.diagnostics.cross_validation import (
        FoldLabel,
        FoldSplit,
        FoldPlan,
        FoldPrediction,
        CrossValidationResult,
        fold_plan,
        cross_validate,
    )

    from quivers.diagnostics.arviz_io import (
        to_datatree,
        to_datatree_any,
        to_datatree_from_sbc,
        to_datatree_from_svi,
    )
    from quivers.diagnostics.posterior import (
        PosteriorAnalysis,
        PosteriorCovariance,
        PosteriorSummary,
        WAICResult,
        waic,
    )
    from quivers.diagnostics.regression import bayesian_r2
    from quivers.diagnostics.conditional import (
        GridValue,
        PredictionGrid,
        ConditionalEffects,
        prediction_grid,
    )
    from quivers.diagnostics.density import (
        PointNullAssumptions,
        PointNullSpec,
        DensityOrdinate,
        DensityRatio,
        log_density_ordinate,
        density_ratio,
    )
    from quivers.diagnostics.contrast import ContrastExpression
    from quivers.diagnostics.hypothesis import (
        HypothesisSummary,
        RegionEvidence,
        hypothesis_summary,
    )
    from quivers.diagnostics.comparison import compare, loo
    from quivers.diagnostics.predictive_checks import (
        loo_pit,
        posterior_predictive_check,
    )
    from quivers.diagnostics._ppc_data import (
        STATISTICS,
        km_data,
        loo_pit_values,
        pit_ecdf_data,
        pit_values,
        ppc_bars_data,
        ppc_data,
        ppc_error_binned_data,
        ppc_error_data,
        ppc_intervals_data,
        ppc_rootogram_data,
        ppc_scatter_avg_data,
        ppc_scatter_data,
        ppc_stat_data,
        predictive_errors,
    )
    from quivers.diagnostics.posterior_tables import (
        SummaryTable,
        posterior_interval,
        posterior_summary,
        posterior_table,
        predictive_interval,
    )
    from quivers.diagnostics.elpd import ELPDEstimate, LooExpectation, PSISWeights
    from quivers.diagnostics.bridge import (
        BridgeRelativeError,
        BridgeRepetitionSpread,
        BridgeResult,
    )
    from quivers.diagnostics.bridge import (
        BridgeMethod,
        ar_spectrum0,
        bayes_factor_from_logml,
        bridge_sampler,
        effective_size,
        near_pd,
        posterior_model_probabilities,
    )
    from quivers.diagnostics.elpd import (
        Criterion,
        compare_estimates,
        from_arviz,
        good_k,
        information_weights,
        kfold_estimate,
        loo_expectation,
        loo_r2_draws,
        pointwise_lpd,
        psis_weights,
        require_psis_draws,
        require_scores,
        response_hash,
        stacking_weights,
        subsample_estimate,
        waic_estimate,
        weighted_quantile,
    )
    from quivers.diagnostics.posterior_tables import (
        MAD_CONSTANT,
    )
    from quivers.diagnostics.calibration import (
        SBCCoverage,
        SBCPosterior,
        SBCResult,
        SBCStatistic,
        SBCThinning,
        sbc,
        sbc_coverage,
        sbc_ranks,
        sbc_uniformity,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "FoldLabel": "quivers.diagnostics.cross_validation",
    "FoldSplit": "quivers.diagnostics.cross_validation",
    "FoldPlan": "quivers.diagnostics.cross_validation",
    "FoldPrediction": "quivers.diagnostics.cross_validation",
    "CrossValidationResult": "quivers.diagnostics.cross_validation",
    "fold_plan": "quivers.diagnostics.cross_validation",
    "cross_validate": "quivers.diagnostics.cross_validation",
    "GridValue": "quivers.diagnostics.conditional",
    "PredictionGrid": "quivers.diagnostics.conditional",
    "ConditionalEffects": "quivers.diagnostics.conditional",
    "prediction_grid": "quivers.diagnostics.conditional",
    "PointNullAssumptions": "quivers.diagnostics.density",
    "PointNullSpec": "quivers.diagnostics.density",
    "DensityOrdinate": "quivers.diagnostics.density",
    "DensityRatio": "quivers.diagnostics.density",
    "log_density_ordinate": "quivers.diagnostics.density",
    "density_ratio": "quivers.diagnostics.density",
    "ContrastExpression": "quivers.diagnostics.contrast",
    "hypothesis_summary": "quivers.diagnostics.hypothesis",
    "HypothesisSummary": "quivers.diagnostics.hypothesis",
    "RegionEvidence": "quivers.diagnostics.hypothesis",
    "bayesian_r2": "quivers.diagnostics.regression",
    "PosteriorAnalysis": "quivers.diagnostics.posterior",
    "PosteriorSummary": "quivers.diagnostics.posterior",
    "PosteriorCovariance": "quivers.diagnostics.posterior",
    "WAICResult": "quivers.diagnostics.posterior",
    "waic": "quivers.diagnostics.posterior",
    "sbc": "quivers.diagnostics.calibration",
    "sbc_ranks": "quivers.diagnostics.calibration",
    "sbc_uniformity": "quivers.diagnostics.calibration",
    "sbc_coverage": "quivers.diagnostics.calibration",
    "SBCResult": "quivers.diagnostics.calibration",
    "SBCCoverage": "quivers.diagnostics.calibration",
    "SBCStatistic": "quivers.diagnostics.calibration",
    "SBCPosterior": "quivers.diagnostics.calibration",
    "SBCThinning": "quivers.diagnostics.calibration",
    "to_datatree": "quivers.diagnostics.arviz_io",
    "to_datatree_from_sbc": "quivers.diagnostics.arviz_io",
    "to_datatree_any": "quivers.diagnostics.arviz_io",
    "to_datatree_from_svi": "quivers.diagnostics.arviz_io",
    "compare": "quivers.diagnostics.comparison",
    "loo": "quivers.diagnostics.comparison",
    "posterior_predictive_check": "quivers.diagnostics.predictive_checks",
    "loo_pit": "quivers.diagnostics.predictive_checks",
    "STATISTICS": "quivers.diagnostics._ppc_data",
    "km_data": "quivers.diagnostics._ppc_data",
    "loo_pit_values": "quivers.diagnostics._ppc_data",
    "pit_ecdf_data": "quivers.diagnostics._ppc_data",
    "pit_values": "quivers.diagnostics._ppc_data",
    "ppc_bars_data": "quivers.diagnostics._ppc_data",
    "ppc_data": "quivers.diagnostics._ppc_data",
    "ppc_error_binned_data": "quivers.diagnostics._ppc_data",
    "ppc_error_data": "quivers.diagnostics._ppc_data",
    "ppc_intervals_data": "quivers.diagnostics._ppc_data",
    "ppc_rootogram_data": "quivers.diagnostics._ppc_data",
    "ppc_scatter_avg_data": "quivers.diagnostics._ppc_data",
    "ppc_scatter_data": "quivers.diagnostics._ppc_data",
    "ppc_stat_data": "quivers.diagnostics._ppc_data",
    "predictive_errors": "quivers.diagnostics._ppc_data",
    "SummaryTable": "quivers.diagnostics.posterior_tables",
    "posterior_summary": "quivers.diagnostics.posterior_tables",
    "posterior_interval": "quivers.diagnostics.posterior_tables",
    "posterior_table": "quivers.diagnostics.posterior_tables",
    "predictive_interval": "quivers.diagnostics.posterior_tables",
    "ELPDEstimate": "quivers.diagnostics.elpd",
    "PSISWeights": "quivers.diagnostics.elpd",
    "LooExpectation": "quivers.diagnostics.elpd",
    "BridgeResult": "quivers.diagnostics.bridge",
    "BridgeRelativeError": "quivers.diagnostics.bridge",
    "BridgeRepetitionSpread": "quivers.diagnostics.bridge",
    "BridgeMethod": "quivers.diagnostics.bridge",
    "ar_spectrum0": "quivers.diagnostics.bridge",
    "bayes_factor_from_logml": "quivers.diagnostics.bridge",
    "bridge_sampler": "quivers.diagnostics.bridge",
    "effective_size": "quivers.diagnostics.bridge",
    "near_pd": "quivers.diagnostics.bridge",
    "posterior_model_probabilities": "quivers.diagnostics.bridge",
    "Criterion": "quivers.diagnostics.elpd",
    "compare_estimates": "quivers.diagnostics.elpd",
    "from_arviz": "quivers.diagnostics.elpd",
    "good_k": "quivers.diagnostics.elpd",
    "information_weights": "quivers.diagnostics.elpd",
    "kfold_estimate": "quivers.diagnostics.elpd",
    "loo_expectation": "quivers.diagnostics.elpd",
    "loo_r2_draws": "quivers.diagnostics.elpd",
    "pointwise_lpd": "quivers.diagnostics.elpd",
    "psis_weights": "quivers.diagnostics.elpd",
    "require_psis_draws": "quivers.diagnostics.elpd",
    "require_scores": "quivers.diagnostics.elpd",
    "response_hash": "quivers.diagnostics.elpd",
    "stacking_weights": "quivers.diagnostics.elpd",
    "subsample_estimate": "quivers.diagnostics.elpd",
    "waic_estimate": "quivers.diagnostics.elpd",
    "weighted_quantile": "quivers.diagnostics.elpd",
    "MAD_CONSTANT": "quivers.diagnostics.posterior_tables",
}

type _LazyExport = (
    FunctionType
    | type
    | TypeAliasType
    | float
    | UnionType
    | GenericAlias
    | dict[str, Callable[[np.ndarray], float]]
)


def __getattr__(name: str) -> _LazyExport:
    """Bind an SBC or ArviZ-backed export on first access.

    Parameters
    ----------
    name : str
        The attribute requested.

    Returns
    -------
    _LazyExport
        The function, class, type alias, or statistic table the
        defining module binds under that name.

    Raises
    ------
    AttributeError
        If ``name`` is not a lazily bound export.
    ModuleNotFoundError
        If the ``diagnostics`` extra (ArviZ, xarray) is not installed.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: _LazyExport = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "FoldLabel",
    "FoldSplit",
    "FoldPlan",
    "FoldPrediction",
    "CrossValidationResult",
    "fold_plan",
    "cross_validate",
    "GridValue",
    "PredictionGrid",
    "ConditionalEffects",
    "prediction_grid",
    "PointNullAssumptions",
    "PointNullSpec",
    "DensityOrdinate",
    "DensityRatio",
    "log_density_ordinate",
    "density_ratio",
    "ContrastExpression",
    "hypothesis_summary",
    "HypothesisSummary",
    "RegionEvidence",
    "bayesian_r2",
    "PosteriorAnalysis",
    "PosteriorSummary",
    "PosteriorCovariance",
    "WAICResult",
    "waic",
    "rhat",
    "ess",
    "mcse",
    "summary",
    "ConvergenceSummary",
    "EssMethod",
    "McseMethod",
    "sbc",
    "sbc_ranks",
    "sbc_uniformity",
    "sbc_coverage",
    "SBCResult",
    "SBCCoverage",
    "SBCStatistic",
    "SBCPosterior",
    "SBCThinning",
    "to_datatree",
    "to_datatree_from_sbc",
    "to_datatree_any",
    "to_datatree_from_svi",
    "compare",
    "loo",
    "posterior_predictive_check",
    "loo_pit",
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
    "SummaryTable",
    "posterior_summary",
    "posterior_interval",
    "posterior_table",
    "predictive_interval",
    "ELPDEstimate",
    "PSISWeights",
    "LooExpectation",
    "BridgeResult",
    "BridgeRelativeError",
    "BridgeRepetitionSpread",
    "BridgeMethod",
    "ar_spectrum0",
    "bayes_factor_from_logml",
    "bridge_sampler",
    "effective_size",
    "near_pd",
    "posterior_model_probabilities",
    "Criterion",
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
    "MAD_CONSTANT",
]
