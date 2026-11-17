"""Connect retained raw likelihoods and refit recipes to ArviZ's reloo."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING

from quivers.diagnostics.cross_validation import (
    FoldPlan,
    FoldPrediction,
    FoldSplit,
    _prediction_scores,
)

if TYPE_CHECKING:
    from quivers.diagnostics.posterior import PosteriorAnalysis


def reloo(
    analysis: PosteriorAnalysis,
    refit: Callable[[FoldSplit], FoldPrediction],
    *,
    var_name: str | None,
    k_threshold: float | None,
    seed: int,
    pointwise: bool,
) -> object:
    """Adapt one-row refits without implementing another PSIS algorithm.

    Parameters
    ----------
    analysis : PosteriorAnalysis
        Retained raw observation likelihoods and provenance.
    refit : Callable[[FoldSplit], FoldPrediction]
        Existing fitting and held-out prediction adapter.
    var_name : str or None
        Selected response name.
    k_threshold : float or None
        Pareto threshold, or ArviZ's retained threshold when None.
    seed : int
        Base seed for row-specific refit streams.
    pointwise : bool
        Retain per-observation criterion components.

    Returns
    -------
    arviz.ELPDData
        Criterion updated by ArviZ with the supplied refit scores.

    Raises
    ------
    ValueError, TypeError
        If options, axes or held-out score contracts are invalid.
    """
    import arviz as az
    import numpy as np
    import xarray as xr
    from arviz_stats.loo.wrapper import SamplingWrapper
    from arviz_stats.utils import ELPDData

    from quivers._rng import seeded_rng

    if not callable(refit):
        raise TypeError("refit must be a callable returning FoldPrediction")
    if k_threshold is not None and (
        isinstance(k_threshold, bool)
        or not isinstance(k_threshold, (int, float))
        or math.isnan(k_threshold)
    ):
        raise ValueError("k_threshold must be a real number or None")
    if type(pointwise) is not bool:
        raise TypeError("pointwise must be Boolean")
    name = analysis._likelihood_name(var_name)
    raw = analysis.log_likelihood[name]
    if raw.ndim != 3:
        raise ValueError(
            "reloo requires one observation axis with each response event reduced"
        )
    plan = FoldPlan(assignments=tuple(range(raw.shape[-1])), seed=seed)
    tree = analysis.to_datatree()
    original = analysis.loo(var_name=name, pointwise=True)
    assert isinstance(original, ELPDData)
    threshold = original.good_k if k_threshold is None else float(k_threshold)
    if original.pareto_k is None:
        raise ValueError("ArviZ must return pointwise Pareto diagnostics for reloo")
    # Undefined tail fits do not establish that importance sampling is reliable.
    # An explicit all-row request also includes constant-likelihood columns.
    original.pareto_k = (
        xr.full_like(original.pareto_k, math.inf)
        if threshold == -math.inf
        else original.pareto_k.fillna(math.inf)
    )

    class RefitWrapper(SamplingWrapper):
        """Bridge the canonical partition callback to ArviZ's refit protocol."""

        def sel_observations(self, idx):
            """Return the designated split as training and scoring metadata.

            Parameters
            ----------
            idx : int
                Original positional row requested by ArviZ.

            Returns
            -------
            tuple[FoldSplit, FoldSplit]
                One shared immutable partition record.

            Raises
            ------
            TypeError
                If the request does not identify one positional row.
            """
            # ArviZ passes positional indices for a single observation axis;
            # labels remain in its original tree and never become row positions.
            if not isinstance(idx, (int, np.integer)):
                raise TypeError("ArviZ refit indices must identify one original row")
            split = plan.split(int(idx))
            return split, split

        def sample(self, modified_observed_data):
            """Invoke the shared refit callback with its independent seed.

            Parameters
            ----------
            modified_observed_data : FoldSplit
                Original training and held-out row identities.

            Returns
            -------
            xarray.DataArray
                Raw held-out scores on named chain and draw axes.
            """
            with seeded_rng(modified_observed_data.seed):
                prediction = refit(modified_observed_data)
            scores, _ = _prediction_scores(prediction, modified_observed_data, (name,))
            # A one-row score has no residual observation axis. ArviZ owns
            # log-mean-exp, criterion replacement, effective dimension and SE.
            return xr.DataArray(scores[..., 0].cpu().numpy(), dims=("chain", "draw"))

        def get_inference_data(self, fitted_model):
            """Pass already evaluated raw scores through the ArviZ protocol.

            Parameters
            ----------
            fitted_model : xarray.DataArray
                Scores returned from the existing refit adapter.

            Returns
            -------
            xarray.DataArray
                The same retained scores, without another model execution.
            """
            return fitted_model

        def log_likelihood__i(self, excluded_obs, idata__i):
            """Return the designated row's raw likelihood without replay.

            Parameters
            ----------
            excluded_obs : FoldSplit
                Original held-out row identity retained by the adapter.
            idata__i : xarray.DataArray
                Corresponding raw held-out scores.

            Returns
            -------
            xarray.DataArray
                Named sample axes consumed by ArviZ's scoring reduction.
            """
            return idata__i

    return az.reloo(
        RefitWrapper(model=None, idata_orig=tree),
        loo_orig=original,
        var_name=name,
        k_threshold=threshold,
        pointwise=pointwise,
    )
