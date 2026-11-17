"""Posterior component membership of finite mixtures, brms ``pp_mixture``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import torch

from quivers.continuous._observation_events import event_log_prob
from quivers.continuous._zip_hurdle import MixtureNormal
from quivers.continuous.measure import Mixture
from quivers.formulas._likelihood import prepare_likelihood
from quivers.formulas._posterior import collect_draws, replay
from quivers.formulas._prediction import prepare_prediction
from quivers.formulas._response_modifiers import response_input_name
from quivers.inference._posterior import draw_count
from quivers.inference._response import response_distribution
from quivers.inference.trace import Trace

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._group_prediction import NewLevelPolicy


def _membership(
    recorded: Iterator[Trace],
    response: str,
    observations: Mapping[str, torch.Tensor],
    status: str | None,
    upper: str | None,
    bounds: tuple[str | None, str | None],
    rows: int,
    *,
    log: bool,
) -> Iterator[torch.Tensor]:
    """Yield posterior membership probabilities per replayed draw.

    Parameters
    ----------
    recorded : Iterator[Trace]
        Replayed traces.
    response : str
        Response site and input name.
    observations : Mapping[str, torch.Tensor]
        Recorded responses and censoring inputs of the scored rows.
    status, upper : str or None
        Censoring status and interval upper-bound inputs.
    bounds : tuple[str or None, str or None]
        Truncation bound inputs, which the Gaussian kernel applies to each
        component itself.
    rows : int
        Number of scored rows.
    log : bool
        Return log probabilities.

    Yields
    ------
    torch.Tensor
        Shape ``(row, component)`` membership probabilities.

    Raises
    ------
    TypeError
        If the replayed response is not a native mixture.
    """
    value = observations[response]
    for trace in recorded:
        law = response_distribution(trace, response)
        if isinstance(law, MixtureNormal):
            lower, high = (
                None if name is None else observations[name] for name in bounds
            )
            law = (
                Mixture(
                    law.weights,
                    [
                        torch.distributions.Normal(law.loc[..., k], law.scale[..., k])
                        for k in range(law.num_components)
                    ],
                )
                if lower is None and high is None
                else law.componentwise_truncation(lower, high)
            )
        if not isinstance(law, Mixture):
            raise TypeError("pp_mixture requires a fitted native mixture response")
        weights = law.weights
        positive = weights > 0
        log_weights = torch.where(
            positive,
            torch.where(positive, weights, torch.ones_like(weights)).log(),
            torch.full_like(weights, -torch.inf),
        )
        scores = torch.stack(
            torch.broadcast_tensors(
                *(
                    event_log_prob(
                        component,
                        value,
                        status=None if status is None else observations[status],
                        interval_upper=None if upper is None else observations[upper],
                    )
                    for component in law.components
                )
            ),
            dim=-1,
        )
        joint = log_weights + scores
        result = joint - torch.logsumexp(joint, dim=-1, keepdim=True)
        result = result.reshape(-1, result.shape[-1])[-rows:]
        yield result if log else result.exp()


def pp_mixture(
    fit: BayesianFit,
    *,
    new_data: IntoDataFrame | None = None,
    re_formula: str | None = None,
    allow_new_levels: bool = False,
    sample_new_levels: NewLevelPolicy = "uncertainty",
    log: bool = False,
    num_samples: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Posterior probabilities that each observation belongs to each component.

    brms ``pp_mixture``: ``P(K = k | y) = theta_k p_k(y) / sum_j theta_j p_j(y)``
    per posterior draw, with censored rows scored by their recorded event and
    truncated components normalized on their interval. Cost: O(S n K).

    Parameters
    ----------
    fit : BayesianFit
        A fitted mixture model.
    new_data : IntoDataFrame or None
        Rows with recorded responses (and censoring columns); None scores the
        training rows.
    re_formula : str or None
        Varying-effect selector, as for prediction.
    allow_new_levels : bool
        Permit unseen grouping levels in ``new_data``.
    sample_new_levels : str
        Policy for unseen grouping levels.
    log : bool
        Return log probabilities.
    num_samples : int or None
        Draw count under the convention of `BayesianFit.draws`.
    seed : int
        Seed for variational draws and new-level policies.

    Returns
    -------
    torch.Tensor
        Shape ``(chain, draw, row, component)``.

    Raises
    ------
    ValueError
        If the fit is not a mixture model.
    """
    if fit.family.mixture is None:
        raise ValueError("pp_mixture can only be applied to mixture models")
    if new_data is None:
        context = prepare_prediction(
            fit,
            None,
            None,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
            require_trials=False,
        )
    else:
        context = prepare_likelihood(
            fit,
            new_data,
            None,
            re_formula=re_formula,
            allow_new_levels=allow_new_levels,
            sample_new_levels=sample_new_levels,
            seed=seed,
        )
    columns = dict(fit.formula.modifiers.columns())
    status = columns.get("censoring")
    upper = columns.get("censoring_upper")
    lower_bound = columns.get("trunc_lower")
    upper_bound = columns.get("trunc_upper")
    shape = draw_count(fit.posterior, num_samples)
    draws = fit.draws(num_samples=num_samples, seed=seed)
    response = response_input_name(fit.formula)
    with torch.no_grad():
        return collect_draws(
            _membership(
                replay(fit, draws, shape, context=context, score_observations=True),
                response,
                context.observations,
                None if status is None else status.qvr_name,
                None if upper is None else upper.qvr_name,
                (
                    None if lower_bound is None else lower_bound.qvr_name,
                    None if upper_bound is None else upper_bound.qvr_name,
                ),
                context.rows,
                log=log,
            ),
            shape,
        )


__all__ = ["pp_mixture"]
