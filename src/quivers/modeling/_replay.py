"""Prepare one checked source replay for likelihoods and retained analyses."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import torch

from quivers._rng import seeded_rng
from quivers.diagnostics.posterior import PosteriorAnalysis, _sample_stats
from quivers.effects.observation_scoring import observation_scoring
from quivers.inference._likelihood import (
    collect_log_likelihood,
    collect_unit_log_likelihood,
)
from quivers.inference._posterior import constrained_draws, draw_count
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.predictive import _require_replayed_sites
from quivers.inference.prior import PriorDraws
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace, trace
from quivers.inference.units import UnitLogLikelihood

# Pathfinder draws report variational provenance.
from quivers.inference.pathfinder import draw_provenance

if TYPE_CHECKING:
    from quivers.modeling.model import BoundModel

type ScoreUnits = Literal["rows", "factors"]
"""What one score of an analysis covers: a response row, or an independent
unit of the likelihood."""


def _replayed[S](
    bound: BoundModel,
    posterior: MCMCResult | PriorDraws | Guide,
    collect: Callable[[Iterator[Trace], set[str], tuple[int, int]], S],
    *,
    num_samples: int | None,
    seed: int | None,
) -> tuple[S, dict[str, torch.Tensor], tuple[int, int]]:
    """Replay complete joint draws once using the checked source binding.

    Parameters
    ----------
    bound : BoundModel
        Original or compatibly rebound source, responses, and execution input.
    posterior : MCMCResult or PriorDraws or Guide
        Retained count metadata and constrained joint draws.
    collect : Callable
        Reads the conditioned traces, one per chain and draw.
    num_samples : int or None
        Leading draws per recorded chain, or fresh guide draw count.
    seed : int or None
        Isolated stream seed; None retains the caller's stream policy.

    Returns
    -------
    tuple[S, dict[str, torch.Tensor], tuple[int, int]]
        What ``collect`` returned, the constrained draws, and the number of
        chains and draws.

    Raises
    ------
    ValueError
        If required responses or latent values are missing, sample counts or
        shapes disagree, or a retained label is not reached by the program.
    TypeError
        If the binding has no native tensor executor, or ``collect`` refuses
        an observation.
    """
    shape = draw_count(posterior, num_samples)
    program = bound.require_tensor_program()
    observations = set(bound.schema.observations)
    if missing := observations - set(bound.data):
        raise ValueError(f"log likelihood requires observations: {sorted(missing)}")
    values = bound.tensor_data()
    try:
        registry = LatentRegistry.from_model(program, observations)
    except ValueError, NotImplementedError:
        # Dynamic scan addresses and non-bijective latent supports cannot use
        # the static continuous registry. Validate the reached, conditioned
        # execution instead, using that registry's existing trace constructor.
        registry = None
    samples = constrained_draws(
        posterior,
        bound.input,
        num_samples=num_samples,
        seed=seed,
        registry=registry,
    )
    devices = (
        value.device
        for value in (
            bound.input,
            *values.values(),
            *samples.values(),
            *program.parameters(),
            *program.buffers(),
        )
    )
    context = seeded_rng(seed, devices=devices) if seed is not None else nullcontext()
    with context, torch.no_grad(), observation_scoring(True):

        def traces() -> Iterator[Trace]:
            """Yield one fully conditioned trace for each retained position.

            Yields
            ------
            Trace
                Checked native trace, preserving scan occurrence addresses.

            Raises
            ------
            ValueError
                If replay would substitute a newly sampled unresolved latent.
            """
            for chain in range(shape[0]):
                for draw in range(shape[1]):
                    latents = {
                        name: value[chain, draw] for name, value in samples.items()
                    }
                    recorded = trace(program, bound.input, values | latents)
                    _require_replayed_sites(recorded, latents)
                    if registry is None:
                        unresolved = LatentRegistry.from_trace(program, recorded)
                        if unresolved.names:
                            raise ValueError(
                                "posterior is missing latent sites "
                                f"{sorted(unresolved.names)}"
                            )
                    yield recorded

        collected = collect(traces(), observations, shape)
    return collected, samples, shape


def source_analysis(
    bound: BoundModel,
    posterior: MCMCResult | PriorDraws | Guide,
    *,
    num_samples: int | None = None,
    seed: int | None = None,
    units: ScoreUnits = "rows",
) -> PosteriorAnalysis:
    """Prepare a posterior analysis with the raw scores of the observations.

    Parameters
    ----------
    bound : BoundModel
        Original or compatibly rebound source, responses, and execution input.
    posterior : MCMCResult or PriorDraws or Guide
        Retained count metadata and constrained joint draws.
    num_samples : int or None
        Leading draws per recorded chain, or fresh guide draw count.
    seed : int or None
        Isolated stream seed; None retains the caller's stream policy.
    units : ScoreUnits
        ``"rows"`` for one score per response row, and ``"factors"`` for
        one per independent unit of the likelihood.

    Returns
    -------
    PosteriorAnalysis
        Raw scores and the same sampled tensors, without an artificial
        latent variable when the program has none. The analysis holds the
        observed values of the responses whose scores are one per entry.

    Raises
    ------
    ValueError
        If required responses or latent values are missing, sample counts or
        shapes disagree, or a retained label is not reached by the program.
    UnfactoredObservationError
        If scores per row are asked for and the rows of a marginalized
        observation share a latent.
    TypeError
        If nothing scores an observation or the binding has no native tensor
        executor.
    """
    values = bound.tensor_data()
    observed = {
        name: values[name] for name in bound.schema.observations if name in values
    }
    group_dims: dict[str, dict[str, tuple[str, ...]]] = {}
    if units == "rows":
        likelihood, samples, shape = _replayed(
            bound, posterior, collect_log_likelihood, num_samples=num_samples, seed=seed
        )
    else:
        factors, samples, shape = _replayed(
            bound,
            posterior,
            collect_unit_log_likelihood,
            num_samples=num_samples,
            seed=seed,
        )
        likelihood = {key: scores.log_likelihood for key, scores in factors.items()}
        group_dims["log_likelihood"] = {
            key: scores.axes for key, scores in factors.items() if scores.axes
        }
        observed = {
            key: observed[key]
            for key, scores in factors.items()
            if scores.axes is None and key in observed
        }
    result = PosteriorAnalysis(
        samples=samples,
        log_likelihood=likelihood,
        observed_data=observed,
        sample_stats=_sample_stats(posterior, shape[1]),
        kind=draw_provenance(posterior),
        group_dims=group_dims,
    )
    result._validate_axes()
    return result


def source_unit_log_likelihood(
    bound: BoundModel,
    posterior: MCMCResult | PriorDraws | Guide,
    *,
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, UnitLogLikelihood]:
    """Read the log likelihood of each independent unit at posterior draws.

    Parameters
    ----------
    bound : BoundModel
        Original or compatibly rebound source, responses, and execution input.
    posterior : MCMCResult or PriorDraws or Guide
        Retained count metadata and constrained joint draws.
    num_samples : int or None
        Leading draws per recorded chain, or fresh guide draw count.
    seed : int or None
        Isolated stream seed; None retains the caller's stream policy.

    Returns
    -------
    dict[str, UnitLogLikelihood]
        One entry per set of responses whose rows share units.

    Raises
    ------
    ValueError
        If required responses or latent values are missing, sample counts or
        shapes disagree, or the units change between draws.
    TypeError
        If a score's unit weights are not log densities, nothing scores an
        observation, or the binding has no native tensor executor.
    """
    scores, _, _ = _replayed(
        bound,
        posterior,
        collect_unit_log_likelihood,
        num_samples=num_samples,
        seed=seed,
    )
    return scores


__all__: tuple[str, ...] = ()
