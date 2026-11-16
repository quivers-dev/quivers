"""Prepare one checked source replay for likelihoods and retained analyses."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from quivers._rng import seeded_rng
from quivers.diagnostics.posterior import PosteriorAnalysis, _sample_stats
from quivers.effects.observation_scoring import observation_scoring
from quivers.inference._likelihood import collect_log_likelihood
from quivers.inference._posterior import constrained_draws, draw_count
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.predictive import _require_replayed_sites
from quivers.inference.prior import PriorDraws
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace, trace

if TYPE_CHECKING:
    from quivers.modeling.model import BoundModel


def source_analysis(
    bound: BoundModel,
    posterior: MCMCResult | PriorDraws | Guide,
    *,
    num_samples: int | None = None,
    seed: int | None = None,
) -> PosteriorAnalysis:
    """Replay complete joint draws once using the checked source binding.

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
    PosteriorAnalysis
        Raw response scores and the same sampled tensors, without an artificial
        latent variable when the program has none.

    Raises
    ------
    ValueError
        If required responses or latent values are missing, sample counts or
        shapes disagree, or a retained label is not reached by the program.
    TypeError
        If an observation needs a marginal-likelihood observer or the binding
        has no native tensor executor.
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

        likelihood = collect_log_likelihood(traces(), observations, shape)
    result = PosteriorAnalysis(
        samples=samples,
        log_likelihood=likelihood,
        observed_data={name: values[name] for name in observations},
        sample_stats=_sample_stats(posterior, shape[1]),
        kind="mcmc"
        if isinstance(posterior, MCMCResult)
        else "prior"
        if isinstance(posterior, PriorDraws)
        else "variational",
    )
    result._validate_axes()
    return result


__all__: tuple[str, ...] = ()
