"""Replay retained formula bindings at normalized posterior draws."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch.distributions import Distribution

from quivers.continuous.plate import VectorisedObserve
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace, trace

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def draw_count(
    posterior: MCMCResult | Guide, num_samples: int | None
) -> tuple[int, int]:
    """Resolve chain and draw counts without flattening independent chains.

    Parameters
    ----------
    posterior : MCMCResult or Guide
        The retained posterior representation.
    num_samples : int or None
        Maximum draws per chain, or fresh guide draws.

    Returns
    -------
    tuple[int, int]
        The chain count and draw count, preserving independent chains.

    Raises
    ------
    ValueError
        If the requested count is invalid or no samples were retained.
    """
    if num_samples is not None and (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples < 1
    ):
        raise ValueError("num_samples must be a positive integer")
    if isinstance(posterior, MCMCResult):
        count = (
            posterior.num_samples
            if num_samples is None
            else min(num_samples, posterior.num_samples)
        )
        if count < 1:
            raise ValueError("the posterior has no retained draws")
        return posterior.num_chains, count
    return 1, 100 if num_samples is None else num_samples


def posterior_draws(
    fit: BayesianFit, num_samples: int | None, seed: int
) -> dict[str, torch.Tensor]:
    """Return constrained draws with chain and draw axes, preserving CPU RNG.

    Parameters
    ----------
    fit : BayesianFit
        The fitted formula, native program, observations, and posterior.
    num_samples : int or None
        Maximum draws per chain, or fresh guide draws.
    seed : int
        Seed for guide draws; the caller's CPU RNG is preserved.

    Returns
    -------
    dict[str, torch.Tensor]
        Constrained site draws on chain, draw, and event axes.

    Raises
    ------
    ValueError
        If posterior sites or shapes do not match the fitted program.
    """
    chains, count = draw_count(fit.posterior, num_samples)
    registry = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    if isinstance(fit.posterior, MCMCResult):
        result = {
            name: values[:, :count] for name, values in fit.posterior.samples.items()
        }
    else:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            input = torch.zeros(1, 1, dtype=torch.long)
            result = {}
            for draw in range(count):
                sample = fit.posterior.rsample(input)
                if set(sample) != set(registry.names):
                    raise ValueError(
                        "guide sites do not match the fitted formula's latent sites"
                    )
                for name in registry.names:
                    value = sample[name].reshape(registry[name].constrained_shape)
                    if name not in result:
                        result[name] = value.new_empty((1, count, *value.shape))
                    result[name][0, draw].copy_(value)
    if set(result) != set(registry.names):
        raise ValueError(
            "posterior sites do not match the fitted formula's latent sites"
        )
    for name, values in result.items():
        expected = (chains, count, *registry[name].constrained_shape)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"posterior site {name!r} has shape {tuple(values.shape)}, expected {expected}"
            )
    return result


def replay(
    fit: BayesianFit, draws: dict[str, torch.Tensor], shape: tuple[int, int]
) -> Iterator[Trace]:
    """Replay the retained native program once per draw, without recompilation.

    Parameters
    ----------
    fit : BayesianFit
        The retained fitted program and training observations.
    draws : dict[str, torch.Tensor]
        Validated constrained posterior draws.
    shape : tuple[int, int]
        The chain count and draw count.

    Yields
    ------
    Trace
        One native execution trace per chain and draw.
    """
    with torch.no_grad():
        observations = dict(fit.observations)
        input = torch.zeros(1, 1, dtype=torch.long)
        for chain in range(shape[0]):
            for draw in range(shape[1]):
                latents = {name: values[chain, draw] for name, values in draws.items()}
                yield trace(
                    fit.program,
                    input,
                    observations | latents,
                )


def response_distribution(recorded: Trace, response: str) -> Distribution:
    """Read the actual native response distribution constructed by a replay.

    Parameters
    ----------
    recorded : Trace
        A native execution trace with an observed response.
    response : str
        The normalized response-site name.

    Returns
    -------
    torch.distributions.Distribution
        The native response family at its replayed parameters.

    Raises
    ------
    TypeError
        If the response does not expose a native distribution.
    """
    sampleable = recorded.sites[response].sampleable
    if sampleable is None:
        raise TypeError("the response has no native conditional distribution")
    morphism = sampleable.morphism
    if isinstance(morphism, VectorisedObserve):
        morphism = morphism._family
    builder = getattr(morphism, "_get_dist", None)
    if builder is None:
        raise TypeError(
            "the response family does not expose native distribution expectations"
        )
    return builder(sampleable.input)


def row_binding(value: torch.Tensor, rows: int, categories: int = 0) -> torch.Tensor:
    """Canonicalize a scalar or class-valued binding on retained response rows.

    Parameters
    ----------
    value : torch.Tensor
        A native binding, with or without a shared-row axis.
    rows : int
        Number of retained response rows.
    categories : int
        Number of response categories, or zero for a scalar value.

    Returns
    -------
    torch.Tensor
        The binding on row and optional category axes.
    """
    shape = (rows, categories) if categories else (rows,)
    if categories and value.numel() == categories:
        return value.reshape(1, categories).expand(rows, categories)
    if value.numel() == 1 and not categories:
        return value.reshape(1).expand(rows)
    return value.reshape(shape)
