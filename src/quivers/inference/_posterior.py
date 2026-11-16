"""Shared posterior shape contracts for source and formula workflows."""

import math
from collections.abc import Iterable, Mapping
from contextlib import nullcontext

import torch

from quivers.inference.registry import LatentRegistry
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult


def draw_count(
    posterior: MCMCResult | Guide, num_samples: int | None
) -> tuple[int, int]:
    """Resolve chain and draw counts, retaining recorded chain boundaries."""
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


def constrained_draws(
    posterior: MCMCResult | Guide,
    input: torch.Tensor,
    *,
    num_samples: int | None = None,
    seed: int | None = None,
    registry: LatentRegistry | None = None,
) -> dict[str, torch.Tensor]:
    """Collect constrained posterior draws on chain, draw, and event axes.

    Parameters
    ----------
    posterior : MCMCResult or Guide
        Recorded chains or a fitted guide.
    input : torch.Tensor
        One model execution input for guide sampling.
    num_samples : int or None
        Maximum draws per recorded chain, or fresh guide draws.
    seed : int or None
        Isolated CPU seed, or the caller's current random stream.
    registry : LatentRegistry or None
        Expected latent layout. Defaults to the guide's own registry;
        recorded chains otherwise retain their declared event shapes.

    Returns
    -------
    dict[str, torch.Tensor]
        Site tensors with separate chain and draw dimensions. Recorded
        MCMC tensors are views; guide output is preallocated once per site.

    Raises
    ------
    ValueError
        If draw counts, site names, or constrained shapes are inconsistent.
    """
    chains, count = draw_count(posterior, num_samples)
    if isinstance(posterior, MCMCResult):
        result = {name: value[:, :count] for name, value in posterior.samples.items()}
    else:
        registry = posterior.registry if registry is None else registry
        context = (
            torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        )
        result: dict[str, torch.Tensor] = {}
        with context, torch.no_grad():
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            for draw in range(count):
                sample = posterior.rsample(input)
                if set(sample) != set(registry.names):
                    raise ValueError("guide sites do not match the fitted latent sites")
                for name in registry.names:
                    expected = registry[name].constrained_shape
                    if sample[name].numel() != math.prod(expected):
                        raise ValueError(
                            f"guide site {name!r} has shape {tuple(sample[name].shape)}, "
                            f"expected one constrained event of shape {expected}"
                        )
                    value = sample[name].reshape(expected)
                    if name not in result:
                        result[name] = value.new_empty((1, count, *expected))
                    result[name][0, draw].copy_(value)
    if registry is not None and set(result) != set(registry.names):
        raise ValueError("posterior sites do not match the fitted latent sites")
    for name, value in result.items():
        event_shape = (
            registry[name].constrained_shape
            if registry is not None
            else value.shape[2:]
        )
        expected = (chains, count, *event_shape)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"posterior site {name!r} has shape {tuple(value.shape)}, expected {expected}"
            )
    return result


def collect_site_draws(
    values: Iterable[Mapping[str, torch.Tensor]], shape: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Preallocate rectangular posterior outputs once, preserving named sites.

    Every supplied mapping represents one chain/draw position. Site names,
    event shapes, dtypes and devices must remain constant across draws.
    """
    count = shape[0] * shape[1]
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("posterior output requires positive chain and draw counts")
    result: dict[str, torch.Tensor] = {}
    completed = 0
    with torch.no_grad():
        for index, current in enumerate(values):
            if index >= count:
                raise ValueError("posterior replay produced too many draws")
            if index and current.keys() != result.keys():
                raise ValueError("posterior site names change between draws")
            for name, value in current.items():
                if not index:
                    result[name] = value.new_empty((*shape, *value.shape))
                output = result[name]
                if output.shape[2:] != value.shape:
                    raise ValueError(
                        f"posterior site {name!r} changes shape between draws"
                    )
                if output.dtype != value.dtype or output.device != value.device:
                    raise ValueError(
                        f"posterior site {name!r} changes dtype or device between draws"
                    )
                output[index // shape[1], index % shape[1]].copy_(value)
            completed = index + 1
    if completed != count:
        raise ValueError("posterior replay did not produce every retained draw")
    return result


def collect_draws(
    values: Iterable[torch.Tensor], shape: tuple[int, int]
) -> torch.Tensor:
    """Collect one tensor per posterior draw on explicit chain and draw axes."""
    return collect_site_draws(({"value": value} for value in values), shape)["value"]
