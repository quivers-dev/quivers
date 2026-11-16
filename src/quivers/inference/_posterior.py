"""Shared posterior shape contracts for source and formula workflows."""

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
