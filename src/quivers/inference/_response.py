"""Native response laws and moments shared by source and formula prediction."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch.distributions import Distribution, constraints

from quivers.inference.trace import Trace


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
    morphism = getattr(sampleable, "raw_morphism", sampleable.morphism)
    sampling = getattr(morphism, "sampling_distribution", None)
    if sampling is None:
        raise TypeError(
            "the response family does not expose native distribution expectations"
        )
    return sampling(sampleable.input)


@contextmanager
def response_rng(seed: int, tensors: Iterable[torch.Tensor]) -> Iterator[None]:
    """Isolate response randomness on the CPU and all participating devices.

    Parameters
    ----------
    seed : int
        Independent response stream seed.
    tensors : Iterable[torch.Tensor]
        Native posterior/data tensors identifying active CUDA devices.

    Yields
    ------
    None
        Seeded context restoring each caller RNG on exit.
    """
    devices = sorted(
        {tensor.device.index for tensor in tensors if tensor.device.type == "cuda"}
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed % 2**64)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed % 2**64)
        yield


def category_probabilities(distribution: Distribution, categories: int) -> torch.Tensor:
    """Read ordered category probabilities through the existing native law.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        Native discrete response with zero-based category codes.
    categories : int
        Retained category extent.

    Returns
    -------
    torch.Tensor
        Probabilities on batch axes followed by the category axis.
    """
    from quivers.continuous._ordinal import OrdinalDistribution

    if isinstance(distribution, OrdinalDistribution):
        return distribution.category_probs()
    from quivers.continuous._interval import _reference

    reference = _reference(distribution)
    codes = torch.arange(categories, device=reference.device)
    return (
        distribution.log_prob(
            codes.reshape(categories, *([1] * len(distribution.batch_shape)))
        )
        .exp()
        .movedim(0, -1)
    )


def response_mean(
    distribution: Distribution, mc_samples: int | None = None
) -> torch.Tensor:
    """Read a native expectation or explicitly approximate a certified finite mean.

    Parameters
    ----------
    distribution : Distribution
        Native response law after deterministic predictor replay.
    mc_samples : int or None
        Positive Monte Carlo budget for laws without an analytic mean.

    Returns
    -------
    torch.Tensor
        Analytic expectation or sample average, using chunks of at most64.

    Raises
    ------
    ValueError
        If the Monte Carlo budget is invalid, draws are nonfinite, or the
        sample average exceeds numerical capacity.
    TypeError
        If no analytic expectation exists and finite-mean approximation
        was not requested or cannot establish a finite mean.
    """
    if mc_samples is not None and (
        isinstance(mc_samples, bool)
        or not isinstance(mc_samples, int)
        or mc_samples <= 0
    ):
        raise ValueError("mc_samples must be a positive integer")
    try:
        result = distribution.mean
    except NotImplementedError as error:
        support = distribution.support
        bounded = support is constraints.simplex
        lower = getattr(support, "lower_bound", None)
        upper = getattr(support, "upper_bound", None)
        if not bounded and lower is not None and upper is not None:
            bounded = bool(
                torch.isfinite(torch.as_tensor(lower)).all()
                and torch.isfinite(torch.as_tensor(upper)).all()
            )
        finite = bounded or 1 in getattr(distribution, "finite_moment_orders", ())
        if mc_samples is None or not finite:
            raise TypeError(
                "no analytic response expectation; bounded or certified finite-mean response laws permit an explicit mc_samples budget"
            ) from error
    else:
        if not torch.isfinite(result).all():
            raise ValueError(
                "response expectation is undefined or nonfinite at this posterior draw"
            )
        return result
    total = None
    scale = None
    for start in range(0, mc_samples, 64):
        size = min(64, mc_samples - start)
        values = distribution.sample((size,))
        if not torch.isfinite(values).all():
            raise ValueError(
                "Monte Carlo response expectation produced nonfinite draws"
            )
        values = values.to(torch.float64)
        part_scale = values.abs().amax(0)
        next_scale = part_scale if scale is None else torch.maximum(scale, part_scale)
        divisor = torch.where(next_scale > 0, next_scale, torch.ones_like(next_scale))
        part = (values / divisor).sum(0)
        if total is None:
            total = part
        else:
            assert scale is not None
            total = total * (scale / divisor) + part
        scale = next_scale
    assert total is not None and scale is not None
    result = (total / mc_samples) * scale
    if not torch.isfinite(result).all():
        raise ValueError("Monte Carlo response expectation exceeded numerical capacity")
    return result


__all__: tuple[str, ...] = ()
