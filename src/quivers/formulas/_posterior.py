"""Replay retained formula bindings at normalized posterior draws."""

from __future__ import annotations

import math

from collections.abc import Iterator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.distributions import Distribution

from quivers.inference._posterior import constrained_draws
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace, trace

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._prediction import PredictionContext


_MAX_FEASIBLE_GROUP_ATTEMPTS = 1024


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
    registry = LatentRegistry.from_model(
        fit.program, observed_names=set(fit.observations)
    )
    return constrained_draws(
        fit.posterior,
        torch.zeros(1, 1, dtype=torch.long),
        num_samples=num_samples,
        seed=seed,
        registry=registry,
    )


def replay(
    fit: BayesianFit,
    draws: dict[str, torch.Tensor],
    shape: tuple[int, int],
    *,
    context: PredictionContext | None = None,
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
    context : PredictionContext or None
        Prepared fresh-row program and bindings, or None for training rows.

    Yields
    ------
    Trace
        One native execution trace per chain and draw.

    Raises
    ------
    ValueError
        If fresh rows cross retained thresholds, or a conditional new-group
        policy exhausts its proposal budget. Posterior draws remain unchanged.
    """
    with torch.no_grad():
        observations = dict(
            fit.observations if context is None else context.observations
        )
        program = fit.program if context is None else context.program
        input = torch.zeros(1, 1, dtype=torch.long)
        generator = torch.Generator().manual_seed(
            (context.seed + 2) % 2**64 if context is not None else 0
        )
        fixed_donors: dict[str, torch.Tensor] = {}
        specialized_shapes = {}
        if context is not None and context.groups:
            registry = LatentRegistry.from_model(
                program, observed_names=set(observations)
            )
            specialized_shapes = {
                name: registry[name].constrained_shape for name in registry.names
            }
        for chain in range(shape[0]):
            for draw in range(shape[1]):
                original_latents = {
                    name: values[chain, draw] for name, values in draws.items()
                }
                retained: dict[str, torch.Tensor] = {}
                fitted: Trace | None = None
                if context is not None and context.retain_ordinal_cutpoints:
                    fitted = trace(
                        fit.program, input, dict(fit.observations) | original_latents
                    )
                    if "cutpoints" in fitted.sites:
                        retained["cutpoints"] = fitted.sites["cutpoints"].value.reshape(
                            -1
                        )
                feasible = (
                    context is not None
                    and context.new_level_policy.startswith("feasible_")
                    and any(
                        name in context.ordinal_order_groups
                        and -1 in group.fitted_indices
                        for name, group in context.groups.items()
                    )
                )
                attempts = _MAX_FEASIBLE_GROUP_ATTEMPTS if feasible else 1
                recorded: Trace | None = None
                for attempt in range(attempts):
                    latents = original_latents
                    donors: dict[str, torch.Tensor] = {}
                    if context is not None and context.groups:
                        from quivers.formulas._group_prediction import remap_draw

                        latents = remap_draw(
                            context,
                            latents,
                            specialized_shapes,
                            generator,
                            fixed_donors,
                            donors,
                        )
                    if (
                        context is not None
                        and context.retain_ordinal_cutpoints
                        and fitted is not None
                        and "cutpoint_matrix" in fitted.sites
                    ):
                        retained.update(
                            _retained_group_cutpoints(
                                fit,
                                context,
                                fitted,
                                latents,
                                observations,
                                donors,
                                input,
                            )
                        )
                    try:
                        recorded = trace(
                            program, input, observations | latents | retained
                        )
                    except ValueError as exc:
                        if (
                            context is None
                            or not context.retain_ordinal_cutpoints
                            or "ordinal_effective_spacing" not in str(exc)
                        ):
                            raise
                        if feasible and attempt + 1 < attempts:
                            continue
                        raise ValueError(
                            "ordinal prediction requires ordered effective cutpoints; "
                            f"chain {chain}, draw {draw}, requested rows/groups "
                            f"{tuple(context.groups)}: {exc}. "
                            + (
                                f"No feasible joint new-group proposal was obtained in {attempts} attempts; "
                                "the posterior draw was retained, and prediction failed."
                                if feasible
                                else "Fitted cutpoints were retained; use prediction covariates "
                                "inside their fitted order-support domain."
                            )
                        ) from exc
                    break
                assert recorded is not None
                yield recorded


def _retained_group_cutpoints(
    fit: BayesianFit,
    context: PredictionContext,
    fitted: Trace,
    latents: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
    donors: dict[str, torch.Tensor],
    input: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Retain known profiles and obtain genuine conditional new-group proposals.

    Parameters
    ----------
    fit : BayesianFit
        Original grouped cumulative fit.
    context : PredictionContext
        Prepared compact group metadata and prediction program.
    fitted : Trace
        Full original-data replay, before varying-effect selection.
    latents : dict of str to torch.Tensor
        Jointly remapped raw group parameters for one proposal.
    observations : dict of str to torch.Tensor
        Fresh predictor bindings.
    donors : dict of str to torch.Tensor
        Joint donor indices for the non-Gaussian policies.
    input : torch.Tensor
        One native execution input.

    Returns
    -------
    dict of str to torch.Tensor
        Actual retained cutpoint profiles and inert support-bound overrides.
        A Gaussian new group's profile is sampled from its original log-slack
        hierarchical law and subsequently conditioned on fresh-row order by
        the caller's explicit rejection policy.
    """
    assert fit.thresholds_by is not None
    group = fit.thresholds_by
    original_groups = len(fit.formula.group_levels[group])
    matrix = fitted.sites["cutpoint_matrix"].value.reshape(original_groups, -1)
    if group not in context.groups:
        return {"cutpoint_matrix": matrix}
    mapping = context.groups[group]
    indices = torch.tensor(mapping.fitted_indices, dtype=torch.long)
    cuts = matrix[indices.clamp_min(0)].clone()
    gaps = max(0, matrix.shape[-1] - 1)
    overrides: dict[str, torch.Tensor] = {}
    if gaps:
        overrides["ordinal_spacing_bound"] = matrix.new_zeros(len(indices), gaps)
    new = indices < 0
    if bool(new.any()):
        if group in donors:
            cuts[new] = matrix[donors[group][new]]
        elif gaps:
            # Obtain the existing emitter's hierarchical shape transform.
            # Safe contrast overrides suppress response evaluation only in
            # this proposal trace; the final trace validates the actual law.
            transitions = matrix.shape[-1]
            safe = torch.arange(transitions, dtype=matrix.dtype, device=matrix.device)
            proposal = trace(
                context.program,
                input,
                observations
                | latents
                | overrides
                | {
                    "ordinal_contrasts": safe.expand(context.rows, transitions),
                    "ordinal_effective_spacing": matrix.new_ones(context.rows, gaps),
                },
            )
            proposed = proposal.sites["cutpoint_matrix"].value.reshape(len(indices), -1)
            cuts[new] = proposed[new]
        else:
            cuts[new] = 0
    overrides["cutpoint_matrix"] = cuts
    return overrides


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
        bounded = support is torch.distributions.constraints.simplex
        if (
            not bounded
            and hasattr(support, "lower_bound")
            and hasattr(support, "upper_bound")
        ):
            bounded = bool(
                torch.isfinite(torch.as_tensor(support.lower_bound)).all()
                and torch.isfinite(torch.as_tensor(support.upper_bound)).all()
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


def row_binding(
    value: torch.Tensor, rows: int, categories: int | tuple[int, ...] = 0
) -> torch.Tensor:
    """Canonicalize a scalar or class-valued binding on retained response rows.

    Parameters
    ----------
    value : torch.Tensor
        A native binding, with or without a shared-row axis.
    rows : int
        Number of retained response rows.
    categories : int or tuple[int, ...]
        Declared event shape, category width, or zero for a scalar value.

    Returns
    -------
    torch.Tensor
        The binding on row and optional category axes.
    """
    event = (
        categories
        if isinstance(categories, tuple)
        else (categories,)
        if categories
        else ()
    )
    shape = (rows, *event)
    width = math.prod(event)
    if value.numel() == width:
        return value.reshape(1, *event).expand(shape)
    return value.reshape(shape)


def collect_draws(
    values: Iterator[torch.Tensor], shape: tuple[int, int]
) -> torch.Tensor:
    """Collect posterior row values into one preallocated output tensor.

    Parameters
    ----------
    values : Iterator[torch.Tensor]
        One response or predictor array per chain and draw.
    shape : tuple[int, int]
        The chain count and draw count, in iteration order.

    Returns
    -------
    torch.Tensor
        The collected arrays with leading chain and draw axes.

    Raises
    ------
    ValueError
        If the iterator's count or value shapes differ from the contract.
    """
    from quivers.inference._posterior import collect_draws as collect

    return collect(values, shape)
