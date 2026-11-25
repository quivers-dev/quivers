"""Replay retained formula bindings at normalized posterior draws."""

from __future__ import annotations

from contextlib import nullcontext
import math

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch

from quivers._values import ValueModel
from quivers._rng import seeded_rng
from quivers.inference._posterior import constrained_draws
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace, trace
from quivers.inference._response import (
    response_distribution as response_distribution,
    response_rng as response_rng,
    category_probabilities as category_probabilities,
    response_mean as response_mean,
)

from quivers.formulas._prediction import OrdinalReplay

# Constants, centered intercepts and QR coefficients are derived sites.
from quivers.formulas._brms_priors import derived_site_draws
from quivers.formulas._mixture import equated_sites
from quivers.formulas._population_coordinates import reported_coefficients

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
    draws = constrained_draws(
        fit.posterior,
        torch.zeros(1, 1, dtype=torch.long),
        num_samples=num_samples,
        seed=seed,
        registry=registry,
    )
    plan = fit.prediction_plan
    if plan is None:
        return draws
    return derived_site_draws(
        plan.module,
        draws,
        reported_coefficients(fit._predictor_items())
        + [site for target in plan.prior_targets for site in target.sites]
        # mixtures: parameters equated to a sampled one keep their own name.
        + equated_sites(fit.family, draws),
    )


def replay(
    fit: BayesianFit,
    draws: dict[str, torch.Tensor],
    shape: tuple[int, int],
    *,
    context: PredictionContext | None = None,
    score_observations: bool = False,
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
    score_observations : bool
        Evaluate observed densities for raw likelihood analysis. Prediction
        defaults omit scoring placeholder responses while retaining priors.

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
    from quivers.effects.observation_scoring import observation_scoring

    with torch.no_grad(), observation_scoring(score_observations):
        observations = dict(
            fit.observations if context is None else context.observations
        )
        program = fit.program if context is None else context.program
        input = torch.zeros(1, 1, dtype=torch.long)
        generator = torch.Generator().manual_seed(
            (context.seed + 2) % 2**64 if context is not None else 0
        )
        # Fresh latent sites of prediction rows (the latent function of an
        # exact gp() at new inputs) are sampled by the program. Their stream
        # depends on the prediction seed and the draw alone, so every
        # prediction method sees the same latent draws for one seed.
        fresh = (
            torch.Generator().manual_seed((context.seed + 3) % 2**64)
            if context is not None and context.fresh_sites
            else None
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
                    name: values[chain, draw]
                    for name, values in draws.items()
                    if context is None or name not in context.fresh_sites
                }
                retained: dict[str, torch.Tensor] = {}
                fitted: Trace | None = None
                # Retained threshold replays: one per ordinal response, named by
                # its site prefix (empty for a single response, namespaced in
                # a joint program).
                replays = (
                    ()
                    if context is None or not context.retain_ordinal_cutpoints
                    else context.ordinal_replays
                    or (
                        OrdinalReplay(
                            prefix="",
                            thresholds_by=fit.thresholds_by,
                            fitted_levels=len(
                                fit.formula.group_levels[fit.thresholds_by]
                            )
                            if fit.thresholds_by is not None
                            else 0,
                            rows=context.rows,
                        ),
                    )
                )
                if replays:
                    fitted = trace(
                        fit.program, input, dict(fit.observations) | original_latents
                    )
                    for entry in replays:
                        site = entry.prefix + "cutpoints"
                        if site in fitted.sites:
                            retained[site] = fitted.sites[site].value.reshape(-1)
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
                    for entry in replays:
                        if (
                            context is not None
                            and fitted is not None
                            and entry.prefix + "cutpoint_matrix" in fitted.sites
                        ):
                            retained.update(
                                _retained_group_cutpoints(
                                    entry,
                                    context,
                                    fitted,
                                    latents,
                                    observations,
                                    donors,
                                    input,
                                )
                            )
                    try:
                        with (
                            nullcontext()
                            if fresh is None
                            else seeded_rng(
                                int(torch.randint(2**62, (), generator=fresh)),
                                devices=(input.device,),
                            )
                        ):
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
    entry: OrdinalReplay,
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
    entry : OrdinalReplay
        Site prefix, threshold group, fitted level count, and rows of one
        grouped cumulative response.
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
    assert entry.thresholds_by is not None
    group = entry.thresholds_by
    prefix = entry.prefix
    rows = entry.rows
    matrix = fitted.sites[prefix + "cutpoint_matrix"].value.reshape(
        entry.fitted_levels, -1
    )
    if group not in context.groups:
        return {prefix + "cutpoint_matrix": matrix}
    mapping = context.groups[group]
    indices = torch.tensor(mapping.fitted_indices, dtype=torch.long)
    cuts = matrix[indices.clamp_min(0)].clone()
    gaps = max(0, matrix.shape[-1] - 1)
    overrides: dict[str, torch.Tensor] = {}
    if gaps:
        overrides[prefix + "ordinal_spacing_bound"] = matrix.new_zeros(
            len(indices), gaps
        )
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
                    prefix + "ordinal_contrasts": safe.expand(rows, transitions),
                    prefix + "ordinal_effective_spacing": matrix.new_ones(rows, gaps),
                },
            )
            proposed = proposal.sites[prefix + "cutpoint_matrix"].value.reshape(
                len(indices), -1
            )
            cuts[new] = proposed[new]
        else:
            cuts[new] = 0
    overrides[prefix + "cutpoint_matrix"] = cuts
    return overrides


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


class LinpredBinding(ValueModel):
    """The row binding holding a linear predictor, and how it is read.

    Parameters
    ----------
    site : str
        Binding of the program.
    width : int
        Number of coordinates per row; zero for a scalar.
    exponentiate : bool
        Whether the requested value is the exponential of the binding.
    specific : str or None
        Binding of category-specific effects added to every row's value.
    coordinate : int or None
        Coordinate to select.
    """

    site: str
    width: int = 0
    exponentiate: bool = False
    specific: str | None = None
    coordinate: int | None = None

    def read(self, binding: Callable[[str], torch.Tensor], rows: int) -> torch.Tensor:
        """Read the values over the rows.

        Parameters
        ----------
        binding : Callable[[str], torch.Tensor]
            Value of a binding in one replayed program.
        rows : int
            Number of rows.

        Returns
        -------
        torch.Tensor
            One value per row, or per row and coordinate.
        """
        value = binding(self.site)
        if self.exponentiate:
            value = value.exp()
        if self.specific is not None:
            value = row_binding(value, rows).unsqueeze(-1) + row_binding(
                binding(self.specific), rows, self.width
            )
        value = row_binding(value, rows, self.width)
        return value if self.coordinate is None else value[..., self.coordinate]
