"""Independent ancestral prior draws and explicit factor-conditioned alternatives."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain
from typing import Literal, cast

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.effects.observation_scoring import observation_scoring
from quivers.inference.mcmc.driver import MCMC, MCMCResult


type PriorMode = Literal["no", "yes", "only"]
"""Fit observations, additionally retain independent priors, or fit priors only."""


def _ancestral_program(model: MonadicProgram) -> MonadicProgram:
    """Project conditional priors and refuse unnormalized global potentials.

    Parameters
    ----------
    model : MonadicProgram
        Original program, including conditional priors and explicit factors.

    Returns
    -------
    MonadicProgram
        Dependency projection retaining the original conditional draw laws.

    Raises
    ------
    ValueError
        If a global potential prevents independent ancestral sampling.
    """
    from quivers.inference._prior_projection import _has_potential, project_prior_sites

    projected = project_prior_sites(model)
    if _has_potential(projected):
        raise ValueError(
            "independent prior sampling cannot normalize explicit score potentials; "
            "use method=MCMC(...) for prior-only inference"
        )
    return projected


def _prior_mode(value: str) -> PriorMode:
    """Validate the public brms-compatible prior sampling modes.

    Parameters
    ----------
    value : str
        Requested observation/prior sampling policy.

    Returns
    -------
    PriorMode
        The validated mode.

    Raises
    ------
    ValueError
        If the mode is unknown.
    """
    if not isinstance(value, str) or value not in {"no", "yes", "only"}:
        raise ValueError("sample_prior must be 'no', 'yes', or 'only'")
    return cast(PriorMode, value)


class PriorDraws(ValueModel):
    """Independent ancestral samples, without MCMC convergence statistics.

    Parameters
    ----------
    samples : Mapping[str, torch.Tensor]
        Constrained latent samples with axes ``(1, draw, *event_shape)``.
        The leading singleton is an output-layout axis, not a Markov chain.
    num_samples : int
        Number of independently generated joint prior draws.
    """

    samples: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    num_samples: int

    @dx.model_validator()
    def _validate_samples(self) -> None:
        """Require declared independent-draw axes on every retained site.

        Raises
        ------
        ValueError
            If a budget or retained draw axis is inconsistent.
        """
        if isinstance(self.num_samples, bool) or self.num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        for name, value in self.samples.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 2
                or tuple(value.shape[:2]) != (1, self.num_samples)
            ):
                raise ValueError(
                    f"prior site {name!r} must have axes (1, {self.num_samples}, *event_shape)"
                )

    @property
    def num_chains(self) -> int:
        """Return the singleton layout axis, which has no chain diagnostics.

        Returns
        -------
        int
            One independent-draw layout axis, not a Markov chain.
        """
        return 1


def prior_draws(
    model: MonadicProgram,
    input: torch.Tensor,
    observations: Mapping[str, torch.Tensor] | None = None,
    *,
    num_samples: int = 100,
    seed: int = 0,
    method: Literal["ancestral"] | MCMC = "ancestral",
) -> PriorDraws | MCMCResult:
    """Sample the retained prior without evaluating observation likelihoods.

    Ancestral sampling draws every proper conditional prior once per joint
    draw. Arbitrary explicit score potentials require the explicitly selected
    MCMC alternative because their normalized law is not an ancestral kernel.
    Existing measure-valued draw boundaries retain their normalizing semantics.

    Parameters
    ----------
    model : MonadicProgram
        Compiled model with its original priors and support constraints.
    input : torch.Tensor
        One execution's input.
    observations : Mapping[str, torch.Tensor] or None
        Covariates and retained observations needed by data-dependent priors.
    num_samples : int
        Positive number of independent draws under the ancestral method.
    seed : int
        Isolated CPU random stream; the caller's state is restored.
    method : {"ancestral"} or MCMC
        Independent forward sampling, or a configured Markov-chain driver for
        priors containing explicit potentials. The latter retains its budgets.

    Returns
    -------
    PriorDraws or MCMCResult
        Samples with honest provenance and constrained event coordinates.

    Raises
    ------
    ValueError
        If the independent budget is invalid or an active potential precludes
        independent ancestral sampling.
    TypeError
        If an unsupported sampling method is requested.
    """
    from quivers.inference._posterior import collect_site_draws, sample_count
    from quivers.inference._prior_projection import project_prior_sites
    from quivers.inference._response import response_rng
    from quivers.inference.mcmc.driver import MCMC
    from quivers.inference.registry import _require_one_run, _trace_latent_values
    from quivers.inference.trace import trace

    if method != "ancestral" and not isinstance(method, MCMC):
        raise TypeError("prior method must be 'ancestral' or a configured MCMC driver")
    _require_one_run(input, "prior_draws")
    data = dict(observations or {})
    with observation_scoring(False):
        if isinstance(method, MCMC):
            return method.run(project_prior_sites(model), input, data)
        count = sample_count(num_samples)
        projected = _ancestral_program(model)
        with (
            response_rng(
                seed,
                chain(
                    (input,), data.values(), projected.parameters(), projected.buffers()
                ),
            ),
            torch.no_grad(),
        ):

            def values():
                """Yield one independent constrained joint draw.

                Yields
                ------
                dict[str, torch.Tensor]
                    Named latent values without the program run axis.
                """
                for _ in range(count):
                    recorded = trace(projected, input, data)
                    yield _trace_latent_values(recorded)

            samples = collect_site_draws(values(), (1, count))
    return PriorDraws(samples=samples, num_samples=count)


__all__ = ["PriorDraws", "PriorMode", "prior_draws"]
