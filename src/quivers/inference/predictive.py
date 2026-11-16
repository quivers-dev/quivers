"""Posterior predictive sampling.

Given a trained posterior representation (either a variational
guide or an MCMC chain), `Predictive` repeatedly samples
latents from the posterior and traces the model forward to produce
posterior predictive draws of every site.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference._posterior import sample_count
from quivers.inference.guides import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.prior import PriorDraws
from quivers.inference.trace import Trace, trace


def _require_replayed_sites(
    recorded: Trace, latents: Mapping[str, torch.Tensor]
) -> None:
    """Require retained posterior labels to be reached by the current execution.

    Parameters
    ----------
    recorded : Trace
        Replayed program execution.
    latents : Mapping[str, torch.Tensor]
        One retained joint posterior draw.

    Raises
    ------
    ValueError
        If a retained label is absent from the program's sample sites.
    """
    if stray := sorted(set(latents) - set(recorded.sites)):
        raise ValueError(
            f"Predictive: the posterior holds draws of {stray}, which "
            "are not sample sites of the model"
        )


def posterior_expectations(
    model: MonadicProgram,
    posterior: Guide | MCMCResult | PriorDraws,
    x: torch.Tensor,
    observations: Mapping[str, torch.Tensor] | None = None,
    *,
    sites: Sequence[str],
    num_samples: int | None = None,
    mc_samples: int | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Collect native conditional response expectations on retained draw axes.

    Parameters
    ----------
    model : MonadicProgram
        Existing compiled program; no source preparation is performed.
    posterior : Guide or MCMCResult or PriorDraws
        Fitted guide or retained constrained joint draws.
    x : torch.Tensor
        One-run execution input. Multiple independent runs require separate
        requests so their posterior interpretation remains explicit.
    observations : Mapping[str, torch.Tensor] or None
        Covariates and optional response values conditioning the replay.
        Observed densities are not scored. Upstream observed or lexical
        values retain their ordinary conditioning semantics.
    sites : Sequence[str]
        Distinct response names, including optional exact Scan occurrences.
    num_samples : int or None
        Maximum leading draws per recorded chain, or fresh guide draws.
        None retains recorded draws or takes 100 variational draws.
    mc_samples : int or None
        Explicit finite-moment approximation budget when a native analytic
        expectation is unavailable, following the shared response-law policy.
    seed : int
        Isolated posterior and response streams; the caller RNG is restored.

    Returns
    -------
    dict[str, torch.Tensor]
        Conditional native means with chain/draw/native batch/event axes.
        Categorical and ordinal laws return category probabilities. Scan
        occurrences stack in visitation order, as for raw log likelihoods.

    Raises
    ------
    ValueError, TypeError
        If execution/draw axes, selected sites, posterior layout, or response
        moments are invalid. No response-replication average is substituted
        for a native analytic expectation.
    """
    from quivers.continuous._ordinal import OrdinalDistribution
    from quivers.effects.observation_scoring import observation_scoring
    from quivers.inference._likelihood import observation_sites
    from quivers.inference._posterior import (
        constrained_draws,
        draw_count,
        collect_site_draws,
    )
    from quivers.inference._response import (
        response_mean,
        response_distribution,
        response_rng,
    )
    from quivers.inference.registry import _require_one_run
    from quivers.qiec import IndexLiteral, tensor_shape

    _require_one_run(x, "posterior_expectations")
    names = tuple(sites)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("expectations require distinct nonempty response-site names")
    if mc_samples is not None:
        sample_count(mc_samples)
    shape = draw_count(posterior, num_samples)
    values = dict(observations or {})
    draws = constrained_draws(posterior, x, num_samples=num_samples, seed=seed)
    declared_shapes = {}
    if model.input_contract is not None:
        for parameter in model.input_contract.parameters:
            if parameter.name not in names:
                continue
            split = tensor_shape(parameter.type)
            dimensions = () if split is None else split[1]
            if all(
                isinstance(dimension, IndexLiteral) and isinstance(dimension.value, int)
                for dimension in dimensions
            ):
                declared_shapes[parameter.name] = tuple(
                    dimension.value for dimension in dimensions
                )
    category_axes: dict[str, int] = {}

    def expectations() -> Iterator[dict[str, torch.Tensor]]:
        """Read each selected conditional law once per joint posterior draw.

        Yields
        ------
        dict[str, torch.Tensor]
            Response expectations on their unchanged native batch/event axes.

        Raises
        ------
        TypeError
            If a selected response is unavailable or exposes no native law.
        """
        with (
            torch.no_grad(),
            observation_scoring(False),
            response_rng(seed + 1, (x, *values.values(), *draws.values())),
        ):
            for chain in range(shape[0]):
                for draw in range(shape[1]):
                    latents = {
                        name: value[chain, draw] for name, value in draws.items()
                    }
                    recorded = trace(model, x, values | latents)
                    _require_replayed_sites(recorded, latents)
                    grouped = observation_sites(recorded, set(names))
                    if missing := set(names) - set(grouped):
                        raise TypeError(
                            f"response sites {sorted(missing)} expose no native conditional distribution"
                        )
                    result = {}
                    for name, occurrences in grouped.items():
                        moments = []
                        for site in occurrences:
                            distribution = response_distribution(recorded, site.name)
                            if isinstance(distribution, OrdinalDistribution):
                                mean = distribution.category_probs()
                                categories = mean.shape[-1]
                            elif isinstance(
                                distribution, torch.distributions.Categorical
                            ):
                                mean = distribution.probs
                                categories = mean.shape[-1]
                            else:
                                mean = response_mean(distribution, mc_samples)
                                categories = 0
                            if category_axes.setdefault(name, categories) != categories:
                                raise TypeError(
                                    f"response {name!r} changes its native category axis between draws"
                                )
                            moments.append(mean)
                        result[name] = (
                            moments[0] if len(moments) == 1 else torch.stack(moments)
                        )
                    yield result

    output = collect_site_draws(expectations(), shape)
    for name, dimensions in declared_shapes.items():
        if name in output:
            categories = (category_axes[name],) if category_axes[name] else ()
            # Presentation follows the checked input contract; an extra
            # probability axis comes only from the native categorical law.
            output[name] = output[name].reshape(*shape, *dimensions, *categories)
    return output


class Predictive:
    """Posterior predictive sampler.

    Accepts either a trained `Guide` (variational
    posterior), an `MCMCResult` (Monte Carlo posterior), or independent
    `PriorDraws` from the retained prior law.
    Variational case: draws ``num_samples`` fresh guide samples.
    MCMC case: iterates over the recorded posterior draws (one
    forward trace per draw, up to ``num_samples`` if specified).

    An input of ``B`` rows is ``B`` independent runs of the model.
    A guide's `Guide.rsample` gives each run its own draw of a site
    that is not plated, ``(batch, *constrained_shape)``, and one draw
    of a plated site that every run shares. An MCMC draw, at
    `LatentSite.constrained_shape`, is one value every run shares.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    posterior : Guide or MCMCResult or PriorDraws
        Trained posterior representation.
    num_samples : int, optional
        Number of predictive draws. Defaults to ``100`` for guides
        and ``num_chains * num_samples`` for MCMC results (use all
        recorded posterior draws). If supplied as an explicit
        integer for an MCMC result it is capped at the available
        draw count.

    Raises
    ------
    TypeError
        If ``posterior`` is not a supported guide or recorded draw container.
    ValueError
        If ``num_samples`` is not a positive integer, or the posterior
        has no retained draws.
    """

    def __init__(
        self,
        model: MonadicProgram,
        posterior: Guide | MCMCResult | PriorDraws,
        num_samples: int | None = None,
    ) -> None:
        if not isinstance(posterior, (Guide, MCMCResult, PriorDraws)):
            raise TypeError(
                f"Predictive: draws must be Guide or MCMCResult or PriorDraws; "
                f"got {type(posterior).__name__}"
            )
        self.model = model
        self.posterior = posterior
        self.num_samples = sample_count(
            num_samples,
            available=(
                posterior.num_chains * posterior.num_samples
                if isinstance(posterior, (MCMCResult, PriorDraws))
                else None
            ),
        )

    def _iter_recorded_latents(self) -> Iterator[dict[str, torch.Tensor]]:
        """Return ``num_samples`` recorded posterior draws, one dict each,
        flattening the chain and sample axes.

        Yields
        ------
        dict[str, torch.Tensor]
            One dict per selected draw, each site at its
            `LatentSite.constrained_shape`: one value every run of the
            predictive call shares.
        """
        assert isinstance(self.posterior, (MCMCResult, PriorDraws))
        samples = self.posterior.samples
        chain_samples = self.posterior.num_chains * self.posterior.num_samples
        # Random selection without replacement (when num_samples
        # equals the total, this is a permutation).
        perm = torch.randperm(chain_samples)[: self.num_samples]
        for idx in perm.tolist():
            chain = idx // self.posterior.num_samples
            step = idx % self.posterior.num_samples
            yield {name: draws[chain, step] for name, draws in samples.items()}

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior predictive samples.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        observations : dict[str, torch.Tensor] or None
            Additional observed data to condition on.

        Returns
        -------
        dict[str, torch.Tensor]
            One key per site, holding the stacked trace values:
            ``(num_samples, batch, *shape)`` for a site that is not
            plated, and for a plated site ``(num_samples, *shape)`` for
            one run and ``(num_samples, batch, *shape)`` for ``batch >
            1`` runs. Guide and MCMC posteriors produce the same shapes.

        Raises
        ------
        ValueError
            If an `MCMCResult` holds draws of a name that is not a
            sample site of the model.
        """
        observations = observations if observations is not None else {}
        if isinstance(self.posterior, Guide):
            latents_iter = (self.posterior.rsample(x) for _ in range(self.num_samples))
        else:
            latents_iter = self._iter_recorded_latents()

        from quivers.inference._posterior import collect_site_draws

        def values() -> Iterator[dict[str, torch.Tensor]]:
            """Stream site values from the existing predictive traces.

            Yields
            ------
            dict[str, torch.Tensor]
                One execution's site tensors in recorded trace order.

            Raises
            ------
            ValueError
                If a retained posterior name is absent from the execution.
            """
            for latents in latents_iter:
                all_obs = {**latents, **observations}
                recorded = trace(self.model, x, observations=all_obs)
                _require_replayed_sites(recorded, latents)
                yield {
                    name: site.value
                    for name, site in recorded.sites.items()
                    if not site.is_deterministic
                }

        return {
            name: value.squeeze(0)
            for name, value in collect_site_draws(
                values(), (1, self.num_samples)
            ).items()
        }


__all__ = ["Predictive", "posterior_expectations"]
