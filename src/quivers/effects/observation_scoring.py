"""Select likelihood scoring without changing priors, observations, or sampling."""

from __future__ import annotations

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.effects.base import EffectHandler, Installation, RunContext


_POLICY = "__quivers_observation_scoring__"


def _enabled(run: RunContext) -> bool:
    """Read the innermost scoring policy installed for this execution.

    Parameters
    ----------
    run : RunContext
        Isolated execution annotations.

    Returns
    -------
    bool
        Whether observation likelihoods contribute.
    """
    return bool(run.annotations.get(_POLICY, {}).get("enabled", True))


class _UnscoredObservation(ContinuousMorphism):
    """Preserve response generation while omitting likelihood evaluation.

    Parameters
    ----------
    inner : ContinuousMorphism
        Original response law including recording and weight wrappers.
    """

    def __init__(self, inner: ContinuousMorphism) -> None:
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the additive identity without evaluating a response density.

        Parameters
        ----------
        x, y : torch.Tensor
            Native inputs and recorded response, used only for placement.

        Returns
        -------
        torch.Tensor
            A scalar zero score on the execution device.
        """
        dtype = y.dtype if y.is_floating_point() else torch.get_default_dtype()
        return torch.zeros((), dtype=dtype, device=x.device)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw the unchanged underlying response law.

        Parameters
        ----------
        x : torch.Tensor
            Native law inputs.
        sample_shape : torch.Size
            Additional independent sample axes.

        Returns
        -------
        torch.Tensor
            Samples from the original response law.
        """
        return self._inner.rsample(x, sample_shape)

    def sampling_distribution(self, x, event=None):
        """Delegate sampling-law inspection to the unchanged response wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Native law inputs.
        event : dict or None
            Resolved recording/truncation metadata.

        Returns
        -------
        torch.distributions.Distribution
            The original response-generating distribution.
        """
        return self._inner.sampling_distribution(x, event)


class ObservationScoringHandler(EffectHandler):
    """Choose whether declared observations and likelihood factors contribute.

    Latent draws and explicit score potentials retain their ordinary scores.
    Recorded values and raw likelihoods are unchanged; disabling likelihood
    scoring avoids evaluating that likelihood at all.

    Parameters
    ----------
    enabled : bool
        Include likelihood scores when true; omit them when false.
    """

    def __init__(self, enabled: bool = True) -> None:
        if type(enabled) is not bool:
            raise TypeError("observation scoring requires a Boolean policy")
        self.enabled = enabled

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Set this execution's policy, with innermost scopes taking precedence.

        Parameters
        ----------
        run : RunContext
            Isolated per-run state, shared by its native host steps.

        Returns
        -------
        tuple[Installation, ...]
            No effect is replaced; existing host boundaries read the policy.
        """
        run.annotations.setdefault(_POLICY, {})["enabled"] = self.enabled
        return ()


def observation_scoring(enabled: bool = True) -> ObservationScoringHandler:
    """Create a scoped likelihood policy shared by fitting interfaces.

    Parameters
    ----------
    enabled : bool
        Whether observations and classified likelihood factors are scored.

    Returns
    -------
    ObservationScoringHandler
        A reusable context manager; latent priors and score potentials remain.
    """
    return ObservationScoringHandler(enabled)


__all__ = ["ObservationScoringHandler", "observation_scoring"]
