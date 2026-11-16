"""Recorded-event scores over the existing observation and plate interfaces."""

from __future__ import annotations

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.plate import VectorisedObserve


def resolve_event(event, environment):
    """Resolve recording roles from a lexical runtime environment.

    Parameters
    ----------
    event : ObservationEvent
        Typed recording and truncation references.
    environment : Mapping[str, object]
        Lexical values after removing any leading run axis.

    Returns
    -------
    dict[str, object]
        Bound arguments accepted by the common event-scoring operation.
    """
    fields = {
        "status": event.status,
        "interval_upper": event.upper,
        "trunc_lower": event.lower_bound,
        "trunc_upper": event.upper_bound,
    }
    return {
        name: environment[value] if isinstance(value, str) else value
        for name, value in fields.items()
    }


class RecordedObservation(ContinuousMorphism):
    """Bind recording roles while preserving the response sampling law.

    Parameters
    ----------
    inner : ContinuousMorphism
        Response morphism under the current parameter handlers.
    declared : ContinuousMorphism
        Original response morphism, identifying its plate layout.
    event : dict[str, object]
        Bound recording and truncation data.
    """

    def __init__(self, inner, declared, event):
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner
        self._plated = isinstance(declared, VectorisedObserve)
        self._event = event

    def log_prob(self, x, y):
        """Score the recorded events with the inner morphism's reductions.

        Parameters
        ----------
        x : Tensor
            Resolved response-family parameters, shared or one row per observation.
        y : Tensor or None
            Observed values or censoring endpoints; None uses the registered response.

        Returns
        -------
        Tensor
            Recorded-event log likelihoods under the method's row and event reductions.
        """
        return self._inner.log_prob_event(x, y, self._event)

    def log_prob_pointwise(self, x, y):
        """Read unweighted recorded-event scores for each observation row.

        Parameters
        ----------
        x : Tensor
            Resolved response-family parameters, shared or one row per observation.
        y : Tensor or None
            Observed values or censoring endpoints; None uses the registered response.

        Returns
        -------
        Tensor
            Recorded-event log likelihoods under the method's row and event reductions.
        """
        if self._plated:
            return self._inner.log_prob_pointwise_event(x, y, self._event)
        return self._inner.log_prob_event(x, y, self._event)

    def log_prob_pointwise_rows(self, x, y, *rows):
        """Read selected event scores using the existing plate index rules.

        Parameters
        ----------
        x : Tensor
            Resolved response-family parameters, shared or one row per observation.
        y : Tensor or None
            Observed values or censoring endpoints; None uses the registered response.
        *rows : tuple[Tensor, ...]
            Selected plate indices, outermost observation axis first.

        Returns
        -------
        Tensor
            Recorded-event log likelihoods under the method's row and event reductions.
        """
        return self._inner.log_prob_pointwise_rows_event(x, y, self._event, *rows)

    def log_prob_rows(self, x, y, *rows):
        """Sum selected event scores before subsampling rescaling.

        Parameters
        ----------
        x : Tensor
            Resolved response-family parameters, shared or one row per observation.
        y : Tensor or None
            Observed values or censoring endpoints; None uses the registered response.
        *rows : tuple[Tensor, ...]
            Selected plate indices, outermost observation axis first.

        Returns
        -------
        Tensor
            Recorded-event log likelihoods under the method's row and event reductions.
        """
        return self._inner.log_prob_rows_event(x, y, self._event, *rows)

    def rsample(self, x, sample_shape=torch.Size()):
        """Draw latent responses, truncated when specified, before recording.

        Parameters
        ----------
        x : Tensor
            Resolved response-family parameters, shared or one row per observation.
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        Tensor
            Response draws with retained sample, observation, array, and event axes.
        """
        return self._inner.rsample_event(x, sample_shape, self._event)

    def sampling_distribution(self, x, event=None):
        """Expose the actual response law for prediction and expectations.

        Parameters
        ----------
        x : Tensor
            Resolved response-family parameters, shared or one row per observation.
        event : dict[str, object] or None
            Bound recording roles and sampling bounds; None requests ordinary response scoring.

        Returns
        -------
        Distribution
            Underlying response sampling law, normalized on any truncation interval.
        """
        return self._inner.sampling_distribution(x, self._event)

    def score_dependencies(self):
        """Retain parameter-handler dependencies for zero-weight scores.

        Returns
        -------
        tuple[Tensor, ...]
            Live parameter overrides that must remain connected to zero-weight scores.
        """
        method = getattr(self._inner, "score_dependencies", None)
        return method() if callable(method) else ()
