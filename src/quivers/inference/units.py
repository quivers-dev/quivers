"""Log likelihoods of the independent units of a fit's responses."""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel


class UnitLogLikelihood(ValueModel):
    """The log likelihood of each independent unit of a set of responses.

    Given the latents of a draw, the units are independent, so the sum of
    their log likelihoods is that of all the responses together. Leaving
    out one unit is the prediction question these scores answer.

    Parameters
    ----------
    axes : tuple[str, ...] or None
        The objects that index the units, outermost first. ``None`` when
        the units are the entries of one observation.
    responses : tuple[str, ...]
        The responses whose rows the units hold.
    log_likelihood : torch.Tensor
        ``(chains, draws, *units)``, without likelihood powers. A score
        with one unit has one entry.
    row_units : dict[str, torch.Tensor]
        For each response, the unit of each of its rows, counted row-major
        over the unit axes.
    """

    axes: tuple[str, ...] | None
    responses: tuple[str, ...]
    log_likelihood: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    row_units: dict[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )


__all__ = ["UnitLogLikelihood"]
