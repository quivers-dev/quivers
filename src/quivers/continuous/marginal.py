"""Responses whose density a marginalized score carries.

A ``marginalize`` block scores its responses with the latent summed out,
so a response of the block adds no density of its own to the joint. The
response still has a law. `MarginalizedResponse` is that law at the values
of one run: it draws the response with the latent drawn too, and gives
the distribution of each row with the latent summed out, whose mean is
the class-weighted mean of the row.

`GroupLevel` and `draw_classes` state how the latents of nested grouped
blocks are drawn jointly, given what has already been bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import didactic.api as dx
import torch
from torch.distributions import Distribution, constraints

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.morphisms import (
    BoundEvent,
    ContinuousMorphism,
    MarginalizedFactor,
)
from quivers.continuous.program_steps import Environment, ResponseLaw


class MarginalizedResponse(ContinuousMorphism):
    """A marginalized block's response law, bound to one run's values.

    Parameters
    ----------
    declared : MarginalizedFactor
        The response's declared morphism, which scores nothing.
    law : ResponseLaw
        How the response is distributed given an environment.
    environment : Environment
        The values of one run the law reads.
    """

    mappable: bool = True

    def __init__(
        self, declared: MarginalizedFactor, law: ResponseLaw, environment: Environment
    ) -> None:
        super().__init__(declared.domain, declared.codomain)
        self._declared = declared
        self._law = law
        self._environment = environment

    @property
    def support(self) -> constraints.Constraint:
        """The support of the response family.

        Returns
        -------
        constraints.Constraint
            The declared family's support.
        """
        return self._declared.support

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw the latent and then the response.

        Parameters
        ----------
        x : torch.Tensor
            The step's input, which the law does not read.
        sample_shape : torch.Size
            Leading independent sampling dimensions.

        Returns
        -------
        torch.Tensor
            Draws of every row, ``(*sample_shape, rows, *event)``.
        """
        del x
        if not sample_shape:
            return self._law.draw(self._environment)
        draws = [
            self._law.draw(self._environment) for _ in range(math.prod(sample_shape))
        ]
        return torch.stack(draws).reshape(*sample_shape, *draws[0].shape)

    def rsample_event(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size,
        event: BoundEvent | None,
    ) -> torch.Tensor:
        """Draw the response on its declared truncation interval.

        Parameters
        ----------
        x : torch.Tensor
            The step's input, which the law does not read.
        sample_shape : torch.Size
            Leading independent sampling dimensions.
        event : BoundEvent or None
            The recording roles, which the law holds itself.

        Returns
        -------
        torch.Tensor
            Draws of every row.
        """
        del event
        return self.rsample(x, sample_shape)

    def sampling_distribution(
        self, x: torch.Tensor, event: BoundEvent | None = None
    ) -> Distribution:
        """The law of each row with the block's latents summed out.

        Parameters
        ----------
        x : torch.Tensor
            The step's input, which the law does not read.
        event : BoundEvent or None
            The recording roles, which the law holds itself.

        Returns
        -------
        Distribution
            One distribution per row, given the values the block does not
            sum out and no response.
        """
        del x, event
        return self._law.marginal(self._environment)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Score nothing: the block's score step holds the density.

        Parameters
        ----------
        x : torch.Tensor
            The step's input.
        y : torch.Tensor
            The response.

        Returns
        -------
        torch.Tensor
            Zero.
        """
        return self._declared.log_prob(x, y)

    def log_prob_event(
        self, x: torch.Tensor, y: torch.Tensor, event: BoundEvent
    ) -> torch.Tensor:
        """Score nothing: the block's score step holds the recorded events.

        Parameters
        ----------
        x : torch.Tensor
            The step's input.
        y : torch.Tensor
            The recorded response.
        event : BoundEvent
            The recording roles.

        Returns
        -------
        torch.Tensor
            Zero.
        """
        del event
        return self._declared.log_prob(x, y)


class GroupLevel(ValueModel):
    """One block of a nest of grouped blocks, at one run's values.

    Parameters
    ----------
    log_weights : torch.Tensor
        The log weight of each unit and class, ``(units, *enclosing, K)``
        with the class axes of the enclosing blocks outermost first: the
        log prior of the class, with the log likelihood of every response
        of the block that is already bound.
    parent : torch.Tensor or None
        The enclosing block's unit each unit belongs to, ``(units,)``;
        ``None`` for the outermost block.
    """

    log_weights: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    parent: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def draw_classes(levels: Sequence[GroupLevel]) -> tuple[torch.Tensor, ...]:
    """Draw the classes of a nest of grouped blocks jointly.

    The evidence of each inner unit is summed over its class and added to
    the class weights of the unit that encloses it, from the innermost
    block out. The classes are then drawn from the outermost block in, each
    given the classes of the units that enclose it. The draw is exact: the
    classes of the innermost units have the joint distribution that the
    weights of all the levels define.

    Parameters
    ----------
    levels : Sequence[GroupLevel]
        The blocks, outermost first.

    Returns
    -------
    tuple[torch.Tensor, ...]
        For each block, the class of the unit of it that encloses each unit
        of the innermost block, ``(units of the innermost block,)``.
    """
    evidence: list[torch.Tensor] = [level.log_weights for level in levels]
    for index in range(len(levels) - 1, 0, -1):
        parent = levels[index].parent
        assert parent is not None
        summed = torch.logsumexp(evidence[index], dim=-1)
        evidence[index - 1] = evidence[index - 1].index_add(0, parent, summed)
    drawn: list[torch.Tensor] = []
    for index, level in enumerate(levels):
        if index:
            assert level.parent is not None
            drawn = [classes[level.parent] for classes in drawn]
        units = torch.arange(evidence[index].shape[0], device=evidence[index].device)
        drawn.append(
            torch.distributions.Categorical(
                logits=evidence[index][(units, *drawn)]
            ).sample()
        )
    return tuple(drawn)


__all__ = ["GroupLevel", "MarginalizedResponse", "draw_classes"]
