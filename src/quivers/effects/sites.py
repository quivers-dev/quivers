"""Host sampleables: a torch morphism applied to an input, as a distribution.

A `MonadicProgram` step draws from a `ContinuousMorphism` at an input
tensor. The kernel's ``Random.sample`` request takes a sampleable value,
so the pair is wrapped as one: `TorchSampleable` offers the sampling
and scoring interface the prelude handlers use and keeps the morphism
and input a strategy or a trace may want to read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.morphisms import ContinuousMorphism


class TorchSampleable(ValueModel):
    """A morphism applied to an input, offered to the kernel as a sampleable.

    Parameters
    ----------
    morphism
        The morphism the site draws from, under whatever parameters the
        run's ``Param`` handlers answered.
    input
        The input tensor the morphism is conditioned on, carrying the
        batch axis.
    declared
        The morphism as the program declares it, which a trace records
        and a strategy reads the family of; the drawing morphism itself
        when no parameter store rebinds it.
    """

    morphism: ContinuousMorphism = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    input: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    declared: ContinuousMorphism = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def __init__(
        self,
        *,
        morphism: ContinuousMorphism,
        input: torch.Tensor,
        declared: ContinuousMorphism | None = None,
    ) -> None:
        """Construct a sampleable holding live runtime references by identity.

        Parameters
        ----------
        morphism : ContinuousMorphism
            The morphism used for sampling and scoring.
        input : torch.Tensor
            The tensor conditioning the morphism.
        declared : ContinuousMorphism or None
            The declared site morphism, defaulting to the sampling morphism.

        Raises
        ------
        didactic.ValidationError
            If the runtime fields have incompatible types.
        """
        initialize = cast("Callable[..., None]", super().__init__)
        initialize(
            morphism=morphism,
            input=input,
            declared=morphism if declared is None else declared,
        )

    @dx.model_validator()
    def _check_runtime_fields(self) -> None:
        """Validate live morphisms and the conditioning tensor.

        Raises
        ------
        TypeError
            If either morphism or the input has an incompatible type.
        """
        if not isinstance(self.morphism, ContinuousMorphism) or not isinstance(
            self.declared, ContinuousMorphism
        ):
            raise TypeError("sampleable morphisms must be ContinuousMorphism values")
        if not isinstance(self.input, torch.Tensor):
            raise TypeError("sampleable input must be a torch.Tensor")

    def rsample(self) -> torch.Tensor:
        """Draw a reparameterised sample.

        Returns
        -------
        torch.Tensor
            One draw per batch row.
        """
        return self.morphism.rsample(self.input)

    def sample(self) -> torch.Tensor:
        """Draw a sample.

        Returns
        -------
        torch.Tensor
            One draw per batch row.
        """
        return self.morphism.rsample(self.input)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score a value under the morphism at the input.

        Parameters
        ----------
        value : torch.Tensor
            The value, one per batch row.

        Returns
        -------
        torch.Tensor
            The log density per batch row.
        """
        return self.morphism.log_prob(self.input, value)

    def parameters(self) -> dict[str, torch.Tensor]:
        """The distribution parameters the morphism computes at the input.

        Returns
        -------
        dict[str, torch.Tensor]
            The named parameters of the torch distribution the morphism
            builds, or ``loc`` and ``scale`` for a morphism that exposes
            only a location-scale pair.

        Raises
        ------
        TypeError
            If the morphism exposes neither a distribution builder nor a
            location-scale pair.
        """
        get_dist = getattr(self.morphism, "_get_dist", None)
        if callable(get_dist):
            distribution = get_dist(self.input)
            return {
                name: getattr(distribution, name)
                for name in distribution.arg_constraints
                if hasattr(distribution, name)
            }
        get_params = getattr(self.morphism, "_get_params", None)
        if callable(get_params):
            loc, scale = get_params(self.input)
            return {"loc": loc, "scale": scale}
        raise TypeError(
            f"{type(self.morphism).__name__} exposes no distribution parameters"
        )


__all__ = ["TorchSampleable"]
