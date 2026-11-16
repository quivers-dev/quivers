"""Stochastic Variational Inference (SVI) training loop.

SVI optimizes a [`quivers.inference.objectives.Objective`][quivers.inference.objectives.Objective]
(the ELBO, IWAE, Rényi, or VR-IWAE bound) by taking gradient
steps on the guide and model parameters. The ``objective``
parameter accepts any `Objective` subclass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides import Guide
from quivers.inference.objectives import Objective


def _unique_parameters(*sources: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Retain shared model and guide parameters exactly once, in stable order."""
    return list(
        {
            id(parameter): parameter
            for source in sources
            for parameter in source
            if parameter.requires_grad
        }.values()
    )


def _prepare_optimizer(
    *sources: Iterable[torch.Tensor],
    learning_rate: float,
    factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer] | None = None,
) -> torch.optim.Optimizer | None:
    """Prepare one optimizer over unique trainable parameters, if any exist."""
    parameters = _unique_parameters(*sources)
    if not parameters:
        return None
    return (
        factory(parameters)
        if factory is not None
        else torch.optim.Adam(parameters, lr=learning_rate)
    )


class SVI:
    """Stochastic Variational Inference optimizer.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    guide : Guide
        Variational guide.
    optim : torch.optim.Optimizer or None
        Optimizer for model and guide parameters. None is valid only when
        neither has trainable parameters; steps then evaluate the objective
        without constructing phantom parameters or taking a gradient step.
    objective : Objective
        Variational objective (ELBO, IWAEBound, RenyiBound,
        VRIWAEBound, …).
    """

    def __init__(
        self,
        model: MonadicProgram,
        guide: Guide,
        optim: torch.optim.Optimizer | None,
        objective: Objective,
    ) -> None:
        if optim is None and _unique_parameters(model.parameters(), guide.parameters()):
            raise ValueError(
                "an optimizer is required for trainable model or guide parameters"
            )
        self.model = model
        self.guide = guide
        self.optim = optim
        self.objective = objective

    def step(
        self,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> float:
        """One SVI step.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        observations : dict[str, torch.Tensor]
            Observed variable values + host data (the
            non-site keys are exposed to the trace via the
            ``condition`` machinery).

        Returns
        -------
        float
            Scalar loss for this step.
        """
        if self.optim is not None:
            self.optim.zero_grad()
        loss_val = self.objective(self.model, self.guide, x, observations)
        if not bool(torch.isfinite(loss_val).all()):
            raise ValueError("nonfinite variational objective")
        if self.optim is not None:
            loss_val.backward()
            self.optim.step()
        return loss_val.item()

    def run(
        self,
        x: torch.Tensor,
        observations: Mapping[str, torch.Tensor]
        | Callable[[int], Mapping[str, torch.Tensor]],
        *,
        num_steps: int,
        callback: Callable[[int, float], None] | None = None,
    ) -> tuple[float, ...]:
        """Optimize one posterior and retain its finite loss history.

        Parameters
        ----------
        x : torch.Tensor
            Model execution input.
        observations : Mapping[str, torch.Tensor] or Callable
            Fixed bindings, or a step-indexed supplier that recomputes
            differentiable predictors for the current parameter values.
        num_steps : int
            Nonnegative optimization budget. Zero retains initialization
            without evaluating the objective or updating parameters.
        callback : Callable[[int, float], None] or None
            Receives each zero-based step index and its scalar loss.

        Returns
        -------
        tuple[float, ...]
            Losses in optimization order.

        Raises
        ------
        ValueError
            If the budget is invalid or an objective is nonfinite.
        """
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a nonnegative integer")
        losses: list[float] = []
        for index in range(num_steps):
            bindings = observations(index) if callable(observations) else observations
            try:
                loss = self.step(x, dict(bindings))
            except ValueError as exc:
                raise ValueError(
                    f"variational optimization at step {index}: {exc}"
                ) from exc
            losses.append(loss)
            if callback is not None:
                callback(index, loss)
        return tuple(losses)


__all__ = ["SVI"]
