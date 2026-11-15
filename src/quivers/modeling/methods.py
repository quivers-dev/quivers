"""Reusable sampling and variational fitting components for bound QVR models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.inference import (
    AutoNormalGuide,
    ELBO,
    Guide,
    MCMC,
    NUTSKernel,
    Objective,
    SVI,
)
from quivers.modeling.model import BoundModel, ModelFit


class MCMCMethod(ValueModel):
    """Fit with a reusable MCMC driver, defaulting to seeded NUTS.

    Parameters
    ----------
    sampler : MCMC
        Driver carrying kernel, chains, warmup, seed, budgets, and callbacks.
        Held by identity. The default runs four NUTS chains, with 500 warmup
        transitions and 1000 retained draws per chain, seeded with zero.
    """

    sampler: MCMC = dx.field(
        default_factory=lambda: MCMC(
            NUTSKernel(), num_warmup=500, num_samples=1000, seed=0
        ),
        opaque=True,
        extras=OPAQUE_BY_IDENTITY,
    )

    def run(self, model: BoundModel) -> ModelFit:
        """Sample the bound model using this driver's configuration.

        Parameters
        ----------
        model : BoundModel
            Validated data binding.

        Returns
        -------
        ModelFit
            Posterior draws and the original data binding.
        """
        return model.sample(self.sampler)


class VariationalMethod(ValueModel):
    """Optimize a guide and model using a chosen objective and optimizer.

    Parameters
    ----------
    num_steps : int
        Positive optimization step budget.
    learning_rate : float
        Positive Adam learning rate, used when no optimizer factory is given.
    seed : int
        CPU random seed for guide construction and training; restored on exit.
    guide_factory : Callable[[BoundModel], Guide] or None
        Build a fresh guide for each fit. The default is a traced AutoNormalGuide,
        which includes every position of a scan. Held by identity.
    objective : Objective
        Variational objective, held by identity; ELBO by default.
    optimizer_factory : Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer] or None
        Optional optimizer constructor, held by identity. Receives unique guide
        and model parameters so shared modules are optimized once.
    callback : Callable[[int, float], None] or None
        Called after each step with its zero-based index and loss. Held by identity.
    """

    num_steps: int = 1000
    learning_rate: float = 0.01
    seed: int = 0
    guide_factory: Callable[[BoundModel], Guide] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    objective: Objective = dx.field(
        default_factory=ELBO, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    optimizer_factory: (
        Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer] | None
    ) = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    callback: Callable[[int, float], None] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def run(self, model: BoundModel) -> ModelFit:
        """Fit one variational posterior and retain its loss history.

        Parameters
        ----------
        model : BoundModel
            Validated model and data.

        Returns
        -------
        ModelFit
            Bound model, guide, and losses. A guide's draws are approximate;
            optimization does not establish posterior convergence.

        Raises
        ------
        ValueError
            If a budget or rate is invalid, or an objective returns a nonfinite loss.
        """
        if (
            self.num_steps < 1
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("num_steps and learning_rate must be positive and finite")
        program = model.require_tensor_program()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            data = dict(model.data)
            guide = (
                self.guide_factory(model)
                if self.guide_factory is not None
                else AutoNormalGuide(
                    program,
                    observed_names=set(model.schema.observations),
                    x=model.input,
                    observations=data,
                )
            )
            parameters = list(
                {
                    id(p): p for p in (*program.parameters(), *guide.parameters())
                }.values()
            )
            optimizer = (
                self.optimizer_factory(parameters)
                if self.optimizer_factory is not None
                else torch.optim.Adam(parameters, lr=self.learning_rate)
            )
            svi = SVI(program, guide, optimizer, self.objective)
            losses = []
            for index in range(self.num_steps):
                loss = svi.step(model.input, data)
                if not math.isfinite(loss):
                    raise ValueError(f"nonfinite variational loss at step {index}")
                losses.append(loss)
                if self.callback is not None:
                    self.callback(index, loss)
        return ModelFit(model=model, posterior=guide, losses=tuple(losses))


__all__ = ["MCMCMethod", "VariationalMethod"]
