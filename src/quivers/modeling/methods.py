"""Reusable inference components for bound QVR models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
    SMC,
    SMCResult,
    SVI,
)
from quivers.modeling.model import BoundModel, Model, ModelFit
from quivers.family_catalog import FamilyCatalog
from quivers.inference.mcmc.parallel import ChainStates, parallel_chains, run_chains
from quivers.inference.pathfinder import Pathfinder
from quivers.inference.svi import _prepare_optimizer


class SourceChainWorker(ValueModel):
    """Rebuild a bound QVR model in a worker process and run some chains.

    The worker recompiles the source with the binding's data, static
    arguments, input, and compilation seed, so it builds the program the
    parent sampled, then runs its assigned chains with a copy of the parent's
    driver.

    Parameters
    ----------
    source : str
        QVR source text.
    entry : str
        Selected program.
    families : FamilyCatalog or None
        Native family definitions of the compilation.
    data : Mapping[str, object]
        External values of the binding.
    parameters : Mapping[str, object]
        Static arguments of the binding.
    input : torch.Tensor
        One-run input.
    seed : int
        Compilation seed.
    sampler : MCMC
        The parent's driver.
    """

    source: str
    entry: str
    families: FamilyCatalog | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    data: Mapping[str, object] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    parameters: Mapping[str, object] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    input: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    seed: int
    sampler: MCMC = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def __call__(self, chains: tuple[int, ...]) -> ChainStates:
        """Recompile the binding and run the assigned chains.

        Parameters
        ----------
        chains : tuple[int, ...]
            Chain indices.

        Returns
        -------
        ChainStates
            Their records.
        """
        bound = Model(self.source, entry=self.entry, families=self.families).bind(
            self.data, parameters=self.parameters, input=self.input, seed=self.seed
        )
        return run_chains(
            self.sampler,
            bound.require_tensor_program(),
            bound.input,
            bound.tensor_data(),
            chains,
        )


class MCMCMethod(ValueModel):
    """Fit with a reusable MCMC driver, defaulting to seeded NUTS.

    Parameters
    ----------
    sampler : MCMC
        Driver carrying kernel, chains, warmup, seed, budgets, and callbacks.
        Held by identity. The default runs four NUTS chains, with 500 warmup
        transitions and 1000 retained draws per chain, seeded with zero.
    cores : int
        Worker processes for the chains (brms ``cores``). With more than
        one, each worker recompiles the model from its source and binding
        and runs its share of the chains; the result equals the serial run.
    """

    sampler: MCMC = dx.field(
        default_factory=lambda: MCMC(
            NUTSKernel(), num_warmup=500, num_samples=1000, seed=0
        ),
        opaque=True,
        extras=OPAQUE_BY_IDENTITY,
    )
    cores: int = 1

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

        Raises
        ------
        ValueError
            If ``cores`` is not a positive integer, or parallel chains are
            requested for a driver without a seed or with a checkpoint or
            callback, or for a binding that cannot be pickled.
        """
        if isinstance(self.cores, bool) or self.cores < 1:
            raise ValueError("cores must be a positive integer")
        if self.cores == 1 or self.sampler.num_chains == 1:
            return model.sample(self.sampler)
        specification = model.specification
        worker = SourceChainWorker(
            source=specification.source,
            entry=specification.entry,
            families=specification.families,
            data=model.data,
            parameters=model.template_arguments,
            input=model.input,
            seed=model.compilation_seed,
            sampler=self.sampler,
        )
        posterior = parallel_chains(
            self.sampler,
            model.require_tensor_program(),
            model.input,
            model.tensor_data(),
            worker,
            self.cores,
        )
        return ModelFit(model=model, posterior=posterior)


class SMCMethod(ValueModel):
    """Apply likelihood-tempered SMC to a checked data binding.

    Parameters
    ----------
    sampler : SMC
        Existing driver with explicit particle, rejuvenation, adaptation, and
        random-stream settings, held by identity. The default uses NUTS
        rejuvenation, 1024 particles, and seed zero.

    Notes
    -----
    The result retains the driver's particles, temperature ladder, weight ESS,
    acceptance, evaluation count, and estimated log normalizing constant.
    Particles are not recorded MCMC chains. A data-evidence interpretation
    requires the declared prior and likelihood to define that comparison.
    """

    sampler: SMC = dx.field(
        default_factory=lambda: SMC(NUTSKernel(), seed=0),
        opaque=True,
        extras=OPAQUE_BY_IDENTITY,
    )

    def run(self, model: BoundModel) -> SMCResult:
        """Run the existing sampler without converting its result.

        Parameters
        ----------
        model : BoundModel
            Checked program, validated data, and one-run input.

        Returns
        -------
        SMCResult
            Equally weighted particles and the driver's evidence diagnostics.

        Raises
        ------
        TypeError
            If the entry has no tensor executor.
        ValueError
            If the driver cannot partition the target, normalize its prior,
            or use an exact likelihood without observation subsampling.
        """
        return self.sampler.run(
            model.require_tensor_program(), model.input, model.tensor_data()
        )


class PathfinderMethod(ValueModel):
    """Fit with single- or multi-path Pathfinder variational inference.

    Parameters
    ----------
    pathfinder : Pathfinder
        Driver carrying the path count, draw counts, L-BFGS settings,
        initialization, and seed. The default runs CmdStan's defaults: four
        paths of 1000 draws resampled to 1000 draws, history size 5, and
        25 draws per ELBO estimate, seeded with zero. Held by identity.
    """

    pathfinder: Pathfinder = dx.field(
        default_factory=Pathfinder, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def run(self, model: BoundModel) -> ModelFit:
        """Run Pathfinder on the bound model's unconstrained log joint.

        Parameters
        ----------
        model : BoundModel
            Validated data binding.

        Returns
        -------
        ModelFit
            Bound model and `PathfinderDraws`, read as independent
            approximate posterior draws.

        Raises
        ------
        TypeError
            If the entry has no tensor executor.
        ValueError
            If every path fails or an initial point is invalid.
        """
        posterior = self.pathfinder.run(
            model.require_tensor_program(), model.input, model.tensor_data()
        )
        return ModelFit(model=model, posterior=posterior)


class VariationalMethod(ValueModel):
    """Optimize a guide and model using a chosen objective and optimizer.

    Parameters
    ----------
    num_steps : int
        Nonnegative optimization step budget; zero retains the initialized guide.
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
            self.num_steps < 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "num_steps must be nonnegative and learning_rate must be positive and finite"
            )
        program = model.require_tensor_program()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            data = model.tensor_data()
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
            optimizer = _prepare_optimizer(
                program.parameters(),
                guide.parameters(),
                learning_rate=self.learning_rate,
                factory=self.optimizer_factory,
            )
            svi = SVI(program, guide, optimizer, self.objective)
            losses = svi.run(
                model.input, data, num_steps=self.num_steps, callback=self.callback
            )
        return ModelFit(model=model, posterior=guide, losses=tuple(losses))


__all__ = [
    "MCMCMethod",
    "PathfinderMethod",
    "SMCMethod",
    "SourceChainWorker",
    "VariationalMethod",
]
