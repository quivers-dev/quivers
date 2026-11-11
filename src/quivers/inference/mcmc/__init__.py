"""MCMC kernels, adaptation primitives, and the chain driver.

Public surface (the kernels and driver are also re-exported from
`quivers.inference`):

* `MCMCKernel`: ABC for Markov kernels on the flat unconstrained
  latent vector, with `KernelState` as its per-chain state and
  `PotentialFn` as the potential it evaluates.
* `HMCKernel`: Hamiltonian Monte Carlo with leapfrog integration,
  dual-averaging step-size adaptation, and Welford mass-matrix
  adaptation, parameterized by a `MassMatrixKind`.
* `NUTSKernel`: No-U-Turn Sampler with multinomial sampling.
* `DualAveraging`, `WelfordCovariance`, and
  `find_reasonable_step_size`: the warmup adaptation primitives the
  kernels share.
* `AdaptationSchedule`, `AdaptationWindow`, and `AdaptationMode`: the
  Stan-style windowed warmup schedule the kernels follow by default.
* `MetropolisKernel` with a `Proposal` (`RandomWalkProposal`,
  `IndependenceProposal`): Metropolis-Hastings moves on the flat
  vector.
* `GibbsKernel`: Gibbs blocks over named latent sites held outside the
  flat vector, discrete ones included, updated by enumeration or by a
  `GibbsUpdate`.
* `CycleKernel` and `MixtureKernel`: compositions of kernels.
* `MCMC`: chain orchestrator with warmup, parallel chains, an
  `InitStrategy`, and posterior diagnostics (split-:math:`\\hat R`,
  effective sample size).
* `MCMCResult`: posterior samples and per-chain diagnostics.
"""

from __future__ import annotations

from quivers.inference.mcmc.adapt import (
    AdaptationMode,
    AdaptationSchedule,
    AdaptationWindow,
    DualAveraging,
    WelfordCovariance,
    find_reasonable_step_size,
)
from quivers.inference.mcmc.compose import CycleKernel, MixtureKernel
from quivers.inference.mcmc.driver import MCMC, InitStrategy, MCMCResult
from quivers.inference.mcmc.gibbs import GibbsKernel, GibbsUpdate
from quivers.inference.mcmc.hmc import HMCKernel, MassMatrixKind, NUTSKernel
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
)
from quivers.inference.mcmc.metropolis import (
    IndependenceProposal,
    MetropolisKernel,
    Proposal,
    RandomWalkProposal,
)

__all__ = [
    "MCMCKernel",
    "KernelState",
    "PotentialFn",
    "HMCKernel",
    "NUTSKernel",
    "MassMatrixKind",
    "MetropolisKernel",
    "Proposal",
    "RandomWalkProposal",
    "IndependenceProposal",
    "GibbsKernel",
    "GibbsUpdate",
    "CycleKernel",
    "MixtureKernel",
    "AdaptationMode",
    "AdaptationSchedule",
    "AdaptationWindow",
    "DualAveraging",
    "WelfordCovariance",
    "find_reasonable_step_size",
    "MCMC",
    "MCMCResult",
    "InitStrategy",
]
