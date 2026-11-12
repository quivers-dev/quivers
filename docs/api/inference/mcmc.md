# MCMC

[Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) draws a
correlated sequence whose stationary distribution is the posterior.
[Hamiltonian Monte Carlo](../../guides/inference-mcmc.md#mcmc-hmc-and-nuts) (HMC) proposes moves by
simulating momentum with gradients, and the No-U-Turn Sampler (NUTS) adapts the length of each
trajectory [@neal2011mcmc; @hoffman2014nuts]. The module gives the
[`HMC`](#quivers.inference.mcmc.HMCKernel) and [`NUTS`](#quivers.inference.mcmc.NUTSKernel)
kernels, the [`MCMC`](#quivers.inference.mcmc.MCMC) runner, and the
[`MCMCResult`](#quivers.inference.mcmc.MCMCResult) summary. Warmup tunes the step size and mass
matrix before draws are kept, and [windowed warmup](../../guides/inference-mcmc.md#windowed-warmup)
follows Stan's schedule. The target acceptance rate comes from an asymptotic analysis of
high-dimensional product targets [@beskos2013optimal]. Several chains reveal regions one chain
missed, and the
[effective sample size](../../tutorials/python/09-debugging.md#nuts-diagnostics-that-point-at-root-causes)
measures the precision of posterior summaries [@vehtari2021rhat].

The runner targets a `MonadicProgram`. Models that declare every latent
as a `sample` site need no conversion. For models with `nn.Parameter`s
or intermediate latent sites, the
[`bayesian_lift_parameters`](lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
function returns a 3-tuple `(MonadicProgram, torch.Tensor, dict[str, torch.Tensor])`,
whose first element is the program to pass to the runner.

`quivers.inference` re-exports the kernels, the runner, and its result.
`quivers.inference.mcmc` also exports the warmup adaptation primitives
and the type aliases that parameterize the kernels and the runner.

::: quivers.inference.mcmc

## Sequential Monte Carlo

[Tempered sequential Monte Carlo](../../guides/inference-mcmc.md#tempered-sequential-monte-carlo)
moves a population from the prior to the posterior, rejuvenating it with
any of the kernels above, and estimates the evidence
[@delmoral2006sequential].

::: quivers.inference.smc

## Variational warmup

`WarmupThenHMC` fits a guide with SVI and starts the chains from it.

::: quivers.inference.warmup
