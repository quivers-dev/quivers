# MCMC

Markov chain Monte Carlo (MCMC) generates correlated draws targeting a posterior.
Hamiltonian Monte Carlo (HMC) uses gradients and simulated momentum; the No-U-Turn
Sampler (NUTS) adapts the trajectory length. Warmup tunes the algorithm before draws are
retained. Compare chains for unexplored regions and inspect effective sample sizes to
assess the precision of posterior summaries.

See [HMC, NUTS, and their settings](../../guides/inference-mcmc.md#mcmc-hmc-and-nuts) and
[interpreting sampler diagnostics](../../tutorials/python/09-debugging.md#nuts-diagnostics-that-point-at-root-causes).

For HMC and NUTS, see [@neal2011mcmc; @hoffman2014nuts].

The acceptance rate discussed below comes from an asymptotic analysis of
high-dimensional product targets [@beskos2013optimal]. Its assumptions determine
when that tuning result applies.

Markov-chain Monte Carlo: the [`HMC`](#quivers.inference.mcmc.HMCKernel)
and [`NUTS`](#quivers.inference.mcmc.NUTSKernel) kernels, the
[`MCMC`](#quivers.inference.mcmc.MCMC) runner, and the
[`MCMCResult`](#quivers.inference.mcmc.MCMCResult) summary.

The runner targets a `MonadicProgram`. Models that declare every latent
as a `sample` site need no conversion. For models with `nn.Parameter`s
or intermediate latent sites, the
[`bayesian_lift_parameters`](lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
lift returns a 3-tuple `(MonadicProgram, torch.Tensor, dict[str, torch.Tensor])`,
whose first element is the matching `MonadicProgram`. Callers pass
that first element to the runner.

`quivers.inference` re-exports the kernels, the runner, and its result;
`quivers.inference.mcmc` also exports the warmup adaptation primitives
and the type aliases that parameterize the kernels and the runner.

::: quivers.inference.mcmc

## Variational warmup

`WarmupThenHMC` fits a guide with SVI and starts the chains from it.

::: quivers.inference.warmup
