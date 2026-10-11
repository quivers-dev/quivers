# Variational Inference: MCMC and Hybrid Samplers

This page covers gradient-based MCMC (HMC, NUTS), the hybrid
samplers that combine a variational warm-up with HMC chains, and
posterior predictive sampling from an
[`MCMCResult`](../api/inference/mcmc.md). The variational
families and ELBO objectives live in
[the SVI guide](inference-svi.md).

## MCMC: HMC and NUTS

When variational families underfit, fall back to gradient-based
MCMC. The kernel runs on the registry's unconstrained vector;
gradients flow through
[`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).

<!-- python: skip -->
```python
from quivers.inference import NUTSKernel, MCMC

kernel = NUTSKernel(
    target_accept=0.8,
    max_tree_depth=10,
    mass_matrix="diagonal",
)
mcmc = MCMC(
    kernel=kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    seed=2026,
)
result = mcmc.run(model, x, observations)

print(result.r_hat)             # per-site split R-hat (Vehtari et al. 2021)
print(result.ess)               # effective sample size
print(result.divergence_counts) # per-chain divergence count
print(result.total_divergences) # sum across chains
samples = result.samples        # dict[str, Tensor] of shape (chains, draws, ...)
```

Both
[`HMCKernel`](../api/inference/mcmc.md) and
[`NUTSKernel`](../api/inference/mcmc.md) implement
[Nesterov dual-averaging](https://doi.org/10.48550/arXiv.1111.4246)
step-size adaptation during warmup, and Welford-online mass-matrix
adaptation when `mass_matrix` is set to `"diagonal"` or `"dense"`
(the default `"identity"` disables mass-matrix adaptation). Chains
run sequentially: the driver loops over `num_chains`, and within
each chain warmup and sampling advance one leapfrog step at a time.
Each chain derives an independent local generator from `seed`; that
stream controls initialization, momentum, tree direction, subtree
selection, and acceptance. Thus repeated runs with the same seed are
reproducible without changing Torch's global RNG state. Pass a CPU
`torch.Generator` as `generator=` instead when successive calls should
advance a caller-owned stream. `seed` and `generator` are mutually
exclusive. Step-size and mass-matrix adaptation start fresh for every
chain and every call to `run`.

The [`MCMCResult`](../api/inference/mcmc.md) exposes per-site
[split `R̂` and effective sample size (Vehtari et al.
2021)](https://doi.org/10.1214/20-BA1221) via `r_hat` and `ess`,
per-chain divergence counts via `divergence_counts` (with
`total_divergences` summing across chains), and posterior log
densities per draw via `log_densities`.

## Hybrid samplers

### AutoDAIS

[Differentiable annealed importance
sampling](https://doi.org/10.48550/arXiv.2107.10211) wraps a base
guide with $K$ HMC trajectories along an annealing path between
base and target. The base mean and scale, step size, and inverse
temperatures are jointly trained via SVI.

<!-- python: skip -->
```python
from quivers.inference import AutoNormalGuide, AutoDAIS

base = AutoNormalGuide(model, observed_names={"y"})
guide = AutoDAIS(
    base,
    model=model,
    observations=observations,
    num_steps=8,
    init_step_size=0.05,
    init_temperature=0.1,
)
# Plug into SVI exactly like any other guide.
```

### WarmupThenHMC

Train a variational guide for a configured number of SVI steps, then
initialise HMC chains from the guide's posterior mean. This provides
an informed starting point; convergence still depends on the HMC
warmup and diagnostics.

<!-- python: skip -->
```python
from quivers.inference import (
    AutoMultivariateNormalGuide, NUTSKernel, WarmupThenHMC
)

sampler = WarmupThenHMC(
    guide=AutoMultivariateNormalGuide(model, observed_names={"y"}),
    kernel=NUTSKernel(),
    svi_steps=1000,
    mcmc_warmup=500,
    mcmc_samples=2000,
)
svi_losses, result = sampler.run(model, x, observations)
```

## Predictive with MCMC

[`Predictive`](../api/inference/predictive.md) consumes either a
[`Guide`](../api/inference/guide.md) or an
[`MCMCResult`](../api/inference/mcmc.md). With an
`MCMCResult`, it iterates over posterior samples instead of calling
`guide.rsample`.

<!-- python: skip -->
```python
from quivers.inference import Predictive

predictive = Predictive(
    model=conditioned.model,
    posterior=result,
    num_samples=500,
)
samples = predictive(x_new)
```

## See also

- [Inference Foundations](inference-foundations.md): the trace,
  conditioning, and `LatentRegistry` primitives every MCMC kernel
  consumes.
- [SVI guide](inference-svi.md): the variational counterpart, and
  the SVI driver wrapping the `AutoDAIS` hybrid sampler.
- [Analysis Pipelines: Fitting and Diagnostics](analysis-fitting-and-diagnostics.md):
  the high-level `fit(...)` API that wraps the MCMC and SVI
  drivers under one entry point.


## References

- Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul-Christian Bürkner. 2021. [Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC](https://doi.org/10.1214/20-BA1221). *Bayesian Analysis*, 16(2):667–718.
