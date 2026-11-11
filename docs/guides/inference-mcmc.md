# MCMC and Hybrid Samplers

This page covers gradient-based MCMC (HMC, NUTS), the hybrid
samplers that combine a variational warm-up with HMC chains, and
[posterior predictive](inference-svi.md#predictive-sampling) sampling from an
[`MCMCResult`](../api/inference/mcmc.md). The variational
families and [ELBO](inference-svi.md#elbo-evidence-lower-bound) objectives live in
[the SVI guide](inference-svi.md).

## MCMC: HMC and NUTS

The algorithmic treatments are [@neal2011mcmc] for HMC and
[@hoffman2014nuts] for NUTS.

**Markov chain Monte Carlo (MCMC)** generates a sequence of dependent
draws designed to have the posterior as its stationary distribution.
The transition rule uses the model’s log density to propose and accept
new latent values. **Hamiltonian Monte Carlo (HMC)**
uses gradients of the log density and auxiliary momentum to propose
moves through continuous latent space. The **No-U-Turn Sampler (NUTS)**
adapts trajectory length to avoid continuing a path that turns back.

*Warmup* tunes the sampler before the retained draws. Separate *chains*
allow comparison of runs started from different initial values.
Successive draws within each chain remain correlated. The registry's unconstrained coordinates accommodate positive
or bounded continuous latents; these gradient-based kernels do not
automatically sample arbitrary discrete latent sites.

Gradient-based MCMC is an alternative to fitting a variational family. The kernel runs on the registry's unconstrained vector;
gradients flow through
[`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).

MCMC targets the posterior of one run of the program, so `MCMC.run`
takes a one-row input, such as `x = torch.zeros(1, 1)`, and refuses more
rows: [an input of `B` rows is `B` independent runs](inference-foundations.md#input-rows-are-independent-runs),
and the rows a program observes come from its data and its plates. Its
latents are the sites one run at `(x, observations)` reaches, every
position of a scanned program cell included.

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

print(result.r_hat)             # rank-normalized folded split R-hat
print(result.ess)               # rank-normalized bulk ESS
print(result.ess_tail)          # 5% / 95% quantile tail ESS
print(result.divergence_counts) # per-chain divergence count
print(result.total_divergences) # sum across chains
samples = result.samples        # dict[str, Tensor]: (chains, draws, *constrained_shape)
```

Both
[`HMCKernel`](../api/inference/mcmc.md) and
[`NUTSKernel`](../api/inference/mcmc.md) implement
Nesterov dual-averaging [@hoffman2014nuts]
step-size adaptation during warmup, and Welford-online mass-matrix
adaptation when `mass_matrix` is set to `"diagonal"` or `"dense"`
(`"identity"`, the `HMCKernel` default, disables mass-matrix
adaptation; `NUTSKernel` defaults to `"diagonal"`). Chains
run sequentially: the driver loops over `num_chains`, and within
each chain warmup and sampling advance one leapfrog step at a time.
Each leapfrog step evaluates the target once, through
[`PotentialFn.value_and_grad`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn.value_and_grad)
(one forward and one backward pass). The value it returns is the
endpoint log-density in the Hamiltonian, and NUTS carries the
selected proposal's gradient into the next transition, so a
transition costs one target evaluation per leapfrog step and nothing
more.
Each chain derives an independent local generator from `seed`; that
stream controls initialization, momentum, tree direction, subtree
selection, and acceptance. Thus repeated runs with the same seed are
reproducible without changing Torch's global RNG state. Pass a CPU
`torch.Generator` as `generator=` instead when successive calls should
advance a caller-owned stream. `seed` and `generator` are mutually
exclusive. Step-size and mass-matrix adaptation start fresh for every
chain and every call to `run`.

Before sampling, `MCMC.run` scores each chain's initial position with
[`PotentialFn.check_initial`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn.check_initial),
which converts no error: observed values that are not finite or lie
outside a likelihood's support, row counts that disagree, and errors in
a custom score function raise there, with a note naming the chain, and
a non-finite initial log density raises naming the offending terms.
During sampling a position scores $-\infty$, and is rejected, only when
it lies outside the model's support: a `torch.distributions` parameter
or value validation error, recognised by the validator that raised it,
or a `torch.linalg.LinAlgError` from a matrix the position makes
non-positive-definite. Every other error propagates.

The [`MCMCResult`](../api/inference/mcmc.md) holds each site's draws at
`(num_chains, num_samples, *constrained_shape)`, where
`constrained_shape` is the shape of one draw: `()` for a scalar site,
`(d,)` for a vector site, and `(n, ...)` for a plate of size `n`. It exposes per-site convergence and effective sample size diagnostics
[@vehtari2021rhat] via
`r_hat`, `ess`, and `ess_tail`, each of shape `constrained_shape` (zero-dimensional
for a scalar site), computed from the draws on first access by
[`rhat`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.rhat) and
[`ess`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.ess). [Latent site shapes](inference-foundations.md#latent-site-shapes)
relates these shapes to the `(batch, ...)` values that `log_joint`
and the guides exchange. Each `r_hat` coordinate is the conservative
maximum of the rank-normalized and folded split statistics, while
`ess` is the rank-normalized cross-chain bulk ESS and `ess_tail` the
smaller of the 5% and 95% quantile ESS values. The same functions
diagnose derived quantities and draws from custom sampling loops; see
[convergence diagnostics for any draws](analysis-fitting-and-diagnostics.md#convergence-diagnostics-for-any-draws). The result also
exposes per-chain divergence counts via `divergence_counts` (with
`total_divergences` summing across chains), and posterior log
densities per draw via `log_densities`.

## Hybrid samplers

`r_hat` compares variation within chains with variation between chains.
If chains stay in different regions, their means can differ enough to raise
`r_hat`. Chains trapped in the same region can agree, so compare runs from
dispersed initial values as well. `ess` accounts for autocorrelation: 8,000
saved draws might provide precision comparable to only 500 independent
draws. Divergences mark inaccurate numerical trajectories. Locate them in
parameter plots, then assess whether a smaller step size or a different
parameterization improves exploration.

Hybrid methods use variational fitting to construct a proposal
distribution or choose initial chain values, as specified below.

### AutoDAIS

Differentiable annealed importance
sampling [@zhang2021dais; @geffner2021uha] wraps a base
guide with $K$ uncorrected Hamiltonian transitions along an annealing
path between base and target: each transition draws a fresh momentum
and takes leapfrog steps on the bridge
$q_0^{1-\beta_k}\, p^{\beta_k}$, with no Metropolis correction. The
base mean and scale, step sizes, and inverse temperatures are jointly
trained via SVI, differentiating through the whole trajectory.

Each draw comes with the log importance weight of its own trajectory,

$$\log w = \log p(u_K, y) - \log q_0(u_0)
  + \sum_{k=1}^{K}\bigl[\log \mathcal{N}(\hat v_k) - \log \mathcal{N}(v_{k-1})\bigr],$$

where $v_{k-1}$ and $\hat v_k$ are the momenta before and after
transition $k$ and both densities are taken in the unconstrained space
the chain runs in (Eq. 15 of [@zhang2021dais]). Because leapfrog preserves
volume, $\mathbb{E}[w] = p(y)$, so `ELBO` with this guide estimates
$\mathbb{E}[\log w] \le \log p(y)$, a valid lower bound, and
`IWAEBound` gives the importance-weighted bound from the same weights.
The bridges and step sizes shape the chain, not the validity of the
bound. The bound is not guaranteed to exceed the base guide's ELBO.

The endpoint has no closed-form density, so `AutoDAIS.log_prob` raises.
Use the `Reparameterized` estimator: `StickingTheLanding`,
`DoublyReparameterized` (the `IWAEBound` default), `ScoreFunction`, and
`RWS` need the guide density at a fixed point and raise with this
guide.

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
# Plug into SVI with ELBO() or IWAEBound(estimator=Reparameterized()).
```

### WarmupThenHMC

Train a variational guide for a configured number of SVI steps, then
initialize HMC chains from the guide's posterior mean. This provides
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

Posterior draws represent uncertainty about latent quantities. Predictive
draws add the outcome noise: [take a posterior latent draw, then generate
a new response from its likelihood](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html#simulating-from-the-posterior-predictive-distribution) [@stanPredictiveChecks]. Thus an interval for new observations
includes both uncertainty about the latent mean and variability of the
response around it. The same calculation uses draws from either an MCMC
result or a variational guide. Comparing these replicated responses with
the observations gives a [posterior predictive check](analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics).

[`Predictive`](../api/inference/predictive.md) consumes either a
[`Guide`](../api/inference/guide.md) or an
[`MCMCResult`](../api/inference/mcmc.md). With an
`MCMCResult`, it iterates over posterior samples instead of calling
`guide.rsample`; each recorded draw is one value every run of the
predictive input shares, so both posteriors yield predictive draws of
shape `(num_samples, batch, ...)` for non-plate sites.

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

- [Inference Foundations](inference-foundations.md): the [trace](inference-foundations.md#trace-and-sample-sites),
  [conditioning](inference-foundations.md#conditioning-on-observations), and `LatentRegistry` primitives every MCMC kernel
  consumes.
- [SVI guide](inference-svi.md): the variational counterpart, and
  the SVI driver wrapping the `AutoDAIS` hybrid sampler.
- [Analysis Pipelines: Fitting and Diagnostics](analysis-fitting-and-diagnostics.md):
  the high-level `fit(...)` API that wraps the MCMC and SVI
  drivers under one entry point.

## Further reading

- [@vehtari2021rhat]
