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
or bounded continuous latents. The gradient-based kernels move only
those coordinates; a discrete latent site is sampled by a
[Gibbs block](#composing-kernels) composed with them, or marginalized
in the model.

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

### Windowed warmup

By default both kernels adapt in one phase (`adaptation="single"`):
dual averaging runs over the whole warmup under the initial metric, the
covariance estimate accumulates every warmup position, and the metric is
installed once, when warmup ends. Two problems may follow on a target
whose scales differ from one. The estimate includes the initial
transient, which can inflate it by orders of magnitude when a chain
starts far from the typical set, and the frozen step size was tuned
for the identity metric rather than the metric the chain samples with.

`adaptation="windowed"` follows Stan's schedule instead
[@stanHmcTuning]. An
[`AdaptationSchedule`](../api/inference/mcmc.md#quivers.inference.mcmc.AdaptationSchedule)
splits the warmup into three phases: (i) an initial fast interval
(`init_buffer`, default 75 iterations) that adapts only the step size;
(ii) slow windows, the first `base_window` (default 25) iterations long
and each later one twice as long, that each accumulate a fresh
covariance estimate and install it, regularised as
$\frac{n}{n+5}\hat\Sigma + 10^{-3}\frac{5}{n+5} I$, at the window's
end, after which the step size is re-initialized by doubling or halving
under the new metric and dual averaging restarts from it; and (iii) a
terminal fast interval (`term_buffer`, default 50) that tunes the step
size under the final metric. When the window after next would not fit
before the terminal interval, the current slow window is stretched to
reach it. A warmup too short for the buffers
(`init_buffer + base_window + term_buffer > num_warmup`) rescales them
to 15% / 75% / 10% of the warmup, and a warmup of fewer than 20
iterations estimates no metric at all.
[`NUTSKernel.adaptation_windows`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel.adaptation_windows)
returns the windows a given warmup will run, as
[`AdaptationWindow`](../api/inference/mcmc.md#quivers.inference.mcmc.AdaptationWindow)
records, and `inverse_mass_matrix` reports the metric in use.

<!-- python: skip -->
```python
import torch

from quivers.dsl import loads
from quivers.inference.mcmc import MCMC, AdaptationSchedule, NUTSKernel

model = loads(
    "object Obs : FinSet 20\n"
    "program p : Obs -> Obs\n"
    "    sample mu <- Normal(loc=0.0, scale=10.0)\n"
    "    sample tau <- HalfNormal(scale=0.1)\n"
    "    observe y : Obs <- Normal(loc=mu, scale=tau)\n"
    "    return mu\n"
    "export p\n"
).morphism

kernel = NUTSKernel(
    mass_matrix="diagonal",
    adaptation="windowed",
    adaptation_schedule=AdaptationSchedule(init_buffer=75, term_buffer=50, base_window=25),
)
for window in kernel.adaptation_windows(500):
    print(window.start, window.stop, window.adapts_metric)
# 0 75 False, 75 100 True, 100 150 True, 150 250 True, 250 450 True, 450 500 False

result = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=2, seed=0).run(
    model, torch.zeros(1, 1), {"y": 3.0 + 0.1 * torch.randn(20)}
)
print(kernel.step_size, kernel.inverse_mass_matrix)
```

The windowed schedule suits warmups long enough for its buffers, 150
iterations under the defaults. A shorter warmup is rescaled, and its
terminal interval may then last only a few iterations, too few for dual
averaging to settle, which is why `"single"` remains the default. Under
`"single"` mode the fixed-seed draws of a run are those the kernels have
always produced.

## Composing kernels

`MCMC` drives one kernel, and that kernel may be built from several. If
each kernel leaves the posterior invariant, so does applying them in
turn and so does choosing one at random with fixed probabilities
[@tierney1994markov]. Thus a chain may interleave NUTS with cheap
Metropolis-Hastings moves that cross between modes NUTS connects only
slowly, or alternate Gibbs updates of discrete latents with NUTS on the
continuous ones.

- [`MetropolisKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.MetropolisKernel)
  moves the flat unconstrained vector by a
  [`Proposal`](../api/inference/mcmc.md#quivers.inference.mcmc.Proposal),
  which returns a candidate and its log Hastings correction
  $\log q(z \mid z') - \log q(z' \mid z)$ [@hastings1970monte]. The
  kernel scores the candidate by its log-density alone and evaluates
  the gradient only when it accepts, so a gradient kernel that follows
  reuses it.
  [`RandomWalkProposal`](../api/inference/mcmc.md#quivers.inference.mcmc.RandomWalkProposal)
  is a symmetric Gaussian walk, and
  [`IndependenceProposal`](../api/inference/mcmc.md#quivers.inference.mcmc.IndependenceProposal)
  draws from a fixed distribution over the flat vector (a Gaussian fitted
  to a variational guide, for instance), optionally a defensive mixture
  with a heavier-tailed one [@hesterberg1995weighted]. A proposal that
  needs the layout of the flat vector, to swap two exchangeable blocks
  for instance, overrides `Proposal.bind` to receive the registry.
- [`GibbsKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.GibbsKernel)
  updates named latent sites from their full conditional
  [@geman1984stochastic]. `MCMC.run` leaves these sites out of the
  registry, keeps their values in `KernelState.gibbs_values`, and hands
  every kernel the potential conditioned on them
  ([`PotentialFn.condition`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn.condition)).
  A site with a boolean or bounded integer support (a Bernoulli or
  Categorical draw, a plate of them) is updated by enumeration, element
  by element; any other block takes an `update=` function
  ([`GibbsUpdate`](../api/inference/mcmc.md#quivers.inference.mcmc.GibbsUpdate))
  that draws from its full conditional, such as a conjugate update. Every
  discrete latent must belong to a Gibbs block, and `MCMC.run` raises
  naming any that does not.
- [`CycleKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.CycleKernel)
  applies its components in order on every transition, and
  [`MixtureKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.MixtureKernel)
  applies one, chosen with fixed probabilities. Both forward the
  chain-local generator and warmup to every component; a mixture draws
  its warmup choices when warmup starts, so each component adapts over
  exactly the transitions that select it.

<!-- python: skip -->
```python
import torch

from quivers.dsl import loads
from quivers.inference.mcmc import (
    MCMC,
    CycleKernel,
    GibbsKernel,
    MetropolisKernel,
    NUTSKernel,
    RandomWalkProposal,
)

model = loads(
    "object Obs : FinSet 6\n"
    "program p : Obs -> Obs\n"
    "    sample loc <- Normal(loc=0.0, scale=3.0)\n"
    "    sample side <- Categorical(probs=[0.5, 0.5])\n"
    "    let m = loc * sign[side]\n"
    "    observe y : Obs <- Normal(loc=m, scale=1.0)\n"
    "    return loc\n"
    "export p\n"
).morphism

kernel = CycleKernel(
    {
        "side": GibbsKernel(["side"]),
        "nuts": NUTSKernel(),
        "walk": MetropolisKernel(RandomWalkProposal(2.0)),
    }
)
result = MCMC(kernel, num_warmup=300, num_samples=500, num_chains=2, seed=0).run(
    model,
    torch.zeros(1, 1),
    {"y": torch.full((6,), 1.5), "sign": torch.tensor([-1.0, 1.0])},
)
print(result.samples["side"].shape)    # (2, 500): Gibbs sites are in the samples
print(result.component_acceptance)     # {"nuts": ..., "side": ..., "walk": ...}
```

A transition's statistics are in `KernelState.stats`, each component's
under its name (`"nuts.tree_depth"`), and
[`MCMCResult`](../api/inference/mcmc.md#quivers.inference.mcmc.MCMCResult)
reports each component's acceptance rate per chain in
`component_acceptance`. For a composite kernel, `acceptance_rates` counts a
transition as accepted when it moved the chain. Enumeration evaluates the
model's log joint once per value of each element, so a large discrete
block may need an `update=` function that exploits its structure.

## Checkpoints, budgets, and sampler statistics

A long run of an expensive target is exposed to crashes, preemption,
and power loss. With an
[`MCMCCheckpoint`](../api/inference/mcmc.md#quivers.inference.mcmc.MCMCCheckpoint),
`MCMC.run` writes every chain's state to one file in a directory: the
chain's phase and position in it, its
[`KernelState`](../api/inference/mcmc.md#quivers.inference.mcmc.KernelState),
the kernel's adaptation state (`MCMCKernel.state_dict`), the chain
generator's state, and the draws so far. It writes after every `every`
transitions, after `every_seconds`, when a chain finishes, and when a
budget stops it. Each write goes to a temporary file that then replaces
the previous checkpoint atomically, so an interrupted write leaves the
last good checkpoint in place. The file holds only tensors and plain
containers and is read with `torch.load(..., weights_only=True)`, so
reading it runs no code.
[`MCMC.resume`](../api/inference/mcmc.md#quivers.inference.mcmc.MCMC.resume)
continues the run, and its draws, statistics, and diagnostics are bit
for bit those of an uninterrupted run with the same seed. It refuses a
checkpoint written for a different model or data, kernel
configuration, run settings, seed, or checkpoint format, naming which.

`max_seconds` and `max_evaluations` cap each chain's wall-clock time
and potential evaluations in one call. They are checked between
transitions, so a chain stops at a transition boundary, possibly one
transition past the budget. `MCMCResult.stopped_by` records why each chain
stopped, `num_draws` how many draws each made, and `truncated` whether a
budget cut any short; the per-draw arrays then keep the first
`min(num_draws)` draws of every chain. A `callback(chain, iteration,
state, phase)` sees every transition as it happens.

<!-- python: skip -->
```python
import torch

from quivers.dsl import loads
from quivers.inference.mcmc import MCMC, MCMCCheckpoint, NUTSKernel

model = loads(
    "object Obs : FinSet 20\n"
    "program p : Obs -> Obs\n"
    "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
    "    sample tau <- HalfNormal(scale=1.0)\n"
    "    observe y : Obs <- Normal(loc=mu, scale=tau)\n"
    "    return mu\n"
    "export p\n"
).morphism
x, data = torch.zeros(1, 1), {"y": torch.linspace(-1.0, 2.0, 20)}


def make_mcmc(max_evaluations: int | None = None) -> MCMC:
    return MCMC(
        NUTSKernel(adaptation="windowed"),
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        seed=0,
        checkpoint=MCMCCheckpoint(directory="mcmc-run", every=50),
        max_evaluations=max_evaluations,
    )


partial = make_mcmc(max_evaluations=2000).run(model, x, data)
print(partial.stopped_by, partial.num_draws)   # ('evaluations', 'evaluations') ...
result = make_mcmc().resume("mcmc-run", model, x, data)
print(result.stopped_by)                       # ('draws', 'draws')

energy = result.sample_stats["energy"]         # (num_chains, num_samples)
print(result.sample_stats["tree_depth"].max(), result.num_evaluations)
```

`MCMCResult.sample_stats` holds each transition's statistics at
`(num_chains, num_samples)`. For NUTS these are `accept_prob`, `accepted`,
`diverged`, `step_size`, `tree_depth`, `num_steps` (leapfrog steps), and
`energy`, the Hamiltonian at the draw; a composite kernel's appear under
each component's name. `energy` is what the energy plot and the
energy Bayesian fraction of missing information (E-BFMI) need
[@betancourt2016diagnosing], and
[`to_datatree`](../api/diagnostics/arviz_io.md#quivers.diagnostics.arviz_io.to_datatree)
writes these statistics under ArviZ's names (`diverging`, `tree_depth`,
`n_steps`, `energy`, `step_size`, `acceptance_rate`), so divergences
can be located in parameter space and tree-depth saturation checked
draw by draw.

## Tempered sequential Monte Carlo

A single NUTS chain may stay in one mode of a multimodal posterior, and
MCMC gives no estimate of the evidence $p(y)$ for model comparison.
[`SMC`](../api/inference/mcmc.md#quivers.inference.smc.SMC) moves a
population of particles from the prior to the posterior through the
tempered targets $\pi_\beta(z) \propto p(z)\, L(z)^\beta$,
$0 = \beta_0 < \dots < \beta_T = 1$ [@delmoral2006sequential]. Here
$p(z)$ is the density of the model's latent draws, from which the
initial particles are drawn exactly by running the model, and $L(z)$
is everything else the model scores, its observations and its score
steps.
[`TemperedPotential`](../api/inference/mcmc.md#quivers.inference.smc.TemperedPotential)
evaluates $\log p(z) + \beta \log L(z)$ and returns the two parts
separately.

Each step does four things.

1. It chooses the next temperature by bisection, as the largest
   $\beta' \le 1$ whose incremental weights $L(z_i)^{\beta' - \beta}$ keep
   a conditional effective sample size of `target_ess` times the
   population [@jasra2011inference; @zhou2016toward].
2. It reweights, adding the log of the weighted mean increment to the
   log-evidence estimate.
3. It resamples, by `"systematic"`, `"stratified"`, `"residual"`, or
   `"multinomial"` resampling, each of which gives every particle an
   expected number of copies of $N$ times its weight [@douc2005comparison].
4. It rejuvenates every particle by `num_mcmc_steps` transitions of the
   kernel at the new temperature.

Any [composed kernel](#composing-kernels) serves for rejuvenation,
Gibbs blocks over discrete sites included. At each temperature the
kernel is first tuned to the population (`MCMCKernel.tune` sets an HMC
metric to the particles' covariance, and a random walk's scale to
$2.38/\sqrt{D}$ times their spread), then tuned further on a discarded
pilot run of `num_pilot_steps` transitions. It is then frozen, so every
move at that temperature leaves $\pi_{\beta'}$ invariant.

The product of the mean incremental weights is an unbiased estimate of
$p(y)$, and `SMCResult.log_evidence` is its logarithm (so it tends to
underestimate $\log p(y)$ by about half the estimate's variance).
Agreement between independent runs, and with MCMC, is a check on both.
Like `MCMC.run`, `SMC.run` takes a one-row input.

<!-- python: skip -->
```python
import torch

from quivers.dsl import loads
from quivers.inference import SMC
from quivers.inference.mcmc import HMCKernel

model = loads(
    "object Obs : FinSet 10\n"
    "program p : Obs -> Obs\n"
    "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
    "    observe y : Obs <- Normal(loc=mu, scale=1.0)\n"
    "    return mu\n"
    "export p\n"
).morphism
y = torch.linspace(0.5, 2.5, 10)

result = SMC(
    HMCKernel(num_steps=5, mass_matrix="diagonal"),
    num_particles=200,
    num_mcmc_steps=3,
    target_ess=0.5,
    resampling="systematic",
    seed=0,
).run(model, torch.zeros(1, 1), {"y": y})
print(result.log_evidence)          # close to log N(y; 0, I + 1 1^T) = -13.45
print(result.temperatures)          # the adaptive ladder, from 0 to 1
print(result.ess, result.acceptance)
print(result.samples["mu"].shape)   # (200,): equally weighted particles
```

## Hybrid samplers

`r_hat` compares variation within chains with variation between chains.
If chains stay in different regions, their means can differ enough to raise
`r_hat`. Chains trapped in the same region can agree, so compare runs from
dispersed initial values as well. `ess` accounts for autocorrelation: 8,000
saved draws might provide precision comparable to only 500 independent
draws. Divergences mark inaccurate numerical trajectories. Locate them in
parameter plots, then assess whether a smaller step size or a different
parameterization improves exploration.

These diagnostics compare chains with each other, so a sampler that
mixes well around the wrong distribution passes them.
[Simulation-based calibration](analysis-fitting-and-diagnostics.md#simulation-based-calibration)
checks the sampler against the model instead, by fitting data simulated
from the prior and testing whether the true parameters' ranks among the
posterior draws are uniform.

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
