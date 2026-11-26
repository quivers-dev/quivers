# MCMC and Hybrid Samplers

[Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
(MCMC) draws a sequence of dependent values whose stationary distribution is the
posterior, proposing and accepting new latent values by the model's log density.
quivers provides gradient-based MCMC, kernels that compose several moves, tempered
sequential Monte Carlo, samplers that combine a variational fit with MCMC, and
[posterior predictive](inference-svi.md#predictive-sampling) sampling from an
[`MCMCResult`](../api/inference/mcmc.md). [Variational Inference](inference-svi.md)
covers the alternative of fitting a guide.

## MCMC: HMC and NUTS

[Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) (HMC)
proposes moves through a continuous latent space by simulating Hamiltonian dynamics
with an auxiliary momentum, using the gradient of the log density [@neal2011mcmc]. The
simulation advances by *leapfrog* steps of a given step size, and the *mass matrix*
sets the scale of the momentum in each direction. The No-U-Turn Sampler (NUTS) chooses
the length of each trajectory by stopping where the path starts to turn back
[@hoffman2014nuts]. *Warmup* iterations tune the step size and the mass matrix before
the draws that are kept, and separate *chains* started from different values allow
comparing runs. Draws within a chain are correlated.

The kernels run on the latent registry's unconstrained vector
([Inference Foundations](inference-foundations.md#the-variational-pipeline)), so
positive or bounded continuous latents need no special treatment, and gradients come
from
[`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).
They move continuous coordinates only. A discrete latent site is updated by a
[Gibbs block](#composing-kernels) composed with them or summed out in the model.

MCMC targets the posterior of one run of the program, so `MCMC.run` takes an input
with one row, such as `x = torch.zeros(1, 1)`, and refuses more rows
([Input rows are independent runs](inference-foundations.md#input-rows-are-independent-runs)).
Its latents are the sites one run at `(x, observations)` reaches, every position of a
scanned program cell included.

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

[`HMCKernel`](../api/inference/mcmc.md) and [`NUTSKernel`](../api/inference/mcmc.md)
adapt the step size during warmup by Nesterov dual averaging [@hoffman2014nuts], and
adapt the mass matrix when `mass_matrix` is `"diagonal"` or `"dense"`, from a
running estimate of the posterior covariance updated one draw at a time. `"identity"`,
the `HMCKernel` default, turns mass-matrix adaptation off, and `NUTSKernel` defaults
to `"diagonal"`. Chains run sequentially: the driver loops over `num_chains`, and within
each chain warmup and sampling advance one leapfrog step at a time.
Each leapfrog step evaluates the target once, through
[`PotentialFn.value_and_grad`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn.value_and_grad)
(one forward and one backward pass). The value it returns is the
endpoint log-density in the Hamiltonian, and NUTS carries the
selected proposal's gradient into the next transition, so a
transition costs one target evaluation per leapfrog step and nothing
more.

An evaluation costs little beyond the model's own tensor arithmetic. The first
evaluation for a given set of shapes, dtypes, and devices (of the latents, the data,
and the values a Gibbs block conditions on) records the program's scoring run as a
fixed sequence of tensor operations, and every later evaluation replays that sequence
in the same order. Draws, log densities, gradients, and evaluation counts are thus
those of
[`MonadicProgram.log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint),
bit for bit. A program with a scan, whose recursion runs over its
sequence, is scored by `log_joint` at every evaluation, and so is any
evaluation made while an effect handler that does more than fix sites is in scope, or of
a program whose `log_joint` is overridden. An intervention by
[`do`](../api/effects/handlers.md#quivers.effects.do.do), which only fixes sites, is
replayed like any other evaluation.

Each chain derives an independent local generator from `seed`, and that stream
controls initialization, momentum, tree direction, subtree
selection, and acceptance. Thus repeated runs with the same seed are
reproducible without changing Torch's global RNG state. Pass a CPU
`torch.Generator` as `generator=` instead to have successive calls advance a stream
the caller owns. `seed` and `generator` are mutually
exclusive. Step-size and mass-matrix adaptation start fresh for every
chain and every call to `run`.

Before sampling, `MCMC.run` scores each chain's initial position with
[`PotentialFn.check_initial`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn.check_initial),
which lets every error through. Observed values that are not finite or lie outside a
likelihood's support, row counts that disagree, and errors in a custom score function
raise there with a note naming the chain. A position that scores without error but
with a non-finite log density is discarded, and the chain draws another from its own
stream, up to 100 attempts, as Stan does for random initial values. The first accepted
draw is unchanged for a model whose first position is finite. If no attempt is finite,
or the start is deterministic (`init_strategy="zero"`, or named initial values that fix
every coordinate), the error names the offending terms and the attempt count. During
sampling a position scores $-\infty$, and is
rejected, only when it lies outside the model's support: a `torch.distributions`
parameter or value validation error, recognized by the validator that raised it, or a
`torch.linalg.LinAlgError` from a matrix the position makes non-positive-definite.
Every other error propagates.

The [`MCMCResult`](../api/inference/mcmc.md) holds each site's draws at
`(num_chains, num_samples, *constrained_shape)`, where
`constrained_shape` is the shape of one draw: `()` for a scalar site,
`(d,)` for a vector site, and `(n, ...)` for a plate of size `n`. It exposes the
convergence diagnostic $\hat R$ and the effective sample size (ESS) of each site
[@vehtari2021rhat] as `r_hat`, `ess`, and `ess_tail`, each of shape `constrained_shape` (zero-dimensional
for a scalar site), computed from the draws on first access by
[`rhat`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.rhat) and
[`ess`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.ess). [Latent site shapes](inference-foundations.md#latent-site-shapes)
relates these shapes to the `(batch, ...)` values that `log_joint`
and the guides exchange. Each `r_hat` coordinate is the conservative
maximum of the rank-normalized and folded split statistics, while
`ess` is the rank-normalized cross-chain bulk ESS and `ess_tail` the
smaller of the 5% and 95% quantile ESS values. The same functions diagnose derived
quantities and draws from custom sampling loops
([convergence diagnostics for any draws](analysis-fitting-and-diagnostics.md#convergence-diagnostics-for-any-draws)).
The result also gives the number of divergences per chain, `divergence_counts`, their
sum, `total_divergences`, and the posterior log density of each draw,
`log_densities`.

$\hat R$ compares the variation within chains with the variation between them, so
chains that stay in different regions raise it, while chains trapped in the same region
can agree, which is why chains should start from dispersed values. The ESS accounts for
autocorrelation: 8,000 kept draws might give the precision of only 500 independent
ones. A divergence is a trajectory whose numerical integration error blew up, a sign of
curvature the step size cannot follow [@betancourt2016diagnosing]. Located in parameter
plots, divergences show whether a smaller step size or a different parameterization
would help. These diagnostics compare chains with each other, so a sampler that mixes
well around the wrong distribution passes them.
[Simulation-based calibration](analysis-fitting-and-diagnostics.md#simulation-based-calibration)
checks the sampler against the model instead, by fitting data simulated from the
prior and testing whether the true parameters' ranks among the posterior draws are
uniform.

### Named initial values

`MCMC(initial_values={"sigma": 0.5})` sets starting values in the latent
sites' natural coordinates. The driver first runs `init_strategy` (`"jitter"`,
`"prior"`, `"zero"`, or a given `"guide"`) with each chain's seeded generator,
then replaces the named sites in every chain. Unspecified sites retain the
strategy's values, so a partial mapping can set a constrained scale while
allowing dispersed starting locations.

```python
import torch

from quivers.dsl import loads
from quivers.inference import HMCKernel, MCMC

model = loads("""
object Obs : FinSet 3
program model : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    sample offset : Obs <- Normal(loc=0.0, scale=0.3)
    let mean = mu + offset
    observe y : Obs <- Normal(loc=mean, scale=sigma)
    return mu
export model
""").morphism

mcmc = MCMC(
    HMCKernel(step_size=0.03, num_steps=3),
    num_warmup=10,
    num_samples=20,
    num_chains=2,
    seed=2026,
    init_strategy="prior",
    initial_values={"sigma": 0.5, "offset": torch.tensor([-0.2, 0.0, 0.2])},
)
result = mcmc.run(model, torch.zeros(1, 1), {"y": torch.tensor([0.1, 0.4, 0.8])})
assert result.samples["offset"].shape == (2, 20, 3)
```

A scalar broadcasts over a site's declared shape; a tensor must have exactly
that shape, including plate and event axes. Thus `"offset": 0.0` initializes
all three offsets, while the vector above initializes them separately. A scalar
site has shape `()`. Values must be finite, real, non-Boolean, and strictly
inside the prior support: positive scales exceed zero, bounded values exclude
both endpoints, and simplex coordinates are positive and sum to one. Unknown
names, observed sites, and incompatible shapes raise before sampling.

A name may also be a site a [Gibbs block](#composing-kernels) updates. Those
sites sit outside the latent vector, so the chain starts at the given value in
place of a draw from the program, whatever `init_strategy` is. The value has the
shape of one draw of the site in `MCMCResult.samples` (a scalar broadcasts), and
it must be finite and inside the site's support, whose boundary is allowed here
since no bijector is inverted: `initial_values={"side": 1}` starts a
`Categorical(probs=[0.5, 0.5])` site at its second category. The Gibbs sites
left unnamed are drawn from the program given the named ones.

The driver detaches and copies the given tensors, so later changes to the
caller's mapping or tensors do not change its initialization. Named values are
part of [checkpoint compatibility](#checkpoints-budgets-and-sampler-statistics):
resume with the same mapping values as the original run.

### Windowed warmup

By default both kernels adapt in one phase (`adaptation="single"`):
dual averaging runs over the whole warmup under the initial metric, the
covariance estimate accumulates every warmup position, and the metric is
installed once, when warmup ends. Two problems may follow on a target
whose scales differ from one. The estimate includes the initial transient, which can
inflate it by orders of magnitude when a chain starts far from the region where most
of the posterior mass lies, and the frozen step size was tuned
for the identity metric rather than the metric the chain samples with.

`adaptation="windowed"` follows Stan's schedule instead
[@stanHmcTuning]. An
[`AdaptationSchedule`](../api/inference/mcmc.md#quivers.inference.mcmc.AdaptationSchedule)
splits the warmup into three phases: (i) an initial fast interval
(`init_buffer`, default 75 iterations) that adapts only the step size;
(ii) slow windows, the first `base_window` (default 25) iterations long
and each later one twice as long, that each accumulate a fresh
covariance estimate and install it, regularized as
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
[`NUTSKernel.adaptation_windows`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel)
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
averaging to settle, which is why `"single"` is the default.

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
  by element. Any other block takes an `update=` function
  ([`GibbsUpdate`](../api/inference/mcmc.md#quivers.inference.mcmc.GibbsUpdate))
  that draws from its full conditional, such as a conjugate update. Every
  discrete latent must belong to a Gibbs block, and `MCMC.run` raises
  naming any that does not. A block with an `update=` may also hold host
  data, names the program reads (in a `score` step, say) without
  declaring a distribution for them: `GibbsKernel(["z"], update=move,
  host=True)`. `MCMC` conditions the potential on host data as it does on
  sample sites, reports them in `MCMCResult.samples`, and saves them in
  checkpoints. Since the program has no draw of them, every chain's
  starting value comes from `initial_values`, taken with its own shape
  and dtype, and the update must return that shape and dtype. Enumeration
  needs the support of a sample site, so a host block without `update=`
  is refused, and so is a host block in `SMC`, whose particles start from
  the prior.
- [`CycleKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.CycleKernel)
  applies its components in order on every transition, and
  [`MixtureKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.MixtureKernel)
  applies one, chosen with fixed probabilities. Both forward the
  generator local to each chain and warmup to every component, and a mixture draws
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

The chain's state carries the log-density and gradient at its position,
which a gradient kernel reuses from the component before it. A Gibbs
sweep that changes a value leaves them unevaluated, and the next
component that reads them, or the draw the driver records, evaluates
them once. A cycle of several Gibbs blocks followed by NUTS thus pays one
density-and-gradient evaluation for the whole run of blocks, a sweep that
changes nothing pays none, and `KernelState.has_density` tells a callback
whether the state it sees has been evaluated.

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
budget cut any short. The arrays with one entry per draw then keep the first
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
`energy`, the Hamiltonian at the draw, and a composite kernel's appear under
each component's name. `energy` is what the energy plot and the
energy Bayesian fraction of missing information (E-BFMI) need
[@betancourt2016diagnosing], and
[`to_datatree`](../api/diagnostics/arviz_io.md#quivers.diagnostics.arviz_io.to_datatree)
writes these statistics under ArviZ's names (`diverging`, `tree_depth`,
`n_steps`, `energy`, `step_size`, `acceptance_rate`), so divergences
can be located in parameter space and saturation of the tree depth checked
draw by draw.

## Tempered sequential Monte Carlo

A single NUTS chain may stay in one mode of a multimodal posterior, and
MCMC gives no estimate of the evidence $p(y)$ for model comparison.
[`SMC`](../api/inference/mcmc.md#quivers.inference.smc.SMC) moves a
population of particles from the prior to the posterior through the
tempered targets $\pi_\beta(z) \propto p(z)\, L(z)^\beta$,
$0 = \beta_0 < \dots < \beta_T = 1$ [@delmoral2006sequential]. Here
$p(z)$ contains latent draw densities and explicit `Score(role="potential")`
factors; $L(z)$ contains observations and `Score(role="likelihood")`
factors. Roles come from the checked step records, rather than names such
as `lik`. `TemperedPotential` preserves prior potentials at every temperature.
`SMC.run` requires a normalized ancestral prior and refuses explicit prior
potentials whose mass it cannot determine. Its initial particles use the
shared prior dependency projection, retaining conditional prior inputs while
omitting downstream likelihood computations.
[`TemperedPotential`](../api/inference/mcmc.md#quivers.inference.smc.TemperedPotential)
evaluates $\log p(z) + \beta \log L(z)$ and returns the two parts
separately. Tempering requires the complete joint to come from the declared
native steps. An overridden `MonadicProgram.log_joint` is opaque to this
partition and is refused before execution; custom factor callbacks belong
in typed `Score` steps. Ordinary `PotentialFn` scoring still supports its
existing exact fallback for overridden joints.

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

The product of the mean incremental weights estimates the target's
normalizing constant, and `SMCResult.log_evidence` is its logarithm.
This constant is data evidence when the prior and observation factors form
the corresponding normalized model. Non-unit likelihood weights instead
define a power target; a response-dependent prior likewise needs its stated
conditional interpretation. Agreement between independent runs, and with
MCMC, checks the estimate. Like `MCMC.run`, `SMC.run` takes an input with one
row and refuses subsampled plates.

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

## Densities at many positions

A quadrature grid, a set of importance weights, a profile likelihood, or a check that
integrates a model's own conditional needs the log density at thousands of positions at
once.
[`PotentialFn.log_density_batch`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn)
takes positions of shape `(B, D)` on the registry's unconstrained vector and returns the
`(B,)` log densities in one pass over the program, and `value_and_grad_batch` returns each
row's gradient with it. The bijectors, their log-Jacobians, and every step run once for all
rows, and a value computed from the data alone is computed once and shared.

<!-- python: run -->
```python
import torch

from quivers.continuous import Draw, Euclidean, MonadicProgram, Score
from quivers.continuous import make_fixed_logit_beta, make_fixed_normal, reading
from quivers.core import Unit
from quivers.inference import LatentRegistry
from quivers.inference.mcmc import PotentialFn

counts = torch.tensor([3.0, 0.0, 5.0, 1.0])


def likelihood(env):
    # One run holds a and b at (4,), and a batch of positions at (B, 4).
    fit = (counts * torch.nn.functional.logsigmoid(env["a"])).sum(-1)
    return fit - env["b"].square().sum(-1)


model = MonadicProgram(
    Unit,
    Euclidean(name="out", dim=1),
    steps=[
        Draw(
            names=("a",),
            morphism=make_fixed_logit_beta(1.0, 2.0, Euclidean(name="A", dim=4)),
        ),
        Draw(
            names=("b",),
            morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="B", dim=4)),
        ),
        Score(
            name="lik",
            score=reading(likelihood, ("a", "b")),
            batched=True,
        ),
    ],
    return_vars=("a", "b"),
)
registry = LatentRegistry.from_model(model, set())
potential = PotentialFn(model, registry, torch.zeros(1, 1), {})

positions = torch.randn(2000, registry.total_unconstrained_dim)
values = potential.log_density_batch(positions)                  # (2000,)
values, gradients = potential.value_and_grad_batch(positions)    # (2000,), (2000, 8)
```

Each row equals [`PotentialFn`](../api/inference/mcmc.md#quivers.inference.mcmc.PotentialFn)'s
`log_density` at that row up to rounding. A row outside the model's support, where a scale
is negative or a covariance is not positive definite for instance, scores `-inf` with a zero
gradient and leaves the other rows as they are. A defect of the program raises, as it does at
one position.

### Which programs can be evaluated this way

Every step has to declare that it can. The steps the DSL compiler builds, the formula
interface's included, and the library's distributions declare it themselves. A
[`Let`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Let) or
[`Score`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Score) step
written in Python holds a callable the program cannot inspect, and declares it in one of two
ways.

1. `batched=True` on the step says that the callable accepts a leading batch axis. Over `B`
   positions it is called once. A value that depends on the position leads with an axis of
   length `B` ahead of the axes it has in one run, so a scalar site arrives at `(B,)` and a
   vector site at `(B, K)`. A value that every position shares arrives at its one-run shape.
   The result leads with `B` too, and a result that does not raises
   [`PositionLayoutError`](../api/effects/batching.md). The callable is also called for a
   single position, with no batch axis, so it is written against trailing axes.
2. [`mappable`](../api/continuous/program_steps.md#quivers.continuous.program_steps.mappable)
   around the callable says that it is a tensor computation that branches on no value. It is
   then mapped over the positions by `torch.func.vmap`, and sees the shapes of one run.

A distribution defined as a
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
subclass declares the second on its class, with `mappable = True`. A program with a step that
declares neither is refused with an
[`UnbatchableProgramError`](../api/effects/batching.md) that names each such step, and nothing
is evaluated to find out.

<!-- python: run -->
```python
from quivers.effects import UnbatchableProgramError

undeclared = MonadicProgram(
    Unit,
    Euclidean(name="out", dim=1),
    steps=[
        Draw(
            names=("a",),
            morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="A", dim=1)),
        ),
        Score(name="opaque", score=lambda env: -env["a"].square()),
    ],
    return_vars=("a",),
)
opaque = PotentialFn(
    undeclared, LatentRegistry.from_model(undeclared, set()), torch.zeros(1, 1), {}
)
try:
    opaque.log_density_batch(torch.zeros(3, 1))
except UnbatchableProgramError as refusal:
    print(refusal.obstacles[0].step)    # score step 'opaque'
```

A few programs are refused whatever they declare, because the number of terms in their joint
would differ between positions. These are an observation whose weights depend on a latent,
since a zero weight removes its row, and a `filter` over values that depend on one. A program
whose `log_joint` is overridden, and any evaluation made while a handler that does more than
fix sites is on the stack, is refused as well, since its density is not its steps' alone. A
`do` intervention only fixes sites, so a program under one is evaluated at many positions.

[`MonadicProgram.log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint)
over an input of `B` rows scores its independent runs together under the same declarations,
and one by one when a step declares neither, draws at random, reads a host datum given per
run, or runs under a handler that does more than fix sites.

The whole batch is held in memory at once, at `B` times the memory of one evaluation. A grid
of millions of positions is split into calls of a size that fits.

## Hybrid samplers

Hybrid samplers use a variational fit to build a proposal distribution or to choose
the starting values of MCMC chains.

### AutoDAIS

Differentiable annealed importance
sampling [@zhang2021dais; @geffner2021uha] wraps a base
guide with $K$ uncorrected Hamiltonian transitions along an annealing
path between base and target: each transition draws a fresh momentum
and takes leapfrog steps on the bridge
$q_0^{1-\beta_k}\, p^{\beta_k}$, with no Metropolis correction. The
base mean and scale, step sizes, and inverse temperatures are jointly
trained via stochastic variational inference (SVI), differentiating through the whole trajectory.

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

`WarmupThenHMC` trains a variational guide for `svi_steps` SVI steps and then starts
HMC chains at the guide's posterior mean. The chains start near the posterior, and
their convergence still depends on the HMC warmup and its diagnostics.

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

Posterior draws represent uncertainty about the latent quantities, and predictive draws
add the outcome noise: each takes a posterior draw of the latents and generates a new
response from the likelihood [@stanPredictiveChecks]. Thus an interval for new observations
includes both uncertainty about the latent mean and variability of the
response around it. The same calculation uses draws from either an MCMC
result or a variational guide. Comparing these replicated responses with
the observations gives a [posterior predictive check](analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics).

[`Predictive`](../api/inference/predictive.md) consumes either a
[`Guide`](../api/inference/guide.md) or an
[`MCMCResult`](../api/inference/mcmc.md). With an
`MCMCResult`, it iterates over posterior samples instead of calling
`guide.rsample`. Each recorded draw is one value that every run of the
predictive input shares, so both posteriors yield predictive draws of
shape `(num_samples, batch, ...)` for sites outside any plate.

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
