# Continuous State-Space Model

## Overview

A continuous state-space model extends an HMM-style sequence to continuous latent states and observations. This example demonstrates `scan`, a one-step generative program, and a separately learned recognition-reconstruction path. The recognition cell is not derived by Bayesian inversion of the transition and emission kernels.

## QVR source

```qvr
# Continuous State-Space Model
#
# A continuous state-space wiring example with a one-step
# generative program and a separate recurrent recognition path.
# The learned recognition cell resembles a filter, but it is not
# the Bayesian inverse of the transition and emission kernels.
#
# Generative structure:
#
#   s_t  ~ transition(s_{t-1})                    State -> State kernel
#   o_t  ~ emission(s_t)                          State -> Obs kernel
#   h_t  ~ inference_cell(o_t, h_{t-1})           learned summary
#
# scan threads the learned state across the observation sequence;
# filter_and_reconstruct decodes its terminal summary back to
# observation space.

object State : Real 16
object Obs : Real 8

morphism transition : State -> State ~ Normal
morphism emission : State -> Obs ~ Normal

program generative_step : State -> State
    sample s_new <- transition

    observe o <- emission(s_new)
    return s_new

morphism inference_cell : Obs * State -> State ~ Normal

define filter = scan(inference_cell)

morphism decoder : State -> Obs ~ Normal

define filter_and_reconstruct = scan(inference_cell) >> decoder

export generative_step
export filter_and_reconstruct
```

## Understanding the model

A hidden state represents something evolving that is not directly measured,
such as position; an emission represents a noisy measurement of it. The
transition answers where the next state might be given the previous state.
The emission answers what observation might be seen given that new state.
A recognition model asks the reverse inferential question, but must be trained
or derived to answer it.

The fitting blocks supply synthetic `s_new` as well as `o`, teaching the
kernels from known transitions and observations. To infer states from measurements,
`s_new` would need to remain unobserved and the inference method would need
to [approximate](../tutorials/python/05-variational-inference.md#creating-a-guide) its distribution conditional on `o`.

## Walkthrough

`object State : Real 16` and `object Obs : Real 8` introduce the two Euclidean spaces: a 16-dimensional latent and an 8-dimensional observation.

`morphism transition : State -> State ~ Normal` evolves the latent state by one time step under a Normal kernel whose mean and scale are both produced from the previous state by the kernel's [`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource). `morphism emission : State -> Obs ~ Normal` projects a state to an observation under the same kernel family.

`program generative_step : State -> State` is a one-step monadic program: `sample s_new <- transition` draws the new latent state from the transition kernel, `observe o <- emission(s_new)` scores an observation against the emission kernel, and `return s_new` projects the program's joint kernel onto the new state. `repeat(generative_step, n)` can compose this endomorphism a fixed number of times. A `scan` cell instead needs a product domain `Input * State -> State`, as the separate `inference_cell` declaration shows.

`morphism inference_cell : Obs * State -> State ~ Normal` is a learned recurrent cell that incorporates a new observation into a state representation. `define filter = scan(inference_cell)` threads it across a sequence. Using it as an approximate Bayesian filter would require training its state distributions against a filtering objective.

`morphism decoder : State -> Obs ~ Normal` decodes a state back to observation space; `define filter_and_reconstruct = scan(inference_cell) >> decoder` composes the scan with the decoder so the exported pipeline filters and reconstructs in one composite.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Pick ground-truth linear dynamics for the latent state and a random emission matrix, then forward-sample a trajectory of latent states and observations of length `T`. The single-step program `generative_step : State -> State` reads the previous state as input; the per-step pair `(s_new, o)` is supplied as the observation dict.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/continuous_hmm.qvr")
model = prog.morphism

T = 32
state_dim = 16
obs_dim = 8
A = 0.9 * torch.eye(state_dim)
C = 0.3 * torch.randn(obs_dim, state_dim)
s = torch.zeros(T + 1, state_dim)
o = torch.zeros(T, obs_dim)
for t in range(T):
    s[t + 1] = s[t] @ A.T + 0.1 * torch.randn(state_dim)
    o[t] = s[t + 1] @ C.T + 0.1 * torch.randn(obs_dim)
state_prev = s[:T]
sites = {"s_new": s[1:T + 1], "o": o}
```

### SVI fit

The exported program is a single-step [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) with no explicit priors on the kernel parameter networks; [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts each leaf parameter into a unit-Normal sample site so [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) can build a mean-field surrogate. With both `s_new` and `o` observed, the ELBO bounds the log evidence for that supplied trajectory under the lifted parameter model; it is not the evidence itself.

```python
from quivers.inference import bayesian_lift_parameters
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
prog = load("docs/examples/source/continuous_hmm.qvr")
inner = prog.morphism
model, x_lift, obs_lift = bayesian_lift_parameters(
    inner, state_prev, sites, prior_scale=1.0,
)

guide = AutoNormalGuide(model, observed_names={"s_new", "o"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_lift, obs_lift)
losses = [svi.step(x_lift, obs_lift) for _ in range(300)]
loss_final = sum(losses[-20:]) / 20.0
oracle_ll = inner.log_joint(state_prev, sites).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p:     {-oracle_ll:.1f}")
```

### NUTS posterior

The lifted model exposes one Normal sample site per leaf parameter; [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly under small warmup and sample budgets.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
prog = load("docs/examples/source/continuous_hmm.qvr")
model, x_lift, obs_lift = bayesian_lift_parameters(
    prog.morphism, state_prev, sites, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_lift, obs_lift)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical perspective

The `scan` combinator implements [Kleisli composition](https://en.wikipedia.org/wiki/Kleisli_category) threaded through time. Given a cell $f : O \times S \to S$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category), it consumes an observation sequence, threads the sampled state, and returns the terminal state. The current `inference_cell` defines a learned conditional Normal update; an exact filtering posterior would require the Bayesian conditioning calculation for the transition and observation model. [@giry1982categorical]

The generative program uses `transition` and `emission`; the recognition path uses distinct `inference_cell` and `decoder` morphisms. The source does not tie their parameters or call `bayes_invert`, so the two paths are not the same kernels run in opposite directions.
