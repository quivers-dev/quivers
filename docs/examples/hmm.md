# Finite-state path composition

## Overview

The exported `hmm` is a finite-state path-composition example inspired by a discrete [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model). Its top-level morphisms live under `product_fuzzy`, have sigmoid-constrained entries, and have no `Dirichlet` declarations. They are thus not row-stochastic HMM kernels. The same source also contains `hmm_program`, which draws Dirichlet rows but currently leaves `transition_rows` unused in its marginalized observation body.

## QVR source

```qvr
# Finite-State Path Composition
#
# A finite-state path-composition example inspired by a K-state
# hidden Markov model. The top-level arrows live in the
# product-fuzzy algebra: their entries are sigmoid-constrained,
# but their rows are not normalized probability distributions.
# The composed path thus demonstrates runtime-variable
# powering, not the sum-product HMM forward algorithm.
#
# Structural form:
#
#   initial    : State -> State          fuzzy initial relation
#   transition : State -> State          fuzzy transition relation
#   emission   : State -> Obs            fuzzy emission relation
#   hmm        = initial >> repeat(transition) >> emission
#
# The separate hmm_program below demonstrates Dirichlet rows and
# exact marginalization for a single shared emission state. Its
# transition_rows draw is present for parameter-shape comparison
# but does not enter that likelihood; it is not a multi-step HMM.

composition product_fuzzy [level=algebra]

object State : FinSet 8
object Obs : FinSet 16
object Step : FinSet 12
object StateDist : Real 8

morphism initial, transition : State -> State [role=latent]
morphism emission : State -> Obs [role=latent]

define n_step = repeat(transition) >> emission
define hmm = initial >> n_step

# Probabilistic model for transpilation: the initial, transition,
# and emission parameter blocks are row-stochastic Dirichlet
# draws. The initial-state vector is one draw on the State
# simplex; transition_rows allocates one State-simplex row per
# source state; emission_rows allocates one Obs-simplex row per
# latent state. The marginalized body below uses initial_row and
# emission_rows only, so transition_rows remains prior-only.
#
# Step is the plate the emitted sequence is observed over and
# Obs is the alphabet each emission takes its value in, so the
# two are separate objects. StateDist is the value space of the
# returned initial-state vector, a point of the State simplex
# embedded in R^8.
program hmm_program : Step -> StateDist
    sample initial_row <- Dirichlet(concentration=1.0) [over=State]
    sample transition_rows : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission_rows : State <- Dirichlet(concentration=1.0) [over=Obs, iid_over=State]

    marginalize state <- Categorical(probs=initial_row) [reduction=logsumexp]
        observe obs : Step <- Categorical(probs=emission_rows[state])

    return initial_row

export hmm
export hmm_program
```

## Understanding the model

A state-to-state relation assigns a number to each possible step. Repeating
it combines routes with intermediate states; following it by an emission
relation connects starting states to observations. The arithmetic matters:
HMM probabilities sum mutually exclusive paths, whereas product-fuzzy
composition combines route strengths with [noisy-OR](../tutorials/python/01-first-quiver.md#composition).

Read the two constructions in this source separately. The exported relation
demonstrates repeated composition. The probabilistic `hmm_program` gives one
shared hidden state an emission sequence and leaves its [transition draw](../tutorials/qvr/05-time-series.md#finite-state-path-composition)
unused. It thus cannot learn transitions from that likelihood. Inspecting
which sampled values actually reach an observation is as important as
inspecting their shapes.

## Walkthrough

`composition product_fuzzy [level=algebra]` selects product and noisy-OR aggregation for fuzzy relations. It is not the sum-product semiring used by the HMM forward algorithm. A Viterbi implementation would also require log weights and max aggregation; changing this declaration alone is not a validated conversion of the current parameters.

`object State : FinSet 8` and `object Obs : FinSet 16` are finite discrete spaces: the latent state set and the emission alphabet. `object Step : FinSet 12` is the separate plate the emitted sequence is observed over, and `object StateDist : Real 8` is the value space of the returned initial-state vector, a point of the State simplex embedded in $\mathbb{R}^8$. The three `[role=latent]` declarations introduce learned relation tensors. To build a normalized categorical kernel, add a row-wise [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution#Symmetric_case) family or change base through a row-normalizing transformation.

`repeat` builds the algebraic power of `transition` by repeated squaring, with $n$ supplied via `prog(n_steps=N)`. Under the current algebra, the result is a product-fuzzy relation. Also note that `initial : State -> State`, so the exported domain is `State`, not a singleton initial-distribution object.

`hmm_program` is the separate probabilistic model exported from the same source. Its three `sample` steps draw the initial-state vector, the transition kernel, and the emission kernel from symmetric `Dirichlet(1.0)` priors: `[over=State]` names the simplex axis of the initial row, which carries no plate annotation because it is a single draw, and `iid_over=State` allocates one independent Dirichlet row per source state for `transition_rows` and per latent state for `emission_rows`. The `marginalize` block integrates the latent `state` out of the observation by log-sum-exp over the eight `State` atoms, so the observation body scores `obs` under `Categorical(probs=emission_rows[state])` alone and `transition_rows` never enters the likelihood.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Draw the three row-stochastic parameter blocks from their own Dirichlet priors, draw one latent state from the initial row, and emit a `Step`-long sequence from that state's emission row. `transition_rows` never enters the likelihood under this body, so it is drawn from the prior and clamped alongside the other two; the emitted sequence is a single-state categorical mixture rather than a full transition sequence.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")
model = prog.hmm_program

n_state, n_emit, n_step = 8, 16, 12

true_initial_row = torch.distributions.Dirichlet(torch.ones(n_state)).sample()
true_transition_rows = torch.distributions.Dirichlet(
    torch.ones(n_state),
).sample((n_state,))
true_emission_rows = torch.distributions.Dirichlet(
    torch.ones(n_emit),
).sample((n_state,))

state = torch.distributions.Categorical(true_initial_row).sample()
obs = torch.distributions.Categorical(true_emission_rows[state]).sample(
    (n_step,),
)

observations = {"obs": obs}
x_in = torch.zeros(1, 1)
```

### SVI fit

Reinitialize the raw relation logits and recover the n-step joint that produced the synthetic histogram by minimizing the cross-entropy between the normalized `prog(n_steps=n_steps)` and the observed counts. The optimizer walks the sigmoid-constrained entries of the product-fuzzy algebra directly.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))
n_steps = 5
true_marginal = prog(n_steps=n_steps).detach()
probs = (true_marginal / true_marginal.sum()).flatten()
N = 500
obs_counts = torch.distributions.Multinomial(
    total_count=N, probs=probs,
).sample().reshape(true_marginal.shape)

# Fresh logits for fitting.
torch.manual_seed(1)
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))

optim = torch.optim.Adam(list(prog.parameters()), lr=5e-2)
losses = []
for _ in range(150):
    optim.zero_grad()
    pred = prog(n_steps=n_steps)
    pred_norm = pred / (pred.sum() + 1e-12)
    loss = -(obs_counts * (pred_norm + 1e-12).log()).sum()
    loss.backward()
    optim.step()
    losses.append(float(loss.detach()))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

The exported `hmm` composition has no `sample` priors of its own; its `initial`, `transition`, and `emission` latents are plain algebra parameters with sigmoid-constrained entries. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts each `nn.Parameter` into a Normal-prior sample site so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) has a continuous unconstrained state space. The likelihood scores the observed histogram against the normalized n-step joint emitted by `prog(n_steps=K)`.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel, lift_from_log_prob

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))
n_steps = 5
true_marginal = prog(n_steps=n_steps).detach()
probs = (true_marginal / true_marginal.sum()).flatten()
N = 500
obs_counts = torch.distributions.Multinomial(
    total_count=N, probs=probs,
).sample().reshape(true_marginal.shape)

def hmm_log_prob(x, counts):
    pred = prog(n_steps=n_steps)
    pred_norm = pred / (pred.sum() + 1e-12)
    return (counts * (pred_norm + 1e-12).log()).sum().expand(x.shape[0])

lifted, lx, lobs = lift_from_log_prob(
    prog,
    log_prob_fn=hmm_log_prob,
    parameter_prior_scale=1.0,
    target_key="obs_counts",
    x=torch.zeros(1, 1),
    observations={"obs_counts": obs_counts},
)
kernel = NUTSKernel(step_size=0.01, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

HMM forward and Viterbi recurrences can be expressed as matrix products over sum-product and max-plus semirings. The current example instead demonstrates runtime-variable powering under Quivers' product-fuzzy rule. A normalized HMM needs different parameter constraints and aggregation.
