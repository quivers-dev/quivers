# GRU-shaped language model

## Overview

This stochastic recurrent cell is shaped after a [GRU](https://doi.org/10.3115/v1/D14-1179): it samples update and reset gates and mixes a candidate with the previous hidden state. It is not the canonical GRU equation. In particular, the candidate is centered directly on `r * h_prev` and does not use `x_t` or a learned candidate transformation. [@cho2014encoder]

## QVR source

```qvr
# GRU-Shaped Stochastic Language Model
#
# A stochastic gated recurrence wrapped in scan and used as a
# causal language model. It borrows the update and reset gates
# of a GRU, but its candidate is centered directly on the
# reset-gated previous state rather than on a learned transform
# of (x_t, r_t * h_{t-1}).
#
# Generative structure:
#
#   z_t      ~ LogitNormal(gate_z(x_t, h_{t-1}))  update gate
#   r_t      ~ LogitNormal(gate_r(x_t, h_{t-1}))  reset gate
#   h_cand   ~ Normal(r_t * h_{t-1}, 0.5)         simplified candidate
#   h_t      = (1 - z_t) * h_{t-1} + z_t * h_cand gated update
#   next_t   ~ Categorical(logits=lm_head(h_t))   next-token target
#
# scan threads hidden state across each sequence and returns its
# terminal state; a Categorical lm_head projects that summary
# onto the Token vocabulary.
#
# Resp is the plate: it indexes the 32 scored rows of the corpus,
# one next-token target per context. Token is the vocabulary, so it
# is the value space of what lm_head draws and of what the program
# returns.
#
# Reference: [Cho et al. 2014](https://doi.org/10.3115/v1/D14-1179).

object Token : FinSet 256
object Resp : FinSet 32
object Embedded : Real 64
object Hidden : Real 128

morphism tok_embed : Token -> Embedded [role=embed]
morphism gate_z, gate_r : Embedded * Hidden -> Hidden ~ LogitNormal
morphism lm_head : Hidden -> Token ~ Categorical

program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    sample z <- gate_z(x_t, h_prev)
    sample r <- gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev

    sample h_cand <- Normal(loc=reset_hidden, scale=0.5)

    let z_complement = 1.0 - z
    let h_new = z_complement * h_prev + z * h_cand
    return h_new

define backbone = tok_embed >> scan(gru_cell)

program gru_lm : Token -> Token
    sample h <- backbone

    observe next_token : Resp <- lm_head(h)
    return next_token

export gru_lm
```

## Understanding the model

A gate near zero retains little of the term it multiplies; a gate near one
retains almost all of it. In the update equation, a small `z` keeps the
[previous hidden state](vanilla-rnn-lm.md#understanding-the-model), while a
large `z` favors the new candidate. The reset gate `r` controls how much
previous state informs that candidate. In a [standard GRU cell](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#variants-on-long-short-term-memory),
the candidate also depends on the current input through a learned
transformation [@olah2015lstm].

Here the gates and candidate are [random draws](../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample), so two executions with the
same inputs can have different hidden states. The categorical head converts
the terminal summary to probabilities over the next token. The table below
should be read with that sampling interpretation: the gate means motivate
the architecture, but a sampled LogitNormal gate is not a deterministic sigmoid.

## Walkthrough

### Cell equations

| Step | DSL | Meaning |
|---|---|---|
| update gate | `z <- gate_z(x_t, h_prev)` | $z_t \sim \mathrm{LogitNormal}(\mu_z(x_t,h_{t-1}), \sigma_z(x_t,h_{t-1}))$ |
| reset gate | `r <- gate_r(x_t, h_prev)` | $r_t \sim \mathrm{LogitNormal}(\mu_r(x_t,h_{t-1}), \sigma_r(x_t,h_{t-1}))$ |
| reset-gated state | `let reset_hidden = r * h_prev` | $r_t \odot h_{t-1}$ |
| candidate | `h_cand <- Normal(reset_hidden, 0.5)` | $\tilde h_t \sim \mathcal{N}(r_t \odot h_{t-1}, 0.5)$ |
| update | `let h_new = z_complement * h_prev + z * h_cand` | $h_t = (1 - z_t)\,h_{t-1} + z_t \,\tilde h_t$ |

The candidate is drawn from a Normal centered on the reset-gated previous state; the update-gate convex combination $(1 - z_t)\,h_{t-1} + z_t \,\tilde h_t$ interpolates between persistence and the new candidate.

### State threading

`scan(gru_cell)` threads the hidden state $h_t$ across the sequence; the `Categorical` `lm_head` scores the next-token target from the terminal state $h_T$.

The two `FinSet` objects play different roles. `Resp : FinSet 32` sits in the observe step's index slot, so it is the plate: 32 scored rows, one next-token target per context. `Token : FinSet 256` sits in `lm_head`'s codomain and in the program's own codomain, so it is the value space the draw ranges over.

```mermaid
flowchart LR
    x_t["x_t"] --> gate_z["gate_z"]
    x_t["x_t"] --> gate_r["gate_r"]
    h_prev["h_prev"] --> gate_z["gate_z"]
    h_prev["h_prev"] --> gate_r["gate_r"]
    gate_r["gate_r"] --> reset_hidden["reset_hidden"]
    h_prev["h_prev"] --> reset_hidden["reset_hidden"]
    reset_hidden["reset_hidden"] --> h_cand["h_cand"]
    h_prev["h_prev"] --> h_new["h_new"]
    gate_z["gate_z"] --> h_new["h_new"]
    h_cand["h_cand"] --> h_new["h_new"]
    h_new["h_new"] --> scan["scan"]
    class x_t,h_prev qv-input
    class gate_z,gate_r,reset_hidden,h_cand qv-checked
    class h_new qv-foundation
    class scan qv-output
```

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Trace the fixed GRU-shaped recurrence once to obtain its latent trajectory and target. The scan reaches the embedding and the cell's draws once per position, at the sites `h$x_in@t`, named under the scan's binding `h`, and the cell's own labels at `@t`, and the terminal state `h` is a function of them. Reuse the trajectory in `observations` so later likelihood evaluations score that same realization; otherwise each evaluation draws a new one. Each `Resp` row contains an int64 prompt of length `seq_len` and one next-token label.

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/gru_lm.qvr")
model = prog.morphism

# Fix the model's kernel parameters to chosen values, then run one
# forward trace so the captured hidden state and the next-token target it
# generated are jointly consistent under the same weights.
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

rows, seq_len, vocab = 32, 8, 256
prompts = torch.randint(0, vocab, (rows, seq_len))
with torch.no_grad():
    forward = trace(model, prompts)
# Every position's embedding and cell draws, `h$x_in@t` and the cell's own
# sites at `@t`, are the latent trajectory the terminal state is a
# function of.
trajectory = {
    name: site.value.detach()
    for name, site in forward.sites.items()
    if "@" in name and not site.is_deterministic
}
next_token = forward.sites["next_token"].value.detach()

x_in = prompts
observations = {"next_token": next_token, **trajectory}
print("prompts:", prompts.shape, prompts.dtype)
print("trajectory sites:", len(trajectory))
print("next_token:", next_token.shape, next_token.dtype)
```

### SVI fit

Reinitialize `tok_embed`, `gate_z`, `gate_r`, and `lm_head`, then fit a mean-field posterior over their parameters and the latent trajectory with an [`AutoNormalGuide`](../api/inference/guide.md) and [`SVI`](../api/inference/svi.md). The scan draws the embedding and the cell's sites at every position, so the trajectory is lifted beside the parameters with [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters), each label stacked over positions. The [`ELBO`](../api/inference/elbo.md) scores the Categorical `next_token` site for each prompt.

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI, bayesian_lift_parameters
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/gru_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, seq_len, vocab = 32, 8, 256
prompts = torch.randint(0, vocab, (rows, seq_len))
targets = model.rsample(prompts)
observations = {"next_token": targets}

torch.manual_seed(1)
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

# The scan's draws at every position are latents too: collect each
# label, stacked as (rows, seq_len, ...), and lift it beside the
# kernel parameters so the guide covers the whole trajectory.
forward = trace(model, prompts, observations)
per_position: dict[str, list[torch.Tensor]] = {}
for name, site in forward.sites.items():
    label, at, _ = name.partition("@")
    if at and not site.is_deterministic and not site.is_observed:
        per_position.setdefault(label, []).append(site.value)
trajectory_shapes = {
    label: tuple(torch.stack(values, dim=1).shape)
    for label, values in per_position.items()
}
lifted, lx, lobs = bayesian_lift_parameters(
    model, prompts, observations,
    prior_scale=1.0,
    additional_latents=trajectory_shapes,
)

guide = AutoNormalGuide(lifted, observed_names=set(lobs))
optim = torch.optim.Adam(guide.parameters(), lr=1e-2)
svi = SVI(lifted, guide, optim, ELBO(num_particles=1))

losses = [svi.step(lx, lobs)]
for _ in range(30):
    losses.append(svi.step(lx, lobs))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

The lifted Bayesian model treats both the parameters $\theta$ and the latent trajectory $z$ as latents: $p(\theta, z \mid x, y) \propto p(\theta) \, p(z \mid x, \theta) \, p(y \mid z, \theta)$, where $z$ collects every draw the scan makes at every position, the embedding `h$x_in@t` and the cell's own sites, and the terminal state is a function of $z$. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) declares Normal priors on every learnable parameter and accepts an `additional_latents` mapping that lifts each per-position label, stacked over positions as `(rows, seq_len, ...)`, as a NUTS variable with a placeholder Normal prior. The score step substitutes both into the inner program and cancels the placeholder, leaving $\log p(\theta) + \log p_{\text{inner}}(z, y \mid x, \theta)$. Given the full $(\theta, z)$ state, this log density is deterministic and introduces no Monte Carlo estimate during leapfrog steps.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel, bayesian_lift_parameters
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/gru_lm.qvr")
model = prog.morphism
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, seq_len, vocab = 32, 8, 256
prompts = torch.randint(0, vocab, (rows, seq_len))
targets = model.rsample(prompts)
observations = {"next_token": targets}

# The latent trajectory: each label the scan reaches per position,
# stacked as (rows, seq_len, ...), the layout a scanned site's values
# take under its bare label.
forward = trace(model, prompts, observations)
per_position: dict[str, list[torch.Tensor]] = {}
for name, site in forward.sites.items():
    label, at, _ = name.partition("@")
    if at and not site.is_deterministic and not site.is_observed:
        per_position.setdefault(label, []).append(site.value)
trajectory_shapes = {
    label: tuple(torch.stack(values, dim=1).shape)
    for label, values in per_position.items()
}
lifted, lx, lobs = bayesian_lift_parameters(
    model, prompts, observations,
    prior_scale=1.0,
    additional_latents=trajectory_shapes,
)
kernel = NUTSKernel(step_size=0.005, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The GRU cell is a Kleisli morphism $\mathrm{Embedded} \times \mathrm{Hidden} \to \mathcal{G}(\mathrm{Hidden})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category; `scan(gru_cell)` is its iterated composition along the sequence. The Categorical head and `observe` step close the composite into the LM likelihood by accumulating per-batch categorical log-probabilities. [@giry1982categorical]

## Sources and further reading

- *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation* [@cho2014encoder].
- *A categorical approach to probability theory* [@giry1982categorical].
